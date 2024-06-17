import os
import signal
import time

import numpy as np
import psutil as psu
import netCDF4 as nc
from omuse.units import units
from pytams.fmodel import ForwardModel

from popomo.poputils import (
    disableCheckPoint,
    getLastRestart,
    getPOPinstance,
    setCheckPoint,
    setRestart,
    setStoichForcingAmpl,
)
from popomo.utils import (
    daysincurrentyear,
    monthly_reference,
    remainingdaysincurrentmonth,
)


class PopomoError(Exception):
    """Exception class for Popomo."""

    pass


class POPOmuseModel(ForwardModel):
    """A forward model for pyTAMS based on POP-Omuse."""

    def __init__(self, params: dict = None, ioprefix: str = None) -> None:
        """Override the template."""
        # Stash away POP-specific parameters
        self._pop_params = params.get("pop", {})

        # In this model, the state is a pointer to a POP restart file
        # Upon initialization, pass a initial solution if one is provided
        self._state = None
        init_file = self._pop_params.get("init_file", None)
        if init_file:
            assert os.path.exists(init_file) is True
            self._state = init_file

        # POP simulation is not starting at t=0
        # keep track of initial time
        self._init_time = self._pop_params.get("init_time", 0.0)

        # Forcing method
        # Default is "baseline-frac" which only requires a single
        # random number
        self._forcing_method = self._pop_params.get("forcing_method", "baseline-frac")
        # For ERA5 based forcing, need many more random
        self._nr_eofs_ep = 0
        self._nr_eofs_t2m = 0

        # POP domain parameters
        self._popDomain = self._pop_params.get("domain_dict", None)
        assert self._popDomain is not None

        # The actual POP object
        self.pop = None

        # Database information
        self.checkpoint_prefix = None
        if params.get("database", {}).get("DB_save", False):
            nameDB = "{}.tdb".format(
                params.get("database", {}).get("DB_prefix", "TAMS")
            )
            model = "pop_{}x{}x{}".format(
                self._popDomain.get("Nx", 120),
                self._popDomain.get("Ny", 56),
                self._popDomain.get("Nz", 12),
            )
            self.run_folder = "{}/trajectories/{}".format(nameDB, ioprefix)
            self.checkpoint_prefix = "{}/trajectories/{}/{}".format(
                nameDB, ioprefix, model
            )
            self.era5data_file = "{}/ARSpinUpData.nc".format(self.run_folder)
            if not os.path.exists(self.run_folder):
                os.mkdir(self.run_folder)

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        # On the first call to advance, initialize POP
        if self.pop is None:
            self.spinup_AR_ERA5(self.era5data_file)
            self.pop = getPOPinstance(
                pop_domain_dict=self._popDomain,
                pop_options=self._pop_params,
                root_run_folder=self.run_folder,
            )
            if self.checkpoint_prefix:
                setCheckPoint(self.pop, int(dt), self.checkpoint_prefix)
            else:
                disableCheckPoint(self.pop)
            if self._state is not None:
                setRestart(self.pop, self._state)
                if not os.path.exists(self._state):
                    raise PopomoError(
                        "State file {} do not exists".format(self._state)
                    )
            print("Start advancing with init. state {}".format(self._state))

        # Time stepping is month based, so exact length varies
        tstart = self.pop.model_time
        date = self.pop.get_model_date()
        days_left = remainingdaysincurrentmonth(date)
        year_length = daysincurrentyear(date)
        tnow = tstart
        dt_d = days_left * 1 | units.day
        tstoich_end = tstart + dt_d

        print(
            "Start stoich step: tstart = {}, t_end = {}".format(
                tstart.value_in(units.day), tstoich_end.value_in(units.day)
            )
        )

        # Set stoichastic amplitude
        self._noise = np.random.randn(1)
        sfwf_ampl = (
            forcingAmpl
            * np.sqrt(dt_d.value_in(units.day) / year_length)
            * self._noise
        )
        setStoichForcingAmpl(self.pop, sfwf_ampl)

        # Outer loop might not be necessary
        # since we rely on POP internal sub-stepping
        # but keep it for now
        while tnow < tstoich_end:
            tend = tnow + dt_d
            self.pop.evolve_model(tend)
            tnow = self.pop.model_time

        # Retrieve the actual time for which POP was integrated
        actual_dt = tnow - tstart

        # State is handled by a pointer to a check file
        self._state = getLastRestart(self.pop)

        # Convert to year
        return actual_dt.value_in(units.day) / year_length

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state) -> None:
        """Override the template."""
        self._state = state

    def score(self) -> float:
        """Normalized AMOC strength minus monthly reference."""
        # When called, the date is the first of
        # month following the month of interest
        date = self.pop.get_model_date()
        score_ref = monthly_reference(date)
        score = (26.0 - self.pop.get_amoc_strength()) / 26.0
        return score - score_ref

    def getNoise(self):
        """Return last generated noise."""
        return self._noise

    def setNoise(self, a_noise):
        """Set stochastic noise."""
        self._noise = a_noise

    def spinup_AR_ERA5(self, ARdatafile : str) -> None:
        """Spinup data for AR model when using ERA5 forcing."""
        if (self._forcing_method != "ERA-Data"):
            return

        # E-P data
        lag_re_ep = np.load('./e_p_lags.npy')       # Lags
        l_m_ep = int(np.amax(lag_re_ep))            # maximum lag
        rho_ep = np.load('./e_p_yw_rho.npy')        # dim = nr_eofs x max lag
        sig_ep = np.load('./e_p_yw_sigma.npy')      # dime = nr_eofs
        nr_ep = sig_ep.shape[0]
        hist_ep = np.zeros([nr_ep,l_m])           # spinup history

        for nr_i in range(nr_ep): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_ep[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_ep[nr_i,l_m-1] =  np.dot(hist_ep[nr_i,:lag],rho_ep[nr_i,:lag])
                                     + np.random.normal(0,sig_ep[nr_i])
                
                # Roll the time series to keep the history: 
                # last item (just computed) becomes the first,
                # first becomes second, etc.
                hist_ep[nr_i,:] = np.roll(hist_ep[nr_i,:],1)

        # t2m data
        lag_re_t2m = np.load('./t2m_lags.npy')      # Lags
        l_m_t2m = int(np.amax(lag_re_t2m))          # maximum lag
        rho_t2m = np.load('./t2m_yw_rho.npy')       # dim = nr_eofs x max lag
        sig_t2m = np.load('./t2m_yw_sigma.npy')     # dime = nr_eofs
        nr_t2m = sig_t2m.shape[0]
        hist_t2m = np.zeros([nr_t2m,l_m])           # spinup history

        for nr_i in range(nr_t2m): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_t2m[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_t2m[nr_i,l_m-1] =  np.dot(hist_t2m[nr_i,:lag],rho_t2m[nr_i,:lag])
                                     + np.random.normal(0,sig_t2m[nr_i])
                
                # Roll the time series to keep the history: 
                # last item (just computed) becomes the first,
                # first becomes second, etc.
                hist_t2m[nr_i,:] = np.roll(hist_t2m[nr_i,:],1)

        # Store AR data
        nc_out = nc.Dataset(ARdatafile, 'w')
        # dims
        lag_out = nc_out.createDimension('lag_d_ep',l_m_ep)
        eof_out = nc_out.createDimension('eof_d_ep',nr_eofs_ep)
        lag_out = nc_out.createDimension('lag_d_t2m',l_m_t2m)
        eof_out = nc_out.createDimension('eof_d_t2m',nr_eofs_t2m)
        # lags (as int)
        nc_lags = nc_out.createVariable('lags_ep', 'i4', 'eof_d_ep')
        for i in range(nr_ep):
            nc_lags[i] = int(lag_re_ep[i])
        nc_lags = nc_out.createVariable('lags_t2m', 'i4', 'eof_d_t2m')
        for i in range(nr_t2m):
            nc_lags[i] = int(lag_re_t2m[i])
        # rho
        nc_rho = nc_out.createVariable('rho_ep', 'f8', ['eof_d_ep','lag_d_ep'])
        nc_rho[:,:] = rho_ep[:,:]
        nc_rho = nc_out.createVariable('rho_t2m', 'f8', ['eof_d_t2m','lag_d_t2m'])
        nc_rho[:,:] = rho_t2m[:,:]
        # sig
        nc_sigs = nc_out.createVariable('sig_ep', 'f8', 'eof_d_ep')
        nc_sigs[:] = sig_ep[:]
        nc_sigs = nc_out.createVariable('sig_t2m', 'f8', 'eof_d_t2m')
        nc_sigs[:] = sig_t2m[:]
        # Next set of random number used in POP
        nc_rnd = nc_out.createVariable('rnd_ep', 'f8', 'eof_d_ep')
        nc_rnd[:] = np.random.randn(nr_ep)
        nc_rnd = nc_data_out.createVariable('rnd_t2m', 'f8', 'eof_d_t2m')
        nc_rnd[:] = np.random.randn(nr_t2m)
        # Spinup history data
        hist_out = nc_out.createVariable("hist_ep", 'f8', ["eof_d_ep","lag_d_ep"])
        hist_out[:,:] = hist_ep[:,:]
        hist_out = nc_out.createVariable("hist_t2m", 'f8', ["eof_d_t2m","lag_d_t2m"])
        hist_out[:,:] = hist_t2m[:,:]
        nc_out.close()



    @classmethod
    def name(self):
        """Return the model name."""
        return "POPOmuseModel"

    def listProcess(self):
        """List process currently running."""
        for p in psu.process_iter():
            if "pop_worker" in p.name():
                print(p, flush=True)

    def killSleepingProcess(self):
        """Kill pop process currently sleeping."""
        for p in psu.process_iter():
            if "pop_worker" in p.name() and "sleep" in p.status():
                os.kill(p.pid(), signal.SIGKILL)

    def clear(self):
        """Clear any model internals."""
        print("Clearing model internals", flush=True)
        time.sleep(2.0)
        if self.pop:
            self.pop.cleanup_code()
            self.pop.clear()
            del self.pop
            self.pop = None
        self._state = None
        # print("Process after del()", flush=True)
        # self.listProcess()
        time.sleep(2.0)
        # self.killSleepingProcess()
