import os
import signal
import time
import numpy as np
import psutil as psu
import netCDF4 as nc
import logging
from typing import Any
from omuse.units import units
from pytams.fmodel import ForwardModelBaseClass

from popomo.poputils import (
    disableCheckPoint,
    getLastRestart,
    getPOPinstance,
    setCheckPoint,
    setRestart,
    setStochForcingAmpl,
    setStochForcingFWF,
    setStochForcingARfile,
    setStochERA5Forcingfile,
)
from popomo.utils import (
    daysincurrentyear,
    monthly_reference,
    remainingdaysincurrentmonth,
    year_month_day,
)
from popomo.forcing_ERA5 import (
    ERA5ForcingGenerator
)
from popomo.plotutils import (
    plot_globe,
    plot_globe_old,
)

_logger = logging.getLogger(__name__)


class PopomoError(Exception):
    """Exception class for Popomo."""

    pass


class POPOmuseModel(ForwardModelBaseClass):
    """A forward model for pyTAMS based on POP-Omuse."""

    def _init_model(self,
                    params: dict,
                    ioprefix: str = None) -> None:
        """Override the template."""
        # Setup the RNG
        if params["model"]["deterministic"]:
            seed = int(ioprefix[4:])
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # Stash away POP-specific parameters
        self._pop_params = params.get("pop", {})

        # In this model, the state is a pointer to a POP restart file
        # Upon initialization, pass a initial solution if one is provided
        self._state = None
        init_file = self._pop_params.get("init_file", None)
        if init_file:
            assert os.path.exists(init_file) is True
            self._state = init_file

        # Forcing method
        # Default is "baseline-frac" which only requires a single
        # random number
        self._forcing_method = self._pop_params.get("forcing_method", "baseline-frac")
        if self._forcing_method == "ERA5-dataAR":
            # For ERA5 AR based forcing, need many more random
            self._nr_eofs_ep = 0
            self._nr_eofs_t2m = 0
        elif self._forcing_method == "ERA5-data":
            # For ERA5 data, use the generator
            self._era5_prefix = self._pop_params.get("ERA5-basefile", "ERA5_Noise")
            self._era5_data_folder = self._pop_params.get("ERA5-dataloc", None)
            self._era5_init_year = self._pop_params.get("ERA5-inityear", None)
            self._era5_gen = ERA5ForcingGenerator(self._era5_data_folder)

        # POP domain parameters
        self._popDomain = self._pop_params.get("domain_dict", None)
        assert self._popDomain is not None

        # The actual POP object
        self.pop = None

        # Database information
        self.run_folder = "./"
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
            self.run_folder = f"{nameDB}/trajectories/{ioprefix}"
            self.checkpoint_prefix = "{}/trajectories/{}/{}".format(
                nameDB, ioprefix, model
            )
            if self._forcing_method == "ERA5-dataAR":
                self.era5data_file = f"{self.run_folder}/ARSpinUpData.nc"
            if not os.path.exists(self.run_folder):
                os.mkdir(self.run_folder)

    def _advance(self,
                 step: int,
                 time : float,
                 dt: float,
                 noise: Any,
                 forcingAmpl: float):
        """Override the template."""
        # On the first call to advance, initialize POP
        if self.pop is None:
            _logger.debug("Start POP itself")
            # Initialize Omuse-POP object
            self.pop = getPOPinstance(
                pop_domain_dict=self._popDomain,
                pop_options=self._pop_params,
                root_run_folder=self.run_folder,
            )
            _logger.debug("Done starting POP itself")

            # ERA5-dataAR forcing model need some spin-up
            if self._forcing_method == "ERA5-dataAR":
                self.spinup_AR_ERA5(self.era5data_file)
            # ERA5-data uses a forcing file
            # Formatted with {prefix}_{year} containing all 12 months.
            # only generate a new file if step == 0
            # otherwise rely on restart mechanism internals to ensure a forcing
            # data file exists
            elif self._forcing_method == "ERA5-data":
                if self._step == 0:
                    # Generate a forcing file for the starting year
                    # Can't use POP to get the year yet, so use input param
                    year = self._era5_init_year
                    forcing_file_base = f"{self.run_folder}/{self._era5_prefix}_{year:04}"
                    self._era5_gen.generate_forcing_file(forcing_file_base, year)
                    dbg_msg = f"Generating noise file {forcing_file_base}"
                    _logger.debug(dbg_msg)
                forcing_file_base = f"{self.run_folder}/{self._era5_prefix}"
                setStochERA5Forcingfile(self.pop, forcing_file_base)

            if self.checkpoint_prefix:
                setCheckPoint(self.pop, 12, self.checkpoint_prefix)
            else:
                disableCheckPoint(self.pop)
            if self._state is not None:
                setRestart(self.pop, self._state)
                if not os.path.exists(self._state):
                    err_msg = f"State file {self._state} do not exists"
                    _logger.error(err_msg)
                    raise PopomoError(err_msg)
            dbg_msg = f"Start advancing with init. state {self._state}"
            _logger.debug(dbg_msg)

        # Time stepping is month based, so exact length varies
        tstart = self.pop.model_time
        date = self.pop.get_model_date()
        days_left = remainingdaysincurrentmonth(date)
        year_length = daysincurrentyear(date, False)
        tnow = tstart
        dt_d = days_left * 1 | units.day
        tstoch_end = tstart + dt_d

        tstart_d = tstart.value_in(units.day)
        tend_d = tstoch_end.value_in(units.day)
        inf_msg = f"Start stoch step: tstart = {tstart_d}, t_end = {tend_d}"
        _logger.info(inf_msg)

        # Set stochastic amplitude
        if (self._forcing_method == "baseline-frac"):
            # Amplitude is a random number, scaled by user-provided forcingAmpl
            sfwf_ampl = (
                forcingAmpl
                * np.sqrt(dt_d.value_in(units.day) / year_length)
                * noise
            )
            setStochForcingAmpl(self.pop, sfwf_ampl)
        elif (self._forcing_method == "ERA5-data"):
            # Amplitude is set to user-provided forcingAmpl
            # Local data read from file provided in pop_in
            setStochForcingAmpl(self.pop, forcingAmpl)
        elif (self._forcing_method == "ERA5-dataAR"):
            # Random number for noise generated and passed here
            # user-provided forcingAmpl is used for amplitude scaling
            setStochForcingFWF(self.pop, noise)
            setStochForcingAmpl(self.pop, forcingAmpl)

        #sfwf_flux = self.pop.elements.surface_fresh_water_flux.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_flux, "kg/m2/s", "sfwf_flux", elements=True)
        #sfwf_flux_st = self.pop.elements.surface_freshwater_stoch_flux.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_flux_st, "kg/m2/s", "sfwf_flux_st", elements=True)
        #sfwf_precip = self.pop.element_forcings.sfwf_precip.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_precip, "kg/m2/s", "sfwf_precip", elements=True)
        #exit()

        # Outer loop might not be necessary
        # since we rely on POP internal sub-stepping
        # but keep it for now
        while tnow < tstoch_end:
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
        if self.pop:
            date = self.pop.get_model_date()
            amoc_ref = monthly_reference(date)
            score = (amoc_ref - self.pop.get_amoc_strength()) / amoc_ref
            dbg_msg = f"Date: {date}, score: {score}, Ref: {amoc_ref}"
            _logger.debug(dbg_msg)
            return score
        return 0.0

    def _make_noise(self) -> Any:
        """Override the template."""
        if (self._forcing_method == "baseline-frac"):
            # Single number for scaling of baseline forcing data
            return self._rng.standard_normal(1)
        elif (self._forcing_method == "ERA5-data"):
            # Return a file name and month index
            if self.pop:
                date = self.pop.get_model_date()
                year, month, _ =  year_month_day(date)
                # Generate new forcing files for the year
                # if they don't exists already
                forcing_file_base = f"{self.run_folder}/{self._era5_prefix}_{year:04}"
                self._era5_gen.generate_forcing_file(forcing_file_base, year)
                return f"{forcing_file_base}_{month}"
            else:
                return f"None"
        elif (self._forcing_method == "ERA5-dataAR"):
            # A set of random numbers for each EOF of both E-P and T 
            return self._rng.standard_normal(self._nr_eofs_ep)

    def spinup_AR_ERA5(self, ARdatafile : str) -> None:
        """Spinup data for AR model when using ERA5 forcing."""
        if (self._forcing_method != "ERA5-dataAR"):
            return

        # E-P data
        lag_re_ep = np.load('./e_p_lags.npy')       # Lags
        l_m_ep = int(np.amax(lag_re_ep))            # maximum lag
        rho_ep = np.load('./e_p_yw_rho.npy')        # dim = nr_eofs x max lag
        sig_ep = np.load('./e_p_yw_sigma.npy')      # dime = nr_eofs
        nr_ep = sig_ep.shape[0]
        hist_ep = np.zeros([nr_ep,l_m_ep])          # spinup history
        self._nr_eofs_ep = nr_ep

        for nr_i in range(nr_ep): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_ep[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_ep[nr_i,l_m_ep-1] = (np.dot(hist_ep[nr_i,:lag],rho_ep[nr_i,:lag])
                                        + np.random.normal(0,sig_ep[nr_i]))
                
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
        hist_t2m = np.zeros([nr_t2m,l_m_t2m])       # spinup history
        self._nr_eofs_t2m = nr_t2m

        for nr_i in range(nr_t2m): # Loop over all EOFs
            # Select the lag corresponding to the partial autocorrelation function
            lag = int(lag_re_t2m[nr_i])
            for spin_it in range(lag):
                # Construct the AR(lag) process where the white noise
                # is scaled with 'sig'
                hist_t2m[nr_i,l_m_t2m-1] = (np.dot(hist_t2m[nr_i,:lag],rho_t2m[nr_i,:lag])
                                          + np.random.normal(0,sig_t2m[nr_i]))
                
                # Roll the time series to keep the history: 
                # last item (just computed) becomes the first,
                # first becomes second, etc.
                hist_t2m[nr_i,:] = np.roll(hist_t2m[nr_i,:],1)

        # Store AR data
        nc_out = nc.Dataset(ARdatafile, 'w')
        # dims
        lag_out = nc_out.createDimension('lag_d_ep',l_m_ep)
        eof_out = nc_out.createDimension('eof_d_ep',nr_ep)
        lag_out = nc_out.createDimension('lag_d_t2m',l_m_t2m)
        eof_out = nc_out.createDimension('eof_d_t2m',nr_t2m)
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
        nc_rnd = nc_out.createVariable('rnd_t2m', 'f8', 'eof_d_t2m')
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
        _logger.debug("Clearing model internals")
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
