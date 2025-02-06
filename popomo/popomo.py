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
    getPOPinstance,
    setCheckPoint,
    disableCheckPoint,
    getLastRestart,
    setRestart,
    setStochForcingAmpl,
    setERA5PCARForcing,
    setERA5PCARSpinUpfile,
    setERA5NIGForcingfile,
)
from popomo.utils import (
    daysincurrentyear,
    monthly_reference,
    remainingdaysincurrentmonth,
    year_month_day,
)
from popomo.forcing_ERA5 import (
    ERA5NIGForcingGenerator,
    ERA5PCARForcingGenerator,
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
    """A forward model for pyTAMS based on POP-Omuse.

    This class implements a forward model compatible with pyTAMS
    for the Parallel Ocean Program (POP), a Fortran90 GCM ocean
    model, through the OMUSE framework.

    POP is driven in lock-step with an outer stochastic loop
    with a fixed step length of a month (calendar month), such
    that POP monthly average fields are used to build a TAMS score
    function.

    Stochastic forcing can take various form, but the specific
    version of POP supporting the OMUSE interface and addons
    for TAMS forcing is open-source and available here:
    https://github.com/nlesc-smcm/eSalsa-POP

    In this model:
    - the state is a path to a POP restart file
    - the score is a float in R
    - the noise type depends on the forcing type selected:
        - baseline-frac: float
        - ERA5-NIG: a path to NetCDF file, appended with _MM
        - ERA5-PC-NIG: numpy array for E-P and T@2m EOFs
        - ERA5-PC-AR: numpy array for E-P and T@2m EOFs as AR coefficients
    """
    def _init_model(self,
                    params: dict,
                    ioprefix: str = None) -> None:
        """Override the template.

        Args:
            params : a parameters dictionary
            ioprefix : the name of the trajectory using this model instance
        """
        # Setup the random number generator, with seed if deterministic run
        if params["model"]["deterministic"]:
            seed = int(ioprefix[4:])
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # Stash away POP-specific parameters
        self._pop_params = params.get("pop", {})

        # Get POP domain parameters
        self._popDomain = self._pop_params.get("domain_dict", None)
        assert self._popDomain is not None

        # We always need the database when running this model
        save_db = params.get("database", {}).get("DB_save", False)
        if not save_db:
            err_msg = "POPOmuseModel for TAMS always needs the database"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        # Setup the run folder in the DB
        nameDB = "{}.tdb".format(
            params.get("database", {}).get("DB_prefix", "TAMS")
        )
        model = "pop_{}x{}x{}".format(
            self._popDomain.get("Nx", 120),
            self._popDomain.get("Ny", 56),
            self._popDomain.get("Nz", 12),
        )
        self.run_folder = f"{nameDB}/trajectories/{ioprefix}"
        self.checkpoint_prefix = f"{nameDB}/trajectories/{ioprefix}/{model}"
        if not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)

        # In this model, the state is a pointer to a POP restart file
        # Upon initialization, pass a initial solution if one is provided
        self._state = None
        init_file = self._pop_params.get("init_file", None)
        if init_file:
            assert os.path.exists(init_file) is True
            self._state = init_file

        # Forcing method
        # Default is "baseline-frac" which only requires a single random number
        self._forcing_method = self._pop_params.get("forcing_method", "baseline-frac")
        if self._forcing_method == "ERA5-PC-AR":
            # For ERA5 PC-AR based forcing: needs as many random numbers as
            # the number of EOFs. Pass in the model RNG and spinup file.
            era5_data_folder = self._pop_params.get("ERA5-dataloc", None)
            self._era5_gen = ERA5PCARForcingGenerator(era5_data_folder)
            self._era5_gen.set_rng(self._rng)
            era5_spinup_file = f"{self.run_folder}/ARSpinUpData.nc"
            self._era5_gen.set_spinup_file(era5_spinup_file)
        elif self._forcing_method == "ERA5-PC-NIG":
            # For ERA5 PC-NIG based forcing: needs as many random numbers as
            # the number of EOFs
            self._nr_eofs_ep = 0
            self._nr_eofs_t2m = 0
        elif self._forcing_method == "ERA5-NIG":
            # For ERA5 NIG based, use the generator
            self._era5_prefix = self._pop_params.get("ERA5-basefile", "ERA5_Noise")
            era5_data_folder = self._pop_params.get("ERA5-dataloc", None)
            self._era5_init_year = self._pop_params.get("ERA5-inityear", None)
            self._era5_gen = ERA5NIGForcingGenerator(era5_data_folder)

        # The actual POP object
        # It is initialized only when calling advance for the first time
        self.pop = None


    def _advance(self,
                 step: int,
                 time : float,
                 dt: float,
                 noise: Any,
                 forcingAmpl: float) -> float:
        """Advance the model for one stochastic step.

        Args:
            step : current trajectory time step
            time : starting time
            dt : stochastic step size in year
            noise : the noise for the current step
            forcingAmpl : a noise amplitude factor

        Returns:
            The actual step size advanced by the model
        """
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

            # Handle noise generator initialization
            if self._forcing_method == "ERA5-PC-AR":
                # ERA5-PC-AR forcing model needs some spin-up
                self._era5_gen.spinup_AR()
                setERA5PCARSpinUpfile(self.pop, self._era5_gen.get_spinup_file())
            elif self._forcing_method == "ERA5-PC-NIG":
                # ERA5-PC-NIG
                print("TODO")
            elif self._forcing_method == "ERA5-NIG":
                # ERA5-NIG uses a forcing file
                # Formatted with {prefix}_{year} containing all 12 months.
                # only generate a new file if step == 0
                # otherwise rely on restart mechanism internals to ensure a forcing
                # data file exists
                if self._step == 0:
                    # Generate a forcing file for the starting year
                    # Can't use POP to get the year yet, so use input param
                    year = self._era5_init_year
                    forcing_file_base = f"{self.run_folder}/{self._era5_prefix}_{year:04}"
                    self._era5_gen.generate_forcing_file(forcing_file_base, year)
                    dbg_msg = f"Generating noise file {forcing_file_base}"
                    _logger.debug(dbg_msg)
                forcing_file_base = f"{self.run_folder}/{self._era5_prefix}"
                setERA5NIGForcingfile(self.pop, forcing_file_base)

            # Control POP checkpointing
            if self.checkpoint_prefix:
                setCheckPoint(self.pop, 12, self.checkpoint_prefix)
            else:
                disableCheckPoint(self.pop)

            # Set initial restart file (i.e. initial state)
            if self._state is not None:
                setRestart(self.pop, self._state)

            dbg_msg = f"Start advancing with init. state {self._state}"
            _logger.debug(dbg_msg)

        # Time stepping is in year, with a month long step
        # so exact length varies. Get the current step length in days.
        date = self.pop.get_model_date()
        days_left_in_month = remainingdaysincurrentmonth(date)
        dt_d =  days_left_in_month * 1 | units.day

        # Handle incoming noise depending on the selected type
        stochf_ampl = 0.0
        if (self._forcing_method == "baseline-frac"):
            # Amplitude is drawn in a normal distribution
            # scaled by dt (gaussian white noise) and scaled by forcingAmpl
            year_length = daysincurrentyear(date, False)
            stochf_ampl = (
                forcingAmpl
                * np.sqrt(dt_d.value_in(units.day) / year_length)
                * noise
            )
        elif (self._forcing_method == "ERA5-PC-AR"):
            # Random numbers for noise generated are passed to POP
            setERA5PCARForcing(self.pop, noise)
            stochf_ampl = forcingAmpl
        elif (self._forcing_method == "ERA5-PC-NIG"):
            # Amplitude is set to user-provided forcingAmpl
            # Local data read from file provided in pop_in
            stochf_ampl = forcingAmpl
        elif (self._forcing_method == "ERA5-NIG"):
            # Incoming noise is actually handled internally by POP
            # Nothing to do here but set the scaling
            stochf_ampl = forcingAmpl

        # All models use a scaling
        setStochForcingAmpl(self.pop, stochf_ampl)

        #sfwf_flux = self.pop.elements.surface_fresh_water_flux.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_flux, "kg/m2/s", "sfwf_flux", elements=True)
        #sfwf_flux_st = self.pop.elements.surface_freshwater_stoch_flux.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_flux_st, "kg/m2/s", "sfwf_flux_st", elements=True)
        #sfwf_precip = self.pop.element_forcings.sfwf_precip.value_in(units.kg / units.m**2 / units.s)
        #plot_globe_old(self.pop, sfwf_precip, "kg/m2/s", "sfwf_precip", elements=True)
        #exit()

        # Get start and end date
        tstart = self.pop.model_time
        tend = tstart + dt_d
        inf_msg = f"Start stoch step: tstart = {tstart.value_in(units.day)},"\
                  f" t_end = {tend.value_in(units.day)}"
        _logger.info(inf_msg)

        # Advance POP to the end of the stochastic step
        self.pop.evolve_model(tend)

        # Retrieve the actual time for which POP was integrated
        tnow = self.pop.model_time
        actual_dt = tnow - tstart

        # State is handled by a pointer to a check file
        self._state = getLastRestart(self.pop)

        # TAMS runs in unit year -> convert actual_dt to year
        year_length = daysincurrentyear(date, False)
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
        elif (self._forcing_method == "ERA5-PC-AR"):
            # A set of random numbers, one for each EOF of both E-P and T
            return self._era5_gen.generate_normal_noise()
        elif (self._forcing_method == "ERA5-PC-NIG"):
            return 0.0
        elif (self._forcing_method == "ERA5-NIG"):
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

    @classmethod
    def name(self) -> str:
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

    def clear(self) -> None:
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
