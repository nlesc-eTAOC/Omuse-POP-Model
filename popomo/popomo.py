import os
import signal
import time
import shutil
import numpy as np
import psutil as psu
import netCDF4 as nc
import logging
from pathlib import Path
from typing import Any
from omuse.units import units
from pytams.fmodel import ForwardModelBaseClass
import pop_tools

from popomo.poputils import (
    getPOPinstance,
    setCheckPoint,
    disableCheckPoint,
    getLastRestart,
    setRestart,
    setStochForcingAmpl,
    setERA5PCForcing,
    setERA5PCARSpinUpfile,
    setERA5PCNIGInitfile,
    setERA5NIGForcingfile,
)
from popomo.utils import (
    daysincurrentyear,
    monthly_reference,
    remainingdaysincurrentmonth,
    year_month_day,
    random_file_in_list,
)
from popomo.forcing_ERA5 import (
    ERA5NIGForcingGenerator,
    ERA5PCForcingGenerator,
)
from popomo.plotutils import (
    plot_globe,
)

_logger = logging.getLogger(__name__)


class PopomoError(Exception):
    """Exception class for Popomo."""

    pass


class POPOmuseModel(ForwardModelBaseClass):
    """A forward model for pyTAMS based on POP-Omuse.

    This class implements a forward model compatible with pyTAMS
    for the Parallel Ocean Program (POP), a Fortran90 global circulation
    model, through the OMUSE framework.

    POP is driven in lock-step by an outer stochastic loop
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
            seed = int(ioprefix[4:10])
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # Stash away POP-specific parameters
        self._pop_params = params.get("pop", {})

        # Get POP domain parameters
        self._popDomain = self._pop_params.get("domain_dict", None)
        assert self._popDomain is not None

        self._noise_amplitude = self._pop_params.get("noise_amplitude",1.0)
        self._timedep_score = self._pop_params.get("time_dependent_score", False)
        self._amoc_str_target = self._pop_params.get("AMOC_str_target",0.0)
        self._hosing_reference = self._pop_params.get("baseline_hosing", "Sv0p26")

        # We always need the database when running this model
        save_db = params.get("database", {}).get("path", None)
        if not save_db:
            err_msg = "POPOmuseModel for TAMS always needs the on-disk database"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        model = "pop_{}x{}x{}".format(
            self._popDomain.get("Nx", 120),
            self._popDomain.get("Ny", 56),
            self._popDomain.get("Nz", 12),
        )
        self.checkpoint_prefix = f"{str(self._workdir)}/{model}"

        # Handling path is still a bit tricky
        # The base forward model class provide a self._workdir
        # Get the database path from there:
        self._db_path = self._workdir.parents[1]

        # Create the workdir if not existent
        if not self._workdir.exists():
            os.mkdir(self._workdir)

        # In this model, the state is a pointer to a POP restart file
        # Upon initialization, pass a initial solution if one is provided
        # Init can be selected at random in a list of files
        self._state = None
        init_file = self._pop_params.get("init_file", None)
        if init_file:
            assert Path(init_file).exists() is True
            self.set_current_state(init_file)
        init_files_list = self._pop_params.get("init_files_list", None)
        if init_files_list:
            assert Path(init_files_list).exists() is True
            self.set_current_state(random_file_in_list(init_files_list))

        # Forcing method
        # Default is "baseline-frac" which only requires a single random number
        self._forcing_method = self._pop_params.get("forcing_method", "baseline-frac")
        if self._forcing_method == "ERA5-PC-AR":
            # For ERA5 PC-AR based forcing: needs as many random numbers as
            # the number of EOFs. Pass in the model RNG and spinup file.
            era5_data_folder = self._pop_params.get("ERA5-dataloc", None)
            self._era5_gen = ERA5PCForcingGenerator(era5_data_folder, "PC-AR")
            self._era5_gen.set_rng(self._rng)
            era5_spinup_file = f"{self._workdir.as_posix()}/ARSpinUpData.nc"
            self._era5_gen.set_spinup_file(era5_spinup_file)
        elif self._forcing_method == "ERA5-PC-NIG":
            # For ERA5 PC-NIG based forcing: needs as many random numbers as
            # the number of EOFs
            era5_data_folder = self._pop_params.get("ERA5-dataloc", None)
            self._era5_gen = ERA5PCForcingGenerator(era5_data_folder, "PC-NIG")
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
                 a_step: int,
                 a_time : float,
                 a_dt: float,
                 noise: Any) -> float:
        """Advance the model for one stochastic step.

        Args:
            a_step : current trajectory time step
            a_time : starting time
            a_dt : stochastic step size in year
            noise : the noise for the current step

        Returns:
            The actual step size advanced by the model
        """
        # On the first call to advance, initialize POP
        if self.pop is None:
            # Enable multiple attempts at initializing POP
            _attempts = 3
            while _attempts:
                inf_msg = f"Start POP itself: try #{3-_attempts}"
                _logger.info(inf_msg)
                try:
                    # Initialize Omuse-POP object
                    self.pop = getPOPinstance(
                        pop_domain_dict = self._popDomain,
                        pop_options = self._pop_params,
                        root_run_folder = self._workdir,
                    )
                    break
                except Exception as e:
                    _attempts -= 1
                    if _attempts == 0:
                        err_msg = "POP initialization failed repeatedly"
                        _logger.error(err_msg)
                        raise

                    warn_msg = f"Warning: {e} while initializing POP at try #{3-_attempts}"
                    _logger.warning(warn_msg)

                    # Give it a few seconds to try again
                    time.sleep(5)
            _logger.info("Done starting POP itself")

            # Handle noise generator initialization
            if self._forcing_method == "ERA5-PC-AR":
                # ERA5-PC-AR forcing model needs some spin-up
                self._era5_gen.spinup_AR()
                setERA5PCARSpinUpfile(self.pop, self._era5_gen.get_spinup_file())
            if self._forcing_method == "ERA5-PC-NIG":
                # ERA5-PC-NIG forcing model needs a first set of RNG early on.
                # Pass incoming noise to POP in a file
                init_rnd_file = f"{self._workdir.as_posix()}/PCNIG_init_noise.nc"
                self._era5_gen.generate_noise_init_file(noise, init_rnd_file)
                setERA5PCNIGInitfile(self.pop, init_rnd_file)
            elif self._forcing_method == "ERA5-NIG":
                # ERA5-NIG uses a forcing file
                # Formatted with {prefix}_{year} containing all 12 months.
                # only generate a new file if a_step == 0
                # otherwise rely on restart mechanism internals to ensure a forcing
                # data file exists
                if self._step == 0:
                    # Generate a forcing file for the starting year
                    # Can't use POP to get the year yet, so use input param
                    year = self._era5_init_year
                    forcing_file_base = f"{self._workdir.as_posix()}/{self._era5_prefix}_{year:04}"
                    self._era5_gen.generate_forcing_file(forcing_file_base, year)
                    dbg_msg = f"Generating noise file {forcing_file_base}"
                    _logger.debug(dbg_msg)
                forcing_file_base = f"{self._workdir.as_posix()}/{self._era5_prefix}"
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
            # scaled by dt (gaussian white noise) and scaled by self._noise_amplitude
            year_length = daysincurrentyear(date, False)
            stochf_ampl = (
                self._noise_amplitude
                * np.sqrt(dt_d.value_in(units.day) / year_length)
                * noise
            )
        elif (self._forcing_method == "ERA5-PC-AR"):
            # Random numbers for noise generated are passed to POP
            setERA5PCForcing(self.pop, noise)
            stochf_ampl = self._noise_amplitude
        elif (self._forcing_method == "ERA5-PC-NIG"):
            # Random numbers for noise generated are passed to POP
            setERA5PCForcing(self.pop, noise)
            stochf_ampl = self._noise_amplitude
        elif (self._forcing_method == "ERA5-NIG"):
            # Incoming noise is actually handled internally by POP
            # Nothing to do here but set the scaling
            stochf_ampl = self._noise_amplitude

        # All models use a scaling
        setStochForcingAmpl(self.pop, stochf_ampl)

        # Get start and end date
        tstart = self.pop.model_time
        tend = tstart + dt_d
        inf_msg = f"TAMS time: tstart = {a_time},"\
                  f" t_end = {a_time+a_dt}"
        _logger.info(inf_msg)
        inf_msg = f"POP time: tstart = {tstart.value_in(units.day)},"\
                  f" t_end = {tend.value_in(units.day)}"
        _logger.info(inf_msg)

        # Advance POP to the end of the stochastic step
        self.pop.evolve_model(tend)

        #grid = pop_tools.get_grid('POP_gx1v6')
        #sfwf_flux = self.pop.elements.surface_fresh_water_flux.value_in(units.kg / units.m**2 / units.s).transpose()
        #plot_globe(grid, "T", sfwf_flux, "kg/m2/s", f"sfwf_flux_{self._step:04}", "SFWF")
        #sfwf_flux_st = self.pop.elements.surface_freshwater_stoch_flux.value_in(units.kg / units.m**2 / units.s).transpose()
        #plot_globe(grid, "T", sfwf_flux_st, "kg/m2/s", f"sfwf_flux_st_{self._step:04}", "SFWF Stoch.")
        #surf_temp_st = self.pop.elements.surface_temp_stoch.value_in(units.Celsius)
        #plot_globe(self.pop, surf_temp_st, "C", f"surf_temp_st_{self._step:04}", elements=True)

        # Retrieve the actual time for which POP was integrated
        tnow = self.pop.model_time
        actual_dt = tnow - tstart

        # State is handled by a pointer to a check file
        self.set_current_state(getLastRestart(self.pop))

        # TAMS runs in unit year -> convert actual_dt to year
        year_length = daysincurrentyear(date, False)
        return actual_dt.value_in(units.day) / year_length

    def get_current_state(self):
        """Return a string representation of the current POP check file path.

        POP produces absolute path, but it is more convenient
        to store relative path in the database so that we can move the full database around.
        Return the full path if it is not a parent of the self._db_path
        """
        state_path = Path(self._state)
        if state_path.is_relative_to(self._db_path):
            return state_path.relative_to(self._db_path).as_posix()
        else:
            return self._state

    def set_current_state(self, state) -> None:
        """Set the current state of the model.

        i.e. a string representation to the current POP check file path.
        Preprend the workdir if it is not an absolute path.
        """
        if not state:
            self._state = None
        else:
            state_path = Path(state)
            if state_path.is_relative_to(self._db_path):
                self._state = state_path.relative_to(self._db_path).as_posix()
            else:
                self._state = state

    def score(self) -> float:
        """Score function.

        The model only support AMOC strength-based score
        function at this point.
        A monthly reference data is used to de-seasonalize, and
        the target strength value can be specified in the input file.
        """
        # When called, the date is the first of
        # month following the month of interest
        if self.pop:
            date = self.pop.get_model_date()
            amoc_ref = monthly_reference(date, self._hosing_reference)
            cur_amoc = self.pop.get_amoc_strength()
            score = (amoc_ref - cur_amoc) / (amoc_ref - self._amoc_str_target)
            if self._timedep_score:
                weight = np.fmin((100.0 - self._time) / 100, 1.0)
                score = score * weight
            inf_msg = f"Date: {date}, score: {score}, cur : {cur_amoc}, ref: {amoc_ref}"
            _logger.info(inf_msg)
            return score
        return 0.0

    def _make_noise(self) -> Any:
        """Generate a new noise increment."""
        if (self._forcing_method == "baseline-frac"):
            # Single number for scaling of baseline forcing data
            return self._rng.standard_normal(1)
        elif (self._forcing_method == "ERA5-PC-AR"):
            # A set of random numbers, one for each EOF of both E-P and T
            return self._era5_gen.generate_normal_noise()
        elif (self._forcing_method == "ERA5-PC-NIG"):
            return self._era5_gen.generate_nig_noise()
        elif (self._forcing_method == "ERA5-NIG"):
            # Return a file name and month index
            if self.pop:
                date = self.pop.get_model_date()
                year, month, _ =  year_month_day(date)
                # Generate new forcing files for the year
                # if they don't exists already
                forcing_file_base = f"{self._workdir}/{self._era5_prefix}_{year:04}"
                self._era5_gen.generate_forcing_file(forcing_file_base, year)
                return f"{forcing_file_base}_{month}"
            else:
                return "None"

    def check_convergence(self,
                      step: int,
                      time: float,
                      current_score: float,
                      target_score: float) -> bool:
        """Check for convergence of the trajectory.


        Overwrite the default. This enable a time-dependent
        target when the score function is scaled by the
        remaining time.

        Args:
            step: the current step counter
            time: the time of the simulation
            current_score: the current score
            target_score: the target score
        """
        #
        if self._timedep_score:
            target = target_score * (100.0 - time) / 100.0
            return current_score >= target
        else:
            return current_score >= target_score


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
