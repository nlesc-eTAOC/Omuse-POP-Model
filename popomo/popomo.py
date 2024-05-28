import os
import signal
import time

import numpy as np
import psutil as psu
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
            if not os.path.exists(self.run_folder):
                os.mkdir(self.run_folder)

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        # On the first call to advance, initialize POP
        if self.pop is None:
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

    def noise(self):
        """Return last generated noise."""
        return self._noise

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
