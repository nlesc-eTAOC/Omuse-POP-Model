import os

import numpy as np
from omuse.units import units
from pytams.fmodel import ForwardModel

from popomo.poputils import (
    getLastRestart,
    getPOPinstance,
    setCheckPoint,
    setRestart,
)


class POPOmuseModel(ForwardModel):
    """A forward model for pyTAMS based on POP-Omuse."""

    def __init__(self, params: dict = None, ioprefix: str = None) -> None:
        """Override the template."""
        # In this model, the state is a pointer to a POP restart file
        self._state = None
        self._nml_file = params.get("nml_file", "pop_in")
        self._nProc_pop = params.get("nProc_POP", 1)
        self._popDomain = params.get("domain_dict", None)
        assert self._popDomain is not None

        self.pop = None
        self.checkpoint_prefix = None
        if params.get("DB_save", False):
            nameDB = "{}.tdb".format(params.get("DB_prefix", "TAMS"))
            model = "pop_{}x{}x{}".format(
                self._popDomain.get("Nx", 120),
                self._popDomain.get("Ny", 56),
                self._popDomain.get("Nz", 12),
            )
            checkpoint_path = "{}/trajectories/{}".format(nameDB, ioprefix)
            self.checkpoint_prefix = "{}/trajectories/{}/{}".format(
                nameDB, ioprefix, model
            )
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        if self.pop is None:
            self.pop = getPOPinstance(
                nml_file=self._nml_file,
                nworkers=self._nProc_pop,
                domain_dict=self._popDomain,
            )
            if self.checkpoint_prefix:
                setCheckPoint(self.pop, int(dt), self.checkpoint_prefix)
            if self._state is not None:
                setRestart(self.pop, self._state)
            print("advance with init. State is {}".format(self._state))

        tnow = self.pop.model_time
        print(tnow.value_in(units.hour))
        dt_s = dt * 1 | units.hour
        tend = tnow + dt_s
        self.pop.evolve_model(tend)

        # Retrieve the actual dt performed by POP
        actual_dt = self.pop.model_time - tnow

        # State is handled by a pointer to a check file
        self._state = getLastRestart(self.pop)

        return actual_dt.value_in(units.hour)

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state) -> None:
        """Override the template."""
        self._state = state

    def score(self) -> float:
        """Override the template."""
        tnow = self.pop.model_time
        return tnow.value_in(units.hour) * np.random.rand(1).item() / 1000.0

    @classmethod
    def name(self):
        """Return the model name."""
        return "POPOmuseModel"

    def clear(self):
        """Clear any model internals."""
        self.pop.cleanup_code()
        self.pop.clear()
        del self.pop
        self.pop = None
        self._state = None
