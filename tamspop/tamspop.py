import time
import numpy as np
from pytams.fmodel import ForwardModel
from tamspop.poputils import getPOPinstance
from omuse.community.pop.interface import POP
from omuse.units import units


class OmusePOPModel(ForwardModel):
    """A forward model for pyTAMS based on Omuse-POP.
    """

    def __init__(self) -> None:
        """Override the template."""
        self._state = None

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        p = getPOPinstance()

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state) -> None:
        """Override the template."""
        self._state = state

    def score(self) -> float:
        """Override the template."""
        return 0.0

    def name(self):
        """Return the model name."""
        return "OmusePOPModel"
