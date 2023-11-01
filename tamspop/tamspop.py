from pytams.fmodel import ForwardModel

from tamspop.poputils import getPOPinstance


class OmusePOPModel(ForwardModel):
    """A forward model for pyTAMS based on Omuse-POP."""

    def __init__(self, params: dict = None) -> None:
        """Override the template."""
        self._state = None
        self._nml_file = params.get("nml_file", "pop_in")
        self._topo_file = params.get("topo_file", None)

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        p = getPOPinstance(nml_file=self._nml_file, topo_file=self._topo_file)
        tnow = p.model_time
        print(tnow)

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
