from pytams.fmodel import ForwardModel

from popomo.poputils import getPOPinstance
from popomo.poputils import plot_depth

from omuse.units import units


class POPOmuseModel(ForwardModel):
    """A forward model for pyTAMS based on POP-Omuse."""

    def __init__(self, params: dict = None) -> None:
        """Override the template."""
        self._state = None
        self._nml_file = params.get("nml_file", "pop_in")
        self._topo_file = params.get("topo_file", None)
        self._nProc_pop = params.get("nProc_POP", 1)
        self._popDomain = {"lonMin" : params.get("lonmin", -180) | units.deg,
                           "lonMax" : params.get("lonmax",  180) | units.deg,
                           "latMin" : params.get("latmin", -84) | units.deg,
                           "latMax" : params.get("latmax",  84) | units.deg,
                           "Nx" : params.get("Nx", 120),
                           "Ny" : params.get("Ny", 56),
                           "Nz" : params.get("Nz", 12)}

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        p = getPOPinstance(
            nml_file = self._nml_file,
            nworkers = self._nProc_pop,
            topo_file = self._topo_file,
            domain_dict = self._popDomain,
        )
        tnow = p.model_time
        print(tnow)
        plot_depth(p, "depth.png")

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
        return "POPOmuseModel"
