from tamspop import __version__
from tamspop.tamspop import OmusePOPModel


def test_version():
    """Test the version."""
    assert __version__ == "0.0.0"


def test_initModel():
    """Init an Omuse-POP model."""
    pop_params = {}
    fmodel = OmusePOPModel(params=pop_params)
    assert fmodel.name() == "OmusePOPModel"


def test_advanceModel():
    """Advance an Omuse-POP model over a step."""
    pop_params = {
        "nml_file": "./data/pop_in",
        "topo_file": "./data/KMT_120_56.csv",
    }
    fmodel = OmusePOPModel(params=pop_params)
    fmodel.advance(0.1, 0.1)
