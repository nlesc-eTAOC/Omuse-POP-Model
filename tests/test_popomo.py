from popomo import __version__
from popomo.popomo import POPOmuseModel


def test_version():
    """Test the version."""
    assert __version__ == "0.0.0"


def test_initModel():
    """Init an POPOmuseModel model."""
    pop_params = {}
    fmodel = POPOmuseModel(params=pop_params)
    assert fmodel.name() == "POPOmuseModel"


def test_advance3degModel():
    """Advance an POP-Omuse model over a step."""
    pop_params = {
        "nml_file": "./data/pop_in",
        "topo_file": "./data/KMT_120_56.csv",
        "nProc_POP": 1,
    }
    fmodel = POPOmuseModel(params=pop_params)
    fmodel.advance(0.1, 0.1)


def test_advance1degModel():
    """Advance an POP-Omuse model over a step."""
    pop_params = {
        "nml_file": "./data/pop_in",
        "topo_file": "./data/KMT_360_168.csv",
        "nProc_POP": 1,
        "Nx": 360,
        "Ny": 168,
        "Nz": 24,
    }
    fmodel = POPOmuseModel(params=pop_params)
    fmodel.advance(0.1, 0.1)
