from popomo import __version__
from popomo.popomo import POPOmuseModel


def test_version():
    """Test the version."""
    assert __version__ == "0.0.0"


def test_initModel():
    """Init an POPOmuseModel model."""
    pop_params = {"domain_dict": {}}
    fmodel = POPOmuseModel(params=pop_params)
    assert fmodel.name() == "POPOmuseModel"


def test_advance3degModel():
    """Advance an POP-Omuse model over a step."""
    domain_amuse = {"topography_file": "./data/KMT_120_56.csv"}
    pop_params = {
        "nml_file": "./data/pop_in",
        "domain_dict": domain_amuse,
        "nProc_POP": 1,
    }
    fmodel = POPOmuseModel(params=pop_params)
    fmodel.advance(0.1, 0.1)


def test_advance1degModel():
    """Advance an POP-Omuse model over a step."""
    domain_amuse = {
        "topography_file": "./data/KMT_360_168.csv",
        "Nx": 360,
        "Ny": 168,
        "Nz": 24,
    }
    pop_params = {
        "nml_file": "./data/pop_in",
        "domain_dict": domain_amuse,
        "nProc_POP": 1,
    }
    fmodel = POPOmuseModel(params=pop_params)
    fmodel.advance(0.1, 0.1)
