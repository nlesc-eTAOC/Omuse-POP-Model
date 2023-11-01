from tamspop import __version__
from tamspop.tamspop import OmusePOPModel

def test_version():
    """Test the version."""
    assert __version__ == "0.0.0"


def test_initModel():
    """Init an Omuse-POP model."""
    fmodel = OmusePOPModel()
    assert fmodel.name() == "OmusePOPModel"

def test_advanceModel():
    """Advance an Omuse-POP model over a step."""
    fmodel = OmusePOPModel()
    fmodel.advance(0.1,0.1)
