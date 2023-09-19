from opsicl import __version__
from opsicl.popsicl import popsicl


def test_version():
    """Test the version."""
    assert __version__ == "0.0.1"


def test_popinit():
    """Test pop instance"""
    mode = "96x120x12"
    cola = popsicl(nworkers=1, mode=mode, nml_file="./tests/poptest_in")
