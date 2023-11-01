from omuse.community.pop.interface import POP 
from omuse.units import units
import numpy as np

def depth_levels(N:int, stretch_factor:float = 1.8) -> np.ndarray:
    """Set linear or streched depth level grid on [0:1].

    Args:
        N: number of levels
        stretch_factor: a strech factor for tanh stretching

    Return:
        Numpy 1D ndarray with levels
    """
    z = np.arange(N) / (1.0 * (N - 1))
    if stretch_factor == 0:
        return z
    else:
        return 1 - np.tanh(stretch_factor * (1 - z)) / np.tanh(stretch_factor)

def getPOPinstance(nworkers: int = 2,
                   Nx : int = 120,
                   Ny : int = 56,
                   Nz : int = 12,
                   nml_file : str = "./pop_in") -> POP:
    """Return an instance of POP."""
    mode = "{}x{}x{}".format(Nx,Ny,Nz)
    p = POP(number_of_workers = nworkers,
            mode = mode,
            nml_file = nml_file,
            redirection = "none")

    # Prepare grid data
    levels = depth_levels(Nz+1) * 5000 | units.m
    depth = np.zeros((Nx, Ny), dtype=int)
    depth_in = np.loadtxt("./KMT_{}_{}.csv".format(Nx,Ny),
                          delimiter=",",
                          dtype=int)
    dz = levels[1:] - levels[:-1]

    p.parameters.topography_option = "amuse"
    p.parameters.depth_index = np.flip(depth_in.T,1)
    p.parameters.horiz_grid_option = "amuse"
