import numpy as np
from omuse.community.pop.interface import POP
from omuse.units import units


def depth_levels(N: int, stretch_factor: float = 1.8) -> np.ndarray:
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


def getPOPinstance(
    nworkers: int = 2,
    Nx: int = 120,
    Ny: int = 56,
    Nz: int = 12,
    nml_file: str = "./pop_in",
    topo_file: str = None,
) -> POP:
    """Return an instance of POP."""
    mode = "{}x{}x{}".format(Nx, Ny, Nz)
    p = POP(
        number_of_workers=nworkers,
        mode=mode,
        namelist_file=nml_file,
        redirection="none",
    )

    # Prepare grid data
    levels = depth_levels(Nz + 1) * 5000 | units.m
    depth_in = np.loadtxt(topo_file, delimiter=",", dtype=int)
    dz = levels[1:] - levels[:-1]

    p.parameters.topography_option = "amuse"
    p.parameters.vert_grid_option = "amuse"
    p.parameters.depth_index = np.flip(depth_in.T, 1)
    p.parameters.vertical_layer_thicknesses = dz

    p.parameters.horiz_grid_option = "amuse"

    return p
