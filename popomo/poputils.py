import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from omuse.community.pop.interface import POP
from omuse.units import units

class POPUtilsError(Exception):
    """Exception class for POPUtils."""

    pass


def plot_globe(p, value, unit, name, elements=False):
    """Plot value of the globe."""
    matplotlib.use('agg')
    mask = p.elements.depth.value_in(units.km) == 0
    value = np.ma.array(value, mask=mask)

    if elements:
        x = p.elements3d.lon[:, 0, 0].value_in(units.deg)
        y = p.elements3d.lat[0, :, 0].value_in(units.deg)
    else:
        x = p.nodes3d.lon[:, 0, 0].value_in(units.deg)
        y = p.nodes3d.lat[0, :, 0].value_in(units.deg)

    for i in range(len(x)):
        if x[i] > 180:
            x[i] = x[i] - 360

    i = np.argsort(x)
    i = np.insert(i, 0, i[-1])

    value = value[i, :]
    x = x[i]
    x[0] -= 360

    pyplot.figure(figsize=(7, 3.5))

    pyplot.contourf(x, y, value.T)

    pyplot.xticks(
        [-180, -120, -60, 0, 60, 120, 180],
        ["180°W", "120°W", "60°W", "0°", "60°E", "120°E", "180°E"],
    )
    pyplot.yticks(
        [-60, -30, 0, 30, 60], ["60°S", "30°S", "0°", "30°N", "60°N"]
    )
    pyplot.colorbar(label=unit)
    pyplot.ylim(y[1], y[-2])
    pyplot.savefig(name)
    pyplot.close()


def plot_depth(p, name="depth.png"):
    """Plot the ocean depth level."""
    h = p.nodes.depth.value_in(units.km)
    plot_globe(p, h, "km", name)


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
    nworkers: int = 1,
    nml_file: str = "./pop_in",
    domain_dict: dict = None,
) -> POP:
    """Return an instance of POP."""
    assert domain_dict is not None

    # Mode is based on resolution
    mode = "{}x{}x{}".format(
        domain_dict["Nx"], domain_dict["Ny"], domain_dict["Nz"]
    )
    p = POP(
        number_of_workers=nworkers,
        mode=mode,
        namelist_file=nml_file,
        redirection="none",
    )

    # Grid option: either amuse or pop_files
    grid_option = domain_dict.get("grid_option", "amuse")

    if (grid_option == "amuse"):
        # Prepare grid data
        levels = depth_levels(domain_dict["Nz"] + 1) * 5000 | units.m
        depth_in = np.loadtxt(domain_dict["topography_file"], delimiter=",", dtype=int)
        dz = levels[1:] - levels[:-1]

        p.parameters.topography_option = "amuse"
        p.parameters.vert_grid_option = "amuse"
        p.parameters.depth_index = np.flip(depth_in.T, 1)
        p.parameters.vertical_layer_thicknesses = dz

        p.parameters.horiz_grid_option = "amuse"
        p.parameters.lonmin = domain_dict.get("lonmin",-180) | units.deg
        p.parameters.lonmax = domain_dict.get("lonmax",180) | units.deg
        p.parameters.latmin = domain_dict.get("latmin",-84) | units.deg
        p.parameters.latmax = domain_dict.get("latmax",84) | units.deg

    elif (grid_option == "pop_files"):
        p.parameters.topography_option = "file"
        p.parameters.topography_file = domain_dict.get("topography_file", None)
        p.parameters.vert_grid_option = "file"
        p.parameters.vert_grid_file = domain_dict.get("vert_grid_file", None)
        p.parameters.horiz_grid_option = "file"
        p.parameters.horiz_grid_file = domain_dict.get("horiz_grid_file", None)
    else:
        raise POPUtilsError(
                "Unknown grid_option {}. Can only be 'amuse' or 'pop_files'".format(grid_option)
        )

    return p


def setCheckPoint(p: POP, freq: int, chkprefix: str):
    """Set the POP instance checkpointing options."""
    # Set restart such that a single restart is called at
    # the end of the advance function.
    p.parameters.restart_option = "nhour"
    p.parameters.restart_freq_option = freq
    p.parameters.restart_file = chkprefix


def setRestart(p: POP, rstFile: str):
    p.parameters.ts_file = rstFile


def getLastRestart(p: POP) -> str:
    return p.get_last_restart()
