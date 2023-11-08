import numpy as np
from matplotlib import pyplot
from omuse.community.pop.interface import POP
from omuse.units import units

def plot_globe(p, value, unit, name, elements=False):
    """Plot value of the globe."""
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

    pyplot.xticks([-180, -120, -60, 0, 60, 120, 180],
                  ['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])
    pyplot.yticks([-60, -30, 0, 30, 60],
                  ['60°S', '30°S', '0°', '30°N', '60°N'])
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
    topo_file: str = None,
    domain_dict: dict = None,
) -> POP:
    """Return an instance of POP."""
    assert domain_dict is not None
    mode = "{}x{}x{}".format(domain_dict["Nx"], domain_dict["Ny"], domain_dict["Nz"])
    p = POP(
        number_of_workers=nworkers,
        mode=mode,
        namelist_file=nml_file,
        redirection="none",
    )

    # Prepare grid data
    levels = depth_levels(domain_dict["Nz"] + 1) * 5000 | units.m
    depth_in = np.loadtxt(topo_file, delimiter=",", dtype=int)
    dz = levels[1:] - levels[:-1]

    p.parameters.topography_option = "amuse"
    p.parameters.vert_grid_option = "amuse"
    p.parameters.depth_index = np.flip(depth_in.T, 1)
    p.parameters.vertical_layer_thicknesses = dz

    p.parameters.horiz_grid_option = "amuse"
    p.parameters.lonmin = domain_dict["lonMin"]
    p.parameters.lonmax = domain_dict["lonMax"]
    p.parameters.latmin = domain_dict["latMin"]
    p.parameters.latmax = domain_dict["latMax"]

    return p
