# Use oceanDB data
import oceansdb as odb
import argparse
import csv
import numpy
from matplotlib import pyplot
from itertools import product


def depth_levels(N, stretch_factor=1.8):
    """Define a tanh stretched between 0 and 1"""
    z = numpy.arange(N) / (1.0 * (N - 1))
    if stretch_factor == 0:
        return z
    else:
        return 1 - numpy.tanh(stretch_factor * (1 - z)) / numpy.tanh(stretch_factor)


def getGrid(xdeg, ydeg):
    lX = len(xdeg)
    lY = len(ydeg)
    x = numpy.zeros((lX, lY))
    y = numpy.zeros((lX, lY))
    for i, j in product(range(lX), range(lY)):
        x[i, j] = xdeg[i]
        y[i, j] = ydeg[j]

    return x, y


def plot_depth(x, y, depth):
    aspect_ratio = (y[0, -1] - y[0, 0]) / (x[-1, 0] - x[0, 0])
    pyplot.figure(figsize=(7, 7 * aspect_ratio))
    pyplot.contourf(x, y, depth)
    pyplot.colorbar(label="Topography [m]")
    pyplot.savefig("Depth_{}_{}.png".format(x.shape[0], x.shape[1]))
    pyplot.close()


def plot_KMT(x, y, kmt):
    aspect_ratio = (y[0, -1] - y[0, 0]) / (x[-1, 0] - x[0, 0])
    pyplot.figure(figsize=(7, 7 * aspect_ratio))
    pyplot.contourf(x, y, kmt)
    pyplot.colorbar(label="KMT field [-]")
    pyplot.savefig("KMT_{}_{}.png".format(x.shape[0], x.shape[1]))
    pyplot.close()


if __name__ == "__main__":
    # ---------------
    # Parse input
    # ---------------
    parser = argparse.ArgumentParser(
        description="A tool to generate KMT field for Omuse-POP topography from ocean's DB data. \nDefault parameters give a 3 deg grid over the entire globe."
    )
    parser.add_argument(
        "--Nx",
        required=False,
        type=int,
        dest="Nx",
        default=120,
        help="Number of longitudinal grid cells",
    )
    parser.add_argument(
        "--Ny",
        required=False,
        type=int,
        dest="Ny",
        default=56,
        help="Number of latitudinal grid cells",
    )
    parser.add_argument(
        "--Nz",
        required=False,
        type=int,
        dest="Nz",
        default=12,
        help="Number depth layers",
    )
    parser.add_argument(
        "--lgmin",
        required=False,
        type=float,
        dest="lonmin",
        default=-180,
        help="west boundary of domain (deg)",
    )
    parser.add_argument(
        "--lgmax",
        required=False,
        type=float,
        dest="lonmax",
        default=180,
        help="east boundary of domain (deg)",
    )
    parser.add_argument(
        "--ltmin",
        required=False,
        type=float,
        dest="latmin",
        default=-84,
        help="south boundary of domain (deg)",
    )
    parser.add_argument(
        "--ltmax",
        required=False,
        type=float,
        dest="latmax",
        default=84,
        help="north boundary of domain (deg)",
    )
    parser.add_argument(
        "--plot",
        required=False,
        type=int,
        dest="plot_topo",
        default=0,
        help="plot the topography",
    )
    args = parser.parse_args()

    # Move to easier to handle variables
    Nx = args.Nx
    Ny = args.Ny
    Nz = args.Nz
    lonmin = args.lonmin
    lonmax = args.lonmax
    latmin = args.latmin
    latmax = args.latmax

    # Generate the depth levels
    levels = depth_levels(Nz + 1) * 5000

    # Get amuse-type grid positions
    dlon = (lonmax - lonmin) / float(Nx)
    dlat = (latmax - latmin) / float(Ny)
    xdeg = []
    ydeg = []
    for i in range(Nx):
        xdeg.append(lonmin + (float(i) + 0.5) * dlon)
    for j in range(Ny):
        ydeg.append(latmin + (float(j) + 0.5) * dlat)

    # Get the topography out of ocean's DB
    # interpolating on the grid
    with odb.ETOPO() as depth_db:
        real_depth = depth_db["topography"].extract(lat=ydeg, lon=xdeg)["height"]

    # Compute the KMT field
    kmt = numpy.zeros((Nx, Ny), dtype=int)

    for i, j in product(range(Nx), range(Ny)):
        for k in range(Nz):
            if (
                real_depth[j, i] < -1 * levels[k]
                and real_depth[j, i] >= -1 * levels[k + 1]
            ):
                kmt[i, j] = k + 1
                break
        if real_depth[j, i] < -1 * levels[Nz]:
            kmt[i, j] = Nz

    with open("./KMT_{}_{}.csv".format(Nx, Ny), "w") as myfile:
        wr = csv.writer(myfile)
        wr.writerows(numpy.flip(kmt.T, 0))

    if args.plot_topo:
        x, y = getGrid(xdeg, ydeg)
        plot_depth(x, y, real_depth.T)
        plot_KMT(x, y, kmt)
