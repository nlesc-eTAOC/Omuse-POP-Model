import os
import shutil
import numpy as np
import logging
from omuse.community.pop.interface import POP
from omuse.units import units

_logger = logging.getLogger(__name__)


class POPUtilsError(Exception):
    """Exception class for POPUtils."""

    pass


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


def getPOPinstance(pop_domain_dict: dict = None,
                   pop_options: dict = None,
                   root_run_folder: str = None) -> POP:
    """Return an instance of POP given a parameter dict."""
    assert pop_domain_dict is not None

    # Mode and worker name are based on resolution
    mode = "{}x{}x{}".format(
        pop_domain_dict["Nx"], pop_domain_dict["Ny"], pop_domain_dict["Nz"]
    )

    # Set POP output redirection
    redirect = pop_options.get("redirect", "none")
    redirect_file = "pop_log.txt"
    if root_run_folder:
        redirect_file = "{}/pop_log.txt".format(root_run_folder)

    # Copy the nml option file to the run folder
    orig_nml_file = pop_options.get("nml_file", "pop_in")
    target_nml_file = "{}/{}".format(
        root_run_folder, os.path.basename(orig_nml_file)
    )
    shutil.copy(orig_nml_file, target_nml_file)

    inf_msg = f"Initiating pop_{mode} with input nml {target_nml_file}"
    _logger.info(inf_msg)

    # Instantiate POP
    p = POP(
        number_of_workers=pop_options.get("nProc", 1),
        mode=mode,
        namelist_file=target_nml_file,
        redirection=redirect,
        redirect_file=redirect_file,
        channel_type=pop_options.get("channel", "mpi"),
    )

    # Grid option: either amuse or pop_files
    grid_option = pop_domain_dict.get("grid_option", "amuse")

    if grid_option == "amuse":
        # Amuse generate a cartesian lat/long grid
        levels = depth_levels(pop_domain_dict["Nz"] + 1) * 5000 | units.m
        depth_in = np.loadtxt(
            pop_domain_dict["topography_file"], delimiter=",", dtype=int
        )
        dz = levels[1:] - levels[:-1]

        p.parameters.topography_option = "amuse"
        p.parameters.vert_grid_option = "amuse"
        p.parameters.depth_index = np.flip(depth_in.T, 1)
        p.parameters.vertical_layer_thicknesses = dz

        p.parameters.horiz_grid_option = "amuse"
        p.parameters.lonmin = pop_domain_dict.get("lonmin", -180) | units.deg
        p.parameters.lonmax = pop_domain_dict.get("lonmax", 180) | units.deg
        p.parameters.latmin = pop_domain_dict.get("latmin", -84) | units.deg
        p.parameters.latmax = pop_domain_dict.get("latmax", 84) | units.deg

    elif grid_option == "pop_files":
        # Standard pop file use fancy grid
        p.parameters.topography_option = "file"
        p.parameters.topography_file = pop_domain_dict.get(
            "topography_file", None
        )
        p.parameters.vert_grid_option = "file"
        p.parameters.vert_grid_file = pop_domain_dict.get(
            "vert_grid_file", None
        )
        p.parameters.horiz_grid_option = "file"
        p.parameters.horiz_grid_file = pop_domain_dict.get(
            "horiz_grid_file", None
        )
    else:
        err_msg = f"Unknown grid_option {grid_option}. Either 'amuse' or 'pop_files'"
        raise POPUtilsError(err_msg)

    # Set the POP average output prefix
    p.parameters.tavg_file = root_run_folder + "/tavg"


    # Set the diag and transport output prefix
    p.parameters.diagout_file = root_run_folder + "/diag"
    p.parameters.trandiagout_file = root_run_folder + "/tran"

    return p


def setStochForcingAmpl(p: POP,
                        a_ampl: float) -> None:
    """Set the stochastic forcing amplitude coefficient.

    Args:
        p : a POP instance
        a_ampl : a float passed to POP for scaling noise
    """
    inf_msg = f"Stoch. amplitude factor is: {a_ampl}"
    _logger.info(inf_msg)
    p.parameters.stoch_ampl = a_ampl


def setERA5PCForcing(p: POP,
                     noise) -> None:
    """Set the stochastic noise for ERA5-PC-* forcing.

    Args:
        p : a POP instance
        noise : an array of random numbers
    """
    inf_msg = f"Setting ERA5PC random {len(noise)} numbers"
    _logger.info(inf_msg)
    p.set_era5pc_noise(noise)


def setERA5PCARSpinUpfile(p: POP,
                          ERA5PCARSpinUpfile : str) -> None:
    """Set the AR spinup data file path.

    Args:
        p : a POP instance
        ERA5PCARSpinUpfile : the spinup AR coefficients file
    """
    p.set_era5pcar_spinupfile(ERA5PCARSpinUpfile)


def setERA5PCNIGInitfile(p: POP,
                         ERA5PCNIGinitfile : str) -> None:
    """Set the PC-NIG initial noise data file path.

    Args:
        p : a POP instance
        ERA5NIGARSpinUpfile : the spinup AR coefficients file
    """
    p.set_era5pcnig_initfile(ERA5PCNIGinitfile)


def setERA5NIGForcingfile(p: POP,
                          ERA5NIGbasefile : str) -> None:
    """Set the ERA5-NIG file base path.

    Args:
        p : a POP instance
        ERA5NIGbasefile : a string containing the root of NIG forcing files
    """
    p.set_era5nig_forcing_file(ERA5NIGbasefile)


def setCheckPoint(p: POP,
                  freq: int,
                  chkprefix: str) -> None:
    """Set the POP instance checkpointing options.

    Args:
        p : a POP instance
        freq : the restart frequency option (in month)
        chkprefix : the restart file prefix (incl. path)
    """
    # Set restart such that a single restart is called at
    # the end of the advance function.
    p.parameters.restart_option = "nmonth"
    p.parameters.restart_freq_option = freq
    p.parameters.restart_file = chkprefix


def disableCheckPoint(p: POP) -> None:
    """Disable restart file IO in POP.

    Args:
        p : a POP instance
    """
    p.parameters.restart_option = "never"


def setRestart(p: POP,
               rstFile: str) -> None:
    """Set POP restart file.

    Args:
        p : a POP instance
        rstFile : a restart file path
    """
    if not os.path.exists(rstFile):
        err_msg = f"State file {rstFile} does not exists"
        _logger.error(err_msg)
        raise ValueError(err_msg)
    p.parameters.ts_file = rstFile
    p.parameters.ts_file_format = "bin"


def getLastRestart(p: POP) -> str:
    """Get the last generated restart file from POP.

    Args:
        p : a POP instance

    Returns:
        The last restart file generated by POP
    """
    return p.get_last_restart()
