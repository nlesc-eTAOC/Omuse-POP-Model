from pytams.tams import TAMS

from popomo.popomo import POPOmuseModel

if __name__ == "__main__":
    domain_amuse = {
        "grid_option" : "amuse",
        "topography_file" : "../data/KMT_320_384.csv",
        "Nx": 320,
        "Ny": 384,
        "Nz": 40,
    }
    domain_popfiles = {
        "grid_option" : "pop_files",
        "topography_file" : "./x1_data/grid/topography_20010702.ieeei4",
        "vert_grid_file" : "./x1_data/grid/in_depths.dat",
        "horiz_grid_file" : "./x1_data/grid/horiz_grid_20010402.ieeer8",
        "Nx": 320,
        "Ny": 384,
        "Nz": 40,
    }
    params = {
        "nml_file": "./x1_data/pop_in",
        "domain_dict" : domain_popfiles,
        "nProc_POP": 4,
        "traj.end_time": 12.0,
        "traj.step_size": 0.5,
        "nTrajectories": 1,
        "nSplitIter": 4,
        "Verbose": True,
        "dask.nworker": 1,
        "DB_save": False,
        "DB_prefix": "Tams-POP",
    }

    # Test TAMS
    tams = TAMS(fmodel_t=POPOmuseModel, parameters=params)
    transition_proba = tams.compute_probability()
