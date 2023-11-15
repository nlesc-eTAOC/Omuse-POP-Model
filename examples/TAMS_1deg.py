from pytams.tams import TAMS

from popomo.popomo import POPOmuseModel

if __name__ == "__main__":
    params = {
        "nml_file": "../data/pop_in",
        "topo_file": "../data/KMT_240_110.csv",
        "nProc_POP": 1,
        "Nx": 240,
        "Ny": 110,
        "Nz": 12,
        "traj.end_time": 36.0,
        "traj.step_size": 6.0,
        "nTrajectories": 8,
        "nSplitIter": 4,
        "Verbose": True,
        "nProc": 1,
        "DB_save": True,
        "DB_prefix": "Tams-POP",
    }

    # Test TAMS
    tams = TAMS(fmodel_t=POPOmuseModel, parameters=params)
    transition_proba = tams.compute_probability()
