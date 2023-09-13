import argparse

import numpy as np
from netCDF4 import Dataset


def compNCDF(NetCDF1, NetCDF2, eps):
    # Open NetCDF files
    Dataset1 = Dataset(NetCDF1, "r")
    Dataset2 = Dataset(NetCDF2, "r")

    # Check variables consistency
    VarsNetCDF1 = [v.strip() for v in list(Dataset1.variables)]
    VarsNetCDF2 = [v.strip() for v in list(Dataset2.variables)]
    VarsNetCDF1.sort()
    VarsNetCDF2.sort()

    assert VarsNetCDF1 == VarsNetCDF2

    # Compute and print l2norm error for each variables
    filesAgree = True

    print(
        "{:12s}    {:>10s}   {:>10s}".format(
            "Variable", "Abs. Diff", "Rel. Diff [%]"
        )
    )
    for Var in VarsNetCDF1:
        NetCDF_array1 = Dataset1[Var][:]
        NetCDF_array2 = Dataset2[Var][:]

        NetCDF_array1 = np.array(NetCDF_array1, dtype=np.float32)
        NetCDF_array2 = np.array(NetCDF_array2, dtype=np.float32)
        diff = NetCDF_array1 - NetCDF_array2

        absmax = np.abs(max(NetCDF_array1.min(), NetCDF_array1.max(), key=abs))
        l2norm = np.linalg.norm(diff)

        if absmax > 0.0:
            print(
                "{:>12s}:   {:>7.4e}   {:>7.4e}".format(
                    Var, l2norm, l2norm / absmax * 0.01
                )
            )
        else:
            print("{:>12s}:   {:>7.4e}   {:>7.4e}".format(Var, 0.0, 0.0))

        filesAgree = filesAgree and l2norm < eps

    # Clean-up
    Dataset1.close()
    Dataset2.close()

    # Handle comparison result
    if filesAgree:
        print("NetCDF files agree !")
    else:
        raise Exception("NetCDF files differ !")


if __name__ == "__main__":
    # ---------------
    # Parse
    # ---------------
    parser = argparse.ArgumentParser(
        description="A plain NetCDF files comparison tool."
    )
    parser.add_argument(
        "-f1",
        "--file1",
        required=True,
        type=str,
        dest="file1",
        help="first netcdf file",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        required=True,
        type=str,
        dest="file2",
        help="second netcdf file",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        required=False,
        type=str,
        dest="epsilon",
        default=1e-12,
        help="Epsilon maximum allowable difference",
    )
    args = parser.parse_args()

    compNCDF(args.file1, args.file2, args.epsilon)
