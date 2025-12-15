# POP-Omuse MOdel - POPOMO

## Overview

Within the [eTAOC](https://research-software-directory.org/projects/etaoc) project, our objective is
to use rare events techniques on high-dimensional system to evaluate the probability of collapse of
the AMOC within a finite time-window.

The [pyTAMS](https://github.com/nlesc-eTAOC/pyTAMS) framework, also developed within eTAOC,
provides a generic implementation of the trajectory-adaptive multilevel splitting (TAMS)
technique by defining an abstract model class. The present
repository is a concrete model class implementation for running the parallel ocean program
[POP](https://www.cesm.ucar.edu/models/pop) and relying on the [OMUSE](https://research-software-directory.org/software/omuse)
infrastructure to link between POP Fortran code base and Python.

## Installation

### Getting the sources

To install *popomo*, first use git to get the sources:

```console
git clone git@github.com:nlesc-eTAOC/Omuse-POP-Model.git
```

and submodules to get the direct dependencies:

```console
cd Omuse-POP-Model
git submodule init
git submodule update
```

You now have the OMUSE and [AMUSE](https://research-software-directory.org/software/amuse) packages available locally.

### Setting up your environment




## Acknowledgements

The development of *popomo* was supported by the Netherlands eScience Center
in collaboration with the Institute for Marine and Atmospheric research Utrecht [IMAU](https://www.uu.nl/onderzoek/imau).
