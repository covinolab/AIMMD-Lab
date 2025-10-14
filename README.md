[![pytest](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml/badge.svg)](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml)

# AIMMD
AI for Molecular Mechanism Discovery

AI-enhanced path sampling simulations, both on a workstation and on HPC clusters. Massive parallelization supported!

## Required (main) packages
- `tqdm`
- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `MDAnalysis`
- `mdtraj`
- `psutil`
- GROMACS
From the point of view of the requirements, the code is rather light.

## The structure of a system's folder
- `run.gro`, which can be loaded by either `MDAnalysis` or `mdtraj` to infer the system's topology;
- `params.py`, containing the definitions of states, descriptors, the neural network architecture, and other parameters specific to an AIMMD run;
- `run.mdp` (GROMACS engine only) to run the simulations;
- `randomvelocities.mdp` (GROMACS engine only) to initializing the starting point's velocities;
- `topol.top`, force-field folder (GROMACS engine only);
- `integrator.py` (custom engine for toy systems);
- an `initial` trajectory, to be copied inside the `run`'s folder;
- the actual scripts, which are currently **not a python package yet**, and instead are copied from the `aimmd/core` folder.

## Tests
Two general tests for running AIMMD/analyzing results are currently implemented:
- `test_toy_1d.py`
- `test_retinal.py`
You can adapt the former to toy systems of arbitrary dimensions with a custom-written "integrator" engine, and the latter to molecular dynamics systems with the GROMACS engine. Please use the content of the `toy_1d` and `retinal` folders as templates.
