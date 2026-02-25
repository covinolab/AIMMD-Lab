[![pytest](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml/badge.svg)](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml)

# AIMMD
AI for Molecular Mechanism Discovery

AI-enhanced path sampling simulations, both on a workstation and on HPC clusters. Massive parallelization supported!

## Required (main) packages
- `dill`
- `tqdm`
- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `MDAnalysis`
- `mdtraj`
- `psutil`
- GROMACS
- 
From the point of view of the requirements, the code is rather light.

## How to build you "params" file
[WIP]

## Tests
Two general tests for running AIMMD/analyzing results are currently implemented:
- `test_toy_1d.py`
- `test_retinal.py`
You can adapt the former to toy systems of arbitrary dimensions with a custom-written "integrator" engine, and the latter to molecular dynamics systems with the GROMACS engine. Please use the content of the `toy_1d` and `retinal` folders as templates.

## Example of AIMMD with the new architecture
Create a system given a parameters' file `params.py`, and run it in `run1` folder (will be created by AIMMD).
```
import aimmd
folder = 'run1'  # where
params = aimmd.Params.load('params.py')  # automatically checks loaded functions
launcher = aimmd.Launcher(params, folder)

# locally (spawn parallel processes)
launcher.run(n=1, nA=1, nB=1, nsteps=10, walltime=300)

# on cluster (creates SLURM submission scripts)
launcher.create_job('job.sh', n=1, nA=1, nB=1, nsteps=10)

# on terminal
Worker(params, folder).train()
```

## Left to do
- All `PathEnsemble` to inherit functions from `Params` for more convenient reloading.
