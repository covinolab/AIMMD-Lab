[![pytest](https://github.com/covinolab/AIMMD/actions/workflows/pytest.yml/badge.svg)](https://github.com/covinolab/AIMMD/actions/workflows/pytest.yml)
[![Documentation Status](https://readthedocs.org/projects/aimmd-lab/badge/?version=latest)](https://aimmd-lab.readthedocs.io/en/latest/?badge=latest)
[![Read the Docs](https://img.shields.io/badge/Read_the_Docs-8CA1AF?logo=readthedocs&logoColor=white)](https://aimmd-lab.readthedocs.io/en/latest/)

# AIMMD

AIMMD (AI for Molecular Mechanism Discovery) implements **AI-enhanced path sampling** for molecular systems in a parallel way to run both on local systems and HPC clusters, interfacing with the gromacs MD engine. For details, please see our publications:


(1)	Jung, H.; Covino, R.; Arjun, A.; Leitold, C.; Dellago, C.; Bolhuis, P. G.; Hummer, G. Machine-Guided Path Sampling to Discover Mechanisms of Molecular Self-Organization. Nat. Comput. Sci. 2023, 3 (4), 334–345. https://doi.org/10.1038/s43588-023-00428-z.

(2)	Lazzeri, G.; Jung, H.; Bolhuis, P. G.; Covino, R. Molecular Free Energies, Rates, and Mechanisms from Data-Efficient Path Sampling Simulations. J. Chem. Theory Comput. 2023, 19 (24), 9060–9076. https://doi.org/10.1021/acs.jctc.3c00821.

(3)		Lazzeri, G.; Bolhuis, P. G.; Covino, R. Optimal Rejection-Free Path Sampling. arXiv. https://arxiv.org/html/2503.21037v1.


# Installation instructions

AIMMD is a python package, and we recommend installing 
it in a fresh conda environment.

```bash
conda create -n aimmd python=3.13
```

## Prerequisites

AIMMD depends on the gromacs MD engine. Any installation that adds working `gmx` or `gmx_mpi` to your path will do. For testing purposes and lightweight tasks, gromacs can be installed via conda:

```
conda install conda-forge::gromacs
```

For production runs and optimal performnace, however, it is highly recommended to install gromacs from source. See [here](https://manual.gromacs.org/2025.4/install-guide/index.html) for instructions.

Additionally, if you wish to use AIMMD with graph neural networks (GNNs), ```torch-geometric```, ```torch-cluster``` and ```mlcolvar``` are additional dependencies. Since ```torch-cluster``` can be tricky to install depending on your machine's configuration, AIMMD will not install these dependencies by default. On a Linux machine with an NVIDIA GPU compatible with cuda 11.8, we have confirmed that the following works in a python 3.13 environment:

```
pip install torch==2.7.1 -f https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp313-cp313-manylinux_2_28_x86_64.whl
pip install torch-geometric==2.7.0
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp313-cp313-linux_x86_64.whl
pip install mlcolvar
```

### Installing AIMMD

The package has not yet been deposited on PyPi. To install it, you can clone the github repository, and - once in the folder - do:

```
pip install -e .
```

### Verification of the installation

Lastly, we recommend you run the tests to verify that AIMMD was installed correctly.

```
pip install pytest
pytest tests/
```


# Introduction to the implementation

The core task implemented in this package is **committor-guided shooting** in the reactive region of molecular systems. Short unbiased simulations
are launched from **shooting points** selected using a learned committor model
(typically a neural network). This produces a **diverse ensemble of reactive
trajectories**, and - in most cases - **more transition events than equilibrium**
sampling at comparable cost.

The resulting path ensemble can be inspected and reweighted to obtain estimates
of **free-energy profiles** and **transition rates** and, together with the
learned reaction coordinate (committor model), to support mechanistic
interpretation. For rejection-free path sampling workflows, reweighting and
bin/density adaptation are also performed **on the fly**. We provide tools
(and example notebooks) to support this analysis.

AIMMD organizes computation into **workers** launched by a **launcher**, either using multiprocessing or ```srun``` on HPC architectures.

## Worker tasks
Workers execute exactly one task at a time:

- **`shoot`**: the core path-sampling loop.  
  Selects a shooting point from a selection pool (committor-guided), runs a backward and forward simulation, merges them into a new path, and registers it in a shooting chain. Supports TPS-style acceptance, or the rejection-free sampling algorithm.

- **`free`**: runs equilibrium MD simulations around a chosen state.  
  Used to provide additional sampling/statistics and (optionally) candidate frames for shooting-point selection (“overriding” frames).

- **`train`**: updates the committor model and adaptive sampling state.  
  Trains the network on the current ensemble, computes/updates values on frames, builds adaptive bins, estimates densities, and writes artifacts.

These workers run simultaneously, so that the **`train`** worker provides frequent updates to the committor model, using always the most recent available training data.

## Setting parameters

Parameters guiding the AIMMD execution, including state definitions, neural network architectures, etc., are set in parameter python files. See the tutorials for examples.

## Sweep mode (validation)
The launcher can set `reactive_region_mode='sweep'`. In this mode, workers
deterministically cycle through a fixed set of frames and repeatedly shoot from
them to estimate committor values by brute force (ratio of outcomes reaching one
end state vs the other). This is mainly for **validating** the committor model.

