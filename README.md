# AIMMD-Lab

[![pytest](https://github.com/covinolab/AIMMD/actions/workflows/pytest.yml/badge.svg)](https://github.com/covinolab/AIMMD/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/covinolab/AIMMD/graph/badge.svg?token=0B7E24VODY)](https://codecov.io/github/covinolab/AIMMD)
[![Documentation Status](https://readthedocs.org/projects/aimmd-lab/badge/?version=latest)](https://aimmd-lab.readthedocs.io/en/latest/?badge=latest)
[![Read the Docs](https://img.shields.io/badge/Read_the_Docs-8CA1AF?logo=readthedocs&logoColor=white)](https://aimmd-lab.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/aimmd-lab.svg)](https://pypi.org/project/aimmd-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

This repository implements **AI for Molecular Mechanism Discovery (AIMMD)** — AI-enhanced path sampling for rare
molecular transitions.

AIMMD generates **short unbiased simulations** from configurations chosen by a **machine-learned** 
committor model, building a diverse ensemble of trajectories that approximates a **long equilibrirum trajecotory** at a **fraction of the computational cost**.
That ensemble is reweighted to recover **free-energy profiles** and **transition
rates**, while the learned committor doubles as an interpretable **reaction
coordinate**. Runs scale from a laptop (multiprocessing) to HPC clusters (SLURM
`srun`), driving the GROMACS MD engine or a built-in toy engine.

📖 **Documentation:** <https://aimmd-lab.readthedocs.io>

A **alternative** package implementing AIMMD as published in
Jung _et al._ (2023) is available at <https://github.com/bio-phys/aimmd>.

## Features

- **Committor-guided two-way shooting** in the reactive region.
- **Rejection-free path sampling** (`rfps`) *and* classic TPS acceptance (`tps`).
- **On-the-fly training and reweighting** — the committor model, adaptive bins,
  and densities update continuously as data arrives.
- **Free energies and rates** from a single self-consistent path ensemble.
- **Multi-system / multi-ligand runs** trained with one shared committor network.
- **Biased dynamics** (OPES/PLUMED) with Tiwary–Parrinello rate reweighting.
- **Graph-neural-network** committor models (optional).
- **Local or HPC** execution from the same parameter file.

## Installation

### From PyPI

```bash
pip install aimmd-lab
```

The distribution is named `aimmd-lab`, but the import name is `aimmd`:

```python
import aimmd
```

We recommend a clean conda environment (Python 3.13 is tested):

```bash
conda create -n aimmd python=3.13
conda activate aimmd
pip install aimmd-lab
```

### Prerequisites: GROMACS

AIMMD drives the GROMACS MD engine and expects `gmx` or `gmx_mpi` on your
`PATH`. For testing and lightweight work it can be installed from conda-forge:

```bash
conda install conda-forge::gromacs
```

For production runs we recommend building GROMACS from source for optimal
performance; see the [GROMACS install guide](https://manual.gromacs.org/current/install-guide/index.html).
(The built-in toy engine needs no GROMACS.)

### Optional: graph neural networks

Graph-based committor models need extra packages (`torch-cluster` can be awkward
to build). Install the `graphs` extra plus `mlcolvar`:

```bash
pip install "aimmd-lab[graphs]"
pip install mlcolvar
```

If the default wheels do not match your CUDA/Python build, install them
explicitly — see the [installation guide](https://aimmd-lab.readthedocs.io/en/latest/installation.html)
for a confirmed-working CUDA 11.8 / Python 3.13 example.

### Development install

```bash
git clone https://github.com/covinolab/AIMMD.git
cd AIMMD
pip install -e ".[tests,docs]"
pytest tests/          # verify the installation
```

## Quickstart

An AIMMD run is defined by a `params.py` file (states, features, network,
engine, sampling controls) and a short launch script:

```python
import aimmd

params   = aimmd.Params.load('params.py')   # load the run configuration
launcher = aimmd.Launcher(params, 'run1')   # attach a working directory

# 5 shooting workers, 1 free-A, 1 free-B; stop after 25k frames.
launcher.run(5, 1, 1, nframes=25000)

# Reload and analyze the resulting path ensemble.
ensemble = params.pathensemble('run1')
ensemble.report()
ensemble.reweight()      # free energies / rates
```

See the [Quickstart](https://aimmd-lab.readthedocs.io/en/latest/quickstart.html)
and the runnable [tutorial notebooks](https://aimmd-lab.readthedocs.io/en/latest/tutorials/index.html)
(a 1-D toy system and a multi-ligand run, both GROMACS-free) for complete
examples.

## Documentation

Full documentation is hosted at <https://aimmd-lab.readthedocs.io>, including:

- [Concepts](https://aimmd-lab.readthedocs.io/en/latest/concepts.html) and the
  [scientific background](https://aimmd-lab.readthedocs.io/en/latest/scientific_background.html),
- the [workflow](https://aimmd-lab.readthedocs.io/en/latest/workflow.html) and
  [advanced usage](https://aimmd-lab.readthedocs.io/en/latest/advanced.html)
  (multi-system, biased dynamics, validation),
- the full [parameter reference](https://aimmd-lab.readthedocs.io/en/latest/parameters.html) and
  [API reference](https://aimmd-lab.readthedocs.io/en/latest/api/index.html).

## Citation

If you use AIMMD in your work, please cite the relevant papers:

1. Jung, H.; Covino, R.; Arjun, A.; Leitold, C.; Dellago, C.; Bolhuis, P. G.;
   Hummer, G. *Machine-Guided Path Sampling to Discover Mechanisms of Molecular
   Self-Organization.* Nat. Comput. Sci. **2023**, 3 (4), 334–345.
   <https://doi.org/10.1038/s43588-023-00428-z>
2. Lazzeri, G.; Jung, H.; Bolhuis, P. G.; Covino, R. *Molecular Free Energies,
   Rates, and Mechanisms from Data-Efficient Path Sampling Simulations.*
   J. Chem. Theory Comput. **2023**, 19 (24), 9060–9076.
   <https://doi.org/10.1021/acs.jctc.3c00821>
3. Lazzeri, G.; Bolhuis, P. G.; Covino, R. *Optimal Rejection-Free Path
   Sampling.* arXiv **2025**. <https://arxiv.org/abs/2503.21037>

A `CITATION.cff` file is included for GitHub's "Cite this repository" feature.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[developer guide](https://aimmd-lab.readthedocs.io/en/latest/developer_guide.html).

## License

AIMMD is released under the MIT License. See [LICENSE](LICENSE).
