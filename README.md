[![pytest](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml/badge.svg)](https://github.com/gl95/AIMMD/actions/workflows/pytest.yml)

# AIMMD
**AI for Molecular Mechanism Discovery**

AIMMD implements **AI-enhanced path sampling** for molecular mechanisms discovery.
The core workflow is **committor-guided shooting**: short unbiased simulations
are launched from **shooting points** selected using a learned committor model
(typically a neural network), producing a **diverse ensemble of reactive
trajectories** and, in general, **more transition events than equilibrium**
sampling at comparable cost.

AIMMD runs:
- locally (multi-process execution),
- on HPC clusters (SLURM `srun` job scripts, massive parallelization).

---

## Concepts (what AIMMD runs)

AIMMD organizes computation into **workers** launched by a **launcher**.

### Worker tasks
Workers execute exactly one task at a time:

- **`shoot`**: the core path-sampling loop.  
  Selects a shooting point from a selection pool (committor-guided), runs a
  backward and forward simulation, merges them into a new path, and registers it
  in a shooting chain. Supports TPS-style acceptance when configured.

- **`free`**: runs free trajectories around a chosen state.  
  Used to provide additional sampling/statistics and (optionally) candidate
  frames for shooting-point selection (“overriding” frames).

- **`train`**: updates the committor model and adaptive sampling state.  
  Trains the network on the current ensemble, computes/updates values on frames,
  builds adaptive bins, estimates densities, and writes artifacts.

### Sweep mode (validation)
The launcher can set `reactive_region_mode='sweep'`. In this mode, workers
deterministically cycle through a fixed set of frames and repeatedly shoot from
them to estimate committor values by brute force (ratio of outcomes reaching one
end state vs the other). This is mainly for **validating** the committor model.

---

## Requirements

Main Python dependencies:
- `numpy`, `scipy`
- `torch`
- `tqdm`
- `matplotlib`
- `psutil`
- `dill`
- `MDAnalysis`
- `mdtraj`

Engines:
- **GROMACS** for MD-based workflows (optional but typical).
- A lightweight “toy” engine can be used for low-dimensional models/tests.

AIMMD itself is relatively lightweight; heavy requirements come from the MD
engine and trajectory stack.

---

## Installation

This repository is a Python package. Install in editable mode:

```bash
pip install -e .
