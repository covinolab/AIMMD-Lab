# AIMMD – AI Agent Onboarding Guide

## Environment & Installation

AIMMD is installed in the **`new_aimmd` conda environment**:

```bash
conda activate new_aimmd
```

Key details:
- Environment path: `/home/lichtinger/anaconda3/envs/new_aimmd`
- Python: 3.13
- AIMMD version: 0.1.0 (installed as a package under `site-packages`)
- PyTorch: 2.7.1+cu118 (CUDA 11.8)

To run a script:
```bash
conda activate new_aimmd
python run_1.py
```

Or without activating first:
```bash
conda run -n new_aimmd python run_1.py
```

When submitting SLURM jobs, include environment activation in the job script:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate new_aimmd
```

And set the GROMACS executable PATH explicitly in the same script (it is not inherited from the login shell; see `aimmd/_init.py` for how the path is resolved at runtime).

---

## What is AIMMD?

AIMMD (AI for Molecular Mechanism Discovery) is a Python package for **adaptive importance sampling of rare molecular transitions** using AI-guided path sampling. It learns a committor model (a neural network predicting the probability of reaching state B before state A) and uses it to bias where new trajectories are shot, forming a feedback loop that accelerates sampling of reactive paths.

Core loop:
1. Start with one or more transition paths
2. Shoot new trajectories from selected frames (guided by current committor model)
3. Train neural network on collected paths
4. Use updated model to guide future shooting-point selection
5. Repeat until convergence

---

## Repository Structure

```
AIMMD/
├── aimmd/                    # Main Python package
│   ├── __init__.py           # Public API re-exports
│   ├── _init.py              # Runtime initialization (GROMACS path, caches)
│   ├── _config.py            # Global configuration
│   ├── params/               # Params class (run configuration)
│   ├── path/                 # Path class (single trajectory)
│   ├── pathensemble/         # PathEnsemble class (collection of paths)
│   ├── worker/               # Worker class (atomic execution unit)
│   ├── launcher/             # Launcher class (orchestration)
│   ├── network/              # Neural network training utilities
│   ├── engines/              # Simulation backends (GROMACS, toy)
│   ├── core/                 # Base utilities and decorators
│   ├── analysis/             # Binning, reweighting, confidence intervals
│   ├── resources/            # CPU/GPU resource binding
│   ├── execute/              # Subprocess management
│   └── cache/                # Trajectory reader and NumPy array caching
├── docs/                     # Sphinx documentation (RST)
├── examples/notebooks/       # Jupyter notebooks (1_toy_1d.ipynb)
├── tests/                    # Unit tests
└── implementation_trials/    # Experimental code
```

Almost all public classes use a **mixin-based architecture**: each mixin implements one concern (I/O, magic methods, properties, compute). Do not expect a single monolithic class file — look across the numbered mixin files in each subdirectory.

---

## Key Classes

### `Params` (`aimmd/params/`)

Central configuration object. Users define a `params.py` file and load it:

```python
params = aimmd.Params.load('params.py')
```

Critical fields:
- `states_function(trajectory)` → array of per-frame state labels (e.g. `'A'`, `'B'`, `'S'`)
- `descriptors_function(trajectory)` → `(N_frames, N_descriptors)` array
- `values_function(descriptors)` → `(N_frames,)` committor logit array
- `network` → PyTorch `nn.Module` (or subclass) used for training
- `fit` → callable training function with signature `fit(network, pathensemble, ...)`
- `topology`, `trajectory_extension` (`.xtc` or `.trr`)
- `gmx_grompp`, `gmx_mdrun`, `gmx_eneconv` → GROMACS command prefixes
- `chain_type` → `'tps'` (acceptance/rejection) or `'rfps'` (rejection-free)
- `nbins`, `selection_pool_size`, `cutoff_min`, `cutoff_max`
- `max_length`, `gen_temperature`
- `slurm_header` → list of SBATCH directives for HPC submission
- **Multi-system (multi-ligand)**: `multi_system` (bool), `multi_system_share_network` (bool), `system_ids` (per-system labels), `atom_types` (fixed shared graph encoding), `trainers_share_gpu` (bool). When `multi_system=True`, `topology`/`initial_paths` become lists (one per system), the data functions receive a `system_id=` keyword, and each system runs in `<run>/<system_id>/`. See `docs/source/workflow.rst` ("Multi-System Runs") and `examples/notebooks/2_multi_system.ipynb`.
- **Bias recording (OPES/PLUMED)**: `record_bias` (bool), `bias_function`, `bias_source` (`'reader'`|`'file'`), `bias_reactive_threshold` (float). The trainer caches per-frame bias (`<traj>.bias.npy`) and prints Tiwary-Parrinello bias-reweighted rates `k = 1/Σ(wᵢ·Lᵢ·γᵢ)`. **Works with `multi_system`**: the bias enters GROMACS via the per-system (list-valued) `gmx_mdrun` `-plumed` string, `bias_reactive_threshold` may be a per-system list, and per-system bias-reweighted rates are reported. Each system's PLUMED `PRINT STRIDE` must equal its `nstxout-compressed` (COLVAR↔frame alignment).
- **Value-pass subsampling**: `subsample_caps` (dict or per-system list of dicts; default `None`). Bounds the per-round committor value pass + bins + reweighting by running them on a fresh random subsample (`PathEnsemble.subsample`), while `fit` keeps the full ensemble. Keys: `'shot'`/`'free'` cap PATHS *per direction-type* (`'shot':100` ⇒ up to 4×100=400 shot paths/system), `'in_state'` caps FRAMES per state. Uniform-within-category so rates stay consistent; in-state-only paths carry zero reweight. `nbins=0` skips bin generation but keeps the (capped) value pass + rates.
- **Shooting-pool / density heuristics** (from `flexible_sampling`): `always_select_inside_the_bins` (bool, default `False`) — restrict shooting-point selection to paths with values inside the current bin range; `density_adjustment` (`Number`, default `inf`) — how many recent selection points to density-correct (old `bool` configs are converted: `True`→`inf`, `False`→`0`); `shared_density_adjustment` (bool, default `False`) — apply the density correction across all chains a worker manages. All default to the previous behaviour, so old params files are unaffected.

**Important:** GROMACS executables are found at runtime via `aimmd._init.py`. If running via subprocess or SLURM, the PATH must include `gmx`/`gmx_mpi`. Set this explicitly — it is not inherited automatically.

---

### `Path` (`aimmd/path/`)

Lightweight reference to a trajectory on disk. Stores an ordered list of segments (file, start frame, end frame). Heavy data (coordinates, descriptors, committor values) are lazily loaded and cached as `.npy` files alongside the trajectory.

```python
path[i]          # single frame
path[i:j]        # slice → new Path
path + path2     # concatenate
path.states      # per-frame state labels (cached)
path.descriptors # per-frame descriptors (cached)
path.values      # per-frame committor logits (cached)
path.write(filename)
```

Cache files live next to trajectories:
- `traj.xtc.states.npy`
- `traj.xtc.descriptors.npy`
- `traj.xtc.values.npy`

---

### `PathEnsemble` (`aimmd/pathensemble/`)

Collection of `Path` objects with vectorized access, filtering, projection, and reweighting:

```python
ensemble = params.pathensemble('run_directory')
ensemble[i]                    # single path
ensemble[ensemble.are_excursions()]  # filter
ensemble.project(values, bins) # histogram committor values
ensemble.reweight(...)         # equilibrium reweighting
ensemble.shooting_results(...) # empirical committor from sweep data
```

---

### `Worker` (`aimmd/worker/`)

Atomic execution unit. Each worker process runs one task type:

1. **`shoot`** — core path sampling: selects a shooting point via committor + adaptive bins, runs backward and forward MD, merges into a path, registers it
2. **`free`** — unbiased long MD simulations from metastable states (used for additional equilibrium sampling or shooting-point override)
3. **`train`** — loads path ensemble, fits network, updates cached values, rebuilds adaptive bins

Workers check stop conditions periodically (walltime, nsteps, nframes) and shut down cleanly. They do not need explicit synchronization — paths and network weights are exchanged via the filesystem.

---

### `Launcher` (`aimmd/launcher/`)

Orchestrates one or more simultaneous runs:

```python
launcher = aimmd.Launcher(params, 'run_directory')
# Local execution:
launcher.run(n=5, n1=1, n2=1, nframes=25000, walltime=3600)
# SLURM script generation:
launcher.create_job(n=5, n1=1, n2=1, nframes=100000, walltime=86400, jobname='my_run')
```

`n` = shooting workers (reactive region), `n1` = free workers (state A), `n2` = free workers (state B). These accept a scalar or a list of length `len(launcher)` (one per run). In a `multi_system` run they apply **per system** (each system gets `n`/`n1`/`n2` workers), and the launcher expands one run into per-system subfolders, emitting per-system shoot/free workers plus either one shared trainer (at the run root) or one trainer per system.

`nchains_per_worker` (default `1`, from `flexible_sampling`) makes each shooting worker manage that many independent chains (folders `chain{t}{k}`, `chain{t}{k+1}`, …); a higher value regularizes the training set and reduces correlation when `selection_pool_size=1`. It is a launcher argument (not a persisted `Params` field) and is **appended at the end** of `run()`/`create_job()` so old positional/keyword calls are unaffected; with the default `1` the on-disk layout is identical to before. It is ignored in sweep mode (each sweep worker manages one chain).

The launcher monitors all workers and stops everything if any worker exits with an error.

---

## Typical Run Setup

A minimal production run consists of two files:

**`params.py`** — defines states, descriptors, network, and all Params fields

**`run.py`** — loads params and launches:
```python
import aimmd
params = aimmd.Params.load('params.py')
launcher = aimmd.Launcher(params, 'run1')
launcher.run(5, 1, 1, nframes=25000)
```

Alternatively, use `launcher.create_job(...)` to produce a SLURM submission script.

---

## Output Directory Structure

After a run, the working directory contains:

```
run1/
├── initialARB/                   # Exported initial paths
├── chainR0/, chainR1/, ...       # One subdir per shooting worker
│   ├── path000001.xtc            # Accepted trajectory
│   ├── path000001.xtc.states.npy
│   ├── path000001.xtc.descriptors.npy
│   ├── path000001.xtc.values.npy
│   ├── back.xtc, forw.xtc        # Backward/forward segments during shooting
│   └── back.tpr, forw.tpr, ...   # GROMACS files
├── freeA/, freeB/                # Free simulation trajectories
├── networkARB.h5                 # Current network weights
├── networkARB.step0010.h5        # Periodic snapshots
├── binsARB.npy                   # Adaptive bin boundaries
└── densitiesARB.npy              # Bin densities
```

File locks (`*.lock`) and atomic renames (`.new.npy` → `.npy`) ensure concurrent safety.

**Multi-system runs** nest one level: each system gets a subfolder `run1/<system_id>/` containing that system's `chainR*/`, `freeA/`, `freeB/`, `binsARB.npy`, `densitiesARB.npy`. With a shared network the single `networkARB.h5` lives at the run root (`run1/`); with separate networks each subfolder has its own `networkARB.h5`.

---

## Real-World Example: Calixarene Binding

The calixarene runs at `/home/covino-shared/data/lichtinger/LigandAIMMD/new_aimmd_runs/calixarene/validation_datasets/` are a reference for production usage:

- **States** defined by ligand–host COM distance (5 states: B=bound, Z, S, A, R=unbound)
- **Network**: PaiNNModel (E(n)-equivariant GNN), `cutoff=3.0 Å`, 3 layers, 32 hidden channels
- **Descriptors**: heavy-atom coordinates; graph structure cached in SQLite
- **Fit**: 2000 epochs, lr=5e-4, Bayesian factor=20, regularization=1e-8
- **Workers**: 5 shooting + 1 freeA + 1 freeB, stop at 25 000 frames
- **Validation** (`MB_sweep/`): `committor_sweep.py` samples validation frames by committor bin, runs sweep mode (repeated shooting from fixed frames), computes empirical committor vs. model prediction (RMSE, R²)

---

## Graph Neural Networks

When using GNN-based networks (like PaiNN):

- `descriptors_function` returns raw atom coordinates, not featurized distances
- `values_function` must batch frames for efficient GPU inference (use PyTorch Geometric batching)
- Graph construction (edges based on cutoff) is typically cached in SQLite to avoid recomputing every call
- The network subclasses `torch.nn.Module`; implement `forward(batch)` returning per-graph scalars
- Use `aimmd.network.fit` utilities for the training loop; a custom `fit` callable can be passed to `Params`
- For a **shared multi-ligand network**, pass a fixed `atom_types` table (params field, forwarded to `get_graphs_pyg`) so all systems featurize into the same one-hot columns, and call `fit` with a *list* of PathEnsembles (it pools them balanced). Different atom counts are fine: graphs are batched per system via `Batch.from_data_list`, and the in-memory descriptor transform is applied per system before pooling — see `aimmd/network/fit.py` (`_assemble_inmemory_multi`, `_load_batch_descriptors_routed`).

---

## GROMACS Integration

AIMMD drives GROMACS via subprocess calls:

1. `gmx grompp -f run.mdp -p topol.top -c conf.gro -o tpr` — prepares a run input file
2. `gmx mdrun -deffnm <name> -maxh <hours>` — executes dynamics

The engine reads frames incrementally from `.xtc` as they appear. GROMACS stop conditions are communicated via cooperative process management (not hard kill).

**Always set the GROMACS PATH explicitly** when submitting jobs — it is not inherited from the login shell.

---

## Key Design Patterns

- **Mixin-based classes**: each public class is assembled from multiple numbered mixins (e.g. `_fields.py`, `_io.py`, `_magic.py`, `_methods.py`). Read all of them to understand the full API.
- **Filesystem-backed state**: paths are references to `.xtc` files; heavy data cached as `.npy` beside them. Workers communicate through shared disk, not memory.
- **Atomic writes**: temp files + rename for cache updates; `.lock` files for mutual exclusion.
- **Cooperative termination**: workers check `must_stop()` after each step; SIGINT/SIGTERM are caught and handled cleanly.
- **Feedback loop without synchronization**: shooting and training workers run independently; shooting picks up the latest network weights from disk at each selection step.

---

## Common Tasks for AI Agents

| Task | Where to look |
|------|--------------|
| Understand Params fields | `aimmd/params/_fields.py` + module docstring |
| Understand shooting logic | `aimmd/worker/_shoot.py` |
| Understand training loop | `aimmd/worker/_train.py`, `aimmd/network/fit.py` |
| Understand path representation | `aimmd/path/_helpers.py`, `_compute.py` |
| Understand reweighting | `aimmd/pathensemble/_reweight.py` |
| Add a new network architecture | Subclass `torch.nn.Module`; implement `values_function`; optionally provide custom `fit` |
| Add a new engine | Follow `aimmd/engines/toy.py` as template |
| Validate a trained model | See `committor_sweep.py` in calixarene example |
| Understand adaptive bins | `aimmd/worker/_train.py` + `aimmd/analysis/` |
