"""
aimmd.params._fields
===================

Dataclass field definitions for :class:`aimmd.params.Params`.

This module defines :class:`~aimmd.params._fields.ParamsFields`, a dataclass-like
mixin containing the full set of AIMMD parameters and their default values.

Each field includes a human-readable description in `metadata['description']`,
used by:
- `Params.__str__` to generate a verbose parameters file representation,
- `Params.save` to write a reproducible `params.py` script.

Notes
-----
- This mixin is not intended to be used standalone.
- Type annotations are used for validation by `ParamsHelpers._setattr`.
- Some fields are callables that must be serializable by `ParamsIO.save`;
  see `aimmd.params.utils.update_source`.
"""

# external
from abc import ABC
from math import inf
from typing import List, Callable
from numbers import Number
from pathlib import Path as PosixPath
from torch.nn import Module as NeuralNetworkModule
from dataclasses import dataclass, field

# aimmd imports
from .._config import GROMACS
from ..network.fit import default as default_fit
from ..network.utils import placeholder as placeholder_network


@dataclass
class ParamsFields(ABC):
    """
    Dataclass field mixin defining the AIMMD parameter set.

    This mixin contains:
    - system definitions (topology, trajectory extension),
    - engine configuration (GROMACS/toy),
    - neural network components and data pipeline functions,
    - path sampling / shooting selection options,
    - saving and SLURM-related parameters,
    - internal bookkeeping (`path`).

    Notes
    -----
    - `states_function` has no default and must be provided.
    - `path` is set at load time and is not meant to be user-assigned directly.
    """

    # ------------------------------------------------------------------
    # States and state mapping
    # ------------------------------------------------------------------

    states_function: Callable = field(
        metadata={'description':
"""Map an MDAnalysis trajectory or Timestep to state labels.
This callable must accept an MDAnalysis trajectory-like object (or a Timestep,
depending on your implementation) and return an array of state identifiers
(e.g., one integer or one single-character label per frame). Used to:
- detect whether a trajectory contains transitions,
- classify frames as belonging to metastable states or the reactive region."""
                 })

    states: str = field(
        default='ARB',
        metadata={'description':
"""State label specification. A compact string defining:
- the first metastable state (e.g., 'A'),
- the reactive/intermediate region label (e.g., 'R'),
- the final metastable state (e.g., 'B').
Example: 'ARB' means transitions are defined from A to B through R."""
                 })

    # ------------------------------------------------------------------
    # System information
    # ------------------------------------------------------------------

    name: str = field(
        default='AIMMD',
        metadata={'description':
"""System name, used when creating SLURM job names."""
        })

    topology: str = field(
        default='run.gro',
        metadata={'description':
"""Topology/structure file used for engine setup and mass lookup.
Typically a GROMACS .gro file. Used by:
- mass assignment routines,
- (in case `engine = 'gromacs') `grompp` when constructing .tpr files."""
                 })

    trajectory_extension: str = field(
        default='.xtc',
        metadata={'description':
"""Trajectory file extension written and read by the engine.
Common values:
- '.xtc' : compressed coordinates (positions only),
- '.trr' : full-precision coordinates and (optionally) velocities.
Must be consistent with the engine configuration and your analysis pipeline."""
                 })

    # ------------------------------------------------------------------
    # Engine configuration
    # ------------------------------------------------------------------

    engine: str = field(
        default='gromacs',
        metadata={'description':
"""Simulation engine backend.
Supported values:
- 'gromacs' : external GROMACS runs,
- 'toy'     : lightweight Python integrator (see `toy_mdrun`)."""
                 })

    # ------------------------------------------------------------------
    # GROMACS engine configuration
    # ------------------------------------------------------------------

    gmx_mdp: str = field(
        default='run.mdp',
        metadata={'description':
"""GROMACS .mdp file used for production segments.
Used when running shooting trajectories and free simulations.
Must be compatible with `trajectory_extension` (e.g., if you require velocities,
use settings consistent with '.trr')."""
                 })

    gmx_grompp: str = field(
        default=f'{GROMACS} grompp -maxwarn 1',
        metadata={'description':
"""Base `grompp` command used to build .tpr files.
You may extend this command (e.g., add `-n index.ndx`), but do NOT include flags
that AIMMD injects automatically (e.g., input coordinates, output file names,
and `-nobackup` handling)."""
                 })

    gmx_mdrun: str = field(
        default=f'{GROMACS} mdrun -v -maxh 4',
        metadata={'description':
"""Base `mdrun` command used to run dynamics. You may tune performance flags
here (MPI/OMP/GPU), but do no include `-deffnm`, `-nobackup`, and `-noappend`,
as those are add automatically by AIMMD.
Performance note: to prevent CPU oversubscription and ensure smooth
communication between GROMACS and AIMMD, set `-ntmpi` to `(cpus_per_task - 1)`.
This reserves the final CPU core exclusively for computing CVs and orchestrating
the simulation logic."""
                 })

    gmx_eneconv: str = field(
        default=f'printf "c\nc\n" | {GROMACS} -nobackup eneconv -settime',
        metadata={'description':
"""Command used to merge GROMACS energy (.edr) files after two-way shooting.
This command should merge backward/forward segment energy files into a single
time-consistent .edr file (commonly via `eneconv -settime`).
Set to an empty string to disable energy file merging/saving."""
                 })

    # ------------------------------------------------------------------
    # Toy engine configuration
    # ------------------------------------------------------------------

    toy_mdrun: Callable = field(
        default=None,
        metadata={'description':
"""Toy-engine integrator step function.
Callable that advances the system by one step for the toy engine.
It is expected to take an MDAnalysis Timestep (or equivalent) as input
and evolve it in-place, based on the chosen law of motion."""
                 })

    toy_slowdown: float = field(
        default=0.01,
        metadata={'description':
"""Artificial delay per toy integration step (seconds).
Used to slow down the toy engine to emulate wall-clock behavior or to reduce
CPU usage during debugging, while allowing AIMMD manager tasks to keep up with
the simulation speed."""
                 })

    # ------------------------------------------------------------------
    # Neural network and computation options
    # ------------------------------------------------------------------

    network: NeuralNetworkModule = field(
        default=placeholder_network,
        metadata={'description':
"""Neural network model used to estimate logit-committor-like values.
The network is evaluated on descriptors (or positions if descriptors are not
provided). The default placeholder returns a trivial output.
Important:
If you load parameters from a Python params file, the network class must be
importable or defined in that same file so it can be reconstructed."""
                 })

    values_function: Callable = field(
        default=None,
        metadata={'description':
"""Map descriptors to scalar values (typically logit committor).
If None, AIMMD evaluates `network(descriptors)` directly.
If provided, this callable must accept an array of descriptors and return a
1D array of values (one per frame)."""
                 })

    descriptors_function: Callable = field(
        default=None,
        metadata={'description':
"""Compute descriptors from an MDAnalysis trajectory.
If None, AIMMD uses raw positions (potentially more expensive and higher
dimensional). If provided, the callable should return an array of descriptors
(one descriptor vector per frame)."""
                 })

    descriptor_transform: Callable = field(
        default=None,
        metadata={'description':
"""Transform applied to descriptors immediately before network evaluation.
Typical uses:
- normalization/standardization,
- feature selection,
- dimensionality transforms.
If None, no transform is applied."""
                 })

    network_batch_size: int = field(
        default=4096,
        metadata={'description':
"""Maximum number of frames evaluated by the network in a single batch.
Reduce this value if you run out of GPU/CPU memory during inference."""
                 })

    # ------------------------------------------------------------------
    # Initial paths
    # ------------------------------------------------------------------

    initial_paths: List = field(
        default_factory=lambda: [],
        metadata={'description':
"""Initial transition paths used to seed the PathEnsemble.
Accepted elements include:
- trajectory filenames (string or list of strings, regular expressions allowed),
- MDAnalysis trajectory objects,
- `aimmd.Path` objects.
Paths are validated to contain transitions according to `states_function` and
`states`. These paths initialize an `aimmd.PathEnsemble`."""
        })

    # ------------------------------------------------------------------
    # Shooting point selection options
    # ------------------------------------------------------------------

    chain_type: str = field(
        default='rfps',
        metadata={'description':
"""Path-sampling chain type.
Supported values:
- 'tps'  : transition path sampling,
- 'rfps' : rejection-free path sampling."""
                 })

    always_select_inside_the_bins: bool = field(
        default=True,
        metadata={'description':
"""If True: ensure you always select from paths with values in the current
bins. This to prevent the simulations from getting stuck close to the state
boundaries."""
                 })

    retry_with_state_definition_glitches: bool = field(
        default=False,
        metadata={'description':
"""If True, automatically recover from transient state-definition glitches
during shooting. Occasionally the first frame of back.xtc/forw.xtc — the
shooting point itself — is classified in a different state than expected
due to slightly different PBC handling in GROMACS vs MDAnalysis, producing
a RuntimeError like "... 0 in state A, should be in R; consider deleting
the trajectory file to allow AIMMD to recreate it". When this flag is True,
AIMMD logs a warning, deletes back* and forw* in the offending chain
directory, and reselects a new shooting point on the next iteration.
When False (default), the error is re-raised and the worker exits as
before. Only the specific "should be in" error is caught; all other errors
propagate unchanged."""
                 })

    nbins: int = field(
        default=10,
        metadata={'description':
"""Number of bins used to discretize value space in the reactive region.
Bins are constructed in (logit) committor/value space and used to guide
shooting point selection. Must be >= 0. 0 disables binning."""
                 })

    cutoff_min: float = field(
        default=0.5,
        metadata={'description':
"""Minimum absolute value for finite bin boundaries.
Ensures that the first/last finite boundaries are not placed too close to zero
in absolute value."""
                 })

    cutoff_max: float = field(
        default=20.0,
        metadata={'description':
"""Maximum absolute value for finite bin boundaries.
Finite boundaries are clipped to lie within [-cutoff_max, +cutoff_max].
If free simulations are disabled, the finite bin range typically spans
approximately `[-cutoff_max, +cutoff_max]` (with optional +-inf bins)."""
                 })

    terminal_bin_extension: str = field(
        default='',
        metadata={'description':
"""Extends the first and/or last bin edges to the state interfaces (+-inf).
This forces the outermost bins to capture all configurations close to the
states, increasing exploration at the cost of reduced exploitation.
Values: '' (disable), 'all' (apply to both edges), or directly the state
names towards which you want the extension to happen ("A", "B", "AB", etc.)."""
                 })
    
    local_density_adjustment: Number = field(
        default=inf,
        metadata={'description':
"""Apply a correction to the density during selection to accelerate
convergence. For each shooting chain, in each bin: multiply the density by
the number of the latest `local_density_adjustment` shooting points already
selected in the bin. Stacking with `global_density_adjustment`."""
                 })

    global_density_adjustment: bool = field(
        default=False,
        metadata={'description':
"""If True: apply a correction to the density during selection to accelerate
convergence. For each shooting chain, in each bin: multiply the density by
the number of shooting points from all chains already selected in the bin.
Stacking with `local_density_adjustment`."""
                 })
    
    lorentzian: float = field(
        default=inf,
        metadata={'description':
"""Lorentzian width controlling the target distribution in value space.
If finite, shooting points are biased toward the center of the distribution
according to a Lorentzian in logit/value space.
If `inf`, shooting points are sampled approximately uniformly between the first
and last finite bin boundaries."""
                 })

    free_overriding_states: str = field(
        default='',
        metadata={'description':
"""Enable occasional shooting point selection from free simulations near states.
If non-empty, AIMMD may override the usual selection and draw shooting points
from free-simulation segments around selected states.
- If 'all': allow overriding around all states.
- If empty: disable overriding.
Warning: enabling this can slow down selection since you must reload the
free simulations every time."""
                 })
    
    free_overriding_attempts: int = field(
        default=100,
        metadata={'description':
"""Number of frames from the free simulations considered for overriding.
Higher values increase the chance of finding a usable free-simulation shooting
point but can increase overhead."""
                 })
    
    free_overriding_bins: List = field(
        default_factory=lambda: [0, -1],
        metadata={'description':
"""Bins where overriding is allowed, following the same logic as numpy array
indexing. For example, `free_overriding_bins = [0, 1]` will consider only the
first and last selection bin for overrding. `free_overriding_bins = None` will
consider all bins for overriding."""
        })
    
    free_overriding_recovery_rate: float = field(
        default=0.05,
        metadata={'description':
"""Probability of overriding within the same bin as the previous shooting point.
This is a “recovery” mechanism that can preserve local continuity, as orverriding
is intended to happen only if the new shooting point has changed selection bin
compared to the previous one. Too large values may disrupt the Markov chain
(excessive overrides)."""
                 })

    restart_free_simulations_with_transitions: str = field(
        default='',
        metadata={'description':
"""Restart free simulations from AIMMD-sampled transition for selected states.
If non-empty, free simulations targeting specified states are be restarted from
the last frames of randomly sampled transition paths rather than from the most
recent state crossing observed in the previous free simulation.
Accepted elements include:
- a list of states in capital letters (eg. 'AB'),
- 'all', which means that you consider *all* free simulations, regardless of their
target state."""
                 })

    # ------------------------------------------------------------------
    # Shooting point initialization options
    # ------------------------------------------------------------------

    gen_temperature: float = field(
        default=300.0,
        metadata={'description':
"""Velocity generation temperature (Kelvin).
If > 0: generate new velocities at this temperature.
If < 0: reuse velocities from the parent trajectory (requires
`trajectory_extension == '.trr'` so velocities are available)."""
                 })

    # ------------------------------------------------------------------
    # Simulation options
    # ------------------------------------------------------------------

    max_length: int = field(
        default=50_000,
        metadata={'description':
"""Maximum allowed trajectory length (frames) for a single path.
Prevents simulations from running indefinitely in long-lived intermediates.
Paths exceeding this length are typically truncated/terminated by the engine
control logic."""
                 })

    extra_free_frames: int = field(
        default=0,
        metadata={'description':
"""Extra frames to continue after reaching the target state in free simulations.
If > 0, free simulations do not stop immediately upon reaching the target
state, but continue for `extra_free_frames` additional frames."""
                 })

    # ------------------------------------------------------------------
    # Logit committor fit options
    # ------------------------------------------------------------------

    fit: Callable = field(
        default=staticmethod(default_fit),
        metadata={'description':
"""Callable that fits network parameters to PathEnsemble data.
Expected signature (conceptually):
- positional: (network, pathensemble)
- optional keyword args: verbose, worker
This callable is invoked by AIMMD training logic to update `params.network`."""
                 })

    rescale_committor: bool = field(
        default=False,
        metadata={'description':
"""Rescale network outputs to enforce expected committor boundary behavior.
If True, AIMMD rescales the model output to better match the expected crossing
probabilities near the states (heuristically ~1/p from A and ~1/(1-p) from B,
where p is the committor), assuming sufficiently small time between frames.
Requires that `params.network` is a subclass of `aimmd.network.Rescalable`.
Experimental: recommended only when also running free simulations."""
                 })

    # ------------------------------------------------------------------
    # Paths reweighting options
    # ------------------------------------------------------------------

    reweight_parameters: dict = field(
        default_factory=lambda: {'free_threshold': 20},
        metadata={'description':
"""Dictionary of parameters for path reweighting and rate/free-energy estimation.
Passed to `pathensemble.reweight(...)`. Typical entries control thresholds for
including free-simulation data and regularization choices."""
                 })

    # ------------------------------------------------------------------
    # Save options
    # ------------------------------------------------------------------

    trajectory_update_batch_size: int = field(
        default=1000,
        metadata={'description':
"""Number of frames registered/processed per update batch.
Controls how many frames are appended or indexed per internal update step.
During an AIMMD production run, the simulations stop if states and descriptors
computations are not catching up, and getting behind by `trajectory_update_batch_size`
or more. Reduce this when the simulation engine produces new frames much faster than
you can analyze them or if memory spikes while loading and analyzing trajectories.
"""
                 })
    
    network_save_interval: int = field(
        default=10,
        metadata={'description':
"""Save network parameters every N shooting iterations.
If 10, the model is saved after every 10 accepted/attempted shooting moves
(depending on the higher-level training loop)."""
                 })

    # ------------------------------------------------------------------
    # SLURM configuration
    # ------------------------------------------------------------------

    slurm_header: str = field(
        default='#SBATCH --mail-type=FAIL',
        metadata={'description':
"""Default SLURM header lines inserted into generated job scripts.
Do NOT include:
- shebang (`#!/bin/bash`),
- job name,
- walltime,
- node count.
Those are set automatically by AIMMD.
Scheduling model:
- each shooting/free worker uses one SLURM task,
- trainer takes an additional task."""
                 })

    # ------------------------------------------------------------------
    # Bias recording options
    # ------------------------------------------------------------------

    record_bias: bool = field(
        default=False,
        metadata={'description':
"""If True, record the instantaneous bias potential for each trajectory frame.
The bias is extracted by `bias_function` and cached as `<traj>.bias.npy` alongside
each trajectory file. Only when `record_bias=True` are bias caches computed and
Tiwary-Parrinello-corrected rate estimates printed during training.
Set to False (default) for unbiased runs; existing runs are fully unaffected."""
                 })

    bias_function: Callable = field(
        default=None,
        metadata={'description':
"""Callable that extracts the bias potential for each frame, in kT units (dimensionless).
Two calling conventions are supported (selected by `bias_source`):

- `bias_source = 'reader'` (default, toy/position-based):
      bias_function(trajectory_reader) -> ndarray of shape (n_frames,)
  Same convention as `states_function`. Called per-batch by `path.compute(...)`.

- `bias_source = 'file'` (PLUMED/GROMACS):
      bias_function(fname: str) -> ndarray of shape (n_frames_in_file,)
  Receives the trajectory file path; the function locates the associated PLUMED
  output (e.g. COLVAR file) and returns bias values for ALL frames in the file.
  No re-running of PLUMED is required.

In both modes, results are cached as `<traj>.bias.npy` and accessed via `path.bias`."""
                 })

    bias_source: str = field(
        default='reader',
        metadata={'description':
"""Determines the calling convention of `bias_function`. Supported values:
- 'reader' : bias_function(trajectory_reader) -> ndarray  [default; toy/position-based]
- 'file'   : bias_function(fname: str) -> ndarray         [PLUMED/GROMACS file-based]
See `bias_function` for details on each mode."""
                 })

    bias_reactive_threshold: float = field(
        default=0.5,
        metadata={'description':
"""Maximum acceptable mean |bias| (in kT) in the reactive region R.
After bias computation, the training worker checks the mean absolute bias over all
frames whose state label equals the reactive-region label (e.g. 'R' in 'ARB').
If the mean exceeds this threshold, a warning is printed. This validates the
Tiwary-Parrinello assumption that the bias is negligible inside R."""
                 })

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------

    path: PosixPath = field(
        init=False,
        default=PosixPath('.'),
        metadata={'description':
"""Base working directory for engine operations.
Engine commands (GROMACS/toy) are executed relative to this path.
Set automatically on load; typically not user-assigned."""
                 })
