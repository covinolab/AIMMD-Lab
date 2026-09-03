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

    selection_pool_size: int = field(
        default=10,
        metadata={'description':
"""Number of candidate paths for shooting point selection considered
at each selection step. For standard TPS (`chain_type='tps'`), this must be 1.
For RFPS, when running with `nchains_per_worker=1`, `selection_pool_size>1`
enables pool-based selection and bin rebalancing, while improving the
homogeneity of the sampled chain. As an alternative, you can run multiple
chains per worker (`nchains_per_worker>1`)"""
                 })

    at_least_one_transition_in_pool: bool = field(
        default=False,
        metadata={'description':
"""If True: ensure each selection pool contains at least 1 transition.
If True, detailed balance is not strictly preserved, but it can reduce
stagnation near state boundaries and improve exploration.
When `selection_pool_size=1`, this option effectively reduces to TPS-like
selection, where transitions have 100% acceptance rate in the pool, and
non-transitions are always rejected.""" 
                 })
    
    always_select_inside_the_bins: bool = field(
        default=False,
        metadata={'description':
"""If True: ensure you always select from paths with values in the current
selection bins. This to prevent the simulations from getting stuck close to the
state boundaries."""
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
Recommended for `selection_pool_size > 1`. Values: '' (disable),
'all' (apply to both edges), or directly the state names towards which you
want the extension to happen ("A", "B", "AB", etc.)."""
                 })
    
    density_adjustment: Number = field(
        default=inf,
        metadata={'description':
"""Apply a correction to the density during selection to accelerate
convergence. For each shooting chain, in each bin: multiply the density by
the number of the latest `density_adjustment` shooting points already
selected in the bin. It can be combined with `density_adjustment`."""
                 })
    
    shared_density_adjustment: bool = field(
        default=False,
        metadata={'description':
"""If True: apply a correction to the density during selection to accelerate
convergence. For each shooting chain, in each bin: multiply the density by
the number of shooting points already selected in the bin from all chains
managed by the same worker, plus with that of all the shooting points currently
being employed in a path sampling simulation. It can be combined with
`local_density_adjustment`."""
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

    free_seeding_position: str = field(
        default='boundary',
        metadata={'description':
"""Where inside the state the FIRST free simulation of each state starts.

Applies only to the first free trajectory of a state - the one seeded from
`initial_paths`, before any AIMMD ensemble exists. Where the later trajectories
restart from is `free_restart_source`, which is independent of this.

The candidates are the frames of the initial path that belong to the target
state and lie on that state's side of the transition boundary. This field picks
one of them by fractional position, measured from the far side of the state
towards the reactive region:

- 0.0  the frame furthest from the reactive region - as deep into the state as
       the initial path reaches.
- 0.5  the middle of the run.
- 1.0  the frame adjacent to the reactive region. DEFAULT, and the behaviour of
       every AIMMD version so far.

The fraction is a position among that state's own frames, so 0.0 always means
"deep" and 1.0 always means "at the boundary", for A and for B alike. For a
state sitting at the end of the path (normally B) that is the mirror image of
the file order. The index is `int(fraction * (n - 1) + 0.5)` over the candidates
ordered far-side-first, so a single candidate always yields itself, no value can
fall outside the state, and two candidates tie towards the boundary, i.e.
towards the historical behaviour.

A brief recrossing does NOT truncate the candidates. If the path pops out of the
state for a frame or two and comes straight back, those frames are skipped and
the ones behind them still count. calixarene-G5's initial path is the reason:

    BBBBBBBRRRRRRRRRRAAAAAAAAA R AAAAAAAAAAAAAAAAAAAAAAA
           boundary=17 ^       ^26                    ^49

Frame 26 sits at 6.35 A against a 5.5 A boundary - one frame - and drops back to
5.37 A. Stopping at it would leave 'deepest' pointing at frame 25 (4.97 A) and
hide the 23 deeper frames behind it, the deepest being frame 49 at 3.64 A: 1.33 A
of depth given up to a single-frame excursion, more than the entire seed offset
this field exists to remove. Frames outside the state are excluded from the
candidates rather than merely traversed, so an intermediate position can never
land on a reactive frame and start the free simulation outside the basin.

A path with NO transition at all - what a brute-force shooting setup uses, in
practice every frame in the reactive state - has neither a boundary nor an
in-state side, so a position cannot be placed in it. Setting anything other
than 'boundary' for such a run raises ValueError: the choice is meaningless
there rather than merely awkward.

Accepted values:

- a float in [0, 1], or a string that parses as one, e.g. 0.25
- 'boundary'   = 1.0   DEFAULT
- 'middle'     = 0.5
- 'deepest'    = 0.0
- 'random'     a frame drawn uniformly from the run, drawn from a seed derived
               from the worker index so a requeued worker reproduces its own
               choice
- a dict keyed by state letter for per-state control, e.g.
  {'A': 'deepest', 'B': 'boundary'}. States not named keep the default.

Only the two end states have an in-state run; a value given for the reactive
state `states[1]` is ignored.

Ignored for a state whose `free_restart_source` is 'transitions', which - as the
deprecated `restart_free_simulations_with_transitions` did - replaces the first
seed too, with a randomly chosen initial path.

Any value other than 'boundary' needs initial-path frames that the transition
trim in `Params._process_and_check` removes, so the untrimmed paths are
retained internally at load time; 'boundary' uses the trimmed path unchanged
and takes the historical code path.

Why it matters. The trim replaces each initial path by its transition block,
and because `Path.split()` overlaps neighbouring blocks by two frames that
block begins at the LAST in-state frame before the reactive region. The first
free simulation therefore starts at the state boundary, however deep the
initial path reached - which is fine when the boundary sits just outside the
bound minimum, and badly wrong when it does not."""
                 })

    free_restart_source: str = field(
        default='crossing',
        metadata={'description':
"""Where each free simulation AFTER the first one restarts from.

A free simulation stops when it commits to a state, and AIMMD immediately starts
the next one. This field says where that next configuration comes from. It is
independent of `free_seeding_position`, which governs only the first trajectory
of each state.

Accepted values:

- 'crossing'     The last frame the previous trajectory spent in the target
                 state, i.e. the configuration it escaped from. DEFAULT, and the
                 historical behaviour. That frame lies ON the state boundary, so
                 every first passage starts from the boundary-entry distribution
                 rather than from the equilibrium distribution inside the state.
                 The two agree only when relaxation inside the state is fast
                 compared with the escape time; when the state holds
                 sub-populations that interconvert slowly (a deep core and a
                 weakly bound outer shell, say) this restarts every observation
                 in the escape-prone shell, the first-passage times stop being
                 exponential, and the rate estimate is biased fast.
- 'seed'         The same frame the first seeding used, re-derived from the
                 initial path through `free_seeding_position`. Every trajectory
                 of the state then starts from one fixed configuration with
                 fresh velocities.
- 'basin'        A frame drawn uniformly from the in-state frames AIMMD has
                 accumulated, i.e. from the *biased* equilibrium inside the
                 state - the occupancy measure the Tiwary-Parrinello boosted
                 clock assumes. Needs no bias data.
- 'equilibrium'  The same pool, drawn with probability proportional to
                 exp(bias), i.e. from the *unbiased* (Boltzmann) equilibrium
                 inside the state. This matches the textbook rate - mean first
                 passage from equilibrium in A - and is what an unbiased-MD or
                 OPES-flooding reference measures. With no recorded bias every
                 weight is 1 and this coincides with 'basin', which is correct:
                 an unbiased trajectory already samples Boltzmann.
- 'transitions'  The end frames of a randomly sampled AIMMD transition path.
                 This is what the deprecated
                 `restart_free_simulations_with_transitions` selected, and like
                 that flag it also replaces the FIRST seed, falling back to a
                 randomly chosen initial path while no transition has been
                 sampled yet.
- a dict keyed by state letter for per-state control, e.g.
  {'A': 'equilibrium', 'B': 'crossing'}. States not named keep the default.

'seed', 'basin' and 'equilibrium' draw from inside the state and are meaningless
for the reactive state `states[1]`; requesting one of them there leaves that
state on 'crossing'.

See also `free_restart_min_frames`."""
                 })

    restart_free_simulations_with_transitions: str = field(
        default='',
        metadata={'description':
"""DEPRECATED - use `free_restart_source` instead.

Restart free simulations from AIMMD-sampled transitions for selected states.
Accepted elements are a list of states in capital letters (eg. 'AB'), or 'all'.

This is the same choice as `free_restart_source = 'transitions'`, so it is now
read through that field: `'all'` means `free_restart_source = 'transitions'` and
`'AB'` means `{'A': 'transitions', 'B': 'transitions'}`. Setting it to a
non-empty value still works and still does exactly what it used to - including
replacing the first seed and falling back to a randomly chosen initial path
while no transition has been sampled - but raises a DeprecationWarning naming
the replacement. Setting both it and a non-default `free_restart_source` for the
same state is an error."""
                 })

    free_restart_min_frames: int = field(
        default=0,
        metadata={'description':
"""Minimum number of in-state frames the pool must hold before the `'basin'` /
`'equilibrium'` sources of `free_restart_source` are used. Below it, the free
worker falls back to the boundary exit frame (`'crossing'`) for that restart.
Default 0 (no requirement). Raise it if you would rather keep the historical
behaviour until a meaningful in-basin sample exists."""
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
Tiwary-Parrinello assumption that the bias is negligible inside R.

In a multi-system run this may be a single float (applied to every system) or a
list of floats, one per entry of `system_ids` (each system's bias is checked
against its own threshold)."""
                 })

    # ------------------------------------------------------------------
    # Multi-system (multi-ligand) shared-committor options
    # ------------------------------------------------------------------

    multi_system: bool = field(
        default=False,
        metadata={'description':
"""Enable multi-system (multi-ligand) mode. When False (default) AIMMD behaves
exactly as a single-system run. When True, one params file orchestrates several
chemical systems at once: per-system fields (`topology`, `initial_paths`) become
lists (one entry per system), each system runs in its own subfolder
`<run>/<system_id>/`, and the user data functions (`states_function`,
`descriptors_function`, `values_function`) receive an extra `system_id` keyword
(passed only if their signature accepts it, so existing single-system functions
keep working unchanged)."""
                 })

    multi_system_share_network: bool = field(
        default=False,
        metadata={'description':
"""Train ONE shared committor network across all systems (only meaningful when
`multi_system=True`). When False, every system gets its own network file and its
own trainer (the trainers may share a GPU, see `trainers_share_gpu`). When True,
a single network is trained by one trainer that passes the `fit` function a LIST
of per-system PathEnsembles; the shared network is stored once at the run-folder
root and read by every system's shooting workers. Rates/kinetics are still
computed per system, in sequence."""
                 })

    system_ids: List = field(
        default_factory=lambda: [],
        metadata={'description':
"""Per-system labels for a multi-system run (e.g. ['G2', 'G4']). They name the
per-system subfolders `<run>/<system_id>/` and index the per-system entries of
list-valued fields (`topology`, `initial_paths`). If left empty in multi-system
mode, defaults to ['0', '1', ...] inferred from the number of topologies."""
                 })

    atom_types: List = field(
        default=None,
        metadata={'description':
"""Fixed, ordered list of MDAnalysis atom-type strings defining the shared
one-hot graph node encoding (e.g. ['H','C','N','O','F','NA','P','S','CL','BR','I']).
When None (default) the graph encoding is derived per-universe from
`sorted(set(universe.atoms.types))` (legacy single-system behaviour). A fixed
table is what lets one graph network consume graphs from multiple systems: every
system encodes into the same columns and the network input width must equal
`len(atom_types)`. Forwarded to `aimmd.network.graph_utils.get_graphs_pyg`."""
                 })

    trainers_share_gpu: bool = field(
        default=True,
        metadata={'description':
"""When `multi_system=True` and `multi_system_share_network=False`, each system
has its own trainer. If True (default), all of a run's per-system trainers are
bound to the SAME GPU (one shared device); if False they are spread across
distinct GPUs. Ignored when a single shared network is trained (then there is
only one trainer)."""
                 })

    # ------------------------------------------------------------------
    # Training-time value-pass subsampling (optional; speeds up large runs)
    # ------------------------------------------------------------------

    subsample_caps: dict = field(
        default=None,
        metadata={'description':
"""Optional caps that bound the per-round committor *value pass* (and the bin
generation / reweighting / rate estimate that consume it) by evaluating them on a
randomly subsampled slice of the path ensemble instead of every reactive frame.
`fit` (network training) is unaffected and still sees the full ensemble.

When None (default) NO subsampling happens and behaviour is identical to before.
When a dict, the recognised keys are:

- 'shot'     : max number of PATHS kept *per shot-excursion direction-type*.
               Each of sAA, sAB, sBA, sBB is capped independently, so a value of
               100 keeps up to 4*100 = 400 shot paths per system.
- 'free'     : max number of PATHS kept *per free-excursion direction-type*.
               Each of fAA, fAB, fBA, fBB is capped independently, so a value of
               500 keeps up to 4*500 = 2000 free paths per system.
- 'in_state' : max number of FRAMES kept per state (the in-A and in-B paths are
               kept until this many frames are reached, per state, per system).

A missing key means that category is left uncapped. The subsample is drawn fresh
each round (uniformly within each category, so reweighting stays consistent;
in-state-only paths carry zero reweight so never bias the rate). In a multi-system
run this may be a single dict (applied to every system) or a list of dicts/None,
one per entry of `system_ids`. Pick caps generously (e.g. shot=100, free=500)."""
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
