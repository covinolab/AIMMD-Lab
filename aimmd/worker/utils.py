"""
aimmd.worker.utils
=================

Low-level utilities for AIMMD worker tasks.

This module collects helper functions used by worker task implementations,
primarily :mod:`aimmd.worker._shoot` and :mod:`aimmd.worker._train`.

Scope and role in AIMMD
-----------------------
The functions in this file support **path sampling simulations**, which are the
core of AIMMD's enhanced-sampling strategy. In particular, they implement the
mechanics needed to:

- process the initial paths for shooting chains, sweeping, and free simulations,
- maintain a *selection pool* of candidate paths/frames,
- select shooting points in a way that is guided by the current committor model
  (typically a neural network) and by adaptive density targets,
- register newly generated paths on disk and in memory,
- (optionally) apply TPS-style acceptance/rejection steps.

Importantly, the objective is not simply to generate equilibrium transition
events, but to produce a **diverse set of reactive trajectories** and enrich
sampling in the reactive region (often yielding more transition-like path
segments than would be observed in an equilibrium trajectory of comparable cost).

Key concepts
------------
Selection pool
    A limited-size :class:`~aimmd.pathensemble.PathEnsemble` that stores a set of
    candidate paths used to propose shooting points.

    The pool is updated as sampling proceeds. Conceptually, it serves as a
    short-memory buffer that keeps the proposal distribution responsive to the
    most recent sampling outcomes while avoiding repeated use of the same source
    path. In particular, pool updates:

    - **append the latest valid chain path** (typically the most recent path with
      non-zero weight), so newly discovered reactive behavior can immediately
      influence where future shooting points are drawn;

    - **re-seed the pool** from ``initial_paths`` when underfilled, ensuring the
      algorithm can continue proposing shooting points even early in a run or
      after aggressive pruning;

    - **enforce a hard pool size** by discarding the oldest entries, keeping the
      memory footprint bounded and preventing the pool from drifting into a
      long-history archive;

    - optionally **enforce presence of transitions**: if requested, the pool is
      guaranteed to contain at least one transition-like path (in either
      direction) by re-inserting a suitable element from the chain. This is a
      practical safeguard to prevent the proposal distribution from collapsing
      onto non-transition excursions when transitions are rare.

Committor-guided shooting-point selection
    When shooting from the reactive state, the selection procedure uses current
    network-evaluated values (committor-like) together with adaptive bins and
    density targets to bias the choice toward informative regions. The algorithm
    can additionally adjust for current chain populations and apply optional
    Lorentzian penalization around the origin in value space.

Atomic persistence of paths and caches
    Path registration writes trajectories via a temporary file and then renames
    it, minimizing disruption from interruptions. Associated cached arrays
    (states/descriptors) are written as `.npy` files, and the relevant MDAnalysis
    readers are evicted from the global caches to ensure subsequent reads see
    the final files.

Notes
-----
- This module relies on the global caches exposed in :mod:`aimmd._config`
  (notably :data:`~aimmd._config.NPY_CACHE` and :data:`~aimmd._config.MDA_CACHE`).
  Several functions deliberately clear or evict entries to avoid stale reads.
- Some functions operate on "private" attributes of :class:`PathEnsemble`
  (e.g., ``._paths``). This is intentional in the worker layer, where
  performance and control over sampling bookkeeping take precedence.
"""

# external
import os
import numpy as np
import torch
from glob import glob
from math import inf
from pathlib import Path as PosixPath

# aimmd imports
from ..path import Path
from .._config import NPY_CACHE, MDA_CACHE, print
from ..cache.npy import save_npy, load_npy
from ..core.utils import now, process_state, remove
from ..path.utils import get_cache_fname, read_sweep_frame, write_sweep_frame
from ..pathensemble import PathEnsemble
from ..params.utils import (SEEDING_POSITION_ALIASES,
                             SEEDING_POSITION_RANDOM)
from ..execute.utils import execute_command
from ..analysis.utils import bin_centers, merge_empty_bins
from ..network.rescale_utils import rescale


def get_initial_transitions_for_shooting_chain(initial_paths, states='ARB'):
    """
    Given an `aimmd.path.PathEnsemble instance`, extract a transition from
    each path for the purpose of initializing a shooting chain.

    Parameters
    ----------
    initial_paths : aimmd.path.PathEnsemble or aimmd.path.Path
        The pathensemble from which the transitions is extracted.
    states : str, optional
        3-char string containing the initial, reactive, and final states in
        order, default is `'ARB'`.
    
    Returns
    -------
    transitions : aimmd.path.PathEnsemble
        The transition segments collected in a `PathEnsemble` instance.
    
    Notes
    -----
    If each path contains more than one transition, only the first one will be
    considered.
    """
    initial_paths = PathEnsemble(initial_paths)
    transitions = PathEnsemble()
    for path in initial_paths:
        try:
            transition = path.split().extract(states, states[::-1])[0]
        except Exception as exception:
            raise RuntimeError(f'could not extract {states} transition '
                               f'from {path}: {exception}')
        transitions._paths.append(transition)
    return transitions


def state_run_locs(states, boundary_loc, at_start):
    """
    Locations of the in-state run a transition departs from, far side first.

    Parameters
    ----------
    states : array of str
        Per-frame state labels of the untrimmed initial path.
    boundary_loc : int
        Location of the in-state frame adjacent to the reactive region, i.e.
        the first frame of the transition block the initial-path trim keeps.
    at_start : bool
        True when the state's run precedes the reactive region in file order
        (the usual case for `states[0]`), False when it follows it (`states[-1]`).

    Returns
    -------
    list of int
        The maximal run of the same state label containing `boundary_loc`,
        ordered **far side first**: index 0 is the frame furthest from the
        reactive region and the last index is `boundary_loc`. A leading or
        trailing excursion out of the state is therefore never swept in.
    """
    states = np.asarray(states)
    label = states[boundary_loc]
    if at_start:
        low = boundary_loc
        while low > 0 and states[low - 1] == label:
            low -= 1
        return list(range(low, boundary_loc + 1))
    high = boundary_loc
    while high + 1 < len(states) and states[high + 1] == label:
        high += 1
    return list(range(high, boundary_loc - 1, -1))


def seed_index_in_run(n, position):
    """
    Index into a state's run of frames for a fractional seeding position.

    `position` is measured over the run ordered far-side-first, so 0.0 is the
    frame furthest from the reactive region and 1.0 the one adjacent to it.

    Uses ``int(position * (n - 1) + 0.5)`` rather than `round`, whose
    banker's rounding sends ``round(0.5)`` to 0 and would make a two-frame run
    seed at the *far* frame for ``position=0.5``. Ties therefore go towards the
    boundary, i.e. towards the historical behaviour.
    """
    if n <= 1:
        return 0
    return int(position * (n - 1) + 0.5)


def transition_block(path, states):
    """
    The transition block of `path`, i.e. what the initial-path trim keeps.

    Reproduces `Params._process_and_check`: the first `path.split()` block
    whose ``type[:3]`` is a transition. Returned as a view on `path`, so its
    `locs` are locations in `path`'s own file - which is what makes it usable
    to place a seed inside the untrimmed path. Returns None when the path holds
    no transition.
    """
    for block in path.split():
        # `type` is normally the compact str `_process_and_check` compares
        # against; coerce so a per-frame array cannot raise an ambiguous-truth
        # ValueError out of the `in` test
        block_type = ''.join(np.atleast_1d(block.type).astype(str))[:3]
        if block_type in (states, states[::-1]):
            return block
    return None


def _match_untrimmed(untrimmed_paths, path, states):
    """
    The untrimmed path `path` was trimmed and written out from.

    `untrimmed_paths` is keyed by the base name of the source file, as
    returned by `params.untrimmed_initial_paths()`. A worker reads its initial
    paths back from ``<run>/initial<states>/``, where the Launcher wrote them
    with the trajectory extension appended, so the worker-side base name has
    the source base name as a prefix rather than being equal to it.
    """
    if not untrimmed_paths:
        raise TypeError(
            'seeding away from the state boundary needs the initial paths as '
            'they were before the transition trim, which removes exactly '
            'those frames; pass '
            'untrimmed_paths=params.untrimmed_initial_paths()')
    base = os.path.basename(str(path.fname))
    if base in untrimmed_paths:
        return untrimmed_paths[base]
    matches = [value for key, value in untrimmed_paths.items()
               if base.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if len(untrimmed_paths) == 1:
        return next(iter(untrimmed_paths.values()))
    raise TypeError(
        f'cannot tell which initial path {base!r} was trimmed from; '
        f'candidates are {sorted(untrimmed_paths)}')


def _couple_in_state_run(untrimmed, target_state, states, position, rng):
    """
    Build a ``(history, seed)`` couple at `position` inside the state's run.

    The couple's last frame is the seed `Params.initialize_simulation` starts
    the MD from; the first is its neighbour on the boundary side, written as
    the ``.part0000`` history segment. When the seed is the boundary frame that
    neighbour is the adjacent reactive frame, exactly as in the historical
    behaviour, so the couple is always two frames long.

    The state boundary is located by re-deriving the transition block inside
    `untrimmed`, NOT from the trimmed path the caller holds: a worker reads its
    initial paths back from the copies the Launcher wrote, whose `locs` restart
    at 0 and so cannot index the untrimmed file.
    """
    block = transition_block(untrimmed, states)
    if block is None:
        raise RuntimeError(
            f'{untrimmed.fname} holds no {states!r} transition, so there is no '
            f'state boundary to place a seed relative to')
    at_start = block.initial('states') == target_state
    boundary_loc = block.locs[0] if at_start else block.locs[-1]
    run = state_run_locs(untrimmed.states, boundary_loc, at_start)

    if position == SEEDING_POSITION_RANDOM:
        if rng is None:
            index = int(np.random.randint(len(run)))
        else:
            index = int(rng.integers(len(run)))
    else:
        index = seed_index_in_run(len(run), position)
    seed_loc = run[index]

    if at_start:
        # [seed, seed+1] reversed -> (history on the boundary side, seed)
        return untrimmed[seed_loc:seed_loc + 2][::-1]
    return untrimmed[seed_loc - 1:seed_loc + 1]


def get_initial_frames_for_free_simulations(
        initial_paths, target_state, reactive_state,
        position=1.0, untrimmed_paths=None, states=None, rng=None):
    """
    Given an `aimmd.path.PathEnsemble` instance, extract two consecutive frames
    from each path for the purporse of launching free simulations. The first
    frame of each couple is the `.part0000` history frame, and the second is
    internal to `target_state` and is the frame the MD starts from.

    Parameters
    ----------
    initial_paths : aimmd.path.PathEnsemble or aimmd.path.Path
        The paths from which the frames are extracted. These are the *trimmed*
        initial paths, i.e. transition blocks, so each one already begins (or
        ends) at the in-state frame adjacent to the reactive region.
    target_state : str of size 1
        The state of the second frame, e.g. `'A'`.
    reactive_state : str of size 1
        The state of the first frame, e.g. `'R'`.
    position : float or str, optional
        Where inside the state's run of frames to seed, as a fraction ordered
        far-side-first: 0.0 is the frame furthest from the reactive region and
        1.0 (the default) the one adjacent to it. ``'random'`` draws uniformly
        over the run. See `params.free_seeding_position`.
    untrimmed_paths : dict, optional
        The initial paths before the transition trim, keyed by the base name of
        the source file, as returned by `params.untrimmed_initial_paths()`.
        Required for every `position` other than 1.0, because the trim removes
        exactly the frames those positions ask for. Ignored when
        ``position == 1.0``.
    states : str, optional
        ``params.states``. Required for every `position` other than 1.0, to
        re-derive where the transition block starts inside the untrimmed path.
    rng : numpy.random.Generator, optional
        Source of randomness for ``position='random'``. Defaults to NumPy's
        global RNG, like the rest of the worker layer.

    Returns
    -------
    initial_frames : aimmd.path.PathEnsemble
        `PathEnsemble` with the same length of `initial_paths`, where each
        element contains the frame couples extracted from the paths.

    Notes
    -----
    With ``position == 1.0`` this is the historical implementation, unchanged
    and taking the same code path: the two frames in a couple are either at the
    beginning or at the end of the origin path, and `untrimmed_paths` is never
    consulted. If the path does not allow for it, this function throws an
    error.
    """
    initial_paths = PathEnsemble(initial_paths)
    initial_frames = PathEnsemble()
    at_boundary = (not isinstance(position, str)
                   and float(position) == SEEDING_POSITION_ALIASES['boundary'])

    if not at_boundary and target_state != reactive_state and not states:
        raise TypeError(
            f'seeding at position {position!r} needs states=params.states to '
            f'locate the transition block inside the untrimmed initial path')

    # extract frames for each path
    for path in initial_paths:
        if target_state == reactive_state:
            # the reactive state has no in-state run to place a seed in
            if np.random.random() > .5:
                initial_frames._paths.append(path[:+2])
            else:
                initial_frames._paths.append(path[:-3:-1])
        elif at_boundary:
            if path.initial('states') == target_state:
                initial_frames._paths.append(path[1::-1])
            else:
                initial_frames._paths.append(path[-2:])
        else:
            initial_frames._paths.append(
                _couple_in_state_run(
                    _match_untrimmed(untrimmed_paths, path, states),
                    target_state, states, position, rng))

        # check
        states = initial_frames[-1].states
        if states[0] != reactive_state and states[1] != target_state:
            raise RuntimeError(f'{path.fname} must allow to extract a'
                            f'"{reactive_state}{target_state}" segment')
    return initial_frames


def get_basin_frames_for_free_restart(
        free_trajectories, target_state, reactive_state,
        weighting='occupancy', min_frames=0):
    """
    Draw a two-frame free-simulation restart segment from inside a basin.

    The free worker's default restart takes the last frame the previous free
    trajectory spent in the target state — the configuration it escaped from,
    which lies on the state boundary. This function instead draws the restart
    configuration from the frames the accumulated free trajectories of that
    state actually spent *inside* it, so a new first passage starts from the
    state's own occupancy measure rather than from its boundary.

    Parameters
    ----------
    free_trajectories : iterable of aimmd.path.Path
        Free trajectories of the target state (unsplit), most usefully every
        trajectory of that state seen so far, including the one that has just
        finished. Trajectories with no usable state labels are skipped.
    target_state : str of size 1
        The state to draw from, e.g. `'A'`.
    reactive_state : str of size 1
        The reactive-region label, e.g. `'R'`. Drawing is refused when it
        equals `target_state`.
    weighting : {'occupancy', 'unbiased'}, optional
        `'occupancy'` (default) draws uniformly over in-state frames, i.e. from
        the biased equilibrium inside the state — the distribution the boosted
        Tiwary-Parrinello clock assumes. `'unbiased'` draws with probability
        proportional to `exp(bias)`, i.e. from the unbiased equilibrium; it
        degrades to `'occupancy'` if any candidate lacks a bias cache.
    min_frames : int, optional
        Refuse to draw when fewer than this many in-state frames are available.
        Default 0 (draw as soon as one frame qualifies).

    Returns
    -------
    initial_frames : aimmd.path.Path or None
        Two consecutive frames `(j - 1, j)` of one candidate trajectory, with
        frame `j` in `target_state`. `Params.initialize_simulation` starts from
        the last frame and writes the first as the `.part0000` history frame, so
        the pair is in forward time order: the history frame is the seed's real
        predecessor. None when no draw was possible; the caller must then fall
        back to its own behaviour.
    seed_bias : numpy.ndarray or None
        One-element array holding the bias (kT) of the `.part0000` history
        frame, when it is known. The caller writes it to the seed's bias cache;
        an in-basin history frame carries a large bias and approximating it as 0
        would bias short trajectories' γ downward. None when unknown.

    Notes
    -----
    - Frames are selected from `path._get('states')`, which masks frames beyond
      `path._exclude_from`, so indicted (corrupt) frames are never drawn.
    - Frame index 0 of a trajectory is never drawn: a history frame is needed.
    - Weights are normalised in a two-level draw (trajectory, then frame within
      it) which is equivalent to one draw over the pooled frames.
    - Cost is one pass over the cached state arrays of the pool, paid once per
      completed free trajectory.
    - Uses NumPy's global RNG, like the rest of the worker layer; seed it with
      `np.random.seed(...)` for reproducibility.
    """
    t = target_state
    if t == reactive_state:
        return None, None

    # collect candidates: (path, in-state indices, bias array or None)
    candidates = []
    have_bias = True
    for path in free_trajectories:
        if path is None or len(path) < 2:
            continue
        try:
            states = np.asarray(path._get('states'))
        except Exception:
            continue
        if len(states) < 2:
            continue
        # index 0 cannot be drawn: it has no predecessor to act as history
        indices = np.flatnonzero(states[1:] == t) + 1
        if not indices.size:
            continue
        bias = None
        if weighting == 'unbiased':
            try:
                bias = np.asarray(
                    path._get('bias', raise_if_missing=True), dtype=float)
            except Exception:
                bias = None
            if bias is None or len(bias) < len(states):
                have_bias = False
        candidates.append((path, indices, bias))

    if not candidates:
        return None, None
    n_total = int(sum(len(indices) for _, indices, _ in candidates))
    if n_total < max(1, int(min_frames)):
        return None, None

    # per-frame weights
    if weighting == 'unbiased' and have_bias:
        # subtract the global maximum before exponentiating: the fill can be
        # many kT deep and exp() would otherwise overflow
        shift = max(float(np.max(bias[indices]))
                    for _, indices, bias in candidates)
        per_frame = [np.exp(bias[indices] - shift)
                     for _, indices, bias in candidates]
    else:
        if weighting == 'unbiased':
            print('Warning: the \'equilibrium\' free restart source needs a '
                  'bias cache for every candidate trajectory; some are missing, '
                  'so this draw falls back to \'basin\' (occupancy) '
                  'weighting instead')
        per_frame = [np.ones(len(indices), dtype=float)
                     for _, indices, _ in candidates]

    totals = np.array([float(w.sum()) for w in per_frame])
    if not np.isfinite(totals).all() or totals.sum() <= 0.0:
        return None, None

    # two-level draw == one draw over the pooled frames
    k = int(np.random.choice(len(candidates), p=totals / totals.sum()))
    path, indices, bias = candidates[k]
    w = per_frame[k]
    j = int(np.random.choice(indices, p=w / w.sum()))

    initial_frames = path[j - 1:j + 1]
    try:
        if len(initial_frames) != 2 or initial_frames.states[1] != t:
            return None, None
    except Exception:
        return None, None

    # bias of the .part0000 history frame (frame j - 1), when known
    seed_bias = None
    if bias is None:
        try:
            bias = np.asarray(
                path._get('bias', raise_if_missing=True), dtype=float)
        except Exception:
            bias = None
    if bias is not None and len(bias) > j - 1:
        seed_bias = np.array([float(bias[j - 1])], dtype=float)
    return initial_frames, seed_bias


def get_initial_frames_for_training(initial_paths, states='ARB'):
    """
    Given an `aimmd.path.PathEnsemble` instance, extract all frames in the
    reactive and product states for the purpose of priming a machine learning
    model of the committor between the two.

    Parameters
    ----------
    initial_paths : aimmd.path.PathEnsemble or aimmd.path.Path
        The pathensemble from which the frames are extracted.
    states : str, optional
        3-char string containing the initial, reactive, and final states in
        order, default is `'ARB'`. The middle character is not considered.
    
    Returns
    -------
    frames : aimmd.path.PathEnsemble
        The training set frames collected in a `PathEnsemble` instance.
    
    Notes
    -----
    If there are no frames in either state, the function throws an error.
    """
    initial_paths = PathEnsemble(initial_paths).join()
    initial_states = initial_paths.states
    keepers1 = np.flatnonzero(initial_states == states[+0])
    keepers2 = np.flatnonzero(initial_states == states[-1])
    if not len(keepers1):
        raise RuntimeError(f'no frames in state {states[+0]} for input paths')
    if not len(keepers2):
        raise RuntimeError(f'no frames in state {states[-1]} for input paths')
    return PathEnsemble(
        [initial_paths[k:k+1] for k in keepers1] +
        [initial_paths[k:k+1] for k in keepers2]
    )


def is_initial_path(path):
    """
    Checks wether `path` comes from the initial paths' folder.

    Parameters
    ----------
    path : aimmd.path.Path

    Returns
    -------
    is_initial : bool
    """
    return PosixPath(path.fname).root.startswith('initial')


def update_selection_pool(pool, size, chain,
                          initial_paths=None,
                          at_least_one='',
                          boundaries=[-inf, +inf]):
    """
    Update a selection pool of candidate paths.

    The pool is used by shooting-point selection to propose candidates. The
    update procedure:

    1) Appends the most recently produced valid path from ``chain`` (if present
       and not already the pool's current source).
    2) If the pool is underfilled, optionally pre-pends paths from
       ``initial_paths`` to ensure at least roughly half of the desired pool
       size is available.
    3) Trims the pool to the requested maximum size.
    4) Optionally enforces that at least one transition is present in the pool
       (if ``at_least_one`` is non-empty).

    Parameters
    ----------
    pool : aimmd.pathensemble.PathEnsemble
        The current selection pool (mutated in place).
    size : int
        Maximum pool size after update.
    chain : aimmd.pathensemble.PathEnsemble
        Shooting chain that may contain the most recent accepted/valid path.
        The attribute ``chain.path`` is used as the candidate "last path".
    initial_paths : aimmd.pathensemble.PathEnsemble, optional
        Fallback path ensemble used to seed the pool when it is underfilled.
        If ``None``, underfilling is not corrected.
    at_least_one : str, optional
        If non-empty, interpreted as a transition "type" signature. The pool is
        checked for at least one transition of type ``at_least_one`` or its
        reverse. If absent, a transition is searched in ``chain`` (from newest
        to oldest) and prepended to the pool. Default is ``''`` (no constraint).
    boundaries : [float, float] iterable, optional
        Paths in pool must have at least one frame between boundaries.
        Default is [-inf, +inf].

    Returns
    -------
    aimmd.pathensemble.PathEnsemble
        The updated pool (same object as input).

    Notes
    -----
    - This function may manipulate ``pool._paths`` directly for performance.
    - The "last ok path" criterion is implemented as ``chain.path`` and a check
      that the path's ``fname`` differs from ``pool.fname`` (avoid duplicates).
    """

    # update with last ok path (weight != 0)
    if chain and (path := chain.path) and path.fname != pool.fname:
        # pool element that produced new path was already removed
        # thus we can just safely append to current pool
        pool.append(path)

    # replicate up to half size
    missing = max((size + 1) // 2 - len(pool), 0)
    if missing and initial_paths is not None:
        length = len(initial_paths)
        while missing:
            missing -= 1
            pool._paths.insert(0, initial_paths._paths[missing % length])

    # remove selected element in the chain or first one
    while len(pool) > size:
        pool.pop(0)

    # re-add transition if required
    if at_least_one and chain and (
        not pool.extract(at_least_one, at_least_one[::-1])):
        for path in chain._paths[::-1]:
            if path.type[:3] in (at_least_one, at_least_one[::-1]):
                pool._paths = [path] + pool._paths
                break
    
    # replace elements without values between vmin, vmax
    if boundaries is not None:
        vmin, vmax = boundaries[0], boundaries[-1]
    else:
        vmin, vmax = -inf, +inf
    if vmin > -inf or vmax < +inf:
        pool_fnames = [path.fname for path in pool] 
        j = len(chain) - 1  # current chain index
        for i in range(len(pool)):
            if is_initial_path(pool[i]):
                continue
            values = pool[i].internal('values')
            if not ((values >= vmin) * (values <= vmax)).any():
                # find new element going backwards
                replaced = False
                for j in range(j, -1, -1):
                    path = chain._paths[j]
                    # do not re-add old elements
                    if path.fname in pool_fnames:
                        continue
                    values = path.values
                    if not ((values >= vmin) * (values <= vmax)).any():
                        continue
                    # replace
                    print(f'!!! replacing selection pool element '
                          f'{pool_fnames[i]} with values outside the '
                          f'[{vmin:.3f}, {vmax:.3f}] range with {path.fname}')
                    pool[i] = path
                    replaced = True
                    j -= 1
                    break
                # no replacement? use one of the initial paths (if any)
                if not replaced:
                    if initial_paths:
                        path = initial_paths[
                            np.random.choice(len(initial_paths))]
                        print(f'!!! replacing selection pool element '
                          f'{pool_fnames[i]} with values inside the '
                          f'[{vmin:.3f}, {vmax:.3f}] range with {path.fname}')
                        pool[i] = path
                    else:
                        print(f'!!! warning: could not replace selection pool '
                              f'element {pool_fnames[i]} with values inside '
                              f'the [{vmin:.3f}, {vmax:.3f}] range')

    return pool


def rescale_bins(bins, knots, values):
    """
    Rescale bin edges in place using a committor rescaling map.

    This helper is used when the committor (or committor-like) values are
    rescaled (e.g., to match an estimated crossing probability). The intent is
    to keep the binning range aligned with the rescaled value range.

    The function:

    - identifies the finite range of the bin array (skipping ``±inf`` edges if
      present and if interior finite edges exist),
    - rescales the lower/upper endpoints through :func:`rescale`,
    - replaces the finite portion with a new ``np.linspace`` between the
      rescaled endpoints.

    Parameters
    ----------
    bins : numpy.ndarray
        Bin boundaries (mutated in place).
    knots : array-like
        Knot locations defining the rescaling interpolation.
    values : array-like
        Values at the knots defining the rescaling interpolation.

    Returns
    -------
    None
    """
    if not len(bins):
        return
    i = 0
    j = len(bins)
    a, b = bins[[i, j - 1]]
    if j - 1:
        if a == -inf and not np.isinf(bins[+1]):
            a = bins[+1]
            i = 1
        if b == +inf and not np.isinf(bins[-2]):
            b = bins[-2]
            j -= 1
    a, b = rescale([a, b], knots, values)
    bins[i:j] = np.linspace(a, b, j - i)


def _read_colvar_for_register(traj_fname, bias_function):
    """Read a COLVAR file associated with a trajectory segment.

    Derives the COLVAR filename via ``traj_fname.replace('.xtc', '_COLVAR')``
    (consistent with the rename performed by ``params.run_simulation`` when
    ``bias_source='file'``), reads the raw data rows and the ``#! FIELDS``
    header, and calls ``bias_function(traj_fname)`` to obtain the per-frame
    bias in kT (unit conversion is the caller's responsibility).

    Parameters
    ----------
    traj_fname : str
        Trajectory segment filename (e.g. ``'chainR0/back.xtc'``).
    bias_function : callable
        ``params.bias_function`` — maps trajectory filename → ndarray in kT.

    Returns
    -------
    bias_kT : numpy.ndarray, shape (n_frames,)
        Per-frame bias in kT as returned by ``bias_function``.
    all_rows : numpy.ndarray, shape (n_frames, n_cols)
        All COLVAR columns (time, CV(s), bias, …) as a float array.
    header : str
        The ``#! FIELDS …`` header line, without trailing newline.
    """
    # Derive COLVAR filename from trajectory name (drop extension, add _COLVAR)
    ext = os.path.splitext(traj_fname)[1]            # e.g. '.xtc'
    colvar_fname = traj_fname.replace(ext, '_COLVAR')

    # Read all data rows and header
    header = ''
    with open(colvar_fname) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#'):
                header = line
            else:
                break
    all_rows = np.loadtxt(colvar_fname, comments='#')
    if all_rows.ndim == 1:
        all_rows = all_rows[None, :]

    # kT-converted bias via user-supplied bias_function
    bias_kT = np.asarray(bias_function(traj_fname), dtype=float)

    return bias_kT, all_rows, header


def _write_colvar(fname, header, rows):
    """Write a merged COLVAR text file in PLUMED format.

    The time column (column 0) is recalculated as sequential multiples of
    ``dt`` (derived from the first two rows of ``rows``), so the file is
    self-consistent regardless of the original timestamps.

    Parameters
    ----------
    fname : str
        Output filename.
    header : str
        ``#! FIELDS …`` header line (written verbatim as the first line).
    rows : numpy.ndarray, shape (n_frames, n_cols)
        Data rows in the desired frame order.
    """
    if len(rows) == 0:
        return
    dt = rows[1, 0] - rows[0, 0] if len(rows) > 1 else rows[0, 0]
    rows = rows.copy()
    rows[:, 0] = np.arange(len(rows)) * dt
    with open(fname, 'w') as fh:
        fh.write(header + '\n')
        for row in rows:
            fh.write(' '.join(f'{v:.6f}' for v in row) + '\n')


def register_path(path, chain, eneconv=None, bias_function=None):
    """
    Register a newly generated path into the shooting chain and persist it.

    This function is called after a two-way shooting step has completed and a
    new full path has been assembled (typically as backward+forward segments).

    It performs three persistence steps:

    1) Writes per-frame cached arrays:
       - ``...states.npy`` (always),
       - ``...descriptors.npy`` (if present).

       Arrays are taken from :data:`NPY_CACHE` for the temporary segment files
       (``back`` / ``forw``) and then concatenated/selected according to the
       indices used in the assembled path.

    2) Writes the trajectory file itself using an atomic pattern:
       ``.{name}{ext}`` is written first, then renamed to ``{name}{ext}``.
       This reduces the chance that readers observe a partially written file.

    3) Evicts stale readers from :data:`MDA_CACHE` for the segment files, then
       appends the path to ``chain._paths``.

    If ``eneconv`` is provided (GROMACS), energies from ``back.edr`` and
    ``forw.edr`` are merged into ``{name}.edr``.

    Parameters
    ----------
    path : aimmd.path.Path
        Newly generated path. This object is mutated in place to point to the
        final single-file trajectory (``path._fnames``, ``path._first``,
        ``path._last`` are rewritten).
    chain : aimmd.pathensemble.PathEnsemble
        Shooting chain to which the path is appended (mutated in place).
    eneconv : str, optional
        Command (or full command prefix) for GROMACS ``eneconv``. If provided,
        the function attempts to merge ``back.edr`` and ``forw.edr`` into a
        path-level ``.edr`` file.
    bias_function : callable or None, optional
        ``params.bias_function`` for ``bias_source='file'`` runs.  When
        provided, the function reads the COLVAR files associated with the
        ``back`` and ``forw`` segments (renamed by ``params.run_simulation``
        to ``back_COLVAR`` / ``forw_COLVAR``), applies the same frame-index
        selection used for states/descriptors, and writes two additional
        files alongside the registered path:

        - ``{path}_COLVAR`` — merged COLVAR text file in PLUMED format with
          rows in path frame order and a recalculated time column.
        - ``{path}.bias.npy`` — per-frame bias in kT (extracted via
          ``bias_function``).

        If ``None`` (default), no bias files are written.

    Returns
    -------
    None

    Notes
    -----
    - This function uses cache filenames produced by
      :func:`~aimmd.path.utils.get_cache_fname`.
    - It assumes the segment filenames are ``back{ext}`` and ``forw{ext}``
      under the same folder as the generated output.
    """

    length = len(path)

    # get file names
    back_fname = path._fnames[0]
    back_fname = PosixPath(back_fname)
    folder = back_fname.parent
    ext = back_fname.suffix
    name = f'path{len(chain) + 1:06g}'
    temp_fname = f'{folder}/.{name}{ext}'
    fname = f'{folder}/{name}{ext}'
    fname_states = get_cache_fname(fname, 'states')
    fname_descr = get_cache_fname(fname, 'descriptors')

    # backward and forward
    forw_fname = f'{folder}/back{ext}'
    forw_fname = f'{folder}/forw{ext}'
    back_fname_states = get_cache_fname(back_fname, 'states')
    forw_fname_states = get_cache_fname(forw_fname, 'states')
    back_fname_descr = get_cache_fname(back_fname, 'descriptors')
    forw_fname_descr = get_cache_fname(forw_fname, 'descriptors')

    # only back
    if path.n_files == 1:

        # save states time series (delete cached)
        states = NPY_CACHE.pop(back_fname_states)[path.locs]
        save_npy(fname_states, states)

        # save descriptors time series (if existing)
        descriptors = NPY_CACHE.pop(back_fname_descr)
        if descriptors is not None:
            descriptors = descriptors[path.locs]
            save_npy(fname_descr, descriptors)

    else:  # backward and forward

        # get frame indices in backward and forward trajectories
        back_indices = path._extract(0, 'locs')
        forw_indices = path._extract(1, 'locs')

        # save states time series (delete cached)
        states = np.concatenate([
            NPY_CACHE.pop(back_fname_states)[back_indices],
            NPY_CACHE.pop(forw_fname_states)[forw_indices]])
        save_npy(fname_states, states)

        # save descriptors time series (if existing)
        descriptors = NPY_CACHE.pop(forw_fname_descr)
        if descriptors is not None:
            descriptors = np.concatenate([
                NPY_CACHE.pop(back_fname_descr)[back_indices],
                descriptors[forw_indices]])
            save_npy(fname_descr, descriptors)

        # merge energies (if existing)
        if eneconv:
            back_edr = f'{folder}/back.edr'
            forw_edr = f'{folder}/forw.edr'
            fname_edr = f'{folder}/{name}.edr'
            command = f'{eneconv} -f {back_edr} {forw_edr} -o {fname_edr}'
            execute_command(command, log_file='')
            print(f'+++ created {fname_edr}')

    # Save COLVAR + bias.npy for PLUMED file-mode bias tracking.
    # Uses the same back/forw COLVAR files (renamed by run_simulation) and
    # the same index selection already computed for states/descriptors above.
    if bias_function is not None:
        fname_bias = get_cache_fname(fname, 'bias')
        fname_colvar = f'{folder}/{name}{ext}'.replace(ext, '_COLVAR')
        try:
            if path.n_files == 1:
                bias_kT, rows, header = _read_colvar_for_register(
                    str(back_fname), bias_function)
                sel_bias = bias_kT[path.locs]
                sel_rows = rows[path.locs]
            else:
                bias_kT_b, rows_b, header = _read_colvar_for_register(
                    str(back_fname), bias_function)
                bias_kT_f, rows_f, _     = _read_colvar_for_register(
                    forw_fname, bias_function)
                sel_bias = np.concatenate([bias_kT_b[back_indices],
                                           bias_kT_f[forw_indices]])
                sel_rows = np.concatenate([rows_b[back_indices],
                                           rows_f[forw_indices]])
            save_npy(fname_bias, sel_bias)
            _write_colvar(fname_colvar, header, sel_rows)
            print(f'+++ saved {fname_colvar} and bias cache')
        except Exception as exc:
            print(f'!!! could not save COLVAR/bias for {fname}: {exc}')

    # save/load trajectory through temp file
    # in this way, you minimize potential disruptions
    # from sudden interruptions
    # also: modify in place
    path.write(temp_fname, overwrite=True)
    os.rename(temp_fname, fname)
    path._fnames = [fname]
    path._first = [0]
    path._last = [length - 1]

    # remove cache (if existing) - important
    MDA_CACHE.pop(back_fname)
    MDA_CACHE.pop(forw_fname)

    # add to chain
    chain._paths.append(path)

    # report
    print(f'+++ added {fname} ({path.type[:3]}) '
          f'with {len(path)} frames {now()}')


def select_shooting_point(pool, params, folder,
                          chain=None,
                          free_trajectories=[],
                          shooting_chains=[],
                          target_state=1):
    """
    Select a shooting point (frame) for committor-guided path sampling.

    This function implements the core *proposal* step of AIMMD shooting when the
    target corresponds to the reactive region. The objective is to select a
    shooting point that improves exploration and diversity in the reactive
    region by using:

    - current network-evaluated committor-like values for frames in the pool,
    - adaptive bins and target densities (loaded from disk),
    - current chain populations (to discourage oversampling already common
      regions),
    - optional additional biasing/regularization (Lorentzian factor),
    - optional "overriding" candidate frames from free trajectories.

    The function has two regimes:

    1) **Internal shooting** (``t != states[1]``)
       The target is not the reactive label, so the method samples a random
       internal frame from a path in the pool (no bin/density logic).

    2) **Reactive shooting** (``t == states[1]``)
       The method:
       - loads network/bins/densities (unless ``nbins == 1``),
       - computes missing values for pool and overriding candidates on the same
         network snapshot,
       - forms per-path histograms of values in the bins,
       - constructs bin selection weights approximately proportional to the
         inverse target density, corrected by current chain populations and
         optional Lorentzian penalization,
       - selects a bin, then selects a frame whose value falls in that bin.

    Parameters
    ----------
    pool : aimmd.pathensemble.PathEnsemble
        Current selection pool. This function may remove the chosen path from
        the pool (when the pool is at capacity) to avoid repeatedly shooting
        from the same path.
    params : aimmd.params.Params
        Parameters object providing the network, binning configuration, and
        control flags for selection.
    folder : str
        Folder for the current shooting worker (used to locate the relevant
        network/bins/densities state and, for TPS, to persist selection-time
        artifacts).
    chain : aimmd.pathensemble.PathEnsemble, optional
        Current shooting chain. Used to adjust selection probabilities based on
        `params.density_adjustment`.
    free_trajectories : list, optional
        List of free trajectories that may provide additional candidate frames
        for "overriding" selection.
    shooting_chains : list, optional
        List of shooting chains. Used to adjust selection probabilities based on
        `params.shared_density_adjustment`.
    target_state : int or str, optional
        Target state label or index, normalized via :func:`process_state`.

    Returns
    -------
    aimmd.path.Path
        A single-frame Path slice representing the chosen shooting point.

    Notes
    -----
    - The function clears :data:`NPY_CACHE` before reading values to reduce the
      risk of using stale arrays when the network/bins have just changed.
    - When ``params.chain_type == 'tps'``, the function persists the network
      state, bins, and densities used at selection time so that TPS acceptance
      can compute the correct selection bias for the chosen shooting point.
    """

    # params
    lorentzian = params.lorentzian
    states = params.states
    t = process_state(target_state, states)
    states = params.sorted_states
    pool_size = params.selection_pool_size

    # easy situation: internal shooting
    if t != states[1]:
        pool_index = np.random.choice(len(pool))
        path = pool[pool_index]
        fname = path.fname
        if path.middle('states') == t:
            indices = path.internal('indices')
        else:
            indices = np.flatnonzero(path.states == t)
        if not len(indices):
            raise RuntimeError(f'no frames available for {fname} in {t}')
        index = np.random.choice(indices)
        k, loc = path._get_local_loc(index)
        print(f'=== selecting path {fname!r}')
        print(f'=== selecting frame {loc}')

        # remove from pool and return
        if len(pool) >= pool_size:
            print(f'xxx removed {fname} from pool')
            pool.pop(pool_index)

        return path[index]

    # next params
    free_overriding_states = params.free_overriding_states
    if free_overriding_states == 'all':
        free_overriding_states = '.'
    overriding_types = [f'{s}{t}' for s in free_overriding_states]
    overriding_attempts = params.free_overriding_attempts
    overriding_rate = params.free_overriding_recovery_rate
    compute_values_args = params.compute_values_args
    ext = params.trajectory_extension
    nbins = params.nbins
    density_adjustment = max(params.density_adjustment, 0)
    shared_density_adjustment = params.shared_density_adjustment
    if nbins <= 1:  # no density adjustment in this case
        density_adjustment = 0
        shared_density_adjustment = False
    overriding_bins = np.zeros(nbins, dtype=bool)
    overriding_bins[params.free_overriding_bins] = True
    
    # process chain
    if chain is not None:
        chain = chain[chain.accepted]
    else:
        chain = PathEnsemble()
    
    # shared density adjustment: obtain shared_shooting_points
    shared_shooting_points = Path()
    if shared_density_adjustment:
        shooting_chains = PathEnsemble(shooting_chains)
        shared_shooting_points = shooting_chains.shooting('self').join()
        for fname in glob(f'{folder}/../chain{t}*/back{ext}'):
            shared_shooting_points += Path(fname, stop=1)

    # overriding configurations
    candidate_paths = PathEnsemble()
    for trajectory in free_trajectories:
        candidate_paths += trajectory.split().extract(*overriding_types)
    overriding = overriding_unique = candidate_paths.sample(
        overriding_attempts)
    if overriding_attempts:
        print(f'*** shortlisted {len(overriding)} overriding frames '
              f'from {len(candidate_paths)} candidates')
        overriding_unique = PathEnsemble(overriding).merge()

    # clear cache to avoid picking the wrong values
    NPY_CACHE.clear()

    # network parameters, bins, densities
    if nbins > 1:
        params.update_network(f'{folder}/..')
        bins, densities = params.load_bins_and_densities(f'{folder}/..')

        # compute simulated values only where there are none (yet)
        n1 = pool.compute(*compute_values_args,
                          raise_if_error=True)
        n2 = overriding_unique.compute(*compute_values_args,
                                       raise_if_error=True)
        
        # also recompute initial paths' values if they still feature in pool
        initial_paths_in_pool = PathEnsemble(
            [path for path in pool if is_initial_path(path)])
        n1 += initial_paths_in_pool.compute(
            *compute_values_args, overwrite=True)

        print(f'*** updated {n1 + n2} frame values ({n1} from pool)')
        # need to compute overriding values already here to be sure that
        # both pool and overriding values are evaluated on the same NN model
        # this is because I am not re-evaluating pre-existing values

    else:
        bins = np.array([-inf, +inf])
        densities = np.array([1.])

    # immediately get all values & population histograms
    # (such that there is a lower risk of desync)
    pool_values = pool.values
    pool_shooting_values = pool.shooting('values')
    overriding_values = overriding.values
    chain_shooting_values = chain.shooting('values')
    populations = np.histogram(chain_shooting_values, bins)[0]
    # for the purpose of density adjustment
    if 0 < density_adjustment < inf:
        populations_for_adjustment = np.histogram(
            chain_shooting_values[-density_adjustment:], bins)[0]
    elif density_adjustment:
        populations_for_adjustment = populations
    else:
        populations_for_adjustment = np.zeros_like(populations)
    if shared_density_adjustment:
        try:  # catch instabilities in try/except loop
            shared_populations_for_adjustment = np.histogram(
                shared_shooting_points.compute(
                    compute_values_args[0], '',
                    *compute_values_args[2:]),
                bins)[0]
        except Exception as exception:
            shared_populations_for_adjustment = np.zeros_like(populations)
            print("!!! Warning: in computing shared_populations_for_adjustment:"
                  f" {exception}")
    else:
        shared_populations_for_adjustment = np.zeros_like(populations)

    # report selection pool
    report, histograms = pool.report(bins=bins, values=pool_values)
    print(f'\nSelection pool\n{report}')
    if nbins > 1:
        print(f'*** current pool shooting interfaces: {pool_shooting_values}')
        print(f'*** populations  {populations}')
        print(f'*** for adjust.  {populations_for_adjustment}')
        print(f'*** shared adj.  {shared_populations_for_adjustment}')
    
    # normalize histograms, average in "combined" histogram
    norms = np.maximum(histograms.sum(axis=1), 1.0)
    histograms /= norms[:, None]
    combined_histograms = histograms.mean(axis=0)
    
    # density adjustment (lorentzian)
    densities /= densities.sum()
    print(f'*** densities    {densities}')
    if lorentzian < inf:
        centers = bin_centers(bins)
        densities *= centers ** 2 + lorentzian ** 2
        densities /= densities.sum()
        print(f'    (after applying the Loretzian) {densities}')
    
    # merge empty bins, update histograms and densities
    keepers = combined_histograms > 0
    processed_overriding_bins = overriding_bins
    merged_bin_counts = np.ones(len(densities))
    if not keepers.all():
        (bins, merged_bin_counts,
         processed_overriding_bins, *merged_histograms
        ) = merge_empty_bins(
            bins, keepers, overriding_bins,
            *histograms, combined_histograms,
            densities, populations,
            populations_for_adjustment, shared_populations_for_adjustment
        )
        processed_overriding_bins = processed_overriding_bins.astype(bool)
        histograms = merged_histograms[:len(histograms)]
        combined_histograms, densities, populations = merged_histograms[-5:-2]
        populations_for_adjustment = merged_histograms[-2]
        shared_populations_for_adjustment = merged_histograms[-1]
        if len(bins) - 1 < nbins:
            print(f'*** merged {nbins - len(bins) + 1} internal empty bins:')
            print(f'    bins         {bins}')
            print(f'    merged count {merged_bin_counts}')
            print(f'    populations  {populations}')
            print(f'    for adjust.  {populations_for_adjustment}')
            print(f'    shared adj.  {shared_populations_for_adjustment}')
            print(f'    densities    {densities}')          
        
        # to preserve target distribution: divide densities by merged bin counts
        densities /= merged_bin_counts
    
    # density adjustment (populations)
    if density_adjustment:
        densities *= populations_for_adjustment + merged_bin_counts
        densities *= shared_populations_for_adjustment + merged_bin_counts
    densities /= densities.sum()
    print(f'    (adjusted)   {densities}')

    # report also overriding
    if overriding_bins.any() and overriding_attempts:
        print(f'*** overriding   {processed_overriding_bins}')
    
    # choose path
    pool_index = np.random.choice(len(pool))
    path = pool[pool_index]
    values = pool_values[pool_index]
    if not path.is_excursion(states):  # check for consistency
        raise RuntimeError(
            f'{path.fname!r} should be an excursion, found {path.type} instead')
    indices = path.internal('indices')
    locs = path.internal('locs')
    fname = path.fname
    print(f'=== selecting path {fname!r}')
    
    # assign selection probabilities (weights)
    histogram = histograms[pool_index]
    mask = histogram > 0
    if mask.any():
        bin_weights = np.zeros(len(histogram))
        bin_weights[mask] = combined_histograms[mask] / densities[mask]
        bin_weights /= bin_weights.sum()
        print(f'*** sel weights  {bin_weights}')
        
        # select bin
        k = np.random.choice(len(bin_weights), p=bin_weights)
        print(f'=== selecting bin {k}: {bins[k:k+2]}')

        # select shooting point among candidates in bin
        candidates = np.flatnonzero(np.digitize(values, bins) - 1 == k)
        i = np.random.choice(candidates)

    else:  # special situation: FFS-like
        i = np.argmin(np.abs(values))
        print(f'!!! outside bins range')
        k = None

    # get shooting point and report info
    shooting_point = path[indices[i]]
    loc = locs[i]
    if len(bins) - 1 > 1:
        value = values[i]
        print(f'=== selecting frame {loc} (value: {value:.3f})')
    else:
        print(f'=== selecting frame {loc}')
    
    if overriding_types and k is not None:
        if not processed_overriding_bins[k]:
            print('*** skipped overriding because the SP bin is '
                  'not in overriding_bins')
            overriding = None
        elif np.digitize(pool_shooting_values[pool_index], bins) - 1 == k:
            if np.random.random() > overriding_rate:
                print(f'*** skipped overriding because the old SP is in the '
                      f'same bin (rec. rate = {overriding_rate})')
                overriding = None
            else:
                # on a very rare occasion: still override
                print(f'*** rescued overriding with '
                      f'rec. rate = {overriding_rate}')
    
    if overriding:
        candidates = np.flatnonzero(np.digitize(
            overriding_values, bins) - 1 == k)
        if len(candidates):
            i = np.random.choice(candidates)
            path = overriding
            loc = overriding.locs[i]
            name = overriding.filenames[i]
            shooting_point = path[i]  # ALL
            if nbins > 1:
                value = overriding_values[i]
                print(f'=== overriding with {name}, {loc} '
                      f'(value: {value:.3f})')
            else:
                print(f'=== overriding with {name}, {loc}')
        else:
            print(f'*** no overriding candidates in bin {k} '
                  f'({len(overriding)} frames in total)')

    # remove from pool
    if len(pool) >= pool_size:
        print(f'xxx removed {fname} from pool')
        pool.pop(pool_index)

    # save params for TPS
    if params.chain_type == 'tps':
        torch.save(params.network.state_dict(),
                   f'{folder}/network{states}.h5')
        save_npy(f'{folder}/bins{states}.npy', bins)
        save_npy(f'{folder}/densities{states}.npy', densities)
    
    print(f'Shooting initialization completed {now()}\n')
    return shooting_point


def accept_or_reject_last_path(chain, params):
    """
    Apply TPS acceptance/rejection to the most recently generated path.

    This function implements a Metropolis-like acceptance rule for TPS chains,
    correcting for *selection bias* introduced by committor-guided (bin/density
    weighted) shooting-point selection.

    The logic is:

    - If the current path is not a transition between the end states, reject.
    - If it is the first sampled transition, accept.
    - Otherwise, compute the selection probability (bias) of the chosen shooting
      point in both the current and the previously leading transition path, and
      accept with probability:

      ``acc = bias(current) / bias(leading)``

    where the "bias" is derived from the bin weights used at selection time.

    Parameters
    ----------
    chain : aimmd.pathensemble.PathEnsemble
        TPS shooting chain. The newest path is ``chain[-1]`` and the previously
        leading accepted transition is ``chain.path``.
    params : aimmd.params.Params
        Parameters object providing end-state definitions and the value function.

    Returns
    -------
    None

    Side Effects
    ------------
    - Modifies ``current.weight`` and/or ``leading.weight`` in place.
    - Prints acceptance diagnostics.

    Notes
    -----
    To reproduce the exact selection bias used when the shooting point was
    chosen, this function reloads the network/bins/densities saved in the
    worker folder at selection time (see :func:`select_shooting_point` when
    ``params.chain_type == 'tps'``).
    """
    
    # retrieve states info
    states = params.states

    current = chain[-1]
    current.weight = 0.
    leading = chain.path

    # if "current" is not a transition: reject
    if not chain[-1:].types(states, states[::-1])[0]:
        if leading:
            leading.weight += 1.
        print(f'=== acceptance probability: {0:.3f}')
        print('*** rejected')
        return

    # if "current" is the first sampled transition: accept
    if leading is None:
        current.weight = 1.
        print(f'=== acceptance probability: {1:.3f}')
        print('*** accepted')
        return

    # get model, bins, densities at the time of shooting point selection
    folder = PosixPath(chain.fname).parent
    params.update_network(folder)
    bins, densities = params.load_bins_and_densities(folder)

    # get bin weights at the time of selection
    # (population, lorentzian corrections were already incorporated
    #  in f'{folder}/density.npy')
    bin_weights = np.array(list(1 / densities) + [0.])
    # the last bin is for handling special cases outside of bin range

    # get (internal) values
    source = 'descriptors' if params.descriptors_function else 'reader'
    batch_size = params.network_batch_size
    current_values = current[1:-1].compute(
        params.values_function, source=source, batch_size=batch_size)
    leading_values = leading[1:-1].compute(
        params.values_function, source=source, batch_size=batch_size)

    # get bins
    current_bin_indices = np.digitize(current_values, bins) - 1
    leading_bin_indices = np.digitize(leading_values, bins) - 1

    # selection biases
    current_selection_biases = bin_weights[current_bin_indices]
    leading_selection_biases = bin_weights[leading_bin_indices]
    current_selection_biases /= (current_selection_biases.sum() or 1.0)
    leading_selection_biases /= (leading_selection_biases.sum() or 1.0)
    
    # of shooting point
    current_shooting_point_bias = (current_selection_biases[
        current.shooting_index - 1] or 1.0)
    leading_shooting_point_bias = (leading_selection_biases[
        leading.shooting_index - 1] or 1.0)
    
    # compute acceptance probability
    acceptance = current_shooting_point_bias / leading_shooting_point_bias
    print(f'=== acceptance probability: {acceptance:.3f}')
    
    # finally run acceptance/rejection
    if np.random.random() < acceptance:
        current.weight = 1.
        print(f'*** accepted')
    else:
        leading.weight += 1.
        print(f'*** rejected')


# ----------------------------------------------------------------------------
# Sweep-mode coordination (brute-force committor validation)
# ----------------------------------------------------------------------------
# Sweep workers cooperate purely through the shared filesystem (no central
# coordinator, no shared mutable ledger): each worker writes only into its own
# ``sweep{t}{k}`` folder, and derives global progress by reading every worker's
# committed shots (the ``path??????`` files, tagged with their source frame via
# :func:`aimmd.path.utils.write_sweep_frame`). This mirrors how the trainer
# already computes a global total by scanning all chain folders.


def sweep_marker_fname(folder):
    """Filename of a worker's in-flight 'currently shooting this frame' marker.

    Single-writer per folder, so it never contends on the filesystem.
    """
    return f'{folder}/.current_frame.npy'


def write_sweep_marker(folder, index):
    """Record (in ``folder``) the validation frame this worker is now shooting.

    Read by *other* workers as in-flight coverage so concurrent shots spread
    across frames instead of piling onto the same one; read by *this* worker on
    resume to recover the frame of an interrupted, not-yet-registered shot.
    """
    save_npy(sweep_marker_fname(folder), np.array([int(index)], dtype=int))


def read_sweep_marker(folder):
    """Read a worker's in-flight frame marker, or ``None`` if absent."""
    array = load_npy(sweep_marker_fname(folder))
    if array is None:
        return None
    try:
        return int(np.asarray(array).reshape(-1)[0])
    except (IndexError, ValueError, TypeError):
        return None


def clear_sweep_marker(folder):
    """Remove a worker's in-flight frame marker (shot finished / not running)."""
    remove(sweep_marker_fname(folder), verbose=False)


def sweep_coverage(directory, t, ext, sweep_size, seen=None,
                   include_in_flight=True):
    """Global per-frame shot counts across all sweep workers of state ``t``.

    Scans every ``{directory}/sweep{t}*`` folder for committed shots
    (``path??????{ext}``) and attributes each to a validation frame: the frame
    tag written by :func:`aimmd.path.utils.write_sweep_frame` when present, else
    the legacy positional rule ``i % sweep_size`` (the i-th shot of a folder),
    which is exactly the frame the old strictly-sequential sweep code shot. So
    campaigns started before frame tagging existed are attributed correctly
    without any backfill.

    Parameters
    ----------
    directory : str
        Run directory containing the ``sweep{t}{k}`` folders.
    t : str
        Reactive-region state label (folder infix, e.g. ``'R'``).
    ext : str
        Trajectory extension (e.g. ``'.xtc'``), including the dot.
    sweep_size : int
        Number of validation frames.
    seen : dict, optional
        Cache ``{path_fname: frame_index}`` reused across calls so each shot's
        tag is read at most once (committed shots never change frame). Mutated
        in place when provided.
    include_in_flight : bool, optional
        If True (default) the returned ``effective`` histogram also counts each
        worker's in-flight marker (frames being shot right now). The
        ``committed`` histogram never counts in-flight shots.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int]
        ``(committed, effective, committed_total)`` where ``committed`` and
        ``effective`` have shape ``(sweep_size,)`` and ``committed_total`` is
        ``int(committed.sum())`` (the global stop criterion).
    """
    sweep_size = int(sweep_size)
    if seen is None:
        seen = {}
    committed = np.zeros(sweep_size, dtype=int)
    folders = sorted(glob(f'{directory}/sweep{t}*'))
    for folder in folders:
        for i, fname in enumerate(sorted(glob(f'{folder}/path??????{ext}'))):
            if fname not in seen:
                frame = read_sweep_frame(fname)
                seen[fname] = frame if frame is not None else (i % sweep_size)
            committed[seen[fname] % sweep_size] += 1
    effective = committed.copy()
    if include_in_flight:
        for folder in folders:
            marker = read_sweep_marker(folder)
            if marker is not None:
                effective[marker % sweep_size] += 1
    return committed, effective, int(committed.sum())


def least_covered_frame(effective, k):
    """Pick the next sweep frame: least-covered, decorrelated across workers.

    Among the frames currently at the minimum (effective) coverage, worker ``k``
    takes the ``k``-th (cyclically). When coverage is flat -- notably at a cold
    start where every count is zero -- this spreads the workers across distinct
    frames immediately instead of all starting at frame 0 (the original bug).

    Parameters
    ----------
    effective : numpy.ndarray
        Per-frame coverage histogram (committed + in-flight).
    k : int
        Worker index (within this reactive-region state).

    Returns
    -------
    int
        Chosen validation-frame index.
    """
    candidates = np.flatnonzero(effective == effective.min())
    return int(candidates[int(k) % len(candidates)])
