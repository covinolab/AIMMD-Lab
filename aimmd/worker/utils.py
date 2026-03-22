"""
aimmd.worker.utils
=================

Low-level utilities for AIMMD worker tasks.

This module collects helper functions used by worker task implementations,
primarily :mod:`aimmd.worker._shoot` and :mod:`aimmd.worker._train`.

Scope and role in AIMMD
-----------------------
The functions in this file support **path sampling simulations**, which are the
core of AIMMD’s enhanced-sampling strategy. In particular, they implement the
mechanics needed to:

- maintain a *selection pool* of candidate paths/frames,
- select shooting points in a way that is guided by the current committor model
  (typically a neural network) and by adaptive density targets,
- register newly generated paths on disk and in memory,
- (optionally) apply TPS-style acceptance/rejection steps,

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
import time
import numpy as np
import torch
import random
from math import inf
from pathlib import PosixPath

# aimmd imports
from ..path import Path
from .._config import NPY_CACHE, MDA_CACHE, print
from ..cache.npy import save_npy
from ..core.utils import now, process_state
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble
from ..execute.utils import execute_command
from ..analysis.utils import bin_centers, merge_empty_bins
from ..pathensemble.utils import match_patterns, assemble_pathensemble
from ..network.rescale_utils import rescale


def update_selection_pool(pool, size, chain,
                          initial_paths=None,
                          at_least_one=''):
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


def register_path(path, chain, eneconv=None):
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
        Current shooting chain. When provided, only accepted paths are used to
        estimate current "populations" in value bins.
    free_trajectories : list, optional
        List of free trajectories that may provide additional candidate frames
        for "overriding" selection.
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

    # easy situation: internal shooting
    if t != states[1]:
        path = pool.pop()
        index = np.random.choice(path.internal('indices'))
        k, loc = path._get_local_loc(index)
        print(f'=== selecting frame {path._fnames[k]}, {loc}')
        return path[index]

    # next params
    free_overriding_states = params.free_overriding_states
    if free_overriding_states == 'all':
        free_overriding_states = '.'
    overriding_types = [f'{s}{t}' for s in free_overriding_states]
    overriding_attempts = params.free_overriding_attempts
    overriding_rate = params.free_overriding_recovery_rate
    compute_values_args = params.compute_values_args
    density_adjustment = params.density_adjustment
    pool_size = params.selection_pool_size
    len_ext = len(params.trajectory_extension)
    nbins = params.nbins
    overriding_bins = np.arange(nbins - 1)[params.free_overriding_bins]
    
    # process chain
    if chain is not None:
        chain = chain[chain.accepted]
    else:
        chain = PathEnsemble()

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

        # compute only where there are no values (yet)
        n1 = pool.compute(*compute_values_args,
                          raise_if_error=True)
        n2 = overriding_unique.compute(*compute_values_args,
                                       raise_if_error=True)
        print(f'*** updated {n1 + n2} frame values ({n1} from pool)')
        # need to compute overriding values already here to be sure that
        # both pool and overriding values are evaluated on the same NN model
        # this is because I am not re-evaluating pre-existing values

    else:
        bins = np.array([-inf, +inf])
        densities = np.array([1.])

    # immediately get all values & populations histogram
    # (such that there is a lower risk of desync)
    pool_values = pool.values
    pool_shooting_values = pool.shooting('values')
    overriding_values = overriding.values
    chain_shooting_values = chain.shooting('values')
    populations = np.histogram(chain_shooting_values, bins)[0]

    # report selection pool
    report, histograms = pool.report(bins=bins, values=pool_values)
    print(f'\nSelection pool\n{report}')
    if nbins > 1:
        print(f'*** current pool shooting interfaces: {pool_shooting_values}')
        print(f'*** populations  {populations}')
    
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
    if not keepers.all():
        bins, merged_bin_counts, *merged_histograms = merge_empty_bins(
            bins, keepers,
            *histograms, combined_histograms, densities, populations
        )
        histograms = merged_histograms[:len(histograms)]
        combined_histograms, densities, populations = merged_histograms[-3:]
        if len(bins) - 1 < nbins:
            print(f'*** merged {nbins - len(bins) + 1} internal empty bins:')
            print(f'    bins         {bins}')
            print(f'    merged count {merged_bin_counts}')
            print(f'    populations  {populations}')
            print(f'    densities    {densities}')
        
        # to preserve target distribution: divide densities by merged bin counts
        densities /= merged_bin_counts
    
    # density adjustment (populations)
    if density_adjustment:
        densities *= populations + 1
    densities /= densities.sum()
    print(f'    (adjusted)   {densities}')
    
    # choose path
    pool_index = np.random.choice(len(pool))
    path = pool[pool_index]
    values = pool_values[pool_index]
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
    if nbins > 1:
        value = values[i]
        print(f'=== selecting frame {loc} (value: {value:.3f})')
    else:
        print(f'=== selecting frame {loc}')
    
    if overriding_types and k is not None:
        if k not in overriding_bins:
            print(f'*** skipped overriding because the SP bin is '
                  f'not in free_overriding_bins={overriding_bins!r}')
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
