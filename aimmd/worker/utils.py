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
from re import split
from glob import glob
from math import inf
from pathlib import Path as PosixPath

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


def get_initial_frames_for_free_simulations(path, target_state, reactive_state):
    """
    given a path, extract two frames such that the first is always
    reactive, the second is internal to the target_state.
    """
    if target_state == reactive_state:
        if np.random.random() > .5:
            initial_frames = path[:+2]
        else:
            initial_frames = path[:-3:-1]
    elif path.initial('states') == target_state:
        initial_frames = path[1::-1]
    else:
        initial_frames = path[-2:]

    # check and return
    states = initial_frames.states
    if states[0] != reactive_state and states[1] != target_state:
        raise RuntimeError(f'{path.fname} must allow to extract a'
                           f'"{reactive_state}{target_state}" segment')
    return initial_frames


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


def select_shooting_point(chain, params, shots=[], target_state=1):
    """
    Select a shooting point (frame) for committor-guided path sampling
    from latest path in chain with nonzero weight.

    This function implements the core *proposal* step of AIMMD shooting when the
    target corresponds to the reactive region. The objective is to select a
    shooting point that improves exploration and diversity in the reactive
    region by using:

    - current network-evaluated committor-like values for frames in the path,
    - adaptive bins and target densities (loaded from disk),
    - current sampled shooting point population (to discourage oversampling
      already common regions),
    - optional additional biasing/regularization
      (Lorentzian factor, global, and local populations),
    - optional "overriding" candidate frames from free trajectories.
    
    The function has two regimes:

    1) **Internal shooting** (``t != states[1]``)
       The target is not the reactive label, so the method samples a random
       internal frame fro path
    
    2) **Reactive shooting** (``t == states[1]``)
       The method:
       - loads network/bins/densities (unless ``nbins == 1``),
       - computes missing values for path,
       - forms per-path histograms of values in the bins,
       - constructs bin selection weights approximately proportional to the
         inverse target density, corrected by current chain populations and
         optional Lorentzian penalization,
       - selects a bin, then selects a frame whose value falls in that bin.
    
    Parameters
    ----------
    chain : aimmd.pathensemble.Pathensemble
        The shooting path chains. You will select from the latest path in
        chain with nonzero weight. The inital path is assumed to be there
        in first position.
    params : aimmd.params.Params
        Parameters object providing the network, binning configuration, and
        control flags for selection.
    shots : list of aimmd.pathensemble.Pathensemble
        For computing the global shooting point populations in bins.
    target_state : int or str, optional
        Target state label or index, normalized via :func:`process_state`.
    
    Returns
    -------
    aimmd.path.Path
        A single-frame Path slice representing the chosen shooting point.
    
    Notes
    -----
    - When ``params.chain_type == 'tps'``, the function copies the network
      state, bins, and densities used at selection time so that TPS acceptance
      can compute the correct selection bias for the chosen shooting point.
    """
    
    # get states
    states = params.states
    t = process_state(target_state, states)
    states = params.sorted_states

    # which path?
    path = chain.path
    fname = path.fname
    folder = '/'.join(split(r'[\\/]', path.fname)[:-1]) or '.'
    print(f'*** choosing from {fname} {now()}')
    
    # easy situation: internal shooting
    if t != states[1]:
        index = np.random.choice(path.internal('indices'))
        k, loc = path._get_local_loc(index)
        print(f'=== selecting frame {path._fnames[k]}, {loc}')
        return path[index]
    
    # get the rest of the params
    chain_type = params.chain_type
    ext = params.trajectory_extension
    compute_values_args = params.compute_values_args
    lorentzian = params.lorentzian
    free_overriding_states = params.free_overriding_states
    if free_overriding_states == 'all':
        free_overriding_states = '.'
    overriding_types = [f'{s}{t}' for s in free_overriding_states]
    overriding_attempts = params.free_overriding_attempts
    overriding_rate = params.free_overriding_recovery_rate
    always_select_inside_the_bins = params.always_select_inside_the_bins
    
    # get bins and overriding bins info
    nbins = params.nbins
    overriding_bins = np.zeros(nbins, dtype=bool)
    overriding_bins[params.free_overriding_bins] = True

    # get density adjustment info
    local_density_adjustment = params.local_density_adjustment * (nbins > 1)
    global_density_adjustment = params.global_density_adjustment and nbins > 1
    
    # get path info
    indices = path.internal('indices')
    locs = path.locs[indices]
    si = path.shooting_index - indices[0]
    
    # initialize SP populations
    global_populations = None
    local_populations = None
    
    if nbins > 1:
        
        # collect shooting points from every chain
        if global_density_adjustment:
            shooting_points = PathEnsemble()
            for current_chain in shots:
                if not current_chain:
                    continue
                shooting_points += current_chain.shooting('self')
            shooting_points = shooting_points.join()
            current_points = PathEnsemble()
            for fname in glob(f'{folder}/../chain{t}*/back{ext}'):
                shooting_point = Path()
                shooting_point.extend(fname, 1)
                current_points += shooting_point
            current_points = current_points.join()
        
        # load network params, bins, and densities
        params.update_network(f'{folder}/..')
        bins = NPY_CACHE.load(f'{folder}/../bins{states}.npy')
        densities = NPY_CACHE.load(f'{folder}/../densities{states}.npy').copy()
        
        # compute (new) values
        path.compute(*compute_values_args, raise_if_error=True,
                     overwrite=path==chain[0])
        values = path.values[indices]
        
        # global shooting point populations
        if global_density_adjustment:
            shooting_points.compute(*compute_values_args)
            global_populations = np.histogram(shooting_points.values, bins)[0]
            try:  # catch instabilities in try/except loop
                global_populations += np.histogram(
                    current_points.compute(compute_values_args[0],
                                           compute_values_args[2]), bins)[0]
            except Exception as exception:
                print("!!! Warning: in computing current shooting points' "
                      f"values: {exception}")
            global_populations = global_populations.astype(float)
        
        # local shooting point populations
        if local_density_adjustment >= len(chain) - 1:
            local_populations = np.histogram(
                chain[1:].shooting('values'), bins)[0]
            local_populations = local_populations.astype(float)
        elif local_density_adjustment >= 1:
            n = int(local_density_adjustment)
            local_populations = np.histogram(
                chain[-n:].shooting('values'), bins)[0]
            local_populations = local_populations.astype(float)
        
        # compute path histogram
        histogram = np.histogram(values, bins)[0].astype(float)
    
    else: # all default
        bins = np.array([-inf, +inf])
        densities = np.array([1.])
        values = np.zeros(len(indices))
        histogram = np.array([1.])
    
    # report
    print(f'*** selection bins     {bins}')
    if overriding_bins.any() and overriding_attempts:
        print(f'    overriding bins    {overriding_bins}') 
    print(f'*** path histogram     {histogram}')
    if global_populations is not None:
        print(f'*** global populations {global_populations}')
    if local_populations is not None:
        print(f'*** local populations  {local_populations}')
    print(f'*** ensemble densities {densities}')
    
    # density adjustment (lorentzian)
    if lorentzian < inf:
        centers = bin_centers(bins)
        densities *= centers ** 2 + lorentzian ** 2
        densities /= densities.sum()
        print(f'    (after applying the Loretzian) {densities}')
    
    # merge empty bins, update histograms and densities
    keepers = histogram > 0
    processed_overriding_bins = overriding_bins
    merged_bin_counts = np.ones(len(densities))
    if not keepers.all():
        (bins, merged_bin_counts,
         processed_overriding_bins,
         histogram, densities,
         global_populations, local_populations
        ) = merge_empty_bins(
            bins, keepers, overriding_bins,
            histogram, densities,
            global_populations, local_populations)
        # only if populations
        processed_overriding_bins = processed_overriding_bins.astype(bool)
        if len(bins) - 1 < nbins:
            print(f'*** merged {nbins - len(bins) + 1} internal empty bins:')
            print(f'    selection bins     {bins}')
            if processed_overriding_bins.any() and overriding_attempts:
                print(f'    overriding bins    {processed_overriding_bins}') 
            print(f'    path histogram     {histogram}')
            print(f'    ensemble densities {densities}')           
        
        # to preserve target distribution: divide densities by merged bin counts
        densities /= merged_bin_counts
    
    # density adjustment by SP populations
    if global_populations is not None:
        densities *= global_populations + .5 * merged_bin_counts
    if local_populations is not None:
        densities *= local_populations + .5 * merged_bin_counts
    
    # final normalization and report
    densities /= densities.sum()
    print(f'    adjusted densities {densities}')
    
    # assign selection probabilities (weights)
    mask = histogram > 0
    if mask.any():
        bin_weights = np.zeros(len(histogram))
        bin_weights[mask] = histogram[mask] / densities[mask]
        bin_weights /= bin_weights.sum()
        print(f'*** selection weights {bin_weights}')
        
        # select bin
        k = np.random.choice(len(bin_weights), p=bin_weights)
        bin_info = f'bin {k}: {bins[k:k+2]}'
        print(f'=== selecting {bin_info}')
        
        # select shooting point among candidates in bin
        candidates = np.flatnonzero(np.digitize(values, bins) - 1 == k)
        i = np.random.choice(candidates)

    elif always_select_inside_the_bins and len(chain) > 1:
        
        # use previous path in chain
        return select_shooting_point(chain[:-1], params, shots, target_state)
    
    else: # special situation: FFS-like
        i = np.argmin(np.abs(values))
        print(f'!!! outside bins range')
        k = None
        bin_info = 'bin: outside'
    
    # get shooting point and report info
    shooting_point = path[indices[i]]
    loc = locs[i]
    if nbins > 1:
        value = values[i]
        print(f'=== selecting frame {loc} (value: {value:.3f})')
    else:
        print(f'=== selecting frame {loc}')
    
    # determine wether you are overriding
    if overriding_types:
        if k is None or not processed_overriding_bins[k]:
            print(f'*** skipped overriding because the SP {bin_info} '
                  f'is not in overriding_bins')
            overriding_types = []
        elif np.digitize(values[si], bins) - 1 == k:
            if np.random.random() > overriding_rate:
                print(f'*** skipped overriding because the old SP is in the '
                      f'same {bin_info} (rec. rate = {overriding_rate})')
                overriding_types = []
            else:
                print(f'*** rescued overriding in {bin_info} '
                      f'(rec. rate = {overriding_rate})')
    
    if overriding_types:
        
        # get free configurations from overriding
        candidate_paths = PathEnsemble()
        for trajectory in params.free_trajectories(f'{folder}/..'):
            candidate_paths += trajectory.split().extract(*overriding_types)
        overriding = overriding_unique = candidate_paths.sample(
            overriding_attempts)
        if overriding_attempts:
            print(f'*** shortlisted {len(overriding)} overriding frames '
                  f'from {len(candidate_paths)} candidates')
            overriding_unique = PathEnsemble(overriding).merge()
            overriding_unique.compute(
                *compute_values_args, raise_if_error=True)
        overriding_values = overriding.values
        
        candidates = np.flatnonzero(
            np.digitize(overriding_values, bins) - 1 == k)
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
            print(f'*** no overriding candidates in {bin_info} '
                  f'({len(overriding)} frames in total)')
    
    # save params for TPS
    if chain_type == 'tps':
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
