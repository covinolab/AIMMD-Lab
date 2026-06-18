"""
aimmd.network.fit
================

Training utilities for AIMMD committor networks.

This module provides :func:`fit`, the routine that trains/updates the
neural-network model stored in ``params.network`` (a `torch.nn.Module`) to
predict the *logit committor* from AIMMD path-sampling data stored in a
:class:`~aimmd.pathensemble.PathEnsemble`.

In an AIMMD run, workers generate trajectories and accumulate them in a
:class:`~aimmd.pathensemble.PathEnsemble`. Periodically, AIMMD calls a *training
hook* to improve the model that guides subsequent sampling. By default (when
``Params.fit`` is not set), AIMMD uses :func:`aimmd.network.fit.default`, which
forwards directly to :func:`fit`.

Core idea
---------
Training examples are assembled from multiple categories of paths/frames
(internal state frames, free trajectories, and shooting trajectories). Each
frame is assigned:

- a 2-component outcome vector ``r = (r_to_state1, r_to_state2)`` encoding
  fractional contributions towards reaching each end state, and
- a selection probability that controls how batches are drawn.


The training objective is a log-binomial loss (optionally modified near the end
states by a Bayesian-like quadratic penalty), with optional smoothness and L1
regularization terms.

Side effects
------------
:func:`fit` **modifies the network in-place**. Training begins after calling
``params.network.reset_parameters()`` and proceeds with Adam. Depending on stop
criteria and early stopping settings, the function may restore a previously
saved state dict.
"""

# external imports
import os
import copy
import time
import torch
import numpy as np
from math import inf
from tqdm import tqdm
from scipy.special import expit

# aimmd imports
from .utils import extract_indices_and_series, extract_lsr_pairs, extract_mar_sequences
from ..core.utils import concatenate, now, accepts_system_id
from ..analysis.utils import compute_bins, merge_marginal_bins
from ..path.utils import get_cache_fname
from .._config import NPY_CACHE


def _load_batch_descriptors(npy_paths, locs):
    """Load a batch of raw descriptors by indexing per-trajectory NPY cache files.

    Groups frame lookups by file to avoid redundant disk reads within a batch.

    Parameters
    ----------
    npy_paths : array-like of str
        Descriptor ``.npy`` file path for each frame in the batch.
    locs : array-like of int
        Absolute frame index within the corresponding ``.npy`` file.

    Returns
    -------
    numpy.ndarray
        Raw descriptor array, shape ``(batch, *frame_shape)``.
    """
    result = [None] * len(npy_paths)
    cache = {}
    for i, (npy_path, loc) in enumerate(zip(npy_paths, locs)):
        if npy_path not in cache:
            arr = NPY_CACHE.get(npy_path)
            if arr is None:
                raise RuntimeError(
                    f'Could not load descriptor cache file: {npy_path!r}.' \
                    'The descriptor caches are likely corrupted.')
            cache[npy_path] = arr
        result[i] = cache[npy_path][loc]
    return np.array(result)


# ----------------------------------------------------------------------------
# Multi-system (multi-ligand) helpers
#
# ``fit`` accepts either a single PathEnsemble (single-system, unchanged
# behaviour) or a LIST of PathEnsembles (one per chemical system). For a list,
# every system's frames are pooled into the same flat training arrays and a
# parallel ``system_id`` array tags each frame, so the existing
# selection-probability machinery balances the systems (each carries 1/N of the
# selection weight in every bin, including the in-state anchors). The helpers
# below factor the per-system extraction and the per-batch / per-block
# descriptor handling so that systems with *different atom counts* never get
# stacked into a single dense array before they are featurized into the shared
# network's input space.
# ----------------------------------------------------------------------------

# The 10 frame categories, in the fixed order used by every flat training array
# (selection_probabilities, results, system_id, ...).
CATEGORY_NAMES = ['in1', 'in2', 'free1to1', 'free2to2', 'free1to2',
                  'free2to1', 'shot1to1', 'shot2to2', 'shot1to2', 'shot2to1']


def _category_specs(i, t, f, s, a, r, b):
    """Return ``(name, frame_mask, has_values)`` for each of the 10 categories.

    ``has_values`` marks the reactive (free/shot) categories that carry committor
    'values'; the two in-state anchor categories (in1/in2) do not.
    """
    return [
        ('in1',      (i == r) & (t == a),                       False),
        ('in2',      (i == r) & (t == b),                       False),
        ('free1to1', (i == a) & (t == r) & (f != b) & (s == a), True),
        ('free2to2', (i == b) & (t == r) & (f != a) & (s == b), True),
        ('free1to2', (i == a) & (t == r) & (f == b) & (s == a), True),
        ('free2to1', (i == b) & (t == r) & (f == a) & (s == b), True),
        ('shot1to1', (i == a) & (t == r) & (f != b) & (s == r), True),
        ('shot2to2', (i == b) & (t == r) & (f != a) & (s == r), True),
        ('shot1to2', (i == a) & (t == r) & (f == b) & (s == r), True),
        ('shot2to1', (i == b) & (t == r) & (f == a) & (s == r), True),
    ]


def _extract_categories(pathensemble, a, r, b, desc_series, must_stop):
    """Extract the per-frame arrays for all 10 categories from ONE PathEnsemble.

    Returns ``{name: {back, forw, values, desc_ref, npaths, n}}`` (``values`` is
    None for the in-state categories) or None if the worker requested
    termination. This is the per-PathEnsemble unit that ``fit`` concatenates
    across systems for multi-system (list-of-PathEnsembles) training.
    """
    types = pathensemble.types()
    i, t, f, s = types.view('U1').reshape(len(types), 4).T
    out = {}
    for name, mask, has_values in _category_specs(i, t, f, s, a, r, b):
        series = ('values', *desc_series) if has_values else desc_series
        result = extract_indices_and_series(
            pathensemble, np.flatnonzero(mask), *series)
        if has_values:
            indices, back, forw, values, *desc_ref, npaths = result
        else:
            indices, back, forw, *desc_ref, npaths = result
            values = None
        back = np.asarray(back)
        out[name] = dict(back=back, forw=np.asarray(forw), values=values,
                         desc_ref=desc_ref, npaths=npaths, n=len(back))
        if must_stop():
            return None
    return out


def _assign_balanced_uniform(selection_probabilities, start, stop,
                             system_id, total_mass):
    """Spread ``total_mass`` uniformly over a contiguous block of frames, split
    equally across the systems present in the block.

    Each present system carries ``total_mass / n_present`` (uniform over its own
    frames). With a single system this reduces exactly to the plain
    ``total_mass / block_size`` uniform weighting used in single-system fits.
    """
    if stop <= start:
        return
    sids = system_id[start:stop]
    present = np.unique(sids)
    share = total_mass / len(present)
    for sid in present:
        sub = np.flatnonzero(sids == sid)
        selection_probabilities[start + sub] = share / len(sub)


def _load_batch_descriptors_routed(npy_paths, locs, system_id, system_labels,
                                   descriptor_transform, transform_takes_sid,
                                   graphs):
    """Per-batch descriptor build for multi-system, ``in_memory=False``.

    A training batch mixes frames from several systems (different topologies /
    atom counts), so they cannot be stacked and transformed together. Frames are
    grouped by system, each group is loaded and transformed with its own
    ``system_id`` into the shared network's input space, then reassembled in the
    original batch order. Returns a list of ``Data`` (graphs) or a 2D array
    (dense).
    """
    out = [None] * len(npy_paths)
    for j in np.unique(system_id):
        sel = np.flatnonzero(system_id == j)
        raw = _load_batch_descriptors(npy_paths[sel], locs[sel])
        label = system_labels[j]
        if transform_takes_sid:
            transformed = descriptor_transform(raw, system_id=label)
        else:
            transformed = descriptor_transform(raw)
        for k, idx in enumerate(sel):
            out[idx] = transformed[k]
    if graphs:
        return out
    return np.asarray(out)


def _assemble_inmemory_multi(desc_raw_blocks, keepers, descriptor_transform,
                             transform_takes_sid, system_labels, graphs):
    """Build the upfront-transformed descriptors for multi-system, in_memory=True.

    ``desc_raw_blocks`` is a list of ``(system_index, raw_2D_array)`` in the same
    global frame order as ``keepers``/``selection_probabilities``. Each block is
    homogeneous (one system), so we filter it by its slice of ``keepers``,
    transform it with that system's ``system_id`` (graphs -> list of Data; dense
    -> fixed-width rows), and concatenate. This keeps the in_memory benefit
    (transform once, up front) while never stacking different atom counts raw.
    """
    parts = []
    offset = 0
    for system_index, raw in desc_raw_blocks:
        length = len(raw)
        sub = keepers[offset:offset + length]
        offset += length
        if not sub.any():
            continue
        block = raw[sub]
        label = system_labels[system_index]
        if transform_takes_sid:
            transformed = descriptor_transform(block, system_id=label)
        else:
            transformed = descriptor_transform(block)
        parts.append(transformed)
    if graphs:
        return [graph for part in parts for graph in part]
    if parts:
        return np.concatenate(parts, axis=0)
    return np.array([])


def fit(params,
        pathensemble,
        
        # values binning
        nbins=0,
        cutoff_min=0.5,
        cutoff_max=20.,
        state_bins='',
        max_adjustment_in_bin=10.0,
        transition_path_upweighting=1.0,
        end_state_factor=1.0,
        sparse_update_max_frames=-1,
        
        # data augmentation
        augment='no',
        
        # learning
        lr=2e-4,
        loss_bayesian_factor=0,
        loss_smoothening_weight=0,
        loss_regularization_weight=0,
        loss_regularization_exponent=1,
        epochs=500,
        batch_size=4096,
        batching_strategy='draw-replace',
        
        # stopping
        stop=80.,
        train_validation_early_stopping=False,
        early_stopping_patience=10,
        early_stopping_min_samples=1000,
        early_stopping_split=0.1,
        
        # processing
        in_memory=True,
        graphs=False,

        # latent space regularization
        lsr_weight=0.0,
        lsr_lagtime=1,
        lsr_batch_size=None,
        lsr_epsilon=1e-6,

        # minimum action regularization
        mar_weight=0.0,
        mar_lagtime=1,
        mar_batch_size=None,
        mar_epsilon=1e-8,

        # misc
        verbose=False,
        worker=None,
        loss_log_path=None):
    
    """
    Train ``params.network`` to predict the logit committor from AIMMD data.

    The function builds a supervised dataset from `pathensemble` and performs an
    iterative optimization of the network parameters using Adam. The target
    labels are *fractional* shooting outcomes (not just hard 0/1), constructed
    from shooting results and (optionally) augmented with additional trajectory
    data.

    Important
    ---------
    The network is modified **in-place**. Training begins after calling
    ``network.reset_parameters()``; depending on termination criteria, the
    function may restore earlier saved weights.

    Parameters
    ----------
    params : aimmd.Params
        AIMMD parameter container. This function uses at least the following
        attributes:

        - ``params.network`` : torch.nn.Module
            The network to train. Must have parameters so that
            ``next(network.parameters())`` is valid.
        - ``params.sorted_states`` : tuple[str, str, str]
            State labels in the order ``(state1, reactant, state2)`` assigned to
            variables ``a, r, b`` in the implementation.
        - ``params.descriptors_function`` : bool or callable-like
            Used to decide whether descriptors are read from ``'descriptors'`` or
            ``'positions'`` in the path storage model.
        - ``params.descriptor_transform`` : callable or None
            Optional transformation applied to raw descriptors to obtain network
            inputs. If None, an identity transform is used.


    pathensemble : PathEnsemble
        Collection of sampled paths providing the following interface:

        - ``pathensemble.types()`` returning an array-like where each entry
          encodes the path type as 4 characters (initial, internal, final, shoot).
        - Path access compatible with :func:`aimmd.network.utils.extract_indices_and_series`,
          i.e. supports indexing and provides the necessary per-path interface
          (`type`, `internal('indices')`, `indices`, `shooting_index`, `get(...)`).

    nbins : int, default=0
        If > 0, define committor-space bins (via :func:`compute_bins`) and build
        selection probabilities such that batches are more uniformly distributed
        across bins. Also regularizes rare outcomes within bins.

    cutoff_min : float, default=0.5
        Passed to :func:`compute_bins` as `cutoff_min` (lower cutoff in committor
        space).

    cutoff_max : float, default=20.0
        Passed to :func:`compute_bins` as `cutoff_max` (upper cutoff in logit
        scale).

    state_bins : str, default=''
        Controls inclusion of end-state bins. Examples:

        - ``'AB'`` includes both end-state bins (where `A` and `B` are the end
          state labels in `params.sorted_states`).
        - ``'all'`` includes all state bins.

        When state bins are excluded, the corresponding end-state frames may be
        set to zero selection probability depending on data presence.

    max_adjustment_in_bin : float, default=10.0
        Caps how strongly rare outcomes inside each bin are upweighted in
        selection probability (and correspondingly downweighted in results).

    transition_path_upweighting : float, default=1.0
        Multiplier applied to selection probabilities of frames belonging to
        transition paths (free and shooting transition segments).

    end_state_factor : float, default=1.0
        Base factor used for end-state (in-`a`, in-`b`) selection probability
        before bin-based adjustments.

    sparse_update_max_frames : int, default=-1
        Currently only ``-1`` is accepted (enforced by a check). Present for
        forward compatibility with sparse updates.

    augment : {'no', 'yes', 'experimental'}, default='no'
        Data augmentation mode.

        - ``'no'``: train only on shooting-point outcomes (using the intersection
          of backward and forward masks around the shooting point).
        - ``'yes'``: include free-trajectory frames and use backward/forward
          segments to assign fractional outcomes.
        - ``'experimental'``: additional heuristic augmentation with fractional
          transition contributions derived from free trajectories.

    lr : float, default=1e-3
        Base learning rate for Adam. The effective LR is ramped up over the first
        ~5% of epochs (see implementation).

    loss_bayesian_factor : float, default=0
        If non-zero, use a modified squared-deviation loss (Bayesian-like) in
        logit space to better handle discrete/imbalanced data near the states.
        If zero, use a standard binomial (cross-entropy-like) loss.

    loss_smoothening_weight : float, default=0
        If non-zero, add a smoothness penalty based on the gradient of the
        network output w.r.t. the input descriptors (requires `d.requires_grad`).

    loss_regularization_weight : float, default=0
        If non-zero, add Ln regularization over network parameters, where n is given by `loss_regularization_exponent` (default: L1).

    loss_regularization_exponent : float, default=1
        Exponent n for the Ln regularization term (see `loss_regularization_weight`).

    epochs : int, default=500
        Target number of epochs. The loop may extend up to 1.5× epochs while
        tracking best losses and may stop early due to `stop` scale or early
        stopping criteria.

    batch_size : int, default=4096
        Number of samples per batch. With draw-with-replacement batching, the
        batch size is fixed even for small datasets.

    batching_strategy : {'draw-replace', 'loop-all'}, default='draw-replace'
        Strategy for building batches.

        - ``'draw-replace'`` draws with replacement according to selection
          probabilities.
        - ``'loop-all'`` is declared but not implemented (raises NotImplementedError).

    stop : float, default=50.0
        Stop training if the output scale (max absolute logit) reaches or exceeds
        this value, or if NaNs appear.

    train_validation_early_stopping : bool, default=False
        If True, create a validation split and stop when validation loss does not
        improve for `early_stopping_patience` epochs (after scale/range conditions
        are met).

    early_stopping_patience : int, default=10
        Number of consecutive no-improvement validation checks before stopping.

    early_stopping_min_samples : int, default=1000
        Minimum training set size required to enable early stopping.

    early_stopping_split : float, default=0.1
        Fraction of the dataset used as validation set when early stopping is on.

    in_memory : bool, default=True
        Descriptor loading strategy:

        - If ``True``: all raw descriptors are loaded into one concatenated
          numpy array at the start of training, then ``descriptor_transform``
          is applied to the whole set at once and the result is kept in memory.
          Fast per-epoch access but requires peak RAM proportional to the full
          training-set size (raw array + transformed representation).
        - If ``False``: **no large descriptor array is kept in memory**.
          Instead, only per-frame file references ``(npy_path, loc)`` are
          stored.  Each training batch (and each LSR batch) loads its raw
          frames on-the-fly from the NPY cache via
          :func:`_load_batch_descriptors`, then applies
          ``descriptor_transform`` immediately.  Peak RAM is proportional to
          one batch rather than the full dataset, at the cost of repeated disk
          I/O and transform overhead each epoch.

    graphs : bool, default=False
        If True, descriptors are assumed to be graph objects in an mlcolvar-like
        DataDict format, requiring `torch_geometric` for batching.

    lsr_weight : float, default=0.0
        If non-zero, add a latent space regularization (LSR) term to the loss,
        weighted by this factor. The LSR term is computed from time-lagged pairs
        drawn from all continuous (non-internal) trajectories in the path
        ensemble and acts on the network's latent representation (via
        ``network.forward_latent`` if available, otherwise the scalar network
        output). The term encourages the latent features to capture slow
        dynamical modes of the system. A value of 0.0 (default) disables LSR
        entirely and incurs no overhead.

    lsr_lagtime : int, default=1
        Lag in frames used to form time-lagged pairs (x_t, x_{t+lagtime})
        within each continuous trajectory for the LSR term. Larger values
        target slower dynamics but reduce the number of available pairs.

    lsr_batch_size : int or None, default=None
        Number of time-lagged pairs sampled per LSR gradient step. If None,
        defaults to ``batch_size``. Should be at least 2× the latent dimension
        for well-conditioned cross-covariance estimates.

    lsr_epsilon : float, default=1e-6
        Floor value for the per-sample L2 norm during row-normalisation of
        latent features inside the LSR loss. Prevents division by zero for
        frames near the network's zero surface.

    verbose : bool, default=False
        If True, show progress via `tqdm` and print more frequent diagnostics.

    loss_log_path : str or None, default=None
        If not None, save a CSV with one row per epoch containing the columns
        ``epoch``, ``total_loss``, ``committor_loss``, ``vamp_loss``, and
        ``scale``. The committor and LSR terms are each already multiplied by
        their respective weights so that ``total_loss ≈ committor_loss +
        vamp_loss`` (plus any regularisation terms).  The file is written only
        after training completes.

    worker : aimmd.Worker or None, optional
        If provided, training periodically checks for a termination signal via
        ``getattr(worker, 'termination_signal', False)`` and returns empty outputs
        if termination is requested.

    Returns
    -------
    losses : list[float]
        Training loss per optimizer step (epoch).
    scales : list[float]
        Output scale per step, defined as ``max(max(q), -min(q))`` over the last
        computed batch output `q`.
    values : numpy.ndarray
        1D array of "values" for the training points (logit committor-like
        coordinate assembled from trajectory `values` sources). Only includes
        frames kept after selection probability masking.
    selection_probabilities : numpy.ndarray
        Per-sample selection probabilities (normalized to sum to 1) used for
        drawing training batches.
    results : numpy.ndarray, shape (N, 2)
        Per-sample fractional outcomes to each end state. These are adjusted
        in bins for imbalance and may include augmented contributions.
    """
    # Input consistency checks (fail fast)
    if batching_strategy not in ('draw-replace', 'loop-all'):
        raise TypeError(f"Invalid batching strategy: {batching_strategy}")
    if augment not in ('no', 'yes', 'experimental'):
        raise TypeError(f"Invalid augment: {augment!r}")
    if sparse_update_max_frames != -1:
        raise TypeError('In this version of aimmd, only '
                        'sparse_update_max_frames = -1 is supported')
    
    # Optional dependency only when graph descriptors are enabled
    if graphs:
        # only need to import this torch_geometric Batch if graphs
        # are used as descriptors. Otherwise, avoid the dependency.
        from torch_geometric.data import Batch
    
    # Initialization: network, optimizer, descriptor handling
    t0 = time.time()
    losses, scales = [], []
    loss_log = []                   # one dict per epoch for CSV output
    _epoch_losses = [0.0, 0.0, 0.0]  # [committor_loss, lsr_loss, mar_loss] — mutated inside closure()
    network = params.network
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # states ordering: a (state1), r (reactant), b (state2)
    a, r, b = states = params.sorted_states

    # choose where descriptors are sourced from in Path storage
    descriptors_source = ('descriptors' if params.descriptors_function else
                          'coordinates')
    descriptor_transform = params.descriptor_transform
    if params.descriptor_transform is None:
        descriptor_transform = lambda x: x

    # Multi-system: ``pathensemble`` may be a LIST of PathEnsembles, one per
    # system. Normalize to a list and detect the multi-system case. The single
    # PathEnsemble case (``multi`` False) runs the original code paths verbatim.
    pes = (list(pathensemble)
           if isinstance(pathensemble, (list, tuple)) else [pathensemble])
    n_systems = len(pes)
    multi = n_systems > 1
    if not multi:
        # a single PathEnsemble (or a 1-element list) runs the original,
        # single-system code paths below operating on this one ensemble.
        pathensemble = pes[0]
    system_labels = list(getattr(params, 'system_ids', None) or range(n_systems))
    if len(system_labels) < n_systems:                  # pad missing labels
        system_labels = list(range(n_systems))
    transform_takes_sid = accepts_system_id(descriptor_transform)
    if multi and (lsr_weight or mar_weight):
        raise NotImplementedError(
            'LSR/MAR regularization is not yet supported for multi-system '
            '(list of PathEnsembles) fits; set lsr_weight=mar_weight=0')
    if multi and train_validation_early_stopping:
        raise NotImplementedError(
            'train/validation early stopping is not yet supported for '
            'multi-system fits')
    # system_id tags each pooled frame; built in the extraction step below.
    system_id = None

    # legacy placeholders (kept as-is; only set when `augment` is falsy)
    if not augment:
        th1 = None
        th2 = None

    # Stop condition hook (worker-driven)
    must_stop = lambda : getattr(worker, 'termination_signal', False)
    
    # Classify paths by their type codes
    types = pes[0].types()
    i, t, f, s = types.view('U1').reshape(len(types), 4).T
    # i: initial states of path
    # f: final states of path
    # s: shooting states of path
    # t: internal states path

    # When in_memory=False: collect per-frame file references (fname + loc)
    # instead of the full raw descriptor array. Raw data is loaded per batch.
    _desc_series = (descriptors_source,) if in_memory else ('filenames', 'locs')

    # desc_raw_blocks (in_memory multi) holds (system_index, raw_array) in the
    # global frame order; built only in the multi branch.
    desc_raw_blocks = None

    if not multi:
        # ---- single-system extraction (original behaviour, unchanged) ----
        # Collect indices + descriptors/values for each relevant category
        # (each call may skip paths that cannot provide the requested series)
        (in1, in1_back, in1_forw, *in1_desc_ref,
         npaths_in1) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == r) & (t == a)), *_desc_series)
        if must_stop():  # responsiveness
            return [], [], [], [], []
        (in2, in2_back, in2_forw, *in2_desc_ref,
         npaths_in2) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == r) & (t == b)), *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (free1to1, free1to1_back, free1to1_forw,
         free1to1_values, *free1to1_desc_ref,
         npaths_free1to1) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == a) & (t == r) & (f != b) & (s == a)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (free2to2, free2to2_back, free2to2_forw,
         free2to2_values, *free2to2_desc_ref,
         npaths_free2to2) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == b) & (t == r) & (f != a) & (s == b)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (free1to2, free1to2_back, free1to2_forw,
         free1to2_values, *free1to2_desc_ref,
         npaths_free1to2) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == a) & (t == r) & (f == b) & (s == a)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (free2to1, free2to1_back, free2to1_forw,
         free2to1_values, *free2to1_desc_ref,
         npaths_free2to1) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == b) & (t == r) & (f == a) & (s == b)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (shot1to1, shot1to1_back, shot1to1_forw,
         shot1to1_values, *shot1to1_desc_ref,
         npaths_shot1to1) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == a) & (t == r) & (f != b) & (s == r)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (shot2to2, shot2to2_back, shot2to2_forw,
         shot2to2_values, *shot2to2_desc_ref,
         npaths_shot2to2) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == b) & (t == r) & (f != a) & (s == r)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (shot1to2, shot1to2_back, shot1to2_forw,
         shot1to2_values, *shot1to2_desc_ref,
         npaths_shot1to2) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == a) & (t == r) & (f == b) & (s == r)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        (shot2to1, shot2to1_back, shot2to1_forw,
         shot2to1_values, *shot2to1_desc_ref,
         npaths_shot2to1) = extract_indices_and_series(pathensemble,
            np.flatnonzero((i == b) & (t == r) & (f == a) & (s == r)),
            'values', *_desc_series)
        if must_stop():
            return [], [], [], [], []
        in1_n = len(in1_back)

    else:
        # ---- multi-system extraction: pool all systems, tag each frame ----
        # Extract every system independently, then concatenate per category in
        # the fixed [category: system0, system1, ...] order so that values,
        # results, system_id and descriptors all share one global frame order.
        per_pe = []
        for pe in pes:
            cats = _extract_categories(pe, a, r, b, _desc_series, must_stop)
            if cats is None:
                return [], [], [], [], []
            per_pe.append(cats)

        def _cat(name, key):
            arrs = [np.asarray(pp[name][key]) for pp in per_pe
                    if pp[name][key] is not None]
            nonempty = [x for x in arrs if len(x)]
            if nonempty:
                return np.concatenate(nonempty)
            return arrs[0] if arrs else np.array([])

        # back/forw masks and (reactive) values, per category, pooled
        in1_back, in1_forw = _cat('in1', 'back'), _cat('in1', 'forw')
        in2_back, in2_forw = _cat('in2', 'back'), _cat('in2', 'forw')
        free1to1_back, free1to1_forw = _cat('free1to1', 'back'), _cat('free1to1', 'forw')
        free2to2_back, free2to2_forw = _cat('free2to2', 'back'), _cat('free2to2', 'forw')
        free1to2_back, free1to2_forw = _cat('free1to2', 'back'), _cat('free1to2', 'forw')
        free2to1_back, free2to1_forw = _cat('free2to1', 'back'), _cat('free2to1', 'forw')
        shot1to1_back, shot1to1_forw = _cat('shot1to1', 'back'), _cat('shot1to1', 'forw')
        shot2to2_back, shot2to2_forw = _cat('shot2to2', 'back'), _cat('shot2to2', 'forw')
        shot1to2_back, shot1to2_forw = _cat('shot1to2', 'back'), _cat('shot1to2', 'forw')
        shot2to1_back, shot2to1_forw = _cat('shot2to1', 'back'), _cat('shot2to1', 'forw')
        free1to1_values = _cat('free1to1', 'values')
        free2to2_values = _cat('free2to2', 'values')
        free1to2_values = _cat('free1to2', 'values')
        free2to1_values = _cat('free2to1', 'values')
        shot1to1_values = _cat('shot1to1', 'values')
        shot2to2_values = _cat('shot2to2', 'values')
        shot1to2_values = _cat('shot1to2', 'values')
        shot2to1_values = _cat('shot2to1', 'values')
        npaths_in1 = sum(pp['in1']['npaths'] for pp in per_pe)
        npaths_in2 = sum(pp['in2']['npaths'] for pp in per_pe)
        npaths_free1to1 = sum(pp['free1to1']['npaths'] for pp in per_pe)
        npaths_free2to2 = sum(pp['free2to2']['npaths'] for pp in per_pe)
        npaths_free1to2 = sum(pp['free1to2']['npaths'] for pp in per_pe)
        npaths_free2to1 = sum(pp['free2to1']['npaths'] for pp in per_pe)
        npaths_shot1to1 = sum(pp['shot1to1']['npaths'] for pp in per_pe)
        npaths_shot2to2 = sum(pp['shot2to2']['npaths'] for pp in per_pe)
        npaths_shot1to2 = sum(pp['shot1to2']['npaths'] for pp in per_pe)
        npaths_shot2to1 = sum(pp['shot2to1']['npaths'] for pp in per_pe)
        in1_n = len(in1_back)

        # per-frame system_id in the global [category: system0, system1, ...] order
        sid_chunks = []
        for name in CATEGORY_NAMES:
            for j, pp in enumerate(per_pe):
                sid_chunks.append(np.full(pp[name]['n'], j, dtype=int))
        system_id = (np.concatenate(sid_chunks) if sid_chunks
                     else np.array([], dtype=int))

        # descriptor handling (never stacks different atom counts raw)
        if in_memory:
            desc_raw_blocks = []
            for name in CATEGORY_NAMES:
                for j, pp in enumerate(per_pe):
                    desc_raw_blocks.append(
                        (j, np.asarray(pp[name]['desc_ref'][0])))
        else:
            fn_chunks, loc_chunks = [], []
            for name in CATEGORY_NAMES:
                for j, pp in enumerate(per_pe):
                    fn_chunks.append(np.asarray(pp[name]['desc_ref'][0]))
                    loc_chunks.append(np.asarray(pp[name]['desc_ref'][1]))
            _fn = [c for c in fn_chunks if len(c)]
            _lc = [c for c in loc_chunks if len(c)]
            desc_fnames = np.concatenate(_fn) if _fn else np.array([])
            desc_locs = (np.concatenate(_lc) if _lc
                         else np.array([], dtype=int))
            desc_npy_paths = np.array(
                [get_cache_fname(f, 'descriptors') for f in desc_fnames])

    # Report collection statistics
    lengths = [len(in1_back), len(in2_back),
               len(free1to1_back), len(free2to2_back),
               len(free1to2_back), len(free2to1_back),
               len(shot1to1_back), len(shot2to2_back),
               len(shot1to2_back), len(shot2to1_back)]
    print(f'\nCollected {lengths[0]:9} in {a} frames '
                   f'({npaths_in1:6} paths),\n'
            f'          {lengths[1]:9} in {b} frames '
                   f'({npaths_in2:6} paths),\n'
            f'          {lengths[2]:9} free {a} to {a} frames '
                   f'({npaths_free1to1:6} paths),\n'
            f'          {lengths[3]:9} free {b} to {b} frames '
                   f'({npaths_free2to2:6} paths),\n'
            f'          {lengths[4]:9} free {a} to {b} frames '
                    f'({npaths_free1to2:6} paths),\n'
            f'          {lengths[5]:9} free {b} to {a} frames '
                   f'({npaths_free2to1:6} paths),\n'
            f'          {lengths[6]:9} shot {a} to {a} frames '
                   f'({npaths_shot1to1:6} paths),\n'
            f'          {lengths[7]:9} shot {b} to {b} frames '
                   f'({npaths_shot2to2:6} paths),\n'
            f'          {lengths[8]:9} shot {a} to {b} frames '
                   f'({npaths_shot1to2:6} paths), and\n'
            f'          {lengths[9]:9} shot {b} to {a} frames '
                   f'({npaths_shot2to1:6} paths)\n'
            f'   TOTAL: {sum(lengths):9} frames')

    # Multi-system: per-system breakdown of the pooled totals above (one compact
    # line per system: per-category frame counts + that system's totals).
    if multi:
        codes = [f'in{a}', f'in{b}', f'f{a}{a}', f'f{b}{b}', f'f{a}{b}',
                 f'f{b}{a}', f's{a}{a}', f's{b}{b}', f's{a}{b}', f's{b}{a}']
        print('   Per-system breakdown:')
        for j, pp in enumerate(per_pe):
            cells = ' '.join(f'{codes[k]}={pp[name]["n"]}'
                             for k, name in enumerate(CATEGORY_NAMES))
            total_frames = sum(pp[name]['n'] for name in CATEGORY_NAMES)
            total_paths = sum(pp[name]['npaths'] for name in CATEGORY_NAMES)
            print(f'     [system {system_labels[j]}] {cells}  '
                  f'| TOTAL {total_frames} frames ({total_paths} paths)')

    print(f'\nCalculating shooting results and sel. probabilities {now()}')
    
    # Allocate per-category result arrays (fractional outcomes to a/b)
    in1_results = np.zeros((lengths[0], 2))
    in2_results = np.zeros((lengths[1], 2))
    free1to1_results = np.zeros((lengths[2], 2))
    free2to2_results = np.zeros((lengths[3], 2))
    free1to2_results = np.zeros((lengths[4], 2))
    free2to1_results = np.zeros((lengths[5], 2))
    shot1to1_results = np.zeros((lengths[6], 2))
    shot2to2_results = np.zeros((lengths[7], 2))
    shot1to2_results = np.zeros((lengths[8], 2))
    shot2to1_results = np.zeros((lengths[9], 2))

    # Deterministic labels for end-state frames
    # (will put them to zero if required)
    # in 1
    in1_results[:, 0] = 1.
    
    # in 2
    in2_results[:, 1] = 1.

    # Assign shooting/free results depending on augmentation mode...    
    # ...at shooting points
    if augment == 'no':
        shot1to1_results[shot1to1_back & shot1to1_forw, 0] += 2.0
        shot2to2_results[shot2to2_back & shot2to2_forw, 1] += 2.0
        shot1to2_results[shot1to2_back & shot1to2_forw] += 1.0
        shot2to1_results[shot2to1_back & shot2to1_forw] += 1.0

    # augment
    else:
        free1to1_results[:, 0] += 1.0
        free2to2_results[:, 1] += 1.0
        free1to2_results[:, 1] += 1.0
        free2to1_results[:, 0] += 1.0        
        shot1to1_results[shot1to1_back, 0] += 1.0
        shot1to1_results[shot1to1_forw, 0] += 1.0
        shot2to2_results[shot2to2_back, 1] += 1.0
        shot2to2_results[shot2to2_forw, 1] += 1.0
        shot1to2_results[shot1to2_back, 0] += 1.0
        shot1to2_results[shot1to2_forw, 1] += 1.0
        shot2to1_results[shot2to1_back, 1] += 1.0
        shot2to1_results[shot2to1_forw, 0] += 1.0

    # experimental augmentation adds heuristic fractional transition weights
    if augment == 'experimental':
        conversion1to2 =    + expit(free1to1_values + free1to2_values).sum()
        conversion2to1 = (1 - expit(free2to2_values + free2to1_values)).sum()
        conversion1to2 /= len(shot1to2_results) + len(shot1to2_results)
        conversion2to1 /= len(shot1to2_results) + len(shot1to2_results)
        free1to1_results[:, 0] += 1.0
        free1to2_results[:, 1] += 1.0
        free2to2_results[:, 1] += 1.0
        free2to1_results[:, 0] += 1.0
        shot1to2_results[:, 1] += conversion1to2
        shot1to2_results[:, 0] += conversion2to1
        shot2to1_results[:, 1] += conversion1to2
        shot2to1_results[:, 0] += conversion2to1
        print(f'Augmentation factor from {a} to {b}: {conversion1to2:.3e}')
        print(f'Augmentation factor from {b} to {a}: {conversion2to1:.3e}')
    
    # Latent space regularization: collect time-lagged pairs split by origin
    # state for balanced batching.
    # A-origin: paths starting from a (i == a, t == r)
    # B-origin: paths starting from b (i == b, t == r)
    # Keeping them separate prevents batch bias when one side has far more
    # trajectories than the other (common early in an AIMMD run).
    lsr_desc_t_A = lsr_desc_tau_A = None
    lsr_desc_t_B = lsr_desc_tau_B = None
    lsr_npy_t_A = lsr_locs_t_A = lsr_npy_tau_A = lsr_locs_tau_A = None
    lsr_npy_t_B = lsr_locs_t_B = lsr_npy_tau_B = lsr_locs_tau_B = None
    n_lsr_A = n_lsr_B = 0
    n_lsr_pairs = 0
    if lsr_weight:
        lsr_key_A = np.flatnonzero((i == a) & (t == r))
        lsr_key_B = np.flatnonzero((i == b) & (t == r))
        print(f'\nCollecting time-lagged pairs for LSR (lagtime={lsr_lagtime}) {now()}')
        if in_memory:
            lsr_desc_t_A, lsr_desc_tau_A, n_sel_A, n_lsr_A = extract_lsr_pairs(
                pathensemble, lsr_key_A, lsr_lagtime, descriptors_source)
            lsr_desc_t_B, lsr_desc_tau_B, n_sel_B, n_lsr_B = extract_lsr_pairs(
                pathensemble, lsr_key_B, lsr_lagtime, descriptors_source)
        else:
            # Collect file references instead of raw arrays; load per batch.
            fn_t_A, fn_tau_A, n_sel_A, n_lsr_A = extract_lsr_pairs(
                pathensemble, lsr_key_A, lsr_lagtime, 'filenames')
            lc_t_A, lc_tau_A, _, _ = extract_lsr_pairs(
                pathensemble, lsr_key_A, lsr_lagtime, 'locs')
            fn_t_B, fn_tau_B, n_sel_B, n_lsr_B = extract_lsr_pairs(
                pathensemble, lsr_key_B, lsr_lagtime, 'filenames')
            lc_t_B, lc_tau_B, _, _ = extract_lsr_pairs(
                pathensemble, lsr_key_B, lsr_lagtime, 'locs')
            _gcf = lambda arr: np.array([get_cache_fname(f, 'descriptors') for f in arr]) if len(arr) else arr
            lsr_npy_t_A, lsr_locs_t_A     = _gcf(fn_t_A),   lc_t_A
            lsr_npy_tau_A, lsr_locs_tau_A = _gcf(fn_tau_A), lc_tau_A
            lsr_npy_t_B, lsr_locs_t_B     = _gcf(fn_t_B),   lc_t_B
            lsr_npy_tau_B, lsr_locs_tau_B = _gcf(fn_tau_B), lc_tau_B
        n_lsr_pairs = n_lsr_A + n_lsr_B
        print(f'   {n_lsr_A} A-origin pairs ({n_sel_A} paths), '
              f'{n_lsr_B} B-origin pairs ({n_sel_B} paths)')
        if must_stop():
            return [], [], [], [], []

    # Minimum action regularization: collect ordered sequences from reactive paths.
    # Falls back to all available paths (e.g. initial seed paths) if no reactive
    # paths from TPS sampling are found yet, so MAR acts from the first training step.
    mar_sequences_in_mem = []   # in_memory=True : list of raw descriptor arrays (one per path)
    mar_sequences_refs = []     # in_memory=False: list of (npy_paths, locs) tuples (one per path)
    n_mar_sequences = 0
    if mar_weight:
        mar_key = np.flatnonzero(
            ((i == a) & (t == r) & (f == b)) |   # free1to2 and shot1to2
            ((i == b) & (t == r) & (f == a))      # free2to1 and shot2to1
        )
        # Fallback: if no reactive paths from TPS exist yet (e.g. round 1 with only
        # initial paths that may not be classified as reactive), expand to all paths.
        mar_key_used = mar_key if len(mar_key) > 0 else None
        mar_label = 'reactive' if len(mar_key) > 0 else 'all (fallback)'
        print(f'\nCollecting MAR sequences ({mar_label}, lagtime={mar_lagtime}) {now()}')
        if in_memory:
            mar_sequences_in_mem, n_mar_sel, n_mar_sequences = extract_mar_sequences(
                pathensemble, mar_key_used, mar_lagtime, descriptors_source)
        else:
            fn_seqs, n_mar_sel, n_mar_sequences = extract_mar_sequences(
                pathensemble, mar_key_used, mar_lagtime, 'filenames')
            lc_seqs, _, _ = extract_mar_sequences(
                pathensemble, mar_key_used, mar_lagtime, 'locs')
            _gcf_mar = lambda f: get_cache_fname(f, 'descriptors')
            for fn_seq, lc_seq in zip(fn_seqs, lc_seqs):
                npy_seq = np.array([_gcf_mar(f) for f in fn_seq])
                mar_sequences_refs.append((npy_seq, lc_seq))
        print(f'   {n_mar_sequences} sequences ({n_mar_sel} paths selected)')
        if must_stop():
            return [], [], [], [], []

    # Concatenate into global training vectors
    values = concatenate([free1to1_values, free2to2_values,  # only reactive
                          free1to2_values, free2to1_values,
                          shot1to1_values, shot2to2_values,
                          shot1to2_values, shot2to1_values])
    # Build the descriptor data structure.
    # in_memory=True : a single concatenated numpy array of raw descriptors
    #                  (transformed up-front on line ~716 below).
    # in_memory=False: two compact arrays storing per-frame file references
    #                  (desc_npy_paths[i], desc_locs[i]) so raw data is
    #                  loaded on demand per batch via _load_batch_descriptors.
    # Multi-system already built desc_raw_blocks / desc_npy_paths above (it must
    # not stack different atom counts into one raw array).
    if not multi:
        _all_desc_ref0 = [in1_desc_ref[0], in2_desc_ref[0],
                          free1to1_desc_ref[0], free2to2_desc_ref[0],
                          free1to2_desc_ref[0], free2to1_desc_ref[0],
                          shot1to1_desc_ref[0], shot2to2_desc_ref[0],
                          shot1to2_desc_ref[0], shot2to1_desc_ref[0]]
        if in_memory:
            descriptors = concatenate(_all_desc_ref0)
        else:
            desc_fnames   = concatenate(_all_desc_ref0)
            desc_locs     = concatenate([in1_desc_ref[1], in2_desc_ref[1],
                                         free1to1_desc_ref[1], free2to2_desc_ref[1],
                                         free1to2_desc_ref[1], free2to1_desc_ref[1],
                                         shot1to1_desc_ref[1], shot2to2_desc_ref[1],
                                         shot1to2_desc_ref[1], shot2to1_desc_ref[1]])
            desc_npy_paths = np.array(
                [get_cache_fname(f, 'descriptors') for f in desc_fnames])
    results = concatenate([in1_results, in2_results,
                           free1to1_results, free2to2_results,
                           free1to2_results, free2to1_results,
                           shot1to1_results, shot2to2_results,
                           shot1to2_results, shot2to1_results])
    selection_probabilities = np.zeros(sum(lengths))
    if must_stop():
        return [], [], [], [], []

    # Compute committor-space bins to drive selection probability shaping.
    # Multi-system pools all systems (one shared binning) by computing extremes
    # over the concatenation of the per-system ensembles.
    if multi:
        bins_pe = pes[0]
        for pe in pes[1:]:
            bins_pe = bins_pe + pe
    else:
        bins_pe = pathensemble
    bins = compute_bins(bins_pe, max(nbins, 1),
                        cutoff_max, cutoff_min,
                        find_extremes_with='transitions',
                        source='values',
                        states=states,
                        terminal_bin_extension='all')
    if must_stop():
        return [], [], [], [], []

    # uniformize in bins
    # in 1, in 2 data: each system carries 1/N of the in-state (anchor) mass
    # (multi); a single system gets the plain uniform end_state_factor/length.
    n_internal_frames = lengths[0] + lengths[1]
    if multi:
        _assign_balanced_uniform(selection_probabilities, 0, lengths[0],
                                 system_id, end_state_factor)
        _assign_balanced_uniform(selection_probabilities, lengths[0],
                                 n_internal_frames, system_id, end_state_factor)
    else:
        if lengths[0]:
            selection_probabilities[:lengths[0]] = end_state_factor / lengths[0]
        if lengths[1]:
            selection_probabilities[lengths[0]:n_internal_frames
                ] = end_state_factor / lengths[1]

    # merge sparse bins together
    print(f'\nBins {bins}')
    bins, counts = merge_marginal_bins(bins,
        values[results[n_internal_frames:, 0].astype(bool)],
        values[results[n_internal_frames:, 1].astype(bool)], min_values=3)
    if len(bins) < nbins + 1:
        print(f'*** merged {nbins + 2 - len(bins)} bins together '
              f'to avoid overfitting: {bins}')
    
    # Assign selection probabilities and adjust results within each bin
    indices = np.digitize(values, bins) - 1
    present = results[n_internal_frames:].sum(axis=1).astype(bool)
    for i, bin_counts in enumerate(counts):
        # which data are we talking about?
        mask = n_internal_frames + np.flatnonzero((indices == i) & present)
        if not mask.any():
            continue
        
        # correct for imbalance
        # let selection probability absorb imbalance from states 1 and 2
        bin_results = results[mask]
        mask0 = mask[bin_results.prod(axis=1) > 0]  # both
        mask1 = mask[bin_results[:, 1] == 0]  # only 1
        mask2 = mask[bin_results[:, 0] == 0]  # only 2
        adjust0 = adjust1 = adjust2 = 0.
        if mask0.size:
            adjust0 = min(max_adjustment_in_bin, mask.size / mask0.size)
            results[mask0] /= adjust0
            selection_probabilities[mask0] = adjust0
        if mask1.size:
            adjust1 = min(max_adjustment_in_bin, mask.size / mask1.size)
            results[mask1] /= adjust1
            selection_probabilities[mask1] = adjust1
        if mask2.size:
            adjust2 = min(max_adjustment_in_bin, mask.size / mask2.size)
            results[mask2] /= adjust2
            selection_probabilities[mask2] = adjust2
        results[mask] /= results[mask].mean()
        # multi-system balancing: within this bin, give each system present an
        # equal share (1/n_present) of the bin's selection mass, preserving the
        # outcome-imbalance weighting *within* each system. Single-system is a
        # no-op. The existing per-bin normalization below then carries through.
        if multi:
            bin_sids = system_id[mask]
            present_systems = np.unique(bin_sids)
            for sid in present_systems:
                sub = mask[bin_sids == sid]
                sub_sum = selection_probabilities[sub].sum()
                if sub_sum > 0:
                    selection_probabilities[sub] *= (
                        (1.0 / len(present_systems)) / sub_sum)
        selection_probabilities[mask] /= selection_probabilities.sum()
        selection_probabilities[mask] *= bin_counts
        # while merged together, preserve measure
        
        if not verbose:
            continue
        
        r1, r2 = np.average(results[mask], axis=0,
                            weights=selection_probabilities[mask])
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
        
        print(f'    bin {i}: ({expit(bins[i]):.3e}, {expit(bins[i+1]):.3e}) '
              f'[weight {bin_counts}]')
        print(f'    ... {len(mask):<9} frames')
        print(f'    ... {r1:.3e} average result to {a}')
        print(f'    ... {r2:.3e} average result to {b}')
        print(f'    ... {adjust0 if mask0.size else 0:.3e} ∝sel prob of [x,y]')
        print(f'    ... {adjust1 if mask1.size else 0:.3e} ∝sel prob of [x,0]')
        print(f'    ... {adjust2 if mask2.size else 0:.3e} ∝sel prob of [0,y]')
    
    # remove training set in A and B only if required
    # and if there is enough sampling
    r1, r2 = results[n_internal_frames:].T
    if state_bins != 'all' and (r1 > r2).any() and (r2 > r1).any():
        if a not in state_bins:
            selection_probabilities[:lengths[0]] = 0.
        if b not in state_bins:
            selection_probabilities[lengths[0]:n_internal_frames] = 0.
    
    # modulate selection probabilities for transition paths
    cumsum_lengths = np.cumsum(lengths)
    free_tp_indices = range(cumsum_lengths[3], cumsum_lengths[5])
    shot_tp_indices = range(cumsum_lengths[7], cumsum_lengths[9])
    selection_probabilities[free_tp_indices] *= transition_path_upweighting
    selection_probabilities[shot_tp_indices] *= transition_path_upweighting

    # restrict to selection_probabilities > 0
    keepers = selection_probabilities > 0
    selection_probabilities = selection_probabilities[keepers]
    values = values[keepers[n_internal_frames:]]
    if multi:
        # Build the upfront-transformed descriptors per system (in_memory) and
        # filter the per-frame system_id, BEFORE filtering by keepers.
        if in_memory:
            descriptors = _assemble_inmemory_multi(
                desc_raw_blocks, keepers, descriptor_transform,
                transform_takes_sid, system_labels, graphs)
        else:
            desc_npy_paths = desc_npy_paths[keepers]
            desc_locs      = desc_locs[keepers]
        system_id = system_id[keepers]
    elif in_memory:
        descriptors = descriptors[keepers]
    else:
        desc_npy_paths = desc_npy_paths[keepers]
        desc_locs      = desc_locs[keepers]
    results = results[keepers]
    k = results[:, 0] > 0
    training_set_size = len(selection_probabilities)
    if not keepers.any() or must_stop():
        return [], [], [], [], []
    
    # Early stopping setup (optional)
    # disable early stopping, if threshold of training set size is not met
    if training_set_size < early_stopping_min_samples:
        train_validation_early_stopping = False
        print(f"\nDisabling early stopping since "
              f"total samples {training_set_size} < "
              f"{early_stopping_min_samples} min samples.")

    if train_validation_early_stopping:
        state_dict_early_stopping = copy.deepcopy(network.state_dict())
        no_improvement_steps = 0
        min_validation_loss = np.inf
        validation_losses = []
        validation_set_size = int(training_set_size * early_stopping_split)
        print(f"\nUsing early stopping with {validation_set_size} "
              f"samples for validation set.")
        validation_indices = np.random.choice(
            training_set_size, size=validation_set_size, replace=False)
        training_set_size -= validation_set_size
        # set selection probabilities of validation set to zero in training set
        selection_probabilities[validation_indices] = 0.
        
        # create validation vectors
        if not graphs:
            if in_memory:
                raw_val = descriptors[validation_indices]
            else:
                raw_val = _load_batch_descriptors(
                    desc_npy_paths[validation_indices], desc_locs[validation_indices])
            d_val = descriptor_transform(raw_val)
            d_val = torch.tensor(d_val, dtype=dtype, device=device)
            d_val.requires_grad = True
        else:
            # when using graphs, we need to process the DataDict objects
            # instead of arrays
            if in_memory:
                # descriptors is already a list of Data objects after the pre-transform
                d_val_list = [descriptors[i] for i in validation_indices]
            else:
                d_val_list = descriptor_transform(_load_batch_descriptors(
                    desc_npy_paths[validation_indices], desc_locs[validation_indices]))
            d_val = Batch.from_data_list(d_val_list).to(device).to_dict()
        r_val = torch.tensor(results[validation_indices], dtype=dtype, device=device)
    
    print(f'\nTraining set size {training_set_size}')
    selection_probabilities /= selection_probabilities.sum()
    
    # optional global transform (single-system only; multi-system already
    # transformed per system in _assemble_inmemory_multi above).
    if in_memory and not multi:
        print(f'Transforming descriptors {now()}')
        descriptors = descriptor_transform(descriptors)
        if lsr_weight and n_lsr_pairs > 0 and not graphs:
            if n_lsr_A > 0:
                lsr_desc_t_A = descriptor_transform(lsr_desc_t_A)
                lsr_desc_tau_A = descriptor_transform(lsr_desc_tau_A)
            if n_lsr_B > 0:
                lsr_desc_t_B = descriptor_transform(lsr_desc_t_B)
                lsr_desc_tau_B = descriptor_transform(lsr_desc_tau_B)
    
    # LSR helpers (defined here so they close over network, device, dtype)
    lsr_bs = lsr_batch_size if lsr_batch_size is not None else batch_size
    mar_bs = mar_batch_size if mar_batch_size is not None else 8

    def _get_latent(net, d):
        """Return latent features: forward_latent if available, else forward."""
        if hasattr(net, 'forward_latent'):
            return net.forward_latent(d)
        return net(d)

    def _lsr_loss(chi_0, chi_tau):
        """Latent space regularization loss (negated; minimise to regularise).

        Computes the negative squared Frobenius norm of the time-lagged
        cross-covariance matrix of row-normalised latent features. This is a
        surrogate for the VAMP-2 score (Mardt et al., 2018; Wu & Noé, 2020)
        that avoids explicit matrix inversion. Row-normalisation makes the loss:

        - **scale-invariant**: the loss cannot be driven to -inf by scaling up
          feature magnitudes, which would otherwise cause gradient explosion;
        - **bounded**: ||C01||_F^2 ≤ 1 after row-normalisation (by the
          submultiplicativity of the Frobenius norm), so the loss lies in [-1, 0];
        - **gradient-stable**: no Cholesky factorisation or eigendecomposition
          enters the backward pass.

        The surrogate equals the true VAMP-2 score exactly when the equal-time
        covariances satisfy C00 = C11 = I (trivially true for d=1; an
        approximation for d>1 that improves as the latent distribution becomes
        more isotropic on the unit sphere).
        """
        n = chi_0.shape[0]
        # Centre features
        chi_0 = chi_0 - chi_0.mean(0, keepdim=True)
        chi_tau = chi_tau - chi_tau.mean(0, keepdim=True)
        # Row-normalise each sample to unit norm
        chi_0 = chi_0 / chi_0.norm(dim=1, keepdim=True).clamp(min=lsr_epsilon)
        chi_tau = chi_tau / chi_tau.norm(dim=1, keepdim=True).clamp(min=lsr_epsilon)
        # LSR loss = -||C01||_F^2  (exact VAMP-2 surrogate when C00 = C11 = I)
        C01 = chi_0.T @ chi_tau / n
        return -(C01 * C01).sum()

    def _mar_loss(latent_sequences):
        """Normalized action ratio loss for Minimum Action Regularization (MAR).

        For each reactive transition path with latent sequence z_0, ..., z_{n-1},
        computes the normalized Onsager-Machlup action ratio:

            ratio_i = (n-1) * sum_{t=0}^{n-2} ||z_{t+1} - z_t||^2
                      / ||z_{n-1} - z_0||^2.clamp(min=mar_epsilon)

        By Cauchy-Schwarz, ratio_i >= 1 for non-degenerate paths, with equality
        iff the path is a geodesic (straight line) in latent space. The loss is
        mean(relu(ratio_i - 1)) >= 0, equal to 0 only for geodesics or degenerate
        (all-equal) paths. The relu prevents the degenerate case (path_action=0,
        endpoint_sq clamped to mar_epsilon) from producing a negative loss and
        creating an attractive gradient toward feature collapse.

        Properties:
        - Scale-invariant: multiplying all z by a constant leaves ratio unchanged,
          so the loss cannot be driven to zero by feature shrinkage.
        - No matrix operations: purely element-wise squared distances.
        - Works for latent_dim=1 (in which case it penalises non-monotone paths).
        - Collapse-safe: the relu ensures zero gradient when the network collapses.
        """
        per_path = []
        for z in latent_sequences:
            n = z.shape[0]
            steps = z[1:] - z[:-1]                          # (n-1, d)
            path_action = (steps * steps).sum()              # scalar
            endpoint_sq = ((z[-1] - z[0]) ** 2).sum().clamp(min=mar_epsilon)
            # clamp at 0: Cauchy-Schwarz guarantees ratio>=1 for non-degenerate paths;
            # the relu prevents the collapsed (all-equal) case from giving -1 and
            # creating an attractive gradient toward feature collapse.
            per_path.append(torch.relu((n - 1) * path_action / endpoint_sq - 1.0))
        return torch.stack(per_path).mean()

    """
    Training loop.
    """
    
    # Training loop: state backups and stopping logic
    state_dict0 = copy.deepcopy(network.state_dict())  # fixed
    state_dict1 = copy.deepcopy(network.state_dict())  # linked to min_loss1
    state_dict2 = copy.deepcopy(network.state_dict())  # linked to min_loss2
    min_loss1 = inf
    min_loss2 = inf
    min_loss_step1 = 0
    min_loss_step2 = 0
    
    print(f'Resetting the network parameters {now()}\n')
    network.reset_parameters()
    
    # actual loop
    print(f'Starting the training cycle {now()}')
    counter = tqdm(total=epochs, disable=not verbose)
    while True:
        
        if must_stop():
            return [], [], [], [], []
        
        for param_group in optimizer.param_groups:
            # slowly increase lr
            param_group['lr'] = lr * min(1, (counter.n + 1) / (epochs / 20))
        
        # sample batch
        if batching_strategy == 'draw-replace':
            indices = np.random.choice(training_set_size, batch_size,
                p=selection_probabilities)
        elif batching_strategy == 'loop-all':
            # zero selection probabilities are ALREADY not in the training set
            training_indices = np.random.permutation(training_set_size)
            batches = [training_indices[i:i + batch_size]
                       for i in range(0, len(training_indices), batch_size)]
            raise NotImplementedError("can't to adapt this right now")
        
        # build descriptors batch (array or graph)
        if not graphs:
            if in_memory:
                d = descriptors[indices]          # already transformed upfront
            elif multi:  # mixed-system batch: load + transform per system
                d = _load_batch_descriptors_routed(
                    desc_npy_paths[indices], desc_locs[indices],
                    system_id[indices], system_labels, descriptor_transform,
                    transform_takes_sid, graphs=False)
            else:  # load raw frames on-the-fly from NPY_CACHE, then transform
                d = descriptor_transform(
                    _load_batch_descriptors(desc_npy_paths[indices], desc_locs[indices]))
            d = torch.tensor(d, dtype=dtype, device=device)
            d.requires_grad = True
        else:
            # when using graphs, we need to process the DataDict objects
            # instead of arrays
            if in_memory:
                # descriptors is already a list of Data objects after the pre-transform
                d = [descriptors[i] for i in indices]
            elif multi:  # mixed-system batch: load + transform per system
                d = _load_batch_descriptors_routed(
                    desc_npy_paths[indices], desc_locs[indices],
                    system_id[indices], system_labels, descriptor_transform,
                    transform_takes_sid, graphs=True)
            else:  # load raw frames on-the-fly from NPY_CACHE
                d = descriptor_transform(
                    _load_batch_descriptors(desc_npy_paths[indices], desc_locs[indices]))
            d = Batch.from_data_list(d).to(device).to_dict()

        # flatten non-graph descriptors for dense networks
        if not graphs:
            d = torch.flatten(d, start_dim=1)

        # build result batch
        r = torch.tensor(results[indices], dtype=dtype, device=device)
        
        # define loss function
        # Note: refactored this to allow for computation of validation loss
        def loss_function(q, r):
            if loss_bayesian_factor:
                
                q1 = - (torch.log(1 + torch.exp(-q[:, 0])) +
                        loss_bayesian_factor)
                q2 = + (torch.log(1 + torch.exp(+q[:, 0])) +
                        loss_bayesian_factor)
                
                to1_contribution = (q[:, 0] - q1) ** 2
                to2_contribution = (q[:, 0] - q2) ** 2
                
                q_bis = q[:,0].detach()
                
                loss = torch.sum(q_bis ** 2 *
                    (r[:, 0] * to1_contribution +
                    r[:, 1] * to2_contribution))
                
                # normalize
                loss /= torch.sum(q_bis ** 2 * (r[:, 0] + r[:, 1]) *
                                loss_bayesian_factor ** 2)
                loss -= 1.0
            
            # standard binomial loss
            else:
                exp_pos_q = torch.exp(+q[:, 0])
                exp_neg_q = torch.exp(-q[:, 0])
                to1_contrib = r[:, 0] * torch.log(1. + exp_pos_q)
                to2_contrib = r[:, 1] * torch.log(1. + exp_neg_q)
                loss = torch.sum(to1_contrib + to2_contrib) / torch.sum(r)
            
            # Compute the smoothness penalty
            if loss_smoothening_weight:
                q_grad = torch.autograd.grad(
                    outputs=q.sum(), inputs=d, create_graph=True)[0]
                smoothness_loss = (torch.abs(q_grad) ** 2).mean()
                loss += loss_smoothening_weight * smoothness_loss
            
            # Calculate Ln regularization
            if loss_regularization_weight:
                ln_norm = sum(p.abs().pow(loss_regularization_exponent).sum() for p in network.parameters())

                # Combine original loss with Ln regularization termls
                loss += loss_regularization_weight * ln_norm
            return loss

        def closure():
            optimizer.zero_grad()
            q = network(d)
            committor_term = loss_function(q, r)
            loss = committor_term

            # Latent space regularization (LSR)
            lsr_contrib = 0.0
            if lsr_weight and n_lsr_pairs > 0:
                # Balanced sampling: draw half from A-origin and half from
                # B-origin trajectories to prevent bias toward the more
                # populated state.
                half = lsr_bs // 2
                raw_t_segs, raw_tau_segs = [], []
                if in_memory:
                    _lsr_sides = ((lsr_desc_t_A, lsr_desc_tau_A, n_lsr_A),
                                  (lsr_desc_t_B, lsr_desc_tau_B, n_lsr_B))
                else:
                    _lsr_sides = (
                        (lsr_npy_t_A, lsr_npy_tau_A, lsr_locs_t_A, lsr_locs_tau_A, n_lsr_A),
                        (lsr_npy_t_B, lsr_npy_tau_B, lsr_locs_t_B, lsr_locs_tau_B, n_lsr_B))
                for side in _lsr_sides:
                    n_side = side[-1]
                    if n_side == 0:
                        continue
                    n_draw = half if (n_lsr_A > 0 and n_lsr_B > 0) else lsr_bs
                    v_idx = np.random.choice(n_side, n_draw, replace=True)
                    if in_memory:
                        desc_t_side, desc_tau_side = side[0], side[1]
                        raw_t_segs.append(desc_t_side[v_idx])
                        raw_tau_segs.append(desc_tau_side[v_idx])
                    else:
                        npy_t, npy_tau, locs_t, locs_tau = side[0], side[1], side[2], side[3]
                        raw_t_segs.append(
                            _load_batch_descriptors(npy_t[v_idx], locs_t[v_idx]))
                        raw_tau_segs.append(
                            _load_batch_descriptors(npy_tau[v_idx], locs_tau[v_idx]))
                raw_t = np.concatenate(raw_t_segs, axis=0)
                raw_tau = np.concatenate(raw_tau_segs, axis=0)

                if not graphs:
                    if in_memory:
                        dv_t = raw_t
                        dv_tau = raw_tau
                    else:
                        dv_t = descriptor_transform(raw_t)
                        dv_tau = descriptor_transform(raw_tau)
                    dv_t = torch.flatten(
                        torch.tensor(dv_t, dtype=dtype, device=device), start_dim=1)
                    dv_tau = torch.flatten(
                        torch.tensor(dv_tau, dtype=dtype, device=device), start_dim=1)
                else:
                    if in_memory:
                        dv_t_list = descriptor_transform(raw_t)
                        dv_tau_list = descriptor_transform(raw_tau)
                    else:
                        dv_t_list = descriptor_transform(raw_t)
                        dv_tau_list = descriptor_transform(raw_tau)
                    dv_t = Batch.from_data_list(dv_t_list).to(device).to_dict()
                    dv_tau = Batch.from_data_list(dv_tau_list).to(device).to_dict()

                network.train()
                chi_t = _get_latent(network, dv_t)
                chi_tau = _get_latent(network, dv_tau)
                lsr_term = lsr_weight * _lsr_loss(chi_t, chi_tau)
                lsr_contrib = float(lsr_term.detach())
                loss = loss + lsr_term

            # Minimum Action Regularization (MAR)
            mar_contrib = 0.0
            if mar_weight and n_mar_sequences > 0:
                n_avail = n_mar_sequences
                path_idxs = np.random.choice(n_avail, min(mar_bs, n_avail), replace=False)
                latent_seqs = []
                for pidx in path_idxs:
                    if in_memory:
                        raw_seq = mar_sequences_in_mem[pidx]
                    else:
                        npy_seq, lc_seq = mar_sequences_refs[pidx]
                        raw_seq = _load_batch_descriptors(npy_seq, lc_seq)
                    if not graphs:
                        dv_mar = torch.flatten(
                            torch.tensor(descriptor_transform(raw_seq),
                                         dtype=dtype, device=device),
                            start_dim=1)
                    else:
                        graph_list = descriptor_transform(raw_seq)
                        dv_mar = Batch.from_data_list(graph_list).to(device).to_dict()
                    network.train()
                    z_seq = _get_latent(network, dv_mar)   # (n_frames_i, latent_dim)
                    latent_seqs.append(z_seq)
                mar_term = mar_weight * _mar_loss(latent_seqs)
                mar_contrib = float(mar_term.detach())
                loss = loss + mar_term

            # Store individual components for per-epoch logging (list mutation
            # is visible outside the closure without nonlocal)
            _epoch_losses[0] = float(committor_term.detach())
            _epoch_losses[1] = lsr_contrib
            _epoch_losses[2] = mar_contrib

            loss.backward()
            return loss
        
        # update network
        network.train()
        loss = optimizer.step(closure)
        losses.append(float(loss.detach()))

        # report scales
        q = network(d).detach()
        scales.append(max(float(torch.max(q)), -float(torch.min(q))))

        # per-epoch loss log
        loss_log.append({
            'epoch': counter.n + 1,
            'total_loss': losses[-1],
            'committor_loss': _epoch_losses[0],
            'lsr_loss': _epoch_losses[1],
            'mar_loss': _epoch_losses[2],
            'scale': scales[-1],
        })
        Range = float(torch.min(q)), float(torch.max(q))
        
        # update counter
        if verbose:
            counter.update(1)
        else:
            counter.n += 1
        
        # handle termination: too high scales
        if scales[-1] >= stop or np.isnan(scales[-1]):
            print(f'!!! stopping early since scale '
                  f'{scales[-1]:.3f} > {stop:.3f}')
            if counter.n < 1.25 * epochs:
                print(f'    restoring lowest loss\' ({min_loss1:.3e}) '
                      f'weights, step {min_loss_step1 + 1}')
                network.load_state_dict(state_dict1)
            else:
                print(f'    restoring lowest loss\' ({min_loss2:.3e}) '
                      f'weights, step {min_loss_step2 + 1}')
                network.load_state_dict(state_dict2)
            break
        
        # save model if goood
        if losses[-1] <= min_loss1:
            min_loss1 = losses[-1]
            min_loss_step1 = counter.n - 1
            state_dict1 = copy.deepcopy(network.state_dict())
        
        # new min loss after reaching target epochs
        if counter.n >= epochs and losses[-1] <= min_loss2:
            min_loss2 = losses[-1]
            min_loss_step2 = counter.n - 1
            state_dict2 = copy.deepcopy(network.state_dict())
        
        # handle termination: lowest loss
        if counter.n >= epochs and losses[-1] <= min_loss1:
            break
        
        # new target after 1.25 * epochs
        if counter.n >= 1.25 * epochs and losses[-1] <= min_loss2:
            break
        
        # at most 1.5 * epochs
        if counter.n >= 1.5 * epochs:
            print(f'    restoring lowest loss\' ({min_loss2:.3e}) '
                  f'weights, step {min_loss_step2 + 1}')
            network.load_state_dict(state_dict2)
            break
        
        # handle early stopping with validation set
        if train_validation_early_stopping:
            # compute validation loss
            network.eval()
            with torch.no_grad():
                pred_val = network(d_val)
                val_loss = loss_function(pred_val, r_val)
                val_loss = float(val_loss.detach())
                validation_losses.append(val_loss)
            network.train()

            # early stopping logic. Only start when Range is acceptable
            if (Range[0] <= 0 and Range[1] >= 0 and scales[-1] >= 1):
                if val_loss < min_validation_loss:
                    min_validation_loss = val_loss
                    no_improvement_steps = 0
                    state_dict_early_stopping = copy.deepcopy(network.state_dict())
                else:
                    no_improvement_steps += 1
                    if verbose:
                        print(f'    Early stopping report, epoch {i}: no improvement for '
                        f'{no_improvement_steps} steps, loss {loss:.3e} validation loss {val_loss:.3e} '
                        f'(min val loss: {min_validation_loss:.3e})')
                    if no_improvement_steps >= early_stopping_patience:
                        print(f'    Early stopping triggered after {no_improvement_steps} steps without improvement, '
                            f'restoring best model from early stopping '
                            f'(val loss: {min_validation_loss:.3e})')
                        network.load_state_dict(state_dict_early_stopping)
                        # exit training loop
                        break

        # report
        if verbose and counter.n % max(epochs // 20, 1) == 0:
            print(f'\n    loss {losses[-1]:.3e}, '
                  f'scale {scales[-1]:.3f}, '
                  f'range ({Range[0]:.3f}, {Range[1]:.3f})')
    
    # close counter
    counter.close()
    
    # Final sanity check: restore original weights if range is inconsistent
    if Range[0] > 0 or Range[1] < 0 or scales[-1] < 1:
        print(f'!!! bad range ({Range[0]:.3f}, {Range[1]:.3f}), '
              f'restoring original parameters')
        network.load_state_dict(state_dict0)
    
    # recompute scales and range in case they changed
    q = network(d).detach()
    scales[-1] = max(float(torch.max(q)), -float(torch.min(q)))
    Range = float(torch.min(q)), float(torch.max(q))
    
    # Save per-epoch loss log as CSV if requested
    if loss_log_path is not None and loss_log:
        import csv
        with open(loss_log_path, 'w', newline='') as _f:
            _writer = csv.DictWriter(
                _f, fieldnames=['epoch', 'total_loss', 'committor_loss',
                                'lsr_loss', 'mar_loss', 'scale'])
            _writer.writeheader()
            _writer.writerows(loss_log)
        print(f'    loss history saved to {loss_log_path}')

    # Report and return
    print(f'\nTraining took {time.time()-t0:.1f}s')
    print(f'    {counter.n} epochs')
    print(f'    last loss {losses[-1]:.3e}')
    print(f'    last scale {scales[-1]:.3f}')
    print(f'    last range ({Range[0]:.3f}, {Range[1]:.3f})')
    return losses, scales, values, selection_probabilities, results#, D, R


def default(params, pathensemble, verbose=True, worker=None):
    """Default training hook used when `Params.fit` is not set.

    AIMMD keeps the full run configuration in :class:`~aimmd.Params`. This
    includes the neural-network model to be trained and all settings needed to
    run the simulation (system, engines, sampling, I/O).

    If `Params` does not specify a custom training callable, AIMMD calls this
    function. It is a thin wrapper that forwards its arguments to :func:`fit`
    unchanged, so the training uses **all default hyperparameters of**
    :func:`fit` (e.g. learning rate, batch size, number of epochs, binning and
    augmentation settings) unless a custom training function is provided.

    Parameters
    ----------
    params : aimmd.Params
        Complete AIMMD configuration, including the network model.
    pathensemble : PathEnsemble
        The trajectories generated so far; used by :func:`fit` to assemble the
        training set.
    verbose : bool, optional
        Controls training log/progress output. If True (default), :func:`fit`
        may print progress information; if False, it should run quietly.
    worker : aimmd.Worker or None, optional
        Worker context for this run. If provided, :func:`fit` may use it for
        coordinating with other workers. If None (default), :func:`fit` runs
        without a worker context.

    Returns
    -------
    tuple
        Exactly the return value of :func:`fit`.

    Notes
    -----
    This is a thin wrapper around :func:`fit`, which trains or updates the
    AIMMD model from the current :class:`~aimmd.pathensemble.PathEnsemble`.
    """
    return fit(params=params,
               pathensemble=pathensemble,
               verbose=verbose,
               worker=worker)
