"""
aimmd.cgaimmd.bioemu2.fit
=========================

The online fit of a :class:`~aimmd.cgaimmd.bioemu2.network.BioEmu2Network`:
the *sequential* recipe, committor head first, then the generator's adapters,
both trained from reset in every round.

A params file calls it from its ``fit`` (AIMMD's trainer passes ``worker``)::

    from aimmd.cgaimmd.bioemu2 import sequential_fit

    def fit(params, pathensemble, verbose=False, worker=None):
        return sequential_fit(params, pathensemble, verbose=verbose, worker=worker,
                              min_shots_for_generator=100, adapter_steps=3000)

One round (:func:`sequential_round`):

1. **Data.** :func:`aimmd.cgaimmd.dataset.extract_dataset` on the run's path
   ensemble (descriptor series ``params.descriptors_series``): shooting points
   with their two-way outcomes, frames with their states. The extraction is
   reused across the rounds of one trainer process (an explicit cache keyed by
   the run folder, the states, the series and the atom count); nothing else is.
2. **Features** of unmodified BioEmu2, computed once (stored in float16): every
   shooting point with an outcome clean (``sigma_ref``) and ``noisy_copies``
   times noised, at most ``n_anchor_max`` in-state frames clean and once
   noised (noise levels log-uniform in :data:`~.head.SIGMA_RANGE`), at most
   ``pool`` frames (the generator's pool) clean.
3. **Head:** a fresh MLP with the frozen scaler, loss
   ``L_sp + 0.5 L_in-state`` (two-way outcomes; in-state frames toward
   ``sigmoid(+-10)``, as many first-state as last-state frames per step),
   AdamW, EMA 0.999, early-stopped on a random ``head_holdout`` share of the
   shooting points (at least ``min_held``; else ``head_steps_without_holdout``
   steps). Head and EMA head end as the best EMA head.
4. **Generator** (once ``min_shots_for_generator`` shooting points and
   ``min_frames_for_generator`` frames exist): fresh adapters (zero output
   layers: plain BioEmu2), ``adapter_steps`` steps of BioEmu2's EDM denoising
   loss (with the model's data scale ``edm_schedule.sigma_data``) on the pool,
   each frame at its label ``clip(head, +-10)`` from the frozen head of step
   3, the frames drawn so that the 10 label bins on ``[-10, 10]`` are equally
   likely (uniform within a bin), label dropout ``p_uncond``. The per-atom
   loss weights the ligand atoms ``ligand_weight`` times. AdamW, warm-up and
   cosine decay, gradient clip 1, a divergence guard (roll back to the last
   good window, halve the learning rate), and on CUDA running out of memory
   the micro-batch is halved.
5. **Label probe** (logged, decides nothing): the trained adapters' loss on
   fixed pool frames with their own labels, shuffled labels and none.

The weights are reset every round (PI, 26 Oct 2026: never warm-start); the
round counter and the dataset are the only state carried between rounds.

Options (all off by default; DESIGN 4.8):

``label_time_window_ps``
    The labels of the reactive frames become the logit of the mean committor
    of the reactive frames of the same trajectory within ``+- window`` ps
    (:func:`aimmd.cgaimmd.dataset.time_averaged_labels`). The frame interval
    of every trajectory is read from its trajectory files
    (:func:`trajectory_frame_intervals`), so shots and free MD may be written
    at different intervals. It changes the generator's labels only, never the
    head.
``bin_min``
    Bins with fewer than ``bin_min`` pool frames get a proportionally smaller
    share of the draws (a safeguard against one frame taking a large share;
    never screened).
``p_uncond`` with ``BioEmu2Network(learned_null=True)``
    Label dropout that trains the learned null (needed for guidance against
    it). Without a learned null the dropped rows give no gradient: the
    production chignolin campaign ran ``p_uncond=0.2`` that way.
``ligand_weight``
    The trypsin campaign used 5 (without it the adapter ignored the label).
The pair adapter and the adapter widths are network options.

Seeds. Round ``r`` uses ``seed + 1000 r`` (data and draws), ``+ 1`` (head) and
``+ 2``, ``+ 3`` (adapters). :func:`sequential_fit` takes ``seed`` from its
argument, else from ``params.cgaimmd_parameters['generation_seed']``, else
from a hash of the run folder, so replicates in different folders differ and
a run restarted in the same folder reproduces its seeds. A run that is moved
(local disk, archive, cluster) gets other seeds after the move unless
``generation_seed`` is set. The seed and every setting go into the round's
metadata.

Ported from ``bioemu2_committor/aimmd_flow_fit.py`` (``sequential_round``,
``bioemu2_fit_sequential``, ``fresh_head``, ``fresh_adapter``,
``_rotations``, ``_previous_round``) and ``aimmd_flow_holo_fit.py``
(``holo_sequential_round``, ``holo_fit_sequential``, ``_label_probe``),
merged; the warm-start fit, the density-weighted pool and the patch of AIMMD's
reweighting were dropped, and the round counter reads ``round`` where AIMMD
stores it.
"""

import copy
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from ..dataset import DEFAULT_SERIES, extract_dataset, time_averaged_labels
from .head import LAMBDA_CLIP, SIGMA_RANGE, in_state_nll, shooting_point_nll

__all__ = ['sequential_fit', 'sequential_round', 'previous_round', 'fit_seed', 'trajectory_frame_intervals',
           'clear_dataset_cache', 'N_LABEL_BINS']

N_LABEL_BINS = 10
"""Label bins on ``[-LAMBDA_CLIP, LAMBDA_CLIP]`` of the generator's balanced draws."""

_DATASETS = {}
"""The extracted dataset and the last round of every (run folder, states, series, atoms, stride) of this process."""


def clear_dataset_cache():
    """Forget the datasets :func:`sequential_fit` keeps between rounds (tests)."""
    _DATASETS.clear()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _rotations(n, generator, device):
    q = torch.randn(n, 4, generator=generator, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _log(verbose, **record):
    if verbose:
        print(json.dumps(record, default=str), flush=True)


def _bin_draw_probabilities(labels, bin_min=None):
    """Draw probability of every pool frame: label bins equally likely, uniform within a bin.

    With `bin_min`, a bin with ``c < bin_min`` frames gets the share
    ``c / bin_min`` of a full bin.
    """
    edges = torch.linspace(-LAMBDA_CLIP, LAMBDA_CLIP, N_LABEL_BINS + 1, device=labels.device)
    bins = torch.clamp(torch.bucketize(labels, edges[1:-1]), 0, N_LABEL_BINS - 1)
    counts = torch.bincount(bins, minlength=N_LABEL_BINS).float()
    if not bin_min:
        weights = 1.0 / (torch.count_nonzero(counts) * counts[bins])
    else:
        share = torch.clamp(counts / float(bin_min), max=1.0)
        weights = share[bins] / counts[bins]
    return weights / weights.sum(), counts


def _round_seed(seed, round_index):
    return int(seed) + 1000 * int(round_index)


# ---------------------------------------------------------------------------
# the run: round counter, seed, frame intervals
# ---------------------------------------------------------------------------

def _run_directory(worker):
    """The run folder of the trainer (relative to the params folder, where the task runs), or None."""
    if worker is None:
        return None
    # inside a worker task the run is read through ``_directory`` (``Worker.run`` changes into the params
    # folder; ``directory`` is the name the worker was given), as the shoot and free tasks do
    return getattr(worker, '_directory', None) or getattr(worker, 'directory', None)


def previous_round(run_directory, states):
    """The round of the network the run holds, from ``network{states}.json``; -1 without one.

    AIMMD's trainer stores the fit's ``network_metadata`` at the top level of
    the file (``{'round': ..., ...}``); the old BioEmu2 fit looked for
    ``network_metadata.round`` and so restarted its count at every submission.
    Both layouts are read.
    """
    if run_directory is None:
        return -1
    fname = os.path.join(os.fspath(run_directory), f'network{states}.json')
    try:
        with open(fname) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return -1
    value = record.get('round', (record.get('network_metadata') or {}).get('round'))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _word62(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 2


def fit_seed(params, run_directory, seed=None):
    """The base seed of the fit and where it came from.

    Returns
    -------
    (int, str)
        `seed` (``'argument'``) if given; else
        ``params.cgaimmd_parameters['generation_seed']`` (``'generation_seed'``)
        if set; else a 62-bit hash of the absolute run folder (``'run folder'``),
        so that replicates in different folders differ and a run restarted in
        the same folder reproduces its seeds; 0 (``'default'``) without a run
        folder.

    Notes
    -----
    The hash is of the folder's path: a run moved between machines or disks
    (``/tmp``, an archive, a cluster) gets other seeds after the move. Set
    ``generation_seed`` for a run that will move.
    """
    if seed is not None:
        return int(seed), 'argument'
    settings = getattr(params, 'cgaimmd_settings', None) or dict(getattr(params, 'cgaimmd_parameters', None) or {})
    if settings.get('generation_seed') is not None:
        return int(settings['generation_seed']), 'generation_seed'
    if run_directory is not None:
        return _word62(os.path.abspath(os.fspath(run_directory))), 'run folder'
    return 0, 'default'


def _file_interval(fname):
    """Time between consecutive frames of a trajectory file (ps), or None (fewer than two frames)."""
    from ... import _config
    reader = _config.MDA_CACHE.get(fname) if getattr(_config, 'MDA_CACHE', None) is not None else None
    if reader is None:
        from MDAnalysis.coordinates.core import reader as open_reader
        reader = open_reader(fname)
    times = []
    for k in range(min(len(reader), 3)):
        times.append(float(reader[k].time))
    steps = [abs(b - a) for a, b in zip(times[:-1], times[1:]) if abs(b - a) > 0]
    return steps[0] if steps else None


def trajectory_frame_intervals(data):
    """``(n_trajectories,)`` float64: the time between frames (ps) of every trajectory of a dataset.

    Read from the trajectory files (the first two frames of the first file of
    a trajectory with at least two frames, through AIMMD's reader cache), so
    that each trajectory has its own: shots and free MD may be written at
    different intervals. NaN where no file of a trajectory has two frames.
    """
    out = np.full(len(data.trajectories), np.nan)
    for k, files in enumerate(data.trajectories):
        for fname in files:
            interval = _file_interval(fname)
            if interval is not None:
                out[k] = interval
                break
    return out


# ---------------------------------------------------------------------------
# one round
# ---------------------------------------------------------------------------

@torch.no_grad()
def _label_probe(den_loss, pool_x, labels, device, micro_batch, n=16, sigmas=(4.0, 16.0)):
    """The trained adapters' loss on `n` pool frames spread over the labels: own, shuffled and no labels.

    The same noise for the three (one generator seed per micro-batch); logged, decides nothing. A pool
    of fewer than `n` frames gives repeated frames.
    """
    order = torch.argsort(labels)
    pick = order[torch.linspace(0, len(order) - 1, n, device=device).round().long()]
    x, lam = pool_x[pick], labels[pick]
    shuffled = lam[torch.roll(torch.arange(n, device=device), n // 2)]
    out = {}
    for sigma in sigmas:
        result = {}
        for name, use in (('own', lam), ('shuffled', shuffled), ('none', None)):
            total = 0.0
            for i in range(0, n, micro_batch):
                generator = torch.Generator(device=device)
                generator.manual_seed(4242 + i)
                k = len(x[i:i + micro_batch])
                total += float(den_loss(x[i:i + micro_batch], None if use is None else use[i:i + micro_batch],
                                        None if use is not None else torch.ones(k, dtype=torch.bool, device=device),
                                        torch.full((k,), sigma, device=device), generator)) * k / n
            result[name] = total
        result['own_vs_shuffled_percent'] = 100 * (result['shuffled'] - result['own']) / max(result['shuffled'], 1e-12)
        out[f'sigma{sigma:g}'] = result
    return out


def _averaged_labels(data, pool_index, pool_labels, labels_of, window_ps, frame_intervals_ps):
    """The pool labels with the reactive frames' labels averaged over ``+- window_ps`` along their trajectories."""
    fr_state = np.asarray(data.fr_state)
    trajectory = np.asarray(data.fr_trajectory)
    if frame_intervals_ps is None:
        frame_intervals_ps = trajectory_frame_intervals(data)
    intervals = np.asarray(frame_intervals_ps, dtype=np.float64)
    time_ps = np.asarray(data.fr_frame, dtype=np.float64) * intervals[trajectory]
    reactive = (fr_state == 0) & np.isfinite(time_ps)
    members = np.union1d(pool_index, np.flatnonzero(reactive))
    labels = np.full(len(fr_state), np.nan)
    labels[pool_index] = pool_labels
    others = np.setdiff1d(members, pool_index)
    if len(others):
        labels[others] = labels_of(others)
    averaged = time_averaged_labels(labels[members], trajectory[members], np.where(
        np.isfinite(time_ps[members]), time_ps[members], 0.0), window_ps, include=reactive[members])
    labels[members] = averaged
    shift = labels[pool_index] - pool_labels
    info = dict(time_window_ps=float(window_ps), n_labelled=int(len(members)), n_averaged=int(reactive.sum()),
                n_without_interval=int(((fr_state == 0) & ~np.isfinite(time_ps)).sum()),
                frame_intervals_ps=sorted({round(float(v), 6) for v in intervals if np.isfinite(v)}),
                n_pool_changed=int(np.count_nonzero(shift)),
                pool_shift_median=float(np.median(shift)) if len(shift) else None,
                pool_shift_abs_median=float(np.median(np.abs(shift))) if len(shift) else None)
    return labels[pool_index], info


def sequential_round(network, data, round_index=0, *, seed=0, verbose=False, batch_sp=64, batch_anchor=64,
                     batch_den=16, micro_batch=8, n_anchor_max=20000, pool=20000, min_shots_for_generator=100,
                     min_frames_for_generator=0, noisy_copies=4, lr_head=3e-4, wd_head=1e-4, ema=0.999,
                     head_holdout=0.1, min_held=15, head_eval_every=250, head_patience=8, head_max_steps=20000,
                     head_steps_without_holdout=1000, lr_adapter=1e-3, adapter_steps=3000,
                     adapter_lr_schedule='cosine', adapter_log_every=500, guard_window=100, guard_factor=2.0,
                     p_uncond=0.0, ligand_weight=1.0, bin_min=None, label_time_window_ps=None,
                     frame_intervals_ps=None, feature_batch=64, label_probe=True):
    """One round of the sequential recipe on a dataset (module docstring); the network is trained in place.

    Parameters
    ----------
    network : BioEmu2Network
    data : TrainingDataset
        Or any object with its fields ``sp_rows``, ``sp_r_first``,
        ``sp_r_last``, ``fr_rows``, ``fr_state`` (and ``fr_trajectory``,
        ``fr_frame``, ``trajectories`` for `label_time_window_ps`). Rows are
        the flat generator atoms.
    round_index : int
        The round (seeds ``seed + 1000 round_index``, ...).
    seed : int, optional
        The base seed.
    label_time_window_ps : float, optional
        Average the reactive frames' generator labels over this half-window
        (ps) along their trajectories (off by default).
    frame_intervals_ps : array-like, optional
        The frame interval of every trajectory of `data` (default: read from
        the files, :func:`trajectory_frame_intervals`); only with
        `label_time_window_ps`.

    The other settings are those of the module docstring; the defaults are
    the production chignolin campaign's except ``p_uncond`` (0) and
    ``ligand_weight`` (1).

    Returns
    -------
    (list of dict, dict)
        The head losses of every step (``L_sp``, ``L_anchor``; empty when
        there was nothing to train on) and the round's record.
    """
    from . import _api
    if adapter_lr_schedule not in ('cosine', 'constant'):
        raise ValueError(f"adapter_lr_schedule must be 'cosine' or 'constant', got {adapter_lr_schedule!r}")
    if not 0.0 <= float(p_uncond) < 1.0:
        raise ValueError(f'p_uncond must be in [0, 1), got {p_uncond!r}')
    if bin_min is not None and not bin_min >= 0:
        raise ValueError(f'bin_min must be >= 0 or None, got {bin_min!r}')
    if label_time_window_ps is not None and not (math.isfinite(label_time_window_ps) and label_time_window_ps >= 0):
        raise ValueError(f'label_time_window_ps must be a finite number >= 0 or None, got {label_time_window_ps!r}')
    r = int(round_index)
    round_seed = _round_seed(seed, r)
    device = network.device
    engine = network.engine()
    features_of, denoiser = engine.features, engine.denoiser
    n_real, n_pad = network.system.n_bioemu2_atoms, denoiser.n_pad
    sigma_data = float(denoiser.model.edm_schedule.sigma_data)      # BioEmu2's data scale (Ångström)
    t0 = time.time()
    rng = np.random.default_rng(round_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(round_seed)
    lo, hi = np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1])

    def bioemu2_atoms(rows):
        x = network.to_bioemu2(np.asarray(rows, np.float32))
        return x - x.mean(1, keepdim=True)

    @torch.no_grad()
    def feats(x, sigma):
        out = [features_of.features(x[i:i + feature_batch], sigma[i:i + feature_batch]).half()
               for i in range(0, len(x), feature_batch)]
        return torch.cat(out) if out else torch.zeros(0, network.n_features, dtype=torch.half, device=device)

    @torch.no_grad()
    def noised(x, sigma):
        return features_of.canonical(x) + sigma[:, None, None] * torch.randn(x.shape, generator=gen, device=device)

    def log_sigma(n):
        return torch.exp(torch.empty(n, device=device).uniform_(lo, hi, generator=gen))

    def ref(m):
        return torch.full((m,), network.sigma_ref, device=device)

    # ---- data and features
    sp_rows = np.asarray(data.sp_rows)
    keep = ((np.asarray(data.sp_r_first) + np.asarray(data.sp_r_last)) > 0 if len(sp_rows)
            else np.zeros(0, bool))
    sp_index = np.flatnonzero(keep)
    n_sp = int(len(sp_index))
    fr_state = np.asarray(data.fr_state)
    fr_rows = np.asarray(data.fr_rows)
    anchors = np.flatnonzero(fr_state > 0)
    if len(anchors) > n_anchor_max:
        anchors = np.sort(rng.choice(anchors, n_anchor_max, replace=False))
    n_first = torch.as_tensor(np.asarray(data.sp_r_first, np.float32)[sp_index], device=device)
    n_last = torch.as_tensor(np.asarray(data.sp_r_last, np.float32)[sp_index], device=device)
    shots = {}
    if n_sp:
        sp_x = bioemu2_atoms(sp_rows[sp_index])
        shots['clean'] = feats(sp_x, ref(n_sp))
        shots['sigma'] = log_sigma(n_sp * noisy_copies)
        noisy = []
        for c in range(noisy_copies):                     # copy by copy; row c + noisy_copies * shot
            sigma_c = shots['sigma'][c::noisy_copies]
            noisy.append(feats(noised(sp_x, sigma_c), sigma_c))
        shots['noisy'] = torch.stack(noisy, 1).reshape(n_sp * noisy_copies, -1)
        del sp_x, noisy
    in_state = dict(state=torch.as_tensor(fr_state[anchors], device=device))
    if len(anchors):
        anchor_x = bioemu2_atoms(fr_rows[anchors])
        in_state['clean'] = feats(anchor_x, ref(len(anchors)))
        in_state['sigma'] = log_sigma(len(anchors))
        in_state['noisy'] = feats(noised(anchor_x, in_state['sigma']), in_state['sigma'])
        del anchor_x
    use_generator = n_sp >= min_shots_for_generator and len(fr_state) >= max(1, min_frames_for_generator)
    if use_generator:
        pool_index = np.sort(rng.choice(len(fr_state), min(pool, len(fr_state)), replace=False))
        pool_x = bioemu2_atoms(fr_rows[pool_index])
        pool_f = feats(pool_x, ref(len(pool_x)))
    t_features = time.time() - t0

    # ---- head: fresh, early-stopped on held-out shooting points
    head, head_ema = network.head, network.head_ema
    head.reset_parameters(round_seed + 1)
    head_ema.load_state_dict(head.state_dict())
    n_held = int(round(head_holdout * n_sp))
    permutation = rng.permutation(n_sp)
    held = np.sort(permutation[:n_held]) if n_held >= min_held else np.zeros(0, int)
    train = np.sort(permutation[len(held):])
    train_t, held_t = torch.as_tensor(train, device=device), torch.as_tensor(held, device=device)
    first_state = (torch.nonzero(in_state['state'] == 1).reshape(-1) if len(anchors)
                   else torch.zeros(0, dtype=torch.long, device=device))
    last_state = (torch.nonzero(in_state['state'] == 2).reshape(-1) if len(anchors)
                  else torch.zeros(0, dtype=torch.long, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_head, weight_decay=wd_head)
    max_steps = head_max_steps if len(held) else head_steps_without_holdout

    def held_nll():
        with torch.no_grad():
            lam = head_ema(shots['clean'][held_t].float(), ref(len(held_t)))
            return float(shooting_point_nll(lam, n_first[held_t], n_last[held_t]))

    def draw(index, k):
        return index[torch.randint(len(index), (k,), generator=gen, device=device)]

    best = (float('inf'), None, -1)
    trace, losses = [], []
    head.train()
    step = -1
    for step in range(max_steps):
        for group in optimizer.param_groups:
            group['lr'] = lr_head * min(1.0, (step + 1) / 100)
        optimizer.zero_grad(set_to_none=True)
        record, parts = {}, []
        if len(train):
            c = draw(train_t, batch_sp)
            ci = c * noisy_copies + torch.randint(noisy_copies, (batch_sp,), generator=gen, device=device)
            lam = head(torch.cat([shots['clean'][c], shots['noisy'][ci]]).float(),
                       torch.cat([ref(batch_sp), shots['sigma'][ci]]))
            l_sp = 0.5 * (shooting_point_nll(lam[:batch_sp], n_first[c], n_last[c])
                          + shooting_point_nll(lam[batch_sp:], n_first[c], n_last[c]))
            parts.append(l_sp)
            record['L_sp'] = float(l_sp.detach())
        if len(anchors):
            half = batch_anchor // 2
            c = torch.cat([draw(first_state, half) if len(first_state) else draw(last_state, half),
                           draw(last_state, half) if len(last_state) else draw(first_state, half)])
            lam = head(torch.cat([in_state['clean'][c], in_state['noisy'][c]]).float(),
                       torch.cat([ref(len(c)), in_state['sigma'][c]]))
            l_in = 0.5 * in_state_nll(lam, torch.cat([in_state['state'][c], in_state['state'][c]]))
            parts.append(0.5 * l_in)
            record['L_anchor'] = float(l_in.detach())
        if not parts:
            step -= 1
            break
        sum(parts).backward()
        optimizer.step()
        with torch.no_grad():
            for p_ema, p in zip(head_ema.parameters(), head.parameters()):
                p_ema.mul_(ema).add_(p.detach(), alpha=1 - ema)
        losses.append(record)
        if len(held) and (step + 1) % head_eval_every == 0:
            nll = held_nll()
            trace.append((step + 1, nll))
            if nll < best[0]:
                best = (nll, copy.deepcopy(head_ema.state_dict()), step + 1)
            elif (step + 1 - best[2]) >= head_patience * head_eval_every:
                break
    head_steps = step + 1
    if best[1] is not None:
        head_ema.load_state_dict(best[1])
    head.load_state_dict(head_ema.state_dict())
    head.eval()
    t_head = time.time() - t0 - t_features
    _log(verbose, round=r, phase='head', n_shooting_points=n_sp, n_held=len(held), steps=head_steps,
         best_step=best[2], best_heldout_nll=best[0] if best[1] is not None else None, seconds=round(t_head, 1))

    # ---- generator: fresh adapters on the frozen head's labels
    adapters = network.adapters()
    for j, adapter in enumerate(adapters):
        adapter.reset_parameters(round_seed + 2 + j)
    counts, gen_losses, rollbacks, probe, label_info, oom = None, [], [], None, None, []
    mb = int(micro_batch)
    if use_generator:
        with torch.no_grad():
            labels = torch.cat([head_ema(pool_f[i:i + 4096].float(), ref(len(pool_f[i:i + 4096])))
                                for i in range(0, len(pool_f), 4096)])
        if label_time_window_ps is not None:
            def labels_of(index):
                out = []
                for i in range(0, len(index), feature_batch):
                    x = bioemu2_atoms(fr_rows[index[i:i + feature_batch]])
                    with torch.no_grad():
                        out.append(head_ema(feats(x, ref(len(x))).float(), ref(len(x))).double().cpu().numpy())
                return np.concatenate(out)
            averaged, label_info = _averaged_labels(data, pool_index, labels.double().cpu().numpy(),
                                                    labels_of, label_time_window_ps, frame_intervals_ps)
            labels = torch.as_tensor(averaged, dtype=torch.float32, device=device)
            _log(verbose, round=r, phase='time-averaged labels', **label_info)
        labels = labels.clamp(-LAMBDA_CLIP, LAMBDA_CLIP)
        probs, counts = _bin_draw_probabilities(labels, bin_min)
        atom_weight = torch.ones(n_real, device=device)
        ligand_atoms = network.system.ligand_bioemu2_atoms
        if len(ligand_atoms):
            atom_weight[torch.as_tensor(ligand_atoms, device=device)] = float(ligand_weight)
        atom_weight = atom_weight / atom_weight.mean()
        mask1 = denoiser.batch.atom_mask

        def den_loss(x, lam, uncond, sigma, g):
            m = len(x)
            rotation = _rotations(m, g, device)
            x = x @ rotation.transpose(1, 2) + torch.randn(m, 1, 3, generator=g, device=device)
            xt = x + sigma[:, None, None] * torch.randn(x.shape, generator=g, device=device)
            pad = torch.zeros(m, n_pad, 3, device=device)
            d = denoiser.denoise(torch.cat([xt, pad[:, n_real:]], 1), sigma, lam, uncond)[:, :n_real]
            target = torch.cat([x, pad[:, n_real:]], 1)
            dp = torch.cat([d, pad[:, n_real:]], 1)
            mask = mask1.repeat_interleave(m, 0).float()
            with torch.autocast('cuda', enabled=False):
                aligned, _, _ = _api.weighted_rigid_align(target.float(), dp.float(), mask, mask)
            se = ((d - aligned[:, :n_real]) ** 2).sum(-1)
            mse = (se * atom_weight).mean(-1)
            return ((sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2 * mse).mean()

        parameters = [p for adapter in adapters for p in adapter.parameters()]
        optimizer_a = torch.optim.AdamW(parameters, lr=lr_adapter, weight_decay=0.0)
        for adapter in adapters:
            adapter.train()
        lr_scale, window, history = 1.0, [], []
        good_state = [copy.deepcopy(adapter.state_dict()) for adapter in adapters]
        for step in range(adapter_steps):
            f = min(1.0, (step + 1) / 100)
            if adapter_lr_schedule == 'cosine':
                f *= 0.5 * (1 + np.cos(np.pi * step / adapter_steps))
            for group in optimizer_a.param_groups:
                group['lr'] = lr_adapter * lr_scale * f
            index = torch.multinomial(probs, batch_den, replacement=True, generator=gen)
            sigma = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_den, generator=gen, device=device))
            uncond = torch.rand(batch_den, generator=gen, device=device) < p_uncond
            while True:
                try:
                    optimizer_a.zero_grad(set_to_none=True)
                    loss = 0.0
                    for i in range(0, batch_den, mb):
                        part = den_loss(pool_x[index[i:i + mb]], labels[index[i:i + mb]], uncond[i:i + mb],
                                        sigma[i:i + mb], gen) * (len(index[i:i + mb]) / batch_den)
                        part.backward()
                        loss += float(part.detach())
                    break
                except torch.OutOfMemoryError:
                    if mb == 1:
                        raise
                    mb = max(1, mb // 2)
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                    oom.append(dict(step=step + 1, micro_batch=mb))
                    _log(verbose, round=r, phase='adapter', event='CUDA out of memory', micro_batch=mb)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer_a.step()
            gen_losses.append(loss)
            window.append(loss)
            if len(window) == guard_window:
                mean = float(np.mean(window))
                window = []
                if len(history) >= 3 and mean > guard_factor * float(np.median(history)):
                    for adapter, state in zip(adapters, good_state):
                        adapter.load_state_dict(state)
                    optimizer_a = torch.optim.AdamW(parameters, lr=lr_adapter, weight_decay=0.0)
                    lr_scale *= 0.5
                    rollbacks.append(dict(step=step + 1, window_loss=mean, median_before=float(np.median(history)),
                                          lr_scale=lr_scale))
                    _log(verbose, round=r, phase='adapter', event='divergence: rolled back', **rollbacks[-1])
                else:
                    history.append(mean)
                    good_state = [copy.deepcopy(adapter.state_dict()) for adapter in adapters]
            if (step + 1) % adapter_log_every == 0:
                _log(verbose, round=r, phase='adapter', step=step + 1, micro_batch=mb,
                     L_den=float(np.mean(gen_losses[-adapter_log_every:])),
                     out_weight_norm=[float(adapter.out.weight.detach().norm()) for adapter in adapters],
                     seconds=round(time.time() - t0 - t_features - t_head, 1))
        for adapter in adapters:
            adapter.eval()
        if label_probe:
            probe = _label_probe(den_loss, pool_x, labels, device, mb)
            _log(verbose, round=r, phase='label probe', **probe)
    t_generator = time.time() - t0 - t_features - t_head
    network.eval()
    info = dict(round=r, recipe='sequential, fresh every round', seed=int(seed), round_seed=round_seed,
                n_shooting_points=int(len(sp_rows)), n_with_outcome=n_sp, n_anchor=int(len(anchors)),
                n_pool=int(len(pool_index)) if use_generator else 0,
                head=dict(n_held=int(len(held)), steps=int(head_steps), best_step=int(best[2]),
                          best_heldout_nll=best[0] if best[1] is not None else None, trace=trace),
                generator_trained=bool(use_generator), adapter_steps=int(adapter_steps if use_generator else 0),
                sigma_data=sigma_data,
                p_uncond=float(p_uncond), learned_null=bool(network.learned_null), ligand_weight=float(ligand_weight),
                bin_min=bin_min, label_note=(
                    f'p_uncond={p_uncond:g} without a learned null: the unlabelled rows give no gradient'
                    if p_uncond and not network.learned_null else None),
                adapter_final_loss=float(np.mean(gen_losses[-200:])) if gen_losses else None,
                adapter_rollbacks=rollbacks, out_of_memory=oom, micro_batch=mb,
                adapter_out_weight_norm=[float(adapter.out.weight.detach().norm()) for adapter in adapters],
                label_counts=counts.tolist() if counts is not None else None, label_probe=probe,
                time_averaged_labels=label_info,
                seconds=dict(features=round(t_features, 1), head=round(t_head, 1),
                             adapter=round(t_generator, 1)))
    return losses, info


# ---------------------------------------------------------------------------
# the params file's fit
# ---------------------------------------------------------------------------

def sequential_fit(params, pathensemble, verbose=False, worker=None, *, seed=None, free_stride=5, shot_stride=1,
                   shot_stride_min_frames=0, **round_settings):
    """The ``fit`` of a BioEmu2 C-GAIMMD params file: one :func:`sequential_round` on the run's data so far.

    Parameters
    ----------
    params : aimmd.Params
        ``params.network`` is the :class:`.network.BioEmu2Network`; its
        ``sorted_states``, ``descriptors_series`` and ``cgaimmd_parameters``
        (``generation_seed``, ``training_device``) are read.
    pathensemble : aimmd.PathEnsemble
        What AIMMD's trainer passes (shot chains, free trajectories, margins).
    verbose : bool, optional
        Print the round's JSON log lines.
    worker : aimmd.Worker, optional
        The trainer (AIMMD passes it): its run folder holds
        ``network{states}.json``, the round counter.
    seed : int, optional
        The base seed (:func:`fit_seed`).
    free_stride, shot_stride, shot_stride_min_frames : int, optional
        Of :func:`aimmd.cgaimmd.dataset.extract_dataset`.
    **round_settings
        Of :func:`sequential_round` (``min_shots_for_generator``,
        ``adapter_steps``, ``ligand_weight``, ``label_time_window_ps``, ...).

    Returns
    -------
    (list, dict)
        The head losses (empty: nothing was trained) and ``{'status',
        'reason', 'round', 'record', 'network_metadata'}``;
        ``network_metadata`` (``round``, ``n_trained_shots``, ``fit``,
        ``seed``, ``seed_source``, ``settings``) is what AIMMD stores next to
        the network.
    """
    from ..devices import device_for
    network = params.network
    device = device_for('training', params)
    if device is not None and network.device != device:
        network.to(device)
    run_directory = _run_directory(worker)
    states = params.sorted_states
    series = getattr(params, 'descriptors_series', None) or DEFAULT_SERIES
    key = (None if run_directory is None else os.path.abspath(os.fspath(run_directory)), states, series,
           int(network.n_atoms), int(free_stride), int(shot_stride), int(shot_stride_min_frames))
    cached = _DATASETS.get(key, {})
    last = cached.get('round', -1)
    r = max(previous_round(run_directory, states), last) + 1
    base_seed, seed_source = fit_seed(params, run_directory, seed)
    data = extract_dataset(pathensemble, states=states, n_atoms=network.n_atoms, series=series,
                           free_stride=free_stride, shot_stride=shot_stride,
                           shot_stride_min_frames=shot_stride_min_frames, previous=cached.get('dataset'))
    _DATASETS[key] = dict(dataset=data, round=last)          # only extracted data is reused, never weights
    losses, record = sequential_round(network, data, r, seed=base_seed, verbose=verbose, **round_settings)
    if losses:                                                # a round that trained counts
        _DATASETS[key]['round'] = r
    record['seed_source'] = seed_source
    record['summary'] = data.summary()
    settings = dict(round_settings, free_stride=int(free_stride), shot_stride=int(shot_stride),
                    shot_stride_min_frames=int(shot_stride_min_frames), series=series)
    metadata = dict(round=r, n_trained_shots=int(record['n_shooting_points']), fit='aimmd.cgaimmd.bioemu2.fit',
                    seed=int(base_seed), seed_source=seed_source,
                    settings=json.loads(json.dumps(settings, default=str)), network=network.settings())
    _log(verbose, **record)
    status = 'trained' if losses else 'skipped'
    return losses, dict(status=status, reason=None if losses else 'no shooting points and no in-state frames',
                        round=r, record=record, network_metadata=metadata)
