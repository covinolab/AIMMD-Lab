"""
aimmd.cgaimmd.flow_matching.pretrain
====================================

End-state pretraining: train the generator and the committor head of a
flow-matching network on unbiased MD of the two end states, before any shot
exists, and write the result as the ``initial_network`` of a C-GAIMMD run.

Before shooting there are no committor labels, so the only conditions are "in
the first state" and "in the last state": the generator learns the two basins
(flow matching conditioned at ``lambda = -+L``) and the committor head
separates them (anchors with the soft targets ``sigmoid(-+L)``). Generated
configurations at intermediate levels are interpolations until the online fit
has seen shooting points. The online fit
(:mod:`aimmd.cgaimmd.flow_matching.fit`) starts every round again from the
network written here.

The inputs MUST be unbiased MD of the end states (no umbrella sampling,
metadynamics or steered frames): the flow learns the density of what it is
given. The training core is the online fit's
(:class:`aimmd.cgaimmd.flow_matching.fit.FlowTrainer`) without shooting
points. The prior is the network's own (the Gaussian of
:class:`~aimmd.cgaimmd.flow_matching.painn.JointPaiNN`).

Typical use, in a script next to the params file:

>>> frames = collect_end_state_frames(['free_A.xtc', 'free_B.xtc'], structure='conf.gro',
...                                   descriptors_function=descriptors_function,
...                                   states_function=states_function, states='ARB')
>>> scale = estimate_coord_scale(frames['coordinates'])       # pass it to the network
>>> report = pretrain_end_states(network, frames, 'pretrained', states='ARB', steps=30000)

Ported from ``aimmd/flow/pretrain.py`` of the flow branch. Pretraining
keeps the network's own prior; the keywords of the flow branch's other priors
are refused with the reason (:data:`aimmd.cgaimmd._legacy.DROPPED_KEYWORDS`).
"""

import json
import os
import time

import numpy as np

from .. import _legacy, protocol


REPORT_NOTE = ('inputs must be unbiased MD of the end states (no umbrella sampling, '
               'metadynamics or steered frames); the input files are listed in trajectories')
"""The note every pretraining report carries."""

MAX_CONSECUTIVE_SKIPPED_STEPS = 1000
"""Consecutive steps with a non-finite gradient after which pretraining stops with ``RuntimeError``."""


def _sorted_states(states):
    states = str(states)
    if len(states) != 3:
        raise ValueError(f'states must be three letters (e.g. "ARB"), got {states!r}')
    return states[::-1] if states[0] > states[2] else states


def estimate_coord_scale(coordinates):
    """Å per flow unit for the PaiNN network: the per-coordinate RMS of centered frames.

    Parameters
    ----------
    coordinates : array-like
        ``(n, M, 3)`` whole frames, Å.

    Returns
    -------
    float
        ``sqrt(mean over frames and atoms of |x - centroid|^2 / 3)``.

    Notes
    -----
    With this ``coord_scale`` centered training frames have per-coordinate
    RMS 1 in flow units, comparable with the standard normal prior. Measure
    it once, on whole molecules, and keep it: changing it invalidates every
    checkpoint.
    """
    x = np.asarray(coordinates, dtype=np.float64)
    centered = x - x.mean(axis=-2, keepdims=True)
    return float(np.sqrt((centered ** 2).sum(-1).mean() / 3.0))


def collect_end_state_frames(trajectories, *, structure, descriptors_function, states_function,
                             states, stride=1, max_frames=200_000, seed=0, chunk_frames=1000):
    """End-state frames of unbiased trajectories, as generator-atom coordinates with their states.

    Parameters
    ----------
    trajectories : sequence of str
        Trajectory files (unbiased MD in the end states).
    structure : str
        Topology or structure file of the full system (MDAnalysis).
    descriptors_function : callable
        ``params.descriptors_function``: a trajectory to the flat Å coordinates of the generator's atoms.
    states_function : callable
        ``params.states_function``: a trajectory to one state letter per frame.
    states : str
        ``params.states``; frames outside the first and last state are dropped.
    stride : int, optional
        Every `stride`-th frame of each file.
    max_frames : int or None, optional
        Random cap on the number of frames kept (time order preserved); None: no cap.
    seed : int, optional
        Seed of the cap.
    chunk_frames : int, optional
        Strided frames of the full system held in memory at a time.

    Returns
    -------
    dict
        ``coordinates`` ``(n, M, 3)`` float32 Å, ``state`` ``(n,)`` ``'<U1'``, ``source`` ``(n,)`` int (index into ``trajectories``), ``trajectories`` (the input files).

    Raises
    ------
    ValueError
        For ``stride < 1`` or ``chunk_frames < 1``, or functions that do not return one row and one state per frame.

    Notes
    -----
    Each file is read in blocks of `chunk_frames` strided frames: a block is
    loaded into an in-memory MDAnalysis universe (with its boxes) and given
    to both functions as one trajectory, as AIMMD's caches would, and only
    the rows and states of its end-state frames are kept. The peak memory is
    therefore one block of the full system (a solvated system of 20 000
    atoms: about 0.25 GB per 1000 frames) plus the kept rows; the functions
    must work frame by frame (every C-GAIMMD descriptor does), so that the
    blocks give the same result as one trajectory.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    sorted_states = _sorted_states(states)
    first, last = sorted_states[0], sorted_states[2]
    stride = int(stride)
    if stride < 1:
        raise ValueError(f'stride must be >= 1, got {stride}')
    chunk_frames = int(chunk_frames)
    if chunk_frames < 1:
        raise ValueError(f'chunk_frames must be >= 1, got {chunk_frames}')
    names = [str(t) for t in trajectories]
    coords, labels, source = [], [], []
    memory = None

    def describe(fname, k, positions, boxes):
        nonlocal memory
        if memory is None:
            memory = mda.Universe(str(structure))
        dimensions = None if any(b is None for b in boxes) else np.stack(boxes)
        memory.load_new(np.stack(positions), format=MemoryReader, dimensions=dimensions)
        descriptors = np.asarray(descriptors_function(memory.trajectory))
        state = np.asarray(states_function(memory.trajectory)).astype('<U1')
        if len(descriptors) != len(positions) or len(state) != len(positions):
            raise ValueError(f'{fname}: {len(positions)} frames but {len(descriptors)} '
                             f'descriptor rows and {len(state)} states')
        keep = (state == first) | (state == last)
        n = int(keep.sum())
        width = descriptors.reshape(len(descriptors), -1).shape[1]
        coords.append(descriptors[keep].reshape(n, width // 3, 3).astype(np.float32))
        labels.append(state[keep])
        source.append(np.full(n, k, dtype=np.int64))

    for k, fname in enumerate(names):
        universe = mda.Universe(str(structure), fname)
        positions, boxes = [], []
        for ts in universe.trajectory[::stride]:
            positions.append(ts.positions.astype(np.float32).copy())
            boxes.append(None if ts.dimensions is None else np.asarray(ts.dimensions).copy())
            if len(positions) == chunk_frames:
                describe(fname, k, positions, boxes)
                positions, boxes = [], []
        if positions:
            describe(fname, k, positions, boxes)
        universe.trajectory.close()
    if coords:
        coordinates = np.concatenate(coords)
        state = np.concatenate(labels)
        src = np.concatenate(source)
    else:
        coordinates = np.zeros((0, 0, 3), np.float32)
        state = np.zeros(0, '<U1')
        src = np.zeros(0, np.int64)
    if max_frames is not None and len(state) > int(max_frames):
        keep = np.sort(np.random.default_rng(seed).choice(len(state), int(max_frames),
                                                          replace=False))
        coordinates, state, src = coordinates[keep], state[keep], src[keep]
    return {'coordinates': coordinates, 'state': state, 'source': src, 'trajectories': names}


def _is_count(value, minimum):
    from numbers import Integral
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _holdout_split(source, holdout_fraction, rng):
    """Held-out mask: whole sources with >= 5 of them, else the last fraction of each (in order)."""
    source = np.asarray(source)
    sources = np.unique(source)
    held = np.zeros(len(source), dtype=bool)
    if len(sources) >= 5:
        n_hold = max(1, int(round(holdout_fraction * len(sources))))
        chosen = rng.choice(sources, size=min(n_hold, len(sources)), replace=False)
        held = np.isin(source, chosen)
    else:
        for s in sources:
            idx = np.flatnonzero(source == s)
            held[idx[int((1.0 - holdout_fraction) * len(idx)):]] = True
    return held


def _prior_report(network):
    """What the network's prior is (its ``settings()`` when it has one)."""
    prior = getattr(network, 'prior', None)
    if prior is None:
        return {'kind': 'the network\'s own sample_prior', 'network': type(network).__name__}
    settings = getattr(prior, 'settings', None)
    if callable(settings):
        return dict(settings())
    return {'kind': type(prior).__name__}


def pretrain_end_states(network, frames, out_dir, *, states, steps, batch_size=64, lr=3e-4,
                        weight_decay=1e-5, warmup_steps=0, lr_schedule='cosine', grad_clip=1.0,
                        ema_decay=None, holdout_fraction=0.1, p_uncond=0.15,
                        t_schedule='uniform', fm_max_sample_loss=1e3, checkpoint_steps=(),
                        device='auto', seed=0, must_stop=None, lambda_clip=None,
                        **dropped_keywords):
    """Pretrain a flow-matching network on end-state frames and write it as a network file.

    Parameters
    ----------
    network : object
        A flow-matching C-GAIMMD network (the params file's instance); trained in place.
    frames : dict
        :func:`collect_end_state_frames` output (``coordinates``, ``state``, ``source``, optionally ``trajectories``).
    out_dir : str
        Output folder (created).
    states : str
        ``params.states``.
    steps : int
        Optimizer steps; required, system dependent (NANMA: 30 000 cosine steps).
    batch_size : int, optional
        Anchors and flow-matching samples per step (stratified 50/50 by state).
    lr, weight_decay, warmup_steps, grad_clip : optional
        AdamW and its schedule, as in the online fit.
    lr_schedule : {'cosine', 'constant'}, optional
        Learning-rate schedule over `steps` after the warm-up.
    ema_decay : float, optional
        Keep an EMA of the weights and save it; None saves the live weights.
    holdout_fraction : float, optional
        Held-out share (whole trajectories with >= 5 of them, else the last part of each).
    p_uncond : float, optional
        Probability of the null token in flow matching.
    t_schedule : str, optional
        Flow-time distribution (:func:`aimmd.cgaimmd.flow_matching.flowmath.sample_time`).
    fm_max_sample_loss : float or None, optional
        Runaway guard of flow matching.
    checkpoint_steps : sequence of int, optional
        Steps after which ``checkpoint_step<k:06d>.pt`` (the saved weights) is written.
    device : str, optional
        ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'cuda:N'`` (pretraining runs outside a worker task).
    seed : int, optional
        Seed of the split and of the trainer's random streams.
    must_stop : callable, optional
        Polled before every step; True ends the training early (outputs are still written).
    lambda_clip : float, optional
        The conditioning range ``L`` for a network without a ``lambda_clip`` member (as
        ``cgaimmd_parameters['lambda_clip']``).
    **dropped_keywords
        Old keywords of the flow branch's other priors (``prior``,
        ``prior_sigma``, ``prior_bank``, ...): None, and ``prior='gaussian'``,
        are accepted with one ``DeprecationWarning``; any other value raises
        ``TypeError`` with the reason
        (:data:`aimmd.cgaimmd._legacy.DROPPED_KEYWORDS`).

    Returns
    -------
    dict
        The report written to ``out_dir/pretrain.json``.

    Raises
    ------
    TypeError
        For a network that is not a flow-matching C-GAIMMD network, without a
        conditioning range (neither the network's ``lambda_clip`` nor the
        keyword, or the two disagree), or for an old prior keyword.
    ValueError
        For bad keyword values, or a state without training frames.
    RuntimeError
        After :data:`MAX_CONSECUTIVE_SKIPPED_STEPS` consecutive steps with a non-finite gradient (nothing is written).

    Notes
    -----
    The trainer is the online fit's core with ``n_sp = 0``, ``n_anchor =
    n_fm = batch_size`` and every flow-matching sample a basin frame
    conditioned at ``-+L`` (``instate_fm_share = 1``); the other training
    settings are the fit's defaults
    (:data:`aimmd.cgaimmd.flow_matching.fit.FLOW_FIT_DEFAULTS`). The committor
    terms use :func:`aimmd.cgaimmd.protocol.coordinate_committor` (the frames
    are in memory). Only protocol and flow-matching members are used, so any
    flow-matching network can be pretrained (a network without
    ``lambda_clip`` takes ``L`` from the keyword, as the online fit takes it
    from ``cgaimmd_parameters``). ``network<s>.h5`` holds tensors only (the
    EMA with `ema_decay`, else the live weights) and loads into the params
    file's network as ``initial_network``.
    """
    import torch
    from . import fit as ff
    from ..devices import resolve_device
    from ..io import atomic_torch_save, atomic_write_json

    _legacy.dropped_keywords('pretrain_end_states', dropped_keywords)
    t_start = time.time()
    problems = []
    if not _is_count(steps, 1):
        problems.append(f'steps must be an int >= 1, got {steps!r}')
    if not _is_count(batch_size, 2):
        problems.append(f'batch_size must be an int >= 2 (one anchor of each end state per '
                        f'step), got {batch_size!r}')
    if not _is_count(seed, 0):
        problems.append(f'seed must be an int >= 0, got {seed!r}')
    if isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, (int, float)) \
            or not 0.0 <= float(holdout_fraction) < 1.0:
        problems.append(f'holdout_fraction must be a number in [0, 1), got {holdout_fraction!r}')
    if isinstance(checkpoint_steps, (str, bytes)) \
            or not all(_is_count(k, 1) for k in checkpoint_steps):
        problems.append(f'checkpoint_steps must be ints >= 1, got {checkpoint_steps!r}')
    if problems:
        raise ValueError('pretrain_end_states: ' + '; '.join(problems))
    protocol.check_network_protocol(network)
    member_problems = ff._fit_member_problems(network)
    if member_problems:
        raise TypeError(f'pretrain_end_states: {type(network).__name__} is not a flow-matching '
                        f'network: ' + '; '.join(member_problems))
    lambda_clip_keyword = lambda_clip
    try:
        lambda_clip = protocol.lambda_clip_of(network, {'lambda_clip': lambda_clip})
    except TypeError as exc:                     # the online fit's rule, with this keyword's name
        raise TypeError('pretrain_end_states: ' + str(exc).replace(
            "cgaimmd_parameters['lambda_clip']", 'the lambda_clip keyword')) from None
    sorted_states = _sorted_states(states)
    first, last = sorted_states[0], sorted_states[2]
    # the online fit's validated settings: pretraining is its trainer core without shooting
    # points; the keywords that differ in name (steps, batch_size, seed) are checked above
    try:
        settings = ff._settings_from({
            'lr': lr, 'weight_decay': weight_decay, 'warmup_steps': warmup_steps,
            'lr_schedule': lr_schedule, 'total_steps': steps, 'grad_clip': grad_clip,
            'ema_decay': ema_decay, 'n_sp': 0, 'n_anchor': batch_size, 'n_fm': batch_size,
            'instate_fm_share': 1.0, 'p_uncond': p_uncond, 't_schedule': t_schedule,
            'fm_max_sample_loss': fm_max_sample_loss, 'trainer_seed': seed, 'grad_penalty': 0.0})
    except ValueError as exc:
        raise ValueError(str(exc).replace('flow_matching_fit: ', 'pretrain_end_states: ', 1)) \
            from None
    steps = int(steps)

    coordinates = np.asarray(frames['coordinates'], dtype=np.float32)
    state = np.asarray(frames['state']).astype('<U1')
    source = np.asarray(frames.get('source', np.zeros(len(state), dtype=np.int64)))
    trajectories = [str(t) for t in frames.get('trajectories', [])]
    in_state = (state == first) | (state == last)
    counts = {s: int((state == s).sum()) for s in sorted(set(state.tolist()))}
    rng = np.random.default_rng(seed)
    held = _holdout_split(source, float(holdout_fraction), rng)
    train = np.flatnonzero(in_state & ~held)
    holdout = np.flatnonzero(in_state & held)
    for s in (first, last):
        if not (state[train] == s).any():
            raise ValueError(f'no training frames of state {s!r}; pretraining needs both end '
                             f'states populated')
    n_sources = len(np.unique(source))
    split = {'kind': (f'by trajectory ({len(np.unique(source[held]))} of {n_sources})'
                      if n_sources >= 5 else
                      f'last {holdout_fraction:.0%} of each of {n_sources} trajectories'),
             'n_train': int(len(train)), 'n_holdout': int(len(holdout)),
             'held_out_sources': sorted(int(s) for s in np.unique(source[held]))}

    dev = resolve_device(device)
    network.to(dev)
    codes = np.where(state == first, 1, np.where(state == last, 2, 0)).astype(np.int8)
    n = len(train)
    names = np.array([trajectories[int(s)] if int(s) < len(trajectories) else str(int(s))
                      for s in source[train]], dtype=object)
    from ..dataset import TrainingDataset
    dataset = TrainingDataset(
        sp_rows=np.zeros((0, coordinates.shape[1] * 3), np.float32),
        sp_r_first=np.zeros(0, np.float32), sp_r_last=np.zeros(0, np.float32),
        sp_group=np.zeros(0, np.int64), sp_fname=[],
        fr_rows=coordinates[train].reshape(n, -1), fr_state=codes[train],
        fr_w=np.ones(n), fr_group=np.full(n, -1, np.int64), fr_fname=names,
        fr_loc=train.astype(np.int64), notes={'source': 'pretrain_end_states'})

    def committor_fn(net, rows, requires_grad):
        x = torch.as_tensor(np.asarray(rows), dtype=torch.float32,
                            device=protocol.network_device(net)).reshape(len(rows), -1, 3)
        if requires_grad:
            x = x.clone().requires_grad_(True)
        return protocol.coordinate_committor(net, x), x

    trainer = ff.FlowTrainer(network, settings, lambda_clip=lambda_clip, device=dev,
                             committor_fn=committor_fn, seed=seed)
    trainer.set_data(dataset)
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = sorted({int(k) for k in checkpoint_steps})
    written = []
    history, block = [], []
    stopped = False
    consecutive_skipped = 0
    t_train = time.time()
    for _ in range(steps):
        if must_stop is not None and must_stop():
            stopped = True
            break
        total, terms = trainer.step()
        block.append({'total': total, **terms})
        if terms.get('skipped'):
            consecutive_skipped += 1
            if consecutive_skipped % 100 == 0:
                print(f'WARNING: pretrain_end_states: {consecutive_skipped} consecutive steps '
                      f'skipped (non-finite gradient) up to step {trainer.step_count}', flush=True)
            if consecutive_skipped >= MAX_CONSECUTIVE_SKIPPED_STEPS:
                raise RuntimeError(f'pretrain_end_states: training diverged: '
                                   f'{consecutive_skipped} consecutive steps with a non-finite '
                                   f'gradient up to step {trainer.step_count}; no network file '
                                   f'was written (lower lr, or check the frames)')
        else:
            consecutive_skipped = 0
        if trainer.step_count in checkpoints:
            fname = os.path.join(out_dir, f'checkpoint_step{trainer.step_count:06d}.pt')
            atomic_torch_save(network.state_dict(), fname)
            written.append(os.path.basename(fname))
        if len(block) == 1000:
            history.append(_summarize(block, trainer.step_count))
            block = []
    if block:
        history.append(_summarize(block, trainer.step_count))
    train_seconds = time.time() - t_train

    holdout_nll = None
    if len(holdout):
        lam = torch.as_tensor(trainer.committor_of_rows(network,
                                                        coordinates[holdout].reshape(
                                                            len(holdout), -1)))
        edge = 1.0 / (1.0 + np.exp(lambda_clip))
        y = torch.as_tensor(np.where(state[holdout] == last, 1.0 - edge, edge))
        holdout_nll = float(ff.committor_nll(lam, y, torch.ones_like(y)))
    network_file = os.path.join(out_dir, f'network{sorted_states}.h5')
    sha = atomic_torch_save(network.state_dict(), network_file)
    report = {
        'note': REPORT_NOTE,
        'settings': {'states': states, 'steps': steps, 'batch_size': batch_size, 'lr': lr,
                     'weight_decay': weight_decay, 'warmup_steps': warmup_steps,
                     'lr_schedule': lr_schedule, 'grad_clip': grad_clip, 'ema_decay': ema_decay,
                     'holdout_fraction': holdout_fraction, 'p_uncond': p_uncond,
                     't_schedule': t_schedule, 'fm_max_sample_loss': fm_max_sample_loss,
                     'checkpoint_steps': checkpoints, 'device': device, 'seed': seed,
                     'lambda_clip': lambda_clip_keyword},
        'states': states, 'sorted_states': sorted_states,
        'trajectories': trajectories, 'n_frames': int(len(state)), 'state_counts': counts,
        'split': split, 'prior': _prior_report(network), 'lambda_clip': float(lambda_clip),
        'coord_scale_network': (float(getattr(network, 'coord_scale'))
                                if getattr(network, 'coord_scale', None) is not None else None),
        'coord_scale_measured': (estimate_coord_scale(coordinates[train]) if n else None),
        'steps_done': int(trainer.step_count), 'stopped': stopped,
        'skipped_steps': int(trainer.skipped_steps), 'loss_history': history,
        'holdout_anchor_nll': holdout_nll,
        'saved_weights': 'ema' if ema_decay else 'live',
        'network_file': os.path.basename(network_file), 'network_sha256': sha,
        'checkpoints': written, 'device': str(dev),
        'seconds': {'train': round(train_seconds, 3), 'total': round(time.time() - t_start, 3)}}
    atomic_write_json(os.path.join(out_dir, 'pretrain.json'), report)
    return json.loads(json.dumps(report))


def _summarize(block, step):
    keys = sorted({k for entry in block for k in entry if k != 'skipped'})
    out = {'steps': [int(step - len(block)), int(step)]}
    for key in keys:
        values = [entry[key] for entry in block if key in entry]
        out[key] = float(np.mean(values))
    out['skipped'] = int(sum(entry.get('skipped', 0.0) for entry in block))
    return out
