"""
aimmd.cgaimmd.flow_matching.fit
===============================

The online fit of a flow-matching C-GAIMMD network: one training round per
call of AIMMD's trainer, every round trained again from the same start.

The network is the PaiNN network of C-GAIMMD
(:class:`~aimmd.cgaimmd.flow_matching.painn.JointPaiNN`), or any network with
a committor ``forward`` and the flow-matching members of
:mod:`~aimmd.cgaimmd.flow_matching.sampler` (``velocity`` and
``sample_prior``, both with the keyword ``uncond_mask``). The params file
wraps :func:`flow_matching_fit` in the four-argument ``fit`` AIMMD calls::

    from aimmd.cgaimmd.flow_matching.fit import flow_matching_fit

    def fit(params, pathensemble, verbose=False, worker=None):
        return flow_matching_fit(params, pathensemble, verbose=verbose, worker=worker,
                                 steps_per_round=4000, relabel_every=1000)

**Every round starts from the same weights**: a round never continues from
the previous round's weights. The start is
``params.initial_network`` (typically the end-state pretraining of
:mod:`~aimmd.cgaimmd.flow_matching.pretrain`) when it is set, else
``network.reset_parameters()`` under the run's seed. The optimizer, the
exponential moving average (EMA) of the weights and the loss normalizer are
new in every round. A round trains ``steps_per_round`` steps, or stops
earlier on the held-out chains (``early_stopping_patience``), and leaves its
EMA in ``params.network``, which AIMMD's trainer saves as
``network{states}.h5`` and scores. Nothing of the previous round's weights
enters a round: not as its start, and not as its flow-matching labels, which
come from the round's own EMA. What is carried between rounds is bookkeeping
(the round counter, the held-out chains, the first-round no-op; in
``flow_fit{states}/state.json``) and, in memory, the extracted dataset.

One step minimizes a single loss:

.. math::

   L = N_q(w_{sp} L_{sp} + w_{gp} L_{gp} + w_{a} L_{anchor}) + w_{fm} N_{fm}(L_{FM})

- ``L_sp``: the binomial negative log-likelihood of the two-way outcomes of
  the shooting points (AIMMD's shooting likelihood); ``L_gp``: the gradient
  penalty ``mean((d lambda / d x)^2)`` on the same batch; ``L_anchor``: the
  negative log-likelihood of in-state frames with the soft targets
  ``sigmoid(-+L)`` (``L`` = ``lambda_clip``, the conditioning range);
- ``L_FM``: conditional flow matching on frames of the reweighted ensemble,
  conditioned on the EMA's own committor (in-state frames at ``-+L``), with
  the network's optimal-transport pairing and label dropout (``p_uncond``:
  the share of rows trained with the null token, which classifier-free
  guidance needs);
- ``N_q``, ``N_fm``: division by a running magnitude of each term
  (:class:`LossNormalizer`), so that the weights mean the same for every
  system and stage of training.

The committor terms use AIMMD's input path, exactly as the trainer's value
pass does: dense descriptor rows, or ``params.descriptor_transform`` in graph
mode (``graphs=True``).

The flow-matching labels of the reactive frames are the committor of the
round's EMA, computed at the start of the round and again every
``relabel_every`` steps (default 1,000 of ``steps_per_round`` 4,000, the
recipe of the PaiNN evidence). Relabelling within the round is what lets the
generator's conditioning learn from the shooting points: at the start of a
round the EMA is the start network, and with the end-state pretraining as the
start that is the committor of a network that has seen no shooting point.
With ``relabel_every=0`` (or not below ``steps_per_round``) the labels stay
the start network's for the whole round, and the fit warns once.

Options, off by default: ``basin_draw='density'`` (in-state frames drawn by
their density weight), ``holdout_every`` with
``early_stopping_patience`` (early stopping on the held-out chains' shooting
likelihood), ``split_backward`` (two backward passes per step, for a small
GPU) and ``graphs`` (graph mode).
"""

# Provenance: Ported from ``aimmd/network/flow_fit.py`` of the flow branch (``FlowTrainer``,
# ``LossNormalizer``, ``committor_nll``, ``flow_fit``); the warm start (the
# round continued the previous round's live weights, optimizer and EMA), the
# side checkpoint of those weights and its reconciliation with the network
# file, and the flow-matching labels read from the values caches (the previous
# round's network) were dropped. ``split_backward`` comes from the run-local
# patch of the Goethe-dataset campaign (16 Oct 2026).

import copy
import json
import math
import os
import time
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch

from .. import protocol
from .._once import once
from ..dataset import (DEFAULT_SERIES, extract_dataset, is_initial_path, path_weights,
                       resolve_exclusions)
from ..fitting import (FitSettings, fit_seed, previous_round, run_directory,
                       settings_only_active)
from ..io import atomic_write_json, file_sha256
from . import flowmath
from .sampler import flow_matching_problems, pair, to_flow


__all__ = ['flow_matching_fit', 'FlowTrainer', 'LossNormalizer', 'committor_nll',
           'release_trainer', 'FLOW_FIT_DEFAULTS', 'BASIN_DRAWS', 'STATE_SCHEMA', 'FIT_NAME',
           'labels_from_start_problem']

FLOW_FIT_DEFAULTS = {
    'graphs': False,
    'steps_per_round': 4000,
    'max_round_seconds': None,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 500,
    'lr_schedule': 'constant',
    'total_steps': None,
    'grad_clip': 1.0,
    'ema_decay': 0.999,
    'n_sp': 64,
    'n_anchor': 32,
    'n_fm': 64,
    'committor_weight': 1.0,
    'anchor_weight': 0.5,
    'grad_penalty': 0.1,
    'fm_weight': 1.0,
    'normalize_losses': True,
    'normalizer_decay': 0.99,
    'normalizer_warmup': 50,
    'normalizer_max_ratio': 10.0,
    'instate_fm_share': 0.2,
    'basin_draw': 'uniform',
    'n_lambda_bins': 20,
    'p_uncond': 0.15,
    't_schedule': 'uniform',
    'fm_max_sample_loss': 1e3,
    'relabel_max_frames': 100000,
    'relabel_chunk': 128,
    'relabel_every': 1000,
    'holdout_fraction': 0.1,
    'holdout_seed': 12345,
    'holdout_chains': None,
    'hold_out_anchor_frames': False,
    'holdout_every': 0,
    'early_stopping_patience': None,
    'free_stride': 5,
    'shot_stride': 1,
    'shot_stride_min_frames': 0,
    'exclude_frames': None,
    'fill_missing_values': True,
    'min_basin_frames': 1,
    'idle_sleep_s': 10.0,
    'round_oom_retries': 30,
    'round_oom_wait_s': 60.0,
    'trainer_seed': None,
    'split_backward': False,
    'log_every': 0,
    'state_dir': None,
    'cache_dataset': True,
}
"""Keyword arguments of :func:`flow_matching_fit` with their defaults.

``lr_schedule='cosine'`` decays to 0 at ``total_steps`` (None: the round's
``steps_per_round``); the schedule and the warm-up restart with every round,
so ``total_steps`` larger than ``steps_per_round`` is refused (the cosine
would never reach 0).
``trainer_seed`` None takes the run's seed (:func:`aimmd.cgaimmd.fitting.fit_seed`).
``steps_per_round`` 4,000 and ``relabel_every`` 1,000 are the online recipe
of the PaiNN evidence (4,000 steps gave the best held-out committor on
chignolin; every PaiNN recipe relabelled every 1,000 steps); with every round
trained from the start, relabelling within the round is what brings the
shooting points into the generator's labels.
"""

BASIN_DRAWS = ('uniform', 'density')
"""The values of ``basin_draw``: how the anchors and the in-state share of flow matching are drawn."""

STATE_SCHEMA = 'aimmd.flow_fit/2'
"""The ``schema`` entry of ``flow_fit{states}/state.json`` (the fit's bookkeeping between rounds)."""

OLD_CHECKPOINT_SCHEMA = 'aimmd.flow_fit/1'
"""The schema of the side checkpoint ``state.pt`` of earlier versions (bookkeeping read, weights not)."""

FIT_NAME = 'aimmd.cgaimmd.flow_matching.fit'
"""The ``fit`` entry of the network metadata."""

ROUND_SEED_STRIDE = 1000
"""Round ``r`` (1, 2, ...) seeds its random streams with ``seed + ROUND_SEED_STRIDE (r - 1)``."""

_RUNS = {}
# what a fit keeps between rounds in this process, per fit-state folder (or ('memory', id(network))
# without one): {key: _RunMemory}; cleared by tests/conftest.py

_clock = time.monotonic      # wall clock of max_round_seconds and the round records


# --------------------------------------------------------------------------------------------
# loss pieces
# --------------------------------------------------------------------------------------------

def committor_nll(logits, y, n):
    """AIMMD's shooting likelihood with soft targets and multiplicities.

    Parameters
    ----------
    logits : torch.Tensor
        ``(B,)`` logit committors.
    y : torch.Tensor
        ``(B,)`` targets in ``[0, 1]`` (``r_last / (r_first + r_last)``, or ``sigmoid(-+L)`` for anchors).
    n : torch.Tensor
        ``(B,)`` multiplicities (``r_first + r_last``, or 1).

    Returns
    -------
    torch.Tensor
        ``sum(n * BCE_logits(logits, y)) / sum(n)``.

    Notes
    -----
    Through ``binary_cross_entropy_with_logits``, which stays finite for the
    large ``|lambda|`` of the committor's flanks.
    """
    nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * n
    return nll.sum() / n.sum().clamp_min(1e-8)


class _RunningMagnitude:
    """Exponential moving average of a loss term's magnitude, outlier-capped."""

    floor = 1e-8

    def __init__(self, decay, max_ratio, value=0.0, count=0):
        self.decay, self.max_ratio = float(decay), float(max_ratio)
        self.value, self.count = float(value), int(count)

    def update(self, x):
        if not math.isfinite(x):
            return
        if self.count >= 1 and self.value > 0:
            x = min(x, self.max_ratio * self.value)
        self.count += 1
        self.value = x if self.count == 1 else self.decay * self.value + (1 - self.decay) * x

    @property
    def scale(self):
        return max(self.value, self.floor)


class LossNormalizer:
    """Divides each named loss term by its running magnitude, after a warm-up.

    Parameters
    ----------
    decay : float, optional
        Decay of the exponential moving average of each term's magnitude.
    warmup : int, optional
        Steps (counted by :meth:`advance`) during which terms are returned unscaled.
    max_ratio : float, optional
        An update is capped at this multiple of the current magnitude (outlier guard).
    enabled : bool, optional
        False returns every term unscaled (the magnitudes are still tracked).

    Notes
    -----
    Without normalization the relative weights of the terms would mean
    something different for every system and every stage of training (the
    flow-matching term falls by orders of magnitude, the committor terms
    barely move). The normalizer is detached: it scales gradients, it does
    not contribute to them.
    """

    def __init__(self, decay=0.99, warmup=50, max_ratio=10.0, enabled=True):
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.max_ratio = float(max_ratio)
        self.enabled = bool(enabled)
        self.step = 0
        self.terms = {}

    def __call__(self, name, value):
        """`value` divided by the running magnitude of term `name` (identity in the warm-up)."""
        magnitude = self.terms.setdefault(name, _RunningMagnitude(self.decay, self.max_ratio))
        magnitude.update(float(value.detach()))
        if not self.enabled or self.step < self.warmup:
            return value
        return value / magnitude.scale

    def advance(self):
        """Count one optimizer step."""
        self.step += 1

    def state_dict(self):
        """``{'step': int, 'terms': {name: {'value': float, 'count': int}}}``."""
        return {'step': int(self.step),
                'terms': {name: {'value': float(m.value), 'count': int(m.count)}
                          for name, m in self.terms.items()}}

    def load_state_dict(self, state):
        """Restore a :meth:`state_dict`."""
        self.step = int(state['step'])
        self.terms = {name: _RunningMagnitude(self.decay, self.max_ratio, t['value'], t['count'])
                      for name, t in state['terms'].items()}


# --------------------------------------------------------------------------------------------
# the trainer core
# --------------------------------------------------------------------------------------------

class FlowTrainer:
    """The training state of one round: live weights, EMA, optimizer, normalizer, random streams, data.

    Parameters
    ----------
    network : torch.nn.Module
        The EMA container (``params.network``); it holds the EMA of the live weights after every step.
    settings : dict
        :data:`FLOW_FIT_DEFAULTS` updated with the fit's keywords (optimizer, loss, batch, normalizer keys are read).
    lambda_clip : float
        The conditioning range ``L``.
    device : torch.device
        Where the live weights, the network and the random generator live.
    committor_fn : callable
        ``committor_fn(net, rows, requires_grad) -> (logits (B,), positions)`` for descriptor rows of the dataset.
    seed : int, optional
        Seed of the trainer's random streams; None: ``settings['trainer_seed']``, else 0.

    Notes
    -----
    ``live`` is a ``copy.deepcopy`` of `network`, made when the trainer is
    created; the optimizer is AdamW over its parameters. The EMA update after
    every finite step is ``theta_ema <- d theta_ema + (1 - d) theta`` on the
    network's parameters (the network follows the live weights for
    ``ema_decay`` None or 0), buffers copied from the live network. The
    trainer draws only from its own numpy ``Generator`` (``rng``) and
    ``torch.Generator`` (``generator``), both seeded with `seed`. The
    flow-matching fit creates a new trainer for every round (after loading
    the round's start weights into the network), so the optimizer, the EMA
    and the normalizer never outlive a round; end-state pretraining uses one
    for its whole run.

    Basin frames (the anchors and the in-state share of flow matching) are
    drawn among the in-state frames of positive weight, uniformly
    (``basin_draw='uniform'``, the default) or with probabilities
    proportional to their AIMMD density weight ``fr_w``
    (``basin_draw='density'``).

    With ``split_backward`` (``settings``, default off) a step backpropagates
    the committor part right after it is formed and frees its graph before
    the flow-matching forward pass, then backpropagates the flow-matching
    part: the gradient is the same sum (to float32 rounding), the peak memory
    the larger part instead of the sum. Off, a step is the ported step bit
    for bit.
    """

    def __init__(self, network, settings, *, lambda_clip, device, committor_fn, seed=None):
        self.settings = dict(settings)
        self.lambda_clip = float(lambda_clip)
        self.device = torch.device(device) if device is not None \
            else protocol.network_device(network)
        self.committor_fn = committor_fn
        self.network = network
        network.to(self.device)
        self.live = copy.deepcopy(network).to(self.device)
        self.optimizer = torch.optim.AdamW(self.live.parameters(), lr=float(self.settings['lr']),
                                           weight_decay=float(self.settings['weight_decay']))
        s = self.settings
        self.normalizer = LossNormalizer(decay=s['normalizer_decay'], warmup=s['normalizer_warmup'],
                                         max_ratio=s['normalizer_max_ratio'],
                                         enabled=s['normalize_losses'])
        if seed is None:
            seed = self.settings.get('trainer_seed')
        self.seed = int(seed or 0)
        self.rng = np.random.default_rng(self.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.step_count = 0
        self.skipped_steps = 0
        self.last_terms = {}
        self.dataset = None
        self.held_out_chains = ()
        empty = np.zeros(0, dtype=np.int64)
        self.sp_train = self.sp_hold = self.anchor_first = self.anchor_last = empty
        self._basin_p = {1: None, 2: None}     # draw probabilities of the basin frames
        self.fm_candidates = self.fm_pool = empty
        self.fm_lambda = np.zeros(0)
        self._strata, self._strata_p = [], []

    # -- data ------------------------------------------------------------------------------------

    def set_data(self, dataset, held_out_chains=(), *, hold_out_anchor_frames=False):
        """Use `dataset` from the next step on.

        Parameters
        ----------
        dataset : aimmd.cgaimmd.dataset.TrainingDataset
            Shooting points and frames.
        held_out_chains : sequence of int, optional
            Chains whose shooting points leave the NLL and whose frames leave the flow-matching pool.
        hold_out_anchor_frames : bool, optional
            Also drop the held-out chains' in-state frames from the anchors.

        Notes
        -----
        Clears the flow-matching targets: call :meth:`set_fm_targets` before
        the next step (without targets every flow-matching sample is a basin
        frame). With ``basin_draw='density'`` the basin frames' draw
        probabilities (their ``fr_w``, normalized per state) are set here.
        """
        held = sorted({int(c) for c in held_out_chains})
        self.dataset = dataset
        self.held_out_chains = tuple(held)
        sp_held = np.isin(dataset.sp_group, held)
        self.sp_train = np.flatnonzero(~sp_held)
        self.sp_hold = np.flatnonzero(sp_held)
        fr_held = np.isin(dataset.fr_group, held) & (dataset.fr_group >= 0)
        anchor_ok = ~fr_held if hold_out_anchor_frames else np.ones(len(fr_held), dtype=bool)
        positive = dataset.fr_w > 0
        self.anchor_first = np.flatnonzero((dataset.fr_state == 1) & anchor_ok & positive)
        self.anchor_last = np.flatnonzero((dataset.fr_state == 2) & anchor_ok & positive)
        density = self.settings.get('basin_draw', 'uniform') == 'density'
        self._basin_p = {1: None, 2: None}
        if density:
            for code, members in ((1, self.anchor_first), (2, self.anchor_last)):
                w = np.asarray(dataset.fr_w[members], dtype=np.float64)
                if len(w) and w.sum() > 0:
                    self._basin_p[code] = w / w.sum()
        self.fm_candidates = np.flatnonzero((dataset.fr_state == 0) & ~fr_held & positive)
        self.set_fm_targets(np.zeros(0, dtype=np.int64), np.zeros(0))

    def set_fm_targets(self, pool, lambda_hat):
        """Flow-matching pool of reactive frames and their conditioning levels.

        Parameters
        ----------
        pool : numpy.ndarray
            ``(n,)`` indices into the dataset's frames.
        lambda_hat : numpy.ndarray
            ``(n,)`` committor logits of the pool frames (clipped to ``[-L, L]`` here).

        Notes
        -----
        Strata ``np.linspace(-L, L, n_lambda_bins + 1)``, empty strata
        dropped; a stratum is drawn uniformly among the non-empty ones and a
        frame within it with probability proportional to its weight.
        """
        pool = np.asarray(pool, dtype=np.int64)
        lam = np.clip(np.asarray(lambda_hat, dtype=np.float64), -self.lambda_clip,
                      self.lambda_clip)
        if len(pool) != len(lam):
            raise ValueError(f'{len(pool)} pool frames for {len(lam)} values')
        self.fm_pool, self.fm_lambda = pool, lam
        n_bins = int(self.settings['n_lambda_bins'])
        edges = np.linspace(-self.lambda_clip, self.lambda_clip, n_bins + 1)
        which = np.clip(np.digitize(lam, edges) - 1, 0, n_bins - 1)
        members = [np.flatnonzero(which == k) for k in range(n_bins)]
        self._strata = [m for m in members if len(m)]
        w = self.dataset.fr_w[pool] if self.dataset is not None and len(pool) else np.zeros(0)
        self._strata_p = [w[m] / w[m].sum() if w[m].sum() > 0 else None for m in self._strata]

    def _draw_basin(self, code, n):
        """`n` basin frames of state `code` (1 first, 2 last): uniform, or proportional to ``fr_w``."""
        members = self.anchor_first if code == 1 else self.anchor_last
        return self.rng.choice(members, n, p=self._basin_p[code])

    def _draw_fm(self, n):
        """Frame indices and levels of a flow-matching batch: basin share (half per state) plus strata."""
        L = self.lambda_clip
        n_in = int(round(self.settings['instate_fm_share'] * n)) if self._strata else n
        n_first = n_in // 2
        idx = [self._draw_basin(1, n_first), self._draw_basin(2, n_in - n_first)]
        lam = [np.full(n_first, -L), np.full(n_in - n_first, L)]
        n_r = n - n_in
        if n_r > 0:
            strata = self.rng.integers(0, len(self._strata), n_r)
            picked = np.empty(n_r, dtype=np.int64)
            for k in np.unique(strata):
                where = np.flatnonzero(strata == k)
                members = self._strata[k]
                p = self._strata_p[k]
                picked[where] = (self.rng.choice(members, len(where), p=p) if p is not None
                                 else self.rng.choice(members, len(where)))
            idx.append(self.fm_pool[picked])
            lam.append(self.fm_lambda[picked])
        return np.concatenate(idx), np.concatenate(lam).astype(np.float32)

    # -- one step ----------------------------------------------------------------------------

    def _lr_factor(self, k):
        """Learning-rate factor of step `k`: linear warm-up, then constant or a cosine to 0 at ``total_steps``."""
        s = self.settings
        warmup = int(s['warmup_steps'])
        if warmup and k < warmup:
            return (k + 1) / warmup
        if s['lr_schedule'] == 'cosine':
            total = s['total_steps'] if s['total_steps'] is not None else s['steps_per_round']
            fraction = (k - warmup) / max(int(total) - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(fraction, 0.0), 1.0)))
        return 1.0

    def _fm_loss(self, x_t, t, lam, uncond, target):
        """Flow-matching loss without runaway samples: ``(loss or None, n_dropped)``."""
        live = self.live
        m = target.shape[-2]
        v = live.velocity(x_t, t, lam, uncond_mask=uncond)
        per_sample = (v - target).pow(2).sum(dim=(-1, -2)) / m
        limit = self.settings['fm_max_sample_loss']
        if limit is None:
            return per_sample.mean(), 0
        with torch.no_grad():
            keep = per_sample.detach() <= float(limit)          # NaN compares False: dropped
        n_bad = int((~keep).sum())
        if n_bad == 0:
            return per_sample.mean(), 0
        del v, per_sample                                       # free the poisoned graph
        if not bool(keep.any()):
            return None, n_bad
        v = live.velocity(x_t[keep], t[keep], lam[keep], uncond_mask=uncond[keep])
        return ((v - target[keep]).pow(2).sum(dim=(-1, -2)) / m).mean(), n_bad

    def step(self):
        """One optimizer step of the single loss (see the module docstring).

        Returns
        -------
        tuple
            ``(float(total), terms)``: the normalized total loss and a dict of the raw terms and flags.

        Notes
        -----
        ``L_q = committor_weight L_sp + grad_penalty L_gp + anchor_weight
        L_anchor``, ``total = N_q(L_q) + fm_weight N_fm(L_FM)``. A step whose
        gradient norm is not finite is skipped (no optimizer or EMA update)
        and counted. With ``split_backward`` the committor part is
        backpropagated before the flow-matching part is formed (see the class
        notes); everything else, including the random draws, is the same.
        """
        s, data, device, L = self.settings, self.dataset, self.device, self.lambda_clip
        if data is None:
            raise RuntimeError('FlowTrainer.step: call set_data first')
        split = bool(s.get('split_backward', False))
        factor = self._lr_factor(self.step_count)
        for group in self.optimizer.param_groups:
            group['lr'] = float(s['lr']) * factor
        live = self.live
        live.train()
        terms = {}
        lq = torch.zeros((), device=device)
        if len(self.sp_train) and s['n_sp'] > 0:
            i = self.rng.choice(self.sp_train, int(s['n_sp']))
            penalty = s['grad_penalty'] > 0
            lam, pos = self.committor_fn(live, data.sp_rows[i], penalty)
            first = torch.as_tensor(data.sp_r_first[i], dtype=lam.dtype, device=lam.device)
            last = torch.as_tensor(data.sp_r_last[i], dtype=lam.dtype, device=lam.device)
            n = first + last
            l_sp = committor_nll(lam, last / n, n)
            lq = lq + s['committor_weight'] * l_sp
            terms['L_sp'] = float(l_sp.detach())
            if penalty:
                (grad,) = torch.autograd.grad(lam.sum(), pos, create_graph=True)
                l_gp = grad.pow(2).mean()
                lq = lq + s['grad_penalty'] * l_gp
                terms['L_gradpen'] = float(l_gp.detach())
        if s['n_anchor'] > 0:
            n_first = int(s['n_anchor']) // 2
            n_last = int(s['n_anchor']) - n_first
            ia = self._draw_basin(1, n_first)
            ib = self._draw_basin(2, n_last)
            rows = np.concatenate([data.fr_rows[ia], data.fr_rows[ib]])
            lam_a, _ = self.committor_fn(live, rows, False)
            edge = 1.0 / (1.0 + math.exp(L))
            y = torch.as_tensor(np.r_[np.full(n_first, edge), np.full(n_last, 1.0 - edge)],
                                dtype=lam_a.dtype, device=lam_a.device)
            l_anchor = committor_nll(lam_a, y, torch.ones_like(y))
            lq = lq + s['anchor_weight'] * l_anchor
            terms['L_anchor'] = float(l_anchor.detach())
        total = self.normalizer('q', lq)
        backward_done = False
        if split:
            # the committor part's backward now, before the flow-matching forward pass
            self.optimizer.zero_grad(set_to_none=True)
            if torch.is_tensor(total) and total.requires_grad:
                total.backward()
                backward_done = True
            total = total.detach()
            del lq
        if s['fm_weight'] > 0 and s['n_fm'] > 0:
            j, lam_hat = self._draw_fm(int(s['n_fm']))
            n = len(j)
            with torch.no_grad():
                x1 = to_flow(live, torch.as_tensor(data.fr_x[j], dtype=torch.float32,
                                                   device=device))
                lam_t = torch.as_tensor(lam_hat, dtype=torch.float32, device=device)
                uncond = torch.rand(n, device=device, generator=self.generator) < s['p_uncond']
                x0 = live.sample_prior(lam_t, n, uncond_mask=uncond, generator=self.generator)
                x1 = pair(live, x1, x0)
                t = flowmath.sample_time(n, s['t_schedule'], device=device,
                                         generator=self.generator)
                x_t, target = flowmath.interpolate(x0, x1, t)
            l_fm, dropped = self._fm_loss(x_t, t, lam_t, uncond, target)
            if dropped:
                terms['fm_dropped'] = float(dropped)
            if l_fm is not None:
                terms['L_FM'] = float(l_fm.detach())
                if split:
                    # the flow-matching part's backward: its gradients add to the committor part's
                    fm_total = s['fm_weight'] * self.normalizer('fm', l_fm)
                    if fm_total.requires_grad:
                        fm_total.backward()
                        backward_done = True
                    total = total + fm_total.detach()
                else:
                    total = total + s['fm_weight'] * self.normalizer('fm', l_fm)
        finite = False
        if split:
            if backward_done:
                norm = torch.nn.utils.clip_grad_norm_(live.parameters(), float(s['grad_clip']))
                finite = bool(torch.isfinite(norm))
        else:
            self.optimizer.zero_grad(set_to_none=True)
            if torch.is_tensor(total) and total.requires_grad:
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(live.parameters(), float(s['grad_clip']))
                finite = bool(torch.isfinite(norm))
        if finite:
            self.optimizer.step()
            self._update_ema()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            terms['skipped'] = 1.0
        self.normalizer.advance()
        self.step_count += 1
        self.last_terms = terms
        return float(total.detach()), terms

    def _update_ema(self):
        decay = self.settings['ema_decay']
        with torch.no_grad():
            for ema, live in zip(self.network.parameters(), self.live.parameters()):
                if decay:
                    ema.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
                else:
                    ema.copy_(live.detach())
            for ema, live in zip(self.network.buffers(), self.live.buffers()):
                ema.copy_(live)

    # -- evaluation -------------------------------------------------------------------------------

    def committor_of_rows(self, net, rows):
        """Logits of descriptor rows under `net`, in eval mode, without autograd, in chunks of ``relabel_chunk``."""
        chunk = int(self.settings.get('relabel_chunk', 128))
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                out = [self.committor_fn(net, rows[i:i + chunk], False)[0].detach()
                       .to(torch.float64).cpu().numpy() for i in range(0, len(rows), chunk)]
        finally:
            net.train(was_training)
        return np.concatenate(out) if out else np.zeros(0)

    def holdout_nll(self, net):
        """Shooting NLL of the held-out chains' shooting points under `net`, None without any.

        Parameters
        ----------
        net : torch.nn.Module
            ``self.network`` (the EMA) or ``self.live``.

        Returns
        -------
        float or None
            :func:`committor_nll` of the held-out shooting points.
        """
        if self.dataset is None or not len(self.sp_hold):
            return None
        data = self.dataset
        lam = torch.as_tensor(self.committor_of_rows(net, data.sp_rows[self.sp_hold]))
        first = torch.as_tensor(data.sp_r_first[self.sp_hold], dtype=torch.float64)
        last = torch.as_tensor(data.sp_r_last[self.sp_hold], dtype=torch.float64)
        return float(committor_nll(lam, last / (first + last), first + last))


# --------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------

def _is_int(value, minimum=None):
    ok = isinstance(value, Integral) and not isinstance(value, bool)
    return ok and (minimum is None or value >= minimum)


def _is_number(value, minimum=None, strict=False):
    ok = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    if ok and minimum is not None:
        ok = value > minimum if strict else value >= minimum
    return ok


def _settings_from(kwargs):
    """FLOW_FIT_DEFAULTS updated with `kwargs`, validated."""
    unknown = sorted(set(kwargs) - set(FLOW_FIT_DEFAULTS))
    if unknown:
        raise TypeError(f'flow_matching_fit() got unexpected keyword argument(s) '
                        f'{", ".join(repr(k) for k in unknown)}; the keywords are those of '
                        f'aimmd.cgaimmd.flow_matching.fit.FLOW_FIT_DEFAULTS')
    s = {**FLOW_FIT_DEFAULTS, **kwargs}
    problems = []

    def need(ok, key, what):
        if not ok:
            problems.append(f'{key} must be {what}, got {s[key]!r}')

    for key in ('graphs', 'normalize_losses', 'hold_out_anchor_frames', 'fill_missing_values',
                'cache_dataset', 'split_backward'):
        need(isinstance(s[key], bool), key, 'a bool')
    for key in ('steps_per_round', 'n_lambda_bins', 'relabel_max_frames', 'relabel_chunk',
                'free_stride', 'shot_stride', 'min_basin_frames'):
        need(_is_int(s[key], 1), key, 'an int >= 1')
    for key in ('warmup_steps', 'n_sp', 'n_anchor', 'n_fm', 'normalizer_warmup',
                'shot_stride_min_frames', 'round_oom_retries', 'log_every', 'holdout_seed',
                'relabel_every', 'holdout_every'):
        need(_is_int(s[key], 0), key, 'an int >= 0')
    for key in ('lr', 'grad_clip'):
        need(_is_number(s[key], 0, strict=True), key, 'a number > 0')
    for key in ('weight_decay', 'committor_weight', 'anchor_weight', 'grad_penalty', 'fm_weight',
                'idle_sleep_s', 'round_oom_wait_s'):
        need(_is_number(s[key], 0), key, 'a number >= 0')
    for key in ('instate_fm_share', 'p_uncond', 'holdout_fraction'):
        need(_is_number(s[key], 0) and s[key] <= 1, key, 'a number in [0, 1]')
    need(_is_number(s['normalizer_decay'], 0, strict=True) and s['normalizer_decay'] < 1,
         'normalizer_decay', 'a number in (0, 1)')
    need(_is_number(s['normalizer_max_ratio'], 1), 'normalizer_max_ratio', 'a number >= 1')
    need(s['max_round_seconds'] is None or _is_number(s['max_round_seconds'], 0),
         'max_round_seconds', 'None or a number >= 0')
    need(s['ema_decay'] is None or (_is_number(s['ema_decay'], 0) and s['ema_decay'] < 1),
         'ema_decay', 'None or a number in [0, 1)')
    need(s['fm_max_sample_loss'] is None or _is_number(s['fm_max_sample_loss'], 0, strict=True),
         'fm_max_sample_loss', 'None or a number > 0')
    need(s['lr_schedule'] in ('constant', 'cosine'), 'lr_schedule', "'constant' or 'cosine'")
    need(s['basin_draw'] in BASIN_DRAWS, 'basin_draw', "'uniform' or 'density'")
    need(s['total_steps'] is None or _is_int(s['total_steps'], 1), 'total_steps',
         'None or an int >= 1')
    if s['lr_schedule'] == 'cosine' and _is_int(s['total_steps'], 1) \
            and _is_int(s['steps_per_round'], 1) and s['total_steps'] > s['steps_per_round']:
        # every round restarts the schedule: a cosine longer than the round never anneals
        problems.append(
            f'total_steps={s["total_steps"]} is larger than steps_per_round='
            f'{s["steps_per_round"]}: every round trains again from the start with a new '
            f'optimizer, so the cosine schedule restarts every round and would stop short of 0; '
            f'leave total_steps None (the cosine then ends at steps_per_round) or set it to at '
            f'most steps_per_round')
    need(s['trainer_seed'] is None or _is_int(s['trainer_seed'], 0), 'trainer_seed',
         'None or an int >= 0')
    need(s['early_stopping_patience'] is None or _is_int(s['early_stopping_patience'], 1),
         'early_stopping_patience', 'None or an int >= 1')
    if s['early_stopping_patience'] is not None and _is_int(s['holdout_every'], 0) \
            and s['holdout_every'] == 0:
        problems.append('early_stopping_patience needs holdout_every >= 1 (the held-out '
                        'likelihood is evaluated every holdout_every steps)')
    try:
        flowmath.parse_time_schedule(s['t_schedule'])
    except ValueError as exc:
        problems.append(f't_schedule: {exc}')
    chains = s['holdout_chains']
    need(chains is None or (not isinstance(chains, (str, bytes))
                            and all(_is_int(c, 0) for c in chains)),
         'holdout_chains', 'None or a list of chain numbers')
    need(s['exclude_frames'] is None or isinstance(s['exclude_frames'], (dict, str, os.PathLike)),
         'exclude_frames', 'None, a mapping or the path of a JSON file')
    need(s['state_dir'] is None or isinstance(s['state_dir'], (str, os.PathLike)), 'state_dir',
         'None or a path')
    if problems:
        raise ValueError('flow_matching_fit: ' + '; '.join(problems))
    return s


def _json_default(obj):
    """JSON form of what ``json`` cannot write (numpy scalars and arrays, sets sorted, paths, iterables, else ``str``)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    try:
        return list(obj)
    except TypeError:
        return str(obj)


def _canonical(value):
    """`value` as plain JSON data (lists, never sets or tuples)."""
    return json.loads(json.dumps(value, default=_json_default))


# --------------------------------------------------------------------------------------------
# helpers of flow_matching_fit
# --------------------------------------------------------------------------------------------

def _transform_input(out, device):
    """A graph transform's output as a batch dict on `device` (a dict, or a list of PyG ``Data``)."""
    if isinstance(out, Mapping):
        return {key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in out.items()}
    from torch_geometric.data import Batch
    return Batch.from_data_list(out).to(device).to_dict()


class _MDFrameInput:
    """The committor of descriptor rows through the input the trainer's value pass builds."""

    def __init__(self, params, graphs):
        self.graphs = bool(graphs)
        self.transform = getattr(params, 'descriptor_transform', None)

    def input(self, net, rows, requires_grad=False):
        """``(input, positions)``: the dense tensor, or the graph batch dict and its positions."""
        device = protocol.network_device(net)
        if not self.graphs:
            x = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=device)
            x = torch.flatten(x, start_dim=1)
            if requires_grad:
                x = x.detach().clone().requires_grad_(True)
            return x, x
        graphs = _transform_input(self.transform(rows), device)
        if requires_grad:
            graphs['positions'] = graphs['positions'].detach().clone().requires_grad_(True)
        return graphs, graphs['positions']

    def __call__(self, net, rows, requires_grad):
        inp, positions = self.input(net, rows, requires_grad)
        return net(inp)[:, 0], positions

    def check(self, rows, network=None):
        """TypeError when ``graphs`` does not match what ``params.descriptor_transform`` returns.

        In graph mode the transform must return a graph batch dict (the
        default route of :class:`~aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`)
        or a list of PyG ``Data`` (its graph-cache routes); its edges are also
        compared with the edges the network builds itself (a network with a
        numeric ``cutoff``): a transform with another cutoff raises
        ``ValueError``.
        """
        if self.graphs:
            if self.transform is None:
                raise TypeError('flow_matching_fit(graphs=True) needs params.descriptor_transform '
                                '(graph mode: a transform returning a graph batch dict or a list '
                                'of graph Data, e.g. aimmd.cgaimmd.flow_matching.graphs.'
                                'FlowGraphTransform)')
            out = self.transform(rows)
            if not isinstance(out, (list, Mapping)):
                raise TypeError(f'flow_matching_fit(graphs=True): params.descriptor_transform '
                                f'must return a graph batch dict or a list of graph Data, got '
                                f'{type(out).__name__}')
            cutoff = getattr(network, 'cutoff', None)
            if isinstance(cutoff, Real) and not isinstance(cutoff, bool) and len(out):
                _check_graph_edges(out, float(cutoff),
                                   int(getattr(network, 'max_num_neighbors', 128)))
        elif self.transform is not None:
            out = self.transform(rows)
            if isinstance(out, list):
                what = 'graph Data (pass graphs=True)'
            elif isinstance(out, Mapping):
                what = 'a graph batch dict (pass graphs=True)'
            else:
                what = type(out).__name__
            raise TypeError(f'flow_matching_fit(graphs=False) gives forward the flat descriptors, '
                            f'but params.descriptor_transform is set and returns {what}: the '
                            f'values function would score other inputs than the fit trains on')


def _graphs_of(out):
    """``(positions, edge_index)`` of every graph of a transform output (a batch dict or a list of Data)."""
    if not isinstance(out, Mapping):
        for data in out:
            yield (torch.as_tensor(data.positions, dtype=torch.float32),
                   torch.as_tensor(data.edge_index, dtype=torch.long))
        return
    positions = torch.as_tensor(out['positions'], dtype=torch.float32).cpu()
    edges = torch.as_tensor(out['edge_index'], dtype=torch.long).cpu()
    ptr = torch.as_tensor(out['ptr'], dtype=torch.long).cpu()
    for k in range(len(ptr) - 1):
        lo, hi = int(ptr[k]), int(ptr[k + 1])
        mine = (edges[1] >= lo) & (edges[1] < hi)
        yield positions[lo:hi], edges[:, mine] - lo


def _check_graph_edges(graphs, cutoff, cap, slack=1e-4):
    """ValueError when the transform's graphs lack edges within `cutoff` or have edges beyond it.

    Pairs within `slack` Å of the cutoff are ignored (the two edge builders
    round distances differently); graphs with a node at the neighbour cap are
    skipped (their neighbour lists may be truncated either way). Bond and
    topology edges beyond the cutoff (a network's own extra edges) are not
    graph edges of the transform and are not looked at.
    """
    from .graph_batch import radius_edges
    for k, (pos, edges) in enumerate(_graphs_of(graphs)):
        if edges.shape[1] and int(torch.bincount(edges[1]).max()) >= cap:
            continue
        own = radius_edges(pos, torch.zeros(len(pos), dtype=torch.long), cutoff,
                           max_num_neighbors=cap, backend='torch')
        theirs_set = set(map(tuple, edges.t().tolist()))
        own_set = set(map(tuple, own.t().tolist()))
        differ = [(i, j) for i, j in theirs_set ^ own_set
                  if abs(float((pos[i] - pos[j]).norm()) - cutoff) > slack]
        if differ:
            i, j = differ[0]
            raise ValueError(f'flow_matching_fit(graphs=True): graph {k} of '
                             f'params.descriptor_transform has {len(theirs_set)} edges, the network '
                             f'builds {len(own_set)} within its cutoff of {cutoff:g} A (pair '
                             f'{i}-{j} at {float((pos[i] - pos[j]).norm()):.3f} A differs); the '
                             f'transform\'s cutoff must be the network\'s')


def _fit_member_problems(network):
    """What `network` lacks for this fit: the flow-matching members, with the keyword ``uncond_mask``.

    The trainer drops labels with ``uncond_mask`` in both ``velocity`` and
    ``sample_prior`` (the ``'ode'`` generator needs it only for guidance).
    """
    problems = flow_matching_problems(network, uncond_mask=True)
    member = getattr(network, 'sample_prior', None)
    if callable(member) and not any(p.startswith('sample_prior') for p in problems):
        error = protocol._bind(member, ('lam', 'n'), {'uncond_mask': None, 'generator': None})
        if error is not None:
            problems.append(f'sample_prior must accept sample_prior(lam, n, *, uncond_mask=None, '
                            f'generator=None) ({error})')
    return problems


def _cgaimmd_settings(params):
    settings = getattr(params, 'cgaimmd_settings', None)
    if settings is not None:
        return settings
    from ...params.utils import CGAIMMD_PARAMETER_DEFAULTS
    return {**CGAIMMD_PARAMETER_DEFAULTS, **(getattr(params, 'cgaimmd_parameters', None) or {})}


def _series_of(params):
    """The descriptor series the fit reads: ``params.descriptors_series`` (else ``'descriptors'``)."""
    return getattr(params, 'descriptors_series', None) or DEFAULT_SERIES


def _initial_network_file(params):
    """The file of ``params.initial_network`` (relative to the params file), or None."""
    method = getattr(params, '_initial_network_fname', None)
    if callable(method):
        return method()
    value = getattr(params, 'initial_network', None)
    return None if value is None else os.fspath(value)


def _check_start(params, network):
    """The kind of start the rounds will have; TypeError when there is none."""
    if _initial_network_file(params) is not None:
        return 'initial_network'
    if callable(getattr(network, 'reset_parameters', None)):
        return 'reset_parameters'
    raise TypeError(f'flow_matching_fit trains every round from the same start: '
                    f'params.initial_network (e.g. the end-state pretraining of '
                    f'aimmd.cgaimmd.flow_matching.pretrain) or network.reset_parameters(); '
                    f'{type(network).__name__} has no reset_parameters, so set initial_network')


def _load_start(params, network, seed, device):
    """Load the round's start weights into `network` (then on `device`); returns what they were.

    ``params.initial_network`` (read on the CPU, tensors only, loaded
    strictly) when it is set, else ``network.reset_parameters()`` on the CPU
    with the CPU random stream seeded with `seed` inside a fork: the start is
    the same in every round and on every device, and the caller's random
    streams are unchanged.
    """
    fname = _initial_network_file(params)
    if fname is not None:
        state = torch.load(fname, map_location='cpu', weights_only=True)
        network.to(device)
        network.load_state_dict(state)
        return {'kind': 'initial_network', 'file': os.path.abspath(fname),
                'sha256': file_sha256(fname)}
    network.to('cpu')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        network.reset_parameters()
    network.to(device)
    return {'kind': 'reset_parameters', 'seed': int(seed)}


def _stop_poller(worker):
    if worker is None:
        return lambda: False
    return lambda: bool(getattr(worker, 'must_stop', False))


def _sleep(seconds, stopping):
    """Sleep `seconds` in steps of at most 1 s, polling `stopping`; True if a stop was requested."""
    remaining = float(seconds)
    while remaining > 0:
        if stopping():
            return True
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk
    return bool(stopping())


def _cpu_state(network):
    return {k: v.detach().to('cpu', copy=True) for k, v in network.state_dict().items()}


def _params_directory(params):
    """The directory relative paths of a params file refer to (``params.parent``, else the cwd)."""
    try:
        parent = params.parent
    except Exception:                                       # noqa: BLE001
        parent = None
    return os.path.abspath(os.fspath(parent)) if parent is not None else os.getcwd()


# --------------------------------------------------------------------------------------------
# what a fit keeps between rounds
# --------------------------------------------------------------------------------------------

class _RunMemory:
    """The bookkeeping of a run's fit (also in ``state.json``) and its dataset (memory only)."""

    def __init__(self):
        self.round = 0
        self.held_out_chains = ()
        self.noop_done = False
        self.n_trained_shots = 0
        self.previous = None            # the dataset of the previous round (cache_dataset)
        self.checked = None             # id of the network the start-up checks passed for

    def load(self, saved):
        self.round = int(saved.get('round', 0))
        self.held_out_chains = tuple(int(c) for c in saved.get('held_out_chains') or ())
        self.noop_done = bool(saved.get('noop_done', False))
        self.n_trained_shots = int(saved.get('n_trained_shots', 0))


def release_trainer(state_dir):
    """Drop what the fit keeps in this process for a fit-state folder (the cached dataset, the bookkeeping).

    Parameters
    ----------
    state_dir : str or os.PathLike
        The fit-state folder.

    Returns
    -------
    bool
        Whether the process kept anything for it.

    Notes
    -----
    The next fit with this folder reads its bookkeeping from
    ``state.json`` again and extracts its dataset from scratch. No weights
    are kept between rounds, so nothing else is lost.
    """
    return _RUNS.pop(os.path.abspath(os.fspath(state_dir)), None) is not None


def _read_state(state_dir):
    """The bookkeeping saved in `state_dir`: ``state.json``, else that of an old ``state.pt``, else None."""
    fname = os.path.join(state_dir, 'state.json')
    if os.path.exists(fname):
        try:
            with open(fname) as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f'WARNING: flow-matching fit: {fname} could not be read ({exc}); the round '
                  f'counter and the held-out chains start again', flush=True)
            return None
        if isinstance(state, dict) and state.get('schema') == STATE_SCHEMA:
            return state
        print(f'WARNING: flow-matching fit: {fname} is not a fit state ({STATE_SCHEMA}); '
              f'ignored', flush=True)
        return None
    for name in ('state.pt', 'state.pt.prev'):
        old = os.path.join(state_dir, name)
        if not os.path.exists(old):
            continue
        try:
            checkpoint = torch.load(old, map_location='cpu', weights_only=True)
        except Exception:                                   # noqa: BLE001
            continue
        if not isinstance(checkpoint, dict) or checkpoint.get('schema') != OLD_CHECKPOINT_SCHEMA:
            continue
        if once(('flow-matching fit: old side checkpoint', os.path.abspath(old))):
            print(f'flow-matching fit: {old} is the warm-start checkpoint of the flow branch; its '
                  f'round counter and held-out chains are continued, its weights are not used '
                  f'(every round starts from initial_network or reset weights)', flush=True)
        return {key: checkpoint.get(key) for key in ('round', 'held_out_chains', 'noop_done',
                                                     'n_trained_shots')}
    return None


def _write_state(context):
    state_dir, memory = context['state_dir'], context['memory']
    if state_dir is None:
        return
    os.makedirs(state_dir, exist_ok=True)
    state = {'schema': STATE_SCHEMA, 'round': int(memory.round),
             'held_out_chains': [int(c) for c in memory.held_out_chains],
             'noop_done': bool(memory.noop_done),
             'n_trained_shots': int(memory.n_trained_shots),
             'seed': int(context['seed']), 'seed_source': context['seed_source'],
             'keywords': _canonical(context['kwargs']), 'time': time.time()}
    atomic_write_json(os.path.join(state_dir, 'state.json'), state)


def _append_record(state_dir, record):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, 'rounds.jsonl'), 'a') as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + '\n')
        handle.flush()


# --------------------------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------------------------

def flow_matching_fit(params, pathensemble, verbose=False, worker=None, **kwargs):
    """One training round of a flow-matching network, from the run's fixed start.

    Parameters
    ----------
    params : aimmd.Params
        The run's params; ``params.network`` is the EMA container that AIMMD saves and scores.
    pathensemble : aimmd.PathEnsemble
        What the trainer passes (``pathensemble + margins``); a list (multi-system) is refused.
    verbose : bool, optional
        Print a one-line summary of every round.
    worker : aimmd.worker.WorkerTrain, optional
        The trainer (run directory, ``must_stop``, task); None: no files unless ``state_dir`` is given.
    **kwargs
        Keywords of :data:`FLOW_FIT_DEFAULTS` (unknown keys raise ``TypeError``).

    Returns
    -------
    tuple
        ``(losses, info)``: the per-step totals (non-empty: "trained, save the
        network") and a dict with ``status``, ``reason``, ``round``,
        ``record`` (the round record, also appended to
        ``flow_fit{states}/rounds.jsonl``) and ``network_metadata`` (what
        AIMMD's trainer stores next to the network: the round, every setting,
        the seed and the start; None for a round that did not count).

    Raises
    ------
    NotImplementedError
        For a list ensemble or ``params.multi_system``.
    TypeError
        For unknown keywords, ``params.rescale_committor``, a network that is
        not a C-GAIMMD network or lacks the flow-matching members, a network
        without a start (no ``params.initial_network`` and no
        ``reset_parameters``), or ``graphs`` not matching
        ``params.descriptor_transform``.
    ValueError
        For bad keyword values or a descriptor cache of the wrong width.

    Notes
    -----
    One call, in order:

    1. **Refusals and set-up.** The training device
       (``device_for('training', params)``; the network's device outside a
       worker task), the conditioning range ``L =
       protocol.lambda_clip_of(params.network, params.cgaimmd_settings)``,
       the fit-state folder (``state_dir``, default
       ``{worker._directory}/flow_fit{sorted_states}``) and its bookkeeping
       (``state.json``), the round number (one more than the larger of the
       saved round and the ``round`` of ``network{states}.json``), the seed
       (``trainer_seed``, else :func:`aimmd.cgaimmd.fitting.fit_seed`: the
       run's ``generation_seed`` or a hash of the run folder) and the round
       seed ``seed + 1000 (round - 1)``.
    2. **Data.** Missing committor values of reactive frames are computed
       with the run's network as it is (``fill_missing_values``; the only use
       of the network's current weights), AIMMD's reweighting (with the
       trainer's ``bins{states}.npy``) gives every path its density weight,
       and :func:`aimmd.cgaimmd.dataset.extract_dataset` (descriptor series
       ``params.descriptors_series``) the shooting points and frames. The
       held-out chains are ``holdout_chains``, else those saved, else a
       ``holdout_fraction`` of the chains drawn once with ``holdout_seed``.
    3. **No data.** With fewer than ``min_basin_frames`` in-state frames of
       either state: the first round of a fresh run is a no-op
       (``([0.0], info)``, the network unchanged) so that the trainer writes
       the first bins; later rounds return ``[]`` after ``idle_sleep_s``.
    4. **Start.** The start weights are loaded into ``params.network``
       (``initial_network``, or ``reset_parameters()`` under the run's seed:
       the same start every round), then a new :class:`FlowTrainer` (new
       optimizer, EMA and normalizer, streams seeded with the round seed).
    5. **Training.** The flow-matching labels from the EMA (at the start and
       every ``relabel_every`` steps), then ``steps_per_round`` steps; with
       ``holdout_every`` the EMA's held-out NLL is evaluated every that many
       steps and at the end, and with ``early_stopping_patience`` the round
       stops after that many evaluations without improvement and the network
       ends as the EMA with the lowest held-out NLL. ``max_round_seconds``
       counts from the start of the call, but at least one step runs.
    6. **Records.** ``state.json``, a line of ``rounds.jsonl``, the return value.

    ``status`` is ``'trained'``, ``'no-op'``, ``'skipped'`` (no data) or
    ``'stopped'`` (``must_stop`` during the round: the losses so far, the
    network as trained so far). A CUDA out-of-memory error restarts the round
    from step 4 (the same start and seed, so the result does not depend on
    it), at most ``round_oom_retries`` times, ``round_oom_wait_s`` apart; an
    error while the round record is written repeats only that part.

    Inside :func:`aimmd.cgaimmd.fitting.settings_only` (the preflight) a
    call stops after step 1's refusals and the keyword checks and raises
    :class:`aimmd.cgaimmd.fitting.FitSettings` with the settings, and with
    :func:`labels_from_start_problem` as a problem; nothing is read or
    trained.

    In the ``kinetics_convergence`` task the trainer fits data fractions:
    the fit then uses ``worker.kinetics_convergence_state_dir`` as its state
    folder when the trainer sets one, else it keeps nothing between calls;
    it never reads the run's round counter, has no first-round no-op and
    does not sleep in a round without data.
    """
    t_call = _clock()
    settings = _settings_from(kwargs)
    kcv = worker is not None and getattr(worker, 'task', None) == 'kinetics_convergence'
    if isinstance(pathensemble, (list, tuple)) or getattr(params, 'multi_system', False):
        raise NotImplementedError('flow_matching_fit supports single systems only')
    if getattr(params, 'rescale_committor', False):
        raise TypeError('rescale_committor is not supported by flow_matching_fit: it rescales '
                        'the network in place, behind the fit')
    network = params.network
    protocol.check_network_protocol(network)
    problems = _fit_member_problems(network)
    if problems:
        raise TypeError(f'flow_matching_fit: {type(network).__name__} is not a flow-matching '
                        f'network: ' + '; '.join(problems))
    start_kind = _check_start(params, network)
    if settings_only_active():
        # the preflight: the keywords and the start are checked, nothing is read or trained
        problem = labels_from_start_problem(settings)
        raise FitSettings(FIT_NAME, settings, keywords=kwargs,
                          problems=[problem] if problem else [],
                          notes=[f'every round starts from {start_kind}'])
    states = params.sorted_states
    lambda_clip = protocol.lambda_clip_of(network, _cgaimmd_settings(params))
    from ..devices import device_for
    device = device_for('training', params) or protocol.network_device(network)
    device = torch.device(device)
    run_dir = run_directory(worker)
    state_dir = settings['state_dir']
    if kcv:
        state_dir = getattr(worker, 'kinetics_convergence_state_dir', None)
    elif state_dir is None and run_dir is not None:
        state_dir = os.path.join(run_dir, f'flow_fit{states}')
    state_dir = None if state_dir is None else os.fspath(state_dir)
    memory = _memory(state_dir, network, kcv)
    round_number = memory.round
    if not kcv:
        round_number = max(round_number, previous_round(run_dir, states))
    round_number += 1
    seed, seed_source = fit_seed(params, run_dir, settings['trainer_seed'])
    context = dict(params=params, pathensemble=pathensemble, worker=worker, settings=settings,
                   kwargs=kwargs, memory=memory, network=network, states=states,
                   run_dir=run_dir, state_dir=state_dir, device=device,
                   stopping=_stop_poller(worker),
                   frames_input=_MDFrameInput(params, settings['graphs']), t_call=t_call, verbose=verbose, lambda_clip=lambda_clip, kcv=kcv,
                   round_number=round_number, seed=seed, seed_source=seed_source,
                   round_seed=seed + ROUND_SEED_STRIDE * (round_number - 1),
                   start_kind=start_kind, start=None, data=None, t_data=None, trainer=None,
                   losses=[], terms=[], oom_retries=0, round_counted=False, noop=False,
                   finished=None, held=None)
    while True:
        try:
            return _round(context)
        except torch.OutOfMemoryError as exc:
            if context['oom_retries'] >= settings['round_oom_retries']:
                raise
            context['oom_retries'] += 1
            if context['finished'] is None:              # the round restarts: free its trainer
                context['trainer'] = None
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            where = 'the record of the round' if context['finished'] is not None else 'the round'
            print(f'flow-matching fit: CUDA out of memory ({str(exc)[:160]}); {where} restarts in '
                  f'{settings["round_oom_wait_s"]:.0f} s (retry {context["oom_retries"]}/'
                  f'{settings["round_oom_retries"]})', flush=True)
            if _sleep(settings['round_oom_wait_s'], stopping=context['stopping']):
                if context['finished'] is not None:
                    return _finish(context, *context['finished'])
                context['losses'], context['terms'] = [], []
                return _finish(context, 'stopped', 'stop requested during the wait after a CUDA '
                               'out-of-memory error', context['data'], {})
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def _memory(state_dir, network, kcv):
    """The run memory of this call: from the registry (and ``state.json``), or new."""
    if kcv and state_dir is None:
        return _RunMemory()                  # a data fraction of its own: nothing is carried
    key = os.path.abspath(state_dir) if state_dir is not None else ('memory', id(network))
    memory = _RUNS.get(key)
    if memory is None:
        memory = _RUNS[key] = _RunMemory()
    if state_dir is not None:
        saved = _read_state(state_dir)
        if saved is not None:
            memory.load(saved)
    return memory


def _assemble(context):
    """Values, weights and the dataset of this round."""
    params, pe, s = context['params'], context['pathensemble'], context['settings']
    memory, states, run_dir = context['memory'], context['states'], context['run_dir']
    r = states[1]
    if s['fill_missing_values'] and len(pe):
        function, _, source = params.compute_values_args
        function = _cast_to_values_caches(function, pe)
        try:
            pe.compute(function=function, target='values', source=source,
                       conditions={'states': lambda state: state == r},
                       batch_size=params.network_batch_size)
        except RuntimeError as exc:
            if 'Dtype mismatch' not in str(exc):
                raise
            raise RuntimeError(f'flow-matching fit: the missing values could not be written: '
                               f'{exc}. The values caches of this run have more than one dtype, '
                               f'or params.values_function returns another one than they hold '
                               f'(AIMMD\'s default values function returns float32, the C-GAIMMD '
                               f'values function float64); make the caches agree, or pass '
                               f'fill_missing_values=False') from exc
    bins = None
    if run_dir is not None and os.path.exists(os.path.join(run_dir, f'bins{states}.npy')):
        from ...cache.npy import load_npy
        bins = load_npy(os.path.join(run_dir, f'bins{states}.npy'))
    from ...worker.utils import reweight_parameters_with_bins
    rw = reweight_parameters_with_bins(dict(getattr(params, 'reweight_parameters', None) or {}),
                                       bins)
    try:
        weights = path_weights(pe, states, rw)
    except Exception as exc:                                # noqa: BLE001
        print(f'flow-matching fit: reweighting failed ({exc}); frames weighted uniformly this '
              f'round', flush=True)
        weights = np.array([0.0 if is_initial_path(p) else 1.0 for p in pe])
    base = run_dir if run_dir is not None else os.getcwd()
    exclusions = resolve_exclusions(s['exclude_frames'], base, _params_directory(params))
    n_atoms = protocol.n_atoms_of(context['network'])
    data = extract_dataset(
        pe, states=states, n_atoms=n_atoms, series=_series_of(params), weights=weights,
        free_stride=s['free_stride'], shot_stride=s['shot_stride'],
        shot_stride_min_frames=s['shot_stride_min_frames'], exclude_frames=exclusions,
        previous=memory.previous if s['cache_dataset'] else None)
    memory.previous = data if s['cache_dataset'] else None
    return data


def _values_cache_dtype(pathensemble):
    """The dtype of the first existing values cache of the ensemble's files, None if there is none."""
    from ...path.utils import get_cache_fname
    seen = set()
    for path in pathensemble:
        for fname in path.fnames:
            fname = str(fname)
            if fname in seen:
                continue
            seen.add(fname)
            cache = get_cache_fname(fname, 'values')
            if not os.path.exists(cache):
                continue
            try:
                return np.dtype(np.load(cache, mmap_mode='r').dtype)
            except (OSError, ValueError):
                continue
    return None


def _cast_to_values_caches(function, pathensemble):
    """`function` with its output cast to the dtype of the run's values caches.

    ``update_npy`` refuses to write another dtype into a cache; a run whose
    caches hold several dtypes still fails, with the message of
    :func:`_assemble`.
    """
    dtype = _values_cache_dtype(pathensemble)
    if dtype is None:
        return function

    def values_function(*args, **kwargs):
        return np.asarray(function(*args, **kwargs)).astype(dtype, copy=False)
    for name in ('__name__', '__qualname__', '__doc__', '__module__'):
        try:
            setattr(values_function, name, getattr(function, name))
        except (AttributeError, TypeError):
            pass
    import inspect
    try:
        values_function.__signature__ = inspect.signature(function)
    except (TypeError, ValueError):
        pass
    return values_function


def _held_out(memory, data, settings):
    if settings['holdout_chains'] is not None:
        return tuple(sorted(int(c) for c in settings['holdout_chains']))
    if memory.held_out_chains:
        return memory.held_out_chains
    chains = np.unique(data.sp_group[data.sp_group >= 0])
    if len(chains) < 2:
        return ()
    n = max(1, int(round(settings['holdout_fraction'] * len(chains))))
    drawn = np.random.default_rng(settings['holdout_seed']).choice(chains, n, replace=False)
    return tuple(sorted(int(c) for c in drawn))


def _relabel(context, data):
    """The flow-matching pool and its labels: the committor of the round's EMA, recomputed."""
    trainer, s = context['trainer'], context['settings']
    pool = trainer.fm_candidates
    if len(pool) > s['relabel_max_frames']:
        pool = np.sort(trainer.rng.choice(pool, s['relabel_max_frames'], replace=False))
    lam = trainer.committor_of_rows(trainer.network, data.fr_rows[pool])
    trainer.set_fm_targets(pool, lam)


def labels_from_start_problem(settings):
    """Why `settings` would condition the generator on the start network only, or None.

    Every round starts again from ``initial_network`` (or reset weights), and
    the flow-matching labels of the reactive frames are the committor of the
    round's EMA, taken at the start of the round and then every
    ``relabel_every`` steps. With ``relabel_every`` 0, or not below
    ``steps_per_round``, they stay the start network's for the whole round
    and the generator never learns from the shooting points. None when the
    labels are refreshed within the round, or when the flow-matching term
    uses no reactive frames (``fm_weight`` 0, ``n_fm`` 0 or
    ``instate_fm_share`` 1).

    Parameters
    ----------
    settings : dict
        :data:`FLOW_FIT_DEFAULTS` updated with the fit's keywords.

    Returns
    -------
    str or None
    """
    s = settings
    if s['fm_weight'] == 0 or s['n_fm'] == 0 or s['instate_fm_share'] >= 1:
        return None
    if s['relabel_every'] and s['relabel_every'] < s['steps_per_round']:
        return None
    return (f'relabel_every={s["relabel_every"]} with steps_per_round={s["steps_per_round"]}: '
            f'every round starts again from the same network, so the flow-matching labels of '
            f'the reactive frames would stay that start network\'s committor for the whole '
            f'round and the generator would never learn from the shooting points; set '
            f'relabel_every between 1 and steps_per_round - 1 (default 1000 of 4000)')


def _warn_labels_from_start(context):
    """Once per process: the reactive frames' labels are the start network's for the whole round."""
    s, trainer = context['settings'], context['trainer']
    if not len(trainer.fm_pool) or s['fm_weight'] == 0 or s['n_fm'] == 0 \
            or s['instate_fm_share'] >= 1:
        return
    if s['relabel_every'] and s['relabel_every'] < s['steps_per_round']:
        return
    if once('flow-matching fit: labels from the start network'):
        print(f'Warning: flow-matching fit: relabel_every={s["relabel_every"]} with '
              f'steps_per_round={s["steps_per_round"]}: the flow-matching labels of the reactive '
              f'frames are the committor of the start network ({context["start_kind"]}) for the '
              f'whole round, because every round starts again from it; set relabel_every below '
              f'steps_per_round (default 1000, the PaiNN recipes)',
              flush=True)


def _metadata(context, entry):
    """The network metadata of a counted round: what AIMMD's trainer stores next to the network."""
    memory, trainer, s = context['memory'], context['trainer'], context['settings']
    network = context['network']
    try:
        n_parameters = int(sum(p.numel() for p in network.parameters()))
    except Exception:                                       # noqa: BLE001
        n_parameters = None
    metadata = {
        'round': int(memory.round), 'n_trained_shots': int(memory.n_trained_shots),
        'steps': int(trainer.step_count) if trainer is not None else 0,
        'fit': FIT_NAME, 'status': entry['status'] if entry else None,
        'ema_decay': s['ema_decay'], 'seed': int(context['seed']),
        'seed_source': context['seed_source'], 'round_seed': int(context['round_seed']),
        'start': context['start'], 'series': _series_of(context['params']),
        'held_out_chains': [int(c) for c in memory.held_out_chains],
        'settings': _canonical({key: s[key] for key in FLOW_FIT_DEFAULTS}),
        'keywords': _canonical(context['kwargs']),
        'network': {'class': f'{type(network).__module__}.{type(network).__qualname__}',
                    'n_parameters': n_parameters,
                    'n_atoms': protocol.n_atoms_of(network),
                    'lambda_clip': float(context['lambda_clip'])}}
    if entry is not None and entry.get('early_stopping') is not None:
        metadata['early_stopping'] = entry['early_stopping']
    return metadata


def _record(context, status, reason, data, seconds):
    """The round record (``rounds.jsonl``)."""
    memory, trainer, terms = context['memory'], context['trainer'], context['terms']
    names = sorted({k for t in terms for k in t if k.startswith('L_')})
    record = {
        'round': int(memory.round), 'status': status, 'reason': reason,
        'steps': int(trainer.step_count) if trainer is not None else 0,
        'skipped_steps': int(sum(t.get('skipped', 0.0) for t in terms)),
        'fm_dropped': int(sum(t.get('fm_dropped', 0.0) for t in terms)),
        'loss_terms': {k: float(np.mean([t[k] for t in terms if k in t])) for k in names},
        'loss_total': float(np.mean(context['losses'])) if context['losses'] else None,
        'holdout_nll_ema': (trainer.holdout_nll(trainer.network)
                            if trainer is not None and data is not None else None),
        'holdout_nll_live': (trainer.holdout_nll(trainer.live)
                             if trainer is not None and data is not None else None),
        'early_stopping': context.get('early_stopping'),
        'n_trained_shots': int(memory.n_trained_shots),
        'held_out_chains': list(memory.held_out_chains),
        'relabels_in_round': int(context.get('relabels_in_round', 0)),
        'start': context['start'], 'seed': int(context['seed']),
        'round_seed': int(context['round_seed']),
        'data': data.summary() if data is not None else None,
        'seconds': seconds, 'device': str(context['device']),
        'oom_retries': int(context['oom_retries']), 'time_end': time.time()}
    return record


def _finish(context, status, reason, data=None, seconds=None, record=True):
    """Round counter, state file, round record and the return value."""
    memory, losses, trainer = context['memory'], context['losses'], context['trainer']
    counted = status == 'no-op' or (trainer is not None and trainer.step_count > 0)
    if counted and not context['round_counted']:
        memory.round = int(context['round_number'])          # once per call, even after an OOM
        context['round_counted'] = True
    total = round(_clock() - context['t_call'], 3)
    seconds = {**(seconds or {}), 'total': total}
    entry = _record(context, status, reason, data, seconds) if (counted or record) else None
    if status != 'skipped':
        _write_state(context)
        if counted and context['state_dir'] is not None:
            _append_record(context['state_dir'], entry)
    if context['verbose'] and entry is not None:
        terms = ', '.join(f'{k} {v:.4g}' for k, v in entry['loss_terms'].items())
        print(f'flow-matching fit round {memory.round}: {status}, {entry["steps"]} steps '
              f'({terms}), held-out NLL {entry["holdout_nll_ema"]}, {total} s', flush=True)
    info = {'status': status, 'reason': reason, 'round': int(memory.round), 'record': entry,
            'network_metadata': _metadata(context, entry) if counted else None}
    return (list(losses) if status != 'no-op' else [0.0]), info


def _round(context):
    """Data, no-data rule, start, steps (one attempt; repeated after a CUDA out-of-memory error)."""
    if context['finished'] is not None:                       # the record failed: only that again
        return _finish(context, *context['finished'])
    s, memory, network = context['settings'], context['memory'], context['network']
    t0 = _clock()
    data = context['data']
    if data is None:
        data = _assemble(context)
        context['data'] = data
        context['t_data'] = round(_clock() - t0, 3)
        context['held'] = _held_out(memory, data, s)
    seconds = {'data': context['t_data']}
    held = context['held']
    memory.held_out_chains = held
    memory.n_trained_shots = int((~np.isin(data.sp_group, held)).sum())
    n_first, n_last = _basin_counts(data, held, s['hold_out_anchor_frames'])
    if min(n_first, n_last) < s['min_basin_frames']:
        return _no_data(context, data, n_first, n_last, seconds)
    t_start = _clock()
    context['start'] = _load_start(context['params'], network, context['seed'], context['device'])
    if memory.checked != id(network):
        rows = data.fr_rows[:4]
        context['frames_input'].check(rows, network)
        forward_input, _ = context['frames_input'].input(network, rows)
        protocol.check_network_numerics(network, data.fr_x[:4].astype(np.float32),
                                        forward_input)
        memory.checked = id(network)
    trainer = FlowTrainer(network, s, lambda_clip=context['lambda_clip'], device=context['device'],
                          committor_fn=context['frames_input'], seed=context['round_seed'])
    context.update(trainer=trainer, losses=[], terms=[], relabels_in_round=0,
                   relabel_in_round_s=0.0, early_stopping=None)
    trainer.set_data(data, held, hold_out_anchor_frames=s['hold_out_anchor_frames'])
    t_relabel0 = _clock()
    _relabel(context, data)
    _warn_labels_from_start(context)
    t_relabel = _clock()
    status, reason = _train(context, data)
    t_train = _clock()
    seconds.update({'start': round(t_relabel0 - t_start, 3),
                    'relabel': round(t_relabel - t_relabel0, 3),
                    'train': round(t_train - t_relabel, 3)})
    if context['relabels_in_round']:
        seconds['relabel_in_round'] = round(context['relabel_in_round_s'], 3)   # within 'train'
    context['finished'] = (status, reason, data, seconds)
    return _finish(context, status, reason, data, seconds)


def _basin_counts(data, held, hold_out_anchor_frames):
    """In-state frames of positive weight of the first and the last state that may be anchors."""
    ok = data.fr_w > 0
    if hold_out_anchor_frames:
        ok &= ~(np.isin(data.fr_group, list(held)) & (data.fr_group >= 0))
    return int(((data.fr_state == 1) & ok).sum()), int(((data.fr_state == 2) & ok).sum())


def _no_data(context, data, n_first, n_last, seconds):
    """The no-data rule: the first round of a fresh run is a no-op, later rounds are skipped."""
    s, memory = context['settings'], context['memory']
    reason = (f'fewer than min_basin_frames={s["min_basin_frames"]} basin frames '
              f'(first state {n_first}, last state {n_last})')
    run_dir, states = context['run_dir'], context['states']
    bins_missing = run_dir is not None and \
        not os.path.exists(os.path.join(run_dir, f'bins{states}.npy'))
    if context['kcv']:                      # no bins to write, nothing to wait for
        return _finish(context, 'skipped', reason, data, seconds, record=False)
    if context['noop'] or (getattr(context['params'], 'nbins', 0) > 1 and bins_missing
                           and not memory.noop_done):
        if not context['noop']:                         # (not again after an OOM in _finish)
            memory.noop_done = True
            context['noop'] = True
            print(f'flow-matching fit: {reason} and no bins file yet: round '
                  f'{context["round_number"]} is a no-op (the network is unchanged) so that the '
                  f'trainer writes the first bins', flush=True)
        context['finished'] = ('no-op', reason + '; first round of a fresh run', data, seconds)
        return _finish(context, *context['finished'])
    if _sleep(s['idle_sleep_s'], context['stopping']):
        return _finish(context, 'skipped', reason + '; stop requested', data, seconds,
                       record=False)
    return _finish(context, 'skipped', reason, data, seconds, record=False)


def _train(context, data):
    """The steps of the round; returns ``(status, reason)``."""
    s, trainer, network = context['settings'], context['trainer'], context['network']
    stopping = context['stopping']
    status, reason = 'trained', 'steps_per_round reached'
    every = s['holdout_every']
    patience = s['early_stopping_patience']
    evaluate = bool(every) and len(trainer.sp_hold) > 0
    curve, best, bad = [], None, 0

    def evaluate_now(done):
        nonlocal best, bad
        nll = trainer.holdout_nll(network)
        curve.append([int(done), nll])
        if best is None or nll < best[0]:
            best, bad = (nll, int(done), _cpu_state(network) if patience else None), 0
        else:
            bad += 1
        return patience is not None and bad >= patience

    while trainer.step_count < s['steps_per_round']:
        if stopping():
            status, reason = 'stopped', 'stop requested'
            break
        if s['max_round_seconds'] is not None and trainer.step_count > 0 \
                and _clock() - context['t_call'] > s['max_round_seconds']:
            reason = f'max_round_seconds={s["max_round_seconds"]} reached'   # after >= 1 step
            break
        value, terms = trainer.step()
        context['losses'].append(value)
        context['terms'].append(terms)
        done = trainer.step_count
        if s['relabel_every'] and done % s['relabel_every'] == 0 and done < s['steps_per_round']:
            t_mid = _clock()
            _relabel(context, data)                 # the EMA of this step
            context['relabels_in_round'] += 1
            context['relabel_in_round_s'] += _clock() - t_mid
        if s['log_every'] and done % s['log_every'] == 0 and context['state_dir'] is not None:
            progress = _record(context, 'progress', '', data, {'data': context['t_data']})
            progress['progress'] = True
            progress['round'] = int(context['round_number'])
            _append_record(context['state_dir'], progress)
        if evaluate and done % every == 0 and evaluate_now(done):
            reason = (f'early stopping: no lower held-out NLL in {patience} evaluations '
                      f'(every {every} steps)')
            break
    if evaluate and (not curve or curve[-1][0] != trainer.step_count) and trainer.step_count:
        evaluate_now(trainer.step_count)
    if every:
        info = {'every': int(every), 'patience': patience, 'curve': curve,
                'best_step': best[1] if best else None,
                'best_holdout_nll': best[0] if best else None, 'restored': False}
        if not evaluate:
            info['note'] = 'no held-out shooting points: nothing evaluated'
        elif patience is not None and best is not None and best[1] != trainer.step_count:
            network.load_state_dict(best[2])
            info['restored'] = True
        context['early_stopping'] = info
    return status, reason
