"""
aimmd.cgaimmd.analysis.validate
===============================

Committor accuracy of any AIMMD network against a brute-force reference, and
the validation protocol of a trained C-GAIMMD model.

Committor parity
----------------
A *committor reference* (:class:`CommittorReference`) is a set of
configurations, each with commitment counts: how many trajectories started
from it ended first in the first end state (``n_first``) and in the last end
state (``n_last``; the committor is the probability of reaching this one).
Loaders build one from

- an npz file that holds the coordinates and the counts
  (:meth:`CommittorReference.from_npz`; by default coordinates ``xyz_nm`` in
  nm and counts ``nA``, ``nB``);
- a trajectory file with one frame per configuration and an npz of its
  counts (:meth:`CommittorReference.from_trajectory` with
  :func:`shot_counts`; counts as ``n_A``, ``n_B``, or as a fraction and a
  number of shots);
- an AIMMD sweep folder (:func:`sweep_reference`), the held-out chains of a run
  (:func:`heldout_chain_reference`) or the free simulations of a run
  (:meth:`CommittorReference.from_free_md`).

Any network can be scored on it (:func:`score_committor`): the committor head
of a C-GAIMMD model on the coordinates it reads (:func:`network_committor`),
or any AIMMD network through its params, from the full configurations, their
descriptors and the values function (:func:`params_committor`; an ordinary
descriptor network such as the distance-matrix baseline of
:mod:`aimmd.cgaimmd.analysis.committor_baseline`).

The parity metrics (:func:`committor_metrics`), with ``p = sigmoid(lambda)``
the model's committor, ``q = n_last / n`` the observed fraction and ``n``
the number of trajectory ends (endpoints) of a configuration:

- **log-likelihood per endpoint**: the mean of ``log p`` over endpoints in
  the last state and ``log(1 - p)`` over endpoints in the first. It rewards a
  model for being confident and right and punishes it hard for being
  confident and wrong. Two lines to read it against: a *constant model*
  (``p`` = the pooled fraction of last-state endpoints, a model that knows
  nothing about structure) and the *ceiling* (``p = q`` at every
  configuration; it fits the counts' own noise, so a correct model scores a
  little below it, about half a nat per configuration with mixed outcomes);
- **calibration line** ``q = sigmoid(a + b lambda)``, fitted by maximum
  likelihood on the counts: ``a = 0, b = 1`` is a calibrated model; ``b < 1``
  means its logits are too steep (overconfident), ``b > 1`` too flat;
- **Spearman correlation** of ``p`` and ``q``: whether the model orders the
  configurations correctly, whatever its scale (1: same order);
- **R²** of ``p`` against ``q``, ``1 - sum (q - p)^2 / sum (q - mean q)^2``,
  and its **noise ceiling**: ``q`` is itself a noisy estimate (binomial
  variance ``q (1 - q) / (n - 1)`` per configuration), so even the exact
  committor has an R² below 1 on these counts; the ceiling is ``1 - sum
  noise / sum (q - mean q)^2``. It needs two or more endpoints per
  configuration, and endpoints that are independent given the configuration
  (separate shots); the two halves of one two-way shot are not, so a
  reference of held-out chains gets no ceiling;
- in the transition region ``0.1 < q < 0.9``: the root mean square and the
  mean of ``p - q`` (endpoint weighted), and how many predictions there are
  saturated (``p < 0.05`` or ``p > 0.95``).

:func:`parity_figure` draws the parity of several models side by side, with
the binomial uncertainty of every configuration's ``q``.

The validation protocol of a C-GAIMMD model
-------------------------------------------
A C-GAIMMD model has a committor model and a generator that can fail
independently: the generator can produce invalid or copied structures, or
structures at another committor than requested, and the committor model can
be wrong. :func:`validate` tests each part against something it was not
trained on, writes a JSON report and a one-page model card:

- **M1, validity** per requested level: the valid share of the raw samples
  (bond graph, chirality, clashes: a ``cgaimmd_validity_function`` such as
  :class:`aimmd.cgaimmd.preparation.validity.TopologyValidity`), intact
  waters of a cluster (:func:`waters_intact`), optionally again after a
  restrained energy-minimization polish
  (:func:`aimmd.cgaimmd.preparation.minimize.polish`);
- **M2, committor accuracy** of the model on every reference (above);
- **M3, generator against committor**: the committor of the valid samples
  against the request (:func:`generator_consistency` and
  :func:`aimmd.cgaimmd.analysis.adherence.adherence_metrics`), judged by the
  model's own committor and by an independent committor (a second opinion
  that shares no weights with the generator);
- **guard against copies**: the symmetry-minimized RMSD of every valid sample
  to its nearest training frame, against the same distance for held-out MD
  frames (:func:`nearest_frames`, :func:`guard_summary`).

Samples are drawn through the generation interface
(:func:`aimmd.cgaimmd.generation.make_generator`), so any generator works.
Nothing here shoots trajectories: shooting generated structures is the job of
:mod:`aimmd.cgaimmd.analysis.shooting_validation`. ``torch``, ``scipy`` and
``matplotlib`` are imported inside the functions that need them.

>>> from aimmd.cgaimmd.analysis import validate
>>> reference = validate.CommittorReference.from_npz('committor_benchmark.npz')   # doctest: +SKIP
>>> report, logits = validate.score_committor(validate.params_committor(params),  # doctest: +SKIP
...                                           reference)
>>> card = validate.validate(network, 'card/model', references=[reference],     # doctest: +SKIP
...                          validity=validity, training_frames=training,
...                          heldout_frames=heldout)
"""

import datetime
import hashlib
import inspect
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...analysis._data_access import without_writing
from ._stats import (json_float, jsonable, level_seed, logit as _logit, sigmoid as _sigmoid,
                     spearman as _spearman)

__all__ = ['SCHEMA', 'DEFAULT_LEVELS', 'REFERENCE_KINDS', 'CommittorReference', 'Samples',
           'committor_metrics', 'calibration_fit', 'shot_counts', 'score_committor',
           'network_committor', 'params_committor', 'heldout_chain_reference',
           'sweep_reference', 'draw_samples', 'check_validity', 'validity_summary',
           'waters_intact', 'generator_consistency', 'symmetric_rmsd', 'nearest_frames',
           'guard_summary', 'validate', 'load_report', 'redraw_card', 'leaderboard',
           'parity_figure']

SCHEMA = 'aimmd.cgaimmd.validate/1'
"""Schema name written into every report; bumped when a key changes meaning."""

DEFAULT_LEVELS = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)
"""Requested committor logits of M1 and M3, both end states and the transition region."""

TRANSITION_Q = (0.1, 0.9)
"""Brute-force committor range of the transition-region statistics of M2 (open interval)."""

TRANSITION_LEVEL = 2.0
"""Requested levels with ``abs(lambda) <=`` this value form the transition region of M1 and M3."""

SATURATION = 0.05
"""A prediction counts as saturated when ``p < SATURATION`` or ``p > 1 - SATURATION``."""

PROB_CLIP = 1e-6
"""Predicted probabilities are clipped to ``[PROB_CLIP, 1 - PROB_CLIP]`` in log-likelihoods."""

REASONS = ('non_finite', 'bond_count', 'not_isomorphic', 'vf2_timeout', 'chirality', 'clash')
"""Reason codes of :class:`aimmd.cgaimmd.preparation.validity.TopologyValidity`, in the order they are checked."""

REFERENCE_KINDS = ('brute_force', 'heldout_chains', 'free_md_forward')
"""Kinds of :class:`CommittorReference`: many shots per configuration, one two-way
shot per shooting point of a run, or one forward outcome per free-simulation frame."""

_OLD_SCHEMA_PREFIX = 'aimmd.flow.validate/'


# ════════════════════════════════════════════════════════════════════════════
# M2: committor accuracy
# ════════════════════════════════════════════════════════════════════════════

def _counts(n_first, n_last):
    a = np.asarray(n_first, dtype=float).reshape(-1)
    b = np.asarray(n_last, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f'n_first and n_last differ in shape: {a.shape} and {b.shape}')
    if np.any(a < 0) or np.any(b < 0) or not np.all(np.isfinite(a + b)):
        raise ValueError('commitment counts must be finite and >= 0')
    return a, b


def calibration_fit(lam, n_first, n_last, *, max_iter=200, ridge=0.0):
    """Maximum-likelihood calibration line ``q = sigmoid(a + b lam)`` on commitment counts.

    Parameters
    ----------
    lam : array_like
        ``(n,)`` model logits toward ``sorted_states[2]``.
    n_first, n_last : array_like
        ``(n,)`` endpoints in ``sorted_states[0]`` and ``sorted_states[2]``
        (fractional counts are allowed).
    max_iter : int, optional
        Newton iterations at most.
    ridge : float, optional
        Penalty ``ridge (a^2 + b^2) / 2`` added to the negative
        log-likelihood (0: the plain maximum likelihood). A small ridge keeps
        the fit finite when the counts are perfectly separated by `lam`.

    Returns
    -------
    tuple
        ``(a, b)`` floats; ``(None, None)`` when the line is not identifiable:
        `lam` is constant, or every endpoint lies in the same end state.

    Raises
    ------
    ValueError
        If a logit of an entry with endpoints is not finite.

    Notes
    -----
    Newton's method on the binomial log-likelihood of the counts, from
    ``(0, 1)``, with step halving; ``(0, 1)`` is a calibrated model, ``b < 1``
    a model whose logits are too steep. Without a ridge, for perfectly
    separated counts the slope grows until the iterations run out (the
    maximum is at infinity).
    """
    x = np.asarray(lam, dtype=float).reshape(-1)
    a_n, b_n = _counts(n_first, n_last)
    if x.shape != a_n.shape:
        raise ValueError(f'{len(x)} logits for {len(a_n)} entries')
    n = a_n + b_n
    keep = n > 0
    x, a_n, b_n, n = x[keep], a_n[keep], b_n[keep], n[keep]
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{int(np.sum(~np.isfinite(x)))} logits are not finite')
    if len(x) == 0 or np.ptp(x) < 1e-12 or not (a_n.sum() > 0 and b_n.sum() > 0):
        return None, None
    ridge = float(ridge)

    def nll(theta):
        s = theta[0] + theta[1] * x
        return float(np.sum(b_n * np.logaddexp(0.0, -s) + a_n * np.logaddexp(0.0, s))
                     + 0.5 * ridge * np.dot(theta, theta))

    theta = np.array([0.0, 1.0])
    value = nll(theta)
    design = np.stack([np.ones_like(x), x], 1)
    for _ in range(int(max_iter)):
        p = _sigmoid(theta[0] + theta[1] * x)
        grad = design.T @ (n * p - b_n) + ridge * theta
        hess = (design * (n * p * (1 - p))[:, None]).T @ design + ridge * np.eye(2)
        try:
            step = np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        while scale > 1e-8:
            trial = theta - scale * step
            trial_value = nll(trial)
            if trial_value <= value + 1e-15:
                break
            scale *= 0.5
        else:
            break
        converged = np.max(np.abs(trial - theta)) < 1e-12
        theta, value = trial, trial_value
        if converged:
            break
    return float(theta[0]), float(theta[1])


def committor_metrics(lam, n_first, n_last, *, prob_clip=PROB_CLIP, transition=TRANSITION_Q,
                      saturation=SATURATION, independent_endpoints=True):
    """Accuracy of committor logits against commitment counts.

    Parameters
    ----------
    lam : array_like
        ``(n,)`` model logits toward ``sorted_states[2]``.
    n_first, n_last : array_like
        ``(n,)`` endpoints in ``sorted_states[0]`` and ``sorted_states[2]`` (one two-way shot has two).
    prob_clip : float, optional
        Probabilities are clipped to ``[prob_clip, 1 - prob_clip]`` in the log-likelihoods.
    transition : tuple of float, optional
        Open range of the brute-force committor ``q`` that is the transition region.
    saturation : float, optional
        A prediction ``p < saturation`` or ``p > 1 - saturation`` is saturated.
    independent_endpoints : bool, optional
        Whether the endpoints of one entry are independent given its
        configuration (separate shots, default). False for the two halves of
        one two-way shot (held-out chains): the noise ceiling, which assumes
        independent endpoints, is then not computed.

    Returns
    -------
    dict
        The statistics listed in the Notes, plus ``settings``.

    Raises
    ------
    ValueError
        For counts of another shape, negative counts, no endpoints at all, or
        a logit that is not finite (of an entry with endpoints).

    Notes
    -----
    With ``p = sigmoid(lam)``, ``q = n_last / n`` and ``n = n_first + n_last``
    (entries without endpoints are dropped); the module docstring explains
    what each number says:

    - ``loglik_per_endpoint``: ``sum(n_last log p + n_first log(1 - p)) / sum(n)``;
    - ``loglik_saturated``: the same with ``p = q``, the *in-sample* ceiling
      (``q`` fits the counts' own noise, so the true committor scores less;
      ``loglik_true_committor_estimate`` subtracts the chi-squared optimism,
      half a nat per entry with ``0 < q < 1``); with two endpoints per entry
      (held-out chains) it is not attainable;
    - ``loglik_constant``: ``p`` = the pooled fraction of last-state endpoints;
    - ``calibration_intercept``, ``calibration_slope``: :func:`calibration_fit`;
    - ``spearman``: rank correlation of ``p`` and ``q`` over the entries;
    - ``rmse_transition``, ``bias_transition`` (mean ``p - q``): endpoint
      weighted over ``transition[0] < q < transition[1]``,
      ``n_saturated_transition`` saturated predictions there out of
      ``n_transition``; None without transition entries;
    - ``r2``: ``1 - sum((q - p)^2) / sum((q - mean q)^2)``, unweighted (the
      form of AIMMD-Lab's validation plots); ``r2_weighted`` and
      ``rmse_weighted`` weight each entry by its endpoints;
    - ``mse_binomial_noise``: the mean of ``q (1 - q) / (n - 1)`` over the
      entries with ``n >= 2`` (an unbiased estimate of the variance of
      ``q`` around the true committor when the ``n`` endpoints are
      independent), and ``r2_noise_ceiling``: ``1 - sum of that noise /
      sum((q - mean q)^2)`` over the same entries, the R² the exact committor
      would reach on these counts on average; ``r2_noise_ceiling_entries``:
      the model's R² over those same entries (with ``n_noise_ceiling_entries``
      of them), the number to hold against the ceiling (it equals ``r2`` when
      every entry has two or more endpoints); all None without such entries
      or with `independent_endpoints` False;
    - ``logit_rmse_mid``: RMS of ``logit(p) - logit(q)`` over ``0.05 < q < 0.95``;
    - ``n_entries``, ``n_endpoints``, ``endpoints_per_entry`` (min, median, max).
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    a_n, b_n = _counts(n_first, n_last)
    if lam.shape != a_n.shape:
        raise ValueError(f'{len(lam)} logits for {len(a_n)} entries')
    n = a_n + b_n
    if n.sum() <= 0:
        raise ValueError('the reference has no endpoints')
    keep = n > 0
    lam, a_n, b_n, n = lam[keep], a_n[keep], b_n[keep], n[keep]
    if not np.all(np.isfinite(lam)):
        raise ValueError(f'{int(np.sum(~np.isfinite(lam)))} logits are not finite; leave those '
                         f'entries out (or fix the model) before scoring')
    clip = float(prob_clip)
    p_raw = _sigmoid(lam)
    p = np.clip(p_raw, clip, 1 - clip)
    q = b_n / n
    total = n.sum()
    loglik = float(np.sum(b_n * np.log(p) + a_n * np.log1p(-p)) / total)
    qc = np.clip(q, clip, 1 - clip)
    saturated = float(np.sum(b_n * np.log(qc) + a_n * np.log1p(-qc)) / total)
    qbar = b_n.sum() / total
    constant = 0.0
    if 0 < qbar < 1:
        constant = float((b_n.sum() * math.log(qbar) + a_n.sum() * math.log(1 - qbar)) / total)
    mixed = int(np.sum((b_n > 0) & (a_n > 0)))
    ss = np.sum((q - q.mean()) ** 2)
    r2 = None if ss == 0 else float(1 - np.sum((q - p_raw) ** 2) / ss)
    qw = np.sum(n * q) / total
    ssw = np.sum(n * (q - qw) ** 2)
    r2w = None if ssw == 0 else float(1 - np.sum(n * (q - p_raw) ** 2) / ssw)
    several = (n >= 2) & bool(independent_endpoints)
    noise = q[several] * (1 - q[several]) / (n[several] - 1)
    ss_several = np.sum((q[several] - q[several].mean()) ** 2) if several.any() else 0.0
    r2_several = (None if ss_several == 0 else
                  float(1 - np.sum((q[several] - p_raw[several]) ** 2) / ss_several))
    tr = (q > transition[0]) & (q < transition[1])
    if tr.any():
        rmse_tr = float(np.sqrt(np.sum(n[tr] * (p_raw[tr] - q[tr]) ** 2) / n[tr].sum()))
        bias_tr = float(np.sum(n[tr] * (p_raw[tr] - q[tr])) / n[tr].sum())
    else:
        rmse_tr = bias_tr = None
    n_sat = int(np.sum(tr & ((p_raw < saturation) | (p_raw > 1 - saturation))))
    mid = (q > 0.05) & (q < 0.95)
    logit_rmse = (float(np.sqrt(np.mean((_logit(p[mid]) - _logit(q[mid])) ** 2)))
                  if mid.any() else None)
    a, b = calibration_fit(lam, a_n, b_n)
    return {
        'loglik_per_endpoint': loglik, 'loglik_saturated': saturated,
        'loglik_true_committor_estimate': saturated - 0.5 * mixed / total,
        'loglik_constant': constant, 'pooled_fraction_last': float(qbar),
        'calibration_intercept': a, 'calibration_slope': b,
        'spearman': _spearman(p_raw, q),
        'rmse_transition': rmse_tr, 'bias_transition': bias_tr,
        'n_saturated_transition': n_sat, 'n_transition': int(tr.sum()),
        'r2': r2, 'r2_weighted': r2w,
        'rmse_weighted': float(np.sqrt(np.sum(n * (p_raw - q) ** 2) / total)),
        'mse_binomial_noise': float(noise.mean()) if several.any() else None,
        'r2_noise_ceiling': (None if not several.any() or ss_several == 0
                             else float(1 - noise.sum() / ss_several)),
        'r2_noise_ceiling_entries': r2_several,
        'n_noise_ceiling_entries': int(several.sum()),
        'logit_rmse_mid': logit_rmse, 'n_mid': int(mid.sum()),
        'n_entries': int(len(q)), 'n_endpoints': int(round(total)),
        'endpoints_per_entry': [float(n.min()), float(np.median(n)), float(n.max())],
        'settings': {'prob_clip': clip, 'transition': [float(transition[0]), float(transition[1])],
                     'saturation': float(saturation),
                     'independent_endpoints': bool(independent_endpoints)}}


def _sha256(fname):
    digest = hashlib.sha256()
    with open(fname, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def shot_counts(fname, *, n_first='n_A', n_last='n_B', fraction_last=None, n_total=None):
    """Commitment counts from an npz file.

    Parameters
    ----------
    fname : str or os.PathLike
        The npz file.
    n_first, n_last : str, optional
        The arrays of endpoints in the first and in the last end state.
    fraction_last, n_total : str, optional
        Instead: the arrays of the fraction of endpoints in the last state and
        of the number of endpoints (the counts are rounded to integers).

    Returns
    -------
    tuple of numpy.ndarray
        ``(n_first, n_last)`` as floats.

    Examples
    --------
    >>> n_first, n_last = shot_counts('outcomes.npz')                          # doctest: +SKIP
    >>> n_first, n_last = shot_counts('ground_truth.npz', fraction_last='sampled',  # doctest: +SKIP
    ...                               n_total='nshots')
    """
    with np.load(os.fspath(fname), allow_pickle=False) as z:
        if fraction_last is not None or n_total is not None:
            if fraction_last is None or n_total is None:
                raise ValueError('fraction_last and n_total go together')
            total = np.asarray(z[n_total], dtype=float).reshape(-1)
            last = np.round(np.asarray(z[fraction_last], dtype=float).reshape(-1) * total)
            return total - last, last
        return (np.asarray(z[n_first], dtype=float).reshape(-1),
                np.asarray(z[n_last], dtype=float).reshape(-1))


@dataclass
class CommittorReference:
    """Configurations with commitment counts: the reference of M2."""

    name: str
    """Short descriptive name, the key of the report's ``m2`` section."""

    positions: np.ndarray
    """``(n, M, 3)`` coordinates in Å: of the atoms the committor reads (the network's atoms
    for :func:`network_committor`, the run's whole system for :func:`params_committor`)."""

    n_first: np.ndarray
    """``(n,)`` endpoints in ``sorted_states[0]``."""

    n_last: np.ndarray
    """``(n,)`` endpoints in ``sorted_states[2]``."""

    kind: str = 'brute_force'
    """One of :data:`REFERENCE_KINDS`."""

    include: Optional[np.ndarray] = None
    """``(n,)`` bool, the entries that are scored (None scores all); excluded entries are still evaluated and saved."""

    groups: Optional[np.ndarray] = None
    """``(n,)`` int labels (the chain of a held-out shooting point, the excursion of a free-MD frame), saved with the logits."""

    notes: dict = field(default_factory=dict)
    """Provenance (files, selection, exclusions), copied into the report."""

    dimensions: Optional[np.ndarray] = None
    """``(n, 6)`` box of every configuration (Å, degrees), when known; handed to committors that take it."""

    def __post_init__(self):
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(f'kind must be one of {REFERENCE_KINDS}, got {self.kind!r}')
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f'positions must have shape (n, M, 3), got {self.positions.shape}')
        self.n_first, self.n_last = _counts(self.n_first, self.n_last)
        n = len(self.positions)
        if self.n_first.shape != (n,):
            raise ValueError(f'counts have shape {self.n_first.shape} for {n} configurations')
        if self.include is not None:
            self.include = np.asarray(self.include, dtype=bool).reshape(-1)
            if self.include.shape != (n,):
                raise ValueError(f'include has shape {self.include.shape} for {n} configurations')
        if self.groups is not None:
            self.groups = np.asarray(self.groups).reshape(-1)
            if self.groups.shape != (n,):
                raise ValueError(f'groups has shape {self.groups.shape} for {n} configurations')
        if self.dimensions is not None:
            box = np.asarray(self.dimensions, dtype=np.float32)
            if box.shape == (6,):
                box = np.repeat(box[None], n, axis=0)
            if box.shape != (n, 6):
                raise ValueError(f'dimensions has shape {box.shape} for {n} configurations')
            self.dimensions = box

    def __len__(self):
        return int(len(self.positions))

    @property
    def committor(self):
        """``(n,)`` observed committor ``n_last / (n_first + n_last)`` (NaN without endpoints)."""
        n = self.n_first + self.n_last
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, self.n_last / np.where(n > 0, n, 1), np.nan)

    @classmethod
    def from_arrays(cls, name, positions, n_first, n_last, *, units='angstrom', atoms=None,
                    **kwargs):
        """A reference from arrays, with the coordinates in Å or nm.

        Parameters
        ----------
        name : str
            The reference's name.
        positions : array_like
            ``(n, N, 3)`` coordinates.
        n_first, n_last : array_like
            ``(n,)`` commitment counts.
        units : {'angstrom', 'nm'}, optional
            Units of `positions`.
        atoms : array_like of int, optional
            Keep only these atoms (in this order).
        **kwargs
            The other fields (``kind``, ``include``, ``groups``, ``notes``, ``dimensions``).
        """
        if units not in ('angstrom', 'nm'):
            raise ValueError(f"units must be 'angstrom' or 'nm', got {units!r}")
        x = np.asarray(positions, dtype=float) * (10.0 if units == 'nm' else 1.0)
        if atoms is not None:
            x = x[:, np.asarray(atoms, dtype=int)]
        return cls(name, x, n_first, n_last, **kwargs)

    @classmethod
    def from_npz(cls, fname, *, name=None, positions='xyz_nm', n_first='nA', n_last='nB',
                 units=None, atoms=None, kind='brute_force', include=None):
        """A reference from an npz file that holds the coordinates and the counts.

        Parameters
        ----------
        fname : str or os.PathLike
            The npz file.
        name : str, optional
            The reference's name (default: the file's stem).
        positions, n_first, n_last : str, optional
            The arrays of the coordinates ``(n, N, 3)`` and of the endpoints in
            the first and in the last end state (default ``xyz_nm``, ``nA``,
            ``nB``).
        units : {'angstrom', 'nm'}, optional
            Units of the coordinates (default: ``'nm'`` when the array name
            ends with ``_nm``, else ``'angstrom'``).
        atoms : array_like of int, optional
            Keep only these atoms.
        kind : str, optional
            One of :data:`REFERENCE_KINDS`.
        include : array_like of bool, optional
            The entries that are scored.

        Notes
        -----
        The ``notes`` record the file, its sha256, the array names, the units
        and the file's own description when it has a ``meta`` string. Shots
        that ended in no state are not endpoints and are left out; their
        number is noted when the file has an array ``n_none``.
        """
        fname = os.path.abspath(os.fspath(fname))
        if units is None:
            units = 'nm' if positions.endswith('_nm') else 'angstrom'
        with np.load(fname, allow_pickle=False) as z:
            x = np.asarray(z[positions], dtype=float)
            a = np.asarray(z[n_first], dtype=float)
            b = np.asarray(z[n_last], dtype=float)
            meta = str(z['meta']) if 'meta' in z.files else None
            unfinished = int(np.sum(z['n_none'])) if 'n_none' in z.files else None
        notes = {'file': fname, 'sha256': _sha256(fname),
                 'arrays': {'positions': positions, 'n_first': n_first, 'n_last': n_last},
                 'units': units, 'atoms': None if atoms is None else
                 [int(i) for i in np.asarray(atoms).reshape(-1)]}
        if meta is not None:
            try:
                notes['meta'] = json.loads(meta)
            except ValueError:
                notes['meta'] = meta
        if unfinished is not None:
            notes['shots_without_end_state'] = unfinished
        return cls.from_arrays(name or os.path.splitext(os.path.basename(fname))[0], x, a, b,
                               units=units, atoms=atoms, kind=kind, include=include, notes=notes)

    @classmethod
    def from_trajectory(cls, trajectory, n_first, n_last, *, name=None, atoms=None, frames=None,
                        kind='brute_force', include=None, notes=None):
        """A reference from a trajectory of the configurations and their counts.

        Parameters
        ----------
        trajectory : str or sequence of str
            The trajectory file(s) of the configurations, one frame each
            (several files are concatenated in order).
        n_first, n_last : array_like
            ``(n,)`` commitment counts, one per frame (see :func:`shot_counts`).
        name : str, optional
            The reference's name (default: the first file's stem).
        atoms : array_like of int, optional
            Keep only these atoms (e.g. the network's atoms in the network's
            order for :func:`network_committor`; all atoms for
            :func:`params_committor`).
        frames : array_like of int, optional
            Use only these frames of the trajectory.
        kind : str, optional
            One of :data:`REFERENCE_KINDS`.
        include : array_like of bool, optional
            The entries that are scored.
        notes : dict, optional
            More provenance.

        Notes
        -----
        The frames are read without writing an index file next to the
        trajectory. Their boxes are kept (:attr:`dimensions`).
        """
        from ._trajectory import read_frames
        files = ([os.fspath(trajectory)] if isinstance(trajectory, (str, os.PathLike))
                 else [os.fspath(f) for f in trajectory])
        parts = [read_frames(f) for f in files]
        x = np.concatenate([p[0] for p in parts]).astype(float)
        boxes = [p[1] for p in parts]
        box = None if any(b is None for b in boxes) else np.concatenate(boxes)
        if frames is not None:
            pick = np.asarray(frames, dtype=int).reshape(-1)
            x = x[pick]
            box = None if box is None else box[pick]
        if atoms is not None:
            x = x[:, np.asarray(atoms, dtype=int)]
        record = {'trajectory': [os.path.abspath(f) for f in files],
                  'sha256': [_sha256(f) for f in files],
                  'frames': None if frames is None else [int(i) for i in pick],
                  'atoms': None if atoms is None else
                  [int(i) for i in np.asarray(atoms).reshape(-1)]}
        record.update(notes or {})
        return cls(name or os.path.splitext(os.path.basename(files[0]))[0], x, n_first, n_last,
                   kind=kind, include=include, notes=record, dimensions=box)

    @classmethod
    def from_free_md(cls, reference, *, positions=None, name=None, n_atoms=None):
        """A reference from the free simulations of a run (one forward outcome per reactive frame).

        Parameters
        ----------
        reference : aimmd.analysis.free_md.FreeMDReference
            :func:`aimmd.analysis.free_md.free_md_reference` of the run.
        positions : array_like, optional
            ``(n, M, 3)`` Å coordinates of its frames; default: its descriptor
            rows reshaped to ``(n, -1, 3)`` (the run's descriptors must then be
            flat coordinates, as in C-GAIMMD runs; ask for them with
            ``free_md_reference(..., descriptors=True)``).
        name : str, optional
            Default: the free-MD reference's name.
        n_atoms : int, optional
            Check that the coordinates have this many atoms.

        Notes
        -----
        ``kind='free_md_forward'``; ``groups`` are the excursions (frames of one
        excursion share their outcome, so they are not independent). The
        outcome of a frame estimates its committor only when the velocities
        there are distributed as in equilibrium: true for equilibrium free
        dynamics, approximate shortly after a restart with fresh velocities
        (``notes['free_restart_source']`` says how the run restarted).
        """
        if positions is None:
            if reference.descriptors is None:
                raise ValueError('the free-MD reference has no descriptor rows; pass positions, '
                                 'or build it with free_md_reference(..., descriptors=True)')
            rows = np.asarray(reference.descriptors, dtype=float)
            if rows.shape[1] % 3:
                raise ValueError(f'descriptor rows of width {rows.shape[1]} are not flat '
                                 f'coordinates; pass positions')
            positions = rows.reshape(len(rows), -1, 3)
        positions = np.asarray(positions, dtype=float)
        if n_atoms is not None and positions.shape[1] != int(n_atoms):
            raise ValueError(f'the coordinates have {positions.shape[1]} atoms, not {n_atoms}')
        notes = dict(reference.notes)
        notes.update(files=list(reference.files), source='aimmd.analysis.free_md.free_md_reference')
        return cls(name or reference.name, positions, reference.n_first, reference.n_last,
                   kind='free_md_forward', groups=np.asarray(reference.groups), notes=notes)


def _accepts_dimensions(committor):
    try:
        parameters = inspect.signature(committor).parameters
    except (TypeError, ValueError):
        return False
    return 'dimensions' in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def score_committor(committor, reference, *, prob_clip=PROB_CLIP, transition=TRANSITION_Q,
                    saturation=SATURATION):
    """M2 of one committor on one reference.

    Parameters
    ----------
    committor : callable
        ``committor(positions (n, M, 3) Å) -> (n,)`` logits toward
        ``sorted_states[2]``; a committor that takes a ``dimensions`` keyword
        also gets the reference's boxes (:func:`params_committor` does).
    reference : CommittorReference
        The configurations and their counts.
    prob_clip, transition, saturation : optional
        As for :func:`committor_metrics`.

    Returns
    -------
    tuple
        ``(report, lam)``: the :func:`committor_metrics` of the included
        entries with ``kind``, ``n_configurations``, ``n_scored``,
        ``n_excluded``, ``seconds`` and ``notes``; the ``(n,)`` logits of
        every entry. A reference of held-out chains gets no noise ceiling
        (its two endpoints per entry are the halves of one two-way shot).

    Raises
    ------
    ValueError
        If the committor returns another number of logits or non-finite ones.
    """
    start = time.time()
    if reference.dimensions is not None and _accepts_dimensions(committor):
        out = committor(reference.positions, dimensions=reference.dimensions)
    else:
        out = committor(reference.positions)
    lam = np.asarray(out, dtype=float).reshape(-1)
    if lam.shape != (len(reference.positions),):
        raise ValueError(f'the committor returned {lam.shape} logits for '
                         f'{len(reference.positions)} configurations')
    if not np.all(np.isfinite(lam)):
        raise ValueError(f'the committor returned non-finite logits on {reference.name!r}')
    include = (np.ones(len(lam), dtype=bool) if reference.include is None
               else reference.include)
    metrics = committor_metrics(lam[include], reference.n_first[include],
                                reference.n_last[include], prob_clip=prob_clip,
                                transition=transition, saturation=saturation,
                                independent_endpoints=reference.kind != 'heldout_chains')
    report = {'kind': reference.kind, 'n_configurations': int(len(lam)),
              'n_scored': int(include.sum()), 'n_excluded': int((~include).sum()),
              **metrics, 'seconds': round(time.time() - start, 3),
              'notes': jsonable(reference.notes)}
    return report, lam


def network_committor(network, *, batch_size=None, descriptor_transform=None):
    """The committor of a C-GAIMMD network, as a numpy callable on the network's atoms.

    Parameters
    ----------
    network : object
        A C-GAIMMD network (PaiNN, BioEmu2 or any network of the protocol).
    batch_size : int, optional
        Frames per call (None: one call).
    descriptor_transform : callable, optional
        As for :func:`aimmd.cgaimmd.protocol.coordinate_committor` (None: dense mode).

    Returns
    -------
    callable
        ``f(positions (n, M, 3) Å, dimensions=None) -> (n,)`` float64 logits,
        computed in eval mode without gradients (``dimensions`` is not used).
    """
    def committor(positions, dimensions=None):
        import torch

        from ..protocol import coordinate_committor
        x = np.asarray(positions, dtype=np.float32)
        if len(x) == 0:
            return np.zeros(0)
        was_training = getattr(network, 'training', False)
        network.eval()
        try:
            with torch.no_grad():
                lam = coordinate_committor(network, x, batch_size=batch_size,
                                           descriptor_transform=descriptor_transform)
        finally:
            if was_training:
                network.train(True)
        return lam.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return committor


def params_committor(params, *, network=None, descriptor_transform='params', batch_size=256,
                     system_id=None):
    """The committor of any AIMMD network through its params, as a numpy callable on whole configurations.

    Parameters
    ----------
    params : aimmd.Params
        The params of the run the network belongs to: its
        ``descriptors_function``, ``descriptor_transform`` and
        ``values_function`` are applied as the trainer applies them.
    network : object or path-like, optional
        None: ``params.network`` as it is. A network file: loaded into
        ``params.network`` (``params.update_network``; `params` is changed). A
        network object: scored with its own values function (the C-GAIMMD
        values function for C-GAIMMD networks, AIMMD's default otherwise),
        with `descriptor_transform`.
    descriptor_transform : callable or None, optional
        For a network object: the transform between descriptors and network
        (default ``'params'``: ``params.descriptor_transform``; None: the
        descriptor rows themselves, e.g. for a
        :class:`~aimmd.cgaimmd.analysis.committor_baseline.DistanceMLP` on
        coordinate descriptors).
    batch_size : int, optional
        Configurations per descriptors call.
    system_id : optional
        For multi-system params: the system (handed to the functions that take it).

    Returns
    -------
    callable
        ``f(positions (n, N, 3) Å, dimensions=None) -> (n,)`` float64 logits:
        the configurations as a trajectory in memory (the reader AIMMD hands to
        descriptors functions, with the boxes `dimensions`), their descriptors
        (``params.descriptors_function``; the coordinates without one) and
        the values of the descriptors. `positions` hold the atoms the
        descriptors function expects: the run's whole system.

    Notes
    -----
    This scores any committor model AIMMD can train, an ordinary descriptor
    network as well as a C-GAIMMD model, from the same reference
    configurations, which is what a side-by-side parity plot needs.
    """
    from ...core.utils import accepts_system_id
    from ._trajectory import frames_reader
    if network is not None and isinstance(network, (str, os.PathLike)):
        params.update_network(os.fspath(network), timeout=0)
        network = None
    if network is None:
        values_function = params.values_function
    else:
        from ..protocol import is_cgaimmd_network
        transform = (params.descriptor_transform if isinstance(descriptor_transform, str)
                     and descriptor_transform == 'params' else descriptor_transform)
        if is_cgaimmd_network(network):
            from ..values import protocol_values_function
            values_function = protocol_values_function(network, transform, params=params)
        else:
            from ...params.utils import create_default_values_function
            values_function = create_default_values_function(network, transform)
    descriptors_function = params.descriptors_function

    def call(function, data):
        if system_id is not None and accepts_system_id(function):
            return function(data, system_id=system_id)
        return function(data)

    def committor(positions, dimensions=None):
        x = np.asarray(positions, dtype=np.float32)
        out = []
        for start in range(0, len(x), int(batch_size)):
            block = x[start:start + int(batch_size)]
            box = None
            if dimensions is not None:
                box = np.asarray(dimensions, dtype=np.float32)
                box = box[start:start + int(batch_size)] if box.ndim == 2 else box
            if descriptors_function:
                rows = np.asarray(call(descriptors_function, frames_reader(block, box)))
            else:
                rows = block
            out.append(np.asarray(call(values_function, rows), dtype=np.float64).reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)
    return committor


@without_writing
def heldout_chain_reference(params, run_dir, chains, *, n_atoms=None, exclude_frames=None,
                            name=None, allow_biased_frames=False):
    """The two-way shots of held-out reactive chains of a run folder, as a reference for M2.

    Parameters
    ----------
    params : aimmd.Params
        Params of the run (``shot_chains``, ``sorted_states`` and
        ``descriptors_series`` are used).
    run_dir : str
        The run folder.
    chains : sequence of int
        Chain numbers ``k`` of the held-out reactive chains ``chain{r}{k}``.
    n_atoms : int, optional
        Number ``M`` of the network's atoms (the descriptor caches must be
        their flat coordinates); default: the atoms of ``params.network``.
    exclude_frames : mapping or str, optional
        ``{file: frame indices | 'all'}`` or a JSON file of such a mapping (key
        ``'exclude'``), as the fit's option.
    name : str, optional
        Reference name (default ``'held-out chains <list>'``).
    allow_biased_frames : bool, optional
        Use the shots of a run that records a bias (default False: such a run
        is refused, as frames of biased simulations do not become committor
        references unless that is decided explicitly; a static bias that is
        zero in the reactive region leaves the shots' outcomes those of the
        unbiased system, which can justify it).

    Returns
    -------
    CommittorReference
        ``kind='heldout_chains'``; ``groups`` holds the chain of every shooting point.

    Raises
    ------
    ValueError
        For a run that records a bias (``params.record_bias``), unless `allow_biased_frames`.

    Notes
    -----
    The shooting points and outcomes are exactly those the C-GAIMMD fits
    train on (:func:`aimmd.cgaimmd.dataset.extract_dataset`: complete two-way
    shots of reactive chains with the shooting frame in the reactive state),
    read from the run's descriptor series; with all path weights 0 no other
    frame is read. A frame listed in `exclude_frames` is never a shooting
    point. Only the folders of `chains` are read, and the entries the call
    adds to AIMMD's array and trajectory caches are evicted before it returns
    (the array cache's budget is half the available memory, so a held-out
    chain of a large run would otherwise stay resident).
    """
    from ... import _config
    from ...analysis._data_access import refuse_biased_frames
    from ...pathensemble.utils import assemble_pathensemble
    from ..dataset import extract_dataset, resolve_exclusions
    from ..protocol import n_atoms_of
    refuse_biased_frames(params, 'a committor reference', allow_biased_frames)

    if n_atoms is None:
        n_atoms = n_atoms_of(params.network)
    run_dir = os.path.abspath(os.fspath(run_dir))
    held = sorted({int(c) for c in chains})
    series = getattr(params, 'descriptors_series', None) or 'descriptors'
    caches = [c for c in (getattr(_config, 'NPY_CACHE', None), getattr(_config, 'MDA_CACHE', None))
              if c is not None and hasattr(c, '_cache')]
    resident = [set(c._cache) for c in caches]
    try:
        shot_chains = params.shot_chains(run_dir, None, held)       # only the held-out chains
        pathensemble = assemble_pathensemble(shot_chains, [])
        exclusions = resolve_exclusions(exclude_frames, run_dir, os.getcwd())
        data = extract_dataset(pathensemble, states=params.sorted_states, n_atoms=int(n_atoms),
                               series=series, weights=np.zeros(len(pathensemble)),
                               exclude_frames=exclusions)
        del shot_chains, pathensemble
    finally:
        for cache, keys in zip(caches, resident):
            for key in [k for k in list(cache._cache) if k not in keys]:
                cache.remove(str(key))
    keep = np.isin(data.sp_group, held)
    first, last = data.sp_r_first[keep], data.sp_r_last[keep]
    notes = {'run_dir': run_dir, 'chains': held, 'states': params.sorted_states,
             'descriptors_series': series,
             'exclude_frames': (os.path.abspath(os.fspath(exclude_frames))
                                if isinstance(exclude_frames, (str, os.PathLike))
                                else ('mapping' if exclude_frames else None)),
             'excluded_shooting_points': int(data.notes.get('excluded_shooting_points', 0)),
             'n_shooting_points_read': int(len(data.sp_group)),
             'outcomes': {'(2,0)': int(((first == 2) & (last == 0)).sum()),
                          '(1,1)': int(((first == 1) & (last == 1)).sum()),
                          '(0,2)': int(((first == 0) & (last == 2)).sum())},
             'source': 'aimmd.cgaimmd.dataset.extract_dataset (shot chains only)'}
    return CommittorReference(
        name or f'held-out chains {", ".join(str(c) for c in held)}',
        positions=data.sp_x[keep].astype(float), n_first=first, n_last=last,
        kind='heldout_chains', groups=data.sp_group[keep], notes=notes)


def sweep_reference(params, sweep_dir, *, frames=None, atoms=None, name=None,
                    first_launched=False, tolerance=0.02, allow_biased_frames=False):
    """The brute-force committor of the frames of an AIMMD sweep folder.

    Parameters
    ----------
    params : aimmd.Params
        The sweep's params (states, trajectory extension, the rule that finds
        the shooting frame, the states function for paths without a states
        cache).
    sweep_dir : str or os.PathLike
        The sweep's run folder (``sweep{R}0``, ``sweep{R}1``, ...).
    frames : str or sequence of str, optional
        The sweep frames as trajectory file(s), in the order the sweep read
        them; default: the sweep's initial paths,
        ``initial{sorted_states}/*{ext}`` in name order (as the sweep workers
        read them).
    atoms : array_like of int, optional
        Keep only these atoms in the reference's coordinates (the shots are
        still checked on all atoms).
    name : str, optional
        Default: ``'sweep <folder name>'``.
    first_launched : bool, optional
        Count only the first launched shot of every frame (default: every
        registered shot, as :meth:`aimmd.PathEnsemble.shooting_results` does).
    tolerance : float, optional
        As for :func:`aimmd.cgaimmd.analysis.shooting_validation.sweep_outcomes`.
    allow_biased_frames : bool, optional
        Use a sweep whose params record a bias (default False: refused, as
        for :func:`heldout_chain_reference`).

    Returns
    -------
    CommittorReference
        Per frame: its coordinates and boxes, and the shot halves that ended
        in the first and in the last end state (uncommitted halves are not
        endpoints). ``notes`` record the sweep, the frames file(s), the number
        of shots, of uncommitted and unsimulated halves and the folders with a
        shot still in flight.

    Notes
    -----
    Every shot is attributed to its frame by the tag AIMMD writes next to it,
    and checked against the frame's coordinates
    (:func:`~aimmd.cgaimmd.analysis.shooting_validation.sweep_outcomes`).
    Nothing is written into the sweep folder.
    """
    import glob

    from ...analysis._data_access import refuse_biased_frames
    from ._trajectory import count_frames, read_frames
    from .shooting_validation import _first_launched, sweep_outcomes
    refuse_biased_frames(params, 'a committor reference', allow_biased_frames)
    sweep_dir = os.path.abspath(os.fspath(sweep_dir))
    if frames is None:
        frames = sorted(glob.glob(os.path.join(glob.escape(sweep_dir),
                                               f'initial{params.sorted_states}',
                                               f'*{params.trajectory_extension}')))
        if not frames:
            raise FileNotFoundError(f'no initial paths in {sweep_dir}/initial'
                                    f'{params.sorted_states}; pass the sweep frames')
    files = [os.fspath(frames)] if isinstance(frames, (str, os.PathLike)) else \
        [os.fspath(f) for f in frames]
    n_frames = sum(count_frames(f) for f in files)
    sweep = sweep_outcomes(sweep_dir, n_frames, params=params, frames=files, tolerance=tolerance)
    states = params.sorted_states
    first = np.zeros(n_frames)
    last = np.zeros(n_frames)
    by_frame = {}
    for shot in sweep['shots']:
        by_frame.setdefault(shot['frame'], []).append(shot)
    open_halves = 0
    for frame, shots in by_frame.items():
        used = [_first_launched(shots)[0]] if first_launched else shots
        for shot in used:
            for end in (shot['backward_end'], shot['forward_end']):
                if end == states[0]:
                    first[frame] += 1
                elif end == states[2]:
                    last[frame] += 1
                else:
                    open_halves += 1
    parts = [read_frames(f) for f in files]
    x = np.concatenate([p[0] for p in parts]).astype(float)
    boxes = [p[1] for p in parts]
    box = None if any(b is None for b in boxes) else np.concatenate(boxes)
    if atoms is not None:
        x = x[:, np.asarray(atoms, dtype=int)]
    notes = {'sweep_dir': sweep_dir, 'frames': [os.path.abspath(f) for f in files],
             'n_shots': len(sweep['shots']), 'open_halves': int(open_halves),
             'first_launched_only': bool(first_launched),
             'frames_without_shot': int(n_frames - len(by_frame)),
             'folders_with_a_shot_in_flight': sweep['in_flight'],
             'atoms': None if atoms is None else [int(i) for i in np.asarray(atoms).reshape(-1)],
             'source': 'aimmd.cgaimmd.analysis.shooting_validation.sweep_outcomes'}
    return CommittorReference(name or f'sweep {os.path.basename(sweep_dir)}', x, first, last,
                              kind='brute_force', notes=notes, dimensions=box)


# ════════════════════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Samples:
    """Generated samples per requested level."""

    levels: np.ndarray
    """``(L,)`` requested committor logits."""

    positions: np.ndarray
    """``(L, n, M, 3)`` float64 Å raw generator output (network atom order)."""

    seconds: float = 0.0
    """Wall time of the generation."""

    settings: dict = field(default_factory=dict)
    """Generator method and settings, samples per level, chunk and seed."""


def draw_samples(network, levels, n_per_level, *, generator=None, chunk=None, seed=0):
    """Generate ``n_per_level`` samples at every requested level through the generation interface.

    Parameters
    ----------
    network : object
        A C-GAIMMD network (weights loaded); used in eval mode on its device.
    levels : sequence of float
        Requested committor logits.
    n_per_level : int
        Samples per level.
    generator : str, dict or Generator, optional
        The generation method (:func:`aimmd.cgaimmd.generation.make_generator`;
        None: the network's default generator).
    chunk : int, optional
        Requests per ``generate`` call (default: the generator's ``chunk``
        setting, else all samples of a level at once).
    seed : int, optional
        Level ``l`` draws from a private ``torch.Generator`` seeded with
        ``seed + 1000 (round(2 l) + 100)``, the same for every generator, so
        that samples of different generators pair by rank; the global random
        stream is untouched.

    Returns
    -------
    Samples
        The raw samples.

    Raises
    ------
    ValueError
        If the generator's check finds the network unusable.
    """
    import torch

    from ..generation import check_generator, make_generator
    from ..protocol import n_atoms_of, network_device

    levels = np.asarray(levels, dtype=float).reshape(-1)
    n = int(n_per_level)
    if n < 1:
        raise ValueError(f'n_per_level must be >= 1, got {n_per_level}')
    gen = make_generator(generator, network)
    problems = check_generator(gen, network)
    if problems:
        raise ValueError(f'the generator {gen.method!r} cannot generate with this network: '
                         + '; '.join(problems))
    settings = dict(gen.settings())
    batch = int(chunk or settings.get('chunk') or n)
    device = network_device(network)
    m = n_atoms_of(network)
    positions = np.zeros((len(levels), n, m, 3))
    was_training = getattr(network, 'training', False)
    network.eval()
    start = time.time()
    try:
        with torch.no_grad():
            for li, level in enumerate(levels):
                stream = torch.Generator(device=device)
                stream.manual_seed(level_seed(seed, level))
                for lo in range(0, n, batch):
                    hi = min(lo + batch, n)
                    lam = torch.full((hi - lo,), float(level), dtype=torch.float32, device=device)
                    x = gen.generate(network, lam, generator=stream)
                    positions[li, lo:hi] = torch.as_tensor(x).detach().cpu().double().numpy()
    finally:
        if was_training:
            network.train(True)
    return Samples(levels=levels, positions=positions, seconds=time.time() - start,
                   settings={'generator': {'method': gen.method, 'settings': settings},
                             'chunk': batch, 'seed': int(seed), 'n_per_level': n})


# ════════════════════════════════════════════════════════════════════════════
# M1: validity
# ════════════════════════════════════════════════════════════════════════════

def check_validity(positions, validity, *, context=None):
    """Run a ``cgaimmd_validity_function`` on samples of one or many levels.

    Parameters
    ----------
    positions : array_like
        ``(L, n, M, 3)`` or ``(n, M, 3)`` Å raw samples.
    validity : callable or None
        ``validity(positions (n, M, 3), context) -> ValidityResult`` (None: no check, every sample counts as valid).
    context : aimmd.cgaimmd.types.CandidateContext, optional
        Passed to `validity` (None is accepted by :class:`~aimmd.cgaimmd.preparation.validity.TopologyValidity`).

    Returns
    -------
    dict
        ``ok`` bool, ``reason`` object (``''`` where ok), ``positions``
        relabelled into topology order (unchanged without an isomorphism),
        ``checked``, ``reflected`` bool (mirror images reflected back by a
        validity function with ``reflect_mirror_images``; False otherwise).
    """
    x = np.asarray(positions, dtype=float)
    single = x.ndim == 3
    if single:
        x = x[None]
    shape = x.shape[:2]
    if validity is None:
        out = {'ok': np.ones(shape, dtype=bool), 'reason': np.full(shape, '', dtype=object),
               'positions': x.copy(), 'checked': False, 'reflected': np.zeros(shape, dtype=bool)}
    else:
        flat = x.reshape(-1, *x.shape[2:])
        result = validity(flat, context)
        ok = np.asarray(result.ok, dtype=bool).reshape(shape)
        reason = np.asarray(result.reason, dtype=object).reshape(shape)
        relabelled = np.asarray(result.positions, dtype=float).reshape(x.shape)
        info = list(getattr(result, 'info', None) or [])
        reflected = np.array([bool(r.get('reflected', False)) if isinstance(r, dict) else False
                              for r in info]
                             if len(info) == len(flat) else np.zeros(len(flat), dtype=bool)
                             ).reshape(shape)
        out = {'ok': ok, 'reason': reason, 'positions': relabelled, 'checked': True,
               'reflected': reflected}
    if single:
        out.update(ok=out['ok'][0], reason=out['reason'][0], positions=out['positions'][0],
                   reflected=out['reflected'][0])
    return out


def validity_summary(levels, check, *, waters=None):
    """Per-level fractions of valid samples and of each rejection reason.

    Parameters
    ----------
    levels : sequence of float
        ``(L,)`` requested logits.
    check : dict
        :func:`check_validity` of ``(L, n, M, 3)`` samples.
    waters : array_like, optional
        ``(L, n)`` bool, :func:`waters_intact` of the raw samples.

    Returns
    -------
    list of dict
        One per level: ``lambda``, ``n``, ``valid``, ``reasons`` (fraction
        per code, nonzero only), ``waters_intact`` (None without waters),
        ``reflected`` (fraction of all samples that are valid only after
        reflecting a mirror image back; 0 without ``reflect_mirror_images``).
    """
    out = []
    for li, level in enumerate(np.asarray(levels, dtype=float).reshape(-1)):
        ok = np.asarray(check['ok'][li], dtype=bool)
        reason = np.asarray(check['reason'][li], dtype=object)
        codes = [str(r).split(':')[0] for r in reason[~ok]]
        reasons = {}
        for code in list(REASONS) + sorted(set(codes) - set(REASONS)):
            count = codes.count(code)
            if count:
                reasons[code] = count / len(ok)
        reflected = check.get('reflected')
        out.append({'lambda': float(level), 'n': int(len(ok)), 'valid': float(ok.mean()),
                    'reasons': reasons,
                    'reflected': 0.0 if reflected is None
                    else float((np.asarray(reflected[li], dtype=bool) & ok).mean()),
                    'waters_intact': None if waters is None
                    else float(np.asarray(waters[li], dtype=bool).mean())})
    return out


def waters_intact(positions, elements, *, n_solute, n_waters, atoms_per_water=3,
                  cutoff_scale=None):
    """Whether every cluster water of a sample is an intact, separate molecule.

    Parameters
    ----------
    positions : array_like
        ``(n, M, 3)`` Å samples.
    elements : array_like of int
        ``(M,)`` atomic numbers (network order).
    n_solute : int
        The solute atoms come first; the waters follow.
    n_waters : int
        Number of cluster waters (0: every sample passes).
    atoms_per_water : int, optional
        Atoms per water; the element pattern is taken from the first water.
    cutoff_scale : float, optional
        Bond rule of :func:`aimmd.cgaimmd.preparation.validity.covalent_bond_edges` (default its own).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool.

    Notes
    -----
    Label free: the connected components of the covalent bond graph are
    counted that have the water's elements and ``atoms_per_water - 1``
    bonds (an O with its two H, no H-H bond); a sample passes when there are
    `n_waters` of them. Waters that traded hydrogens therefore pass, a torn
    or a fused water fails. The bond-graph check of the validity function
    already rejects broken waters; this separates them from solute failures.
    """
    from ..preparation.validity import covalent_bond_edges

    x = np.asarray(positions, dtype=float)
    if x.ndim == 2:
        x = x[None]
    if int(n_waters) == 0:
        return np.ones(len(x), dtype=bool)
    z = np.asarray(elements, dtype=int).reshape(-1)
    s, apw = int(n_solute), int(atoms_per_water)
    template = sorted(z[s:s + apw].tolist())
    kwargs = {} if cutoff_scale is None else {'cutoff_scale': float(cutoff_scale)}
    out = np.zeros(len(x), dtype=bool)
    for k, sample in enumerate(x):
        if not np.all(np.isfinite(sample)):
            continue
        edges = covalent_bond_edges(sample, z, **kwargs)
        parent = np.arange(len(z))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in edges:
            ri, rj = root(int(i)), root(int(j))
            if ri != rj:
                parent[ri] = rj
        roots = np.array([root(i) for i in range(len(z))])
        edge_roots = roots[edges[:, 0]] if len(edges) else np.zeros(0, dtype=int)
        n_edges = np.bincount(edge_roots, minlength=len(z))
        count = 0
        for r in np.unique(roots):
            members = np.flatnonzero(roots == r)
            if (len(members) == apw and sorted(z[members].tolist()) == template
                    and n_edges[r] == apw - 1):
                count += 1
        out[k] = count == int(n_waters)
    return out


# ════════════════════════════════════════════════════════════════════════════
# M3: generator against committor
# ════════════════════════════════════════════════════════════════════════════

def generator_consistency(requested, measured, *, tolerance=1.0, transition_level=TRANSITION_LEVEL):
    """How well the committor of generated samples follows the requested level.

    Parameters
    ----------
    requested : array_like
        ``(n,)`` requested logits.
    measured : array_like
        ``(n,)`` committor logits of the samples.
    tolerance : float, optional
        Half-width of the ``within_tolerance`` band (logit units).
    transition_level : float, optional
        Requested levels with ``abs(lambda) <=`` it form the transition region.

    Returns
    -------
    dict
        ``n``, ``spearman``, ``within_tolerance``, ``median_abs_error``,
        ``median_error``, ``per_level``, ``transition_region``, ``tolerance``.

    Notes
    -----
    ``spearman`` is the rank correlation of requested and measured logits
    (average ranks for the ties of the requested levels; None when either
    is constant). ``within_tolerance`` is the fraction with
    ``abs(measured - requested) <= tolerance``. ``per_level`` holds the
    median and quartiles of the measured logits at every level. The finer
    measures of a dial (resolution, slopes per region) are in
    :func:`aimmd.cgaimmd.analysis.adherence.adherence_metrics`.
    """
    req = np.asarray(requested, dtype=float).reshape(-1)
    got = np.asarray(measured, dtype=float).reshape(-1)
    if req.shape != got.shape:
        raise ValueError(f'{len(req)} requested levels for {len(got)} measured logits')
    keep = np.isfinite(got)
    req, got = req[keep], got[keep]

    def block(r, g):
        if len(r) == 0:
            return {'n': 0, 'spearman': None, 'within_tolerance': None, 'median_abs_error': None,
                    'median_error': None}
        err = g - r
        return {'n': int(len(r)), 'spearman': _spearman(r, g),
                'within_tolerance': float(np.mean(np.abs(err) <= tolerance)),
                'median_abs_error': float(np.median(np.abs(err))),
                'median_error': float(np.median(err))}

    out = block(req, got)
    per = []
    for level in np.unique(req):
        sel = req == level
        g = got[sel]
        per.append({'lambda': float(level), 'n': int(sel.sum()), 'median': float(np.median(g)),
                    'q25': float(np.percentile(g, 25)), 'q75': float(np.percentile(g, 75)),
                    'within_tolerance': float(np.mean(np.abs(g - level) <= tolerance)),
                    'median_abs_error': float(np.median(np.abs(g - level)))})
    tr = np.abs(req) <= transition_level
    out.update(per_level=per, transition_region=block(req[tr], got[tr]),
               tolerance=float(tolerance), transition_level=float(transition_level))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Guard against copies: symmetry-minimized distances
# ════════════════════════════════════════════════════════════════════════════

def _ot_kwargs(symmetry):
    if symmetry is None:
        return {}
    kwargs = dict(symmetry.kwargs()) if hasattr(symmetry, 'kwargs') else dict(symmetry)
    out = {}
    for key in ('groups', 'factors', 'molecule_blocks'):
        value = kwargs.get(key)
        if value is not None and len(value):
            out[key] = value
    return out


def _moved_atoms(kwargs):
    """Every atom some symmetry element of an OT spec's kwargs can move."""
    moved = set()
    for group in kwargs.get('groups', ()) or ():
        moved.update(int(i) for i in group)
    factors = kwargs.get('factors', ()) or ()
    for support, _ in getattr(factors, 'factors', factors):
        moved.update(int(i) for i in support)
    blocks = kwargs.get('molecule_blocks', ()) or ()
    if len(blocks) > 1:
        for block in blocks:
            moved.update(int(i) for i in block)
    return moved


def _weights(m, atoms):
    import torch
    w = torch.zeros(m, dtype=torch.float64)
    if atoms is None:
        w[:] = 1.0
    else:
        idx = np.asarray(atoms, dtype=int).reshape(-1)
        if len(idx) < 3:
            raise ValueError(f'distances need at least 3 atoms, got {len(idx)}')
        w[torch.as_tensor(idx)] = 1.0
    return w


def _kabsch_rmsd(x, y, w):
    from ..flow_matching import flowmath
    xc, yc = flowmath.center(x, w), flowmath.center(y, w)
    xr = flowmath.kabsch_align(xc, yc, w)
    return ((((xr - yc) ** 2).sum(-1) * w).sum(-1) / w.sum()).sqrt()


def symmetric_rmsd(x, y, *, atoms=None, symmetry=None, n_rounds=2):
    """RMSD of paired structures after the best proper rotation and symmetry permutation.

    Parameters
    ----------
    x, y : array_like
        ``(n, M, 3)`` Å structures, compared row by row.
    atoms : array_like of int, optional
        Atoms the RMSD and the rotation are computed on (None: all).
    symmetry : object, optional
        ``aimmd.cgaimmd.preparation.symmetry.OTSpec`` or anything with
        ``kwargs()`` giving ``groups``, ``factors`` and ``molecule_blocks``
        (None: rotations only).
    n_rounds : int, optional
        Alternations of rotation and permutations.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64 Å.

    Notes
    -----
    Rotations are proper (Kabsch with the reflection guard), so a mirror
    image is not matched. The permutations of `symmetry` are searched as the
    optimal-transport pairing of flow matching does
    (:func:`aimmd.cgaimmd.flow_matching.flowmath.ot_pair`: exact factors,
    Hungarian groups, whole-molecule blocks), alternated with the rotation
    fitted on `atoms` (the first rotation on the atoms of `atoms` that no
    symmetry element moves, when there are three or more, so that a swapped
    pair does not bias it); the result is the smaller of that and the
    unpermuted Kabsch RMSD, so it never exceeds the plain RMSD. Like the
    pairing, the alternation is a local search: an upper bound on the true
    minimum over the group.
    """
    import torch

    from ..flow_matching import flowmath
    x = torch.as_tensor(np.asarray(x, dtype=float))
    y = torch.as_tensor(np.asarray(y, dtype=float))
    if x.dim() == 2:
        x, y = x[None], y[None]
    if x.shape != y.shape:
        raise ValueError(f'x and y differ in shape: {tuple(x.shape)} and {tuple(y.shape)}')
    w = _weights(x.shape[1], atoms)
    plain = _kabsch_rmsd(x, y, w)
    kwargs = _ot_kwargs(symmetry)
    if not kwargs:
        return plain.numpy()
    fixed = w.clone()
    moved = sorted(_moved_atoms(kwargs))
    if moved:
        fixed[torch.as_tensor(moved)] = 0.0
    first = fixed if int((fixed > 0).sum()) >= 3 else w
    yc = flowmath.center(y, w)
    xs = flowmath.center(x, w)
    for r in range(max(1, int(n_rounds))):
        fit = first if r == 0 else w
        xs = flowmath.kabsch_align(flowmath.center(xs, fit), flowmath.center(yc, fit), fit)
        xs = flowmath.ot_pair(xs, yc, rotate=False, n_rounds=1, **kwargs)
    permuted = _kabsch_rmsd(xs, yc, w)
    return torch.minimum(plain, permuted).numpy()


def _rmsd_matrix(x, frames, w, chunk):
    """``(S, T)`` Kabsch RMSD of every sample to every frame on the weighted atoms."""
    import torch

    from ..flow_matching import flowmath
    sel = w > 0
    xs = flowmath.center(x, w)[:, sel]
    ys = flowmath.center(frames, w)[:, sel]
    m = xs.shape[1]
    gx = (xs ** 2).sum((1, 2))
    out = torch.empty(len(xs), len(ys), dtype=torch.float64)
    for lo in range(0, len(ys), int(chunk)):
        yb = ys[lo:lo + int(chunk)]
        cov = torch.einsum('sai,taj->stij', xs, yb)
        sv = torch.linalg.svdvals(cov)
        sign = torch.sign(torch.linalg.det(cov))
        trace = sv[..., 0] + sv[..., 1] + sign * sv[..., 2]
        msd = (gx[:, None] + (yb ** 2).sum((1, 2))[None] - 2 * trace) / m
        out[:, lo:lo + len(yb)] = msd.clamp_min(0).sqrt()
    return out


def nearest_frames(x, frames, *, atoms=None, symmetry=None, k=8, chunk=256):
    """The nearest of `frames` to every structure of `x`, by symmetry-minimized RMSD.

    Parameters
    ----------
    x : array_like
        ``(S, M, 3)`` Å structures (samples or held-out frames).
    frames : array_like
        ``(T, M, 3)`` Å candidate frames (training frames).
    atoms : array_like of int, optional
        Atoms of the RMSD (None: all).
    symmetry : object, optional
        As for :func:`symmetric_rmsd` (None: rotations only).
    k : int, optional
        Candidates per structure refined with the symmetry permutations.
    chunk : int, optional
        Frames per block of the rotation-only distance matrix.

    Returns
    -------
    tuple
        ``(distance (S,), index (S,))``: RMSD in Å and the row of the nearest frame.

    Notes
    -----
    The rotation-only Kabsch RMSD to every frame comes from the singular
    values of the 3 x 3 covariances (with the reflection guard). Without
    `symmetry` its minimum is the answer. With `symmetry` the `k` nearest
    frames by the Kabsch RMSD over the selected atoms that no symmetry
    element moves (all selected atoms if fewer than three are left), a
    distance the permutations do not change, are refined by
    :func:`symmetric_rmsd`. A nearer frame outside these `k` is missed, so the
    distance is an upper bound.
    """
    import torch
    x = torch.as_tensor(np.asarray(x, dtype=float))
    frames = torch.as_tensor(np.asarray(frames, dtype=float))
    if len(x) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    if len(frames) == 0:
        raise ValueError('no frames to compare with')
    if x.shape[1:] != frames.shape[1:]:
        raise ValueError(f'structures {tuple(x.shape[1:])} and frames {tuple(frames.shape[1:])} '
                         f'differ in shape')
    w = _weights(x.shape[1], atoms)
    kwargs = _ot_kwargs(symmetry)
    if not kwargs:
        distance, index = _rmsd_matrix(x, frames, w, chunk).min(1)
        return distance.numpy(), index.numpy().astype(int)
    fixed = w.clone()
    moved = sorted(_moved_atoms(kwargs))
    if moved:
        fixed[torch.as_tensor(moved)] = 0.0
    prefilter = _rmsd_matrix(x, frames, fixed if int((fixed > 0).sum()) >= 3 else w, chunk)
    k = min(int(k), len(frames))
    candidates = prefilter.topk(k, dim=1, largest=False).indices            # (S, k)
    pairs_x = x.repeat_interleave(k, 0)
    pairs_y = frames[candidates.reshape(-1)]
    refined = torch.as_tensor(symmetric_rmsd(pairs_x.numpy(), pairs_y.numpy(), atoms=atoms,
                                             symmetry=symmetry)).reshape(len(x), k)
    best = refined.argmin(1)
    rows = torch.arange(len(x))
    return refined[rows, best].numpy(), candidates[rows, best].numpy().astype(int)


def guard_summary(distances, baseline=None):
    """Distances of samples against the held-out baseline.

    Parameters
    ----------
    distances : array_like
        ``(n,)`` Å distances of samples to their nearest training frame.
    baseline : array_like, optional
        ``(n_b,)`` Å distances of held-out MD frames to the nearest training frame.

    Returns
    -------
    dict
        ``n_samples``, ``median``, ``p5``, ``p95``, ``baseline_n``,
        ``baseline_p5``, ``baseline_median``, ``fraction_below_baseline_p5``.

    Notes
    -----
    ``fraction_below_baseline_p5`` is the fraction of samples closer than
    the 5th percentile of the baseline: about 0.05 for a generator whose
    samples are as new as held-out MD, near 1 for one that copies training
    frames.
    """
    d = np.asarray(distances, dtype=float).reshape(-1)
    d = d[np.isfinite(d)]
    out = {'n_samples': int(len(d)), 'median': json_float(np.median(d)) if len(d) else None,
           'p5': json_float(np.percentile(d, 5)) if len(d) else None,
           'p95': json_float(np.percentile(d, 95)) if len(d) else None,
           'baseline_n': 0, 'baseline_p5': None, 'baseline_median': None,
           'fraction_below_baseline_p5': None}
    if baseline is not None and len(np.asarray(baseline)):
        b = np.asarray(baseline, dtype=float).reshape(-1)
        p5 = float(np.percentile(b, 5))
        out.update(baseline_n=int(len(b)), baseline_p5=p5, baseline_median=float(np.median(b)),
                   fraction_below_baseline_p5=json_float(np.mean(d < p5)) if len(d) else None)
    return out


# ════════════════════════════════════════════════════════════════════════════
# One call per checkpoint
# ════════════════════════════════════════════════════════════════════════════

def _environment():
    import torch

    from ..sidecar import aimmd_provenance
    provenance = aimmd_provenance()
    commit = provenance.get('git_commit')
    if commit and provenance.get('git_dirty'):
        commit = f'{commit} (modified)'
    return {'aimmd_commit': commit, 'aimmd_version': provenance.get('version'),
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'host': platform.node(),
            'torch_threads': int(torch.get_num_threads())}


def _network_info(network):
    info = {'class': f'{type(network).__module__}.{type(network).__qualname__}',
            'n_parameters': int(sum(p.numel() for p in network.parameters()))}
    for key in ('n_atoms', 'lambda_clip', 'coord_scale', 'cutoff', 'parity_breaking'):
        value = getattr(network, key, None)
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            info[key] = value.item() if hasattr(value, 'item') else value
    return info


def _elements_of(network, elements):
    if elements is not None:
        return np.asarray(elements, dtype=int).reshape(-1)
    value = getattr(network, 'atomic_numbers', None)
    if value is None:
        return None
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value,
                      dtype=int).reshape(-1)


def _guard_atoms(guard_atoms, elements, water_layout, m):
    if guard_atoms is not None:
        idx = np.asarray(guard_atoms, dtype=int).reshape(-1)
        return idx, f'given atoms ({len(idx)})'
    n_solute = None if water_layout is None else int(water_layout[0])
    if elements is None:
        idx = np.arange(m if n_solute is None else n_solute)
        return idx, f'all {"solute" if n_solute is not None else "generator"} atoms ({len(idx)})'
    heavy = np.flatnonzero(elements != 1)
    if n_solute is not None:
        heavy = heavy[heavy < n_solute]
        return heavy, f'heavy solute atoms ({len(heavy)})'
    return heavy, f'heavy atoms of the generator atoms ({len(heavy)})'


def _subsample(frames, maximum):
    frames = np.asarray(frames, dtype=float)
    if maximum is None or len(frames) <= int(maximum):
        return frames, None
    pick = np.unique(np.linspace(0, len(frames) - 1, int(maximum)).round().astype(int))
    return frames[pick], pick


def _adherence(requested, judged, n_boot, transition_level):
    from .adherence import adherence_metrics
    if not len(requested):
        return None
    report = adherence_metrics(requested, judged, band=transition_level, n_boot=n_boot)
    return {'resolution_all': report['all_requests']['resolution'],
            'resolution_band': (report['band'] or {}).get('resolution'),
            'resolution_wings': (report['wings'] or {}).get('resolution'),
            'slope_all': report['all_requests']['slope'],
            'median_slopes': {k: v['slope'] for k, v in report['median_slopes'].items()},
            'neighbour_auc_all': report['all_requests']['neighbour_auc_mean'],
            'large_miss_share': report['all_requests']['large_miss_share'],
            'intervals': report['intervals']}


def validate(network, out, *, label=None, system=None, levels=DEFAULT_LEVELS, n_per_level=256,
             generator=None, chunk=None, validity=None, references=(),
             independent_committor=None, independent_name=None, minimizer=None,
             polish_per_level=0, polish_params=None, polish_box=None, polish_workers=1,
             training_frames=None, heldout_frames=None, guard_atoms=None, guard_per_level=64,
             guard_max_training=5000, guard_max_heldout=500, symmetry='network',
             water_layout=None, elements=None, seed=0, committor_batch_size=None,
             prob_clip=PROB_CLIP, tolerance=1.0, transition_level=TRANSITION_LEVEL,
             adherence_boot=200, metadata=None, card=True, save_samples=True):
    """Validate one C-GAIMMD checkpoint: JSON report, sample arrays and model card.

    Parameters
    ----------
    network : object
        The C-GAIMMD network (weights loaded); used in eval mode on its current device.
    out : str
        Output path stem: ``<out>.json``, ``<out>_samples.npz``,
        ``<out>_card.png`` and ``.pdf``; nothing else is written. A stem
        inside a run folder is refused.
    label : str, optional
        Descriptive name of the model (default: the stem's name).
    system : str, optional
        Descriptive name of the system.
    levels : sequence of float, optional
        Requested committor logits of M1 and M3.
    n_per_level : int, optional
        Samples drawn per level.
    generator : str, dict or Generator, optional
        The generation method (default: the network's own default generator).
    chunk : int, optional
        Requests per generation call (:func:`draw_samples`).
    validity : callable, optional
        A ``cgaimmd_validity_function`` (e.g.
        :class:`~aimmd.cgaimmd.preparation.validity.TopologyValidity`); None
        counts every sample as valid.
    references : sequence of CommittorReference, optional
        The references of M2 (brute-force sets, held-out chains, free MD), in
        the network's atoms.
    independent_committor : callable, optional
        ``f(positions (n, M, 3) Å) -> (n,)`` logits of a committor that shares
        no weights with the network (M3's second opinion).
    independent_name : str, optional
        Descriptive name of the independent committor.
    minimizer : callable, optional
        A ``cgaimmd_minimize_function`` (e.g.
        :class:`~aimmd.cgaimmd.preparation.minimize.GromacsMinimizer`) for the
        polish of M1.
    polish_per_level : int, optional
        The first this many draws of every level are polished (0: no polish).
    polish_params : object, optional
        The ``params`` the minimizer reads (``universe``, GROMACS commands);
        required with a minimizer.
    polish_box : array_like, optional
        ``(6,)`` box (Å, degrees) the samples are centred in (default
        ``polish_params.universe.dimensions``).
    polish_workers : int, optional
        Minimizations run in parallel threads.
    training_frames : array_like, optional
        ``(T, M, 3)`` Å frames the network was trained on (the guard's nearest-frame search).
    heldout_frames : array_like, optional
        ``(H, M, 3)`` Å unbiased MD frames the network was not trained on (the guard's baseline).
    guard_atoms : array_like of int, optional
        Atoms of the guard's RMSD (default: the heavy solute atoms, see Notes).
    guard_per_level : int, optional
        Valid samples per level that enter the guard.
    guard_max_training, guard_max_heldout : int, optional
        Larger frame sets are subsampled evenly to this many frames.
    symmetry : object or str, optional
        ``'network'`` (the network's ``ot_symmetry``), an ``OTSpec``-like
        object, or None (rotations only).
    water_layout : tuple, optional
        ``(n_solute, n_waters, atoms_per_water)`` of a cluster: enables the waters-intact check.
    elements : array_like of int, optional
        ``(M,)`` atomic numbers (default ``network.atomic_numbers``).
    seed : int, optional
        Seed of the generation (:func:`draw_samples`).
    committor_batch_size : int, optional
        Frames per committor call.
    prob_clip : float, optional
        Probability clip of the log-likelihoods of M2.
    tolerance : float, optional
        The ``within_tolerance`` band of M3, logit units.
    transition_level : float, optional
        Levels with ``abs(lambda) <=`` it are the transition region of M1 and M3.
    adherence_boot : int, optional
        Bootstrap resamples of the adherence numbers of M3 (0: none).
    metadata : dict, optional
        Copied into the report; the model card shows ``checkpoint`` (a path,
        or a dict with ``path`` and ``sha256``; a path's hash may be
        ``checkpoint_sha256``).
    card : bool, optional
        Draw the model card.
    save_samples : bool, optional
        Write the per-sample arrays.

    Returns
    -------
    dict
        The report (also written to ``<out>.json``); ``report['files']`` lists what was written.

    Raises
    ------
    ValueError
        For a minimizer without `polish_params`, or frame sets of another atom count.

    Notes
    -----
    Steps: :func:`draw_samples`; M1 with :func:`check_validity` and
    :func:`waters_intact` (raw), then the polish of the first
    `polish_per_level` draws of each level (the relabelled sample, centred in
    `polish_box`, is minimized alone and checked again; a failed minimization
    counts as invalid); M2 with the network's committor
    (:func:`network_committor`) on every reference; M3 on the valid samples
    (relabelled) with the network's committor and the independent committor,
    and with a polish also on the samples valid after it; the guard on up to
    `guard_per_level` valid samples per level: :func:`nearest_frames` to the
    training frames, and the nearest-training-frame distances of the held-out
    frames as the baseline.

    The default guard atoms are the heavy atoms of the generator atoms (of
    the solute only with a `water_layout`), from `elements` or the network's
    ``atomic_numbers``; without elements all atoms. With ``symmetry=
    'network'`` and a network without ``ot_symmetry`` only rotations are
    searched (reported as ``'none (rotations only)'``).

    The report is JSON with ``schema`` :data:`SCHEMA`; NaN and infinities
    are written as null. The model card is one page, readable on its own.
    """
    import torch

    from ..io import atomic_write_json
    from ..protocol import n_atoms_of

    from ...analysis.reweighting import check_output_outside

    t0 = time.time()
    out = os.path.abspath(os.fspath(out))
    check_output_outside(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if minimizer is not None and int(polish_per_level) > 0 and polish_params is None:
        raise ValueError('a minimizer needs polish_params (the params it reads: universe, '
                         'GROMACS commands)')
    m = n_atoms_of(network)
    elements = _elements_of(network, elements)
    for name, frames in (('training_frames', training_frames), ('heldout_frames', heldout_frames)):
        if frames is not None and np.asarray(frames).shape[1:] != (m, 3):
            raise ValueError(f'{name} must have shape (n, {m}, 3), got {np.asarray(frames).shape}')
    timing = {}
    arrays = {}

    # --- samples and M1 --------------------------------------------------------------------
    samples = draw_samples(network, levels, n_per_level, generator=generator, chunk=chunk,
                           seed=seed)
    timing['generate_seconds'] = round(samples.seconds, 2)
    levels = samples.levels
    report = {'schema': SCHEMA, 'label': label or os.path.basename(out), 'system': system,
              'created': datetime.datetime.now().isoformat(timespec='seconds'),
              'network': _network_info(network),
              'settings': {'levels': [float(v) for v in levels], 'n_per_level': int(n_per_level),
                           'generator': samples.settings['generator'],
                           'chunk': samples.settings['chunk'], 'seed': int(seed),
                           'prob_clip': float(prob_clip), 'transition_q': list(TRANSITION_Q),
                           'saturation': SATURATION, 'tolerance': float(tolerance),
                           'transition_level': float(transition_level),
                           'polish_per_level': int(polish_per_level) if minimizer else 0,
                           'guard_per_level': int(guard_per_level),
                           'water_layout': None if water_layout is None else list(water_layout),
                           'adherence_boot': int(adherence_boot)},
              'metadata': jsonable(metadata or {})}
    t = time.time()
    check = check_validity(samples.positions, validity)
    timing['validity_seconds'] = round(time.time() - t, 2)
    waters = None
    if water_layout is not None and elements is not None:
        s, k, apw = (list(water_layout) + [3])[:3]
        waters = np.stack([waters_intact(samples.positions[li], elements, n_solute=s, n_waters=k,
                                         atoms_per_water=apw) for li in range(len(levels))])
    rows = validity_summary(levels, check, waters=waters)
    own = network_committor(network, batch_size=committor_batch_size)
    polish_arrays = None
    n_polish = min(int(polish_per_level), int(n_per_level)) if minimizer is not None else 0
    if n_polish:
        from ..preparation.minimize import polish
        t = time.time()
        box = polish_box
        if box is None:
            universe = getattr(polish_params, 'universe', None)
            box = None if universe is None else universe.dimensions
            if box is None:
                raise ValueError('polish_box is needed when polish_params has no universe box')
        scratch = tempfile.mkdtemp(prefix='validate_polish_', dir=os.path.dirname(out))
        try:
            polish_arrays = {'positions': np.full((len(levels), n_polish, m, 3), np.nan),
                             'minimized': np.zeros((len(levels), n_polish), dtype=bool),
                             'ok': np.zeros((len(levels), n_polish), dtype=bool),
                             'reason': np.full((len(levels), n_polish), '', dtype=object)}
            for li in range(len(levels)):
                before = check['positions'][li, :n_polish]
                results = polish(before, minimizer, network, polish_params, box, scratch,
                                 polish_workers, seed + 100000 * li)
                done = np.array([r.ok for r in results])
                after = np.full_like(before, np.nan)
                for j, r in enumerate(results):
                    if r.ok:
                        after[j] = r.positions
                ok_after = np.zeros(n_polish, dtype=bool)
                reason_after = np.array([r.reason for r in results], dtype=object)
                if done.any():
                    again = check_validity(after[done], validity)
                    ok_after[done] = again['ok']
                    reason_after[done] = again['reason']
                    after[done] = again['positions']
                shifts = np.full(n_polish, np.nan)
                if done.any():
                    shifts[done] = np.abs(own(after[done]) - own(before[done]))
                displacement = [r.record.get('heavy_rms_displacement_A') for r in results if r.ok]
                displacement = [d for d in displacement if d is not None]
                rows[li]['polish'] = {
                    'n': n_polish, 'minimized': int(done.sum()),
                    'minimize_failed': int((~done).sum()),
                    'valid_after': float(ok_after.mean()),
                    'valid_after_given_minimized': json_float(ok_after[done].mean())
                    if done.any() else None,
                    'valid_before': float(check['ok'][li, :n_polish].mean()),
                    'median_heavy_rms_displacement_A': json_float(np.median(displacement))
                    if displacement else None,
                    'median_own_lambda_shift': json_float(np.nanmedian(shifts))
                    if done.any() else None}
                polish_arrays['positions'][li] = after
                polish_arrays['minimized'][li] = done
                polish_arrays['ok'][li] = ok_after
                polish_arrays['reason'][li] = reason_after
        finally:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)
        timing['polish_seconds'] = round(time.time() - t, 2)
    tr_rows = [r for r in rows if abs(r['lambda']) <= transition_level]
    end_rows = [r for r in rows if abs(r['lambda']) > transition_level]
    summary = {'valid_min': min(r['valid'] for r in rows),
               'valid_mean': float(np.mean([r['valid'] for r in rows])),
               'valid_transition_mean': json_float(np.mean([r['valid'] for r in tr_rows]))
               if tr_rows else None,
               'valid_end_mean': json_float(np.mean([r['valid'] for r in end_rows]))
               if end_rows else None,
               'yield': float(check['ok'].mean()),
               'seconds_per_valid_sample': json_float(samples.seconds / check['ok'].sum())
               if check['ok'].any() else None,
               'waters_intact_min': None if waters is None else float(waters.mean(1).min())}
    if n_polish:
        summary['valid_after_polish_mean'] = float(np.mean([r['polish']['valid_after']
                                                            for r in rows]))
        summary['valid_after_polish_min'] = min(r['polish']['valid_after'] for r in rows)
    report['m1'] = {'checked': check['checked'],
                    'validity': jsonable(getattr(validity, 'settings', None))
                    if validity is not None else None,
                    'levels': rows, 'summary': summary}

    # --- M2 ----------------------------------------------------------------------------------
    t = time.time()
    report['m2'] = {}
    arrays['m2_names'] = np.array([r.name for r in references], dtype=str)
    for k, reference in enumerate(references):
        if reference.name in report['m2']:
            raise ValueError(f'two references are named {reference.name!r}')
        result, lam = score_committor(own, reference, prob_clip=prob_clip)
        report['m2'][reference.name] = result
        arrays[f'm2_logits_{k}'] = lam
        arrays[f'm2_n_first_{k}'] = reference.n_first
        arrays[f'm2_n_last_{k}'] = reference.n_last
        arrays[f'm2_include_{k}'] = (np.ones(len(lam), dtype=bool) if reference.include is None
                                     else reference.include)
        if reference.groups is not None:
            arrays[f'm2_groups_{k}'] = reference.groups
    timing['m2_seconds'] = round(time.time() - t, 2)

    # --- M3 ----------------------------------------------------------------------------------
    t = time.time()
    ok = check['ok']
    requested = np.broadcast_to(levels[:, None], ok.shape)
    own_lam = np.full(ok.shape, np.nan)
    ind_lam = np.full(ok.shape, np.nan)
    if ok.any():
        own_lam[ok] = own(check['positions'][ok])
        if independent_committor is not None:
            ind_lam[ok] = np.asarray(independent_committor(check['positions'][ok]),
                                     dtype=float).reshape(-1)
    report['m3'] = {
        'own': generator_consistency(requested[ok], own_lam[ok], tolerance=tolerance,
                                     transition_level=transition_level),
        'independent': None if independent_committor is None else generator_consistency(
            requested[ok], ind_lam[ok], tolerance=tolerance, transition_level=transition_level),
        'independent_name': independent_name if independent_committor is not None else None,
        'own_vs_independent_median_abs_difference':
            json_float(np.median(np.abs(own_lam[ok] - ind_lam[ok])))
            if independent_committor is not None and ok.any() else None,
        'own_adherence': _adherence(requested[ok], own_lam[ok], adherence_boot,
                                    transition_level),
        'independent_adherence': None if independent_committor is None else _adherence(
            requested[ok], ind_lam[ok], adherence_boot, transition_level)}
    if polish_arrays is not None:
        ok_p = polish_arrays['ok']
        req_p = np.broadcast_to(levels[:, None], ok_p.shape)
        own_p = np.full(ok_p.shape, np.nan)
        ind_p = np.full(ok_p.shape, np.nan)
        if ok_p.any():
            own_p[ok_p] = own(polish_arrays['positions'][ok_p])
            if independent_committor is not None:
                ind_p[ok_p] = np.asarray(independent_committor(polish_arrays['positions'][ok_p]),
                                         dtype=float).reshape(-1)
        report['m3']['own_after_polish'] = generator_consistency(
            req_p[ok_p], own_p[ok_p], tolerance=tolerance, transition_level=transition_level)
        report['m3']['independent_after_polish'] = None if independent_committor is None else \
            generator_consistency(req_p[ok_p], ind_p[ok_p], tolerance=tolerance,
                                  transition_level=transition_level)
        polish_arrays.update(own_lambda=own_p, independent_lambda=ind_p)
    timing['m3_seconds'] = round(time.time() - t, 2)

    # --- guard -----------------------------------------------------------------------------
    t = time.time()
    report['guard'] = None
    d_train = np.full(ok.shape, np.nan)
    i_train = np.full(ok.shape, -1, dtype=np.int64)
    in_guard = np.zeros(ok.shape, dtype=bool)
    for li in range(len(levels)):
        in_guard[li, np.flatnonzero(ok[li])[:int(guard_per_level)]] = True
    if training_frames is not None:
        atoms, atoms_label = _guard_atoms(guard_atoms, elements, water_layout, m)
        if isinstance(symmetry, str):
            if symmetry != 'network':
                raise ValueError(f"symmetry must be 'network', an OTSpec or None, got {symmetry!r}")
            spec = getattr(network, 'ot_symmetry', None)
            symmetry_label = 'network' if spec is not None else 'none (rotations only)'
        else:
            spec = symmetry
            symmetry_label = 'none (rotations only)' if spec is None else 'given'
        guard = {'atoms': atoms_label, 'symmetry': symmetry_label, 'training': None,
                 'baseline': None, 'n_training_frames': 0, 'n_heldout_frames': 0}
        baseline = None
        train, pick = _subsample(training_frames, guard_max_training)
        guard['n_training_frames'] = int(len(train))
        guard['training_frames_subsampled_from'] = (None if pick is None
                                                    else int(len(training_frames)))
        if heldout_frames is not None:
            held, _ = _subsample(heldout_frames, guard_max_heldout)
            baseline, _ = nearest_frames(held, train, atoms=atoms, symmetry=spec)
            guard['n_heldout_frames'] = int(len(held))
            guard['baseline'] = guard_summary(baseline)
            arrays['guard_baseline'] = baseline
        if in_guard.any():
            d, i = nearest_frames(check['positions'][in_guard], train, atoms=atoms, symmetry=spec)
            d_train[in_guard], i_train[in_guard] = d, i
        guard['training'] = guard_summary(d_train[in_guard], baseline)
        guard['training']['per_level'] = [
            {'lambda': float(level), **guard_summary(d_train[li][in_guard[li]], baseline)}
            for li, level in enumerate(levels)]
        report['guard'] = guard
    timing['guard_seconds'] = round(time.time() - t, 2)

    # --- write -------------------------------------------------------------------------------
    timing['total_seconds'] = round(time.time() - t0, 2)
    report['timing'] = timing
    report['environment'] = _environment()
    files = {'json': out + '.json'}
    if save_samples:
        files['samples'] = out + '_samples.npz'
        arrays.update(
            levels=levels, raw=samples.positions.astype(np.float32),
            relabelled=check['positions'].astype(np.float32), ok=ok,
            reason=check['reason'].astype(str), own_lambda=own_lam, independent_lambda=ind_lam,
            in_guard=in_guard, d_train=d_train, i_train=i_train)
        if waters is not None:
            arrays['waters_intact'] = waters
        if polish_arrays is not None:
            arrays.update(polish_positions=polish_arrays['positions'].astype(np.float32),
                          polish_minimized=polish_arrays['minimized'], polish_ok=polish_arrays['ok'],
                          polish_reason=polish_arrays['reason'].astype(str),
                          polish_own_lambda=polish_arrays['own_lambda'],
                          polish_independent_lambda=polish_arrays['independent_lambda'])
        tmp = files['samples'] + '.tmp.npz'
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, files['samples'])
    if card:
        from ._validate_plots import model_card
        files['card_png'], files['card_pdf'] = model_card(
            report, {'levels': levels, 'ok': ok, 'own_lambda': own_lam,
                     'independent_lambda': ind_lam, 'd_train': d_train, 'in_guard': in_guard,
                     'baseline': arrays.get('guard_baseline'),
                     'references': [(r, arrays[f'm2_logits_{k}'])
                                    for k, r in enumerate(references)]},
            out + '_card')
    report['files'] = files
    report = jsonable(report)
    atomic_write_json(files['json'], report)
    if torch.cuda.is_available() and str(getattr(network, 'device', 'cpu')).startswith('cuda'):
        report['environment']['cuda_peak_MB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return report


def load_report(path):
    """Read a report written by :func:`validate` (also one of the old ``aimmd.flow.validate`` schema).

    Parameters
    ----------
    path : str
        The ``.json`` file.

    Returns
    -------
    dict
        The report; an old report's ``schema`` is translated to the new name
        (with one ``DeprecationWarning``).

    Raises
    ------
    ValueError
        If the file is not a validation report.
    """
    with open(os.fspath(path)) as handle:
        report = json.load(handle)
    schema = str(report.get('schema', ''))
    if schema.startswith(_OLD_SCHEMA_PREFIX):
        from .. import _legacy
        if schema not in _legacy.VALIDATION_SCHEMAS:
            raise ValueError(f'{path} has the unknown old schema {schema!r}')
        report['schema'] = _legacy.translate(_legacy.VALIDATION_SCHEMAS, schema,
                                             'validation report schema')
        settings = report.get('settings') or {}
        if 'generate' in settings and 'generator' not in settings:
            method = _legacy.translate(_legacy.GENERATOR_LABELS, settings['generate'],
                                       'generation label')
            settings['generator'] = {'method': method, 'settings': {
                key: settings[key] for key in ('sampler', 'n_steps', 'chunk') if key in settings}}
    elif not schema.startswith('aimmd.cgaimmd.validate/'):
        raise ValueError(f'{path} is not an aimmd.cgaimmd.validate report')
    return report


def redraw_card(path, *, stem=None):
    """Draw the model card of a written report again, from its JSON and its sample arrays.

    Parameters
    ----------
    path : str
        The report's ``.json`` file (its ``_samples.npz`` must exist).
    stem : str, optional
        Output path stem (default: the report's own ``<out>_card``); not
        inside a run folder.

    Returns
    -------
    tuple
        ``(png, pdf)`` paths.

    Notes
    -----
    Nothing is generated or evaluated: the card is drawn from the saved
    logits, counts and distances, so a change of the card's layout can be
    applied to existing reports.
    """
    from types import SimpleNamespace

    from ._validate_plots import model_card
    report = load_report(path)
    samples = report.get('files', {}).get('samples')
    if not samples or not os.path.exists(samples):
        raise ValueError(f'{path}: the sample arrays are needed and were not found ({samples})')
    with np.load(samples, allow_pickle=False) as z:
        references = []
        for k, name in enumerate(z['m2_names'].tolist()):
            references.append((SimpleNamespace(name=str(name), kind=report['m2'][name]['kind'],
                                               n_first=z[f'm2_n_first_{k}'],
                                               n_last=z[f'm2_n_last_{k}'],
                                               include=z[f'm2_include_{k}']),
                               z[f'm2_logits_{k}']))
        data = {key: z[key] for key in ('levels', 'ok', 'own_lambda', 'independent_lambda',
                                        'd_train', 'in_guard')}
        data.update(baseline=z['guard_baseline'] if 'guard_baseline' in z.files else None,
                    references=references)
    from ...analysis.reweighting import check_output_outside
    if stem is None:
        stem = os.path.splitext(os.path.abspath(os.fspath(path)))[0] + '_card'
    check_output_outside(stem)
    return model_card(report, data, stem)


# ════════════════════════════════════════════════════════════════════════════
# Leaderboard
# ════════════════════════════════════════════════════════════════════════════

_M2_COLUMNS = ('loglik_per_endpoint', 'loglik_saturated', 'loglik_constant',
               'calibration_slope', 'calibration_intercept', 'spearman', 'rmse_transition',
               'bias_transition', 'n_saturated_transition', 'n_transition', 'r2',
               'r2_noise_ceiling', 'r2_noise_ceiling_entries')
_M3_COLUMNS = ('spearman', 'within_tolerance', 'median_abs_error')


def _row(report):
    generator = ((report.get('settings') or {}).get('generator') or {})
    row = {'label': report.get('label'), 'system': report.get('system'),
           'generator': generator.get('method') if isinstance(generator, dict) else None,
           'json': (report.get('files') or {}).get('json')}
    m1 = (report.get('m1') or {}).get('summary') or {}
    for key in ('valid_min', 'valid_mean', 'valid_transition_mean', 'valid_end_mean',
                'waters_intact_min', 'valid_after_polish_mean'):
        row[f'm1:{key}'] = m1.get(key)
    for name, m2 in (report.get('m2') or {}).items():
        for key in _M2_COLUMNS:
            row[f'm2:{name}:{key}'] = m2.get(key)
    m3 = report.get('m3') or {}
    for which in ('own', 'independent', 'own_after_polish', 'independent_after_polish'):
        if which.endswith('after_polish') and not m3.get(which):
            continue
        block = m3.get(which) or {}
        for key in _M3_COLUMNS:
            row[f'm3:{which}:{key}'] = block.get(key)
        row[f'm3:{which}:transition_within_tolerance'] = (block.get('transition_region')
                                                          or {}).get('within_tolerance')
    for which in ('own', 'independent'):
        block = m3.get(f'{which}_adherence') or {}
        row[f'm3:{which}:resolution_all'] = block.get('resolution_all')
        row[f'm3:{which}:core_slope'] = (block.get('median_slopes') or {}).get('core')
    guard = report.get('guard') or {}
    block = guard.get('training') or {}
    row['guard:training:median'] = block.get('median')
    row['guard:training:fraction_below_baseline_p5'] = block.get('fraction_below_baseline_p5')
    row['guard:baseline_median'] = (guard.get('baseline') or {}).get('median')
    row['seconds'] = (report.get('timing') or {}).get('total_seconds')
    return row


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


def leaderboard(reports, out, *, title=None, figure=True):
    """Tabulate many reports: CSV, Markdown and JSON tables and a comparison figure.

    Parameters
    ----------
    reports : sequence
        Report dicts or paths of report JSON files, in the order of the table.
    out : str
        Output path stem (``.csv``, ``.md``, ``.json``, ``.png``, ``.pdf``);
        not inside a run folder.
    title : str, optional
        Title of the figure and the Markdown table.
    figure : bool, optional
        Draw the figure.

    Returns
    -------
    dict
        ``rows`` (one flat dict per report), ``columns`` and ``files``.

    Notes
    -----
    Columns are ``label``, ``system``, ``generator``, ``json``, then
    ``m1:<summary key>``, ``m2:<reference>:<metric>`` for every reference of
    any report, ``m3:<own|independent>:<metric>``,
    ``guard:training:<metric>`` and ``seconds``; a report without a value
    leaves the cell empty.
    """
    from ...analysis.reweighting import check_output_outside
    from ..io import atomic_write_bytes, atomic_write_json

    check_output_outside(os.path.abspath(os.fspath(out)))
    loaded = [load_report(r) if isinstance(r, (str, os.PathLike)) else r for r in reports]
    rows = [_row(r) for r in loaded]
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    rows = [{key: row.get(key) for key in columns} for row in rows]
    out = os.path.abspath(os.fspath(out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    files = {'csv': out + '.csv', 'markdown': out + '.md', 'json': out + '.json'}

    def csv_cell(value):
        text = _cell(value)
        return f'"{text}"' if (',' in text or '"' in text) else text

    lines = [','.join(columns)] + [','.join(csv_cell(row[c]) for c in columns) for row in rows]
    atomic_write_bytes(files['csv'], ('\n'.join(lines) + '\n').encode())
    shown = [c for c in columns if c != 'json']
    md = [f'# {title}', ''] if title else []
    md += ['| ' + ' | '.join(shown) + ' |', '|' + '---|' * len(shown)]
    md += ['| ' + ' | '.join(_cell(row[c]) for c in shown) + ' |' for row in rows]
    atomic_write_bytes(files['markdown'], ('\n'.join(md) + '\n').encode())
    atomic_write_json(files['json'], jsonable({'title': title, 'columns': columns, 'rows': rows}))
    if figure:
        from ._validate_plots import leaderboard_figure
        files['png'], files['pdf'] = leaderboard_figure(loaded, rows, out, title=title)
    return {'rows': rows, 'columns': columns, 'files': files}


def parity_figure(entries, stem, *, title=None):
    """Committor parity of several models on references, side by side.

    Parameters
    ----------
    entries : sequence
        One per panel: ``(label, reference, logits)`` with a
        :class:`CommittorReference` and the model's ``(n,)`` logits on it
        (the second value of :func:`score_committor`).
    stem : str or os.PathLike
        Output path stem: ``<stem>.png`` and ``<stem>.pdf``; not inside a run folder.
    title : str, optional
        Figure title.

    Returns
    -------
    tuple
        ``(png, pdf)`` paths.

    Notes
    -----
    Top row, one panel per entry: for a brute-force reference, every
    configuration's observed committor ``q`` (with its 68 % binomial interval)
    against the model's ``p``, with the log-likelihood per endpoint, the
    calibration line, Spearman and R² (with its noise ceiling) written in;
    for references with one or two endpoints per configuration, the observed
    fraction in ten bins of ``p`` instead. Bottom: the reliability curves of
    every entry in one panel (observed fraction against ``p`` in ten equal
    bins), and the distribution of the logits.
    """
    from ...analysis.reweighting import check_output_outside
    from ._validate_plots import parity_panels
    stem = os.path.abspath(os.fspath(stem))
    check_output_outside(stem)
    return parity_panels(entries, stem, title=title)
