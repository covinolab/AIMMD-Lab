"""
aimmd.cgaimmd.analysis.shooting_validation
==========================================

Does a generated configuration have the committor it was requested at? The
direct test: shoot from it.

The procedure
-------------
1. Generate samples at a few requested committor logits ``lambda*`` (the
   requests) and prepare each exactly as a C-GAIMMD run would before shooting
   it (:func:`write_validation_frames`: the run's validity function, its
   preparation and minimization hooks, the state check). Which samples are
   used is decided before any outcome is known: the first ``n`` per request,
   in seed order, never chosen by a score.
2. Shoot one two-way shot from every prepared frame that lies in the reactive
   region, in an AIMMD *sweep* (``reactive_region_mode='sweep'``: shots from
   a fixed set of frames, no training; the frames written in step 1 are the
   sweep's initial path).
3. Read the outcomes (:func:`sweep_outcomes`, :func:`validation_rows`) and
   summarise them per request (:func:`level_summary`) and over all requests
   (:func:`dial_slope`).

The run's own shots are a free version of the same measurement
(:func:`from_shot_table`): every C-GAIMMD shot is one two-way shot from one
generated configuration. They are filtered, though: once the network is
trained, the run accepts a candidate only if the network puts it within
``lambda_tolerance`` of its request (the final check), so they measure the
generator together with that check, not the generator alone.

What is computed
----------------
A two-way shot sends two trajectories (halves) from a configuration, with
opposite velocities; each half ends in the first end state, the last end
state (the committor is the probability of reaching the last one), or still
in the reactive region when it hits the maximum length (uncommitted).

- **Committor mean per request**, ``p``: the mean over frames of the fraction
  of a frame's committed halves that end in the last state, with a bootstrap
  interval over frames, and bounds that count every uncommitted or
  unsimulated half as the first (lower bound) or the last state (upper bound).
  For a perfect generator ``p = sigmoid(lambda*)``.
- **Transition share** ``T``: the share of fully committed shots whose halves
  end in different states.
- **Committor variance at a fixed request**, ``Var(p) = (2 p (1 - p) - T) / 2``,
  with ``p`` over the same fully committed shots: zero when every sample of a
  request has the same committor, ``p (1 - p)`` at most. It assumes that the
  two halves of a shot are independent draws given the configuration, so that
  a configuration of committor ``p`` makes a transition with probability
  ``2 p (1 - p)``. That holds for diffusive dynamics. Under inertial
  dynamics the halves start from opposite velocities of one draw and are
  correlated; then ``T`` is inflated and the variance comes out too low, even
  negative (``committor_sd_if_positive`` is None then). It is also a
  small-sample estimate, biased low by about the variance of ``p``'s own
  estimate.
- **Dial slope** ``b``: a logistic fit of every committed half's outcome
  (1: last state, 0: first state) on the request, ``P(last | lambda*) =
  sigmoid(a + b lambda*)``. ``b = 1`` is a perfect dial; spread at a fixed
  request flattens it (``b < 1``); ``b > 1`` overshoots, which is not better
  adherence. It comes with a bootstrap interval (frames resampled within
  requests) and with bounds from assigning the uncommitted halves every way.
- **Comparison of two generators** (:func:`compare_dials`): the difference of
  their slopes with a bootstrap interval, independent or paired by sample
  rank (the same rank of a request has the same random seed for every
  generator written by :func:`write_validation_frames`).

Frames whose prepared configuration is already in an end state count as one
shot with both halves committed to that state at time zero. Samples whose
preparation failed have no outcome; they are counted per request.

Reading
-------
Nothing here writes into a run or sweep folder: trajectory files are read with
MDAnalysis's low-level file objects, which keep no frame index next to the
file, and caches are read with numpy. :func:`write_validation_frames` writes
only into the folder its caller names.
"""

import datetime
import glob
import json
import math
import os
import re

import numpy as np

from ._stats import jsonable, level_seed, percentile_interval, sigmoid

__all__ = ['NOT_SIMULATED', 'UNCOMMITTED_RULES', 'VARIANCE_NOTE', 'frame_outcomes',
           'level_statistics', 'level_summary', 'half_outcomes', 'logistic_dial',
           'dial_slope', 'resample_within_levels', 'compare_dials', 'matched_ranks',
           'sweep_outcomes', 'validation_rows', 'from_shot_table', 'write_validation_frames',
           'load_validation_table']

NOT_SIMULATED = '-'
"""The end of a half that was never run (AIMMD skips the forward half when the
backward half used the whole ``max_length`` of the two-way path)."""

UNCOMMITTED_RULES = ('excluded', 'as_first', 'as_last', 'toward_request', 'against_request')
"""What :func:`half_outcomes` does with uncommitted and unsimulated halves:
leave them out, count them as the first or the last end state, or count them
toward the request (last state above ``lambda* = 0``, first below, a half at
0) or against it."""

VARIANCE_NOTE = ('committor variance (2 p (1 - p) - T) / 2, p and T over the same fully '
                 'committed shots; assumes the two halves of a shot are independent draws given '
                 'the configuration (diffusive dynamics); a small-sample estimate that can be '
                 'negative')
"""The caveat of the variance estimate, written into every summary."""

VALIDATION_TABLE = 'frames_table.json'
"""The manifest :func:`write_validation_frames` writes."""

VALIDATION_FRAMES = 'shooting_frames'
"""The stem of the frames file :func:`write_validation_frames` writes."""


# ════════════════════════════════════════════════════════════════════════════
# Per frame and per request
# ════════════════════════════════════════════════════════════════════════════

def frame_outcomes(halves, *, states='ARB'):
    """The outcome counts of one frame from its two-way shots.

    Parameters
    ----------
    halves : sequence of tuple
        One ``(backward end, forward end)`` per shot of the frame; an end is a
        state letter, the reactive state (uncommitted at the maximum length) or
        :data:`NOT_SIMULATED`.
    states : str, optional
        The sorted states: ``states[0]`` is the first end state, ``states[2]``
        the last (the committor is the probability of reaching it).

    Returns
    -------
    dict
        ``n_shots``, ``n_first_halves``, ``n_last_halves``,
        ``n_uncommitted_halves``, ``n_not_simulated_halves``,
        ``n_fully_committed_shots`` (both halves in an end state),
        ``n_transition_shots`` (of those, halves in different states),
        ``fraction_last`` (last-state share of the committed halves),
        ``fraction_last_fully_committed`` (the same over the fully committed
        shots), ``transition_share`` (transitions per fully committed shot),
        ``fraction_last_lower``/``_upper`` (every open half counted as the
        first/last state), ``transition_share_lower``/``_upper`` (every
        incomplete shot counted as no transition/a transition). NaN where
        undefined.
    """
    first, last = states[0], states[2]
    halves = [tuple(h) for h in halves]
    m = len(halves)
    ends = [end for shot in halves for end in shot]
    n_first, n_last = ends.count(first), ends.count(last)
    n_open = sum(1 for end in ends if end not in (first, last, NOT_SIMULATED))
    n_not = ends.count(NOT_SIMULATED)
    full = [s for s in halves if s[0] in (first, last) and s[1] in (first, last)]
    n_transition = sum(s[0] != s[1] for s in full)
    n_last_full = sum(end == last for s in full for end in s)
    nan = math.nan
    return {'n_shots': m, 'n_first_halves': n_first, 'n_last_halves': n_last,
            'n_uncommitted_halves': n_open, 'n_not_simulated_halves': n_not,
            'n_fully_committed_shots': len(full), 'n_transition_shots': n_transition,
            'fraction_last': n_last / (n_first + n_last) if n_first + n_last else nan,
            'fraction_last_fully_committed': n_last_full / (2 * len(full)) if full else nan,
            'transition_share': n_transition / len(full) if full else nan,
            'fraction_last_lower': n_last / (2 * m) if m else nan,
            'fraction_last_upper': (n_last + n_open + n_not) / (2 * m) if m else nan,
            'transition_share_lower': n_transition / m if m else nan,
            'transition_share_upper': (n_transition + m - len(full)) / m if m else nan}


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def _level_numbers(fraction, transition, fraction_full):
    p, T, p_full = _nanmean(fraction), _nanmean(transition), _nanmean(fraction_full)
    expected = 2 * p_full * (1 - p_full)
    return p, T, p_full, expected, 0.5 * (expected - T)


_LEVEL_KEYS = ('committor_mean', 'transition_share', 'committor_mean_fully_committed',
               'expected_transition_share', 'committor_variance')


def level_statistics(rows, *, n_boot=2000, confidence=0.95, seed=0, rng=None):
    """The committor mean, transition share and committor variance of the frames of one request.

    Parameters
    ----------
    rows : sequence of dict
        :func:`frame_outcomes` of the frames of one request that have an
        outcome (frames weighted equally).
    n_boot : int, optional
        Bootstrap resamples of the frames (0: no intervals).
    confidence : float, optional
        Width of the intervals.
    seed : int, optional
        Seed of the resampling (ignored when `rng` is given).
    rng : numpy.random.Generator, optional
        The random stream to draw from (several calls sharing one stream).

    Returns
    -------
    dict
        ``n_frames_with_outcome``; ``committor_mean``, ``transition_share``,
        ``committor_mean_fully_committed`` (the ``p`` of the variance),
        ``expected_transition_share`` (``2 p (1 - p)``) and
        ``committor_variance``, each with ``<key>_interval``;
        ``committor_mean_bounds`` (open halves counted as first or last state);
        ``transition_share_bounds`` (incomplete shots counted as no transition
        or as a transition); ``committor_sd_if_positive`` (None when the
        variance estimate is not positive); ``n_frames_with_committed_half``,
        ``n_frames_with_fully_committed_shot``. The module docstring explains
        the variance and its assumption.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    out = {'n_frames_with_outcome': len(rows)}
    if not len(rows):
        return out
    fraction = np.array([r['fraction_last'] for r in rows], dtype=float)
    transition = np.array([r['transition_share'] for r in rows], dtype=float)
    fraction_full = np.array([r['fraction_last_fully_committed'] for r in rows], dtype=float)
    draws = {key: [] for key in _LEVEL_KEYS}
    with np.errstate(all='ignore'):
        values = _level_numbers(fraction, transition, fraction_full)
        if n_boot and len(rows) > 1:
            for pick in rng.integers(0, len(rows), size=(int(n_boot), len(rows))):
                for key, value in zip(_LEVEL_KEYS, _level_numbers(fraction[pick], transition[pick],
                                                                  fraction_full[pick])):
                    draws[key].append(value)
    for key, value in zip(_LEVEL_KEYS, values):
        out[key] = value
        out[f'{key}_interval'] = (percentile_interval(draws[key], confidence) if draws[key]
                                  else [math.nan, math.nan])
    variance = out['committor_variance']
    out.update(
        committor_mean_bounds=[_nanmean([r['fraction_last_lower'] for r in rows]),
                               _nanmean([r['fraction_last_upper'] for r in rows])],
        transition_share_bounds=[_nanmean([r['transition_share_lower'] for r in rows]),
                                 _nanmean([r['transition_share_upper'] for r in rows])],
        committor_sd_if_positive=(float(np.sqrt(variance)) if np.isfinite(variance)
                                  and variance > 0 else None),
        n_frames_with_committed_half=int(np.isfinite(fraction).sum()),
        n_frames_with_fully_committed_shot=int(np.isfinite(transition).sum()))
    return out


def _level_of(row):
    return float(row['level'] if row.get('level') is not None else row['request'])


def level_summary(rows, *, levels=None, n_boot=2000, confidence=0.95, seed=0):
    """Per request: what happened to the samples, and :func:`level_statistics` of their outcomes.

    Parameters
    ----------
    rows : sequence of dict
        One per sample (:func:`validation_rows`, :func:`from_shot_table`): its
        ``level`` (the request it is grouped under), ``outcome_class``
        (``'shoot'``, ``'first_state_at_time_zero'``,
        ``'last_state_at_time_zero'`` or ``'failed_<reason>'``) and the
        :func:`frame_outcomes` keys when it has an outcome.
    levels : sequence of float, optional
        The requests to report (default: every level of `rows`).
    n_boot, confidence, seed : optional
        As for :func:`level_statistics` (one random stream for all levels).

    Returns
    -------
    list of dict
        Per level: ``level``, ``requested_committor`` (``sigmoid(level)``),
        ``n_samples``, ``n_failed_preparation``, ``failed_reasons``,
        ``n_first_state_at_time_zero``, ``n_last_state_at_time_zero``,
        ``n_shot`` (frames in the reactive region with a shot),
        ``n_without_shot`` (frames in the reactive region without one),
        ``halves`` (counts of first, last, uncommitted and unsimulated halves)
        and the keys of :func:`level_statistics`.
    """
    rng = np.random.default_rng(seed)
    if levels is None:
        levels = sorted({_level_of(r) for r in rows})
    out = []
    for level in [float(v) for v in levels]:
        mine = [r for r in rows if np.isclose(_level_of(r), level)]
        with_outcome = [r for r in mine if r.get('n_shots')]
        classes = [str(r.get('outcome_class', 'shoot')) for r in mine]
        entry = {'level': level, 'requested_committor': float(sigmoid(level)),
                 'n_samples': len(mine),
                 'n_failed_preparation': sum(c.startswith('failed') for c in classes),
                 'failed_reasons': sorted({str(r.get('reason')) for r, c in zip(mine, classes)
                                           if c.startswith('failed')}),
                 'n_first_state_at_time_zero': classes.count('first_state_at_time_zero'),
                 'n_last_state_at_time_zero': classes.count('last_state_at_time_zero'),
                 'n_shot': sum(c == 'shoot' and bool(r.get('n_shots'))
                               for r, c in zip(mine, classes)),
                 'n_without_shot': sum(c == 'shoot' and not r.get('n_shots')
                                       for r, c in zip(mine, classes)),
                 'halves': {key: sum(int(r.get(f'n_{key}_halves', 0)) for r in with_outcome)
                            for key in ('first', 'last', 'uncommitted', 'not_simulated')}}
        entry.update(level_statistics(with_outcome, n_boot=n_boot, confidence=confidence,
                                      rng=rng))
        out.append(entry)
    return out


# ════════════════════════════════════════════════════════════════════════════
# The dial: a logistic fit of the halves
# ════════════════════════════════════════════════════════════════════════════

def half_outcomes(rows, uncommitted='excluded'):
    """Every half of the samples with an outcome, as ``(request, outcome)``.

    Parameters
    ----------
    rows : sequence of dict
        Samples with ``request`` (or ``level``) and the :func:`frame_outcomes` keys.
    uncommitted : str, optional
        One of :data:`UNCOMMITTED_RULES`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``: the request of each half and its outcome, 1 for the last
        end state, 0 for the first, 0.5 for an open half counted at a request
        of exactly 0 by the rules toward or against the request.
    """
    if uncommitted not in UNCOMMITTED_RULES:
        raise ValueError(f'uncommitted must be one of {UNCOMMITTED_RULES}, got {uncommitted!r}')
    x, y = [], []
    for r in rows:
        if not r.get('n_shots'):
            continue
        request = float(r['request'] if r.get('request') is not None else r['level'])
        n_first, n_last = int(r['n_first_halves']), int(r['n_last_halves'])
        x += [request] * (n_first + n_last)
        y += [0.0] * n_first + [1.0] * n_last
        n_open = int(r['n_uncommitted_halves']) + int(r['n_not_simulated_halves'])
        if n_open and uncommitted != 'excluded':
            toward = 1.0 if request > 0 else 0.0 if request < 0 else 0.5
            value = {'as_first': 0.0, 'as_last': 1.0, 'toward_request': toward,
                     'against_request': 1.0 - toward}[uncommitted]
            x += [request] * n_open
            y += [value] * n_open
    return np.array(x, dtype=float), np.array(y, dtype=float)


def logistic_dial(x, y, *, ridge=1e-4):
    """``(a, b)`` of ``P(last | lambda*) = sigmoid(a + b lambda*)`` by maximum likelihood.

    Parameters
    ----------
    x : array_like
        Requests of the halves.
    y : array_like
        Outcomes in ``[0, 1]`` (fractional values count as fractional halves).
    ridge : float, optional
        A small penalty ``ridge (a^2 + b^2) / 2`` that keeps the fit finite
        for perfectly separated outcomes.

    Returns
    -------
    tuple
        ``(a, b)``; NaN when there are fewer than two halves, one request
        only, or every half ended in the same end state (then the outcomes
        say nothing about the slope; with the ridge alone a fit would return
        a number that only reflects the penalty).

    Notes
    -----
    The halves are summed per request into counts and fitted with
    :func:`aimmd.cgaimmd.analysis.validate.calibration_fit`, the same
    binomial maximum likelihood as the calibration line of a committor model.
    """
    from .validate import calibration_fit
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 2 or np.ptp(x) == 0:
        return math.nan, math.nan
    levels, inverse = np.unique(x, return_inverse=True)
    n_last = np.bincount(inverse, weights=y)
    n_first = np.bincount(inverse, weights=1.0 - y)
    a, b = calibration_fit(levels, n_first, n_last, ridge=ridge)
    return (math.nan, math.nan) if a is None else (a, b)


def resample_within_levels(rows, rng):
    """The samples with an outcome, resampled with replacement within each request level."""
    by_level = {}
    for r in rows:
        if r.get('n_shots'):
            by_level.setdefault(_level_of(r), []).append(r)
    return [group[i] for group in by_level.values()
            for i in rng.integers(0, len(group), len(group))]


def dial_slope(rows, *, n_boot=2000, confidence=0.95, seed=0, ridge=1e-4):
    """The dial slope ``b``: a logistic fit of the half outcomes on the request.

    Parameters
    ----------
    rows : sequence of dict
        Samples (:func:`validation_rows`, :func:`from_shot_table`).
    n_boot : int, optional
        Bootstrap resamples (samples resampled within requests).
    confidence : float, optional
        Width of the interval.
    seed : int, optional
        Seed of the resampling.
    ridge : float, optional
        As for :func:`logistic_dial`.

    Returns
    -------
    dict
        ``intercept`` ``a``, ``slope`` ``b`` and ``slope_interval``;
        ``n_committed_halves``; ``slope_with_open_halves`` (the slope with the
        uncommitted and unsimulated halves assigned by each rule of
        :data:`UNCOMMITTED_RULES`) and ``slope_bounds`` (the smallest and
        largest of these and ``b``); ``large_slope`` (True when ``|b| > 20``:
        the outcomes are nearly separated by the request and ``b`` is set by
        the ridge); ``single_outcome`` (True when every committed half ended
        in the same end state: then ``a`` and ``b`` are NaN); ``confidence``.
    """
    rng = np.random.default_rng(seed)
    x, y = half_outcomes(rows)
    a, b = logistic_dial(x, y, ridge=ridge)
    draws = [logistic_dial(*half_outcomes(resample_within_levels(rows, rng)), ridge=ridge)[1]
             for _ in range(int(n_boot))]
    with_open = {rule: logistic_dial(*half_outcomes(rows, rule), ridge=ridge)[1]
                 for rule in UNCOMMITTED_RULES[1:]}
    finite = [v for v in list(with_open.values()) + [b] if np.isfinite(v)]
    single_outcome = bool(len(y)) and bool(np.all(y == y[0])) and y[0] in (0.0, 1.0)
    return {'intercept': a, 'slope': b, 'single_outcome': single_outcome,
            'slope_interval': (percentile_interval(draws, confidence) if draws
                               else [math.nan, math.nan]),
            'confidence': float(confidence), 'n_committed_halves': int(len(x)),
            'slope_with_open_halves': with_open,
            'slope_bounds': [min(finite), max(finite)] if finite else [math.nan, math.nan],
            'large_slope': bool(np.isfinite(b) and abs(b) > 20)}


def _resample_ranks(rows, rng, picks=None):
    """Samples with an outcome resampled within levels by rank; `picks` reuses given ranks."""
    by_level = {}
    for r in rows:
        if r.get('n_shots'):
            by_level.setdefault(_level_of(r), {})[int(r['rank'])] = r
    out, used = [], {}
    for level, group in by_level.items():
        ranks = sorted(group)
        chosen = (picks[level] if picks is not None and level in picks
                  else [ranks[i] for i in rng.integers(0, len(ranks), len(ranks))])
        used[level] = chosen
        out += [group[k] for k in chosen if k in group]
    return out, used


def compare_dials(baseline_rows, candidate_rows, *, n_boot=5000, confidence=0.90, paired=False,
                  seed=0, ridge=1e-4):
    """The difference of two generators' dial slopes, ``b_candidate - b_baseline``, with its interval.

    Parameters
    ----------
    baseline_rows, candidate_rows : sequence of dict
        The samples of the two generators (with ``rank``, the sample's place
        among the samples of its request).
    n_boot : int, optional
        Bootstrap resamples.
    confidence : float, optional
        Width of the interval of the difference (default 0.90).
    paired : bool, optional
        Resample the same ranks for both generators (the same rank of a
        request had the same random seed for both); default: independent
        resamples.
    seed : int, optional
        Seed of the resampling.
    ridge : float, optional
        As for :func:`logistic_dial`.

    Returns
    -------
    dict
        ``delta_slope`` and ``delta_slope_interval``;
        ``fraction_of_resamples_above_zero``; ``candidate_slope`` and
        ``candidate_slope_interval``; ``baseline_slope``;
        ``delta_slope_bounds_open_halves`` (the open halves of each
        generator assigned toward and against the request in the directions
        that make the difference smallest and largest); ``bootstrap``
        (``'paired ranks'`` or ``'independent'``), ``n_boot``,
        ``confidence``.

    Notes
    -----
    An interval of the difference that excludes 0 resolves the difference.
    A candidate slope above 1 overshoots the requests: a larger slope is then
    not better adherence. Whether a difference matters is the reader's call.
    """
    rng = np.random.default_rng(seed)

    def slope(rows, rule='excluded'):
        return logistic_dial(*half_outcomes(rows, rule), ridge=ridge)[1]

    deltas, candidate = [], []
    for _ in range(int(n_boot)):
        base, picks = _resample_ranks(baseline_rows, rng)
        cand, _ = _resample_ranks(candidate_rows, rng, picks if paired else None)
        value = slope(cand)
        candidate.append(value)
        deltas.append(value - slope(base))
    b_base, b_cand = slope(baseline_rows), slope(candidate_rows)
    largest = slope(candidate_rows, 'toward_request') - slope(baseline_rows, 'against_request')
    smallest = slope(candidate_rows, 'against_request') - slope(baseline_rows, 'toward_request')
    deltas = np.asarray(deltas, dtype=float)
    return {'delta_slope': b_cand - b_base,
            'delta_slope_interval': (percentile_interval(deltas, confidence) if len(deltas)
                                     else [math.nan, math.nan]),
            'fraction_of_resamples_above_zero': (float(np.mean(deltas[np.isfinite(deltas)] > 0))
                                                 if np.isfinite(deltas).any() else math.nan),
            'candidate_slope': b_cand,
            'candidate_slope_interval': (percentile_interval(candidate, confidence) if candidate
                                         else [math.nan, math.nan]),
            'baseline_slope': b_base,
            'delta_slope_bounds_open_halves': [float(smallest), float(largest)],
            'bootstrap': 'paired ranks' if paired else 'independent', 'n_boot': int(n_boot),
            'confidence': float(confidence)}


def matched_ranks(baseline_rows, candidate_rows):
    """Both sets restricted, per request, to the sample ranks that both have an outcome for.

    Returns
    -------
    tuple
        ``(baseline rows, candidate rows, number of matched samples)``: a
        sensitivity check of :func:`compare_dials` when the generators lost
        different samples in preparation.
    """
    def has(rows):
        return {(_level_of(r), int(r['rank'])) for r in rows if r.get('n_shots')}
    keep = has(baseline_rows) & has(candidate_rows)
    return ([r for r in baseline_rows if (_level_of(r), int(r['rank'])) in keep],
            [r for r in candidate_rows if (_level_of(r), int(r['rank'])) in keep], len(keep))


# ════════════════════════════════════════════════════════════════════════════
# Reading a sweep
# ════════════════════════════════════════════════════════════════════════════

_SELECT = re.compile(r'Selecting shooting point for (\S+)/path(\d+) \((.+)\)\s*$')
_FRAME = re.compile(r'=== selecting frame (.+), (\d+) \(global coverage: total (\d+)')


def _launch_records(folder):
    """``{path number: launch}`` from a sweep folder's ``worker.log`` (the last selection before registration)."""
    out = {}
    log = os.path.join(folder, 'worker.log')
    if not os.path.exists(log):
        return out
    pending = None
    with open(log, errors='replace') as handle:
        for line in handle:
            match = _SELECT.search(line)
            if match:
                try:
                    when = datetime.datetime.strptime(' '.join(match.group(3).split()),
                                                      '%a %b %d %H:%M:%S %Y')
                except ValueError:
                    when = None
                pending = (int(match.group(2)), when)
                continue
            match = _FRAME.search(line)
            if match and pending is not None:
                out[pending[0]] = {'time': pending[1], 'frame': int(match.group(2)),
                                   'registered_at_launch': int(match.group(3))}
                pending = None
    return out


def _states_of(fname, states_function):
    """The cached states of a path file (read without a lock), else `states_function` on its frames."""
    from ...path.utils import get_cache_fname
    from ._trajectory import frames_reader, read_frames
    try:
        return np.asarray(np.load(get_cache_fname(fname, 'states'), allow_pickle=False)
                          ).astype(str).reshape(-1), 'cache'
    except (OSError, ValueError):
        pass
    if states_function is None:
        raise FileNotFoundError(f'{fname!r} has no states cache; pass the states function '
                                f'(params) to compute the states')
    positions, dimensions, _ = read_frames(fname)
    labels = states_function(frames_reader(positions, dimensions))
    return np.asarray(labels).astype(str).reshape(-1), 'states function'


def _sweep_tag(fname):
    """The sweep frame tagged on a shot (``aimmd.path.utils.write_sweep_frame``), read without a lock; None if untagged."""
    from ...path.utils import get_cache_fname
    try:
        return int(np.asarray(np.load(get_cache_fname(fname, 'sweep_frame'),
                                      allow_pickle=False)).reshape(-1)[0])
    except (OSError, ValueError, IndexError):
        return None


def _minimum_image_deviation(x, y, box):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if box is not None:
        lengths = np.asarray(box[:3], dtype=float)
        if np.all(lengths > 0) and np.allclose(box[3:6], 90.0):
            d = d - np.round(d / lengths) * lengths
    return float(np.abs(d).max())


def sweep_outcomes(sweep_dir, n_frames, *, params=None, states='ARB', trajectory_extension='.xtc',
                   frames=None, atoms=None, tolerance=0.02, round_shooting_index=False):
    """Every registered shot of an AIMMD sweep, attributed to the frame it was shot from.

    Parameters
    ----------
    sweep_dir : str or os.PathLike
        The sweep's run folder (it holds ``sweep{R}0``, ``sweep{R}1``, ...).
    n_frames : int
        The number of sweep frames (the frames of the sweep's initial paths).
    params : aimmd.Params, optional
        The sweep's params: their ``sorted_states``, ``trajectory_extension``,
        ``round_shooting_index`` and ``states_function`` (for paths without a
        states cache) are used.
    states : str, optional
        Without `params`: the sorted states.
    trajectory_extension : str, optional
        Without `params`: the trajectory extension.
    frames : str or sequence of str, optional
        The sweep frames as trajectory file(s), concatenated in the order the
        sweep read them (the files of its initial paths). When given, every
        shot's shooting frame is compared with the frame it is attributed to
        and an untagged shot is attributed by its coordinates.
    atoms : array_like of int, optional
        The atoms compared (default all).
    tolerance : float, optional
        Largest coordinate difference (Å, minimum image in a rectangular box)
        of a shooting frame and its sweep frame; XTC files store 0.01 Å.
    round_shooting_index : bool, optional
        Without `params`: how the shooting frame is found from the times
        (``Params.round_shooting_index``).

    Returns
    -------
    dict
        ``shots``: one dict per registered shot with ``path`` (relative to
        `sweep_dir`), ``folder``, ``number``, ``frame``, ``frame_source``
        (``'tag'``, ``'tag, coordinates agree'``, ``'coordinates'`` or
        ``'legacy rule'``), ``backward_end``, ``forward_end``
        (:data:`NOT_SIMULATED` when the path ends at its shooting frame),
        ``backward_frames``, ``forward_frames``, ``deviation`` (Å, with
        `frames`), ``launch_time`` and ``registered_at_launch`` (from the
        folder's ``worker.log``; None if not found), ``states_source``;
        ``folders``; ``in_flight`` (folders with a shot still running or cut
        off: a ``.current_frame.npy``, ``*back*`` or ``*forw*`` file);
        ``coordinates_checked``.

    Raises
    ------
    ValueError
        If a shooting frame is not in the reactive state, if a shot's
        coordinates match no sweep frame, or if its tag names another frame
        than its coordinates.

    Notes
    -----
    The frame of a shot is the tag AIMMD writes next to it
    (``path??????.xtc.sweep_frame.npy``); an untagged shot (registration
    interrupted between the rename and the tag) is attributed by its
    coordinates when `frames` is given, otherwise by AIMMD's old rule (the
    ``i``-th path of a folder belongs to frame ``i mod n_frames``), as
    :meth:`aimmd.PathEnsemble.shooting_results` does. Only reading: no
    trajectory index file is written next to the shots.
    """
    from ._trajectory import count_frames, read_frames, shooting_index_from_times
    states_function = None
    if params is not None:
        states = params.sorted_states
        trajectory_extension = params.trajectory_extension
        round_shooting_index = bool(getattr(params, 'round_shooting_index', False))
        states_function = getattr(params, 'states_function', None)
    first, reactive, last = states
    sweep_dir = os.fspath(sweep_dir)
    n_frames = int(n_frames)
    reference = None
    if frames is not None:
        files = [frames] if isinstance(frames, (str, os.PathLike)) else list(frames)
        parts = [read_frames(f) for f in files]
        reference = np.concatenate([p[0] for p in parts])
        if len(reference) != n_frames:
            raise ValueError(f'the sweep frames hold {len(reference)} frames, not {n_frames}')
    selection = None if atoms is None else np.asarray(atoms, dtype=int).reshape(-1)
    shots = []
    folders = sorted(f for f in glob.glob(os.path.join(glob.escape(sweep_dir), f'sweep{reactive}*'))
                     if os.path.isdir(f))
    for folder in folders:
        launches = _launch_records(folder)
        pattern = os.path.join(glob.escape(folder), f'path??????{trajectory_extension}')
        for i, fname in enumerate(sorted(glob.glob(pattern))):
            number = int(os.path.basename(fname)[4:10])
            labels, source = _states_of(fname, states_function)
            n = count_frames(fname)
            if n != len(labels):
                raise ValueError(f'{fname!r}: {n} frames but {len(labels)} states')
            _, _, times = read_frames(fname, [0, 1] if n > 1 else [0])
            shooting = shooting_index_from_times(times, n, nearest=round_shooting_index)
            if labels[shooting] != reactive:
                raise ValueError(f'{fname!r}: the shooting frame {shooting} is in '
                                 f'{labels[shooting]!r}, not in {reactive!r}')
            tag = _sweep_tag(fname)
            deviation = None
            if reference is not None:
                x, box, _ = read_frames(fname, [shooting])
                x = x[0] if selection is None else x[0][selection]
                box = None if box is None else box[0]
                ref = reference if selection is None else reference[:, selection]
                deviations = np.array([_minimum_image_deviation(x, r, box) for r in ref])
                best = int(np.argmin(deviations))
                deviation = float(deviations[best])
                if deviation > tolerance:
                    raise ValueError(f'{fname!r}: its shooting frame matches no sweep frame '
                                     f'(closest: frame {best}, {deviation:.3f} A)')
                if tag is not None and tag % n_frames != best:
                    raise ValueError(f'{fname!r} is tagged with frame {tag}, but its shooting '
                                     f'frame is frame {best} ({deviation:.3f} A)')
                frame = best
                frame_source = 'tag, coordinates agree' if tag is not None else 'coordinates'
            elif tag is not None:
                frame, frame_source = tag % n_frames, 'tag'
            else:
                frame, frame_source = i % n_frames, 'legacy rule'
            launch = launches.get(number)
            if launch is not None and launch['frame'] != frame:
                launch = None                       # a stale selection line (e.g. a retry)
            shots.append({
                'path': os.path.relpath(fname, sweep_dir), 'folder': os.path.basename(folder),
                'number': number, 'frame': int(frame), 'frame_source': frame_source,
                'backward_end': str(labels[0]),
                'forward_end': str(labels[-1]) if shooting < n - 1 else NOT_SIMULATED,
                'backward_frames': int(shooting + 1),
                'forward_frames': int(n - shooting) if shooting < n - 1 else 0,
                'deviation': deviation, 'states_source': source,
                'launch_time': (launch['time'].isoformat() if launch and launch['time']
                                else None),
                'registered_at_launch': launch['registered_at_launch'] if launch else None})
    in_flight = [os.path.basename(f) for f in folders
                 if os.path.exists(os.path.join(f, '.current_frame.npy'))
                 or glob.glob(os.path.join(glob.escape(f), '*back*'))
                 or glob.glob(os.path.join(glob.escape(f), '*forw*'))]
    return {'shots': shots, 'folders': [os.path.basename(f) for f in folders],
            'in_flight': in_flight, 'coordinates_checked': reference is not None,
            'states': states}


def _first_launched(shots):
    """The first launched of a frame's shots, and how that was decided."""
    if len(shots) == 1:
        return shots[0], 'single shot'
    if all(s['launch_time'] is not None for s in shots):
        ordered = sorted(shots, key=lambda s: (s['launch_time'], s['registered_at_launch'] or 0,
                                               s['folder'], s['number']))
        return ordered[0], 'first launched (worker.log)'
    ordered = sorted(shots, key=lambda s: (s['folder'], s['number']))
    return ordered[0], 'launch order unknown: lowest folder and path number'


def load_validation_table(fname):
    """Read the manifest of :func:`write_validation_frames` (a ``frames_table.json``)."""
    with open(os.fspath(fname)) as handle:
        return json.load(handle)


def validation_rows(table, sweep, *, first_launched=True):
    """One row per generated sample: its request, what preparation did, and its shot outcomes.

    Parameters
    ----------
    table : dict or str
        The manifest of :func:`write_validation_frames` (or its file).
    sweep : dict
        :func:`sweep_outcomes` of the sweep that shot the frames.
    first_launched : bool, optional
        Use only the first launched shot of every frame (default; the
        pre-registered one shot per sample: a sweep can shoot a frame twice at
        its end, when a worker starts another shot of a covered frame).
        False: every registered shot (a sensitivity check).

    Returns
    -------
    list of dict
        Per sample: the manifest's ``level``, ``request``, ``rank``,
        ``outcome_class``, ``reason``, ``frame``; ``n_registered_shots``,
        ``shot_rule`` and ``shots`` (the paths used); and the
        :func:`frame_outcomes` keys (a frame prepared in an end state counts
        as one shot with both halves there; a failed sample has
        ``n_shots`` 0).

    Raises
    ------
    ValueError
        If the sweep has shots of frames the table does not list.
    """
    if not isinstance(table, dict):
        table = load_validation_table(table)
    states = sweep.get('states', table.get('meta', {}).get('states', 'ARB'))
    by_frame = {}
    for shot in sweep['shots']:
        by_frame.setdefault(int(shot['frame']), []).append(shot)
    listed = {int(s['frame']) for s in table['samples'] if s.get('frame') is not None}
    orphans = sorted(set(by_frame) - listed)
    if orphans:
        raise ValueError(f'the sweep has shots of frames {orphans}, which the table does not '
                         f'list: the table does not describe this sweep')
    rows = []
    for sample in table['samples']:
        row = {key: sample.get(key) for key in ('level', 'request', 'rank', 'outcome_class',
                                                'reason', 'frame', 'lambda_raw',
                                                'lambda_prepared', 'seed')}
        oc = str(sample['outcome_class'])
        if oc in ('first_state_at_time_zero', 'last_state_at_time_zero'):
            end = states[0] if oc.startswith('first') else states[2]
            row.update(frame_outcomes([(end, end)], states=states), n_registered_shots=0,
                       shot_rule='committed at time zero', shots=[])
        elif oc == 'shoot':
            mine = by_frame.get(int(sample['frame']), [])
            if not mine:
                used, rule = [], 'no shot'
            elif first_launched:
                one, rule = _first_launched(mine)
                used = [one]
            else:
                used, rule = mine, 'all registered shots'
            row.update(frame_outcomes([(s['backward_end'], s['forward_end']) for s in used],
                                      states=states),
                       n_registered_shots=len(mine), shot_rule=rule,
                       shots=[s['path'] for s in used])
        else:
            row.update(n_shots=0, n_registered_shots=0, shot_rule='not shot', shots=[])
        rows.append(row)
    return rows


# ════════════════════════════════════════════════════════════════════════════
# The run's own shots
# ════════════════════════════════════════════════════════════════════════════

def from_shot_table(rows, *, states='ARB', level_edges=None):
    """The shots of a C-GAIMMD run as samples of a shooting validation.

    Parameters
    ----------
    rows : list of dict
        :func:`aimmd.cgaimmd.sidecar.shot_table` of the run.
    states : str, optional
        The run's sorted states (``params.sorted_states``).
    level_edges : array_like, optional
        Bin edges of the request: a shot is grouped under the middle of the
        bin its ``lambda_target`` falls in (default: bins of width 1
        centred on the integers, as the requests of a run are drawn from a
        continuous law). The dial fit uses the exact requests.
    Returns
    -------
    list of dict
        Per shot with a C-GAIMMD record (shots without one, RFPS shots of a
        continued run, have no request and are left out): ``request`` (its
        ``lambda_target``), ``level`` (the bin
        middle), ``rank`` (its place among the shots of its bin, in shot
        order), ``outcome_class`` ``'shoot'``, ``shot_index``, ``chain``,
        ``path``, ``acceptance_enforced`` and the :func:`frame_outcomes` of
        its one two-way shot (backward end ``first_state``, forward end
        ``last_state``).

    Notes
    -----
    Under a committor acceptance rule (``'target'`` or ``'window'``), once the
    network is trained, the run accepted a candidate only if the network put
    it near its request (the final check, ``acceptance_enforced`` True). Such
    shots measure generation and that check together, not the generator
    alone; select the rows with ``acceptance_enforced`` False (every shot of
    a run under the default rule ``'transition_region'``, which never filters
    by the committor), or validate with fresh frames
    (:func:`write_validation_frames`), for the generator alone.
    """
    out = []
    counts = {}
    for row in rows:
        target = row.get('lambda_target')
        if target is None:                      # no C-GAIMMD record: an RFPS shot, no request
            continue
        target = float(target)
        if level_edges is None:
            level = float(np.floor(target + 0.5))
        else:
            edges = np.asarray(level_edges, dtype=float)
            k = int(np.clip(np.searchsorted(edges, target, side='right') - 1, 0, len(edges) - 2))
            level = float(0.5 * (edges[k] + edges[k + 1]))
        rank = counts.get(level, 0)
        counts[level] = rank + 1
        first, last = row.get('first_state'), row.get('last_state')
        halves = [(first if first is not None else states[1],
                   last if last is not None else states[1])]
        entry = {'request': target, 'level': level, 'rank': rank, 'outcome_class': 'shoot',
                 'shot_index': row.get('shot_index'), 'chain': row.get('chain'),
                 'path': row.get('path'), 'acceptance_enforced': row.get('acceptance_enforced')}
        entry.update(frame_outcomes(halves, states=states))
        out.append(entry)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Writing the frames to validate
# ════════════════════════════════════════════════════════════════════════════

def _preparation_seed(seed, level, rank):
    """The seed of the preparation of one sample (its random stream in the hooks)."""
    return level_seed(seed, level) * 4096 + int(rank)


def write_validation_frames(params, out_dir, levels, n_per_level, *, network=None,
                            generator=None, seed=0, chunk=None, run_dir=None,
                            write_raw=True):
    """Generate samples at given requests, prepare them as the run would, and write the frames to shoot.

    Parameters
    ----------
    params : aimmd.Params
        The C-GAIMMD run's params: its generator (``cgaimmd_generator``), its
        validity, preparation and minimization hooks, its states,
        descriptors and values functions.
    out_dir : str or os.PathLike
        A new folder for the output (created; it must not be inside the run folder).
    levels : sequence of float
        The requests (committor logits).
    n_per_level : int
        Samples per request; all of them are kept, whatever happens to them.
    network : object, optional
        The network to generate with (default ``params.network``, with the
        weights it holds).
    generator : str, dict or Generator, optional
        The generation method (default ``params.cgaimmd_generator``, else the
        network's own default), see :func:`aimmd.cgaimmd.generation.make_generator`.
    seed : int, optional
        Seed: request ``l`` draws from ``level_seed(seed, l)``, the same for
        every generator, so that sample ranks pair across generators.
    chunk : int, optional
        Requests per generator call (default: the generator's ``chunk``
        setting, else all samples of a request at once).
    run_dir : str or os.PathLike, optional
        The run folder; `out_dir` must not be inside it (nor inside any other
        folder that holds AIMMD chain, free-simulation or sweep folders).
    write_raw : bool, optional
        Also write the raw samples (``raw_samples.npz``).

    Returns
    -------
    dict
        The manifest (also written to ``frames_table.json``): ``meta``
        (requests, samples per request, seed, generator method and
        settings, the frames file, its frame count and its sha256, the
        network's class) and ``samples``, one per sample in request and rank
        order, with ``level``, ``request``, ``rank``, ``seed`` (of its
        request), ``preparation_seed`` (of its preparation hooks), ``outcome_class`` (``'shoot'``: written to the frames file
        as frame ``frame``; ``'first_state_at_time_zero'`` or
        ``'last_state_at_time_zero'``: the prepared frame is already in an end
        state, not written; ``'failed_<reason>'``: validity or preparation
        failed), ``reason``, ``frame``, ``lambda_raw`` (the network's committor
        of the raw sample), ``lambda_prepared`` (of the prepared frame) and
        ``prepared_state``.

    Notes
    -----
    Each sample goes through what the shooting worker does with a candidate:
    ``params.cgaimmd_validity_function`` (its relabelled positions are
    used), then :func:`aimmd.cgaimmd.source.prepare_candidate` (preparation,
    minimization, the state check). The committor part of the final check is
    *not* applied: a validation measures the generator without that filter,
    so every prepared frame in the reactive region is written. Frames are
    written to ``shooting_frames<ext>`` (the run's trajectory extension) with
    their boxes; this file is the initial path of the sweep that shoots them.
    """
    import hashlib

    import torch

    from ...analysis.reweighting import check_output_outside
    from ..generation import check_generator, make_generator
    from ..protocol import coordinate_committor, n_atoms_of, network_device
    from ..source import candidate_context, prepare_candidate
    from ..types import ValidityResult

    out_dir = os.path.abspath(os.fspath(out_dir))
    check_output_outside(out_dir, run_dir)
    os.makedirs(out_dir, exist_ok=True)
    network = params.network if network is None else network
    spec = generator if generator is not None else getattr(params, 'cgaimmd_generator', None)
    gen = make_generator(spec, network)
    problems = check_generator(gen, network)
    if problems:
        raise ValueError(f'the generator {gen.method!r} cannot generate with this network: '
                         + '; '.join(problems))
    settings = dict(gen.settings())
    chunk = int(chunk or settings.get('chunk') or 0) or None
    n_per_level = int(n_per_level)
    levels = [float(v) for v in levels]
    m = n_atoms_of(network)
    device = network_device(network)
    states = params.sorted_states
    validity = getattr(params, 'cgaimmd_validity_function', None)
    ext = getattr(params, 'trajectory_extension', '.xtc') or '.xtc'
    frames_fname = os.path.join(out_dir, VALIDATION_FRAMES + ext)
    scratch = os.path.join(out_dir, 'preparation')
    raw = np.full((len(levels), n_per_level, m, 3), np.nan)
    samples, written = [], []
    was_training = getattr(network, 'training', False)
    network.eval()
    try:
        for li, level in enumerate(levels):
            stream = torch.Generator(device=device)
            stream.manual_seed(level_seed(seed, level))
            batch = chunk or n_per_level
            parts = []
            for start in range(0, n_per_level, batch):
                lam = torch.full((min(batch, n_per_level - start),), float(level),
                                 dtype=torch.float32, device=device)
                with torch.no_grad():
                    x = gen.generate(network, lam, generator=stream)
                parts.append(torch.as_tensor(x).detach().to('cpu', torch.float64).numpy())
            positions = np.concatenate(parts)
            if positions.shape != (n_per_level, m, 3):
                raise ValueError(f'the generator returned shape {positions.shape}, expected '
                                 f'({n_per_level}, {m}, 3)')
            raw[li] = positions
            folder = os.path.join(scratch, f'level{li:03d}')

            def context_of(k):
                """The candidate context of sample `k` (its own seed: preparation is reproducible)."""
                context = candidate_context(
                    params, folder, path_name=f'validation_level{li:03d}_rank{k:04d}',
                    seed=_preparation_seed(seed, level, k), network=network)
                # the state check only: no committor filter, whatever the run's rule
                context.settings['acceptance'] = 'transition_region'
                context.settings['raw_lambda_tolerance'] = None
                return context

            context = context_of(0)
            ok = np.ones(n_per_level, dtype=bool)
            reasons = np.full(n_per_level, '', dtype=object)
            if validity is not None:
                result = validity(positions.copy(), context)
                if not isinstance(result, ValidityResult):
                    raise TypeError('cgaimmd_validity_function must return a ValidityResult')
                positions = np.asarray(result.positions, dtype=float)
                ok = np.asarray(result.ok, dtype=bool).reshape(-1)
                reasons = np.asarray(result.reason, dtype=object).reshape(-1)
            with torch.no_grad():
                raw_lambda = coordinate_committor(network, positions).detach().cpu().numpy()
            for k in range(n_per_level):
                entry = {'level': level, 'request': level, 'rank': k,
                         'seed': level_seed(seed, level), 'outcome_class': None, 'reason': '',
                         'preparation_seed': _preparation_seed(seed, level, k), 'frame': None,
                         'lambda_raw': float(raw_lambda[k]), 'lambda_prepared': None,
                         'prepared_state': None}
                if not ok[k]:
                    reason = str(reasons[k]).split(':')[0] or 'invalid'
                    entry.update(outcome_class=f'failed_{reason}', reason=str(reasons[k]))
                    samples.append(entry)
                    continue
                prepared = prepare_candidate(positions[k], level, context_of(k))
                check = prepared.check or {}
                entry.update(lambda_prepared=check.get('lambda'),
                             prepared_state=check.get('state'))
                if prepared.ok:
                    frame = prepared.frame
                    entry.update(outcome_class='shoot', frame=len(written))
                    written.append((np.asarray(frame.positions, dtype=np.float32),
                                    None if frame.dimensions is None
                                    else np.asarray(frame.dimensions, dtype=np.float32)))
                elif check.get('state') in (states[0], states[2]):
                    which = 'first' if check['state'] == states[0] else 'last'
                    entry.update(outcome_class=f'{which}_state_at_time_zero',
                                 reason=prepared.reason)
                else:
                    entry.update(outcome_class=f'failed_{prepared.reason or "preparation"}',
                                 reason=prepared.reason)
                samples.append(entry)
    finally:
        if was_training:
            network.train(True)
    digest = None
    if written:
        from MDAnalysis import Universe, Writer
        n_system = len(written[0][0])
        if any(len(x) != n_system for x, _ in written):
            raise ValueError('the prepared frames differ in their number of atoms')
        universe = Universe.empty(n_system, trajectory=True)
        with Writer(frames_fname, n_atoms=n_system) as writer:
            for index, (x, box) in enumerate(written):
                ts = universe.trajectory.ts
                ts.positions = x
                ts.time = float(index)
                if box is not None:
                    ts.dimensions = box
                writer.write(universe.atoms)
        with open(frames_fname, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    if write_raw:
        np.savez_compressed(os.path.join(out_dir, 'raw_samples.npz'), levels=np.array(levels),
                            positions=raw.astype(np.float32))
    table = {'meta': {'levels': levels, 'n_per_level': n_per_level, 'seed': int(seed),
                      'level_seeds': [level_seed(seed, v) for v in levels],
                      'generator': {'method': gen.method, 'settings': settings},
                      'network': f'{type(network).__module__}.{type(network).__qualname__}',
                      'states': states, 'frames_file': frames_fname if written else None,
                      'n_frames': len(written), 'frames_sha256': digest,
                      'committor_window_applied': False,
                      'created': datetime.datetime.now().isoformat(timespec='seconds')},
             'samples': samples}
    table = jsonable(table)
    from ..io import atomic_write_json
    atomic_write_json(os.path.join(out_dir, VALIDATION_TABLE), table)
    return table
