"""
aimmd.cgaimmd.analysis.adherence
================================

How closely a committor-conditioned generator follows its requests, and what
its samples look like next to molecular dynamics (MD).

A C-GAIMMD generator is asked for configurations at a requested committor,
given as a logit ``lambda* = ln(p_B / p_A)`` (the request). Its samples are
then *judged*: each gets a committor logit ``lambda`` from some model, the
generator's own committor head, an independent committor model, or a shooting
estimate. The functions here turn the pairs ``(lambda*, lambda)`` into numbers
that say how good the generator is as a *dial*, and compare the samples'
structures with MD frames. They take arrays and callables only, so they work
for any system and any generator; they decide nothing: thresholds are
arguments with documented defaults, and what counts as good enough is left to
the caller.

The dial
--------
- **Resolution** (the lead number). Fit ``lambda = a + b lambda*`` by least
  squares; ``b`` is the *slope* (1 for a perfect dial, 0 for a generator that
  ignores the request). The residual spread, measured robustly
  (:func:`robust spread <aimmd.cgaimmd.analysis._stats.robust_sd>`: 1.4826
  times the median absolute deviation, the standard deviation for normal
  residuals), divided by the slope, is the resolution in request units: how far
  apart two requests must be before their samples can be told apart. Smaller
  is better; a flat dial (slope at or below 1e-3) has an infinite resolution.
  It is reported over all requests, inside the *band* ``|lambda*| <= 2`` (the
  transition region) and over the *wings* ``|lambda*| > 2``. The wings get one
  shared slope and a separate intercept per wing, so that a jump between the
  low and the high wing does not count as slope.
- **Median slopes per region.** The least-squares slope of the per-request
  *medians* of ``lambda`` against ``lambda*``, over the *core* ``-1..+1``, the
  band ``-2..+2`` and each wing. A generator can have a steep band slope only
  because its samples jump at ``+-2`` while its core is flat; the core slope
  shows that.
- **Neighbour AUC.** For two neighbouring requests, the probability that a
  sample of the higher request is judged above a sample of the lower one (ties
  count half; the Mann-Whitney statistic). 0.5: the two requests cannot be told
  apart; 1: they never overlap.
- **Large misses.** The share of samples judged more than 3 logits away from
  their request.

Every number above comes with a bootstrap interval in which the samples are
resampled within each request (:func:`adherence_metrics`).

The samples next to MD
----------------------
Distances between structures are distances between *feature vectors*: the
root mean square of the feature differences. With pair distances between
atoms as features (:func:`pair_distance_features`) that is the distance RMS
(dRMS), which needs no alignment. Every function takes features, so any
other structural description works too.

- **Variety.** At one request, the median distance between two samples,
  divided by the same median for independent MD frames whose committor label
  lies within ``+-0.5`` of the request (:func:`label_binned_reference`). About
  1: the samples are as varied as MD at that committor; well below 1: the
  generator collapses onto a few structures.
- **Near-copies.** The share of samples whose nearest training frame is closer
  than the typical distance between two consecutive MD frames
  (:func:`consecutive_frame_threshold`): such a sample is, for practical
  purposes, a copy of a training frame. Compared with a baseline generator's
  share, since even a good generator lands near its training data sometimes.
- **Departure.** The median distance of the samples to their nearest training
  frame, divided by the same for the label-binned MD frames, then the median
  over requests. About 1: the samples sit as close to the training data as
  fresh MD does; well above 1: the samples leave the region MD visits.
- **Structural match.** At one request, the Wasserstein-1 distance (the mean
  shift needed to turn one distribution into the other) between a structural
  collective variable of the samples and of the label-binned MD frames
  (:func:`structural_match`). 0: the same distribution.
- **Data ceiling.** Training frames drawn at their own label (within
  ``+-0.25`` of each request, :func:`data_ceiling`) and judged by the same
  judge: what a generator that reproduced its training data perfectly would
  score. Every adherence number should be read against it.

Read the judged committor of your own head with care: a generator can
follow its own head and still miss the true committor. An independent judge
(a committor model that shares no weights with the generator) or a shooting
validation (:mod:`aimmd.cgaimmd.analysis.shooting_validation`) is the test.
When samples of two methods are compared structurally, compare them at
matched committor (by their own model and by shooting outcome), not along a
collective variable alone.

numpy only at import time; scipy is imported inside :func:`structural_match`.
"""

import math

import numpy as np

from ._stats import level_groups, percentile_interval, robust_sd, spearman

__all__ = ['TRANSITION_BAND', 'REGIONS', 'LARGE_MISS', 'affine_fit', 'neighbour_auc',
           'wing_resolution', 'median_slopes', 'adherence_metrics', 'structural_match',
           'pair_distance_features', 'feature_distances', 'pairwise_feature_distances',
           'nearest_distances', 'consecutive_frame_threshold', 'label_binned_reference',
           'sample_quality', 'data_ceiling']

TRANSITION_BAND = 2.0
"""Requests with ``|lambda*| <=`` this value form the band (the transition region)."""

REGIONS = {'core': (-1.0, 1.0), 'band': (-2.0, 2.0), 'low_wing': (-5.0, -2.0),
           'high_wing': (2.0, 5.0)}
"""The request regions of :func:`median_slopes` (closed intervals of ``lambda*``)."""

LARGE_MISS = 3.0
"""A sample judged further than this many logits from its request is a large miss."""

FLAT_SLOPE = 1e-3
"""A fitted slope at or below this value counts as a flat dial (infinite resolution)."""


# ════════════════════════════════════════════════════════════════════════════
# The dial
# ════════════════════════════════════════════════════════════════════════════

def _pairs(requested, judged):
    r = np.asarray(requested, dtype=float).reshape(-1)
    j = np.asarray(judged, dtype=float).reshape(-1)
    if r.shape != j.shape:
        raise ValueError(f'{len(r)} requests for {len(j)} judged values')
    if not np.all(np.isfinite(r)):
        raise ValueError('the requests must be finite')
    return r, j


def neighbour_auc(lower, higher):
    """The probability that a value of `higher` lies above a value of `lower` (ties count half).

    Parameters
    ----------
    lower, higher : array_like
        Judged values of the samples of two requests, the lower request first.

    Returns
    -------
    float
        The Mann-Whitney statistic divided by ``len(lower) * len(higher)``:
        0.5 when the two sets cannot be told apart, 1 when every value of
        `higher` lies above every value of `lower`; NaN if a set is empty.
    """
    lower = np.asarray(lower, dtype=float).reshape(-1)
    higher = np.asarray(higher, dtype=float).reshape(-1)
    if not len(lower) or not len(higher):
        return math.nan
    above = (higher[:, None] > lower[None, :]).sum()
    ties = (higher[:, None] == lower[None, :]).sum()
    return float((above + 0.5 * ties) / (len(lower) * len(higher)))


def affine_fit(requested, judged, *, large_miss=LARGE_MISS):
    """The least-squares line of judged on requested committor logits, and the spread around it.

    Parameters
    ----------
    requested, judged : array_like
        ``(n,)`` requested logits ``lambda*`` and judged logits ``lambda``
        (finite; :func:`adherence_metrics` drops non-finite judged values).
    large_miss : float, optional
        The large-miss distance in logits.

    Returns
    -------
    dict
        ``n``; ``slope`` ``b`` and ``offset_at_zero`` ``a`` of
        ``lambda = a + b lambda*``; ``residual_robust_sd`` and
        ``residual_sd`` of the residuals; ``resolution``
        (``residual_robust_sd / slope``, infinite for a slope at or below
        :data:`FLAT_SLOPE`); ``variance_explained`` (``1 - var(residuals) /
        var(lambda)``); ``neighbour_auc`` (one value per pair of neighbouring
        requests, :func:`neighbour_auc`) and ``neighbour_auc_mean``;
        ``spearman`` (rank correlation of request and judged value);
        ``large_miss_share`` (``|lambda - lambda*| > large_miss``),
        ``large_residual_share`` (residual beyond ``large_miss``),
        ``within_one_logit`` (``|lambda - lambda*| <= 1``) and ``median_miss``
        (median of ``lambda - lambda*``). With fewer than two distinct
        requests the line is undefined: slope and resolution are NaN.
    """
    r, j = _pairs(requested, judged)
    out = {'n': int(len(r)), 'slope': math.nan, 'offset_at_zero': math.nan,
           'residual_robust_sd': math.nan, 'residual_sd': math.nan, 'resolution': math.nan,
           'variance_explained': math.nan, 'neighbour_auc': [], 'neighbour_auc_mean': math.nan,
           'spearman': None, 'large_miss_share': math.nan, 'large_residual_share': math.nan,
           'within_one_logit': math.nan, 'median_miss': math.nan}
    if not len(r):
        return out
    miss = j - r
    out.update(large_miss_share=float(np.mean(np.abs(miss) > large_miss)),
               within_one_logit=float(np.mean(np.abs(miss) <= 1.0)),
               median_miss=float(np.median(miss)))
    levels = np.unique(r)
    if len(levels) < 2:
        return out
    design = np.column_stack([np.ones(len(r)), r])
    coef, *_ = np.linalg.lstsq(design, j, rcond=None)
    residual = j - design @ coef
    slope = float(coef[1])
    spread = robust_sd(residual)
    neighbours = [neighbour_auc(j[r == lo], j[r == hi]) for lo, hi in zip(levels[:-1], levels[1:])]
    out.update(slope=slope, offset_at_zero=float(coef[0]), residual_robust_sd=spread,
               residual_sd=float(np.std(residual)),
               resolution=float(spread / slope) if slope > FLAT_SLOPE else math.inf,
               variance_explained=(float(1 - np.var(residual) / np.var(j)) if np.var(j) > 0
                                   else math.nan),
               neighbour_auc=[float(v) for v in neighbours],
               neighbour_auc_mean=float(np.nanmean(neighbours)),
               spearman=spearman(r, j),
               large_residual_share=float(np.mean(np.abs(residual) > large_miss)))
    return out


def wing_resolution(requested, judged, *, band=TRANSITION_BAND):
    """The resolution over the wings ``|lambda*| > band``, without rewarding the gap between them.

    Parameters
    ----------
    requested, judged : array_like
        ``(n,)`` requested and judged logits (finite); requests inside the band are ignored.
    band : float, optional
        Half-width of the band.

    Returns
    -------
    dict or None
        ``n``, ``slope`` (shared by both wings), ``intercepts`` (low wing,
        high wing) and ``resolution`` (robust residual spread / slope;
        infinite for a flat dial); None without requests outside the band.

    Notes
    -----
    One line per wing with a shared slope: ``lambda = c_low + b lambda*``
    below ``-band`` and ``c_high + b lambda*`` above ``+band``. A generator
    that puts every low request at -5 and every high request at +5 has two
    flat wings: its shared slope is 0 and its wing resolution infinite,
    although one line through both wings would have a steep slope.
    """
    r, j = _pairs(requested, judged)
    outside = np.abs(r) > float(band)
    r, j = r[outside], j[outside]
    if not len(r):
        return None
    low = (r < 0).astype(float)
    design = np.column_stack([low, 1.0 - low, r])
    coef, *_ = np.linalg.lstsq(design, j, rcond=None)
    slope = float(coef[2])
    spread = robust_sd(j - design @ coef)
    resolution = (math.nan if len(np.unique(r)) < 2 else
                  float(spread / slope) if slope > FLAT_SLOPE else math.inf)
    return {'n': int(len(r)), 'slope': slope,
            'intercepts': [float(coef[0]), float(coef[1])], 'resolution': resolution}


def _region_slopes(medians, regions):
    levels = np.array(sorted(medians))
    out = {}
    for name, (lo, hi) in regions.items():
        sel = [level for level in levels if lo <= level <= hi]
        out[name] = (float(np.polyfit(sel, [medians[level] for level in sel], 1)[0])
                     if len(sel) >= 2 else math.nan)
    return out


def median_slopes(requested, judged, *, regions=None, n_boot=500, confidence=0.95, seed=0):
    """The slope of the per-request medians over regions of requests.

    Parameters
    ----------
    requested, judged : array_like
        ``(n,)`` requested and judged logits (finite).
    regions : dict, optional
        ``{name: (low, high)}`` closed intervals of ``lambda*`` (default
        :data:`REGIONS`: core ``-1..+1``, band ``-2..+2`` and the two wings).
    n_boot : int, optional
        Bootstrap resamples (samples resampled within each request; 0: no intervals).
    confidence : float, optional
        Width of the bootstrap intervals.
    seed : int, optional
        Seed of the resampling.

    Returns
    -------
    dict
        ``{name: {'slope', 'interval', 'requests'}}``: the least-squares slope
        of the medians against the requests in the region (NaN with fewer than
        two requests there), its bootstrap interval and the requests used.
    """
    regions = dict(REGIONS if regions is None else regions)
    r, j = _pairs(requested, judged)
    groups = level_groups(r)
    estimate = _region_slopes({level: float(np.median(j[idx])) for level, idx in groups}, regions)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        boots.append(_region_slopes(
            {level: float(np.median(j[rng.choice(idx, len(idx))])) for level, idx in groups},
            regions))
    out = {}
    for name, (lo, hi) in regions.items():
        out[name] = {'slope': estimate[name],
                     'interval': (percentile_interval([b[name] for b in boots], confidence)
                                  if boots else [math.nan, math.nan]),
                     'requests': [level for level, _ in groups if lo <= level <= hi]}
    return out


def _lead(r, j, band, large_miss):
    """The numbers that get bootstrap intervals."""
    full = affine_fit(r, j, large_miss=large_miss)
    inside = np.abs(r) <= band
    in_band = affine_fit(r[inside], j[inside], large_miss=large_miss) if inside.sum() > 2 else None
    wings = wing_resolution(r, j, band=band)
    return full, in_band, wings, {
        'resolution_all': full['resolution'],
        'resolution_band': in_band['resolution'] if in_band else math.nan,
        'resolution_wings': wings['resolution'] if wings else math.nan,
        'neighbour_auc_all': full['neighbour_auc_mean'],
        'neighbour_auc_band': in_band['neighbour_auc_mean'] if in_band else math.nan,
        'large_miss_share': full['large_miss_share']}


def adherence_metrics(requested, judged, *, band=TRANSITION_BAND, regions=None,
                      large_miss=LARGE_MISS, n_boot=200, confidence=0.95, seed=0):
    """How well judged committors of generated samples follow the requests.

    Parameters
    ----------
    requested : array_like
        ``(n,)`` requested committor logits ``lambda*`` of the samples
        (usually a few distinct levels, many samples each).
    judged : array_like
        ``(n,)`` committor logits of the samples by some judge. Non-finite
        values are left out of every number and counted in ``n_missing``.
    band : float, optional
        Requests with ``|lambda*| <= band`` form the band (default 2).
    regions : dict, optional
        Regions of :func:`median_slopes` (default :data:`REGIONS`).
    large_miss : float, optional
        The large-miss distance in logits (default 3).
    n_boot : int, optional
        Bootstrap resamples for the intervals (samples resampled within each
        request); 0 gives no intervals.
    confidence : float, optional
        Width of the intervals (default 0.95).
    seed : int, optional
        Seed of the resampling.

    Returns
    -------
    dict
        - ``n``, ``n_missing``;
        - ``all_requests``: :func:`affine_fit` over every request;
        - ``band``: the same inside the band (None with three samples or fewer there);
        - ``wings``: :func:`wing_resolution` (None without requests outside the band);
        - ``median_slopes``: :func:`median_slopes`;
        - ``per_request``: per request ``request``, ``n``, ``median``,
          ``q25``, ``q75``, ``iqr``, ``robust_sd``, ``large_miss_share``;
        - ``median_request_iqr``: the median over requests of the
          interquartile range;
        - ``intervals``: bootstrap intervals of ``resolution_all``,
          ``resolution_band``, ``resolution_wings``, ``neighbour_auc_all``,
          ``neighbour_auc_band`` and ``large_miss_share``;
        - ``settings``.

    Notes
    -----
    The module docstring defines every number. A bootstrap resample in which
    a resolution is infinite keeps that value, so an interval can reach
    infinity.
    """
    r, j = _pairs(requested, judged)
    finite = np.isfinite(j)
    n_missing = int((~finite).sum())
    r, j = r[finite], j[finite]
    band = float(band)
    regions = dict(REGIONS if regions is None else regions)
    full, in_band, wings, lead = _lead(r, j, band, large_miss)
    per = []
    for level, idx in level_groups(r):
        v = j[idx]
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        per.append({'request': level, 'n': int(len(v)), 'median': float(q50), 'q25': float(q25),
                    'q75': float(q75), 'iqr': float(q75 - q25), 'robust_sd': robust_sd(v),
                    'large_miss_share': float(np.mean(np.abs(v - level) > large_miss))})
    rng = np.random.default_rng(seed)
    groups = [idx for _, idx in level_groups(r)]
    draws = {key: [] for key in lead}
    for _ in range(int(n_boot)):
        pick = np.concatenate([rng.choice(idx, len(idx)) for idx in groups]) if groups else []
        values = _lead(r[pick], j[pick], band, large_miss)[3]
        for key, value in values.items():
            draws[key].append(value)
    intervals = {key: (percentile_interval(values, confidence) if values
                       else [math.nan, math.nan]) for key, values in draws.items()}
    return {'n': int(len(r)), 'n_missing': n_missing, 'all_requests': full, 'band': in_band,
            'wings': wings,
            'median_slopes': median_slopes(r, j, regions=regions, n_boot=n_boot,
                                           confidence=confidence, seed=seed),
            'per_request': per,
            'median_request_iqr': float(np.median([p['iqr'] for p in per])) if per else math.nan,
            'intervals': intervals,
            'settings': {'band': band, 'regions': {k: list(v) for k, v in regions.items()},
                         'large_miss': float(large_miss), 'n_boot': int(n_boot),
                         'confidence': float(confidence), 'seed': int(seed)}}


def structural_match(requested, sample_cv, reference_labels, reference_cv, *, half_width=0.5,
                     min_reference=20, band=TRANSITION_BAND):
    """Whether the samples at a request look like MD frames at that committor, along one CV.

    Parameters
    ----------
    requested : array_like
        ``(n,)`` requested logits of the samples.
    sample_cv : array_like
        ``(n,)`` a structural collective variable of each sample (e.g. the
        fraction of native contacts).
    reference_labels : array_like
        ``(m,)`` committor logits of independent MD frames (labelled by a
        committor model).
    reference_cv : array_like
        ``(m,)`` the same collective variable of those frames.
    half_width : float, optional
        MD frames with ``|label - lambda*| <= half_width`` are the reference of a request.
    min_reference : int, optional
        Fewer reference frames than this at a request give NaN there.
    band : float, optional
        Requests with ``|lambda*| <= band`` enter ``median_band``.

    Returns
    -------
    dict
        ``per_request`` (``{request: Wasserstein-1 distance}``, NaN where the
        reference is too small), ``n_reference`` (``{request: frames}``),
        ``median_all`` and ``median_band`` (medians over requests, NaN
        ignored), ``settings``.

    Notes
    -----
    The Wasserstein-1 distance between two one-dimensional distributions is
    the mean distance each value has to move to turn one into the other, in
    the units of the CV: 0 for identical distributions, the shift for two
    copies of one distribution shifted against each other.
    """
    from scipy.stats import wasserstein_distance
    r = np.asarray(requested, dtype=float).reshape(-1)
    cv = np.asarray(sample_cv, dtype=float).reshape(-1)
    labels = np.asarray(reference_labels, dtype=float).reshape(-1)
    ref_cv = np.asarray(reference_cv, dtype=float).reshape(-1)
    if r.shape != cv.shape or labels.shape != ref_cv.shape:
        raise ValueError('requested and sample_cv, and reference_labels and reference_cv, '
                         'must have the same length')
    per, counts = {}, {}
    for level, idx in level_groups(r):
        values = cv[idx][np.isfinite(cv[idx])]
        reference = ref_cv[np.abs(labels - level) <= half_width]
        reference = reference[np.isfinite(reference)]
        counts[level] = int(len(reference))
        per[level] = (float(wasserstein_distance(values, reference))
                      if len(reference) >= int(min_reference) and len(values) else math.nan)
    values = np.array(list(per.values()), dtype=float)
    in_band = np.array([v for k, v in per.items() if abs(k) <= band], dtype=float)
    median = (lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else math.nan)
    return {'per_request': per, 'n_reference': counts, 'median_all': median(values),
            'median_band': median(in_band),
            'settings': {'half_width': float(half_width), 'min_reference': int(min_reference),
                         'band': float(band)}}


# ════════════════════════════════════════════════════════════════════════════
# Distances between structures
# ════════════════════════════════════════════════════════════════════════════

def pair_distance_features(atom_indices):
    """A feature map: the distances between all pairs of the given atoms.

    Parameters
    ----------
    atom_indices : array_like of int
        The atoms (indices into the configurations' atoms), e.g. the heavy atoms.

    Returns
    -------
    callable
        ``features(positions (n, M, 3)) -> (n, P)`` float32 distances,
        ``P = k (k - 1) / 2`` for ``k`` atoms, pairs ``i < j`` in the order of
        `atom_indices`. The distance between two feature vectors
        (:func:`feature_distances`) is then the distance RMS (dRMS), which
        needs no alignment of the structures.
    """
    idx = np.asarray(atom_indices, dtype=int).reshape(-1)
    if len(idx) < 2:
        raise ValueError('pair distances need at least two atoms')
    first, second = np.triu_indices(len(idx), 1)
    i, k = idx[first], idx[second]

    def features(positions):
        x = np.asarray(positions, dtype=np.float32)
        if x.ndim == 2:
            x = x.reshape(len(x), -1, 3)
        return np.linalg.norm(x[:, i] - x[:, k], axis=-1)

    features.atom_indices = idx
    return features


def _as_features(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(len(x), -1)


def feature_distances(a, b):
    """Row-wise distance of two sets of feature vectors: the RMS of the feature differences.

    Parameters
    ----------
    a, b : array_like
        ``(n, P)`` features (the same rows are compared; ``(P,)`` broadcasts).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` ``sqrt(mean((a - b)^2))``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def _squared_distance_block(a, b):
    """``(len(a), len(b))`` mean squared feature differences (clipped at 0)."""
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0) / a.shape[1]


def pairwise_feature_distances(features):
    """``(n, n)`` RMS feature differences between every pair of rows of `features`."""
    f = _as_features(features)
    d = np.sqrt(_squared_distance_block(f, f))
    np.fill_diagonal(d, 0.0)
    return d


def nearest_distances(features, reference_features, *, chunk=2048):
    """The nearest reference row to every row of `features`, by RMS feature difference.

    Parameters
    ----------
    features : array_like
        ``(n, P)`` features of the structures to place (samples, MD frames).
    reference_features : array_like
        ``(m, P)`` features of the reference structures (training frames).
    chunk : int, optional
        Reference rows per block (memory: ``n x chunk`` doubles).

    Returns
    -------
    tuple
        ``(distance (n,), index (n,))``: the distance to the nearest
        reference row and its index.
    """
    f = _as_features(features)
    ref = _as_features(reference_features)
    if len(f) and not len(ref):
        raise ValueError('no reference rows to compare with')
    if len(f) and f.shape[1] != ref.shape[1]:
        raise ValueError(f'features of width {f.shape[1]} and reference of width {ref.shape[1]}')
    best = np.full(len(f), np.inf)
    index = np.full(len(f), -1, dtype=int)
    for start in range(0, len(ref), int(chunk)):
        block = _squared_distance_block(f, ref[start:start + int(chunk)])
        where = block.argmin(1)
        value = block[np.arange(len(f)), where]
        better = value < best
        best[better] = value[better]
        index[better] = where[better] + start
    return np.sqrt(best), index


def consecutive_frame_threshold(features_t, features_next):
    """The near-copy line: the median distance between MD frames and their successors.

    Parameters
    ----------
    features_t, features_next : array_like
        ``(n, P)`` features of MD frames and of the frame that follows each
        (e.g. 5 ps later in the same trajectory).

    Returns
    -------
    float
        The median of :func:`feature_distances` of the pairs. A generated
        sample closer than this to a training frame is as close to it as the
        next frame of the same trajectory would be: a near-copy.
    """
    return float(np.median(feature_distances(_as_features(features_t),
                                              _as_features(features_next))))


def _median_pair_distance(matrix, pick=None):
    """Median of the distinct pairs of a distance matrix (rows `pick`, pairs of one row with itself left out)."""
    n = len(matrix) if pick is None else len(pick)
    first, second = np.triu_indices(n, 1)
    if pick is None:
        values = matrix[first, second]
    else:
        keep = pick[first] != pick[second]
        values = matrix[pick[first[keep]], pick[second[keep]]]
    return float(np.median(values)) if len(values) else math.nan


def _median_pair_distance_interval(matrix, n_boot, confidence, rng):
    loc = np.arange(len(matrix))
    draws = [_median_pair_distance(matrix, rng.choice(loc, len(loc))) for _ in range(int(n_boot))]
    return percentile_interval(draws, confidence) if draws else [math.nan, math.nan]


def label_binned_reference(labels, features, levels, *, half_width=0.5, n_max=200,
                           min_frames=10, training_features=None, near_copy_threshold=None,
                           n_boot=200, confidence=0.95, seed=0):
    """Independent MD frames binned by their committor label: the yardstick of variety and departure.

    Parameters
    ----------
    labels : array_like
        ``(m,)`` committor logits of independent MD frames (frames no
        generator was trained on), from the same committor model that labels
        the training data.
    features : array_like
        ``(m, P)`` their features.
    levels : sequence of float
        The requests.
    half_width : float, optional
        Frames with ``|label - level| <= half_width`` belong to a level.
    n_max : int, optional
        At most this many frames per level are used (drawn at random).
    min_frames : int, optional
        A level with fewer frames in its window gets only ``n_in_window``.
    training_features : array_like, optional
        ``(t, P)`` features of the training frames, for the nearest-frame distances.
    near_copy_threshold : float, optional
        The near-copy line (:func:`consecutive_frame_threshold`); with
        `training_features` the share of reference frames below it is reported.
    n_boot : int, optional
        Bootstrap resamples of the median pair distance.
    confidence : float, optional
        Width of its interval.
    seed : int, optional
        Seed of the frame draw and the resampling.

    Returns
    -------
    dict
        ``{level: entry}``; every entry has ``n_in_window``; levels with at
        least `min_frames` frames also ``indices`` (the rows used),
        ``median_pair_distance`` and ``median_pair_distance_interval``, and
        with `training_features` ``nearest_training_median`` and
        ``near_copy_share`` (None without a threshold).
    """
    labels = np.asarray(labels, dtype=float).reshape(-1)
    f = _as_features(features)
    if len(f) != len(labels):
        raise ValueError(f'{len(labels)} labels for {len(f)} feature rows')
    rng = np.random.default_rng(seed)
    out = {}
    for level in [float(v) for v in levels]:
        window = np.flatnonzero(np.abs(labels - level) <= half_width)
        entry = {'n_in_window': int(len(window))}
        if len(window) >= int(min_frames):
            take = np.sort(rng.choice(window, min(int(n_max), len(window)), replace=False))
            matrix = pairwise_feature_distances(f[take])
            entry.update(indices=take, median_pair_distance=_median_pair_distance(matrix),
                         median_pair_distance_interval=_median_pair_distance_interval(
                             matrix, n_boot, confidence, rng))
            if training_features is not None:
                nearest, _ = nearest_distances(f[take], training_features)
                entry['nearest_training_median'] = float(np.median(nearest))
                entry['near_copy_share'] = (None if near_copy_threshold is None else
                                            float(np.mean(nearest < near_copy_threshold)))
        out[level] = entry
    return out


def sample_quality(requested, sample_features, reference, *, training_features=None,
                   near_copy_threshold=None, baseline_near_copy_share=None,
                   variety_threshold=0.8, near_copy_margin=0.05, departure_threshold=1.25,
                   n_boot=200, confidence=0.95, seed=0):
    """Variety, near-copies and departure of generated samples against label-binned MD.

    Parameters
    ----------
    requested : array_like
        ``(n,)`` requested logits of the samples.
    sample_features : array_like
        ``(n, P)`` features of the samples (non-finite rows are left out).
    reference : dict
        :func:`label_binned_reference` at the requests (with the same features).
    training_features : array_like, optional
        ``(t, P)`` features of the generator's training frames (needed for
        near-copies and departure).
    near_copy_threshold : float, optional
        The near-copy line (:func:`consecutive_frame_threshold`).
    baseline_near_copy_share : float, optional
        The near-copy share of a baseline generator, for comparison.
    variety_threshold : float, optional
        A level whose whole variety interval lies below this ratio is listed
        in ``variety_interval_below`` (default 0.8: samples clearly less
        varied than MD at that committor).
    near_copy_margin : float, optional
        ``near_copy_line`` is the baseline share plus this margin (default 0.05).
    departure_threshold : float, optional
        Recorded with ``departure_ratio`` for the reader (default 1.25).
    n_boot : int, optional
        Bootstrap resamples (samples resampled within each request).
    confidence : float, optional
        Width of the intervals.
    seed : int, optional
        Seed of the resampling.

    Returns
    -------
    dict
        - ``per_request``: per request ``n``, ``variety`` (median pair
          distance of the samples), ``variety_ratio`` and its
          ``variety_ratio_interval`` (against the reference's median pair
          distance; None where the reference has none), and with training
          features ``nearest_training_median``, ``departure_ratio`` (against
          the reference's ``nearest_training_median``) and
          ``near_copy_share``;
        - ``variety_ratio_lowest`` and ``variety_interval_below`` (the
          requests whose whole interval lies below `variety_threshold`);
        - ``near_copy_share`` (all samples), its ``near_copy_share_interval``,
          ``baseline_near_copy_share``, ``near_copy_line`` (baseline +
          margin, None without a baseline) and ``near_copy_excess`` (share
          minus baseline);
        - ``departure_ratio``: the median over requests of the per-request ratio;
        - ``settings``: the thresholds and the resampling.

    Notes
    -----
    The numbers are measurements: whether a generator is acceptable is the
    reader's decision, with the thresholds recorded next to them.
    """
    r = np.asarray(requested, dtype=float).reshape(-1)
    f = _as_features(sample_features)
    if len(f) != len(r):
        raise ValueError(f'{len(r)} requests for {len(f)} feature rows')
    finite = np.all(np.isfinite(f), axis=1)
    rng = np.random.default_rng(seed)
    nearest = np.full(len(f), np.nan)
    if training_features is not None and finite.any():
        nearest[finite], _ = nearest_distances(f[finite], training_features)
    per = {}
    for level, idx in level_groups(r):
        idx = idx[finite[idx]]
        ref = reference.get(level) or reference.get(float(level)) or {}
        entry = {'n': int(len(idx)), 'variety': math.nan, 'variety_ratio': None,
                 'variety_ratio_interval': None}
        if len(idx) >= 2:
            matrix = pairwise_feature_distances(f[idx])
            entry['variety'] = _median_pair_distance(matrix)
            scale = ref.get('median_pair_distance')
            if scale:
                interval = _median_pair_distance_interval(matrix, n_boot, confidence, rng)
                entry['variety_ratio'] = entry['variety'] / scale
                entry['variety_ratio_interval'] = [interval[0] / scale, interval[1] / scale]
        if training_features is not None and len(idx):
            entry['nearest_training_median'] = float(np.median(nearest[idx]))
            scale = ref.get('nearest_training_median')
            entry['departure_ratio'] = (entry['nearest_training_median'] / scale if scale
                                        else None)
            entry['near_copy_share'] = (None if near_copy_threshold is None else
                                        float(np.mean(nearest[idx] < near_copy_threshold)))
        per[level] = entry
    ratios = {k: v['variety_ratio'] for k, v in per.items() if v['variety_ratio'] is not None}
    below = [k for k, v in per.items() if v['variety_ratio_interval'] is not None
             and v['variety_ratio_interval'][1] < variety_threshold]
    out = {'per_request': per,
           'variety_ratio_lowest': min(ratios.values()) if ratios else None,
           'variety_ratio_lowest_at': min(ratios, key=ratios.get) if ratios else None,
           'variety_interval_below': below,
           'near_copy_share': None, 'near_copy_share_interval': None,
           'baseline_near_copy_share': baseline_near_copy_share, 'near_copy_line': None,
           'near_copy_excess': None, 'departure_ratio': None,
           'settings': {'variety_threshold': float(variety_threshold),
                        'near_copy_threshold': near_copy_threshold,
                        'near_copy_margin': float(near_copy_margin),
                        'departure_threshold': float(departure_threshold),
                        'n_boot': int(n_boot), 'confidence': float(confidence),
                        'seed': int(seed)}}
    if training_features is not None and finite.any():
        departures = [v['departure_ratio'] for v in per.values()
                      if v.get('departure_ratio') is not None]
        out['departure_ratio'] = float(np.median(departures)) if departures else None
        if near_copy_threshold is not None:
            copies = nearest < near_copy_threshold
            groups = [idx[finite[idx]] for _, idx in level_groups(r)]
            groups = [g for g in groups if len(g)]
            share = float(np.mean(copies[finite]))
            draws = [float(np.mean(np.concatenate([copies[rng.choice(g, len(g))]
                                                   for g in groups])))
                     for _ in range(int(n_boot))]
            out.update(near_copy_share=share,
                       near_copy_share_interval=(percentile_interval(draws, confidence)
                                                 if draws else None))
            if baseline_near_copy_share is not None:
                out['near_copy_line'] = float(baseline_near_copy_share) + float(near_copy_margin)
                out['near_copy_excess'] = share - float(baseline_near_copy_share)
    return out


def data_ceiling(labels, levels, *, half_width=0.25, n_per_level=64, valid=None, seed=0):
    """Training frames at their own label: the samples of a perfect generator.

    Parameters
    ----------
    labels : array_like
        ``(m,)`` committor labels (logits) of the training frames, as the
        generator was trained on them.
    levels : sequence of float
        The requests.
    half_width : float, optional
        A frame with ``|label - level| <= half_width`` stands for a sample at that level.
    n_per_level : int, optional
        Frames drawn per level (fewer where the window holds fewer).
    valid : array_like of bool, optional
        ``(m,)`` frames that may be drawn (e.g. leaving out held-out chains).
    seed : int, optional
        Seed of the draw.

    Returns
    -------
    dict
        ``indices`` (the drawn frames, per level sorted), ``requested`` (the
        level each frame stands for), ``labels`` (their labels) and
        ``n_in_window`` (``{level: frames available}``).

    Notes
    -----
    Judge the drawn frames with the same judge as the samples and pass
    ``(requested, judged)`` to :func:`adherence_metrics`: a generator that
    reproduced the conditional distribution of its training data would score
    about this, so it is the line to read a generator's numbers against.
    """
    labels = np.asarray(labels, dtype=float).reshape(-1)
    ok = np.isfinite(labels)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).reshape(-1)
    rng = np.random.default_rng(seed)
    indices, requested, picked, available = [], [], [], {}
    for level in [float(v) for v in levels]:
        window = np.flatnonzero(ok & (np.abs(labels - level) <= half_width))
        available[level] = int(len(window))
        take = (np.sort(rng.choice(window, min(int(n_per_level), len(window)), replace=False))
                if len(window) else np.zeros(0, dtype=int))
        indices.append(take)
        requested.append(np.full(len(take), level))
        picked.append(labels[take])
    return {'indices': np.concatenate(indices).astype(int) if indices else np.zeros(0, int),
            'requested': np.concatenate(requested) if requested else np.zeros(0),
            'labels': np.concatenate(picked) if picked else np.zeros(0),
            'n_in_window': available}
