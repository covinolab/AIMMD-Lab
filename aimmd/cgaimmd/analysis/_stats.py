"""Small numerical helpers shared by the analysis modules of C-GAIMMD.

Private: numpy only, no state. The public modules document what they compute.
"""

import math
import os

import numpy as np


def sigmoid(x):
    """``1 / (1 + exp(-x))`` without overflow, as float64."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def logit(p):
    """``log(p / (1 - p))``."""
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def json_float(value):
    """A JSON float: None for None, NaN and infinities."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def jsonable(obj):
    """Numpy scalars and arrays, NaN and paths turned into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return json_float(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def ranks(x):
    """Average ranks, 1-based (ties share the mean of their positions)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='mergesort')
    out = np.empty(len(x))
    out[order] = np.arange(1, len(x) + 1)
    _, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=out)
    return (sums / np.bincount(inverse))[inverse]


def spearman(x, y):
    """Spearman's rank correlation of ``x`` and ``y`` (None when either is constant or n < 2)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def robust_sd(x):
    """``1.4826 * median(|x - median(x)|)``: the standard deviation of a normal sample, insensitive to outliers."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if not len(x):
        return math.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def percentile_interval(values, confidence):
    """The central `confidence` interval of `values`, NaN left out (``[nan, nan]`` without any).

    Infinite values take part: an interval reaches infinity when more than
    its tail of the values is infinite (an infinite resolution, for example).
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[~np.isnan(values)]
    if not len(values):
        return [math.nan, math.nan]
    finite = values[np.isfinite(values)]
    top = finite.max() if len(finite) else 0.0
    bottom = finite.min() if len(finite) else 0.0
    tail = 50.0 * (1.0 - float(confidence))
    bounds = np.percentile(np.clip(values, bottom - 1.0, top + 1.0), [tail, 100.0 - tail])
    return [math.inf if b > top else -math.inf if b < bottom else float(b) for b in bounds]


def level_groups(levels):
    """``[(level, indices)]`` of the distinct values of `levels`, sorted."""
    levels = np.asarray(levels, dtype=float).reshape(-1)
    return [(float(level), np.flatnonzero(levels == level)) for level in np.unique(levels)]


def level_seed(seed, level):
    """The random seed of the samples at one request: the same for every generator at that request.

    ``seed + 1000 (round(2 level) + 100)``: requests on a half-integer grid
    get distinct seeds, and a request keeps its seed whatever other requests
    are drawn with it, so that sample ranks pair across generators.
    """
    return int(seed) + 1000 * (int(round(2 * float(level))) + 100)
