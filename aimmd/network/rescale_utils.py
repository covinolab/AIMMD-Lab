"""
aimmd.network.rescale_utils
==========================

Rescaling utilities for committor-like coordinates.

This module contains small numerical helpers used to *re-map* the logit-committor `q`
in order to improve sampling uniformity in the reactive region. It does so by
making the rescaled crossing probabilities from both states as close as possible
to the theoretical `P1->2 = 1 / expit(+q)`, `P2-1 = 1 / expit(-q)`.

The two main operations are:

- :func:`find_knots_and_values`: infer a piecewise-linear remapping in logit space
  from per-side "extreme" statistics and cumulative crossing-probability measures.
- :func:`rescale`: apply that remapping to an array/tensor of `q` values
  (performed in-place).

Conventions
-----------
- `q`, `extremes1`, `extremes2`, `knots`, and `values` are in *logit-committor*
  units (i.e. real numbers where `expit(q)` maps to (0, 1)).
- Masks and interpolation are performed in logit space, but interpolation is
  carried out on `log(expit(...))` to keep monotonic behavior in probability
  space.

Notes
-----
- This module prints informative messages via `aimmd._config.print`.
- The logic is heuristic and intentionally conservative: several early returns
  produce empty knot/value sets when rescaling is not applicable.

"""

# external
import numpy as np
import torch
from math import inf
from torch import Tensor
from scipy.special import expit


def find_knots_and_values(extremes1, extremes2, xP1, xP2):
    """Infer rescaling knots and target values in logit space.

    This routine constructs a *nonlinear* coordinate transformation represented
    as a piecewise-linear map in logit space:

    - `knots`: x-positions (in the original logit coordinate) where slope changes,
    - `values`: y-positions (in the rescaled coordinate) at the corresponding knots.

    The transformation is later applied by :func:`rescale`.

    This transformation will make crossing probabilities from both states as close as
    possible to the theoretical `P1->2 = 1 / expit(+q)`, `P2-1 = 1 / expit(-q)`.
    In this way, the shooting point selection will be become more uniform in the
    reactive region, hopefully improving the sampling.

    Inputs represent statistics from two "sides" (1 and 2) of an ensemble.
    Conceptually, the function:

    1) normalizes the provided crossing-probability series (`xP1`, `xP2`)
       into codomain curves (`N1`, `N2`) in probability space;
    2) determines the domain of action (`kmin`, `kmax`) from finite extremes;
    3) interpolates both sides onto a fine grid `q` in logit space (101 points);
    4) estimates a transition-state shift `ts` and an overall rescaling factor `r`;
    5) generates a set of non-unique candidate knot locations and associated
       target values by matching geometric levels between the “actual” and
       “theoretical” curves;
    6) removes redundant / non-monotone knots.

    Parameters
    ----------
    extremes1 : array-like of float
        Logit-coordinate "extreme" values for side 1. May contain `math.inf` values
        (treated as state points and removed before the action domain is computed).
        Converted with ``np.asarray(extremes1)``.
    extremes2 : array-like of float
        Logit-coordinate "extreme" values for side 2. Same conventions as
        `extremes1`. Converted with ``np.asarray(extremes2)``.
    xP1 : array-like of float
        Cumulative or aggregated crossing-probability measure for side 1.
        Must be indexable; if empty, the function falls back to a trivial
        normalization for that side. Converted with ``np.asarray(xP1)``.
    xP2 : array-like of float
        Crossing-probability measure for side 2. Same conventions as `xP1`.
        Converted with ``np.asarray(xP2)``.

    Returns
    -------
    knots : numpy.ndarray, dtype=float
        Knot positions (x-coordinates) in the *original* logit coordinate.
        May be empty if no rescaling is applicable.
    values : numpy.ndarray, dtype=float
        Target positions (y-coordinates) in the *rescaled* logit coordinate,
        aligned with `knots`. Same length as `knots`.

    Notes
    -----
    - Several “nothing to do” checks return empty arrays:
      - codomain interval invalid (``vmin >= vmax``),
      - action domain invalid (``kmin >= kmax``).
    - `np.isinf(extremes*)` entries are removed before computing the domain of
      action and interpolation.
    - The printed diagnostics include the estimated transition-state shift `ts`
      and total rescaling factor `r`.

    """
    # normalize inputs
    extremes1 = np.asarray(extremes1)
    extremes2 = np.asarray(extremes2)
    xP1 = np.asarray(xP1)
    xP2 = np.asarray(xP2)

    # process crossing probability
    # codomain of action
    if len(xP1):
        # N1 is normalized so that N1[-1] * expit(extremes1)[-1] matches xP1[-1]
        N1 = xP1 / (xP1[-1] * expit(extremes1)[-1])
    else:
        # trivial fallback (also empties extremes1 to avoid downstream min/max issues)
        N1 = np.ones(1)
        extremes1 = np.zeros(0)
    if len(xP2):
        N2 = xP2 / (xP2[-1] * expit(extremes2)[-1])
    else:
        N2 = np.ones(1)
        extremes2 = np.zeros(0)

    # codomain bounds derived from the first values of both sides
    N10 = N1[0]
    N20 = N2[0]
    vmax = N10
    vmin = 4 / N20

    # nothing to do here
    if vmin >= vmax:
        return np.array([]), np.array([])

    # remove values at states (infinite extremes)
    mask = ~np.isinf(extremes1)
    N1 = N1[mask]
    extremes1 = extremes1[mask]
    mask = ~np.isinf(extremes2)
    N2 = N2[mask]
    extremes2 = extremes2[mask]

    # domain of action (intersection of available finite ranges)
    kmin = min(+np.min(extremes1, initial=0.), -np.max(extremes2, initial=0.))
    kmax = min(-np.min(extremes2, initial=0.), +np.max(extremes1, initial=0.))

    # nothing to do here
    if kmin >= kmax:
        return np.array([]), np.array([])

    # turn into fine grained interpolation in logit committor space
    # x axis
    q = np.linspace(kmin, kmax, 101)

    # interpolate side 1 in log(expit(.)) space to keep probability-space monotonicity
    x = extremes1[::+1]
    y = N1[::+1]
    N1 = np.exp(np.interp(np.log(expit(q)), np.log(expit(x)), np.log(y)))

    # interpolate side 2 on mirrored axis (-extremes2), reversing for increasing order
    x = -extremes2[::-1]
    y = N2[::-1]
    N2 = np.exp(np.interp(np.log(expit(q)), np.log(expit(x)), np.log(y)))

    # TS shift and rescaling computation
    ts = 0.  # initialization
    r = 1.
    from_1_wins = N1 >= N2
    from_2_wins = N1 <  N2
    if N10 > 2 and N20 > 2 and\
        np.sum(from_1_wins) and np.sum(from_2_wins):
        # bracket the crossing region of the two curves and shift TS to its midpoint
        ts = (q[from_1_wins][-1] + q[from_2_wins][0]) / 2
        # choose r so that the winning side at TS maps to ~2
        r = 2. / N1[from_1_wins][-1]
    elif N10 > 2 and np.sum(from_1_wins):
        # fall back: pick q where N1/2 is closest to 1
        ts = q[np.argmin(np.abs(N1 / 2 - 1.))]
    elif N20 > 2 and np.sum(from_2_wins):
        # fall back: pick q where N2/2 is closest to 1, with conservative clipping
        ts = np.clip(q[np.argmin(np.abs(N2 / 2 - 1.))], -5., 5.)
    print(f'*** transition state shift by {ts:.3f}, '
          f'total xP rescaling by {r:.3f}')

    # theoretical line (piecewise definition around TS)
    y = np.zeros(len(q))
    y[q <= ts] = 1 / expit(q[q <= ts] - ts)
    y[q >  ts] = 4 * (1 - expit(q[q > ts] - ts))

    # actual line (apply r symmetrically to the two sides)
    y0 = np.append(r * N1[q <= ts], 4 / N2[q > ts] / r)

    # determine number of knots / values
    vmin /= r
    vmax *= r
    drop = max(1, N10 * N20 * r ** 2)
    n = min(round(np.log(drop)), 100)
    if not n:
        print(f'!!! rescaling is not possible (yet)')
    else:
        print(f'*** generating {n} (non unique) knots')

    # fill knot/value candidates by matching geometric levels between y0 and y
    knots = np.zeros(n)
    values = np.zeros(n)
    for i, v in enumerate(
        np.geomspace(vmin, vmax, n + 2)[::-1][1:-1]):
        knots[i] = q[np.argmin(np.abs(v - y0))]
        values[i] = q[np.argmin(np.abs(v - y))]

    # remove redundancies in knots (keep first occurrence)
    knots, indices = np.unique(knots, return_index=True)
    values = np.array(values)[indices]

    # remove non-growing or even decreasing knots (enforce monotone values)
    if len(values) > 1:
        keepers = np.diff(values) > 0
        values = np.append(values[0], values[1:][keepers])
        knots = np.append(knots[0], knots[1:][keepers])

    return knots, values


def rescale(q, knots, values):
    """Apply a piecewise-linear rescaling to `q` (in-place).

    The rescaling is defined by matching `knots` (x-coordinates in the original
    space) to `values` (y-coordinates in the rescaled space). Between knots the
    mapping is linear, and the end segments extrapolate using the slope of the
    first/last interior segment.

    This function mutates `q` in-place and returns it for convenience.

    Parameters
    ----------
    q : numpy.ndarray or torch.Tensor
        1D (or broadcastable) array/tensor of values in logit space to be
        transformed. If `q` is a `torch.Tensor`, bucketization uses
        `torch.bucketize`; otherwise NumPy digitization is used.
    knots : array-like of float
        Knot positions (x-coordinates) in the original space. Typically produced
        by :func:`find_knots_and_values`.
    values : array-like of float
        Target positions (y-coordinates) corresponding to `knots`. Must have the
        same length as `knots`.

    Returns
    -------
    q : numpy.ndarray or torch.Tensor
        The same object as the input `q`, modified in-place.

    Notes
    -----
    - If ``len(knots) < 1``, the function returns `q` unchanged.
    - If ``len(knots) == 1``, the mapping is a pure shift about `knots[0]`:
      ``q[:] = (q - knots[0]) + values[0]``.
    - For ``len(knots) >= 2``, each bucket gets its own affine transform
      ``a * (q - x0) + b`` computed from adjacent knot/value pairs.

    """
    # number of knots
    I = len(knots)
    if I < 1:
        return q
    if I < 2:
        # single-knot special case: shift about x0
        x0 = knots[0]
        a = 1
        b = values[0]
        q[:] = a * (q - x0) + b
        return q

    # assign each q to a segment index
    if isinstance(q, Tensor):
        indices = torch.bucketize(q, knots)
    else:
        q = np.asarray(q)
        indices = np.digitize(q, knots)

    # apply piecewise affine mapping per segment
    for i in range(I + 1):
        # choose the neighboring knot pair to define slope on this segment
        if 0 < i < I:
            j = i
        elif i == 0:
            j = 1
        elif i == I:
            j = I - 1
        x0 = knots[max(i - 1, 0)]
        a = ((values[j] - values[j - 1]) /
             (knots[j] - knots[j - 1]))
        b = values[max(i - 1, 0)]
        mask = indices == i
        q[mask] = a * (q[mask] - x0) + b
    return q
