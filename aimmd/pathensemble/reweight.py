"""
aimmd.pathensemble.reweight
===========================

Statistical core for path-ensemble reweighting.

This module provides the low-level numerical routines used by
:class:`aimmd.pathensemble._reweight.PathEnsembleReweight`. The functions here
implement the statistical pieces of AIMMD reweighting:

- local shooting-point density estimation,
- optional local normalization ("uniformization") of correction factors,
- construction of a crossing-probability curve as a function of a path extreme,
- computation of excursion weights from (shooting value, extreme, factor) data.

These functions are written to be called on *arrays* collected from an ensemble.
They do not depend on PathEnsemble directly and perform no I/O.

Conventions
-----------
AIMMD uses special sentinel values in the arrays passed to these functions:

- ``shooting_value == -inf``
  Marks an equilibrium/free excursion with no shooting-point correction.
  These contribute to the "equilibrium" part of the reweighting and to the
  initial part of the crossing-probability curve.

- ``extreme == +inf``
  Marks a trajectory that certainly crossed (or is treated as such). This is
  used to make the final normalization stable.

The meaning of "shooting value" and "extreme" is defined upstream in
:meth:`aimmd.pathensemble._reweight.PathEnsembleReweight.reweight`. In
particular, AIMMD uses a signed convention so that "progress toward the product"
corresponds to increasing extreme values in both directions.

Notes
-----
- All computations here are purely numerical and operate on 1D arrays.
- The returned weights are normalized such that transitions have weight ~1
  (see :func:`reweight_excursions`).
"""

# external
import numpy as np
from math import inf, nan
from scipy.special import expit


def compute_shooting_density(values, shooting_value, neighbors=10):
    """
    Estimate the local path density at the shooting value.

    This function estimates a 1D density around `shooting_value` using a simple
    nearest-neighbor window. It is used to correct the path weight at the shooting
    interface, to restore detailed balance: paths with high density are downweighted
    and viceversa.

    The estimator is:

        n* / Δ

    where:
    - n* is the number of neighbors retained in a window around `shooting_value`,
    - Δ is the window width, defined by the midpoints to the next values outside
      the window.

    Parameters
    ----------
    values : array-like
        Per-frame scalar values for a single path. The function converts to
        float and removes NaN and inf entries. The remaining values are treated
        as the support of the 1D distribution along that path.
    shooting_value : float
        Value at the shooting point. If NaN or inf, the density is undefined and
        the function returns +inf (so the caller typically assigns factor 0 or
        leaves the factor unchanged depending on usage).
    neighbors : int, default=10
        Target number of nearest neighbors retained in the window. The algorithm
        shrinks [begin, end) until the window contains at most `neighbors` values.

    Returns
    -------
    float
        Estimated local density. Returns +inf when the estimate is undefined
        (insufficient data, zero window width, or invalid shooting value).

    Notes
    -----
    The code forces the first and last value in the sorted list to be the global
    min/max of the path values. This stabilizes boundary handling when the
    shooting value lies near the extremes of the sampled range.
    """
    
    # process values
    values = values.astype(float)
    values = values[(~np.isinf(values)) * (~np.isnan(values))]
    if (len(values) < 3 or
        np.isinf(shooting_value) or
        np.isnan(shooting_value)):
        return inf
    
    # min and max value
    min_value = np.min(values)
    max_value = np.max(values)
    values = np.concatenate(
        [[min_value], np.sort(values[1:-1]), [max_value]])
    differences = np.abs(shooting_value - values)
    begin = 0
    end = len(values)
    while end - begin > neighbors:
        if differences[begin] > differences[end - 1]:
            begin += 1
        else:
            end -= 1
    
    # computation and handling of special case
    upper_boundary = (
        values[end - 1] + values[min(end, len(values) - 1)]) / 2
    lower_boundary = (
        values[max(begin - 1, 0)] + values[begin]) / 2
    delta = upper_boundary - lower_boundary
    if not delta:
        return inf
    n_of_neighbors = end - begin
    return n_of_neighbors / delta


def compute_crossing_probability(
    shooting_values, extremes,
    free_extremes=None,
    free_extremes_factors=None,
    free_threshold=0,
    theoretical_threshold=None,
    crossing_probability_cutoff=0.):
    """
    Construct a crossing-probability curve as a function of the extreme value.

    This function estimates a monotonically non-increasing "survival" / crossing
    probability curve xP(E) defined on a set of extreme values E. It is used by
    :func:`reweight_excursions` to map each excursion extreme to a probability
    factor.

    The algorithm combines:
    - an "equilibrium/free" part derived from `free_extremes` and their factors,
    - a "drop" recursion for the remaining (shot) data, applied after sorting by
      the extreme value.

    Parameters
    ----------
    shooting_values : array-like
        Per-trajectory shooting values for the (shot) dataset that will be used
        in the "drop" recursion. These are filtered by the threshold logic below.
    extremes : array-like
        Per-trajectory extremes corresponding to `shooting_values`. The returned
        crossing probability is computed on these extreme values (and on the
        free extremes if provided).
    free_extremes : array-like, optional
        Extreme values for free excursions (equilibrium-like samples). These
        dominate the low-extreme part of the curve up to a threshold.
    free_extremes_factors : array-like, optional
        Per-free-trajectory factors (weights) used when computing the equilibrium
        part. If omitted, all free factors are 1.
    free_threshold : int, default=0
        If ``len(free_extremes) >= free_threshold > 0``, define the threshold
        extreme as ``free_extremes[-free_threshold]`` after sorting. Extremes
        below this value are treated as purely equilibrium/free.
        Otherwise, fall back to ``extremes.min()`` (or +inf if no data).
    theoretical_threshold : float, optional
        If provided, once the sorted extreme exceeds this threshold the curve
        is continued using a theoretical approximation based on ``expit(E)``.
        If None, this is set to +inf (no theoretical continuation).
    crossing_probability_cutoff : float, default=0.
        Robustness cutoff for the "drop" recursion. At each step j, only
        trajectories with ``shooting_value < extreme[j] - cutoff`` contribute to
        the normalization. Increasing this cutoff discards near-tangent shots
        (more robust, but uses less data).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        extremes_out : numpy.ndarray
            Sorted extreme values on which the curve is defined. This includes
            the included free extremes (below threshold) concatenated with the
            remaining extremes.
        xP : numpy.ndarray
            Crossing probability values aligned with `extremes_out`.

    Notes
    -----
    The "drop" recursion is a heuristic that enforces monotonic decay. It
    propagates xP forward in sorted extreme order by applying a multiplicative
    ratio derived from the fraction of remaining weight that can still "drop"
    before reaching the current extreme.
    """

    shooting_values = np.asarray(shooting_values)
    extremes = np.asarray(extremes)
    
    # free data (sort)
    if free_extremes is not None:
        free_extremes = np.asarray(free_extremes)
        order = np.argsort(free_extremes)
        free_extremes = free_extremes[order]
    else:
        free_extremes = np.array([])
    
    # free factors
    if free_extremes_factors is not None:
        factors = np.asarray(free_extremes_factors)[order]
    else:
        factors = np.ones(len(free_extremes))
    
    # get threshold: up until which only equilibrium counts
    if len(free_extremes) >= free_threshold > 0:
        threshold = free_extremes[-free_threshold]
    elif len(extremes):
        threshold = extremes.min()
    else:
        threshold = inf
    
    # get upper threshold: from which only theoretical counts
    if theoretical_threshold is None:
        theoretical_threshold = + inf
    
    # completely "free" part of crossing probability
    n = np.count_nonzero(free_extremes < threshold)
    if n:
        xP_free = factors.sum() - np.cumsum([0.] + list(factors[:n]))
        xP_free /= xP_free[0]
        i = n - 1
    else:
        xP_free = np.array([])
        threshold = -inf
        i = 0
    
    # join the rest
    extremes = extremes[shooting_values >= threshold]
    shooting_values = shooting_values[shooting_values >= threshold]
    factors = factors[n:]
    shooting_values = np.append(np.full(len(factors), -inf), shooting_values)
    factors = np.append(factors, np.ones(len(extremes)))
    extremes = np.append(free_extremes[n:], extremes)    
    
    # do you need to go on?
    if not len(extremes):
        return free_extremes, xP_free[:-1]
    
    # sort
    order = np.argsort(extremes)
    extremes = extremes[order]
    shooting_values = shooting_values[order]
    factors = factors[order]
    xP = np.ones(len(order))
    
    # compute the rest of the crossing probability with the "drop" method     
    for j in range(len(extremes) - 1):

        # theoretical part
        if extremes[j] >= theoretical_threshold:
            xP[j + 1:] = xP[j] * expit(extremes[j]) / expit(extremes[j + 1:])
            break

        # the rest
        mask = shooting_values[j:] < (
            extremes[j] - crossing_probability_cutoff)
        norm = factors[j:][mask].sum()
        if norm:
            if mask[0]:
                ratio = max(1 - factors[j] / norm, 0.5)
            else:
                ratio = 1.
        else:
            ratio = .5
        xP[j + 1] = xP[j] * ratio
    
    # combine
    if len(xP_free):
        xP = np.append(xP_free, xP[1:] * xP_free[-1])
        extremes = np.append(free_extremes[:n], extremes)
    
    return extremes, xP


def uniformize_factors(factors, shooting_values, cutoff=1., norm=10):
    """
    Locally normalize factors as a function of the shooting value.

    This function reduces large factor fluctuations by rescaling each factor by
    the mean factor of nearby shooting values.

    Parameters
    ----------
    factors : array-like
        Per-trajectory correction factors (typically density corrections).
    shooting_values : array-like
        Per-trajectory shooting values associated with `factors`.
    cutoff : float, default=1.
        Width of the local window in shooting-value space. The initial mask is:

            (sv > shooting_value - cutoff/2) & (sv < shooting_value + cutoff/2)

    norm : int, default=10
        Minimum number of neighbors required for the local window. If fewer than
        `norm` values fall in the cutoff window, the `norm` closest shooting
        values are used instead.

    Returns
    -------
    numpy.ndarray
        Rescaled factors. The output has the same shape as the input.

    Notes
    -----
    The normalization is applied pointwise:

        new[i] = factors[i] / mean(factors[neighbors(i)])

    This does not enforce a global mean; it only smooths local variability.
    """
    if not len(factors):
        return factors
    new = factors.copy()
    # iterate over paths
    for i, shooting_value in enumerate(shooting_values):
        # obtain mask: paths whose shooting value is close
        # to the one considered here
        mask = ((shooting_values > (shooting_value - cutoff / 2)) &
                (shooting_values < (shooting_value + cutoff / 2)))
        if mask.sum() < norm:
            # take the factors_norm-closest
            mask = np.argsort(np.abs(shooting_values - shooting_value))
            mask = mask[:norm]
        
        # properly rescale based on the count
        new[i] /= np.mean(factors[mask])
    return new


def reweight_excursions(shooting_values, extremes, factors, xP_extremes, xP):
    """
    Compute normalized excursion weights from factors and crossing probabilities.

    This function implements the final step of AIMMD excursion reweighting. It
    assumes:
    - each excursion is described by a shooting value, an extreme, and a factor,
    - a crossing probability curve xP(E) has already been computed on `xP_extremes`.

    The output weights are normalized such that transitions (largest extremes)
    have weight ~1. The normalization uses the last xP point and a logistic
    correction ``expit(xP_extremes[-1])`` exactly as in the original code.

    Parameters
    ----------
    shooting_values : array-like
        Per-excursion shooting values. ``-inf`` marks equilibrium/free excursions.
    extremes : array-like
        Per-excursion extreme values (sorted internally).
    factors : array-like
        Per-excursion correction factors (density correction, uniformized, etc.).
    xP_extremes : array-like
        Extreme values defining the crossing probability curve.
    xP : array-like
        Crossing probability values aligned with `xP_extremes`.

    Returns
    -------
    (weights, order, shooting_values, extremes, xP_at_extremes, factors, m)
        weights : numpy.ndarray
            Excursion weights in the original input order.
        order : numpy.ndarray
            Indices that sort excursions by `extremes`.
        shooting_values : numpy.ndarray
            Shooting values sorted by extremes.
        extremes : numpy.ndarray
            Extremes sorted.
        xP_at_extremes : numpy.ndarray
            Crossing probabilities mapped to each excursion extreme.
        factors : numpy.ndarray
            Factors sorted by extremes.
        m : numpy.ndarray
            Denominator term used in the weight definition.

    Notes
    -----
    The core quantity is `m[i]`, defined (after sorting by extremes) as a
    weighted count of trajectories whose shooting value is not larger than the
    current extreme. Free excursions (shooting value -inf) are treated as an
    "equilibrium" prefix whose cumulative m is precomputed.

    The final weights are:

        w[i] = factors[i] * xP(extreme[i]) / m[i]

    followed by a global normalization so that the final transition weight is ~1.
    """
    shooting_values = np.asarray(shooting_values)
    extremes = np.asarray(extremes)
    factors = np.asarray(factors)
    
    if not len(shooting_values):
        return (np.zeros(0), np.zeros(0, dtype=int), np.zeros(0),
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    
    # order if not already
    order = np.argsort(extremes)
    shooting_values = shooting_values[order]
    extremes = extremes[order]
    factors = factors[order]
    
    # compute m
    m = np.zeros(len(extremes))

    # equilibrium part
    mask = shooting_values == -inf
    n = np.count_nonzero(mask)
    m[:n] = np.cumsum(factors[mask][::-1])[::-1]
    threshold = -inf
    if not mask.any():
        threshold = np.min(shooting_values[~mask])
    
    # actual computation
    current_extreme = -inf
    current_m = 1.
    for i in range(len(m)):
        if extremes[i] > current_extreme:  # need to update
            current_extreme = extremes[i]
            if current_extreme < threshold:
                current_m = m[i]
                continue  # precomputed value
            else:
                current_m = (factors[i:] * 
                    (shooting_values[i:] <= current_extreme)).sum()
        m[i] = current_m
        if current_extreme == +inf:
            m[i:] = current_m
            break
    
    # crossing probability at the extremes
    xP = xP[np.clip(np.searchsorted(xP_extremes, extremes, 'left'),
                    0, len(xP) - 1)]
    
    # weights of excursions (normalized such that transitions = 1)
    weights = np.zeros(len(m))
    mask = m > 0
    weights[mask] = factors[mask] * xP[mask] / m[mask]
    if len(xP) and (norm := xP[-1] * expit(xP_extremes[-1])):
        weights /= norm
    
    # return with right order
    invert_order = np.empty_like(order)
    invert_order[order] = np.arange(len(order))
    return (weights[invert_order], order,
            shooting_values, extremes,
            xP, factors, m)
