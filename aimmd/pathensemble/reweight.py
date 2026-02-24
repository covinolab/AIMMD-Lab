"""
...
"""

# external
import numpy as np
from math import inf, nan
from scipy.special import expit

# functions
def compute_shooting_density(values, shooting_value, neighbors=10):
    """n^star of values at shooting point"""
    
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
    """crossing_probability_cutoff:
        if > 0 in principle more robust but less data
        discard if shooting_value > extreme - cutoff
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
