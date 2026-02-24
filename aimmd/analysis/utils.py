"""
...
"""

# external
import numpy as np
from math import inf, nan
from scipy.stats import beta
from scipy.interpolate import RegularGridInterpolator

# aimmd imports
from ..pathensemble.utils import match_patterns


def find_extremes_with_free_simulations(pathensemble,
                                        states='ARB',
                                        source='values'):
    """begin, end used for computing bins"""
    
    # get types and states
    types = pathensemble.types()
    a, r, b = states

    # extract trajectory types
    free_excursions_from1 = pathensemble[match_patterns(types, f'{a}{r}.{a}')]
    free_excursions_from2 = pathensemble[match_patterns(types, f'{b}{r}.{b}')]
    n_transitions = match_patterns(types, states, states[-1]).sum()
    
    # (inverse) free crossing probability histogram from A and from B
    e1 = np.sort(free_excursions_from1.max(source, source))[::-1]
    if not len(e1):
        e1 = [-inf]
    e2 = np.sort(free_excursions_from2.min(source, source))
    if not len(e2):
        e2 = [+inf]
    
    # assign
    limit = n_transitions + 1
    return e1[min(limit, len(e1) - 1)], e2[min(limit, len(e2) - 1)]


def find_extremes_with_transitions(pathensemble,
                                   states='ARB',
                                   source='values'):
    """begin, end used for computing bins"""
    
    # get types and states
    types = pathensemble.types()
    a, r, b = states

    mask = np.flatnonzero(match_patterns(types, states, states[::-1]))
    if not len(mask):
        return 0., 0.

    begin = []
    end = []
    for k, path_type in zip(mask, types[mask]):
        if path_type[:3] == states:
            begin.append(pathensemble[k]._position(1, source))
            end.append(pathensemble[k]._position(-2, source))
        else:
            end.append(pathensemble[k]._position(1, source))
            begin.append(pathensemble[k]._position(-2, source))
    
    return np.median(begin), np.median(end)
    
# functions
def compute_bins(pathensemble,
                 nbins,
                 cutoff_max=20.,
                 cutoff_min=.5,
                 find_extremes_with='free',
                 source='values',
                 states='ARB',
                 marginal_bins='all'):
    """
    nbins including marginal
    find_extremes_with: str, either "free" or "transitions"
    Cut bins only in case of "transitions" being used.
    Assumes "states" are ordered.
    """
    
    if nbins <= 0:
        return np.array([])

    # extension to states
    a, r, b = states
    left = False
    right = False
    if marginal_bins == 'all':
        left = True
        right = True
    elif a in marginal_bins:
        left = True
    elif b in marginal_bins:
        right = True

    # ===reactive region
    if nbins == 1 and left and right:
        return np.array([-inf, +inf])

    # find extremes
    begin1, end1 = -inf, +inf
    if find_extremes_with == 'free':
        begin1, end1 = find_extremes_with_free_simulations(
            pathensemble, states)        

    begin2, end2 = -inf, +inf
    if (find_extremes_with == 'transitions' or
        begin1 < -cutoff_min or
        end1 > +cutoff_max):
        begin2, end2 = find_extremes_with_transitions(
            pathensemble, states)
        delta = (end2 - begin2) / (nbins + 1)
        if left:
            begin2 += delta
        if right:
            end2 -= delta

    if begin1 > -cutoff_max:
        begin = min(begin1, -cutoff_min)
    else:
        begin = np.clip(begin2, -cutoff_max, -cutoff_min)
    if end1 < +cutoff_max:
        end = max(end1, +cutoff_min)
    else:
        end = np.clip(end2, +cutoff_min, +cutoff_max)
    
    # # further correction (try to avoid empty bins)
    # if e1[-1] > begin:
    #     begin = np.clip(e1[-1], -cutoff_max, -cutoff_min)
    # if e2[-1] < end:
    #     end = np.clip(e2[-1], +cutoff_min, +cutoff_max)

    bins = np.linspace(begin, end, nbins + 1 - left - right)
    if left:
        bins = np.append([-inf], bins)
    if right:
        bins = np.append(bins, [+inf])
    
    # return
    return bins


def bin_centers(bins):
    """Also with +- inf"""
    bins = np.asarray(bins)
    length = len(bins)
    if length < 2:
        raise TypeError('"bins" must have at least size 2')
    if length == 2:
        if bins[0] == -inf and bins[1] == +inf:
            return np.zeros(1, dtype=bins.dtype)
        if bins[0] == -inf:
            return bins[-1:]
        if bins[1] == +inf:
            return bins[:1]
    if length == 3 and bins[0] == -inf and bins[2] == +inf:
        return np.repeat(bins[1], 2)
    result = np.zeros(length - 1)
    if bins[+0] == -inf:
        result[+0] = 1.5 * bins[+1] - 0.5 * bins[+2]
    else:
        result[+0] = (bins[+0] + bins[+1]) / 2
    if bins[-1] == +inf:
        result[-1] = 1.5 * bins[-2] - 0.5 * bins[-3]
    else:
        result[-1] = (bins[-1] + bins[-2]) / 2
    result[1:-1] = (bins[1:-2] + bins[2:-1]) / 2
    return result


def merge_marginal_bins(bins, *values, min_values=3):
    """
    Merge marginal histogram bins containing fewer than
    `min_values` data points, for each data set "values"
    
    Parameters
    ----------
    bins : np.ndarray
        Array of shape (k+1,) containing bin boundaries.
        Bin i spans [bins[i], bins[i+1]).
    values : list of np.ndarray
        list of 1D array of raw data points.
    min_n_values : int
        Minimum number of values for each data list required for a bin
        to remain independent.
    
    Merging rule
    ------------
    From left and from right, up until there are enough data.
    
    Returns
    -------
    new_bins : np.ndarray
        Updated bin boundaries after merging.
    merged_bin_counts : np.ndarray
        Number of original bins merged into each new bin.
        Such that you can give right selection weight.
    """
    
    histograms = np.array([np.histogram(v, bins=bins)[0]
                           for v in values]).min(axis=0)

    i = 0
    for i, h in enumerate(histograms[1:]):
        if h >= min_values:
            break
    j = 0
    for j, h in enumerate(histograms[::-1][1:]):
        if h >= min_values:
            break
    
    bins = np.concatenate([[bins[0]], bins[i+1:len(bins)-j-1], [bins[-1]]])
    counts = np.ones(len(bins) - 1, dtype=int)
    counts[+0] = i + 1
    counts[-1] = j + 1
    if len(counts) == 1:
        counts += 1
    return bins, counts


def binomial_mean_and_confidence_interval(r1, r2, alpha=0.95):
    n = r1 + r2
    if not n:
        return nan, nan, nan
    k = r2

    a = 1 - alpha
    
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(a/2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - a/2, k + 1, n - k)

    return k / n, lower, upper


def solve_committor_by_relaxation(
        X, Y, Fx, Fy, A, B, P0, progress=[5, 4, 2, 1]):
    """
    Compute committor in 2D with relaxation method (Brownian dynamics).

    Parameters
    ----------
    X: x-coordinates on a 2D grid
    Y: y-coordinates on a 2D grid
    Fx: force's x component on a 2D grid
    Fy: force's y component on a 2D grid
    A: points in A on a 2D grid
    B: points in B on a 2D grid
    P0: initial guess for committor on a 2D grid
    progress: iteratively increase the resolution based on the vector's values

    Returns
    -------
    P0: committor estimate
    """
    for split in tqdm(progress):
        X1 = X[::split, ::split]
        Y1 = Y[::split, ::split]
        P1 = P0[::split, ::split]
        Fx1 = Fx[::split, ::split]
        Fy1 = Fy[::split, ::split]
        A1 = A[::split, ::split]
        B1 = B[::split, ::split]
        dFx = np.diff(X1, axis=1)[1:-1, :-1] * Fx1[1:-1, 1:-1]
        dFy = np.diff(Y1, axis=0)[:-1, 1:-1] * Fy1[1:-1, 1:-1]
        dFx[dFx > +1.] = +1.
        dFx[dFx < -1.] = -1.
        dFy[dFy > +1.] = +1.
        dFy[dFy < -1.] = -1.
        r = np.max(np.abs(
            (((P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
              (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
             (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
              dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)
        ))
        while True:
            r1 = 0 + r
            for i in range(100):
                P1[1:-1, 1:-1] = (2 * (P1[2:, 1:-1] + P1[:-2, 1:-1] +
                                       P1[1:-1, 2:] + P1[1:-1, :-2]) +
                               (dFy * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                                dFx * (P1[1:-1, 2:] - P1[1:-1, :-2]))) / 8
                P1[:, 0] = P1[:, 1]
                P1[:, -1] = P1[:, -2]
                P1[0, :] = P1[1, :]
                P1[-1, :] = P1[-2, :]
                P1[P1 < 0] = 0
                P1[P1 > 1] = 1
                P1[A1] = 0
                P1[B1] = 1
            r = np.max(np.abs(((
                 (P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
                 (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
                 (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                  dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)))
            if np.abs(r - r1) < 1e-16:
                break
        if split > 1:
            interp = RegularGridInterpolator((Y1[:, 0], X1[0, :]), P1, bounds_error=False,
                                             fill_value=None)
            P0 = interp(np.column_stack((Y.ravel(), X.ravel()))).reshape(X.shape)
        else:
            P0 = P1.copy()
    return P0
