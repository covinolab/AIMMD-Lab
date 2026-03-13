"""
aimmd.analysis.utils
===================

Numerical analysis helpers for AIMMD.

This module collects lightweight, standalone routines used by the trainer and by
analysis workflows. Functions operate on NumPy arrays and on PathEnsemble-like
objects (AIMMD’s path-sampling convention), without depending on high-level
AIMMD classes.

Overview
--------
Bins and grid helpers
    - :func:`compute_bins` constructs 1D bin boundaries (optionally including
      marginal outer bins at ``-inf`` and/or ``+inf``).
    - :func:`bin_centers` returns bin centers and supports bins including
      plus and minus infinity.
    - :func:`merge_empty_bins` merges low-occupancy bins with the closest
      ones moving away from the transition state.
    - :func:`merge_marginal_bins` merges low-occupancy marginal bins.

Extremes used to define bins
    - :func:`find_extremes_with_free_simulations` estimates left/right extremes
      from free excursions started from each terminal state.
    - :func:`find_extremes_with_transitions` estimates left/right extremes from
      transition paths by sampling values adjacent to the endpoints.

Simple statistics
    - :func:`binomial_mean_and_confidence_interval` computes a binomial mean and
      a two-sided confidence interval using Beta quantiles.

Rate-estimate extraction
    - :func:`extract_rate_estimates_from_log_file` parses AIMMD training
      logs and returns the time series of forward/backward rate estimates.

      The parser is tailored to AIMMD ``train*.log`` files. It scans the file in
      order and recognizes a three-line pattern:

      - a line containing ``"k12 estimate"``: extracts the forward estimate
        following ``"estimate:"`` (up to the next ``"["`` token),
      - the next relevant line: extracts the backward estimate in the same way,
      - the next relevant line: extracts the time coordinate as the number
        preceding the token ``"frames"``.

      The returned arrays ``(t, k12, k21)`` therefore reflect the sequence of
      estimates printed by the training logger and the corresponding cumulative
      simulated time/frames at which they were reported.

2D committor solver
    - :func:`solve_committor_by_relaxation` solves a committor field on a 2D grid
      by relaxation (finite-difference discretization) with coarse-to-fine
      refinement.

Dependencies and expectations
-----------------------------
- PathEnsemble conventions:
  ``pathensemble.types()`` returns per-path type strings and ``pathensemble[...]``
  supports boolean mask selection.
- Type-pattern matching is delegated to
  :func:`aimmd.pathensemble.utils.match_patterns`.

Notes
-----
The bin construction functions assume that the projected reaction-coordinate
values order terminal states consistently (as in AIMMD training). Bounds are
made robust by explicit cutoffs and by falling back from free-excursion
statistics to transition statistics when required.
"""

# external
import numpy as np
from math import inf, nan
from tqdm import tqdm
from scipy.stats import beta
from scipy.interpolate import RegularGridInterpolator

# aimmd imports
from ..core.utils import longest_true_segment
from ..pathensemble.utils import match_patterns


def find_extremes_with_free_simulations(pathensemble,
                                        states='ARB',
                                        source='values'):
    """
    Estimate bin extremes using free excursions.

    This helper is used by :func:`compute_bins` to obtain left/right limits for
    a bin range using only *free* excursions originating from each terminal state.

    It extracts two subsets:

    - free excursions originating from `a` and returning to `a`
      (pattern ``f'{a}{r}.{a}'``)
    - free excursions originating from `b` and returning to `b`
      (pattern ``f'{b}{r}.{b}'``)

    From these, it computes:
    - `e1`: sorted per-path maxima (descending) from `a`-side free excursions
    - `e2`: sorted per-path minima (ascending) from `b`-side free excursions

    The index used is `n_transitions + 1`, where `n_transitions` counts paths
    matching either direction of the requested transition.

    Parameters
    ----------
    pathensemble : PathEnsemble-like
        Ensemble supporting:
        - ``types()`` returning per-path type strings
        - boolean-mask selection ``pathensemble[mask]``
        - per-path extrema selectors ``max`` and ``min``.
    states : str, default='ARB'
        Triplet of state labels `(a, r, b)`:
        - `a`: left terminal state
        - `r`: reactive region label
        - `b`: right terminal state
    source : str, default='values'
        Name of the per-frame stream used when computing per-path extrema.

    Returns
    -------
    (float, float)
        (begin, end) suggested extremes for binning.

    Notes
    -----
    If no free excursions are present on a side, the corresponding list is
    replaced with ``[-inf]`` or ``[+inf]`` to keep the selection logic stable.
    """
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
    """
    Estimate bin extremes using transition trajectories.

    This helper is used by :func:`compute_bins` when free excursions do not give
    reasonable bounds (or when explicitly requested via `find_extremes_with`).

    It identifies transition paths in either direction (A→B or B→A) using
    type-pattern matching and then collects representative boundary-adjacent
    values from those paths:

    - For paths with type prefix equal to `states` (A→R→B):
        - begin uses ``_position(1, source)`` (near the start, excluding boundary)
        - end   uses ``_position(-2, source)`` (near the end, excluding boundary)

    - For the reverse direction:
        - begin/end are swapped accordingly.

    The returned extremes are the medians over the collected values.

    Parameters
    ----------
    pathensemble : PathEnsemble-like
        Ensemble supporting:
        - ``types()`` returning per-path type strings
        - integer indexing returning Path objects
        - Path method ``_position(i, source)``.
    states : str, default='ARB'
        Triplet `(a, r, b)` defining which transitions are considered.
    source : str, default='values'
        Stream name passed to ``_position``.

    Returns
    -------
    (float, float)
        (begin, end) suggested extremes for binning.

    Notes
    -----
    If no transition trajectories are present, returns ``(0., 0.)``.
    """
    
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
    

def compute_bins(pathensemble,
                 nbins,
                 cutoff_max=20.,
                 cutoff_min=.5,
                 find_extremes_with='free',
                 source='values',
                 states='ARB',
                 marginal_bins='all'):
    """
    Construct 1D bin boundaries for projection/analysis.

    This function returns a 1D array of bin *boundaries* of length ``nbins + 1``.
    The finite (non plus or minus inf) part of the grid spans the interval
    ``[begin, end]``, where (begin, end) are inferred from the ensemble using one
    of the extreme estimators:

    - :func:`find_extremes_with_free_simulations` (default)
    - :func:`find_extremes_with_transitions` (fallback or explicit)

    The returned boundaries are then optionally given *marginal* outer bins by
    replacing the first and/or last boundary with ``-inf`` and/or ``+inf``.
    The finite interior boundaries are still uniformly spaced between `begin`
    and `end`.

    Parameters
    ----------
    pathensemble : PathEnsemble-like
        Ensemble used to infer the finite interval.
    nbins : int
        Number of bins (including marginal bins if requested).
        The returned array has length ``nbins + 1``. If ``nbins <= 0``,
        returns an empty array.
    cutoff_max : float, default=20.
        Hard clip for the finite interval. If the free-based extremes exceed
        this range, transition-based extremes are used and clipped.
    cutoff_min : float, default=0.5
        Inner clip to keep the finite interval away from zero:
        `begin <= -cutoff_min` and `end >= +cutoff_min` when possible.
    find_extremes_with : {'free', 'transitions'}, default='free'
        Preferred heuristic for obtaining (begin, end). Even when 'free' is
        requested, transition-based extremes are used if the free-based bounds
        fall outside the configured cutoffs.
    source : str, default='values'
        Passed to the extreme estimators.
    states : str, default='ARB'
        Triplet `(a, r, b)` used by the extreme estimators.
    marginal_bins : {'all'} or str, default='all'
        Which marginal bins to include by replacing outer boundaries:

        - 'all' : replace both ends (first boundary = -inf, last boundary = +inf)
        - string containing `a` : replace the left boundary with -inf
        - string containing `b` : replace the right boundary with +inf

        The replacement changes the *outermost* bin to be unbounded. The finite
        interior bins still span [begin, end].

    Returns
    -------
    numpy.ndarray
        1D array of bin boundaries. If marginal bins are enabled, the first and/or
        last element is ±inf. Otherwise all boundaries are finite.

    Notes
    -----
    Implementation detail:
    - The number of *finite* boundaries generated by ``np.linspace`` is
      ``nbins + 1 - left - right`` where `left/right` indicate marginal bins.
    - If marginal bins are requested, `-inf` and/or `+inf` are then inserted as
      the first/last boundary, replacing the corresponding finite boundary.
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

    # reactive region
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
    
    bins = np.linspace(begin, end, nbins + 1 - left - right)
    if left:
        bins = np.append([-inf], bins)
    if right:
        bins = np.append(bins, [+inf])
    
    # return
    return bins


def bin_centers(bins):
    """
    Return bin centers, supporting `±inf` marginal bins.
    
    Parameters
    ----------
    bins : array-like
        1D array of bin boundaries of length >= 2. Bin i spans [bins[i], bins[i+1]).
    
    Returns
    -------
    numpy.ndarray
        Array of length ``len(bins) - 1`` with one center per bin.
    
    Notes
    -----
    For bins with infinite boundaries, finite extrapolated centers are produced
    using linear extrapolation from the nearest finite bins.
    """
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


def merge_empty_bins(bins, keepers, *histograms, center=0):
    """
    Merge low-occupancy bins with the closest ones moving away from
    the "center" (the transtion state). Additionally merge the provided
    histograms.
    
    Will not merge an empty bin if there are no occupied bins which are
    more external.
    
    Parameters
    ----------
    bins : np.ndarray
        1D array of bin boundaries of shape (k+1,). Bin i spans [bins[i], bins[i+1]).
    keepers : np.ndarray
        Identifies which bins are not empty (the others will be merged).
    histograms : tuple of np.ndarray, shape `len(bins) - 1`
        Histograms to be merged to correspond to `merged_bins`.
    
    Returns
    -------
    merged_bins : np.ndarray
        Updated bin boundaries after merging.
    merged_histograms : tuple of np.ndarray, shape `len(merged_bins) - 1`
        Updated histograms after merging.
    """
    
    # convert keepers to mask
    mask = np.zeros(len(bins) - 1, dtype=bool)
    mask[keepers] = True
    
    # trivial case: nothing to do
    if mask.all():
        return (bins, ) + tuple(histograms)
    
    # need to merge
    centers = bin_centers(bins)
    
    # initialize conversion: each bin maps to itself unless merged
    bins_to_merged_bins = np.arange(len(bins) - 1)
    
    for i in np.flatnonzero(~mask):
        
        # look for more external bins
        if centers[i] <= center:  # going left
            j = i - 1
            while j >= 0 and not mask[j]:
                j -= 1
            if j >= 0:
                bins_to_merged_bins[i] = j
        
        else:  # going right
            j = i + 1
            while j < len(mask) and not mask[j]:
                j += 1
            if j < len(mask):
                bins_to_merged_bins[i] = j
    
    # build merged_bins based on bins_to_merged_bins
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(bins_to_merged_bins))]
    ends = np.r_[starts[1:], len(mask)]
    merged_bins = np.r_[bins[starts], bins[ends[-1]]]
    
    # collapse histograms based on bins_to_merged_bins
    merged_histograms = [
        np.add.reduceat(histogram, starts)
        for histogram in histograms
    ]
    
    # return
    return (merged_bins, ) + tuple(merged_histograms)


def merge_marginal_bins(bins, *values, min_values=3):
    """
    Merge low-occupancy marginal bins.
    
    This helper is used to stabilize analysis when the outermost bins are too
    sparse. It merges bins from the *left edge* and from the *right edge*
    such that:
    1) all bins from the second to the second-to-last are a contiguous
       subset of the original bins;
    2) the above bins contain at least `min_values` counts for all the
       datasets provided;
    3) given the above conditions, the number of bins is maximum.
    
    The occupancy criterion is computed from the per-dataset histograms:
    ``np.histogram(v, bins=bins)[0]`` and then taking the minimum over datasets.
    
    Parameters
    ----------
    bins : np.ndarray
        1D array of bin boundaries of shape (k+1,). Bin i spans [bins[i], bins[i+1]).
    *values : np.ndarray
        One or more 1D arrays of raw samples. Each provided dataset contributes
        its histogram to the "minimum occupancy" decision.
    min_values : int, default=3
        Minimum required histogram count in each marginal bin.
    
    Returns
    -------
    merged_bins : np.ndarray
        Updated bin boundaries after merging.
    merged_bin_counts : np.ndarray
        Number of original bins merged into each new bin. This can be used to
        scale selections consistently when downstream code depends on the
        original binning density.
    
    Notes
    -----
    The function always preserves the first and last boundary of `bins`.
    """
    
    # how many bins to start with?
    nbins = len(bins) - 1
    bin_counts = np.ones(nbins, dtype=int)
    
    # trivial case
    if nbins <= 1:
        return bins, bin_counts
    
    # compute histogram
    histograms = np.array(
        [np.histogram(v, bins=bins)[0] for v in values]).min(axis=0)
    
    # where is the occupacy condition respected
    condition = histograms >= min_values
    
    # find maximum contiguous segment
    b, e = longest_true_segment(condition)
        
    if b == 0:
        if e == nbins:
            # nothing to do: bins and counts are already ok
            merged_bins = bins
            merged_bin_counts = bin_counts
        else:
            # merge on the right
            merged_bins = np.append(bins[:e + 1], [bins[-1]])
            merged_bin_counts = bin_counts[:e + 1]
            merged_bin_counts[-1] = nbins - e
    elif e == nbins:
        # merge on the left
        merged_bins = np.append([bins[0]], bins[b:])
        merged_bin_counts = bin_counts[b - 1:]
        merged_bin_counts[0] = b
    else:
        # merge on both sides
        merged_bins = np.concatenate([[bins[0]], bins[b:e + 1], [bins[-1]]])
        merged_bin_counts = bin_counts[b - 1:e + 1]
        merged_bin_counts[0] = b
        merged_bin_counts[-1] = nbins - e
        
    return merged_bins, merged_bin_counts


def binomial_mean_and_confidence_interval(r1, r2, alpha=0.95):
    """
    Binomial mean and two-sided confidence interval.

    Parameters
    ----------
    r1, r2 : int
        Two outcome counts. The code interprets:
        - n = r1 + r2
        - k = r2
        and computes the confidence interval for p = k/n.
    alpha : float, default=0.95
        Confidence level.

    Returns
    -------
    (float, float, float)
        (p, lower, upper), where p = k/n and (lower, upper) is a two-sided
        beta-based confidence interval.

    Notes
    -----
    This uses the standard Beta quantile construction:
    - lower = Beta(a/2; k, n-k+1)
    - upper = Beta(1-a/2; k+1, n-k)
    with special-case handling for k=0 and k=n.
    """
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


def extract_rate_estimates_from_log_file(fname):
    """
    Parse rate estimates from an AIMMD training log.

    This helper reads AIMMD ``train*.log`` files produced during training and
    extracts the time series of rate-constant estimates printed in the log.

    Parsed quantities
    -----------------
    The function returns three arrays:

    - ``t``   : cumulative simulated time (as printed by the logger)
    - ``k12`` : rate constant estimate for the state 1 to state 2 transition
    - ``k21`` : rate constant estimate for the state 2 to state 1 transition

    Parameters
    ----------
    fname : str or path-like
        Path to a training log file (e.g., ``trainARB.log``).

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        ``(t, k12, k21)`` as NumPy arrays of dtype float.

    Notes
    -----
    This parser assumes the log format where:
    - a line containing ``'k12 estimate'`` holds the k12 value after ``'estimate:'``
    - the next relevant line holds the k21 value in the same format
    - the subsequent relevant line begins with the number of frames followed by
      the token ``'frames'`` (used as the time coordinate)

    The parsing logic is intentionally minimal and matches the existing logger
    output exactly.
    """

    # initialize output
    k12 = []
    k21 = []
    t = []
    with open(fname, 'r') as file:
        step = -1
        for line in file:
            if 'k12 estimate' in line:
                k12.append(float(line.split('estimate:')[1].split('[')[0]))
                step = 1
            elif step == 1:
                k21.append(float(line.split('estimate:')[1].split('[')[0]))
                step = 2
            elif step == 2:
                t.append(float(line.split('frames')[0]))
                step = 0
    return np.array(t), np.array(k12), np.array(k21)

    # initialize output
    k12 = []
    k21 = []
    t = []
    with open(fname, 'r') as file:
        step = -1
        for line in file:
            if 'k12 estimate' in line:
                k12.append(float(line.split('estimate:')[1].split('[')[0]))
                step = 1
            elif step == 1:
                k21.append(float(line.split('estimate:')[1].split('[')[0]))
                step = 2
            elif step == 2:
                t.append(float(line.split('frames')[0]))
                step = 0
    return np.array(t), np.array(k12), np.array(k21)



def solve_committor_by_relaxation(
        X, Y, Fx, Fy, A, B, P0, progress=[5, 4, 2, 1]):
    """
    Compute committor in 2D with a relaxation method.

    This routine solves the steady-state committor equation on a 2D grid using
    a finite-difference discretization and iterative relaxation.

    The method enforces:
    - P=0 on region A
    - P=1 on region B
    - reflecting (zero-gradient) boundary conditions at the grid edges

    Parameters
    ----------
    X, Y : np.ndarray
        2D arrays defining the grid coordinates. Shapes must match.
    Fx, Fy : np.ndarray
        2D arrays of drift/force components on the same grid.
    A, B : np.ndarray of bool
        Boolean masks on the grid indicating the A and B regions.
    P0 : np.ndarray
        Initial guess for the committor field. Updated in-place by coarse-to-fine
        passes.
    progress : list[int], default=[5, 4, 2, 1]
        Coarsening schedule. For each `split`, the solver runs on the subgrid
        ``[::split, ::split]`` and, if `split>1`, interpolates back to the full
        grid before continuing at higher resolution.

    Returns
    -------
    np.ndarray
        Final committor estimate on the full grid.

    Notes
    -----
    - This function uses ``tqdm(progress)`` but does not import tqdm.
      The caller must ensure that `tqdm` is available in scope.
    - The update rule clamps P to [0,1] each iteration and enforces boundary
      conditions after each sweep.
    - When refining, interpolation uses :class:`scipy.interpolate.RegularGridInterpolator`.
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
