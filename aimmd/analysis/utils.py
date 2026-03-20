"""
aimmd.analysis.utils
===================

Numerical analysis helpers for AIMMD.

This module collects standalone routines used by the trainer and by analysis
workflows. Functions operate on NumPy arrays and on PathEnsemble-like
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

Track and plot paths lineage
    - :func:`find_path_lineages` reconstructs path lineages by parsing the specific
      worker.log file produced by AIMMD for each shooting chain.
    - :func:`plot_path_lineages` visualizes path lineages in a compact way with nice
      graphics.

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
import os
import numpy as np
import matplotlib.pyplot as plt
from math import inf, nan
from tqdm import tqdm
from scipy.stats import beta
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Patch

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

    # check cutoffs
    if cutoff_min < 0:
        raise TypeError(f'cutoff_min must be > 0, got {cutoff_min}')
    if cutoff_max <= cutoff_min:
        raise TypeError(f'cutoff_max must be >= cutoff_min ({cutoff_min}), got {cutoff_max}')
    
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
        # ensure no nans
        if np.isnan(begin1):
            begin1 = -cutoff_min
        if np.isnan(end1):
            end1 = +cutoff_min
    
    begin2, end2 = -inf, +inf
    if (find_extremes_with == 'transitions' or
        begin1 < -cutoff_min or
        end1 > +cutoff_max):
        begin2, end2 = find_extremes_with_transitions(
            pathensemble, states)
        # ensure no nans
        if np.isnan(begin2):
            begin2 = -cutoff_min
        if np.isnan(end2):
            end2 = +cutoff_min
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
    Merge low-occupancy bins into the closest occupied bins moving away
    from the reference ``center``. Any additional histograms are merged
    accordingly by summation.

    An empty bin is merged only if there is an occupied bin further
    outward on the same side of ``center``. Thus, bins left of or at
    ``center`` merge leftward, while bins right of ``center`` merge
    rightward. Empty edge bins with no occupied bin further outward are
    left unchanged.

    Parameters
    ----------
    bins : np.ndarray
        One-dimensional array of bin boundaries with shape ``(k + 1,)``.
        Bin ``i`` spans ``[bins[i], bins[i + 1])``.
    keepers : np.ndarray
        Indices or boolean mask identifying bins that are occupied and
        therefore kept as merge targets.
    *histograms : np.ndarray
        One or more one-dimensional histograms of shape ``(len(bins) - 1,)``
        defined on the original bins. Each histogram is merged to match
        the returned ``merged_bins`` by summing the values of all original
        bins contributing to each merged bin.
    center : float, default=0
        Reference value defining the outward merge direction. Bins whose
        centers are less than or equal to ``center`` merge toward smaller
        values; bins whose centers are greater than ``center`` merge
        toward larger values.

    Returns
    -------
    merged_bins : np.ndarray
        Updated bin boundaries after merging.
    merged_bin_counts : np.ndarray
        Number of original bins contributing to each merged bin.
    *merged_histograms : np.ndarray
        The input histograms after merging by summation, returned in the
        same order as provided.

    """

    # convert keepers to mask
    mask = np.zeros(len(bins) - 1, dtype=bool)
    mask[keepers] = True

    # trivial case: nothing to do
    if mask.all():
        return bins, tuple(histograms), np.ones(len(mask), dtype=int)

    # bin centers determine whether a bin is left or right of the center
    centers = bin_centers(bins)

    # each bin initially maps to itself; empty bins may be reassigned
    bins_to_merged_bins = np.arange(len(mask))

    for i in np.flatnonzero(~mask):

        # empty bins on the left side merge into the closest occupied bin
        # further to the left (more external)
        if centers[i] <= center:
            j = i - 1
            while j >= 0 and not mask[j]:
                j -= 1
            if j >= 0:
                bins_to_merged_bins[i] = j

        # empty bins on the right side merge into the closest occupied bin
        # further to the right (more external)
        else:
            j = i + 1
            while j < len(mask) and not mask[j]:
                j += 1
            if j < len(mask):
                bins_to_merged_bins[i] = j

    # consecutive bins with the same target define one merged bin
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(bins_to_merged_bins))]
    merged_bins = np.r_[bins[starts], bins[-1]]

    # number of original bins contributing to each merged bin
    merged_bin_counts = np.diff(np.r_[starts, len(mask)])

    # merge all histograms by summation
    merged_histograms = tuple(
        np.add.reduceat(histogram, starts) for histogram in histograms
    )

    return merged_bins, merged_bin_counts, *merged_histograms


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


def find_path_lineages(*shooting_chains, verbose=False):
    """
    Reconstructs path lineages by parsing the specific worker.log for each chain.
    
    This function attaches a `._previous` attribute to path objects. 
    It dynamically identifies the correct log file for every path by looking 
    at its directory.
    
    Parameters
    ----------
    *shooting_chains : list of PathChain objects
        One or more chains containing path objects to be linked.
    verbose : bool, default=False
        If True, displays a progress bar for the parsing process.
    """
    
    # --- 1. PRE-PROCESSING ---
    # Flatten all paths from all chains into a single lookup map
    for chain in shooting_chains:
        fnames = [path.fname for path in chain._paths]
        
        if not fnames:
            return
        
        # current folder
        folder = '/'.join(fnames[-1].split('/')[:-1])
        suffix = '.' + fnames[-1].split('.')[-1]
        
        # --- 2. LOG PARSING HELPER ---
        def extract_name(line, prefix, add_suffix=''):
            """Extracts and cleans a path name from a log line."""
            if prefix in line:
                # Splits by prefix, takes name before '(', removes quotes
                part = line.split(prefix)[1].split('(')[0]
                clean_name = part.strip().replace("'", "").replace('"', "")
                return f"{clean_name}{add_suffix}"
            return None
        
        # --- 3. MULTI-LOG TRAVERSAL ---
        done_fnames = set()
        pbar = tqdm(total=len(fnames), disable=not verbose, desc=folder)
            
        # Iterate through folders (and their respective logs)
        log_path = os.path.join(folder, 'worker.log')
        if not os.path.exists(log_path):
            if verbose:
                print(f"\nWarning: Log not found in {folder}")
            continue
        
        # State trackers for the current log file
        current_path = None
        parent_path = None
        
        with open(log_path, 'r') as log_file:
            for line in log_file:
                # Identify the 'child' being created
                current_path = extract_name(
                    line, 'Selecting shooting point for', suffix
                ) or current_path
                
                # Identify the 'parent' being shot
                parent_path = (
                    extract_name(line, '=== selecting path') or 
                    extract_name(line, '=== overriding with')
                ) or parent_path
                
                # Commit the link when initialization is logged as complete
                if line.startswith('Shooting initialization completed'):
                    # Only process if the child is in our provided chains
                    
                    if current_path in fnames:
                        path = chain._paths[fnames.index(current_path)]
                        
                        # Link to parent object if found in chains, else store string
                        if parent_path in fnames:
                            path._previous = chain._paths[
                                fnames.index(parent_path)]
                        else:
                            path._previous = parent_path
                        
                        if current_path not in done_fnames:
                            pbar.update(1)
                            done_fnames.add(current_path)
        pbar.close()


def plot_path_lineages(*shooting_chains, out='path_tree.pdf',
                       states='ARB', source='values', 
                       vmin=-20, vmax=20, fields=None, show=False):
    """
    Visualizes the genealogical history of AIMMD path sampling results
    ensembles using an independently-packed grid layout. 
    
    Each lineage is treated as an autonomous vertical unit, allowing 
    columns to stack tightly without being constrained by the length 
    of chains in adjacent tiles. Labels are centered relative to the 
    data axis of each column.
    """
    
    # --- 1. CONFIGURATION & STATE NORMALIZATION ---
    states = sorted([states[0], states[-1]])
    s_A, s_B = states[0], states[-1]
    done, fnames, values = [], [], []
    
    color_map = {
        'AA': '#EE8866', 'BB': '#77AADD', 
        'AB': '#44AA99', 'incomplete': '#BBBBBB',
        'free_A': '#FF0000',  # Bright Red
        'free_B': '#0000FF'   # Bright Blue
    }
    
    # --- 2. LINEAGE EXTRACTION ---
    for chain in shooting_chains:
        i = len(chain) - 1
        while i >= 0:
            path = chain[i]
            if path in done:
                i -= 1
                continue
                        
            curr_fnames, curr_vals = [], []
            while True:
                done.append(path)
                s_val = path.shooting(source)
                min_v = -inf if s_A in path.type else path.min(source)
                max_v = inf if s_B in path.type else path.max(source)
                
                is_A, is_B = (min_v == -inf), (max_v == inf)
                if is_A and not is_B: p_type = 'AA'
                elif is_B and not is_A: p_type = 'BB'
                elif is_A and is_B: p_type = 'AB'
                else: p_type = 'incomplete'
                
                extra_info = ""
                if fields:
                    info_bits = [f"{f}:{getattr(path, f)}" 
                                 for f in fields if hasattr(path, f)]
                    extra_info = " | ".join(info_bits)
                                
                curr_vals.insert(0, (min_v, s_val, max_v, p_type, extra_info))
                curr_fnames.insert(0, path.fname)
                
                if not hasattr(path, '_previous'):
                    i -= 1
                    break
                
                prev = path._previous
                if isinstance(prev, str):
                    if 'initial' in prev: 
                        curr_vals.insert(0, (-inf, s_val, inf, 'AB', ""))
                    elif f'free{s_A}' in prev: 
                        # Specific Color: Bright Red for free state A
                        curr_vals.insert(0, (-inf, s_val, s_val, 'free_A', ""))
                    elif f'free{s_B}' in prev: 
                        # Specific Color: Bright Blue for free state B
                        curr_vals.insert(0, (s_val, s_val, inf, 'free_B', ""))
                    curr_fnames.insert(0, prev)
                    i -= 1
                    break
                path = prev
            
            fnames.append(curr_fnames)
            values.append(curr_vals)
    
    # --- 3. DYNAMIC BOUNDARY CALCULATION ---
    flat_coords = []
    for col in values:
        for p in col:
            flat_coords.extend([p[0], p[1], p[2]])
    flat_coords = np.array(flat_coords)
    real_coords = flat_coords[~np.isinf(flat_coords)]
    
    p_min = min(vmin, np.min(real_coords)) if real_coords.size > 0 else vmin
    p_max = max(vmax, np.max(real_coords)) if real_coords.size > 0 else vmax
    plot_width = p_max - p_min
    
    # --- 4. TILING GEOMETRY ---
    num_total_chains = len(values)
    max_cols = int(np.ceil(np.sqrt(num_total_chains))) if num_total_chains > 3 else num_total_chains
    
    col_heights = [0.0] * max_cols
    v_buffer = 2.5 
    col_width = plot_width * 1.3
    
    positions = []
    for idx, col_v in enumerate(values):
        tile_c = idx % max_cols
        chain_len = len(col_v)
        y_start = col_heights[tile_c]
        positions.append((tile_c, y_start))
        col_heights[tile_c] += (chain_len + v_buffer)
        
    total_max_height = max(col_heights)
    fig, ax = plt.subplots(figsize=(max_cols * 4.5, total_max_height * 0.35 + 1))
    
    # --- 5. RENDERING ---
    for idx, (col_f, col_v, (tile_c, y_base)) in enumerate(zip(fnames, values, positions)):
        x_off = tile_c * col_width
        y_top = total_max_height - y_base
        chain_len = len(col_v)
        
        if tile_c < max_cols - 1:
            sep_x = x_off + p_max + (col_width - plot_width) / 2
            ax.vlines(sep_x, y_top - chain_len - 1, y_top + 1, color='#EEEEEE', lw=1.0, zorder=0)

        for r_idx, (v_min, v_s, v_max, p_type, extra) in enumerate(col_v):
            y = y_top - r_idx
            color = color_map.get(p_type, '#BBBBBB')
            d_min = p_min if v_min == -inf else v_min
            d_max = p_max if v_max == inf else v_max
            
            ax.plot([d_min + x_off, d_max + x_off], [y, y], color=color, lw=3.0, zorder=2)
            
            if v_min == -inf:
                ax.text(d_min + x_off - 0.1, y, s_A, ha='right', va='center', weight='bold', color=color, fontsize=6)
            if v_max == inf:
                ax.text(d_max + x_off + 0.1, y, s_B, ha='left', va='center', weight='bold', color=color, fontsize=6)
            
            ax.text(d_min + x_off, y - 0.22, f"{v_min:.2f}" if v_min != -inf else s_A, fontsize=5, ha='left', color='#666666', va='top')
            ax.text(d_max + x_off, y - 0.22, f"{v_max:.2f}" if v_max != inf else s_B, fontsize=5, ha='right', color='#666666', va='top')
            ax.text(v_s + x_off, y + 0.1, f"{v_s:.2f}", fontsize=5, ha='center', weight='bold', va='bottom')
            
            ax.scatter(v_s + x_off, y, color='white', ec='#333333', s=25, zorder=4)
            
            if r_idx > 0: 
                ax.vlines(v_s + x_off, y, y + 1, color='#BBBBBB', ls=':', lw=0.8, zorder=1)
            
            full_label = f"{col_f[r_idx]}  {extra}" if extra else col_f[r_idx]
            center_x = x_off + (p_min + p_max) / 2
            ax.text(center_x, y + 0.25, full_label, fontsize=6.5, ha='center', va='bottom', alpha=0.9, family='monospace')
    
    # --- 6. CLEANUP ---
    ax.axis('off')
    ax.set_xlim(p_min - 2, max_cols * col_width)
    ax.set_ylim(-1, total_max_height + 1)
    
    leg_handles = [Patch(color=c, label=k) for k, c in color_map.items() if not k.startswith('free_')]
    ax.legend(handles=leg_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
