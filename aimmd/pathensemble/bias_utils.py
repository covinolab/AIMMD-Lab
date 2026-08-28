"""
aimmd.pathensemble.bias_utils
==============================

Bias reweighting utilities for :class:`aimmd.pathensemble.PathEnsemble`.

This module provides four public functions used by the AIMMD training worker when
``params.record_bias = True``:

- :func:`check_reactive_bias` — validates that the applied bias is negligible inside
  the reactive region R, which is required for the reweighting formula to be valid.

- :func:`compute_bias_corrections` — computes the per-path bias correction factor
  ``γᵢ = ⟨exp(bias)⟩`` used to recover unbiased kinetics from a biased path
  ensemble, and reports how much of the ensemble it could actually correct.

- :func:`bias_reweighted_rates` — the whole reweighted-rate step in one call
  (reactive-bias check, γ, and ``k = 1/Σ(w·L·γ)`` in both directions), so the
  four call sites in :mod:`aimmd.worker._train` cannot drift apart.

- :func:`format_bias_cache_coverage` — renders that coverage diagnostic for the
  training log.

- :func:`derive_bias_from_cumulative_colvar` — read-only fallback that recovers a
  still-running free trajectory's bias from the cumulative PLUMED ``COLVAR``.

Physical background
-------------------
The bias is applied **only within the metastable states** A and B (not in R). This
raises the energy inside the states, making transitions faster. Bias reweighting
corrects rates back to the unbiased values.

Bias convention
---------------
``path.bias`` stores the bias in **kT units** (dimensionless ``β·V_bias``).
A positive value means the bias raised the energy by that many kT at that frame.

Bias-reweighted rate correction
--------------------------------
For each path ``i``::

    γᵢ = mean(exp(path.bias))   over the Lᵢ frames the ensemble counts

The corrected rate estimate is::

    k₁₂_unbiased = 1 / Σᵢ(wᵢ · Lᵢ · γᵢ)

where ``wᵢ`` are the existing RFPS/TPS weights and ``Lᵢ`` is the counted path
length in frames, i.e. :attr:`~aimmd.pathensemble.PathEnsemble.n_frames`.

The frame set matters. ``Lᵢ`` is *not* ``len(path)``: `Path.split` produces blocks
that overlap by two frames, and the ensemble drops each block's boundary frames so
that the blocks of one trajectory partition it. ``Lᵢ · γᵢ`` is meant to be the
boosted (unbiased) residence time of those ``Lᵢ`` frames, i.e.
``Σ_{counted frames} exp(bias)``. Averaging ``exp(bias)`` over the *whole* block
while multiplying by the *trimmed* count is that sum for no frame set at all: a
boundary frame of an in-state dwell block sits in the bias-free reactive region and
contributes ``exp(0) = 1``, so it dilutes ``γ`` and shortens the reweighted dwell
time — one-signed toward faster rates, and worst for short blocks, where the two
margin frames are most of the block. ``γᵢ`` is therefore averaged over exactly the
window :attr:`~aimmd.pathensemble.PathEnsemble.frame_windows` reports, which is the
same window ``Lᵢ`` counts by construction.

Because the bias is negligible in R (verified by :func:`check_reactive_bias`),
``exp(bias) ≈ 1`` for R frames and the correction mainly comes from A/B frames::

    γᵢ ≈ (n_A_i · ⟨exp(bias)⟩_A_i + n_R_i) / n_total_i

Backward compatibility
-----------------------
These functions are only imported inside ``_train.py`` when ``params.record_bias`` is
``True``. Existing runs (``record_bias=False``) are completely unaffected.

If bias cache files (``<traj>.bias.npy``) are missing for some paths (e.g. mixing old
and new runs), :func:`compute_bias_corrections` falls back to ``γᵢ = 1.0`` for those
paths and prints a one-time warning.

Bias-cache coverage
-------------------
That fallback is silent in the rate: an uncorrected path still contributes ``w · L``
to ``Σ(w · L · γ)``, just without its ``γ``. Because the estimator is a sum over
``w · L · γ``, the damage scales with the *weighted path length* running on
``γ = 1.0``, not with the number of affected paths — a single long free-basin
trajectory can be one path in a thousand and still carry most of the dwell time.
For a missing weighted fraction ``f`` and a typical correction ``Γ ≫ 1``, the
reweighted rate is overestimated by ``≈ 1 / (1 - f)``.

The usual cause is structural rather than accidental. AIMMD writes the per-part
PLUMED ``_COLVAR`` slice that ``bias_function`` reads only once an ``mdrun``
segment returns, so a free-basin worker that never escapes its state within a job
never produces one — and that is exactly the deepest, most heavily biased dwell
time. The failure therefore grows with residence time, hitting hardest the slow
systems the in-basin bias exists to accelerate. It is easy to miss because
unrelated things hide it: a short queue limit keeps coverage complete simply by
recycling the job, as does a free trajectory that reaches the other state every
few tens of nanoseconds. A long queue limit combined with a long residence time
removes both, and coverage can then fall to a few per cent.

Two things address it. :func:`derive_bias_from_cumulative_colvar` recovers the
bias for a still-running part directly from the cumulative ``COLVAR``, so the
correction no longer waits for the segment to end — read-only, and without
touching the slicing machinery. And :func:`compute_bias_corrections` runs the
coverage check on every call by default, printing the covered fraction and
warning when it is problematic, so any gap the fallback cannot close is visible.
Coverage is counted in frames, and frames with no bias value contribute
``exp(bias) = 1`` to ``γ`` rather than inheriting the covered frames' average.
"""

# external
import os
import re
import shutil
import tempfile
import textwrap
import warnings
from glob import glob
import numpy as np

# aimmd imports — kept lazy to avoid import-time coupling
# (this module is only imported when record_bias=True)


def check_reactive_bias(pathensemble, states, threshold=0.5):
    """
    Check that the bias is negligible in the reactive region R.

    Validates the Tiwary-Parrinello assumption: the bias potential must be small
    (ideally zero) inside the reactive region R so that the path length correction
    ``γᵢ = ⟨exp(bias)⟩`` is well-defined and the reweighted kinetics are reliable.

    Parameters
    ----------
    pathensemble : PathEnsemble
        The path ensemble to inspect.
    states : str
        Three-character state string, e.g. ``'ARB'``. The middle character ``states[1]``
        is taken as the reactive-region label (e.g. ``'R'``).
    threshold : float, default 0.5
        Maximum acceptable mean ``|bias|`` (in kT) in the reactive region.
        A warning is printed if the mean exceeds this value.

    Returns
    -------
    mean_abs_bias : float
        Mean ``|bias|`` (in kT) over all reactive-region frames across all paths.
        Returns ``nan`` if no reactive frames were found or no bias data is available.
    max_abs_bias : float
        Maximum ``|bias|`` (in kT) over those frames.
    """
    r = states[1] if len(states) >= 2 else states[0]
    all_r_bias = []

    for path in pathensemble:
        try:
            path_states = path._get('states', raise_if_missing=False)
            path_bias = path._get('bias', raise_if_missing=False)
        except Exception:
            continue

        if path_bias is None or path_states is None:
            continue

        n = min(len(path_states), len(path_bias))
        if n == 0:
            continue

        r_mask = path_states[:n] == r
        if r_mask.any():
            all_r_bias.append(path_bias[:n][r_mask])

    if not all_r_bias:
        return float('nan'), float('nan')

    all_r_bias = np.concatenate(all_r_bias)
    mean_abs = float(np.mean(np.abs(all_r_bias)))
    max_abs = float(np.max(np.abs(all_r_bias)))

    if mean_abs > threshold:
        warnings.warn(
            f'Bias check FAILED: mean |bias| in reactive region '
            f'{r!r} is {mean_abs:.3f} kT (threshold {threshold:.3f} kT). '
            f'The bias should be negligible in {r!r} for bias reweighting to be valid. '
            f'Max |bias| in {r!r}: {max_abs:.3f} kT.',
            UserWarning,
            stacklevel=2)
    else:
        print(f'    Bias check passed: mean |bias| in {r!r} = '
              f'{mean_abs:.3f} kT ≤ {threshold:.3f} kT (max {max_abs:.3f} kT)')

    return mean_abs, max_abs


BIAS_CACHE_COVERAGE_THRESHOLD = 0.05
"""Default missing weighted fraction above which the coverage check escalates:
:func:`format_bias_cache_coverage` adds its warning block and
:func:`compute_bias_corrections` raises a ``UserWarning``."""


def _path_label(path):
    """Best-effort human-readable identifier for a path (for diagnostics)."""
    fnames = getattr(path, 'fnames', None)
    if fnames is not None:
        try:
            if len(fnames):
                return str(fnames[0])
        except TypeError:
            pass
    return repr(path)


_REMEDIATION_NOTE = (
    'A still-running free trajectory is normally not bias-cached yet; the trainer '
    'derives its bias out-of-cache from the cumulative COLVAR, so a high figure '
    'here means that fallback also failed.')
"""One-sentence note shared by the warning and the printed report.

Deliberately says nothing about the state definition or the bias fill. Coverage is
set by how often a free ``mdrun`` segment returns (reaching another state,
``params.max_length``, or a worker restart), not by where the state boundary sits:
a short queue limit alone is enough to keep coverage complete.
"""


def _inflation_text(inflation):
    """Format a ``max_inflation`` value, keeping ``inf`` readable."""
    if not np.isfinite(inflation):
        return 'an unbounded factor'
    return f'{inflation:.1f}x'


_PART_RE = re.compile(r'^(?P<base>.+)\.part(?P<part>\d{4})$')


def derive_bias_from_cumulative_colvar(fname, trajectory_extension,
                                       bias_function):
    """
    Derive a free-trajectory part's bias from the cumulative PLUMED COLVAR.

    A per-part ``_COLVAR`` slice is only written once the part's ``mdrun``
    returns, so a still-running free segment has no bias cache and would enter
    the rate estimate with ``γ = 1.0``. This reads the rows belonging to *fname*
    straight out of the trajectory's cumulative ``COLVAR`` instead.

    It is a **read-only** fallback: nothing is written into the trajectory's
    directory, and :func:`aimmd.params._methods._split_cumulative_colvar` is
    neither called nor affected. The extracted rows are handed to
    *bias_function* through a throwaway file in a temporary directory, so the
    ``params.bias_function`` contract (it receives a trajectory filename and
    reads the sibling ``_COLVAR``) is unchanged.

    Parameters
    ----------
    fname : str
        Trajectory part, e.g. ``run1/freeA/traj000002.part0001.xtc``.
    trajectory_extension : str
        Extension including the dot, e.g. ``'.xtc'`` (``params.trajectory_extension``).
    bias_function : callable
        Called as ``bias_function(tmp_trajectory_path)``; must return the
        per-frame bias in kT for the rows it finds, or None.

    Returns
    -------
    numpy.ndarray or None
        Per-frame bias in kT, aligned with the part's frames from frame 0. May
        be **shorter** than the part if PLUMED has not flushed all rows yet —
        callers must treat the uncovered tail as ``exp(bias) = 1``. None when
        the alignment cannot be established (see Notes).

    Notes
    -----
    Returns None when:

    - *fname* is not a ``.partNNNN`` file (shooting paths cache their own bias);
    - the cumulative ``COLVAR`` is absent, or the part is not on disk;
    - *fname* belongs to an **older** trajectory than the one that owns the live
      ``COLVAR``. The file is rotated away when a new trajectory starts, so
      resolving an older part against it would read another trajectory's rows;
    - the ``COLVAR`` holds at least twice the rows the parts account for, which
      means ``PRINT STRIDE`` does not match ``nstxout-compressed`` (warns).

    Guessing an offset in those last two cases would silently misalign bias with
    frames, which is worse than no correction at all.
    """
    from ..params._methods import _part_row_ranges

    stem = os.path.basename(fname)
    if not stem.endswith(trajectory_extension):
        return None
    match = _PART_RE.match(stem[:-len(trajectory_extension)])
    if match is None:
        return None  # not a free-trajectory part (shooting paths cache their own)
    base = match.group('base')
    part = int(match.group('part'))

    deffnm_dir = os.path.dirname(fname) or '.'
    cum_colvar = os.path.join(deffnm_dir, 'COLVAR')
    if not os.path.exists(cum_colvar):
        return None

    # The cumulative COLVAR is per trajectory: when a new trajectory starts, the
    # old file is rotated away. Only the newest trajectory in the directory owns
    # the live COLVAR, so deriving for an earlier one would silently read another
    # trajectory's rows.
    present = set()
    for pf in glob(os.path.join(deffnm_dir, f'*.part????{trajectory_extension}')):
        m = _PART_RE.match(os.path.basename(pf)[:-len(trajectory_extension)])
        if m:
            present.add(m.group('base'))
    if present and base != max(present):
        return None

    header = '#! FIELDS time'
    with open(cum_colvar) as fh:
        for line in fh:
            if line.startswith('#'):
                header = line.rstrip('\n')
                break
    rows = np.loadtxt(cum_colvar, comments='#')
    if rows.ndim == 1:
        rows = rows[None, :]
    if not len(rows):
        return None

    ranges = _part_row_ranges(deffnm_dir, base, trajectory_extension,
                              len(rows), allow_partial=True)
    if part not in ranges:
        return None

    # A stride mismatch (PRINT STRIDE not equal to nstxout-compressed) leaves the
    # COLVAR holding an integer multiple of the rows the parts account for, and
    # every offset is then wrong by that factor. A small surplus is normal —
    # PLUMED can be a row or two ahead of the safely readable frames — so refuse
    # only from the 2x floor, which no flush lag can reach.
    accounted = max(stop for _, stop, _ in ranges.values())
    if accounted and len(rows) >= 2 * accounted:
        warnings.warn(
            f'{cum_colvar!r} holds {len(rows)} rows but the parts of {base!r} '
            f'account for only {accounted}; PRINT STRIDE most likely does not '
            f'match nstxout-compressed, so the row-to-frame offsets cannot be '
            f'trusted. Not deriving bias for {os.path.basename(fname)!r}.',
            UserWarning,
            stacklevel=2)
        return None

    row_start, row_stop, _ = ranges[part]
    sel = rows[row_start:row_stop]
    if not len(sel):
        return None

    tmpdir = tempfile.mkdtemp(prefix='aimmd_bias_')
    try:
        tmp_traj = os.path.join(tmpdir, stem)
        with open(tmp_traj.replace(trajectory_extension, '_COLVAR'), 'w') as fh:
            fh.write(header + '\n')
            for row in sel:
                fh.write(' '.join(f'{v:.6f}' for v in row) + '\n')
        result = bias_function(tmp_traj)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if result is None:
        return None
    return np.asarray(result, dtype=float)


def counted_frame_windows(pathensemble, windows=None, trim_margins=True):
    """
    Resolve the per-path frame window that the rate estimate counts.

    Parameters
    ----------
    pathensemble : PathEnsemble
        Ensemble to inspect.
    windows : (array, array) or None, optional
        Pre-computed ``(start, stop)`` arrays; returned unchanged (as int
        arrays) when given.
    trim_margins : bool, default True
        When False, returns None: ``γ`` then averages over whole paths, which is
        the pre-fix behaviour and is kept only to reproduce old numbers.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) or None
        The half-open windows, or None when whole paths should be used.

    Notes
    -----
    Returns None (whole paths) when *pathensemble* does not expose
    :attr:`~aimmd.pathensemble.PathEnsemble.frame_windows` — e.g. a plain list of
    paths, or a test double. That keeps the function total, but a real
    ``PathEnsemble`` always supplies windows.
    """
    if not trim_margins:
        return None
    if windows is not None:
        starts, stops = windows
        return (np.asarray(starts, dtype=int), np.asarray(stops, dtype=int))
    try:
        starts, stops = pathensemble.frame_windows
    except Exception:
        return None
    return (np.asarray(starts, dtype=int), np.asarray(stops, dtype=int))


def compute_bias_corrections(pathensemble, weights, lengths=None,
                             check=True,
                             threshold=BIAS_CACHE_COVERAGE_THRESHOLD,
                             return_coverage=False,
                             windows=None, trim_margins=True):
    """
    Compute per-path bias correction factors.

    For each path ``i``, computes::

        γᵢ = mean(exp(bias_i))   over the frames the ensemble counts

    where ``bias_i`` is the per-frame bias in kT units loaded from the
    ``path.bias`` cache (``<traj>.bias.npy``). Paths with missing bias data
    fall back to ``γᵢ = 1.0`` (no correction).

    The average runs over :func:`counted_frame_windows`, i.e. over exactly the
    ``Lᵢ = `` :attr:`~aimmd.pathensemble.PathEnsemble.n_frames` frames that the
    rate estimate multiplies ``γᵢ`` by, so that ``Lᵢ · γᵢ`` is
    ``Σ_{counted frames} exp(bias)``. See the module docstring for why the
    difference is one-signed toward faster rates.

    Parameters
    ----------
    pathensemble : PathEnsemble
        The path ensemble to process.
    weights : numpy.ndarray, shape (len(pathensemble),)
        Per-path weights from :meth:`PathEnsemble.reweight`. Only paths with
        ``weights[i] != 0`` need non-trivial correction; the rest are set to 1.0.
    lengths : numpy.ndarray, optional
        Per-path frame counts, as used in the rate estimate
        ``k = 1 / Σ(w · L · γ)`` — normally ``pathensemble.n_frames``. Used
        only to weight the coverage diagnostic. Defaults to ``len(path)``.
    check : bool, default True
        Run the bias-cache coverage check: print the coverage summary and warn
        when part of the ensemble ran on ``γ = 1.0``. On by default so the
        check cannot be forgotten — pass False only to silence a repeat call on
        the same ensemble (e.g. the reverse-direction weights).
    threshold : float, default 0.05
        Missing weighted fraction above which the check escalates from a plain
        report to an explicit "these kinetics are unreliable" warning.
    return_coverage : bool, default False
        If True, also return the bias-cache coverage diagnostic (see below).
        Defaults to False so existing callers keep receiving a bare array.
    windows : (array, array) or None, optional
        Pre-computed ``(start, stop)`` frame windows, one entry per path. When
        omitted they are taken from ``pathensemble.frame_windows``.
    trim_margins : bool, default True
        When False, ``γ`` averages over whole paths — the pre-fix behaviour.
        Kept only to reproduce numbers from before this fix; it makes
        ``Lᵢ · γᵢ`` inconsistent with ``Lᵢ``.

    Returns
    -------
    gammas : numpy.ndarray, shape (len(pathensemble),)
        Per-path correction factors. ``gammas[i] = 1.0`` for paths with zero
        weight or missing bias data.
    coverage : dict
        Only when ``return_coverage=True``. Keys:

        ``n_paths``
            Number of paths with non-zero weight (the ones that reach the rate).
        ``n_missing``
            How many of those had no usable bias array.
        ``n_missing_files``
            Distinct trajectory files behind those paths (several paths commonly
            share one).
        ``frac_paths``
            ``n_missing / n_paths``.
        ``frac_weighted_length``
            Share of ``Σ |w| · L`` carried by paths with **no** bias cache at
            all. **This is the number that matters** — see Notes. A cache that
            is merely shorter than its path is not counted, because
            :meth:`Path._get` zero-pads it (its tail then contributes
            ``exp(0) = 1``, which is the correct weight anyway).
        ``max_inflation``
            ``1 / (1 - frac_weighted_length)``: the largest factor by which the
            reweighted rate can be overestimated because of the gap.
        ``weighted_length_total``, ``weighted_length_missing``
            The raw sums behind ``frac_weighted_length``.
        ``missing_examples``
            Up to three identifiers of uncached paths, for the log.

    Notes
    -----
    - The corrected rate is ``k = 1 / Σ(w · L · γ)`` where ``L`` is the counted
      path length ``pathensemble.n_frames`` and ``γ`` averages ``exp(bias)`` over
      those same frames.
    - For unbiased paths (bias ≈ 0 everywhere), ``exp(0) = 1`` so ``γ = 1`` and
      the formula reduces exactly to the standard AIMMD estimator.
    - Unless ``check=False``, the coverage is printed on every call, and a single
      terse ``UserWarning`` is raised when the missing weighted fraction exceeds
      *threshold*. Smaller gaps are reported but not warned about: a still-running
      free segment and the deliberate ``part0000`` seed produce them every round.
    - Coverage is weighted by ``|w| · L``, not counted per path, because the
      estimator is a sum over ``w · L · γ``: one long uncached free-basin
      trajectory can be 1 path in 1000 and still carry most of the dwell time.
      With a typical correction ``Γ ≫ 1``, a missing weighted fraction ``f``
      inflates the rate by ``≈ 1 / (1 - f)``, hence ``max_inflation``.
    """
    gammas = np.ones(len(pathensemble))
    windows = counted_frame_windows(pathensemble, windows, trim_margins)

    n_weighted = 0
    n_missing = 0
    length_total = 0.0
    length_missing = 0.0
    missing_files = set()

    for i, path in enumerate(pathensemble):
        if weights[i] == 0.0:
            continue

        n_weighted += 1

        if lengths is not None:
            n_frames = float(lengths[i])
        else:
            try:
                n_frames = float(len(path))
            except TypeError:
                n_frames = 0.0
        weighted_length = abs(float(weights[i])) * n_frames
        length_total += weighted_length

        try:
            bias = path._get('bias', raise_if_missing=True)
        except Exception:
            # Cache file missing → fall back to γ = 1.0.
            bias = None

        if bias is None or len(bias) == 0:
            gammas[i] = 1.0
            n_missing += 1
            length_missing += weighted_length
            missing_files.add(_path_label(path))
            continue

        # γᵢ = mean(exp(bias)) over the frames this path contributes to the
        # rate, i.e. the window `n_frames` counts — not the whole path. The two
        # differ by the boundary frames `Path.split` leaves on each block; those
        # sit in the neighbouring state, carry ~zero bias, and would dilute γ
        # while not being counted in L.
        # `Path._get('bias')` zero-pads to the path length (path/_get.py:167,205
        # -> core/utils.extend_array), so a cache shorter than the path already
        # contributes exp(0) = 1 for its tail — no separate frame weighting is
        # needed, and a short cache is indistinguishable from a full one here.
        # Coverage below therefore detects *absent* caches, not short ones.
        if windows is not None:
            start, stop = int(windows[0][i]), int(windows[1][i])
            counted = bias[start:stop]
        else:
            counted = bias
        if len(counted) == 0:
            # nothing counted -> w * L * γ is 0 whatever γ is
            gammas[i] = 1.0
            continue
        gammas[i] = float(np.mean(np.exp(counted)))

    frac_weighted = (length_missing / length_total) if length_total > 0 else 0.0
    coverage = {
        'n_paths': n_weighted,
        'n_missing': n_missing,
        'frac_paths': (n_missing / n_weighted) if n_weighted else 0.0,
        'frac_weighted_length': frac_weighted,
        'max_inflation': (1.0 / (1.0 - frac_weighted)
                          if frac_weighted < 1.0 else float('inf')),
        'weighted_length_total': length_total,
        'weighted_length_missing': length_missing,
        'missing_examples': sorted(missing_files)[:3],
        'n_missing_files': len(missing_files),
    }

    # The check is default behaviour: report every call, and warn — after the
    # loop, so the message can quantify the gap it is warning about.
    if check:
        print(format_bias_cache_coverage(coverage, threshold=threshold))
        if frac_weighted > threshold:
            # Terse, catchable signal for library callers. The explanation and
            # the remediation live in the printed report; saying both twice only
            # taught readers to skim them.
            warnings.warn(
                f'Bias cache coverage {1.0 - frac_weighted:.1%}: '
                f'{frac_weighted:.1%} of the weighted path length entered the '
                f'rate estimate with γ = 1.0, across {n_missing} of '
                f'{n_weighted} paths ({len(missing_files)} files), which can '
                f'overestimate the reweighted rate by up to '
                f'{_inflation_text(coverage["max_inflation"])}. See the bias '
                f'cache coverage report in the log for what to check.',
                UserWarning,
                stacklevel=2)

    if return_coverage:
        return gammas, coverage
    return gammas


def format_bias_cache_coverage(coverage,
                               threshold=BIAS_CACHE_COVERAGE_THRESHOLD,
                               indent='    '):
    """
    Render the bias-cache coverage diagnostic as printable log lines.

    Always reports how much of the weighted path length actually carried a
    Tiwary-Parrinello correction. When the uncorrected share exceeds
    *threshold*, it escalates to a warning block naming the consequence and
    what to change.

    Parameters
    ----------
    coverage : dict
        As returned by :func:`compute_bias_corrections` with
        ``return_coverage=True``.
    threshold : float, default 0.05
        Missing weighted fraction above which the remediation note is shown.
    indent : str, default four spaces
        Prefix for every line, matching the surrounding training log.

    Returns
    -------
    str
        One line for a healthy ensemble; a multi-line block otherwise. No
        trailing newline.

    Notes
    -----
    A high missing fraction is usually a symptom, not a cause: AIMMD writes the
    per-part PLUMED ``_COLVAR`` slice that ``bias_function`` reads only after an
    ``mdrun`` segment returns. A free-basin worker that never leaves its state
    within a job never produces that slice, so the deepest, most heavily biased
    dwell time is exactly the data that ends up uncorrected.
    """
    n_paths = coverage['n_paths']
    frac_missing = coverage['frac_weighted_length']
    inflation = coverage['max_inflation']

    if n_paths == 0:
        return f'{indent}Bias cache coverage: no weighted paths to check'

    lines = [
        f'{indent}Bias cache coverage: {1.0 - frac_missing:.1%} of the weighted '
        f'path length ({n_paths - coverage["n_missing"]}/{n_paths} paths)'
    ]
    if frac_missing <= threshold:
        return '\n'.join(lines)

    body = (
        f'{frac_missing:.1%} of the weighted path length has no bias cache and '
        f'entered the rate estimate with γ = 1.0 — no Tiwary-Parrinello '
        f'correction at all. This can overestimate the reweighted kinetics by up '
        f'to {_inflation_text(inflation)}; treat the bias-reweighted rates as '
        f'unreliable until it is fixed. ' + _REMEDIATION_NOTE)
    wrapped = textwrap.wrap(body, width=78, break_on_hyphens=False)
    lines.append(f'{indent}*** WARNING: {wrapped[0]}')
    lines += [f'{indent}***   {line}' for line in wrapped[1:]]
    if coverage['missing_examples']:
        shown = ', '.join(coverage['missing_examples'])
        more = coverage['n_missing_files'] - len(coverage['missing_examples'])
        suffix = f' (+{more} more files)' if more > 0 else ''
        lines.append(f'{indent}***   uncovered: {shown}{suffix}')
    return '\n'.join(lines)


def bias_reweighted_rates(pathensemble, weights1, weights2, lengths=None,
                          states='ARB', reactive_threshold=0.5,
                          coverage_threshold=BIAS_CACHE_COVERAGE_THRESHOLD,
                          windows=None, trim_margins=True, label=''):
    """
    Run the whole Tiwary-Parrinello reweighting step and report both rates.

    This is the single implementation of the bias-reweighted rate estimate. It
    exists because the same six lines used to be written out at four places in
    :mod:`aimmd.worker._train` (the single-system trainer, the multi-system
    trainer, and the two kinetics-convergence scans), which is how the frame-set
    mismatch between ``L`` and ``γ`` could be fixed in one place and survive in
    three.

    Parameters
    ----------
    pathensemble : PathEnsemble
        The ensemble whose rates are wanted. Bias caches must already be in
        place (``Worker._cache_bias_files`` in file mode).
    weights1, weights2 : numpy.ndarray
        Reweighting weights for ``states`` and for ``states[::-1]``.
    lengths : numpy.ndarray, optional
        Counted frames per path. Defaults to
        :attr:`~aimmd.pathensemble.PathEnsemble.n_frames`, which is what the
        estimator is defined with. A caller-supplied array that disagrees with
        the frame windows is reported, because then ``L · γ`` is not a frame sum.
    states : str, optional
        Three-character state string, e.g. ``'ARB'``.
    reactive_threshold : float, optional
        Passed to :func:`check_reactive_bias`.
    coverage_threshold : float, optional
        Passed to :func:`compute_bias_corrections`.
    windows, trim_margins
        Passed to :func:`counted_frame_windows`.
    label : str, optional
        Prefix for the printed lines, e.g. ``"[system 'G2'] "`` in a
        multi-system run. Kept so the log format is unchanged.

    Returns
    -------
    k12_rw : float
        ``1 / Σ(w1 · L · γ1)`` in ``[1/dt]``, or nan if the sum is zero.
    k21_rw : float
        The same for the reverse direction.
    gamma1, gamma2 : numpy.ndarray
        The per-path correction factors, for callers that want to inspect them.
    """
    # the flushing print, as used by the trainer these lines moved out of
    from .._config import print

    windows = counted_frame_windows(pathensemble, windows, trim_margins)
    if lengths is None:
        if windows is not None:
            lengths = windows[1] - windows[0]
        else:
            lengths = np.array([len(path) for path in pathensemble])
    lengths = np.asarray(lengths)
    if windows is not None:
        counted = windows[1] - windows[0]
        if len(counted) == len(lengths) and not np.array_equal(counted,
                                                              lengths):
            warnings.warn(
                f'{label}the supplied path lengths differ from the counted '
                f'frame windows for '
                f'{int(np.count_nonzero(counted != lengths))} of '
                f'{len(lengths)} paths, so L * gamma is not a sum of exp(bias) '
                f'over any single frame set. Pass lengths=pathensemble.n_frames '
                f'(the default) to keep the estimator consistent.',
                UserWarning, stacklevel=2)

    check_reactive_bias(pathensemble, states, reactive_threshold)
    gamma1 = compute_bias_corrections(
        pathensemble, weights1, lengths=lengths,
        threshold=coverage_threshold,
        windows=windows, trim_margins=trim_margins)
    # the coverage report is identical for both directions: print it once
    gamma2 = compute_bias_corrections(
        pathensemble, weights2, lengths=lengths, check=False,
        threshold=coverage_threshold,
        windows=windows, trim_margins=trim_margins)

    denominator1 = float(np.sum(weights1 * lengths * gamma1))
    denominator2 = float(np.sum(weights2 * lengths * gamma2))
    k12_rw = 1.0 / denominator1 if denominator1 else float('nan')
    k21_rw = 1.0 / denominator2 if denominator2 else float('nan')
    print(f'    {label}k12 bias-reweighted: {k12_rw:.3e} [1/dt]')
    print(f'    {label}k21 bias-reweighted: {k21_rw:.3e} [1/dt]')
    return k12_rw, k21_rw, gamma1, gamma2
