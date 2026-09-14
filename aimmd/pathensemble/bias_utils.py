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

- :func:`check_bias_zero_point` — validates that the *recorded* bias is zero where
  no bias was applied, catching a wrong constant offset in ``params.bias_function``
  (which multiplies every γ, and therefore the rate, by a constant factor).

- :func:`report_nonequilibrium_seeds` — reports whether the free first passages
  behind the rate were accelerated as much as an equilibrated trajectory in that
  basin would have been, and whether their bias-reweighted durations are
  memoryless. Both fail loudly when the free worker's restart scheme is re-seeding
  at the state boundary faster than the basin can equilibrate.

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
import shlex
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
    for attribute in ('fnames', '_fnames'):
        fnames = getattr(path, attribute, None)
        if fnames is None:
            continue
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


# ════════════════════════════════════════════════════════════════════════════
# Bias zero point
# ════════════════════════════════════════════════════════════════════════════

BIAS_ZERO_POINT_TOLERANCE = 0.05
"""Largest |median| (kT) accepted as "zero", and the width within which two
cache files count as sharing one zero point. exp(0.05) = 1.05."""

ZERO_POINT_MIN_FILE_FRAMES = 3
"""Reactive frames a cache file needs before it may vote on a zero point.
The floor of one or two frames is those frames' value, tail or not."""

ZERO_POINT_MIN_FLOOR_MASS = 0.5
"""Share of a file's reactive frames that must sit within `tolerance` of its own
floor before that floor counts as the file's zero point.

This is the shape test that separates a stale cache from a file whose reactive
frames happen to lie in the genuine near-boundary band. A cache written with a
different `bias_function` has ALL of its bias-free frames displaced together, so
its floor is where its mass is; a file that only clipped the fill has its frames
spread over several kT with nothing at the floor."""

ZERO_POINT_MIN_LEVEL_FILES = 2
"""Voting files that make a group a zero point regardless of its frame share."""

ZERO_POINT_MIN_LEVEL_FRACTION = 0.01
"""Frame share that makes a group a zero point regardless of its file count.

The two criteria are an OR on purpose, and both directions are documented: the
file count catches a small stale batch that a share threshold alone would drop
(the silent-pass this check exists to close), the frame share catches a SINGLE
stale file that carries a lot of the ensemble - the .part0001 of a long free
trajectory. A group meeting neither is reported as an informational line rather
than silently discarded."""

ZERO_POINT_MAX_FILES_LISTED = 20
"""Longest explicit `rm` printed in the report. The complete list is always in
`result['fix_command']`, which is never truncated."""

_STOP_DELETE_RESTART = (
    'STOP the run, DELETE the caches, then RESTART - in that order. Deleting '
    'under a live worker changes nothing: the in-memory NPY cache never '
    're-stats the file it memoised and is only cleared when a worker starts. '
    'The values are then recomputed from the untouched *_COLVAR files; a cache '
    'whose COLVAR is gone comes back with gamma = 1 and is named by the bias '
    'cache coverage report instead. This applies to bias_source=\'file\'; under '
    '\'reader\' the bias lives on the Path and PathEnsemble.compute refills only '
    'falsy frames, so a stale non-zero frame is not revisited by a delete at '
    'all.')

_NOT_A_REPAIR = (
    'This report makes the failure legible; it does not repair the run, and it '
    'will keep firing every round until the caches are rebuilt. Nothing '
    'fingerprints params.bias_function at any writer of a *.bias.npy, so a '
    'future mid-campaign edit produces this same mixture again.')


# ── helpers ─────────────────────────────────────────────────────────────────

def _plural(count, singular, plural=None):
    """``3 cache files`` / ``1 cache file``, so report text reads as English."""
    return f'{count} {singular if count == 1 else (plural or singular + "s")}'


def _warning_block(body, indent):
    """Wrap *body* into the module's ``*** WARNING:`` / ``***   `` block."""
    wrapped = textwrap.wrap(body, width=78, break_on_hyphens=False)
    return ([f'{indent}*** WARNING: {wrapped[0]}']
            + [f'{indent}***   {line}' for line in wrapped[1:]])


def _wrap_lines(body, indent, width=74):
    """Wrap *body* into continuation lines of an already-open warning block."""
    return [f'{indent}***   {line}'
            for line in textwrap.wrap(body, width=width, break_on_hyphens=False)]


def _frame_files(path, n):
    """Per-frame source-file names for the first *n* frames, and whether the
    per-file census had to fall back to one pseudo-file per path.

    ``aimmd.path.Path.filenames`` is per-frame and aligned with the arrays
    ``_get`` returns, which is what lets the census attribute a frame to the
    cache file it came from. When it is unavailable the census degrades to the
    per-path unit -- still a working (weaker) detector, but the file-level
    remedy must then be suppressed rather than printed wrong.
    """
    try:
        names = [str(name) for name in path.filenames]
    except Exception:
        names = []
    if len(names) < n:
        return np.asarray([_path_label(path)] * n, dtype=object), True
    return np.asarray(names[:n], dtype=object), False


def _cache_name(fname):
    """The bias cache written next to trajectory *fname*."""
    return f'{fname}.bias.npy'


def _delete_command(fnames):
    """A pasteable ``rm`` of those trajectories' bias caches, never truncated."""
    return 'rm -f ' + ' '.join(shlex.quote(_cache_name(f)) for f in fnames)


def _zero_point_levels(files, tolerance,
                       min_file_frames=ZERO_POINT_MIN_FILE_FRAMES,
                       min_floor_mass=ZERO_POINT_MIN_FLOOR_MASS,
                       min_level_files=ZERO_POINT_MIN_LEVEL_FILES,
                       min_fraction=ZERO_POINT_MIN_LEVEL_FRACTION):
    """Group per-file reactive-bias floors into the distinct zero points present.

    Returns ``(levels, unplaced, no_reactive, dropped)``:

    levels
        ascending in ``value``; each ``{'value', 'n_files', 'n_frames',
        'files'}`` with every ASSIGNED file, not only the voting ones.
    unplaced
        files that have reactive frames but whose floor matches no level -
        anomalous, and what makes a deletion list a lower bound.
    no_reactive
        files with no bias-free frame at all. Benign and structural (every
        pure-A or pure-B block, and the deliberate ``part0000`` seeds): their
        zero point is simply not readable, and their presence must not escalate
        the remedy.
    dropped
        groups that were too small to call a zero point, reported rather than
        discarded.
    """
    voters = []
    for name, entry in files.items():
        if entry['n_frames'] < min_file_frames or not np.isfinite(entry['floor']):
            continue
        if entry['floor_mass'] < min_floor_mass:
            continue                       # a tail, not a zero point
        voters.append((entry['floor'], name))
    voters.sort()
    if not voters:
        return ([],
                sorted(n for n, e in files.items() if e['n_frames']),
                sorted(n for n, e in files.items() if not e['n_frames']),
                [])

    # a level is at most one tolerance wide: opened at its lowest floor and
    # closed at the first floor beyond it, never chained neighbour-to-neighbour,
    # so a slow drift cannot merge two real zero points into one wide level
    groups = []
    for floor, name in voters:
        if not groups or floor - groups[-1][0][0] > tolerance:
            groups.append([])
        groups[-1].append((floor, name))

    n_voted = sum(files[n]['n_frames'] for _, n in voters)
    levels, dropped = [], []
    for group in groups:
        n_frames = sum(files[n]['n_frames'] for _, n in group)
        value = float(np.median([f for f, _ in group]))
        if len(group) >= min_level_files or n_frames >= min_fraction * n_voted:
            levels.append({'value': value, 'n_files': 0, 'n_frames': 0,
                           'files': []})
        else:
            dropped.append({'value': value, 'n_files': len(group),
                            'n_frames': n_frames,
                            'files': sorted(n for _, n in group)})

    unplaced, no_reactive = [], []
    for name in sorted(files):
        entry = files[name]
        if not entry['n_frames']:
            no_reactive.append(name)
            continue
        nearest = (min(levels, key=lambda lv: abs(entry['floor'] - lv['value']))
                   if levels and np.isfinite(entry['floor']) else None)
        if nearest is None or abs(entry['floor'] - nearest['value']) > tolerance:
            unplaced.append(name)
            continue
        nearest['files'].append(name)
        nearest['n_files'] += 1
        nearest['n_frames'] += entry['n_frames']
    return levels, unplaced, no_reactive, dropped


# ── the check ───────────────────────────────────────────────────────────────

def check_bias_zero_point(pathensemble, states,
                          tolerance=BIAS_ZERO_POINT_TOLERANCE,
                          indent='    '):
    """
    Check that the *recorded* bias is zero where no bias was applied.

    ``path.bias`` must hold ``beta*V_bias`` with the bias-free region at exactly
    zero, because gamma = <exp(bias)> is absolute: adding a constant ``c`` to
    every recorded value multiplies every gamma, and therefore the reweighted
    rate, by ``exp(c)``. Nothing else in the pipeline can notice.

    Three separate questions, because their remedies differ:

    - **Is there one zero point?** Each ``<traj>.bias.npy`` is written in one
      call with one ``params.bias_function``, so the floor of a file's reactive
      frames is the zero point that file carries. More than one level is the
      signature of a ``bias_function`` edited mid-campaign: nothing invalidates
      a cache that is already complete. A mixture has **no single offset** - one
      subset of the gammas is wrong by a constant factor while the rest are
      right - so the remedy is a list of caches to rebuild, not a number.
    - **Is that zero point 0?** The median over the reactive frames. The only
      failure a scalar correction repairs.
    - **Is any recorded bias negative?** A fill that raises the energy inside
      the states gives V >= 0, over every frame, not only the reactive ones.

    Contract for callers::

        np.isfinite(factor)  <=>  `census_conclusive`
                             <=>  the per-file census placed every cache file
                                  that has reactive frames on one and the same
                                  zero point
                             <=>  dividing every rate by `factor` is a valid
                                  correction (a no-op when `median_ok`)

    which is why :func:`bias_reweighted_rates` prints its corrected rates under
    ``not median_ok and np.isfinite(factor)``. The census is deliberately the
    gate rather than ``uniform_ok``: ``len(levels) <= 1`` is also what an
    ensemble the census could not read looks like - every cache below the
    voting bar of ``ZERO_POINT_MIN_FILE_FRAMES``, or a floor matching no level
    - and a mixture like that must not be published as one clean offset.

    Parameters
    ----------
    pathensemble : PathEnsemble
        Ensemble to inspect (bias caches already in place).
    states : str
        Three-character state string, e.g. ``'ARB'``; ``states[1]`` is the
        bias-free reactive region.
    tolerance : float, default 0.05
        Largest ``|median|`` (kT) accepted as "zero". Doubles as the width
        within which two cache files count as sharing one zero point, and as
        the slack on the ``V >= 0`` test.
    indent : str, default four spaces
        Prefix for the printed lines.

    Returns
    -------
    dict
        ``median``, ``minimum``, ``n_frames``, ``report`` - as before.
        ``ok`` - ``median_ok and uniform_ok and positive_ok``.
        ``median_ok``, ``uniform_ok``, ``positive_ok`` - its three components.
        ``census_conclusive`` - the census placed every cache file with
        reactive frames on a single level; what gates ``offset``/``factor``.
        ``offset``, ``factor`` - the constant to subtract and ``exp(-offset)``;
        **nan unless** ``census_conclusive``, i.e. nan for a mixture and for an
        ensemble in which the census established nothing.
        ``gap``, ``gap_factor`` - kT from the level at zero (from zero itself
        when no level is there) to the furthest other level, and ``exp(|gap|)``;
        nan unless mixed.
        ``levels``, ``n_levels``, ``n_files``.
        ``mismatched_files`` - every file off the level that is at zero
        (every placed file when no level is; trajectory names, so their caches
        are those names + ``.bias.npy``).
        ``mismatched_fraction``.
        ``negative_files``, ``n_negative_frames``.
        ``unplaced_files`` - files with reactive frames whose floor matched no
        level; any at all makes ``mismatched_files`` a LOWER BOUND.
        ``no_reactive_files`` - files with no bias-free frame (benign).
        ``dropped_levels`` - groups too small to call a zero point.
        ``census_degraded`` - the per-file census fell back to per-path.
        ``fix_command`` - complete, never truncated ``rm`` of the caches to
        rebuild, or None; ``fix_command_is_complete`` says whether it suffices.
    """
    r = states[1] if len(states) >= 2 else states[0]
    r_bias = []
    minimum = np.inf
    n_negative = 0
    files = {}
    degraded = False

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
        path_bias = np.asarray(path_bias[:n], dtype=float)
        minimum = min(minimum, float(np.min(path_bias)))
        n_negative += int(np.count_nonzero(path_bias < -tolerance))
        mask = np.asarray(path_states[:n]) == r
        if mask.any():
            r_bias.append(path_bias[mask])

        names, fell_back = _frame_files(path, n)
        degraded = degraded or fell_back
        for name in set(names.tolist()):
            here = names == name
            entry = files.setdefault(str(name),
                                     {'n_frames': 0, 'floor': np.inf,
                                      'floor_mass': 0.0, 'minimum': np.inf,
                                      '_r': []})
            entry['minimum'] = min(entry['minimum'],
                                   float(np.min(path_bias[here])))
            here_r = here & mask
            if here_r.any():
                entry['_r'].append(path_bias[here_r])

    for entry in files.values():
        if entry['_r']:
            block = np.concatenate(entry['_r'])
            entry['n_frames'] = int(len(block))
            entry['floor'] = float(np.min(block))
            entry['floor_mass'] = float(
                np.mean(np.abs(block - entry['floor']) <= tolerance))
        del entry['_r']

    if not r_bias:
        result = _empty_result(indent, files)
        print(result['report'])
        return result

    r_bias = np.concatenate(r_bias)
    median = float(np.median(r_bias))
    minimum = float(minimum) if np.isfinite(minimum) else float('nan')

    levels, unplaced, no_reactive, dropped = _zero_point_levels(files, tolerance)
    negative_files = sorted(name for name, e in files.items()
                            if e['minimum'] < -tolerance)

    # The correct zero point is 0 by definition - that is the premise the whole
    # check rests on - so the reference level is the one AT zero, never the one
    # carrying the most frames. Picking by population inverts the remedy exactly
    # in the regime this check was written for: once the stale caches hold more
    # than half the placed reactive frames, every already-correct cache lands in
    # `mismatched_files` and in the printed `rm`, and the direction words invert
    # with it. On calixarene_G2_opes_v5 that was true for most of the campaign.
    if len(levels) <= 1:
        reference = levels[0] if levels else None
        others = []
    else:
        at_zero = [lv for lv in levels if abs(lv['value']) <= tolerance]
        # no level at zero (or two of them inside one tolerance): no level is
        # the right one, so every placed cache file has to be rebuilt
        reference = at_zero[0] if len(at_zero) == 1 else None
        others = [lv for lv in levels if lv is not reference]
    mismatched_files = sorted(f for lv in others for f in lv['files'])
    n_mismatched = sum(lv['n_frames'] for lv in others)
    n_placed = sum(lv['n_frames'] for lv in levels)
    mismatched_fraction = (n_mismatched / n_placed) if n_placed else 0.0

    median_ok = abs(median) <= tolerance
    uniform_ok = len(levels) <= 1
    positive_ok = not (minimum < -tolerance)
    ok = median_ok and uniform_ok and positive_ok

    # One level is evidence of ONE zero point only when every cache file that
    # has reactive frames sits ON it. A file the census could not place - too
    # few reactive frames to vote, or a floor matching no level - carries a zero
    # point the census never saw, so `len(levels) <= 1` is then the absence of a
    # finding rather than a finding, and a mixture whose caches all fall below
    # the voting bar would otherwise be published as a single clean offset. A
    # degraded (per-path) census is excluded for the same reason: two paths
    # sharing a first file collapse into one pseudo-cache and hide a mixture
    # between them.
    census_conclusive = len(levels) == 1 and not unplaced and not degraded

    # bound here, where the levels are - not inside the branch that prints them
    if others:
        base = reference['value'] if reference is not None else 0.0
        worst = max(others, key=lambda lv: abs(lv['value'] - base))
        gap = float(worst['value'] - base)
        gap_factor = float(np.exp(abs(gap)))
    else:
        gap = float('nan')
        gap_factor = float('nan')

    # `offset`/`factor` describe a shift of ONE zero point, so they exist only
    # when the census established that there is one. nan otherwise - which is
    # the second guard that stops bias_reweighted_rates printing a "corrected"
    # rate for a mixture, or for an ensemble the census could not read.
    # errstate: a units error in bias_function (kJ/mol read as kT) overflows
    # exp() to inf, which prints as 'inf' and is the right answer; a numpy
    # RuntimeWarning next to it in the log is not.
    with np.errstate(over='ignore', under='ignore'):
        offset = median if census_conclusive else float('nan')
        factor = float(np.exp(-offset)) if census_conclusive else float('nan')
        # What a constant offset of `median` would do to every rate. Report
        # text only, and taken from the median directly: `factor` is withheld
        # above whenever one constant is not established, and exp(|median|)
        # also cannot divide by an underflowed exp(-median) the way
        # max(factor, 1.0 / factor) did (ZeroDivisionError above ~746 kT).
        median_inflation = float(np.exp(abs(median)))

    # what the per-file census may and may not be quoted as having found
    if census_conclusive:
        census_clause = (f'the census over {_plural(len(files), "cache file")} '
                         f'finds no second zero point')
    else:
        census_clause = (f'the per-file census could not establish that this is '
                         f'the only zero point '
                         f'({_plural(len(levels), "level")} found, '
                         f'{_plural(len(unplaced), "cache file")} on none of '
                         f'them)')

    complete = not unplaced and not degraded
    if mismatched_files:
        fix_command = _delete_command(mismatched_files)
    elif not median_ok and files and not degraded:
        fix_command = _delete_command(sorted(files))
    else:
        fix_command = None

    lines = [f'{indent}Bias zero point: median recorded bias in {r!r} = '
             f'{median:+.3f} kT over {len(r_bias)} frames '
             f'(min over all frames {minimum:+.3f} kT)']
    warning = None

    if not uniform_ok:
        # every phrase that depended on "the majority" is phrased against zero
        anchor = ('the level that is at zero, where the bias-free region '
                  'belongs' if reference is not None else
                  'zero - and NO level here is at zero, so every placed cache '
                  'carries a wrong one')
        rest = ('while the rest are right' if reference is not None else
                'and none of the others is right either')
        body = (
            f'the recorded bias has {len(levels)} different zero points, '
            f'{abs(gap):.3f} kT apart: {len(mismatched_files)} of '
            f'{_plural(len(files), "bias cache file")} ({n_mismatched} of '
            f'{n_placed} placed frames in {r!r}, {mismatched_fraction:.1%}) '
            f'were written with a floor {abs(gap):.3f} kT '
            f'{"below" if gap < 0 else "above"} {anchor}. gamma = '
            f'<exp(bias)> is absolute, so those files\' gamma are a factor '
            f'{gap_factor:.3f} too {"small" if gap < 0 else "large"} {rest}, '
            f'and the bias-reweighted rates below are too '
            f'{"fast" if gap < 0 else "slow"} by somewhere between 1 and '
            f'{gap_factor:.3f}, depending on how much of Sum(w*L*gamma) those '
            f'files carry. A mixture has NO single offset, so no zero-point '
            f'corrected rate is printed below: scaling the whole ensemble by '
            f'one number would leave the correct files wrong in the other '
            f'direction. This is what editing params.bias_function '
            f'mid-campaign leaves behind - register_path writes a shooting '
            f'path\'s cache once and never revisits it, and _cache_bias_files '
            f'rewrites a <traj>.bias.npy only while it is SHORTER than its '
            f'trajectory, so every cache that was already complete kept the '
            f'old zero point. For a PLUMED OPES fill floored at -BARRIER, one '
            f'BARRIER change in the shift is exactly this gap.')
        lines += _warning_block(body, indent)
        for level in levels:
            tag = ' (at zero)' if level is reference else ''
            lines.append(f'{indent}***   zero point {level["value"]:+.3f} kT: '
                         f'{_plural(level["n_files"], "file")}, '
                         f'{_plural(level["n_frames"], "frame")} '
                         f'in {r!r}{tag}')
        lines += _rebuild_lines(mismatched_files, files, indent,
                                complete=complete, degraded=degraded,
                                unplaced=unplaced)
        warning = (
            f'Bias zero point is not unique: {len(levels)} zero points in the '
            f'recorded bias, {abs(gap):.3f} kT apart. {len(mismatched_files)} '
            f'of {_plural(len(files), "bias cache")} carry the wrong one, so '
            f'their gamma is wrong by a factor {gap_factor:.3f} and no single '
            f'offset can repair it - those caches have to be rebuilt. See the '
            f'bias zero point report in the log.')

    elif not median_ok:
        # the offset is only a correction if it applies to every cache, which
        # is exactly what an inconclusive census has not shown
        hedge = ('' if census_conclusive else
                 'No zero-point corrected rate is printed below: one constant '
                 'is a correction only if every cache carries the same zero '
                 'point, and the census could not establish that. ')
        body = (
            f'the recorded bias is not zero where no bias was applied: its '
            f'median over the {len(r_bias)} frames in {r!r} is {median:+.3f} kT '
            f'(min over all frames {minimum:+.3f} kT), and {census_clause}. '
            f'gamma = <exp(bias)> is absolute, so a constant offset c '
            f'multiplies every gamma by exp(c): the bias-reweighted rates '
            f'printed below are too {"fast" if median < 0 else "slow"} by a '
            f'factor {median_inflation:.3f}. {hedge}Subtract {median:+.3f} '
            f'kT ({-median:+.3f} to be added) inside params.bias_function - '
            f'for a PLUMED OPES fill floored at -BARRIER this is exactly a '
            f'wrong BARRIER in the shift - and then rebuild every bias cache, '
            f'because a cache that is already complete keeps the old zero '
            f'point and the ensemble would otherwise mix two of them.')
        lines += _warning_block(body, indent)
        lines += _rebuild_lines(sorted(files), files, indent,
                                complete=complete, degraded=degraded,
                                unplaced=unplaced, whole_run=True)
        warning = (
            f'Bias zero point off by {median:+.3f} kT: every gamma, and every '
            f'bias-reweighted rate, is wrong by a factor '
            f'{median_inflation:.3f}. See the bias zero point report '
            f'in the log.')

    elif not positive_ok:
        body = (
            f'{_plural(n_negative, "recorded bias value")} '
            f'{"is" if n_negative == 1 else "are"} negative (minimum '
            f'{minimum:+.3f} kT, over all frames, not only {r!r}), which a '
            f'fill that raises the energy inside the states cannot produce: '
            f'V >= 0 by construction. The median in {r!r} is {median:+.3f} kT '
            f'and {census_clause}, so this is neither a constant offset nor a '
            f'mixture and no scalar correction applies - none is printed '
            f'below. Look for a wrong sign, a wrong column or a wrong kT in '
            f'params.bias_function, or a truncated cache, in:')
        lines += _warning_block(body, indent)
        shown = negative_files[:3]
        more = len(negative_files) - len(shown)
        suffix = f' (+{more} more files)' if more > 0 else ''
        lines.append(f'{indent}***   negative in: {", ".join(shown)}{suffix}')
        warning = (
            f'Bias zero point: {_plural(n_negative, "recorded bias value")} '
            f'{"is" if n_negative == 1 else "are"} negative (minimum '
            f'{minimum:+.3f} kT) across '
            f'{_plural(len(negative_files), "cache file")}, which a fill that '
            f'only raises the energy cannot produce. The zero point itself is '
            f'fine, so no constant offset applies. See the bias zero point '
            f'report in the log.')

    elif levels:
        # a complete partition of the cache files: on the level, no bias-free
        # frame at all, and - the category this line used to omit - on no
        # recognised level, which is what withholds `factor` above
        unaccounted = (f', {len(unplaced)} on no recognised zero point'
                       if unplaced else '')
        lines.append(f'{indent}Bias zero point: one zero point at '
                     f'{levels[0]["value"]:+.3f} kT across '
                     f'{levels[0]["n_files"]} of {len(files)} cache files '
                     f'({len(no_reactive)} hold no bias-free frame'
                     f'{unaccounted})')

    for group in dropped:
        lines.append(f'{indent}    note: {group["n_files"]} cache file(s) floor '
                     f'at {group["value"]:+.3f} kT over {group["n_frames"]} '
                     f'frames in {r!r} - below both level thresholds, reported '
                     f'not dropped: {", ".join(group["files"][:3])}')
    if unplaced:
        lines.append(f'{indent}    note: {len(unplaced)} cache file(s) have '
                     f'reactive frames on no recognised zero point: '
                     f'{", ".join(unplaced[:3])} - any deletion list above is '
                     f'a lower bound')
    if degraded:
        lines.append(f'{indent}    note: per-frame filenames unavailable, so '
                     f'the census ran per PATH, not per cache file; no '
                     f'deletion list is printed from a degraded census')

    result = {
        'median': median, 'minimum': minimum, 'n_frames': len(r_bias),
        'offset': offset, 'factor': factor, 'ok': bool(ok),
        'median_ok': bool(median_ok), 'uniform_ok': bool(uniform_ok),
        'positive_ok': bool(positive_ok),
        'census_conclusive': bool(census_conclusive),
        'gap': gap, 'gap_factor': gap_factor,
        'levels': levels, 'n_levels': len(levels), 'n_files': len(files),
        'mismatched_files': mismatched_files,
        'mismatched_fraction': float(mismatched_fraction),
        'negative_files': negative_files, 'n_negative_frames': n_negative,
        'unplaced_files': unplaced, 'no_reactive_files': no_reactive,
        'dropped_levels': dropped, 'census_degraded': bool(degraded),
        'fix_command': fix_command,
        'fix_command_is_complete': bool(fix_command is not None and complete),
        'report': '\n'.join(lines),
    }
    print(result['report'])
    if warning is not None:
        warnings.warn(warning, UserWarning, stacklevel=2)
    return result


def _empty_result(indent, files):
    """Early return for an ensemble with no reactive frames.

    Carries EVERY key, so a caller may index the dict unconditionally.
    """
    return {
        'median': float('nan'), 'minimum': float('nan'), 'n_frames': 0,
        'offset': 0.0, 'factor': 1.0, 'ok': True,
        'median_ok': True, 'uniform_ok': True, 'positive_ok': True,
        # no reactive frame anywhere: there is no zero point to get wrong, so
        # the no-op correction below is valid and the contract on `factor` holds
        'census_conclusive': True,
        'gap': float('nan'), 'gap_factor': float('nan'),
        'levels': [], 'n_levels': 0, 'n_files': len(files),
        'mismatched_files': [], 'mismatched_fraction': 0.0,
        'negative_files': [], 'n_negative_frames': 0,
        'unplaced_files': [], 'no_reactive_files': sorted(files),
        'dropped_levels': [], 'census_degraded': False,
        'fix_command': None, 'fix_command_is_complete': False,
        'report': f'{indent}Bias zero point: no reactive frames to check',
    }


def _rebuild_lines(fnames, files, indent, *, complete, degraded, unplaced,
                   whole_run=False):
    """Name the caches to rebuild, and give the exact command only when it is
    provably the whole fix.

    An explicit ``rm`` is printed only when the list is complete and short: it
    is the least destructive repair and keeps every already-correct cache.
    Otherwise the remedy is prose, never a partial ``rm`` and never a
    ``find ... -delete`` (whose root would be a guess, and which in a
    multi-system layout can reach outside the run). The complete list always
    stays in ``result['fix_command']``.
    """
    if not fnames:
        return _wrap_lines('no cache file could be attributed, so no rebuild '
                           'list can be given. ' + _STOP_DELETE_RESTART, indent)

    caches = [shlex.quote(_cache_name(name)) for name in fnames]
    if complete and len(fnames) <= ZERO_POINT_MAX_FILES_LISTED:
        head = (f'rebuild {_plural(len(fnames), "bias cache")}'
                f'{" - every cache in the run" if whole_run else ""}. '
                + _STOP_DELETE_RESTART)
        lines = _wrap_lines(head, indent)
        body = (['rm -f \\'] + [f'  {c} \\' for c in caches[:-1]]
                + [f'  {caches[-1]}'])
        lines += [f'{indent}***     {line}' for line in body]
    else:
        if degraded:
            why = ('the per-file census degraded to per-path, so a file-level '
                   'list would name paths, not caches')
        elif unplaced:
            why = (f'{_plural(len(unplaced), "further cache file")} '
                   f'{"has" if len(unplaced) == 1 else "have"} '
                   f'reactive frames on no recognised level, so this list is a '
                   f'LOWER BOUND and deleting only these would leave a mixture')
        else:
            why = (f'too many to list every round (the complete list is in '
                   f'the returned fix_command)')
        head = (f'{len(fnames)} of the {len(files)} bias caches must be '
                f'rebuilt, starting with '
                f'{", ".join(_cache_name(f) for f in fnames[:3])} - {why}, so '
                f'delete every *.bias.npy under the run directory instead and '
                f'let them all be recomputed; a cache that was already correct '
                f'recomputes to the same values. ' + _STOP_DELETE_RESTART)
        lines = _wrap_lines(head, indent)
    lines += _wrap_lines(_NOT_A_REPAIR, indent)
    return lines


# ════════════════════════════════════════════════════════════════════════════
# Non-equilibrium free-basin seeds
# ════════════════════════════════════════════════════════════════════════════

SEED_BOOST_FRACTION = 0.5
"""Smallest realised/equilibrium acceleration accepted for one first passage.

Per free trajectory, ``Γᵢ = Σ(L·γ)/Σ(L)`` over its in-basin blocks is the boost
it actually got; ``Γ_eq`` is the frame-weighted pooled value over every free
trajectory of that basin, i.e. what an equilibrated trajectory gets. A passage
that escaped without ever sampling the basin's bias distribution has
``Γᵢ/Γ_eq ≪ 1``. Measured: G4 calixarene 0.69-1.14 across 68 passages in three
replicates; calixarene-G2 0.13-0.24 for the boundary-limited ones against
1.05-1.28 for the equilibrated ones. 0.5 sits in an empty gap on both sides."""

SEED_BOOST_MISS_THRESHOLD = 0.2
"""Fraction of first passages that may fall below ``SEED_BOOST_FRACTION``.

G4 control: 1 of 68 (1.5 %). G2-v2: 4 of 5 (80 %). G2-v3: 10 of 12 (83 %)."""

SEED_MEMORYLESS_THRESHOLD = 0.35
"""Smallest median/mean ratio of the reweighted first-passage times accepted.

An exponential has median/mean = ln2 = 0.693; sampling noise at n ~ 30 keeps it
roughly within 0.5-0.9. Values far below indicate a spike of near-zero passages
on top of a few long ones, i.e. a start distribution that is not the basin's.
Measured: G4 control 0.850, OPES flooding 0.779, G2-v2 0.023, G2-v3 0.068."""

_FREE_PATH_RE = re.compile(
    r'(?P<prefix>.*)free(?P<state>[^/]+)/(?P<traj>traj\d+)\.part\d+$')


def _free_trajectory_key(path):
    """Identify the free trajectory a split block came from.

    Returns ``(key, state)`` where *key* uniquely names the free trajectory
    (directory + ``traj??????``) and *state* is its target-state folder label, or
    ``(None, None)`` for anything that is not a free-simulation part (shooting
    chain paths, initial paths, in-memory paths).
    """
    fnames = getattr(path, '_fnames', None) or getattr(path, 'fnames', None)
    if fnames is None:
        return None, None
    try:
        fname = str(fnames[0])
    except (IndexError, TypeError):
        return None, None
    match = _FREE_PATH_RE.match(os.path.splitext(fname)[0])
    if match is None:
        return None, None
    state = match.group('state')
    if len(state) != 1:
        return None, None
    return f"{match.group('prefix')}free{state}/{match.group('traj')}", state


def _ks_exponential(times):
    """One-sample KS distance of *times* from an exponential fitted to them.

    Returns ``(D, D_crit)``. The rate is estimated from the sample, so the
    textbook Kolmogorov p-value does not apply; *D_crit* is the approximate 5 %
    Lilliefors critical value for the exponential with estimated mean,
    ``1.094/sqrt(n)``. Both are nan for fewer than three finite times.
    """
    x = np.sort(np.asarray([t for t in times if np.isfinite(t)], dtype=float))
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    mean = float(np.mean(x))
    if not mean > 0.0:
        return float('nan'), float('nan')
    cdf = 1.0 - np.exp(-x / mean)
    i = np.arange(1, n + 1, dtype=float)
    d = max(float(np.max(i / n - cdf)), float(np.max(cdf - (i - 1) / n)))
    return d, 1.094 / np.sqrt(n)


def report_nonequilibrium_seeds(pathensemble, lengths, gammas, states='ARB',
                                boost_fraction=SEED_BOOST_FRACTION,
                                miss_threshold=SEED_BOOST_MISS_THRESHOLD,
                                skew_threshold=SEED_MEMORYLESS_THRESHOLD,
                                indent='    '):
    """
    Report whether the free first passages behind the rate started in equilibrium.

    ``k = N / Σ(w·L·γ)`` is a mean-first-passage estimator, and it is the escape
    rate from a state only if each first passage started from the equilibrium
    distribution *inside* that state. AIMMD's free worker restarts every new free
    trajectory from the frame the previous one escaped from, which sits on the
    state boundary (see
    :func:`aimmd.worker.utils.get_basin_frames_for_free_restart` and
    ``params.free_restart_source``). That is harmless when in-state
    relaxation is fast compared with the escape time, and badly biased when it is
    not: the observations pile up at short times, the mean is carried by the rare
    trajectory that did settle into the basin, and the rate comes out too fast.

    Two symptoms are checked, both computed from the numbers already in hand.

    1. **Realised acceleration.** Per free trajectory, ``Γᵢ = Σ(L·γ)/Σ(L)`` over
       its in-basin blocks is the boost it actually got; the frame-weighted pooled
       value over all free trajectories of that basin, ``Γ_eq``, is the boost an
       equilibrated trajectory gets. Reported is the fraction of completed first
       passages with ``Γᵢ/Γ_eq`` below *boost_fraction*.

       This is deliberately *not* a "did it reach the deep well" test on
       ``max(bias)``. The fill is not monotonic in depth — for calixarene-G2 the
       recorded bias peaks at 6.8 kT around d ≈ 0.42 nm and falls back to ~0 for
       d < 0.27 nm, a region the frozen bias never filled — so a depth criterion
       built on the deepest bias mis-ranks trajectories. ``Γᵢ/Γ_eq`` asks the
       question the estimator actually cares about: was this passage's clock
       boosted the way the basin's equilibrium clock is?

    2. **Memorylessness.** The bias-reweighted durations of a Poisson escape
       process are exponential: median/mean = ln 2 = 0.693, and the KS distance
       from a fitted exponential is small. Reported are both, with the approximate
       5 % Lilliefors critical value.

    Parameters
    ----------
    pathensemble : PathEnsemble
        The ensemble the rate was computed from.
    lengths : numpy.ndarray
        Counted frames per path (``pathensemble.n_frames``).
    gammas : numpy.ndarray
        Per-path bias corrections from :func:`compute_bias_corrections`.
    states : str, optional
        Three-character state string, e.g. ``'ARB'``.
    boost_fraction, miss_threshold, skew_threshold : float, optional
        See the module constants.
    indent : str, default four spaces
        Prefix for the printed lines.

    Returns
    -------
    dict
        Keyed by target-state label, each value a dict with ``n_passages``,
        ``n_censored``, ``n_low_boost``, ``frac_low_boost``, ``boost_ratios``,
        ``boost_equilibrium``, ``median_over_mean``, ``ks_distance``,
        ``ks_critical``, ``times`` (reweighted first-passage durations in ``dt``)
        and ``ok``. Also prints the report and raises a single ``UserWarning``
        if any direction fails.

    Notes
    -----
    - The durations are the free trajectories' own boosted clocks, ``Σ L·γ`` over
      their blocks, deliberately *without* the reweighting weights: that keeps the
      number defined for both directions at once and comparable between them.
      Free trajectories are unmodified dynamics and enter ``Σ(w·L·γ)`` through the
      same ``L·γ``, so a distortion of this distribution is a distortion of the
      denominator.
    - A trajectory still running (or cut at ``params.max_length``) has not
      completed a first passage. Those are counted as ``n_censored`` and excluded
      from both statistics; they are legitimate right-censored observations, and
      excluding them makes the reported rate an *upper* bound.
    - ``Γ_eq`` is estimated from the same trajectories, so it is only a valid
      equilibrium reference once at least one of them has equilibrated. When none
      has, the ratios all sit near 1 and the memorylessness statistic is the
      backstop.
    - Cost is one pass over the per-path arrays; no trajectory or cache file is
      read.
    """
    r = states[1] if len(states) >= 2 else states[0]
    ends = [s for s in states if s != r]

    lengths = np.asarray(lengths, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    # group the split blocks back into free trajectories
    groups = {}
    for i, path in enumerate(pathensemble):
        key, state = _free_trajectory_key(path)
        if key is None or state not in ends:
            continue
        try:
            path_type = path.type
        except Exception:
            continue
        group = groups.setdefault(key, {'state': state, 'time': 0.0,
                                        'boost_sum': 0.0, 'boost_n': 0.0,
                                        'done': False})
        group['time'] += lengths[i] * gammas[i]
        if len(path_type) > 1 and path_type[1] == state:
            group['boost_sum'] += lengths[i] * gammas[i]
            group['boost_n'] += lengths[i]
        if any(other in path_type[:3] for other in ends if other != state):
            group['done'] = True

    results = {}
    lines = []
    failed = []
    for state in ends:
        mine = [g for g in groups.values() if g['state'] == state]
        if not mine:
            continue
        pooled_sum = sum(g['boost_sum'] for g in mine)
        pooled_n = sum(g['boost_n'] for g in mine)
        boost_eq = pooled_sum / pooled_n if pooled_n > 0 else float('nan')

        done = [g for g in mine if g['done']]
        times = np.array([g['time'] for g in done], dtype=float)
        n_censored = len(mine) - len(done)

        ratios = np.array(
            [((g['boost_sum'] / g['boost_n']) / boost_eq)
             if (g['boost_n'] > 0 and np.isfinite(boost_eq) and boost_eq > 0)
             else 0.0
             for g in done], dtype=float)
        if len(ratios) and np.isfinite(boost_eq) and boost_eq > 1.0 + 1e-9:
            low = int(np.count_nonzero(ratios < boost_fraction))
            frac_low = low / len(ratios)
        else:
            # no fill at all (unbiased run): the ratio carries no information
            low, frac_low = 0, float('nan')

        finite = times[np.isfinite(times) & (times > 0)]
        ratio = (float(np.median(finite) / np.mean(finite))
                 if len(finite) else float('nan'))
        ks, ks_crit = _ks_exponential(finite)

        ok = True
        if np.isfinite(frac_low) and frac_low > miss_threshold:
            ok = False
        if np.isfinite(ratio) and ratio < skew_threshold:
            ok = False
        if np.isfinite(ks) and np.isfinite(ks_crit) and ks > ks_crit:
            ok = False

        results[state] = {
            'n_passages': len(done), 'n_censored': n_censored,
            'n_low_boost': low, 'frac_low_boost': frac_low,
            'boost_ratios': ratios, 'boost_equilibrium': boost_eq,
            'median_over_mean': ratio, 'ks_distance': ks,
            'ks_critical': ks_crit, 'times': finite, 'ok': bool(ok)}

        others = '/'.join(o for o in ends if o != state)
        lines.append(
            f'{indent}Free {state}->{others} first passages: '
            f'{len(done)} completed'
            + (f' (+{n_censored} still open)' if n_censored else '')
            + f'; {low} got under {boost_fraction:.0%} of the equilibrium '
            f'boost <exp(bias)>_{state} = {boost_eq:.1f}'
            + ('' if not np.isfinite(frac_low) else f' ({frac_low:.0%})')
            + f'; median/mean {ratio:.3f} (0.693 if memoryless), KS D {ks:.3f}'
            + ('' if not np.isfinite(ks_crit) else f' (5% crit {ks_crit:.3f})'))
        if not ok:
            failed.append(state)

    if not results:
        report = f'{indent}Free-basin seed check: no free first passages found'
        print(report)
        return results

    if failed:
        body = (
            f'the free first passages in {", ".join(sorted(failed))} did not '
            f'start from an equilibrated basin. k = N / sum(w*L*gamma) is a '
            f'mean-first-passage estimator and only measures the escape rate if '
            f'each passage starts from the equilibrium distribution INSIDE the '
            f'state; a trajectory re-seeded at the state boundary escapes before '
            f'it has sampled the basin, so it enters the denominator with almost '
            f'none of the boosted dwell time it should carry and the rate is '
            f'biased fast by roughly 1/(1 - fraction of low-boost passages). Set '
            f'params.free_restart_source to draw restarts from inside the '
            f'basin, and treat these bias-reweighted rates as an upper bound '
            f'until the low-boost fraction and median/mean recover.')
        wrapped = textwrap.wrap(body, width=78, break_on_hyphens=False)
        lines.append(f'{indent}*** WARNING: {wrapped[0]}')
        lines += [f'{indent}***   {line}' for line in wrapped[1:]]

    print('\n'.join(lines))
    if failed:
        warnings.warn(
            f'Non-equilibrium free-basin seeds in {", ".join(sorted(failed))}: '
            f'the bias-reweighted rate is an upper bound. See the free-basin '
            f'seed report in the log.', UserWarning, stacklevel=2)
    return results

def bias_reweighted_rates(pathensemble, weights1, weights2, lengths=None,
                          states='ARB', reactive_threshold=0.5,
                          coverage_threshold=BIAS_CACHE_COVERAGE_THRESHOLD,
                          zero_point_tolerance=BIAS_ZERO_POINT_TOLERANCE,
                          seed_diagnostics=True,
                          windows=None, trim_margins=True, label=''):
    """
    Run the whole Tiwary-Parrinello reweighting step and report both rates.

    This is the single implementation of the bias-reweighted rate estimate. It
    exists because the same six lines used to be written out at four places in
    :mod:`aimmd.worker._train` (the single-system trainer, the multi-system
    trainer, and the two kinetics-convergence scans), which is how the frame-set
    mismatch between ``L`` and ``γ`` could be fixed in one place and survive in
    three.

    Being the single implementation, it is also where the always-on validity
    checks live: :func:`check_reactive_bias`, :func:`check_bias_zero_point`, the
    bias-cache coverage report inside :func:`compute_bias_corrections`, and
    :func:`report_nonequilibrium_seeds`. A rate that fails any of them is still
    printed — nothing is silently withheld — but the log then says what is wrong
    with it, and for a bias zero-point offset it also prints the corrected value.

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
    zero_point_tolerance : float, optional
        Passed to :func:`check_bias_zero_point`.
    seed_diagnostics : bool, optional
        Run :func:`report_nonequilibrium_seeds` (default True). It reads no
        files, so there is little reason to switch it off.
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
    zero_point = check_bias_zero_point(pathensemble, states,
                                       tolerance=zero_point_tolerance)
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

    # A constant zero-point offset c scales every gamma by exp(c), hence the
    # rate by exp(-c) exactly. Print the corrected value so the log carries a
    # usable number without anyone editing params.bias_function mid-run.
    # Gated on the *median* test, which is where `factor` comes from, and not
    # on `ok`, which also carries the mixed-zero-point and negative-bias
    # failures that `factor` says nothing about: those two used to print
    # `offset +0.000 kT` and a rate identical to the uncorrected one under a
    # "corrected" label. `factor` is nan for a mixture, so `isfinite` is the
    # second guard, and the invariant documented on check_bias_zero_point
    # holds: isfinite(factor) <=> dividing every rate by it is valid.
    if not zero_point['median_ok'] and np.isfinite(zero_point['factor']):
        scale = zero_point['factor']
        print(f'    {label}k12 zero-point corrected '
              f'(offset {zero_point["offset"]:+.3f} kT): '
              f'{k12_rw / scale:.3e} [1/dt]')
        print(f'    {label}k21 zero-point corrected '
              f'(offset {zero_point["offset"]:+.3f} kT): '
              f'{k21_rw / scale:.3e} [1/dt]')

    if seed_diagnostics:
        # gamma is only filled where the direction's weight is non-zero, so
        # merge the two directions: a free A->B trajectory is weighted in one of
        # them, a free B->A trajectory in the other.
        gammas = np.where(np.asarray(weights1, dtype=float) != 0.0,
                          gamma1, gamma2)
        report_nonequilibrium_seeds(pathensemble, lengths, gammas,
                                    states=states)
    return k12_rw, k21_rw, gamma1, gamma2
