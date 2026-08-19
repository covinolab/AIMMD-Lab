"""
aimmd.pathensemble.bias_utils
==============================

Bias reweighting utilities for :class:`aimmd.pathensemble.PathEnsemble`.

This module provides four public functions used by the AIMMD training worker when
``params.record_bias = True``:

- :func:`check_reactive_bias` — validates that the applied bias is negligible inside
  the reactive region R, which is required for the reweighting formula to be valid.

- :func:`compute_bias_corrections` — computes the per-path bias correction factor
  ``γᵢ = ⟨exp(bias)⟩_path_i`` used to recover unbiased kinetics from a biased path
  ensemble, and reports how much of the ensemble it could actually correct.

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

    γᵢ = mean(exp(path.bias))   over all frames of path i

The corrected rate estimate is::

    k₁₂_unbiased = 1 / Σᵢ(wᵢ · Lᵢ · γᵢ)

where ``wᵢ`` are the existing RFPS/TPS weights and ``Lᵢ`` is the path length in frames.

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
"""Default missing-fraction above which :func:`format_bias_cache_coverage`
escalates from a one-line summary to an explicit warning block."""


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


def _default_frame_counter(fname):
    """Frame count of a trajectory file, via AIMMD's reader cache.

    Uses the same idiom as :func:`aimmd.params._methods._split_cumulative_colvar`
    (``len(MDA_CACHE.get(f))``); the cache rebuilds offsets on every open, so a
    file still being written by ``mdrun`` reports its currently readable length.
    Returns 0 if the file cannot be opened.
    """
    from .._config import MDA_CACHE
    try:
        reader = MDA_CACHE.get(fname)
    except Exception:
        return 0
    if reader is None:
        return 0
    try:
        return len(reader)
    except TypeError:
        return 0


def derive_bias_from_cumulative_colvar(fname, trajectory_extension,
                                       bias_function, frame_counter=None):
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
    frame_counter : callable, optional
        ``frame_counter(path) -> int`` frame count. Defaults to AIMMD's reader
        cache; injectable for testing.

    Returns
    -------
    numpy.ndarray or None
        Per-frame bias in kT, aligned with the part's frames from frame 0. May
        be **shorter** than the part if PLUMED has not flushed all rows yet —
        callers must treat the uncovered tail as ``exp(bias) = 1``. None when
        the alignment cannot be established (see Notes).

    Notes
    -----
    Returns None, warning where the cause is ambiguous, when:

    - *fname* is not a ``.partNNNN`` file (shooting paths have their own cache);
    - the cumulative ``COLVAR`` is absent, or the part is not on disk;
    - the ``COLVAR`` holds **more** rows than the trajectory's parts account for.
      That means it is not this trajectory's COLVAR (it is rotated to
      ``bck.0.COLVAR`` when a new trajectory starts) or ``PRINT STRIDE`` does not
      match ``nstxout-compressed``. Guessing an offset here would silently
      misalign bias with frames, which is worse than no correction at all.
    """
    if frame_counter is None:
        frame_counter = _default_frame_counter

    stem = os.path.basename(fname)
    if not stem.endswith(trajectory_extension):
        return None
    match = _PART_RE.match(stem[:-len(trajectory_extension)])
    if match is None:
        return None  # not a free-trajectory part
    base = match.group('base')
    part = int(match.group('part'))

    deffnm_dir = os.path.dirname(fname) or '.'
    cum_colvar = os.path.join(deffnm_dir, 'COLVAR')
    if not os.path.exists(cum_colvar):
        return None

    # Parts in order, excluding the python-written part0000 seed (no PLUMED rows).
    parts = []
    for pf in sorted(glob(os.path.join(
            deffnm_dir, f'{base}.part????{trajectory_extension}'))):
        n = int(os.path.basename(pf)[len(base) + len('.part'):][:4])
        if n == 0:
            continue
        parts.append((n, frame_counter(pf)))
    if part not in [n for n, _ in parts]:
        return None  # our part is not on disk (yet)

    offset = sum(c for n, c in parts if n < part)
    n_frames = dict(parts)[part]
    total = sum(c for _, c in parts)
    if n_frames <= 0:
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
    if len(rows) == 0:
        return None

    if len(rows) > total:
        warnings.warn(
            f'{cum_colvar!r} has {len(rows)} rows but the parts of {base!r} '
            f'account for only {total} frames, so the row-to-frame alignment is '
            f'unknown (a rotated COLVAR from another trajectory, or PRINT STRIDE '
            f'not matching nstxout-compressed). Not deriving bias out-of-cache '
            f'for {os.path.basename(fname)!r}.',
            UserWarning,
            stacklevel=2)
        return None

    available = len(rows) - offset
    if available <= 0:
        return None
    sel = rows[offset:offset + min(n_frames, available)]

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


def compute_bias_corrections(pathensemble, weights, lengths=None,
                             check=True,
                             threshold=BIAS_CACHE_COVERAGE_THRESHOLD,
                             return_coverage=False):
    """
    Compute per-path bias correction factors.

    For each path ``i``, computes::

        γᵢ = mean(exp(bias_i))

    where ``bias_i`` is the per-frame bias in kT units loaded from the
    ``path.bias`` cache (``<traj>.bias.npy``). Paths with missing bias data
    fall back to ``γᵢ = 1.0`` (no correction).

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
            How many of those had no usable bias array, or only a partial one.
        ``frac_paths``
            ``n_missing / n_paths``.
        ``frac_weighted_length``
            Share of ``Σ |w| · L`` carried by **frames** running on
            ``exp(bias) = 1`` for want of a bias value. **This is the number
            that matters** — see Notes. Counted per frame, so a path whose bias
            array covers only part of it contributes its uncovered fraction.
        ``max_inflation``
            ``1 / (1 - frac_weighted_length)``: the largest factor by which the
            reweighted rate can be overestimated because of the gap.
        ``weighted_length_total``, ``weighted_length_missing``
            The raw sums behind ``frac_weighted_length``.
        ``missing_examples``
            Up to three identifiers of uncached paths, for the log.

    Notes
    -----
    - The corrected rate is ``k = 1 / Σ(w · L · γ)`` where ``L`` is path length.
    - For unbiased paths (bias ≈ 0 everywhere), ``exp(0) = 1`` so ``γ = 1`` and
      the formula reduces exactly to the standard AIMMD estimator.
    - Unless ``check=False``, a single ``UserWarning`` is emitted if any path with
      non-zero weight is missing its bias cache.
    - Coverage is weighted by ``|w| · L``, not counted per path, because the
      estimator is a sum over ``w · L · γ``: one long uncached free-basin
      trajectory can be 1 path in 1000 and still carry most of the dwell time.
      With a typical correction ``Γ ≫ 1``, a missing weighted fraction ``f``
      inflates the rate by ``≈ 1 / (1 - f)``, hence ``max_inflation``.
    """
    gammas = np.ones(len(pathensemble))

    n_weighted = 0
    n_missing = 0
    length_total = 0.0
    length_missing = 0.0
    missing_examples = []

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
            if len(missing_examples) < 3:
                missing_examples.append(_path_label(path))
            continue

        n_bias = len(bias)
        if n_bias >= n_frames or n_frames <= 0:
            # Fully covered. Note `pathensemble.n_frames` excludes boundary
            # frames, so len(bias) > n_frames is the normal case; γ stays the
            # mean over the whole path exactly as it always was.
            # For unbiased paths (bias = 0 everywhere), exp(0) = 1, so γ = 1.
            gammas[i] = float(np.mean(np.exp(bias)))
        else:
            # Partially covered — e.g. the out-of-cache fallback found the
            # cumulative COLVAR still short of the part's frames. Frames with no
            # bias information carry exp(0) = 1 rather than inheriting the
            # covered frames' average.
            gammas[i] = float(
                (np.sum(np.exp(bias)) + (n_frames - n_bias)) / n_frames)
            n_missing += 1
            length_missing += weighted_length * (n_frames - n_bias) / n_frames
            if len(missing_examples) < 3:
                missing_examples.append(
                    f'{_path_label(path)} ({n_bias}/{int(n_frames)} frames)')

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
        'missing_examples': missing_examples,
    }

    # The check is default behaviour: report every call, and warn — after the
    # loop, so the message can quantify the gap it is warning about.
    if check:
        print(format_bias_cache_coverage(coverage, threshold=threshold))
        if n_missing:
            message = (
                f'{n_missing} of {n_weighted} paths with non-zero weight are '
                f'missing bias cache files, carrying {frac_weighted:.1%} of the '
                f'weighted path length. Those paths fall back to γ = 1.0 (no bias '
                f'correction), which can overestimate the reweighted rate by up to '
                f'{_inflation_text(coverage["max_inflation"])}.')
            if frac_weighted > threshold:
                message += ' ' + _REMEDIATION_NOTE
            else:
                message += (' Ensure bias_function was applied before computing '
                            'bias corrections.')
            warnings.warn(message, UserWarning, stacklevel=2)

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
        more = coverage['n_missing'] - len(coverage['missing_examples'])
        suffix = f' (+{more} more)' if more > 0 else ''
        lines.append(f'{indent}***   uncached: {shown}{suffix}')
    return '\n'.join(lines)
