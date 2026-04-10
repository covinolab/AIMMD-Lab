"""
aimmd.pathensemble.bias_utils
==============================

Bias reweighting utilities for :class:`aimmd.pathensemble.PathEnsemble`.

This module provides two public functions used by the AIMMD training worker when
``params.record_bias = True``:

- :func:`check_reactive_bias` — validates that the applied bias is negligible inside
  the reactive region R, which is required for the reweighting formula to be valid.

- :func:`compute_bias_corrections` — computes the per-path bias correction factor
  ``γᵢ = ⟨exp(bias)⟩_path_i`` used to recover unbiased kinetics from a biased path
  ensemble.

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
"""

# external
import warnings
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


def compute_bias_corrections(pathensemble, weights):
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
        ``weights[i] > 0`` need non-trivial correction; the rest are set to 1.0.

    Returns
    -------
    gammas : numpy.ndarray, shape (len(pathensemble),)
        Per-path correction factors. ``gammas[i] = 1.0`` for paths with zero
        weight or missing bias data.

    Notes
    -----
    - The corrected rate is ``k = 1 / Σ(w · L · γ)`` where ``L`` is path length.
    - For unbiased paths (bias ≈ 0 everywhere), ``exp(0) = 1`` so ``γ = 1`` and
      the formula reduces exactly to the standard AIMMD estimator.
    - A single ``UserWarning`` is emitted if any path with non-zero weight is
      missing its bias cache.
    """
    gammas = np.ones(len(pathensemble))
    missing_warned = False

    for i, path in enumerate(pathensemble):
        if weights[i] == 0.0:
            continue

        try:
            bias = path._get('bias', raise_if_missing=True)
        except Exception:
            # Cache file missing → fall back to γ = 1.0 with one-time warning.
            if not missing_warned:
                warnings.warn(
                    'Some paths with non-zero weight are missing bias cache files. '
                    'Falling back to γ = 1.0 (no bias correction) for those paths. '
                    'Ensure bias_function was applied before computing bias corrections.',
                    UserWarning,
                    stacklevel=2)
                missing_warned = True
            gammas[i] = 1.0
            continue

        if bias is None or len(bias) == 0:
            gammas[i] = 1.0
            continue

        # γᵢ = mean(exp(bias)) over all frames of this path
        # For unbiased paths (bias = 0 everywhere), exp(0) = 1, so γ = 1 naturally.
        gammas[i] = float(np.mean(np.exp(bias)))

    return gammas
