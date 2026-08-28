"""
Unit tests for aimmd.pathensemble.bias_utils.

Tests
-----
test_bias_formula_values          — sigmoid bias has correct values at key positions
test_check_reactive_bias_pass     — validation passes when R bias is small
test_check_reactive_bias_fail     — validation warns when R bias is large
test_check_reactive_bias_no_data  — returns 0.0 / nan when no bias data available
test_bias_corrections_no_bias       — γ = 1.0 when bias is all zeros (no warning)
test_bias_corrections_constant      — γ = exp(V0) when bias is constant V0
test_bias_corrections_zero_weight   — zero-weight paths get γ = 1.0
test_bias_corrections_mixed         — correct average for mixed A/R/B bias
test_bias_corrections_fallback      — γ = 1.0 + UserWarning for missing bias cache

Bias-cache coverage (the fraction of data running on γ = 1.0)
test_coverage_opt_in_preserves_return_type   — default call still returns a bare array
test_coverage_all_cached_reports_zero        — fully cached → 0 % missing, inflation 1
test_coverage_weighted_by_path_length_not_path_count
                                             — one long uncached path dominates
test_coverage_counts_empty_bias_array_as_missing — empty array counts as uncached
test_coverage_ignores_zero_weight_paths      — zero-weight paths excluded entirely
test_coverage_falls_back_to_len_when_lengths_omitted — len(path) fallback
test_coverage_all_missing_is_infinite_inflation — nothing corrected → inf
test_report_reads_well_when_nothing_is_cached — 'unbounded', not 'infx'
test_coverage_empty_ensemble_is_safe         — no weighted paths → no ZeroDivisionError
test_report_clean_run_is_one_reassuring_line — no remediation note when healthy
test_report_flags_high_missing_fraction_with_remediation — names cause and remedies
test_report_threshold_is_respected           — threshold controls escalation

The check is default behaviour, not opt-in
test_check_runs_by_default_and_prints_coverage — plain call reports coverage
test_check_by_default_escalates_when_problematic — plain call escalates
test_check_can_be_suppressed                 — check=False silences print+warning
test_warning_is_silent_below_threshold       — small gaps are logged, not warned
test_warning_does_not_repeat_the_printed_report — one channel per message
test_uncovered_file_list_has_no_duplicates   — each file named once
test_check_threshold_is_configurable         — caller can tighten the threshold
test_no_bias_anywhere_stays_quiet_about_remediation — unbiased run not flagged

Out-of-cache bias derivation (fallback for a still-running free segment)
test_derive_happy_path_middle_part           — a middle part gets its own rows
test_derive_trailing_part_under_colvar_lag   — short slice, never borrowed rows
test_derive_tolerates_surplus_colvar_rows    — surplus rows are simply unreached
test_derive_refuses_a_part_of_an_older_trajectory — rotated COLVAR is not ours
test_derive_returns_none_without_colvar      — no COLVAR, no derivation
test_derive_skips_the_part0000_seed          — seed part carries no PLUMED rows
test_derive_writes_nothing_into_the_trajectory_directory — read-only
test_derive_ignores_non_part_filenames       — shooting paths untouched
test_derive_handles_single_row_colvar        — 1-D loadtxt result
test_derive_returns_none_when_part_not_yet_on_disk — no crash
test_cache_bias_files_falls_back_to_out_of_cache   — trainer wiring

Gamma over a path's bias array
test_a_short_bias_cache_needs_no_special_handling  — _get pads, so one branch suffices
test_gamma_unchanged_when_bias_covers_the_whole_path — regression

L and gamma must count the same frames
test_frame_windows_reproduce_n_frames        — n_frames is stop - start, by definition
test_gamma_averages_only_the_counted_frames  — margins excluded from L are excluded from γ
test_margin_frames_used_to_dilute_gamma      — the defect this fixes, quantified
test_trim_margins_false_reproduces_old_numbers — escape hatch for old runs
test_windows_fall_back_to_whole_paths        — objects without frame_windows still work
test_reweighted_rates_are_a_frame_sum        — Σ(w·L·γ) == Σ(w·Σexp(bias))
test_reweighted_rates_flag_inconsistent_lengths — caller-supplied L that is not the window
"""

import os
import warnings
import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def _bias_value(x, V0=3.0, kappa=20.0, xb=0.5):
    """Reference sigmoid bias: V0 * σ(κ · (|x| - x_b))  [kT]."""
    return V0 * _sigmoid(kappa * (abs(x) - xb))


# ════════════════════════════════════════════════════════════════════════════
# Mock path / pathensemble
# ════════════════════════════════════════════════════════════════════════════

class MockPath:
    """Minimal mock exposing .bias and .states as arrays."""

    def __init__(self, states, bias=None):
        self._states = np.asarray(states, dtype='<U1')
        self._bias = (np.asarray(bias, dtype=float)
                      if bias is not None else None)

    def _get(self, attribute, raise_if_missing=False):
        if attribute == 'states':
            return self._states.copy()
        if attribute == 'bias':
            if self._bias is None:
                if raise_if_missing:
                    raise TypeError('no bias')
                return np.zeros(len(self._states))
            return self._bias.copy()
        raise AttributeError(attribute)

    def __len__(self):
        return len(self._states)


class MockPathEnsemble:
    """Minimal mock iterable over MockPath objects."""

    def __init__(self, paths):
        self._paths = list(paths)

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — bias formula
# ════════════════════════════════════════════════════════════════════════════

def test_bias_formula_values():
    """Sigmoid bias should be ≈V0 deep in the state and ≈0 deep in R."""
    V0, kappa, xb = 3.0, 20.0, 0.5

    assert _bias_value(-1.5, V0, kappa, xb) > 2.9, "Bias not near V0 in state A"
    assert _bias_value(0.0, V0, kappa, xb) < 0.01, "Bias not near 0 in R at x=0"

    val_at_boundary = _bias_value(0.4, V0, kappa, xb)
    assert val_at_boundary < 0.5, (
        f"Bias at extended boundary ({val_at_boundary:.3f} kT) "
        f"exceeds 0.5 kT threshold — bias check would fail")


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — check_reactive_bias
# ════════════════════════════════════════════════════════════════════════════

def test_check_reactive_bias_pass():
    """check_reactive_bias should pass (no warning) when R bias is small."""
    from aimmd.pathensemble.bias_utils import check_reactive_bias

    states_arr = list('AARRRRB')
    bias_arr = [3.0, 3.0, 0.05, 0.1, 0.05, 0.1, 2.9]
    pe = MockPathEnsemble([MockPath(states_arr, bias_arr)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mean_bias, max_bias = check_reactive_bias(pe, 'ARB', threshold=0.5)

    assert not any(issubclass(w.category, UserWarning) for w in caught), \
        "Unexpected warning: bias in R should be below threshold"
    assert mean_bias < 0.5, f"mean_bias in R = {mean_bias:.3f}"


def test_check_reactive_bias_fail():
    """check_reactive_bias should warn when R bias is large."""
    from aimmd.pathensemble.bias_utils import check_reactive_bias

    states_arr = list('ARRRB')
    bias_arr = [3.0, 1.0, 1.5, 1.2, 3.0]
    pe = MockPathEnsemble([MockPath(states_arr, bias_arr)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mean_bias, _ = check_reactive_bias(pe, 'ARB', threshold=0.5)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) > 0, "Expected UserWarning for large R bias"
    assert mean_bias > 0.5, f"mean_bias in R = {mean_bias:.3f}"


def test_check_reactive_bias_no_data():
    """check_reactive_bias returns 0.0 when no explicit bias is set."""
    from aimmd.pathensemble.bias_utils import check_reactive_bias

    pe = MockPathEnsemble([MockPath(list('ARB'), None)])  # no bias → all zeros
    mean_bias, _ = check_reactive_bias(pe, 'ARB', threshold=0.5)
    assert mean_bias == 0.0 or np.isnan(mean_bias), \
        f"Unexpected mean_bias = {mean_bias}"


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — compute_bias_corrections
# ════════════════════════════════════════════════════════════════════════════

def test_bias_corrections_no_bias():
    """γ = 1.0 for all paths when bias is all zeros (exp(0)=1, no warning)."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    paths = [MockPath(list('ARB'), [0.0, 0.0, 0.0]),
             MockPath(list('ARA'), [0.0, 0.0, 0.0])]
    pe = MockPathEnsemble(paths)
    weights = np.array([1.0, 1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gammas = compute_bias_corrections(pe, weights)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, \
        f"Unexpected warning for all-zero bias: {[str(w.message) for w in user_warnings]}"
    np.testing.assert_allclose(gammas, [1.0, 1.0], atol=1e-10)


def test_bias_corrections_constant():
    """γ = exp(V0) when bias is constant V0 everywhere."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    V0 = 3.0
    pe = MockPathEnsemble([MockPath(list('ARRB'), [V0, V0, V0, V0])])
    gammas = compute_bias_corrections(pe, np.array([1.0]))
    np.testing.assert_allclose(gammas[0], np.exp(V0), rtol=1e-6)


def test_bias_corrections_zero_weight():
    """Paths with zero weight get γ = 1.0 without computing bias."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPath(list('ARB'), [3.0, 0.0, 3.0])])
    gammas = compute_bias_corrections(pe, np.array([0.0]))
    assert gammas[0] == 1.0, "Zero-weight path should have γ = 1.0"


def test_bias_corrections_mixed():
    """Average is correct for mixed bias (high in state, low in R)."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0]
    pe = MockPathEnsemble([MockPath(list('AARRRRBB'), bias)])
    gammas = compute_bias_corrections(pe, np.array([1.0]))
    expected = np.mean(np.exp(np.array(bias)))
    np.testing.assert_allclose(gammas[0], expected, rtol=1e-6)


def test_bias_corrections_fallback():
    """Paths missing bias cache produce γ = 1.0 with a UserWarning."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    class MockPathNoBias(MockPath):
        def _get(self, attribute, raise_if_missing=False):
            if attribute == 'bias' and raise_if_missing:
                raise TypeError('no bias')
            return super()._get(attribute, raise_if_missing=False)

    pe = MockPathEnsemble([MockPathNoBias(list('ARB'), None)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gammas = compute_bias_corrections(pe, np.array([1.0]))

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1, "Expected UserWarning for missing bias cache"
    assert gammas[0] == 1.0, "Fallback γ should be 1.0"


# ════════════════════════════════════════════════════════════════════════════
# Mocks for bias-cache coverage
# ════════════════════════════════════════════════════════════════════════════

class MockPathMissingBias(MockPath):
    """Path whose bias cache is absent — ``_get`` raises when required."""

    def __init__(self, states, fname='missing.xtc'):
        super().__init__(states, bias=None)
        self.fnames = [fname]

    def _get(self, attribute, raise_if_missing=False):
        if attribute == 'bias':
            if raise_if_missing:
                raise TypeError('no bias cache')
            return None
        return super()._get(attribute, raise_if_missing=raise_if_missing)


class MockPathEmptyBias(MockPath):
    """Path whose bias cache exists but is empty (zero-length array)."""

    def __init__(self, states, fname='empty.xtc'):
        super().__init__(states, bias=None)
        self.fnames = [fname]

    def _get(self, attribute, raise_if_missing=False):
        if attribute == 'bias':
            return np.zeros(0)
        return super()._get(attribute, raise_if_missing=raise_if_missing)


def _flat(text):
    """Collapse whitespace so content assertions ignore line wrapping."""
    return ' '.join(text.split())


def _biased_path(n_frames, bias_value=5.0, fname='ok.xtc'):
    """A path of *n_frames* frames carrying a uniform bias, cache present."""
    p = MockPath(list('R' * n_frames), [bias_value] * n_frames)
    p.fnames = [fname]
    return p


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — bias-cache coverage accounting
# ════════════════════════════════════════════════════════════════════════════

def test_coverage_opt_in_preserves_return_type():
    """Without return_coverage the function still returns a bare array."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(4)])
    out = compute_bias_corrections(pe, np.array([1.0]))
    assert isinstance(out, np.ndarray), \
        'default call must stay backward compatible (bare ndarray)'


def test_coverage_all_cached_reports_zero():
    """Fully cached ensemble → zero missing fraction, inflation factor 1."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(4), _biased_path(6)])
    _, cov = compute_bias_corrections(
        pe, np.array([1.0, 1.0]), lengths=np.array([4, 6]),
        return_coverage=True)

    assert cov['n_missing'] == 0
    assert cov['frac_weighted_length'] == pytest.approx(0.0)
    assert cov['max_inflation'] == pytest.approx(1.0)


def test_coverage_weighted_by_path_length_not_path_count():
    """One long uncached path must dominate, even though it is 1 path of 5.

    The failure mode this guards against: a single free-basin trajectory that
    never terminated carries most of the dwell time, so a path-count metric
    reads 20 % while the quantity that actually enters the rate is 85 %.
    """
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    lengths = np.array([100, 100, 100, 100, 2267])
    paths = [_biased_path(100) for _ in range(4)]
    paths.append(MockPathMissingBias(list('A' * 2267), fname='freeA/traj2.xtc'))
    pe = MockPathEnsemble(paths)
    weights = np.ones(5)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        gammas, cov = compute_bias_corrections(
            pe, weights, lengths=lengths, return_coverage=True)

    assert cov['frac_paths'] == pytest.approx(1 / 5)
    assert cov['frac_weighted_length'] == pytest.approx(2267 / 2667, rel=1e-6)
    assert cov['max_inflation'] == pytest.approx(2667 / 400, rel=1e-6)
    assert gammas[-1] == 1.0
    assert 'freeA/traj2.xtc' in ' '.join(cov['missing_examples'])


def test_coverage_counts_empty_bias_array_as_missing():
    """A zero-length bias array is as uncorrected as a missing file."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10), MockPathEmptyBias(list('A' * 10))])
    _, cov = compute_bias_corrections(
        pe, np.ones(2), lengths=np.array([10, 10]), return_coverage=True)

    assert cov['n_missing'] == 1
    assert cov['frac_weighted_length'] == pytest.approx(0.5)


def test_coverage_ignores_zero_weight_paths():
    """Zero-weight paths contribute to neither numerator nor denominator."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 1000))])
    _, cov = compute_bias_corrections(
        pe, np.array([1.0, 0.0]), lengths=np.array([10, 1000]),
        return_coverage=True)

    assert cov['n_paths'] == 1, 'zero-weight path must not be counted'
    assert cov['n_missing'] == 0
    assert cov['frac_weighted_length'] == pytest.approx(0.0)


def test_coverage_falls_back_to_len_when_lengths_omitted():
    """Without an explicit lengths array, len(path) is used."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 30))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(pe, np.ones(2), return_coverage=True)

    assert cov['frac_weighted_length'] == pytest.approx(30 / 40)


def test_coverage_all_missing_is_infinite_inflation():
    """Nothing corrected → fraction 1.0 and an unbounded inflation factor."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPathMissingBias(list('A' * 10))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(
            pe, np.ones(1), lengths=np.array([10]), return_coverage=True)

    assert cov['frac_weighted_length'] == pytest.approx(1.0)
    assert np.isinf(cov['max_inflation'])


def test_report_reads_well_when_nothing_is_cached():
    """An infinite inflation factor must not render as 'infx' in the report."""
    from aimmd.pathensemble.bias_utils import (
        compute_bias_corrections, format_bias_cache_coverage)

    pe = MockPathEnsemble([MockPathMissingBias(list('A' * 10))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(pe, np.ones(1),
                                          lengths=np.array([10]),
                                          return_coverage=True)

    text = format_bias_cache_coverage(cov)
    assert 'infx' not in text, f'unreadable inflation factor: {text!r}'
    assert 'unbounded' in text, f'expected an explicit wording: {text!r}'


def test_coverage_empty_ensemble_is_safe():
    """No weighted paths at all → zeros, no exception, no division by zero."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([])
    gammas, cov = compute_bias_corrections(
        pe, np.array([]), lengths=np.array([]), return_coverage=True)

    assert len(gammas) == 0
    assert cov['n_paths'] == 0
    assert cov['frac_weighted_length'] == pytest.approx(0.0)
    assert cov['max_inflation'] == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — the coverage check runs BY DEFAULT
# ════════════════════════════════════════════════════════════════════════════

def test_check_runs_by_default_and_prints_coverage(capsys):
    """A plain call reports coverage without the caller wiring anything up."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10)])
    compute_bias_corrections(pe, np.ones(1), lengths=np.array([10]))

    out = capsys.readouterr().out
    assert 'Bias cache coverage' in out, \
        f'the check must run by default, got: {out!r}'
    assert '100.0%' in out, out


def test_check_by_default_escalates_when_problematic(capsys):
    """Above threshold the default call prints the full warning block."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 90))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([10, 90]))

    out = capsys.readouterr().out
    assert '90.0%' in out, out
    assert 'out-of-cache' in _flat(out).lower(), out


def test_check_can_be_suppressed(capsys):
    """check=False silences both the print and the warning (for repeat calls)."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 90))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([10, 90]),
                                 check=False)

    assert capsys.readouterr().out == '', 'check=False must not print'
    assert not [w for w in caught if issubclass(w.category, UserWarning)], \
        'check=False must not warn'


def test_warning_is_silent_below_threshold():
    """A negligible gap is reported in the log, not raised as a warning.

    A still-running free segment and the deliberate part0000 seed both produce
    small gaps every round; warning on them trains the reader to ignore warnings.
    """
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(99),
                           MockPathMissingBias(list('A'))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([99, 1]),
                                 threshold=0.05)

    assert not [w for w in caught if issubclass(w.category, UserWarning)], \
        '1 % missing must not warn'


def test_warning_does_not_repeat_the_printed_report():
    """One channel per message: the warning must not restate the remediation."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 90))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([10, 90]),
                                 threshold=0.05)

    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert msgs, 'a material gap must still raise a catchable warning'
    assert 'still-running' not in msgs[0], \
        f'the remediation prose belongs in the printed report only: {msgs[0]!r}'
    assert '90.0%' in msgs[0], msgs[0]
    # the actionable numbers must survive in the warnings channel: a caller may
    # capture warnings while discarding stdout.
    assert '10.0x' in msgs[0], f'inflation factor lost from the warning: {msgs[0]!r}'
    assert '1 of 2 paths' in msgs[0], msgs[0]


def test_uncovered_file_list_has_no_duplicates():
    """Several paths can share a first file; the list names each file once."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    shared = 'freeA/traj000003.part0001.xtc'
    paths = [MockPathMissingBias(list('A' * 10), fname=shared) for _ in range(4)]
    paths.append(MockPathMissingBias(list('A' * 10), fname='freeB/other.xtc'))
    pe = MockPathEnsemble(paths)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(
            pe, np.ones(5), lengths=np.full(5, 10), return_coverage=True)

    ex = cov['missing_examples']
    assert len(ex) == len(set(ex)), f'duplicated entries: {ex}'
    assert shared in ex and 'freeB/other.xtc' in ex


def test_check_threshold_is_configurable(capsys):
    """The caller can tighten the threshold that triggers escalation."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(99), MockPathMissingBias(list('A'))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([99, 1]),
                                 threshold=0.001)

    assert 'out-of-cache' in _flat(capsys.readouterr().out).lower()


def test_no_bias_anywhere_stays_quiet_about_remediation(capsys):
    """An unbiased run (all γ = 1 legitimately) must not be flagged."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPath(list('ARB'), [0.0, 0.0, 0.0])])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(1), lengths=np.array([3]))

    out = capsys.readouterr().out
    assert '100.0%' in out, out
    assert 'out-of-cache' not in _flat(out).lower(), out
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — coverage report formatting
# ════════════════════════════════════════════════════════════════════════════

def test_report_clean_run_is_one_reassuring_line():
    """A fully cached ensemble reports coverage without a remediation note."""
    from aimmd.pathensemble.bias_utils import (
        compute_bias_corrections, format_bias_cache_coverage)

    pe = MockPathEnsemble([_biased_path(10)])
    _, cov = compute_bias_corrections(pe, np.ones(1), lengths=np.array([10]),
                                      return_coverage=True)
    text = format_bias_cache_coverage(cov)

    assert '100.0%' in text, text
    assert 'state definition' not in text, \
        'the remediation note must not fire on a clean run'


def test_report_flags_high_missing_fraction_with_remediation():
    """Above threshold the report names the consequence and the remedies."""
    from aimmd.pathensemble.bias_utils import (
        compute_bias_corrections, format_bias_cache_coverage)

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 90))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(
            pe, np.ones(2), lengths=np.array([10, 90]), return_coverage=True)

    text = format_bias_cache_coverage(cov, threshold=0.05)
    flat = _flat(text).lower()

    assert '90.0%' in text, text
    for expected in ('kinetics', 'still-running', 'out-of-cache'):
        assert expected in flat, f'report should mention {expected!r}:\n{text}'
    assert 'state definition' not in flat, (
        'the note must not blame the state definition — coverage is set by segment '
        f'turnover, not by where A* sits:\n{text}')
    assert '10.0x' in text or '10x' in text, \
        f'report should state the implied overestimation factor:\n{text}'


def test_report_threshold_is_respected():
    """A small gap stays quiet; the same gap above threshold does not."""
    from aimmd.pathensemble.bias_utils import (
        compute_bias_corrections, format_bias_cache_coverage)

    pe = MockPathEnsemble([_biased_path(99),
                           MockPathMissingBias(list('A'))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(
            pe, np.ones(2), lengths=np.array([99, 1]), return_coverage=True)

    assert 'out-of-cache' not in _flat(
        format_bias_cache_coverage(cov, threshold=0.05))
    assert 'out-of-cache' in _flat(
        format_bias_cache_coverage(cov, threshold=0.001))


# ════════════════════════════════════════════════════════════════════════════
# Fixtures for the out-of-cache bias fallback
# ════════════════════════════════════════════════════════════════════════════

_KT = 8.314462618e-3 * 300.0


def _fake_bias_function(fname, ext='.xtc'):
    """Mirror of a real params.bias_function: reads the sibling _COLVAR."""
    import os as _os
    colvar = fname.replace(ext, '_COLVAR')
    if not _os.path.exists(colvar):
        return None
    data = np.loadtxt(colvar, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return (data[:, 2] + 15.0) / _KT


def _make_traj_dir(tmp_path, monkeypatch, n_colvar_rows, part_frames,
                   base='traj000001', ext='.xtc', write_colvar=True):
    """Build a fake free-trajectory directory.

    ``part_frames`` maps part number -> frame count; a zero-byte placeholder
    trajectory file is created for each. The cumulative COLVAR gets
    ``n_colvar_rows`` rows whose column 2 encodes the row index, so a slice can
    be checked against the exact rows it should have taken.
    """
    d = tmp_path / 'freeA'
    d.mkdir(exist_ok=True)
    for part in part_frames:
        (d / f'{base}.part{part:04d}{ext}').write_text('')
    if write_colvar:
        lines = ['#! FIELDS time d opes.bias']
        for i in range(n_colvar_rows):
            # column 2 = i - 15.0 so that bias_function returns exactly i / kT
            lines.append(f'{i * 10.0:.6f} 0.500000 {i - 15.0:.6f}')
        (d / 'COLVAR').write_text('\n'.join(lines) + '\n')
    counts = dict(part_frames)

    def frame_counter(path):
        import re as _re
        m = _re.search(r'\.part(\d{4})', path)
        return counts.get(int(m.group(1)), 0) if m else 0

    # the shared row-range helper reads frame counts through this
    monkeypatch.setattr('aimmd.params._methods._part_frame_count', frame_counter)
    return str(d), base


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — out-of-cache bias derivation
# ════════════════════════════════════════════════════════════════════════════

def test_derive_happy_path_middle_part(tmp_path, monkeypatch):
    """A middle part gets exactly its own slice of the cumulative COLVAR."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 10, 2: 10, 3: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function)

    assert bias is not None
    assert len(bias) == 10
    np.testing.assert_allclose(bias * _KT, np.arange(10, 20), atol=1e-6)


def test_derive_trailing_part_under_colvar_lag(tmp_path, monkeypatch):
    """A lagging COLVAR yields a SHORT slice, never rows from the previous part."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    # 25 rows on disk but 30 frames across parts -> the last part is 5 rows short
    d, base = _make_traj_dir(tmp_path, monkeypatch, 25, {1: 10, 2: 10, 3: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0003.xtc'), '.xtc', _fake_bias_function)

    assert bias is not None
    assert len(bias) == 5, 'must not pad, and must not borrow earlier rows'
    np.testing.assert_allclose(bias * _KT, np.arange(20, 25), atol=1e-6)


def test_derive_tolerates_surplus_colvar_rows(tmp_path, monkeypatch):
    """Rows beyond what the parts need are simply not reached.

    PLUMED can be a little ahead of the readable trajectory, so a surplus is a
    normal steady state. Ranges are resolved from the front, so it is harmless.
    """
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 40, {1: 10, 2: 10, 3: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function)

    assert bias is not None
    np.testing.assert_allclose(bias * _KT, np.arange(10, 20), atol=1e-6)


def test_derive_refuses_a_part_of_an_older_trajectory(tmp_path, monkeypatch):
    """The live COLVAR belongs to the newest trajectory only.

    When a new trajectory starts, the previous cumulative COLVAR is rotated
    away. Resolving an older trajectory's part against the live file would read
    a different trajectory's rows and produce plausible, misaligned bias.
    """
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, _ = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 30}, base='traj000002')
    # an older trajectory whose own COLVAR is gone
    open(os.path.join(d, 'traj000001.part0001.xtc'), 'w').close()

    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, 'traj000001.part0001.xtc'), '.xtc', _fake_bias_function) is None, \
        'must not resolve an older trajectory against the live COLVAR'

    # the newest one is still served
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, 'traj000002.part0001.xtc'), '.xtc', _fake_bias_function) is not None


def test_derive_returns_none_without_colvar(tmp_path, monkeypatch):
    """No cumulative COLVAR => nothing to derive from."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 0, {1: 10}, write_colvar=False)
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0001.xtc'), '.xtc', _fake_bias_function) is None


def test_derive_skips_the_part0000_seed(tmp_path, monkeypatch):
    """part0000 is the python-written seed: no PLUMED rows, so no offset."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 20, {0: 1, 1: 10, 2: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function)

    assert bias is not None
    np.testing.assert_allclose(bias * _KT, np.arange(10, 20), atol=1e-6)


def test_derive_writes_nothing_into_the_trajectory_directory(tmp_path, monkeypatch):
    """The whole point: no _COLVAR, no cache, no writes to the worker's dir."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 10, 2: 10, 3: 10})
    before = sorted(os.listdir(d))
    derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function)
    assert sorted(os.listdir(d)) == before, 'must not touch the run directory'


def test_derive_ignores_non_part_filenames(tmp_path, monkeypatch):
    """Shooting paths (path000001.xtc) have no cumulative COLVAR semantics."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 10})
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, 'path000001.xtc'), '.xtc', _fake_bias_function) is None


def test_derive_handles_single_row_colvar(tmp_path, monkeypatch):
    """A one-row COLVAR must not collapse to a 1-D indexing error."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 1, {1: 1})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0001.xtc'), '.xtc', _fake_bias_function)
    assert bias is not None and len(bias) == 1


def test_derive_returns_none_when_part_not_yet_on_disk(tmp_path, monkeypatch):
    """Asking for a part with no trajectory file yet is a no-op, not a crash."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 10, 2: 10})
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0009.xtc'), '.xtc', _fake_bias_function) is None


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — frame-weighted γ for partially covered paths
# ════════════════════════════════════════════════════════════════════════════

def test_a_short_bias_cache_needs_no_special_handling():
    """A cache shorter than its path is already weighted correctly.

    `Path._get('bias')` zero-pads to the path length (path/_get.py:167 and :205
    -> core.utils.extend_array, covered by tests/test_core_utils.py), so the
    tail of a short cache arrives as zeros and contributes exp(0) = 1. γ is then
    the same number a frame-weighted formula would produce, which is why
    `compute_bias_corrections` needs only one branch.
    """
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    v = np.log(10.0)
    padded = [v] * 4 + [0.0] * 6          # what _get hands over for a 4/10 cache
    pe = MockPathEnsemble([MockPath(list('A' * 10), padded)])
    gammas, cov = compute_bias_corrections(
        pe, np.ones(1), lengths=np.array([10]), return_coverage=True)

    np.testing.assert_allclose(gammas[0], (4 * 10.0 + 6 * 1.0) / 10, rtol=1e-9)
    assert cov['frac_weighted_length'] == pytest.approx(0.0), \
        'a padded short cache is not a coverage gap'


def test_gamma_unchanged_when_bias_covers_the_whole_path():
    """Regression: the fully covered case keeps the existing mean(exp) formula."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = [3.0, 0.0, 0.0, 3.0]
    pe = MockPathEnsemble([MockPath(list('ARRB'), bias)])
    gammas, cov = compute_bias_corrections(
        pe, np.ones(1), lengths=np.array([4]), return_coverage=True)

    np.testing.assert_allclose(gammas[0], np.mean(np.exp(bias)), rtol=1e-9)
    assert cov['frac_weighted_length'] == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════════════════
# L and gamma must count the same frames
#
# `PathEnsemble.n_frames` drops the boundary frames that `Path.split` leaves on
# each block, so that the blocks of one trajectory partition it. `L * gamma` is
# meant to be the boosted residence time of those L frames, i.e.
# sum_{counted} exp(bias). Averaging exp(bias) over the whole block instead
# makes `L * gamma` a sum over no frame set at all, and because the dropped
# frames sit in the bias-free reactive region it is one-signed: gamma is diluted
# toward 1, the reweighted dwell time shrinks, and the rate comes out too fast.
# The effect is largest for short blocks, where the two margins are most of the
# block — i.e. exactly the in-basin dwell segments of a fast-escaping run.
#
# An earlier revision of this file asserted the opposite ("reinterpreting it
# would change every existing biased run's numbers"). It would, and it should:
# the old numbers were not a sum of exp(bias) over any frame set. Use
# `trim_margins=False` to reproduce them.
# ════════════════════════════════════════════════════════════════════════════

class MockWindowedEnsemble(MockPathEnsemble):
    """MockPathEnsemble that also exposes `frame_windows`, as PathEnsemble does."""

    def __init__(self, paths, windows):
        super().__init__(paths)
        starts, stops = zip(*windows)
        self.frame_windows = (np.array(starts, dtype=int),
                              np.array(stops, dtype=int))


class TypedPath:
    """Stub with just what `PathEnsemble.frame_windows` reads."""

    def __init__(self, length, type_):
        self._length = length
        self.type = type_

    def __len__(self):
        return self._length


def test_frame_windows_reproduce_n_frames():
    """`n_frames` must be exactly `stop - start`, for every block shape."""
    from aimmd.pathensemble import PathEnsemble

    pe = PathEnsemble()
    # `path.type` is (first, middle, last, shooting)
    pe._paths = [TypedPath(1, 'AAAA'),      # single frame: whole path
                 TypedPath(4, 'AARA'),      # trailing boundary only
                 TypedPath(4, 'RARA'),      # both boundaries: an A dwell
                 TypedPath(5, 'ARBR'),      # a transition block
                 TypedPath(6, 'AAAA')]      # no boundary at all
    starts, stops = pe.frame_windows
    np.testing.assert_array_equal(stops - starts, pe.n_frames)
    np.testing.assert_array_equal(pe.n_frames, [1, 3, 2, 3, 6])


def test_gamma_averages_only_the_counted_frames():
    """γ must be the mean over the window, not over the whole block."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    # 'RAAR': two bias-free margins around a two-frame in-state dwell
    bias = [0.0, 6.0, 6.0, 0.0]
    pe = MockWindowedEnsemble([MockPath(list('RAAR'), bias)], [(1, 3)])
    gammas = compute_bias_corrections(pe, np.ones(1), lengths=np.array([2]))
    np.testing.assert_allclose(gammas[0], np.exp(6.0), rtol=1e-9)


def test_margin_frames_used_to_dilute_gamma():
    """Quantify the defect: L*γ was 0.5x the frame sum for a 2-frame dwell."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = np.array([0.0, 6.0, 6.0, 0.0])
    paths = [MockPath(list('RAAR'), bias)]
    exact = float(np.sum(np.exp(bias[1:3])))          # what L*γ should be

    fixed = compute_bias_corrections(
        MockWindowedEnsemble(paths, [(1, 3)]), np.ones(1),
        lengths=np.array([2]))[0] * 2
    old = compute_bias_corrections(
        MockWindowedEnsemble(paths, [(1, 3)]), np.ones(1),
        lengths=np.array([2]), trim_margins=False)[0] * 2

    np.testing.assert_allclose(fixed, exact, rtol=1e-9)
    assert old / exact == pytest.approx(
        (2 * (1.0 + np.exp(6.0)) / 4) / np.exp(6.0), rel=1e-9)
    assert old < exact, 'the old convention shortened the dwell time'


def test_trim_margins_false_reproduces_old_numbers():
    """The escape hatch must give back the pre-fix γ, bit for bit."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = list(np.linspace(0.0, 2.0, 10))
    pe = MockWindowedEnsemble([MockPath(list('A' * 10), bias)], [(1, 9)])
    gammas = compute_bias_corrections(
        pe, np.ones(1), lengths=np.array([8]), trim_margins=False)
    np.testing.assert_allclose(gammas[0], np.mean(np.exp(bias)), rtol=1e-9)


def test_windows_fall_back_to_whole_paths():
    """An object with no `frame_windows` keeps working (γ over whole paths)."""
    from aimmd.pathensemble.bias_utils import (compute_bias_corrections,
                                               counted_frame_windows)

    bias = [0.0, 6.0, 6.0, 0.0]
    pe = MockPathEnsemble([MockPath(list('RAAR'), bias)])
    assert counted_frame_windows(pe) is None
    gammas = compute_bias_corrections(pe, np.ones(1), lengths=np.array([2]))
    np.testing.assert_allclose(gammas[0], np.mean(np.exp(bias)), rtol=1e-9)


def test_reweighted_rates_are_a_frame_sum():
    """1/k must equal Σ_paths w · Σ_{counted frames} exp(bias), exactly."""
    from aimmd.pathensemble.bias_utils import bias_reweighted_rates

    bias_a = np.array([0.0, 6.0, 6.0, 6.0, 0.0])
    bias_b = np.array([0.0, 0.0, 0.0])
    paths = [MockPath(list('RAAAR'), bias_a), MockPath(list('ARB'), bias_b)]
    windows = [(1, 4), (1, 2)]
    pe = MockWindowedEnsemble(paths, windows)
    w1 = np.array([1.0, 2.0])
    lengths = np.array([3, 1])

    k12, k21, gamma1, gamma2 = bias_reweighted_rates(
        pe, w1, w1, lengths=lengths, states='ARB')

    expected = (1.0 * np.sum(np.exp(bias_a[1:4]))
                + 2.0 * np.sum(np.exp(bias_b[1:2])))
    np.testing.assert_allclose(1.0 / k12, expected, rtol=1e-9)
    np.testing.assert_allclose(k21, k12, rtol=1e-9)


def test_reweighted_rates_flag_inconsistent_lengths():
    """A caller-supplied L that is not the window is a silent estimator bug."""
    from aimmd.pathensemble.bias_utils import bias_reweighted_rates

    pe = MockWindowedEnsemble(
        [MockPath(list('RAAR'), [0.0, 6.0, 6.0, 0.0])], [(1, 3)])
    with pytest.warns(UserWarning, match='counted frame windows'):
        bias_reweighted_rates(pe, np.ones(1), np.ones(1),
                              lengths=np.array([4]), states='ARB')


# ════════════════════════════════════════════════════════════════════════════
# Integration test — the trainer falls back to out-of-cache derivation
# ════════════════════════════════════════════════════════════════════════════

def test_cache_bias_files_falls_back_to_out_of_cache(tmp_path, monkeypatch):
    """`_cache_bias_files` must derive bias for a part with no `_COLVAR` slice.

    The situation this guards against: the free segment's mdrun has not
    returned, so no slice exists, and before this fallback the whole segment
    entered the rate estimate with γ = 1.0.
    """
    import types
    from aimmd.worker import _train as train_mod

    d, base = _make_traj_dir(tmp_path, monkeypatch, 30, {1: 30})
    part = os.path.join(d, f'{base}.part0001.xtc')
    assert not os.path.exists(part.replace('.xtc', '_COLVAR')), \
        'the premise is that no slice exists'

    saved = {}
    monkeypatch.setattr(train_mod, 'MDA_CACHE',
                        types.SimpleNamespace(get=lambda f: types.SimpleNamespace(
                            trajectory=[None] * 30)))
    monkeypatch.setattr(train_mod, 'NPY_CACHE',
                        types.SimpleNamespace(get=lambda *a, **k: None,
                                              remove=lambda *a, **k: None))
    monkeypatch.setattr(train_mod, 'save_npy',
                        lambda fname, arr: saved.__setitem__(fname, arr))

    class _P:
        _fnames = [part]

    worker = types.SimpleNamespace(
        params=types.SimpleNamespace(trajectory_extension='.xtc'))
    train_mod.WorkerTrain._cache_bias_files(
        worker, [_P()], _fake_bias_function)

    assert saved, 'no bias cache was produced from the cumulative COLVAR'
    (arr,) = saved.values()
    assert len(arr) == 30
    np.testing.assert_allclose(arr * _KT, np.arange(30), atol=1e-6)


def test_derive_refuses_a_stride_mismatch(tmp_path, monkeypatch):
    """PRINT STRIDE not matching nstxout-compressed must not be papered over.

    Then the COLVAR holds an integer multiple of the rows the parts account for,
    and every offset is wrong by that factor. A small surplus is normal (PLUMED
    can be a row or two ahead of the readable frames), so the canary sits at the
    2x floor rather than at any surplus at all.
    """
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    # 60 rows for 3 parts x 10 frames: one row per half-frame
    d, base = _make_traj_dir(tmp_path, monkeypatch, 60, {1: 10, 2: 10, 3: 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bias = derive_bias_from_cumulative_colvar(
            os.path.join(d, f'{base}.part0003.xtc'), '.xtc', _fake_bias_function)

    assert bias is None, 'a 2x row surplus means the offsets are all wrong'
    assert [w for w in caught if issubclass(w.category, UserWarning)], \
        'refusing silently would hide a misconfigured PRINT STRIDE'
