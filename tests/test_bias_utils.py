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
test_coverage_warning_reads_well_when_nothing_is_cached — 'unbounded', not 'infx'
test_coverage_empty_ensemble_is_safe         — no weighted paths → no ZeroDivisionError
test_coverage_warning_quantifies_the_gap     — UserWarning carries the fraction
test_report_clean_run_is_one_reassuring_line — no remediation note when healthy
test_report_flags_high_missing_fraction_with_remediation — names cause and remedies
test_report_threshold_is_respected           — threshold controls escalation

The check is default behaviour, not opt-in
test_check_runs_by_default_and_prints_coverage — plain call reports coverage
test_check_by_default_escalates_when_problematic — plain call escalates
test_check_can_be_suppressed                 — check=False silences print+warning
test_warning_carries_remediation_only_when_problematic — warning text escalates
test_check_threshold_is_configurable         — caller can tighten the threshold
test_no_bias_anywhere_stays_quiet_about_remediation — unbiased run not flagged

Out-of-cache bias derivation (fallback for a still-running free segment)
test_derive_happy_path_middle_part           — a middle part gets its own rows
test_derive_trailing_part_under_colvar_lag   — short slice, never borrowed rows
test_derive_refuses_when_colvar_longer_than_parts — unknown alignment -> None
test_derive_returns_none_without_colvar      — no COLVAR, no derivation
test_derive_skips_the_part0000_seed          — seed part carries no PLUMED rows
test_derive_writes_nothing_into_the_trajectory_directory — read-only
test_derive_ignores_non_part_filenames       — shooting paths untouched
test_derive_handles_single_row_colvar        — 1-D loadtxt result
test_derive_returns_none_when_part_not_yet_on_disk — no crash
test_cache_bias_files_falls_back_to_out_of_cache   — trainer wiring

Frame-weighted gamma for partially covered paths
test_gamma_is_frame_weighted_for_a_short_bias_array — uncovered frames -> exp(0)
test_short_bias_array_counts_as_partial_coverage    — coverage counted in frames
test_gamma_unchanged_when_bias_covers_the_whole_path — regression
test_gamma_unchanged_when_bias_longer_than_margin_excluded_length — regression
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


def test_coverage_warning_reads_well_when_nothing_is_cached():
    """An infinite inflation factor must not render as 'infx' in the warning."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPathMissingBias(list('A' * 10))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(1), lengths=np.array([10]))

    msg = str(caught[0].message)
    assert 'infx' not in msg, f'unreadable inflation factor: {msg!r}'
    assert 'unbounded' in msg, f'expected an explicit wording, got: {msg!r}'


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


def test_coverage_warning_quantifies_the_gap():
    """The existing UserWarning now carries the weighted fraction."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(10),
                           MockPathMissingBias(list('A' * 90))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([10, 90]),
                                 return_coverage=True)

    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert msgs, 'expected a UserWarning for the missing cache'
    assert '90.0%' in msgs[0] or '90%' in msgs[0], \
        f'warning should quantify the gap, got: {msgs[0]!r}'


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


def test_warning_carries_remediation_only_when_problematic():
    """A small gap warns plainly; a large one warns with what to change."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    def _warn_text(lengths, threshold):
        pe = MockPathEnsemble([_biased_path(int(lengths[0])),
                               MockPathMissingBias(list('A' * int(lengths[1])))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            compute_bias_corrections(pe, np.ones(2), lengths=np.asarray(lengths),
                                     threshold=threshold)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
        assert msgs, 'a missing bias cache must always warn'
        return msgs[0]

    small = _warn_text([99, 1], 0.05)      # 1 % missing — below threshold
    assert 'out-of-cache' not in small, small
    assert '1.0%' in small, small

    large = _warn_text([10, 90], 0.05)     # 90 % missing — above threshold
    assert 'out-of-cache' in large, large
    assert 'still-running' in large, large


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


def _make_traj_dir(tmp_path, n_colvar_rows, part_frames, base='traj000001',
                   ext='.xtc', write_colvar=True):
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

    return str(d), base, frame_counter


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — out-of-cache bias derivation
# ════════════════════════════════════════════════════════════════════════════

def test_derive_happy_path_middle_part(tmp_path):
    """A middle part gets exactly its own slice of the cumulative COLVAR."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 30, {1: 10, 2: 10, 3: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc)

    assert bias is not None
    assert len(bias) == 10
    np.testing.assert_allclose(bias * _KT, np.arange(10, 20), atol=1e-6)


def test_derive_trailing_part_under_colvar_lag(tmp_path):
    """A lagging COLVAR yields a SHORT slice, never rows from the previous part."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    # 25 rows on disk but 30 frames across parts -> the last part is 5 rows short
    d, base, fc = _make_traj_dir(tmp_path, 25, {1: 10, 2: 10, 3: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0003.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc)

    assert bias is not None
    assert len(bias) == 5, 'must not pad, and must not borrow earlier rows'
    np.testing.assert_allclose(bias * _KT, np.arange(20, 25), atol=1e-6)


def test_derive_refuses_when_colvar_longer_than_parts(tmp_path):
    """More COLVAR rows than the parts account for => refuse, do not guess.

    That happens when the COLVAR belongs to a different trajectory (it is
    rotated to bck.0.COLVAR when a new one starts) or when PRINT STRIDE does
    not match nstxout-compressed. Either way the alignment is unknown.
    """
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 40, {1: 10, 2: 10, 3: 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bias = derive_bias_from_cumulative_colvar(
            os.path.join(d, f'{base}.part0003.xtc'), '.xtc', _fake_bias_function,
            frame_counter=fc)

    assert bias is None
    assert [w for w in caught if issubclass(w.category, UserWarning)], \
        'refusing silently would hide a misaligned COLVAR'


def test_derive_returns_none_without_colvar(tmp_path):
    """No cumulative COLVAR => nothing to derive from."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 0, {1: 10}, write_colvar=False)
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0001.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc) is None


def test_derive_skips_the_part0000_seed(tmp_path):
    """part0000 is the python-written seed: no PLUMED rows, so no offset."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 20, {0: 1, 1: 10, 2: 10})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc)

    assert bias is not None
    np.testing.assert_allclose(bias * _KT, np.arange(10, 20), atol=1e-6)


def test_derive_writes_nothing_into_the_trajectory_directory(tmp_path):
    """The whole point: no _COLVAR, no cache, no writes to the worker's dir."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 30, {1: 10, 2: 10, 3: 10})
    before = sorted(os.listdir(d))
    derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0002.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc)
    assert sorted(os.listdir(d)) == before, 'must not touch the run directory'


def test_derive_ignores_non_part_filenames(tmp_path):
    """Shooting paths (path000001.xtc) have no cumulative COLVAR semantics."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 30, {1: 10})
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, 'path000001.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc) is None


def test_derive_handles_single_row_colvar(tmp_path):
    """A one-row COLVAR must not collapse to a 1-D indexing error."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 1, {1: 1})
    bias = derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0001.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc)
    assert bias is not None and len(bias) == 1


def test_derive_returns_none_when_part_not_yet_on_disk(tmp_path):
    """Asking for a part with no trajectory file yet is a no-op, not a crash."""
    from aimmd.pathensemble.bias_utils import derive_bias_from_cumulative_colvar

    d, base, fc = _make_traj_dir(tmp_path, 30, {1: 10, 2: 10})
    assert derive_bias_from_cumulative_colvar(
        os.path.join(d, f'{base}.part0009.xtc'), '.xtc', _fake_bias_function,
        frame_counter=fc) is None


# ════════════════════════════════════════════════════════════════════════════
# Unit tests — frame-weighted γ for partially covered paths
# ════════════════════════════════════════════════════════════════════════════

def test_gamma_is_frame_weighted_for_a_short_bias_array():
    """A short bias array must not be averaged as if it covered the path.

    The out-of-cache fallback returns a short array when PLUMED has not flushed
    the tail yet. Averaging exp(bias) over only the covered frames would apply
    the well's correction to frames that have no bias information at all.
    """
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    v = np.log(10.0)                      # exp(bias) = 10 on covered frames
    pe = MockPathEnsemble([MockPath(list('A' * 10), [v] * 4)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        gammas, cov = compute_bias_corrections(
            pe, np.ones(1), lengths=np.array([10]), return_coverage=True)

    # (4 covered frames * 10 + 6 uncovered frames * exp(0)) / 10
    np.testing.assert_allclose(gammas[0], (4 * 10.0 + 6 * 1.0) / 10, rtol=1e-9)
    assert gammas[0] != pytest.approx(10.0), 'must not be the covered-only mean'


def test_short_bias_array_counts_as_partial_coverage():
    """Coverage is measured in frames, so a short array is partly missing."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPath(list('A' * 10), [1.0] * 4)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, cov = compute_bias_corrections(
            pe, np.ones(1), lengths=np.array([10]), return_coverage=True)

    assert cov['frac_weighted_length'] == pytest.approx(6 / 10)
    assert cov['n_missing'] == 1, 'a partially covered path counts as affected'


def test_gamma_unchanged_when_bias_covers_the_whole_path():
    """Regression: the fully covered case keeps the existing mean(exp) formula."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = [3.0, 0.0, 0.0, 3.0]
    pe = MockPathEnsemble([MockPath(list('ARRB'), bias)])
    gammas, cov = compute_bias_corrections(
        pe, np.ones(1), lengths=np.array([4]), return_coverage=True)

    np.testing.assert_allclose(gammas[0], np.mean(np.exp(bias)), rtol=1e-9)
    assert cov['frac_weighted_length'] == pytest.approx(0.0)


def test_gamma_unchanged_when_bias_longer_than_margin_excluded_length():
    """`pe.n_frames` excludes boundary frames, so len(bias) > L is normal.

    In that case γ must stay the mean over the whole path exactly as before —
    reinterpreting it would change every existing biased run's numbers.
    """
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    bias = list(np.linspace(0.0, 2.0, 10))
    pe = MockPathEnsemble([MockPath(list('A' * 10), bias)])
    gammas, cov = compute_bias_corrections(
        pe, np.ones(1), lengths=np.array([8]), return_coverage=True)

    np.testing.assert_allclose(gammas[0], np.mean(np.exp(bias)), rtol=1e-9)
    assert cov['frac_weighted_length'] == pytest.approx(0.0)


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

    d, base, fc = _make_traj_dir(tmp_path, 30, {1: 30})
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
    # real frame counts come from MDA_CACHE inside the helper; inject ours
    monkeypatch.setattr('aimmd.pathensemble.bias_utils._default_frame_counter', fc)

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
