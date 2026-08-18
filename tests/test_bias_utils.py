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
"""

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

    This is the calixarene-G2 failure mode: a single free-basin trajectory
    that never terminated carries most of the dwell time, so a path-count
    metric reads 20 % while the quantity that actually enters the rate is
    85 %.
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
    assert 'state definition' in _flat(out).lower(), out


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
    assert 'state definition' not in small, small
    assert '1.0%' in small, small

    large = _warn_text([10, 90], 0.05)     # 90 % missing — above threshold
    assert 'state definition' in large, large
    assert 'from-basin excursions' in large, large


def test_check_threshold_is_configurable(capsys):
    """The caller can tighten the threshold that triggers escalation."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([_biased_path(99), MockPathMissingBias(list('A'))])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(2), lengths=np.array([99, 1]),
                                 threshold=0.001)

    assert 'state definition' in _flat(capsys.readouterr().out).lower()


def test_no_bias_anywhere_stays_quiet_about_remediation(capsys):
    """An unbiased run (all γ = 1 legitimately) must not be flagged."""
    from aimmd.pathensemble.bias_utils import compute_bias_corrections

    pe = MockPathEnsemble([MockPath(list('ARB'), [0.0, 0.0, 0.0])])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_bias_corrections(pe, np.ones(1), lengths=np.array([3]))

    out = capsys.readouterr().out
    assert '100.0%' in out, out
    assert 'state definition' not in _flat(out).lower(), out
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
    for expected in ('kinetics', 'state definition', 'excursion'):
        assert expected in flat, f'report should mention {expected!r}:\n{text}'
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

    assert 'state definition' not in _flat(
        format_bias_cache_coverage(cov, threshold=0.05))
    assert 'state definition' in _flat(
        format_bias_cache_coverage(cov, threshold=0.001))
