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
