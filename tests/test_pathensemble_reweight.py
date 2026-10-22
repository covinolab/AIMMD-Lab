import numpy as np

from aimmd.pathensemble import PathEnsemble
from aimmd.pathensemble.reweight import (
    compute_crossing_probability,
    compute_shooting_density,
    reweight_excursions,
    uniformize_factors,
)
from tests._helpers_unit import build_path


def test_low_level_reweight_helpers():
    """Exercise the numerical core used by path-ensemble reweighting."""

    density = compute_shooting_density(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), 0.0, neighbors=3)
    assert np.isfinite(density)

    # Uniformization rescales factors locally in shooting-value space, so it
    # should keep them positive while damping local amplitude differences.
    factors = uniformize_factors(
        np.array([1.0, 2.0, 4.0]), np.array([-1.0, 0.0, 1.0]), norm=2, cutoff=10.0
    )
    assert np.all(factors > 0)

    extremes, xP = compute_crossing_probability(
        shooting_values=np.array([0.1, 0.3, 0.8]),
        extremes=np.array([0.5, 1.0, 1.5]),
        free_extremes=np.array([0.2, 0.4]),
        free_threshold=1,
    )
    assert len(extremes) == len(xP)

    result = reweight_excursions(
        np.array([0.0, 0.2, 0.5]),
        np.array([0.5, 1.0, np.inf]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.2, 0.5, 1.0]),
        np.array([1.0, 0.5, 0.1]),
    )
    weights, order = result[0], result[1]
    assert len(weights) == len(order) == 3
    assert np.all(np.isfinite(weights))


def test_free_transitions_are_at_risk_at_every_level(tmp_path):
    """A free transition leaves A like every free excursion, so it belongs to the
    at-risk mass m(e) at every level. Its first frame is in the state, where the
    value pass leaves 0, which is no shooting value: taken at face value it hid
    the transition from m(e) at every e < 0, and each free A->A excursion with an
    extreme below 0 weighed 1/N_AB + 1/n(>=e) instead of 1/N_AB."""
    from aimmd.pathensemble.utils import assemble_pathensemble

    # one free trajectory from A: three A->A excursions (extremes -3, -1, -4,
    # all below 0) and one A->B transition; 0 outside R, as the value pass leaves
    states = list('AARAARRAARAARRRBB')
    values = [0, 0, -3, 0, 0, -2, -1, 0, 0, -4, 0, 0, -2, 0, 2, 0, 0]
    positions = np.array([[[v / 10, 0, 0]] for v in values], dtype=np.float32)
    path = build_path(tmp_path, stem="free", positions=positions, states=states,
                      values=values, shooting_index=0)
    ensemble = assemble_pathensemble([path])
    types = ensemble.types()
    assert np.sum(types == 'ARAA') == 3 and np.sum(types == 'ARBA') == 1

    # free-only: every free excursion from A weighs 1/N_AB (N_AB = 1); before
    # the fix the A->A excursions weighed 4/3, 3/2 and 2
    weights = ensemble.reweight("ARB", free_threshold=1)[0]
    assert np.allclose(weights[types == 'ARAA'], 1.0)
    assert np.allclose(weights[types == 'ARBA'], 1.0)


def test_pathensemble_reweight_wrapper(tmp_path):
    """The high-level wrapper should return aligned diagnostics and weights."""

    path1 = build_path(tmp_path, stem="rw1", positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32))
    path2 = build_path(tmp_path, stem="rw2", positions=np.array([[[-1, 0, 0]], [[0.2, 0, 0]], [[-1, 0, 0]]], dtype=np.float32))
    ensemble = PathEnsemble(path1, path2)

    result = ensemble.reweight("ARB", free_threshold=1)
    weights, indices, factors, shooting_values, extremes, xP_extremes, xP = result[:7]
    # The first output is always one weight per stored path, while the following
    # arrays describe only the excursion subset that actually enters the
    # reweighting calculation.
    assert weights.shape == (2,)
    assert np.all(np.isfinite(factors))
    assert len(indices) == len(factors) == len(shooting_values) == len(extremes)
    assert len(xP_extremes) == len(xP)
