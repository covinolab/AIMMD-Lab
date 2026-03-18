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
