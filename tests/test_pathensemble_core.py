import numpy as np

from aimmd.pathensemble import PathEnsemble
from tests._helpers_unit import build_path


def test_pathensemble_collection_and_filters(tmp_path):
    """A `PathEnsemble` should expose vectorized views over its member paths."""

    path1 = build_path(tmp_path, stem="p1", positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32))
    path2 = build_path(tmp_path, stem="p2", positions=np.array([[[1, 0, 0]], [[0, 0, 0]], [[-1, 0, 0]]], dtype=np.float32))
    ensemble = PathEnsemble(path1, path2)

    assert ensemble.n_paths == 2
    # `type` is the compact AIMMD path signature: initial, middle, final, and
    # shooting states. We assert the exact current encoding here.
    np.testing.assert_array_equal(ensemble.types(), np.array(["ARBR", "BRAR"]))
    assert ensemble.are_transitions().all()
    np.testing.assert_allclose(ensemble.frame(0).positions, path1[0].positions)

    # Filtering by type patterns should return the expected subset and joining
    # should concatenate path frames into one larger `Path`.
    extracted = ensemble.extract("ARB.")
    assert len(extracted) == 1
    joined = ensemble.join()
    assert len(joined) == len(path1) + len(path2)


def test_pathensemble_weights_and_shooting_results(tmp_path):
    """Weight views should read and write through to the underlying paths."""

    path1 = build_path(tmp_path, stem="w1", weight=1.5)
    path2 = build_path(tmp_path, stem="w2", weight=0.5)
    ensemble = PathEnsemble(path1, path2)

    np.testing.assert_allclose(ensemble.weights, np.array([1.5, 0.5]))
    # Assigning through the ensemble property broadcasts back to each stored path.
    ensemble.weights[:] = 2.0
    np.testing.assert_allclose(ensemble.weights, np.array([2.0, 2.0]))
    assert ensemble.shooting_results().shape == (2, 2)
