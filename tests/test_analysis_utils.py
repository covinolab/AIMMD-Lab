import numpy as np

from aimmd.analysis.utils import (
    bin_centers,
    binomial_mean_and_confidence_interval,
    compute_bins,
    extract_rate_estimates_from_log_file,
    find_path_lineages,
    merge_empty_bins,
    merge_marginal_bins,
    solve_committor_by_relaxation,
)
from aimmd.pathensemble import PathEnsemble
from tests._helpers_unit import build_path


def test_bin_helpers_and_statistics(tmp_path):
    """Check bin construction/merging and simple statistical helpers."""

    path1 = build_path(tmp_path, stem="a", positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32))
    path2 = build_path(tmp_path, stem="b", positions=np.array([[[1, 0, 0]], [[0, 0, 0]], [[-1, 0, 0]]], dtype=np.float32))
    ensemble = PathEnsemble(path1, path2)

    # `compute_bins` should create an infinite left/right margin when requested.
    bins = compute_bins(ensemble, nbins=4, marginal_bins="all")
    assert np.isneginf(bins[0]) and np.isposinf(bins[-1])
    centers = bin_centers(np.array([-np.inf, -1.0, 1.0, np.inf]))
    np.testing.assert_allclose(centers, np.array([-2.0, 0.0, 2.0]))

    # Empty central bins are merged into the nearest occupied bins moving away
    # from the transition-state center, matching the current implementation.
    merged = merge_empty_bins(np.array([-2, -1, 0, 1, 2], dtype=float), [0, 3], np.array([1, 0, 0, 2]))
    np.testing.assert_array_equal(merged[0], np.array([-2.0, 0.0, 2.0]))

    marginal = merge_marginal_bins(np.array([-2, -1, 0, 1, 2], dtype=float), np.array([-1.8, -1.7, 1.7, 1.8]), min_values=1)
    assert len(marginal[0]) <= 5

    mean, lo, hi = binomial_mean_and_confidence_interval(6, 10)
    assert lo <= mean <= hi


def test_log_parsing_relaxation_and_lineages(tmp_path):
    """Cover the heterogeneous analysis helpers that operate on files and grids."""

    log_file = tmp_path / "train.log"
    log_file.write_text(
        "k12 estimate: 1.0 [1/dt]\n"
        "k21 estimate: 2.0 [1/dt]\n"
        "100 frames simulated\n"
    )
    t, k12, k21 = extract_rate_estimates_from_log_file(log_file)
    np.testing.assert_array_equal(t, np.array([100.0]))
    np.testing.assert_array_equal(k12, np.array([1.0]))
    np.testing.assert_array_equal(k21, np.array([2.0]))

    # A zero-drift toy system on a small grid is enough to exercise the
    # relaxation solver without requiring any physical interpretation.
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6))
    A = np.zeros_like(X, dtype=bool)
    B = np.zeros_like(X, dtype=bool)
    A[:, 0] = True
    B[:, -1] = True
    q = solve_committor_by_relaxation(
        X,
        Y,
        np.zeros_like(X),
        np.zeros_like(Y),
        A,
        B,
        np.full_like(X, 0.5),
        progress=[1],
    )
    assert q.shape == X.shape

    # `find_path_lineages` does not infer genealogy from paths directly: it
    # parses the worker log, so we create the minimal on-disk chain/log layout
    # that mirrors what AIMMD writes during shooting.
    chain_dir = tmp_path / "chainR0"
    chain_dir.mkdir()
    p0 = build_path(chain_dir, stem="path000000")
    p1 = build_path(chain_dir, stem="path000001")
    (chain_dir / "worker.log").write_text(
        f"Selecting shooting point for '{p1.fname[:-4]}' (value: 0.0)\n"
        f"=== selecting path '{p0.fname}'\n"
        "Shooting initialization completed\n"
    )
    chain = PathEnsemble(p0, p1)
    find_path_lineages(chain)
    assert chain[1].__dict__["_previous"] == p0
