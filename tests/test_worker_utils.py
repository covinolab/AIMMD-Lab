import numpy as np

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker.utils import (
    accept_or_reject_last_path,
    register_path,
    rescale_bins,
    select_shooting_point,
    update_selection_pool,
)
from tests._helpers_unit import build_path


def test_update_selection_pool_and_rescale_bins(tmp_path):
    """Selection-pool maintenance should preserve recent useful sampling paths."""

    chain = PathEnsemble(
        build_path(tmp_path, stem="chain0"),
        build_path(tmp_path, stem="chain1", positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32)),
    )
    pool = PathEnsemble(build_path(tmp_path, stem="pool0"))
    initial = PathEnsemble(build_path(tmp_path, stem="init0"), build_path(tmp_path, stem="init1"))

    updated = update_selection_pool(pool, size=2, chain=chain, initial_paths=initial, at_least_one="ARB")
    assert len(updated) == 2
    # The pool should still contain at least one transition-like path when that
    # safeguard is requested.
    assert updated.extract("ARB.", "BRA.")

    bins = np.array([-np.inf, -1.0, 1.0, np.inf])
    rescale_bins(bins, [-1.0, 1.0], [-2.0, 2.0])
    assert np.isneginf(bins[0]) and np.isposinf(bins[-1])
    np.testing.assert_allclose(bins[1:-1], np.array([-2.0, 2.0]))


def test_register_select_and_accept_path(tmp_path, monkeypatch):
    """Cover low-level path registration and simplified shooting/TPS logic."""

    path = build_path(tmp_path, stem="back")
    forw = build_path(tmp_path, stem="forw")
    path = path + forw[1:]
    chain = PathEnsemble()

    register_path(path, chain, eneconv=None)
    assert len(chain) == 1
    # `register_path` writes a final single-file path into the chain directory.
    assert chain[0].fname.endswith(".xtc")

    params = aimmd.Params.placeholder
    params.__dict__["states"] = "ARB"
    params.__dict__["selection_pool_size"] = 1
    shooting_point = select_shooting_point(
        PathEnsemble(chain[0]), params, str(tmp_path), target_state="A"
    )
    # For non-reactive targets the selector picks a random internal frame from
    # the pool, so the returned timestep simply needs to be a valid frame.
    assert shooting_point.n_atoms > 0

    last = build_path(tmp_path, stem="last")
    new_chain = PathEnsemble(last)
    accept_or_reject_last_path(new_chain, params)
    # A non-transition path is rejected immediately in the TPS acceptance step,
    # which the code represents by setting its weight to zero.
    assert new_chain[-1].weight == 0.0
