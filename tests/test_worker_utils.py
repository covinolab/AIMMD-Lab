import numpy as np

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker.utils import (
    accept_or_reject_last_path,
    register_path,
    rescale_bins,
    select_shooting_point,
)
from tests._helpers_unit import build_path


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
    shooting_point = select_shooting_point(
        chain, params, target_state="A"
    )
    assert shooting_point.n_atoms > 0

    last = build_path(tmp_path, stem="last")
    new_chain = PathEnsemble(last)
    accept_or_reject_last_path(new_chain, params)
    # A non-transition path is rejected immediately in the TPS acceptance step,
    # which the code represents by setting its weight to zero.
    assert new_chain[-1].weight == 0.0
