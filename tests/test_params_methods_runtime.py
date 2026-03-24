"""Runtime-oriented unit tests for `aimmd.params._methods`.

These tests focus on the engine-facing methods that were previously almost
untouched by coverage. They use tiny synthetic files and heavy monkeypatching
so we can validate behavior without running real GROMACS jobs.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cache.npy import save_npy
from tests._helpers_unit import TinyNetwork, build_path, write_trajectory


def _toy_params(tmp_path):
    """Start from `Params.placeholder` and fill only runtime fields we need."""

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        engine="toy",
        # Use zero-temperature generation in the tests so the toy initializer
        # does not require velocity data on the synthetic input frames.
        gen_temperature=0,
        masses=None,
        trajectory_extension=".xtc",
        toy_mdrun=lambda ts: ts,
        toy_slowdown=0.0,
        topology="unused.top",
        sorted_states="ARB",
        network=TinyNetwork(),
        parent=Path(tmp_path),
    )
    return params


def test_initialize_simulation_toy_writes_reusable_files(tmp_path):
    """Toy-engine initialization should write a small trajectory immediately.

    The method supports both a single frame and a short `Path` history. When a
    `Path` is supplied, the earlier frames are written as `part0000` so the toy
    engine can continue from a tiny synthetic history segment.
    """

    params = _toy_params(tmp_path)
    path = build_path(
        tmp_path,
        stem="history",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )

    params.initialize_simulation(path, str(Path(tmp_path) / "seed"))
    assert (tmp_path / "seed.part0000.xtc").exists()


def test_run_simulation_gromacs_builds_expected_command(monkeypatch, tmp_path):
    """The GROMACS branch should assemble command-line flags deterministically."""

    params = _toy_params(tmp_path)
    params.__dict__.update(engine="gromacs", gmx_mdrun="gmx mdrun")
    called = {}

    monkeypatch.setattr(
        "aimmd.params._methods.execute_command",
        lambda command, **kwargs: called.update(command=command, kwargs=kwargs) or 17,
    )

    result = params.run_simulation("prod", backup=False, cpt=0.2, noappend=True, walltime=5)
    assert result == 17
    assert "-deffnm prod" in called["command"]
    assert "-nobackup" in called["command"]
    assert "-cpi prod.cpt -cpt 0.2" in called["command"]
    assert "-noappend" in called["command"]
    assert called["kwargs"]["walltime"] == 5


def test_check_if_initialized_and_copy_behave_consistently(tmp_path):
    """Initialization checks should match the on-disk conventions exactly."""

    params = _toy_params(tmp_path)
    deffnm = tmp_path / "traj"
    deffnm.with_suffix(".xtc").write_bytes(b"x")

    assert params.check_if_initialized(str(deffnm))
    cloned = params.copy()
    assert cloned is not params
    assert cloned.sorted_states == params.sorted_states
    assert cloned.initial_paths == params.initial_paths


def test_update_network_loading_roundtrip(tmp_path):
    """Saved network/bins artifacts should be reloadable through Params helpers."""

    params = _toy_params(tmp_path)
    states = params.sorted_states
    network_fname = tmp_path / f"network{states}.h5"

    torch.save(params.network.state_dict(), network_fname)
    
    params.update_network(str(tmp_path), timeout=0.0, raise_if_failure=True)


def test_check_engine_returns_success_and_failure(monkeypatch, tmp_path):
    """`check_engine` should map its internal exceptions onto 0/1 return codes."""

    params = _toy_params(tmp_path)
    params.__dict__["parent"] = tmp_path
    params.__dict__["_universe"] = None
    initial = build_path(
        tmp_path,
        stem="initial_engine",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]]], dtype=np.float32),
    )
    params.__dict__["initial_paths"] = aimmd.PathEnsemble(initial)

    def fake_initialize(ts, deffnm, timeout=10.0, verbose=True):
        Path(f"{deffnm}.xtc").write_bytes(b"x")

    params.__dict__["initialize_simulation"] = fake_initialize
    params.__dict__["run_simulation"] = lambda deffnm, walltime=10.0: None

    assert params.check_engine(deffnm=".params_check_engine", timeout=0.0) == 0

    params.__dict__["run_simulation"] = lambda deffnm, walltime=10.0: (_ for _ in ()).throw(RuntimeError("boom"))
    assert params.check_engine(deffnm=".params_check_engine", timeout=0.0) == 1


def test_minimize_energy_rejects_non_gromacs(tmp_path):
    """Energy minimization is a strict GROMACS-only helper."""

    params = _toy_params(tmp_path)
    trajectory = write_trajectory(tmp_path, stem="traj_for_em")

    with pytest.raises(TypeError):
        params.minimize_energy(trajectory, tmp_path / "out.xtc")
