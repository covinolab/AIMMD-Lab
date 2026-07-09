"""Tests for GROMACS availability behavior.

``import aimmd`` must not hard-fail when GROMACS is absent: it emits a
``RuntimeWarning`` and leaves ``_config.GROMACS = None`` so import, analysis, and
training stay usable. The hard requirement is moved to the GROMACS-engine
sampling entry points (``Launcher.run`` / ``Launcher.create_job`` /
shoot+free workers), which raise ``EnvironmentError`` via
``_config.require_gromacs()``. Toy-engine sampling needs no GROMACS.

All tests force the gmx-absent state via monkeypatch, so they pass with or
without a real GROMACS install (e.g. a gmx-free CI lane).
"""

import shutil
import warnings
from types import SimpleNamespace

import pytest

import aimmd
import aimmd._config as _config
import aimmd._init as _init
from aimmd.launcher import Launcher
from aimmd.worker import Worker
from aimmd.worker._run import WorkerRun
from tests._helpers_unit import build_path


# --------------------------------------------------------------------------- #
# (a) import-time resolution: warn + None, never raise
# --------------------------------------------------------------------------- #

def test_resolve_gromacs_warns_and_sets_none(monkeypatch):
    """No gmx on PATH -> RuntimeWarning, GROMACS left None, no raise."""
    # Register the attribute so monkeypatch restores the real value on teardown
    # (the function below overwrites _config.GROMACS directly).
    monkeypatch.setattr(_config, "GROMACS", "sentinel", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.warns(RuntimeWarning, match="GROMACS exec not found"):
        result = _init._resolve_gromacs()

    assert result is None
    assert _config.GROMACS is None


def test_resolve_gromacs_sets_path_when_present(monkeypatch):
    """gmx present -> returns its path, sets _config.GROMACS, emits no warning."""
    monkeypatch.setattr(_config, "GROMACS", "sentinel", raising=False)
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/gmx" if name == "gmx" else None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        result = _init._resolve_gromacs()

    assert result == "/usr/bin/gmx"
    assert _config.GROMACS == "/usr/bin/gmx"


# --------------------------------------------------------------------------- #
# (b) the guard helper
# --------------------------------------------------------------------------- #

def test_require_gromacs_raises_when_none(monkeypatch):
    monkeypatch.setattr(_config, "GROMACS", None)
    with pytest.raises(EnvironmentError, match="GROMACS exec not found"):
        _config.require_gromacs()


def test_require_gromacs_noop_when_present(monkeypatch):
    monkeypatch.setattr(_config, "GROMACS", "/usr/bin/gmx")
    assert _config.require_gromacs() is None


# --------------------------------------------------------------------------- #
# (c) launcher entry points require GROMACS for the gromacs engine
# --------------------------------------------------------------------------- #

def _make_gromacs_launcher(tmp_path, monkeypatch):
    """Minimal single-run Launcher whose placeholder params use the gmx engine.

    Construction itself must succeed (it requires initial paths and a saved
    params file); the guard we test is the first statement of run()/create_job().
    Mirrors the fixture in tests/test_worker_launcher_lowlevel.py.
    """
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_cpus", lambda: 4)
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_gpus", lambda: 0)
    monkeypatch.setattr(
        "aimmd.params._io.ParamsIO.save",
        lambda self, *args, **kwargs: str(tmp_path / "params.py"))
    params = aimmd.Params.placeholder  # `engine` defaults to 'gromacs'
    params.__dict__["initial_paths"] = aimmd.PathEnsemble(
        build_path(tmp_path, stem="guard_init"))
    params.__dict__["path"] = tmp_path / "params.py"
    params.__dict__["states"] = "ARB"
    (tmp_path / "params.py").write_text("# placeholder params file\n")
    assert params.engine == "gromacs"
    return Launcher([params], [str(tmp_path / "run")])


def test_launcher_run_requires_gromacs(tmp_path, monkeypatch):
    launcher = _make_gromacs_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr(_config, "GROMACS", None)
    with pytest.raises(EnvironmentError, match="GROMACS exec not found"):
        launcher.run(1, walltime=1)


def test_launcher_create_job_requires_gromacs(tmp_path, monkeypatch):
    launcher = _make_gromacs_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr(_config, "GROMACS", None)
    job = tmp_path / "job.sh"
    with pytest.raises(EnvironmentError, match="GROMACS exec not found"):
        launcher.create_job(str(job))
    # The guard fires before anything is written to disk.
    assert not job.exists()


# --------------------------------------------------------------------------- #
# (c') worker shoot/free require GROMACS for the gromacs engine
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("task", ["shoot", "free"])
def test_worker_run_shoot_free_require_gromacs(task, monkeypatch):
    worker = Worker(aimmd.Params.placeholder, ".", log_file="stdout")
    assert worker.params.engine == "gromacs"
    monkeypatch.setattr(_config, "GROMACS", None)
    with pytest.raises(EnvironmentError, match="GROMACS exec not found"):
        worker.run(task)


# --------------------------------------------------------------------------- #
# (d) engine-aware exemptions: train never needs gmx; toy sampling doesn't either
# --------------------------------------------------------------------------- #

class _DummyWorker(WorkerRun):
    """Minimal concrete WorkerRun to exercise run()'s guard/dispatch branch
    without running real MD (mirrors DummyRunWorker in test_worker_runtime_unit)."""

    def __init__(self, tmp_path, engine):
        self.directory = str(tmp_path / "run")
        self._directory = self.directory
        self.params = SimpleNamespace(parent=str(tmp_path), engine=engine)
        self.log_file = self.original_stdout = "stdout"
        self.localid = 0

    def _bind_resources(self):
        pass

    def _update_stop_condition(self, **kwargs):
        pass

    def _terminate_operations(self):
        pass

    def _reset_stop_condition(self):
        pass

    def _shoot(self, *args, **kwargs):
        return "shot"

    def _free(self, *args, **kwargs):
        return "free"

    def _train(self, *args, **kwargs):
        return "train"


def _silence_caches(monkeypatch):
    monkeypatch.setattr("aimmd.worker._run.MDA_CACHE.clear", lambda: None)
    monkeypatch.setattr("aimmd.worker._run.NPY_CACHE.clear", lambda: None)


def test_worker_train_does_not_require_gromacs(tmp_path, monkeypatch):
    """`train` runs with the gromacs engine and no gmx (training is not sampling)."""
    _silence_caches(monkeypatch)
    monkeypatch.setattr(_config, "GROMACS", None)
    worker = _DummyWorker(tmp_path, engine="gromacs")
    assert worker.run("train") == "train"


@pytest.mark.parametrize("task,expected", [("shoot", "shot"), ("free", "free")])
def test_worker_toy_engine_sampling_needs_no_gromacs(task, expected, tmp_path, monkeypatch):
    """Toy-engine shoot/free run without gmx (README: the toy engine needs no GROMACS)."""
    _silence_caches(monkeypatch)
    monkeypatch.setattr(_config, "GROMACS", None)
    worker = _DummyWorker(tmp_path, engine="toy")
    assert worker.run(task) == expected
