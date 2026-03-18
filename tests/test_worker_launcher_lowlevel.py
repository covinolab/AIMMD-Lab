from pathlib import Path

import aimmd
from aimmd.launcher import Launcher
from aimmd.worker import Worker
from tests._helpers_unit import build_path


def test_worker_stop_condition_helpers(monkeypatch):
    """The helper mixin should update and reset worker stop conditions cleanly."""

    worker = Worker(aimmd.Params.placeholder, ".", log_file="stdout")
    worker._update_stop_condition(nsteps=3, nframes=5, walltime=7)
    assert worker.nsteps == 3.0
    assert worker.nframes == 5.0
    assert worker.walltime == 7.0
    worker._terminate_handler(15)
    assert worker.termination_signal == 15
    worker._reset_stop_condition()
    assert worker.nsteps == float("inf")


def test_launcher_build_creates_expected_plan(tmp_path, monkeypatch):
    """The launcher build step should emit one worker plan for one simple run."""

    params = aimmd.Params.placeholder
    params.__dict__["initial_paths"] = aimmd.PathEnsemble(build_path(tmp_path, stem="launcher_init"))
    params.__dict__["path"] = tmp_path / "params.py"
    params.__dict__["states"] = "ARB"
    (tmp_path / "params.py").write_text("# placeholder params file\n")

    monkeypatch.setattr("aimmd.launcher._helpers.get_num_cpus", lambda: 4)
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_gpus", lambda: 0)
    monkeypatch.setattr("aimmd.params._io.ParamsIO.save", lambda self, *args, **kwargs: str(tmp_path / "params.py"))

    launcher = Launcher([params], [str(tmp_path / "run")])
    launcher._update(n=1, n1=0, n2=0, nrounds=0, cpus_per_task=1, gpus_per_task=0, ntasks_per_node=1)
    args, descriptions = launcher._build()
    # One configured sampling process should translate into exactly one worker
    # argument tuple and one human-readable description.
    assert len(args) == 1
    assert len(descriptions) == 1
    assert Path(tmp_path / "run").exists()
