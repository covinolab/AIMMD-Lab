"""Runtime mixin tests for worker task implementations.

The worker modules account for a large fraction of uncovered lines because they
mostly orchestrate filesystem state, process control, and repeated simulation
loops. These tests replace the expensive boundaries with tiny fakes so the
branch logic itself becomes testable.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker._free import WorkerFree
from aimmd.worker._run import WorkerRun
from aimmd.worker._shoot import WorkerShoot
from aimmd.worker._simulate import WorkerSimulate
from aimmd.worker._train import WorkerTrain
from tests._helpers_unit import TinyNetwork, build_path


class DummyRunWorker(WorkerRun):
    """Small concrete wrapper to exercise `WorkerRun.run` bookkeeping."""

    def __init__(self, tmp_path):
        self.directory = str(tmp_path / "run")
        self._directory = self.directory
        self.params = SimpleNamespace(parent=str(tmp_path))
        self.log_file = self.original_stdout = "stdout"
        self.localid = 0
        self.events = []

    def _bind_resources(self):
        self.events.append("bind")

    def _update_stop_condition(self, **kwargs):
        self.events.append(("stop", kwargs))

    def _terminate_operations(self):
        self.events.append("terminate")

    def _reset_stop_condition(self):
        self.events.append("reset")

    def _shoot(self, *args, **kwargs):
        self.events.append(("shoot", args, kwargs))
        return "shot"

    def _free(self, *args, **kwargs):
        self.events.append(("free", args, kwargs))
        return "free"

    def _train(self, *args, **kwargs):
        self.events.append(("train", args, kwargs))
        return "train"


class FakeTrajectory:
    """Configurable trajectory stub used by `_simulate`.

    Each call to `check_stop` and `extend` consumes one pre-baked result from a
    queue so the tests can steer `_simulate` through specific branches.
    """

    def __init__(self, check_results, extend_results, lengths=None):
        self._check_results = list(check_results)
        self._extend_results = list(extend_results)
        self.lengths = lengths or [0]
        self._last_check = [None, 0, "", 0]

    def __len__(self):
        return 0

    def check_stop(self, **kwargs):
        if self._check_results:
            self._last_check = list(self._check_results.pop(0))
        return list(self._last_check)

    def extend(self, pattern, batch_size, remove_overlapping_frames=True, pipeline=None):
        return self._extend_results.pop(0)


class DummySimWorker(WorkerSimulate):
    """Concrete host for `_simulate` with controllable params/stop flags."""

    def __init__(self, params):
        self.params = params
        self.must_stop = False
        self.termination_timeout = 1.0


class TinyFreeWorker(WorkerFree):
    """Concrete host for `_free` with just enough state for the control loop."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ""


class TinyShootWorker(WorkerShoot):
    """Concrete host for `_shoot` that records calls instead of simulating."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ""
        self.log_file = sys.stdout
        self.original_stdout = sys.stdout


class TinyTrainWorker(WorkerTrain):
    """Concrete host for `_train` using monkeypatched collaborators."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self._directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0


def test_worker_run_dispatches_and_cleans_up(monkeypatch, tmp_path):
    """`run` should bind resources, dispatch, and always perform cleanup."""

    worker = DummyRunWorker(tmp_path)
    monkeypatch.setattr("aimmd.worker._run.MDA_CACHE.clear", lambda: worker.events.append("mda"))
    monkeypatch.setattr("aimmd.worker._run.NPY_CACHE.clear", lambda: worker.events.append("npy"))

    result = worker.run("shoot", 1, nsteps=3)
    assert result == "shot"
    assert "bind" in worker.events
    assert "mda" in worker.events and "npy" in worker.events
    assert "terminate" in worker.events and "reset" in worker.events


def test_worker_run_rejects_unknown_task(tmp_path):
    """Unknown task names should raise after the standard setup/cleanup logic."""

    worker = DummyRunWorker(tmp_path)
    with pytest.raises(TypeError):
        worker.run("unknown")


def test_simulate_returns_without_running_when_frames_are_still_buffered(tmp_path):
    """If `extend` reports unread frames left, `_simulate` should pause early.

    That branch exists so the worker can keep up with a still-growing trajectory
    file instead of immediately re-entering the engine call.
    """

    params = SimpleNamespace(
        engine="toy",
        trajectory_extension=".xtc",
        max_length=10,
        trajectory_update_batch_size=2,
        pipeline=["states", "descriptors", "values"],
        states="ARB",
        run_simulation=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    worker = DummySimWorker(params)
    deffnm = str(tmp_path / "traj")
    Path(f"{deffnm}.part0000.xtc").write_bytes(b"x")
    trajectory = FakeTrajectory(
        # Report a batch-sized frame count so `_simulate` takes the "enough
        # frames have accumulated" branch, then notices `frames_left=True`.
        check_results=[(None, 2, "", 0)],
        extend_results=[(1, True)],
    )

    result = worker._simulate(deffnm, trajectory, "A", mode="free")
    assert result[0] is None


def test_simulate_runs_engine_once_when_inputs_are_ready(tmp_path):
    """When the input files exist, `_simulate` should call `run_simulation`."""

    calls = []
    params = SimpleNamespace(
        engine="toy",
        trajectory_extension=".xtc",
        max_length=10,
        trajectory_update_batch_size=2,
        pipeline=["states", "descriptors", "values"],
        states="ARB",
        run_simulation=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    worker = DummySimWorker(params)
    deffnm = str(tmp_path / "traj")
    Path(f"{deffnm}.xtc").write_bytes(b"x")
    trajectory = FakeTrajectory(
        check_results=[(None, 0, "", 0), (None, 0, "", 0)],
        extend_results=[(0, False)],
    )

    worker._simulate(deffnm, trajectory, "A", mode="shoot")
    assert calls
    assert calls[0][0][0] == deffnm


def test_free_initializes_and_then_stops_after_one_completed_segment(monkeypatch, tmp_path):
    """`_free` should seed a new trajectory and advance to the next name.

    The internal loop is driven with a two-step `_simulate` stub:
    - first call: nothing has happened yet, so initialization is needed;
    - second call: a completed segment is reported, so the worker increments
      counters and prepares the next trajectory basename.
    """

    initial = build_path(
        tmp_path,
        stem="free_initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    params = aimmd.Params.placeholder.copy()
    
    params.__dict__.update(
        states="ARB",
        nbins=1,
        # 2. Add the initial paths here
        initial_paths=aimmd.PathEnsemble(initial), 
        trajectory_extension=".xtc",
        trajectory_update_batch_size=2,
        pipeline=["states", "descriptors", "values"],
        topology='run.gro',
        extra_free_frames=0,
        restart_free_simulations_with_transitions="",
        check_if_initialized=lambda deffnm: False,
        shot_chains=lambda directory, r, old=None: [],
        initialize_simulation=lambda frames, deffnm: init_calls.append((frames, deffnm)),
        parent=Path('.').resolve(),
    )
    params.save()
    
    init_calls = []
    worker = TinyFreeWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    # `_free` later slices the mutable trajectory object to pick restart frames.
    # Seed empty `Path()` constructions with a tiny real path so that branch has
    # concrete states/frames to work with after our synthetic `_simulate` says
    # the segment completed.
    monkeypatch.setattr(
        "aimmd.worker._free.Path",
        lambda *args, **kwargs: initial.copy() if not args else aimmd.Path(*args, **kwargs),
    )
    simulate_results = iter([(None, 0, "", 0), (0, 2, "A", 2), (None, 0, "", 0)])

    def fake_simulate(*args, **kwargs):
        result = next(simulate_results)
        if worker.total_steps >= 1:
            # Let `_free` finish the first completed trajectory, then stop on
            # the next loop iteration before another initialization cycle.
            worker.must_stop = True
        return result

    monkeypatch.setattr(worker, "_simulate", fake_simulate, raising=False)
    monkeypatch.setattr("aimmd.worker._free.remove", lambda *args, **kwargs: None)

    worker._free(target_state="A", k=0, total=1, wait=False)
    assert init_calls
    assert worker.total_steps == 1


def test_shoot_registers_completed_path_and_updates_non_tps_weight(monkeypatch, tmp_path):
    """`_shoot` should assemble a completed path and register it into the chain."""

    initial = build_path(
        tmp_path,
        stem="shoot_initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    chain = PathEnsemble()
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states="ARB",
        sorted_states="ARB",
        chain_type="rfps",
        always_select_inside_the_bins=True,
        nbins=1,
        max_length=10,
        free_overriding_states="",
        topology='run.gro',
        engine="toy",
        initial_paths = PathEnsemble(initial),
        check_if_initialized=lambda *deffnms: False,
        shot_chains=lambda directory, t, k=None: chain,
        shot_paths=lambda directory, prefix, t, k=None: chain,
        free_trajectories=lambda directory: [],
        initialize_simulation=lambda shooting_point, *deffnms: None,
        compute_values_args=(lambda x: np.array([0.0]), "values", "positions"),
        parent=Path('.').resolve(),
    )
    params.save()
    worker = TinyShootWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    monkeypatch.setattr("aimmd.worker._shoot.select_shooting_point", lambda *args, **kwargs: initial[1:2])
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *args, **kwargs: None)
    back = build_path(tmp_path, stem="back_seg")
    forw = build_path(tmp_path, stem="forw_seg")
    simulate_results = iter([(0, len(back), "A", len(back)), (0, len(forw), "B", len(forw))])
    monkeypatch.setattr(worker, "_simulate", lambda *args, **kwargs: next(simulate_results), raising=False)

    def stop_after_one_register(path, chain_, eneconv, **kwargs):
        chain_.append(path)
        worker.must_stop = True

    monkeypatch.setattr("aimmd.worker._shoot.register_path", stop_after_one_register)
    # Use prebuilt `back`/`forw` paths by swapping them in after each reset.
    paths = iter([back, forw, aimmd.Path(), aimmd.Path()])
    monkeypatch.setattr("aimmd.worker._shoot.Path", lambda *args, **kwargs: next(paths) if not args else aimmd.Path(*args, **kwargs))

    worker._shoot(target_state="R", k=0, sweep=False)
    assert len(chain) == 1
    assert worker.total_steps == 1


def test_train_performs_one_round_and_saves_outputs(monkeypatch, tmp_path):
    """`_train` should fit once, update bins/densities, and persist artifacts."""

    initial = build_path(
        tmp_path,
        stem="train_initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    class FakeEnsemble:
        """Pathensemble-like object with the exact API `_train` needs."""

        def __init__(self):
            self.weights = np.array([1.0])
            self.fnames = np.array([initial.fname])
            self.n_frames = np.array([len(initial)])

        def __len__(self):
            return 1

        def __add__(self, other):
            return self

        def compute(self, *args, **kwargs):
            return 1

        def reweight(self, *args, **kwargs):
            return (
                np.array([1.0]),
                None,
                None,
                None,
                np.array([-1.0, 1.0]),
                np.array([0.2, 0.8]),
            )
        
        def shooting(self, attribute="values"):
            return np.array([0.])
        
        def internal(self, attribute):
            return np.array([0])

        def project(self, bins, source="values"):
            return np.array([2.0, 1.0], dtype=float)

        def types(self, pattern=None):
            return np.array([True])

    ensemble = FakeEnsemble()
    save_calls = []
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        sorted_states="ARB",
        chain_type="rfps",
        fit=lambda params, pathensemble, **kwargs: ([1.0], [1.0], np.array([0.0]), np.array([1.0]), np.array([[1.0, 0.0]])),
        nbins=2,
        cutoff_min=0.5,
        cutoff_max=5.0,
        marginal_bins="all",
        network_batch_size=4,
        rescale_committor=False,
        reweight_parameters={},
        trajectory_extension=".xtc",
        compute_values_args=(lambda x: np.array([0.0]), "values", "positions"),
        network_save_interval=1,
        update_network=lambda directory, timeout=0, raise_if_failure=False: None,
        shot_chains=lambda directory, target_state=None, old=None: [ensemble],
        free_trajectories=lambda directory: [],
        network=TinyNetwork(),
    )
    worker = TinyTrainWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    monkeypatch.setattr("aimmd.worker._train.assemble_pathensemble", lambda *args, **kwargs: ensemble)
    monkeypatch.setattr("aimmd.worker._train.compute_bins", lambda *args, **kwargs: np.array([-np.inf, 0.0, np.inf]))
    monkeypatch.setattr("aimmd.worker._train.save_npy", lambda fname, arr: save_calls.append((fname, np.asarray(arr).shape)))
    monkeypatch.setattr("aimmd.worker._train.replace_in_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr("aimmd.worker._train.torch.save", lambda state, fname: save_calls.append((fname, "torch")))
    monkeypatch.setattr("aimmd.worker._train.shutil.copyfile", lambda src, dst: save_calls.append((src, dst)))
    worker._train(nrounds=1, keep_running=False)
    assert any("networkARB.h5" in str(item[0]) for item in save_calls)
    assert any("binsARB.npy" in str(item[0]) for item in save_calls)
