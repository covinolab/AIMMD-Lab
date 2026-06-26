"""Unit tests for the brute-force committor *sweep* coordination.

These cover the round-robin/global-coverage rewrite of sweep-mode shooting:

* per-shot source-frame tagging (`write_sweep_frame` / `read_sweep_frame`),
* global per-frame coverage scanning across all sweep workers
  (`sweep_coverage`), including legacy positional fallback and in-flight markers,
* least-covered-frame selection (`least_covered_frame`),
* tag-aware aggregation (`PathEnsemble.shooting_results`),
* the launcher emitting a global sweep target + a plain ``wait`` job tail
  (no ``wait -n; scancel``) for trainerless sweep jobs.
"""

import os
from math import inf
from pathlib import Path as FsPath

import numpy as np
import pytest

import aimmd
from aimmd.launcher import Launcher
from aimmd.pathensemble import PathEnsemble
from aimmd.path.utils import read_sweep_frame, write_sweep_frame
from aimmd.worker.utils import (
    clear_sweep_marker,
    least_covered_frame,
    read_sweep_marker,
    sweep_coverage,
    sweep_marker_fname,
    write_sweep_marker,
)
from tests._helpers_unit import build_path, write_trajectory


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_shot(directory, k, n, ext=".xtc", frame=None):
    """Create ``{directory}/sweepR{k}/path{n:06d}{ext}`` and optionally tag it."""
    folder = os.path.join(directory, f"sweepR{k}")
    os.makedirs(folder, exist_ok=True)
    fname = write_trajectory(folder, stem=f"path{n:06d}", ext=ext)
    if frame is not None:
        write_sweep_frame(fname, frame)
    return fname


# ---------------------------------------------------------------------------
# frame tagging
# ---------------------------------------------------------------------------
def test_sweep_frame_roundtrip_and_missing(tmp_path):
    """A written tag reads back; an untagged shot reports ``None``."""
    fname = write_trajectory(tmp_path, stem="path000001")
    assert read_sweep_frame(fname) is None  # no tag yet
    write_sweep_frame(fname, 7)
    assert read_sweep_frame(fname) == 7
    # overwriting updates it
    write_sweep_frame(fname, 3)
    assert read_sweep_frame(fname) == 3


# ---------------------------------------------------------------------------
# least-covered-frame selection
# ---------------------------------------------------------------------------
def test_least_covered_frame_cold_start_spreads_workers():
    """With flat (e.g. all-zero) coverage, worker k takes the k-th frame."""
    flat = np.zeros(8, dtype=int)
    assert [least_covered_frame(flat, k) for k in range(4)] == [0, 1, 2, 3]
    # k wraps around the candidate set
    assert least_covered_frame(flat, 9) == 1


def test_least_covered_frame_targets_the_minimum():
    """The chosen frame is always among the least-covered ones."""
    hist = np.array([2, 0, 1, 0, 3])
    # minima are frames 1 and 3
    assert least_covered_frame(hist, 0) == 1
    assert least_covered_frame(hist, 1) == 3
    assert least_covered_frame(hist, 2) == 1  # wraps the 2 candidates


# ---------------------------------------------------------------------------
# global coverage scanning
# ---------------------------------------------------------------------------
def test_sweep_coverage_legacy_positional(tmp_path):
    """Untagged (old-code) shots are attributed positionally: i % sweep_size,
    per folder. This reproduces the exact frame the sequential sweep shot."""
    directory = str(tmp_path)
    # worker 0: 3 shots -> frames 0,1,0 ; worker 1: 2 shots -> frames 0,1
    for n in range(1, 4):
        _make_shot(directory, 0, n)
    for n in range(1, 3):
        _make_shot(directory, 1, n)
    committed, effective, total = sweep_coverage(directory, "R", ".xtc", 2)
    np.testing.assert_array_equal(committed, [3, 2])  # f0: 2+1, f1: 1+1
    np.testing.assert_array_equal(effective, committed)  # no in-flight markers
    assert total == 5


def test_sweep_coverage_tags_override_position(tmp_path):
    """Explicit frame tags are honoured regardless of file order."""
    directory = str(tmp_path)
    # positions would say frames 0,1,0 but tags pin all three to frame 1
    _make_shot(directory, 0, 1, frame=1)
    _make_shot(directory, 0, 2, frame=1)
    _make_shot(directory, 0, 3, frame=1)
    committed, _effective, total = sweep_coverage(directory, "R", ".xtc", 2)
    np.testing.assert_array_equal(committed, [0, 3])
    assert total == 3


def test_sweep_coverage_mixed_legacy_and_tagged(tmp_path):
    """A folder may mix old untagged shots (positional) with new tagged ones."""
    directory = str(tmp_path)
    _make_shot(directory, 0, 1)            # untagged, position 0 -> frame 0
    _make_shot(directory, 0, 2)            # untagged, position 1 -> frame 1
    _make_shot(directory, 0, 3, frame=1)   # tagged   -> frame 1
    _make_shot(directory, 0, 4, frame=1)   # tagged   -> frame 1
    committed, _effective, total = sweep_coverage(directory, "R", ".xtc", 2)
    np.testing.assert_array_equal(committed, [1, 3])
    assert total == 4


def test_sweep_coverage_counts_in_flight_markers(tmp_path):
    """In-flight markers add to *effective* coverage only, never *committed*."""
    directory = str(tmp_path)
    _make_shot(directory, 0, 1, frame=0)
    folder1 = os.path.join(directory, "sweepR1")
    os.makedirs(folder1, exist_ok=True)
    write_sweep_marker(folder1, 1)  # worker 1 is shooting frame 1 right now
    committed, effective, total = sweep_coverage(directory, "R", ".xtc", 2)
    np.testing.assert_array_equal(committed, [1, 0])
    np.testing.assert_array_equal(effective, [1, 1])
    assert total == 1  # marker does not count toward the global stop


def test_sweep_marker_write_read_clear(tmp_path):
    """Markers round-trip and clear cleanly."""
    folder = str(tmp_path)
    assert read_sweep_marker(folder) is None
    write_sweep_marker(folder, 4)
    assert read_sweep_marker(folder) == 4
    assert os.path.exists(sweep_marker_fname(folder))
    clear_sweep_marker(folder)
    assert read_sweep_marker(folder) is None


def test_sweep_coverage_seen_cache_is_consistent(tmp_path):
    """Passing the ``seen`` cache yields the same counts as a cold scan and
    only grows by the number of committed shots."""
    directory = str(tmp_path)
    for n in range(1, 4):
        _make_shot(directory, 0, n)
    seen = {}
    c1, _e1, t1 = sweep_coverage(directory, "R", ".xtc", 2, seen=seen)
    assert len(seen) == 3
    # a fresh shot appears; cached scan must pick it up and stay correct
    _make_shot(directory, 0, 4, frame=1)
    c2, _e2, t2 = sweep_coverage(directory, "R", ".xtc", 2, seen=seen)
    c_cold, _e, t_cold = sweep_coverage(directory, "R", ".xtc", 2)
    np.testing.assert_array_equal(c2, c_cold)
    assert t2 == t_cold == 4
    assert len(seen) == 4


# ---------------------------------------------------------------------------
# tag-aware aggregation
# ---------------------------------------------------------------------------
def test_shooting_results_is_tag_aware_with_positional_fallback(tmp_path):
    """`shooting_results` bins by frame tag when present, else by position."""
    # three identical A-R-B shots (each contributes one count to A and to B);
    # the A-R-B positions make `shooting_result` non-zero (middle frame in R).
    arb = np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32)
    paths = [build_path(tmp_path, stem=f"path{n:06d}", positions=arb)
             for n in (1, 2, 3)]
    ensemble = PathEnsemble(*paths)

    # untagged -> positional i % 2 -> frames 0,1,0
    positional = ensemble.shooting_results(states="ARB", sweep_size=2)
    np.testing.assert_array_equal(positional, [[2, 2], [1, 1]])

    # tag them out of order: frames 1,1,0 -> swaps the distribution
    write_sweep_frame(paths[0].fname, 1)
    write_sweep_frame(paths[1].fname, 1)
    write_sweep_frame(paths[2].fname, 0)
    tagged = ensemble.shooting_results(states="ARB", sweep_size=2)
    np.testing.assert_array_equal(tagged, [[1, 1], [2, 2]])


# ---------------------------------------------------------------------------
# launcher: global target + job-tail policy
# ---------------------------------------------------------------------------
def _sweep_launcher(tmp_path, monkeypatch, stem):
    params = aimmd.Params.placeholder
    params.__dict__["initial_paths"] = aimmd.PathEnsemble(build_path(tmp_path, stem=stem))
    params.__dict__["path"] = tmp_path / "params.py"
    params.__dict__["states"] = "ARB"
    (tmp_path / "params.py").write_text("# placeholder params file\n")
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_cpus", lambda: 4)
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_gpus", lambda: 0)
    monkeypatch.setattr("aimmd.params._io.ParamsIO.save", lambda self, *a, **k: str(tmp_path / "params.py"))
    return Launcher([params], [str(tmp_path / "run")])


def test_build_sweep_sets_global_target_and_disables_per_worker_caps(tmp_path, monkeypatch):
    """Sweep workers get a global target (= n * nsteps) and inf step/frame caps."""
    launcher = _sweep_launcher(tmp_path, monkeypatch, stem="sweep_build_init")
    launcher._update(n=3, n1=0, n2=0, reactive_region_mode="sweep",
                     nsteps=5, nrounds=0, cpus_per_task=1, gpus_per_task=0,
                     ntasks_per_node=3)
    args, _descriptions = launcher._build()

    shoot = [a for a in args if a[10] == "shoot"]
    assert len(shoot) == 3            # one per sweep worker, no trainer
    assert all(a[10] == "shoot" for a in shoot)
    for a in shoot:
        # conditions are (walltime, nsteps, nframes) at indices 6,7,8
        assert a[7] == inf and a[8] == inf       # per-worker caps disabled
        assert a[13] is True                     # sweep flag
        assert a[14] == 3 * 5                     # global target = n * nsteps
    # no trainer task in a sweep job
    assert not any(a[10] == "train" for a in args)


def test_create_job_sweep_waits_for_all_workers(tmp_path, monkeypatch):
    """A trainerless sweep job ends with a plain ``wait`` (no wait -n/scancel)."""
    launcher = _sweep_launcher(tmp_path, monkeypatch, stem="sweep_job_init")
    job = tmp_path / "job.sh"
    launcher.create_job(str(job), n=2, n1=0, n2=0, reactive_region_mode="sweep",
                        nsteps=4, nrounds=0, walltime=3600,
                        cpus_per_task=1, gpus_per_task=0)
    script = job.read_text()
    assert "wait -n" not in script
    assert "scancel" not in script
    assert "\nwait\n" in script
    # the global sweep target (n * nsteps = 8) is passed to the workers
    assert "shoot" in script and '"8.0"' in script


def test_create_job_with_trainer_keeps_scancel(tmp_path, monkeypatch):
    """A trainer-coordinated job keeps the wait -n; scancel teardown."""
    launcher = _sweep_launcher(tmp_path, monkeypatch, stem="trainer_job_init")
    job = tmp_path / "job.sh"
    launcher.create_job(str(job), n=1, n1=0, n2=0, reactive_region_mode="chain",
                        nrounds=1, walltime=3600, cpus_per_task=1, gpus_per_task=0)
    script = job.read_text()
    assert "wait -n" in script
    assert "scancel" in script


# ---------------------------------------------------------------------------
# _shoot sweep loop (integration with mocked engine / registration)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from aimmd.worker._shoot import WorkerShoot  # noqa: E402


class _TinySweepWorker(WorkerShoot):
    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ""


def _sweep_params():
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states="ARB",
        sorted_states="ARB",
        chain_type="rfps",
        at_least_one_transition_in_pool=False,
        nbins=1,
        max_length=10,
        selection_pool_size=2,
        free_overriding_states="",
        engine="toy",
        trajectory_extension=".xtc",
        record_bias=False,
        retry_with_state_definition_glitches=False,
        check_if_initialized=lambda *deffnms: False,
        initialize_simulation=lambda shooting_point, *deffnms: None,
    )
    return params


def _all_R_frames(tmp_path, n_frames, stem):
    """Build an initial path of `n_frames` frames, all in state R (x in (-0.5, 0.5))."""
    pos = np.zeros((n_frames, 1, 3), dtype=np.float32)
    pos[:, 0, 0] = np.linspace(0.0, 0.3, n_frames)
    return build_path(tmp_path, stem=stem, positions=pos, shooting_index=0)


def test_shoot_sweep_round_robin_tags_and_global_stop(tmp_path, monkeypatch):
    """Sweep shooting picks least-covered frames, tags each shot, and stops at
    the global committed-shot target."""
    initial = _all_R_frames(tmp_path, n_frames=4, stem="cval")
    params = _sweep_params()
    chain = PathEnsemble()
    params.__dict__["shot_paths"] = lambda directory, prefix, t, k=None: chain
    worker = _TinySweepWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    folder = os.path.join(str(tmp_path), "sweepR0")

    # one completed back + forw per shot; engine is fully mocked
    monkeypatch.setattr(worker, "_simulate",
                        lambda *a, **k: (0, 1, "B", 1), raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    seg = initial.copy()
    monkeypatch.setattr("aimmd.worker._shoot.Path",
                        lambda *a, **k: seg.copy() if not a else aimmd.Path(*a, **k))

    counter = {"n": 0}

    def fake_register(path, chain_, eneconv, **kwargs):
        counter["n"] += 1
        fname = write_trajectory(folder, stem=f"path{counter['n']:06d}")
        path._fnames = [fname]
        path._first = [0]
        path._last = [0]
        chain_.append(path)

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)

    # stop the campaign after 3 committed shots (sweep_size is 4)
    worker._shoot(target_state="R", k=0, sweep=True, sweep_target=3)

    # exactly the target number of shots was produced, then it stopped
    assert counter["n"] == 3
    shot_files = sorted(FsPath(folder).glob("path??????.xtc"))
    assert len(shot_files) == 3
    # each shot is tagged, and a single worker fills the least-covered frames
    # in order: 0, 1, 2 (frame 3 is never reached because the target is hit).
    tags = [read_sweep_frame(str(f)) for f in shot_files]
    assert tags == [0, 1, 2]
    # the in-flight marker is cleared once the worker stops
    assert read_sweep_marker(folder) is None


def test_shoot_sweep_neutralizes_inherited_per_worker_caps(tmp_path, monkeypatch):
    """Sweep ignores any per-worker nsteps/nframes inherited from the worker
    args (e.g. an old job.sh that hard-coded nsteps), so an already-"full"
    folder does not instant-stop the worker. Governed only by the global target.
    """
    initial = _all_R_frames(tmp_path, n_frames=4, stem="cval_caps")
    params = _sweep_params()
    chain = PathEnsemble()
    params.__dict__["shot_paths"] = lambda directory, prefix, t, k=None: chain
    worker = _TinySweepWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    # simulate the per-worker caps an old job.sh would pass (e.g. nsteps=2)
    worker.nsteps = 2
    worker.nframes = 2
    folder = os.path.join(str(tmp_path), "sweepR0")

    monkeypatch.setattr(worker, "_simulate",
                        lambda *a, **k: (0, 1, "B", 1), raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    seg = initial.copy()
    monkeypatch.setattr("aimmd.worker._shoot.Path",
                        lambda *a, **k: seg.copy() if not a else aimmd.Path(*a, **k))

    counter = {"n": 0}

    def fake_register(path, chain_, eneconv, **kwargs):
        counter["n"] += 1
        fname = write_trajectory(folder, stem=f"path{counter['n']:06d}")
        path._fnames = [fname]
        path._first = [0]
        path._last = [0]
        chain_.append(path)

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)

    # target of 3 > the inherited nsteps=2: if the cap were still honoured the
    # real must_stop would fire at 2; here we assert sweep got 3 shots and the
    # caps were lifted to inf.
    worker._shoot(target_state="R", k=0, sweep=True, sweep_target=3)
    assert counter["n"] == 3
    assert worker.nsteps == inf
    assert worker.nframes == inf


def test_shoot_sweep_resume_recovers_frame_from_marker(tmp_path, monkeypatch):
    """On resume mid-shot (back/forw already initialized), the worker recovers
    the in-flight frame from its marker and tags the finished shot with it."""
    initial = _all_R_frames(tmp_path, n_frames=5, stem="cval_resume")
    params = _sweep_params()
    # pretend a shot is already initialized on disk (process restarted mid-shot)
    params.__dict__["check_if_initialized"] = lambda *deffnms: True
    chain = PathEnsemble()
    params.__dict__["shot_paths"] = lambda directory, prefix, t, k=None: chain
    worker = _TinySweepWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    folder = os.path.join(str(tmp_path), "sweepR0")
    os.makedirs(folder, exist_ok=True)
    # the interrupted shot was launched from frame 3 (recorded in the marker)
    write_sweep_marker(folder, 3)

    monkeypatch.setattr(worker, "_simulate",
                        lambda *a, **k: (0, 1, "B", 1), raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    seg = initial.copy()
    monkeypatch.setattr("aimmd.worker._shoot.Path",
                        lambda *a, **k: seg.copy() if not a else aimmd.Path(*a, **k))

    registered = {}

    def fake_register(path, chain_, eneconv, **kwargs):
        fname = write_trajectory(folder, stem="path000001")
        path._fnames = [fname]
        path._first = [0]
        path._last = [0]
        chain_.append(path)
        registered["fname"] = fname
        worker.must_stop = True  # stop after this single resumed shot

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)

    worker._shoot(target_state="R", k=0, sweep=True, sweep_target=inf)

    # the resumed shot is tagged with the recovered frame (3), not a fresh pick
    assert read_sweep_frame(registered["fname"]) == 3
    assert read_sweep_marker(folder) is None
