"""A backward half that reaches ``max_length`` without committing must still produce a path.

Regression test for the kcmpd09 trial3 crash on JUPITER: a chain's backward half ran to max_length
in R ("back completed after 50014 frames in R"), the loop skipped the forward half, and assembling the
path raised ``UnboundLocalError: nframes_forw`` (aimmd/worker/_shoot.py). Within a long-running worker
a stale ``nframes_forw`` from the previous shot hid this; it fails whenever the first shot of a process
is such a shot -- i.e. on every restart while that shot is still on disk.
"""
import os

import numpy as np

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.path.utils import write_sweep_frame
from aimmd.worker.utils import write_sweep_marker
from tests._helpers_unit import build_path, write_trajectory
from tests.test_sweep_validation import _TinySweepWorker, _sweep_params


def _all_R(tmp_path, n_frames, stem):
    pos = np.zeros((n_frames, 1, 3), dtype=np.float32)
    pos[:, 0, 0] = np.linspace(0.0, 0.3, n_frames)
    return build_path(tmp_path, stem=stem, positions=pos, shooting_index=0)


def test_backward_half_at_max_length_on_resume_registers_a_backward_only_path(tmp_path, monkeypatch):
    initial = _all_R(tmp_path, n_frames=12, stem="seed")
    params = _sweep_params()                      # max_length = 10
    params.__dict__["check_if_initialized"] = lambda *deffnms: True   # restart with a shot in flight
    chain = PathEnsemble()
    params.__dict__["shot_paths"] = lambda directory, prefix, t, k=None: chain
    worker = _TinySweepWorker(params, aimmd.PathEnsemble(initial), tmp_path)
    folder = os.path.join(str(tmp_path), "sweepR0")
    os.makedirs(folder, exist_ok=True)
    write_sweep_marker(folder, 2)

    calls = []

    def fake_simulate(deffnm, path, t, mode, offset=0):
        calls.append(os.path.basename(deffnm))
        # backward half: reached max_length (10 frames) without committing to a state
        return (9, 10, "R", 1)

    monkeypatch.setattr(worker, "_simulate", fake_simulate, raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    seg = initial.copy()
    monkeypatch.setattr("aimmd.worker._shoot.Path",
                        lambda *a, **k: seg.copy() if not a else aimmd.Path(*a, **k))

    registered = []

    def fake_register(path, chain_, eneconv, **kwargs):
        registered.append(path.n_frames)
        fname = write_trajectory(folder, stem=f"path{len(registered):06d}")
        write_sweep_frame(fname, 2)
        path._fnames = [fname]; path._first = [0]; path._last = [0]
        chain_.append(path)
        worker.must_stop = True             # stop after this single resumed shot

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)

    worker._shoot(target_state="R", k=0, sweep=True, sweep_target=float("inf"))

    assert calls == ["back"]            # the forward half is skipped, as intended
    assert registered == [10]           # the max_length backward half alone, no forward frames
