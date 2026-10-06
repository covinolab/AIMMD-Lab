"""The "already computed" ledger of `Path.compute` reads only the rows it needs.

A cached per-trajectory series (``<traj>.<target>.npy``) doubles as the ledger
of which frames are done: a row counts as computed when any element is
non-zero. Before computing, `Path.compute` used to load the whole file through
`NPY_CACHE` even when it asked about a handful of rows, e.g. the tail of a
growing trajectory in every ingestion cycle of a worker (794 MB read for 100
rows of a 64.5k-frame node-table file).

When the requested rows are a small part of the file and the file is not
resident in `NPY_CACHE`, the ledger now reads the `.npy` header and only those
rows, under the same file lock `update_npy` takes; rows beyond the file's
length count as missing without being read. These tests check that this gives
exactly the result of the whole-file path in every situation the ledger meets
(tail rows, zero rows mid-file, short files, missing files, a file growing
under a concurrent writer) and that it really reads only the requested rows.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import aimmd
import aimmd.path._compute as compute_module
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import load_npy, read_npy_rows, save_npy, update_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import write_trajectory

WIDTH = 6           # columns of the synthetic descriptor series
N_FRAMES = 60       # frames in the synthetic trajectory


def _trajectory(tmp_path, n_frames=N_FRAMES, stem="traj"):
    """One-atom trajectory whose frame `i` has time `i`."""
    positions = np.zeros((n_frames, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = np.linspace(-1.0, 1.0, n_frames)
    return write_trajectory(tmp_path, stem=stem, positions=positions)


def _row(i):
    """Content of row `i` in every synthetic series: non-zero everywhere."""
    return np.full(WIDTH, i + 1.0)


def _series(n_rows, zero_rows=()):
    """`n_rows` filled rows, except `zero_rows` (= "not computed yet")."""
    array = np.array([_row(i) for i in range(n_rows)])
    array[list(zero_rows)] = 0.0
    return array


class _Recorder:
    """Descriptor function that records which frames it was asked for."""

    def __init__(self):
        self.frames = []

    def __call__(self, reader):
        frames = [int(round(ts.time)) for ts in reader]
        self.frames.extend(frames)
        return np.array([_row(i) for i in frames])


def _run(fname, path_slice, initial, whole_file, monkeypatch, target="descriptors"):
    """Run `Path.compute` from a fresh copy of `initial`; report what happened.

    ``whole_file=True`` disables the partial read, i.e. reproduces the old
    ledger, which always loaded the whole file through `NPY_CACHE`.
    """
    targ_fname = get_cache_fname(fname, target)
    if os.path.exists(targ_fname):
        os.remove(targ_fname)
    if initial is not None:
        save_npy(targ_fname, initial)
    NPY_CACHE.clear()
    partial_reads = []
    real = compute_module.read_npy_rows

    def spy(*args, **kwargs):
        if whole_file:
            return None
        result = real(*args, **kwargs)
        partial_reads.append(result is not None)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(compute_module, "read_npy_rows", spy)
        recorder = _Recorder()
        path = aimmd.Path(fname)[path_slice]
        n = path.compute(recorder, target, source="reader")
    final = load_npy(targ_fname)
    NPY_CACHE.clear()
    return n, recorder.frames, final, partial_reads


CASES = {
    # the ingestion cycle of a worker: the last 2 cached rows + 8 new frames
    "tail rows": (slice(50, None), _series(52)),
    # rows that were zero-filled (e.g. by register_path) in the middle
    "zero rows mid-file": (slice(20, 30), _series(N_FRAMES, zero_rows=(22, 23, 27))),
    # the file stops inside the requested range
    "short file": (slice(5, 15), _series(10)),
    # the requested range lies entirely beyond the end of the file
    "beyond the end": (slice(40, 50), _series(30)),
    # a single frame (zero-weight shooting point)
    "single frame": (slice(33, 34), _series(N_FRAMES, zero_rows=(33,))),
    # most of the file: falls back to the whole-file path
    "most of the file": (slice(1, -1), _series(N_FRAMES, zero_rows=(4, 41))),
    # no series yet
    "missing file": (slice(10, 20), None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_partial_ledger_matches_whole_file_ledger(tmp_path, monkeypatch, case):
    """Same frames computed, same return value, same file afterwards."""
    fname = _trajectory(tmp_path)
    path_slice, initial = CASES[case]

    fast = _run(fname, path_slice, initial, False, monkeypatch)
    whole = _run(fname, path_slice, initial, True, monkeypatch)

    assert fast[0] == whole[0]
    assert fast[1] == whole[1]
    if whole[2] is None:
        assert fast[2] is None
    else:
        np.testing.assert_array_equal(fast[2], whole[2])

    # and both are right: exactly the frames whose rows were missing
    locs = np.arange(N_FRAMES)[path_slice]
    n_rows = 0 if initial is None else len(initial)
    expected = [int(i) for i in locs
                if i >= n_rows or not initial[i].any()]
    assert fast[1] == expected

    # the partial read is used whenever the request is a small part of the file
    in_file = np.count_nonzero(locs < n_rows)
    small = initial is not None and (
        in_file <= compute_module.LEDGER_PARTIAL_READ_FRACTION * n_rows)
    assert fast[3] == [small]


@pytest.mark.parametrize("target, dtype, empty", [
    ("states", "<U1", ""),
    ("values", float, 0.0),
])
def test_partial_ledger_one_dimensional_series(tmp_path, monkeypatch, target, dtype, empty):
    """States ('' = missing) and values (0 = missing) follow the same rules."""
    fname = _trajectory(tmp_path)
    initial = np.array(["A"] * 40, dtype=dtype) if dtype == "<U1" else np.arange(1.0, 41.0)
    initial[[31, 35]] = empty

    def function(reader):
        frames = [int(round(ts.time)) for ts in reader]
        function.frames.extend(frames)
        if dtype == "<U1":
            return np.array(["B"] * len(frames), dtype=dtype)
        return np.array(frames, dtype=float) + 100.0

    results = []
    for whole_file in (False, True):
        targ_fname = get_cache_fname(fname, target)
        save_npy(targ_fname, initial)
        NPY_CACHE.clear()
        function.frames = []
        with monkeypatch.context() as patch:
            if whole_file:
                patch.setattr(compute_module, "read_npy_rows", lambda *a, **k: None)
            n = aimmd.Path(fname)[30:45].compute(function, target, source="reader")
        results.append((n, list(function.frames), load_npy(targ_fname)))
        NPY_CACHE.clear()

    (n_fast, frames_fast, final_fast), (n_whole, frames_whole, final_whole) = results
    assert n_fast == n_whole
    assert frames_fast == frames_whole == [31, 35, 40, 41, 42, 43, 44]
    np.testing.assert_array_equal(final_fast, final_whole)


def test_partial_ledger_reads_only_the_requested_rows(tmp_path, monkeypatch):
    """Two rows of a 12 MB series: two rows read, no np.load of the file."""
    fname = _trajectory(tmp_path, n_frames=10)
    targ_fname = get_cache_fname(fname, "descriptors")
    n_rows, width = 1000, 1500
    series = np.ones((n_rows, width), dtype=np.float64)
    save_npy(targ_fname, series)
    NPY_CACHE.clear()
    path = aimmd.Path(fname)[3:5]           # rows 3 and 4, both computed already

    preads = []
    real_pread = os.pread

    def counting_pread(fd, n, offset):
        data = real_pread(fd, n, offset)
        preads.append(len(data))
        return data

    def no_load(*args, **kwargs):
        raise AssertionError("the ledger loaded the whole file")

    monkeypatch.setattr(os, "pread", counting_pread)
    monkeypatch.setattr(np, "load", no_load)
    rchar0 = _rchar()
    n = path.compute(lambda reader: 1 / 0, "descriptors", source="reader")
    rchar1 = _rchar()

    assert n == 0
    assert sum(preads) == 2 * width * 8
    assert targ_fname not in NPY_CACHE._cache
    if rchar0 is not None:
        # everything this process read meanwhile, header and lock included
        assert rchar1 - rchar0 < 2 * width * 8 + 64 * 1024 < series.nbytes


def test_resident_series_is_used_without_reading_the_file(tmp_path, monkeypatch):
    """A long enough copy in NPY_CACHE is used as before (no disk access)."""
    fname = _trajectory(tmp_path)
    targ_fname = get_cache_fname(fname, "descriptors")
    save_npy(targ_fname, _series(N_FRAMES, zero_rows=(12,)))
    NPY_CACHE.clear()
    NPY_CACHE.get(targ_fname)
    monkeypatch.setattr(compute_module, "read_npy_rows",
                        lambda *a, **k: pytest.fail("read the file"))
    recorder = _Recorder()
    aimmd.Path(fname)[10:14].compute(recorder, "descriptors", source="reader")
    assert recorder.frames == [12]
    NPY_CACHE.clear()


def test_read_npy_rows_contract(tmp_path):
    """Rows in the order asked for; None whenever np.load would be needed."""
    fname = str(tmp_path / "series.npy")
    array = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    np.save(fname, array)

    length, rows = read_npy_rows(fname, np.array([7, 3, 39, 40, 3, 100]))
    assert length == 40
    np.testing.assert_array_equal(rows, array[[7, 3, 39, 3]])
    length, rows = read_npy_rows(fname, np.array([45, 46]))
    assert length == 40 and rows.shape == (0, 3)

    # more than max_fraction of the rows: the caller should load the file
    assert read_npy_rows(fname, np.arange(11), max_fraction=0.25) is None
    assert read_npy_rows(fname, np.arange(10), max_fraction=0.25) is not None

    # missing file
    assert read_npy_rows(str(tmp_path / "missing.npy"), [0]) is None
    # Fortran order
    fortran = str(tmp_path / "fortran.npy")
    np.save(fortran, np.asfortranarray(array))
    assert read_npy_rows(fortran, [0]) is None
    # version 2.0 header
    v2 = str(tmp_path / "v2.npy")
    with open(v2, "wb") as file:
        np.lib.format.write_array(file, array, version=(2, 0))
    assert read_npy_rows(v2, [0]) is None
    # object dtype
    objects = str(tmp_path / "objects.npy")
    np.save(objects, np.array([None, 1], dtype=object), allow_pickle=True)
    assert read_npy_rows(objects, [0]) is None
    # data shorter than the header says
    truncated = str(tmp_path / "truncated.npy")
    np.save(truncated, array)
    with open(truncated, "r+b") as file:
        file.truncate(os.path.getsize(truncated) - 5)
    assert read_npy_rows(truncated, [0]) is None


def _rchar():
    """Bytes read by this process so far (Linux), or None."""
    try:
        with open("/proc/self/io") as file:
            for line in file:
                if line.startswith("rchar"):
                    return int(line.split()[1])
    except OSError:
        return None


_APPENDER = """
import sys, time
import numpy as np
from aimmd.cache.npy import update_npy
fname, start, stop, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for i in range(start, stop):
    update_npy(fname, np.full((1, width), i + 1.0), np.array([i]))
    time.sleep(0.002)
"""


def _start_appender(fname, start, stop):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(aimmd.__file__)),
         env.get("PYTHONPATH", "")])
    process = subprocess.Popen(
        [sys.executable, "-c", _APPENDER, fname, str(start), str(stop), str(WIDTH)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # wait until the writer is running (its import takes a few seconds)
    deadline = time.time() + 120
    while (read_npy_rows(fname, [])[0] == start and process.poll() is None
           and time.time() < deadline):
        time.sleep(0.01)
    return process


def _check_snapshot(length, rows, requested, previous):
    """A consistent snapshot: every row below `length` is complete."""
    assert length >= previous
    np.testing.assert_array_equal(
        rows, np.array([_row(i) for i in requested if i < length]).reshape(-1, WIDTH))
    return length


def test_ledger_under_concurrent_appends(tmp_path):
    """A second process appends rows with update_npy while the ledger reads.

    Every read under the lock is a consistent snapshot (no torn rows, a length
    that never shrinks), for the partial read and the whole-file path alike.
    """
    fname = str(tmp_path / "growing.xtc.descriptors.npy")
    start, stop = 50, 450
    save_npy(fname, _series(start))
    process = _start_appender(fname, start, stop)
    lengths = set()
    previous = 0
    try:
        while process.poll() is None:
            length = read_npy_rows(fname, [])[0]
            requested = np.arange(max(length - 5, 0), length + 5)
            # partial read (the tail of a growing file)
            length, rows = read_npy_rows(fname, requested, max_fraction=0.25)
            previous = _check_snapshot(length, rows, requested, previous)
            lengths.add(length)
            # the ledger of Path.compute, partial and whole-file
            NPY_CACHE.clear()
            length, rows = compute_module._ledger_rows(fname, requested)
            previous = _check_snapshot(length, rows, requested, previous)
            NPY_CACHE.clear()
            whole = load_npy(fname)
            requested = np.arange(len(whole))
            previous = _check_snapshot(len(whole), whole, requested, previous)
    finally:
        process.wait(timeout=120)
    assert process.returncode == 0, process.stderr.read().decode()
    assert len(lengths) >= 3, "the reader never overlapped with the writer"
    np.testing.assert_array_equal(load_npy(fname), _series(stop))


def test_compute_races_a_concurrent_writer_of_the_same_rows(tmp_path):
    """Ingestion of the tail while another process fills the same rows.

    Both write identical rows, so whatever the interleaving, the file ends up
    complete and correct, and Path.compute never fails.
    """
    fname = _trajectory(tmp_path, n_frames=300)
    targ_fname = get_cache_fname(fname, "descriptors")
    save_npy(targ_fname, _series(40))
    NPY_CACHE.clear()
    process = _start_appender(targ_fname, 40, 300)
    try:
        for begin in range(40, 300, 20):
            aimmd.Path(fname)[begin:begin + 20].compute(
                _Recorder(), "descriptors", source="reader")
    finally:
        process.wait(timeout=120)
    assert process.returncode == 0, process.stderr.read().decode()
    NPY_CACHE.clear()
    np.testing.assert_array_equal(load_npy(targ_fname), _series(300))
