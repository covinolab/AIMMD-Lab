"""`read_npy_rows_unlocked` reads rows known to be complete without the lock.

On a network file system the per-file lock of `read_npy_rows` costs more than
the read itself (about 8 of 24 ms for two rows on NFS). Fit draws only rows the
ledger of the round has computed, and no process rewrites a computed row, so
fit reads them without the lock and falls back to `read_npy_rows` whenever the
lock-free read cannot vouch for its rows: a header it cannot parse, a file
whose size is not what the header says (truncated, or growing right now), a
requested row beyond the length in the header, or a requested row that is all
zero (not computed). These tests check that the rows equal the locked ones,
that each of those cases gives None, and that a concurrent appender never
makes the lock-free read return a torn or zero row.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import aimmd
import aimmd.cache.npy as npy_module
from aimmd.cache.npy import read_npy_rows, read_npy_rows_unlocked, save_npy

WIDTH = 512         # float64 columns: a row spans a page boundary


def _row(i, width=WIDTH):
    """Row `i` of the synthetic series: non-zero everywhere."""
    return np.full(width, i + 1.0)


def _series(n_rows, width=WIDTH):
    return np.array([_row(i, width) for i in range(n_rows)]).reshape(
        n_rows, width)


@pytest.mark.parametrize("array", [
    np.arange(1, 40 * 3 + 1, dtype=np.float32).reshape(40, 3),
    np.arange(1, 40 * 6 + 1, dtype=np.float64).reshape(40, 2, 3),
    np.arange(1.0, 41.0),
    np.arange(1, 41, dtype=np.int16),
], ids=["float32", "3-d", "1-d", "int16"])
def test_unlocked_rows_equal_locked_rows(tmp_path, array):
    fname = str(tmp_path / "series.npy")
    save_npy(fname, array)
    requested = np.array([7, 3, 39, 3, 0, 8, 7])

    length, rows = read_npy_rows_unlocked(fname, requested)

    locked_length, locked_rows = read_npy_rows(fname, requested)
    assert length == locked_length == 40
    assert rows.dtype == locked_rows.dtype == array.dtype
    assert rows.shape == locked_rows.shape == (7,) + array.shape[1:]
    np.testing.assert_array_equal(rows, locked_rows)
    np.testing.assert_array_equal(rows, array[requested])
    # np.save's own header (not save_npy's) reads the same
    np.save(fname, array)
    np.testing.assert_array_equal(
        read_npy_rows_unlocked(fname, requested)[1], array[requested])


def test_unlocked_rows_do_not_take_the_lock(tmp_path, monkeypatch):
    fname = str(tmp_path / "series.npy")
    save_npy(fname, _series(10, 4))

    def no_lock(*args, **kwargs):
        raise AssertionError("the lock was taken")

    monkeypatch.setattr(npy_module, "FileLock", no_lock)
    length, rows = read_npy_rows_unlocked(fname, [9, 2])
    assert length == 10
    np.testing.assert_array_equal(rows, _series(10, 4)[[9, 2]])
    assert read_npy_rows(fname, [9, 2]) is None    # the locked read locks


def test_a_zero_row_gives_none(tmp_path):
    fname = str(tmp_path / "series.npy")
    array = _series(10, 4)
    array[4] = 0.0
    array[6, :3] = 0.0             # partly zero: computed
    save_npy(fname, array)

    assert read_npy_rows_unlocked(fname, [3, 4]) is None
    assert read_npy_rows_unlocked(fname, [4]) is None
    np.testing.assert_array_equal(
        read_npy_rows_unlocked(fname, [3, 6, 5])[1], array[[3, 6, 5]])
    # -0.0 is zero as well (the ledger's "not computed")
    array[4] = -0.0
    save_npy(fname, array)
    assert read_npy_rows_unlocked(fname, [4]) is None


def test_rows_beyond_the_length_give_none(tmp_path):
    fname = str(tmp_path / "series.npy")
    save_npy(fname, _series(10, 4))
    assert read_npy_rows_unlocked(fname, [9, 10]) is None
    assert read_npy_rows_unlocked(fname, [100]) is None
    assert read_npy_rows_unlocked(fname, [-1]) is None
    assert read_npy_rows_unlocked(fname, [1.0]) is None
    assert read_npy_rows_unlocked(fname, [9])[0] == 10


def test_unusual_files_give_none(tmp_path):
    array = _series(40, 3)

    def variant(name, write):
        fname = str(tmp_path / f"{name}.npy")
        write(fname)
        return fname

    assert read_npy_rows_unlocked(str(tmp_path / "missing.npy"), [0]) is None
    # data shorter than the header says
    truncated = variant("truncated", lambda f: np.save(f, array))
    with open(truncated, "r+b") as file:
        file.truncate(os.path.getsize(truncated) - 5)
    assert read_npy_rows_unlocked(truncated, [0]) is None
    # data longer than the header says (a file growing right now)
    grown = variant("grown", lambda f: np.save(f, array))
    with open(grown, "r+b") as file:
        file.truncate(os.path.getsize(grown) + 3 * 8)
    assert read_npy_rows_unlocked(grown, [0]) is None
    # a header that does not parse, or is cut short
    for k, (at, damage) in enumerate([
            (0, b"\x00" * 6),                       # magic
            (8, b"\xff\xff"),                       # length beyond the file
            (10, b"{'descr': '<f8', 'fortran_order': False, 'shapx'"),
            (10, b"garbage")]):
        bad = variant(f"bad{k}", lambda f: np.save(f, array))
        with open(bad, "r+b") as file:
            file.seek(at)
            file.write(damage)
        assert read_npy_rows_unlocked(bad, [0]) is None
    short = variant("short", lambda f: open(f, "wb").write(b"\x93NUMPY\x01"))
    assert read_npy_rows_unlocked(short, [0]) is None
    # version 2.0 header, Fortran order, object dtype, 0-d, structured
    v2 = str(tmp_path / "v2.npy")
    with open(v2, "wb") as file:
        np.lib.format.write_array(file, array, version=(2, 0))
    assert read_npy_rows_unlocked(v2, [0]) is None
    fortran = variant("fortran", lambda f: np.save(f, np.asfortranarray(array)))
    assert read_npy_rows_unlocked(fortran, [0]) is None
    objects = variant("objects", lambda f: np.save(
        f, np.array([None, 1], dtype=object), allow_pickle=True))
    assert read_npy_rows_unlocked(objects, [0]) is None
    scalar = variant("scalar", lambda f: np.save(f, np.float64(1.0)))
    assert read_npy_rows_unlocked(scalar, [0]) is None
    structured = variant("structured", lambda f: np.save(
        f, np.ones(4, dtype=[("a", "f4"), ("b", "i4")])))
    assert read_npy_rows_unlocked(structured, [0]) is None
    empty_rows = variant("empty_rows", lambda f: np.save(f, np.ones((4, 0))))
    assert read_npy_rows_unlocked(empty_rows, [0]) is None
    # the locked read still reads the files it read before
    assert read_npy_rows(grown, [0]) is not None


_APPENDER = """
import sys, time
import numpy as np
from aimmd.cache.npy import update_npy
fname, start, stop, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for i in range(start, stop):
    update_npy(fname, np.full((1, width), i + 1.0), np.array([i]))
    time.sleep(0.001)
"""


def _start_appender(fname, start, stop):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(aimmd.__file__)),
         env.get("PYTHONPATH", "")])
    process = subprocess.Popen(
        [sys.executable, "-c", _APPENDER, fname, str(start), str(stop),
         str(WIDTH)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # wait until the writer is running (its import takes a few seconds)
    deadline = time.time() + 120
    while (read_npy_rows(fname, [])[0] == start and process.poll() is None
           and time.time() < deadline):
        time.sleep(0.01)
    return process


def test_a_concurrent_appender_never_gives_torn_or_zero_rows(tmp_path):
    """A second process appends rows with update_npy while rows are read.

    Rows complete before the read (below a length read earlier) and rows at
    the growing end alike: the lock-free read gives either None or exactly
    the rows written, never a torn or zero row.
    """
    fname = str(tmp_path / "growing.xtc.descriptors.npy")
    start, stop = 20, 600
    save_npy(fname, _series(start))
    process = _start_appender(fname, start, stop)
    lengths, found, refused = set(), 0, 0
    try:
        while process.poll() is None:
            known = read_npy_rows(fname, [])[0]
            for requested in (np.arange(max(known - 6, 0), known),
                              np.arange(max(known - 3, 0), known + 3),
                              np.array([known - 1, 0, known, known + 1])):
                result = read_npy_rows_unlocked(fname, requested)
                if result is None:
                    refused += 1
                    continue
                length, rows = result
                assert requested.max() < length
                np.testing.assert_array_equal(
                    rows, np.array([_row(i) for i in requested]))
                lengths.add(length)
                found += 1
    finally:
        process.wait(timeout=120)
    assert process.returncode == 0, process.stderr.read().decode()
    assert len(lengths) >= 3, "the reader never overlapped with the writer"
    assert found and refused
    np.testing.assert_array_equal(
        read_npy_rows_unlocked(fname, np.arange(stop))[1], _series(stop))
