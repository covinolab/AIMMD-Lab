"""
Tests for the cache helpers that roll a trajectory's per-frame data back when
the trajectory loses frames (see aimmd.worker.rollback):
aimmd.cache.npy.truncate_npy cuts a cache file to its leading rows, and
aimmd.cache.forget drops what a process holds in memory about a file.
"""

import glob
import os

import numpy as np
import pytest
from filelock import FileLock

from aimmd import MDA_CACHE, NPY_CACHE
from aimmd.cache import forget
from aimmd.cache.npy import load_npy, save_npy, truncate_npy, update_npy
from tests._helpers_unit import write_trajectory


@pytest.mark.parametrize('array', [
    np.arange(1, 11, dtype=float),                         # values, bias
    np.array(list('ARRRRBBBBA'), dtype='<U1'),             # states
    np.arange(1, 41, dtype=np.float32).reshape(10, 4),     # 2-D descriptor rows
    np.arange(1, 61, dtype=np.float32).reshape(10, 3, 2),  # 3-D series
])
def test_keeps_the_leading_rows(tmp_path, array):
    fname = str(tmp_path / 'back.xtc.series.npy')
    save_npy(fname, array)
    assert truncate_npy(fname, 4) == 10
    cut = np.load(fname)
    assert cut.dtype == array.dtype and cut.shape == (4,) + array.shape[1:]
    np.testing.assert_array_equal(cut, array[:4])
    assert not os.path.exists(str(tmp_path / 'temp.back.xtc.series.npy'))


def test_leaves_a_short_file_alone(tmp_path):
    fname = str(tmp_path / 'back.xtc.states.npy')
    save_npy(fname, np.array(list('RRR'), dtype='<U1'))
    before = os.stat(fname)
    assert truncate_npy(fname, 3) == 3
    assert truncate_npy(fname, 50) == 3
    after = os.stat(fname)
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_cut_to_zero_rows_keeps_dtype_and_trailing_shape(tmp_path):
    fname = str(tmp_path / 'back.xtc.descriptors.npy')
    save_npy(fname, np.ones((5, 3), dtype=np.float32))
    assert truncate_npy(fname, 0) == 5
    assert truncate_npy(fname, -3) == 0
    cut = np.load(fname)
    assert cut.shape == (0, 3) and cut.dtype == np.float32
    # the empty cache still grows like any other
    update_npy(fname, np.full((2, 3), 7, dtype=np.float32), [0, 1])
    np.testing.assert_array_equal(load_npy(fname), np.full((2, 3), 7))


def test_missing_and_0d_files(tmp_path):
    assert truncate_npy(str(tmp_path / 'none.npy'), 1) is None
    fname = str(tmp_path / 'scalar.npy')
    np.save(fname, np.float64(3.0))
    assert truncate_npy(fname, 0) is None
    assert np.load(fname) == 3.0


def test_replaces_the_file_and_rows_grow_again_after_the_cut(tmp_path):
    fname = str(tmp_path / 'back.xtc.cv.npy')
    update_npy(fname, np.arange(1, 9, dtype=float), np.arange(8))
    inode = os.stat(fname).st_ino
    assert truncate_npy(fname, 3) == 8
    assert os.stat(fname).st_ino != inode   # atomic replace: a new file
    update_npy(fname, np.array([40.0, 50.0]), [3, 4])
    np.testing.assert_array_equal(load_npy(fname), [1, 2, 3, 40, 50])


def test_waits_for_the_lock(tmp_path):
    fname = str(tmp_path / 'back.xtc.states.npy')
    save_npy(fname, np.array(list('RRRR'), dtype='<U1'))
    with FileLock(str(tmp_path / '.back.xtc.states.npy.lock')):
        with pytest.raises(RuntimeError, match='could not cut'):
            truncate_npy(fname, 1, timeout=0.01)
    assert len(np.load(fname)) == 4


def test_forget_evicts_readers_and_arrays_of_the_file_only(tmp_path):
    a = write_trajectory(tmp_path, stem='a')
    b = write_trajectory(tmp_path, stem='b')
    for xtc in (a, b):
        save_npy(f'{xtc}.states.npy', np.array(list('RRR')))
        MDA_CACHE.get(xtc)
        NPY_CACHE.get(f'{xtc}.states.npy')
    assert a in MDA_CACHE._cache and f'{a}.states.npy' in NPY_CACHE._cache
    forget(os.path.relpath(a))       # keys are matched by absolute path
    assert a not in MDA_CACHE._cache and f'{a}.states.npy' not in NPY_CACHE._cache
    assert b in MDA_CACHE._cache and f'{b}.states.npy' in NPY_CACHE._cache
    assert not glob.glob(str(tmp_path / '.a.xtc_offsets.*'))
    forget()                         # nothing to forget
