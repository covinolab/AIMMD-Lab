import numpy as np

from aimmd.cache.npy import NpyReaderCache, load_npy, save_npy, update_npy


def test_save_load_and_update_npy(tmp_path):
    """Cover the basic file-level contract of the lightweight NumPy cache."""

    fname = tmp_path / "values.npy"
    save_npy(str(fname), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(load_npy(str(fname)), np.array([1.0, 2.0, 3.0]))

    # `update_npy` performs sparse in-place updates by index, so only the chosen
    # entries should change while untouched positions retain their old values.
    update_npy(str(fname), np.array([9.0, 8.0]), np.array([0, 2]))
    np.testing.assert_allclose(load_npy(str(fname)), np.array([9.0, 2.0, 8.0]))


def test_npy_reader_cache_get_pop_and_remove(tmp_path):
    """Show how the in-memory reader cache mirrors the on-disk `.npy` files."""

    fname = tmp_path / "cached.npy"
    save_npy(str(fname), np.array([1, 2, 3]))

    cache = NpyReaderCache()
    arr = cache.get(str(fname))
    np.testing.assert_array_equal(arr, np.array([1, 2, 3]))
    assert str(fname) in cache._cache

    popped = cache.pop(str(fname))
    np.testing.assert_array_equal(popped, np.array([1, 2, 3]))
    assert str(fname) not in cache._cache

    # Re-loading repopulates the cache, and `remove` only clears the in-memory
    # entry rather than deleting the underlying file.
    cache.get(str(fname))
    cache.remove(str(fname))
    assert str(fname) not in cache._cache
