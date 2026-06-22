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


def test_npy_reader_cache_evicts_when_max_size_exceeded(tmp_path):
    """Loading more than `max_size` bytes must trigger FIFO eviction.

    The cache budget is enforced via `sys.getsizeof(instance)`. For arrays
    returned by `np.load` the data buffer is owned by an internal mmap, so
    `sys.getsizeof` reports only the ndarray wrapper (~128 B) instead of the
    real `nbytes`. As a result `total_size` never crosses `max_size`, no
    eviction ever fires, and every loaded `.npy` stays resident — which can
    pin tens of GB on real workloads (descriptor caches per trajectory).

    This test loads three 1-MB arrays into a 2-MB cache; if the budget were
    accounted correctly, at least one of the older entries would have been
    evicted by the time the third array is in.
    """
    payload = np.zeros(250_000, dtype=np.float32)  # 1 MB per file
    fnames = []
    for i in range(3):
        f = tmp_path / f"big{i}.npy"
        save_npy(str(f), payload)
        fnames.append(str(f))

    cache = NpyReaderCache()
    cache.max_size = 2 * payload.nbytes  # budget = 2 MB; three loads = 3 MB

    for f in fnames:
        arr = cache.get(f)
        assert arr is not None

    # All three entries are still resident even though their combined buffers
    # (3 MB) exceed the 2 MB budget — the bug.
    assert len(cache) <= 2, (
        f"expected FIFO eviction once cache exceeds max_size "
        f"({cache.max_size} B), but all {len(cache)} entries are still cached "
        f"(total_size reported as {cache.total_size} B while real buffers "
        f"hold {sum(a.nbytes for a in cache._cache.values())} B)"
    )


def _save_payload(path, n_floats):
    """Write an `n_floats`-element float32 array and return its file path."""
    arr = np.arange(n_floats, dtype=np.float32)
    save_npy(str(path), arr)
    return str(path), arr.nbytes


def test_eviction_is_fifo_oldest_first(tmp_path):
    """The oldest entry must be the one evicted when the budget tightens."""
    f0, sz = _save_payload(tmp_path / "a.npy", 250_000)        # 1 MB
    f1, _ = _save_payload(tmp_path / "b.npy", 250_000)         # 1 MB
    f2, _ = _save_payload(tmp_path / "c.npy", 250_000)         # 1 MB

    cache = NpyReaderCache()
    cache.max_size = 2 * sz + sz // 2  # room for ~2 entries, not 3

    cache.get(f0); cache.get(f1); cache.get(f2)

    # f0 is the oldest, so it must have been evicted first.
    assert f0 not in cache._cache
    assert f1 in cache._cache
    assert f2 in cache._cache


def test_get_on_cached_entry_does_not_reload(tmp_path):
    """A second `get` for the same key must hit the cache without re-opening."""
    fname, _ = _save_payload(tmp_path / "x.npy", 1_000)

    cache = NpyReaderCache()
    cache.get(fname)
    size_after_first = cache.total_size
    n_after_first = len(cache)

    # Spy on _open: any call here would mean we missed the cache.
    calls = []
    real_open = cache._open
    cache._open = lambda f, _calls=calls, _real=real_open: (
        _calls.append(f) or _real(f)
    )

    cache.get(fname)

    assert calls == []
    assert cache.total_size == size_after_first
    assert len(cache) == n_after_first


def test_mixed_sizes_evict_until_room(tmp_path):
    """One large `get` must evict however many smaller entries it takes to fit."""
    small_a, small_sz = _save_payload(tmp_path / "small_a.npy", 50_000)   # 0.2 MB
    small_b, _ = _save_payload(tmp_path / "small_b.npy", 50_000)
    small_c, _ = _save_payload(tmp_path / "small_c.npy", 50_000)
    big, big_sz = _save_payload(tmp_path / "big.npy", 500_000)            # 2 MB

    cache = NpyReaderCache()
    # Budget fits `big + one small` with a little slack — the eviction check
    # is `>=`, so we need strict room beyond `big + small` to keep the newest
    # small alive.
    cache.max_size = big_sz + small_sz + 100_000

    cache.get(small_a); cache.get(small_b); cache.get(small_c)
    assert len(cache) == 3  # all three fit comfortably

    cache.get(big)

    # The big entry is in; we kept the newest small (small_c); older ones
    # were evicted in arrival order until the budget held.
    assert big in cache._cache
    assert small_c in cache._cache
    assert small_a not in cache._cache
    assert small_b not in cache._cache
    # Accounting reflects what's actually resident.
    assert cache.total_size == sum(cache._size(a) for a in cache._cache.values())
    assert cache.total_size <= cache.max_size


def test_readd_after_eviction_works_and_recounts(tmp_path):
    """Re-getting an evicted file reloads it and is reflected in total_size."""
    f0, sz = _save_payload(tmp_path / "a.npy", 250_000)
    f1, _ = _save_payload(tmp_path / "b.npy", 250_000)

    cache = NpyReaderCache()
    cache.max_size = sz + sz // 2  # only one entry fits at a time

    cache.get(f0)
    cache.get(f1)
    assert f0 not in cache._cache and f1 in cache._cache
    size_with_one = cache.total_size

    # Re-fetch the evicted file: now f1 must go.
    cache.get(f0)
    assert f0 in cache._cache and f1 not in cache._cache
    assert cache.total_size == size_with_one  # back to a single entry


def test_pop_removes_entry_and_decrements_total_size(tmp_path):
    """`pop` should both surface the array and update the budget bookkeeping."""
    f0, sz = _save_payload(tmp_path / "a.npy", 100_000)
    f1, _ = _save_payload(tmp_path / "b.npy", 100_000)

    cache = NpyReaderCache()
    cache.get(f0); cache.get(f1)
    total_before = cache.total_size
    assert total_before > 0

    popped = cache.pop(f0)
    np.testing.assert_array_equal(popped, np.arange(100_000, dtype=np.float32))
    assert f0 not in cache._cache
    assert f1 in cache._cache
    # total_size dropped by exactly one entry's accounting.
    assert cache.total_size == total_before - cache._size(popped)


def test_pop_oldest_when_called_with_no_argument(tmp_path):
    """`pop(None)` returns and removes the oldest entry (FIFO semantics)."""
    f0, _ = _save_payload(tmp_path / "a.npy", 1_000)
    f1, _ = _save_payload(tmp_path / "b.npy", 1_000)

    cache = NpyReaderCache()
    cache.get(f0); cache.get(f1)

    cache.pop()  # drop the oldest

    assert f0 not in cache._cache
    assert f1 in cache._cache


def test_oversized_single_entry_is_cached_and_does_not_loop(tmp_path):
    """A file larger than `max_size` should still be cached without spinning.

    Earlier the eviction loop was `while total_size + size >= max_size: …`,
    which ran forever when the cache was already empty and the new entry
    couldn't fit. The fix gates the loop on `self._cache and …`.
    """
    fname, sz = _save_payload(tmp_path / "huge.npy", 500_000)  # 2 MB

    cache = NpyReaderCache()
    cache.max_size = sz // 4  # smaller than a single entry on purpose

    arr = cache.get(fname)  # must return; previously this would spin

    assert arr is not None
    assert fname in cache._cache
