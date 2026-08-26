"""Why Path._extract cannot replace its force-reload with a stat guard.

`_extract` routes values/bias/new/kcv through `NPY_CACHE.load` -- which never
short-circuits -- with the comment "force reload because values can change".
Replacing that with a cached `get` gated on `(st_dev, st_ino, st_size,
st_mtime_ns)` would remove the dominant cost of the post-training phase, since
every per-path reader currently pays a fresh FileLock + np.load.

It is not safe, and the reason is measurable rather than theoretical:
`update_npy` rewrites rows *in place* (`temp = fname`, then `r+b` + seek +
write + fsync), changing content while leaving size and inode untouched -- so
mtime is the only remaining signal. But Linux updates file timestamps on a
kernel-tick granularity of milliseconds, even though `st_mtime_ns` is a
nanosecond-width field. A rewrite landing in the same tick as a read is
therefore invisible, and the reader would serve a stale array into training
with no error anywhere.

These tests pin that fact. They use `os.utime` to reproduce the collision
deterministically instead of racing the clock, which would be flaky.
"""
import os

import numpy as np

from aimmd.cache.npy import NpyReaderCache, save_npy, update_npy


def _stat_key(f):
    st = os.stat(f)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def test_update_npy_rewrites_in_place_without_changing_size_or_inode(tmp_path):
    """Size and inode are useless here: only mtime could ever move."""
    f = str(tmp_path / 'v.npy')
    save_npy(f, np.zeros(10))
    before = os.stat(f)

    update_npy(f, np.array([7.0]), np.array([5]))
    after = os.stat(f)

    assert np.load(f)[5] == 7.0, 'content must have changed'
    assert after.st_size == before.st_size, 'in-place: size is unchanged'
    assert after.st_ino == before.st_ino, 'in-place: inode is unchanged'


def test_stat_tuple_does_not_identify_content(tmp_path):
    """The whole (dev, ino, size, mtime_ns) tuple can repeat across a rewrite.

    Reproduced deterministically with os.utime; in production the same state
    arises whenever a rewrite lands in the same millisecond-scale kernel
    timestamp tick as the preceding read.
    """
    f = str(tmp_path / 'v.npy')
    save_npy(f, np.zeros(10))
    key_before = _stat_key(f)
    stamp = os.stat(f).st_mtime_ns

    update_npy(f, np.array([7.0]), np.array([5]))
    os.utime(f, ns=(stamp, stamp))          # collapse into one tick

    assert _stat_key(f) == key_before, 'stat identity repeats'
    assert np.load(f)[5] == 7.0, 'but the content is different'


def test_a_stat_gated_cache_would_serve_stale_values(tmp_path):
    """Demonstrates the failure a stat guard would introduce, on the real class.

    Kept as an explicit demonstration rather than an xfail so that anyone
    implementing the guard sees exactly which invariant it must not break: a
    correct implementation needs a signal that is not derived from the mtime
    clock -- for instance a version counter written into the npy header, or a
    monotonic sidecar bumped under the same FileLock as the write.
    """
    f = str(tmp_path / 'v.npy')
    save_npy(f, np.zeros(10))

    cache = NpyReaderCache()
    cache.max_size = 10 ** 9
    stamp = os.stat(f).st_mtime_ns
    first = cache.get(f)
    assert first[5] == 0.0
    key_at_load = _stat_key(f)

    update_npy(f, np.array([7.0]), np.array([5]))
    os.utime(f, ns=(stamp, stamp))

    assert _stat_key(f) == key_at_load, (
        'a stat guard would see no change here and serve the cached array, '
        'which no longer matches the file'
    )
    # The current code is safe precisely because _extract does not consult the
    # cache for these attributes at all -- it reloads unconditionally.
    assert np.load(f)[5] == 7.0
