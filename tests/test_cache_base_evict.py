"""The miss path of AbstractCache must not do I/O for a value it discards.

`load()` opens the file, then calls `remove()` -> `pop()`, whose miss branch
opens it *again* purely so the result can be closed and dropped. Three
independent audits reached this from different directions; measured at 2 `_open`
calls for a first `get` on an uncached key, and at 0.374 s versus 0.0005 s for
`remove(fname)` on a 200 MB non-resident npy file.

The fix must live in `load()` and `remove(fname)` and NOT in `pop()`:
aimmd/worker/utils.py:577,581,594-595,599,602 index pop's return value directly
(`NPY_CACHE.pop(back_fname_states)[path.locs]`), so a pop that stopped opening on
a miss would raise TypeError -- precisely in the case that matters, since
NPY_CACHE.clear() fires at every task start.
"""
import pytest

from aimmd.cache.base import AbstractCache


class CountingCache(AbstractCache):
    """Minimal concrete cache that records every hook call."""

    def __init__(self, max_size=10_000, fail=()):
        super().__init__()
        self.max_size = max_size
        self.opens, self.closes = [], []
        self.fail = set(fail)

    def _open(self, fname):
        self.opens.append(fname)
        if fname in self.fail:
            raise OSError(f'cannot open {fname}')
        return [fname]          # a 1-element list stands in for the payload

    def _close(self, instance):
        self.closes.append(instance)

    def _size(self, instance):
        return 1


def test_first_get_opens_once():
    """The whole point: one open per miss, not two."""
    c = CountingCache()
    assert c.get('a') == ['a']
    assert c.opens == ['a'], f'expected a single open, got {c.opens}'
    assert c.closes == [], 'nothing was evicted, so nothing may be closed'
    assert len(c) == 1 and c.total_size == 1


def test_cache_hit_opens_nothing():
    c = CountingCache()
    c.get('a')
    c.opens.clear()
    assert c.get('a') == ['a']
    assert c.opens == []


def test_remove_on_a_miss_opens_nothing():
    """`remove(fname)` is used as a bare invalidation primitive.

    Call sites: worker/_train.py:164 (in a loop over every trajectory file,
    right after save_npy, so a guaranteed miss), :1346, :1544, and
    path/_compute.py:276. None can use the return value -- `remove` returns None
    unconditionally -- so opening on a miss is pure waste.
    """
    c = CountingCache()
    c.remove('never-seen')
    assert c.opens == [], f'expected no open, got {c.opens}'
    assert c.closes == []


def test_open_failure_still_yields_none_and_caches_nothing():
    """Regression guard for the None-guard concern.

    `load()` already guards at its own lines (`instance = self.open(fname)` /
    `if instance is None: return`) *before* reaching `remove()`, so the second
    open never served this purpose -- but the invariant it protects is real and
    must keep holding.
    """
    c = CountingCache(fail={'bad'})
    assert c.get('bad') is None
    assert c.opens == ['bad'], 'exactly one attempt'
    assert c.closes == [], 'nothing to close'
    assert len(c) == 0 and c.total_size == 0
    assert 'bad' not in c._cache, 'no None may be inserted'


def test_pop_still_opens_on_a_miss():
    """pop's contract is load-bearing for worker/utils.py:577+ -- do not change it."""
    c = CountingCache()
    got = c.pop('uncached')
    assert got == ['uncached'], 'pop must still return an opened instance'
    assert c.opens == ['uncached']
    assert len(c) == 0, 'and must not cache it'


def test_pop_on_a_hit_removes_and_returns():
    c = CountingCache()
    c.get('a')
    assert c.pop('a') == ['a']
    assert len(c) == 0 and c.total_size == 0


def test_fifo_eviction_still_closes():
    """The `remove()` (no argument) path must keep closing what it evicts."""
    c = CountingCache(max_size=3)
    for k in 'abcd':
        c.get(k)
    assert c.closes, 'eviction must call _close'
    assert len(c) < 4
    assert c.total_size == len(c), 'accounting must stay consistent'


def test_close_failure_does_not_escape():
    """MDAReaderCache._close raises AttributeError on *every* instance.

    `_open` returns `reader[:n]`, a FrameIteratorAll, which has no `close()`.
    An unguarded `_close` would therefore make `load()` start raising on every
    MDA reload, so the evictor must swallow it exactly as `remove`'s
    try/finally did.
    """
    class Boom(CountingCache):
        def _close(self, instance):
            raise AttributeError("'FrameIteratorAll' object has no attribute 'close'")

    c = Boom(max_size=2)
    c.get('a')
    c.get('b')
    c.get('c')          # forces an eviction, whose _close raises
    assert c.get('c') == ['c']
