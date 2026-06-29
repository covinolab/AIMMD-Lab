"""Unit tests for ``PathEnsemble.subsample`` (value-pass subsampling caps).

``subsample`` is the building block of the optional ``subsample_caps`` feature:
it returns a real, randomly down-sampled slice of a path ensemble with caps
applied *per path category* (per shot/free direction-type) and a frame budget
for in-state paths. These tests pin the selection logic deterministically using
a lightweight stand-in that exposes the three things ``subsample`` touches
(``types()``, ``n_frames``, ``paths``/``__getitem__``) and binds the *real*
``PathEnsemble.subsample`` method, so no GROMACS/toy run is needed.
"""
import collections

import numpy as np
import pytest

from aimmd.pathensemble import PathEnsemble


class _FakePE:
    """Minimal stand-in exposing exactly what ``subsample`` reads."""

    def __init__(self, types, n_frames):
        self._paths = list(types)
        self._types = np.array(types, dtype='<U4')
        self._nf = np.asarray(n_frames)

    def types(self, *patterns):
        assert not patterns
        return self._types

    @property
    def n_frames(self):
        return self._nf

    @property
    def paths(self):
        return np.array(self._paths, dtype=object)

    def __getitem__(self, i):
        return _FakePE(list(self._types[i]), self._nf[i])

    # bind the real method under test
    subsample = PathEnsemble.subsample


def _ensemble():
    # 4-char Path.type = (initial, middle, final, shooting)
    types = (['ARBR'] * 250      # shot A->B
             + ['ARAR'] * 30      # shot A->A
             + ['ARAA'] * 800     # free A->A
             + ['ARBA'] * 5       # free A->B
             + ['AAAA'] * 20      # in-A (5 frames each -> 100)
             + ['BBBB'] * 10)     # in-B (10 frames each -> 100)
    n_frames = ([7] * 250 + [7] * 30 + [9] * 800 + [9] * 5
                + [5] * 20 + [10] * 10)
    return _FakePE(types, n_frames)


def _counts(pe):
    return collections.Counter(pe.types())


def test_subsample_caps_per_direction_type():
    """Each shot/free direction-type is capped independently; in-state by frames."""
    pe = _ensemble()
    caps = {'shot': 100, 'free': 500, 'in_state': 60}
    sub = pe.subsample(caps, 'ARB', np.random.default_rng(0))
    c = _counts(sub)

    assert c['ARBR'] == 100          # shot A->B capped 250 -> 100
    assert c['ARAR'] == 30           # shot A->A below cap -> all kept
    assert c['ARAA'] == 500          # free A->A capped 800 -> 500
    assert c['ARBA'] == 5            # free A->B below cap -> all kept

    # in-state capped by FRAMES: A has 5 frames/path -> 12 paths == 60 frames;
    # B has 10 frames/path -> 6 paths == 60 frames. Always keeps >= 1 path.
    assert 1 <= c['AAAA'] and c['AAAA'] * 5 <= 60 + 5
    assert 1 <= c['BBBB'] and c['BBBB'] * 10 <= 60 + 10


def test_subsample_no_caps_is_identity():
    pe = _ensemble()
    assert pe.subsample(None, 'ARB') is pe
    assert pe.subsample({}, 'ARB') is pe


def test_subsample_missing_key_leaves_category_uncapped():
    pe = _ensemble()
    # only cap shot; free / in-state untouched
    sub = pe.subsample({'shot': 10}, 'ARB', np.random.default_rng(1))
    c = _counts(sub)
    assert c['ARBR'] == 10 and c['ARAR'] == 10        # both shot types capped
    assert c['ARAA'] == 800 and c['ARBA'] == 5        # free untouched
    assert c['AAAA'] == 20 and c['BBBB'] == 10        # in-state untouched


def test_subsample_reproducible_with_seed():
    pe = _ensemble()
    a = pe.subsample({'shot': 50}, 'ARB', np.random.default_rng(7)).types()
    b = pe.subsample({'shot': 50}, 'ARB', np.random.default_rng(7)).types()
    assert np.array_equal(np.sort(a), np.sort(b))


def test_subsample_small_ensemble_untouched():
    # caps larger than the counts -> everything kept
    pe = _ensemble()
    sub = pe.subsample({'shot': 10_000, 'free': 10_000, 'in_state': 10_000},
                       'ARB', np.random.default_rng(0))
    assert len(sub.types()) == len(pe.types())


if __name__ == '__main__':
    for fn in (test_subsample_caps_per_direction_type,
               test_subsample_no_caps_is_identity,
               test_subsample_missing_key_leaves_category_uncapped,
               test_subsample_reproducible_with_seed,
               test_subsample_small_ensemble_untouched):
        fn()
        print(fn.__name__, 'OK')
