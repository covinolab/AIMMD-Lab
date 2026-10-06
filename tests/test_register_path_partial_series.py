"""register_path copes with a backward or forward half that lacks descriptors.

A shooting move joins the reversed backward half and the forward half into
one path and copies their per-frame descriptor rows into the path's series.
When one half had no series file, or a shorter one, register_path used to
crash (``None[back_indices]``, IndexError) or, when only the forward half was
missing, drop the rows of both halves. Missing rows are now zero rows, which
the descriptor ledger treats as "not computed" and refills on its next pass.
"""
import os

import numpy as np
import pytest

from aimmd._config import NPY_CACHE
from aimmd.cache.npy import load_npy, save_npy
from aimmd.pathensemble import PathEnsemble
from aimmd.path.utils import get_cache_fname
from aimmd.worker.utils import register_path
from tests._helpers_unit import build_path, simple_descriptors_function

# backward half R -> A, forward half R -> B, two atoms
BACK_X = np.array([0.0, -0.3, -0.6, -0.9])
FORW_X = np.array([0.0, 0.3, 0.6, 0.9])


def _positions(x):
    zeros = np.zeros_like(x)
    return np.stack([np.stack([x, zeros, zeros], 1),
                     np.stack([x + 2.0, zeros, zeros], 1)], 1).astype(np.float32)


def _halves(tmp_path):
    back = build_path(tmp_path, stem="back", positions=_positions(BACK_X))
    forw = build_path(tmp_path, stem="forw", positions=_positions(FORW_X))
    return back, forw


def _descriptors_file(path):
    return get_cache_fname(path.fname, "descriptors")


def _register(path):
    NPY_CACHE.clear()
    chain = PathEnsemble()
    register_path(path, chain, eneconv=None)
    registered = chain[-1]
    NPY_CACHE.clear()
    return registered, load_npy(_descriptors_file(registered))


def _featurized(path):
    """Descriptors of the registered path, computed from its own xtc."""
    return np.asarray(simple_descriptors_function(path.reader), dtype=float)


def _refill(path):
    """The descriptor ledger fills the zero rows (as the trainer would)."""
    NPY_CACHE.clear()
    n = path.compute(simple_descriptors_function, "descriptors")
    NPY_CACHE.clear()
    return n, load_npy(_descriptors_file(path))


@pytest.mark.parametrize("missing", ["back", "forw"])
def test_one_half_without_series_gives_zero_rows(tmp_path, missing):
    back, forw = _halves(tmp_path)
    os.remove(_descriptors_file(back if missing == "back" else forw))
    path = back[::-1] + forw[1:]

    registered, descriptors = _register(path)

    expected = _featurized(registered)
    n_back = len(back)
    assert descriptors is not None and descriptors.shape == expected.shape
    if missing == "back":
        assert not descriptors[:n_back].any()
        np.testing.assert_array_equal(descriptors[n_back:], expected[n_back:])
    else:
        np.testing.assert_array_equal(descriptors[:n_back], expected[:n_back])
        assert not descriptors[n_back:].any()

    # the zero rows are "missing" for the ledger, which refills exactly them
    n, refilled = _refill(registered)
    assert n == (n_back if missing == "back" else len(forw) - 1)
    np.testing.assert_array_equal(refilled, expected)


def test_short_half_series_gives_zero_rows_for_the_missing_frames(tmp_path):
    back, forw = _halves(tmp_path)
    # the backward series stops after 2 of 4 frames (e.g. written mid-segment)
    save_npy(_descriptors_file(back), np.asarray(
        simple_descriptors_function(back.reader), dtype=float)[:2])
    path = back[::-1] + forw[1:]

    registered, descriptors = _register(path)

    expected = _featurized(registered)
    # reversed backward half: frames 3, 2, 1, 0 -> only 1 and 0 were cached
    assert not descriptors[:2].any()
    np.testing.assert_array_equal(descriptors[2:], expected[2:])
    n, refilled = _refill(registered)
    assert n == 2
    np.testing.assert_array_equal(refilled, expected)


def test_backward_only_path_with_short_series(tmp_path):
    """A backward half that reached max_length is registered alone."""
    back, forw = _halves(tmp_path)
    save_npy(_descriptors_file(back), np.asarray(
        simple_descriptors_function(back.reader), dtype=float)[:3])
    path = back[len(back) - 1::-1] + forw[1:1]     # as the shoot loop builds it
    assert path.n_files == 1

    registered, descriptors = _register(path)

    expected = _featurized(registered)
    assert not descriptors[0].any()
    np.testing.assert_array_equal(descriptors[1:], expected[1:])


def test_no_series_on_either_half_writes_none(tmp_path):
    back, forw = _halves(tmp_path)
    os.remove(_descriptors_file(back))
    os.remove(_descriptors_file(forw))
    registered, descriptors = _register(back[::-1] + forw[1:])
    assert descriptors is None
    assert not os.path.exists(_descriptors_file(registered))


def test_complete_halves_are_copied_unchanged(tmp_path):
    back, forw = _halves(tmp_path)
    registered, descriptors = _register(back[::-1] + forw[1:])
    np.testing.assert_array_equal(descriptors, _featurized(registered))
