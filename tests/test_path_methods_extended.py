"""Additional coverage for the high-level convenience methods on `Path`."""

from pathlib import Path
import shutil

import numpy as np
import pytest

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import build_path, write_trajectory


def test_path_classification_and_shooting_result(tmp_path):
    """The semantic helpers should agree on a simple A-R-B transition path."""

    path = build_path(
        tmp_path,
        stem="transition",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )

    assert path.is_complete("R", "ARB")
    assert path.is_transition("ARB")
    assert path.is_excursion("ARB")
    assert not path.is_internal("ARB")
    np.testing.assert_allclose(path.shooting_result("ARB"), np.array([1.0, 1.0]))


def test_check_stop_handles_bad_starts_exclusions_and_length_limits(tmp_path):
    """`check_stop` scans state blocks and reports the first stopping reason."""

    path = build_path(
        tmp_path,
        stem="check_stop",
        positions=np.array(
            [[[-1, 0, 0]], [[0, 0, 0]], [[0.1, 0, 0]], [[1, 0, 0]]], dtype=np.float32
        ),
    )

    with pytest.raises(RuntimeError):
        path.check_stop(allowed_states="R", check_first_frame=True)

    # `check_stop` treats the fully connected non-empty state string as one
    # contiguous block, so a short `max_length` stops from the first frame.
    stop_index, nframes, last_state, block_length = path.check_stop(max_length=1)
    assert stop_index == 0
    assert nframes == 4
    assert last_state == "B"
    assert block_length == 4

    # Because the trajectory is still one contiguous non-empty block, raising
    # `_exclude_from` alone does not create a new split point.
    path._exclude_from = 2
    stop_index, *_ = path.check_stop()
    assert stop_index is None


def test_path_partial_split_memory_copy_and_sampling(tmp_path):
    """Exercise the higher-level data-management helpers on a tiny path."""

    fname = write_trajectory(
        tmp_path,
        stem="times",
        positions=np.array(
            [[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]], [[0, 0, 0]], [[-1, 0, 0]]],
            dtype=np.float32,
        ),
        times=np.array([2.0, 1.0, 0.0, 1.0, 2.0]),
    )
    save_npy(get_cache_fname(fname, "states"), np.array(list("ARBRA"), dtype="<U1"))
    save_npy(get_cache_fname(fname, "values"), np.array([-1.0, 0.0, 1.0, 0.0, -1.0]))
    save_npy(
        get_cache_fname(fname, "descriptors"),
        np.array([[-1.0], [0.0], [1.0], [0.0], [-1.0]], dtype=float),
    )
    path = aimmd.Path(fname, shooting_index="find")

    # `partial` should return a list when selecting multiple file segments.
    partial = path.partial("self", slice(None))
    assert isinstance(partial, list)
    assert len(partial) == path.n_files

    split_paths = path.split()
    assert len(split_paths) == 3

    assert not path.in_memory()
    path.to_memory()
    assert path.in_memory()
    path.from_files()
    assert not path.in_memory()

    copy = path.copy()
    assert copy.fname == path.fname
    assert copy is not path

    # The path starts on the backward branch because times decrease from the
    # first frame, so the inferred shooting point should be the middle frame.
    assert path.find_shooting_index() == 2

    sampled = path.sample(2, source="values", vmin=-0.1, vmax=0.1)
    assert len(sampled) == 2


def test_update_exclude_from_reads_log_file(tmp_path):
    """The exclude log parser should match on basename and optional index."""

    path = build_path(tmp_path, stem="logged")
    log_file = tmp_path / "indicted.log"
    log_file.write_text(f"{Path(path.fname).name} 4\n")

    path.update_exclude_from(log_file)
    assert path._exclude_from == 4


# ------------------------------------------------ _extreme: one read only --
def _count_gets(path, monkeypatch):
    """Record every attribute `_get` fetches. `_range` reads `states` first."""
    calls = []
    orig = type(path)._get
    monkeypatch.setattr(
        type(path), '_get',
        lambda self, attr, *a, **k: (calls.append(attr),
                                     orig(self, attr, *a, **k))[1])
    return calls


def test_extreme_reads_the_source_once_when_attribute_is_source(tmp_path, monkeypatch):
    """`path.max('values','values')` must read `values` once, not twice.

    A missing `else:` in PathHelpers._extreme made `series = values` dead, so
    the same attribute was fetched twice. That is not a cache hit: for anything
    other than descriptors/states, Path._extract deliberately routes through
    NPY_CACHE.load, which never short-circuits -- so the duplicate was a second
    FileLock + np.load, measured at 97.9 ms on the campaign's NFS mount, 81 ms
    of it the lock alone.

    `min`/`max` select `where='internal'`, so expectations are taken over that
    range -- and computed *before* the counter is installed, since reading them
    would otherwise be counted too.
    """
    path = build_path(tmp_path, values=[0.0, 3.0, 1.0])
    vals = path._get('values', *path._range('internal'))
    want_max, want_min = np.max(vals), np.min(vals)

    calls = _count_gets(path, monkeypatch)
    assert path.max('values', 'values') == want_max
    assert calls.count('values') == 1, f'expected one read, got {calls}'

    calls.clear()
    assert path.min('values', 'values') == want_min
    assert calls.count('values') == 1, f'expected one read, got {calls}'


def test_extreme_still_reads_both_when_attribute_differs(tmp_path, monkeypatch):
    """The `attribute != source` path must keep fetching both arrays."""
    path = build_path(tmp_path, values=[0.0, 3.0, 1.0])
    start, stop = path._range('internal')
    vals = path._get('values', start, stop)
    want = path._get('times', start, stop)[np.argmax(vals)]

    calls = _count_gets(path, monkeypatch)
    got = path.max('times', 'values')

    assert calls.count('values') == 1 and calls.count('times') == 1, calls
    assert got == want, 'times at the argmax of values'


def test_extreme_one_arg_form_is_unchanged(tmp_path):
    """`path.max(source)` -- the analysis/utils.py:995 shape -- still works.

    Note this defaults to source='values', so it is the *same* branch as the
    two-argument form above, which is why it must not regress.
    """
    path = build_path(tmp_path, values=[0.0, 3.0, 1.0])
    vals = path._get('values', *path._range('internal'))
    assert path.max('values') == np.max(vals)
    assert path.min('values') == np.min(vals)
