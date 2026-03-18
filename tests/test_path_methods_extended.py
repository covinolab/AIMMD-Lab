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
