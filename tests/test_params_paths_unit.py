"""Coverage-focused tests for `aimmd.params._paths` run-layout loaders."""

from pathlib import Path
import shutil

import numpy as np
import pytest

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import build_path


def _params(tmp_path):
    """Create a tiny Params object configured for file-layout loading tests."""

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states="ARB",
        sorted_states="ARB",
        trajectory_extension=".xtc",
        chain_type="rfps",
        parent=tmp_path,
    )
    return params


def test_free_trajectories_group_parts_and_apply_indicted_log(tmp_path):
    """Free trajectories are reconstructed by grouping `trajXXXXXX.partXXXX` files."""

    params = _params(tmp_path)
    source = build_path(
        tmp_path,
        stem="source",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    free_dir = tmp_path / "run" / "freeA"
    free_dir.mkdir(parents=True)
    shutil.copy2(source.fname, free_dir / "traj000001.part0000.xtc")
    shutil.copy2(source.fname, free_dir / "traj000001.part0001.xtc")
    (free_dir / "indicted.log").write_text("traj000001 2\n")

    trajectories = params.free_trajectories(str(tmp_path / "run"))
    assert len(trajectories) == 1
    assert trajectories[0]._exclude_from == 2


def test_shot_paths_reuses_old_objects_and_reads_tps_weights(tmp_path):
    """`shot_paths` should reuse already-loaded paths and fill TPS weights."""

    params = _params(tmp_path)
    params.__dict__["chain_type"] = "tps"
    source = build_path(
        tmp_path,
        stem="shot_source",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    chain_dir = tmp_path / "run" / "chainR0"
    chain_dir.mkdir(parents=True)
    path_fname = chain_dir / "path000001.xtc"
    shutil.copy2(source.fname, path_fname)
    # Copy the cached low-level arrays as well so the reconstructed path keeps
    # its transition classification when loaded from the new run folder.
    for target in ("states", "values", "descriptors"):
        shutil.copy2(get_cache_fname(source.fname, target), get_cache_fname(str(path_fname), target))
    save_npy(str(chain_dir / "tps_weights.npy"), np.array([0.75]))

    old = aimmd.PathEnsemble(aimmd.Path(str(path_fname), shooting_index="find"))
    # TPS weights are only filled for paths whose weight is still NaN, which is
    # how the loader distinguishes "new, weight not assigned yet" paths.
    old[0].weight = np.nan
    loaded = params.shot_paths(str(tmp_path / "run"), "chain", "R", 0, old=old)
    assert len(loaded) == 1
    # The loader should prefer the already-instantiated Path object from `old`
    # rather than creating a redundant second copy.
    assert loaded[0] is old[0]
    assert loaded.weights[0] == 0.75


def test_shot_paths_scan_modes_and_pathensemble_assembly(tmp_path):
    """The loader should support broad scans and compose a full pathensemble."""

    params = _params(tmp_path)
    source = build_path(
        tmp_path,
        stem="scan_source",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    run_dir = tmp_path / "run"
    for folder_name in ("chainR0", "chainR1"):
        folder = run_dir / folder_name
        folder.mkdir(parents=True)
        shutil.copy2(source.fname, folder / "path000001.xtc")
    free_dir = run_dir / "freeA"
    free_dir.mkdir(parents=True)
    shutil.copy2(source.fname, free_dir / "traj000001.part0000.xtc")

    chains = params.shot_paths(str(run_dir), "chain", "R", None)
    assert len(chains) == 2

    ensemble = params.pathensemble(str(run_dir))
    assert len(ensemble) >= 2


def test_shot_paths_rejects_invalid_k_type(tmp_path):
    """Non-integral, non-iterable chain selectors should fail clearly."""

    params = _params(tmp_path)
    with pytest.raises(TypeError):
        params.shot_paths(str(tmp_path), "chain", "R", object())
