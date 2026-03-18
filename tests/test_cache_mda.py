from pathlib import Path

import numpy as np

from aimmd.cache.mda import MDAReaderCache, count_safe_frames, remove_offset_files
from tests._helpers_unit import write_trajectory


def test_remove_offset_files(tmp_path):
    """MDAnalysis offset sidecars should be cleaned up deterministically."""

    fname = Path(write_trajectory(tmp_path, stem="offset_test"))
    (tmp_path / f".{fname.name}_offsets.npz").write_text("x")
    (tmp_path / f".{fname.name}_offsets.lock").write_text("x")
    remove_offset_files(str(fname))
    assert not (tmp_path / f".{fname.name}_offsets.npz").exists()
    assert not (tmp_path / f".{fname.name}_offsets.lock").exists()


def test_mda_reader_cache_and_safe_frames(tmp_path):
    """A small synthetic trajectory should round-trip through the reader cache."""

    fname = write_trajectory(tmp_path, stem="traj")
    cache = MDAReaderCache()
    reader = cache.get(fname)
    assert len(reader) == 3
    # The whole file is readable, so the "safe" prefix should equal the full
    # frame count and the final coordinates should match the written data.
    assert count_safe_frames(reader) == 3
    np.testing.assert_allclose(reader[2].positions[0], np.array([1.0, 0.0, 0.0]))
