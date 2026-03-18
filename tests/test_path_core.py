import numpy as np

import aimmd
from tests._helpers_unit import build_path, write_trajectory


def test_path_basic_accessors_and_slicing(tmp_path):
    """Exercise the most common `Path` accessors on a tiny three-frame path."""

    path = build_path(
        tmp_path,
        stem="core",
        positions=np.array(
            [[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
        shooting_index=1,
    )
    assert len(path) == 3
    assert path.initial("states") == "A"
    assert path.middle("states") == "R"
    assert path.final("states") == "B"
    assert path.shooting("states") == "R"
    # The cached state series is aligned one-to-one with path frames.
    np.testing.assert_array_equal(path.all("states"), np.array(["A", "R", "B"]))

    # Slicing should preserve frame order and return another `Path` view.
    sliced = path[1:]
    assert len(sliced) == 2
    np.testing.assert_array_equal(sliced.states, np.array(["R", "B"]))


def test_path_compute_extend_and_write(tmp_path):
    """Check growth, on-the-fly computation, and trajectory export."""

    fname = write_trajectory(
        tmp_path,
        stem="extend_source",
        positions=np.array(
            [[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    )
    path = aimmd.Path(fname, stop=2, shooting_index=1)
    added, frames_left = path.extend(fname, nframes=10)
    # The initial path was created from only the first two frames; extending
    # against the same file should therefore append the remaining two.
    assert added == 2
    assert frames_left == 0

    result = path.compute(lambda coords: coords[:, 0, 0], source="positions", return_result=True)
    # The test trajectory was built with monotonic x coordinates, so extracting
    # `coords[:, 0, 0]` should recover those values exactly.
    np.testing.assert_allclose(result, np.array([-1.0, 0.0, 1.0, 2.0]))

    out = tmp_path / "written.xtc"
    times = path.write(out)
    assert len(times) == len(path)
    written = aimmd.Path(str(out))
    # Writing and reading back should preserve the frame count.
    assert len(written) == len(path)
