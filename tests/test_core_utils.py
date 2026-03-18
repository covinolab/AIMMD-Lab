from pathlib import Path

import numpy as np
from MDAnalysis import Universe
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.coordinates.timestep import Timestep

import aimmd.core.utils as core_utils
from aimmd.core.base import AbstractArray
from aimmd.core.decorators import classproperty
from aimmd.core.utils import (
    concatenate,
    cycle,
    extend_array,
    extract_folder_and_name,
    get_local_index,
    guess_masses,
    longest_true_segment,
    memory_reader_from_timesteps,
    merge_ranges,
    process_state,
    replace_in_cache,
)


def test_local_index_and_collection_helpers():
    """Exercise the small pure helpers that define AIMMD's indexing semantics."""

    # `offsets` is a cumulative end-position array. These assertions pin down
    # the exact "global index -> (block, local index)" behavior used throughout
    # `Path` and `PathEnsemble`.
    offsets = np.array([2, 5, 8])
    assert get_local_index(0, offsets) == (0, 0)
    assert get_local_index(4, offsets) == (1, 2)
    # With `clip=True`, this helper returns the final valid *global* index in
    # the special out-of-range case, which is a slightly unusual but real quirk
    # of the current implementation that downstream code relies on.
    assert get_local_index(99, offsets, clip=True) == 7
    assert cycle([1, 2, 3, 4], 1) == [2, 3, 4, 1]

    # Empty arrays are skipped by `concatenate`, so the non-empty payload is
    # preserved without the caller having to special-case empty inputs.
    arrays = [np.array([]), np.array([[1], [2]]), np.array([[3]])]
    np.testing.assert_array_equal(concatenate(arrays), np.array([[1], [2], [3]]))

    # `extend_array` zero-pads along axis 0 and freezes the result, matching how
    # AIMMD exposes cache-like arrays that should not be mutated by callers.
    extended = extend_array(np.array([1, 2]), 4)
    np.testing.assert_array_equal(extended, np.array([1, 2, 0, 0]))
    assert not extended.flags.writeable

    # These helpers are used for path/range bookkeeping and for detecting the
    # longest "active" segment in boolean masks.
    assert merge_ranges([(4, 5), (1, 2), (2, 4), (10, 11)]) == [(1, 5), (10, 11)]
    assert longest_true_segment([False, True, True, False, True]) == (1, 3)


def test_misc_helpers(tmp_path):
    """Check string/path helpers and cache-key migration behavior."""

    assert process_state("b") == "B"
    assert process_state("1") == "R"
    assert extract_folder_and_name("foo/bar.xtc") == ("foo", "bar.xtc")

    old = Path(tmp_path) / "old.txt"
    new = Path(tmp_path) / "new.txt"
    old.write_text("x")

    class DummyCache:
        def __init__(self):
            self._cache = {str(old): "cached"}

    cache = DummyCache()
    replace_in_cache(cache, str(old), str(new))
    assert new.exists()
    # The file is renamed on disk and the in-memory cache entry follows it, so
    # later lookups by the new name still see the existing cached payload.
    assert cache._cache[str(new)] == "cached"


def test_memory_reader_classproperty_and_abstract_array():
    """Verify low-level abstractions used by the rest of the codebase."""

    core_utils.MemoryReader = MemoryReader
    core_utils.Timestep = Timestep

    universe = Universe.empty(2, trajectory=True)
    ts0 = universe.trajectory.ts.copy()
    ts0.positions = np.zeros((2, 3), dtype=np.float32)
    ts0.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=np.float32)
    ts1 = ts0.copy()
    ts1.positions = np.ones((2, 3), dtype=np.float32)

    reader = memory_reader_from_timesteps(ts0, [ts1])
    assert len(reader) == 2
    # The helper copies timestep data into a `MemoryReader`, so the second frame
    # should reproduce the positions we injected into `ts1`.
    np.testing.assert_allclose(reader[1].positions, np.ones((2, 3)))

    class Demo:
        @classproperty
        def name(cls):
            return cls.__name__.lower()

    # `classproperty` should resolve on the class itself without instantiation.
    assert Demo.name == "demo"

    class DemoArray(AbstractArray):
        def __init__(self):
            self._values = np.array([1, 2, 3])

        def _array(self):
            return self._values

    arr = DemoArray()
    # `AbstractArray` subclasses are expected to behave like thin NumPy-backed
    # proxies once `_array()` is implemented.
    np.testing.assert_array_equal(np.asarray(arr), np.array([1, 2, 3]))
    assert len(arr) == 3


def test_guess_masses_prefers_martini_beads():
    """Document the Martini-specific mass heuristic used by AIMMD."""

    universe = Universe.empty(2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
    universe.add_TopologyAttr("name", ["BB1", "SC1"])
    universe.add_TopologyAttr("masses", [12.0, 14.0])
    # Once a Martini-like bead name is detected, AIMMD deliberately assigns the
    # same default bead mass to every atom instead of trusting topology masses.
    np.testing.assert_array_equal(guess_masses(universe.atoms), np.array([72.0, 72.0]))
