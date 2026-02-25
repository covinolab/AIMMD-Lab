"""
aimmd.path
=========

Path representation for AIMMD trajectory segments.

This package defines :class:`aimmd.path.Path`, the core container representing
a sequence of frames that may span one or more underlying trajectory files.
A Path behaves like a lightweight, indexable time series with additional
bookkeeping required by path sampling (shooting point index, acceptance,
path type, etc.).

Implementation overview
-----------------------
`Path` is assembled via internal mixins:

- PathHelpers      : initialization and indexing helpers
- PathMagic        : magic methods and attribute access
- PathProperties   : derived properties and setters
- PathMethods      : higher-level operations and utilities
- PathExtract      : per-file extraction from readers/caches
- PathGet          : global retrieval across multi-file Paths
- PathPositions    : convenience accessors (initial/final/...)
- PathCompute      : batch computation and caching
- PathIO           : extending from files and writing output

Notes
-----
`Path.__init__` is assigned to `PathHelpers._init` to keep construction logic
centralized.
"""

from ._io import PathIO
from ._get import PathGet
from ._magic import PathMagic
from ._extract import PathExtract
from ._helpers import PathHelpers
from ._compute import PathCompute
from ._methods import PathMethods
from ._positions import PathPositions
from ._properties import PathProperties

class Path(
    PathHelpers,
    PathMagic,
    PathProperties,
    PathMethods,
    PathExtract,
    PathGet,
    PathPositions,
    PathCompute,
    PathIO):
    """
    Trajectory-backed path object with segment-aware frame indexing.

    A :class:`Path` represents a sequence of frames assembled from one or more
    (possibly not contiguous) trajectory *segments* on disk. It is the basic
    unit manipulated by AIMMD path sampling:

    - workers incrementally extend it while a simulation is running,
    - shooting tasks slice and concatenate it to assemble new paths,
    - analysis tasks compute and cache per-frame series (states, descriptors,
      values) aligned with the path frames.

    Storage model
    -------------
    A Path stores an ordered list of trajectory segments:

    - ``_fnames`` : list[str]
      One filename per segment (trajectory file path).
    - ``_first`` / ``_last`` : list[int]
      Per-segment local frame bounds (inclusive). A segment may be forward
      (``first <= last``) or backward (``first > last``).

    The global path frame index space is the concatenation of per-segment
    intervals in the order stored in ``_fnames``. Most operations treat the path
    as a linear sequence of frames, hiding the segment structure from callers.

    Caching and derived series
    --------------------------
    Per-frame quantities associated with a path (e.g., ``states``,
    ``descriptors``, ``values``) are commonly cached in `.npy` files derived
    from the trajectory filename. These cached arrays are assumed to be aligned
    with trajectory *frames* and are subset/concatenated according to the path’s
    segment slices.

    In AIMMD, trajectory readers are obtained through a global MDAnalysis reader
    cache (``aimmd._config.MDA_CACHE``). Methods that mutate the underlying
    trajectory files should evict affected entries from that cache.

    Key operations and semantics
    ----------------------------
    - Slicing returns a new Path describing the corresponding subsequence of
      frames, preserving segment semantics and direction.
    - Concatenation (``p + q``) returns a new Path that is the frame-wise
      concatenation of the two paths (typically used to join backward and
      forward segments in shooting).
    - :meth:`extend` mutates the path by appending frames discovered on disk,
      and is the core primitive used while a simulation is running.
    - :meth:`write` exports the frames represented by the path into a new
      trajectory file.

    Notes
    -----
    - Many AIMMD workflows assume that extension operates on forward segments
      only (monotonic frame growth within each segment). Backward segments may
      exist as a *representation* (e.g., reversed slices) but are typically not
      extended in place.
    - The path object is intentionally lightweight: it stores indexing metadata
      and filenames, while heavy data access is delegated to cached readers and
      computed arrays.
    """
    
    __init__ = PathHelpers._init

# visible objects
__all__ = ['Path']
