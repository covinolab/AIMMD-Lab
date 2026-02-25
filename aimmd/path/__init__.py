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
    
    __init__ = PathHelpers._init

# visible objects
__all__ = ['Path']
