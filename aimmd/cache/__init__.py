"""
aimmd.cache
===========

Cache implementations used by AIMMD.

This subpackage provides cache objects that manage:
- safe, concurrent access to `.npy` files (with file locks),
- robust opening/slicing of MDAnalysis trajectories (handling partial writes),
- optional extension/padding behavior for cached arrays.

Public classes
--------------
NpyReaderCache
    Cache for read-only NumPy arrays loaded from `.npy` files.

MDAReaderCache
    Cache for MDAnalysis trajectory readers (robust open and safe slicing).

Notes
-----
These caches are instantiated during :func:`aimmd._init.initialize` and stored
in :mod:`aimmd._config` as shared singletons.
"""

from .npy import NpyReaderCache
from .mda import MDAReaderCache
