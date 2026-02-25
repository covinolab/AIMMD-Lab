"""
aimmd.cache
===========

Caching utilities for AIMMD.

This subpackage groups small caches used to reduce repeated disk I/O. The caches
are instantiated during package initialization (:mod:`aimmd._init`) and exposed
via :mod:`aimmd._config` as global singletons.

Public API
----------
- :class:`~aimmd.cache.npy.NpyReaderCache`
- :class:`~aimmd.cache.mda.MDAReaderCache`
"""

from .npy import NpyReaderCache
from .mda import MDAReaderCache
