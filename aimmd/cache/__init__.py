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

Functions
---------
forget
    Drop what this process caches about trajectory files (their readers and
    per-frame arrays), e.g. after the files were cut or replaced.

Notes
-----
These caches are instantiated during :func:`aimmd._init.initialize` and stored
in :mod:`aimmd._config` as shared singletons.
"""

from .npy import NpyReaderCache
from .mda import MDAReaderCache


def forget(*trajectories):
    """
    Drop what this process caches in memory about trajectory files.

    Evicts each file's reader from ``MDA_CACHE`` and all its per-frame arrays
    (``<file>.<series>.npy``) from ``NPY_CACHE``, matching the cache keys by
    absolute path, and removes the MDAnalysis offsets files beside it, so that
    the next read sees what is on disk now. Both caches reuse an entry for as
    long as it is long enough, so a file that was cut, moved away or replaced
    by another one of the same name must be forgotten explicitly.

    Parameters
    ----------
    *trajectories : str
        Trajectory files.
    """
    import os
    from .. import _config
    from .mda import remove_offset_files

    targets = [os.path.abspath(str(fname)) for fname in trajectories]
    if not targets:
        return
    for cache in (getattr(_config, 'NPY_CACHE', None),
                  getattr(_config, 'MDA_CACHE', None)):
        if cache is None:
            continue
        for key in list(cache._cache):
            path = os.path.abspath(str(key))
            if any(path == target or (path.startswith(f'{target}.')
                                      and path.endswith('.npy'))
                   for target in targets):
                cache.remove(key)
    for fname in trajectories:
        remove_offset_files(str(fname))
