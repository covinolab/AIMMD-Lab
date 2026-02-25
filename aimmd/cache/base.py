"""
aimmd.cache.base
================

Base cache abstractions used across AIMMD.

AIMMD uses small, process-local, in-memory caches to avoid repeatedly opening
and parsing expensive on-disk resources, such as:

- NumPy `.npy` arrays (see :class:`aimmd.cache.npy.NpyReaderCache`)
- MDAnalysis trajectory readers (see :class:`aimmd.cache.mda.MDAReaderCache`)

These caches are instantiated during package initialization in :mod:`aimmd._init`
and exposed via :mod:`aimmd._config` (e.g. ``_config.NPY_CACHE`` and
``_config.MDA_CACHE``).

Design
------
The cache is a simple size-limited mapping:

- Key: filename/path (string)
- Value: opened/loaded instance (any object)
- Storage: :class:`collections.OrderedDict` to preserve insertion order
- Eviction: FIFO (oldest entry removed first)
- Memory accounting: heuristic via :func:`sys.getsizeof`

Subclass contract
-----------------
Subclasses implement:
- :meth:`_open`: open/load the resource.
- Optionally :meth:`_close`: cleanup hook for evicted entries.
- Optionally :meth:`_extend`: extension/padding hook used by :meth:`get`.

Important notes
---------------
- Concurrency: this cache is process-local and does not implement internal locks.
  When filesystem-level concurrency matters, use explicit file locks (see
  :mod:`aimmd.cache.npy`).
- Memory accounting via `sys.getsizeof` is heuristic; it is preserved as-is.
"""

# external
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable


class AbstractCache(ABC):
    """
    Minimal size-limited cache base class.

    Attributes
    ----------
    max_size : int or None
        Maximum allowed approximate cache size in bytes. Subclasses set this.
    _cache : collections.OrderedDict
        Cached mapping. Oldest entries are evicted first.
    total_size : int
        Approximate total cached size computed by summing `sys.getsizeof`.

    Notes
    -----
    - This cache assumes cached instances support `__len__`, because :meth:`get`
      compares `len(instance)` against `min_length`.
    - The cache is intentionally minimal and not thread/process safe.
    """
    max_size = None
    
    def __init__(self):
        # Maintain insertion order for FIFO eviction.
        self._cache = OrderedDict()
        # Track approximate memory budget consumption.
        self.total_size = 0
    
    def __len__(self):
        # Number of cached entries (not bytes).
        return len(self._cache)
    
    @abstractmethod
    def _open(self, fname):
        """
        Open/load the resource identified by `fname`.

        Parameters
        ----------
        fname : str
            File path or cache key.

        Returns
        -------
        object
            The opened resource instance.
        """
        pass

    def _close(self, instance):
        """
        Close/cleanup a cached instance.

        Parameters
        ----------
        instance : object
            Cached instance to close.

        Notes
        -----
        Default: no-op. Subclasses override to close file handles/readers.
        """
        # no-op by default
        return

    def _extend(self, instance, min_length):
        """
        Optionally extend/pad an instance to satisfy a requested minimum length.

        Parameters
        ----------
        instance : object
            Cached instance.
        min_length : int
            Desired minimum length for `len(instance)`.

        Returns
        -------
        object
            Extended instance.

        Notes
        -----
        Default: return the instance unchanged.
        """
        return instance
    
    def get(self, fname, min_length=0, extend=False):
        """
        Retrieve an instance from cache, reloading if missing/too short.

        Parameters
        ----------
        fname : str
            File path used as cache key.
        min_length : int, default 0
            If cached instance is missing or `len(instance) < min_length`,
            reload via :meth:`load`.
        extend : bool, default False
            If True, run `_extend` after retrieval.

        Returns
        -------
        object or None
            Cached/loaded instance, or None if open/load failed.
        """
        # Attempt fast-path lookup
        instance = self._cache.get(fname, None)

        # Reload if not present or insufficient length
        if instance is None or len(instance) < min_length:
            new = self.load(fname)
            if new is not None:
                instance = new

        # Optional extension/padding step
        if instance is not None and extend:
            instance = self._extend(instance, min_length)

        return instance

    def open(self, fname):
        """
        Safe wrapper around `_open`.

        Returns
        -------
        object or None
            Instance on success, or None if `_open` raises.
        """
        try:
            return self._open(fname)
        except:
            # Preserve permissive behavior: failures yield None.
            return None
    
    def load(self, fname):
        """Updates cache with fname"""
        # Open resource (may return None)
        instance = self.open(fname)
        if instance is None:
            return

        # Remove any existing entry for this key (refresh insertion order + size)
        self.remove(fname)

        # Estimate the memory cost of the new instance
        size = sys.getsizeof(instance)

        # Evict oldest entries until there is room
        while self.total_size + size >= self.max_size:
            self.remove()

        # Insert instance and account memory
        self._cache[fname] = instance
        self.total_size += size
        return instance
    
    def reload(self):
        """
        Reload every currently cached filename.

        Returns
        -------
        list
            List of newly loaded instances (may include None if failures occur).
        """
        return [self.load(fname) for fname in self._cache]
    
    def pop(self, fname=None):
        """
        Remove an instance from cache and return it.

        Behavior
        --------
        - If `fname` is a string and cached: remove and return cached instance.
        - If `fname` is a string and not cached: open it (not cached) and return it.
        - If `fname` is None: evict and return the oldest cached instance (FIFO).
        - Otherwise: convert `fname` to str and retry.

        Returns
        -------
        object or None
            Popped instance or opened instance, or None if nothing could be returned.

        Notes
        -----
        Adjusts `total_size` only when removing a cached entry.
        """
        """Remove from cache if there, opens if not there.
        Fallback: returns None."""
        if isinstance(fname, str):
            instance = self._cache.pop(fname, None)
            if instance is not None:
                self.total_size -= sys.getsizeof(instance)
            else:
                # Not cached: open without caching
                instance = self.open(fname)
            return instance

        if fname is None:
            if len(self):
                # Pop oldest (FIFO)
                instance = self._cache.popitem(last=False)[1]
                self.total_size -= sys.getsizeof(instance)
                return instance
            return

        # Fallback: stringify and retry
        return self.pop(str(fname))
    
    def remove(self, fname=None):
        """
        Remove an entry and call the `_close` hook.

        Parameters
        ----------
        fname : str or None
            If None, removes the oldest entry; otherwise removes that key.

        Notes
        -----
        Errors in `_close` are not propagated due to the `try/finally` structure.
        """
        try:
            instance = self.pop(fname)
            self._close(instance)
            del instance
        finally:
            return

    def clear(self):
        """
        Clear the cache completely, closing all cached entries.
        """
        while self._cache:
            self.remove()
