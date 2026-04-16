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
    - Cached instances are assumed to implement `__len__`, because :meth:`get`
      compares `len(instance)` against `min_length`.
    - The cache is intentionally minimal and not thread/process safe.
    """
    max_size = None

    def __init__(self):
        """
        Initialize an empty cache instance.

        Notes
        -----
        - `_cache` preserves insertion order for FIFO eviction.
        - `total_size` tracks a heuristic byte budget via `sys.getsizeof`.
        """
        # Maintain insertion order for FIFO eviction.
        self._cache = OrderedDict()
        # Track approximate memory budget consumption.
        self.total_size = 0

    def __len__(self):
        """
        Number of cached entries.

        Returns
        -------
        int
            Count of keys currently stored in the cache.
        """
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
            Opened resource instance.

        Raises
        ------
        Exception
            Subclasses may raise any exception to signal open/load failure.
            Callers should generally use :meth:`open` which converts failures
            to `None` (preserved behavior).
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
        Default is a no-op. Subclasses override this to close file handles,
        readers, etc. Errors are intentionally not propagated by :meth:`remove`.
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
        Default: return the instance unchanged. Subclasses may implement padding
        or reallocation behavior (e.g., for arrays).
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
            If True, run :meth:`_extend` after retrieval.

        Returns
        -------
        object or None
            Cached/loaded instance, or None if open/load failed.

        Notes
        -----
        This method preserves permissive behavior: failures to open/load are not
        raised here; they surface as `None`.
        """
        # Attempt fast-path lookup.
        instance = self._cache.get(fname, None)
        
        # Reload if not present or insufficient length.
        try:
            needs_reload = instance is None or len(instance) < min_length
        except:  # error in case instance is not accessible anymore
            needs_reload = True
        if needs_reload:
            new = self.load(fname)
            if new is not None:
                instance = new
        
        # Optional extension/padding step.
        if instance is not None and extend:
            instance = self._extend(instance, min_length)

        return instance

    def open(self, fname):
        """
        Safe wrapper around :meth:`_open`.

        Parameters
        ----------
        fname : str
            File path or cache key.

        Returns
        -------
        object or None
            Instance on success, or None if :meth:`_open` raises.

        Notes
        -----
        This method intentionally catches all exceptions to keep cache usage
        non-fatal in production workflows.
        """
        try:
            return self._open(fname)
        except:
            # Preserve permissive behavior: failures yield None.
            return None

    def load(self, fname):
        """
        Open and insert `fname` into the cache, evicting older entries if needed.

        Parameters
        ----------
        fname : str
            File path used as cache key.

        Returns
        -------
        object or None
            The loaded instance, or None if opening failed.

        Side Effects
        ------------
        - May evict entries to satisfy the `max_size` heuristic budget.
        - Updates `_cache` insertion order and `total_size`.
        """
        # Open resource (may return None).
        instance = self.open(fname)
        if instance is None:
            return

        # Remove any existing entry for this key (refresh insertion order + size).
        self.remove(fname)

        # Estimate the memory cost of the new instance.
        size = sys.getsizeof(instance)

        # Evict oldest entries until there is room.
        while self.total_size + size >= self.max_size:
            self.remove()

        # Insert instance and account memory.
        self._cache[fname] = instance
        self.total_size += size
        return instance

    def reload(self):
        """
        Reload every currently cached filename.

        Returns
        -------
        list
            List of newly loaded instances in the current cache iteration order.

        Notes
        -----
        This preserves the current permissive behavior: if a reload fails for a
        given entry, `None` may appear in the returned list.
        """
        return [self.load(fname) for fname in self._cache]

    def pop(self, fname=None):
        """
        Remove an instance from cache and return it (or open it if not cached).

        Parameters
        ----------
        fname : str or None, optional
            - If `str`: pop that key if present; otherwise open it (not cached).
            - If `None`: evict and return the oldest cached entry (FIFO).
            - Otherwise: `fname` is converted to `str` and retried.

        Returns
        -------
        object or None
            Popped cached instance, opened instance, or None if nothing can be returned.

        Notes
        -----
        `total_size` is decreased only when removing a cached entry.
        """
        # Preserve original inline comment/doc behavior.
        if isinstance(fname, str):
            instance = self._cache.pop(fname, None)
            if instance is not None:
                self.total_size -= sys.getsizeof(instance)
            else:
                # Not cached: open without caching.
                instance = self.open(fname)
            return instance

        if fname is None:
            if len(self):
                # Pop oldest (FIFO).
                instance = self._cache.popitem(last=False)[1]
                self.total_size -= sys.getsizeof(instance)
                return instance
            return

        # Fallback: stringify and retry.
        return self.pop(str(fname))

    def remove(self, fname=None):
        """
        Remove an entry and call the :meth:`_close` hook.

        Parameters
        ----------
        fname : str or None, optional
            If None, removes the oldest entry; otherwise removes that key.

        Side Effects
        ------------
        - Evicts an entry from `_cache` (if present).
        - Calls :meth:`_close` on the removed instance.

        Notes
        -----
        Errors in :meth:`_close` are intentionally not propagated due to the
        `try/finally` structure (preserved behavior).
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

        Side Effects
        ------------
        Calls :meth:`remove` repeatedly until `_cache` is empty.
        """
        while self._cache:
            self.remove()
