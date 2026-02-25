"""
aimmd.cache.base
================

Base cache abstractions for AIMMD.

This subpackage provides small, process-local, in-memory caches used to avoid
repeated expensive I/O operations (e.g., repeatedly reading trajectory files or
NumPy arrays). These caches are initialized during package startup in
:func:`aimmd._init.initialize` and stored on :mod:`aimmd._config` as global
singletons (e.g., ``_config.NPY_CACHE`` and ``_config.MDA_CACHE``).

High-level behavior
-------------------
The cache is a simple size-limited mapping:

- Keys: file names (strings)
- Values: loaded "instances" (e.g., numpy arrays or MDAnalysis readers)
- Eviction: FIFO order using :class:`collections.OrderedDict` (oldest first)
- Budget: ``total_size`` tracks approximate memory usage via ``sys.getsizeof``

Contract for subclasses
-----------------------
Subclasses implement:

- :meth:`_open(fname)`:
    Actually opens/loads the resource, returning an instance or raising.
- Optionally :meth:`_close(instance)`:
    Cleanup hook (e.g., close file handles / readers).
- Optionally :meth:`_extend(instance, min_length)`:
    Used when clients ask for at least ``min_length`` items and want to extend.

Notes / caveats
---------------
- ``sys.getsizeof`` is used as a heuristic for memory accounting. For complex
  objects (NumPy arrays, MDAnalysis readers), it may under/overestimate true
  memory usage. The current logic keeps this behavior as-is.

This file contains only the cache base class and avoids heavyweight imports.
"""

# external
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable

# abstract cache
class AbstractCache(ABC):
    """
    Abstract size-limited cache.

    Attributes
    ----------
    max_size : int or None
        Approximate cache memory budget (bytes). Subclasses typically set this.
    _cache : OrderedDict
        Stores cached instances; eviction is FIFO (popitem(last=False)).
    total_size : int
        Approximate total size (bytes) of cached instances.

    Notes
    -----
    - This cache is process-local; it is not shared across processes.
    - It is intentionally minimal and does not implement concurrency control.
      For filesystem-level locking, see :mod:`aimmd.cache.npy`.
    """
    max_size = None
    
    def __init__(self):
        # OrderedDict preserves insertion order for FIFO eviction.
        self._cache = OrderedDict()
        # Heuristic memory accounting (sum of sys.getsizeof(instance)).
        self.total_size = 0
    
    def __len__(self):
        # Number of cached entries (not total frames or bytes).
        return len(self._cache)
    
    @abstractmethod
    def _open(self, fname):
        """
        Open/load a resource.

        Parameters
        ----------
        fname : str
            File name / path.

        Returns
        -------
        object or None
            Loaded instance. Returning None indicates failure to open.

        Notes
        -----
        Subclasses may raise exceptions; :meth:`open` catches them and returns None.
        """
        pass
    
    def _close(self, instance):
        """
        Cleanup hook called when removing an instance from cache.

        Parameters
        ----------
        instance : object
            Cached instance previously returned by `_open`.

        Notes
        -----
        Default is a no-op. Subclasses can override, e.g. call `instance.close()`.
        """
        pass

    def _extend(self, instance, min_length):
        """
        Optional hook: extend/pad an instance to at least `min_length`.

        This is used when callers request `extend=True` in :meth:`get`.

        Default implementation returns `instance` unchanged.
        """
        return instance
    
    def get(self, fname, min_length=0, extend=False):
        """
        Retrieve an instance from cache, optionally ensuring a minimum length.

        Parameters
        ----------
        fname : str
            Resource identifier (typically a filepath).
        min_length : int, default 0
            If the cached instance is missing or shorter than this length,
            the resource is reloaded via :meth:`load`.
        extend : bool, default False
            If True and an instance is available, call :meth:`_extend` to ensure
            at least `min_length`.

        Returns
        -------
        object or None
            Cached (or reloaded) instance, possibly extended; None on failure.
        """
        # Fast path: try cache first
        instance = self._cache.get(fname, None)
        # Reload if not cached or too short
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
        Safe open wrapper around :meth:`_open`.

        Returns
        -------
        object or None
            Instance on success, None on any exception.
        """
        try:
            return self._open(fname)
        except:
            # Keep permissive behavior: any failure returns None.
            return None
    
    def load(self, fname):
        """
        Load a resource and update the cache.

        Behavior
        --------
        - Opens the resource via :meth:`open`.
        - Removes any existing cached entry for `fname`.
        - Evicts oldest entries until there is space for the new one.
        - Inserts the new instance and updates `total_size`.

        Returns
        -------
        object or None
            The loaded instance, or None if opening failed.

        Notes
        -----
        Uses `sys.getsizeof` for instance size estimation.
        """
        instance = self.open(fname)
        if instance is None:
            return
        # Remove existing cached version (if any) to refresh insertion order
        self.remove(fname)
        size = sys.getsizeof(instance)
        # Evict FIFO until there is enough budget
        while self.total_size + size >= self.max_size:
            self.remove()
        # Insert and account
        self._cache[fname] = instance
        self.total_size += size
        return instance
    
    def reload(self):
        """
        Reload all currently cached file names (refresh the cache contents).

        Returns
        -------
        list
            List of reloaded instances (may include None entries).
        """
        return [self.load(fname) for fname in self._cache]
    
    def pop(self, fname=None):
        """
        Remove an entry from the cache and return the instance.

        Parameters
        ----------
        fname : str or None
            - If str: pop that key if present, otherwise open it without caching.
            - If None: pop the *oldest* cached entry (FIFO).
            - Otherwise: coerces to str and recurses.

        Returns
        -------
        object or None
            The removed (or opened) instance, or None if nothing found/opened.

        Notes
        -----
        This method updates `total_size` when removing cached entries.
        """
        # Remove a specific entry (or open it if not cached)
        if isinstance(fname, str):
            instance = self._cache.pop(fname, None)
            if instance is not None:
                self.total_size -= sys.getsizeof(instance)
            else:
                instance = self.open(fname)
            return instance
        # Pop oldest entry
        if fname is None:
            if len(self):
                instance = self._cache.popitem(last=False)[1]
                self.total_size -= sys.getsizeof(instance)
                return instance
            return
        # Fallback: try string conversion
        return self.pop(str(fname))
    
    def remove(self, fname=None):
        """
        Remove an entry from the cache and run the cleanup hook.

        Parameters
        ----------
        fname : str or None
            Passed to :meth:`pop`. If None, removes the oldest entry.

        Notes
        -----
        Uses `try/finally` to preserve current behavior (always returns).
        """
        try:
            instance = self.pop(fname)
            self._close(instance)
            del instance
        finally:
            return

    def clear(self):
        """
        Clear the entire cache, calling cleanup hooks on all entries.
        """
        while self._cache:
            self.remove()
    
