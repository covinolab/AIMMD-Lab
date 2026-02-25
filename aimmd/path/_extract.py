"""
aimmd.path._extract
===================

Low-level extraction of per-file time series for :class:`aimmd.path.Path`.

This module defines :class:`~aimmd.path._extract.PathExtract`, a mixin providing
the internal method `_extract(...)`.

The Path object in AIMMD may span multiple underlying trajectory files (stored
in `self._fnames`) and may traverse each file forward or backward (via per-file
`_first` and `_last` indices). `_extract` is responsible for extracting a slice
of data **from a single underlying file** `k`, and mapping a user-facing key
(slice/indices) to the correct on-disk frame locations.

Extraction modes
----------------
1) In-memory attributes
   If `attribute` already exists in `self.__dict__`, it is assumed to be a
   global concatenated array indexed by `self.indices`. `_extract` maps local
   per-file indices into that global index space.

2) Reader-derived series (MDAnalysis)
   For attributes in:
   ('reader', 'frames', 'positions', 'times', 'coordinates',
    'velocities', 'dimensions')
   the method loads frames from `MDA_CACHE` and constructs numpy arrays.

3) File-based cached series
   For other attribute names, `_extract` attempts to load a `.npy` array from
   `NPY_CACHE` at the file produced by `get_cache_fname(fname, attribute)`.

Special attributes
------------------
- 'self':
  Returns a Path object restricted to the single file `k`, optionally sliced by
  `key`. This is used internally for per-file operations.

- 'locs':
  The per-file frame locations (absolute indices into the underlying trajectory).

- 'indices':
  The per-file indices in the *global concatenated index space* of the Path.

Missing data
------------
If the requested attribute is not available and `raise_if_missing` is False,
the method returns a default:
- for 'states': array of empty strings
- for everything else: array of zeros

Notes and caveats
-----------------
- This function assumes `MDA_CACHE.get(fname, min_length)[locs]` returns an
  indexable MDAnalysis trajectory slice.
- The code path for `attribute.endswith('_mtimes')` uses `os.path.getmtime`
  (requires `os` to be available in scope in the actual runtime module).
"""

# external
import numpy as np
from abc import ABC
from math import inf
from itertools import islice

# aimmd imports
from .utils import get_cache_fname
from .._config import NPY_CACHE, MDA_CACHE, DEFAULT_DIMENSIONS

# class with _extract function
class PathExtract(ABC):
    def _extract(self, k, attribute='self', key=None,
                 raise_if_missing=True):
        """
        Extract a per-file time series for the k-th underlying file of this Path.

        Parameters
        ----------
        k : int
            Index of the underlying filename in `self._fnames`.
        attribute : str, optional
            Name of the series to extract. Supported categories:

            - Structural:
              * 'self'     : return a Path restricted to file k
              * 'locs'     : absolute frame locations within the file
              * 'indices'  : global indices of these frames within the Path

            - Reader-derived (from MDAnalysis via `MDA_CACHE`):
              * 'reader', 'frames', 'positions', 'times', 'coordinates',
                'velocities', 'dimensions'

            - Cached array (from `NPY_CACHE`):
              * any other string, interpreted as a `.npy` series stored under
                `get_cache_fname(fname, attribute)`

            - Modification times:
              * '<name>_mtimes' returns an array filled with the mtime of the
                cached file for '<name>'.

        key : slice, array-like, or None, optional
            Per-file selection. If provided, only the selected frames are
            returned. For 'self', slicing is applied to the returned Path.
        raise_if_missing : bool, optional
            If True (default), missing series raise `TypeError`.
            If False, return a default array (empty strings for 'states',
            zeros otherwise).

        Returns
        -------
        object
            Depending on `attribute`:
            - Path for 'self'
            - numpy.ndarray for most series (filled with zeros where data are missing)
            - list of timesteps for 'frames'
            - MDAnalysis trajectory slice for 'reader'

        Raises
        ------
        TypeError
            If the requested series cannot be obtained and `raise_if_missing`
            is True.

        Notes
        -----
        - This method does not apply any state post-processing (e.g. label mapping).
        - It does not extend caches unless `min_length` forces cache loaders to
          refresh and read more frames.
        """

        if attribute == 'self':
            from . import Path
            result = object.__new__(Path)
            result._fnames = [self._fnames[k]]
            result._first = [self._first[k]]
            result._last = [self._last[k]]
            result._weight = self._weight

            # Compute global start offset: number of frames in files < k.
            start = 0
            for first, last in zip(self._first[:k], self._last[:k]):
                start += abs(last - first) + 1

            # Global stop index range for the single-file path.
            # NOTE: This uses `result.last[0]` which is flagged in-code as TODO.
            stop = start + result.last[0]  # TODO FIX HERE

            # Propagate exclusion boundary into the sub-path coordinate system.
            if self._exclude_from >= 0:
                exclude_from = max(0, self._exclude_from - start)
                if exclude_from > len(result):
                    result._exclude_from = -1
                else:
                    result._exclude_from = exclude_from
            else:
                result._exclude_from = -1

            # Map global shooting index into sub-path coordinates.
            if self._shooting_index < start:
                result._shooting_index = 0
            elif self._shooting_index < stop:
                result._shooting_index = self._shooting_index - start
            else:
                result._shooting_index = stop - start - 1

            # Copy cached arrays for this segment only (assumes arrays align
            # with global indices).
            for attribute, value in islice(
                self.__dict__.items(), 6, None):
                result.__dict__[attribute] = value[start:stop].copy()

            if key is not None:
                result = result[key]
            return result
        
        start = self._first[k]
        last = self._last[k]
        step = 1 if start <= last else -1
        stop = last + step

        if attribute == 'locs':
            result = np.arange(start, stop, step)
            if key is not None:
                result = result[key].flatten()
            return result
        
        if attribute == 'indices':
            # Global index offset is the cumulative length of previous file segments.
            offset = self.offsets[k - 1] if k else 0
            length = abs(last - start) + 1
            result = np.arange(length)
            if key is not None:
                result = result[key].flatten()
            return offset + result
        
        if attribute in self.__dict__:
            # Attribute is already stored globally on the Path; map local->global.
            indices = self._extract(k, 'indices', key)
            return self.__dict__[attribute][indices]
        
        # from keys to locs
        if key is None:
            length = abs(last - start) + 1
            locs = slice(start, stop if stop >= 0 else None, step)
        elif isinstance(key, slice):
            locs = range(start, stop, step)[key]
            start = locs.start  # updated
            stop = locs.stop
            step = locs.step
            length = len(locs)
            locs = slice(start, stop if stop >= 0 else None, step)
        else:
            locs = np.arange(start, stop, step)[key].flatten()
            length = len(locs)
        
        if attribute == 'filenames':
            return np.repeat(self._fnames[k], length)
        
        fname = self._fnames[k]
        min_length = max(start + 1, stop)
        
        # reader-based
        if attribute in ('reader', 'frames', 'positions', 'times',
                         'coordinates', 'velocities', 'dimensions'):
            reader = MDA_CACHE.get(fname, min_length)[locs]
            if attribute == 'reader':
                return reader
            if attribute == 'frames':
                return [ts.copy() for ts in reader]
            if attribute == 'times':
                result = np.array([ts.time for ts in reader])
            if attribute == 'positions':
                result = np.array([ts.positions.copy() for ts in reader])
            if attribute == 'coordinates':
                result = np.array([ts.positions.flatten() for ts in reader])
            if attribute == 'velocities':
                n_atoms = reader.trajectory.n_atoms
                result = np.array([ts.velocities.copy()
                                   if hasattr(ts, 'velocities') else
                                   np.zeros((n_atoms, 3)) for ts in reader])
            if attribute == 'dimensions':
                result = np.array([ts.dimensions.copy()
                                   if ts.dimensions is not None else
                                   DEFAULT_DIMENSIONS for ts in reader])
            result.flags.writeable = False
            return result

        # modification times
        if attribute.endswith('_mtimes'):
            fname = get_cache_fname(fname, attribute[:-7])
            return np.repeat(os.path.getmtime(fname), length)

        # file based (already extended up to element)
        data = NPY_CACHE.get(get_cache_fname(fname, attribute),
                             min_length=min_length)
        if data is not None:
            return data[locs]
        
        # process exceptions
        if raise_if_missing:
            raise TypeError(f'could not obtain {attribute!r} '
                            f'time series for {self._fnames[k]}')
        if attribute == 'states':
            result = np.full(length, '')
        else:
            result = np.zeros(length)
        result.flags.writeable = False  # cannot change here
        return result
