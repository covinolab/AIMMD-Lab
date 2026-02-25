"""
aimmd.path._get
===============

High-level attribute retrieval for :class:`aimmd.path.Path`.

This module defines :class:`~aimmd.path._get.PathGet`, a mixin that implements
the internal `_get(...)` method used by Path properties like:

- `path.positions`, `path.velocities`, `path.coordinates`,
- `path.states`, `path.frames`, `path.reader`,
- and any cached arrays stored in `path.__dict__` (e.g., descriptors/values).

Key features
------------
- Supports global slicing over a Path that may span multiple underlying files.
- Optimizes the “single-file” case (no concatenation needed).
- Supports “in-memory” mode:
  if `path.in_memory()` is True and the user requests `reader` or `frames`,
  returns a `MemoryReader` built from cached positions/velocities/dimensions.
- Supports state masking via `exclude_from`:
  if `path._exclude_from >= 0`, states beyond that point are replaced by ''.

Terminology
-----------
- `start, stop, step` are normalized using `slice(...).indices(len(self))`.
- Local indices are mapped into file segments using `self._get_local_index(...)`.
- Per-file extraction is delegated to `self._extract(k, attribute, key, ...)`.
- Multi-file access is represented via :class:`~aimmd.path.chainreader.ChainReader`.

Notes
-----
- `_get` returns different types depending on `attribute`:
  - 'reader' -> MDAnalysis reader-like object (or ChainReader of readers)
  - 'frames' -> list of Timestep copies
  - 'states' -> numpy array of dtype '<U1' (or empty strings when excluded,
                usually filled with '' where data are missing)
  - everything else -> numpy arrays (usually filled with zeros where data
    are missing)
"""

# external
import numpy as np
from abc import ABC
from MDAnalysis.coordinates.memory import MemoryReader

# aimmd imports
from .chainreader import ChainReader
from ..core.utils import extend_array


# class with _get function
class PathGet(ABC):

    def _get(self, attribute, start=0, stop=None, step=None,
             raise_if_missing=False):
        """
        Retrieve a Path attribute over a global slice.

        Parameters
        ----------
        attribute : str
            Name of the series to retrieve. Common values include:
            - 'indices'       : return global indices (0..len(path)-1)
            - 'reader'        : reader slice (single reader or ChainReader)
            - 'frames'        : list of Timestep copies
            - 'states'        : cached states array if present, otherwise extracted
            - 'true_states'   : alias for 'states' that bypasses exclusion masking
            - 'positions', 'coordinates', 'velocities', 'times', 'dimensions'
              (resolved via `_extract` when not cached)
            - any name stored in `self.__dict__` as a cached array

        start, stop, step : int or None, optional
            Slice bounds expressed in the *global Path index space*.
            They are normalized via `slice(start, stop, step).indices(len(self))`.

        raise_if_missing : bool, optional
            If True, `_extract` is asked to raise when a requested series is not
            available on disk. If False, missing series may be replaced with
            default arrays depending on `_extract` behavior.

        Returns
        -------
        object
            Depending on `attribute`:
            - numpy.ndarray (most attributes)
            - list of timesteps for 'frames'
            - reader-like object for 'reader'

        Notes
        -----
        - If the Path is empty (start == stop), returns empty containers with
          appropriate dtype for 'states'.
        - If `attribute == 'states'` and `self._exclude_from >= 0`, states from
          `exclude_from` onward are replaced with '' (empty string).
        - For in-memory mode (`self.in_memory()`), requesting 'reader'/'frames'
          constructs a `MemoryReader` from cached arrays.
        """

        # get start, stop, step
        start, stop, step = slice(start, stop, step).indices(len(self))
        length = (stop - start) // step

        # indices
        if attribute == 'indices':
            return np.arange(start, stop, step)

        # states -> processed exclude from
        exclude_from = -1
        if attribute == 'true_states':
            attribute = 'states'
        elif attribute == 'states' and self._exclude_from >= 0:
            exclude_from = max(0, self._exclude_from - start)

        # attribute in dictionary (fast path: already cached globally)
        if attribute in self.__dict__:
            key = slice(start, stop if stop >= 0 else None, step)
            result = self.__dict__[attribute][key]
            if exclude_from >= 0:
                result = result.copy()
                result[exclude_from:] = ''
                result.flags.writeable = False
            return result

        # memory reader (only if Path is already fully in memory)
        if attribute in ('reader', 'frames') and self.in_memory():
            key = slice(start, stop if stop >= 0 else None, step)
            result = MemoryReader(
                self.positions[key],
                velocities=self.velocities[key],
                dimensions=self.dimensions[key], dt=self.dt)
            if attribute == 'reader':
                return result
            if attribute == 'frames':
                return [frame for frame in result]

        # no files or data (empty slice)
        if start == stop:
            if attribute in ('reader', 'frames'):
                return []
            if attribute == 'states':
                result = np.array([], dtype='<U1')
            else:
                result = np.array([])
            result.flags.writeable = False
            return result

        # find limits (determine first/last file segments touched)
        last = range(start, stop, step)[-1]
        k_first, i_first = self._get_local_index(start)
        k_last, i_last = self._get_local_index(last)

        # just one file (faster: no concatenation)
        if k_first == k_last:
            start = i_first
            stop = i_last + step
            key = slice(start, stop if stop >= 0 else None, step)
            result = self._extract(k_first, attribute, key, raise_if_missing)
            if attribute == 'reader':
                return result
            if attribute == 'frames':
                return [frame.copy() for frame in result]
            if exclude_from >= 0:
                result = result.copy()
                result[exclude_from:] = ''
                result.flags.writeable = False
            return extend_array(result, length)

        # must concatenate across multiple files
        results = []
        start = i_first
        k_step = 1 if step > 0 else -1
        for k in range(k_first, k_last, k_step):
            if start < 0:
                start += self.lengths[k]
                continue
            key = slice(start, None, step)
            result = self._extract(k, attribute, key, raise_if_missing)
            results.append(result)

            # find the next "start" in the next file's local index space
            next_index = start + len(result) * step
            if step > 0:
                start = next_index - self.lengths[k]
            else:
                start = self.lengths[k + k_step] + next_index

        # last segment
        stop = i_last + step
        key = slice(start, stop if stop >= 0 else None, step)
        results.append(
            self._extract(k_last, attribute, key, raise_if_missing))

        # reconstruct final object
        if attribute == 'reader':
            return ChainReader(*results)
        if attribute == 'frames':
            return [frame.copy() for frame in ChainReader(*results)]
        result = np.concatenate(results, axis=0)

        # preserve size and apply exclusion masking for states
        if exclude_from >= 0:
            result[exclude_from:] = ''
        result.flags.writeable = False
        return extend_array(result, length)
