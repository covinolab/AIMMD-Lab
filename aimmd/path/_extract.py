"""
...
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
        """Only from specific file.
        fname for faster computation and customization.
        Does not update cache unless bigger size requested.
        States processing not applied at this level.
        raise_if_missing: if False, give default if missing
        """
        
        if attribute == 'self':
            from . import Path
            result = object.__new__(Path)
            result._fnames = [self._fnames[k]]
            result._first = [self._first[k]]
            result._last = [self._last[k]]
            result._weight = self._weight
            start = 0
            for first, last in zip(self._first[:k], self._last[:k]):
                start += abs(last - first) + 1
            stop = start + result.last[0]  # TODO FIX HERE
            if self._exclude_from >= 0:
                exclude_from = max(0, self._exclude_from - start)
                if exclude_from > len(result):
                    result._exclude_from = -1
                else:
                    result._exclude_from = exclude_from
            else:
                result._exclude_from = -1
            if self._shooting_index < start:
                result._shooting_index = 0
            elif self._shooting_index < stop:
                result._shooting_index = self._shooting_index - start
            else:
                result._shooting_index = stop - start - 1
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
            offset = self.offsets[k - 1] if k else 0
            length = abs(last - start) + 1
            result = np.arange(length)
            if key is not None:
                result = result[key].flatten()
            return offset + result
        
        if attribute in self.__dict__:
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
