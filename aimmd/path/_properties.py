"""
...
"""

# external imports
import numpy as np
from abc import ABC
from numbers import Integral

# aimmd imports
from .._config import MDA_CACHE

# Path's properties
class PathProperties(ABC):

    @property
    def fnames(self):
        return np.array(self._fnames, dtype=str)

    @property
    def first(self):
        return np.array(self._first, dtype=int)

    @property
    def last(self):
        return np.array(self._last, dtype=int)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = float(weight)
    
    @property
    def exclude_from(self):
        return self._exclude_from
    
    @exclude_from.setter
    def exclude_from(self, exclude_from):
        self._exclude_from = min(max(-1, int(exclude_from)), len(self))
    
    @property
    def shooting_index(self):
        return self._shooting_index
    
    @shooting_index.setter
    def shooting_index(self, shooting_index):
        if not isinstance(shooting_index, Integral):
            self._shooting_index = self.find_shooting_index()
            return
        self._shooting_index = min(max(0, shooting_index), len(self) - 1)
    
    @property
    def fname(self):
        """Last ("active")"""
        if not self.n_files:
            return ''
        return self._fnames[-1]
    
    @property
    def lengths(self):
        return np.abs(self.first - self.last) + 1
    
    @property
    def offsets(self):
        return np.cumsum(self.lengths)
    
    @property
    def dt(self):
        if len(self) <= 1:
            return 1.
        if 'times' in self.__dict__:
            return np.diff(self.__dict__['times'][:2])[0]
        return self.middle('times') - self.initial('times')
    
    @property
    def indices(self):
        return np.arange(len(self))
    
    @property
    def locs(self):
        if not len(self):
            return np.array([], dtype=int)
        result = []
        for start, last in zip(self._first, self._last):
            step = 1 if start <= last else -1
            stop = last + step
            result.append(range(start, stop, step))
        result = np.concatenate(result, dtype=int)
        result.flags.writeable = False
        return result
    
    @property
    def filenames(self):
        return np.repeat(self._fnames, self.lengths).astype(str)
    @property
    def accepted(self):
        return self._exclude_from < 0
    
    @accepted.setter
    def accepted(self, accepted):
        if accepted:
            self._exclude_from = -1
        elif self._exclude_from >= 0:
            self._exclude_from = 0
    
    @property
    def type(self):
        if not self.accepted:
            return '....'
        n_files = self.n_files
        if not n_files:
            return '....'
        if n_files == 1:
            states = self.states
            if len(states) > 1:
                return ((states[0] or '.') +
                        (states[1] or '.') +
                        (states[-1] or '.') +
                        (states[self._shooting_index] or '.'))
            return (states[0] or '.') * 4
        try:
            return ((self.initial('states') or '.') +
                    (self.middle('states') or '.') +
                    (self.final('states') or '.') +
                    (self.shooting('states') or '.'))
        except:
            return '....'
    
    @property
    def n_atoms(self):
        if self.n_files:
            return MDA_CACHE.get(self._fnames[0])[0].n_atoms
        return 0
    
    @property
    def n_files(self):
        return len(self._fnames)
    
    @property
    def n_frames(self):
        return len(self)
