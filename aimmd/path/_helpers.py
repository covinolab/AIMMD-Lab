"""
...
"""

# external imports
import os
import time
import numpy as np
from abc import ABC
from math import inf, nan
from collections.abc import Iterable

# aimmd imports
from ..core.utils import get_local_index

# Path's helper methods
class PathHelpers(ABC):

    def _init(self, fnames=[],
              start=None,
              stop=None,
              remove_overlapping_frames=False,
              pipeline=(),
              weight=1.0,
              exclude_from=-1,
              shooting_index=0):
        """exclude: frames after which bad, >= 0: trajectory rejected
        shooting_index: 'find' makes you find over time
        """
        
        if isinstance(fnames, Iterable):
            
            # initialize
            self._fnames = []
            self._first = []
            self._last = []
    
            # extend: process start, stop
            if start is None:
                start = 0
            else:
                start = max(0, start)
            if stop is None:
                stop = inf
            nframes = stop - start
            self.extend(fnames, nframes, start,
                        remove_overlapping_frames, pipeline)

        else:  # just update with the path
            from . import Path
            path = fnames
            if not isinstance(path, Path):
                raise TypeError(f'input to Path is either a string, '
                                f'a list of strings, or another path, '
                                f'got {path!r}')
            # just update
            self.__dict__.update(path[start:stop].__dict__)
            for args in pipeline:
                path.compute(*args)
        
        # assign attributes
        self.weight = weight
        self.exclude_from = exclude_from
        self.shooting_index = shooting_index
    
    def _get_local_loc(self, i, clip=False):
        k, i = self._get_local_index(i, clip=clip)
        start, last = self._first[k], self._last[k]
        step = 1 if start <= last else -1
        return k, start + i * step
    
    def _get_local_index(self, i, clip=False):
        return get_local_index(i, self.offsets, clip=clip)
    
    def _extreme(self, attribute, operation, where, source='values'):
        start, stop = self._range(where)
        values = self._get(source, start, stop)
        if attribute == source:
            series = values
        series = self._get(attribute, start, stop)
        return series[operation(values)]

    def _range(self, where):
        start = None
        stop = None
        
        # all
        if where == 'all':
            return start, stop

        # internal
        if where == 'internal':
            if not (abcd := self.type):
                return start, len(self)
            if abcd[0] != abcd[1]:
                start = 1
            if abcd[1] != abcd[2]:
                stop = len(self) - 1
            return start, stop
        
        # backward as a modification of "internal"
        if where == 'backward':
            shooting_index = self._shooting_index
            start, stop = self._range('internal')
            stop = min(stop or len(self), shooting_index + 1)
            return start, stop
        
        # forward as a modification of "internal"
        if where == 'forward':
            shooting_index = self._shooting_index
            start, stop = self._range('internal')
            start = max(start or 0, shooting_index)
            return start, stop
        
        raise ValueError(
            f'"where" must be one of ("all", "internal", "backward", '
            f'"forward"), got {where!r} instead')

    def _position(self, i, attribute='indices'):
        if attribute == 'indices':
            return i
        if attribute == 'locs':
            return self.locs[i]
        if attribute in ('reader', 'frames'):
            return self[i]
        if (attribute == 'states' and
            self._exclude_from >= 0 and
            i >= self._exclude_from):
            return '.'
        if attribute in self.__dict__:
            result = self.__dict__[attribute][i]
            return result
        if i == -1:
            k = -1
            i = self.lengths[-1] - 1
        else:
            if i < 0:
                i += len(self)
            if self.n_files == 1:
                k = 0
            else:
                k, i = self._get_local_index(i)
        if attribute == 'filenames':
            return self._fnames[k]
        try:
            return self._extract(k, attribute)[i]
        except:
            if attribute == 'states':
                return ''
            return nan
