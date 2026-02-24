"""
...
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
        
        # attribute in dictionary
        if attribute in self.__dict__:
            key = slice(start, stop if stop >= 0 else None, step)
            result = self.__dict__[attribute][key]
            if exclude_from >= 0:
                result = result.copy()
                result[exclude_from:] = ''
                result.flags.writeable = False 
            return result

        # memory reader
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
        
        # no files or data
        if start == stop:
            if attribute in ('reader', 'frames'):
                return []
            if attribute == 'states':
                result = np.array([], dtype='<U1')
            else:
                result = np.array([])
            result.flags.writeable = False
            return result
        
        # find limits
        last = range(start, stop, step)[-1]
        k_first, i_first = self._get_local_index(start)
        k_last, i_last = self._get_local_index(last)
        
        # just one file (faster)
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
        
        # must concatenate
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
            
            # find the next "start"
            next_index = start + len(result) * step
            if step > 0:
                start = next_index - self.lengths[k]
            else:
                start = self.lengths[k + k_step] + next_index
        
        # last
        stop = i_last + step
        key = slice(start, stop if stop >= 0 else None, step)
        results.append(
            self._extract(k_last, attribute, key, raise_if_missing))
        
        # reconstruct
        if attribute == 'reader':
            return ChainReader(*results)
        if attribute == 'frames':
            return [frame.copy() for frame in ChainReader(*results)]
        result = np.concatenate(results, axis=0)

        # preserve size
        if exclude_from >= 0:
            result[exclude_from:] = ''
        result.flags.writeable = False
        return extend_array(result, length)
