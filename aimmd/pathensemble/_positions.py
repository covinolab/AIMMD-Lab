"""
...
"""

# external
import numpy as np
from abc import ABC

# aimmd imports
from ..core.utils import memory_reader_from_timesteps

# position methods for path ensemble class
class PathEnsemblePositions(ABC):
    def initial(self, attribute):
        result = [path.initial(attribute) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def shooting(self, attribute):
        result = [path.shooting(attribute) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def final(self, attribute):
        result = [path.final(attribute) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def middle(self, attribute):
        result = [path.middle(attribute) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def min(self, attribute, source='values'):
        result = [path.min(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def max(self, attribute, source='values'):
        result = [path.max(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def min_backward(self, attribute, source='values'):
        result = [path.min_backward(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def max_backward(self, attribute, source='values'):
        result = [path.max_backward(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def min_forward(self, attribute, source='values'):
        result = [path.min_forward(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def max_forward(self, attribute, source='values'):
        result = [path.max_forward(attribute, source) for path in self._paths]
        if attribute == 'frames':
            return result
        if attribute == 'reader':
            return memory_reader_from_timesteps(result)
        return np.array(result)
    def backward(self, attribute):
        return [path.backward(attribute) for path in self._paths]
    def forward(self, attribute):
        return [path.forward(attribute) for path in self._paths]
    def all(self, attribute):
        return [path.all(attribute) for path in self._paths]
    def internal(self, attribute):
        return [path.internal(attribute) for path in self._paths]
