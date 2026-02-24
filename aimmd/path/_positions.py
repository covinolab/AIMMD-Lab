"""
...
"""

# external
import numpy as np
from abc import ABC

# position methods for path class
class PathPositions(ABC):
    
    def initial(self, attribute):
        return self._position(0, attribute)

    def final(self, attribute):
        return self._position(-1, attribute)
    
    def middle(self, attribute):
        if attribute == 'indices':
            return min(len(self), 1)
        return self._position(self.middle('indices'), attribute)
    
    def shooting(self, attribute):
        return self._position(self._shooting_index, attribute)

    def all(self, attribute):
        return self._get(attribute)
    
    def backward(self, attribute):
        return self._get(attribute, *self._range('backward'))
    
    def forward(self, attribute):
        return self._get(attribute, *self._range('forward'))
    
    def internal(self, attribute):
        return self._get(attribute, *self._range('internal'))

    def min(self, attribute, source='values'):
        return self._extreme(attribute, np.argmin, 'internal', source)

    def max(self, attribute, source='values'):
        return self._extreme(attribute, np.argmax, 'internal', source)

    def min_backward(self, attribute, source='values'):
        return self._extreme(attribute, np.argmin, 'backward', source)

    def max_backward(self, attribute, source='values'):
        return self._extreme(attribute, np.argmax, 'backward', source)

    def min_forward(self, attribute, source='values'):
        return self._extreme(attribute, np.argmin, 'forward', source)

    def max_forward(self, attribute, source='values'):
        return self._extreme(attribute, np.argmax, 'forward', source)
