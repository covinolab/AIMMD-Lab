"""
...
"""

# external
from abc import ABC, abstractmethod

# abstract array
class AbstractArray(ABC):

    @abstractmethod
    def _array(self):
        pass
    
    def __getitem__(self, key):
        return self._array()[key]
    
    def __len__(self):
        return len(self._array())

    def __repr__(self):
        return repr(self._array())
    
    def __array__(self, dtype=None):
        array = self._array()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __iter__(self):
        return self._array().__iter__()

    def __getattr__(self, attribute):
        return getattr(self._array(), attribute)
    
    def __eq__(self, other):
        return self._array() == other

    def __ne__(self, other):
        return self._array() != other

    def __lt__(self, other):
        return self._array() < other

    def __le__(self, other):
        return self._array() <= other

    def __gt__(self, other):
        return self._array() > other

    def __ge__(self, other):
        return self._array() >= other

    def __req__(self, other):
        return other == self._array()

    def __rne__(self, other):
        return other != self._array()

    def __rlt__(self, other):
        return other < self._array()

    def __rle__(self, other):
        return other <= self._array()

    def __rgt__(self, other):
        return other > self._array()

    def __rge__(self, other):
        return other >= self._array()

    def __bool__(self):
        raise ValueError(
            "The truth value of an {self.__class__.name} "
            "is ambiguous. Use any() or all().")
    
    def __iadd__(self, other):
        new = self._array() + other
        self[:] = new
        return self

    def __isub__(self, other):
        new = self._array() - other
        self[:] = new
        return self

    def __imul__(self, other):
        new = self._array() * other
        self[:] = new
        return self

    def __itruediv__(self, other):
        new = self._array() / other
        self[:] = new
        return self
    
    def __invert__(self):
        return ~self._array()
