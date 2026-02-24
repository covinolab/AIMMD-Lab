"""
...
"""

# external
import numpy as np
import bisect
from numbers import Integral

# chainreader class
class ChainReader:
    
    def __init__(self, *readers):
        self.readers = readers
        self._keys = iter(range(self.lengths.sum()))
        self._length = self.lengths.sum()

    @property
    def lengths(self):
        return np.array([len(reader) for reader in self.readers], dtype=int)

    @property
    def n_atoms(self):
        if self.readers:
            return self.readers[0].trajectory.n_atoms
    
    def __len__(self):
        return self._length

    def __iter__(self):
        return self

    def __next__(self):
        try:
            i = next(self._keys)
            k, i = self._get_local_index(i)
            return self.readers[k][i]
        except StopIteration:
            # reset keys
            self._keys = iter(range(self.lengths.sum()))
            self._length = self.lengths.sum()
            raise StopIteration
    
    @property
    def offsets(self):
        return np.cumsum(self.lengths)

    def _get_local_index(self, i):
        offsets = self.offsets
        k = bisect.bisect_right(offsets, i)
        if k >= len(offsets):
            raise IndexError(i)
        if k:
            return k, i - offsets[k - 1]
        return k, i
    
    def __getitem__(self, key):
        if isinstance(key, Integral):
            k, i = self._get_local_index(key)
            return self.readers[k][i]
        result = object.__new__(ChainReader)
        result.readers = self.readers
        keys = np.arange(len(self))[key]
        result._keys = iter(keys)
        result._length = len(keys)
        return result
