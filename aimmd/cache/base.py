"""
...
"""

# external
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable

# abstract cache
class AbstractCache(ABC):
    max_size = None
    
    def __init__(self, timeout=10.):
        self._cache = OrderedDict()
        self.total_size = 0
    
    def __len__(self):
        return len(self._cache)
    
    @abstractmethod
    def _open(self, fname):
        pass
    
    def _close(self, instance):
        pass

    def _extend(self, instance, min_length):
        return instance
    
    def get(self, fname, min_length=0, extend=False):
        instance = self._cache.get(fname, None)
        if instance is None or len(instance) < min_length:
            new = self.load(fname)
            if new is not None:
                instance = new
        if instance is not None and extend:
            instance = self._extend(instance, min_length)
        return instance

    def open(self, fname):
        try:
            return self._open(fname)
        except:
            return None
    
    def load(self, fname):
        """Updates cache with fname"""
        instance = self.open(fname)
        if instance is None:
            return
        self.remove(fname)
        size = sys.getsizeof(instance)
        while self.total_size + size >= self.max_size:
            self.remove()
        self._cache[fname] = instance
        self.total_size += size
        return instance
    
    def reload(self):
        return [self.load(fname) for fname in self._cache]
    
    def pop(self, fname=None):
        """Remove from cache if there, opens if not there.
        Fallback: returns None."""
        if isinstance(fname, str):
            instance = self._cache.pop(fname, None)
            if instance is not None:
                self.total_size -= sys.getsizeof(instance)
            else:
                instance = self.open(fname)
            return instance
        if fname is None:
            if len(self):
                instance = self._cache.popitem(last=False)[1]
                self.total_size -= sys.getsizeof(instance)
                return instance
            return
        return self.pop(str(fname))
    
    def remove(self, fname=None):
        try:
            instance = self.pop(fname)
            self._close(instance)
            del instance
        finally:
            return

    def clear(self):
        while self._cache:
            self.remove()
