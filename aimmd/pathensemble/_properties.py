"""
...
"""

# external
import numpy as np
from abc import ABC
from numbers import Integral

# aimmd imports
from .utils import get_paths
from ..core.base import AbstractArray

# path properties class
class PathProperties(AbstractArray):
    def __init__(self, pathensemble, attribute, dtype=float):
        self.pathensemble = pathensemble
        self.attribute = attribute
        self.dtype = dtype
    
    def __getitem__(self, key):
        if isinstance(key, Integral):
            path = self.pathensemble._paths[key]
            return getattr(path, self.attribute)
        return np.array([
            getattr(self.pathensemble._paths[i], self.attribute)
            for i in np.arange(len(self.pathensemble))[key]],
                        dtype=self.dtype)
    
    def _array(self):
        return np.array([
            getattr(path, self.attribute) for path in
            self.pathensemble._paths])
    
    def __setitem__(self, key, values):
        if isinstance(key, Integral):
            path = self.pathensemble._paths[key]
            setattr(path, self.attribute, values)
            return
        
        key = np.arange(len(self.pathensemble))[key]
        values = np.asarray(values, dtype=self.dtype)
        
        # broadcast scalar or single-element
        if values.ndim == 0:
            values = np.full(key.shape, values)
        elif values.shape[0] == 1 and key.shape[0] != 1:
            values = np.full(key.shape, values.item())
        
        # assign
        for i, value in zip(key, values):
            path = self.pathensemble._paths[i]
            setattr(path, self.attribute, value)

# path ensemble properties
class PathEnsembleProperties(ABC):
    @property
    def path(self):
        # last ok path
        for path in self._paths[::-1]:
            if path._weight and path._exclude_from < 0:
                return path
    @property
    def fname(self):
        if path := self.path:
            return path.fname
        return ''
    @property
    def n_paths(self):
        return len(self)
    @property
    def paths(self):
        return np.fromiter(self._paths, dtype=object)
    @paths.setter
    def paths(self, paths):
        self._paths = get_paths(paths)
    @property
    def offsets(self):
        return np.cumsum([len(path) for path in self.paths])
    @property
    def weights(self):
        return PathProperties(self, 'weight', float)
    @property
    def accepted(self):
        return PathProperties(self, 'accepted', bool)
    @property
    def exclude_from(self):
        return PathProperties(self, 'exclude_from', int)
    @property
    def true_states(self):
        return self._get('true_states')
    @property
    def shooting_indices(self):
        return PathProperties(self, 'shooting_index', int)
    @property
    def fnames(self):
        if not len(self._paths):
            return np.array([], dtype=str)
        return np.concatenate([path._fnames for path in self._paths])
    @property
    def n_files(self):
        return sum([len(path.files) for path in self._paths])
    @weights.setter
    def weights(self, weights):
        PathProperties(self, 'weight', float)[:] = weights
    @accepted.setter
    def accepted(self, accepted):
        exclude_from = - np.asarray(accepted).astype(int)
        PathProperties(self, 'exclude_from', int)[:] = exclude_from
    @exclude_from.setter
    def exclude_from(self, exclude_from):
        PathProperties(self, 'exclude_from', int)[:] = exclude_from
    @shooting_indices.setter
    def shooting_indices(self, shooting_indices):
        PathProperties(self, 'shooting_index', int)[:] = shooting_indices
    @property
    def lengths(self):
        return np.array([len(path) for path in self._paths])
    @property
    def n_frames(self):
        n_frames = []
        for path in self._paths:
            length = len(path)
            if length <= 1:
                n_frames.append(length)
            else:
                states = path.type
                if states[0] != states[1]:
                    length -= 1
                if states[1] != states[2]:
                    length -= 1
                n_frames.append(length)
        return np.array(n_frames, dtype=int)
