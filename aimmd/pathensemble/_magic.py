"""
...
"""

# external
import numpy as np
import bisect
from abc import ABC
from numbers import Integral
from collections.abc import Iterable

# aimmd imports
from .utils import get_paths
from ..path import Path
from ._properties import PathEnsembleProperties
from ..core.utils import get_local_index

# magic methods for PathEnsemble class
class PathEnsembleMagic(ABC):
    def _get_local_index(self, i, clip=False):
        return get_local_index(i, self.offsets, clip=clip)

    def __bool__(self):
        if not len(self):
            return False
        for path in self._paths:
            if not path.weight:
                return False
        return True

    def __getitem__(self, i):
        if isinstance(i, Integral):
            return self._paths[i]
        from . import PathEnsemble
        result = object.__new__(PathEnsemble)
        result._paths = list(self.paths[i].flatten())
        return result
    
    def __setitem__(self, key, value):
        paths = get_paths(value)
        if not len(paths):
            raise TypeError('no paths found')
        self._paths[key] = paths[0]
    
    def __iter__(self):
        return self._paths.__iter__()
    
    def __len__(self):
        return len(self._paths)
    
    def __repr__(self):
        return f'PathEnsemble with {len(self)} paths'

    def __eq__(self, other):
        return np.array_equal(self.paths, other.paths)
    
    def __add__(self, other):
        from . import PathEnsemble
        if isinstance(other, PathEnsemble):
            paths = self._paths + other._paths
        elif isinstance(other, Path):
            paths = self._paths + [other]
        elif not isinstance(other, Iterable) or len(other):
            raise TypeError(f'Cannot add {other!r} to {self}.')
        else:
            paths = self._paths
        from . import PathEnsemble
        result = object.__new__(PathEnsemble)
        result._paths = paths
        return result
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
    
    def __getattr__(self, attribute):
        """Internal concatenated attributes."""
        if (attribute in ('first', 'last', 'weight', 'cache') or
            attribute.startswith('_')):
            raise AttributeError(f"can't get {attribute!r}")
        return self._get(attribute)
    
    def __setattr__(self, attribute, value):
        # private attributes
        if attribute[0] == '_':
            self.__dict__[attribute] = value
            return

        # property
        path_property = getattr(PathEnsembleProperties, attribute, None)
        if isinstance(path_property, property):
            if path_property.fset is None:
                raise AttributeError(
                    f"can't set aimmd.Path property {attribute!r}")
            # dispatch to the property setter
            path_property.fset(self, value)
            return
    
        # delete
        if value is None:
            for i, path in enumerate(self._paths):
                setattr(path, attribute, None)
            return
        
        # raise
        raise AttributeError(f"can't set {attribute!r} to aimmd.PathEnsemble"
                             ", must do that path-by-path")
    
    def __bool__(self):
        for path in self._paths:
            if len(path):
                return True
        return False
