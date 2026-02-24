"""
...
"""

# external
import numpy as np
from abc import ABC
from tqdm import tqdm

# aimmd imports
from .utils import match_patterns
from ..path import Path
from ._helpers import PathEnsembleHelpers
from ..core.utils import merge_ranges

# methods for PathEnsemble class
class PathEnsembleMethods(ABC):

    def append(self, path):
        self._paths.append(path)
    
    def extend(self, paths):
        self._paths.extend(paths)

    def are_complete(self, states='ARB'):
        return np.array([path.is_complete(states) for path in self])

    def are_transitions(self, states='ARB'):
        return np.array([path.is_transition(states) for path in self])

    def are_excursions(self, states='ARB'):
        return np.array([path.is_excursion(states) for path in self])

    def are_internal(self, states='ARB'):
        return np.array([path.is_internal(states) for path in self])
    
    def extract(self, *types):
        """Based on types"""
        if not len(types):
            return PathEnsemble()
        return self[self.types(*types)]
    
    def pop(self, i=None):
        return self._paths.pop(i if i is not None else -1)
    
    def remove(self, path):
        self._paths.remove(path)
    
    def index(self, path):
        return self._paths.index(path)

    def frame(self, i):
        if i < 0:
            i += self.n_frames
        k, i = self._get_local_index(i)
        return self.paths[k][i]
    
    def types(self, *patterns):
        types = np.array([path.type for path in self._paths])
        if not len(patterns):
            return types
        return match_patterns(types, *patterns)
    
    def to_memory(self):
        for path in self._paths:
            path.to_memory()
    
    def from_files(self):
        for path in self._paths:
            path.from_files()
    
    def shooting_results(self, states='ARB', sweep_size=0):
        
        # find size
        if sweep_size <= 0:
            sweep_size = len(self)
        
        # initialize results
        results = np.zeros((sweep_size, 2))
        
        # populate results, return
        for i, path in enumerate(self._paths):
            results[i % sweep_size] += path.shooting_result(states)
        return results
    
    def copy(self):
        copied_paths = [path.copy() for path in self._paths]
        from . import PathEnsemble
        result = PathEnsemble()
        result._paths = copied_paths
        return result

    def split(self, verbose=False):
        split_paths = []
        for path in tqdm(self._paths,
                         disable=not verbose, position=0):
            split_paths.extend(path.split()._paths)
        from . import PathEnsemble
        result = PathEnsemble()
        result._paths = split_paths
        return result

    def in_memory(self, attribute='reader'):
        return np.array([path.in_memory(attribute) for path in self._paths])
    
    def merge(self):
        """
        Attention!
        Only first exclude_from inherited.
        Makes backward paths go forward.
        Also info in memory lost.
        """
        ranges = {}
        for path in self._paths:
            for fname, first, last in zip(
                path._fnames, path._first, path._last):
                if fname not in ranges:
                    ranges[fname] = []
                if first <= last:
                    start = first
                    stop = last + 1
                else:
                    start = last
                    stop = first + 1
                ranges[fname].append((start, stop))
        fnames = []
        first = []
        last = []
        for fname, fname_ranges in ranges.items():
            start, stop = np.array(merge_ranges(fname_ranges)).T
            fnames.extend([fname] * len(start))
            first.extend(start)
            last.extend(stop - 1)  # they are all growing
        if not self.accepted.all():
            # could be better, but it does not make sense
            exclude_from = 0
        else:
            exclude_from = -1
        result = object.__new__(Path)
        result._fnames = fnames
        result._first = first
        result._last = last
        result._exclude_from = exclude_from
        if len(self):
            result._shooting_index = self._paths[0]._shooting_index
        else:
            result._shooting_index = 0
        return result
    
    def compute(self, *args, **kwargs):
        """Applies compute on merged path. TODO borrow documentation
        """
        return self.merge().compute(*args, **kwargs)

    def sample(self, n_samples, state='internal'):
        from ..path import Path
        result = Path()
        if not n_samples or not len(self):
            return result

        # one element per index
        paths = []
        indices = []
        for path in self._paths:
            if state == 'internal':
                this = path.internal('indices')
            else:
                this = np.flatnonzero(path.states == state)
            indices.extend(this)
            paths.extend([path] * len(this))
        if not indices:
            return result
        fnames = []
        first = []
        last = []
        for i in np.random.choice(len(indices), n_samples):
            path, i = paths[i], indices[i]
            k, i = path._get_local_loc(i)
            fnames.append(path._fnames[k])
            first.append(i)
            last.append(i)
        result._fnames = fnames
        result._first = first
        result._last = last
        return result

    get = PathEnsembleHelpers._get
