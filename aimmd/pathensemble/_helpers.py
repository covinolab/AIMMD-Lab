"""
...
"""

# external
import numpy as np
from abc import ABC
from tqdm import tqdm

# aimmd imports
from .utils import get_paths
from ..core.utils import get_local_index

class PathEnsembleHelpers(ABC):
    def _init(self, *paths, find_shooting_indices=False, pipeline=()):
        
        # process kwargs
        path_kwargs = {}
        if pipeline:
            path_kwargs['pipeline'] = tuple(pipeline)
        if find_shooting_indices:
            path_kwargs['shooting_index'] = 'find'
        
        # get it
        self._paths = get_paths(paths, initialize=True, **path_kwargs)
    
    def _get(self, attribute, where='internal', verbose=False):
        return [path._get(attribute, *path._range(where))
                for path in tqdm(self._paths, total=len(self),
                                 position=0, disable=not verbose)]
    
    def _get_local_index(self, i, clip=False):
        return get_local_index(i, self.offsets, clip=clip)
