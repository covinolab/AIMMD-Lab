"""
...
"""

# external
import os
from abc import ABC
from pathlib import PosixPath

# aimmd imports
from ..core.decorators import class_or_instancemethod

# input/output methods of class PathEnsemble
class PathEnsembleIO(ABC):
    def save(self, fname):
        parent = PosixPath(fname).resolve().parent
        with open(fname, 'w') as file:
            file.write('\n'.join(
                [os.path.relpath(fname, parent)
                 for fname in self.fnames]))
    
    @class_or_instancemethod
    def load(self_or_cls, filename,
             find_shooting_indices=False, pipeline=()):
        
        # do we need to create or select an instance of Params?
        from . import PathEnsemble
        instance = PathEnsemble(filename,
            find_shooting_indices, pipeline)
        if isinstance(type(self_or_cls), type):
            return instance
        self += instance
        return self
