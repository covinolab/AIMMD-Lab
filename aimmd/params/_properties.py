"""
...
"""

# external
import numpy as np
import torch
from abc import ABC
from dataclasses import MISSING

# aimmd imports
from .utils import create_default_values_function
from ..core.utils import guess_masses
from ..network.fit import placeholder as placeholder_fit
from ..core.decorators import classproperty

# params' properties
class ParamsProperties(ABC):
    
    @classproperty
    def placeholder(cls):
        """Create a placeholder Params instance that bypasses __init__ and
        __setattr__, and initializes all defaults including mutable lists."""
        from . import Params
        self = object.__new__(Params)
        
        # initialize all dataclass fields with defaults
        for name in cls.__dataclass_fields__:
            field = cls.__dataclass_fields__[name]
            if (not hasattr(self, name) and
                field.default_factory is not MISSING):
                    self.__dict__[name] = field.default_factory()
        
        # assign directly
        self.__dict__['_universe'] = None
        self.__dict__['_default_values_function'] = True
        self.__dict__['states_function'] = lambda x: np.full(len(x), 'R')
        self.__dict__['values_function'] = create_default_values_function(
            self.network, None)
        self.__dict__['fit'] = placeholder_fit
        self.__dict__['topology'] = ''
        
        return self
    
    @property
    def parent(self):
        if self.path.is_file():
            return self.path.parent
        return self.path

    @property
    def sorted_states(self):
        if self.states[0] > self.states[2]:
            return self.states[::-1]
        return self.states

    @property
    def universe(self):
        return self._universe
    
    @property
    def masses(self):
        if self._universe is None:
            return
        return np.maximum(guess_masses(self._universe.atoms), 72.)
    
    @property
    def compute_states_args(self):
        return self.states_function, 'states'

    @property
    def compute_descriptors_args(self):
        if not self.descriptors_function:
            return
        return self.descriptors_function, 'descriptors'

    @property
    def compute_values_args(self):
        if not self.descriptors_function:
            return self.values_function, 'values', 'coordinates'
        return self.values_function, 'values', 'descriptors'
    
    @property
    def pipeline(self):
        if not self.descriptors_function:
            return self.compute_states_args, self.compute_values_args
        return (self.compute_descriptors_args,  # better first
                self.compute_states_args, self.compute_values_args)
