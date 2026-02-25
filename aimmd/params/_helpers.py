"""
...
"""

# external
import numpy as np
import torch
import inspect
from abc import ABC
from types import MethodType as Method
from typing import List, Callable
from pathlib import PosixPath
from MDAnalysis import Universe

# aimmd imports
from .utils import update_source, create_default_values_function
from ..path import Path
from ..pathensemble import PathEnsemble
from ..network.rescalable import Rescalable as RescalableNetwork

# params' helpers
class ParamsHelpers(ABC):
    def _init(self, *args, **kwargs):
        """
        kwargs: parameters field -> value.
        If len(args) > 0: first element must be a filename.
        Then load parameters in filename.
        Parameters not in filename and kwargs are given the default value.
        
        If filename in kwargs, then treat it as a params file.
        
        Returns
        -------
        aimmd.params.Params
        """
        
        # determine whether loading from file
        try:
            filename = PosixPath(args[0]).resolve()
            if not filename.exists():
                raise FileNotFoundError
            args = args[1:]
        except (IndexError, TypeError, FileNotFoundError):
            filename = 'params.py'
        
        # create separate instance
        from . import Params
        instance = Params.load(filename, *args, **kwargs).__dict__
        
        # assign instance's fields to self (in right order)
        self.__dict__.update(instance)
    
    def _setattr(self, name, value, process_and_check=True):
        """Helper function."""
        
        # assign fields
        if name in self.__dataclass_fields__:
            expected_type = self.__dataclass_fields__[name].type
            
            # functions
            if expected_type is Callable:
                if value is None and name in (
                    'descriptors_function', 'descriptor_transform',
                    'values_function', 'toy_mdrun'):
                    pass
                
                elif not callable(value):
                    raise TypeError(f"'values_function' must be callable, "
                                    f'got {type(value).__name__}: {value}')

                elif getattr(value, '__name__', None) == "<lambda>":
                    raise TypeError(
                        "lambda functions don't work in params (for now)")
                
                elif isinstance(value, Method):
                    value = value.__func__
                    update_source(value, name)
                
                else:
                    update_source(value, name)

                fit_args = ('params', 'pathensemble',
                            'key', 'verbose', 'worker')
                if name == 'fit':
                    args = tuple(inspect.signature(value).parameters)
                    if args != fit_args:
                        raise TypeError(f"'fit' must have args {fit_args} "
                                        f"in order, got {args} instead")
            
            elif name == 'states':
                value = str(value).upper().replace(' ','')
                if not (value.isalpha() and len(set(value)) == 3): 
                    raise TypeError(
                        f'{name!r} must be three distinct upper alpha chars')

            elif name in ('extra_bins', 'free_overriding_states'):
                value = str(value).replace(' ','')
                if value.lower() != 'all':
                    value = value.upper()
                if value and not value.isalpha():
                    raise TypeError(
                        f'{name!r} must be distinct upper alpha chars') 
            
            # topology (update universe)
            elif name == 'topology':
                try:
                    universe = Universe(value)
                except:
                    universe = None
                self.__dict__['_universe'] = universe
            
            # initial paths (converted in real-time)
            elif name == 'initial_paths':
                if process_and_check:  # otw bug in reloading paths
                    self.__dict__['_reload_initial_paths'] = \
                        not isinstance(value, (Path, PathEnsemble))
                initial_paths = PathEnsemble(value)
                if value and not initial_paths:
                    raise TypeError(
                        f'could not get any initial paths from {value!r}')
                value = initial_paths
            
            # engine
            elif name == 'engine':
                value = value.lower()
                if value not in ['gromacs', 'toy']:
                    raise TypeError(f'{name} must be either "gromacs" or '
                                    f'"toy", got {value}')
            
            # network
            elif name == 'network':
                for attribute in ('forward', 'state_dict',
                                  'load_state_dict', 'parameters'):
                    if not hasattr(value, attribute) or not callable(
                        getattr(value, attribute)):
                        raise TypeError(
                            f'{name} must have method "{attribute}"')
                update_source(value, name)
            
            # chain type
            elif name == 'chain_type':
                value = value.lower()
                if value.startswith('tp'):
                    values = 'tps'
                elif value.startswith('rf'):
                    values = 'rfps'
                else:
                    raise TypeError(f'{name} must be either "tps" or '
                                    f'"rfps", got {value}')
            
            # list of strings
            elif expected_type is List[str]:
                if type(value) is not list:
                    raise TypeError(f'{name} must be list of strings, '
                                    f'got {type(value).__name__}')
                elif np.any([type(element) is not str for element in value]):
                    raise TypeError(f'{name} must be list of strings, '
                                    f'at least one of its elements is not')
            
            # path
            elif name == 'path':
                if not PosixPath(value).exists():
                    raise TypeError(f'Source path {value} does not exist.')
            
            # all the rest
            elif not isinstance(value, expected_type):
                raise TypeError(f'{name} must be {expected_type}, '
                                f'got {type(value).__name__}')

        elif not name.startswith('_'):
            raise TypeError(f'{name!r} is not a valid Params dataclass field '
                            f'or a hidden parameter')
        
        # assign
        backup = self.__dict__.get(name, 'absent')
        self.__dict__[name] = value

        # special checks (after assignment)
        if process_and_check:
            try:
                self._process_and_check([name])
            except Exception as exception:  # go back in case of error
                if not isinstance(backup, str) or backup != 'absent':
                    self.__dict__[name] = backup
                raise TypeError(f'can\'t update {name!r} with {value!r}')
    
    def _process_and_check(self, fields=[]):
        """Check a posteriori (after assignment is done)"""
        
        # check network
        if self.rescale_committor:
            if not isinstance(self.network, RescalableNetwork):
                raise TypeError(f"params' network {self.network} "
                                "is not aimmd.network.Rescalable")
        
        # check chain type
        if self.chain_type == 'tps' and self.selection_pool_size > 1:
            raise TypeError(f"when chain_type is 'tps', "
                            f"selection pool size must be 1")

        # check nbins
        if 'nbins' in fields or 'extra_bins' in fields:
            if self.nbins > 0:
                pass
            elif self.nbins == 0 and (
                self.extra_bins == 'all' or (
                self.states[+0] in self.extra_bins and
                self.states[-1] in self.extra_bins)):
                pass
            else:
                states = f'{self.states[+0]}{self.states[-1]}'
                raise TypeError(f'`nbins` must be > 0, or >= 0 when '
                                f'`extra_bins in ({states!r}, \'all\')`')

        # redefine values function
        default_values_function = self._default_values_function
        if self.values_function is None or (
            'network' in fields and default_values_function):
            default_values_function = True
            self.__dict__['values_function'] = create_default_values_function(
                self.network, self.descriptor_transform)
            check_values = True
        
        # no need: there is no default values function
        elif 'values_function' in fields:
            default_values_function = False

        # reassign default values function
        self.__dict__['_default_values_function'] = default_values_function
        
        # initial paths-related checks
        initial_paths = getattr(self, 'initial_paths', None)
        if not initial_paths:
            return
        check_paths = 'initial_paths' in fields
        check_states = ('states' in fields or
                        'states_function' in fields or
                        check_paths)
        check_descrs = self.descriptors_function is not None and (
                        'descriptors_function' in fields or
                        check_paths)
        if self.descriptors_function is None:
            check_values = ('values_function' in fields or
                            check_paths)
        else:
            check_values = ('values_function' in fields or
                            'descriptors_function' in fields or
                            check_paths)
        
        # go through paths
        for i, path in enumerate(initial_paths):
            if check_paths:
                try:
                    n_atoms = path.reader.n_atoms
                except:
                    n_atoms = path.reader.trajectory.n_atoms
                universe = self._universe
                if universe and n_atoms != len(universe.atoms):
                    raise TypeError(
                        f'{i}-th initial path {path.fname!r} has '
                        f'{n_atoms} atoms, while the topology file '
                        f'{self.topology!r} has {len(universe.atoms)} atoms')
            
            if check_states:
                if self._reload_initial_paths:
                    path = Path(path.fname)
                    path.states = path.compute(self.states_function)
                    transition_found = False
                    for path in (split := path.split()):
                        if path.type[:3] in (self.states, self.states[::-1]):
                            transition_found = True
                            break
                    if not transition_found:
                        types = ", ".join([t[:3] for t in split.types()])
                        raise TypeError(f'the initial trajectory {path.fname!r} '
                            f'has no {self.states!r} transitions '
                            f'(path types: {types})')
                elif not path.type[:3] in (self.states, self.states[::-1]):
                    raise TypeError(f'the initial trajectory {path.fname!r} '
                            f'is not an {self.states!r} transition '
                            f'(path type: {path.type})')
            
            if check_descrs:
                path.descriptors = path.compute(self.descriptors_function)
            
            if check_values:
                if self.descriptors_function is not None:
                    source = path.descriptors[:1]
                else:
                    source = path.coordinates[:1]
                assert self.values_function(source).shape == (1,)
            
            # reassign path
            self.initial_paths[i] = path
