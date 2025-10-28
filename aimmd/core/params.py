'''
AIMMD parameters management / defaults.
'''

import os
import sys
import torch
import types
import numpy as np
import shutil
import mdtraj as md
import MDAnalysis as mda
from .utils import (absolute_path,
                    class_or_instancemethod, fit,
                    PlaceholderNetwork, execute_command)
from typing import List, Callable
from pathlib import Path, PosixPath
from dill.source import getsource
from dataclasses import dataclass, field, MISSING
from MDAnalysis.coordinates.memory import MemoryReader

from pathlib import Path

def getsourcefile(obj):
    """On current working directory."""
    cwd = Path('.').resolve()
    if isinstance(obj, type):
        result = None
        for name in dir(obj):
            method = getattr(obj, name)
            try:
                if absolute_path(method.__code__.co_filename).parent == cwd:
                    result = method.__code__.co_filename
            except:
                pass
        if not result:
            return str(absolute_path(f'{obj.__module__.split("/")[-1]}.py'))
        return result
    return obj.__code__.co_filename

# find executables
PYTHON = sys.executable
GMX = shutil.which('gmx') or shutil.which('gmx_mpi')
if GMX is None:
    raise EnvironmentError(
        'GROMACS executable not found in PATH. Please install GROMACS and '
        'ensure \'gmx\' or \'gmx_mpi\' is accessible in your environment.'
    )

# all data in a class
@dataclass
class Params:
    """Collects all parameters file for an AIMMD run and analysis.
    
    All fields have a detailed descriptions, most of them come with a default.
    However, "states_function", "descriptors_function", "values_function", and
    "initial_paths" must be specified during initialization.
    
    Usage
    -----
    >>> import aimmd
    >>>
    >>> # minimal initalization without parameters file
    >>> params = aimmd.Params(states_function=states_function,
                              descriptors_function=descriptors_function,
                              values_function=values_function,
                              initial_paths=initial_paths)
    >>> with open('params.py', 'w') as file:
    ...    print(params)  # params.py created on-the-spot
    ...    file.write(f'{params}')
    >>>
    >>> # initialization with parameters file "params.py"...
    >>> # (this allows for more complex functions definitions)
    >>> params = aimmd.Params("params.py")
    >>> params = aimmd.Params.load("params.py")
    >>> # the two methods above are equivalent
    >>>
    >>> # initialization where some "params.py" parameters are oversubscribed
    >>> params = aimmd.Params("params.py", initial_paths=initial_paths)
    """
    
    # user-defined functions for PathEnsemble objects
    
    states_function : Callable = field(
        metadata={'description':
"""From MDanalysis trajectory to array of states."""
                 })
    
    descriptors_function : Callable = field(
        metadata={'description':
"""From MDanalysis trajectory to array of descriptors (used by the model)."""
                 })
    
    values_function : Callable = field(
        metadata={'description':
"""From array of descriptors to their corresponding (logit committor)
values."""
                 })
    
    # initial paths
    
    initial_paths: list = field(
        metadata={'description':
"""List of trajectory filenames or MDAnalysis trajectories. They must
contain transitions (checked automatically). Will replace filenames by
MDAnalysis trajectories."""
        })
    
    # system's name
    
    name: str = field(
        default='AIMMD',
        metadata={'description':
"""System's name (will be used for creating slurm's jobs)."""
        })
    
    # engine configuration
    
    topology : str = field(
        default='run.gro',
        metadata={'description':
"""System's topology (gro file)."""
                 })
    
    trajectory_extension : str = field(
        default='.xtc',
        metadata={'description':
"""Use `xtc` for compressed data, `trr` for full-precision data and also
saving velocities along with the positions."""
                 })
    
    engine : str = field(
        default='gromacs',
        metadata={'description':
"""Either "gromacs" or "toy" engine."""
                 })
    
    # gromacs engine
    
    gmx_init_mdp : str = field(
        default='randomvelocities.mdp',
        metadata={'description':
"""Used for initializing the shooting point's random velocities.
Attention! Velocities will be extracted from trr files."""
                 })
    
    gmx_run_mdp : str = field(
        default='run.mdp',
        metadata={'description':
"""Used for initializing production runs (shooting / free simulations).
Attention! It must be consistent with "trajectory_extension"."""
                 })
    
    gmx_grompp : str = field(
        default=f'{GMX} grompp -maxwarn 1',
        metadata={'description':
"""Gromacs grompp and call options. Change to include index files. Allows to
produce `tpr` files. Attention! "coordinates", "restraints", "output", and
"nobackup" flags are added automatically, do not include them here."""
                 })
    
    gmx_mdrun : str = field(
        default=f'{GMX} mdrun -v -maxh 4',
        metadata={'description':
"""Gromacs mdrun options. Change to optimize performance. Attention! "deffnm",
"nobackup", and "noappend" flags (when required) are added automatically,
do not include them here."""
                 })
    
    gmx_eneconv : str = field(
        default=f'printf "c\nc\n" | {GMX} -nobackup eneconv -settime',
        metadata={'description':
"""Command to merge Gromacs "edr" energy files of the backward and forward
trajectory segments of a two-way shooting simulations. Leave emtpy in case
you do not want to save energy files."""
                 })
    
    # toy engine
    
    toy_mdrun : str = field(
        default=f'{PYTHON} integrator.py',
        metadata={'description':
"""Command to run a custom toy integrator. Attention! It must support the
"noappend" flag, and produce trajectory files with the same naming convention
as Gromacs."""
                 })
    
    # other engines will be defined here
    # [...]
    
    # neural network
    network : torch.nn.Module = field(
        default=PlaceholderNetwork(),
        metadata={'description':
"""Neural network model (used for logit committor estimates in AIMMD).
Placeholder just returns input's first dimension."""
                 })
    
    # shooting point selection options
    
    do_tps : bool = field(
        default=False,
        metadata={'description':
"""If True: transition path sampling; if False: rejection-free sampling."""
                 })
    
    selection_pool_size : int = field(
        default=10,
        metadata={'description':
"""Number of candidate paths per selection step. When `do_tps = True`,
`selection_pool_size = 1` is the only option."""
                 })
    
    at_least_one_transition_in_pool : bool = field(
        default=False,
        metadata={'description':
"""If True: ensure each pool contains at least 1 transition.
This breaks detailed balance, but avoid getting stuck close to the states
and compromise exploitation."""
                 })
    
    nbins: int = field(
        default=10,
        metadata={'description':
"""Number of bins partitioning the reactive region according to the
logit committor."""
                 })
    
    cutoff_max : float = field(
        default=20.0,
        metadata={'description':
"""Do not exceed `cutoff_max` absolute value with the first and last
finite bin boundaries. When not runnig free simulations, bins will
span from `-cutoff_max` to `cutoff_max`."""
                 })
    
    include_marginal_bins : bool = field(
        default=True,
        metadata={'description':
"""Include two extra bins at A/B boundaries, sacrificing a bit exploitation
for exploration. It makes sense only when `selection_pool_size > 1`."""
                 })
    
    lorentzian: float = field(
        default=np.inf,
        metadata={'description':
"""Width of Lorentzian target distribution in logit committor space.
If inf: sample shooting points uniformly between first and last bin."""
                 })
    
    adjust_selection_in_marginal_bins : bool = field(
        default=True,
        metadata={'description':
"""Try to correct over-selection in first and last bin, without breaking
detailed balance."""
                 })
    
    memory : float = field(
        default=1.0,
        metadata={'description':
"""Percent of shooting points to keep when computing the bins' populations,
used to accelerate the convergence of future shooting points selection."""
                 })
    
    equilibrium_overriding_states : str = field(
        default='AB',
        metadata={'description':
"""Allow occasional shooting point selection from the free simulations
around the states in `equilibrium_overriding_states`."""
                 })
    
    equilibrium_overriding_rate : int = field(
        default=100,
        metadata={'description':
"""Number of overriding attempts per every selection."""
                 })

    equilibrium_overriding_recovery_rate : float = field(
        default=0.05,
        metadata={'description':
"""Attempt overriding from the same bin as the old shooting point with
this probability. Too high values break the Markov Chain of paths too often.
0.05 is a good compromise."""
                 })
    
    restart_equilibrium_with_transitions : str = field(
        default='',
        metadata={'description':
"""States where to use the last frames of a random sampled transition
instead of the latest crossing of the target state of the previous free
simulation."""
                 })
    
    # shooting point initialization options
    
    randomize_shooting_velocities : bool = field(
        default=True,
        metadata={'description':
"""If True, resample shooting velocities from a Boltzmann distribution as
defined in `random_velocities`. Otherwise, reuse velocities from the parent
trajectory (works if `trajectory_extension == '.trr'`)."""
                 })
    
    # simulation options
    
    max_excursion_length: int = field(
        default=50_000,
        metadata={'description':
"""Maximum allowed number of frames per path. In this way, avoid the
simulations getting stuck in long-lived intermediates."""
                 })
    
    extra_equilibriumA : List[str] = field(
        default_factory=lambda: [],
        metadata={'description':
"""Add the free simulations in the folders listed in `extra_equilibriumA`
to the free simulations produced by this AIMMD run. Useful for recycling
already produced data."""
                 })
    
    extra_equilibriumA_states_map : List[str] = field(
        default_factory=lambda: [''],
        metadata={'description':
"""When adding the free simulations in the folders listed in
`extra_equilibriumA`, for each of these folders, you may need to redefine
the states, in case of multi-step transitions. For each string in
`extra_equilibriumA_states_map` list, you can do this by adapting the following
example. "AB BC ZA" means that A frames become B frames, B become C, and Z
become A. If the state is not present in the string, it is not converted."""
                 })
    
    extra_equilibriumB : List[str] = field(
        default_factory=lambda: [],
        metadata={'description':
"""Same as `extra_equilibriumA` but for this run's state B."""
                 })
    
    extra_equilibriumB_states_map : List[str] = field(
        default_factory=lambda: [''],
        metadata={'description':
"""Same as `extra_equilibriumB_states_map` but for this run's state B."""
                 })
    
    extra_extend_frames : int = field(
        default=0,
        metadata={'description':
"""When performing extension simulations, do not stop right after reaching
the target state, but rather continue for furhter `extra_extend_frames`.""" 
                 })
    
    # logit committor fit options
    
    fit : Callable = field(
        default=fit,
        metadata={'description':
"""Fit neural network parameters to pathensemble data. It must accept the
following arguments: network, pathensemble (positional arguments),
initial_paths, verbose, worker (otional arguments)."""
                 })
    
    rescale_committor : bool = field(
        default=False,
        metadata={'description':
"""If True: rescale the NN output to recover the expected crossing
probability behavior ~ `1/p` (from A) and ~ `1/(1 - p)` (from B),
where p is the committor, in case of diffusive system with small
enough interval between frames. Attention! For that, `params.network`
must support `rescale_knots` and `rescale_values`!"""
                 })
    
    # paths reweighting options 
    
    reweight_parameters : dict = field(
        default_factory=lambda: {'equilibrium_threshold': 10},
        metadata={'description':
"""Dictionary of parameters used for reweighting the paths in a
`PathEnsemble` or `PathEnsemblesCollection` object, necessary for free
energy and rates estimates. Passed to `pathensemble.reweight`."""
                 })
    
    # save options
    
    save_interval : int = field(
        default=10,
        metadata={'description':
"""Save NN model parameters every N shooting simulations."""
                 })
    
    # SLURM configuration (for HPC clusters)
    
    slurm_header : str = field(
        default='#SBATCH --mail-type=FAIL',
        metadata={'description':
"""Default SLURM configuration. Attention! Never include #!/bin/bash,
job-name, time, or number of nodes (will be determined automatically).
Each two-way shooting and free simulation worker will occupy one task.
Manager and trainer share an extra task together."""
                 })
    
    # (full) path from which params were loaded
    
    path : PosixPath = field(
        init=False,
        default=absolute_path(),
        metadata={'description':
"""Will perform engine operations relative to `path`'s directory."""
                 })
    
    def __init__(self, *args, **kwargs):
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
            filename = absolute_path(args[0])
            args = args[1:]
        except (IndexError, TypeError, FileNotFoundError):
            filename = None
        
        # create separate instance
        instance = Params.load(filename, *args, **kwargs)
        
        # assign instance's fields to self
        for name in self.__dataclass_fields__:
            value = getattr(instance, name)
            super().__setattr__(name, value)

    def _setattr(self, name, value):
        """Helper function."""
        
        # assign fields
        if name in self.__dataclass_fields__:
            expected_type = self.__dataclass_fields__[name].type
            
            # function
            if expected_type is Callable:
                if not callable(value):
                    raise TypeError(f'{name} must be callable, '
                                    f'got {type(value).__name__}: {value}')
                
                # always try to get function source
                try:
                    value.__source__ = getsource(value)
                except:
                    if not hasattr(value, '__source__'):
                        value.__source__ = \
                            f'lambda : raise Exception("not found")'
                
                # process lambda functions
                if (not value.__source__.startswith('def ')
                    and 'lambda' in value.__source__):
                    value.__source__ = f'{name} = lambda' + (
                      'lambda'.join(value.__source__.split('lambda')[1:]))
                
                # function defined not in main
                if value.__module__ != '__main__':
                    value.__source__ = (  # when "local" import, omit path
                        f'from {value.__module__.split("/")[-1]} '
                        f'import {name}\n')
            
            # initial paths (converted in real-time)
            elif name == 'initial_paths':
                if type(value) is str:
                    value = [value]  # from string to list of strings
                if not hasattr(value, '__len__') or not len(value):
                    raise TypeError(f'Need at least one initial path, please '
                                    f'set initial_paths with a list of strings'
                                    f' or MDAnalysis trajectories')
                if hasattr(value, 'filename'):
                    value = [value]  # from MDA trajectory to list of
                value = list(value)
                for i, path in enumerate(value):
                    value[i] = self._convert_path(path)
            
            # engine
            elif name == 'engine':
                value = value.lower()
                if value not in ['gromacs', 'toy']:
                    raise TypeError(f'{name} must be either "gromacs" or '
                                    f'"toy", got {value}')
            
            # network
            elif name == 'network':
                for attribute in ['forward', 'state_dict', 'load_state_dict']:
                    if not hasattr(value, attribute) or not callable(
                        getattr(value, attribute)):
                        raise TypeError(
                            f'{name} must have method "{attribute}"')
                
                # always try to get class source
                try:
                    value.__class__.__source__ = getsource(value.__class__)
                except:
                    if not hasattr(value.__class__, '__source__'):
                        value.__class__.__source__ = (
                            f'class {value.__class__.__name__}:\n'
                            f'    def __init__(self):\n'
                            f'        raise Exception("not found")\n')
                
                # class defined in main
                if value.__class__.__module__ == '__main__':
                    value.__source__ = (
                        f'{value.__class__.__source__}\n'
                        f'network = {value.__class__.__name__}()')
                
                # class defined somewhere else, network defined in main
                elif value.__module__ == '__main__':
                    value.__source__ = (  # when "local" import, omit path
                        f'from {value.__class__.__module__.split("/")[-1]} '
                        f'import {value.__class__.__name__}\n'
                        f'network = {value.__class__.__name__}()')
                
                else:  # both defined somewhere else
                    value.__source__ = (  # when "local" import, omit path
                        f'from {value.__module__.split("/")[-1]} '
                        f'import network\n')
            
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
                if not os.path.exists(value):
                    raise TypeError(f'Source path {value} does not exist.')
            
            # all the rest
            elif name != 'path' and expected_type != type(value):
                raise TypeError(f'{name} must be {expected_type}, '
                                f'got {type(value).__name__}')
            
            # by setting attribute, you loose link to path
            # if hasattr(self, 'path'):
            #     path = self.path
            #     if self.path.is_file():
            #         path = path.parent
            # else:
            #     path = absolute_path()
            # super().__setattr__('path', path)
        
        # assign (path is last in "load" so you will restore it)
        super().__setattr__(name, value)
        
        # special checks (after assignment)
        # you can change states_function and only check that everything
        # ...goes fine after new initial paths are given
        if name == 'initial_paths':
            if hasattr(self, 'states_function'):
                self._check_initial_paths_and_states_function()
            if hasattr(self, 'descriptors_function'):
                self._check_initial_paths_and_states_function()
            if hasattr(self, 'values_function'):
                self._check_initial_paths_and_states_function()
        if name == 'descriptors_function' and hasattr(self, 'initial_paths'):
            self._check_descriptors_function()
        if name == 'values_function' and hasattr(self, 'initial_paths'):
            self._check_values_function()
    
    def __setattr__(self, name, value):
        """Enforce data types when reassigning params."""
        
        # backup old value
        if hasattr(self, name):
            backup = getattr(self, name)
        else:
            backup = None
        
        try:
            self._setattr(name, value)
        
        # in case of errors, back to the old value
        except Exception as exception:
            if backup is not None:
                super().__setattr__(name, backup)
            raise exception
    
    def __eq__(self, params):
        # different working directory
        if self.parent != params.parent:
            return False
        
        # after initialization, all fields are already populated
        for name in self.__dataclass_fields__:
            value1 = getattr(self, name)
            value2 = getattr(params, name)
            
            # check initial paths
            if name == 'initial_paths':
                if len(value1) != len(value2):
                    return False
                for path1, path2 in zip(value1, value2):
                    # after initialization, all paths are already
                    # MDAnalysis trajectories
                    if path1.filename != path2.filename:
                        return False
            
            # check functions / instances of classes
            elif hasattr(value1, '__module__'):
                
                # if they are functions, the check is easy
                if hasattr(value1, '__code__'):
                    if value1.__code__.co_code != value2.__code__.co_code:
                        return False
                    if (value1.__code__.co_argcount !=
                        value2.__code__.co_argcount):
                        return False
                
                # now we are checking instances of classes;
                # when both values are not from main...
                elif (value1.__module__ != '__main__' and
                      value2.__module__ != '__main__'):
                    
                    # ...they must have the same module
                    if value1.__module__ != value2.__module__:
                        return False
                
                else: # otherwise, they must share the same source
                    try:
                        object1 = value1
                        object2 = value2
                        assert object1.__source__ == object2.__source__
                    
                    except:
                        # last chance: the classes must share the same source
                        if hasattr(value1, '__class__') and hasattr(
                            value1.__class__, '__source__') and hasattr(
                            value2.__class__, '__source__'):
                            object1 = value1.__class__
                            object2 = value2.__class__
                            if object1.__source__ != object2.__source__:
                                return False
                            
                            # we'll then reference the class and not
                            # directly the instance
                        
                        else:  # that was your last chance...
                            return False
                    
                    # finally, if not from main, they must be local
                    if object1.__module__ == '__main__':
                        try:
                            path = absolute_path(f'{object2.__module__}.py',
                                                 go_to=self.parent)
                            assert path.parent == self.parent
                        except:
                            return False
                    if object2.__module__ == '__main__':
                        try:
                            path = absolute_path(f'{object1.__module__}.py',
                                                 go_to=self.parent)
                            assert path.parent == self.parent
                        except:
                            return False
            
            # check other types of values (easy)
            elif value1 != value2:
                return False
        
        return True
    
    def __str__(self, go_to=None):
        """Verbose string representation of params with descriptions and
        function bodies."""
        
        if self.path.is_file():
            lines = [f'___ Params in {self.path} ___ ']
        else:
            lines = [f'___ Params ___']
        
        for name in self.__dataclass_fields__:
            if name == 'path':
                continue
            
            field = self.__dataclass_fields__[name]
            value = getattr(self, name)
            
            # function or network
            if hasattr(value, '__source__'):
                lines.append(value.__source__.rstrip('\n'))
            
            # initial paths
            elif name == 'initial_paths':
                if not go_to:
                    go_to = absolute_path()
                filenames = [
                    f'"{os.path.relpath(path.filename, go_to)}"'
                    for path in value]
                lines.append(f'{name} = [{", ".join(filenames)}]')
            
            # all the rest
            else:
                lines.append(f'{name} = {repr(value)}')
            
            # print description
            if desc := field.metadata.get("description", ""):
                lines.append(f"\"\"\"{desc}\"\"\"\n")
        
        return "\n".join(lines)
    
    @property
    def parent(self):
        if self.path.is_file():
            return self.path.parent
        return self.path
    
    @property
    def mdrun(self):
        # engine-dependent mdrun command
        if self.engine == 'gromacs':
            return self.gmx_mdrun
        if self.engine == 'toy':
            return self.toy_mdrun
    
    @property
    def grompp(self):
        if self.engine == 'gromacs':
            return self.gmx_grompp
        if self.engine == 'toy':
            return ''
        
    @property
    def eneconv(self):
        if self.engine == 'gromacs':
            return self.gmx_eneconv
        if self.engine == 'toy':
            return ''
    
    def _convert_path(self, path):
        """Convert path from sting to MDAnalysis `MemoryReader`.
        Otherwise do nothing"""

        if type(path) not in (str, PosixPath):
            return path
        
        # absolute path
        path = absolute_path(path, go_to=self.parent)
        topology = absolute_path(self.topology, go_to=self.parent)
        return mda.Universe(topology, path).trajectory
    
    def _check_initial_paths_and_states_function(
        self, initial_paths=[], crop=False):
        """Run states_function and inspect result. Replace initial_path
        strings with MDAnalysis trajectories. Ensure initial paths are
        transitions. Return processed initial paths.
        crop: if True, crop each trajectory to first transition.
        """
        
        # either new paths or already attributed ones
        if not initial_paths:
            initial_paths = self.initial_paths
        
        # iterate through initial paths
        for i, path in enumerate(initial_paths):
            initial_paths[i] = self._convert_path(path)
            
            # compute states and check if the output of states_function
            states = self.states_function(path)
            if type(states) != np.ndarray or len(states) != len(path)\
            or len(states.shape) > 1\
            or states.dtype != np.dtype('<U1'):
                raise TypeError(f'states_function does not return an '
                                f'equally long array of chars (=states)')
            
            # look for a transition
            crossings = np.where(np.diff((states == 'R').astype(int)))[0]
            transition_found = False
            for b, e in zip(crossings, crossings[1:]):
                if states[b] != states[e + 1]:
                    # transition found
                    transition_found = True
                    if crop:
                        if type(path) == MemoryReader:
                            path = MemoryReader(
                                path.coordinate_array[b:e + 2],
                                dimensions=path.dimensions_array[b:e + 2],
                                velocities=path.velocity_array[b:e + 2]
                                    if path.velocity_array else None,
                                filename=path.filename)
                        else:
                            filename = path.filename
                            path = path[b:e + 2]
                            path.filename = filename
                        initial_paths[i] = path
                    break
            
            # transition not found
            if not transition_found:
                raise TypeError(f'The {i + 1}-th initial_path '
                                f'"{initial_paths[i].filename}" does not '
                                f'contain a transition')
        
        # return processed initial paths
        return initial_paths 
    
    def _check_descriptors_function(self):
        """Run descriptors_function and inspect result."""
        
        descriptors = self.descriptors_function(self.initial_paths[0][:1])
        if type(descriptors) != np.ndarray or len(descriptors) != 1\
        or len(descriptors.shape) != 2:
            raise TypeError(f'descriptors_function does not return '
                            f'an array of size 2 and correct length')
    
    def _check_values_function(self):
        """Run descriptors_function, values_function and inspect result."""
        
        # check descriptors_function
        descriptors = self.descriptors_function(self.initial_paths[0][:1])
        if type(descriptors) != np.ndarray or len(descriptors) != 1\
        or len(descriptors.shape) != 2:
            raise TypeError(f'descriptors_function does not return '
                            f'an array of size 2 and correct length')
        
        # check values_function
        values = self.values_function(descriptors)
        if type(values) != np.ndarray or len(values) != len(descriptors)\
        or len(values.shape) > 1:
            raise TypeError(f'values_function does not return '
                            f'an array of size 1 and correct length')
    
    def _check_engine(self, timeout=10):
        """Will be called by user if necessary."""
        
        # go to the right folder
        cwd = absolute_path()
        os.chdir(self.parent)
        
        try:  # cleanup
            os.system(f'rm -f .params_check_engine*')
            
            if self.engine == 'gromacs':
                
                if self.randomize_shooting_velocities:                    
                    # gromacs grompp: init velocities
                    cmd = (f'{self.gmx_grompp} -nobackup '
                           f'-f {self.gmx_init_mdp} '
                           f'-r {self.topology} -c {self.topology} '
                           f'-o .params_check_engine.tpr')
                    if exit := execute_command(cmd, walltime=timeout):
                        raise RuntimeError(
                            f'{cmd} failed with exit code {exit}')
                    
                    # gromacs grompp: mdrun
                    cmd = (f'{self.gmx_mdrun} -nobackup '
                           f'-deffnm .params_check_engine -nsteps 0')
                    if exit := execute_command(cmd, walltime=timeout):
                        raise RuntimeError(
                            f'{cmd} failed with exit code {exit}')
                        
                # gromacs grompp: mdrun
                cmd = (f'{self.gmx_grompp} -nobackup -f {self.gmx_run_mdp} '
                       f'-r {self.topology} -c {self.topology} '
                       f'-o .params_check_engine.tpr') + (
                       f' -t .params_check_engine.trr'
                    if self.randomize_shooting_velocities else '')
                if exit := execute_command(cmd, walltime=timeout):
                    raise RuntimeError(
                        f'{cmd} failed with exit code {exit}')
                
                # gromacs mdrun
                cmd = (f'{self.gmx_mdrun} -nobackup -deffnm '
                       f'.params_check_engine')
                if exit := execute_command(cmd, walltime=timeout):
                    raise RuntimeError(
                        f'{cmd} failed with exit code {exit}')
                
                # right extension?
                fname = f'.params_check_engine{self.trajectory_extension}'
                if not os.path.exists(fname):
                    raise IOError(
                        f'{self.trajectory_extension} file not generated')
            
            # toy mdrun
            if self.engine == 'toy':
                fname = f'.params_check_engine{self.trajectory_extension}'
                md.load(self.topology).save(fname)
                cmd = f'{self.toy_mdrun} -deffnm .params_check_engine'
                if exit := execute_command(cmd, walltime=timeout):
                    raise RuntimeError(f'{cmd} failed with exit code {exit}')
        
        except Exception as exception:  # cleanup
            os.system(f'rm -f .params_check_engine*')
            raise exception
        
        finally:  # back to the original folder
            os.chdir(cwd)
    
    @class_or_instancemethod
    def load(self_or_cls, filename='params.py', *args, **kwargs):
        """
        Load parameters and functions from a Python file and update Params
        instance *in place*. Load additional `kwargs` with higher priority.
        
        filename: str or None
            if None, just run normal init
        """
        
        # which folder we need to go to?
        if filename:
            path = absolute_path(filename)
            if not path.is_file():
                raise TypeError(f'{filename} must be a file')
            folder = path.parent
        else:
            folder = path = absolute_path()
        
        # go to folder
        cwd = absolute_path()
        os.chdir(folder)
        sys.path.insert(0, f'{folder}')
        
        # in case of problems: restore modules and params fields
        backup_modules = sys.modules.copy()
        backup_params = {}
        
        try:
            # do we need to create or select an instance of Params?
            if isinstance(self_or_cls, type):
                instance = super().__new__(self_or_cls)
            else:
                instance = self_or_cls
                if filename and instance.parent != folder:
                    raise TypeError(
                        f"New params' filename \"{path}\" must be "
                        f"in the same folder associated to this"
                        f"aimmd.Params object: \"{instance.parent}\"")
            
            # fields with already present values or their default
            fields = {name:
                      getattr(instance, name)
                          if hasattr(instance, name) else
                      self_or_cls.__dataclass_fields__[name].default
                          if self_or_cls.__dataclass_fields__[name].default
                          is not MISSING else
                      self_or_cls.__dataclass_fields__[name].default_factory()
                          if self_or_cls.__dataclass_fields__[name
                          ].default_factory is not MISSING else
                      MISSING for name in self_or_cls.__dataclass_fields__
                          if name != 'path'}
            
            # defaults
            new_states_function = False
            new_initial_paths = False
            instance.__setattr__('path', path)
            
            # update fields with args (in the right order)            
            for value, name in zip(args, self_or_cls.__dataclass_fields__):
                if name in kwargs:
                    raise TypeError(f'multiple assignements of {name} '
                                    f'when calling Params.load; either '
                                    f'remove the positional argument or '
                                    f'the keyword argument')
                kwargs[name] = value
                        
            # update fields with kwargs
            for name in kwargs:
                if name in fields:
                    
                    # backup old values
                    if hasattr(instance, name):
                        backup_params[name] = fields[name]
                    
                    # get new values
                    fields[name] = kwargs[name]
                
                # special fields
                if name == 'states_function':
                    new_states_function = True
                if name == 'initial_paths':
                    new_initial_paths = True
                    
                    # process to ensure right relative path is registered
                    value = fields[name]
                    if type(value) is str:
                        value = [value]  # from string to list of strings
                    if not hasattr(value, '__len__') or not len(value):
                        raise TypeError(
                            f'Need at least one initial path, please '
                            f'set initial_paths with a list of strings'
                            f' or MDAnalysis trajectories')
                    if hasattr(value, 'filename'):
                        value = [value]  # from MDA trajectory to list of
                    value = list(value)
                    for i, initial_path in enumerate(value):
                        if type(initial_path) is str:
                            value[i] = absolute_path(initial_path, go_to=cwd)
                    fields[name] = value
            
            # execute the file and extract fields...
            num_fields_from_filename = 0
            if filename:
                
                # assign local modules names
                local_module_names = {}
                temporarily_removed_modules = {}
                
                # find all local py files in the folder except the main script
                for local_path in folder.glob("*.py"):
                    
                    # each file in the current folder will get a "local"
                    # module name, corresponding to its full path without .py
                    #
                    # however, the params' "filename" may also be executed
                    # as a standalone file in a "fresh" python interpreter,
                    # and still give the same results
                    #
                    # in order to achieve this goal, ONLY during the params
                    # loading phase, the "local" module name is replaced
                    # by a temporary module name with the relative path
                    # instead of the full path
                    #
                    # importantly, there may be already imported modules with
                    # the same name as those temporary module names
                    #
                    # the "original" modules with conflicting names
                    # temporarily swap their name with the local module name,
                    # and get their original one back only at the end of
                    # "filename"'s execution
                    #
                    # special care is put in restoring the original modules
                    # situation in case some error happens
                    
                    local_module_name = str(local_path).rstrip('.py')
                    original_module_name = local_module_name.split('/')[-1]
                    local_module_names[original_module_name] = \
                        local_module_name
                    
                    # swap name of original modules
                    if original_module_name in sys.modules:
                        sys.modules[local_module_name] = \
                            sys.modules.pop(original_module_name)
                    
                    # ALL local modules are temporarily removed
                    # incidentally, this means that one may just change the
                    # local parameters files, and those modules are updated
                    # at the next Params.load call, in contrast with the usual
                    # behavior when importing modules with python
                    if local_module_name in sys.modules:
                        temporarily_removed_modules[local_module_name] = \
                            sys.modules.pop(local_module_name)
                
                # add current directory to sys.path
                sys.path.insert(0, '')
                
                # treat filename as a (local) module
                module_name = str(path).split('/')[-1].rstrip('.py')
                module = types.ModuleType(module_name)
                module.__file__ = str(path)
                sys.modules[module_name] = module
                
                # execute the file inside the module’s namespace
                source = path.read_text()
                exec(compile(source, path, "exec"), module.__dict__)
                
                # populate the fields
                for name in module.__dict__:
                    
                    # ...only if not assigned already
                    if name in fields and name not in kwargs:
                        
                        # backup old values
                        if hasattr(instance, name):
                            backup_params[name] = fields[name]
                        
                        # get new values
                        fields[name] = module.__dict__[name]
                        num_fields_from_filename += 1
                    
                    # special fields
                    if name == 'states_function':
                        new_states_function = True
                    if name == 'initial_paths':
                        new_initial_paths = True
                
                # after "filename" execution, we can swap original and local
                # modules name avoiding conflicts
                for original_name, local_name in local_module_names.items():
                    original_module = None
                    local_module = None
                    
                    # retrieve modules
                    if original_name in sys.modules:
                        local_module = sys.modules.pop(original_name)
                    if local_name in sys.modules:
                        original_module = sys.modules.pop(local_name)
                    
                    # actual swapping
                    if original_module is not None:
                        sys.modules[original_name] = original_module
                    if local_module is not None:
                        sys.modules[local_name] = local_module
                        
                        # let classes and functions inherit the change
                        # through their __module__ attribute
                        # (necessary only for local modules, used in __str__
                        #  and save methods)
                        for name, obj in local_module.__dict__.items():
                            if (callable(obj) or isinstance(obj, type)) and (
                                hasattr(obj, '__module__') and
                                obj.__module__ == original_name):
                                obj.__module__ = local_name
                
                # put back temporarily removed modules that were not
                # reimported in the procedure
                for module_name in temporarily_removed_modules:
                    if module_name not in sys.modules:
                        sys.modules[module_name] = \
                            temporarily_removed_modules[module_name]
            
            # assign fields; raise error for missing fields
            for name in list(fields):
                if fields[name] is MISSING:
                    raise TypeError(f'{name} not provided for Params')
                instance.__setattr__(name, fields[name])
            
            # post-init operation: since you set them together, check again
            if new_states_function or new_initial_paths:
                instance._check_initial_paths_and_states_function()
            
            # save new file and set path while saving
            if num_fields_from_filename < len(fields):
                # save it only when params file did not have all fields
                # already defined with no defaults
                instance.save()
            
            return instance
        
        except Exception as exception:
            
            # restore modules
            sys.modules = backup_modules
            
            # restore attributes
            for name in backup_params:
                object.__setattr__(instance, name, backup_params[name])
            
            raise exception
        
        finally:  # back to the original folder & path
            os.chdir(cwd)
            sys.path.pop(0)
    
    def update(self, *args, **kwargs):
        """Like load but without filename."""
        return self.load(None, *args, **kwargs)

    def save(self, path=None, seek_existing_file=True):
        """Save to file and replace params.path.
        MUST be in the same folder as params' working directory.
        path: if None, assign by default
        seek_existing_file: if True, try to replace path with already
        existing file"""
        
        # determine correct path (never overwrite)
        if not path:
            path = self.path
        else:
            path = absolute_path(path, check=False)
        if path == self.parent:
            filename = Path(f'{self.parent}/params.py')
        elif path.parent != self.parent:
            raise TypeError(f'params must be saved be in '
                            f'"{os.path.relpath(str(self.parent), ".")}"')
        else:
            filename = path
        
        while filename.exists():
            filename = str(filename).rstrip('.py')
            i = len(filename)
            while filename[i - 1:].isnumeric() and i:
                i -= 1
            if len(filename[i:]):
                n = int(filename[i:]) + 1
            else:
                n = 1
            filename = Path(f'{filename[:i]}{n}.py')
        
        # create text object
        text = []
            
        # copy main modules
        modules = vars(sys.modules['__main__'])
        text.append(f'# packages\n')
        for name in modules:
            if type(modules[name]) is not type(sys):
                continue
            text.append(f'import {modules[name].__name__} as {name}\n')
        text.append(f'inf = float("inf")\n\n')
        
        # copy params
        text.append('\n'.join(self.__str__().split('\n')[1:]))
        text = "".join(text)

        # was there? then use it
        writing = True
        if seek_existing_file:
            for old_filename in sorted(filename.parent.glob("*.py")):
                if old_filename.read_text() == text:
                    writing = False
                    filename = old_filename
                    break
        
        # only when different: write it
        if writing:
            with open(filename, 'w') as file:
                file.write(text)
            
            # report
            print(f'Written full params and descriptions to '
              f'"{os.path.relpath(filename, absolute_path())}"')
        else:
            print(f'Assigned parameters file '
              f'"{os.path.relpath(filename, absolute_path())}"')
        
        # update path info and report
        self.__setattr__('path', filename)
    
    def crop_initial_paths(self):
        """Leave only the transition parts in `params.inital_paths`, to
        speed up future computations. Attention! May then need to reassign
        `initial transitions` when changing `states_function`."""
        setattr(self, 'initial_paths',
                self._check_initial_paths_and_states_function(crop=True))
    
    def save_initial_paths(self, folder, crop=True):
        """Save (cropped) version of initial paths to `folder` with unique
        names derived from the original files. Attention! Overwrites."""
        
        initial_paths = self._check_initial_paths_and_states_function(
            crop=crop)
        
        # save initial paths
        filenames = [path.filename.split('/')[-1]
                     for path in initial_paths]
        for i, path in enumerate(initial_paths):
            filename = filenames[i]
            
            # avoid duplicates 
            if filename in filenames[:i]:
                filename = (f'{".".join(filenames[i].split(".")[:-1])}'
                            f'-2.{filenames[i].split(".")[-1]}')
                filenames[i] = filename
            
            # report
            print(f'Writing {folder}/{filename}')
            os.system(f'mkdir -p {folder}')
            
            # actual save: get n_atoms
            n_atoms = len(path[0].positions)
            
            # create an empty Universe with n_atoms (no topology required)
            universe = mda.Universe.empty(n_atoms, trajectory=True)
            
            # write positions
            with mda.Writer(f'{folder}/{filename}', n_atoms) as writer:
                for frame in path:
                    universe.trajectory.ts.positions = frame.positions
                    if hasattr(frame, 'velocities'):
                        universe.trajectory.ts.velocities = frame._velocities
                    universe.trajectory.ts.triclinic_dimensions = \
                        frame.triclinic_dimensions
                    writer.write(universe)
