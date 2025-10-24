'''
AIMMD parameters management / defaults.
'''

import os
import sys
import torch
import numpy as np
import shutil
import mdtraj as md
import MDAnalysis as mda
from .utils import (class_or_instancemethod, fit,
                    PlaceholderNetwork, execute_command)
from typing import List, Callable
from pathlib import Path, PosixPath
from dill.source import getsource
from dataclasses import dataclass, field, MISSING
from MDAnalysis.coordinates.memory import MemoryReader

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
        default=Path('.').resolve(),
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
            filename = Path(args[0]).resolve()
            args = args[1:]
        except (IndexError, TypeError, FileNotFoundError):
            filename = None
        
        # create separate instance
        instance = Params.load(filename, *args, **kwargs)
        
        # assign instance's fields to self
        for name in self.__dataclass_fields__:
            value = getattr(instance, name)
            super().__setattr__(name, value)
    
    def __setattr__(self, name, value):
        """Enforce data types when reassigning params."""
        
        # assign fields
        if name in self.__dataclass_fields__:
            expected_type = self.__dataclass_fields__[name].type
            
            # function
            if expected_type is Callable:
                if not callable(value):
                    raise TypeError(f'{name} must be callable, '
                                    f'got {type(value).__name__}')
                if value.__module__ == '__main__':
                    value.__source__ = getsource(value)
                    if (not value.__source__.startswith('def ')
                        and 'lambda' in value.__source__):
                        value.__source__ = f'{name} = lambda ' + (
                          'lambda'.join(value.__source__.split('lambda')[1:]))                      
                else:
                    value.__source__ = (f'from {value.__module__} '
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
                if value.__module__ == '__main__':
                    value.__source__ = (
                        f'{getsource(value.__class__)}\n'
                        f'network = {value.__class__.__name__}()\n')
                else:
                    value.__source__ = (
                        f'from {value.__module__} import '
                        f'{value.__class__.__name__}\n'
                        f'network = {value.__class__.__name__}()\n')
            
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
            if hasattr(self, 'path'):
                path = self.path
                if self.path.is_file():
                    path = path.parent
            else:
                path = Path('.').resolve()
            super().__setattr__('path', path)
        
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
    
    def __eq__(self, params):
        # after initialization, all fields are already populated
        for name in self.__dataclass_fields__:
            value1 = getattr(self, name)
            value2 = getattr(params, name)
            if name == 'initial_paths':
                if len(value1) != len(value2):
                    return False
                for path1, path2 in zip(value1, value2):
                    # after initialization, all paths are already
                    # MDAnalysis trajectories
                    if path1.filename != path2.filename:
                        return False
            elif hasattr(value1, '__module__'):
                # also params.path falls in here,
                # and has always the same module, so it's never False
                if value1.__module__ != value2.__module__:
                    return False
            elif value1 != value2:
                return False
        return True
    
    def __str__(self):
        """Verbose string representation of params with descriptions and
        function bodies."""
        
        if self.path.is_file():
            lines = [f'___ Params from {self.path} ___ ']
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
                filenames = [f'"{path.filename}"' for path in value]
                lines.append(f'{name} = [{", ".join(filenames)}]')
            
            # all the rest
            else:
                lines.append(f'{name} = {repr(value)}')
            
            # print description
            if desc := field.metadata.get("description", ""):
                lines.append(f"\"\"\"{desc}\"\"\"\n")
        
        return "\n".join(lines)
    
    # engine-dependent mdrun command
    @property
    def mdrun(self):
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

        if type(path) is not str:
            return path
        
        # absolute path
        path = f'{Path(path).resolve()}'
        
        # go to the right folder
        cwd = os.getcwd()
        folder = self.path.parent if self.path.is_file() else self.path
        os.chdir(folder)
        
        try:
            # relative path with respect to params' folder
            relpath = os.path.relpath(path, folder)
            return mda.Universe(self.topology, relpath).trajectory
        
        except Exception as exception:
            raise TypeError(f'The initial path "{path}" resulted '
                            f'in the following error:\n{exception}')
        
        finally:  # back to the original folder
            os.chdir(cwd)
    
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
                raise TypeError(f'When loading "{filename}", states_function '
                                f'does not return an equally long array of '
                                f'chars (=states)')
            
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
                raise TypeError(f'The {i + 1}-th trajectory in initial_paths '
                                f'does not contain a transition')
        
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
        cwd = os.getcwd()
        os.chdir(self.path.parent if self.path.is_file() else self.path)
        
        try:
            # reset
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
        
        except Exception as exception:
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
        cwd = os.getcwd()
        
        if filename:
            path = Path(filename).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f'Parameter file {filename} not found.')
            filename = f'{filename}'.split('/')[-1].rstrip('.py')
            folder = path.parent
            os.chdir(folder)
        else:
            folder = Path('.').resolve()
        
        sys.path.insert(0, '')  # allows to see modules in path.parent
        
        try:
            # create or select instance
            if isinstance(self_or_cls, type):
                instance = super().__new__(self_or_cls)
                instance.__setattr__('path', folder)  # temporary path
            else:
                instance = self_or_cls
                if instance.path.is_file():
                    instance_folder = instance.path.parent
                else:
                    instance_folder = instance.path
                if filename and instance_folder != folder:
                    raise TypeError(
                        f"New params' filename \"{path}\" must be "
                        f"in the same folder associated to this"
                        f"aimmd.Params object: \"{instance_folder}\"")
            
            # temporary path
            instance.__setattr__('path', folder)
            
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
                    fields[name] = kwargs[name]
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
                    os.chdir(cwd)
                    for i, initial_path in enumerate(value):
                        if type(initial_path) is str:
                            initial_path = f'{Path(initial_path).resolve()}'
                            value[i] = os.path.relpath(
                                initial_path, f'{folder}')
                    os.chdir(folder)
                    fields[name] = value
            
            # execute the file and extract fields
            num_fields_from_filename = 0
            if filename:
                source = path.read_text()
                exec_namespace = {}
                exec(compile(source, str(path), 'exec'), exec_namespace)
                for name in exec_namespace:
                    if name in fields and name not in kwargs: 
                        # only if not assigned already
                        fields[name] = exec_namespace[name]
                        num_fields_from_filename += 1
                        if name in ['states_function', 'descriptors_function',
                                    'values_function', 'network', 'fit']:
                            # register origin
                            fields[name].__module__ = filename
                        if name == 'states_function':
                            new_states_function = True
                        if name == 'initial_paths':
                            new_initial_paths = True
            
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
                # only when params file did not have all fields
                # already defined with no defaults
                instance.save()
            else:
                instance.__setattr__('path', path)
            
            return instance
        
        except Exception as exception:
            raise exception
        
        finally:  # back to the original folder
            os.chdir(cwd)
            sys.path.pop(0)
    
    def update(self, *args, **kwargs):
        """Like load but without filename."""
        return self.load(None, *args, **kwargs)

    def save(self, path=None):
        """Save to file and replace params.path"""
        
        # determine correct path (never overwrite)
        if not path:
            path = self.path
        if Path(f'{path}').resolve().is_file():
            filename = f'{path}'
        else:
            filename = f'{path}/params.py'
        while Path(filename).resolve().exists():
            filename = filename.rstrip('.py')
            i = len(filename)
            while filename[i - 1:].isnumeric() and i:
                i -= 1
            if len(filename[i:]):
                n = int(filename[i:]) + 1
            else:
                n = 1
            filename = f'{filename[:i]}{n}.py'
        
        # actual save
        with open(filename, 'w') as file:
            
            # copy main modules
            modules = vars(sys.modules['__main__'])
            file.write(f'# packages\n')
            for name in modules:
                if type(modules[name]) is not type(sys):
                    continue
                file.write(f'import {modules[name].__name__} as {name}\n')
            file.write(f'inf = float("inf")\n\n')
            
            # copy params
            file.write('\n'.join(self.__str__().split('\n')[1:]))
        
        # change module and source to new file
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if hasattr(value, '__module__'):
                if value.__module__ == '__main__':
                    value.__module__ = filename.rstrip('.py')
                    if name == 'network':
                        value.__source__ = (
                            f'from {value.__module__} import '
                            f'{value.__class__.__name__}\n'
                            f'network = {value.__class__.__name__}()\n')
                    else:
                        value.__source__ = (f'from {value.__module__} '
                                            f'import {name}\n')
        
        # update path info and report
        path = Path(filename).resolve()
        self.__setattr__('path', path)
        print(f'Written params to {path}')
    
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
