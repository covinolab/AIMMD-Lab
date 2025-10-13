'''
AIMMD parameters management / defaults.
'''

import sys
import dill
import types
import shutil
import linecache
import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
from .utils import fit  # base fit function
from typing import List, Callable
from pathlib import Path
from dill.source import getsource
from dataclasses import dataclass, field, fields

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
contain transitions (checked automatically). Will replace strings by
MDAnalysis trajectories.
"""
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
"""Fit neural network parameters to pathensemble data."""
                 })
    
    rescale_committor : bool = field(
        default=True,
        metadata={'description':
"""If True: rescale the NN output to recover the expected crossing
probability behavior ~ `1/p` (from A) and ~ `1/(1 - p)` (from B),
where p is the committor, in case of diffusive system with small
enough interval between frames."""
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
        default=('#SBATCH --ntasks-per-node=1\n'
                 '#SBATCH --cpus-per-task=16\n'
                 '#SBATCH --mail-type=FAIL'),
        metadata={'description':
"""Default SLURM configuration. Attention! It must include the
ntasks-per-node opton. Do not include #!/bin/bash, job-name, or number
of nodes (will be determined automatically). Each two-way shooting and
free simulation worker uses one task. Manager and trainer share an extra
task together."""
                 })
    
    # engine-dependent mdrun command
    @property
    def mdrun(self):
        if self.engine == 'gromacs':
            return self.gmx_mdrun
        if self.engine == 'toy':
            return self.toy_mdrun
    
    def _check_initial_paths_and_states_function(self, initial_paths=[]):
        """Run states_function and inspect result. Replace initial_path
        strings with MDAnalysis trajectories. Ensure initial paths are
        transitions. Return processed initial paths."""
        
        # either new paths or already attributed ones
        if not initial_paths:
            initial_paths = self.initial_paths
        
        # iterate through initial paths
        for i, path in enumerate(initial_paths):
            
            # replace strings with MDAnalysis trajectories
            if type(path) is str:
                filename = path
                try:
                    path = mda.Universe(
                        self.topology, filename, in_memory=True).trajectory
                except Exception as exception:
                    raise TypeError(f'The initial path "{path}" resulted '
                                    f'in the following error:\n{exception}')
                initial_paths[i] = path
            
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
            
            # transition not found
            if not transition_found:
                raise TypeError(f'The {i + 1}-th trajectory in initial_paths '
                                f'does not contain a transition')
        
        # return processed initial paths
        return initial_paths 
    
    def _check_descriptors_and_values_function(self):
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
    
    def __post_init__(self):
        """Check provided functions. Replace initial_path strings with
        MDAnalysis trajectories. Ensure initial paths are transitions.
        Return processed initial paths."""
        
        self._check_initial_paths_and_states_function()
        self._check_descriptors_and_values_function()
    
    def __str__(self):
        """Verbose string representation of params with descriptions and
        function bodies."""
        
        lines = []
        
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            if not callable(value):
                lines.append(f'{name} = {repr(value)}')
                if desc := f.metadata.get("description", ""):
                    lines.append(f"\"\"\"{desc}\"\"\"\n")
            else:  # if it's a function, show its content
                try:
                    lines.append(value.__source__)
                except Exception:
                    lines.append(
                        f"def {name}\n:\n    # source unavailable\n    pass\n")
        
        return "\n".join(lines)
    
    def __setattr__(self, name, value):
        """Enforce data types when reassigning params."""
        
        # get type hints dynamically from dataclass field definitions
        hints = {f.name: f.type for f in fields(self)}
        
        # check
        if name in hints and name not in ['initial_paths', 'engine']:
            expected_type = hints[name]
            
            # function
            if expected_type is Callable:
                if not callable(value):
                    raise TypeError(f'{name} must be callable, '
                                    f'got {type(value).__name__}')
                try:  # assign source
                    value.__source__ = getsource(value)
                except Exception:
                    value.__source__ = (
                        f"def {name}:\n    # source unavailable\n    pass")
            
            # list of strings
            elif expected_type is List[str]:
                if type(value) is not list:
                    raise TypeError(f'{name} must be list of strings, '
                                    f'got {type(value).__name__}')
                elif np.any([type(element) is not str for element in value]):
                    raise TypeError(f'{name} must be list of strings, '
                                    f'at least one of its elements is not')
            
            # all the rest
            elif expected_type != type(value):
                raise TypeError(f'{name} must be {expected_type}, '
                                f'got {type(value).__name__}')
        
        # special check: initial paths
        if name == 'initial_paths':
            if not len(value):
                raise TypeError(f'Need at least one initial path, please set '
                                f'initial_paths with a list of strings or '
                                f'MDAnalysis trajectories')
                self._check_initial_paths_and_states_function()
        
        # special check: engine (before assignment)
        if name == 'engine':
            value = value.lower()
            if value not in ['gromacs', 'toy']:
                raise TypeError(f'{name} must be either "gromacs" or "toy", '
                                f'got {name}')
        
        # assign
        super().__setattr__(name, value)
        
        # special check: descriptors_function (after assignment)
        if name == 'descriptors_function' and hasattr(self, 'initial_paths'):
            self._check_descriptors_function()
        
        # special check: values_function (after assignment)
        if name == 'values_function' and hasattr(self, 'initial_paths'):
            self._check_values_function()
    
    @class_or_instancemethod
    def update(self_or_cls, filename='params.py'):
        """
        Load parameters and functions found in python file `filename`.
        - If called on the class (Params.load()), creates a new instance.
        - If called on an instance (params.load()), updates it in place.
        """
        
        # load filename's content in "module" (named "_current_aimmd_params")
        path = Path(filename).resolve()
        if not path.exists():
            raise FileNotFoundError(f'Parameter file \'{path}\' not found.')
        
        # read the source first
        source = path.read_text()
        
        # register source code with linecache
        linecache.cache[str(path)] = (
            len(source),  # size
            None,         # modification time
            source.splitlines(True),  # list of lines
            str(path)     # filename
        )
        
        # create a new module object manually
        module = types.ModuleType('_current_aimmd_params')
        module.__file__ = str(path)
        
        # compile and execute code so that functions get correct co_filename
        code = compile(source, str(path), 'exec')
        # TODO different folder change relative path
        exec(code, module.__dict__)
        
        # replace old module in sys.modules
        sys.modules['_current_aimmd_params'] = module
        
        # parameters/functions you can update
        param_fields = {f.name for f in fields(Params)}
        
        # parameters/functions found in "module"
        kwargs = {}
        for name in dir(module):
            if name.startswith("__"):
                continue
            if name in param_fields:
                kwargs[name] = getattr(module, name)
        
        # called on class: initialize
        if isinstance(self_or_cls, type):
            return self_or_cls(**kwargs)
        
        # called on instance: update in place
        for name, value in kwargs.items():
            if name == 'initial_paths':
                continue
            setattr(self_or_cls, name, value)
        
        # initial paths are last, so that it checks with new states_function
        if 'initial_paths' in kwargs:
            setattr(self_or_cls, 'initial_paths', kwargs['initial_paths'])
        
        return self_or_cls
    
    def save(self, filename: str):
        """Save only dataclass attributes (data + functions) to a dill file."""
        # Extract dataclass fields only
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        with open(filename, "wb") as f:
            dill.dump(data, f)
    
    @classmethod
    def load(cls, filename: str) -> "Params":
        """Load Params from a dill file, restoring only dataclass fields."""
        with open(filename, "rb") as f:
            data = dill.load(f)
        return cls(**data)
