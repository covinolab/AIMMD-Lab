"""
...
"""

# external
from abc import ABC
from math import inf
from typing import List, Callable
from pathlib import PosixPath
from torch.nn import Module as NeuralNetworkModule
from dataclasses import dataclass, field

# aimmd imports
from .._config import GROMACS
from ..network.fit import fit
from ..network.utils import placeholder as placeholder_network

# params' fields
@dataclass
class ParamsFields(ABC):
    
    # states and states map
    
    states_function : Callable = field(
        metadata={'description':
"""From MDAnalysis trajectory to array of states."""
                 })
    
    states : str = field(
        default='ARB',
        metadata={'description':
"""Which states are considered first metastable, reactive region,
and final metastable, respectively."""
                 })
    
    # system's main information
    
    name: str = field(
        default='AIMMD',
        metadata={'description':
"""System's name (will be used for creating slurm's jobs)."""
        })

    topology : str = field(
        default='run.gro',
        metadata={'description':
"""System's topology (gro file) used for getting masses and by grompp."""
                 })

    trajectory_extension : str = field(
        default='.xtc',
        metadata={'description':
"""Use `xtc` for compressed data, `trr` for full-precision data and also
saving velocities along with the positions."""
                 })
    
    # engine configuration

    engine : str = field(
        default='gromacs',
        metadata={'description':
"""Either "gromacs" or "toy" engine."""
                 })
    
    # gromacs engine
    
    gmx_mdp : str = field(
        default='run.mdp',
        metadata={'description':
"""Used for initializing production runs (shooting / free simulations).
Attention! It must be consistent with "trajectory_extension"."""
                 })
    
    gmx_grompp : str = field(
        default=f'{GROMACS} grompp -maxwarn 1',
        metadata={'description':
"""Gromacs grompp and call options. Change to include index files. Allows to
produce `tpr` files. Attention! "coordinates", "restraints", "output", and
"nobackup" flags are added automatically, do not include them here."""
                 })
    
    gmx_mdrun : str = field(
        default=f'{GROMACS} mdrun -v -maxh 4',
        metadata={'description':
"""Gromacs mdrun options. Change to optimize performance. Attention! "deffnm",
"nobackup", and "noappend" flags (when required) are added automatically,
do not include them here. To optimize performance: bear in mind one CPU core
per task is dedicated to python for checking wether you have to stop.
Thus, if cpus_per_task=12, you must set -ntmpi 11 to maximize exec speed."""
                 })
    
    gmx_eneconv : str = field(
        default=f'printf "c\nc\n" | {GROMACS} -nobackup eneconv -settime',
        metadata={'description':
"""Command to merge Gromacs "edr" energy files of the backward and forward
trajectory segments of a two-way shooting simulations. Leave emtpy in case
you do not want to save energy files."""
                 })
    
    # toy engine
    
    toy_mdrun : Callable = field(
        default=None,
        metadata={'description':
"""Function for integrating to the next step in toy system.
Transforms a MDAnalysis timestep."""
                 })

    toy_slowdown : float = field(
        default=0.01,
        metadata={'description':
"""How much you slow down between integration steps with toy engine."""
                 })
    
    # other engines will be defined here
    # [...]
    
    # neural network and computation options
    
    network : NeuralNetworkModule = field(
        default=placeholder_network,
        metadata={'description':
"""Neural network model (used for logit committor estimates in AIMMD).
Placeholder just returns input's first dimension. Attention! Network's class
must be defined in the same params file you load."""
                 })

    values_function : Callable = field(
        default=None,
        metadata={'description':
"""From array of descriptors to their corresponding (logit committor)
values. If None: just evaluate the neural network on the descriptors."""
                 })

    descriptors_function : Callable = field(
        default=None,
        metadata={'description':
"""From MDAnalysis trajectory to array of descriptors (used by the model).
If None: just use positions, retrieved from trajectory every time."""
                 })
    
    descriptor_transform : Callable = field(
        default=None,
        metadata={'description':
"""Right before passing through NN input."""
                 })
    
    network_batch_size : int = field(
        default=4096,
        metadata={'description':
"""Compute at most `network_batch_size` frames at a time."""
                 })
    
    # initial paths
    
    initial_paths: List = field(
        default_factory=lambda: [],
        metadata={'description':
"""List of trajectory filenames or MDAnalysis trajectories or aimmd.Path
objects. They must contain transitions (checked automatically). Will
initialize an aimmd.PathEnsemble object."""
        })
    
    # shooting point selection options
    
    chain_type : str = field(
        default='rfps',
        metadata={'description':
"""either: 'tps' (transition path sampling) or 'rfps' (rejection-free
path sampling)."""
                 })
    
    selection_pool_size : int = field(
        default=10,
        metadata={'description':
"""Number of candidate paths per selection step. When `chain_tpye = 'tps'`,
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
logit committor. Must be >= 0."""
                 })

    cutoff_min : float = field(
        default=0.5,
        metadata={'description':
"""Do not go below `cutoff_min` absolute value with the first and last
finite bin boundaries."""
                 })
    
    cutoff_max : float = field(
        default=20.0,
        metadata={'description':
"""Do not exceed `cutoff_max` absolute value with the first and last
finite bin boundaries. When not runnig free simulations, not inf bins will
span from `-cutoff_max` to `cutoff_max`."""
                 })
    
    marginal_bins : str = field(
        default='all',
        metadata={'description':
"""Make additional bins boundaries at the initial/final interface (-inf/+inf),
sacrificing a bit exploitation for the sake of exploration. It makes sense
only when `selection_pool_size > 1`."""
                 })
    
    lorentzian: float = field(
        default=inf,
        metadata={'description':
"""Width of Lorentzian target distribution in logit committor space.
If inf: sample shooting points uniformly between first and last bin."""
                 })
    
    adjust_selection_in_bins : bool = field(
        default=True,
        metadata={'description':
"""Try to correct over-selection in bins, without breaking detailed balance.
Only effective when `selection_pool_size > 1`."""
                 })
    
    memory : float = field(
        default=1.0,
        metadata={'description':
"""Percent of shooting points to keep when computing the bins' populations,
used to accelerate the convergence of future shooting points selection."""
                 })
    
    free_overriding_states : str = field(
        default='',
        metadata={'description':
"""Allow occasional shooting point selection from the free simulations
around the states in `free_overriding_states`. If `all`: from all states.
Warning: unoptimized, will slow down selection process by a bit."""
                 })
    
    free_overriding_attempts : int = field(
        default=100,
        metadata={'description':
"""Number of overriding attempts per every selection."""
                 })

    free_overriding_recovery_rate : float = field(
        default=0.05,
        metadata={'description':
"""Attempt overriding from the same bin as the old shooting point with
this probability. Too high values break the Markov Chain of paths too often.
0.05 is a good compromise."""
                 })
    
    restart_free_simulations_with_transitions : str = field(
        default='',
        metadata={'description':
"""States where to use the last frames of a random sampled transition
instead of the latest crossing of the target state of the previous free
simulation."""
                 })
    
    # shooting point initialization options
    
    gen_temperature : float = field(
        default=300.0,
        metadata={'description':
"""Temperature velocity. If < 0, reuse velocities from the parent
trajectory (works if `trajectory_extension == '.trr'`)."""
                 })
    
    # simulation options
    
    max_length: int = field(
        default=50_000,
        metadata={'description':
"""Maximum allowed number of frames per path. In this way, avoid the
simulations getting stuck in long-lived intermediates."""
                 })
    
    extra_free_frames : int = field(
        default=0,
        metadata={'description':
"""When performing free simulations, do not stop right after reaching the
target state, but rather continue for furhter `extra_free_frames`.""" 
                 })
    
    # logit committor fit options
    
    fit : Callable = field(
        default=staticmethod(fit),
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
enough interval between frames. Attention! For that, `params.network` must be
an instance of `aimmd.network.Rescalable`
*Experimental*: use only if also running free simulations,
and at your own risk."""
                 })
    
    # paths reweighting options 
    
    reweight_parameters : dict = field(
        default_factory=lambda: {'free_threshold': 10},
        metadata={'description':
"""Dictionary of parameters used for reweighting the paths in a
`PathEnsemble` object, necessary for free energy and rates estimates.
Passed to `pathensemble.reweight`."""
                 })
    
    # save options

    trajectory_update_batch_size : int = field(
        default=1000,
        metadata={'description':
"""Register `trajectory_update_batch_size` frames at a time."""
                 })
    
    network_save_interval : int = field(
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
        default=PosixPath('.'),
        metadata={'description':
"""Will perform engine operations relative to `path`'s directory."""
                 })
