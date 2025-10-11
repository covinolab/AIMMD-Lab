'''
AIMMD core parameter definitions.

This module provides default configuration values for both AIMMD sampling
and path sampling analysis. It defines simulation, reweighting, and engine
parameters, as well as placeholder (user-overridable) functions for
state identification, descriptor generation, and network output evaluation.

TODO homogenize syntax
'''

import shutil
import numpy as np
from tqdm import tqdm

# =============================================================================
# Simulation parameters
# =============================================================================
max_excursion_length = 50_000  # Maximum allowed trajectory length (frames)
'''
Maximum number of frames a single path may contain. Acts as a safety cutoff
to prevent runaway simulations reaching long-lived intermediates.
'''

# =============================================================================
# Reweighting parameters
# =============================================================================
reweight_parameters = {'equilibrium_threshold': 10}
'''
Dictionary of parameters used for path reweighting (free energy and rate
estimation). Passed to `pathensemble.reweight`.
'''

# =============================================================================
# Extra sampling parameters
# =============================================================================
extra_equilibriumA = []
extra_equilibriumA_states_map = ['']  # TODO describe
extra_equilibriumB = []
extra_equilibriumB_states_map = ['']
extra_extend_frames = 0  # continue simulations beyond final state

# =============================================================================
# Sampling / shooting parameters
# =============================================================================
do_tps = False
'''
If True: transition path sampling; else, rejection-free sampling.
'''
lorentzian = np.inf
'''
Target logit committor distribution.
If inf: uniform between first and last interface in reactive space.
If < inf: Lorentzian distribution of |lorentzian| width.
'''
nbins = 10  # number of bins in reactive coordinate space
cutoff_max = 20.0  # bounds for first/last bin if no free simulations
rescale_committor = True  # rescale NN outputs to approximate committor ~ 1/p
include_marginal_bins = True  # include two extra bins at A/B boundaries
adjust_selection_in_marginal_bins = True  # try to correct over-selection
memory = 1.0
'''
Percent of shooting points to keep when computing the bins' populations, used
to accelerate the convergence of the selection of the next shooting points.
'''
selection_pool_size = 10  # number of candidate paths per selection step
at_least_one_transition_in_pool = False
'''
If True: ensure each pool contains at least 1 transition.
This breaks detailed balance, but avoid getting stuck close to the states
and compromise exploitation.
'''
equilibrium_overriding_states = 'AB'
'''
Allow occasional shooting point selection from the free simulations around
the states in `equilibrium_overriding_states`.
'''
equilibrium_overriding_rate = 100
'''
Number of overriding attempts per every selection.
'''
restart_equilibrium_with_transitions = ''
'''
Use the last frames of a random sampled transition instead of the latest
crossing of the target state of the previous free simulation.
'''

# =============================================================================
# Initialization and velocity handling
# =============================================================================
randomize_shooting_velocities = True
'''
If True, resample shooting velocities from a Boltzmann distribution as defined
in `random_velocities`. Otherwise, reuse velocities from the parent trajectory
(works if trajectory_extension == '.trr').
'''

# =============================================================================
# Engine configuration
# =============================================================================
topology = 'run.gro'                 # system topology
trajectory_extension = '.xtc'  # 'xtc' or 'trr'

# =============================================================================
# GROMACS engine
# =============================================================================
GMX = shutil.which('gmx') or shutil.which('gmx_mpi')

if GMX is None:
    raise EnvironmentError(
        'GROMACS executable not found in PATH. Please install GROMACS and '
        'ensure \'gmx\' or \'gmx_mpi\' is accessible in your environment.'
    )

mdrun_parameters = 'run.mdp'         # GROMACS MDP file for simulation

random_velocities = 'randomvelocities.mdp'  # MDP for velocity initialization
'''
...
'''
grompp = f'{GMX} grompp -maxwarn 5'  # call grompp options
'''
...
'''
mdrun = f'{GMX} mdrun -v -maxh 4'  # call mdrun options (optimize performance)
eneconv = f'printf 'c\\nc\\n' | {GMX} -nobackup eneconv -settime'
'''
...
'''

# =============================================================================
# Toy engine
# =============================================================================

toyrun = f'...'

# =============================================================================
# Saving options
# =============================================================================
save_interval = 10  # save NN model parameters every N shooting simulations

# =============================================================================
# SLURM configuration (for HPC clusters)
# =============================================================================
slurm_options = '''#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
'''
'''
Default SLURM configuration. Do not include #!/bin/bash, job-name, or nodes.
Each two-way shooting and free simulation worker uses one task; manager and
trainer together share an extra task.
'''

# =============================================================================
# Default user-overridable functions in PathEnsemble objects
# =============================================================================
def states_function(trajectory, verbose=False):
    raise NotImplementedError


def descriptors_function(trajectory, verbose=False):
    raise NotImplementedError


def values_function(descriptors, verbose=False):
    raise NotImplementedError

# =============================================================================
# Fit utility (imported from core.utils)
# =============================================================================
from .utils import fit
