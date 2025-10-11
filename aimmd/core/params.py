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
from .utils import fit  # base fit function

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
    
    # engine configuration
    
    topology = 'run.gro'
    """System's topology (gro file)."""
    
    trajectory_extension = '.xtc'
    """Use `xtc` for compressed data, `trr` for full-precision data and
    also saving velocities along with the positions."""
    
    # gromacs engine
    
    gmx_init_mdp = 'randomvelocities.mdp'
    """Used for initializing the shooting point's random velocities.
    Attention! Velocities will be extracted from trr files."""
    
    gmx_run_mdp = 'run.mdp'
    """Used for initializing production runs (shooting / free simulations).
    Attention! It must be consistent with "trajectory_extension"."""
    
    gmx_grompp = f'{GMX} grompp -maxwarn 1'
    """Gromacs grompp and call options. Change to include index files.
    Allows to produce `tpr` files. Attention! "coordinates", "restraints",
    "output", and "nobackup" flags are added automatically, do not include
    them here."""
    
    gmx_mdrun = f'{GMX} mdrun -v -maxh 4'
    """Gromacs mdrun options. Change to optimize performance. Attention!
    "deffnm", "nobackup", and "noappend" flags (when required) are added
    automatically, do not include them here."""
    
    gmx_eneconv = f'printf "c\nc\n" | {GMX} -nobackup eneconv -settime'
    """Command to merge Gromacs "edr" energy files of the backward and forward
    trajectory segments of a two-way shooting simulations. Leave emtpy in case
    you do not want to save energy files."""
    
    # toy engine
    
    toy_mdrun = f'{PYTHON} integrator.py'
    """Command to run a custom toy integrator. Attention! It must support
    the "noappend" flag, and produce trajectory files with the same naming
    convention as Gromacs."""
    
    # other engines will be defined here
    # [...]
    
    # simulation options
    
    max_excursion_length: int = 50_000
    """Maximum allowed number of frames per path. In this way, avoid the
    simulations getting stuck in long-lived intermediates."""

    # shooting point selection options
    
    do_tps: bool = False
    """If True: transition path sampling; if False: rejection-free sampling."""
    
    nbins: int = 10
    """Number of bins in partitioning the reactive region according to
    the logit committor."""
    
    cutoff_max = 20.0
    """Do not exceed `cutoff_max` absolute value with the first and last
    finite bin boundaries."""
    
    lorentzian: float = np.inf
    """Width of Lorentzian target distribution in logit committor space.
    If inf: sample shooting points uniformly between first and last bin."""

    # shooting point initialization options
    
    randomize_shooting_velocities = True
    """If True, resample shooting velocities from a Boltzmann distribution as defined
    in `random_velocities`. Otherwise, reuse velocities from the parent trajectory
    (works if `trajectory_extension == '.trr'`)."""
        
    # bounds for first/last bin if no free simulations
    include_marginal_bins = True  # include two extra bins at A/B boundaries
    adjust_selection_in_marginal_bins = True  # try to correct over-selection
    memory = 1.0
    
        
    
        reweight_parameters = {'equilibrium_threshold': 10}
        '''
        Dictionary of parameters used for reweighting the paths in a `PathEnsemble`
        or `PathEnsemblesCollection` object, necessary for free energy and rates
        estimates. Passed to `pathensemble.reweight`.
        '''
    
    rescale_committor = True  # rescale NN outputs to approximate committor ~ 1/p
    
    
    # sampling options
    
    
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
    
    # extra sampling parameters
    extra_equilibriumA = []
    extra_equilibriumA_states_map = ['']  # TODO describe
    extra_equilibriumB = []
    extra_equilibriumB_states_map = ['']
    extra_extend_frames = 0  # continue simulations beyond final state
    
    # save options
    
    save_interval = 10  # save NN model parameters every N shooting simulations
    
    # SLURM configuration (for HPC clusters)
    
    slurm_options = '''#SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=12
    #SBATCH --mail-type=FAIL
    #SBATCH --time=24:00:00
    """Default SLURM configuration. Do not include #!/bin/bash, job-name, or nodes.
    Each two-way shooting and free simulation worker uses one task; manager and
    trainer together share an extra task."""
    
    # user-defined functions for PathEnsemble objects
    
    states_function = None
    """From MDanalysis trajectory to array of states."""
    
    descriptors_function = None
    """From MDanalysis trajectory to array of descriptors (used by the model)."""
    
    values_function = None
    """From array of descriptors to their corresponding (logit committor)
    values."""
    
    fit = fit
    """Default fit function from utils."""

# initialize params
params = Params()
