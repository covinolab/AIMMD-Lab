'''
AIMMD default parameter definitions.
'''

import shutil
import numpy as np
from tqdm import tqdm
from typing import Callable
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
class DefaultParams:
    
    # engine configuration
    
    topology : str = 'run.gro'
    """System's topology (gro file)."""
    
    trajectory_extension : str = '.xtc'
    """Use `xtc` for compressed data, `trr` for full-precision data and
    also saving velocities along with the positions."""
    
    # gromacs engine
    
    gmx_init_mdp : str = 'randomvelocities.mdp'
    """Used for initializing the shooting point's random velocities.
    Attention! Velocities will be extracted from trr files."""
    
    gmx_run_mdp : str = 'run.mdp'
    """Used for initializing production runs (shooting / free simulations).
    Attention! It must be consistent with "trajectory_extension"."""
    
    gmx_grompp : str = f'{GMX} grompp -maxwarn 1'
    """Gromacs grompp and call options. Change to include index files.
    Allows to produce `tpr` files. Attention! "coordinates", "restraints",
    "output", and "nobackup" flags are added automatically, do not include
    them here."""
    
    gmx_mdrun : str = f'{GMX} mdrun -v -maxh 4'
    """Gromacs mdrun options. Change to optimize performance. Attention!
    "deffnm", "nobackup", and "noappend" flags (when required) are added
    automatically, do not include them here."""
    
    gmx_eneconv : str = f'printf "c\nc\n" | {GMX} -nobackup eneconv -settime'
    """Command to merge Gromacs "edr" energy files of the backward and forward
    trajectory segments of a two-way shooting simulations. Leave emtpy in case
    you do not want to save energy files."""
    
    # toy engine
    
    toy_mdrun : str = f'{PYTHON} integrator.py'
    """Command to run a custom toy integrator. Attention! It must support
    the "noappend" flag, and produce trajectory files with the same naming
    convention as Gromacs."""
    
    # other engines will be defined here
    # [...]

    # user-defined functions for PathEnsemble objects
    
    states_function : Callable
    """From MDanalysis trajectory to array of states."""
    
    descriptors_function : Callable
    """From MDanalysis trajectory to array of descriptors (used by the
    model)."""
    
    values_function : Callable
    """From array of descriptors to their corresponding (logit committor)
    values."""

    # shooting point selection options
    
    do_tps : bool = False
    """If True: transition path sampling; if False: rejection-free sampling."""

    selection_pool_size : int = 10
    """Number of candidate paths per selection step. When `do_tps = True`,
    `selection_pool_size = 1` is the only option."""
    
    at_least_one_transition_in_pool : bool = False
    """If True: ensure each pool contains at least 1 transition.
    This breaks detailed balance, but avoid getting stuck close to the states
    and compromise exploitation."""
    
    nbins: int = 10
    """Number of bins partitioning the reactive region according to the
    logit committor."""
    
    cutoff_max : int = 20.0
    """Do not exceed `cutoff_max` absolute value with the first and last
    finite bin boundaries. When not runnig free simulations, bins will
    span from `-cutoff_max` to `cutoff_max`."""
    
    include_marginal_bins : bool = True
    """Include two extra bins at A/B boundaries, sacrificing a bit exploitation
    for exploration. It makes sense only when `selection_pool_size > 1`."""
    
    lorentzian: float = np.inf
    """Width of Lorentzian target distribution in logit committor space.
    If inf: sample shooting points uniformly between first and last bin."""

    adjust_selection_in_marginal_bins : bool = True
    """Try to correct over-selection in first and last bin, without breaking
    detailed balance."""
    
    memory : float = 1.0
    """Percent of shooting points to keep when computing the bins' populations,
    used to accelerate the convergence of future shooting points selection."""
    
    equilibrium_overriding_states : str = 'AB'
    """Allow occasional shooting point selection from the free simulations
    around the states in `equilibrium_overriding_states`."""
    
    equilibrium_overriding_rate : int = 100
    """Number of overriding attempts per every selection."""
    
    restart_equilibrium_with_transitions : str = ''
    """States where to use the last frames of a random sampled transition
    instead of the latest crossing of the target state of the previous free
    simulation."""
    
    # shooting point initialization options
    
    randomize_shooting_velocities : bool = True
    """If True, resample shooting velocities from a Boltzmann distribution as
    defined in `random_velocities`. Otherwise, reuse velocities from the parent
    trajectory (works if `trajectory_extension == '.trr'`)."""
    
    # simulation options
    
    max_excursion_length: int = 50_000
    """Maximum allowed number of frames per path. In this way, avoid the
    simulations getting stuck in long-lived intermediates."""
    
    extra_equilibriumA : list = []
    """Add the free simulations in the folders listed in `extra_equilibriumA`
    to the free simulations produced by this AIMMD run. Useful for recycling
    already produced data."""
    
    extra_equilibriumA_states_map : list = ['']
    """When adding the free simulations in the folders listed in
    `extra_equilibriumA`, for each of these folders, you may need to redefine
    the states, in case of multi-step transitions. For each string in
    `extra_equilibriumA_states_map` list, you can do this by adapting the following
    example. "AB BC ZA" means that A frames become B frames, B become C, and Z
    become A."""
    
    extra_equilibriumB : list = []
    """Same as `extra_equilibriumA` but for this run's state B."""
    
    extra_equilibriumB_states_map : list = ['']
    """Same as `extra_equilibriumB_states_map` but for this run's state B."""
    
    extra_extend_frames : int = 0
    """When performing extension simulations, do not stop right after reaching
    the target state, but rather continue for furhter `extra_extend_frames`."""
    
    # logit committor fit options
    
    fit : Callable = fit
    """Default fit function from utils."""
    
    rescale_committor : bool = True 
    """If True: rescale the NN output to recover the expected crossing
    probability behavior ~ `1/p` (from A) and ~ `1/(1 - p)` (from B),
    where p is the committor, in case of diffusive system with small
    enough interval between frames."""
    
    # paths reweighting options 
    
    reweight_parameters : dict = {'equilibrium_threshold': 10}
    '''
    Dictionary of parameters used for reweighting the paths in a `PathEnsemble`
    or `PathEnsemblesCollection` object, necessary for free energy and rates
    estimates. Passed to `pathensemble.reweight`.
    '''
    
    # save options
    
    save_interval : int = 10
    """Save NN model parameters every N shooting simulations."""
    
    # SLURM configuration (for HPC clusters)
    
    slurm_options : str = ('#SBATCH --ntasks-per-node=1\n'
                           '#SBATCH --mail-type=FAIL')
    """Default SLURM configuration. Attention! It must include the
    ntasks-per-node opton. Do not include #!/bin/bash, job-name, or number
    of nodes (will be determined automatically). Each two-way shooting and
    free simulation worker uses one task. Manager and trainer share an extra
    task together."""


# initialize
Params = DefaultParams()
