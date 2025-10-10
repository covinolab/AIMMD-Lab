"""
AIMMD core parameter definitions.

This module provides default configuration values for both AIMMD sampling
and path sampling analysis. It defines simulation, reweighting, and engine
parameters, as well as placeholder (user-overridable) functions for
state identification, descriptor generation, and network output evaluation.
"""

import shutil
import numpy as np
from tqdm import tqdm

# =============================================================================
# GROMACS executable detection
# =============================================================================
GMX = shutil.which("gmx") or shutil.which("gmx_mpi")

if GMX is None:
    raise EnvironmentError(
        "GROMACS executable not found in PATH. Please install GROMACS and ensure "
        "'gmx' or 'gmx_mpi' is accessible in your environment."
    )

# =============================================================================
# Simulation parameters
# =============================================================================
max_excursion_length = 50_000  # Maximum allowed trajectory length (frames)
"""
Maximum number of frames a single path may contain. Acts as a safety cutoff
to prevent runaway simulations reaching long-lived intermediates.
"""

# =============================================================================
# Reweighting parameters
# =============================================================================
reweight_parameters = {"equilibrium_threshold": 10}
"""
Dictionary of parameters used for path reweighting (free energy / rate estimation).
Passed to `pathensemble.reweight`.
"""

# =============================================================================
# Extra sampling parameters
# =============================================================================
extra_equilibriumA = []
extra_equilibriumA_states_map = [""]  # TODO describe
extra_equilibriumB = []
extra_equilibriumB_states_map = [""]
extra_extend_frames = 0  # continue simulations beyond final state

# =============================================================================
# Sampling / shooting parameters
# =============================================================================
do_tps = False  # if True: transition path sampling; else, rejection-free sampling
lorentzian = np.inf  # target logit committor distribution (0-centered Lorentzian if finite)
nbins = 10  # number of bins in reactive coordinate space
cutoff_max = 20.0  # bounds for first/last bin if no free simulations
rescale_committor = True  # rescale NN outputs to approximate committor ~ 1/p
include_marginal_bins = True  # include two extra bins at A/B boundaries
adjust_selection_in_marginal_bins = True  # avoid over-selection in marginal bins
memory = 1.0  # statistical memory of shooting-point distributions (<1 = forget faster)
selection_pool_size = 10  # number of candidate paths per selection step
at_least_one_transition_in_pool = False  # ensure each pool contains ≥1 transition (breaks DB)
equilibrium_overriding_states = "AB"  # allow shooting from equilibrium trajectories
equilibrium_overriding_rate = 100  # number of overriding attempts per selection
restart_equilibrium_with_transitions = ""  # enforce restart from transition frames

# =============================================================================
# Initialization and velocity handling
# =============================================================================
randomize_shooting_velocities = True
"""
If True, resample shooting velocities from a Boltzmann distribution as defined
in `random_velocities`. Otherwise, reuse velocities from the parent trajectory
(works if trajectory_extension == ".trr").
"""

# =============================================================================
# Engine (GROMACS) configuration
# =============================================================================
topology = "run.gro"                 # system topology (all atoms/beads)
mdrun_parameters = "run.mdp"         # GROMACS MDP file for simulation
random_velocities = "randomvelocities.mdp"  # MDP for velocity initialization
grompp = f"{GMX} grompp -maxwarn 5"
mdrun = f"{GMX} mdrun -v -maxh 4"
eneconv = f'printf "c\\nc\\n" | {GMX} -nobackup eneconv -settime'

trajectory_extension = ".xtc"  # "xtc" or "trr"
save_interval = 10  # save NN model parameters every N shooting simulations

# =============================================================================
# SLURM configuration (for HPC clusters)
# =============================================================================
slurm_options = """#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
"""
"""
Default SLURM configuration. Do not include #!/bin/bash, job-name, or nodes.
Each two-way shooting and free simulation worker uses one task; manager and
trainer together share an extra task.
"""

# =============================================================================
# Default user-overridable functions in PathEnsemble objects
# =============================================================================
def states_function(trajectory, verbose=False):
    """
    Identify discrete states for each trajectory frame.

    Parameters
    ----------
    trajectory : MDAnalysis.Universe or list
        Input trajectory or list of frames.
    verbose : bool, optional
        Whether to show progress bar (default False).

    Returns
    -------
    np.ndarray of str
        Array of state identifiers ("A", "B", "R", etc.) for each frame.
    """
    return np.array(
        ["R" for _ in tqdm(trajectory, position=0, disable=not verbose)]
    )


def descriptors_function(trajectory, verbose=False):
    """
    Generate frame-level descriptors from trajectory coordinates.

    Parameters
    ----------
    trajectory : MDAnalysis.Universe or list
        Input trajectory or list of frames.
    verbose : bool, optional
        Whether to show progress bar (default False).

    Returns
    -------
    np.ndarray (n_frames, n_features)
        Descriptor matrix for all frames.
    """
    return np.array(
        [frame.positions.copy() for frame in tqdm(trajectory, position=0, disable=not verbose)]
    )


def values_function(descriptors, verbose=False):
    """
    Compute the raw neural network output ("logit" committor values).

    Parameters
    ----------
    descriptors : np.ndarray
        Output of `descriptors_function`.
    verbose : bool, optional
        Whether to show progress bar (default False).

    Returns
    -------
    np.ndarray
        Raw neural network outputs (logit committor values).
    """
    return descriptors[:, 0]


# =============================================================================
# Fit utility (imported from core.utils)
# =============================================================================
from .utils import fit
