from aimmd.core.utils import *
from aimmd.core.utils import fit as _fit

GMX = shutil.which('gmx') or shutil.which('gmx_mpi')

if GMX is None:
    raise EnvironmentError(
        'GROMACS executable not found in PATH. Please install '
        'GROMACS and ensure \'gmx\' or \'gmx_mpi\' is in your PATH.'
    )

mdtraj_frame = md.load('run.gro')
couples = [(i,j) for i in range(mdtraj_frame.n_atoms)
            for j in range(i+1, mdtraj_frame.n_atoms)]

def cv(trajectory, verbose=False):
    """Isomerization dihedral, good for plots."""
    result = []
    for frame in tqdm(trajectory, disable=not verbose, position=0):
        positions = np.round(frame.positions / 10., 2)
        mdtraj_frame.xyz = positions.reshape((-1, *positions.shape))
        mdtraj_frame.unitcell_vectors = np.round(
            frame.triclinic_dimensions / 10., 2).reshape((-1, 3, 3))
        dihedral = md.compute_dihedrals(mdtraj_frame, [[33,28,26,24]])[0]
        result.append(dihedral[0])
    return np.array(result)


def states_function(trajectory, verbose=False):
    result = []
    for frame in tqdm(trajectory, disable=not verbose, position=0):
        positions = np.round(frame.positions / 10., 2)
        mdtraj_frame.xyz = positions.reshape((-1, *positions.shape))
        mdtraj_frame.unitcell_vectors = np.round(
            frame.triclinic_dimensions / 10., 2).reshape((-1, 3, 3))
        dihedral = md.compute_dihedrals(mdtraj_frame, [[33,28,26,24]])[0]
        if np.abs(dihedral) < np.pi/9:
            result.append('A')  # cis
        elif np.abs(dihedral - np.pi) < np.pi/9:
            result.append('B')  # trans
        elif np.abs(dihedral + np.pi) < np.pi/9:
            result.append('B')  # trans
        else:
            result.append('R')
    return np.array(result, dtype='<U1')


def descriptors_function(trajectory, verbose=False):
    result = []
    for frame in tqdm(trajectory, disable=not verbose, position=0):
        result.append(np.append(frame.positions.ravel(),
                     frame._velocities.ravel()))
    if not len(result):
        result = np.zeros((1, 0))
    return np.array(result)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        n = 64
        self.input = torch.nn.Linear(len(couples), n)
        self.layer = torch.nn.Linear(n, n)
        self.activation = torch.nn.ReLU(n)
        self.output = torch.nn.Linear(n, 1)
        self.reset_parameters()
    def forward(self, x):
        x = self.activation(self.input(x))
        x = self.activation(self.layer(x))
        x = self.output(x)
        return x
    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer.reset_parameters()
        self.output.reset_parameters()

network = Network()


def process_descriptors(descriptors):
    """From positions to distances"""
    positions = descriptors[:, :80*3]
    mdtraj_frame.xyz = positions.reshape((len(descriptors), 80, 3)) / 10.
    mdtraj_frame.unitcell_vectors = np.repeat(
        [mdtraj_frame.unitcell_vectors[0]], len(mdtraj_frame.xyz), axis=0)
    return md.compute_distances(mdtraj_frame, couples)


def values_function(descriptors):
    if not len(descriptors):
        return np.zeros(0)
    
    global network
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    network.eval()
    
    # initialize
    results = []
    descriptors = process_descriptors(descriptors)
    
    # compute in batches
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            descriptors, batch_size=4096, shuffle=False):
            batch = batch.to(device=device, dtype=dtype)
            output = network(batch).detach().cpu().numpy().ravel()
            results.append(output)
    
    # return
    return np.concatenate(results)

def fit(network, pathensemble, initial_path=None, verbose=False,
        keys=None, save_memory=False):
    return _fit(network, pathensemble,
        lr=1e-2,
        epochs=100,
        nbins=9,
        state_bins='AB',
        initial_path=initial_path,
        verbose=verbose,
        keys=keys,
        save_memory=False,
        process_descriptors=process_descriptors,
        augment=True,
        loss_bayesian_factor=100)

"""
We collect all functions and AIMMD parameters in this dictionary.
Attention! The description of what each parameter does is still partial.
Please write to lazzeri@fias.uni-frankfurt.de for clarification.
"""

aimmd_run_params = {
    
    # store functions and objects defined or initialized above
    'states_function': states_function,
    'descriptors_function': descriptors_function,
    'values_function': values_function,
    'network': network,
    'fit': fit,
    
    # simulation options
    'max_excursion_length': 50000,   # maximum number of frames
        # a path is allowed to have (usually never met, it is
        # a safety option in case you reached an unknown long-lived
        # intermediate with your simulations)
    
    # reweighting options
    'reweight_parameters': {'equilibrium_threshold': 5},
       # used for reweighting the AIMMD paths to estimate the free
       # energy and rates of the studied transition, one of the
       # inputs of the pathensemble.reweight function

    # extra sampling options
    # (get free simulations data from other sources than your
    #  current AIMMD sampling folder)
    'extra_equilibriumA': [],
    'extra_equilibriumA_states_map': [''],
    'extra_equilibriumB': [],
    'extra_equilibriumB_states_map': [],
    'extra_extend_frames': 30,  # continue extension
       # simulations for a few frames after reaching
       # the final state
    
    # shooting point selection?
    'do_tps': False,  # if True: do transition path sampling;
       # otherwise, do rejection-free path sampling
       # (Lazzeri, Bolhuis, Covino, arXiv 2025)
    'lorentzian': np.inf,  # if < inf: aim at a shooting point values
       # distribution that follows a 0-centered Lorentzian in the
       # logit committor space; if inf: aim at a distribution where
       # all bins are equally populated
    'nbins': 10,  # partition the reactive space in `nbins` bins;
       # the first and last bin isosurfaces in the reactive space are
       # dinamically adapted based on the free simulations or fixed
       # by the parameter "cutoff_max" in case there are no free
       # simulations in the AIMMD run
    'cutoff_max': 20.,  # the first bin boundary in the reactive space
       # cannot get below -cutoff_max; the last one cannot get above
       # +cutoff_max
    'rescale_committor': False,  # if True, transform the raw
       # neural network output such that the crossing probability
       # follows as much as possible a ~1/p law, as the committor
       # theory for diffusive dynamics states.
       # Attention! For rescaling to be effective, you need to have
       # implemented `rescale_knots` and `rescale_values` parameters
       # in your network architecture.
    'include_marginal_bins': True,  # if True, two additional bins will
       # be included, bringing the total to `nbins + 2`; the first bin
       # will then start at the boundary of state A, and the last one
       # will end at the boundary of state B; otherwise, the first and
       # last boundaries would be deeper in the reactive region
       # True is the recommended choice when `do_tps = False` and
       # `selection_pool_size = 1`, when you do not run free
       # simulations, or when you are mostly interested in the
       # transitions alone, to minimize the risk of shooting
       # simulations getting stuck near the state boundaries
    'adjust_selection_in_marginal_bins': True,  # if True, try to
       # prevent selecting shooting points in the first and the last
       # bin in case you have already selected too many
    'memory': 1.,  # if < 1: (partially) "forget" the shooting point
       # statistics when trying to enforce the desired distribution
    'selection_pool_size': 10,  # for each shooting chain:
       # selection_pool_size is the number of paths in the chain
       # from which you can select the next shooting point
       # selection_pool_size >> 1 prevents the situation where paths
       # paths get stuck close to the states
       # Attention! selection_pool_size > 1 works only with the
       # rejection-free path sampling algorithm (detailed balance
       # still enforced)
    'at_least_one_transition_in_pool': False,  # enforce each selection
       # pool to have at least one transition in it
       # Attention! this move breaks detailed balance; when enforced
       # too frequently, it undermines the results
    'equilibrium_overriding_states': 'AB',  # string of allowed states
       # (e.g., "AB"): allow shooting point selection from free
       # trajectories, thus "breaking" and reinitiating the
       # shooting chain, for the sake of improved exploration
    'equilibrium_overriding_rate': 100.,  # the number of equilibrium
       # overriding attempts during each shooting point selection
       # process; anything between 0 and 100 is fine
    'restart_equilibrium_with_transitions': '',  # string of enforced
       # states (e.g., "AB"): enforce the first frame of a new free
       # trajectory to be initialized from the end of a transition
    
    # initialization
    'randomize_shooting_velocities': True,  # if True, resample the
       # shooting point's initial velocities according to the
       # Boltzmann distribution as specified in the 
       # "random_velocities" gromacs mdp options; otherwise use
       # the same velocities as the original trajectory from which
       # the shooting point was selected (works if trajectory_extension
       # is ".trr")
    
    # engine
    'topology': 'run.gro',  # gro file of the system (ALL atoms/beads)
    'mdrun_parameters': 'run.mdp',  # file with gromacs mdp options 
       # for running the AIMMD simulations; leave empty in case of
       # using a custom-written 2D engine
    'random_velocities': 'randomvelocities.mdp',  # mdp options
       # for (re)initializing the shooting point's velocities
    'grompp': f'{GMX} grompp -maxwarn 3 -p topol.top',  # command
       # for creating gromacs "tpr" files that works in your machine
       # Attention! Do not include "nobackup" and other options
       # already defined in utils.py; leave empty in case of using
       # a custom-written 2D engine
    'mdrun': f'{GMX} mdrun',
       # running/extending simulations that works in your machine
       # Attention! adapt in case of using a custom-written 2D engine
    'eneconv': f'printf "c\nc\n" | {GMX} -nobackup eneconv -settime',
       # command for merging the "edr" energy files of the backward
       # and forward trajectory segments of a two-way shooting
       # simulations; leave emtpy in case you do not want to save
       # energy files
    
    # storage options
    'trajectory_extension': '.trr',  # either "xtc" or "trr"
    'save_interval': 10,  # save neural network model parameters
       # every "save_interval" two-way shooting simulations
    
    'slurm_options': """#SBATCH --partition=booster
#SBATCH --account=sispli
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00"""  # slurm options (if running on cluster)
       # do not include #!/bin/bash and the options job-name, nodes
       # include ntasks-per-node; every two-way shooting and free
       # simulation worker gets one task; manager and trainer together
       # get one additional task
}
