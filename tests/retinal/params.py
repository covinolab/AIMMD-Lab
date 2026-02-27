"""
Directly defined params
"""

trajectory_extension = '.trr'
initial_paths = ['initial.trr']
gmx_grompp = 'gmx grompp -maxwarn 4'
gmx_mdrun = 'gmx mdrun -v -ntmpi 1'
rescale_committor = False
at_least_one_transition_in_pool = True
selection_pool_size = 20

"""
Auxiliary
"""

import numpy as np
couples = np.array([(i,j) for i in range(80) for j in range(i+1, 80)])

from MDAnalysis.lib.distances import calc_dihedrals
def cv(trajectory):
    """Isomerization dihedral, good for plots."""
    result = []
    for frame in trajectory:
        result.append(calc_dihedrals(*frame.positions[[33, 28, 26, 24]]))
    return np.array(result)

"""
States function
"""

def states_function(trajectory):
    result = []
    for frame in trajectory:
        dihedral = calc_dihedrals(*frame.positions[[33, 28, 26, 24]])
        if np.abs(dihedral) < np.pi/9:
            result.append('A')  # cis
        elif np.abs(dihedral - np.pi) < np.pi/9:
            result.append('B')  # trans
        elif np.abs(dihedral + np.pi) < np.pi/9:
            result.append('B')  # trans
        else:
            result.append('R')
    return np.array(result, dtype='<U1')

"""
Descriptors function & transform
"""

def descriptors_function(trajectory):
    result = np.zeros((len(trajectory), 80 * 6))
    i = 0
    for frame in trajectory:
        current = np.append(frame.positions.ravel(), frame._velocities.ravel())
        result[i, :len(current)] = current
        i += 1
    return result

from MDAnalysis.lib.distances import calc_bonds
def descriptor_transform(descriptors, couples=couples):
    return np.array([calc_bonds(
            frame[couples[:, 0]], frame[couples[:, 1]])
        for frame in descriptors[:, :80 * 3].reshape(-1, 80, 3)])

"""
Network
"""

import torch
from aimmd.network.rescalable import Rescalable
class Network(Rescalable):
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
        super().reset_parameters()
        self.input.reset_parameters()
        self.layer.reset_parameters()
        self.output.reset_parameters()

network = Network()

"""
Fit
"""

from aimmd.network import fit as _fit
def fit(params,
        pathensemble,
        verbose=False,
        worker=None):
    return _fit(params,
        pathensemble,
        nbins=10,
        cutoff_min=0.5,
        cutoff_max=20.,
        state_bins='all',
        augment='yes',
        lr=5e-4,
        loss_bayesian_factor=0,
        loss_smoothening_weight=0,
        loss_regularization_weight=0,
        epochs=500,
        batch_size=4096,
        stop=50.,
        train_validation_early_stopping=False,
        early_stopping_patience=10,
        early_stopping_min_samples=1000,
        early_stopping_split=0.1,
        in_memory=True,
        graphs=False,
        verbose=True,
        worker=worker)
