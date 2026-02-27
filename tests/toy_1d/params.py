import sys
import torch
import numpy as np
import warnings
from aimmd.network import fit as _fit
from aimmd.network.rescalable import Rescalable
from MDAnalysis.lib.distances import calc_bonds, calc_dihedrals

engine = 'toy'
initial_paths = 'initial.xtc'
free_overriding_states = 'all'

def toy_mdrun(ts):
    for _ in range(100):
        ts.positions = (ts.positions + .02 * np.random.normal()) % 10


def states_function(trajectory):
    result = []
    for frame in trajectory:
        import time
        x = frame.positions[0,0]
        if x < 1 or x > 9:
            result.append('A')
        elif x < 2:
            result.append('R')
        elif x < 3:
            result.append('B')
        elif x < 4:
            result.append('S')
        elif x < 5:
            result.append('C')
        elif x < 6:
            result.append('T')
        elif x < 7:
            result.append('D')
        elif x < 8:
            result.append('U')
        else:
            result.append('E')
    return np.array(result, dtype='<U1')


class Network(Rescalable):
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        n = 64
        self.input = torch.nn.Linear(1, n)
        self.layer = torch.nn.Linear(n, n)
        self.activation = torch.nn.ReLU(n)
        self.output = torch.nn.Linear(n, 1)
        self.reset_parameters()
    def forward(self, x):
        x = self.activation(self.input(x[:, :1]))
        x = self.activation(self.layer(x))
        x = self.output(x)
        return x
    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer.reset_parameters()
        self.output.reset_parameters()

network = Network()


"""It must be of these inputs"""
def fit(params,
        pathensemble,
        verbose=False,
        worker=None):
    return _fit(params,
        pathensemble,
        nbins=0,
        cutoff_min=0.5,
        cutoff_max=20.,
        state_bins='all',
        augment='no',
        lr=1e-3,
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
