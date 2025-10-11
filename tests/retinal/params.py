"""
Overridden parameters.
"""

trajectory_extension = '.trr'

"""
Functions
"""

from aimmd.core.utils import *
from aimmd.core.utils import fit as _fit

"""
Directly define
"""
couples = [(i,j) for i in range(80) for j in range(i+1, 80)]

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


import importlib.util
from pathlib import Path

def import_params(params_filename: str, ParamsClass):
    """
    Load a params.py file and return an instance of ParamsClass
    with attributes from the file.
    
    Args:
        params_filename: path to the Python file with parameters
        ParamsClass: the dataclass or class to instantiate
    
    Returns:
        An instance of ParamsClass with values from the file
    """
    params_path = Path(params_filename).resolve()

    spec = importlib.util.spec_from_file_location("params_module", str(params_path))
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)

    kwargs = {}
    for name in dir(params_module):
        if not name.startswith("__") and hasattr(ParamsClass, name):
            kwargs[name] = getattr(params_module, name)

    return ParamsClass(**kwargs)
