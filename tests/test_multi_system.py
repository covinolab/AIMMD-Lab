"""
End-to-end test of MULTI-SYSTEM (multi-ligand) AIMMD on the toy engine.

This exercises the first-class multi-system machinery without GROMACS/GPU:

- one params file with ``multi_system=True`` driving TWO systems with DIFFERENT
  atom counts (1 and 2 atoms) — a fixed-width ``descriptor_transform`` maps both
  into the single shared network's input space,
- per-system subfolders ``run/<system_id>/`` created by the launcher,
- a shared network (``multi_system_share_network=True``): ONE trainer that hands
  the params ``fit`` a LIST of PathEnsembles and writes one ``networkARB.h5`` at
  the run root, read by every system's shooting workers,
- separate networks (``multi_system_share_network=False``): one trainer per
  system writing its own ``run/<system_id>/networkARB.h5``,
- per-system ``system_id`` threading into the state/descriptor/values functions,
- per-system kinetics-convergence (a results array with a ``system`` field).

It is a functionality test, not an accuracy test.
"""
import pytest


def _write_initial_xtc(fname, n_atoms):
    """Write a 50-frame toy trajectory whose atom-0 x-coordinate sweeps 0..10,
    i.e. a clean A -> R -> B transition for the test's states_function. Extra
    atoms (n_atoms > 1) sit at a constant position and are ignored by the
    state/descriptor functions (which only read atom 0)."""
    import numpy as np
    import MDAnalysis as mda
    n_frames = 50
    universe = mda.Universe.empty(n_atoms, trajectory=True)
    sweep = np.linspace(0.0, 10.0, n_frames)
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    with mda.Writer(fname, n_atoms) as writer:
        for i in range(n_frames):
            universe.atoms.positions = np.column_stack([
                np.full(n_atoms, sweep[i]),          # x: atom0 sweeps; others same
                np.full(n_atoms, 5.0),
                np.full(n_atoms, 5.0)]).astype(np.float32)
            writer.write(universe.atoms)


PARAMS_SOURCE = '''
import numpy as np
import torch
from aimmd.network import fit as _fit
from aimmd.network.rescalable import Rescalable

engine = 'toy'
multi_system = True
multi_system_share_network = SHARE_NETWORK
system_ids = ['s1', 's2']
topology = ['s1.xtc', 's2.xtc']          # different atom counts (1 vs 2)
initial_paths = [['s1.xtc'], ['s2.xtc']]
trainers_share_gpu = True
extra_free_frames = 0
free_overriding_states = 'all'


def toy_mdrun(ts):
    for _ in range(50):
        ts.positions = (ts.positions + .05 * np.random.normal()) % 10


def states_function(trajectory, system_id=None):
    # per-system A-boundary: s1 uses 2.0, s2 uses 2.5 (exercises system_id)
    cut_a = 2.0 if system_id == 's1' else 2.5
    result = []
    for frame in trajectory:
        x = frame.positions[0, 0]
        if x < cut_a:
            result.append('A')
        elif x > 8.0:
            result.append('B')
        else:
            result.append('R')
    return np.array(result, dtype='<U1')


def descriptor_transform(coordinates, system_id=None):
    # map any atom count to a fixed width (the atom-0 x-coordinate), so the one
    # shared network consumes both the 1-atom and the 2-atom system
    arr = np.asarray(coordinates)
    arr = arr.reshape(arr.shape[0], -1, 3)
    return arr[:, :1, 0]                  # shape (n_frames, 1)


class Network(Rescalable):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(1, 16)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(16, 1)
        self.reset_parameters()
    def forward(self, x):
        return self.output(self.activation(self.input(x[:, :1])))
    def reset_parameters(self):
        self.input.reset_parameters()
        self.output.reset_parameters()

network = Network()


def fit(params, pathensemble, verbose=False, worker=None):
    return _fit(params, pathensemble, nbins=0, state_bins='all', augment='no',
                lr=1e-3, loss_bayesian_factor=0, epochs=30, batch_size=128,
                stop=1e9, in_memory=True, graphs=False, verbose=verbose,
                worker=worker)
'''


def _setup(folder, share_network):
    import os
    os.makedirs(folder, exist_ok=True)
    _write_initial_xtc(f'{folder}/s1.xtc', n_atoms=1)
    _write_initial_xtc(f'{folder}/s2.xtc', n_atoms=2)
    with open(f'{folder}/params.py', 'w') as handle:
        handle.write(PARAMS_SOURCE.replace(
            'SHARE_NETWORK', 'True' if share_network else 'False'))


def test_multi_system_shared_network(tmp_path):
    import os
    import aimmd
    import numpy as np

    folder = str(tmp_path / 'shared')
    _setup(folder, share_network=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        # params plumbing
        assert params.multi_system and params.multi_system_share_network
        assert params.system_ids == ['s1', 's2']
        assert params.universe_of('s1').atoms.n_atoms == 1
        assert params.universe_of('s2').atoms.n_atoms == 2
        assert len(params.initial_paths) == 2          # one group per system

        # run a tiny shared-network campaign
        launcher = aimmd.Launcher('params.py', 'run1')
        launcher.run(n=1, n1=1, n2=1, nsteps=8, walltime=180)

        # per-system subfolders + per-system bins/densities
        for sid in ('s1', 's2'):
            assert os.path.isdir(f'run1/{sid}'), f'missing subfolder {sid}'
            assert os.path.isfile(f'run1/{sid}/binsARB.npy')
            assert os.path.isfile(f'run1/{sid}/densitiesARB.npy')
        # the ONE shared network lives at the run root, not in the subfolders
        assert os.path.isfile('run1/networkARB.h5'), 'shared network missing'
        assert not os.path.isfile('run1/s1/networkARB.h5')
        assert not os.path.isfile('run1/s2/networkARB.h5')
    finally:
        os.chdir(cwd)


def test_multi_system_separate_networks(tmp_path):
    import os
    import aimmd

    folder = str(tmp_path / 'separate')
    _setup(folder, share_network=False)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        assert params.multi_system and not params.multi_system_share_network

        launcher = aimmd.Launcher('params.py', 'run1')
        launcher.run(n=1, n1=1, n2=1, nsteps=8, walltime=180)

        # each system trains its OWN network in its own subfolder
        for sid in ('s1', 's2'):
            assert os.path.isfile(f'run1/{sid}/networkARB.h5'), \
                f'per-system network missing for {sid}'
        # and there is no shared network at the root
        assert not os.path.isfile('run1/networkARB.h5')
    finally:
        os.chdir(cwd)


def test_multi_system_kinetics_convergence(tmp_path):
    import os
    import aimmd
    import numpy as np

    folder = str(tmp_path / 'kcv')
    _setup(folder, share_network=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        # seed a little data with a short shared campaign
        aimmd.Launcher('params.py', 'run1').run(
            n=1, n1=1, n2=1, nsteps=8, walltime=180)
        # per-system kinetics convergence (shared network)
        worker = aimmd.Worker(params, 'run1', walltime=180)
        results = worker.kinetics_convergence(fractions=[0.5, 1.0])
        assert 'system' in results.dtype.names
        assert set(results['system']) == {'s1', 's2'}
        assert len(results) == 2 * 2                   # fractions x systems
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    import tempfile, pathlib
    for fn in (test_multi_system_shared_network,
               test_multi_system_separate_networks,
               test_multi_system_kinetics_convergence):
        with tempfile.TemporaryDirectory() as d:
            fn(pathlib.Path(d))
            print(f'{fn.__name__} OK')
