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


# Optional OPES-style in-state bias (reader mode). The per-frame bias is nonzero
# only inside state A (and per-system via system_id), so it is negligible in the
# reactive region R (Tiwary-Parrinello assumption) and the bias check passes.
# bias_reactive_threshold is given as a per-system LIST to exercise that path.
BIAS_SUFFIX = '''
record_bias = True
bias_source = 'reader'
bias_reactive_threshold = [0.5, 0.3]


def bias_function(trajectory, system_id=None):
    cut_a = 2.0 if system_id == 's1' else 2.5
    result = []
    for frame in trajectory:
        x = frame.positions[0, 0]
        result.append(0.7 if x < cut_a else 0.0)   # bias in kT, inside A only
    return np.array(result, dtype=float)
'''


# Optional value-pass subsampling caps (Feature B). Tiny caps so the bounded
# eval-ensemble slice is exercised through the trainer even on a short toy run.
CAPS_SUFFIX = '''
subsample_caps = {'shot': 2, 'free': 2, 'in_state': 50}
'''


def _setup(folder, share_network, with_bias=False, with_caps=False):
    import os
    os.makedirs(folder, exist_ok=True)
    _write_initial_xtc(f'{folder}/s1.xtc', n_atoms=1)
    _write_initial_xtc(f'{folder}/s2.xtc', n_atoms=2)
    source = PARAMS_SOURCE.replace(
        'SHARE_NETWORK', 'True' if share_network else 'False')
    if with_bias:
        source = source + BIAS_SUFFIX
    if with_caps:
        source = source + CAPS_SUFFIX
    with open(f'{folder}/params.py', 'w') as handle:
        handle.write(source)


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


def test_multi_system_per_system_worker_counts(tmp_path):
    """Per-system worker counts: n may be a scalar (uniform) or a per-system
    list. With n=[[2, 1]] the first system gets 2 shooters and the second 1,
    plus one shared trainer at the run root. Verified at _build() level (no MD)."""
    import os
    import aimmd

    folder = str(tmp_path / 'counts')
    _setup(folder, share_network=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        launcher = aimmd.Launcher('params.py', 'run1')
        launcher._update(n=[[2, 1]], n1=1, n2=1, nrounds=1)
        args, descriptions = launcher._build()
        s1_shoot = sum('run1/s1' in d and 'chainR' in d for d in descriptions)
        s2_shoot = sum('run1/s2' in d and 'chainR' in d for d in descriptions)
        trainers = sum('trainer' in d for d in descriptions)
        assert s1_shoot == 2, descriptions
        assert s2_shoot == 1, descriptions
        assert trainers == 1                     # one shared trainer at the root
        assert any(d == '"run1" ARB trainer' for d in descriptions)
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


def test_multi_system_bias_shared_network(tmp_path):
    """OPES-style in-state bias with a shared multi-system network: the trainer
    builds a per-system bias cache (reader mode, system_id forwarded) and runs
    the Tiwary-Parrinello rate correction. Asserts per-system `<traj>.bias.npy`
    caches appear and the shared network is written."""
    import os
    import glob
    import aimmd

    folder = str(tmp_path / 'bias_shared')
    _setup(folder, share_network=True, with_bias=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        assert params.record_bias and params.bias_source == 'reader'
        # per-system threshold resolves from the list
        assert params.bias_reactive_threshold_of('s1') == 0.5
        assert params.bias_reactive_threshold_of('s2') == 0.3

        aimmd.Launcher('params.py', 'run1').run(
            n=1, n1=1, n2=1, nsteps=8, walltime=180)

        assert os.path.isfile('run1/networkARB.h5'), 'shared network missing'
        # the per-system bias cache must have been written for both systems
        for sid in ('s1', 's2'):
            caches = glob.glob(f'run1/{sid}/**/*.bias.npy', recursive=True)
            assert caches, f'no bias cache written for system {sid}'
    finally:
        os.chdir(cwd)


def test_multi_system_bias_kinetics_convergence(tmp_path):
    """Kinetics convergence with a shared network + in-state bias fills the
    per-system Tiwary-Parrinello k12_rw/k21_rw columns (finite, not nan)."""
    import os
    import aimmd
    import numpy as np

    folder = str(tmp_path / 'bias_kcv')
    _setup(folder, share_network=True, with_bias=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        aimmd.Launcher('params.py', 'run1').run(
            n=1, n1=1, n2=1, nsteps=8, walltime=180)
        worker = aimmd.Worker(params, 'run1', walltime=180)
        results = worker.kinetics_convergence(fractions=[1.0])
        assert set(results['system']) == {'s1', 's2'}
        # the bias-reweighted columns are populated (would stay nan without the
        # multi-system bias path)
        assert np.isfinite(results['k12_rw']).all(), results
        assert np.isfinite(results['k21_rw']).all(), results
    finally:
        os.chdir(cwd)


def test_network_saved_right_after_training(tmp_path, monkeypatch):
    """The trained network is persisted immediately after `fit`, BEFORE the
    (potentially long) value pass + reweighting. We make `compute_bins` explode
    right after training and assert the shared network was still written."""
    import os
    import aimmd
    import aimmd.worker._train as train_mod

    folder = str(tmp_path / 'saveafter')
    _setup(folder, share_network=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        # seed data + an initial trained network, then remove it so a fresh save
        # is required this round.
        aimmd.Launcher('params.py', 'run1').run(
            n=1, n1=1, n2=1, nsteps=8, walltime=180)
        assert os.path.isfile('run1/networkARB.h5')
        os.remove('run1/networkARB.h5')

        # blow up the post-training reweighting step
        def boom(*args, **kwargs):
            raise RuntimeError('boom after training')
        monkeypatch.setattr(train_mod, 'compute_bins', boom)

        params = aimmd.Params.load('params.py')
        try:
            aimmd.Worker(params, 'run1', walltime=180).train(nrounds=1)
        except Exception:
            pass  # the crash is expected; what matters is what's on disk

        # the network exists despite the crash -> saved right after training
        assert os.path.isfile('run1/networkARB.h5'), \
            'network was not saved before the post-training reweighting'
    finally:
        os.chdir(cwd)


def test_multi_system_subsample_caps_smoke(tmp_path):
    """A shared multi-system campaign with `subsample_caps` set runs end-to-end:
    the trainer builds a bounded per-system eval ensemble (the value pass / bins /
    reweighting run on it) while `fit` still uses the full ensemble, and the
    shared network + per-system bins are produced."""
    import os
    import aimmd

    folder = str(tmp_path / 'caps')
    _setup(folder, share_network=True, with_caps=True)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params = aimmd.Params.load('params.py')
        assert params.subsample_caps_of('s1') == {'shot': 2, 'free': 2,
                                                  'in_state': 50}
        aimmd.Launcher('params.py', 'run1').run(
            n=1, n1=1, n2=1, nsteps=8, walltime=180)
        assert os.path.isfile('run1/networkARB.h5')
        for sid in ('s1', 's2'):
            assert os.path.isfile(f'run1/{sid}/binsARB.npy')
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    import tempfile, pathlib
    for fn in (test_multi_system_shared_network,
               test_multi_system_separate_networks,
               test_multi_system_kinetics_convergence,
               test_multi_system_bias_shared_network,
               test_multi_system_bias_kinetics_convergence,
               test_multi_system_subsample_caps_smoke):
        with tempfile.TemporaryDirectory() as d:
            fn(pathlib.Path(d))
            print(f'{fn.__name__} OK')

