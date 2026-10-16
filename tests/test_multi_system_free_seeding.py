"""
Free-simulation seeding away from the boundary in a MULTI-SYSTEM run.

A `free_seeding_position` other than 'boundary' reads the untrimmed initial
path back from the file the params named (`Params.untrimmed_initial_paths`),
in the free worker. That worker is bound to one system (`_bind_system_id`):
its `states_function` evaluates that system's trajectories only, and its
trimmed copies in `<run>/<system>/initial<states>/` find their source by base
name. So it must see its own system's source files and no others.

It used to see one flat list over all systems, recorded by `Params.load`:

- it evaluated another system's file with its own states function, which
  fails when the atom counts differ (a real per-system states function writes
  the frame into that system's universe);
- it kept one file per base name, the first system's, so with every system's
  file named alike (inputs/<system>/initial.xtc) a worker whose system has the
  same atom count as the first one silently seeded from that system's path.

Tests
-----
test_load_records_the_initial_path_names_per_system
test_a_bound_worker_reads_only_its_own_systems_initial_path
test_a_bound_worker_never_takes_another_systems_file_of_equal_size
test_one_base_name_for_two_files_is_refused
test_multi_system_toy_run_seeds_every_system_from_its_own_path
"""
import os

import numpy as np
import pytest

import aimmd
from aimmd.worker._run import WorkerRun


PARAMS_SOURCE = '''
import numpy as np
import torch
from aimmd.network import fit as _fit
from aimmd.network.rescalable import Rescalable

engine = 'toy'
multi_system = True
multi_system_share_network = True
system_ids = ['s1', 's2']
topology = ['inputs/s1/initial.xtc', 'inputs/s2/initial.xtc']
initial_paths = [['inputs/s1/initial.xtc'], ['inputs/s2/initial.xtc']]
trainers_share_gpu = True
extra_free_frames = 0
free_overriding_states = 'all'
free_seeding_position = 'deepest'
free_restart_source = 'equilibrium'      # with the bias below: trial-9-like

# an in-state bias per system (reader mode, the toy stand-in for PLUMED's
# COLVAR), so the 'equilibrium' restart pool is reweighted by exp(bias)
record_bias = True
bias_source = 'reader'
bias_reactive_threshold = 0.5

N_ATOMS = N_ATOMS_LITERAL


def bias_function(trajectory, system_id=None):
    cut_a = 2.0 if system_id == 's1' else 2.5
    return np.array([0.7 if frame.positions[0, 0] < cut_a else 0.0
                     for frame in trajectory], dtype=float)


def toy_mdrun(ts):
    # clipped, not wrapped (% 10): a deepest seed sits at the edge of A, and
    # wrapping would join A to B there, which no physical CV does
    for _ in range(50):
        ts.positions = np.clip(ts.positions + .05 * np.random.normal(), 0, 10)


def states_function(trajectory, system_id=None):
    cut_a = 2.0 if system_id == 's1' else 2.5
    result = []
    for frame in trajectory:
        # a real per-system states function writes the frame into the
        # system's universe, which another system's frame does not fit
        if frame.positions.shape[0] != N_ATOMS[system_id]:
            raise ValueError(f'{system_id}: a frame of '
                             f'{frame.positions.shape[0]} atoms')
        x = frame.positions[0, 0]
        result.append('A' if x < cut_a else ('B' if x > 8.0 else 'R'))
    return np.array(result, dtype='<U1')


def descriptor_transform(coordinates, system_id=None):
    arr = np.asarray(coordinates)
    arr = arr.reshape(arr.shape[0], -1, 3)
    return arr[:, :1, 0]


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


def _write_initial_xtc(fname, n_atoms, start, stop):
    """A 50-frame toy path whose atom-0 x sweeps start..stop; A below 2-2.5,
    B above 8 under the test's states_function."""
    import MDAnalysis as mda
    universe = mda.Universe.empty(n_atoms, trajectory=True)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with mda.Writer(fname, n_atoms) as writer:
        for x in np.linspace(start, stop, 50):
            universe.atoms.positions = np.column_stack([
                np.full(n_atoms, x), np.full(n_atoms, 5.0),
                np.full(n_atoms, 5.0)]).astype(np.float32)
            writer.write(universe.atoms)


def _setup(folder, n_atoms):
    """Two systems, each initial path in its own folder under ONE base name.

    s1 runs A -> B from x = 0.5, s2 runs B -> A down to x = 0.1, so the
    deepest A frame of each is told apart by its position."""
    _write_initial_xtc(f'{folder}/inputs/s1/initial.xtc', n_atoms['s1'],
                       0.5, 10.0)
    _write_initial_xtc(f'{folder}/inputs/s2/initial.xtc', n_atoms['s2'],
                       9.9, 0.1)
    with open(f'{folder}/params.py', 'w') as handle:
        handle.write(PARAMS_SOURCE.replace('N_ATOMS_LITERAL', repr(n_atoms)))
    return f'{folder}/params.py'


class _BoundWorker:
    """What `WorkerRun._bind_system_id` needs of a worker."""

    def __init__(self, params, directory):
        self.params = params
        self._directory = directory


def _worker_params(path, system_id):
    """Params as a worker of `system_id` holds them (aimmd/worker/_helpers.py
    builds them with initial_paths=None, then `run` binds the system)."""
    params = aimmd.Params(str(path), initial_paths=None, save=False)
    WorkerRun._bind_system_id(_BoundWorker(params, f'run1/{system_id}'))
    return params


@pytest.fixture
def in_tmp(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(cwd)


def test_load_records_the_initial_path_names_per_system(in_tmp):
    path = _setup(str(in_tmp), {'s1': 1, 's2': 2})
    params = aimmd.Params(path, initial_paths=None, save=False)
    by_system = params.__dict__['_initial_path_files_by_system']
    assert by_system == [[os.path.abspath('inputs/s1/initial.xtc')],
                         [os.path.abspath('inputs/s2/initial.xtc')]]


def test_a_bound_worker_reads_only_its_own_systems_initial_path(in_tmp):
    path = _setup(str(in_tmp), {'s1': 1, 's2': 2})
    for system_id in ('s1', 's2'):
        params = _worker_params(path, system_id)
        got = params.untrimmed_initial_paths()
        assert set(got) == {'initial.xtc'}
        source = got['initial.xtc']
        assert os.path.abspath(source.fname) == os.path.abspath(
            f'inputs/{system_id}/initial.xtc')
        assert len(source) == 50 and set(source.states) == set('ARB')


def test_a_bound_worker_never_takes_another_systems_file_of_equal_size(in_tmp):
    path = _setup(str(in_tmp), {'s1': 2, 's2': 2})
    params = _worker_params(path, 's2')
    source = params.untrimmed_initial_paths()['initial.xtc']
    assert os.path.abspath(source.fname) == os.path.abspath(
        'inputs/s2/initial.xtc')
    # s2's own path runs B -> A, under s2's own cut
    assert source.states[0] == 'B' and source.states[-1] == 'A'


def test_one_base_name_for_two_files_is_refused(in_tmp):
    """Flattened over systems (or two folders of one system), one base name
    for two files would keep one of them and drop the other in silence."""
    path = _setup(str(in_tmp), {'s1': 1, 's2': 2})
    params = aimmd.Params(path, initial_paths=None, save=False)   # unbound
    with pytest.raises(TypeError, match='initial.xtc'):
        params.untrimmed_initial_paths()


def test_multi_system_toy_run_seeds_every_system_from_its_own_path(in_tmp):
    path = _setup(str(in_tmp), {'s1': 1, 's2': 2})
    launcher = aimmd.Launcher(path, 'run1')
    launcher.run(n=1, n1=1, n2=1, nsteps=4, walltime=180)
    # the deepest A frame of each system's OWN path: s1's A run leads (frame
    # 0), s2's trails (frame 49)
    for system_id, frame in (('s1', 0), ('s2', 49)):
        log = open(f'run1/{system_id}/freeA/worker0.log').read()
        assert 'Traceback' not in log, log[-3000:]
        assert ("First free simulation of 'A' is seeded at position 0.0 "
                f"of the in-A run, i.e. frame {frame} of the initial "
                "path") in log
        assert "free_restart_source = 'equilibrium'" in log
