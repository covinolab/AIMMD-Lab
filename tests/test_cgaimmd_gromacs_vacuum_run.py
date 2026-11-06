"""A short PaiNN C-GAIMMD run on NANMA in vacuum with GROMACS (``--runslow``, GROMACS).

``tests/cgaimmd_nanma`` (alanine dipeptide, CHARMM22 without CMAP, 8 nm box)
with ``chain_type='cgaimmd'``, AIMMD's GROMACS engine and the candidate
preparation of a vacuum system: ``TopologyValidity`` as the validity hook (bond
graph, relabelling, the CA stereocentre, clashes against the unbiased free-MD
frames), no preparation hook (``placement='box_center'``: the generator atoms
are the whole system) and restrained energy minimization with
``GromacsMinimizer`` as the minimization hook. The network is a small
``JointPaiNN`` built with ``from_topology`` (element tokens, topology edges,
parity breaking: the consensus Chiro-PaiNN, at tiny widths), trained by
``flow_matching_fit``.

The PaiNN is untrained, so samples of its own ODE would not pass the validity
hook. The generator is therefore registered by the params file (method
``'test_nanma_stand_in'``, picked by name): it returns unbiased free-MD frames
of basin A with psi turned by 40 degrees (which puts them in R, next to A, so
that the shots are short), randomly rotated (proper rotations), relabelled
(the atoms of each element in a random order, which the validity hook undoes)
and with 0.01 Angstrom of noise. Everything else is the production path: the
committor values of the PaiNN, the target laws, the bootstrap acceptance, the
sidecars, the trainer and the flow-matching fit.

The run, in this process:

0. the preflight of the params file (``aimmd.cgaimmd.preflight``), with two
   generated candidates through validity and candidate preparation;
1. the trainer's first round: without ``initial_network`` and without
   shooting points it is the no-op round of a fresh run, which writes the
   constructed network as ``network<s>.h5`` (with its json) and the first
   bins;
2. free MD in A and B for a few seconds each (in-state frames for the fit);
3. two shots of one chain with that network;
4. a second trainer round: the PaiNN is trained on the two shots (five
   steps) from its seeded ``reset_parameters()`` start, and the trainer
   stores the fit's metadata in ``network<s>.json``.

The minimization hook also saves the frames it was given and returned
(``unminimized_<path>.npy``, ``minimized_<path>.npy``), so that the test can
check that each shot starts from the minimized frame, not from the generated
one. The same run in graph mode (``FlowGraphTransform``'s cache-free batch
dicts, the default of graph mode) is marked ``graph`` as well and must leave
no SQLite file.

Ported from the flow branch's ``tests/test_flow_gromacs_vacuum_run.py``
(bb98943), where the network's own ``generate`` returned the stand-in frames,
the network was a stand-in too, and only the no-op round ran.
"""

import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

mda = pytest.importorskip('MDAnalysis')
pytest.importorskip('networkx')

import aimmd                                                        # noqa: E402
from aimmd.cgaimmd.io import file_sha256                            # noqa: E402
from tests._cgaimmd_helpers import mdrun_thread_options             # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.gromacs]

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'
STATES = 'ARB'
PSI_TURN = 40.0                          # degrees: C7eq (psi ~ 73) -> R (psi ~ 113)
CA, CRP = 8, 14                          # the psi rotation axis (0-based, topol.top order)
PSI_SIDE = list(range(15, 22))           # OR NR HR CR HR1 HR2 HR3: the atoms beyond CRP
SHOT_FRAME_TOLERANCE = 0.02              # Angstrom: XTC rounding (<= 0.005) plus margin
FIT_STEPS = 5
FREE_SECONDS = 10                        # walltime of each free MD worker

PARAMS = '''\
"""PaiNN C-GAIMMD params of the NANMA vacuum run test ({mode} mode): tests/cgaimmd_nanma."""
import numpy as np
import torch

from aimmd.cgaimmd.flow_matching import flowmath
from aimmd.cgaimmd.flow_matching.fit import flow_matching_fit
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.generation import register_generator
from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer
from aimmd.cgaimmd.preparation.topology import read_atomic_numbers, read_masses
from aimmd.cgaimmd.preparation.validity import TopologyValidity
from states import states_function
{graph_imports}
GMX = {gmx!r}
engine = 'gromacs'
chain_type = 'cgaimmd'
states = 'ARB'
topology = 'conf.gro'
gmx_mdp = 'run.mdp'
gmx_grompp = f'{{GMX}} grompp -maxwarn 5 -p topol.top'
gmx_mdrun = f'{{GMX}} mdrun {threads} -nb cpu -pin off'
masses_from_topology = True              # the CHARMM CL/CLP carbons
initial_paths = ['free_md_A.xtc', 'free_md_B.xtc']
round_shooting_index = True
cgaimmd_generator = 'test_nanma_stand_in'
cgaimmd_parameters = {{'window_after_trained_shots': 100, 'generation_seed': 0, 'n_targets': 4,
                      'generation_device': 'cpu', 'values_device': 'cpu',
                      'training_device': 'cpu'}}

DESCRIPTORS = GeneratorDescriptors('conf.gro', GeneratorAtoms(n_solute=22),
                                   masses=read_masses('topol.top'))


def descriptors_function(trajectory):
    return DESCRIPTORS(trajectory)
{graph_transform}


class Network(JointPaiNN):
    """Local subclass: Params.save writes 'from <this module> import network'."""


torch.manual_seed(0)
network = Network.from_topology(
    'topol.top', DESCRIPTORS, embedding='element', topology_edges=True,
    coord_scale=2.0, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=5.0,
    cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
    velocity_radial=6, velocity_min_distance=0.7, parity_breaking=True, edge_backend='torch')

_STAND_IN = np.load({frames!r})          # (n, 22, 3) Angstrom, R frames next to A
_ELEMENTS = read_atomic_numbers('topol.top')
_ELEMENT_GROUPS = [np.flatnonzero(_ELEMENTS == z).tolist() for z in np.unique(_ELEMENTS)]


class StandInGenerator:
    """Rotated, relabelled, noised free-MD frames in R (the PaiNN is untrained)."""

    method = 'test_nanma_stand_in'

    def check(self, network):
        return []

    def settings(self):
        return {{'noise_A': 0.01}}

    @torch.no_grad()
    def generate(self, network, lam, *, generator=None):
        n = len(lam)
        frames = torch.as_tensor(_STAND_IN, dtype=torch.float32)
        pick = torch.randint(len(frames), (n,), generator=generator)
        x = frames[pick]
        x = x - x.mean(dim=1, keepdim=True)
        rotations = flowmath.random_rotations(n, generator=generator)
        x = torch.einsum('nij,naj->nai', rotations, x)
        # relabelled: the atoms of each element in a random order (a generator's slots hold
        # atoms of their own element; which of them is the validity hook's to find)
        order = torch.arange(x.shape[1]).repeat(n, 1)
        for group in _ELEMENT_GROUPS:
            group = torch.as_tensor(group)
            shuffled = torch.argsort(torch.rand((n, len(group)), generator=generator), dim=1)
            order[:, group] = group[shuffled]
        x = torch.gather(x, 1, order[:, :, None].expand(-1, -1, 3))
        return x + 0.01 * torch.randn(x.shape, generator=generator)


register_generator('test_nanma_stand_in', StandInGenerator)

_validity = TopologyValidity('topol.top', DESCRIPTORS,
                             clash_reference=['free_md_A.xtc', 'free_md_B.xtc'])


def cgaimmd_validity_function(positions, context):
    return _validity(positions, context)


_minimizer = GromacsMinimizer(em_mdp='em.mdp', restrain='generator_heavy', restraint_k=1e4,
                              emtol=100.0, nsteps=200)


def cgaimmd_minimize_function(frame, context):
    # the frames before and after EM, for the test (the last candidate of a path is its shot)
    out = _minimizer(frame, context)
    np.save(f'unminimized_{{context.path_name}}.npy', frame.positions)
    np.save(f'minimized_{{context.path_name}}.npy', out.positions)
    return out


def fit(params, pathensemble, verbose=False, worker=None):
    return flow_matching_fit(params, pathensemble, verbose=verbose, worker=worker,
                             graphs={graphs}, steps_per_round={steps}, relabel_every=2,
                             warmup_steps=0, n_sp=4, n_anchor=4, n_fm=4, idle_sleep_s=0.0)
'''

GRAPH_IMPORTS = '''from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
'''

GRAPH_TRANSFORM = '''

_graphs = FlowGraphTransform('conf.gro', 5.0, edge_backend='torch')   # cache-free batch dicts


def descriptor_transform(descriptors):
    return _graphs(descriptors)
'''


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """``Params.load`` leaves the params file's folder in ``sys.path``; take it out after the test.

    Otherwise a later ``import params`` or ``import states`` finds this module's files first.
    """
    monkeypatch.setattr(sys, 'path', list(sys.path))


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def rotate_about(positions, axis_from, axis_to, atoms, degrees):
    """`positions` with `atoms` turned by `degrees` about the axis ``axis_from -> axis_to``."""
    out = np.array(positions, dtype=np.float64)
    axis = out[axis_to] - out[axis_from]
    axis /= np.linalg.norm(axis)
    theta = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k
    moved = out[atoms] - out[axis_to]
    out[atoms] = moved @ rotation.T + out[axis_to]
    return out


def stand_in_frames(folder):
    """Free-MD frames of basin A, whole, with psi turned by PSI_TURN: (n, 22, 3), all in R.

    Read from the campaign `folder` (a copy of ``tests/cgaimmd_nanma``), so that
    MDAnalysis writes no offset files into the test data.
    """
    folder = Path(folder)
    sys.path.insert(0, str(folder))
    try:
        from states import dihedrals, states_function
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop('states', None)
    from MDAnalysis.coordinates.memory import MemoryReader

    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    from aimmd.cgaimmd.preparation.topology import read_masses
    descriptors = GeneratorDescriptors(str(folder / 'conf.gro'), GeneratorAtoms(22),
                                       masses=read_masses(str(folder / 'topol.top')))
    universe = mda.Universe(str(folder / 'conf.gro'), str(folder / 'free_md_A.xtc'))
    frames = []
    for ts in universe.trajectory:
        x = descriptors.select(ts.positions, ts.dimensions)
        frames.append(rotate_about(x, CA, CRP, PSI_SIDE, PSI_TURN))
    frames = np.array(frames, dtype=np.float32)
    box = np.array([80, 80, 80, 90, 90, 90], dtype=np.float32)
    states = states_function(MemoryReader(frames, dimensions=box))
    keep = states == 'R'
    assert keep.sum() >= 5, (states, dihedrals(MemoryReader(frames, dimensions=box)))
    return frames[keep]


def make_campaign(folder, mode, gmx):
    """The NANMA campaign folder of `mode` ('dense' or 'graph'): the test data and the params."""
    folder = Path(folder)
    shutil.copytree(NANMA, folder, ignore=shutil.ignore_patterns('__pycache__', '.*'))
    frames_file = folder / 'standin_frames.npy'
    np.save(frames_file, stand_in_frames(folder))
    graph = mode == 'graph'
    (folder / 'params.py').write_text(PARAMS.format(
        mode=mode, gmx=gmx, threads=mdrun_thread_options(gmx, 1), frames=str(frames_file),
        graphs=graph, steps=FIT_STEPS,
        graph_imports=GRAPH_IMPORTS if graph else '',
        graph_transform=GRAPH_TRANSFORM if graph else ''))
    return folder


def run_preflight_of(folder):
    """``aimmd.cgaimmd.preflight`` on the params file, as a job script runs it first: every
    check passes, two generated candidates go through validity and ``prepare_candidate``."""
    from aimmd.cgaimmd.preflight import run_preflight
    report = run_preflight(str(folder / 'params.py'), candidates=2,
                           scratch=str(folder / 'preflight'))
    assert report.ok, report.text()
    assert [check.status for check in report['candidate']] == ['OK'], report.text()
    return report


def run_campaign(folder, monkeypatch):
    """Preflight, no-op round, free MD in A and B, two shots, a trained round; returns
    (params, run, info)."""
    monkeypatch.chdir(folder)
    monkeypatch.syspath_prepend(str(folder))                 # the params file imports states
    sys.modules.pop('states', None)
    preflight = run_preflight_of(folder)
    sys.modules.pop('states', None)
    np.random.seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert params.chain_type == 'cgaimmd' and params.engine == 'gromacs'
    run = str(folder / 'run')
    aimmd.Worker(params, run, walltime=300).train(nrounds=1)
    network_file = os.path.join(run, f'network{STATES}.h5')
    assert os.path.isfile(network_file)
    assert os.path.isfile(os.path.join(run, f'bins{STATES}.npy'))
    first = {'sha256': file_sha256(network_file),
             'record': json.loads(Path(run, f'network{STATES}.json').read_text())}
    # free MD in A and B for a few seconds each (in-state frames for the fit; vacuum NANMA runs
    # at about 10 us/day, and the free worker counts its frames only between mdrun calls)
    for state in (0, 2):
        aimmd.Worker(params, run, nframes=20, walltime=FREE_SECONDS).free(state, 0, 1)
    aimmd.Worker(params, run, nsteps=2, walltime=900).shoot(1, 0)
    records = {}
    for fname in sorted(glob.glob(os.path.join(run, 'chainR0', 'path??????.xtc'))):
        records[fname] = json.loads(Path(f'{fname}.cgaimmd.json').read_text())
    aimmd.Worker(params, run, walltime=300).train(nrounds=1)
    second = json.loads(Path(run, f'network{STATES}.json').read_text())
    return params, run, dict(records=records, first=first, second=second, preflight=preflight)


def check_shots(params, run, records, first):
    """The two shots: candidate preparation, the GROMACS shot, the sidecar (FL's checks)."""
    assert len(records) == 2
    for fname, record in records.items():
        assert record['source'] == 'cgaimmd' and record['state_final'] == 'R'
        assert math.isfinite(record['lambda_registered'])
        assert record['acceptance']['rule'] == 'bootstrap_any_R'
        assert record['generator']['method'] == 'test_nanma_stand_in'
        assert record['generator']['settings'] == {'noise_A': 0.01}
        assert record['network']['sha256'] == first['sha256']      # the no-op round's network
        assert record['validity']['function'].endswith('cgaimmd_validity_function')
        validity = record['validity']['info']
        assert validity['n_stereocenters'] == 1                      # the CA of L-alanine
        assert validity['permutation_is_identity'] is False           # relabelled, restored
        assert record['prepare']['function'] == 'place_in_box'
        assert record['prepare']['placement'] == 'box_center'
        minimize = record['minimize']
        assert minimize['function'].endswith('cgaimmd_minimize_function') and minimize['ok']
        em = minimize['record']
        assert em['epot_after'] <= em['epot_before']
        assert em['max_displacement_A'] < 0.5                        # the heavy atoms held
        assert record['registration']['complete'] is True
        assert record['cost_seconds']['minimize'] > 0
        assert record['device'] == {'generation': 'cpu', 'values': 'cpu'}
        # the registered path: a GROMACS shot from the generated point, both ends in a basin
        states = aimmd.Path(fname).states
        index = record['registration']['shooting_index']
        assert states[0] in 'AB' and states[-1] in 'AB'
        assert states[index] == 'R'
        # the shot starts from the frame the minimization returned, not from its input
        name = os.path.splitext(record['path'])[0]
        minimized = np.load(os.path.join(params.parent, f'minimized_{name}.npy'))
        unminimized = np.load(os.path.join(params.parent, f'unminimized_{name}.npy'))
        universe = mda.Universe(os.path.join(params.parent, 'conf.gro'), fname)
        shot = universe.trajectory[index].positions.astype(float)
        to_minimized = float(np.abs(shot - minimized).max())
        to_input = float(np.abs(shot - unminimized).max())
        assert to_minimized <= SHOT_FRAME_TOLERANCE < to_input, (name, to_minimized, to_input)
    # the RFPS selection pool is not used under C-GAIMMD
    for pool_log in glob.glob(os.path.join(run, 'chainR*', 'pool.log')):
        assert os.path.getsize(pool_log) == 0


def check_rounds(run, info, graphs):
    """The no-op round and the round trained on the two shots, as network<s>.json records them."""
    from aimmd.cgaimmd.flow_matching.fit import FIT_NAME
    first, second = info['first']['record'], info['second']
    assert first['fit'] == FIT_NAME and first['status'] == 'no-op'
    assert first['round'] == 1 and first['n_trained_shots'] == 0
    assert second['fit'] == FIT_NAME and second['status'] == 'trained', second
    assert second['round'] == 2 and second['n_trained_shots'] == 2
    assert second['steps'] == FIT_STEPS
    assert second['start'] == {'kind': 'reset_parameters', 'seed': second['seed']}
    assert second['seed_source'] == 'generation_seed'
    assert second['settings']['graphs'] is graphs
    assert second['network']['class'].endswith('params.Network')       # a JointPaiNN
    assert second['network']['n_atoms'] == 22
    assert second['sha256'] == file_sha256(os.path.join(run, f'network{STATES}.h5'))
    assert second['sha256'] != info['first']['sha256']
    rounds = [json.loads(line) for line in
              Path(run, f'flow_fit{STATES}', 'rounds.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rounds] == ['no-op', 'trained']
    data = rounds[-1]['data']
    assert data['n_sp'] == 2 and data['zero_row_shooting_points'] == 0
    assert data['frames_by_state']['0'] > 0 and data['frames_by_state']['2'] > 0
    assert rounds[-1]['relabels_in_round'] >= 1


def test_nanma_vacuum_painn_run(tmp_path, monkeypatch, gmx_executable, restore_signals):
    """Two shots with validity, restrained GROMACS EM and the GROMACS engine, then the PaiNN
    trained on them (dense descriptors)."""
    folder = make_campaign(tmp_path / 'nanma', 'dense', gmx_executable)
    params, run, info = run_campaign(folder, monkeypatch)
    check_shots(params, run, info['records'], info['first'])
    check_rounds(run, info, graphs=False)


@pytest.mark.graph
def test_nanma_vacuum_painn_run_graph_mode(tmp_path, monkeypatch, gmx_executable,
                                           restore_signals):
    """The same run in graph mode: values and the fit through the cache-free batch dicts,
    and no SQLite graph cache anywhere."""
    folder = make_campaign(tmp_path / 'nanma_graph', 'graph', gmx_executable)
    params, run, info = run_campaign(folder, monkeypatch)
    check_shots(params, run, info['records'], info['first'])
    check_rounds(run, info, graphs=True)
    assert not glob.glob(str(tmp_path / '**' / '*.sqlite*'), recursive=True)
