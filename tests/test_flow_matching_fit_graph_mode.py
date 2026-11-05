"""The flow-matching fit in graph mode (``graphs=True``): MD frames through ``params.descriptor_transform``.

The default route of graph mode (:class:`~aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`
without ``graph_cache``) returns one batch dict per call and needs neither
PyG nor a cache: its tests run in the default lane. A graph cache written
the flow branch's way (``graph_cache=..., write_graph_cache=True``) needs
``aimmd.network.graph_utils`` and PyG: those tests are marked ``graph``
(``--rungraph``).
"""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd.flow_matching import fit as ff
from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble.utils import assemble_pathensemble
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace
from tests._helpers_unit import write_trajectory


STATES = 'ARB'
CUTOFF = 4.0
FAST = dict(steps_per_round=4, warmup_steps=0, n_sp=4, n_anchor=4, n_fm=4, idle_sleep_s=0.0,
            normalizer_warmup=0, fill_missing_values=False)
TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6,
            cutoff=CUTOFF, cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,),
            velocity_hidden=8, velocity_radial=6, velocity_min_distance=0.7,
            parity_breaking=False, edge_backend='torch')
BONDS = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]


def write_structure(fname):
    """The structure of FlowGraphTransform: a linear three-atom molecule, 1.5 Angstrom apart."""
    import MDAnalysis as mda
    universe = mda.Universe.empty(3, n_residues=1, atom_resindex=[0] * 3, trajectory=True)
    universe.add_TopologyAttr('names', ['C1', 'C2', 'C3'])
    universe.add_TopologyAttr('types', ['C'] * 3)
    universe.add_TopologyAttr('elements', ['C'] * 3)
    universe.add_TopologyAttr('resnames', ['MOL'])
    universe.add_TopologyAttr('resids', [1])
    universe.atoms.positions = np.array([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 3.0, 0.0]])
    universe.dimensions = np.array([30, 30, 30, 90, 90, 90], dtype=np.float32)
    universe.atoms.write(str(fname))
    return str(fname)


class FakeWorker:
    def __init__(self, directory):
        self._directory = str(directory)
        self.task = 'train'
        self.must_stop = False


def _network(seed=0):
    torch.manual_seed(seed)
    net = JointPaiNN(3, [6, 6, 6], bond_distances=BONDS, **TINY)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
    return net


def _setup(tmp_path, transform, network=None):
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=3, n_shots=3, n_atoms=3)
    network = _network() if network is None else network
    params = cgaimmd_params_namespace(network=network, descriptor_transform=transform)
    pe = assemble_pathensemble(params.shot_chains(rd.run, None), params.free_trajectories(rd.run))
    return rd, params, pe


@pytest.fixture
def trainers(monkeypatch):
    made = []

    class Recording(ff.FlowTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(ff, 'FlowTrainer', Recording)
    return made


def test_graph_mode_default_route_trains_without_a_cache(tmp_path, trainers):
    """graphs=True with the default route: batch dicts, the gradient penalty on their positions, no file."""
    structure = write_structure(tmp_path / 'generator_atoms.pdb')
    transform = FlowGraphTransform(structure, CUTOFF, edge_backend='torch')
    rd, params, pe = _setup(tmp_path, transform)
    calls = []

    def descriptor_transform(rows):
        out = transform(rows)
        calls.append(type(out))
        return out
    params.__dict__['descriptor_transform'] = descriptor_transform
    losses, info = ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=True,
                                        grad_penalty=0.1, relabel_every=2, **FAST)
    assert info['status'] == 'trained' and len(losses) == 4 and all(np.isfinite(losses))
    assert info['record']['loss_terms']['L_gradpen'] >= 0.0
    assert calls and set(calls) == {dict}
    assert not list(tmp_path.rglob('*.sqlite*'))
    # the fit's input path equals the dense committor of the same rows
    trainer = trainers[-1]
    rows = trainer.dataset.sp_rows[:5]
    network = params.network.eval()
    with torch.no_grad():
        lam, _ = trainer.committor_fn(network, rows, False)
        dense = network.committor(torch.as_tensor(rows).reshape(-1, 3, 3))
    torch.testing.assert_close(lam, dense, atol=1e-5, rtol=1e-5)
    # positions with a gradient for the penalty
    lam, positions = trainer.committor_fn(network, rows, True)
    (grad,) = torch.autograd.grad(lam.sum(), positions)
    assert grad.shape == (15, 3) and bool(torch.isfinite(grad).all())


def test_graph_mode_refusals(tmp_path):
    """A transform that does not fit the mode, or another cutoff than the network's, is refused at the start."""
    structure = write_structure(tmp_path / 'generator_atoms.pdb')
    transform = FlowGraphTransform(structure, CUTOFF, edge_backend='torch')
    rd, params, pe = _setup(tmp_path, transform)
    with pytest.raises(TypeError, match='returns a graph batch dict \\(pass graphs=True\\)'):
        ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=False, **FAST)
    other = FlowGraphTransform(structure, 2.0, edge_backend='torch')
    params.__dict__['descriptor_transform'] = other
    ff._RUNS.clear()
    with pytest.raises(ValueError, match='cutoff'):
        ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=True, **FAST)
    params.__dict__['descriptor_transform'] = None
    ff._RUNS.clear()
    with pytest.raises(TypeError, match='needs params.descriptor_transform'):
        ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=True, **FAST)
    params.__dict__['descriptor_transform'] = lambda rows: np.asarray(rows)
    ff._RUNS.clear()
    with pytest.raises(TypeError, match='must return a graph batch dict or a list'):
        ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=True, **FAST)


def test_graph_mode_trains_like_dense_mode(tmp_path):
    """The same round in graph mode and in dense mode: the committor inputs differ in form only."""
    structure = write_structure(tmp_path / 'generator_atoms.pdb')
    results = []
    for graphs in (False, True):
        transform = (FlowGraphTransform(structure, CUTOFF, edge_backend='torch') if graphs
                     else None)
        rd, params, pe = _setup(tmp_path / f'graphs{graphs}', transform)
        losses, info = ff.flow_matching_fit(params, pe, worker=FakeWorker(rd.run), graphs=graphs,
                                            grad_penalty=0.1, trainer_seed=3, **FAST)
        results.append((losses, {k: v.clone() for k, v in params.network.state_dict().items()}))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-4, atol=1e-5)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------------------------
# a graph cache written the flow branch's way (graph lane)
# --------------------------------------------------------------------------------------------

PARAMS = '''\
"""C-GAIMMD params of the graph-mode test: a three-atom chain molecule, {mode} mode."""
import numpy as np

from aimmd.cgaimmd.flow_matching.fit import flow_matching_fit
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.preparation.descriptors import coordinates_descriptors
{graph_imports}
engine = 'toy'
chain_type = 'cgaimmd'
states = 'ARB'
initial_paths = ['initial_A.xtc', 'initial_B.xtc']
round_shooting_index = True
gen_temperature = 300.0
cgaimmd_parameters = {{'window_after_trained_shots': 5, 'training_device': 'cpu',
                      'values_device': 'cpu', 'generation_device': 'cpu', 'placement': 'none'}}


def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01


def states_function(trajectory):
    """The states of tests._cgaimmd_helpers.build_cgaimmd_run_dir: x of atom 0."""
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R')).astype('<U1')


def descriptors_function(trajectory):
    return coordinates_descriptors(trajectory)
{graph_transform}

class Network(JointPaiNN):
    """Local subclass: Params.save writes 'from <this module> import network'."""


# a linear three-atom molecule: bonds 0-1 and 1-2 (bond edges)
network = Network(3, [6, 6, 6], bond_distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                  coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6,
                  cutoff={cutoff}, cond_dim=8, lambda_embedding_scale=4.0,
                  committor_hidden=(8,), velocity_hidden=8, velocity_radial=6,
                  velocity_min_distance=0.7, parity_breaking=False)


def fit(params, pathensemble, verbose=False, worker=None):
    return flow_matching_fit(params, pathensemble, verbose=verbose, worker=worker,
                             graphs={graphs}, steps_per_round=10, warmup_steps=0, n_sp=8,
                             n_anchor=8, n_fm=8, normalizer_warmup=0, idle_sleep_s=0.0,
                             grad_penalty=0.1)
'''

GRAPH_IMPORTS = 'from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform\n'

GRAPH_TRANSFORM = f'''

_graphs = FlowGraphTransform('flow.pdb', {CUTOFF}, graph_cache='graphs_flow_M3_rc{CUTOFF:.1f}.sqlite',
                             write_graph_cache=True)


def descriptor_transform(descriptors):
    return _graphs(descriptors)
'''


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def make_campaign(folder, mode):
    """A params folder of `mode` ('graph' or 'dense') with its run (build_cgaimmd_run_dir)."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    graph = mode == 'graph'
    (folder / 'params.py').write_text(PARAMS.format(
        mode=mode, graph_imports=GRAPH_IMPORTS if graph else '',
        graph_transform=GRAPH_TRANSFORM if graph else '', cutoff=CUTOFF, graphs=graph))
    write_structure(folder / 'flow.pdb')
    for kind, x in (('A', -1.5), ('B', 1.5)):
        positions = np.zeros((4, 3, 3), dtype=np.float32)
        positions[:, :, 0] = x + 0.05 * np.arange(4)[:, None]
        positions[:, :, 1] = 1.5 * np.arange(3)[None, :]
        write_trajectory(folder, stem=f'initial_{kind}', positions=positions)
    layout = build_cgaimmd_run_dir(folder, n_chains=3, n_shots=3, n_free=1, n_atoms=3)
    return folder, layout


def cache_count(fname):
    with sqlite3.connect(str(fname)) as conn:
        return conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0]


def frames_by_state(layout):
    """The descriptor rows of every file of the run, R frames and in-state frames apart."""
    r_rows, instate_rows = [], []
    for path in list(layout.shots) + list(layout.free):
        rows = np.load(get_cache_fname(path.fnames[0], 'descriptors'))
        states = np.load(get_cache_fname(path.fnames[0], 'states'))
        r_rows.append(rows[states == 'R'])
        instate_rows.append(rows[states != 'R'])
    return np.concatenate(r_rows), np.concatenate(instate_rows)


def rounds(run):
    return [json.loads(line) for line in
            Path(run, f'flow_fit{STATES}', 'rounds.jsonl').read_text().splitlines()]


def seconds_per_step(run):
    return [r['seconds']['train'] / r['steps'] for r in rounds(run) if r.get('steps')]


@pytest.mark.graph
def test_graph_mode_run_with_a_graph_cache(tmp_path, monkeypatch, restore_signals,
                                           record_property, trainers):
    """The trainer's value pass fills the cache; the fit reads it and adds the anchors only.

    A real trainer (``aimmd.Worker.train``, in this process) runs the fit with
    ``graphs=True`` and a graph cache written the flow branch's way. The
    trainer's value pass puts the graphs of every R frame into the cache; the
    fit reads the R frames and the shooting points from there and adds the
    anchors' graphs (in-state frames, which no value pass scores) at their
    first draw, one row per distinct anchor row; the graphs of ``x_t`` never
    reach the cache. Every round starts again from the same weights.
    """
    pytest.importorskip('torch_geometric')
    pytest.importorskip('aimmd.network.graph_utils')
    from torch_geometric.data import Batch

    folder, layout = make_campaign(tmp_path / 'graph', 'graph')
    monkeypatch.chdir(folder)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    transform = params.descriptor_transform
    cache = folder / f'graphs_flow_M3_rc{CUTOFF:.1f}.sqlite'
    graphs = params.descriptor_transform.__globals__['_graphs']
    r_rows, instate_rows = frames_by_state(layout)
    n_r = len(np.unique(r_rows, axis=0))
    n_instate = len(np.unique(instate_rows, axis=0))
    n_load = cache_count(cache) if cache.exists() else 0     # graphs Params.load made, if any

    aimmd.Worker(params, layout.run, walltime=600).train(nrounds=2)

    records = rounds(layout.run)
    assert [r['status'] for r in records] == ['trained', 'trained']
    assert records[0]['start'] == records[1]['start'] == {'kind': 'reset_parameters',
                                                          'seed': records[0]['seed']}
    assert len(trainers) == 2
    from aimmd.network import shm_cache
    assert shm_cache.pending_count(graphs._conn) == 0        # the trainer wrote everything back
    shm_cache.set_reader_role(False)
    n_after = cache_count(cache)
    added = n_after - n_load - n_r
    trainer = trainers[-1]
    data = trainer.dataset
    anchors = data.fr_rows[np.r_[trainer.anchor_first, trainer.anchor_last]]
    n_anchors = len(np.unique(anchors, axis=0))
    assert 0 < n_anchors <= n_instate
    assert added == n_anchors, (added, n_anchors)          # the anchors, once each
    # every R frame (value passes) and every shooting point (the fit) is a cache hit now
    transform(r_rows)
    transform(data.sp_rows)
    assert cache_count(cache) == n_after
    # and so is every anchor, whose graphs only the fit builds
    transform(anchors)
    assert cache_count(cache) == n_after
    # nothing else reached the cache (no margins, no graphs of x_t, which are built on the fly)
    transform(instate_rows)
    assert cache_count(cache) == n_load + n_r + n_instate

    # forward on the cached graphs equals committor on the coordinates
    network = params.network.eval()
    cached_graphs = transform(r_rows[:16])
    with torch.no_grad():
        dense = network.committor(torch.as_tensor(r_rows[:16]).reshape(-1, 3, 3))
        cached = network(Batch.from_data_list(cached_graphs).to_dict())[:, 0]
    assert torch.allclose(dense, cached, atol=1e-4), (dense - cached).abs().max()

    # a new process (the fit's memory cleared) continues the round counter from state.json
    ff._RUNS.clear()
    aimmd.Worker(params, layout.run, walltime=600).train(nrounds=1)
    state = json.loads(Path(layout.run, f'flow_fit{STATES}', 'state.json').read_text())
    assert state['round'] == 3 and rounds(layout.run)[-1]['status'] == 'trained'

    # seconds per step with the cache against the same campaign in dense mode
    dense_folder, dense_layout = make_campaign(tmp_path / 'dense', 'dense')
    monkeypatch.chdir(dense_folder)
    torch.manual_seed(0)
    dense_params = aimmd.Params.load(str(dense_folder / 'params.py'), save=False)
    assert dense_params.descriptor_transform is None
    aimmd.Worker(dense_params, dense_layout.run, walltime=600).train(nrounds=2)
    graph_seconds = float(np.mean(seconds_per_step(layout.run)))
    dense_seconds = float(np.mean(seconds_per_step(dense_layout.run)))
    record_property('graph_seconds_per_step', graph_seconds)
    record_property('dense_seconds_per_step', dense_seconds)
    assert graph_seconds > 0 and dense_seconds > 0
    assert not list(Path(dense_folder).rglob('*.sqlite*'))
    assert os.path.exists(cache)
