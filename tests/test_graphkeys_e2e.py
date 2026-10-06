"""A toy campaign switched to graph keys and back, end to end.

On the toy engine, with a dict-backed, key-aware graph cache defined in the
params file (no torch_geometric), all workers run in this process:

1. 'npy' (today's mode): train, free, shoot; then a shot is left in flight
   (its back segment ingested with descriptors) -- the state of a job that
   ends mid-shot.
2. switch to 'graphkeys' with ``Params.update``, as a resubmitted job with
   the new params file would; two graphs are lost from the cache. The
   launcher re-exports the seed, the shooting worker resumes the in-flight
   shot, a free worker continues, and the trainer runs a round (key ledger,
   verify, fit on keys, value passes). A guard fails the test on ANY open
   of a ``*.descriptors.npy`` file in this phase, although every npy-era
   trajectory still has one.
3. roll back to 'npy': a TPS shot and a trainer round on a chain whose
   newest paths have keys only. Nothing crashes, and the descriptors of
   the graph-key era are regenerated from the trajectories.
"""

import glob
import os
from pathlib import Path as PosixPath

import numpy as np
import pytest

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.core.graphkey import graph_keys, keys_to_hex
from aimmd.network import graph_keys as gk
from aimmd.network import graph_lookup
from tests._helpers_graphkeys import forbid_descriptor_files


PARAMS_SOURCE = '''
import numpy as np
import torch
from aimmd.core.graphkey import graph_keys, is_key_batch, keys_to_hex
from aimmd.network.graph_lookup import (GraphCacheMiss, collect_graphs,
                                        overlay_get)
from aimmd.network import graph_keys as _graph_keys

engine = 'toy'
toy_slowdown = 0.0
initial_paths = ['initial.xtc']
topology = 'initial.xtc'
chain_type = 'tps'
selection_pool_size = 1
nbins = 3
max_length = 80
extra_free_frames = 0
free_overriding_states = ''

# the graph cache: hex key -> graph (here: a 1-feature vector)
GRAPHS = {}
BUILT = []
FITS = []


def toy_mdrun(ts):
    # every coordinate moves on its own, so frames do not recur
    noise = 0.5 * np.random.normal(size=ts.positions.shape)
    ts.positions = (ts.positions + noise) % 10


def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x < 2, 'A', np.where(x > 8, 'B', 'R')).astype('<U1')


def descriptors_function(trajectory):
    return np.array([ts.positions.ravel().copy() for ts in trajectory],
                    dtype=np.float32).reshape(len(trajectory), -1)


def descriptor_transform(x):
    """Coordinate rows: build or load; key rows: look up only."""
    if is_key_batch(x):
        graphs, missing = [], []
        for h, row in zip(keys_to_hex(x), x):
            graph = overlay_get(h) if row.any() else None
            if graph is None and row.any():
                graph = GRAPHS.get(h)
            if graph is None:
                missing.append(row)
            graphs.append(graph)
        if missing:
            raise GraphCacheMiss(np.unique(np.stack(missing), axis=0))
        return graphs
    x = np.asarray(x, dtype=np.float32)
    hexes = keys_to_hex(graph_keys(x)) if len(x) else []
    graphs = []
    for h, row in zip(hexes, x):
        graph = GRAPHS.get(h)
        if graph is None:
            graph = np.array([row[0] / 10.0], dtype=np.float32)
            GRAPHS[h] = graph
            BUILT.append(h)
        graphs.append(graph)
    collect_graphs(hexes, graphs)
    return graphs


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x)

network = Network()


def values_function(x):
    if not len(x):
        return np.zeros(0)
    features = torch.as_tensor(np.stack(descriptor_transform(x)))
    with torch.no_grad():
        return network(features).numpy().ravel().astype(float)


def fit(params, pathensemble, verbose=False, worker=None):
    """A few steps on the frames in A and B, read the run's way."""
    frames = pathensemble.join()
    before = _graph_keys.repair_stats()
    if params.descriptors_source == 'graphkeys':
        fnames, locs = _graph_keys.frame_refs(frames)
        keys = _graph_keys.load_keys(fnames, locs)
        graphs = _graph_keys.call_with_repair(
            descriptor_transform, keys, fnames, locs,
            params.graphkeys_function)
    else:
        graphs = descriptor_transform(frames.descriptors)
    after = _graph_keys.repair_stats()
    FITS.append((params.descriptors_source, len(graphs),
                 {k: after[k] - before[k] for k in after}))
    states = frames.states
    inside = (states == 'A') | (states == 'B')
    x = torch.as_tensor(np.stack(graphs)[inside])
    y = torch.as_tensor((states[inside] == 'B').astype(np.float32))[:, None]
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            network(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return (losses,)
'''


def _sweep(fname, n_frames=30, n_atoms=10):
    """Atom 0 sweeps x from 0 to 10 (A -> R -> B).

    Ten atoms: from ten on, xtc stores compressed integer coordinates, which
    re-encode exactly when registration rewrites a path (as in production).
    Up to nine it stores raw floats in nm, and the conversion to and from
    Angstrom changes some of them by one ulp on the first rewrite.
    """
    import MDAnalysis as mda
    universe = mda.Universe.empty(n_atoms, trajectory=True)
    others = np.random.default_rng(0).uniform(3.0, 7.0, (n_atoms - 1, 3))
    with mda.Writer(str(fname), n_atoms) as writer:
        for x in np.linspace(0.0, 10.0, n_frames):
            universe.atoms.positions = np.vstack(
                [[x, 5.0, 5.0], others]).astype(np.float32)
            writer.write(universe.atoms)


def _trajectories(run):
    """Every trajectory AIMMD ingested: those with a states file."""
    return sorted(f[:-len('.states.npy')] for f in
                  glob.glob(f'{run}/**/*.xtc.states.npy', recursive=True))


def _rows(fname):
    return np.asarray(aimmd.Path(fname).coordinates, dtype=np.float32)


def _more_steps(run, n):
    """Worker step limit for n more shots (limits count what is on disk)."""
    return len(glob.glob(f'{run}/chainR0/path*.xtc')) + n


def _more_frames(run, n):
    """Worker frame limit for n more free frames."""
    return sum(len(aimmd.Path(f))
               for f in glob.glob(f'{run}/freeA/*.xtc')) + n


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('AIMMD_GRAPHKEYS_VERIFY', raising=False)
    monkeypatch.delenv('AIMMD_GRAPHKEYS_SKIP_SELFTEST', raising=False)
    _sweep(tmp_path / 'initial.xtc')
    (tmp_path / 'params.py').write_text(PARAMS_SOURCE)
    np.random.seed(1)
    NPY_CACHE.clear()
    gk.reset_repair_stats()
    yield tmp_path
    NPY_CACHE.clear()
    assert graph_lookup._OVERLAYS == []


def test_switch_to_graph_keys_and_back(campaign, monkeypatch, capsys):
    run = str(campaign / 'run1')
    params = aimmd.Params.load('params.py', save=False)
    module = params.descriptor_transform.__globals__
    graphs, fits = module['GRAPHS'], module['FITS']
    assert not params.graphkeys_mode

    # ---- 1. npy -----------------------------------------------------------
    aimmd.Worker(params, 'run1', walltime=120).train(nrounds=1)
    aimmd.Worker(params, 'run1', nframes=20, walltime=120).free(0, 0, 1)
    aimmd.Worker(params, 'run1', nsteps=2, walltime=120).shoot(1, 0)
    chain = f'{run}/chainR0'
    assert glob.glob(f'{chain}/path*.xtc.descriptors.npy')
    # leave a shot in flight: its back segment is already ingested
    seed = aimmd.PathEnsemble(f'{run}/initialARB/*')[0]
    params.initialize_simulation(seed[len(seed) // 2], f'{chain}/back',
                                 f'{chain}/forw')
    aimmd.Path(f'{chain}/back.xtc', pipeline=params.pipeline[:-1])
    assert os.path.exists(f'{chain}/back.xtc.descriptors.npy')
    npy_era = _trajectories(run)
    assert all(os.path.exists(f'{f}.descriptors.npy') for f in npy_era)
    assert not glob.glob(f'{run}/**/*.graphkeys.npy', recursive=True)
    capsys.readouterr()

    # ---- 2. graph keys, never a descriptors file --------------------------
    with forbid_descriptor_files(monkeypatch) as opened:
        params.update(descriptor_cache='graphkeys', save=False)
        assert params.graphkeys_mode
        assert 'graphkeys' in params.initial_paths[0].__dict__
        # a resubmitted job: the launcher re-exports the seed, then work
        launcher = aimmd.Launcher(params, 'run1')
        launcher._update(n=0)
        launcher._build()
        assert os.path.exists(f'{run}/initialARB/initial.xtc.graphkeys.npy')
        aimmd.Worker(params, 'run1', nsteps=_more_steps(run, 2),
                     walltime=120).shoot(1, 0)
        aimmd.Worker(params, 'run1', nframes=_more_frames(run, 20),
                     walltime=120).free(0, 0, 1)

        # two graphs of the newest registered path are lost
        newest = sorted(glob.glob(f'{chain}/path*.xtc'))[-1]
        assert newest not in npy_era
        lost = keys_to_hex(np.load(f'{newest}.graphkeys.npy')[[1, 2]])
        for h in lost:
            graphs.pop(h)
        aimmd.Worker(params, 'run1', walltime=120).train(nrounds=1)
        out = capsys.readouterr().out
    assert opened == [], opened

    # the in-flight shot was finished and registered with keys only
    keys_era = [f for f in _trajectories(run) if f not in npy_era]
    assert any(f.startswith(f'{chain}/path') for f in keys_era)
    assert any('/freeA' in f for f in keys_era)
    for fname in keys_era:
        assert not os.path.exists(f'{fname}.descriptors.npy'), fname
    # every ingested trajectory is keyed, completely and correctly, and has
    # its graphs in the cache
    NPY_CACHE.clear()
    for fname in _trajectories(run):
        if os.path.basename(fname) in ('back.xtc', 'forw.xtc'):
            continue
        keys = np.load(f'{fname}.graphkeys.npy')
        assert np.array_equal(keys, graph_keys(_rows(fname))), fname
        assert all(h in graphs for h in keys_to_hex(keys)), fname
    assert all(h in graphs for h in lost)
    # the trainer keyed the npy era, repaired the lost graphs, and fit read
    # every graph by key without a single repair
    assert 'missing graph keys' in out
    assert '2 without a graph; 2 repaired (0 stale)' in out
    source, n_frames, repairs = fits[-1]
    assert source == 'graphkeys' and n_frames > 0
    assert repairs['retries'] == 0 and repairs['fallbacks'] == 0
    assert gk.repair_stats()['fallbacks'] == 0

    # ---- 3. back to npy: the graph-key era gets its descriptors again ------
    params.update(descriptor_cache='npy', save=False)
    assert 'descriptors' in params.initial_paths[0].__dict__
    aimmd.Worker(params, 'run1', nsteps=_more_steps(run, 1),
                 walltime=120).shoot(1, 0)
    aimmd.Worker(params, 'run1', walltime=120).train(nrounds=1)
    assert fits[-1][0] == 'descriptors'
    NPY_CACHE.clear()
    for fname in _trajectories(run):
        if os.path.basename(fname) in ('back.xtc', 'forw.xtc'):
            continue
        rows, expected = np.load(f'{fname}.descriptors.npy'), _rows(fname)
        computed = rows.any(axis=1)
        assert np.array_equal(rows[computed], expected[:len(rows)][computed])
        if '/initialARB/' in fname:      # the trainer needs its margins only
            margins = np.isin(np.load(f'{fname}.states.npy'), ['A', 'B'])
            assert computed[margins[:len(rows)]].all(), fname
        else:
            assert len(rows) == len(expected) and computed.all(), fname
