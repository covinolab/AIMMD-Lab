"""Training on graph keys: aimmd.network.fit with descriptor_cache='graphkeys'.

fit reads every batch through one helper, ``_transform_batch``. Without
graph keys it is ``descriptor_transform(_load_batch_descriptors(...))``, as
fit has always done; these tests pin that down first.

With graph keys a batch is read as key rows from ``<traj>.graphkeys.npy``
and its graphs are looked up by key through ``graph_keys.call_with_repair``:
a frame without a key, or whose graph is missing, is rebuilt from its
trajectory and the batch retried. No ``*.descriptors.npy`` is opened.

Most tests run without torch_geometric: real paths (toy xtc files with
states, values, descriptor and key files) of every training category, a
dict-backed, key-aware graph cache whose graphs are feature vectors, and a
stand-in for torch_geometric's ``Batch`` that stacks them, so that fit runs
its graph code path. With fixed seeds, fit on keys must then give bitwise
the losses and weights of fit on descriptor rows. The test marked ``graph``
(``--rungraph``) repeats this with the real graph cache and ``Batch``.
"""

import functools
import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aimmd import Path, PathEnsemble
from aimmd._config import MDA_CACHE, NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.core.graphkey import graph_keys, is_key_batch, keys_to_hex, pad_keys
from aimmd.network import graph_keys as gk
from aimmd.network import graph_lookup
from aimmd.network.rescalable import Rescalable
from aimmd.path.utils import get_cache_fname
from tests._helpers_graphkeys import (ToyCache, descriptors_function,
                                      forbid_descriptor_files)
from tests._helpers_unit import TinyNetwork, write_trajectory


fit_module = importlib.import_module("aimmd.network.fit")


@pytest.fixture(autouse=True)
def _clean_state():
    NPY_CACHE.clear()
    gk.reset_repair_stats()
    yield
    assert graph_lookup._OVERLAYS == [], 'an overlay leaked out of its block'
    NPY_CACHE.clear()


# --------------------------------------------------------------- npy mode --
def _descriptor_files(tmp_path):
    """Two descriptor files and a batch that interleaves their frames."""
    files = []
    for k, n in enumerate((3, 5)):
        fname = str(tmp_path / f'traj{k}.xtc.descriptors.npy')
        save_npy(fname, np.arange(n * 4, dtype=np.float64).reshape(n, 4) + 100 * k)
        files.append(fname)
    NPY_CACHE.clear()
    batch = np.array([files[1], files[0], files[1], files[1]])
    return batch, np.array([4, 2, 0, 4])


def test_transform_batch_without_keys_transforms_the_loaded_rows(tmp_path):
    """The transform gets exactly what the loader returns, in one call."""
    files, locs = _descriptor_files(tmp_path)
    calls = []

    def transform(x, **kwargs):
        calls.append((x, kwargs))
        return 2 * x

    out = fit_module._transform_batch(transform, files, locs)
    expected = fit_module._load_batch_descriptors(files, locs)
    (x, kwargs), = calls
    assert kwargs == {}
    assert x.dtype == expected.dtype and x.shape == expected.shape
    assert np.array_equal(x, expected)
    assert np.array_equal(out, 2 * expected)

    calls.clear()
    fit_module._transform_batch(transform, files, locs, system_id='s2')
    (x, kwargs), = calls
    assert kwargs == {'system_id': 's2'}
    assert np.array_equal(x, expected)


# ------------------------------------------------------------ toy campaign --
#: stem, atom-0 x of each frame, shooting index -- one path per category
PATHS = [
    ('inA', [0.5, 1.0, 1.5, 1.2], 1),           # AAAA
    ('inB', [8.5, 9.0, 9.5, 8.8], 1),           # BBBB
    ('shot12', [1.0, 3, 4, 5, 6, 7, 9], 3),     # ARBR
    ('shot21', [9.0, 7, 6, 5, 4, 3, 1], 3),     # BRAR
    ('shot11', [1.0, 3, 4, 5, 3, 1], 2),        # ARAR
    ('shot22', [9.0, 7, 6, 5, 7, 9], 2),        # BRBR
    ('free12', [1.0, 3, 5, 7, 9], 0),           # ARBA
    ('free21', [9.0, 7, 5, 3, 1], 0),           # BRAB
]

#: fit settings of the comparisons: streamed graphs, as in production
FIT = dict(nbins=2, state_bins='AB', augment='yes', lr=1e-2, epochs=12,
           batch_size=8, stop=1e9, in_memory=False, graphs=True)


def features(row, sign=1.0):
    """A toy graph: two features of atom 0, the input of TinyNetwork."""
    return np.array([(row[0] - 5) / 2, sign * (row[1] - 5) / 2],
                    dtype=np.float32)


class StackedBatch:
    """torch_geometric's ``Batch`` for toy graphs: stacks the vectors."""

    def __init__(self, x):
        self.x = x

    @classmethod
    def from_data_list(cls, graphs):
        return cls(torch.as_tensor(np.stack(graphs)))

    def to(self, device):
        return StackedBatch(self.x.to(device))

    def to_dict(self):
        return self.x


@pytest.fixture
def stacked_batch(monkeypatch):
    """Make ``from torch_geometric.data import Batch`` give StackedBatch."""
    data = types.ModuleType('torch_geometric.data')
    data.Batch = StackedBatch
    package = types.ModuleType('torch_geometric')
    package.data = data
    monkeypatch.setitem(sys.modules, 'torch_geometric', package)
    monkeypatch.setitem(sys.modules, 'torch_geometric.data', data)


def _rows(fname):
    """Descriptor rows of every frame, decoded as ingestion decodes them."""
    reader = MDA_CACHE.get(fname)
    return descriptors_function(reader[np.arange(len(reader))])


def _ensemble(root, cache, n_atoms=3, seed=0):
    """Paths of every training category, ingested for both modes.

    Each trajectory gets its states, values, descriptor rows and graph keys,
    and the graphs of its frames are in ``cache``, as after ingestion.
    """
    os.makedirs(root, exist_ok=True)
    paths = []
    for k, (stem, xs, shooting_index) in enumerate(PATHS):
        rng = np.random.default_rng(seed + k)
        positions = rng.uniform(3, 7, (len(xs), n_atoms, 3)).astype(np.float32)
        positions[:, 0, 0] = xs
        fname = write_trajectory(root, stem=stem, positions=positions)
        rows = _rows(fname)
        x = rows[:, 0]
        states = np.where(x < 2, 'A', np.where(x > 8, 'B', 'R')).astype('<U1')
        save_npy(get_cache_fname(fname, 'states'), states)
        save_npy(get_cache_fname(fname, 'values'), (x - 5.0).astype(float))
        save_npy(get_cache_fname(fname, 'descriptors'), rows)
        save_npy(get_cache_fname(fname, 'graphkeys'), graph_keys(rows))
        cache.cache(rows)
        paths.append(Path(fname, shooting_index=shooting_index))
    NPY_CACHE.clear()
    return PathEnsemble(paths)


def _params(transform, mode, **extra):
    return SimpleNamespace(network=TinyNetwork(), sorted_states='ARB',
                           descriptors_function=descriptors_function,
                           descriptor_transform=transform,
                           descriptor_cache=mode, **extra)


def _fit(params, pathensemble, **kwargs):
    """fit with fixed seeds; what it returns, and the trained weights."""
    np.random.seed(0)
    torch.manual_seed(0)
    NPY_CACHE.clear()
    losses, scales, values, probabilities, results = fit_module.fit(
        params, pathensemble, **{**FIT, **kwargs})
    weights = torch.cat([p.detach().flatten()
                         for p in params.network.parameters()]).numpy()
    return dict(losses=np.array(losses), scales=np.array(scales),
                values=values, probabilities=probabilities,
                results=results, weights=weights)


def _assert_bitwise_equal(a, b):
    assert len(a['losses']) > 1
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert np.array_equal(a[name], b[name]), name


def _batches(cache, kind):
    """Lengths of the batches the cache's transform got, of one kind."""
    return [n for k, n in cache.calls if k == kind and n]


# ----------------------------------------------------- loading key batches --
def test_key_batches_are_read_by_trajectory_with_zero_rows(tmp_path):
    """With keys, the per-frame files are the trajectories themselves; a
    frame without a key file, or beyond its end, gets a zero row."""
    fnames = [str(tmp_path / f'traj{k}.xtc') for k in range(3)]
    keys = graph_keys(np.arange(5 * 4, dtype=np.float32).reshape(5, 4))
    save_npy(get_cache_fname(fnames[0], 'graphkeys'), keys)
    save_npy(get_cache_fname(fnames[1], 'graphkeys'), keys[:2])   # short
    NPY_CACHE.clear()                                              # [2]: none

    batch = np.array([fnames[0], fnames[1], fnames[1], fnames[2], fnames[0]])
    locs = np.array([4, 1, 3, 0, 0])
    files = fit_module._batch_files(batch, keys=True)
    assert files.tolist() == batch.tolist()
    rows = fit_module._load_batch_descriptors(files, locs, keys=True)
    assert rows.dtype == np.uint8 and rows.shape == (5, 32)
    assert np.array_equal(rows[[0, 1, 4]], keys[[4, 1, 0]])
    assert not rows[[2, 3]].any()

    # without keys: descriptor files, and a missing one is an error, as before
    files = fit_module._batch_files(batch)
    assert files[0] == get_cache_fname(fnames[0], 'descriptors')
    with pytest.raises(RuntimeError, match='descriptor cache file'):
        fit_module._load_batch_descriptors(files, locs)


def test_transform_batch_with_keys_looks_the_graphs_up(tmp_path, monkeypatch):
    cache = ToyCache(graph=features)
    pathensemble = _ensemble(tmp_path, cache)
    fnames, locs = gk.frame_refs(pathensemble)
    keys_function = gk.GraphKeysFunction(descriptors_function, cache.transform)
    order = np.random.default_rng(0).permutation(len(fnames))[:9]

    with forbid_descriptor_files(monkeypatch):
        graphs = fit_module._transform_batch(
            cache.transform, fit_module._batch_files(fnames[order], keys=True),
            locs[order], keys=True, keys_function=keys_function)
    rows = np.concatenate([_rows(path.filenames[0]) for path in pathensemble])
    assert np.array_equal(np.stack(graphs), np.stack(
        [features(row) for row in rows[order]]))
    assert _batches(cache, 'rows') == []
    assert _batches(cache, 'keys') == [9]
    assert gk.repair_stats()['repaired'] == 0


# ------------------------------------------------------ fit on graph keys --
@pytest.mark.parametrize('terms', [
    {}, {'lsr_weight': 0.1}, {'mar_weight': 0.1},
    {'lsr_weight': 0.1, 'mar_weight': 0.1}],
    ids=['committor', 'lsr', 'mar', 'lsr+mar'])
def test_fit_on_keys_equals_fit_on_descriptors(tmp_path, monkeypatch,
                                               stacked_batch, terms):
    """Training, LSR and MAR: the same batches, bitwise the same result."""
    cache = ToyCache(graph=features)
    pathensemble = _ensemble(tmp_path, cache)

    npy = _fit(_params(cache.transform, 'npy'), pathensemble, **terms)
    npy_batches = _batches(cache, 'rows')
    assert _batches(cache, 'keys') == []
    cache.calls.clear()
    with forbid_descriptor_files(monkeypatch):
        keyed = _fit(_params(cache.transform, 'graphkeys'), pathensemble,
                     **terms)

    _assert_bitwise_equal(npy, keyed)
    assert _batches(cache, 'rows') == []          # nothing decoded or hashed
    assert _batches(cache, 'keys') == npy_batches
    assert not any(gk.repair_stats().values())
    if terms:                    # the terms read batches of their own
        assert len(npy_batches) > len(npy['losses']) + 1


def test_fit_on_keys_with_validation(tmp_path, monkeypatch, stacked_batch,
                                     capsys):
    """The validation set is read by key too, and early stopping agrees."""
    cache = ToyCache(graph=features)
    pathensemble = _ensemble(tmp_path, cache)
    validation = dict(train_validation_early_stopping=True,
                      early_stopping_min_samples=1, early_stopping_split=0.3,
                      early_stopping_patience=2, epochs=30,
                      lsr_weight=0.1, mar_weight=0.1)

    npy = _fit(_params(cache.transform, 'npy'), pathensemble, **validation)
    cache.calls.clear()
    with forbid_descriptor_files(monkeypatch):
        keyed = _fit(_params(cache.transform, 'graphkeys'), pathensemble,
                     **validation)

    _assert_bitwise_equal(npy, keyed)
    out = capsys.readouterr().out
    assert out.count('Using early stopping with 9 samples') == 2
    assert 9 in _batches(cache, 'keys')          # the validation set
    assert _batches(cache, 'rows') == []


def test_fit_repairs_graph_cache_misses_and_trains_on(tmp_path, monkeypatch,
                                                      stacked_batch, capsys):
    """Graphs lost from the cache, a lost key file, a short one and a stale
    key: fit repairs them from the trajectories as the batches meet them,
    and trains exactly as on intact data.

    The damage is to frames of the reactive paths, which MAR reads in full
    at every step, so every damaged frame is read.
    """
    cache = ToyCache(graph=features)
    pathensemble = _ensemble(tmp_path, cache)
    terms = dict(lsr_weight=0.1, mar_weight=0.1)
    npy = _fit(_params(cache.transform, 'npy'), pathensemble, **terms)

    paths = {os.path.basename(path.filenames[0])[:-len('.xtc')]: path
             for path in pathensemble}
    intact = {name: graph_keys(_rows(path.filenames[0]))
              for name, path in paths.items()}

    def key_file(name):
        return get_cache_fname(paths[name].filenames[0], 'graphkeys')

    for name, frame in (('shot12', 2), ('shot21', 4)):
        del cache.store[keys_to_hex(intact[name][frame:frame + 1])[0]]
    os.remove(key_file('free12'))
    save_npy(key_file('free21'), intact['free21'][:2])
    stale = intact['shot12'].copy()
    stale[3, 0] ^= 0xFF
    save_npy(key_file('shot12'), stale)
    NPY_CACHE.clear()

    with forbid_descriptor_files(monkeypatch):
        keyed = _fit(_params(cache.transform, 'graphkeys'), pathensemble,
                     **terms)

    _assert_bitwise_equal(npy, keyed)
    stats = gk.repair_stats()
    assert stats['filled'] == 5                   # free12: 3, free21: 2
    assert stats['retries'] > 0 and stats['stale'] == 1
    assert stats['fallbacks'] == 0
    assert 'repaired' in capsys.readouterr().out
    for name, path in paths.items():
        stored = pad_keys(np.load(key_file(name)), len(intact[name]), name)
        right = (stored == intact[name]).all(axis=1)
        zero = ~stored.any(axis=1)
        assert right[path.internal('indices')].all(), name   # every frame read
        assert (right | zero).all(), name   # the others: right, or not computed
    assert set(keys_to_hex(np.concatenate(list(intact.values())))) <= set(
        cache.store)


def test_fit_on_keys_ignores_in_memory(tmp_path, monkeypatch, stacked_batch,
                                       capsys):
    cache = ToyCache(graph=features)
    pathensemble = _ensemble(tmp_path, cache)
    streamed = _fit(_params(cache.transform, 'graphkeys'), pathensemble)
    assert 'in_memory=True is ignored' not in capsys.readouterr().out
    with forbid_descriptor_files(monkeypatch):
        in_memory = _fit(_params(cache.transform, 'graphkeys'), pathensemble,
                         in_memory=True)
    assert 'in_memory=True is ignored' in capsys.readouterr().out
    _assert_bitwise_equal(streamed, in_memory)


def test_fit_on_keys_needs_graphs_and_a_transform():
    """Raised before any data is read."""
    with pytest.raises(ValueError, match='graphs=True'):
        fit_module.fit(_params(ToyCache().transform, 'graphkeys'), None,
                       graphs=False)
    with pytest.raises(ValueError, match='descriptor_transform'):
        fit_module.fit(_params(None, 'graphkeys'), None, graphs=True)


class Systems:
    """Two systems, each with its own graph cache, graphs and atom count.

    The transform and the descriptors function take ``system_id``, as in a
    multi-system params file, and record which system each call was for.
    """

    def __init__(self):
        self.caches = {
            's1': ToyCache(graph=features),
            's2': ToyCache(graph=functools.partial(features, sign=-1.0))}
        self.lookups = {'s1': [], 's2': []}
        self.decoded = []

    def transform(self, x, system_id=None):
        if is_key_batch(x):
            self.lookups[system_id].extend(keys_to_hex(x))
        return self.caches[system_id].transform(x)

    def descriptors_function(self, trajectory, system_id=None):
        self.decoded.append(system_id)
        return descriptors_function(trajectory)


def test_fit_on_keys_routes_multi_system_batches(tmp_path, monkeypatch,
                                                 stacked_batch):
    """Each system's frames are looked up (and repaired) in its own cache,
    with its own functions; the result equals fit on descriptor rows."""
    systems = Systems()
    pathensembles = [
        _ensemble(tmp_path / 's1', systems.caches['s1'], n_atoms=3),
        _ensemble(tmp_path / 's2', systems.caches['s2'], n_atoms=4, seed=50)]
    own = {sid: set(keys_to_hex(np.concatenate(
               [graph_keys(_rows(f)) for f in gk.frame_refs(pe)[0]])))
           for sid, pe in zip(('s1', 's2'), pathensembles)}

    def params(mode):
        out = _params(systems.transform, mode, system_ids=['s1', 's2'])
        out.descriptors_function = systems.descriptors_function
        return out

    npy = _fit(params('npy'), pathensembles)
    systems.caches['s2'].store.clear()          # system 2 lost every graph
    systems.decoded.clear()
    with forbid_descriptor_files(monkeypatch):
        keyed = _fit(params('graphkeys'), pathensembles)

    _assert_bitwise_equal(npy, keyed)
    for sid in ('s1', 's2'):
        assert systems.lookups[sid]
        assert set(systems.lookups[sid]) <= own[sid], sid
        assert set(systems.caches[sid].store) <= own[sid], sid
    assert set(systems.decoded) == {'s2'}       # repairs, all of system 2
    assert systems.caches['s2'].store
    stats = gk.repair_stats()
    assert stats['repaired'] > 0 and stats['fallbacks'] == 0


# ----------------------------------------------------- with graph_utils --
def _graph_utils():
    return pytest.importorskip(
        "aimmd.network.graph_utils",
        reason="graph utility tests require the optional graph/GNN dependencies")


class GraphNetwork(Rescalable):
    """A small model of graphs: a linear function of the node positions,
    summed per graph."""

    def __init__(self):
        super().__init__(max_knots=8)
        self.linear = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.linear.weight[:] = torch.tensor([[0.3, -0.2, 0.1]])

    def forward(self, d):
        nodes = self.linear(d['positions'].float() - 5.0)
        out = torch.zeros(int(d['batch'].max()) + 1, 1, dtype=nodes.dtype)
        return out.index_add(0, d['batch'], nodes)


@pytest.mark.graph
def test_fit_on_keys_equals_fit_on_descriptors_with_real_graphs(
        tmp_path, monkeypatch):
    """The real graph cache: the same losses and weights, bitwise, also
    after a graph is lost from the database and repaired inside fit."""
    from tests.test_graphkeys_repair import _GraphParams
    gu = _graph_utils()
    conn = gu.init_db(str(tmp_path / 'graphs.sqlite'))
    functions = _GraphParams(gu, {None: conn})
    keys_function = gk.GraphKeysFunction(functions.descriptors_function,
                                         functions.descriptor_transform)

    class Cache:                       # ingestion: keys and graphs cached
        @staticmethod
        def cache(rows):
            functions.descriptor_transform(rows)

    pathensemble = _ensemble(tmp_path / 'run', Cache())
    assert conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0]

    def params(mode):
        out = _params(functions.descriptor_transform, mode)
        out.network = GraphNetwork()
        out.descriptors_function = functions.descriptors_function
        out.graphkeys_function = keys_function
        return out

    terms = dict(lsr_weight=0.1, mar_weight=0.1)
    npy = _fit(params('npy'), pathensemble, **terms)
    with forbid_descriptor_files(monkeypatch):
        keyed = _fit(params('graphkeys'), pathensemble, **terms)
    _assert_bitwise_equal(npy, keyed)
    assert not any(gk.repair_stats().values())

    fname = pathensemble[2].filenames[0]
    lost = keys_to_hex(graph_keys(_rows(fname))[3:4])[0]
    conn.execute('DELETE FROM graphs_cache WHERE key = ?', (lost,))
    conn.commit()
    conn._aimmd_memo.clear()
    with forbid_descriptor_files(monkeypatch):
        repaired = _fit(params('graphkeys'), pathensemble, **terms)
    _assert_bitwise_equal(npy, repaired)
    assert gk.repair_stats()['retries'] > 0
    assert gk.repair_stats()['fallbacks'] == 0
    assert conn.execute('SELECT 1 FROM graphs_cache WHERE key = ?',
                        (lost,)).fetchone()
    conn.close()
