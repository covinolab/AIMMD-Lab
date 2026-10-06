"""Tests for per-frame graph keys and their repair (aimmd.network.graph_keys).

Most tests run in the default suite: a dict-backed, key-aware toy cache
stands in for the sqlite graph cache, so no torch_geometric is needed. The
toy transform behaves like ``process_descriptors_pyg``: coordinate rows are
keyed, looked up and built if missing (and handed to an open overlay); key
rows are only looked up and raise ``GraphCacheMiss``.

The tests marked ``graph`` (``--rungraph``) repeat the decisive ones with the
real graph cache, among them the review's R4 regression: a graph store that
does not land must never turn into an exception, in the writer role (the
store gives up after lock contention) and in the reader role (no replica, no
memo).
"""

import os

import numpy as np
import pytest

from aimmd import Path
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.core.graphkey import graph_keys, is_key_batch, keys_to_hex
from aimmd.core.utils import accepts_system_id
from aimmd.network import graph_keys as gk
from aimmd.network import graph_lookup, shm_cache
from aimmd.network.graph_lookup import GraphCacheMiss, collect_graphs, overlay_get
from tests._helpers_unit import write_trajectory


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# ------------------------------------------------------------- toy cache --
def _positions(n_frames, n_atoms=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 9.0, (n_frames, n_atoms, 3)).astype(np.float32)


def descriptors_function(trajectory):
    """Flattened float32 coordinates, as atom_coordinate_descriptors_function."""
    return np.array([ts.positions.ravel().copy() for ts in trajectory])


class ToyCache:
    """A dict-backed graph cache with a key-aware transform.

    ``graph`` of a frame is ``('graph', sum of its coordinates)``, so values
    computed from key rows can be compared with values computed from
    coordinate rows.
    """

    def __init__(self, fail_store=False, collect=True):
        self.store = {}
        self.built = []
        self.calls = []
        self.fail_store = fail_store
        self.collect = collect

    @staticmethod
    def graph(row):
        return ('graph', float(np.asarray(row, dtype=np.float64).sum()))

    def transform(self, x):
        if is_key_batch(x):
            self.calls.append(('keys', len(x)))
            out, missing = [], []
            for h, row in zip(keys_to_hex(x), x):
                graph = None
                if row.any():
                    graph = overlay_get(h)
                    if graph is None:
                        graph = self.store.get(h)
                if graph is None:
                    missing.append(row)
                out.append(graph)
            if missing:
                raise GraphCacheMiss(np.unique(np.stack(missing), axis=0))
            return out
        self.calls.append(('rows', len(x)))
        hexes = keys_to_hex(graph_keys(x))
        graphs = []
        for h, row in zip(hexes, x):
            graph = self.store.get(h)
            if graph is None:
                graph = self.graph(row)
                self.built.append(h)
                if not self.fail_store:
                    self.store[h] = graph
            graphs.append(graph)
        if self.collect:
            collect_graphs(hexes, graphs)
        return graphs

    def values(self, x):
        return np.array([g[1] for g in self.transform(x)])

    def cache(self, rows):
        for h, row in zip(keys_to_hex(graph_keys(rows)), rows):
            self.store[h] = self.graph(row)


def _trajectory(tmp_path, n_frames=6, stem='traj', seed=0):
    return write_trajectory(tmp_path, stem=stem, positions=_positions(n_frames, seed=seed))


def _rows(fname, locs=None):
    path = Path(fname)
    rows = np.asarray(path.coordinates, dtype=np.float32)
    return rows if locs is None else rows[np.asarray(locs)]


def _reader(fname, locs):
    from aimmd._config import MDA_CACHE
    locs = np.asarray(locs)
    return MDA_CACHE.get(fname, int(locs.max()) + 1)[locs]


def _key_file(fname):
    return f'{fname}.graphkeys.npy'


@pytest.fixture(autouse=True)
def _clean_state():
    NPY_CACHE.clear()
    gk.reset_repair_stats()
    yield
    assert graph_lookup._OVERLAYS == [], 'an overlay leaked out of its block'
    NPY_CACHE.clear()


# ------------------------------------------------------- GraphKeysFunction --
def test_graph_keys_function_returns_keys_and_builds_only_missing(tmp_path):
    fname = _trajectory(tmp_path, 5)
    rows = _rows(fname)
    toy = ToyCache()
    toy.cache(rows[:2])
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    keys = kf(_reader(fname, np.arange(5)))
    assert keys.dtype == np.uint8 and keys.shape == (5, 32)
    assert np.array_equal(keys, graph_keys(rows))
    assert toy.built == keys_to_hex(keys[2:])


def test_graph_keys_function_on_no_frames(tmp_path):
    toy = ToyCache()
    kf = gk.GraphKeysFunction(lambda trajectory: np.zeros((1, 0)), toy.transform)
    assert kf([]).shape == (0, 32)


def test_side_effect_graphs_are_not_built_twice(tmp_path):
    """A descriptors_function that builds graphs itself, and a store that gives up.

    Production params build and store graphs inside descriptors_function. If
    that store does not land, the key function must not build (and try to
    store) the same graphs a second time.
    """
    fname = _trajectory(tmp_path, 4)
    toy = ToyCache(fail_store=True)

    def building_descriptors_function(trajectory):
        rows = descriptors_function(trajectory)
        toy.transform(rows)
        return rows

    kf = gk.GraphKeysFunction(building_descriptors_function, toy.transform)
    keys = kf(_reader(fname, np.arange(4)))
    assert np.array_equal(keys, graph_keys(_rows(fname)))
    assert len(toy.built) == 4, toy.built
    assert toy.store == {}


def test_missing_reports_zero_rows_and_absent_graphs(tmp_path):
    fname = _trajectory(tmp_path, 4)
    rows = _rows(fname)
    toy = ToyCache()
    toy.cache(rows[[0, 2]])
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    keys = graph_keys(rows)
    keys[3] = 0
    assert kf.missing(keys).tolist() == [False, True, False, True]
    assert kf.connection() is None, 'a dict-backed transform has no sqlite cache'


def test_missing_probes_a_sub_batching_transform_completely(tmp_path):
    """A transform that stops at its first missing sub-batch is asked again."""
    fname = _trajectory(tmp_path, 6)
    rows = _rows(fname)
    toy = ToyCache()
    toy.cache(rows[[0, 1, 3]])

    def sub_batching(x):
        out = []
        for i in range(0, len(x), 2):
            out.extend(toy.transform(x[i:i + 2]))
        return out

    kf = gk.GraphKeysFunction(descriptors_function, sub_batching)
    assert kf.missing(graph_keys(rows)).tolist() == [
        False, False, True, False, True, True]


# ------------------------------------------------------------ KeyedFunction --
def test_keyed_function_forwards_calls():
    seen = []

    def plain(x, scale=1.0):
        seen.append(('plain', scale))
        return np.asarray(x) * scale

    def per_system(x, system_id=None):
        seen.append(('per_system', system_id))
        return np.asarray(x)

    keys_function = object()
    keyed = gk.KeyedFunction(plain, keys_function)
    assert keyed.keys_function is keys_function
    assert keyed.function is plain
    assert keyed.__name__ == 'plain'
    assert accepts_system_id(keyed)
    assert np.array_equal(keyed([1, 2], system_id='s1', scale=2.0), [2, 4])
    assert gk.KeyedFunction(per_system, keys_function)([1], system_id='s2').tolist() == [1]
    assert seen == [('plain', 2.0), ('per_system', 's2')]


# ------------------------------------------------------------------ repair --
def test_repair_rewrites_zero_and_stale_rows_once_per_frame(tmp_path, monkeypatch):
    fname = _trajectory(tmp_path, 6)
    rows = _rows(fname)
    true = graph_keys(rows)
    toy = ToyCache()
    toy.cache(rows[[0, 2, 5]])                     # frames 1, 3, 4 have no graph

    stored = true.copy()
    stored[1] = 0                                  # never computed
    stored[3] = np.arange(32, dtype=np.uint8)      # stale: some other frame's key
    save_npy(_key_file(fname), stored)
    NPY_CACHE.get(_key_file(fname))                # a cached copy must be evicted

    writes = []
    real_update = gk.update_npy

    def spy(target, data, indices, *args, **kwargs):
        writes.append((target, np.asarray(indices).tolist()))
        return real_update(target, data, indices, *args, **kwargs)
    monkeypatch.setattr(gk, 'update_npy', spy)

    locs = np.array([1, 3, 3, 4, 0, 1])            # frames 1 and 3 drawn twice
    batch = stored[locs]
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    result = gk.repair(batch, np.repeat(fname, len(locs)), locs, kf)

    assert result.repaired == 3                    # distinct frames 1, 3, 4
    assert result.stale == 1                       # frame 3
    assert np.array_equal(result.keys, true[locs])
    assert sorted(result.graphs) == sorted(keys_to_hex(true[[1, 3, 4]]))
    for h, row in zip(keys_to_hex(true[[1, 3, 4]]), rows[[1, 3, 4]]):
        assert result.graphs[h] == ToyCache.graph(row)
    # only rows that changed are written, one per distinct frame
    assert writes == [(_key_file(fname), [1, 3])]
    assert np.array_equal(np.load(_key_file(fname)), true)
    assert _key_file(fname) not in NPY_CACHE._cache
    assert gk.repair_stats()['repaired'] == 3 and gk.repair_stats()['stale'] == 1


def test_repair_creates_a_missing_key_file(tmp_path):
    fname = _trajectory(tmp_path, 5)
    rows = _rows(fname)
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    locs = np.array([4, 1])
    result = gk.repair(np.zeros((2, 32), np.uint8), [fname] * 2, locs, kf)
    assert result.repaired == 2 and result.stale == 0
    stored = np.load(_key_file(fname))
    assert stored.shape == (5, 32) and stored.dtype == np.uint8
    assert np.array_equal(stored[locs], graph_keys(rows[locs]))
    assert not stored[[0, 2, 3]].any()
    with open(_key_file(fname), 'rb') as fh:     # update_npy's fixed header
        assert fh.read(128)[-1:] == b'\n'


def test_repair_across_files_and_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(gk, '_REPAIR_CHUNK', 2)
    a = _trajectory(tmp_path, 5, stem='a', seed=1)
    b = _trajectory(tmp_path, 3, stem='b', seed=2)
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    fnames = np.array([a, b, a, a, b, a])
    locs = np.array([0, 2, 4, 2, 0, 3])
    result = gk.repair(np.zeros((6, 32), np.uint8), fnames, locs, kf,
                       keep_graphs=False)
    assert result.graphs == {}
    assert result.repaired == 6
    expected = np.concatenate([graph_keys(_rows(f, [l])) for f, l in zip(fnames, locs)])
    assert np.array_equal(result.keys, expected)
    assert np.array_equal(np.load(_key_file(a))[[0, 2, 3, 4]], graph_keys(_rows(a, [0, 2, 3, 4])))
    assert np.array_equal(np.load(_key_file(b))[[0, 2]], graph_keys(_rows(b, [0, 2])))


def test_repair_of_an_unreadable_trajectory_raises(tmp_path):
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    with pytest.raises(RuntimeError, match='graph keys'):
        gk.repair(np.zeros((1, 32), np.uint8), [str(tmp_path / 'gone.xtc')], [0], kf)
    fname = _trajectory(tmp_path, 3)
    with pytest.raises(RuntimeError, match='frame 7'):
        gk.repair(np.zeros((1, 32), np.uint8), [fname], [7], kf)


# --------------------------------------------------------- call_with_repair --
def test_call_with_repair_fills_zero_rows_before_calling(tmp_path):
    fname = _trajectory(tmp_path, 4)
    rows = _rows(fname)
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    locs = np.arange(4)
    values = gk.call_with_repair(toy.values, np.zeros((4, 32), np.uint8),
                                 [fname] * 4, locs, kf)
    assert np.array_equal(values, [ToyCache.graph(r)[1] for r in rows])
    assert toy.calls.count(('keys', 4)) == 1, toy.calls      # no miss, no retry
    assert np.array_equal(np.load(_key_file(fname)), graph_keys(rows))
    assert gk.repair_stats()['filled'] == 4


def test_call_with_repair_repairs_a_miss_and_retries_once(tmp_path, capsys):
    fname = _trajectory(tmp_path, 5)
    rows = _rows(fname)
    keys = graph_keys(rows)
    toy = ToyCache()
    toy.cache(rows)
    del toy.store[keys_to_hex(keys[2:3])[0]]       # one graph lost
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    locs = np.array([0, 2, 4, 2])
    values = gk.call_with_repair(toy.values, keys[locs], [fname] * 4, locs, kf)
    assert np.array_equal(values, [ToyCache.graph(r)[1] for r in rows[locs]])
    assert [c for c in toy.calls if c[0] == 'keys' and c[1] == 4] == [('keys', 4)] * 2
    assert toy.built == keys_to_hex(keys[2:3])
    assert 'repaired 1 frame' in capsys.readouterr().out
    assert gk.repair_stats()['retries'] == 1


def test_sub_batching_function_is_repaired_in_one_retry(tmp_path):
    """values_function splits batches; the first miss must not hide the others."""
    fname = _trajectory(tmp_path, 6)
    rows = _rows(fname)
    keys = graph_keys(rows)
    toy = ToyCache()
    toy.cache(rows[[0, 2, 4]])
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    key_calls = []

    def values_function(x):
        if is_key_batch(x):
            key_calls.append(len(x))
        return np.concatenate([toy.values(x[i:i + 2]) for i in range(0, len(x), 2)])

    locs = np.arange(6)
    values = gk.call_with_repair(values_function, keys, [fname] * 6, locs, kf)
    assert np.array_equal(values, [ToyCache.graph(r)[1] for r in rows])
    assert key_calls == [6, 6]
    assert gk.repair_stats()['fallbacks'] == 0


def test_call_with_repair_falls_back_to_coordinates(tmp_path, capsys):
    """A miss that survives the repair is evaluated from the trajectory."""
    fname = _trajectory(tmp_path, 4)
    rows = _rows(fname)
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    calls = []

    def stubborn(x):
        if is_key_batch(x):
            calls.append('keys')
            raise GraphCacheMiss(x[:1])
        calls.append('rows')
        return np.asarray(x, dtype=np.float64).sum(axis=1)

    locs = np.array([3, 0, 3])
    values = gk.call_with_repair(stubborn, graph_keys(rows[locs]), [fname] * 3,
                                 locs, kf)
    assert np.array_equal(values, rows[locs].astype(np.float64).sum(axis=1))
    assert calls == ['keys', 'keys', 'rows']
    assert '!! graph keys' in capsys.readouterr().out
    assert gk.repair_stats()['fallbacks'] == 1


def test_call_with_repair_forwards_system_id(tmp_path):
    fname = _trajectory(tmp_path, 3)
    rows = _rows(fname)
    toys = {'s1': ToyCache(), 's2': ToyCache()}

    def transform(x, system_id=None):
        return toys[system_id].transform(x)

    def values(x, system_id=None):
        return toys[system_id].values(x)

    def per_system_descriptors(trajectory, system_id=None):
        assert system_id == 's2'
        return descriptors_function(trajectory)

    kf = gk.GraphKeysFunction(per_system_descriptors, transform)
    locs = np.arange(3)
    out = gk.call_with_repair(gk.KeyedFunction(values, kf), np.zeros((3, 32), np.uint8),
                              [fname] * 3, locs, kf, system_id='s2')
    assert np.array_equal(out, [ToyCache.graph(r)[1] for r in rows])
    assert toys['s1'].built == [] and len(toys['s2'].built) == 3


def test_writer_store_that_gives_up_is_not_fatal_toy(tmp_path):
    """R4 with the toy cache: no store lands, values are still right."""
    fname = _trajectory(tmp_path, 4)
    rows = _rows(fname)
    for collect in (True, False):
        toy = ToyCache(fail_store=True, collect=collect)
        kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
        locs = np.arange(4)
        keys = kf(_reader(fname, locs))
        assert toy.store == {}
        values = gk.call_with_repair(toy.values, keys, [fname] * 4, locs, kf)
        assert np.array_equal(values, [ToyCache.graph(r)[1] for r in rows])
        assert gk.repair_stats()['fallbacks'] == 0, 'served by the retry'


# ------------------------------------------------------------------ verify --
def test_load_keys_pads_missing_and_short_files(tmp_path):
    a = _trajectory(tmp_path, 4, stem='a')
    b = _trajectory(tmp_path, 4, stem='b')
    stored = graph_keys(_rows(a))[:2]
    save_npy(_key_file(a), stored)
    keys = gk.load_keys([a, a, a, b], [1, 0, 3, 2])
    assert keys.shape == (4, 32) and keys.dtype == np.uint8
    assert np.array_equal(keys[:2], stored[[1, 0]])
    assert not keys[2:].any()
    save_npy(_key_file(b), np.zeros((4, 31), np.uint8))
    with pytest.raises(RuntimeError, match='not a graph-key file'):
        gk.load_keys([b], [0])


def test_frame_refs_of_paths_and_ensembles(tmp_path):
    from aimmd import PathEnsemble
    a = _trajectory(tmp_path, 4, stem='a')
    b = _trajectory(tmp_path, 3, stem='b')
    path = Path(a)[1:3] + Path(b)[::-1]
    fnames, locs = gk.frame_refs(path)
    assert fnames.tolist() == [a, a, b, b, b]
    assert locs.tolist() == [1, 2, 2, 1, 0]
    ensemble = PathEnsemble([Path(a), Path(b)])
    fnames, locs = gk.frame_refs([ensemble, Path(b)[:1]])
    assert fnames.tolist() == [a] * 4 + [b] * 3 + [b]
    assert locs.tolist() == [0, 1, 2, 3, 0, 1, 2, 0]


def test_verify_repairs_zero_rows_and_missing_graphs(tmp_path, monkeypatch):
    monkeypatch.delenv('AIMMD_GRAPHKEYS_VERIFY', raising=False)
    a = _trajectory(tmp_path, 5, stem='a', seed=1)
    b = _trajectory(tmp_path, 4, stem='b', seed=2)
    toy = ToyCache()
    toy.cache(_rows(a))
    toy.cache(_rows(b)[:3])                        # b[3] has no graph
    keys_a = graph_keys(_rows(a))
    keys_a[[1, 4]] = 0
    save_npy(_key_file(a), keys_a)                 # b has no key file at all
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)

    counts = gk.verify([Path(a), Path(b)], kf)
    assert counts['frames'] == 9
    assert counts['zero'] == 2 + 4
    assert counts['missing'] == 0, 'zero rows are counted as zero, not missing'
    assert counts['repaired'] == 6 and counts['stale'] == 0
    assert np.array_equal(np.load(_key_file(a)), graph_keys(_rows(a)))
    assert np.array_equal(np.load(_key_file(b)), graph_keys(_rows(b)))
    assert keys_to_hex(graph_keys(_rows(b)[3:]))[0] in toy.store

    del toy.store[keys_to_hex(graph_keys(_rows(a)[2:3]))[0]]
    counts = gk.verify(Path(a), kf)
    assert counts['zero'] == 0 and counts['missing'] == 1 and counts['repaired'] == 1
    counts = gk.verify(Path(a), kf, repair_missing=False)
    assert counts == {**counts, 'missing': 0, 'repaired': 0}


@pytest.mark.parametrize('value', ['0', 'off', 'false'])
def test_verify_can_be_switched_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv('AIMMD_GRAPHKEYS_VERIFY', value)
    toy = ToyCache()
    kf = gk.GraphKeysFunction(descriptors_function, toy.transform)
    assert gk.verify(Path(_trajectory(tmp_path, 2)), kf) is None
    assert toy.calls == []


# ------------------------------------------------------- with graph_utils --
def _import_graph_utils():
    return pytest.importorskip(
        "aimmd.network.graph_utils",
        reason="graph utility tests require the optional graph/GNN dependencies")


def _universe():
    import MDAnalysis as mda
    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("types", ["C", "H", "O"])
    universe.add_TopologyAttr("names", ["C1", "H1", "O1"])
    universe.add_TopologyAttr("bonds", [(0, 1), (1, 2)])
    universe.trajectory.ts.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=np.float32)
    return universe


class _GraphParams:
    """The functions of a GNN params file, on a 3-atom system."""

    def __init__(self, gu, conns):
        self.gu = gu
        self.conns = conns
        self.universe = _universe()

    def descriptor_transform(self, x, system_id=None):
        return self.gu.process_descriptors_pyg(
            x, mdanalysis_universe=self.universe, system_selection="index 0 1",
            environment_selection="index 2", cutoff=2.0,
            conn=self.conns[system_id])['data_list']

    def values_function(self, x, system_id=None, batchsize=2):
        out = []
        for i in range(0, len(x), batchsize):         # sub-batches, as in production
            for g in self.descriptor_transform(x[i:i + batchsize], system_id=system_id):
                out.append(float(g['positions'].double().sum())
                           + 1000.0 * g['edge_index'].shape[1])
        return np.array(out)

    def descriptors_function(self, trajectory, system_id=None):
        return self.gu.atom_coordinate_descriptors_function(trajectory)


def _graph_trajectory(tmp_path, n_frames=4, stem='traj'):
    rng = np.random.default_rng(3)
    base = np.array([[4.0, 4.0, 4.0], [5.0, 4.0, 4.0], [5.0, 5.0, 4.0]], dtype=np.float32)
    positions = base + rng.uniform(-0.3, 0.3, (n_frames, 3, 3)).astype(np.float32)
    return write_trajectory(tmp_path, stem=stem, positions=positions)


@pytest.mark.graph
@pytest.mark.parametrize('role', ['writer', 'reader', 'reader_dropped_backlog'])
def test_store_that_does_not_land_is_never_fatal(tmp_path, monkeypatch, role):
    """Review R4: the graphs of a store that did not land are still served.

    writer: the store gives up after lock contention (``_store_blobs`` returns
    False), so nothing reaches the database or the memo.
    reader: trainer role with no replica and no memo; new graphs only reach
    the pending backlog. ``reader_dropped_backlog`` also loses that backlog,
    as a flush that loses the lock does.
    Today's coordinate path returns the right values in all three; so must
    the key path, without raising.
    """
    gu = _import_graph_utils()
    if role == 'writer':
        monkeypatch.setattr(gu, '_store_blobs', lambda conn, keys, blobs: False)
    else:
        monkeypatch.setenv('AIMMD_GRAPH_MEMO_BYTES', '0')
        monkeypatch.setenv('AIMMD_SHM_DIR', 'off')
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    if role != 'writer':
        assert conn._aimmd_memo is None
        shm_cache.set_reader_role()
    params = _GraphParams(gu, {None: conn})
    fname = _graph_trajectory(tmp_path, 4)
    locs = np.arange(4)
    rows = _rows(fname)
    expected = params.values_function(rows)        # today's coordinate path

    kf = gk.GraphKeysFunction(params.descriptors_function, params.descriptor_transform)
    keys = kf(_reader(fname, locs))                # ingestion: never raises
    assert np.array_equal(keys, graph_keys(rows))
    assert conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0] == 0
    if role == 'reader_dropped_backlog':
        shm_cache.take_pending(conn)

    values = gk.call_with_repair(gk.KeyedFunction(params.values_function, kf),
                                 keys, [fname] * 4, locs, kf)
    assert np.array_equal(values, expected)
    assert gk.repair_stats()['fallbacks'] == 0
    conn.close()


@pytest.mark.graph
def test_key_values_equal_coordinate_values(tmp_path):
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    params = _GraphParams(gu, {None: conn})
    fname = _graph_trajectory(tmp_path, 5)
    rows = _rows(fname)
    kf = gk.GraphKeysFunction(params.descriptors_function, params.descriptor_transform)
    keys = kf(_reader(fname, np.arange(5)))
    assert conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0] == 5
    assert kf.connection() is conn
    assert np.array_equal(params.values_function(keys), params.values_function(rows))
    conn.close()


@pytest.mark.graph
def test_each_system_asks_only_its_own_cache(tmp_path):
    """Multi-system: a graph cached for another system does not count."""
    gu = _import_graph_utils()
    conns = {'s1': gu.init_db(str(tmp_path / 's1.sqlite')),
             's2': gu.init_db(str(tmp_path / 's2.sqlite'))}
    params = _GraphParams(gu, conns)
    fname = _graph_trajectory(tmp_path, 3)
    kf = gk.GraphKeysFunction(params.descriptors_function, params.descriptor_transform)
    kf(_reader(fname, np.arange(3)), system_id='s2')
    assert kf.connection('s1') is conns['s1']
    assert kf.connection('s2') is conns['s2']
    keys = kf(_reader(fname, np.arange(3)), system_id='s1')
    for conn in conns.values():
        assert conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0] == 3
        assert gu.graphs_present(keys, conn).all()
    for conn in conns.values():
        conn.close()


@pytest.mark.graph
def test_verify_scans_the_database_and_repairs(tmp_path, monkeypatch):
    monkeypatch.delenv('AIMMD_GRAPHKEYS_VERIFY', raising=False)
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    params = _GraphParams(gu, {None: conn})
    fname = _graph_trajectory(tmp_path, 5)
    kf = gk.GraphKeysFunction(params.descriptors_function, params.descriptor_transform)
    keys = kf(_reader(fname, np.arange(5)))
    save_npy(_key_file(fname), keys)
    assert gk.verify(Path(fname), kf)['missing'] == 0

    lost = keys_to_hex(keys[1:2])[0]
    conn.execute('DELETE FROM graphs_cache WHERE key = ?', (lost,))
    conn.commit()
    conn._aimmd_memo.clear()
    counts = gk.verify(Path(fname), kf)
    assert counts['missing'] == 1 and counts['repaired'] == 1
    assert conn.execute('SELECT 1 FROM graphs_cache WHERE key = ?', (lost,)).fetchone()
    plan = ' '.join(str(r) for r in conn.execute(
        'EXPLAIN QUERY PLAN ' + gk._KEY_SCAN).fetchall())
    assert 'COVERING INDEX' in plan
    conn.close()
