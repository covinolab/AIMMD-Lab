"""Tests for looking graphs up by key (aimmd.network.graph_lookup).

With ``descriptor_cache='graphkeys'`` network consumers receive graph keys,
``(n, 32)`` uint8 rows, instead of coordinate rows, and only look graphs up.
These tests cover the torch-free lookup layer in the default suite -- the
cache fixtures are plain sqlite tables of pickled payloads, which is all the
layer sees -- and, with ``--rungraph``, the key-row dispatch of
``graph_utils.process_descriptors_pyg`` on real graphs.

What they pin down:

- ``graphs_present`` asks ONE connection, in batched ``IN`` queries, and
  never reports a zero row as present;
- a lookup consults overlay, memo, pending backlog, replica and database in
  that order, and raises ``GraphCacheMiss`` carrying exactly the missing
  keys (zero rows included) without decoding anything;
- the overlay serves graphs that were just built, so a store that gave up or
  an evicted memo does not lose them (review item R4).
"""

import os
import pickle
import sqlite3

import numpy as np
import pytest

from aimmd.core.graphkey import graph_keys, keys_to_hex
from aimmd.network import graph_lookup, shm_cache


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# --------------------------------------------------------------- fixtures --
def _rows(n, seed=0, width=9):
    return np.random.default_rng(seed).standard_normal((n, width)).astype(np.float32)


def _make_cache(path, payloads):
    """A stand-in graph cache: same schema as init_db, pickled payloads."""
    conn = sqlite3.connect(str(path), factory=shm_cache.CacheConnection)
    conn.execute('CREATE TABLE IF NOT EXISTS graphs_cache'
                 '(key TEXT PRIMARY KEY, data BLOB)')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executemany('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                     [(k, pickle.dumps(v)) for k, v in payloads.items()])
    conn.commit()
    conn._aimmd_db_path = os.path.abspath(str(path))
    shm_cache.register(conn)
    return conn


def _payloads(rows):
    """{hex key: payload} with a payload that names its frame."""
    return {h: {'frame': i} for i, h in enumerate(keys_to_hex(graph_keys(rows)))}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    shm = tmp_path / 'shm'
    shm.mkdir()
    monkeypatch.setenv('AIMMD_SHM_DIR', str(shm))
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    shm_cache._REGISTRY.clear()
    shm_cache._OWNED.clear()
    shm_cache._WARNED.clear()
    for k in shm_cache._STATS:
        shm_cache._STATS[k] = 0
    yield shm
    shm_cache.cleanup_replicas()
    assert graph_lookup._OVERLAYS == [], 'an overlay leaked out of its block'


class _Counting:
    """Count the statements a connection executes."""

    def __init__(self, monkeypatch, conn):
        self.statements = []
        real = conn.execute

        def execute(sql, *args):
            self.statements.append(sql)
            return real(sql, *args)
        monkeypatch.setattr(conn, 'execute', execute, raising=False)


# -------------------------------------------------------- GraphCacheMiss --
def test_graph_cache_miss_carries_keys_and_pickles():
    keys = graph_keys(_rows(3))
    miss = graph_lookup.GraphCacheMiss(keys)
    assert isinstance(miss, LookupError)
    assert miss.keys.dtype == np.uint8 and miss.keys.shape == (3, 32)
    assert np.array_equal(miss.keys, keys)
    assert '3 graph(s)' in str(miss)
    back = pickle.loads(pickle.dumps(miss))
    assert isinstance(back, graph_lookup.GraphCacheMiss)
    assert np.array_equal(back.keys, keys)


# --------------------------------------------------------- graphs_present --
def test_graphs_present_checks_the_database(tmp_path):
    rows = _rows(6)
    keys = graph_keys(rows)
    conn = _make_cache(tmp_path / 'g.sqlite', dict(list(_payloads(rows).items())[:4]))
    present = graph_lookup.graphs_present(keys, conn)
    assert present.dtype == bool
    assert present.tolist() == [True] * 4 + [False] * 2


def test_zero_rows_are_never_present(tmp_path):
    conn = _make_cache(tmp_path / 'g.sqlite', {bytes(32).hex(): 'zero'})
    keys = np.zeros((3, 32), np.uint8)
    assert not graph_lookup.graphs_present(keys, conn).any()


def test_graphs_present_uses_batched_in_queries(tmp_path, monkeypatch):
    rows = _rows(1200)
    conn = _make_cache(tmp_path / 'g.sqlite', _payloads(rows))
    conn._aimmd_memo = None
    counting = _Counting(monkeypatch, conn)
    assert graph_lookup.graphs_present(graph_keys(rows), conn).all()
    selects = [s for s in counting.statements if 'graphs_cache' in s]
    assert 1 <= len(selects) <= 3, selects
    assert all(' IN ' in s for s in selects)


def test_graphs_present_asks_only_the_given_connection(tmp_path):
    rows = _rows(4)
    payloads = _payloads(rows)
    mine = _make_cache(tmp_path / 'mine.sqlite', dict(list(payloads.items())[:2]))
    _make_cache(tmp_path / 'other.sqlite', payloads)     # holds all four
    assert graph_lookup.graphs_present(graph_keys(rows), mine).tolist() == [
        True, True, False, False]


@pytest.mark.parametrize('layer', ['overlay', 'memo', 'pending', 'replica'])
def test_graphs_present_sees_every_layer(tmp_path, layer):
    rows = _rows(2)
    keys = graph_keys(rows)
    hexes = keys_to_hex(keys)
    conn = _make_cache(tmp_path / 'g.sqlite', {})
    blob = pickle.dumps('graph')
    if layer == 'memo':
        conn._aimmd_memo.put(hexes[0], blob)
    elif layer == 'pending':
        shm_cache.buffer_write(conn, [hexes[0]], [blob])
    elif layer == 'replica':
        conn.execute('INSERT INTO graphs_cache VALUES (?,?)', (hexes[0], blob))
        conn.commit()
        assert shm_cache.stage_cache(conn) is not None
        conn.execute('DELETE FROM graphs_cache')    # only the replica has it
        conn.commit()
    if layer == 'overlay':
        with graph_lookup.graph_overlay({hexes[0]: 'graph'}):
            present = graph_lookup.graphs_present(keys, conn)
    else:
        present = graph_lookup.graphs_present(keys, conn)
    assert present.tolist() == [True, False]


def test_graphs_present_on_an_empty_batch(tmp_path):
    conn = _make_cache(tmp_path / 'g.sqlite', {})
    assert graph_lookup.graphs_present(np.zeros((0, 32), np.uint8), conn).shape == (0,)


# ---------------------------------------------------------- lookup_graphs --
def test_lookup_returns_graphs_in_row_order(tmp_path):
    rows = _rows(5)
    conn = _make_cache(tmp_path / 'g.sqlite', _payloads(rows))
    keys = graph_keys(rows)[[3, 0, 3, 4]]                # a frame drawn twice
    graphs = graph_lookup.lookup_graphs(keys, conn, decode=pickle.loads)
    assert graphs == [{'frame': 3}, {'frame': 0}, {'frame': 3}, {'frame': 4}]
    assert graphs[0] is not graphs[2], 'each row gets its own decoded graph'


def test_lookup_raises_with_exactly_the_missing_keys(tmp_path):
    rows = _rows(6)
    keys = graph_keys(rows)
    conn = _make_cache(tmp_path / 'g.sqlite', dict(list(_payloads(rows).items())[:3]))
    batch = np.concatenate([keys[[4, 0, 5, 4]], np.zeros((2, 32), np.uint8),
                            keys[[1]]])
    decoded = []

    def decode(blob):
        decoded.append(blob)
        return pickle.loads(blob)

    with pytest.raises(graph_lookup.GraphCacheMiss) as info:
        graph_lookup.lookup_graphs(batch, conn, decode=decode)
    # distinct, in order of first appearance; the zero row is a miss too
    expected = np.stack([keys[4], keys[5], np.zeros(32, np.uint8)])
    assert np.array_equal(info.value.keys, expected)
    assert decoded == [], 'nothing is decoded for a batch that misses'


def test_lookup_order_overlay_memo_pending_replica_database(tmp_path, monkeypatch):
    rows = _rows(5)
    hexes = keys_to_hex(graph_keys(rows))
    conn = _make_cache(tmp_path / 'g.sqlite', {hexes[3]: 'replica', hexes[4]: 'db'})
    assert shm_cache.stage_cache(conn) is not None
    conn.execute('DELETE FROM graphs_cache WHERE key = ?', (hexes[3],))
    conn.execute('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                 (hexes[4], pickle.dumps('db')))
    conn.commit()
    conn._aimmd_replica.execute('PRAGMA query_only=OFF')
    conn._aimmd_replica.execute('DELETE FROM graphs_cache WHERE key = ?', (hexes[4],))
    conn._aimmd_replica.commit()
    conn._aimmd_memo.put(hexes[1], pickle.dumps('memo'))
    shm_cache.buffer_write(conn, [hexes[2]], [pickle.dumps('pending')])
    # lower layers hold stale stand-ins: the first layer that has a key wins
    shm_cache.buffer_write(conn, [hexes[1]], [pickle.dumps('pending-shadowed')])

    with graph_lookup.graph_overlay({hexes[0]: 'overlay'}):
        graphs = graph_lookup.lookup_graphs(graph_keys(rows), conn,
                                            decode=pickle.loads)
    assert graphs == ['overlay', 'memo', 'pending', 'replica', 'db']
    # replica and database hits are mirrored into the memo, as load_from_sqlite does
    assert conn._aimmd_memo.get(hexes[3]) is not None
    assert conn._aimmd_memo.get(hexes[4]) is not None


def test_lookup_survives_a_broken_replica(tmp_path):
    rows = _rows(2)
    conn = _make_cache(tmp_path / 'g.sqlite', _payloads(rows))
    conn._aimmd_memo = None
    assert shm_cache.stage_cache(conn) is not None
    conn._aimmd_replica.close()          # any sqlite error on the replica
    graphs = graph_lookup.lookup_graphs(graph_keys(rows), conn, decode=pickle.loads)
    assert graphs == [{'frame': 0}, {'frame': 1}]
    assert conn._aimmd_replica is None, 'a broken replica is detached'


def test_lookup_reader_role_serves_the_pending_backlog(tmp_path, monkeypatch):
    """R4, reader side: no replica and no memo, the graph only in pending."""
    monkeypatch.setenv('AIMMD_GRAPH_MEMO_BYTES', '0')
    rows = _rows(2)
    hexes = keys_to_hex(graph_keys(rows))
    conn = _make_cache(tmp_path / 'g.sqlite', {})
    assert conn._aimmd_memo is None
    shm_cache.set_reader_role()
    shm_cache.buffer_write(conn, hexes, [pickle.dumps('a'), pickle.dumps('b')])
    assert graph_lookup.lookup_graphs(graph_keys(rows), conn,
                                      decode=pickle.loads) == ['a', 'b']


def test_lookup_of_an_empty_batch(tmp_path):
    conn = _make_cache(tmp_path / 'g.sqlite', {})
    assert graph_lookup.lookup_graphs(np.zeros((0, 32), np.uint8), conn,
                                      decode=pickle.loads) == []


def test_capture_connection_records_the_cache_a_lookup_uses(tmp_path):
    conn = _make_cache(tmp_path / 'g.sqlite', {})
    with graph_lookup.capture_connection() as seen:
        graph_lookup.lookup_graphs(np.zeros((0, 32), np.uint8), conn,
                                   decode=pickle.loads)
    assert seen == [conn]
    # outside the block nothing is recorded
    graph_lookup.lookup_graphs(np.zeros((0, 32), np.uint8), conn, decode=pickle.loads)
    assert seen == [conn]


# ---------------------------------------------------------------- overlay --
def test_overlay_nesting_and_collection():
    assert graph_lookup.overlay_get('a') is None
    with graph_lookup.graph_overlay({'a': 1}) as outer:
        with graph_lookup.graph_overlay() as inner:
            assert graph_lookup.overlay_get('a') == 1
            graph_lookup.collect_graphs(['b', 'c'], [2, None])
            assert inner == {'b': 2}
            assert graph_lookup.overlay_get('b') == 2
        assert outer == {'a': 1, 'b': 2}, 'collection reaches every open overlay'
        assert graph_lookup.overlay_get('b') == 2
    assert graph_lookup.overlay_get('a') is None
    graph_lookup.collect_graphs(['d'], [4])          # no overlay open: a no-op
    assert graph_lookup._OVERLAYS == []


def test_equal_overlays_close_in_the_right_order():
    """Overlays are removed by identity, not by equality."""
    with graph_lookup.graph_overlay() as first:
        with graph_lookup.graph_overlay() as second:
            assert first == second and first is not second
        assert graph_lookup._OVERLAYS == [first]
        assert graph_lookup._OVERLAYS[0] is first
    assert graph_lookup._OVERLAYS == []


# ------------------------------------------- process_descriptors_pyg (graph) --
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


def _frames(n=4):
    base = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.stack([base * (1 + 0.05 * i) for i in range(n)]).astype(np.float32)


def _pyg(gu, descriptors, conn):
    return gu.process_descriptors_pyg(
        descriptors, mdanalysis_universe=_universe(), system_selection="index 0 1",
        environment_selection="index 2", cutoff=2.0, conn=conn)


def _same_graph(a, b):
    import torch
    return all(torch.equal(a[k], b[k])
               for k in ('positions', 'edge_index', 'node_attrs', 'shifts'))


@pytest.mark.graph
def test_graph_utils_re_exports_the_lookup_layer():
    gu = _import_graph_utils()
    assert gu.GraphCacheMiss is graph_lookup.GraphCacheMiss
    assert gu.graphs_present is graph_lookup.graphs_present
    assert gu.graph_overlay is graph_lookup.graph_overlay
    assert gu.is_key_batch(graph_keys(_frames()))


@pytest.mark.graph
def test_key_rows_return_the_graphs_of_the_coordinate_rows(tmp_path, monkeypatch):
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    rows = _frames()
    built = _pyg(gu, rows, conn)
    keys = graph_keys(rows)

    def _must_not_build(*args, **kwargs):
        raise AssertionError('a key lookup must never build graphs')
    monkeypatch.setattr(gu, 'get_graphs_pyg', _must_not_build)
    conn._aimmd_memo.clear()                         # served from the database
    looked_up = _pyg(gu, keys[[2, 0, 2, 3, 1]], conn)
    assert isinstance(looked_up, gu.GraphList)
    assert looked_up['data_list'] == list(looked_up)
    for got, want in zip(looked_up, [built[i] for i in (2, 0, 2, 3, 1)]):
        assert _same_graph(got, want)
    conn.close()


@pytest.mark.graph
def test_key_rows_raise_graph_cache_miss_and_never_build(tmp_path, monkeypatch):
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    rows = _frames(4)
    _pyg(gu, rows[:2], conn)                          # frames 2 and 3 not cached
    keys = graph_keys(rows)
    monkeypatch.setattr(gu, 'get_graphs_pyg',
                        lambda *a, **k: pytest.fail('built a graph'))
    batch = np.concatenate([keys, np.zeros((1, 32), np.uint8)])
    with pytest.raises(gu.GraphCacheMiss) as info:
        _pyg(gu, batch, conn)
    assert np.array_equal(info.value.keys,
                          np.stack([keys[2], keys[3], np.zeros(32, np.uint8)]))
    assert conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0] == 2
    conn.close()


@pytest.mark.graph
def test_coordinate_rows_fill_an_open_overlay(tmp_path):
    """Graphs the coordinate path returns are collected into an open overlay."""
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    rows = _frames(3)
    with gu.graph_overlay() as overlay:
        built = _pyg(gu, rows, conn)
    assert sorted(overlay) == sorted(keys_to_hex(graph_keys(rows)))
    for h, graph in zip(keys_to_hex(graph_keys(rows)), built):
        assert overlay[h] is graph
    conn.close()


@pytest.mark.graph
def test_writer_whose_store_gave_up_is_served_from_the_overlay(tmp_path, monkeypatch):
    """R4, writer side: the store gives up, the overlay still serves the graphs."""
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    monkeypatch.setattr(gu, '_store_blobs', lambda conn, keys, blobs: False)
    rows = _frames(3)
    keys = graph_keys(rows)
    with gu.graph_overlay():
        built = _pyg(gu, rows, conn)                 # store gives up quietly
        assert gu.graphs_present(keys, conn).all()
        served = _pyg(gu, keys, conn)
    assert [g is b for g, b in zip(served, built)] == [True] * 3
    # outside the overlay the graphs are (correctly) missing
    assert not gu.graphs_present(keys, conn).any()
    with pytest.raises(gu.GraphCacheMiss):
        _pyg(gu, keys, conn)
    conn.close()
