"""Optional tests for the graph utility layer.

These tests are intentionally separated from the default unit-test suite because
`aimmd.network.graph_utils` depends on the larger graph/GNN stack
(`mlcolvar`, `torch_geometric`, `torch_cluster`, `mdtraj`, ...).

How to run
----------
From the repository root, enable these tests explicitly with:

    pytest tests/test_graph_utils_optional.py --rungraph

Or run the full suite including them with:

    pytest --rungraph
"""

import os

import numpy as np
import pytest


# Graph imports can trigger matplotlib cache initialization through the wider
# dependency stack, so we direct that cache to a writable temp location.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


pytestmark = pytest.mark.graph


def _import_graph_utils():
    """Import the graph utilities lazily so the default test run stays light."""

    return pytest.importorskip(
        "aimmd.network.graph_utils",
        reason="graph utility tests require the optional graph/GNN dependencies",
    )


def _graph_test_universe():
    """Create the smallest boxed, bonded MDAnalysis universe that still works.

    The MDAnalysis-based graph path uses `unwrap`/`center_in_box`/`wrap`, which
    means our synthetic system must provide:
    - atom names and types for node features,
    - bonds so fragments exist for `unwrap`,
    - box dimensions so periodic-box transforms can run.
    """

    import MDAnalysis as mda

    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("types", ["C", "H", "O"])
    universe.add_TopologyAttr("names", ["C1", "H1", "O1"])
    universe.add_TopologyAttr("bonds", [(0, 1), (1, 2)])
    universe.trajectory.ts.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=np.float32)
    return universe


def _graph_descriptors():
    """Two tiny coordinate frames, flattened the way AIMMD caches descriptors."""

    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0],
        ],
        dtype=float,
    )


def test_graph_sqlite_roundtrip_and_hash_stability(tmp_path):
    """Graphs stored in SQLite should round-trip for all supported codecs.

    This test documents two low-level invariants:
    - hashing the same configuration twice should give the same cache key;
    - a stored `torch_geometric.data.Data` object should deserialize to the
      same tensor payload regardless of the configured compression backend.
    """

    graph_utils = _import_graph_utils()
    from torch_geometric.data import Data
    import torch

    conn = graph_utils.init_db(str(tmp_path / "graphs.sqlite"))
    try:
        key = graph_utils.get_stable_hash(np.array([1.0, 2.0, 3.0]))
        assert key == graph_utils.get_stable_hash(np.array([1.0, 2.0, 3.0]))

        graph = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        for compression in ("none", "gzip", "lz4"):
            graph_utils.store_in_sqlite(key + compression, graph, conn, compression_lib=compression)
            loaded = graph_utils.load_from_sqlite(key + compression, conn, compression_lib=compression)
            np.testing.assert_allclose(loaded.x.numpy(), graph.x.numpy())
            np.testing.assert_array_equal(loaded.edge_index.numpy(), graph.edge_index.numpy())
    finally:
        conn.close()


def test_atom_coordinate_descriptors_function_reads_frames_verbatim():
    """The descriptor helper should flatten positions frame-by-frame.

    This is the lowest-level graph-facing descriptor function: it takes an
    MDAnalysis trajectory and returns one flat coordinate vector per frame.
    """

    graph_utils = _import_graph_utils()
    universe = _graph_test_universe()
    universe.load_new(
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            ],
            dtype=np.float32,
        )
    )

    descriptors = graph_utils.atom_coordinate_descriptors_function(universe.trajectory)
    assert descriptors.shape == (2, 9)
    np.testing.assert_allclose(descriptors[0], np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float))


def test_get_graphs_pyg_builds_graph_objects_with_expected_payload():
    """`get_graphs_pyg` should turn flat coordinates into PyG graph objects.

    The expectation is intentionally structural rather than overly specific:
    - we should get one graph per input frame,
    - each graph should contain all selected atoms,
    - node attributes should be one-hot encoded atom types,
    - and a non-empty edge list should be produced for this tiny connected
      system at the chosen cutoff.
    """

    graph_utils = _import_graph_utils()
    universe = _graph_test_universe()
    descriptors = _graph_descriptors()

    graphs = graph_utils.get_graphs_pyg(
        descriptors=descriptors,
        mdanalysis_universe=universe,
        system_selection="index 0 1",
        environment_selection="index 2",
        cutoff=2.0,
        verbose=False,
    )

    assert len(graphs) == 2
    first = graphs[0]
    assert first["positions"].shape == (3, 3)
    assert first["node_attrs"].shape[0] == 3
    np.testing.assert_allclose(first["node_attrs"].sum(dim=1).numpy(), np.ones(3))
    assert first.edge_index.shape[1] > 0


def test_get_graphs_pyg_fixed_atom_types_shared_encoding():
    """A fixed ``atom_types`` table gives a shared, wider one-hot encoding.

    This is the multi-system encoding: instead of deriving columns per universe
    (``sorted(set(types))`` -> 3 cols here), pass an explicit table so every
    system featurizes into the SAME columns (unused columns stay zero), and the
    network input width equals ``len(atom_types)``.
    """
    graph_utils = _import_graph_utils()
    universe = _graph_test_universe()           # atom types C, H, O
    descriptors = _graph_descriptors()
    atom_types = ['H', 'C', 'N', 'O', 'F']      # fixed, atomic-number ordered

    graphs = graph_utils.get_graphs_pyg(
        descriptors=descriptors, mdanalysis_universe=universe,
        system_selection="index 0 1", environment_selection="index 2",
        cutoff=2.0, verbose=False, atom_types=atom_types)

    node_attrs = graphs[0]["node_attrs"]
    assert node_attrs.shape[1] == len(atom_types)        # 5 columns
    # still exactly one hot per atom
    np.testing.assert_allclose(node_attrs.sum(dim=1).numpy(), np.ones(3))
    # C -> column index 1, H -> 0, O -> 3 (per the fixed table)
    assert node_attrs[0].argmax().item() == atom_types.index('C')
    assert node_attrs[1].argmax().item() == atom_types.index('H')
    assert node_attrs[2].argmax().item() == atom_types.index('O')

    # an atom type missing from the table raises a clear error
    with pytest.raises(ValueError):
        graph_utils.get_graphs_pyg(
            descriptors=descriptors, mdanalysis_universe=universe,
            system_selection="index 0 1", environment_selection="index 2",
            cutoff=2.0, atom_types=['H', 'N', 'O'])     # missing 'C'


def test_process_descriptors_pyg_populates_and_reuses_cache(tmp_path):
    """The high-level PyG path should populate SQLite once and then reuse it.

    We call the conversion twice with the same descriptors and assert that the
    cache row count stays constant on the second call, which documents the
    intended "load if present, build if missing" behavior.
    """

    graph_utils = _import_graph_utils()
    universe = _graph_test_universe()
    descriptors = _graph_descriptors()

    conn = graph_utils.init_db(str(tmp_path / "graphs.sqlite"))
    try:
        dataset1 = graph_utils.process_descriptors_pyg(
            descriptors=descriptors,
            mdanalysis_universe=universe,
            system_selection="index 0 1",
            environment_selection="index 2",
            cutoff=2.0,
            conn=conn,
            verbose=False,
            compression_lib="none",
        )
        count_after_first = conn.execute("SELECT COUNT(*) FROM graphs_cache").fetchone()[0]

        dataset2 = graph_utils.process_descriptors_pyg(
            descriptors=descriptors,
            mdanalysis_universe=universe,
            system_selection="index 0 1",
            environment_selection="index 2",
            cutoff=2.0,
            conn=conn,
            verbose=False,
            compression_lib="none",
        )
        count_after_second = conn.execute("SELECT COUNT(*) FROM graphs_cache").fetchone()[0]

        assert len(dataset1["data_list"]) == len(descriptors)
        assert len(dataset2["data_list"]) == len(descriptors)
        assert count_after_first == len(descriptors)
        assert count_after_second == count_after_first
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Batched insertion, codec handling, and the /dev/shm replica end to end.
# These need real `Data` objects, so they live here rather than in
# tests/test_shm_cache.py (which is stdlib-only and runs by default).
# ---------------------------------------------------------------------------
def _tiny_graph(gu, n=4):
    """A small but genuine torch_geometric Data, shaped like a real cache entry."""
    import torch
    from torch_geometric.data import Data
    return Data(positions=torch.randn(n, 3),
                edge_index=torch.randint(0, n, (2, n * 3)),
                node_attrs=torch.eye(n),
                shifts=torch.zeros(n * 3, 3))


def _mem_db(gu):
    import sqlite3
    conn = sqlite3.connect(':memory:', factory=gu.shm_cache.CacheConnection)
    conn.execute('CREATE TABLE graphs_cache(key TEXT PRIMARY KEY, data BLOB)')
    return conn


def test_batched_store_lz4_roundtrips():
    """A batch written with lz4 must be readable by load_from_sqlite."""
    import torch
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    graphs = [_tiny_graph(gu) for _ in range(5)]
    keys = [f'k{i}' for i in range(5)]

    gu.store_many_in_sqlite(keys, graphs, conn, compression_lib='lz4')

    for key, original in zip(keys, graphs):
        back = gu.load_from_sqlite(key, conn, compression_lib='lz4')
        assert back is not None
        assert torch.equal(back['positions'], original['positions'])
        assert torch.equal(back['edge_index'], original['edge_index'])


def test_codec_mismatch_is_survivable():
    """Regression guard for a trap that already produced one wrong conclusion.

    The module's read/write defaults are "gzip" while the pyg path passes "lz4",
    so a caller that forgets the argument used to write bytes it could not read
    back. `_decode` now identifies the container from its magic bytes, so a
    mismatched -- or even mixed -- cache stays readable.
    """
    import torch
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    graph = _tiny_graph(gu)

    gu.store_in_sqlite('gz', graph, conn, compression_lib='gzip')
    gu.store_in_sqlite('l4', graph, conn, compression_lib='lz4')
    gu.store_in_sqlite('raw', graph, conn, compression_lib='none')

    # every entry readable regardless of what the caller claims the codec is
    for key in ('gz', 'l4', 'raw'):
        for claimed in ('gzip', 'lz4', 'none'):
            back = gu.load_from_sqlite(key, conn, compression_lib=claimed)
            assert back is not None, f'{key} unreadable when asked for {claimed}'
            assert torch.equal(back['positions'], graph['positions'])


def test_batched_store_retries_on_locked_database():
    """The retry must wrap the whole transaction, and roll back between tries."""
    import sqlite3
    gu = _import_graph_utils()
    real = _mem_db(gu)
    state = {'fails': 2, 'rollbacks': 0}

    class Flaky:
        def executemany(self, *a, **k):
            if state['fails'] > 0:
                state['fails'] -= 1
                raise sqlite3.OperationalError('database is locked')
            return real.executemany(*a, **k)

        def commit(self):
            return real.commit()

        def rollback(self):
            state['rollbacks'] += 1

    gu.store_many_in_sqlite(['a'], [_tiny_graph(gu)], Flaky(), compression_lib='lz4')
    assert state['fails'] == 0
    assert state['rollbacks'] == 2, 'each retry must roll back the partial txn'
    assert real.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_batched_store_reraises_non_lock_errors():
    """Only "database is locked" is retryable; anything else must surface."""
    import sqlite3
    import pytest as _pytest
    gu = _import_graph_utils()

    class Broken:
        def executemany(self, *a, **k):
            raise sqlite3.OperationalError('no such table: graphs_cache')

        def commit(self):
            pass

        def rollback(self):
            pass

    with _pytest.raises(sqlite3.OperationalError, match='no such table'):
        gu.store_many_in_sqlite(['a'], [_tiny_graph(gu)], Broken())


def test_pyg_store_path_uses_one_transaction(monkeypatch):
    """The pyg path must commit once per batch, not once per graph."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    commits = {'n': 0}
    real_commit = conn.commit

    def counting_commit():
        commits['n'] += 1
        return real_commit()
    monkeypatch.setattr(conn, 'commit', counting_commit, raising=False)

    gu.store_many_in_sqlite([f'k{i}' for i in range(20)],
                            [_tiny_graph(gu) for _ in range(20)],
                            conn, compression_lib='lz4')
    assert commits['n'] == 1, f'{commits["n"]} commits for a 20-graph batch'


def test_replica_serves_real_graphs_end_to_end(tmp_path, monkeypatch):
    """Stage a real cache into a fake tmpfs and read genuine graphs back."""
    import sqlite3
    import torch
    gu = _import_graph_utils()
    shm = tmp_path / 'shm'
    shm.mkdir()
    monkeypatch.setenv('AIMMD_SHM_DIR', str(shm))
    gu.shm_cache._REGISTRY.clear()
    gu.shm_cache._OWNED.clear()

    db = tmp_path / 'graphs_cache.sqlite'
    conn = gu.init_db(str(db))
    graphs = [_tiny_graph(gu) for _ in range(6)]
    keys = [f'k{i}' for i in range(6)]
    gu.store_many_in_sqlite(keys, graphs, conn, compression_lib='lz4')

    try:
        assert gu.shm_cache.stage_cache(conn) is not None
        conn._aimmd_memo = None                 # isolate the replica

        calls = []
        real_execute = conn.execute
        monkeypatch.setattr(
            conn, 'execute',
            lambda *a, **k: (calls.append(a), real_execute(*a, **k))[1],
            raising=False)

        for key, original in zip(keys, graphs):
            back = gu.load_from_sqlite(key, conn, compression_lib='lz4')
            assert torch.equal(back['positions'], original['positions'])
        assert calls == [], 'replica hits still queried the real database'
        assert gu.shm_cache.replica_stats()['hits'] == 6
    finally:
        gu.shm_cache.cleanup_replicas()


# ------------------------------------ a locked cache must not kill the job --
def test_persistent_lock_does_not_raise(monkeypatch):
    """A writer that cannot get the lock must give up quietly, not raise.

    The graph cache is a cache: `process_descriptors_pyg` assembles its result
    from the in-memory `new_graphs` list and only writes so that a later process
    can skip the recompute. Raising here aborted the entire campaign -- the
    RuntimeError propagated descriptors_function -> Path.compute ->
    trajectory.extend -> execute_command.stop_condition, killed `gmx mdrun`, and
    cancelled all 36 tasks. Seen in 19 production jobs, the oldest
    calixarene_G2/slurm-852869, so it long predates the /dev/shm cache work.
    """
    import sqlite3
    gu = _import_graph_utils()
    monkeypatch.setattr(gu, '_STORE_RETRY_SECONDS', 0.2, raising=True)

    attempts = {'n': 0, 'rollbacks': 0}

    class AlwaysLocked:
        def executemany(self, *a, **k):
            attempts['n'] += 1
            raise sqlite3.OperationalError('database is locked')

        def commit(self):
            raise AssertionError('must not commit')

        def rollback(self):
            attempts['rollbacks'] += 1

    gu.store_many_in_sqlite(['a'], [_tiny_graph(gu)], AlwaysLocked(),
                            compression_lib='lz4')

    assert attempts['n'] >= 2, 'must retry, not give up on the first lock'
    assert attempts['rollbacks'] == attempts['n'], 'every attempt rolls back'


def test_persistent_lock_does_not_populate_memo_or_replica(monkeypatch):
    """Nothing may be mirrored for rows that never reached the database.

    Keeps the replica a strict subset of the real cache, which is what makes the
    rowid watermark in shm_cache.refresh_replicas sound.
    """
    import sqlite3
    gu = _import_graph_utils()
    monkeypatch.setattr(gu, '_STORE_RETRY_SECONDS', 0.2, raising=True)

    mirrored = []
    monkeypatch.setattr(gu, '_after_store',
                        lambda conn, keys, blobs: mirrored.append(keys))

    class AlwaysLocked:
        def executemany(self, *a, **k):
            raise sqlite3.OperationalError('database is locked')

        def commit(self):
            pass

        def rollback(self):
            pass

    gu.store_many_in_sqlite(['a'], [_tiny_graph(gu)], AlwaysLocked(),
                            compression_lib='lz4')
    assert mirrored == [], 'write-through must not run for an unwritten batch'


def test_single_store_is_also_non_fatal(monkeypatch):
    """`store_in_sqlite` shares the hazard -- load_or_create writes through it."""
    import sqlite3
    gu = _import_graph_utils()
    monkeypatch.setattr(gu, '_STORE_RETRY_SECONDS', 0.2, raising=True)

    class AlwaysLocked:
        def executemany(self, *a, **k):
            raise sqlite3.OperationalError('database is locked')

        def commit(self):
            pass

        def rollback(self):
            pass

    gu.store_in_sqlite('k', _tiny_graph(gu), AlwaysLocked(),
                       compression_lib='lz4')


def test_lock_that_clears_still_stores(monkeypatch):
    """Giving up must be a last resort: a lock that frees still gets written."""
    import sqlite3
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu, '_STORE_RETRY_SECONDS', 30.0, raising=True)
    state = {'fails': 3}
    real_executemany = conn.executemany

    class Clearing:
        def executemany(self, *a, **k):
            if state['fails'] > 0:
                state['fails'] -= 1
                raise sqlite3.OperationalError('database is locked')
            return real_executemany(*a, **k)

        def commit(self):
            return conn.commit()

        def rollback(self):
            pass

    gu.store_many_in_sqlite(['a'], [_tiny_graph(gu)], Clearing(),
                            compression_lib='lz4')
    assert state['fails'] == 0
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_init_db_sets_an_explicit_busy_timeout(tmp_path):
    """Do not inherit Python's 5 s default by accident -- state it."""
    gu = _import_graph_utils()
    conn = gu.init_db(db_path=str(tmp_path / 'g.sqlite'))
    got = conn.execute('PRAGMA busy_timeout').fetchone()[0]
    assert got >= 10000, f'busy_timeout is {got} ms; expected an explicit >=10 s'


# ------------------------------------- init_db must survive the startup herd --
def test_init_db_retries_on_transient_lock(tmp_path, monkeypatch):
    """A locked cache at open time must be retried, not fatal.

    The continuation crash: ~36 processes x 5 caches open at once against caches
    carrying a multi-GB stale WAL left by the SIGKILLed job; the first opener
    holds an exclusive lock for WAL recovery > the busy timeout, and init_db --
    which had no retry -- raised `database is locked` straight through Params
    load, uncaught, killing the job in ~3 min. init_db must retry with backoff.
    """
    import sqlite3
    gu = _import_graph_utils()
    monkeypatch.setattr(gu, '_SQLITE_BUSY_SECONDS', 0.1, raising=True)

    real_connect = sqlite3.connect
    state = {'fails': 3}

    class LockingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a):
            if sql.strip().upper().startswith(('CREATE TABLE', 'SELECT 1 FROM SQLITE_MASTER')) \
                    and state['fails'] > 0:
                state['fails'] -= 1
                raise sqlite3.OperationalError('database is locked')
            return self._inner.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def flaky_connect(*a, **k):
        return LockingConn(real_connect(*a, **k))
    monkeypatch.setattr(sqlite3, 'connect', flaky_connect)

    conn = gu.init_db(db_path=str(tmp_path / 'g.sqlite'))   # must NOT raise
    assert state['fails'] == 0, 'should have retried through the transient locks'
    assert conn is not None


def test_init_db_skips_ddl_when_table_exists(tmp_path):
    """On a continuation the table already exists; opening it must not take the
    WAL writer lock. A read-only existence check replaces the unconditional
    CREATE TABLE, so 179/180 concurrent openers never contend for the writer."""
    import sqlite3
    gu = _import_graph_utils()
    p = str(tmp_path / 'g.sqlite')
    gu.init_db(db_path=p).close()          # first call creates the table

    real_connect = sqlite3.connect
    ddl = {'count': 0}

    class WatchConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a):
            if sql.strip().upper().startswith('CREATE TABLE'):
                ddl['count'] += 1
            return self._inner.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    import unittest.mock as _mock
    with _mock.patch.object(sqlite3, 'connect',
                            lambda *a, **k: WatchConn(real_connect(*a, **k))):
        gu.init_db(db_path=p).close()      # second call: table already present
    assert ddl['count'] == 0, 'CREATE TABLE ran though the table already existed'


def test_init_db_busy_timeout_is_at_least_30s(tmp_path):
    """Recovery of a multi-GB WAL can exceed 10 s; the per-attempt patience
    must be raised well above it."""
    gu = _import_graph_utils()
    conn = gu.init_db(db_path=str(tmp_path / 'g.sqlite'))
    got = conn.execute('PRAGMA busy_timeout').fetchone()[0]
    assert got >= 30000, f'busy_timeout is {got} ms; expected >= 30 s'


def _herd_opener(path, q):
    """Module-level so spawn can pickle it."""
    try:
        import aimmd.network.graph_utils as g
        g.init_db(db_path=path).close()
        q.put('ok')
    except Exception as exc:                                    # noqa: BLE001
        q.put(f'FAIL:{type(exc).__name__}')


def test_init_db_concurrent_openers_all_succeed(tmp_path):
    """A herd of concurrent openers on one existing cache must all succeed."""
    import multiprocessing as mp
    gu = _import_graph_utils()
    path = str(tmp_path / 'g.sqlite')
    gu.init_db(db_path=path).close()

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=_herd_opener, args=(path, q)) for _ in range(16)]
    for pr in ps:
        pr.start()
    for pr in ps:
        pr.join(timeout=120)
    res = [q.get() for _ in range(16)]
    assert res.count('ok') == 16, res


# ----------------------------- the trainer must not write to the shared cache --
def test_reader_role_keeps_graphs_local(tmp_path, monkeypatch):
    """In reader role a store populates memo/replica but never the shared DB.

    The trainer computes descriptors for the whole ensemble at the top of every
    round (worker/_train.py, `pathensembles[k].compute(*compute_descriptors_args)`)
    and the campaign's descriptors_function stores every resulting graph. That
    put a ~30 MB, 4096-row transaction in contention with ~35 MD writers for
    SQLite's single, unfair write lock, and the trainer lost -- 300 s per batch,
    observed in production as a stall indistinguishable from a hang.

    The trainer is a reader by role: it needs the graphs in memory for this
    round, not in the shared cache, and the writers cache them anyway when they
    reach those frames. Keeping them local removes the trainer from the write
    lock entirely.
    """
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    graphs = [_tiny_graph(gu) for _ in range(3)]
    keys = ['r0', 'r1', 'r2']

    mirrored = []
    monkeypatch.setattr(gu, '_after_store',
                        lambda c, k, b: mirrored.append(list(k)))
    wrote = []
    monkeypatch.setattr(gu, '_store_blobs',
                        lambda c, k, b: wrote.append(list(k)) or True)

    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite(keys, graphs, conn, compression_lib='lz4')

    assert wrote == [], 'the trainer must not touch the shared write lock'
    assert mirrored == [keys], 'but the graphs must still be memo/replica-local'
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 0


def test_writer_role_still_writes_to_the_shared_cache(tmp_path, monkeypatch):
    """MD writers are unchanged -- they are what populates the cache."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: False)
    gu.store_many_in_sqlite(['w0'], [_tiny_graph(gu)], conn, compression_lib='lz4')
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_reader_role_single_store_is_also_local(tmp_path, monkeypatch):
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_in_sqlite('r', _tiny_graph(gu), conn, compression_lib='lz4')
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 0


def test_blocked_write_is_reported_while_it_waits(monkeypatch, capsys):
    """A contended write must announce itself, not go silent for 300 s.

    In production the only sign of a stalled trainer was a single line after the
    full retry budget expired, so a 5-minute stall looked identical to a hang.
    """
    import sqlite3
    gu = _import_graph_utils()
    monkeypatch.setattr(gu, '_STORE_RETRY_SECONDS', 1.0, raising=True)
    monkeypatch.setattr(gu, '_STORE_REPORT_EVERY', 0.0, raising=True)

    class AlwaysLocked:
        def executemany(self, *a, **k):
            raise sqlite3.OperationalError('database is locked')

        def commit(self):
            pass

        def rollback(self):
            pass

    gu._store_blobs(AlwaysLocked(), ['k'], [b'blob'])
    out = capsys.readouterr().out
    assert 'blocked' in out, f'no progress report while blocked; got: {out!r}'
    assert 'still retrying' in out


# ------------------- the reader's backlog, and getting it into the cache --
def test_reader_role_buffers_for_a_later_flush(monkeypatch):
    """Keeping graphs local is not enough -- they must also be kept.

    The replica is re-staged from the real database at the top of every round,
    so a graph the trainer computed and kept only in its replica is gone by the
    next round and recomputed from scratch. In production that was ~25,000
    graphs per round for one system, indefinitely, and every recompute was a
    read against the real database that stopped its WAL from ever resetting.
    """
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)

    gu.store_many_in_sqlite(['r0', 'r1'], [_tiny_graph(gu) for _ in range(2)],
                            conn, compression_lib='lz4')

    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 0
    assert gu.shm_cache.pending_count(conn) == 2


def test_flush_pending_writes_persists_them_to_the_shared_cache(monkeypatch):
    """One batched write per round is what closes the hole for good."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    keys = ['r0', 'r1', 'r2']
    gu.store_many_in_sqlite(keys, [_tiny_graph(gu) for _ in keys], conn,
                            compression_lib='lz4')

    written = gu.flush_pending_writes(conn)

    assert sum(written.values()) == 3
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 3
    assert gu.shm_cache.pending_count(conn) == 0
    for key in keys:
        assert gu.load_from_sqlite(key, conn, compression_lib='lz4') is not None


def test_flush_pending_writes_uses_a_single_transaction(monkeypatch):
    """A round's backlog is one commit, not one per graph -- that was the bug."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite([f'k{i}' for i in range(20)],
                            [_tiny_graph(gu) for _ in range(20)], conn,
                            compression_lib='lz4')
    calls = []
    real = gu._store_blobs
    monkeypatch.setattr(gu, '_store_blobs',
                        lambda c, k, b: calls.append(len(k)) or real(c, k, b))

    gu.flush_pending_writes(conn)

    assert calls == [20], f'expected one batched write, got {calls}'


def test_writer_role_buffers_nothing(monkeypatch):
    """MD writers write immediately; the backlog is a trainer-only mechanism."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: False)
    gu.store_many_in_sqlite(['w0'], [_tiny_graph(gu)], conn, compression_lib='lz4')
    assert gu.shm_cache.pending_count(conn) == 0
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_flush_with_nothing_pending_is_a_no_op(monkeypatch):
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    calls = []
    monkeypatch.setattr(gu, '_store_blobs',
                        lambda c, k, b: calls.append(k) or True)
    assert gu.flush_pending_writes(conn) == {}
    assert calls == [], 'an empty backlog must not take the write lock at all'


def test_flush_never_raises_and_drops_the_batch_when_the_write_gives_up(monkeypatch):
    """A lock must never be fatal, and the backlog must never grow unbounded.

    Re-buffering a batch that just lost a 300 s fight would carry it into every
    later round and grow the trainer's memory without bound. Dropping it costs
    one recompute next round -- the cache is content-addressed, so that is free
    of correctness risk.
    """
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite(['r0'], [_tiny_graph(gu)], conn, compression_lib='lz4')
    monkeypatch.setattr(gu, '_store_blobs', lambda c, k, b: False)

    written = gu.flush_pending_writes(conn)

    assert sum(written.values()) == 0
    assert gu.shm_cache.pending_count(conn) == 0, 'must not re-buffer'


def test_flush_survives_an_unwritable_database(monkeypatch):
    """Anything unexpected in the flush is a cache miss, never a crash."""
    import sqlite3
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite(['r0'], [_tiny_graph(gu)], conn, compression_lib='lz4')

    def boom(*a, **k):
        raise sqlite3.OperationalError('attempt to write a readonly database')

    monkeypatch.setattr(gu, '_store_blobs', boom)
    assert gu.flush_pending_writes(conn) == {}
    assert gu.shm_cache.pending_count(conn) == 0


def test_store_flushes_early_when_over_the_byte_cap(monkeypatch):
    """A long round must not accumulate the whole ensemble in memory."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    monkeypatch.setattr(gu.shm_cache, '_PENDING_WRITE_BYTES', 1)

    gu.store_many_in_sqlite(['r0'], [_tiny_graph(gu)], conn, compression_lib='lz4')

    assert gu.shm_cache.pending_count(conn) == 0, 'over the cap: flush at once'
    assert conn.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_flush_without_a_connection_covers_every_registered_cache(monkeypatch):
    """The multi-system trainer holds one connection per system."""
    gu = _import_graph_utils()
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    conns = []
    for i in range(3):
        c = _mem_db(gu)
        c._aimmd_db_path = f'/nowhere/cache{i}.sqlite'
        gu.shm_cache.register(c)
        gu.store_many_in_sqlite([f'k{i}'], [_tiny_graph(gu)], c,
                                compression_lib='lz4')
        conns.append(c)

    written = gu.flush_pending_writes()

    assert sum(written.values()) == 3
    assert len(written) == 3, f'one entry per cache, got {written}'
    for c in conns:
        assert c.execute('SELECT count(*) FROM graphs_cache').fetchone()[0] == 1


def test_flush_reports_what_it_wrote(monkeypatch, capsys):
    """The trainer log must attribute the backlog to a system."""
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    conn._aimmd_db_path = '/nowhere/graphs_cache_G4.sqlite'
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite(['r0', 'r1'], [_tiny_graph(gu) for _ in range(2)],
                            conn, compression_lib='lz4')
    gu.flush_pending_writes(conn, verbose=True)
    out = capsys.readouterr().out
    assert 'graphs_cache_G4.sqlite' in out
    assert '2' in out


def test_flush_rolls_back_a_failed_write(monkeypatch):
    """A failed flush must not leave the connection inside a transaction.

    `_store_blobs` rolls back on its lock branch but re-raises every other error
    with the transaction still open, and pysqlite has already issued BEGIN and
    the INSERT by then. A connection left mid-transaction pins a read snapshot
    for the life of the trainer: it stops seeing rows the MD writers add,
    `refresh_replicas` reads an unchanging MAX(rowid) and stops topping up, and
    the pinned read-mark makes `wal_checkpoint` return busy for *every* process
    on that database -- reproducing the exact failure this whole change exists
    to end. Reachable with a read-only or corrupt cache file, neither of which
    heals on its own.
    """
    import sqlite3
    gu = _import_graph_utils()
    conn = _mem_db(gu)
    monkeypatch.setattr(gu.shm_cache, 'reader_role', lambda: True)
    gu.store_many_in_sqlite(['r0'], [_tiny_graph(gu)], conn, compression_lib='lz4')

    def enters_a_transaction_then_fails(cache, keys, blobs):
        # exactly what the real _store_blobs does before its bare `raise`
        cache.execute('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                      ('wedge', b'x'))
        assert cache.in_transaction, 'precondition: the write opened a txn'
        raise sqlite3.OperationalError('attempt to write a readonly database')

    monkeypatch.setattr(gu, '_store_blobs', enters_a_transaction_then_fails)

    assert gu.flush_pending_writes(conn) == {}
    assert conn.in_transaction is False, (
        'the connection is wedged in a transaction: it will pin a read snapshot '
        'and block WAL resets for every process on this database')
    # and the connection is still usable
    conn.execute('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)', ('ok', b'y'))
    conn.commit()
    assert conn.execute(
        "SELECT count(*) FROM graphs_cache WHERE key='ok'").fetchone()[0] == 1
