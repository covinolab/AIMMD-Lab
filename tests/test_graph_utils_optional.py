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
