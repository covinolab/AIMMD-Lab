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
