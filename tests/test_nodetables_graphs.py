"""`NodeTableFeaturizer`: node-table rows to graphs and batch dicts.

The graphs rebuilt from node-table rows must be bitwise the graphs
`get_graphs_pyg` builds from the coordinates of the same frames (positions,
edge order, one-hot node attributes, zero shifts), `batch_dict` must equal
``Batch.from_data_list(...).to_dict()``, `row_from_graph` must invert the
featurization, the edges must be built on the CPU whatever the device, and
empty rows must never reach a network. Toy system with a ligand split across
the box (tests/_nodetables_toy.py). Needs the graph dependencies
(``--rungraph``).
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

pytestmark = pytest.mark.graph

torch = pytest.importorskip("torch")
graph_utils = pytest.importorskip(
    "aimmd.network.graph_utils",
    reason="node-table graph tests require the optional graph dependencies")

from torch_geometric.data import Batch  # noqa: E402

from aimmd.network import nodetables  # noqa: E402
from aimmd.network.nodetables import (NodeTableFeaturizer,  # noqa: E402
                                      NodeTableOverflowError)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF,  # noqa: E402
                                   ENVIRONMENT_SELECTION, SYSTEM_SELECTION,
                                   toy_frames, toy_universe)

FRAMES = np.concatenate([toy_frames(10, seed=1),
                         toy_frames(6, seed=2, broken=False)])
KEYS = ('positions', 'edge_index', 'node_attrs', 'shifts')


def _featurizer(**kwargs):
    settings = dict(atom_types=ATOM_TYPES, cutoff=CUTOFF)
    settings.update(kwargs)
    return NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                               ENVIRONMENT_SELECTION, **settings)


def _reference_graphs(frames=FRAMES, atom_types=ATOM_TYPES, cutoff=CUTOFF):
    return graph_utils.get_graphs_pyg(
        frames.reshape(len(frames), -1), toy_universe(), SYSTEM_SELECTION,
        ENVIRONMENT_SELECTION, cutoff, atom_types=atom_types)


def _assert_same_tensor(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a, b)


def _assert_same_graph(graph, reference):
    assert sorted(graph.keys()) == sorted(KEYS)
    for key in KEYS:
        _assert_same_tensor(graph[key], reference[key])


@pytest.mark.parametrize('atom_types', [ATOM_TYPES, None])
def test_graphs_equal_get_graphs_pyg_bitwise(atom_types):
    featurizer = _featurizer(atom_types=atom_types)
    rows = featurizer.rows_from_coordinates(FRAMES)
    graphs = featurizer.graphs(rows)
    reference = _reference_graphs(atom_types=atom_types)

    assert isinstance(graphs, graph_utils.GraphList)
    assert len(graphs) == len(reference) == len(FRAMES)
    for graph, expected in zip(graphs, reference):
        _assert_same_graph(graph, expected)
    assert sum(graph['edge_index'].shape[1] for graph in graphs) > 0


def test_graphs_follow_the_cutoff_at_transform_time():
    rows = _featurizer().rows_from_coordinates(FRAMES[:4])
    graphs = _featurizer(cutoff=4.5).graphs(rows)
    for graph, expected in zip(graphs, _reference_graphs(FRAMES[:4],
                                                         cutoff=4.5)):
        _assert_same_graph(graph, expected)


def test_batch_dict_equals_batching_the_graphs():
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES)
    for block in (slice(0, 1), slice(0, 7), slice(3, 16)):
        batch = featurizer.batch_dict(rows[block])
        expected = Batch.from_data_list(_reference_graphs()[block]).to_dict()
        assert set(batch) == set(expected)
        for key in expected:
            _assert_same_tensor(batch[key], expected[key])


def test_rows_from_a_series_file_give_the_same_graphs(tmp_path):
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES)
    np.save(tmp_path / 'rows.npy', rows)
    stored = np.load(tmp_path / 'rows.npy', mmap_mode='r')
    for graph, expected in zip(featurizer.graphs(stored), _reference_graphs()):
        _assert_same_graph(graph, expected)
    batch = featurizer.batch_dict(stored[2:9])
    expected = Batch.from_data_list(_reference_graphs()[2:9]).to_dict()
    for key in expected:
        _assert_same_tensor(batch[key], expected[key])


def test_batch_dict_builds_the_edges_on_the_cpu(monkeypatch):
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES[:5])
    devices = []
    radius_graph = nodetables._featurizer._radius_graph

    def spy(x, r, batch=None, **kwargs):
        devices.append((x.device.type,
                        None if batch is None else batch.device.type))
        return radius_graph(x, r, batch=batch, **kwargs)

    monkeypatch.setattr(nodetables._featurizer, '_radius_graph', spy)
    batch = featurizer.batch_dict(rows, device='meta')

    assert devices == [('cpu', 'cpu')]
    assert all(value.device.type == 'meta' for value in batch.values())
    reference = featurizer.batch_dict(rows)
    assert all(batch[key].shape == reference[key].shape for key in reference)


def test_row_from_graph_inverts_the_featurization():
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES)
    for row, graph in zip(rows, _reference_graphs()):
        inverted = featurizer.row_from_graph(graph)
        assert inverted.dtype == np.float32
        assert np.array_equal(inverted, row)
        _assert_same_graph(featurizer.graphs([inverted])[0], graph)
    # a plain mapping of tensors works too
    graph = _reference_graphs(FRAMES[:1])[0]
    mapping = {'positions': graph['positions'],
               'node_attrs': graph['node_attrs']}
    assert np.array_equal(featurizer.row_from_graph(mapping), rows[0])


def test_row_from_graph_refuses_other_graphs():
    featurizer = _featurizer()
    graph = _reference_graphs(FRAMES[:1])[0]

    with pytest.raises(NodeTableOverflowError):
        _featurizer(n_max=6).row_from_graph(graph)
    other_types = graph.clone()
    other_types['node_attrs'] = torch.cat(
        [graph['node_attrs'], torch.zeros(len(graph['node_attrs']), 1)], 1)
    with pytest.raises(ValueError, match='one-hot'):
        featurizer.row_from_graph(other_types)
    shifted = graph.clone()
    shifted['shifts'] = shifted['shifts'] + 1.0
    with pytest.raises(ValueError, match='shifts'):
        featurizer.row_from_graph(shifted)
    double = graph.clone()
    double['positions'] = double['positions'].double()
    with pytest.raises(ValueError, match='float32'):
        featurizer.row_from_graph(double)


@pytest.mark.parametrize('build', ['graphs', 'batch_dict'])
def test_an_empty_row_never_becomes_a_graph(build):
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES[:4])
    rows[2] = 0.0                                   # never computed
    with pytest.raises(ValueError) as info:
        getattr(featurizer, build)(rows)
    message = str(info.value)
    assert 'n_nodes == 0' in message and '[2]' in message
    assert 'python -m aimmd.network.nodetables repack --n-max N' in message


def test_an_overflowing_frame_stops_training_with_the_repack_command(capsys):
    featurizer = _featurizer()
    counts = featurizer.rows_from_coordinates(FRAMES)[:, 0]
    small = _featurizer(n_max=int(np.median(counts)))
    rows = small.rows_from_coordinates(FRAMES)      # never raises
    assert 'ERROR' in capsys.readouterr().out
    keep = rows[:, 0] > 0
    assert keep.any() and not keep.all()
    small.batch_dict(rows[keep])                    # the fitting frames work
    with pytest.raises(ValueError, match='repack --n-max'):
        small.batch_dict(rows)


@pytest.mark.parametrize('corrupt, match', [
    (lambda rows: rows[:, :-1], 'shape'),
    (lambda rows: rows[0], 'shape'),
    (lambda rows: np.concatenate([rows[:, :1], rows[:, 1:2] + 1, rows[:, 2:]],
                                 axis=1), 'layout'),
    (lambda rows: np.concatenate([rows[:, :1] + 0.5, rows[:, 1:]], axis=1),
     'layout'),
    (lambda rows: np.where(np.arange(rows.shape[1]) == 2 + 3 * 768,
                           len(ATOM_TYPES), rows), 'atom types')])
def test_rows_of_another_layout_are_refused(corrupt, match):
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES[:3])
    with pytest.raises(ValueError, match=match):
        featurizer.graphs(corrupt(rows))


def test_batch_dict_needs_a_row():
    featurizer = _featurizer()
    with pytest.raises(ValueError, match='at least one row'):
        featurizer.batch_dict(np.zeros((0, featurizer.width), np.float32))


def test_graph_utils_reexports_the_featurizer():
    assert graph_utils.NodeTableFeaturizer is NodeTableFeaturizer
