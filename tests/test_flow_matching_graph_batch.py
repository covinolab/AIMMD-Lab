"""Radius graphs of ``aimmd.cgaimmd.flow_matching.graph_batch``.

The torch backend runs in the default lane; the PyG backend and the equality
with ``aimmd.network.graph_utils.get_graphs_pyg`` need ``--rungraph``. Ported
from the flow branch's ``tests/test_graph_batch.py`` (``bb98943``).
"""

import os
import warnings

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.flow_matching import graph_batch


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


def _edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


def _reference_edges(pos, batch, cutoff):
    """Brute force: every ordered pair (source, target) of one sample within the cutoff."""
    pos = pos.double()
    out = set()
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and batch[i] == batch[j] and float((pos[i] - pos[j]).norm()) < cutoff:
                out.add((j, i))
    return out


def test_torch_backend_edges():
    """Within-sample pairs inside the cutoff, both directions, no self loops, no cross-sample pairs."""
    g = torch.Generator().manual_seed(0)
    pos = 3.0 * torch.randn(3 * 7, 3, generator=g)
    batch = torch.arange(3).repeat_interleave(7)
    edges = graph_batch.radius_edges(pos, batch, 4.0, backend='torch')
    assert edges.dtype == torch.long and edges.shape[0] == 2
    assert _edge_set(edges) == _reference_edges(pos, batch, 4.0)
    assert not bool((edges[0] == edges[1]).any())
    assert bool((batch[edges[0]] == batch[edges[1]]).all())
    # ragged batches (different atom counts, unsorted sample ids) take the general path
    ragged = torch.tensor([0] * 5 + [1] * 9 + [2] * 7)[torch.randperm(21, generator=g)]
    edges = graph_batch.radius_edges(pos, ragged, 4.0, backend='torch')
    assert _edge_set(edges) == _reference_edges(pos, ragged, 4.0)
    # no gradient flows through the edge construction; empty input is fine
    pos.requires_grad_(True)
    assert not graph_batch.radius_edges(pos, batch, 4.0, backend='torch').requires_grad
    empty = graph_batch.radius_edges(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 4.0,
                                     backend='torch')
    assert empty.shape == (2, 0)
    with pytest.raises(ValueError):
        graph_batch.radius_edges(pos, batch, 4.0, backend='cuda')


def test_build_graph_batch_matches_batch_layout():
    """(B, M, 3) positions -> the dict Batch.from_data_list(...).to_dict() gives."""
    g = torch.Generator().manual_seed(1)
    x = 2.0 * torch.randn(4, 5, 3, generator=g)
    attrs = torch.eye(5)[:, :3]
    d = graph_batch.build_graph_batch(x, 3.0, node_attrs=attrs, backend='torch')
    assert set(d) == {'positions', 'edge_index', 'batch', 'ptr', 'node_attrs', 'shifts'}
    assert torch.equal(d['positions'], x.reshape(20, 3))
    assert torch.equal(d['batch'], torch.arange(4).repeat_interleave(5))
    assert torch.equal(d['ptr'], torch.arange(0, 21, 5))
    assert torch.equal(d['node_attrs'], attrs.repeat(4, 1))
    assert d['shifts'].shape == (d['edge_index'].shape[1], 3) and not bool(d['shifts'].any())
    assert _edge_set(d['edge_index']) == _reference_edges(x.reshape(20, 3), d['batch'], 3.0)
    none = graph_batch.build_graph_batch(x, 3.0, backend='torch')
    assert none['node_attrs'].shape == (20, 0)


def test_max_neighbours_warning():
    """A node that reaches the neighbour cap warns once; the torch backend keeps the nearest."""
    from aimmd.cgaimmd import _once
    pos = torch.zeros(1, 12, 3)
    pos[0, :, 0] = torch.arange(12, dtype=torch.float32) * 0.1        # everything within 2 A
    batch = torch.zeros(12, dtype=torch.long)
    with pytest.warns(RuntimeWarning, match='max_num_neighbors'):
        edges = graph_batch.radius_edges(pos[0], batch, 2.0, max_num_neighbors=4,
                                         backend='torch')
    incoming = torch.bincount(edges[1], minlength=12)
    assert int(incoming.max()) == 4
    # atom 0 keeps its 4 nearest neighbours
    assert set(edges[0][edges[1] == 0].tolist()) == {1, 2, 3, 4}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        graph_batch.radius_edges(pos[0], batch, 2.0, max_num_neighbors=4, backend='torch')
    _once._reset()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        graph_batch.radius_edges(pos[0], batch, 2.0, max_num_neighbors=128, backend='torch')


@pytest.fixture
def fresh_pyg_check():
    """The once-per-process check of PyG's radius_graph and the warn-once flags, cleared."""
    from aimmd.cgaimmd import _once
    check = graph_batch.pyg_radius_graph_problem              # the cached function (tests replace it)
    check.cache_clear()
    _once._reset()
    yield
    check.cache_clear()
    _once._reset()


def test_auto_backend_follows_the_pyg_check(monkeypatch, fresh_pyg_check):
    """'auto' is the PyG backend exactly when PyG's radius_graph runs; 'pyg' and 'torch' are kept."""
    def boom(*args, **kwargs):
        raise AssertionError('the pyg backend was used')

    monkeypatch.setattr(graph_batch, 'pyg_radius_graph_problem',
                        lambda: 'torch_geometric is not installed')
    monkeypatch.setattr(graph_batch, '_pyg_edges', boom)
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *a, **k: None if name == 'torch_geometric' else real(name, *a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter('error')                      # not installed: no warning
        assert graph_batch._resolve_backend('auto') == 'torch'
        graph_batch.radius_edges(torch.randn(6, 3), torch.zeros(6, dtype=torch.long), 3.0)
    monkeypatch.setattr(graph_batch, 'pyg_radius_graph_problem', lambda: None)
    assert graph_batch._resolve_backend('auto') == 'pyg'
    assert graph_batch._resolve_backend('torch') == 'torch'
    assert graph_batch._resolve_backend('pyg') == 'pyg'


def _fake_pyg_without_pyg_lib(monkeypatch):
    """torch_geometric as 2.8 behaves without pyg-lib: its radius_graph raises ImportError."""
    import importlib.machinery
    import importlib.util
    import sys
    import types

    def radius_graph(*args, **kwargs):
        raise ImportError("'radius_graph' requires 'pyg-lib>=0.6.0'")

    package = types.ModuleType('torch_geometric')
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec('torch_geometric', None, is_package=True)
    nn = types.ModuleType('torch_geometric.nn')
    nn.radius_graph = radius_graph
    package.nn = nn
    monkeypatch.setitem(sys.modules, 'torch_geometric', package)
    monkeypatch.setitem(sys.modules, 'torch_geometric.nn', nn)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *a, **k: package.__spec__ if name == 'torch_geometric'
                        else real(name, *a, **k))


def test_auto_backend_falls_back_when_pyg_cannot_build_radius_graphs(monkeypatch,
                                                                     fresh_pyg_check):
    """torch-geometric 2.8 without pyg-lib: 'auto' warns once and builds the torch edges.

    Before, 'auto' chose PyG whenever torch_cluster was importable, and every
    graph then failed with "'radius_graph' requires 'pyg-lib>=0.6.0'". The
    explicit 'pyg' backend still fails, with what to install.
    """
    _fake_pyg_without_pyg_lib(monkeypatch)
    problem = graph_batch.pyg_radius_graph_problem()
    assert problem is not None and 'pyg-lib' in problem
    g = torch.Generator().manual_seed(3)
    pos = 3.0 * torch.randn(2 * 6, 3, generator=g)
    batch = torch.arange(2).repeat_interleave(6)
    with pytest.warns(RuntimeWarning, match='pyg-lib') as record:
        edges = graph_batch.radius_edges(pos, batch, 4.0)
    assert sum('torch backend builds the edges' in str(w.message) for w in record) == 1
    assert _edge_set(edges) == _reference_edges(pos, batch, 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')                      # once per process
        graph_batch.build_graph_batch(pos.reshape(2, 6, 3), 4.0)
    with pytest.raises(ImportError, match=r"edge backend 'pyg'.*torch-geometric<2.8"):
        graph_batch.radius_edges(pos, batch, 4.0, backend='pyg')


@pytest.mark.graph
def test_pyg_check_passes_with_the_graphs_extra(fresh_pyg_check):
    """With torch-geometric < 2.8 and torch-cluster (the graphs extra), PyG's radius_graph runs."""
    pytest.importorskip('torch_cluster')
    assert graph_batch.pyg_radius_graph_problem() is None
    assert graph_batch._resolve_backend('auto') == 'pyg'


@pytest.mark.graph
def test_pyg_equals_torch():
    """Both backends give the same edge set below the neighbour cap."""
    pytest.importorskip('torch_cluster')
    g = torch.Generator().manual_seed(2)
    pos = 3.0 * torch.randn(5 * 11, 3, generator=g)
    batch = torch.arange(5).repeat_interleave(11)
    a = graph_batch.radius_edges(pos, batch, 3.5, backend='pyg')
    b = graph_batch.radius_edges(pos, batch, 3.5, backend='torch')
    assert _edge_set(a) == _edge_set(b) == _reference_edges(pos, batch, 3.5)


def _universe(n_atoms, box):
    import MDAnalysis as mda
    universe = mda.Universe.empty(n_atoms, n_residues=1, atom_resindex=[0] * n_atoms,
                                  trajectory=True)
    universe.add_TopologyAttr('types', ['C'] * n_atoms)
    universe.add_TopologyAttr('names', [f'C{i}' for i in range(n_atoms)])
    universe.add_TopologyAttr('bonds', [(i, i + 1) for i in range(n_atoms - 1)])
    universe.trajectory.ts.dimensions = np.array([box, box, box, 90, 90, 90], dtype=np.float32)
    return universe


@pytest.mark.graph
def test_radius_edges_equals_get_graphs_pyg():
    """The pyg backend reproduces graph_utils.get_graphs_pyg's edges on whole frames.

    Includes a cluster frame that straddles a face of its original 20 A box:
    graph_utils centers it in the (large) box of the universe, which keeps it
    whole, and the edges of the translated graph equal those of the raw frame.
    """
    graph_utils = pytest.importorskip('aimmd.network.graph_utils')
    rng = np.random.default_rng(3)
    n = 9
    chain = np.cumsum(rng.normal(0.0, 0.9, size=(n, 3)) + [1.2, 0.0, 0.0], axis=0)
    frames = [chain + 5.0, chain + [19.0, 3.0, 3.0]]       # the second one crosses x = 20 A
    descriptors = np.array([f.reshape(-1) for f in frames], dtype=np.float32)
    universe = _universe(n, 1000.0)
    graphs = graph_utils.get_graphs_pyg(descriptors, universe, 'all', 'not all', 4.0)
    for frame, data in zip(frames, graphs):
        raw = torch.as_tensor(frame, dtype=torch.float32)
        zeros = torch.zeros(n, dtype=torch.long)
        pyg = graph_batch.radius_edges(raw, zeros, 4.0, backend='pyg')
        torch_edges = graph_batch.radius_edges(raw, zeros, 4.0, backend='torch')
        assert _edge_set(data.edge_index) == _edge_set(pyg) == _edge_set(torch_edges)
        # the cached positions are the frame translated as a whole
        shift = data.positions.double() - raw.double()
        assert torch.allclose(shift, shift[:1].expand_as(shift), atol=1e-3)
