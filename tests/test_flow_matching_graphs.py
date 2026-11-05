"""Graph mode of flow-matching networks (``aimmd.cgaimmd.flow_matching.graphs``).

``FlowGraphTransform`` turns the flat Å coordinates of the generator atoms into
graphs. By default (PI, 5 Nov) it returns one batch dict built on the fly and
writes nothing: that route needs only torch and runs in the default lane.
Reading an existing SQLite graph cache of AIMMD's graph pipeline, or writing
one, is an explicit switch (``graph_cache``, ``write_graph_cache``); those
routes need the graph stack (``--rungraph``). ``bond_edge_features`` is plain
torch. Ported from the flow branch's ``tests/test_flow_graphs.py``
(``bb98943``), with the cache tests moved to the explicit switch, plus the
tests of the default route.
"""

import hashlib
import os
import sqlite3
import sys

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.flow_matching import graphs as graphs_module
from aimmd.cgaimmd.flow_matching.graph_batch import build_graph_batch, radius_edges
from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform, bond_edge_features
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=3.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=True,
            edge_backend='torch')


def _edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


def test_bond_edge_features():
    """One-hot of the topological distance class, looked up by molecule-local index."""
    # a 4-atom chain 0-1-2-3: distances 1, 2, 3 bonds; class = clip(d, 1, 4) - 1
    distance = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    bond_class = (distance.clamp(1, 4) - 1).to(torch.uint8)
    # two molecules of 4 atoms, edges given in global indices
    edge_index = torch.tensor([[0, 3, 5, 4, 7],
                               [1, 0, 7, 7, 4]])
    batch = torch.arange(2).repeat_interleave(4)
    ptr = torch.tensor([0, 4, 8])
    features = bond_edge_features(edge_index, batch, ptr, bond_class)
    assert features.shape == (5, 4) and features.dtype == torch.float32
    assert features.sum(dim=1).tolist() == [1.0] * 5
    assert features.argmax(dim=1).tolist() == [0, 2, 1, 2, 2]
    # disconnected pairs carry the largest class (> 3 bonds)
    far = torch.full((4, 4), 3, dtype=torch.uint8)
    assert bond_edge_features(edge_index, batch, ptr, far).argmax(dim=1).tolist() == [3] * 5


def _write_structure(path, positions, box=20.0, types=None):
    import MDAnalysis as mda
    n = len(positions)
    types = types or ['C'] * n
    universe = mda.Universe.empty(n, n_residues=1, atom_resindex=[0] * n, trajectory=True)
    universe.add_TopologyAttr('names', [f'{t}{i + 1}' for i, t in enumerate(types)])
    universe.add_TopologyAttr('types', types)
    universe.add_TopologyAttr('elements', types)
    universe.add_TopologyAttr('resnames', ['MOL'])
    universe.add_TopologyAttr('resids', [1])
    universe.atoms.positions = positions
    universe.dimensions = np.array([box, box, box, 90, 90, 90], dtype=np.float32)
    universe.atoms.write(str(path))
    return str(path)


def _chain(n, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.25, size=(n, 3)) + [1.5, 0.0, 0.0]
    return np.cumsum(steps, axis=0).astype(np.float32)


def _rows(seed=0):
    """Three frames of a 6-atom chain as stored descriptor rows."""
    chain = _chain(6, seed)
    frames = np.stack([chain + 3.0, chain + [1.0, 2.0, 3.0], 1.1 * chain])
    return chain, frames.reshape(3, -1).astype(np.float32)


def _files(folder):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file())


def _md5(fname):
    return hashlib.md5(open(fname, 'rb').read()).hexdigest()


# ---------------------------------------------------------------------------
# the default route: one batch dict, no cache, no file
# ---------------------------------------------------------------------------

def test_default_route_returns_one_batch_dict(tmp_path):
    """The rows as one batch dict in the layout of Batch.from_data_list(...).to_dict()."""
    chain, rows = _rows()
    types = ['C', 'C', 'O', 'C', 'N', 'C']
    transform = FlowGraphTransform(_write_structure(tmp_path / 'atoms.pdb', chain + 5.0,
                                                    types=types), 3.0, edge_backend='torch')
    assert transform.n_atoms == 6 and transform.graph_cache is None
    assert 'graph_cache=none' in repr(transform)
    # node attributes: one-hot of the atom types, in graph_utils' columns (sorted types)
    columns = sorted(set(types))
    expected = torch.zeros(6, len(columns))
    expected[torch.arange(6), [columns.index(t) for t in types]] = 1.0
    assert torch.equal(transform.node_attrs, expected)
    out = transform(rows)
    assert isinstance(out, dict)
    assert set(out) == {'positions', 'edge_index', 'batch', 'ptr', 'node_attrs', 'shifts'}
    positions = torch.as_tensor(rows).reshape(3, 6, 3)
    assert out['positions'].dtype == torch.float32
    assert torch.equal(out['positions'], positions.reshape(18, 3))          # not moved
    reference = build_graph_batch(positions, 3.0, node_attrs=expected, backend='torch')
    for key, value in reference.items():
        assert torch.equal(out[key], value), key
    assert torch.equal(out['ptr'], torch.tensor([0, 6, 12, 18]))
    assert _edge_set(out['edge_index']) == _edge_set(radius_edges(
        positions.reshape(18, 3), torch.arange(3).repeat_interleave(6), 3.0, backend='torch'))
    # float64 rows give float32 positions; no rows give an empty batch
    assert torch.equal(transform(rows.astype(np.float64))['positions'], out['positions'])
    empty = transform(rows[:0])
    assert empty['positions'].shape == (0, 3) and empty['edge_index'].shape == (2, 0)
    assert empty['ptr'].tolist() == [0] and empty['node_attrs'].shape == (0, len(columns))
    with pytest.raises(ValueError, match='width'):
        transform(rows[:, :9])


def test_graph_mode_by_default_writes_no_sqlite_file(tmp_path, monkeypatch):
    """Graph mode without a switch writes no file and touches neither SQLite nor graph_utils."""
    chain, rows = _rows()
    structure = _write_structure(tmp_path / 'atoms.pdb', chain + 5.0)
    run = tmp_path / 'run'
    run.mkdir()
    monkeypatch.chdir(run)
    before = _files(tmp_path)

    def no_sqlite(*args, **kwargs):
        raise AssertionError('SQLite was used')

    monkeypatch.setattr(sqlite3, 'connect', no_sqlite)
    import aimmd.network
    monkeypatch.delattr(aimmd.network, 'graph_utils', raising=False)
    monkeypatch.setitem(sys.modules, 'aimmd.network.graph_utils', None)       # import fails
    transform = FlowGraphTransform(structure, 3.0, edge_backend='torch')
    torch.manual_seed(0)
    net = JointPaiNN(6, [6] * 6, **TINY).eval()
    from aimmd.cgaimmd.values import ProtocolValuesFunction
    values = ProtocolValuesFunction(net, descriptor_transform=transform)
    for _ in range(3):
        transform(rows)
        values(rows)
    assert _files(tmp_path) == before
    assert not list(tmp_path.rglob('*.sqlite*'))


def test_graph_mode_with_a_dict_returning_transform(tmp_path):
    """The network, the values function and the trainer's batch helper take the dict as it is."""
    chain, rows = _rows(1)
    structure = _write_structure(tmp_path / 'atoms.pdb', chain + 5.0)
    transform = FlowGraphTransform(structure, TINY['cutoff'], edge_backend='torch')
    x = torch.as_tensor(rows).reshape(3, 6, 3)
    # bond edges and a topology edge between the chain ends (beyond the cutoff)
    from aimmd.cgaimmd.preparation.topology import ReferenceGeometry
    distance = np.minimum(np.abs(np.subtract.outer(np.arange(6), np.arange(6))), 4)
    geometry = ReferenceGeometry(n_atoms=6, bonds=np.array([[i, i + 1] for i in range(5)]),
                                 bond_lengths=np.full(5, 1.5),
                                 pairs=np.array([[0, 1], [0, 2], [0, 3]]),
                                 separation=np.array([1, 2, 3]),
                                 distance_min=np.array([1.5, 2.6, 3.0]),
                                 distance_max=np.array([1.5, 2.6, 4.5]))
    for options in ({}, {'bond_distances': distance},
                    {'bond_distances': distance, 'topology_geometry': geometry,
                     'topology_cutoff': 10.0}):
        torch.manual_seed(0)
        net = JointPaiNN(6, [6] * 6, **TINY, **options).eval()
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for p in net.parameters():
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
            dense = net.committor(x)
            assert float(dense.abs().max()) > 1e-3
            assert torch.allclose(net(transform(rows))[:, 0], dense, atol=1e-5), options
        from aimmd.cgaimmd.values import ProtocolValuesFunction
        graph_values = ProtocolValuesFunction(net, descriptor_transform=transform)(rows)
        dense_values = ProtocolValuesFunction(net)(rows)
        assert np.allclose(graph_values, dense_values, atol=1e-5)
        # AIMMD's own fit takes a batch dict from descriptor_transform as it is
        from aimmd.network.fit import _graph_batch
        with torch.no_grad():
            batch = _graph_batch(transform(rows), torch.device('cpu'))
            assert torch.allclose(net(batch)[:, 0], dense, atol=1e-5)


def test_switch_validation(tmp_path):
    chain, _ = _rows()
    structure = _write_structure(tmp_path / 'atoms.pdb', chain)
    with pytest.raises(ValueError, match='write_graph_cache=True needs graph_cache'):
        FlowGraphTransform(structure, 3.0, write_graph_cache=True)
    with pytest.raises(FileNotFoundError, match='only read unless write_graph_cache=True'):
        FlowGraphTransform(structure, 3.0, graph_cache=tmp_path / 'missing.sqlite')
    assert not (tmp_path / 'missing.sqlite').exists()
    with pytest.raises(ValueError, match='128'):
        FlowGraphTransform(structure, 3.0, graph_cache=tmp_path / 'g.sqlite',
                           write_graph_cache=True, max_num_neighbors=64)
    FlowGraphTransform(structure, 3.0, max_num_neighbors=64)                # no cache: any cap
    for bad in ({'cutoff': -1.0}, {'max_num_neighbors': 0}, {'edge_backend': 'cuda'},
                {'box_length': 0.0}, {'write_graph_cache': 'yes'}):
        kwargs = {'cutoff': 3.0, **bad}
        with pytest.raises(ValueError):
            FlowGraphTransform(structure, **kwargs)
    assert not (tmp_path / 'g.sqlite').exists()


# ---------------------------------------------------------------------------
# the SQLite graph cache: an explicit switch
# ---------------------------------------------------------------------------

@pytest.mark.graph
def test_cache_route_equals_default_route(tmp_path):
    """With write_graph_cache the lab's cached pipeline: graphs of the default route's edges, one row per frame."""
    pytest.importorskip('aimmd.network.graph_utils')
    chain, rows = _rows()
    structure = _write_structure(tmp_path / 'atoms.pdb', chain + 5.0)
    plain = FlowGraphTransform(structure, 3.0)
    batch = plain(rows)
    path = tmp_path / 'graphs_M6_rc3.0.sqlite'
    cached = FlowGraphTransform(structure, 3.0, graph_cache=path, write_graph_cache=True)
    assert not path.exists()                                # opened at the first call
    first = cached(rows)
    assert path.exists() and cached(rows[:0]) == []
    count = lambda: cached._conn.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0]  # noqa: E731
    n_rows = count()
    again = cached(rows)
    assert count() == n_rows == 3
    cached.close()
    for k, (a, b) in enumerate(zip(first, again)):
        assert torch.allclose(a.positions, b.positions)
        assert a.positions.shape == (6, 3)                  # generator atoms, network order
        mine = batch['edge_index'][:, batch['batch'][batch['edge_index'][0]] == k] - 6 * k
        assert _edge_set(a.edge_index) == _edge_set(b.edge_index) == _edge_set(mine)
        # the cached positions are the frame translated as a whole (into the box)
        shift = a.positions.double() - torch.as_tensor(rows[k]).reshape(6, 3).double()
        assert torch.allclose(shift, shift[:1].expand_as(shift), atol=1e-4)
        assert torch.equal(a['node_attrs'], plain.node_attrs)
    with pytest.raises(ValueError, match='width'):
        cached(rows[:, :9])


@pytest.mark.graph
def test_an_old_sqlite_cache_is_read_when_switched_on(tmp_path):
    """A cache written by the flow branch's transform is read; misses are built and never stored.

    The flow branch's ``FlowGraphTransform(..., conn=init_db(path))`` was
    ``process_descriptors_pyg`` on that connection; the cache below is written
    that way. One stored graph is then replaced by the graph of another frame,
    so that a graph that comes back from the cache can be told from one built
    again.
    """
    graph_utils = pytest.importorskip('aimmd.network.graph_utils')
    chain, rows = _rows()
    structure = _write_structure(tmp_path / 'atoms.pdb', chain + 5.0)
    path = tmp_path / 'graphs_M6_rc3.0.sqlite'
    reference = FlowGraphTransform(structure, 3.0)           # only for its box universe
    conn = graph_utils.init_db(str(path))
    old = list(graph_utils.process_descriptors_pyg(rows[:2], reference.universe, 'all', 'not all',
                                                   3.0, conn, compression_lib='lz4'))
    marker = graph_utils.get_graphs_pyg(rows[2:3], reference.universe, 'all', 'not all', 3.0)[0]
    conn.execute('INSERT OR REPLACE INTO graphs_cache (key, data) VALUES (?, ?)',
                 (graph_utils.get_stable_hash(rows[0]), graph_utils._encode(marker, 'lz4')))
    conn.commit()
    conn.close()
    md5 = _md5(path)

    def n_cached():
        reader = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            return reader.execute('SELECT COUNT(*) FROM graphs_cache').fetchone()[0]
        finally:
            reader.close()

    assert n_cached() == 2
    # the default route never opens it
    FlowGraphTransform(structure, 3.0)(rows)
    assert _md5(path) == md5
    # switched on: read only
    transform = FlowGraphTransform(structure, 3.0, graph_cache=path)
    assert transform.write_graph_cache is False and 'read only' in repr(transform)
    out = transform(rows)
    assert len(out) == 3
    assert torch.equal(out[0].positions, marker.positions)            # from the cache
    assert torch.equal(out[1].positions, old[1].positions)            # from the cache
    built = graph_utils.get_graphs_pyg(rows[2:3], transform.universe, 'all', 'not all', 3.0)[0]
    assert torch.equal(out[2].positions, built.positions)              # a miss, built again
    assert _edge_set(out[2].edge_index) == _edge_set(built.edge_index)
    transform(rows[1:])
    transform.close()
    assert n_cached() == 2 and _md5(path) == md5                     # nothing was stored
    # switched on for writing: the miss is stored, the cached graphs are kept
    writer = FlowGraphTransform(structure, 3.0, graph_cache=path, write_graph_cache=True)
    out = writer(rows)
    assert torch.equal(out[0].positions, marker.positions)
    writer.close()
    assert n_cached() == 3
    # an archived cache: a folder this process cannot write, no lock files next to it
    archive = tmp_path / 'archive'
    archive.mkdir()
    copy = archive / path.name
    copy.write_bytes(path.read_bytes())
    os.chmod(copy, 0o444)
    os.chmod(archive, 0o555)
    try:
        transform = FlowGraphTransform(structure, 3.0, graph_cache=copy)
        assert torch.equal(transform(rows[:1])[0].positions, marker.positions)
        transform.close()
        assert sorted(p.name for p in archive.iterdir()) == [path.name]
    finally:
        os.chmod(archive, 0o755)
        os.chmod(copy, 0o644)


@pytest.mark.graph
def test_large_box_keeps_cluster_whole(tmp_path):
    """A cluster whose geometric center is far from its middle stays whole in the cache route's box.

    graph_utils moves the geometric center of the generator atoms to the
    center of the box and wraps every atom into it; an atom that is half the
    box length or farther from that center would be wrapped to the other
    side, so the cache routes refuse such a frame instead of returning a split
    graph. The default route has no box: it takes every frame as it is.
    """
    pytest.importorskip('aimmd.network.graph_utils')
    rng = np.random.default_rng(1)
    ball = rng.uniform(-0.5, 0.5, size=(9, 3))                        # a dense head ...
    tail = np.stack([1.5 * np.arange(1, 13), np.zeros(12), np.zeros(12)], axis=1)   # ... and an 18 A tail
    frame = (np.concatenate([ball, tail]) + 1.0).astype(np.float32)
    structure = _write_structure(tmp_path / 'atoms.pdb', frame, box=20.0)
    descriptors = frame.reshape(1, -1)
    raw = radius_edges(torch.as_tensor(frame), torch.zeros(len(frame), dtype=torch.long), 2.5,
                       backend='torch')

    def cache_route(box_length):
        transform = FlowGraphTransform(structure, 2.5, box_length=box_length,
                                       graph_cache=tmp_path / f'graphs_box{box_length:g}.sqlite',
                                       write_graph_cache=True)
        try:
            return transform(descriptors)
        finally:
            transform.close()

    assert FlowGraphTransform.__init__.__kwdefaults__['box_length'] == 100.0
    (graph,) = cache_route(100.0)
    assert _edge_set(graph.edge_index) == _edge_set(raw)
    shift = graph.positions.double() - torch.as_tensor(frame).double()
    assert torch.allclose(shift, shift[:1].expand_as(shift), atol=1e-4)
    # the structure's own 20 A box: the tail end is more than 10 A from the center
    with pytest.raises(ValueError, match='box_length'):
        cache_route(20.0)
    (whole,) = cache_route(30.0)
    assert _edge_set(whole.edge_index) == _edge_set(raw)
    # without a cache there is no box to fit into
    batch = FlowGraphTransform(structure, 2.5, box_length=20.0, edge_backend='torch')(descriptors)
    assert _edge_set(batch['edge_index']) == _edge_set(raw)
    assert graphs_module._GRAPH_UTILS_CAP == 128
