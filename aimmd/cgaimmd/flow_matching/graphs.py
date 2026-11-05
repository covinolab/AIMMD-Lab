"""
aimmd.cgaimmd.flow_matching.graphs
==================================

Graph mode of flow-matching networks: the graphs of MD frames, and the
bond-edge and topology-edge features of the PaiNN network.

**Graph mode.** MD frames (value passes, final checks, the committor terms of
the fit) reach a graph network through ``params.descriptor_transform``. With
C-GAIMMD the descriptors are the flat Å coordinates of the generator atoms,
and :class:`FlowGraphTransform` turns a block of them into graphs:

- **By default** it returns one batch dict, built on the fly by
  :func:`aimmd.cgaimmd.flow_matching.graph_batch.build_graph_batch` (the
  layout of ``Batch.from_data_list(...).to_dict()``, as the node-table
  featurizer's ``batch_dict``). It needs nothing beyond torch, keeps no cache
  and writes no file. Generated configurations get their graphs from the same
  function, so MD frames and generator states are treated alike.
- **With** ``graph_cache=<file>`` it goes through AIMMD's graph pipeline and
  its SQLite graph cache (:mod:`aimmd.network.graph_utils`) and returns its
  list of PyG ``Data``. The cache is only read unless
  ``write_graph_cache=True``: graphs found there are used and the others are
  built and not stored. With ``write_graph_cache=True`` the cache is created
  when it does not exist and every new graph is stored, AIMMD's cached
  pipeline unchanged (``process_descriptors_pyg``).

:func:`bond_edge_features` is the bond-edge wrapper: a 4-way one-hot of the
topological distance of every edge (1, 2, 3 or more than 3 bonds), which the
PaiNN network appends to the radial features of every graph.

:func:`topology_edge_template` and :func:`merge_topology_edges` give the
PaiNN network its optional *topology edges*: every pair of atoms within three
bonds is an edge at every distance, merged with the radius graph of every
input, so that the bonded structure is visible to the network even when a
prior sample puts bonded atoms far apart.

``graph_utils`` (and with it ``torch_geometric``) is imported only by the
cache routes, when a transform is called.
"""

import os
import sqlite3
from numbers import Integral

import numpy as np
import torch

from .. import _once


N_BOND_CLASSES = 4
"""The number of topological distance classes of a bond edge (1, 2, 3 and more than 3 bonds, disconnected included)."""

_GRAPH_UTILS_CAP = 128      # the max_num_neighbors graph_utils.get_graphs_pyg gives radius_graph

TOPOLOGY_INDICATORS = ('radius_only', 'topology_only', 'both')
"""The channels of the topology-edge indicator, in order (a radius-graph edge more than three bonds apart, a topology edge beyond the cutoff, one within it)."""


def bond_edge_features(edge_index, batch, ptr, bond_class):
    """One-hot topological distance class of every edge.

    Parameters
    ----------
    edge_index : torch.Tensor
        ``(2, E)`` long ``[source, target]`` in batch (global) atom indices.
    batch : torch.Tensor
        ``(N,)`` long sample index of every atom.
    ptr : torch.Tensor
        ``(B + 1,)`` long offset of every sample's first atom.
    bond_class : torch.Tensor
        ``(M, M)`` uint8 class matrix, ``clip(topological distance, 1, 4) - 1``.

    Returns
    -------
    torch.Tensor
        ``(E, 4)`` float32 one-hot of the classes (1, 2, 3, > 3 bonds).

    Notes
    -----
    Each edge is looked up by its molecule-local indices
    ``edge_index - ptr[batch[edge_index]]``, so every sample must hold the
    ``M`` atoms of `bond_class` in flow order. Disconnected pairs carry the
    last class.
    """
    bond_class = bond_class.to(edge_index.device)
    start = ptr.to(edge_index.device)[batch.to(edge_index.device)[edge_index[0]]]
    local = edge_index - start.unsqueeze(0)
    classes = bond_class[local[0], local[1]].long()
    return torch.nn.functional.one_hot(classes, N_BOND_CLASSES).to(torch.float32)


def topology_edge_template(n_atoms, pairs, separation, distance_min, distance_max):
    """The fixed topology edges of one sample, both directions.

    Parameters
    ----------
    n_atoms : int
        Number ``M`` of flow atoms.
    pairs : array_like
        ``(P, 2)`` atom pairs within three bonds (``aimmd.cgaimmd.preparation.topology.ReferenceGeometry.pairs``).
    separation : array_like
        ``(P,)`` their topological distances, 1, 2 or 3.
    distance_min, distance_max : array_like
        ``(P,)`` their force-field reference distances, Å.

    Returns
    -------
    dict
        ``index`` ``(2, 2 P)`` long ``[source, target]`` (the pairs, then the reversed pairs), ``separation`` ``(2 P,)`` uint8, ``reference`` ``(2 P, 2)`` float32 ``(distance_min, distance_max)``, ``pair_mask`` ``(M, M)`` bool.

    Raises
    ------
    ValueError
        For pairs out of range or on one atom, a separation outside 1-3, or arrays of different lengths.
    """
    n_atoms = int(n_atoms)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    separation = np.asarray(separation, dtype=np.int64).reshape(-1)
    reference = np.stack([np.asarray(distance_min, dtype=np.float64).reshape(-1),
                          np.asarray(distance_max, dtype=np.float64).reshape(-1)], axis=1) \
        if len(pairs) else np.zeros((0, 2))
    if not len(pairs) == len(separation) == len(reference):
        raise ValueError('pairs, separation, distance_min and distance_max must have the same '
                         'length')
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= n_atoms
                       or bool((pairs[:, 0] == pairs[:, 1]).any())):
        raise ValueError(f'topology pairs must be distinct atoms in [0, {n_atoms})')
    if len(separation) and (separation.min() < 1 or separation.max() > 3):
        raise ValueError('topology pairs must be 1, 2 or 3 bonds apart')
    index = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
    mask = np.zeros((n_atoms, n_atoms), dtype=bool)
    mask[index[0], index[1]] = True
    return {'index': torch.as_tensor(index, dtype=torch.long),
            'separation': torch.as_tensor(np.concatenate([separation, separation]),
                                          dtype=torch.uint8),
            'reference': torch.as_tensor(np.concatenate([reference, reference]),
                                         dtype=torch.float32),
            'pair_mask': torch.as_tensor(mask)}


def merge_topology_edges(edge_index, batch, ptr, template_index, pair_mask):
    """The radius graph merged with the topology edges of every sample, each edge once.

    Parameters
    ----------
    edge_index : torch.Tensor
        ``(2, E)`` long radius-graph edges ``[source, target]`` in batch (global) atom indices.
    batch : torch.Tensor
        ``(N,)`` long sample index of every atom.
    ptr : torch.Tensor
        ``(B + 1,)`` long offset of every sample's first atom (every sample holds the ``M`` flow atoms in flow order).
    template_index : torch.Tensor
        ``(2, T)`` long topology edges of one sample (molecule-local), from :func:`topology_edge_template`.
    pair_mask : torch.Tensor
        ``(M, M)`` bool, True for the topology pairs.

    Returns
    -------
    tuple of torch.Tensor
        ``(edge_index (2, E'), topology_ids (E',))``: the radius-graph edges that are not topology pairs, then the ``T`` topology edges of every sample in sample order; ``topology_ids`` is the template column of a topology edge and -1 for a radius-only edge.

    Notes
    -----
    A radius-graph edge between a topology pair is dropped and comes back
    as the topology edge, so the result has no duplicates when the radius
    graph has none. Pure torch, on the device of `edge_index`.
    """
    device = edge_index.device
    batch, ptr = batch.to(device), ptr.to(device)
    template_index, pair_mask = template_index.to(device), pair_mask.to(device)
    start = ptr[batch[edge_index[0]]]
    local = edge_index - start.unsqueeze(0)
    radius_only = edge_index[:, ~pair_mask[local[0], local[1]]]
    n_samples, n_template = int(ptr.numel()) - 1, int(template_index.shape[1])
    topology = (template_index.unsqueeze(1) + ptr[:-1].reshape(1, -1, 1)).reshape(2, -1)
    ids = torch.cat([torch.full((radius_only.shape[1],), -1, dtype=torch.long, device=device),
                     torch.arange(n_template, device=device).repeat(n_samples)])
    return torch.cat([radius_only, topology], dim=1), ids


class FlowGraphTransform:
    """``descriptor_transform`` of graph mode: ``(n, 3 M)`` Å rows of the generator atoms to graphs.

    Parameters
    ----------
    structure : str
        Structure file (``.gro``, ``.pdb``) of the ``M`` generator atoms, in network order.
    cutoff : float
        Edge cutoff of the graphs, Å: the network's ``cutoff``.
    graph_cache : str or os.PathLike, optional
        SQLite graph cache of AIMMD's graph pipeline (:mod:`aimmd.network.graph_utils`). None (default): no cache; every call builds a batch dict on the fly and nothing is written.
    write_graph_cache : bool, optional
        With `graph_cache`: store every graph that is not in the cache yet, creating the file if it does not exist. False (default): only read the cache, which must exist.
    max_num_neighbors : int, optional
        Neighbour cap of the radius graphs (default 128, the network's default). The cache routes need 128, the cap ``graph_utils`` applies.
    edge_backend : {'auto', 'pyg', 'torch'}, optional
        Backend of the radius graphs of the default route (:func:`~aimmd.cgaimmd.flow_matching.graph_batch.radius_edges`).
    compression_lib : str, optional
        Codec of the graphs written to the cache (``'lz4'``, ``'gzip'``, ``'none'``); reads detect the codec.
    box_length : float, optional
        Side of the cubic box of the cache routes, Å (see the notes).

    Raises
    ------
    ValueError
        For a cutoff, cap, box length or backend out of range, `write_graph_cache` without `graph_cache`, or a cache route with a cap other than 128.
    FileNotFoundError
        For a `graph_cache` that does not exist and is only read.

    Notes
    -----
    **Default route (no cache).** A call returns the dict of
    :func:`~aimmd.cgaimmd.flow_matching.graph_batch.build_graph_batch` for the
    ``n`` frames: ``positions`` (the rows as float32, not moved),
    ``edge_index`` (every pair of atoms of a frame closer than `cutoff`),
    ``batch``, ``ptr``, ``node_attrs`` (the one-hot atom types of `structure`,
    in the columns ``graph_utils`` uses: its sorted atom types) and
    ``shifts`` (zeros). Graph networks take it as it is
    (``aimmd.network.fit`` and the C-GAIMMD values function accept batch
    dicts). Nothing is written, and ``graph_utils``, ``torch_geometric`` and
    SQLite are not used. With ``edge_backend='pyg'`` (or ``'auto'`` where
    ``torch_cluster`` is installed) the edges are those of the
    ``radius_graph`` call the cache route makes.

    **Cache routes.** A call returns the list of PyG ``Data`` of
    ``graph_utils``, one per frame, as AIMMD's graph pipeline does. With
    ``write_graph_cache=True`` it is ``graph_utils.process_descriptors_pyg(
    descriptors, universe, 'all', 'not all', cutoff, conn,
    compression_lib=...)`` on a connection of ``graph_utils.init_db``:
    AIMMD's cached graph pipeline, unchanged (the trainer's reader role and
    the node-local replica work as with any other graph cache). Without it the
    cache is opened read-only (SQLite ``mode=ro``): graphs found under their
    key are used, the others are built by ``graph_utils.get_graphs_pyg`` and
    not stored, so no row is added and the file is not changed. SQLite may
    leave its ``-wal`` and ``-shm`` lock files next to a cache in WAL mode;
    in a folder this process cannot write (an archive) it cannot create them,
    and the cache is then read as a file that nobody changes (``immutable=1``).
    The cache is opened at the first call, not at construction: every process
    that loads the params file builds the transform, and only those that
    transform frames open the cache.

    The universe of the cache routes holds the generator atoms of
    `structure` with guessed bonds (``unwrap`` needs them) and a cubic box of
    side `box_length`; the environment selection is empty and node order is
    network order. ``center_in_box`` moves the geometric center of every
    frame to the center of that box and ``wrap`` then moves every atom into
    it, so a frame stays whole only if no atom is ``box_length / 2`` or
    farther from the frame's center along an axis; a call raises
    ``ValueError`` for such a frame instead of returning a split graph. The
    positions of these graphs are float32 near ``box_length / 2``, so their
    rounding error grows with the box: the default of 100 Å keeps it at the
    level of the MD coordinates themselves (a 1000 Å box costs about ten
    times more, which a steep committor turns into differences between
    ``forward`` on graphs and ``committor`` on coordinates that the start-up
    checks can reject). The default route has no box and no such limit.

    The cache key is the descriptor row only, so one cache must not be
    shared between different generator atoms, cutoffs, selections or box
    lengths (name the file after them, e.g. ``graphs_M22_rc5.0.sqlite``).
    Rows are passed as stored: their dtype and bytes are part of the key.

    ``graph_utils`` fixes the neighbour cap of ``radius_graph`` at 128, so a
    cache route cannot honour another cap; a warning is printed once when a
    node of a returned graph reaches it (after the warning the neighbour
    counts are no longer checked). The default route warns through
    :func:`~aimmd.cgaimmd.flow_matching.graph_batch.radius_edges`. The
    transform's cutoff and cap must be the network's, or ``forward`` on its
    graphs differs from ``committor`` on the same coordinates.

    Examples
    --------
    In a params file (the default route; nothing is cached)::

        _graphs = FlowGraphTransform('generator_atoms.pdb', 5.0)

        def descriptor_transform(descriptors):
            return _graphs(descriptors)
    """

    def __init__(self, structure, cutoff, *, graph_cache=None, write_graph_cache=False,
                 max_num_neighbors=128, edge_backend='auto', compression_lib='lz4',
                 box_length=100.0):
        from .graph_batch import BACKENDS
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError(f'cutoff must be a finite number > 0, got {cutoff!r}')
        if isinstance(max_num_neighbors, bool) or not isinstance(max_num_neighbors, Integral) \
                or max_num_neighbors < 1:
            raise ValueError(f'max_num_neighbors must be an int >= 1, got {max_num_neighbors!r}')
        if edge_backend not in BACKENDS:
            raise ValueError(f'edge_backend must be one of {BACKENDS}, got {edge_backend!r}')
        if not float(box_length) > 0:
            raise ValueError(f'box_length must be > 0, got {box_length!r}')
        if not isinstance(write_graph_cache, (bool, np.bool_)):
            raise ValueError(f'write_graph_cache must be a bool, got {write_graph_cache!r}')
        if write_graph_cache and graph_cache is None:
            raise ValueError('write_graph_cache=True needs graph_cache, the SQLite file to write '
                             'to (without graph_cache no graph is cached)')
        if graph_cache is not None:
            if not isinstance(graph_cache, (str, os.PathLike)):
                raise ValueError(f'graph_cache must be the path of an SQLite graph cache, got '
                                 f'{type(graph_cache).__name__}')
            if max_num_neighbors != _GRAPH_UTILS_CAP:
                raise ValueError(f'FlowGraphTransform: graph_utils caps the neighbours of every '
                                 f'node at {_GRAPH_UTILS_CAP} and is not changed by C-GAIMMD '
                                 f'code; max_num_neighbors={max_num_neighbors!r} cannot be '
                                 f'honoured with graph_cache')
            # relative to the folder of the params file (Params.load executes it there)
            graph_cache = os.path.abspath(os.fspath(graph_cache))
            if not write_graph_cache and not os.path.isfile(graph_cache):
                raise FileNotFoundError(
                    f'graph_cache {graph_cache} does not exist; it is only read unless '
                    f'write_graph_cache=True (without graph_cache no cache is used)')
        import MDAnalysis as mda
        self.structure = os.path.abspath(str(structure))
        """The structure file of the generator atoms."""
        self.cutoff = cutoff
        """Edge cutoff, Å."""
        self.graph_cache = graph_cache
        """Absolute path of the SQLite graph cache, or None (the default route)."""
        self.write_graph_cache = bool(write_graph_cache)
        """True if new graphs are stored in :attr:`graph_cache`."""
        self.max_num_neighbors = int(max_num_neighbors)
        """Neighbour cap of the radius graphs."""
        self.edge_backend = edge_backend
        """Backend of the radius graphs of the default route."""
        self.compression_lib = compression_lib
        """Codec of the graphs written to the cache."""
        self.box_length = float(box_length)
        """Side of the cubic box of the cache routes, Å."""
        self._conn = None
        self._universe = None
        self._cap_warned = False
        types = list(mda.Universe(self.structure).atoms.types)
        columns = sorted(set(types))
        node_attrs = np.zeros((len(types), len(columns)), dtype=np.float32)
        node_attrs[np.arange(len(types)), [columns.index(t) for t in types]] = 1.0
        self._node_attrs = torch.as_tensor(node_attrs)

    @property
    def n_atoms(self):
        """Number ``M`` of generator atoms."""
        return int(self._node_attrs.shape[0])

    @property
    def universe(self):
        """MDAnalysis universe of the cache routes, built on first use (the default route does not need it).

        The generator atoms of :attr:`structure` with guessed bonds, in a
        cubic box of side :attr:`box_length`.
        """
        if self._universe is None:
            import MDAnalysis as mda
            from MDAnalysis.coordinates.memory import MemoryReader
            universe = mda.Universe(self.structure, guess_bonds=True)
            box = np.array([self.box_length] * 3 + [90.0] * 3, dtype=np.float32)
            universe.load_new(universe.atoms.positions.copy()[None], format=MemoryReader,
                              dimensions=box)
            self._universe = universe
        return self._universe

    @property
    def node_attrs(self):
        """``(M, T)`` float32 one-hot atom types of the default route's graphs (``graph_utils``' columns)."""
        return self._node_attrs.clone()

    def __repr__(self):
        cache = 'none' if self.graph_cache is None else (
            f'{self.graph_cache!r} ({"read and write" if self.write_graph_cache else "read only"})')
        return (f'{type(self).__name__}({self.structure!r}, cutoff={self.cutoff:g}, '
                f'graph_cache={cache})')

    def __call__(self, descriptors):
        """Graphs of `descriptors`.

        Parameters
        ----------
        descriptors : numpy.ndarray
            ``(n, 3 M)`` Å rows as stored (with a cache their bytes are the keys).

        Returns
        -------
        dict or list
            The default route's batch dict, or with a cache the list of PyG ``Data`` (one graph per row; ``[]`` for ``n = 0``).

        Raises
        ------
        ValueError
            If the width is not ``3 M``, or (cache routes) a frame does not fit whole into the box (see the class notes).
        """
        descriptors = np.asarray(descriptors)
        width = 3 * self.n_atoms
        if descriptors.ndim != 2 or descriptors.shape[1] != width:
            raise ValueError(f'FlowGraphTransform: descriptors must have width {width} '
                             f'(3 x {self.n_atoms} generator atoms), got shape '
                             f'{descriptors.shape}')
        if self.graph_cache is None:
            return self._batch_dict(descriptors)
        if len(descriptors) == 0:
            return []
        self._check_box(descriptors)
        if self.write_graph_cache:
            graphs = self._cached_graphs(descriptors)
        else:
            graphs = self._read_only_graphs(descriptors)
        self._warn_cap(graphs)
        return graphs

    def close(self):
        """Close the connection to the graph cache, if one is open (the next call opens it again)."""
        if self._conn is not None:
            conn, self._conn = self._conn, None
            conn.close()

    # -- the default route ------------------------------------------------------------------------

    def _batch_dict(self, descriptors):
        from .graph_batch import build_graph_batch
        n, m = len(descriptors), self.n_atoms
        positions = torch.as_tensor(np.asarray(descriptors, dtype=np.float32)).reshape(n, m, 3)
        if n == 0:
            return {'positions': positions.reshape(0, 3),
                    'edge_index': torch.zeros((2, 0), dtype=torch.long),
                    'batch': torch.zeros(0, dtype=torch.long),
                    'ptr': torch.zeros(1, dtype=torch.long),
                    'node_attrs': self._node_attrs[:0].clone(),
                    'shifts': torch.zeros((0, 3), dtype=torch.float32)}
        return build_graph_batch(positions, self.cutoff, node_attrs=self._node_attrs,
                                 max_num_neighbors=self.max_num_neighbors,
                                 backend=self.edge_backend)

    # -- the cache routes -------------------------------------------------------------------------

    def _check_box(self, descriptors):
        frames = descriptors.reshape(len(descriptors), -1, 3).astype(np.float64)
        spread = np.abs(frames - frames.mean(axis=1, keepdims=True)).max(axis=(1, 2))
        if float(spread.max()) >= 0.5 * self.box_length:
            k = int(spread.argmax())
            raise ValueError(f'FlowGraphTransform: frame {k} of the call reaches '
                             f'{float(spread[k]):.1f} A from its center along an axis, at least '
                             f'half the box_length ({self.box_length:g} A): graph_utils would wrap '
                             f'part of it to the other side of the box; pass a larger box_length '
                             f'(and use a new graph cache for it)')

    def _cached_graphs(self, descriptors):
        from ...network import graph_utils
        if self._conn is None:
            self._conn = graph_utils.init_db(db_path=self.graph_cache)
        return list(graph_utils.process_descriptors_pyg(
            descriptors, self.universe, 'all', 'not all', self.cutoff, self._conn,
            compression_lib=self.compression_lib))

    def _read_only_connection(self):
        """A read-only connection to the cache (``mode=ro``; ``immutable=1`` where that cannot open it)."""
        from pathlib import Path
        uri = Path(self.graph_cache).as_uri() + '?mode=ro'
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        try:
            conn.execute('SELECT 1 FROM graphs_cache LIMIT 1').fetchall()
            return conn
        except sqlite3.OperationalError:
            # a cache in WAL mode in a folder this process cannot write (an archive): SQLite cannot
            # create its lock files there, so it reads the file as one that nobody changes
            conn.close()
        conn = sqlite3.connect(uri + '&immutable=1', uri=True, timeout=30.0)
        conn.execute('SELECT 1 FROM graphs_cache LIMIT 1').fetchall()
        return conn

    def _read_only_graphs(self, descriptors):
        from ...network import graph_utils
        if self._conn is None:
            self._conn = self._read_only_connection()
        keys = [graph_utils.get_stable_hash(row) for row in descriptors]
        graphs = [graph_utils.load_from_sqlite(key, self._conn) for key in keys]
        missing = [k for k, graph in enumerate(graphs) if graph is None]
        if missing:
            built = graph_utils.get_graphs_pyg(descriptors[missing], self.universe, 'all',
                                               'not all', self.cutoff)
            for k, graph in zip(missing, built):
                graphs[k] = graph
        return graphs

    def _warn_cap(self, graphs):
        if self._cap_warned:
            return
        for data in graphs:
            edges = data.edge_index
            if edges.shape[1] and int(torch.bincount(edges[1]).max()) >= self.max_num_neighbors:
                self._cap_warned = True
                if _once.once(('FlowGraphTransform cap', self.structure, self.cutoff)):
                    print(f'WARNING: a node of a graph of {self.structure} has reached '
                          f'{self.max_num_neighbors} neighbours within {self.cutoff} A; its '
                          f'neighbour list may be truncated', flush=True)
                return
