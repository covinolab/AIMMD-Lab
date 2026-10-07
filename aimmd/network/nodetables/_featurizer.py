"""Node tables: graph-network inputs stored as fixed-width descriptor rows.

A node table is the frame-specific part of the graph that
`aimmd.network.graph_utils.get_graphs_pyg` builds for a frame: the positions
of the graph's nodes after the periodic-boundary handling, and the atom type of
each node. Everything else in that graph follows from the node table and a few
settings that do not change from frame to frame:

- ``node_attrs``, the one-hot encoding of the atom types (``atom_types``);
- ``edge_index``, ``radius_graph(positions, r=cutoff, loop=False,
  max_num_neighbors=max_num_neighbors)``, built on the CPU;
- ``shifts``, all zero.

`NodeTableFeaturizer.descriptors_function` stores one fixed-width float32 row
per frame, so the rows live in AIMMD's per-trajectory descriptor series
``{trajectory}.{descriptors_series}.npy`` like any other descriptors: the
compute ledger, the registration of new paths, the initial-path export and the
deletion of rejected segments handle them unchanged. `graphs` and `batch_dict`
rebuild the network input from the rows, bitwise identical to what
`get_graphs_pyg` and ``Batch.from_data_list`` give for the same frames. No
graph cache, no database and no hashing are involved.

Row layout (float32, version `NODE_TABLE_LAYOUT`)::

    [0]                      number of nodes, n_nodes
    [1]                      layout version
    [2 : 2+3*n_max]          node positions (Angstrom), node by node (x, y, z)
    [2+3*n_max : 2+4*n_max]  atom-type index of each node into atom_types
                             (zero padding after n_nodes in both blocks)

The nodes are the system atoms, then the environment atoms, each in atom-index
order, after unwrapping the system, centering it in the reference box and
wrapping every atom into that box, exactly as in `get_graphs_pyg`. A computed
row always has ``n_nodes > 0``; an all-zero row is a frame that has not been
featurized (or could not be, see `NodeTableFeaturizer`).

This module only needs numpy and MDAnalysis to write rows. Turning rows into
graphs (`NodeTableFeaturizer.graphs`, `NodeTableFeaturizer.batch_dict`) needs
torch and the optional ``graphs`` extra (torch_geometric, torch_cluster).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import MDAnalysis.transformations as transformations


#: Version of the row layout, stored in column 1 of every computed row.
NODE_TABLE_LAYOUT = 1

#: Default maximum number of graph nodes per frame (row width 2 + 4 * 768).
DEFAULT_N_MAX = 768

#: Fraction of ``n_max`` above which `NodeTableFeaturizer.descriptors_function`
#: warns that frames come close to the row capacity.
N_MAX_WARNING_FRACTION = 0.8

#: Prefix of every node-table series name (followed by 10 hex characters).
SERIES_PREFIX = 'descriptors-gn'

#: Command that rewrites node-table files to a wider layout.
REPACK_COMMAND = 'python -m aimmd.network.nodetables repack --n-max N'

# Frames named in a log line, at most
_MAX_REPORTED_FRAMES = 5


class NodeTableOverflowError(ValueError):
    """A graph has more nodes than the row layout (``n_max``) can hold."""


def _radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    """`torch_geometric.nn.radius_graph`, imported when first needed."""
    from torch_geometric.nn import radius_graph
    return radius_graph(x=x, r=r, batch=batch, loop=loop,
                        max_num_neighbors=max_num_neighbors)


def _to_numpy(array):
    """A numpy view or copy of a numpy array or a (CPU or device) tensor."""
    if hasattr(array, 'detach'):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _frame_names(frames):
    """Short text naming the first few of `frames` (time labels)."""
    names = ', '.join(str(frame) for frame in frames[:_MAX_REPORTED_FRAMES])
    if len(frames) > _MAX_REPORTED_FRAMES:
        names += ', ...'
    return names


class NodeTableFeaturizer:
    """Featurize frames into node-table rows and rows into graph batches.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe with the full topology, the bonds (for unwrapping the system)
        and the reference box: frame 0 of its trajectory, usually the
        structure file. As in `get_graphs_pyg`, the periodic-boundary handling
        always uses this box, not the box of the featurized frame. The
        featurizer overwrites the universe's positions; do not share the
        universe with code that needs them.
    system_selection : str
        MDAnalysis selection of the system atoms (e.g. the ligand heavy atoms).
    environment_selection : str
        MDAnalysis selection of the environment atoms, evaluated per frame
        after the periodic-boundary handling (e.g. ``'not type H and around
        8.0 (resname INH)'``).
    atom_types : list of str or None
        Ordered atom-type table of the one-hot ``node_attrs`` columns, as in
        `get_graphs_pyg`. None derives it from the universe,
        ``sorted(set(universe.atoms.types))``.
    cutoff : float
        Edge cutoff of the graphs in Angstrom.
    n_max : int, default=768
        Maximum number of graph nodes a row can hold. Rows have
        ``2 + 4 * n_max`` float32 columns.
    max_num_neighbors : int, default=128
        ``max_num_neighbors`` of ``radius_graph``, as in `get_graphs_pyg`.

    Raises
    ------
    ValueError
        If the universe has no box, the system atoms have no bonds, the system
        selection is empty or larger than `n_max`, a system atom has a type
        outside `atom_types`, or a numeric setting is not positive.

    Notes
    -----
    **Series name.** `series` is ``'descriptors-gn'`` followed by the first
    10 hex characters of a SHA-256 over `spec`: the layout version, `n_max`,
    the two selections, `atom_types`, the atom count and a hash of the atom
    names, types and residue names and of the reference box. Any change to
    what a row holds gives a new name, so rows are never read under settings
    they were not written with. `cutoff` and `max_num_neighbors` are
    deliberately not part of the name: the edges are rebuilt from the rows
    every time, so changing them changes every graph without touching the
    stored rows (and requires a new network). Pin the name in the params file
    and check it with `check_series`.

    **Overflow.** `descriptors_function` never raises on a frame that cannot
    be stored, since it runs inside the MD stop-condition loop. A frame with
    more than `n_max` nodes, or with an environment atom whose type is not in
    `atom_types`, is reported as an error on stdout and gets an all-zero row.
    The ledger treats the row as missing, so every later pass featurizes and
    reports the frame again, and `graphs` and `batch_dict` refuse zero rows
    with an error naming ``python -m aimmd.network.nodetables repack --n-max
    N``: training and value passes (the trainer's, and a worker's selection
    or TPS acceptance on a path with such a frame) stop instead of using a
    wrong value. Frames with more than 80 % of `n_max` nodes are reported as
    a warning.

    **Params files.** `aimmd.Params` stores functions, not objects: a bound
    method such as ``FEATURIZER.descriptors_function`` would lose its
    featurizer, and Params refuses it (and a ``functools.partial`` of it).
    Use module-level wrapper functions::

        FEATURIZER = NodeTableFeaturizer(tmp_universe, SYSTEM_SELECTION,
                                         ENVIRONMENT_SELECTION, ATOM_TYPES,
                                         cutoff=CUTOFF)
        descriptors_series = FEATURIZER.check_series('descriptors-gn...')

        def descriptors_function(trajectory):
            return FEATURIZER.descriptors_function(trajectory)

        def descriptor_transform(rows, verbose=False):
            return FEATURIZER.batch_dict(rows)

    Examples
    --------
    >>> featurizer = NodeTableFeaturizer(universe, 'resname LIG and not type H',
    ...                                  'not type H and around 8.0 (resname LIG)',
    ...                                  ['H', 'C', 'N', 'O'], cutoff=4.0)
    >>> rows = featurizer.descriptors_function(universe.trajectory)
    >>> batch = featurizer.batch_dict(rows[:32], device='cuda')
    >>> output = network(batch)
    """

    # Params unwraps a bound method to its bare function, which would drop the
    # featurizer: ParamsHelpers._setattr refuses bound methods of objects that
    # set this flag, with a message pointing to module-level wrappers.
    _params_requires_wrapper = True

    def __init__(self, universe, system_selection, environment_selection,
                 atom_types, cutoff, n_max=DEFAULT_N_MAX,
                 max_num_neighbors=128):
        self.universe = universe
        self.system_selection = str(system_selection)
        self.environment_selection = str(environment_selection)
        self.cutoff = float(cutoff)
        self.n_max = int(n_max)
        self.max_num_neighbors = int(max_num_neighbors)
        if not self.cutoff > 0:
            raise ValueError(f'cutoff must be positive, got {cutoff!r}')
        if self.n_max < 1 or self.n_max != n_max:
            raise ValueError(f'n_max must be a positive integer, got {n_max!r}')
        if self.max_num_neighbors < 1:
            raise ValueError(f'max_num_neighbors must be positive, got '
                             f'{max_num_neighbors!r}')

        # the reference frame: its box is used for every featurized frame
        universe.trajectory[0]
        if universe.dimensions is None:
            raise ValueError(
                'the universe has no box: the periodic-boundary handling of '
                'the node tables needs the reference box (e.g. a .gro file)')
        self.n_atoms = len(universe.atoms)

        # atom-type table, as in get_graphs_pyg
        if atom_types is None:
            atom_types = sorted(set(universe.atoms.types))
        self.atom_types = [str(atom_type) for atom_type in atom_types]
        lookup = {atom_type: i for i, atom_type in enumerate(self.atom_types)}
        self._type_index = np.array(
            [lookup.get(atom_type, -1) for atom_type in universe.atoms.types],
            dtype=np.int64)

        # system: fixed for every frame
        self.system = universe.select_atoms(self.system_selection)
        if not self.system.n_atoms:
            raise ValueError(f'the system selection {self.system_selection!r} '
                             f'selects no atoms')
        if self.system.n_atoms > self.n_max:
            raise ValueError(
                f'the system selection has {self.system.n_atoms} atoms, more '
                f'than n_max={self.n_max} graph nodes: raise n_max')
        unknown = sorted(set(self.system.types[
            self._type_index[self.system.indices] < 0]))
        if unknown:
            raise ValueError(
                f'atom type(s) {unknown} of the system are missing from '
                f'atom_types {self.atom_types}')

        # the transformations of get_graphs_pyg (they keep no per-frame state)
        try:
            self._unwrap = transformations.unwrap(self.system)
        except AttributeError as error:
            raise ValueError(
                'the system atoms have no bonds, which unwrapping needs: build '
                "the universe with bonds (e.g. to_guess=['bonds', ...])"
            ) from error
        self._center = transformations.center_in_box(self.system, wrap=False)
        self._wrap = transformations.wrap(universe.atoms)

        # row offsets of the positions and type blocks
        self._positions_start = 2
        self._types_start = 2 + 3 * self.n_max
        self._warning_nodes = int(N_MAX_WARNING_FRACTION * self.n_max)

        # identity of the stored rows
        self._spec = self._make_spec()
        self._fingerprint = hashlib.sha256(
            json.dumps(self._spec, sort_keys=True).encode()).hexdigest()

    def __repr__(self):
        return (f'{type(self).__name__}(series={self.series!r}, '
                f'n_max={self.n_max}, cutoff={self.cutoff})')

    # ------------------------------------------------------------------
    # identity of the series

    def _make_spec(self):
        universe = self.universe
        topology = hashlib.sha256()
        for values in (universe.atoms.names, universe.atoms.types,
                       universe.atoms.resnames):
            topology.update('\0'.join(map(str, values)).encode())
        topology.update(
            np.asarray(universe.dimensions, dtype=np.float64).tobytes())
        return {'layout': NODE_TABLE_LAYOUT,
                'n_max': self.n_max,
                'system': self.system_selection,
                'environment': self.environment_selection,
                'atom_types': list(self.atom_types),
                'n_atoms': self.n_atoms,
                'topology': topology.hexdigest()}

    def spec(self):
        """Everything that determines the stored rows, as a JSON-able dict.

        Returns
        -------
        dict
            ``layout``, ``n_max``, ``system`` and ``environment`` (the
            selections), ``atom_types``, ``n_atoms`` and ``topology``, a
            SHA-256 over the atom names, types and residue names and the
            reference box. `cutoff` and `max_num_neighbors` are not part of
            it (see the class notes).
        """
        return json.loads(json.dumps(self._spec))

    @property
    def fingerprint(self):
        """str: SHA-256 (hex) of `spec`."""
        return self._fingerprint

    @property
    def series(self):
        """str: the descriptor series of these rows, ``'descriptors-gn'`` and
        the first 10 hex characters of `fingerprint`."""
        return SERIES_PREFIX + self._fingerprint[:10]

    @property
    def width(self):
        """int: number of float32 columns of a row, ``2 + 4 * n_max``."""
        return 2 + 4 * self.n_max

    def check_series(self, descriptors_series):
        """Check a pinned series name against this featurizer.

        Parameters
        ----------
        descriptors_series : str
            The ``descriptors_series`` of the params file.

        Returns
        -------
        str
            `descriptors_series`, unchanged.

        Raises
        ------
        ValueError
            If it is not `series`: the stored rows were written with other
            settings (a changed selection or `n_max`, another topology, or a
            different MDAnalysis type guess on this host).
        """
        if descriptors_series != self.series:
            raise ValueError(
                f'descriptors_series {descriptors_series!r} does not match '
                f'this node-table featurizer, whose series is '
                f'{self.series!r}. The name fingerprints everything a row '
                f'holds: {self._spec}. If you changed the featurizer on '
                f'purpose, pin descriptors_series = {self.series!r}; every '
                f'frame is then featurized again (prefill the new series with '
                f'python -m aimmd.network.nodetables prefill). Otherwise find '
                f'what differs, e.g. an edited selection or atom types that '
                f'MDAnalysis guessed differently on this host.')
        return descriptors_series

    def with_n_max(self, n_max):
        """The featurizer of the same graphs with another row capacity.

        Every setting but `n_max` is kept, so the rows of a frame differ only
        in their padding (see `repack_rows`); the series name changes with
        `n_max`.

        Parameters
        ----------
        n_max : int
            Maximum number of graph nodes per row.

        Returns
        -------
        NodeTableFeaturizer
            A new featurizer on the same universe. Like any featurizer it
            overwrites the universe's positions when it featurizes, so use
            the two one after the other, not concurrently.
        """
        return type(self)(self.universe, self.system_selection,
                          self.environment_selection, self.atom_types,
                          self.cutoff, n_max=n_max,
                          max_num_neighbors=self.max_num_neighbors)

    # ------------------------------------------------------------------
    # frames -> rows

    def descriptors_function(self, trajectory):
        """Node-table rows of the frames of an MDAnalysis trajectory.

        Use it through a module-level ``descriptors_function`` wrapper in the
        params file (see the class notes). It never raises on a frame that
        does not fit the layout: such frames get zero rows and are reported.

        Parameters
        ----------
        trajectory : iterable of MDAnalysis.coordinates.timestep.Timestep
            The frames, e.g. a reader or a slice of one. Each must have
            ``universe``'s atom count.

        Returns
        -------
        numpy.ndarray
            float32, shape ``(n_frames, width)``.
        """
        frames = ((ts.time, ts.positions) for ts in trajectory)
        return self._featurize(frames)

    def rows_from_coordinates(self, coordinates):
        """Node-table rows of coordinate frames.

        Parameters
        ----------
        coordinates : array-like
            Positions of all atoms in Angstrom, shape ``(n_frames, n_atoms *
            3)`` (the rows of a 'descriptors' series of atom coordinates) or
            ``(n_frames, n_atoms, 3)``.

        Returns
        -------
        numpy.ndarray
            float32, shape ``(n_frames, width)``, as `descriptors_function`.
        """
        coordinates = np.asarray(coordinates)
        coordinates = coordinates.reshape(len(coordinates), -1, 3)
        return self._featurize(
            (f'#{i}', frame) for i, frame in enumerate(coordinates))

    def _nodes(self, positions):
        """The graph nodes (an AtomGroup) of a frame, as in get_graphs_pyg."""
        universe = self.universe
        ts = universe.trajectory[0]
        universe.atoms.positions = positions
        ts = self._unwrap(ts)
        ts = self._center(ts)
        ts = self._wrap(ts)
        return self.system + universe.select_atoms(self.environment_selection)

    def _featurize(self, frames):
        """Rows of `frames`, ``(label, positions)`` pairs; reports, never
        raises on, frames that do not fit."""
        rows = []
        overflow, unknown, crowded = [], [], []
        for label, positions in frames:
            if len(positions) != self.n_atoms:
                raise ValueError(f'frame {label} has {len(positions)} atoms, '
                                 f'the featurizer universe {self.n_atoms}')
            row = np.zeros(self.width, dtype=np.float32)
            rows.append(row)
            nodes = self._nodes(positions)
            n_nodes = nodes.n_atoms
            if n_nodes > self.n_max:
                overflow.append((label, n_nodes))
                continue
            types = self._type_index[nodes.indices]
            if (types < 0).any():
                unknown.append((label, sorted(set(nodes.types[types < 0]))))
                continue
            if n_nodes > self._warning_nodes:
                crowded.append((label, n_nodes))
            row[0] = n_nodes
            row[1] = NODE_TABLE_LAYOUT
            start = self._positions_start
            row[start:start + 3 * n_nodes] = nodes.positions.ravel()
            start = self._types_start
            row[start:start + n_nodes] = types
        self._report(overflow, unknown, crowded)
        if not rows:
            return np.zeros((0, self.width), dtype=np.float32)
        return np.stack(rows)

    def _report(self, overflow, unknown, crowded):
        """Print one line per kind of problem found in a featurization."""
        if overflow:
            largest = max(n_nodes for _, n_nodes in overflow)
            print(f'ERROR: node tables: {len(overflow)} frame(s) have more '
                  f'graph nodes than n_max={self.n_max} (largest {largest}; '
                  f'frames at t = '
                  f'{_frame_names([label for label, _ in overflow])}). Their '
                  f'rows are left zero, and training and value passes stop on '
                  f'them. Widen the rows with {REPACK_COMMAND!r} (N >= '
                  f'{largest}), set n_max=N in the featurizer and pin the new '
                  f'descriptors_series.', flush=True)
        if unknown:
            types = sorted(set().union(*(set(t) for _, t in unknown)))
            print(f'ERROR: node tables: {len(unknown)} frame(s) have graph '
                  f'nodes of atom type(s) {types}, which are not in '
                  f'atom_types {self.atom_types} (frames at t = '
                  f'{_frame_names([label for label, _ in unknown])}). Their '
                  f'rows are left zero, and training and value passes stop on '
                  f'them.', flush=True)
        if crowded:
            largest = max(n_nodes for _, n_nodes in crowded)
            print(f'WARNING: node tables: {len(crowded)} frame(s) use more '
                  f'than {N_MAX_WARNING_FRACTION:.0%} of n_max={self.n_max} '
                  f'graph nodes (largest {largest}). A frame above n_max gets '
                  f'a zero row, which stops training: consider '
                  f'{REPACK_COMMAND!r} with a larger N.', flush=True)

    # ------------------------------------------------------------------
    # rows -> graphs

    def _unpack(self, rows):
        """Validated ``(positions, types)`` per row: float32 ``(n, 3)`` and
        int64 ``(n,)`` numpy arrays."""
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError(
                f'node-table rows must have shape (n_frames, {self.width}) '
                f'for n_max={self.n_max}, got {rows.shape}: rows of another '
                f'series or layout?')
        n_nodes = rows[:, 0]
        zero = np.flatnonzero(n_nodes == 0)
        if len(zero):
            raise ValueError(
                f'{len(zero)} node-table row(s) are empty (n_nodes == 0; '
                f'row(s) {zero[:_MAX_REPORTED_FRAMES].tolist()} of this '
                f'batch). A zero row is a frame that was never featurized, or '
                f'one whose graph does not fit the layout (descriptors_function '
                f'then logs an ERROR line). For n_max overflows, widen the '
                f'rows with {REPACK_COMMAND!r}, set n_max=N in the featurizer '
                f'and pin the new descriptors_series; the ledger then '
                f'featurizes the empty rows again. Otherwise compute the '
                f'missing rows first, e.g. '
                f'path.compute(*params.compute_descriptors_args).')
        bad = np.flatnonzero((rows[:, 1] != NODE_TABLE_LAYOUT) |
                             (n_nodes != np.round(n_nodes)) |
                             (n_nodes < 0) | (n_nodes > self.n_max))
        if len(bad):
            raise ValueError(
                f'node-table row(s) {bad[:_MAX_REPORTED_FRAMES].tolist()} of '
                f'this batch are not layout-{NODE_TABLE_LAYOUT} rows with at '
                f'most n_max={self.n_max} nodes')
        unpacked = []
        n_types = len(self.atom_types)
        for i, (row, n) in enumerate(zip(rows, n_nodes.astype(np.int64))):
            start = self._positions_start
            positions = row[start:start + 3 * n].reshape(n, 3)
            start = self._types_start
            types = row[start:start + n]
            if ((types != np.round(types)).any() or (types < 0).any() or
                    (types >= n_types).any()):
                raise ValueError(f'node-table row {i} of this batch has atom '
                                 f'types outside 0..{n_types - 1}')
            unpacked.append((positions, types.astype(np.int64)))
        return unpacked

    def graphs(self, rows):
        """Graphs of node-table rows, as `get_graphs_pyg` builds them.

        Parameters
        ----------
        rows : array-like
            Node-table rows, shape ``(n_frames, width)``.

        Returns
        -------
        aimmd.network.graph_utils.GraphList
            One ``torch_geometric.data.Data`` per row, with ``positions``,
            ``edge_index``, ``node_attrs`` and ``shifts``, bitwise equal to
            the graph `get_graphs_pyg` builds for the same frame. Built on the
            CPU.

        Raises
        ------
        ValueError
            On a zero row (``n_nodes == 0``) or a row of another layout.
        """
        import torch
        from torch_geometric.data import Data
        from ..graph_utils import GraphList

        result = GraphList()
        n_types = len(self.atom_types)
        for positions, types in self._unpack(rows):
            n_nodes = len(types)
            positions = torch.tensor(positions, dtype=torch.float)
            node_attrs = torch.zeros((n_nodes, n_types), dtype=torch.float)
            node_attrs[torch.arange(n_nodes), torch.from_numpy(types)] = 1.0
            edge_index = _radius_graph(
                positions, r=self.cutoff, loop=False,
                max_num_neighbors=self.max_num_neighbors)
            shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.float)
            result.append(Data(positions=positions, edge_index=edge_index,
                               node_attrs=node_attrs, shifts=shifts))
        return result

    def batch_dict(self, rows, device='cpu'):
        """Network input of a batch of node-table rows.

        Equal to ``Batch.from_data_list(self.graphs(rows)).to(device)
        .to_dict()``, built directly with one ``radius_graph`` call over the
        batch. Everything, the edges included, is built on the CPU and only
        then moved to `device`, so the result does not depend on the device.

        Parameters
        ----------
        rows : array-like
            Node-table rows, shape ``(n_frames, width)``, ``n_frames >= 1``.
        device : str or torch.device, default='cpu'
            Device of the returned tensors.

        Returns
        -------
        dict of torch.Tensor
            ``positions``, ``edge_index``, ``node_attrs``, ``shifts``,
            ``batch`` and ``ptr``.

        Raises
        ------
        ValueError
            On a zero row (``n_nodes == 0``), a row of another layout or no
            rows.
        """
        import torch

        unpacked = self._unpack(rows)
        if not unpacked:
            raise ValueError('batch_dict needs at least one row')
        counts = torch.tensor([len(types) for _, types in unpacked],
                              dtype=torch.long)
        positions = torch.from_numpy(
            np.concatenate([positions for positions, _ in unpacked]))
        types = torch.from_numpy(
            np.concatenate([types for _, types in unpacked]))
        n_nodes = len(types)
        batch = torch.repeat_interleave(torch.arange(len(counts)), counts)
        node_attrs = torch.zeros((n_nodes, len(self.atom_types)),
                                 dtype=torch.float)
        node_attrs[torch.arange(n_nodes), types] = 1.0
        edge_index = _radius_graph(
            positions, r=self.cutoff, batch=batch, loop=False,
            max_num_neighbors=self.max_num_neighbors)
        shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.float)
        ptr = torch.zeros(len(counts) + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, 0)
        result = {'positions': positions, 'edge_index': edge_index,
                  'node_attrs': node_attrs, 'shifts': shifts,
                  'batch': batch, 'ptr': ptr}
        return {key: value.to(device) for key, value in result.items()}

    def row_from_graph(self, graph):
        """The node-table row of a graph built by `get_graphs_pyg`.

        The inverse of `graphs`: ``graphs([row_from_graph(g)])[0]`` equals
        ``g`` for a graph built with the same selections, `atom_types` and
        cutoff.

        Parameters
        ----------
        graph : torch_geometric.data.Data or mapping
            With float32 ``positions`` ``(n, 3)``, one-hot ``node_attrs``
            ``(n, len(atom_types))`` and optionally ``shifts`` (all zero).

        Returns
        -------
        numpy.ndarray
            float32, shape ``(width,)``.

        Raises
        ------
        NodeTableOverflowError
            If the graph has more than `n_max` nodes.
        ValueError
            If it is not such a graph (other atom-type table, non-zero shifts,
            float64 positions, no nodes).
        """
        positions = _to_numpy(graph['positions'])
        node_attrs = _to_numpy(graph['node_attrs'])
        n_nodes = len(positions)
        if positions.dtype != np.float32 or positions.shape != (n_nodes, 3):
            raise ValueError(f'graph positions must be float32 of shape (n, '
                             f'3), got {positions.dtype} {positions.shape}')
        if not n_nodes:
            raise ValueError('the graph has no nodes')
        if n_nodes > self.n_max:
            raise NodeTableOverflowError(
                f'the graph has {n_nodes} nodes, more than n_max={self.n_max}')
        if (node_attrs.shape != (n_nodes, len(self.atom_types)) or
                not np.all((node_attrs == 0) | (node_attrs == 1)) or
                not np.all(node_attrs.sum(axis=1) == 1)):
            raise ValueError(
                f'graph node_attrs must be a one-hot encoding over the '
                f'{len(self.atom_types)} atom_types, got shape '
                f'{node_attrs.shape}')
        shifts = graph['shifts'] if 'shifts' in _keys(graph) else None
        if shifts is not None and _to_numpy(shifts).any():
            raise ValueError('the graph has non-zero shifts')
        row = np.zeros(self.width, dtype=np.float32)
        row[0] = n_nodes
        row[1] = NODE_TABLE_LAYOUT
        start = self._positions_start
        row[start:start + 3 * n_nodes] = positions.ravel()
        start = self._types_start
        row[start:start + n_nodes] = node_attrs.argmax(axis=1)
        return row


def _keys(graph):
    """Keys of a mapping or a torch_geometric Data object."""
    keys = graph.keys
    return keys() if callable(keys) else keys


def repack_rows(rows, n_max):
    """Node-table rows rewritten for another row capacity.

    The positions and atom types of each row are copied into a layout of
    capacity `n_max`; zero rows stay zero. The result equals featurizing the
    same frames with ``featurizer.with_n_max(n_max)``, bit for bit, for every
    row that is not zero. No trajectory is read.

    Parameters
    ----------
    rows : numpy.ndarray
        float32 node-table rows, shape ``(n_frames, 2 + 4 * n)`` for any
        capacity ``n``.
    n_max : int
        Capacity of the new rows.

    Returns
    -------
    numpy.ndarray
        float32, shape ``(n_frames, 2 + 4 * n_max)``.

    Raises
    ------
    NodeTableOverflowError
        If a row has more than `n_max` nodes (narrowing).
    ValueError
        If `rows` are not float32 node-table rows of layout
        `NODE_TABLE_LAYOUT`, or `n_max` is not a positive integer.
    """
    if int(n_max) != n_max or n_max < 1:
        raise ValueError(f'n_max must be a positive integer, got {n_max!r}')
    n_max = int(n_max)
    rows = np.asarray(rows)
    if rows.ndim != 2:
        raise ValueError(f'node-table rows must be 2-d, got shape '
                         f'{rows.shape}')
    if rows.dtype != np.float32:
        raise ValueError(f'node-table rows must be float32, got {rows.dtype}')
    width = rows.shape[1]
    if width < 6 or (width - 2) % 4:
        raise ValueError(f'a node-table row has width 2 + 4 * n_max, got '
                         f'width {width}')
    old_n_max = (width - 2) // 4
    n_nodes = rows[:, 0]
    computed = n_nodes != 0
    bad = np.flatnonzero(computed & ((rows[:, 1] != NODE_TABLE_LAYOUT) |
                                     (n_nodes != np.round(n_nodes)) |
                                     (n_nodes < 0) | (n_nodes > old_n_max)))
    if len(bad):
        raise ValueError(f'row(s) {bad[:_MAX_REPORTED_FRAMES].tolist()} are '
                         f'not layout-{NODE_TABLE_LAYOUT} node-table rows of '
                         f'width {width}')
    if len(rows) and n_nodes.max() > n_max:
        raise NodeTableOverflowError(
            f'{int((n_nodes > n_max).sum())} row(s) have more than '
            f'n_max={n_max} nodes (largest {int(n_nodes.max())})')
    keep = min(old_n_max, n_max)
    result = np.zeros((len(rows), 2 + 4 * n_max), dtype=np.float32)
    result[:, :2 + 3 * keep] = rows[:, :2 + 3 * keep]
    result[:, 2 + 3 * n_max:2 + 3 * n_max + keep] = \
        rows[:, 2 + 3 * old_n_max:2 + 3 * old_n_max + keep]
    return result


class MultiSystemNodeTableFeaturizer:
    """One `NodeTableFeaturizer` per system, dispatched on ``system_id``.

    For multi-system runs (``multi_system=True``), where every data function
    receives a ``system_id`` keyword. Each system has its own featurizer (its
    topology, selections and `n_max`, hence its own row width; the series
    files are per trajectory, so per system). All share one `atom_types`
    table, since they feed one network.

    Parameters
    ----------
    featurizers : mapping
        ``{system_id: NodeTableFeaturizer}``, keyed by ``params.system_ids``.
        Keys are compared as strings.

    Raises
    ------
    ValueError
        If `featurizers` is empty or the featurizers' `atom_types` differ.

    Notes
    -----
    `series` is ``'descriptors-gn'`` and 10 hex characters of a SHA-256 over
    every system's fingerprint: one name for the campaign, which changes when
    any system's featurizer changes. As for `NodeTableFeaturizer`, use the
    methods through module-level wrappers in the params file::

        def descriptors_function(trajectory, system_id):
            return FEATURIZERS.descriptors_function(trajectory, system_id)

    A multi-system fit transforms every system separately and reassembles
    the graphs in batch order, so its ``descriptor_transform`` must return
    graphs (`graphs`), not a batch dict; value passes run per path, hence per
    system, and can use `batch_dict`.
    """

    _params_requires_wrapper = True

    def __init__(self, featurizers):
        self.featurizers = {str(system_id): featurizer
                            for system_id, featurizer in dict(featurizers).items()}
        if not self.featurizers:
            raise ValueError('no featurizers given')
        tables = {tuple(featurizer.atom_types)
                  for featurizer in self.featurizers.values()}
        if len(tables) > 1:
            raise ValueError(
                f'the featurizers of all systems must share one atom_types '
                f'table (the network input), got {sorted(tables)}')
        self._fingerprint = hashlib.sha256(json.dumps(
            {system_id: featurizer.fingerprint
             for system_id, featurizer in self.featurizers.items()},
            sort_keys=True).encode()).hexdigest()

    def __repr__(self):
        return (f'{type(self).__name__}(series={self.series!r}, '
                f'system_ids={self.system_ids})')

    def __getitem__(self, system_id):
        try:
            return self.featurizers[str(system_id)]
        except KeyError:
            raise KeyError(f'no node-table featurizer for system_id '
                           f'{system_id!r}; known: {self.system_ids}') from None

    @property
    def system_ids(self):
        """list of str: the systems, in the order given."""
        return list(self.featurizers)

    @property
    def atom_types(self):
        """list of str: the shared atom-type table."""
        return list(next(iter(self.featurizers.values())).atom_types)

    def spec(self):
        """``{system_id: spec}`` of every system (see
        `NodeTableFeaturizer.spec`)."""
        return {system_id: featurizer.spec()
                for system_id, featurizer in self.featurizers.items()}

    @property
    def fingerprint(self):
        """str: SHA-256 (hex) over every system's fingerprint."""
        return self._fingerprint

    @property
    def series(self):
        """str: the campaign's descriptor series name."""
        return SERIES_PREFIX + self._fingerprint[:10]

    def check_series(self, descriptors_series):
        """Check a pinned series name; see `NodeTableFeaturizer.check_series`.

        Raises
        ------
        ValueError
            If `descriptors_series` is not `series`.
        """
        if descriptors_series != self.series:
            raise ValueError(
                f'descriptors_series {descriptors_series!r} does not match '
                f'these node-table featurizers, whose series is '
                f'{self.series!r} (series of the systems: '
                f'{ {sid: f.series for sid, f in self.featurizers.items()} }). '
                f'If you changed a featurizer on purpose, pin '
                f'descriptors_series = {self.series!r}; every frame of every '
                f'system is then featurized again. Otherwise find what '
                f'differs, e.g. an edited selection or atom types that '
                f'MDAnalysis guessed differently on this host.')
        return descriptors_series

    def with_n_max(self, n_max):
        """The featurizers of every system with row capacity `n_max`; see
        `NodeTableFeaturizer.with_n_max`."""
        return type(self)({system_id: featurizer.with_n_max(n_max)
                           for system_id, featurizer in self.featurizers.items()})

    def descriptors_function(self, trajectory, system_id):
        """`NodeTableFeaturizer.descriptors_function` of system
        `system_id`."""
        return self[system_id].descriptors_function(trajectory)

    def rows_from_coordinates(self, coordinates, system_id):
        """`NodeTableFeaturizer.rows_from_coordinates` of system
        `system_id`."""
        return self[system_id].rows_from_coordinates(coordinates)

    def graphs(self, rows, system_id):
        """`NodeTableFeaturizer.graphs` of system `system_id`."""
        return self[system_id].graphs(rows)

    def batch_dict(self, rows, system_id, device='cpu'):
        """`NodeTableFeaturizer.batch_dict` of system `system_id`."""
        return self[system_id].batch_dict(rows, device=device)

    def row_from_graph(self, graph, system_id):
        """`NodeTableFeaturizer.row_from_graph` of system `system_id`."""
        return self[system_id].row_from_graph(graph)
