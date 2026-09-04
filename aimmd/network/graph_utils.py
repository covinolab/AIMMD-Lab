""" Contributes functions used for AIMMD with GNNs, including graph generation and caching."""

# Test if modules are available. If not, this module cannot be used.
try:
    import mlcolvar
    import torch_geometric
    import mdtraj as md
    import numpy as np
    import torch
    from tqdm import tqdm
    from torch_geometric.data import Batch, Data
    import MDAnalysis as mda
    import aimmd
    import hashlib
    import pickle
    import gzip
    import lz4
    import sqlite3
    from mlcolvar.data.dataset import DictDataset
    from mlcolvar.data.graph.utils import create_dataset_from_configurations
    try:
        # old name (older mlcolvar versions)
        from mlcolvar.utils.io import (
            _configures_from_trajectory,
            _z_table_from_top,
            _names_from_top,
        )
    except ImportError:
        # new name (newer mlcolvar versions)
        from mlcolvar.utils.io import (
            _configurations_from_trajectory as _configures_from_trajectory,
            _z_table_from_top,
            _names_from_top,
        )
    from torch_geometric.nn import radius_graph
    import os
    import random
    import time
    import MDAnalysis.transformations as transformations
    from . import shm_cache
except ImportError as e:
    raise ImportError(f"Module {e.name} not found. "
                      f"The module 'aimmd.network.graph_utils'"
                      f"requires additional dependencies.") from e

#: How long a writer keeps retrying a locked graph cache before giving up.
#: Overridable with ``AIMMD_STORE_RETRY_SECONDS``.
_STORE_RETRY_SECONDS = float(os.environ.get('AIMMD_STORE_RETRY_SECONDS', 300.0))

#: Per-attempt SQLite busy timeout, set explicitly in :func:`init_db` rather
#: than inheriting Python's 5 s default. Overridable with
#: ``AIMMD_SQLITE_BUSY_SECONDS``. Must stay well under _STORE_RETRY_SECONDS, or
#: a single attempt consumes the whole budget.
_SQLITE_BUSY_SECONDS = float(os.environ.get('AIMMD_SQLITE_BUSY_SECONDS', 30.0))

#: How often to report that a graph-cache write is still blocked. Without this a
#: contended write is silent for the whole retry budget, which in production was
#: indistinguishable from a hang. Overridable with ``AIMMD_STORE_REPORT_EVERY``.
_STORE_REPORT_EVERY = float(os.environ.get('AIMMD_STORE_REPORT_EVERY', 30.0))


def atom_coordinate_descriptors_function(
        trajectory: mda.coordinates.timestep.Timestep,
        verbose: bool = False,
        atom_indices: np.ndarray | None = None,
) -> np.ndarray:
    """From an MDAnalysis trajectory to descriptors, which will be cached by pathensemble.
    Here, we use atomic positions as descriptors, which are then further processed to graphs.

    Parameters
    ----------
    trajectory : mda.coordinates.timestep.Timestep
        The trajectory to be converted to descriptors.
    verbose : bool, optional
        Whether to show a progress bar, by default False.
    atom_indices : np.ndarray, optional
        Integer indices of atoms whose positions to store. If None, all atom positions
        are stored. Pass heavy-atom indices (e.g. pre-computed via
        ``universe.select_atoms("not type H").indices``) to reduce memory usage by
        roughly 2/3 for typical organic systems.

    Returns
    -------
    np.ndarray
        Array of shape (n_frames, n_selected_atoms * 3) with the descriptors.
        When ``atom_indices`` is None the shape is (n_frames, n_atoms * 3).
    """

    result = []
    for frame in tqdm(trajectory, disable=not verbose, position=0):
        if atom_indices is not None:
            result.append(frame.positions[atom_indices].ravel().copy())
        else:
            result.append(frame.positions.ravel().copy())
    if not len(result):
        result = np.zeros((1, 0))
    return np.array(result)

def init_db(db_path: str = "graphs_cache.sqlite") -> sqlite3.Connection:
    """Initialize (or open) the SQLite database for graph caching.

    Hardened against the campaign-start thundering herd. At job start every one
    of the ~36 processes opens all 5 caches at once (Params exec calls this once
    per system), and a continuation inherits multi-GB caches whose large WAL --
    left un-checkpointed by the SIGKILL that ends a walltime-limited job -- must
    be recovered by the first opener under an exclusive lock that can exceed the
    busy timeout. The previous version took the WAL writer lock unconditionally
    (via `CREATE TABLE IF NOT EXISTS`, even when the table already existed) and
    had no retry, so the losers of that race raised `database is locked` straight
    out through `Params.load`, uncaught, killing the whole job in minutes.

    Three defences:
      - a read-only existence check (`sqlite_master`) before any DDL, so on a
        continuation -- where the table always exists -- almost every opener
        takes no writer lock at all;
      - an explicit ``PRAGMA busy_timeout`` set first, so it covers recovery and
        every statement below (raised from Python's 5 s default);
      - a bounded retry-with-backoff around the whole open, so a transient lock
        during the herd is survived rather than fatal.
    """
    busy_ms = int(_SQLITE_BUSY_SECONDS * 1000)
    deadline = time.monotonic() + _STORE_RETRY_SECONDS
    attempt = 0
    while True:
        conn = None
        try:
            # `CacheConnection` is a sqlite3.Connection subclass, so this is a
            # drop-in: isinstance() still holds and every caller is unaffected.
            # It exists only so per-connection state (a /dev/shm replica, a blob
            # memo) can be attached -- a plain sqlite3.Connection has no
            # __dict__.  See aimmd.network.shm_cache.
            conn = sqlite3.connect(db_path, timeout=_SQLITE_BUSY_SECONDS,
                                   factory=shm_cache.CacheConnection)
            # Set the busy timeout first and explicitly, so it also covers WAL
            # recovery triggered by the statements below.
            conn.execute(f"PRAGMA busy_timeout={busy_ms}")
            # Read-only existence check: a WAL reader, never the writer lock. On
            # a continuation the table already exists, so this is the whole cost.
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='graphs_cache'").fetchone()
            if exists is None:
                conn.execute("CREATE TABLE IF NOT EXISTS graphs_cache"
                             "(key TEXT PRIMARY KEY, data BLOB)")
                conn.commit()
            # Setting journal_mode=WAL on an already-WAL db is a no-op; only pay
            # the mode-change (which needs an exclusive moment) when necessary.
            if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA wal_autocheckpoint=1000;")
            conn.commit()
            # `Params.load` chdir's into the params folder before exec'ing it, so
            # a relative db_path resolves correctly here.
            conn._aimmd_db_path = os.path.abspath(db_path)
            shm_cache.register(conn)
            return conn
        except sqlite3.OperationalError as exc:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            message = str(exc).lower()
            if ('locked' in message or 'busy' in message) \
                    and time.monotonic() < deadline:
                attempt += 1
                time.sleep(min(2.0, 0.05 * 2 ** min(attempt, 6))
                           * (0.5 + random.random()))
                continue
            raise


def _encode(data: torch_geometric.data.Data, compression_lib: str) -> bytes:
    """Serialize and compress one graph. The single write-side codec."""
    raw = pickle.dumps(data)
    if compression_lib == "gzip":
        return gzip.compress(raw)
    elif compression_lib == "lz4":
        return lz4.frame.compress(raw)
    elif compression_lib == "none":
        return raw
    raise ValueError(f"Unknown compression library: {compression_lib}")


def _decode(blob: bytes, compression_lib: str = None) -> torch_geometric.data.Data:
    """Decompress and deserialize one graph, detecting the codec from the bytes.

    Sniffing rather than trusting `compression_lib` closes a real trap: this
    module's read/write defaults are "gzip" while `process_descriptors_pyg`
    passes "lz4", so any caller that forgets the argument writes gzip and then
    fails to read it back. The three container magics are mutually exclusive --
    gzip 1f 8b, lz4 frame 04 22 4d 18, pickle protocol >=2 starts 0x80 -- so the
    encoding is unambiguous, and a cache holding a mixture (a campaign whose
    setting changed part way) becomes readable instead of broken.
    """
    if blob[:2] == b'\x1f\x8b':
        return pickle.loads(gzip.decompress(blob))
    if blob[:4] == b'\x04\x22\x4d\x18':
        return pickle.loads(lz4.frame.decompress(blob))
    return pickle.loads(blob)


def load_from_sqlite(key: str, conn: sqlite3.Connection, compression_lib = "gzip") -> torch_geometric.data.Data | None:
    """ Load a graph from SQLite cache.

    Consults, in order, this connection's in-process blob memo, its /dev/shm
    read-replica if one has been staged, and finally the database itself. The
    fallback lives here, inside the single read choke point, so that a miss
    against a stale replica can never escape as None and make the caller
    recompute a graph that does exist -- see aimmd.network.shm_cache.

    Parameters
    ----------
    key : str
        The key of the graph to be loaded.
    conn : sqlite3.Connection
        The SQLite connection.
    compression_lib : str, optional
        The compression library to use, by default "gzip". Only used when
        writing; reads detect the codec from the stored bytes.
        Supported: "gzip", "lz4", "none".

    Returns
    -------
    torch_geometric.data.Data | None
        The loaded graph, or None if not found.
    """
    memo = getattr(conn, '_aimmd_memo', None)
    if memo is not None:
        blob = memo.get(key)
        if blob is not None:
            shm_cache._STATS['memo_hits'] += 1
            return _decode(blob)

    replica = getattr(conn, '_aimmd_replica', None)
    if replica is None and getattr(conn, '_aimmd_stage_pending', False):
        shm_cache.stage_cache(conn)                  # stage on first use
        replica = getattr(conn, '_aimmd_replica', None)
    if replica is not None:
        try:
            row = replica.execute(
                "SELECT data FROM graphs_cache WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            shm_cache.detach(conn, reason=str(exc))
            row = None
        if row is not None:
            shm_cache._STATS['hits'] += 1
            if memo is not None:
                memo.put(key, row[0])
            return _decode(row[0])
        shm_cache._STATS['misses'] += 1

    cursor = conn.execute("SELECT data FROM graphs_cache WHERE key = ?", (key,))
    row = cursor.fetchone()
    if row is None:
        return None
    if memo is not None:
        memo.put(key, row[0])
    return _decode(row[0])


def _store_blobs(conn, keys, blobs) -> bool:
    """Write encoded graphs, retrying while the database is locked.

    Returns True if they were written, False if the deadline passed first.

    **A lock must never be fatal.** The graph cache is a cache: callers already
    hold the graphs in memory (`process_descriptors_pyg` assembles its result
    from `new_graphs`, `load_or_create` returns the graph it just built) and
    write only so a later process can skip the recompute. Raising propagated out
    of `descriptors_function` into `Path.compute` -> `trajectory.extend` ->
    `execute_command.stop_condition`, killed `gmx mdrun`, and cancelled every
    task in the job -- observed in 19 production jobs, the oldest predating the
    /dev/shm cache work by months. Losing a cache entry costs one recompute.

    Backoff is exponential with jitter so that dozens of concurrent writers do
    not resynchronise onto a single retry cadence.
    """
    t0 = time.monotonic()
    deadline = t0 + _STORE_RETRY_SECONDS
    next_report = _STORE_REPORT_EVERY
    attempt = 0
    while True:
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO graphs_cache (key, data) VALUES (?, ?)",
                zip(keys, blobs))
            conn.commit()
            if attempt:
                print(f'graph cache: wrote {len(keys)} graph(s) after '
                      f'{time.monotonic() - t0:.0f}s of contention '
                      f'({attempt} retries)', flush=True)
            return True
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if 'locked' not in message and 'busy' not in message:
                raise
            # Roll back first, or the partial transaction survives and the next
            # executemany compounds it.
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            if time.monotonic() >= deadline:
                print(f'graph cache: gave up writing {len(keys)} graph(s) '
                      f'after {_STORE_RETRY_SECONDS:.0f}s of lock contention; '
                      f'they will be recomputed when next needed', flush=True)
                return False
            attempt += 1
            waited = time.monotonic() - t0
            if waited >= next_report:
                # Say so while it is happening, not only when it gives up.
                print(f'graph cache: write of {len(keys)} graph(s) blocked '
                      f'{waited:.0f}s by other writers ({attempt} attempts), '
                      f'still retrying up to {_STORE_RETRY_SECONDS:.0f}s',
                      flush=True)
                next_report = waited + _STORE_REPORT_EVERY
            time.sleep(min(5.0, 0.05 * 2 ** min(attempt, 7))
                       * (0.5 + random.random()))


def store_in_sqlite(key: str, data: torch_geometric.data.Data, conn: sqlite3.Connection, compression_lib = "gzip"):
    """ Store a graph in SQLite cache.

    Parameters
    ----------
    key : str
        The key of the graph to be stored.
    data : torch_geometric.data.Data
        The graph to be stored.
    conn : sqlite3.Connection
        The SQLite connection.
    compression_lib : str, optional
        The compression library to use, by default "gzip".
        Supported: "gzip", "lz4", "none".
    """
    compressed_data = _encode(data, compression_lib)
    if shm_cache.reader_role():
        _after_store(conn, [key], [compressed_data])
        return
    if _store_blobs(conn, [key], [compressed_data]):
        _after_store(conn, [key], [compressed_data])


def store_many_in_sqlite(keys: list[str], graphs: list[torch_geometric.data.Data],
                         conn: sqlite3.Connection, compression_lib = "lz4"):
    """ Store many graphs in ONE transaction.

    Same contract as `store_in_sqlite`, but a whole batch per commit. A worker
    featurizes an entire trajectory chunk at once, so committing per graph turns
    one transaction into hundreds, each appending to the WAL and triggering
    checkpoints that scatter writes across the whole file -- which is what erodes
    cache residency for every other process sharing it.

    Parameters
    ----------
    keys : list[str]
        Keys of the graphs, in the same order as `graphs`.
    graphs : list[torch_geometric.data.Data]
        The graphs to be stored.
    conn : sqlite3.Connection
        The SQLite connection.
    compression_lib : str, optional
        The compression library to use, by default "lz4" (what the pyg path
        uses). Threaded through the same `_encode` helper as `store_in_sqlite`,
        so a batch written here is always readable by `load_from_sqlite`.
    """
    if not keys:
        return
    blobs = [_encode(g, compression_lib) for g in graphs]
    if shm_cache.reader_role():
        # The trainer keeps what it computes local (memo + its own tmpfs
        # replica) rather than contending for the shared write lock; see
        # shm_cache.set_reader_role.
        _after_store(conn, keys, blobs)
        return
    if _store_blobs(conn, keys, blobs):
        _after_store(conn, keys, blobs)


def _after_store(conn, keys, blobs):
    """Mirror freshly written graphs into the memo and the /dev/shm replica.

    Without this the trainer's own new graphs miss the replica on every
    subsequent lookup -- and `fit` re-draws frames across thousands of epochs, so
    each one would be paid for again and again against the real database.
    Best-effort: a failure here only costs speed.
    """
    memo = getattr(conn, '_aimmd_memo', None)
    if memo is not None:
        for key, blob in zip(keys, blobs):
            memo.put(key, blob)
    shm_cache.write_through(conn, keys, blobs)


def get_stable_hash(config: mlcolvar.data.graph.atomic.Configurations) -> str:
    """ Get a stable hash for a configuration.
    
    Parameters
    ----------
    config : mlcolvar.data.graph.atomic.Configurations
        The configuration to be hashed.
    
    Returns
    -------
    str
        The stable hash of the configuration.
    """
    encoded = pickle.dumps(config)
    stable_hash = hashlib.sha256(encoded).hexdigest()
    return stable_hash

def create_graph(config: mlcolvar.data.graph.atomic.Configurations, z_table: mlcolvar.data.graph.atomic.AtomicNumberTable, atomnames: list, cutoff: float) -> torch_geometric.data.Data:
    """ Create a graph from a configuration.
    
    Parameters
    ----------
    config : mlcolvar.data.graph.atomic.Configurations
        The configuration to be converted to a graph.
    z_table : dict
        The atomic number table.
    atomnames : list of str
        The list of atom names.
    cutoff : float
        The cutoff distance for edge creation.
    
    Returns
    -------
    torch_geometric.data.Data
        The graph representation of the configuration.
    """
    graph = create_dataset_from_configurations(
        config=[config], z_table=z_table, cutoff=cutoff, buffer=0,
        atom_names=atomnames, remove_isolated_nodes=True, show_progress=False
    )
    return graph['data_list'][0]

def load_or_create(conn: sqlite3.Connection, config: mlcolvar.data.graph.atomic.Configurations, z_table: mlcolvar.data.graph.atomic.AtomicNumberTable, atomnames: list, cutoff: float) -> torch_geometric.data.Data:
    """ Load a graph from SQLite cache, or create it if not present.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        The SQLite connection.
    config : mlcolvar.data.graph.atomic.Configurations
        The configuration to be converted to a graph.
    z_table : dict
        The atomic number table.
    atomnames : list of str
        The list of atom names.
    cutoff : float
        The cutoff distance for edge creation.

    Returns
    -------
    torch_geometric.data.Data
        The graph representation of the configuration.
    """
    stable_hash = get_stable_hash(config)

    graph = load_from_sqlite(stable_hash, conn)
    if graph is not None:
        return graph
    else:
        graph = create_graph(config, z_table, atomnames, cutoff)
        store_in_sqlite(stable_hash, graph, conn)
        return graph

def process_descriptors(descriptors: np.ndarray, mdtraj_frame: md.Trajectory, system_selection: str, environment_selection: str, cutoff: float, conn: sqlite3.Connection, verbose: bool = False) -> DictDataset:
    """ Transform the descriptors to network input.
    Here, we transform the atomic positions to a graph embedding using mlcolvar.

    Parameters
    ----------
    descriptors : np.ndarray
        The input descriptors to be transformed. Shape (n_frames, n_atoms * 3).
    mdtraj_frame : md.Trajectory
        A mdtraj frame with the topology and unit cell information.
    system_selection : str
        The selection string for the system atoms.
    environment_selection : str
        The selection string for the environment atoms.
    cutoff : float
        The cutoff distance for edge creation.
    conn : sqlite3.Connection
        The SQLite connection for caching.
    verbose : bool, optional
        Whether to show a progress bar, by default False.

    Returns
    -------
    mlcolvar.data.dataset.DictDataset
        The transformed descriptors suitable for network input.
    """

    if verbose:
        print(f"Processing descriptors with shape: {descriptors.shape}")

    # First, we need to transform the descriptors to an md.Trajectory object
    n_atoms = mdtraj_frame.n_atoms
    n_frames = descriptors.shape[0]
    assert descriptors.shape[1] == n_atoms * 3, \
        f"Descriptors should have shape (n_frames, {n_atoms * 3}), got {descriptors.shape}"
    xyz = descriptors.reshape((n_frames, n_atoms, 3)) / 10.0  # convert from Angstrom to nm
    # replicate unit cell info
    unit_cell_lengths = np.tile(mdtraj_frame.unitcell_lengths, (n_frames, 1))
    unit_cell_angles = np.tile(mdtraj_frame.unitcell_angles, (n_frames, 1))
    # create trajectory
    traj = md.Trajectory(xyz=xyz, topology=mdtraj_frame.topology, unitcell_lengths=unit_cell_lengths,
                        unitcell_angles=unit_cell_angles)

    # now we get mlcolvar.data.graph.atomic.Configurations from this trajectory
    configurations = _configures_from_trajectory(
        traj,
        system_selection = system_selection,
        environment_selection = environment_selection,
    )

    z_table = _z_table_from_top([traj.topology])
    atomnames = _names_from_top([traj.topology])

    # create graphs list
    graphs_list = []

    for config in tqdm(configurations, disable=not verbose, position=0):
        graph = load_or_create(conn, config, z_table, atomnames, cutoff)
        graphs_list.append(graph)

    dataset = mlcolvar.data.DictDataset(
        dictionary={
            'data_list': graphs_list
        },
        data_type = "graphs",
    )    

    return dataset


def get_graphs_pyg(
        descriptors: np.ndarray,
        mdanalysis_universe: mda.Universe,
        system_selection: str,
        environment_selection: str,
        cutoff: float,
        verbose: bool = False,
        atom_indices: np.ndarray | None = None,
        atom_types: list | None = None,
    ) -> list[torch_geometric.data.Data]:
    """ Process descriptors into torch_geometric graphs using MDAnalysis for pbc handling.

    Parameters
    ----------
    descriptors : np.ndarray
        Array of shape (n_frames, n_atoms * 3) with atomic coordinates.
        When ``atom_indices`` is provided, ``n_atoms`` is ``len(atom_indices)``
        rather than the total atom count of ``mdanalysis_universe``.
    mdanalysis_universe : mda.Universe
        MDAnalysis Universe object with the full system topology. Always used
        for PBC handling and graph construction, regardless of ``atom_indices``.
    system_selection : str
        MDAnalysis selection string for the system of interest.
    environment_selection : str
        MDAnalysis selection string for the environment.
    cutoff : float
        Cutoff distance for graph construction (in Angstrom).
    verbose : bool, optional
        Whether to print progress, by default False.
    atom_indices : np.ndarray, optional
        Integer indices (into ``mdanalysis_universe``) of the atoms whose
        positions are stored in ``descriptors``. When provided, only those
        atoms' positions are updated per frame; all other atoms retain their
        positions from the universe's current state (typically the GRO file).
        Pass heavy-atom indices to avoid touching hydrogen positions while
        still using the full topology for PBC transformations and spatial
        selections — this guarantees identical graphs to the full-atom flow
        because (a) the same universe is used, (b) H atoms are filtered by
        the selection strings, and (c) their stale positions never enter any
        distance calculation that affects the final graph.
    atom_types : list, optional
        Fixed, ordered list of MDAnalysis atom-type strings defining the
        one-hot ``node_attrs`` columns. When ``None`` (default) the columns are
        derived per-universe from ``sorted(set(universe.atoms.types))`` — the
        legacy, single-system behaviour. Pass an explicit list to obtain a
        **fixed, shared encoding across multiple systems** (multi-ligand runs):
        every system featurizes into the same columns, unused columns stay
        zero, and the network's input width must equal ``len(atom_types)``. Any
        atom type present in the selection but absent from ``atom_types`` raises
        a clear ``ValueError``.

    Returns
    -------
    list[torch_geometric.data.Data]
        List of torch_geometric Data objects representing the graphs.
    """
    coordinate_array_reshaped = descriptors.reshape(descriptors.shape[0], -1, 3)

    system = mdanalysis_universe.select_atoms(system_selection)
    surroundings = mdanalysis_universe.select_atoms(environment_selection)
    system_and_surroundings = system + surroundings

    # Atom-type columns for the one-hot node encoding. Default: derive per
    # universe (legacy single-system behaviour). Multi-system runs pass a fixed,
    # shared ``atom_types`` so all systems encode into identical columns.
    if atom_types is None:
        atom_types = list(sorted(set(mdanalysis_universe.atoms.types)))
    else:
        atom_types = list(atom_types)
        present = set(system_and_surroundings.atoms.types)
        unknown = present.difference(atom_types)
        if unknown:
            raise ValueError(
                f"atom type(s) {sorted(unknown)} are present in the selected "
                f"atoms but missing from the fixed atom_types table "
                f"{atom_types}. Extend atom_types to cover every element in "
                f"every system.")

    data_list = []
    for frame in tqdm(coordinate_array_reshaped, disable=not verbose):
        # take care of pbcs, placing the system in the center of the box and restoring environment around it
        ts = mdanalysis_universe.trajectory[0]
        if atom_indices is not None:
            mdanalysis_universe.atoms[atom_indices].positions = frame
        else:
            mdanalysis_universe.atoms.positions = frame
        ts = transformations.unwrap(system)(ts)
        ts = transformations.center_in_box(system, wrap=False)(ts)
        ts = transformations.wrap(mdanalysis_universe.atoms)(ts)

        surroundings = mdanalysis_universe.select_atoms(environment_selection)
        system_and_surroundings = system + surroundings
        positions = system_and_surroundings.positions.copy()

        # Get node attributes: one-hot encoding of atom types
        node_attr = np.zeros((system_and_surroundings.n_atoms, len(atom_types)), dtype=np.float32)
        for i, atom in enumerate(system_and_surroundings.atoms.types):
            node_attr[i, atom_types.index(atom)] = 1.0

        edge_index = radius_graph(
            x=torch.tensor(positions, dtype=torch.float),
            r=cutoff,
            loop=False,
            max_num_neighbors=128,
        )

        # shifts are empty, as GNN is equivariant and positions are given
        shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.float)
        data = Data(
                positions=torch.tensor(positions, dtype=torch.float),
                edge_index=edge_index, node_attrs=torch.tensor(node_attr, dtype=torch.float),
                shifts=shifts
            )
        data_list.append(data)

    return data_list


def process_descriptors_pyg(
        descriptors: np.ndarray,
        mdanalysis_universe: mda.Universe,
        system_selection: str,
        environment_selection: str,
        cutoff: float,
        conn: sqlite3.Connection,
        verbose: bool = False,
        compression_lib: str = "lz4",
        atom_indices: np.ndarray | None = None,
        atom_types: list | None = None,
    ) -> list[torch_geometric.data.Data]:
    """ Transform the descriptors to network input using MDAnalysis and torch_geometric.
    Here, we transform the atomic positions to a graph embedding using MDAnalysis for pbc
    handling and torch_geometric with torch_cluster for graph construction.

    Parameters
    ----------
    descriptors : np.ndarray
        The input descriptors (atomic positions) to be transformed.
        When ``atom_indices`` is provided the shape must be
        (n_frames, len(atom_indices) * 3) rather than (n_frames, n_all_atoms * 3).
    mdanalysis_universe : mda.Universe
        The MDAnalysis universe object for handling periodic boundary conditions
        and graph construction. Always used, regardless of ``atom_indices``.
    system_selection : str
        The selection string for the system atoms.
    environment_selection : str
        The selection string for the environment atoms.
    cutoff : float
        The cutoff distance for constructing the graph.
    conn : sqlite3.Connection
        The SQLite connection for storing/loading graphs.
    verbose : bool, optional
        Whether to print verbose output (default is False).
    compression_lib : str, optional
        The compression library to use for storing graphs (default is "lz4").
        Supported options: "gzip", "lz4", "none".
    atom_indices : np.ndarray, optional
        Integer indices (into ``mdanalysis_universe``) of the atoms whose
        positions are stored in ``descriptors``. When provided, only those
        atoms' positions are updated per frame during graph construction.
        Pass heavy-atom indices to work with heavy-atom-only descriptors
        while still using the full topology for PBC and selections, guaranteeing
        graphs identical to the full-atom workflow.
    atom_types : list, optional
        Fixed, ordered atom-type table for the one-hot ``node_attrs`` columns,
        forwarded to :func:`get_graphs_pyg`. ``None`` (default) keeps the legacy
        per-universe encoding; pass an explicit list for a fixed, shared
        encoding across multiple systems (multi-ligand runs). The cache key is
        coordinate-only, so keep one cache (``conn``) per ``atom_types`` table.

    Returns
    -------
    mlcolvar.data.DictDataset
        The dataset containing the processed graphs.
    """

    if verbose:
        print(f"Processing descriptors with shape: {descriptors.shape}")

    # Expected atom count is either the subset or the full universe
    n_atoms = len(atom_indices) if atom_indices is not None else len(mdanalysis_universe.atoms)
    n_frames = descriptors.shape[0]
    assert descriptors.shape[1] == n_atoms * 3, \
        f"Descriptors should have shape (n_frames, {n_atoms * 3}), got {descriptors.shape}"

    # precompute hashes
    stable_hashes = [get_stable_hash(config) for config in descriptors]
    # get indices of configurations in and not in the database
    loaded_graphs = [load_from_sqlite(stable_hash, conn, compression_lib=compression_lib) for stable_hash in stable_hashes]
    missing_indices = []
    for i, graph in enumerate(loaded_graphs):
        if graph is None:
            missing_indices.append(i)
    if verbose:
        print(f"There are {len(missing_indices)} missing graphs out of {n_frames} total.")

    if len(missing_indices) != 0:
        # process missing graphs
        start = time.time()
        descriptors_missing = np.array([descriptors[i] for i in missing_indices])
        graphs_list_new = get_graphs_pyg(
            descriptors=descriptors_missing,
            mdanalysis_universe=mdanalysis_universe,
            system_selection=system_selection,
            environment_selection=environment_selection,
            cutoff=cutoff,
            verbose=verbose,
            atom_indices=atom_indices,
            atom_types=atom_types,
        )
        end = time.time()
        if verbose:
            print(f"Created new graphs in {end - start:.2f} seconds.")

        # Store new graphs in database -- one transaction for the whole batch,
        # not one commit per graph (see store_many_in_sqlite).
        start = time.time()
        new_graphs = [graphs_list_new[i] for i in range(len(missing_indices))]

        store_many_in_sqlite(
            [stable_hashes[missing_indices[i]] for i in range(len(new_graphs))],
            new_graphs, conn, compression_lib=compression_lib)

        end = time.time()
        if verbose:
            print(f"Stored new graphs in database in {end - start:.2f} seconds.")

        # now assemble the full graphs list
        for i, graph in enumerate(new_graphs):
            loaded_graphs[missing_indices[i]] = graph

    dataset = mlcolvar.data.DictDataset(
        dictionary={
            'data_list': loaded_graphs
        },
        data_type = "graphs",
    )
    return dataset
