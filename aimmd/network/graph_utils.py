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
    import multiprocessing
    from torch_geometric.nn import radius_graph
    import time
    import MDAnalysis.transformations as transformations
except ImportError as e:
    raise ImportError(f"Module {e.name} not found. "
                      f"The module 'aimmd.network.graph_utils'"
                      f"requires additional dependencies.") from e

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
    """Initialize the SQLite database for graph caching."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS graphs_cache"
                "(key TEXT PRIMARY KEY, data BLOB)")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA wal_autocheckpoint=1000;")
    conn.commit()
    return conn  


def load_from_sqlite(key: str, conn: sqlite3.Connection, compression_lib = "gzip") -> torch_geometric.data.Data | None:
    """ Load a graph from SQLite cache.

    Parameters
    ----------
    key : str
        The key of the graph to be loaded.
    conn : sqlite3.Connection
        The SQLite connection.
    compression_lib : str, optional
        The compression library to use, by default "gzip".
        Supported: "gzip", "lz4", "none".

    Returns
    -------
    torch_geometric.data.Data | None
        The loaded graph, or None if not found.
    """
    cursor = conn.execute("SELECT data FROM graphs_cache WHERE key = ?", (key,))
    row = cursor.fetchone()
    if row is None:
        return None
    if compression_lib == "gzip":
        return pickle.loads(gzip.decompress(row[0]))
    elif compression_lib == "lz4":
        return pickle.loads(lz4.frame.decompress(row[0]))
    elif compression_lib == "none":
        return pickle.loads(row[0])
    return None

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
    msgpack_bytes = pickle.dumps(data)
    if compression_lib == "gzip":
        compressed_data = gzip.compress(msgpack_bytes)
    elif compression_lib == "lz4":
        compressed_data = lz4.frame.compress(msgpack_bytes)
    elif compression_lib == "none":
        compressed_data = msgpack_bytes
    else:
        raise ValueError(f"Unknown compression library: {compression_lib}")
    
    # try storing, if it fails due to database lock, retry a few times with some delay
    for _ in range(10):
        try:
            conn.execute("INSERT OR REPLACE INTO graphs_cache (key, data) VALUES (?, ?)", (key, compressed_data))
            conn.commit()
            return # success
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(0.05)
            else:
                raise

    # persistent problem
    raise RuntimeError("Failed to store graph in SQLite after multiple attempts due to persistent database lock.")
    

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

def create_graph(config: mlcolvar.data.graph.atomic.Configurations, z_table: dict, atomnames: list, cutoff: float) -> torch_geometric.data.Data:
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

def load_or_create(conn: sqlite3.Connection, config: mlcolvar.data.graph.atomic.Configurations, z_table: dict, atomnames: list, cutoff: float) -> torch_geometric.data.Data:
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

# Suppress noisy SystemExit traces from multiprocessing + GSD cleanup
def quiet_multiprocessing_cleanup():
    """ Suppress noisy SystemExit traces from multiprocessing + GSD cleanup. """
    try:
        orig_exit_func = multiprocessing.util._exit_function
        def silent_exit_function(*args, **kwargs):
            try:
                orig_exit_func(*args, **kwargs)
            except SystemExit:
                pass
            except Exception:
                pass
        multiprocessing.util._exit_function = silent_exit_function
    except Exception:
        pass

def create_compressed_graphs(configs: list[mlcolvar.data.graph.atomic.Configurations], z_table: dict, atomnames: list, cutoff: float) -> torch_geometric.data.Data:
    """ Create a list of graphs from a list of configurations. Also compress the graphs for storage using gzip.
    
    Parameters
    ----------
    configs : list[mlcolvar.data.graph.atomic.Configurations]
        The configurations to be converted to graphs.
    z_table : dict
        The atomic number table.
    atomnames : list of str
        The list of atom names.
    cutoff : float
        The cutoff distance for edge creation.
    
    Returns
    -------
    list of tuple(torch_geometric.data.Data, bytes)
        The list of tuples containing the graph and its compressed representation.
    """
    graphs = create_dataset_from_configurations(
        config=configs, z_table=z_table, cutoff=cutoff, buffer=0,
        atom_names=atomnames, remove_isolated_nodes=True, show_progress=False
    )
    return [(graph, gzip.compress(pickle.dumps(graph))) for graph in graphs['data_list']]

def store_compressed_list_in_sqlite(keys: list[str], compressed_data: list[bytes], conn: sqlite3.Connection):
    """ Store a list of compressed graphs in SQLite cache.
    Parameters
     ----------
    keys : list[str]
        The list of keys for the graphs.
    compressed_data : list[bytes]
        The list of compressed graph data.
    conn : sqlite3.Connection
        The SQLite connection object.
    """
    for key, data in zip(keys, compressed_data):
        conn.execute("INSERT OR REPLACE INTO graphs_cache (key, data) VALUES (?, ?)", (key, data))
    conn.commit()


def process_descriptors_multiprocessing(
        descriptors: np.ndarray, 
        mdtraj_frame: md.Trajectory,
        system_selection: str,
        environment_selection: str,
        cutoff: float,
        conn: sqlite3.Connection,
        verbose: bool = False, 
        n_workers: int = 8) -> DictDataset:
    """ Transform the descriptors to network input using multiprocessing.
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
    n_workers : int, optional
        The number of workers to use for multiprocessing, by default 8.
    verbose : bool, optional
        Whether to show a progress bar, by default False.

    Returns
    -------
    mlcolvar.data.dataset.DictDataset
        The transformed descriptors suitable for network input.
    """

    quiet_multiprocessing_cleanup()

    if verbose:
        print(f"Processing descriptors with shape: {descriptors.shape} using {n_workers} workers.")

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

    # now we get mlcolvar.data.graph.atomic.Configurations from this trajectory, this is fast
    configurations = _configures_from_trajectory(
        traj,
        system_selection = system_selection,
        environment_selection = environment_selection,
    )

    z_table = _z_table_from_top([traj.topology])
    atomnames = _names_from_top([traj.topology])

    # precompute hashes
    stable_hashes = [get_stable_hash(config) for config in configurations]
    # get indices of configurations in and not in the database
    loaded_graphs = [load_from_sqlite(stable_hash, conn) for stable_hash in stable_hashes]
    missing_indices = []
    for i, graph in enumerate(loaded_graphs):
        if graph is None:
            missing_indices.append(i)
    if verbose:
        print(f"There are {len(missing_indices)} missing graphs out of {n_frames} total.")

    if len(missing_indices) != 0:
        from time import time
        start = time()
        missing_configs = [configurations[i] for i in missing_indices]
        missing_configs_split = np.array_split(missing_configs, n_workers)

        with multiprocessing.Pool(n_workers) as pool:
            graphs_list_new = pool.starmap(create_compressed_graphs, [(missing_configs_split[i], z_table, atomnames, cutoff) for i in range(n_workers)])
        # concatenate lists
        graphs_list_new = [graph for sublist in graphs_list_new for graph in sublist]
        end = time()
        if verbose:
            print(f"Created new graphs with multiprocessing in {end - start:.2f} seconds.")

        # Store new graphs in database
        start = time()
        new_graphs = [graphs_list_new[i][0] for i in range(len(missing_indices))]
        compressed_graphs = [graphs_list_new[i][1] for i in range(len(missing_indices))]
        store_compressed_list_in_sqlite(
            [stable_hashes[missing_indices[i]] for i in range(len(missing_indices))],
            compressed_graphs,
            conn
        )
        end = time()
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

        # Store new graphs in database
        start = time.time()
        new_graphs = [graphs_list_new[i] for i in range(len(missing_indices))]

        for i, graph in enumerate(new_graphs):
            graph_hash = stable_hashes[missing_indices[i]]
            store_in_sqlite(graph_hash, graph, conn, compression_lib=compression_lib)

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
