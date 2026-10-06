"""A toy protein-ligand system for the node-table tests.

A six-heavy-atom ligand (resname LIG, two hydrogens) in a 20 A cubic box,
among protein heavy atoms, waters and two sodium ions. The ligand sits on the
x boundary of the box and, in the "broken" frames, every atom is wrapped into
the box separately, so the ligand is split across the boundary as in a
GROMACS trajectory: the node tables must unwrap, center and wrap it exactly as
`get_graphs_pyg` does. Only numpy and MDAnalysis are needed.
"""
import numpy as np
from MDAnalysis import Universe, Writer

BOX = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float32)
ATOM_TYPES = ['H', 'C', 'N', 'O', 'S', 'NA']
SYSTEM_SELECTION = 'resname LIG and not type H'
ENVIRONMENT_SELECTION = 'not type H and around 5.0 (resname LIG)'
CUTOFF = 3.0

LIGAND_NAMES = ['C1', 'C2', 'N3', 'C4', 'O5', 'C6', 'H1', 'H6']
LIGAND_TYPES = ['C', 'C', 'N', 'C', 'O', 'C', 'H', 'H']
N_PROTEIN = 60          # residues of three heavy atoms
N_WATER = 60
N_IONS = 2
N_ATOMS = len(LIGAND_NAMES) + 3 * N_PROTEIN + 3 * N_WATER + N_IONS

# atom index ranges
LIGAND = slice(0, 8)
PROTEIN = slice(8, 8 + 3 * N_PROTEIN)
WATER = slice(PROTEIN.stop, PROTEIN.stop + 3 * N_WATER)
IONS = slice(WATER.stop, N_ATOMS)


def _topology():
    names, types, resnames, resindex = [], [], [], []
    names += LIGAND_NAMES
    types += LIGAND_TYPES
    resnames.append('LIG')
    resindex += [0] * len(LIGAND_NAMES)
    for i in range(N_PROTEIN):
        last = 'SG' if i % 10 == 0 else 'O'
        names += ['N', 'CA', last]
        types += ['N', 'C', last[0]]
        resnames.append('ALA')
        resindex += [len(resnames) - 1] * 3
    for _ in range(N_WATER):
        names += ['OW', 'HW1', 'HW2']
        types += ['O', 'H', 'H']
        resnames.append('SOL')
        resindex += [len(resnames) - 1] * 3
    for _ in range(N_IONS):
        names.append('NA')
        types.append('NA')
        resnames.append('NA')
        resindex.append(len(resnames) - 1)
    bonds = [(i, i + 1) for i in range(5)] + [(0, 6), (5, 7)]
    for k in range(N_WATER):
        oxygen = WATER.start + 3 * k
        bonds += [(oxygen, oxygen + 1), (oxygen, oxygen + 2)]
    return names, types, resnames, resindex, bonds


def toy_universe(positions=None, box=BOX, bonds=True):
    """The toy Universe, with box and (unless not `bonds`) bonds, at
    `positions` (default: the first toy frame)."""
    names, types, resnames, resindex, bond_list = _topology()
    universe = Universe.empty(N_ATOMS, n_residues=len(resnames),
                              atom_resindex=resindex, trajectory=True)
    universe.add_TopologyAttr('names', names)
    universe.add_TopologyAttr('types', types)
    universe.add_TopologyAttr('resnames', resnames)
    universe.add_TopologyAttr('resids', np.arange(1, len(resnames) + 1))
    if bonds:
        universe.add_TopologyAttr('bonds', bond_list)
    universe.dimensions = box
    universe.atoms.positions = (toy_frames(1)[0] if positions is None
                                else positions)
    return universe


def _ligand(rng, center):
    """Zigzag chain of six heavy atoms (bonds 1.43 A) plus two hydrogens."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    side = np.cross(direction, rng.normal(size=3))
    side /= np.linalg.norm(side)
    heavy = np.array([center + (i - 2.5) * 1.25 * direction
                      + (0.35 if i % 2 else -0.35) * side for i in range(6)])
    hydrogens = np.array([heavy[0] - 1.0 * direction,
                          heavy[5] + 1.0 * direction])
    return np.concatenate([heavy, hydrogens])


def toy_frames(n_frames, seed=0, broken=True, ligand_y=None):
    """Toy coordinates, float32 ``(n_frames, N_ATOMS, 3)``.

    The ligand is centered near x = 20 (the box boundary); `ligand_y` gives
    its y per frame (default: random in 5..15). With `broken`, every atom is
    wrapped into the box separately. The environment heavy atoms lie in
    [-2, 22)^3, partly outside the box.
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((n_frames, N_ATOMS, 3))
    for i in range(n_frames):
        y = rng.uniform(5.0, 15.0) if ligand_y is None else ligand_y[i]
        center = np.array([19.6 + rng.normal(0.0, 0.3), y,
                           rng.uniform(5.0, 15.0)])
        frames[i, LIGAND] = _ligand(rng, center)
        frames[i, PROTEIN] = rng.uniform(-2.0, 22.0, (3 * N_PROTEIN, 3))
        oxygens = rng.uniform(-2.0, 22.0, (N_WATER, 3))
        frames[i, WATER][0::3] = oxygens
        for k in (1, 2):
            offset = rng.normal(size=(N_WATER, 3))
            offset *= 0.96 / np.linalg.norm(offset, axis=1)[:, None]
            frames[i, WATER][k::3] = oxygens + offset
        frames[i, IONS] = rng.uniform(-2.0, 22.0, (N_IONS, 3))
    if broken:
        frames = np.mod(frames, BOX[:3])
    return frames.astype(np.float32)


def write_toy_gro(fname, positions=None):
    """Write the toy topology (names, residues, box) as a .gro file."""
    universe = toy_universe(positions)
    with Writer(str(fname), n_atoms=N_ATOMS) as writer:
        writer.write(universe.atoms)
    return str(fname)


def write_toy_xtc(fname, frames):
    """Write toy coordinate frames, with the toy box, as an .xtc file."""
    universe = toy_universe()
    with Writer(str(fname), n_atoms=N_ATOMS) as writer:
        for i, frame in enumerate(frames):
            universe.atoms.positions = frame
            universe.trajectory.ts.time = float(i)
            universe.dimensions = BOX
            writer.write(universe.atoms)
    return str(fname)


def memory_trajectory(frames):
    """An MDAnalysis trajectory (MemoryReader) of toy frames."""
    universe = toy_universe()
    universe.load_new(np.asarray(frames, dtype=np.float32), order='fac',
                      dimensions=BOX)
    return universe.trajectory


def reference_nodes(universe, positions, system_selection=SYSTEM_SELECTION,
                    environment_selection=ENVIRONMENT_SELECTION):
    """Node positions and type names of a frame, by the steps of
    `get_graphs_pyg` written out with MDAnalysis (no torch needed)."""
    from MDAnalysis import transformations
    system = universe.select_atoms(system_selection)
    ts = universe.trajectory[0]
    universe.atoms.positions = positions
    ts = transformations.unwrap(system)(ts)
    ts = transformations.center_in_box(system, wrap=False)(ts)
    ts = transformations.wrap(universe.atoms)(ts)
    nodes = system + universe.select_atoms(environment_selection)
    return nodes.positions.copy(), list(nodes.types)
