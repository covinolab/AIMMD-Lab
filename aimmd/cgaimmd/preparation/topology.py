"""
aimmd.cgaimmd.preparation.topology
==================================

GROMACS topology reading for the C-GAIMMD tools.

The preparation tools need more of a system than its structure file holds:
the exact masses and elements (MDAnalysis guesses them from atom names, and
reads the CHARMM carbons ``CL``/``CLP`` of NANMA as chlorine), the molecules,
the chemical bonds and the bonded interaction lists (to tell an exact
force-field symmetry from one that only preserves the bond graph).
MDAnalysis' parsers do not read a ``.top`` whose force field lives in
GROMACS' share directory, so this module reads ``[ atomtypes ]``,
``[ moleculetype ]``, ``[ atoms ]``, the bonded sections and
``[ molecules ]`` itself, with GROMACS' preprocessing (``#include``,
``#ifdef``/``#ifndef``/``#else``/``#endif``, ``#define``/``#undef``,
backslash continuations).

It also writes derived topologies: :func:`restrained_topology` adds the
position restraints of restrained energy minimization and equilibration to
a copy of a topology.

The force-field geometry of the generator atoms, for the topology edges of a
flow-matching network, is read from the bonded parameters, never from
coordinates: :func:`reference_geometry` gives the bond lengths (``b0``,
constraint lengths, SETTLE ``doh``) and the reference distances of every
pair one, two or three bonds apart (law of cosines with ``theta0``; the cis
and trans distances of a dihedral), looked up on the interaction lines or by
bonded type in ``[ bondtypes ]``, ``[ constrainttypes ]`` and
``[ angletypes ]``. :func:`random_walk_encoding` gives the random-walk
positional encodings of the bond graph.
"""

# Provenance: Port of ``cfaimmd.topology`` (the prototype of C-GAIMMD) with the config
# coupling removed. Standard library and numpy only.

import math
import os
import re
import shutil
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path
import numpy as np

__all__ = ['AtomRecord', 'MoleculeType', 'Topology', 'ModelTopology', 'ReferenceGeometry',
           'read_topology', 'read_atom_records', 'read_masses', 'read_atomic_numbers',
           'model_topology', 'topological_distances', 'reference_geometry',
           'random_walk_encoding', 'restrained_topology', 'RESTRAINT_DEFINE']

#: element masses as force fields write them (u) -> atomic number
_MASS_TO_Z = {
    1.008: 1, 4.003: 2, 6.941: 3, 9.012: 4, 10.811: 5, 12.011: 6, 14.007: 7, 15.999: 8,
    18.998: 9, 20.180: 10, 22.990: 11, 24.305: 12, 26.982: 13, 28.086: 14, 30.974: 15,
    32.060: 16, 35.450: 17, 39.948: 18, 39.098: 19, 40.078: 20, 54.938: 25, 55.845: 26,
    58.933: 27, 58.693: 28, 63.546: 29, 65.380: 30, 79.904: 35, 85.468: 37, 87.620: 38,
    112.411: 48, 126.904: 53, 132.905: 55, 137.327: 56,
}
#: u; a mass farther than this from every element is refused. It covers the rounding of the force
#: fields shipped with GROMACS 2024 (largest: Zn at 65.40 u in AMBER); 0.3 u read AMBER's iron at
#: 55.00 u as manganese.
_MASS_TOLERANCE = 0.05
_VIRTUAL_SITE_MASS = 0.5       # u; below: a virtual site (element 0)
#: masses that are no element but lie near one: united-atom groups (GROMOS; CH2 is 0.02 u from N,
#: CH4 0.044 u from O) and a hydrogen of four times its mass (the HEAVY_H variants, near He);
#: refused, or in the tolerant mode the element the group is built on
_GROUP_MASSES = {13.019: ('a united-atom CH1', 6), 14.027: ('a united-atom CH2', 6),
                 15.035: ('a united-atom CH3', 6), 16.043: ('a united-atom CH4', 6),
                 4.032: ('a hydrogen of four times its mass (HEAVY_H)', 1)}
_GROUP_TOLERANCE = 0.005       # u

_SECTION = re.compile(r'^\s*\[\s*([a-zA-Z_0-9]+)\s*\]')
_INCLUDE = re.compile(r'^\s*#include\s+[<"]([^">]+)[">]')

#: [ section ] -> number of atom indices that open each line
_INTERACTION_ARITY = {'bonds': 2, 'pairs': 2, 'constraints': 2, 'angles': 3, 'dihedrals': 4,
                      'cmap': 5, 'settles': 1, 'position_restraints': 1}
_PTYPES = {'A', 'D', 'S', 'V'}
#: [ section ] of a bonded type table -> number of bonded-type names that open each line
_TYPE_TABLE_ARITY = {'bondtypes': 2, 'constrainttypes': 2, 'angletypes': 3}
#: bond functs whose first parameter is the reference length b0 (harmonic, G96, Morse, cubic,
#: harmonic without exclusions); 5 (a connection), 7 (FENE), 8/9 (tabulated) and 10 (a
#: restraint) have none
_BOND_LENGTH_FUNCTS = (1, 2, 3, 4, 6)
#: constraint functs (with and without exclusions): the first parameter is the length
_CONSTRAINT_FUNCTS = (1, 2)
#: angle functs whose first parameter is theta0 (harmonic, G96, Urey-Bradley, quartic,
#: restricted bending)
_ANGLE_THETA_FUNCTS = (1, 2, 5, 6, 10)
_NM = 10.0                     # Å per nm

RESTRAINT_DEFINE = 'AIMMD_CGAIMMD_POSRES'
"""The preprocessor symbol that switches on the restraints of :func:`restrained_topology`."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomRecord:
    """One atom of an expanded topology, in system order."""

    index: int
    """Atom index (0-based) in the system (or in the generator atoms of a :class:`ModelTopology`)."""

    type: str
    """Force-field atom type."""

    resnr: int
    """Residue number as written in ``[ atoms ]`` (molecule-local)."""

    resname: str
    """Residue name."""

    name: str
    """Force-field atom name."""

    charge: float
    """Partial charge (e)."""

    mass: float
    """Mass (u), from ``[ atoms ]`` or else ``[ atomtypes ]``."""

    element: int
    """Atomic number derived from the mass (0 for a virtual site)."""

    molecule: int
    """Molecule-instance index (0-based) in ``[ molecules ]`` order; the chain id."""

    moleculetype: str
    """Name of the molecule's ``[ moleculetype ]``."""

    is_solvent: bool = False
    """True for water (residue name equal to the water residue name)."""

    is_ion: bool = False
    """True for an ion (a single charged atom, or a listed residue name)."""

    @property
    def residue_token(self):
        """``'none'`` for water and ions (interchangeable copies), else the residue name."""
        return 'none' if (self.is_solvent or self.is_ion) else self.resname

    @property
    def label(self):
        """The force-field identity ``(type, charge, mass)`` used by :mod:`aimmd.cgaimmd.preparation.symmetry`."""
        return (self.type, round(float(self.charge), 4), round(float(self.mass), 3))


@dataclass
class MoleculeType:
    """A ``[ moleculetype ]``: its atoms and bonded interaction lists."""

    name: str
    """Moleculetype name."""

    nrexcl: int = 3
    """Number of exclusions along bonds."""

    atoms: list = field(default_factory=list)
    """``[ atoms ]`` lines as dicts (``nr``, ``type``, ``resnr``, ``resname``, ``name``, ``cgnr``, ``charge``, ``mass`` or None)."""

    interactions: dict = field(default_factory=dict)
    """Section name -> list of ``(funct, atoms, params)``, atoms 0-based and molecule-local, params the remaining fields as strings."""

    def bonds(self):
        """Chemical bonds of the molecule.

        Returns
        -------
        list of tuple of int
            Molecule-local ``(i, j)`` with ``i < j``, sorted.

        Notes
        -----
        Every ``[ bonds ]`` line (funct 5, a pure connection, included), every
        ``[ constraints ]`` line of funct 1 (a constrained bond) and the two
        O-H bonds of each ``[ settles ]`` line.
        """
        out = set()
        for _, atoms, _ in self.interactions.get('bonds', ()):
            out.add((min(atoms), max(atoms)))
        for funct, atoms, _ in self.interactions.get('constraints', ()):
            if funct == 1:
                out.add((min(atoms), max(atoms)))
        for _, atoms, _ in self.interactions.get('settles', ()):
            oxygen = atoms[0]
            out.update({(oxygen, oxygen + 1), (oxygen, oxygen + 2)})
        return sorted(out)


@dataclass
class Topology:
    """A parsed GROMACS topology."""

    path: Path
    """The topology file."""

    moleculetypes: dict
    """Name -> :class:`MoleculeType`."""

    molecules: list
    """``[ molecules ]`` as ``(name, count)`` pairs."""

    atomtypes: dict
    """The ``[ atomtypes ]`` table, name -> ``{'mass', 'charge'}``."""

    unresolved_includes: list = field(default_factory=list)
    """``#include`` targets that were found nowhere (skipped)."""

    bondtypes: dict = field(default_factory=dict)
    """The ``[ bondtypes ]`` table, ``(bonded type i, bonded type j, funct)`` -> parameters (strings), first entry kept."""

    constrainttypes: dict = field(default_factory=dict)
    """The ``[ constrainttypes ]`` table, keyed as :attr:`bondtypes`."""

    angletypes: dict = field(default_factory=dict)
    """The ``[ angletypes ]`` table, ``(type i, type j, type k, funct)`` -> parameters (strings)."""

    defines: dict = field(default_factory=dict)
    """The ``#define`` symbols after preprocessing, name -> value (macro parameters such as GROMOS' ``gb_*``)."""

    def bonded_type(self, atom_type):
        """The bonded type of an atom type (the ``bond_type`` column of ``[ atomtypes ]``, else the type itself).

        Parameters
        ----------
        atom_type : str
            An atom type of ``[ atoms ]``.

        Returns
        -------
        str
            The name the bonded type tables are keyed by.
        """
        entry = self.atomtypes.get(atom_type)
        return entry.get('bond_type', atom_type) if entry else atom_type

    def instances(self):
        """Iterate over every molecule copy in system order.

        Yields
        ------
        tuple
            ``(instance_index, MoleculeType, first_atom_offset)``.
        """
        k, offset = 0, 0
        for name, count in self.molecules:
            moleculetype = self.moleculetypes[name]
            for _ in range(count):
                yield k, moleculetype, offset
                k += 1
                offset += len(moleculetype.atoms)

    @property
    def n_atoms(self):
        """Number of atoms of the expanded system."""
        return sum(len(self.moleculetypes[name].atoms) * count for name, count in self.molecules)


@dataclass
class ModelTopology:
    """Records, bonds and bonded interactions of the generator atoms (global generator-atom indices)."""

    records: list
    """:class:`AtomRecord` per generator atom, ``index`` and ``molecule`` renumbered."""

    bonds: np.ndarray
    """``(n_bonds, 2)`` int, ``i < j``, sorted."""

    interactions: dict
    """Section name -> list of ``(funct, atoms, params)`` with generator-atom indices."""

    molecules: list
    """Atom indices of every molecule instance."""

    moleculetypes: list
    """Moleculetype name of every molecule instance."""

    @property
    def n_atoms(self):
        """Number of generator atoms."""
        return len(self.records)

    @property
    def elements(self):
        """``(n_atoms,)`` int atomic numbers."""
        return np.asarray([r.element for r in self.records], dtype=int)


@dataclass
class ReferenceGeometry:
    """Force-field reference geometry of the generator atoms (Å), from the topology's bonded parameters.

    :func:`reference_geometry` builds it; every distance comes from the
    force field (``b0``, ``theta0``, constraint and SETTLE lengths), none from
    coordinates. The arrays are converted to numpy on construction.
    """

    n_atoms: int
    """Number of generator atoms."""

    bonds: np.ndarray
    """``(n_bonds, 2)`` int, ``i < j``, sorted, the bonds of the :class:`ModelTopology`."""

    bond_lengths: np.ndarray
    """``(n_bonds,)`` float reference lengths of `bonds`, Å."""

    pairs: np.ndarray
    """``(n_pairs, 2)`` int, ``i < j``, sorted, every pair of atoms one, two or three bonds apart."""

    separation: np.ndarray
    """``(n_pairs,)`` int topological distance of each pair (1, 2 or 3 bonds)."""

    distance_min: np.ndarray
    """``(n_pairs,)`` float shortest reference distance of each pair in Å (the cis distance of a 1-4 pair)."""

    distance_max: np.ndarray
    """``(n_pairs,)`` float longest reference distance of each pair in Å (the trans distance of a 1-4 pair)."""

    def __post_init__(self):
        self.n_atoms = int(self.n_atoms)
        self.bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 2)
        self.bond_lengths = np.asarray(self.bond_lengths, dtype=float).reshape(-1)
        self.pairs = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
        self.separation = np.asarray(self.separation, dtype=int).reshape(-1)
        self.distance_min = np.asarray(self.distance_min, dtype=float).reshape(-1)
        self.distance_max = np.asarray(self.distance_max, dtype=float).reshape(-1)
        if len(self.bond_lengths) != len(self.bonds):
            raise ValueError(f'{len(self.bonds)} bonds but {len(self.bond_lengths)} bond lengths')
        if not (len(self.pairs) == len(self.separation) == len(self.distance_min)
                == len(self.distance_max)):
            raise ValueError('pairs, separation, distance_min and distance_max must have the '
                             'same length')

    def pairs_at(self, separation):
        """The pairs ``separation`` bonds apart and their reference distances.

        Parameters
        ----------
        separation : int
            1, 2 or 3.

        Returns
        -------
        tuple of numpy.ndarray
            ``(pairs (n, 2), distance_min (n,), distance_max (n,))``.
        """
        keep = self.separation == int(separation)
        return self.pairs[keep], self.distance_min[keep], self.distance_max[keep]


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def _strip(line):
    return line.split(';')[0].rstrip()


def _gromacs_share_top():
    """``share/gromacs/top`` of the GROMACS executable AIMMD would use, if it exists."""
    candidates = []
    try:
        from ... import _config
        if getattr(_config, 'GROMACS', None):
            candidates.append(_config.GROMACS)
    except Exception:                                           # noqa: BLE001
        pass
    for name in ('gmx', 'gmx_mpi'):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for executable in candidates:
        top = Path(executable).resolve().parents[1] / 'share' / 'gromacs' / 'top'
        if top.is_dir():
            return top
    return None


def _include_search_path(including):
    """Directories an ``#include`` is looked up in, in GROMACS' order."""
    dirs = [Path(including).parent]
    for directory in os.environ.get('GMXLIB', '').split(os.pathsep):
        if directory:
            dirs.append(Path(directory))
    if os.environ.get('GMXDATA'):
        dirs.append(Path(os.environ['GMXDATA']) / 'top')
    share = _gromacs_share_top()
    if share is not None:
        dirs.append(share)
    return dirs


def _resolve_include(target, including):
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path if target_path.is_file() else None
    for base in _include_search_path(including):
        candidate = base / target
        if candidate.is_file():
            return candidate
    return None


def _join_continuations(raw_lines):
    lines, buffer = [], ''
    for raw in raw_lines:
        body = raw.split(';')[0].rstrip()
        if body.endswith('\\'):
            buffer += body[:-1] + ' '
            continue
        lines.append(buffer + raw if buffer else raw)
        buffer = ''
    if buffer:
        lines.append(buffer)
    return lines


def _preprocess(path, defines, seen=None, unresolved=None):
    """GROMACS-style preprocessing; returns the active lines (includes inlined)."""
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen:
        return []
    seen.add(path)
    out, stack = [], []
    for raw in _join_continuations(path.read_text().splitlines()):
        stripped = raw.strip()
        active = all(stack)
        if stripped.startswith('#'):
            tokens = stripped.split()
            directive = tokens[0]
            if directive in ('#ifdef', '#ifndef'):
                name = tokens[1] if len(tokens) > 1 else ''
                stack.append((name in defines) if directive == '#ifdef' else (name not in defines))
            elif directive == '#else':
                if stack:
                    stack[-1] = not stack[-1]
            elif directive == '#endif':
                if stack:
                    stack.pop()
            elif not active:
                continue
            elif directive == '#define' and len(tokens) > 1:
                defines[tokens[1]] = ' '.join(tokens[2:])
            elif directive == '#undef' and len(tokens) > 1:
                defines.pop(tokens[1], None)
            elif directive == '#include':
                match = _INCLUDE.match(stripped)
                if match:
                    target = _resolve_include(match.group(1), path)
                    if target is None:
                        if unresolved is not None:
                            unresolved.append(match.group(1))
                    else:
                        out.extend(_preprocess(target, defines, seen, unresolved))
            continue
        if active:
            out.append(raw)
    return out


def _parse_lines(lines, path, unresolved=(), type_tables=None):
    moleculetypes, molecules, atomtypes = {}, [], {}
    type_tables = {name: {} for name in _TYPE_TABLE_ARITY} if type_tables is None else type_tables
    section, current = None, None
    for raw in lines:
        line = _strip(raw)
        if not line.strip():
            continue
        match = _SECTION.match(line)
        if match:
            section = match.group(1).lower()
            if section in ('system', 'molecules'):
                current = None
            continue
        fields = line.split()
        if section == 'atomtypes':
            # name [bond_type] [at.num] mass charge ptype V W: ptype is 3 from the end
            if len(fields) >= 6 and fields[-3] in _PTYPES:
                try:
                    entry = {'mass': float(fields[-5]), 'charge': float(fields[-4])}
                except ValueError:
                    continue
                middle = fields[1:-5]
                if len(middle) == 2 or (len(middle) == 1 and not middle[0].isdigit()):
                    entry['bond_type'] = middle[0]
                atomtypes[fields[0]] = entry
        elif section in _TYPE_TABLE_ARITY:
            arity = _TYPE_TABLE_ARITY[section]
            if len(fields) <= arity:
                continue
            try:
                funct = int(fields[arity])
            except ValueError:
                continue
            key = tuple(fields[:arity]) + (funct,)
            type_tables[section].setdefault(key, tuple(fields[arity + 1:]))
        elif section == 'moleculetype':
            try:
                nrexcl = int(fields[1]) if len(fields) > 1 else 3
            except ValueError:
                nrexcl = 3
            current = MoleculeType(name=fields[0], nrexcl=nrexcl)
            moleculetypes[fields[0]] = current
        elif section == 'atoms' and current is not None:
            if len(fields) < 7:
                continue
            try:
                atom = {'nr': int(fields[0]), 'type': fields[1], 'resnr': int(fields[2]),
                        'resname': fields[3], 'name': fields[4], 'cgnr': int(fields[5]),
                        'charge': float(fields[6]),
                        'mass': float(fields[7]) if len(fields) >= 8 else None}
            except ValueError:
                continue
            current.atoms.append(atom)
        elif section in _INTERACTION_ARITY and current is not None:
            arity = _INTERACTION_ARITY[section]
            if len(fields) < arity:
                continue
            try:
                atoms = tuple(int(v) - 1 for v in fields[:arity])
            except ValueError:
                continue
            if section == 'settles':
                funct, params = 1, tuple(fields[1:])
            else:
                try:
                    funct = int(fields[arity]) if len(fields) > arity else 1
                except ValueError:
                    funct = 1
                params = tuple(fields[arity + 1:])
            current.interactions.setdefault(section, []).append((funct, atoms, params))
        elif section == 'molecules':
            if len(fields) >= 2:
                try:
                    molecules.append((fields[0], int(fields[1])))
                except ValueError:
                    continue
    hint = f' (includes found nowhere: {list(unresolved)})' if unresolved else ''
    if not moleculetypes and not molecules:
        raise ValueError(f'{path} contains no [ moleculetype ] / [ atoms ] section{hint}')
    if not molecules:
        molecules = [(name, 1) for name in moleculetypes]
    for name, _ in molecules:
        if name not in moleculetypes:
            raise ValueError(
                f'{path}: [ molecules ] names {name!r}, which has no [ moleculetype ] in this file '
                f'or in any include that could be resolved{hint}')
    for moleculetype in moleculetypes.values():
        moleculetype.atoms.sort(key=lambda atom: atom['nr'])
    return moleculetypes, molecules, atomtypes


def read_topology(path, *, defines=()):
    """Parse a GROMACS topology.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.top`` (or ``.itp``) file.
    defines : sequence of str, optional
        Preprocessor symbols set before reading (``-DPOSRES`` is ``('POSRES',)``).

    Returns
    -------
    Topology
        Moleculetypes, ``[ molecules ]`` and ``[ atomtypes ]``.

    Raises
    ------
    ValueError
        If the file is missing, defines no moleculetype, or ``[ molecules ]`` names an unknown one.

    Notes
    -----
    Only the sections that define atoms and bonded connectivity are
    interpreted (``atomtypes moleculetype atoms bonds pairs constraints angles
    dihedrals cmap settles position_restraints molecules``), plus the bonded
    type tables ``bondtypes constrainttypes angletypes`` that
    :func:`reference_geometry` looks parameters up in. An ``#include`` is
    looked up relative to the including file, then in ``$GMXLIB``,
    ``$GMXDATA/top`` and the ``share/gromacs/top`` directory of the GROMACS
    executable (``aimmd._config.GROMACS`` or ``gmx``/``gmx_mpi`` on ``PATH``);
    one that is found nowhere is skipped and listed in
    :attr:`Topology.unresolved_includes`. A file without ``[ molecules ]`` (an
    ``.itp``) counts each moleculetype once.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'could not read {path}')
    unresolved = []
    symbols = {name: '' for name in defines}
    lines = _preprocess(path, symbols, unresolved=unresolved)
    tables = {name: {} for name in _TYPE_TABLE_ARITY}
    moleculetypes, molecules, atomtypes = _parse_lines(lines, path, unresolved, tables)
    return Topology(path=path, moleculetypes=moleculetypes, molecules=molecules,
                    atomtypes=atomtypes, unresolved_includes=unresolved,
                    bondtypes=tables['bondtypes'], constrainttypes=tables['constrainttypes'],
                    angletypes=tables['angletypes'], defines=dict(symbols))


def _as_topology(topology, defines=()):
    return topology if isinstance(topology, Topology) else read_topology(topology, defines=defines)


# ---------------------------------------------------------------------------
# masses, elements, records
# ---------------------------------------------------------------------------

def _atom_mass(top, moleculetype, atom):
    if atom['mass'] is not None:
        return float(atom['mass'])
    atomtype = top.atomtypes.get(atom['type'])
    if atomtype is None:
        raise ValueError(
            f"{top.path}: atom {atom['name']} of moleculetype {moleculetype.name!r} has no mass "
            f"column and its type {atom['type']!r} is in no [ atomtypes ] that could be read (is "
            "the force-field directory next to the topology, on $GMXLIB or in GROMACS' share "
            'directory?)')
    return float(atomtype['mass'])


def _masses_to_atomic_numbers(masses, where='', strict=True):
    """Atomic number per mass (0 for a virtual site).

    `strict` False (the descriptors' bond rule) gives 0 instead of raising
    for a mass that matches no element, and the element a united-atom group
    or a heavy hydrogen is built on (carbon, hydrogen).
    """
    table = np.array(sorted(_MASS_TO_Z))
    out = []
    for mass in np.asarray(masses, dtype=float):
        if mass < _VIRTUAL_SITE_MASS:
            out.append(0)
            continue
        group = next((value for m, value in _GROUP_MASSES.items()
                      if abs(m - mass) <= _GROUP_TOLERANCE), None)
        if group is not None:
            if not strict:
                out.append(group[1])
                continue
            raise ValueError(
                f'mass {mass:g} u{where} is that of {group[0]}, not of an element (a united-atom '
                'or heavy-hydrogen force field); elements cannot be derived from this topology')
        k = int(np.argmin(np.abs(table - mass)))
        if abs(table[k] - mass) > _MASS_TOLERANCE:
            if not strict:
                out.append(0)
                continue
            raise ValueError(
                f'mass {mass:g} u{where} matches no element within {_MASS_TOLERANCE} u (hydrogen '
                'mass repartitioning?); elements cannot be derived from this topology')
        out.append(_MASS_TO_Z[float(table[k])])
    return np.asarray(out, dtype=int)


def read_masses(topology, *, expected=None):
    """Per-atom masses of the expanded system.

    Parameters
    ----------
    topology : str, os.PathLike or Topology
        The topology.
    expected : int, optional
        The atom count the system must have.

    Returns
    -------
    numpy.ndarray
        ``(n_atoms,)`` float masses (u) in system order.

    Raises
    ------
    ValueError
        If the topology cannot be read, a mass cannot be found, or the atom count differs from `expected`.

    Notes
    -----
    A mass missing from an ``[ atoms ]`` line (water and ion ``.itp`` files of
    the GROMACS force fields omit it) comes from ``[ atomtypes ]``.
    """
    top = _as_topology(topology)
    masses = []
    for _, moleculetype, _ in top.instances():
        masses.extend(_atom_mass(top, moleculetype, atom) for atom in moleculetype.atoms)
    out = np.asarray(masses, dtype=float)
    if expected is not None and len(out) != expected:
        raise ValueError(f'{top.path} expands to {len(out)} atoms but {expected} were expected')
    return out


def read_atomic_numbers(topology, *, expected=None):
    """Per-atom atomic numbers, derived from the topology masses.

    Parameters
    ----------
    topology : str, os.PathLike or Topology
        The topology.
    expected : int, optional
        The atom count the system must have.

    Returns
    -------
    numpy.ndarray
        ``(n_atoms,)`` int atomic numbers (0 for a virtual site).

    Raises
    ------
    ValueError
        If a mass is farther than 0.05 u from every tabulated element or is a united-atom mass (see Notes).

    Notes
    -----
    Masses are unambiguous for the elements of all-atom biomolecular force
    fields (12.011 against 35.45 is not a close call), so a mass farther than
    0.05 u from every element is an error rather than a guess, and so are the
    masses of united-atom groups (``CH1``-``CH4``, 13.019 to 16.043 u) and of
    a hydrogen of four times its mass (4.032 u), which lie near nitrogen,
    oxygen and helium: united-atom and mass-repartitioned topologies need
    explicit elements. Masses below 0.5 u are virtual sites.
    """
    top = _as_topology(topology)
    return _masses_to_atomic_numbers(read_masses(top, expected=expected), where=f' in {top.path}')


def read_atom_records(top, *, water_resname='SOL', ion_resnames=None):
    """Per-atom records of the expanded system.

    Parameters
    ----------
    top : str, os.PathLike or Topology
        The topology.
    water_resname : str, optional
        Residue name of the water model.
    ion_resnames : sequence of str, optional
        Residue names of ions; None: ions are the single charged atoms that are not water.

    Returns
    -------
    list of AtomRecord
        One record per atom, in system order.
    """
    top = _as_topology(top)
    ions = None if ion_resnames is None else {str(r) for r in ion_resnames}
    out = []
    for k, moleculetype, offset in top.instances():
        masses = [_atom_mass(top, moleculetype, atom) for atom in moleculetype.atoms]
        elements = _masses_to_atomic_numbers(masses, where=f' in moleculetype '
                                             f'{moleculetype.name!r} of {top.path}')
        single_ion = (len(moleculetype.atoms) == 1
                      and abs(float(moleculetype.atoms[0]['charge'])) > 1e-6)
        for j, (atom, mass, element) in enumerate(zip(moleculetype.atoms, masses, elements)):
            is_water = atom['resname'] == water_resname
            is_ion = (not is_water) and ((atom['resname'] in ions) if ions is not None
                                         else single_ion)
            out.append(AtomRecord(index=offset + j, type=atom['type'], resnr=atom['resnr'],
                                  resname=atom['resname'], name=atom['name'],
                                  charge=float(atom['charge']), mass=mass, element=int(element),
                                  molecule=k, moleculetype=moleculetype.name,
                                  is_solvent=is_water, is_ion=is_ion))
    return out


# ---------------------------------------------------------------------------
# the generator atoms
# ---------------------------------------------------------------------------

def model_topology(topology, *, n_solute, n_waters=0, water_resname='SOL', atoms_per_water=3,
                   defines=()):
    """Records, bonds and interactions of the generator atoms.

    Parameters
    ----------
    topology : str, os.PathLike or Topology
        The full system's topology.
    n_solute : int
        ``S``: the generator atoms start with the first ``S`` topology atoms (whole molecules only).
    n_waters : int, optional
        ``K``: copies of the water moleculetype that follows the solute.
    water_resname : str, optional
        Residue name of the water model.
    atoms_per_water : int, optional
        Atoms per water molecule (oxygen first).
    defines : sequence of str, optional
        Preprocessor symbols (see :func:`read_topology`).

    Returns
    -------
    ModelTopology
        The ``S + K * atoms_per_water`` generator atoms, renumbered from 0.

    Raises
    ------
    ValueError
        If ``S`` cuts a molecule, the topology has fewer atoms, or no suitable water follows the solute.

    Notes
    -----
    With ``n_waters=0`` the generator atoms are the first ``S`` atoms: the whole
    system in vacuum, the solute of a solute-only system. The ``K`` waters
    are copies of the first water moleculetype after the solute (every water
    of a system is one moleculetype), numbered as molecule instances after the
    solute's.
    """
    top = _as_topology(topology, defines=defines)
    records = read_atom_records(top, water_resname=water_resname)
    n_solute = int(n_solute)
    if len(records) < n_solute:
        raise ValueError(f'{top.path} expands to {len(records)} atoms but {n_solute} solute atoms '
                         'were expected')
    out_records, out_bonds, out_interactions, out_molecules, out_types = [], [], {}, [], []

    def add(moleculetype, offset, new_offset, new_k):
        for j, record in enumerate(records[offset: offset + len(moleculetype.atoms)]):
            out_records.append(_replace(record, index=new_offset + j, molecule=new_k))
        out_bonds.extend((new_offset + i, new_offset + j) for i, j in moleculetype.bonds())
        for section, entries in moleculetype.interactions.items():
            out_interactions.setdefault(section, []).extend(
                (funct, tuple(new_offset + a for a in atoms), params)
                for funct, atoms, params in entries)
        out_molecules.append(list(range(new_offset, new_offset + len(moleculetype.atoms))))
        out_types.append(moleculetype.name)

    position, k_new, water = 0, 0, None
    for _, moleculetype, offset in top.instances():
        size = len(moleculetype.atoms)
        if offset + size <= n_solute:
            add(moleculetype, offset, position, k_new)
            position += size
            k_new += 1
        elif offset < n_solute:
            raise ValueError(f'{top.path}: n_solute = {n_solute} cuts moleculetype '
                             f'{moleculetype.name!r} in two')
        elif (water is None and moleculetype.atoms
              and moleculetype.atoms[0]['resname'] == water_resname):
            water = (moleculetype, offset)
        if position == n_solute and (n_waters == 0 or water is not None):
            break
    if n_waters:
        if water is None:
            raise ValueError(f'{top.path}: n_waters = {n_waters} but no {water_resname!r} molecule '
                             'follows the solute')
        moleculetype, offset = water
        if len(moleculetype.atoms) != atoms_per_water:
            raise ValueError(f'{water_resname} has {len(moleculetype.atoms)} atoms, but '
                             f'atoms_per_water = {atoms_per_water}')
        for _ in range(int(n_waters)):
            add(moleculetype, offset, position, k_new)
            position += len(moleculetype.atoms)
            k_new += 1
    bonds = np.asarray(sorted(out_bonds), dtype=int).reshape(-1, 2)
    return ModelTopology(records=out_records, bonds=bonds, interactions=out_interactions,
                         molecules=out_molecules, moleculetypes=out_types)


def topological_distances(n_atoms, bonds, max_distance=4):
    """Bond-graph distances between all atom pairs.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    bonds : array_like
        ``(n_bonds, 2)`` atom index pairs.
    max_distance : int, optional
        Distances are clipped at this value; disconnected pairs get it too.

    Returns
    -------
    numpy.ndarray
        ``(n_atoms, n_atoms)`` int16, 0 on the diagonal.
    """
    from collections import deque

    neighbours = [[] for _ in range(n_atoms)]
    for i, j in np.asarray(bonds, dtype=int).reshape(-1, 2):
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    out = np.full((n_atoms, n_atoms), max_distance, dtype=np.int16)
    for source in range(n_atoms):
        out[source, source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            d = int(out[source, u])
            if d + 1 >= max_distance:
                continue
            for v in neighbours[u]:
                if v != source and out[source, v] > d + 1:
                    out[source, v] = d + 1
                    queue.append(v)
    return out


# ---------------------------------------------------------------------------
# force-field reference geometry
# ---------------------------------------------------------------------------

def _expand(params, defines):
    """Parameter fields with ``#define`` macros replaced by their values (GROMOS' ``gb_*``)."""
    out = []
    for field_ in params:
        value = defines.get(field_)
        out.extend(value.split() if value else [field_])
    return out


def _type_params(table, types, funct):
    """Parameters of `types` (either orientation) and `funct` in a bonded type table, or None."""
    for key in (tuple(types), tuple(reversed(types))):
        params = table.get(key + (int(funct),))
        if params is not None:
            return params
    return None


def _first_float(params, where):
    try:
        return float(params[0])
    except (IndexError, ValueError):
        raise ValueError(f'cannot read a number from the parameters {list(params)} of '
                         f'{where}') from None


def _atom_label(mt, i):
    r = mt.records[i]
    return f'{r.resname}{r.resnr}.{r.name} (#{i})'


def _cosine_distance(a, b, theta_deg):
    """The third side of a triangle with sides `a`, `b` and the angle `theta_deg` between them."""
    c = math.cos(math.radians(theta_deg))
    return math.sqrt(max(a * a + b * b - 2.0 * a * b * c, 0.0))


def _dihedral_range(b1, b2, b3, theta1, theta2):
    """The cis (dihedral 0) and trans (dihedral 180) 1-4 distances of a chain i-j-k-l."""
    t1, t2 = math.radians(theta1), math.radians(theta2)
    base = (b1 * b1 + b2 * b2 + b3 * b3 - 2.0 * b1 * b2 * math.cos(t1)
            - 2.0 * b2 * b3 * math.cos(t2))
    cis = base + 2.0 * b1 * b3 * math.cos(t1 + t2)
    trans = base + 2.0 * b1 * b3 * math.cos(t1 - t2)
    return math.sqrt(max(cis, 0.0)), math.sqrt(max(trans, 0.0))


def _bond_lengths(top, mt):
    """{(i, j): Å} of every resolvable bond: SETTLE, then constraints, then bonds."""
    bonded = [top.bonded_type(r.type) for r in mt.records]
    lengths, hh = {}, {}
    for _, atoms, params in mt.interactions.get('settles', ()):
        oxygen = atoms[0]
        values = _expand(params, top.defines)
        where = f'the [ settles ] of {_atom_label(mt, oxygen)}'
        try:
            doh, dhh = float(values[1]), float(values[2])
        except (IndexError, ValueError):
            raise ValueError(f'cannot read doh/dhh from {where}: {list(params)}') from None
        lengths[(oxygen, oxygen + 1)] = lengths[(oxygen, oxygen + 2)] = _NM * doh
        hh[(oxygen + 1, oxygen + 2)] = _NM * dhh
    for section, functs, table in (('constraints', _CONSTRAINT_FUNCTS, top.constrainttypes),
                                   ('bonds', _BOND_LENGTH_FUNCTS, top.bondtypes)):
        for funct, atoms, params in mt.interactions.get(section, ()):
            i, j = min(atoms), max(atoms)
            if (i, j) in lengths or funct not in functs:
                continue
            values = _expand(params, top.defines)
            where = f'the [ {section} ] line {_atom_label(mt, i)} - {_atom_label(mt, j)}'
            if not values:
                found = _type_params(table, (bonded[i], bonded[j]), funct)
                if found is None:
                    continue
                values = _expand(found, top.defines)
                where = (f'[ {"bondtypes" if section == "bonds" else "constrainttypes"} ] '
                         f'{bonded[i]} {bonded[j]} {funct}')
            lengths[(i, j)] = _NM * _first_float(values, where)
    return lengths, hh


def _angle_values(top, mt):
    """{(i, j, k) with i < k: theta0 in degrees} of the [ angles ] lines that have one."""
    bonded = [top.bonded_type(r.type) for r in mt.records]
    theta = {}
    for funct, atoms, params in mt.interactions.get('angles', ()):
        i, j, k = atoms
        key = (min(i, k), j, max(i, k))
        if key in theta or funct not in _ANGLE_THETA_FUNCTS:
            continue
        values = _expand(params, top.defines)
        where = f'the [ angles ] line {i + 1} {j + 1} {k + 1}'
        if not values:
            found = _type_params(top.angletypes, (bonded[i], bonded[j], bonded[k]), funct)
            if found is None:
                continue
            values = _expand(found, top.defines)
            where = f'[ angletypes ] {bonded[i]} {bonded[j]} {bonded[k]} {funct}'
        theta[key] = _first_float(values, where)
    return theta


def _angle_from_types(top, mt, i, j, k):
    """theta0 of the triple from [ angletypes ] (any funct that has one), or None."""
    types = tuple(top.bonded_type(mt.records[a].type) for a in (i, j, k))
    for funct in _ANGLE_THETA_FUNCTS:
        found = _type_params(top.angletypes, types, funct)
        if found is not None:
            return _first_float(_expand(found, top.defines),
                                f'[ angletypes ] {" ".join(types)} {funct}')
    return None


def reference_geometry(topology, model):
    """Force-field reference distances of the generator atoms: bonds, 1-3 and 1-4 pairs.

    Parameters
    ----------
    topology : str, os.PathLike or Topology
        The full system's topology (its parameters and bonded type tables); read without defines unless a :class:`Topology` is given.
    model : ModelTopology
        The generator atoms (:func:`model_topology` of the same topology).

    Returns
    -------
    ReferenceGeometry
        Bond lengths and the reference distances of every pair within three bonds, in Å.

    Raises
    ------
    ValueError
        If a bond has no length or a pair one or two bonds from a 1-3 or 1-4 pair has no angle (the message names the atoms).

    Notes
    -----
    - **Bond lengths**, in this order: the ``doh`` of a ``[ settles ]`` line
      (its O-H bonds); the length of a ``[ constraints ]`` line of funct 1 or
      2 (on the line or in ``[ constrainttypes ]``); ``b0`` of a ``[ bonds ]``
      line of funct 1, 2, 3, 4 or 6 (on the line or in ``[ bondtypes ]``). A
      bond constrained by the mdp (``constraints = h-bonds``) keeps its
      ``b0``, which is its constraint length. Parameter fields that are
      ``#define`` macros are expanded; the bonded type of an atom is the
      ``bond_type`` column of ``[ atomtypes ]`` when it has one. A bond that
      is only a connection (funct 5) has no length and is an error.
    - **1-3 pairs** (two bonds apart): the law of cosines with the two bond
      lengths and ``theta0`` of the ``[ angles ]`` line (funct 1, 2, 5, 6 or
      10, on the line or in ``[ angletypes ]``), else of ``[ angletypes ]``
      for the triple's types; the H-H pair of a SETTLE water takes ``dhh``.
    - **1-4 pairs** (three bonds apart): the distances of the chain with the
      dihedral at 0 (cis, the shortest) and 180 degrees (trans, the
      longest), from its three bond lengths and two angles.
    - A pair joined by several shortest paths (rings) gets the smallest and
      the largest distance over them. 1-2 and 1-3 pairs have
      ``distance_min == distance_max`` (up to ring paths).
    - GROMACS' units are converted: nm to Å.
    """
    top = _as_topology(topology)
    mt = model
    n = mt.n_atoms
    lengths, settle_hh = _bond_lengths(top, mt)
    bond_lengths = []
    for i, j in mt.bonds:
        key = (int(i), int(j))
        if key not in lengths:
            raise ValueError(f'{top.path}: no length for the bond {_atom_label(mt, key[0])} - '
                             f'{_atom_label(mt, key[1])}: no [ settles ], [ constraints ] or '
                             '[ bonds ] parameter (on the line or in the bonded type tables) '
                             'gives one (a funct 5 connection has none)')
        bond_lengths.append(lengths[key])
    theta = _angle_values(top, mt)
    neighbours = [[] for _ in range(n)]
    for i, j in mt.bonds:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    distance = topological_distances(n, mt.bonds, max_distance=4)

    def length(a, b):
        return lengths[(min(a, b), max(a, b))]

    def angle(a, b, c):
        key = (min(a, c), b, max(a, c))
        if key not in theta:
            value = _angle_from_types(top, mt, a, b, c)
            if value is None:
                raise ValueError(
                    f'{top.path}: no angle (theta0) for {_atom_label(mt, a)} - '
                    f'{_atom_label(mt, b)} - {_atom_label(mt, c)}: no [ angles ] line with '
                    'theta0 and no [ angletypes ] entry for its types')
            theta[key] = value
        return theta[key]

    ranges = {}

    def add(i, j, lo, hi):
        key = (min(i, j), max(i, j))
        old = ranges.get(key)
        ranges[key] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))

    for i in range(n):
        for a in neighbours[i]:
            if i < a:
                add(i, a, length(i, a), length(i, a))
            for b in neighbours[a]:
                if b == i:
                    continue
                if distance[i, b] == 2 and i < b:
                    key = (i, a, b)
                    if (min(i, b), a, max(i, b)) not in theta and (i, b) in settle_hh:
                        d = settle_hh[(i, b)]
                    else:
                        d = _cosine_distance(length(i, a), length(a, b), angle(*key))
                    add(i, b, d, d)
                for c in neighbours[b]:
                    if c in (a, i) or distance[i, c] != 3 or c < i:
                        continue
                    lo, hi = _dihedral_range(length(i, a), length(a, b), length(b, c),
                                             angle(i, a, b), angle(a, b, c))
                    add(i, c, lo, hi)
    keys = sorted(ranges)
    pairs = np.asarray(keys, dtype=int).reshape(-1, 2)
    return ReferenceGeometry(
        n_atoms=n, bonds=np.asarray(mt.bonds, dtype=int).reshape(-1, 2),
        bond_lengths=np.asarray(bond_lengths, dtype=float),
        pairs=pairs, separation=distance[pairs[:, 0], pairs[:, 1]].astype(int),
        distance_min=np.asarray([ranges[k][0] for k in keys], dtype=float),
        distance_max=np.asarray([ranges[k][1] for k in keys], dtype=float))


def random_walk_encoding(n_atoms, bonds, steps=16):
    """Random-walk positional encodings of the bond graph: k-step return probabilities.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    bonds : array_like
        ``(n_bonds, 2)`` atom index pairs.
    steps : int, optional
        ``K``, the number of walk lengths ``k = 1 .. K``.

    Returns
    -------
    numpy.ndarray
        ``(n_atoms, K)`` float64, column ``k - 1`` the probability that a random walk from the atom is back after ``k`` steps.

    Raises
    ------
    ValueError
        If `steps` is not a positive int.

    Notes
    -----
    ``p_k(i) = ((D^-1 A)^k)_ii`` with ``A`` the adjacency and ``D`` the degrees
    of the bond graph (Dwivedi et al., ICLR 2022). The encodings are
    invariant under the graph's automorphisms (atoms of one orbit get the same
    values) and have no sign or basis ambiguity, unlike Laplacian
    eigenvectors. ``p_1 = 0`` (no self loops); an atom without bonds has
    zeros. Computed per connected component.
    """
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError(f'steps must be an int >= 1, got {steps!r}')
    steps, n_atoms = int(steps), int(n_atoms)
    bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in bonds:
        parent[find(int(i))] = find(int(j))
    roots = np.asarray([find(i) for i in range(n_atoms)])
    out = np.zeros((n_atoms, steps))
    for root in np.unique(roots):
        atoms = np.flatnonzero(roots == root)
        if len(atoms) == 1:
            continue
        local = {int(a): k for k, a in enumerate(atoms)}
        adjacency = np.zeros((len(atoms), len(atoms)))
        for i, j in bonds:
            if int(i) in local and int(j) in local:
                adjacency[local[int(i)], local[int(j)]] = adjacency[local[int(j)], local[int(i)]] = 1.0
        walk = adjacency / adjacency.sum(axis=1, keepdims=True)
        power = np.eye(len(atoms))
        for k in range(steps):
            power = power @ walk
            out[atoms, k] = np.diagonal(power)
    return out


# ---------------------------------------------------------------------------
# derived topologies
# ---------------------------------------------------------------------------

def _format_k(value):
    return f'{float(value):g}'


def _posres_block(define, local_indices, k, comment):
    lines = [f'#ifdef {define}', '[ position_restraints ]',
             f';  ai  funct  fcx  fcy  fcz   ({comment})']
    kk = _format_k(k)
    lines += [f'{i + 1:6d}     1  {kk}  {kk}  {kk}' for i in local_indices]
    lines.append('#endif')
    return lines


def _flat_bottom_block(define, local_index, radius_A, k):
    return [f'#ifdef {define}', '[ position_restraints ]',
            ';  ai  funct  g  r (nm; negative: keep outside)  k   '
            '(aimmd.cgaimmd: bulk water exclusion)',
            f'{local_index + 1:6d}     2  1  {-float(radius_A) / 10.0:.4f}  {_format_k(k)}',
            '#endif']


def _water_copy_lines(top, moleculetype, name, define, local_indices, k):
    """A restrained copy of a SETTLE water with the settles written as three constraints."""
    settles = moleculetype.interactions.get('settles', ())
    if len(moleculetype.atoms) != 3 or len(settles) != 1:
        raise NotImplementedError(
            f'only 3-site waters with one [ settles ] line can be copied, {moleculetype.name!r} '
            f'has {len(moleculetype.atoms)} atoms')
    params = settles[0][2]
    try:
        doh, dhh = params[1], params[2]
        float(doh), float(dhh)
    except (IndexError, ValueError):
        raise ValueError(f'{top.path}: cannot read doh/dhh from the [ settles ] of '
                         f'{moleculetype.name!r}: {params}') from None
    lines = ['', f'; restrained copy of {moleculetype.name} '
             '(aimmd.cgaimmd.preparation.topology.restrained_topology): its SETTLE geometry as three '
             'constraints (GROMACS allows one [ settles ] moleculetype)',
             '[ moleculetype ]', f'{name}   {moleculetype.nrexcl}', '', '[ atoms ]']
    for atom in moleculetype.atoms:
        # charges and masses exactly as read (the shortest repr of the parsed value); no mass
        # column where the original has none, so GROMACS takes the [ atomtypes ] mass as there
        mass = '' if atom['mass'] is None else f" {repr(float(atom['mass'])):>10s}"
        lines.append(f"{atom['nr']:6d} {atom['type']:>8s} {atom['resnr']:5d} "
                     f"{atom['resname']:>6s} {atom['name']:>6s} {atom['cgnr']:5d} "
                     f"{repr(float(atom['charge'])):>10s}{mass}")
    lines += ['', '[ constraints ]', ';  ai  aj  funct  b0 (nm)',
              f'    1   2   1   {doh}', f'    1   3   1   {doh}', f'    2   3   1   {dhh}',
              '', '[ exclusions ]', '1 2 3', '2 1 3', '3 1 2', '']
    if local_indices:
        lines += _posres_block(define, local_indices, k, 'aimmd.cgaimmd: restrained cluster water')
    lines.append('')
    return lines


def _restraint_plan(top, restrained, n_restrained_waters, flat_bottom):
    """(blocks per moleculetype name, water moleculetype or None, water copy local indices)."""
    n_atoms = top.n_atoms
    restrained = sorted({int(i) for i in np.asarray(restrained, dtype=int).reshape(-1)})
    if restrained and (restrained[0] < 0 or restrained[-1] >= n_atoms):
        raise ValueError(f'restrained atom indices out of range for {n_atoms} atoms')
    wanted = set(restrained)
    per_type = {}                       # name -> list of local index sets, one per instance
    for _, moleculetype, offset in top.instances():
        local = tuple(i for i in range(len(moleculetype.atoms)) if offset + i in wanted)
        per_type.setdefault(moleculetype.name, []).append(local)
    waters = [name for name, mt in top.moleculetypes.items() if mt.interactions.get('settles')
              and any(name == used for used, _ in top.molecules)]
    water = waters[0] if len(waters) == 1 else None
    K = int(n_restrained_waters)
    if K and water is None:
        raise ValueError(f'n_restrained_waters = {K} but {top.path} has no single [ settles ] '
                         'water moleculetype')
    if flat_bottom is not None and water is None:
        raise ValueError(f'flat_bottom needs a [ settles ] water moleculetype in {top.path}')
    blocks, copy_local = {}, ()
    for name, locals_ in per_type.items():
        if name == water and K:
            head, tail = locals_[:K], locals_[K:]
            if len(head) < K:
                raise ValueError(f'n_restrained_waters = {K} but {top.path} has only '
                                 f'{len(head)} {name} molecules')
            if any(tail):
                raise NotImplementedError(
                    f'restraints on {name} molecules after the first n_restrained_waters = {K} '
                    'are not supported')
            if len(set(head)) != 1:
                raise NotImplementedError(
                    f'the first {K} {name} molecules must all be restrained on the same atoms')
            copy_local = head[0]
            continue
        if not any(locals_):
            continue
        if len(set(locals_)) != 1:
            raise NotImplementedError(
                f'moleculetype {name!r}: only restraints on the same atoms of every instance are '
                'supported (a partial restraint)')
        blocks[name] = list(locals_[0])
    # the water copy must be the first K waters of the first [ molecules ] entry of the water
    if K:
        first = next(count for used, count in top.molecules if used == water)
        if first < K:
            raise NotImplementedError(
                f'the first [ molecules ] entry of {water} has {first} < n_restrained_waters = {K} '
                'molecules')
    flat = None
    if flat_bottom is not None:
        atom_name, radius_A, k_flat = flat_bottom
        names = [atom['name'] for atom in top.moleculetypes[water].atoms]
        if atom_name not in names:
            raise ValueError(f'flat_bottom atom {atom_name!r} is not an atom of {water} ({names})')
        flat = (names.index(atom_name), float(radius_A), float(k_flat))
    return blocks, water, copy_local, flat


def restrained_topology(top_path, restrained, k, *, define=RESTRAINT_DEFINE, flat_bottom=None,
                        water_copy_suffix='R', n_restrained_waters=0):
    """The text of a topology derived from `top_path` with position restraints.

    Parameters
    ----------
    top_path : str or os.PathLike
        The system's topology; the derived text keeps its ``#include`` lines.
    restrained : array_like of int
        Global indices (0-based) of the atoms to restrain.
    k : float
        Force constant (kJ mol^-1 nm^-2) in x, y and z.
    define : str, optional
        Preprocessor symbol that switches the restraints on (``define = -D<define>`` in the mdp).
    flat_bottom : tuple, optional
        ``(atom name, radius_A, k)``: an inverted flat-bottomed restraint on that atom of every unrestrained water.
    water_copy_suffix : str, optional
        Suffix of the restrained water copy's moleculetype name.
    n_restrained_waters : int, optional
        ``K``: the first ``K`` water molecules become the restrained copy.

    Returns
    -------
    str
        The derived topology.

    Raises
    ------
    NotImplementedError
        For a restraint that is neither on whole moleculetypes nor on the first ``K`` waters.
    ValueError
        For out-of-range indices, a missing water moleculetype or atom, or a copy name that exists.

    Notes
    -----
    - A moleculetype all of whose instances are restrained (on the same
      molecule-local atoms) gets a ``#ifdef <define>`` ``[ position_restraints
      ]`` block (funct 1) where its definition ends, or right after the
      ``#include`` line whose file defines it; it must be the last moleculetype
      that file defines.
    - The water is the moleculetype with ``[ settles ]``. With
      ``n_restrained_waters = K`` its first ``K`` molecules become a copy
      ``<name><suffix>`` (written before ``[ system ]``) whose settles are
      three constraints (O-H, O-H, H-H from ``doh``, ``dhh``: GROMACS allows
      one ``[ settles ]`` moleculetype), restrained on the same atoms; the
      ``[ molecules ]`` line becomes ``<name><suffix> K`` and ``<name> n-K``
      (atom order unchanged).
    - `flat_bottom` restrains that atom of every molecule of the original
      water moleculetype (the unrestrained, bulk waters) with funct 2, g = 1,
      r = -radius (keep outside the sphere), under the same define; the
      reference position (``grompp -r``) of the atom is the sphere center.
    - The derived file is meant to live in another directory than `top_path`:
      grompp then needs ``include = -I<directory of top_path>`` in its mdp so
      that relative includes still resolve (``aimmd.cgaimmd.preparation.gromacs.mdp_text``).
    - Conditionals of `top_path` are evaluated with its own ``#define``
      lines only; symbols that the mdp would define are not known here.
    """
    top_path = Path(top_path)
    top = read_topology(top_path)
    blocks, water, copy_local, flat = _restraint_plan(top, restrained, n_restrained_waters,
                                                      flat_bottom)
    K = int(n_restrained_waters)
    copy_name = f'{water}{water_copy_suffix}' if K else None
    if copy_name is not None and copy_name in top.moleculetypes:
        raise ValueError(f'{top_path} already has a moleculetype {copy_name!r}; choose another '
                         'water_copy_suffix')
    pending = {name: _posres_block(define, local, k, 'aimmd.cgaimmd: restrained generator atoms')
               for name, local in blocks.items()}
    if flat is not None:
        pending.setdefault(water, [])
        pending[water] = pending[water] + _flat_bottom_block(define, flat[0], flat[1], flat[2])
    done = set()

    raw_lines = top_path.read_text().splitlines()
    out = [f'; derived by aimmd.cgaimmd.preparation.topology.restrained_topology from {top_path.resolve()}']
    defines = {}
    stack = []                       # (is_active, index in `out` where the conditional opened)
    current, current_depth = None, 0
    expecting_name = False
    copy_written = molecules_split = False
    section = None

    def flush(position):
        """Insert the pending block of the current moleculetype at `position` of out."""
        nonlocal current
        if current is not None and current in pending and current not in done:
            depth = min(current_depth, len(stack))
            at = stack[depth][1] if depth < len(stack) else position
            out[at:at] = pending[current]
            done.add(current)
        current = None

    for raw in raw_lines:
        stripped = raw.strip()
        body = _strip(raw).strip()
        active = all(flag for flag, _ in stack)
        if stripped.startswith('#'):
            tokens = stripped.split()
            directive = tokens[0]
            if directive in ('#ifdef', '#ifndef'):
                name = tokens[1] if len(tokens) > 1 else ''
                flag = (name in defines) if directive == '#ifdef' else (name not in defines)
                stack.append((flag, len(out)))
            elif directive == '#else':
                if stack:
                    stack[-1] = (not stack[-1][0], stack[-1][1])
            elif directive == '#endif':
                if stack:
                    stack.pop()
            elif active and directive == '#define' and len(tokens) > 1:
                defines[tokens[1]] = ' '.join(tokens[2:])
            elif active and directive == '#undef' and len(tokens) > 1:
                defines.pop(tokens[1], None)
            elif active and directive == '#include':
                match = _INCLUDE.match(stripped)
                target = _resolve_include(match.group(1), top_path) if match else None
                names = []
                if target is not None:
                    sub_lines = _preprocess(target, defines)
                    names = _moleculetype_names(sub_lines)
                if names:
                    flush(len(out))
                    out.append(raw)
                    for name in names[:-1]:
                        if name in pending:
                            raise NotImplementedError(
                                f'moleculetype {name!r} is defined in {match.group(1)} but is not '
                                'the last moleculetype of that file; its restraints cannot be '
                                'placed after the #include')
                    last = names[-1]
                    if last in pending and last not in done:
                        out.extend(pending[last])
                        done.add(last)
                    continue
            out.append(raw)
            continue
        if not active:
            out.append(raw)
            continue
        match = _SECTION.match(body) if body else None
        if match:
            section = match.group(1).lower()
            if section == 'moleculetype':
                flush(len(out))
                expecting_name = True
            elif section in ('system', 'molecules'):
                flush(len(out))
                if section == 'system' and copy_name is not None and not copy_written:
                    out.extend(_water_copy_lines(top, top.moleculetypes[water], copy_name, define,
                                                 list(copy_local), k))
                    copy_written = True
            out.append(raw)
            continue
        if expecting_name and body:
            current, current_depth = body.split()[0], len(stack)
            expecting_name = False
        if (section == 'molecules' and body and copy_name is not None and not molecules_split
                and body.split()[0] == water):
            count = int(body.split()[1])
            out.append(f'{copy_name:<16s} {K}')
            out.append(f'{water:<16s} {count - K}')
            molecules_split = True
            continue
        out.append(raw)
    flush(len(out))
    missing = set(pending) - done
    if missing:
        raise ValueError(f'could not place the restraints of {sorted(missing)} in {top_path}')
    if copy_name is not None and not (copy_written and molecules_split):
        raise ValueError(f'{top_path} has no [ system ] / [ molecules ] section for the water copy')
    return '\n'.join(out) + '\n'


def _moleculetype_names(lines):
    names, expecting = [], False
    for raw in lines:
        line = _strip(raw).strip()
        if not line:
            continue
        match = _SECTION.match(line)
        if match:
            expecting = match.group(1).lower() == 'moleculetype'
            continue
        if expecting:
            names.append(line.split()[0])
            expecting = False
    return names
