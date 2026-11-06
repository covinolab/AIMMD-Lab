"""
aimmd.cgaimmd.bioemu2.system
============================

The system of a BioEmu2 C-GAIMMD run: which MD atoms BioEmu2 generates, in
which order, and how the MD atoms it does not model are completed.

The *generator atoms* are the atoms C-GAIMMD's descriptors and candidates
hold: the first ``n_atoms`` atoms of the MD structure (the solute; see
:class:`aimmd.cgaimmd.types.CandidateContext` and the preparation tools).
BioEmu2 generates heavy atoms only, in its own order. :class:`BioEmu2System`
holds the map between the two and everything else the adaptor needs that
depends on the system:

- the **atom map** ``bioemu2_to_md``: the generator atom of every BioEmu2
  atom. Protein chains are mapped residue by residue (the k-th protein
  residue of BioEmu2 is the k-th protein residue of the MD; residue numbers
  need not agree) and, within a residue, by atom name after translating
  force-field spellings (:data:`RESIDUE_ALIASES`, :data:`ATOM_ALIASES`,
  C-terminal ``OT1``/``OC1``/``O1`` -> ``O``). Ligands are mapped by an
  isomorphism of their element-labelled heavy-atom graphs (RDKit; the first
  isomorphism in sorted order, so a symmetric ligand has a fixed convention;
  the automorphisms are stored);
- the **completion** of the generator atoms BioEmu2 does not model
  (hydrogens, a C-terminal OXT, capping groups, ions): each is placed from a
  template structure (the MD structure, made whole) by a rigid fit of nearby
  known atoms. Unmapped heavy atoms first (from their bonded known atom and
  that atom's known heavy neighbours, shell by shell, until at least three),
  then hydrogens (likewise from their heavy atom), then unbonded atoms such
  as ions (from the known heavy atoms within ``ion_cutoff`` of them and those
  atoms' heavy neighbours). Completing the template's own BioEmu2 atoms
  gives the template back (checked when the system is built);
- the **stereocenters** (generator-atom quadruplets ``(n1, center, n2, n3)``
  for :class:`TopologyValidity`): every non-glycine CA, the CB of Ile and
  Thr, and every ligand atom with four bonded neighbours that no symmetry of
  the ligand graph exchanges;
- the **reference pose** (BioEmu2 atom coordinates of a reference structure)
  and the **core** atoms whose rigid fit onto it puts every structure into
  the canonical pose of the features;
- the **token pools** of the pooled feature readout: the ligand tokens, the
  pocket (protein tokens with an atom within ``pocket_cutoff`` of a ligand
  atom in the reference pose) and all protein tokens;
- the BioEmu2 ``System`` itself, the bioemu2 version the folder was built
  with, the element sequence and bond graph of every ligand as BioEmu2
  ordered them (checked at load: :func:`._api.check_ligand_order`), and the
  provenance of the trunk embeddings.

A system lives in a folder (``python -m aimmd.cgaimmd.bioemu2 build-system``,
:mod:`.build`): ``system.json`` and ``trunk_embeddings.pt``. Loading it and
the coordinate maps need numpy only; building it and the load-time check
need bioemu2.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ['BioEmu2System', 'MDAtoms', 'SCHEMA', 'SYSTEM_FILE', 'TRUNK_FILE', 'RESIDUE_ALIASES',
           'ATOM_ALIASES', 'residue_name', 'md_elements', 'kabsch', 'completion_groups', 'complete_from_groups']

SCHEMA = 'aimmd.cgaimmd.bioemu2.system/1'
"""The schema of ``system.json``."""
SYSTEM_FILE = 'system.json'
TRUNK_FILE = 'trunk_embeddings.pt'

AMINO_ACIDS = frozenset('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'
                        .split())
"""Residue names of the standard amino acids (CCD)."""

RESIDUE_ALIASES = {
    'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS', 'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS', 'HISD': 'HIS',
    'HISE': 'HIS', 'HISH': 'HIS', 'HIS1': 'HIS', 'HIS2': 'HIS', 'CYX': 'CYS', 'CYM': 'CYS', 'CYS2': 'CYS',
    'ASH': 'ASP', 'ASPH': 'ASP', 'GLH': 'GLU', 'GLUH': 'GLU', 'LYN': 'LYS', 'LSN': 'LYS', 'LYSH': 'LYS',
    'ARN': 'ARG',
}
"""Force-field residue names (protonation states, disulfides) -> CCD names. Amber's terminal names
(``NALA``, ``CALA``, ...) are recognised by their prefix."""

ATOM_ALIASES = {('ILE', 'CD'): 'CD1'}
"""``(CCD residue, force-field atom name) -> CCD atom name`` (CHARMM and GROMOS spell Ile CD1 ``CD``)."""

TERMINAL_OXYGENS = (('OT1', 'OT2'), ('OC1', 'OC2'), ('O1', 'O2'))
"""Force-field names of the two C-terminal oxygens; in a residue without ``O`` the first is BioEmu2's
``O`` and the second is completed (BioEmu2 has no OXT)."""

ELEMENT_MASSES = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Na': 22.990,
                  'Mg': 24.305, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078,
                  'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
                  'Se': 78.971, 'Br': 79.904, 'I': 126.904}
"""Standard atomic masses of the elements recognised from a topology's masses."""

HYDROGEN_MAX_MASS = 4.5
"""Atoms lighter than this (and heavier than 0.5) are hydrogens; repartitioned hydrogens weigh about 3."""


def residue_name(name):
    """The CCD name of a force-field residue name (``HIE`` -> ``HIS``, ``NALA`` -> ``ALA``, ...)."""
    name = str(name).upper()
    if name in RESIDUE_ALIASES:
        return RESIDUE_ALIASES[name]
    if len(name) == 4 and name[0] in 'NC':
        rest = name[1:]
        rest = RESIDUE_ALIASES.get(rest, rest)
        if rest in AMINO_ACIDS:
            return rest
    return name


def md_elements(masses, bonds):
    """The element of every atom from its mass (hydrogen-mass repartitioning undone).

    An atom lighter than :data:`HYDROGEN_MAX_MASS` is a hydrogen; a heavy
    atom gets back the mass its hydrogens took from it (each bonded hydrogen
    heavier than 1.008 took the difference) before it is matched to the
    nearest standard mass. A massless atom (a virtual site) is ``'X'``.

    Raises
    ------
    ValueError
        If a heavy mass is not within 0.4 of a known element's.
    """
    masses = np.asarray(masses, dtype=np.float64)
    hydrogen = (masses > 0.5) & (masses < HYDROGEN_MAX_MASS)
    corrected = masses.copy()
    for i, j in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
        if hydrogen[i] and not hydrogen[j]:
            corrected[j] += masses[i] - ELEMENT_MASSES['H']
        elif hydrogen[j] and not hydrogen[i]:
            corrected[i] += masses[j] - ELEMENT_MASSES['H']
    names = np.array(list(ELEMENT_MASSES))
    reference = np.array(list(ELEMENT_MASSES.values()))
    out = []
    for k, mass in enumerate(corrected):
        if masses[k] <= 0.5:
            out.append('X')
        elif hydrogen[k]:
            out.append('H')
        else:
            best = int(np.argmin(np.abs(reference - mass)))
            if abs(reference[best] - mass) > 0.4:
                raise ValueError(f'atom {k}: mass {masses[k]:g} (with its hydrogens {mass:g}) is no known '
                                 f'element {sorted(ELEMENT_MASSES)}')
            out.append(str(names[best]))
    return out


@dataclass
class MDAtoms:
    """The generator atoms as the MD topology describes them (what :meth:`BioEmu2System.build` maps).

    Attributes
    ----------
    names, resnames : list of str
        Atom and residue names of the force field.
    residues : numpy.ndarray
        ``(n,)`` int, the residue of each atom (0, 1, ... in topology order;
        equal for the atoms of one residue).
    resids : numpy.ndarray
        ``(n,)`` int, the residue numbers of the topology (reported, not used
        for the map).
    masses : numpy.ndarray
        ``(n,)``.
    positions : numpy.ndarray
        ``(n, 3)`` Ångström, every molecule whole (the template of the
        completion).
    bonds : numpy.ndarray
        ``(k, 2)`` 0-based pairs within the generator atoms.
    elements : list of str, optional
        Default: from the masses (:func:`md_elements`).
    """

    names: list
    resnames: list
    residues: np.ndarray
    resids: np.ndarray
    masses: np.ndarray
    positions: np.ndarray
    bonds: np.ndarray
    elements: list = field(default=None)

    def __post_init__(self):
        n = len(self.names)
        self.names = [str(name) for name in self.names]
        self.resnames = [str(name) for name in self.resnames]
        self.residues = np.asarray(self.residues, dtype=np.int64).reshape(n)
        self.resids = np.asarray(self.resids, dtype=np.int64).reshape(n)
        self.masses = np.asarray(self.masses, dtype=np.float64).reshape(n)
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(n, 3)
        self.bonds = np.asarray(self.bonds, dtype=np.int64).reshape(-1, 2)
        if len(self.resnames) != n:
            raise ValueError(f'{n} atom names but {len(self.resnames)} residue names')
        if len(self.bonds) and (self.bonds.min() < 0 or self.bonds.max() >= n):
            raise ValueError(f'bonds must join generator atoms 0 to {n - 1}')
        if self.elements is None:
            self.elements = md_elements(self.masses, self.bonds)
        self.elements = [str(e).capitalize() for e in self.elements]

    @property
    def n_atoms(self):
        return len(self.names)


# ---------------------------------------------------------------------------
# rigid fits and the completion
# ---------------------------------------------------------------------------

def kabsch(reference, coords):
    """Rigid fits of `reference` onto every frame of `coords`.

    Parameters
    ----------
    reference : numpy.ndarray
        ``(k, 3)``.
    coords : numpy.ndarray
        ``(n, k, 3)``.

    Returns
    -------
    rotation, reference_centre, coords_centre
        ``(n, 3, 3)``, ``(3,)`` and ``(n, 1, 3)`` with
        ``(reference - reference_centre) @ rotation + coords_centre`` the best
        proper superposition of `reference` on each frame.
    """
    p_centre, q_centre = reference.mean(0), coords.mean(1, keepdims=True)
    h = np.einsum('ka,nkb->nab', reference - p_centre, coords - q_centre)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.ones((len(coords), 3))
    diag[:, 2] = d
    return u @ (diag[:, :, None] * vt), p_centre, q_centre


def complete_from_groups(known, template, groups):
    """Place the atoms of `groups` into `known` (in place) and return it.

    Parameters
    ----------
    known : numpy.ndarray
        ``(n, N, 3)`` float64; the rows of the atoms to place are overwritten.
    template : numpy.ndarray
        ``(N, 3)``.
    groups : list
        ``[(atoms to place, reference atoms), ...]`` in placement order.
    """
    for place, refs in groups:
        rotation, p_centre, q_centre = kabsch(template[refs], known[:, refs])
        known[:, place] = np.einsum('ka,nab->nkb', template[place] - p_centre, rotation) + q_centre
    return known


def _neighbours(n_atoms, bonds):
    out = [[] for _ in range(n_atoms)]
    for i, j in np.asarray(bonds, dtype=np.int64).reshape(-1, 2):
        out[int(i)].append(int(j))
        out[int(j)].append(int(i))
    return [sorted(set(row)) for row in out]


def _reference_atoms(parent, neighbours, usable, template, minimum=3):
    """``[parent]`` + its usable heavy neighbours shell by shell until at least `minimum` atoms
    (then the usable atoms nearest the parent in the template, if the molecule is too small)."""
    refs, shell = [parent], [parent]
    while len(refs) < minimum:
        shell = sorted({j for s in shell for j in neighbours[s] if usable[j]} - set(refs))
        if not shell:
            break
        refs += shell
    if len(refs) < minimum:
        candidates = np.flatnonzero(usable)
        candidates = candidates[~np.isin(candidates, refs)]
        order = np.argsort(np.linalg.norm(template[candidates] - template[parent], axis=1), kind='stable')
        refs += [int(c) for c in candidates[order[:minimum - len(refs)]]]
    if len(refs) < minimum:
        raise ValueError(f'atom {parent}: fewer than {minimum} known atoms to place its neighbours from')
    return [int(r) for r in refs]


def completion_groups(n_atoms, mapped, bonds, elements, template, *, ion_cutoff=3.0, references=None):
    """The placement groups of the generator atoms BioEmu2 does not model.

    Parameters
    ----------
    n_atoms : int
        Number of generator atoms.
    mapped : sequence of int
        The generator atoms BioEmu2 models.
    bonds : array_like
        ``(k, 2)`` bonds between generator atoms.
    elements : sequence of str
        Element of every generator atom (``'H'`` hydrogens, ``'X'`` virtual
        sites).
    template : numpy.ndarray
        ``(n_atoms, 3)`` whole template structure.
    ion_cutoff : float, optional
        Ångström; an unbonded atom is placed from the known heavy atoms within
        this distance of it in the template and their bonded heavy atoms.
    references : dict, optional
        ``{atom: [reference atoms]}`` overriding the references of unbonded
        atoms.

    Returns
    -------
    list
        ``[(atoms, reference atoms), ...]`` in placement order (lists of int).
    """
    template = np.asarray(template, dtype=np.float64)
    neighbours = _neighbours(n_atoms, bonds)
    heavy = np.array([e not in ('H', 'X') for e in elements])
    known = np.zeros(n_atoms, bool)
    known[np.asarray(mapped, dtype=np.int64)] = True
    references = {int(k): [int(r) for r in v] for k, v in (references or {}).items()}
    groups = []

    def add(parent_of):
        by_parent = {}
        for atom, parent in sorted(parent_of.items()):
            by_parent.setdefault(parent, []).append(atom)
        for parent, atoms in sorted(by_parent.items()):
            refs = _reference_atoms(parent, neighbours, known & heavy, template)
            groups.append((atoms, refs))
        for atoms in by_parent.values():
            known[atoms] = True

    while True:                                       # 1. heavy atoms bonded to known atoms, wave by wave
        wave = {}
        for atom in np.flatnonzero(heavy & ~known):
            parents = [j for j in neighbours[atom] if known[j] and heavy[j]]
            if parents:
                wave[int(atom)] = parents[0]
        if not wave:
            break
        add(wave)
    hydrogens = {}                                    # 2. hydrogens and virtual sites bonded to a known heavy atom
    for atom in np.flatnonzero(~heavy & ~known):
        parents = [j for j in neighbours[atom] if heavy[j]]
        if parents and known[parents[0]]:
            hydrogens[int(atom)] = parents[0]
    add(hydrogens)
    for atom in np.flatnonzero(~known):                # 3. unbonded atoms (ions)
        atom = int(atom)
        if neighbours[atom] and atom not in references:
            continue
        refs = references.get(atom)
        if refs is None:
            usable = np.flatnonzero(known & heavy)
            distance = np.linalg.norm(template[usable] - template[atom], axis=1)
            near = [int(a) for a in usable[distance <= ion_cutoff]]
            refs = sorted(set(near) | {j for a in near for j in neighbours[a] if known[j] and heavy[j]})
            if len(refs) < 3:
                refs = sorted(int(a) for a in usable[np.argsort(distance, kind='stable')[:3]])
        if len(refs) < 3:
            raise ValueError(f'atom {atom}: fewer than 3 known atoms to place it from')
        groups.append(([atom], refs))
    for atom in [int(a) for a in np.flatnonzero(~known)]:
        if any(atom in place for place, _ in groups):
            known[atom] = True
    left = np.flatnonzero(~known)
    if len(left):
        raise ValueError(f'{len(left)} generator atoms can neither be generated by BioEmu2 nor placed from '
                         f'atoms it generates (first: {left[:5].tolist()}); are they in another molecule? '
                         f'Then n_atoms is too large, or the molecule is missing from the BioEmu2 system')
    return [([int(a) for a in place], [int(r) for r in refs]) for place, refs in groups]


# ---------------------------------------------------------------------------
# building the map
# ---------------------------------------------------------------------------

def _md_residue_atoms(md, residue_atoms, ccd_resname):
    """``{CCD atom name: generator atom}`` of one MD residue."""
    names = {md.names[a]: a for a in residue_atoms}
    out = {}
    for name, atom in names.items():
        out[ATOM_ALIASES.get((ccd_resname, name), name)] = atom
    if 'O' not in out:
        for first, _ in TERMINAL_OXYGENS:            # the second oxygen keeps its name and is completed
            if first in names:
                out['O'] = names[first]
                out.pop(first, None)
                break
    return out


def _formula(elements):
    counts = {}
    for e in elements:
        counts[e] = counts.get(e, 0) + 1
    return ''.join(f'{e}{counts[e] if counts[e] > 1 else ""}' for e in sorted(counts))


def map_atoms(bioemu2_atoms, bioemu2_bonds, ligand_chains, md):
    """The generator atom of every BioEmu2 atom, and what was matched how.

    Parameters
    ----------
    bioemu2_atoms : list of dict
        :func:`._api.topology_atoms` of the BioEmu2 topology.
    bioemu2_bonds : list
        :func:`._api.topology_bonds` of it.
    ligand_chains : collection of str
        The chains BioEmu2 builds from a ligand definition.
    md : MDAtoms
        The generator atoms.

    Returns
    -------
    bioemu2_to_md : numpy.ndarray
        ``(M,)`` int64.
    ligands : dict
        Per ligand chain: its BioEmu2 atoms, the generator atoms of the MD
        molecule it matched (heavy atoms in increasing order), the number of
        isomorphisms and the automorphisms of its graph.
    polymer_residues : list
        ``[(chain, res_seq, CCD name, generator atoms of the residue)]`` of
        every mapped protein residue.
    """
    from . import _api
    n_bioemu2 = len(bioemu2_atoms)
    bioemu2_to_md = np.full(n_bioemu2, -1, dtype=np.int64)
    ligand_chains = set(ligand_chains)
    # --- proteins: residue by residue, then by atom name
    be_residues = {}
    for k, atom in enumerate(bioemu2_atoms):
        if atom['chain'] not in ligand_chains:
            be_residues.setdefault(atom['residue'], []).append(k)
    be_order = sorted(be_residues)
    be_names = [bioemu2_atoms[be_residues[r][0]]['resname'] for r in be_order]
    md_residues = {}
    for a, r in enumerate(md.residues):
        md_residues.setdefault(int(r), []).append(a)
    polymer_names = AMINO_ACIDS | set(be_names)       # modified residues go by their CCD name
    md_polymer = [r for r in sorted(md_residues) if residue_name(md.resnames[md_residues[r][0]]) in polymer_names]
    md_names = [residue_name(md.resnames[md_residues[r][0]]) for r in md_polymer]
    if be_names != md_names:
        k = next((i for i, (a, b) in enumerate(zip(be_names, md_names)) if a != b),
                 min(len(be_names), len(md_names)))
        raise ValueError(f'the protein residues of the BioEmu2 system ({len(be_names)}) and of the MD '
                         f'generator atoms ({len(md_names)}) differ from residue {k + 1} of the chains on: '
                         f'BioEmu2 {be_names[k:k + 5]}, MD {md_names[k:k + 5]}')
    polymer_residues = []
    for r_be, r_md in zip(be_order, md_polymer):
        first = bioemu2_atoms[be_residues[r_be][0]]
        md_atoms = md_residues[r_md]
        by_name = _md_residue_atoms(md, md_atoms, first['resname'])
        for k in be_residues[r_be]:
            label = f"{first['resname']}{first['res_seq']}.{bioemu2_atoms[k]['name']} (chain {first['chain']})"
            a = by_name.get(bioemu2_atoms[k]['name'])
            if a is None:
                raise ValueError(f'BioEmu2 atom {label} has no MD partner in residue '
                                 f'{md.resnames[md_atoms[0]]}{md.resids[md_atoms[0]]} '
                                 f'(atoms {sorted(md.names[x] for x in md_atoms)})')
            if md.elements[a] != bioemu2_atoms[k]['element']:
                raise ValueError(f"BioEmu2 atom {label} is {bioemu2_atoms[k]['element']}, its MD partner "
                                 f'{md.names[a]} is {md.elements[a]}')
            bioemu2_to_md[k] = a
        polymer_residues.append((first['chain'], first['res_seq'], first['resname'], md_atoms))
    # --- ligands: element-graph isomorphism with an unmapped MD molecule
    heavy = np.array([e not in ('H', 'X') for e in md.elements])
    in_polymer = np.zeros(md.n_atoms, bool)
    for r in md_polymer:
        in_polymer[md_residues[r]] = True
    free = heavy & ~in_polymer
    neighbours = _neighbours(md.n_atoms, md.bonds)
    components, seen = [], np.zeros(md.n_atoms, bool)
    for start in np.flatnonzero(free):
        if seen[start]:
            continue
        stack, members = [int(start)], []
        seen[start] = True
        while stack:
            a = stack.pop()
            members.append(a)
            for b in neighbours[a]:
                if free[b] and not seen[b]:
                    seen[b] = True
                    stack.append(b)
        components.append(sorted(members))
    used, ligands = set(), {}
    bonds = [tuple(b) for b in bioemu2_bonds]
    for chain in sorted(ligand_chains, key=lambda c: min(k for k, a in enumerate(bioemu2_atoms)
                                                        if a['chain'] == c)):
        be_index = [k for k, atom in enumerate(bioemu2_atoms) if atom['chain'] == chain]
        local = {k: i for i, k in enumerate(be_index)}
        be_elements = [bioemu2_atoms[k]['element'] for k in be_index]
        be_bonds = [(local[i], local[j]) for i, j in bonds if i in local and j in local]
        query = _api.element_graph(be_elements, be_bonds)
        found = None
        for c, members in enumerate(components):
            if c in used or len(members) != len(be_index):
                continue
            md_local = {a: i for i, a in enumerate(members)}
            md_bonds = [(md_local[i], md_local[j]) for i, j in md.bonds if i in md_local and j in md_local]
            matches = _api.graph_matches(query, _api.element_graph([md.elements[a] for a in members], md_bonds))
            if matches:
                found = (c, members, matches)
                break
        if found is None:
            raise ValueError(
                f'BioEmu2 ligand chain {chain!r} ({_formula(be_elements)}, {len(be_bonds)} bonds between heavy '
                f'atoms) matches no molecule of the MD generator atoms outside the protein; those are '
                f'{[_formula([md.elements[a] for a in m]) for m in components]}')
        c, members, matches = found
        used.add(c)
        for i, k in enumerate(be_index):
            bioemu2_to_md[k] = members[matches[0][i]]
        ligands[chain] = dict(bioemu2_atoms=be_index, md_atoms=members, n_isomorphisms=len(matches),
                              automorphisms=_api.graph_matches(query, query))
    if (bioemu2_to_md < 0).any():                     # pragma: no cover - every branch above raises first
        raise ValueError(f'BioEmu2 atoms without MD partner: {np.flatnonzero(bioemu2_to_md < 0)[:5]}')
    if len(set(bioemu2_to_md.tolist())) != n_bioemu2:
        raise ValueError('two BioEmu2 atoms map to the same MD atom')
    return bioemu2_to_md, ligands, polymer_residues


def _signed_volume(x, quad):
    a, c, b, d = (x[i] for i in quad)
    return float(np.dot(a - c, np.cross(b - c, d - c)))


def _stereocenters(md, polymer_residues, ligands):
    """Protein rule (non-Gly CA; Ile and Thr CB) and the ligand graph rule; quads in generator atoms."""
    quads = []
    for _, _, resname, atoms in polymer_residues:
        names = _md_residue_atoms(md, atoms, resname)
        if all(n in names for n in ('N', 'CA', 'C', 'CB')):
            quads.append((names['N'], names['CA'], names['C'], names['CB']))
        if resname == 'ILE' and all(n in names for n in ('CA', 'CB', 'CG1', 'CG2')):
            quads.append((names['CA'], names['CB'], names['CG1'], names['CG2']))
        if resname == 'THR' and all(n in names for n in ('CA', 'CB', 'OG1', 'CG2')):
            quads.append((names['CA'], names['CB'], names['OG1'], names['CG2']))
    if ligands:
        from . import _api
        neighbours = _neighbours(md.n_atoms, md.bonds)
        for info in ligands.values():
            members = sorted(set(info['md_atoms']) | {j for a in info['md_atoms'] for j in neighbours[a]
                                                      if md.elements[j] == 'H'})
            local = {a: i for i, a in enumerate(members)}
            graph = _api.element_graph([md.elements[a] for a in members],
                                       [(local[i], local[j]) for i, j in md.bonds if i in local and j in local])
            classes = _api.symmetry_classes(graph)
            for centre in info['md_atoms']:
                bonded = neighbours[centre]
                if len(bonded) != 4 or len({classes[local[b]] for b in bonded if b in local}) != 4:
                    continue
                ordered = sorted(bonded, key=lambda b: (classes[local[b]], b))
                quads.append((ordered[0], centre, ordered[1], ordered[2]))
    keep = [q for q in quads if abs(_signed_volume(md.positions, q)) > 0.05]
    return [tuple(int(i) for i in q) for q in keep]


# ---------------------------------------------------------------------------
# the system
# ---------------------------------------------------------------------------

class BioEmu2System:
    """The atom map, completion, poses and token pools of a BioEmu2 C-GAIMMD system (module docstring).

    Make one with :meth:`build` (or ``python -m aimmd.cgaimmd.bioemu2
    build-system``) and load a saved one with :meth:`load`.

    Attributes
    ----------
    folder : Path or None
        The system folder it was loaded from or saved to.
    data : dict
        The content of ``system.json``.
    """

    def __init__(self, data, folder=None):
        if data.get('schema') != SCHEMA:
            raise ValueError(f"not a C-GAIMMD BioEmu2 system (schema {data.get('schema')!r}, expected {SCHEMA!r})")
        self.data = data
        self.folder = None if folder is None else Path(folder)
        self._bioemu2_to_md = np.asarray(data['bioemu2_to_md'], dtype=np.int64)
        self._template = np.asarray(data['template'], dtype=np.float64).reshape(-1, 3)
        self._groups = [([int(a) for a in place], [int(r) for r in refs]) for place, refs in data['completion']]

    # ------------------------------------------------------------------ sizes and maps
    @property
    def n_atoms(self):
        """Number of generator atoms (MD atoms of the descriptors and candidates)."""
        return int(self.data['n_atoms'])

    @property
    def n_bioemu2_atoms(self):
        """Number of atoms BioEmu2 generates."""
        return len(self._bioemu2_to_md)

    @property
    def bioemu2_to_md(self):
        """``(M,)`` int64: the generator atom of every BioEmu2 atom."""
        return self._bioemu2_to_md.copy()

    @property
    def bioemu2_atoms(self):
        """``[{'chain', 'residue', 'res_seq', 'resname', 'name', 'element'}]`` of the BioEmu2 atoms."""
        keys = self.data['bioemu2_atom_fields']
        return [dict(zip(keys, row)) for row in self.data['bioemu2_atoms']]

    @property
    def md_atoms(self):
        """``[{'resid', 'resname', 'name', 'element'}]`` of the generator atoms."""
        keys = self.data['md_atom_fields']
        return [dict(zip(keys, row)) for row in self.data['md_atoms']]

    @property
    def ligand_bioemu2_atoms(self):
        """``(L,)`` int64: the BioEmu2 atoms of every ligand chain (BioEmu2 order)."""
        atoms = [k for info in self.data['ligands'].values() for k in info['bioemu2_atoms']]
        return np.asarray(sorted(atoms), dtype=np.int64)

    @property
    def ligands(self):
        """Per ligand chain: its spec, element sequence, bonds, BioEmu2 and MD atoms, automorphisms."""
        return self.data['ligands']

    @property
    def template(self):
        """``(N, 3)`` Ångström: the whole template structure of the completion."""
        return self._template.copy()

    @property
    def completion(self):
        """``[(atoms, reference atoms), ...]``: the placement groups, in order."""
        return [(list(p), list(r)) for p, r in self._groups]

    @property
    def reference(self):
        """``(M, 3)`` Ångström: the BioEmu2 atoms of the reference structure (the canonical pose)."""
        return np.asarray(self.data['reference'], dtype=np.float64).reshape(-1, 3)

    @property
    def core(self):
        """``(C,)`` int64: the BioEmu2 atoms whose rigid fit onto :attr:`reference` gives the canonical pose."""
        return np.asarray(self.data['core'], dtype=np.int64)

    @property
    def token_idx_at_atom(self):
        """``(M,)`` int64: the BioEmu2 token of every BioEmu2 atom."""
        return np.asarray(self.data['token_idx_at_atom'], dtype=np.int64)

    @property
    def n_tokens(self):
        return int(self.data['n_tokens'])

    @property
    def token_pools(self):
        """``{'ligand': tokens, 'pocket': tokens, 'protein': tokens}`` (apo systems: protein only)."""
        return {name: np.asarray(tokens, dtype=np.int64) for name, tokens in self.data['token_pools'].items()}

    @property
    def bioemu2_version(self):
        """The bioemu2 version the folder was built with."""
        return self.data['bioemu2_version']

    @property
    def trunk_path(self):
        """The trunk embeddings of the folder (``trunk_embeddings.pt``)."""
        if self.folder is None:
            raise ValueError('this system has no folder yet (save it, or build it with build_system)')
        return self.folder / self.data['trunk']['file']

    def stereocenters(self):
        """``[(n1, center, n2, n3), ...]`` generator atoms, for ``TopologyValidity(stereocenters=...)``."""
        return [tuple(int(i) for i in quad) for quad in self.data['stereocenters']]

    def bioemu2_system(self):
        """The ``bioemu2.System`` (needs bioemu2)."""
        from . import _api
        return _api.make_system(self.data['bioemu2_system'])

    # ------------------------------------------------------------------ coordinates
    def to_bioemu2(self, x):
        """Generator atoms ``(n, N, 3)`` (or flat ``(n, 3N)``, or one frame) -> BioEmu2 atoms ``(n, M, 3)``."""
        x = np.asarray(x)
        single = x.ndim == 2 and x.shape == (self.n_atoms, 3) or x.ndim == 1
        x = x.reshape(-1, self.n_atoms, 3)
        out = x[:, self._bioemu2_to_md]
        return out[0] if single else out

    def complete(self, x_bioemu2):
        """BioEmu2 atoms ``(n, M, 3)`` (BioEmu2 padding rows allowed) -> generator atoms ``(n, N, 3)`` float64.

        The BioEmu2 atoms are copied to their generator atoms; every other
        generator atom is placed from the template (module docstring).
        """
        x = np.asarray(x_bioemu2, dtype=np.float64)
        single = x.ndim == 2
        x = x.reshape((1,) + x.shape) if single else x
        if x.ndim != 3 or x.shape[1] < self.n_bioemu2_atoms or x.shape[2] != 3:
            raise ValueError(f'expected (n, {self.n_bioemu2_atoms}, 3) BioEmu2 coordinates, got {np.shape(x_bioemu2)}')
        out = np.zeros((len(x), self.n_atoms, 3))
        out[:, self._bioemu2_to_md] = x[:, :self.n_bioemu2_atoms]
        complete_from_groups(out, self._template, self._groups)
        return out[0] if single else out

    # ------------------------------------------------------------------ checks
    def check(self, topology=None, *, allow_version_mismatch=False):
        """Check the folder against the installed bioemu2 (needs bioemu2).

        BioEmu2 must build the same atoms, in the same order, as when the folder
        was built (the atom names, elements and residues of every atom; the
        element sequence and bonds of every ligand), and the bioemu2 version
        must be the folder's unless `allow_version_mismatch` (the trunk
        embeddings depend on it).

        Raises
        ------
        aimmd.cgaimmd.bioemu2._api.LigandOrderError
            If a ligand is ordered or built differently.
        ValueError
            On any other mismatch.
        """
        from . import _api
        version = _api.check_version()
        if version != self.bioemu2_version and not allow_version_mismatch:
            raise ValueError(f'{self._where()} was built with bioemu2 {self.bioemu2_version}, bioemu2 {version} '
                             f'is installed: its atom map and trunk embeddings may not hold. Rebuild it, or pass '
                             f'allow_version_mismatch=True')
        if topology is None:
            topology, _ = _api.sampling_inputs(self.bioemu2_system(), 'cpu')
        _api.check_ligand_order({chain: {'elements': info['elements'], 'bonds': info['bonds']}
                                 for chain, info in self.ligands.items()}, topology, where=self._where())
        atoms = _api.topology_atoms(topology)
        stored = self.bioemu2_atoms
        if len(atoms) != len(stored):
            raise ValueError(f'bioemu2 {version} builds {len(atoms)} atoms for the system of {self._where()}, '
                             f'which holds {len(stored)}')
        for k, (have, want) in enumerate(zip(atoms, stored)):
            if any(have[key] != want[key] for key in ('chain', 'resname', 'name', 'element')):
                raise ValueError(f'bioemu2 {version} builds atom {k} as {have}, {self._where()} holds {want}')
        return self

    def _where(self):
        return f'the system folder {self.folder}' if self.folder is not None else 'this system'

    # ------------------------------------------------------------------ files
    def save(self, folder):
        """Write ``system.json`` into `folder` (created); return its path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / SYSTEM_FILE
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, path)
        self.folder = folder
        return path

    @classmethod
    def load(cls, folder, *, check=True, allow_version_mismatch=False):
        """Load a system folder.

        Parameters
        ----------
        folder : str or Path
            The folder ``build-system`` wrote.
        check : bool, optional
            Check it against the installed bioemu2 (:meth:`check`; needs
            bioemu2). ``False`` loads the maps for offline use (no bioemu2
            needed).
        allow_version_mismatch : bool, optional
            See :meth:`check`.
        """
        folder = Path(folder)
        system = cls(json.loads((folder / SYSTEM_FILE).read_text()), folder=folder)
        if check:
            system.check(allow_version_mismatch=allow_version_mismatch)
        return system

    def summary(self):
        """A short dict for logs."""
        return dict(folder=str(self.folder) if self.folder is not None else None, n_atoms=self.n_atoms,
                    n_bioemu2_atoms=self.n_bioemu2_atoms, n_tokens=self.n_tokens,
                    ligands=sorted(self.ligands), n_completed=sum(len(p) for p, _ in self._groups),
                    n_stereocenters=len(self.data['stereocenters']), core=len(self.core),
                    pools={k: len(v) for k, v in self.data['token_pools'].items()},
                    bioemu2_version=self.bioemu2_version)

    def __repr__(self):
        return f'BioEmu2System({self.summary()})'

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, bioemu2_system, md, *, reference=None, core=None, pocket_cutoff=6.0, ion_cutoff=3.0,
              placement_references=None, sources=None):
        """Map `md` onto the atoms BioEmu2 builds for `bioemu2_system` (needs bioemu2 and RDKit).

        Parameters
        ----------
        bioemu2_system : bioemu2.System, dict or path-like
            The BioEmu2 system (a YAML file, as ``System.from_yaml`` reads).
        md : MDAtoms
            The generator atoms, whole (the template of the completion).
        reference : numpy.ndarray, optional
            ``(N, 3)`` generator atoms of the reference structure (canonical
            pose and pocket); default ``md.positions``.
        core : sequence of int, optional
            BioEmu2 atoms of the rigid fit onto the reference; default all.
        pocket_cutoff : float, optional
            Ångström; protein tokens with an atom this close to a ligand atom
            in the reference form the pocket pool.
        ion_cutoff : float, optional
            See :func:`completion_groups`.
        placement_references : dict, optional
            ``{generator atom: [reference atoms]}`` for unbonded atoms.
        sources : dict, optional
            Provenance written to ``system.json``.

        Returns
        -------
        BioEmu2System
            Not saved yet (:func:`.build.build_system` saves it with the trunk
            embeddings).
        """
        from . import _api
        system = _api.make_system(bioemu2_system)
        ligand_specs = _api.ligand_chains(system)
        topology, batch = _api.sampling_inputs(system, 'cpu')
        be_atoms = _api.topology_atoms(topology)
        be_bonds = _api.topology_bonds(topology)
        bioemu2_to_md, ligands, polymer_residues = map_atoms(be_atoms, be_bonds, ligand_specs, md)
        # every heavy-atom bond of the MD between mapped atoms must be a BioEmu2 bond and vice versa
        md_to_be = {int(a): k for k, a in enumerate(bioemu2_to_md)}
        heavy = np.array([e not in ('H', 'X') for e in md.elements])
        md_heavy_bonds = {tuple(sorted((md_to_be[int(i)], md_to_be[int(j)]))) for i, j in md.bonds
                          if int(i) in md_to_be and int(j) in md_to_be}
        be_bond_set = set(map(tuple, be_bonds))
        missing = sorted(md_heavy_bonds - be_bond_set)
        extra = sorted(be_bond_set - md_heavy_bonds)
        if missing or extra:
            describe = lambda b: '-'.join(f"{be_atoms[k]['resname']}{be_atoms[k]['res_seq']}.{be_atoms[k]['name']}"
                                          for k in b)                                      # noqa: E731
            raise ValueError(f'the MD and BioEmu2 bond graphs differ under the atom map: '
                             f'{len(missing)} MD bonds BioEmu2 lacks {[describe(b) for b in missing[:4]]}, '
                             f'{len(extra)} BioEmu2 bonds the MD lacks {[describe(b) for b in extra[:4]]} '
                             f'(a disulfide missing from the BioEmu2 system?)')
        groups = completion_groups(md.n_atoms, bioemu2_to_md, md.bonds, md.elements, md.positions,
                                   ion_cutoff=ion_cutoff, references=placement_references)
        self_test = np.zeros((1, md.n_atoms, 3))
        self_test[:, bioemu2_to_md] = md.positions[bioemu2_to_md]
        error = float(np.abs(complete_from_groups(self_test, md.positions, groups)[0] - md.positions).max())
        if error > 1e-6:                              # pragma: no cover - a bug, not a user error
            raise RuntimeError(f'the completion does not reproduce its template ({error:.2e} A)')
        reference = md.positions if reference is None else np.asarray(reference, dtype=np.float64)
        reference = reference.reshape(md.n_atoms, 3)
        reference_be = reference[bioemu2_to_md]
        tokens = batch.token_idx_at_atom[0, :len(be_atoms)].cpu().numpy().astype(np.int64)
        n_tokens = int(batch.token_pad_mask[0].sum())
        is_ligand = np.array([atom['chain'] in ligand_specs for atom in be_atoms])
        pools = {}
        if is_ligand.any():
            lig = np.flatnonzero(is_ligand)
            distance = np.linalg.norm(reference_be[~is_ligand][:, None] - reference_be[lig][None], axis=-1).min(1)
            pools['ligand'] = np.unique(tokens[lig])
            pools['pocket'] = np.unique(tokens[~is_ligand][distance < pocket_cutoff])
            if not len(pools['pocket']):
                raise ValueError(f'no protein atom lies within {pocket_cutoff} A of a ligand atom in the reference '
                                 f'structure: the pocket pool would be empty (is the ligand bound there?)')
        pools['protein'] = np.unique(tokens[~is_ligand])
        core = np.arange(len(be_atoms)) if core is None else np.asarray(core, dtype=np.int64)
        if core.ndim != 1 or len(core) < 3 or core.min() < 0 or core.max() >= len(be_atoms):
            raise ValueError(f'core must hold at least 3 BioEmu2 atoms (0 to {len(be_atoms) - 1})')
        signature = _api.ligand_signature(topology, list(ligands))
        data = dict(
            schema=SCHEMA,
            created=datetime.datetime.now().isoformat(timespec='seconds'),
            bioemu2_version=_api.bioemu2_version(),
            bioemu2_system=_api.system_to_dict(system),
            n_atoms=md.n_atoms,
            md_atom_fields=['resid', 'resname', 'name', 'element'],
            md_atoms=[[int(md.resids[a]), md.resnames[a], md.names[a], md.elements[a]] for a in range(md.n_atoms)],
            bioemu2_atom_fields=['chain', 'residue', 'res_seq', 'resname', 'name', 'element'],
            bioemu2_atoms=[[a['chain'], a['residue'], a['res_seq'], a['resname'], a['name'], a['element']]
                           for a in be_atoms],
            bioemu2_bonds=[list(b) for b in be_bonds],
            bioemu2_to_md=bioemu2_to_md.tolist(),
            md_heavy_unmapped=[int(a) for a in np.flatnonzero(heavy) if int(a) not in md_to_be],
            ligands={chain: dict(spec=ligand_specs[chain], elements=signature[chain]['elements'],
                                 bonds=signature[chain]['bonds'], bioemu2_atoms=info['bioemu2_atoms'],
                                 md_atoms=info['md_atoms'], n_isomorphisms=info['n_isomorphisms'],
                                 automorphisms=[list(m) for m in info['automorphisms']])
                     for chain, info in ligands.items()},
            completion=[[place, refs] for place, refs in groups],
            template=md.positions.tolist(),
            stereocenters=[list(q) for q in _stereocenters(md, polymer_residues, ligands)],
            reference=reference_be.tolist(),
            core=core.tolist(),
            token_idx_at_atom=tokens.tolist(),
            n_tokens=n_tokens,
            token_pools={name: values.tolist() for name, values in pools.items()},
            pocket_cutoff=float(pocket_cutoff),
            ion_cutoff=float(ion_cutoff),
            trunk=dict(file=TRUNK_FILE),
            sources=dict(sources or {}),
        )
        return cls(data)
