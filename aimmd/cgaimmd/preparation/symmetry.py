"""
aimmd.cgaimmd.preparation.symmetry
==================================

Permutation symmetries of the generator atoms, read off the topology.

The *labelled bond graph* has one node per generator atom, labelled by its
force-field identity ``(atom type, charge, mass)``, and one edge per chemical
bond. Its automorphism group is the set of relabellings of identical particles
that leave the bonded topology unchanged. Three things are derived from it:

- **orbits**: atoms that some automorphism exchanges (the hydrogens of a
  methyl, the oxygens of a carboxylate, the ring atoms of a Tyr flip); 1-WL
  color refinement bounds them and VF2 checks them pair by pair, so they are
  exact. The embedding ties tokens along them (:mod:`aimmd.cgaimmd.flow_matching.embedding`);
- **factors**: the group as a direct product of groups with disjoint supports,
  which makes the OT pairing exact and cheap (the best element of a direct
  product is the product of the best element of each factor);
- **exact or approximate**: an automorphism of the labelled bond graph can
  still change the energy when an interaction list orders the atoms it swaps
  (CHARMM's carboxylate improper ``CG CB OD2 OD1``). Every group element is
  checked against the bonded interaction lists; the elements that pass form
  the *exact* subgroup.

How the factors are found without enumerating the group: atoms in trivial
orbits are fixed by every automorphism; the moving atoms split into the
connected components they induce, components that share an orbit form a
cluster, and because a cluster touches the rest of the molecule only through
fixed atoms, the group is the direct product of the cluster groups. Each
cluster group is enumerated by VF2 up to ``max_factor_order`` elements and
split further along its own orbits when that is a direct product too; a
larger one is not enumerated and its orbits go to Hungarian assignment.

Large molecules (``twins_only_min_atoms``): the orbit search can take very
long on highly symmetric molecules (a 196-atom calixarene host did not finish
in 15 minutes), so a moleculetype with at least that many atoms gets only its
*twin* swaps (atoms with equal label and equal neighbour set: the hydrogens of
a CH2 or CH3, the oxygens of a carboxylate), a subgroup of the full group.
Swaps that move a bonded skeleton (ring flips) are then lost.

The analysis runs once per moleculetype and is replicated to the other
copies. Whole identical solvent molecules (waters, ions) are reported as
molecule blocks, which the OT pairing assigns as blocks; identical non-solvent
molecules are not exchanged (the embedding's chain token tells them apart).

Port of ``cfaimmd.symmetry`` (the prototype of C-GAIMMD) with the config
coupling removed. ``networkx`` is imported inside the functions that need it.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import _once

__all__ = ['SYMMETRY_MODES', 'Factor', 'SymmetryReport', 'analyse', 'OTSpec', 'ot_spec',
           'check_ot_tokens', 'OTTokenMismatch']

SYMMETRY_MODES = ('topology', 'topology+approximate', 'groups', 'none')
"""Which permutations the OT pairing searches over (rotations always).

``'topology'`` the exact factors of the labelled bond-graph group;
``'topology+approximate'`` all its factors; ``'groups'`` user-given
Hungarian groups; ``'none'`` rotations only."""

#: sections whose interactions decide whether a permutation is exact
_BONDED_SECTIONS = ('bonds', 'pairs', 'constraints', 'angles', 'dihedrals', 'cmap', 'settles')


class OTTokenMismatch(ValueError):
    """An OT permutation would exchange atoms the embedding tells apart."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Factor:
    """One direct factor of a symmetry group, a permutation group acting on its support alone."""

    support: tuple
    """The atoms the factor moves (generator-atom indices)."""

    elements: Optional[tuple]
    """Group elements as position tuples within the support, identity first; None if not enumerated."""

    order: int
    """Order of the group."""

    exact: Optional[bool]
    """True if every element preserves every bonded interaction list, False if not, None if not enumerated."""

    orbits: tuple
    """Non-trivial orbits of the factor."""

    molecule: int
    """Molecule instance of the support."""

    label: str = ''
    """Atom names of the support, for messages."""

    @property
    def enumerated(self):
        """Whether the elements are known (the order was within the cap)."""
        return self.elements is not None

    @property
    def kind(self):
        """``'exact'``, ``'approximate'`` or ``'unenumerated'``."""
        return {True: 'exact', False: 'approximate'}.get(self.exact, 'unenumerated')

    def atom_maps(self):
        """The elements as atom maps.

        Returns
        -------
        list of dict
            Per element ``{slot atom: source atom}``: element ``e`` sends the coordinates of atom ``support[e[j]]`` to slot ``support[j]``.

        Raises
        ------
        ValueError
            If the factor was not enumerated.
        """
        if self.elements is None:
            raise ValueError('factor was not enumerated')
        return [{self.support[j]: self.support[e[j]] for j in range(len(self.support))}
                for e in self.elements]

    def _shifted(self, delta, molecule):
        return Factor(tuple(i + delta for i in self.support), self.elements, self.order,
                      self.exact, tuple(tuple(i + delta for i in o) for o in self.orbits),
                      molecule, self.label)


@dataclass
class SymmetryReport:
    """The symmetry analysis of a set of generator atoms (see the module docstring)."""

    n_atoms: int
    """Number of generator atoms."""

    orbits: list
    """Non-trivial orbits (each sorted, sorted by first atom) of the intramolecular group."""

    factors: list
    """Direct factors of the full labelled bond-graph group (exact, approximate or unenumerated)."""

    exact_factors: list
    """Direct factors of the exact subgroup."""

    molecule_block_sets: list = field(default_factory=list)
    """Identical solvent molecules, one list of atom blocks per solvent moleculetype."""

    wl_equals_orbits: bool = True
    """Whether the 1-WL color classes equal the orbits."""

    max_factor_order: int = 720
    """Enumeration cap of the analysis."""

    atom_names: Optional[list] = None
    """``'<resname><resnr>.<name>'`` of every generator atom."""

    twins_only_min_atoms: Optional[int] = None
    """Size threshold of the twin-swap fast path, or None for the full analysis everywhere."""

    twins_only_moleculetypes: list = field(default_factory=list)
    """Moleculetypes that got only their twin swaps."""

    def select(self, include=('exact',)):
        """The factors of the exact subgroup or of the full group.

        Parameters
        ----------
        include : tuple of str, optional
            ``('exact',)`` or ``('exact', 'approximate')``.

        Returns
        -------
        list of Factor
            The selected factors.

        Raises
        ------
        ValueError
            For another `include`.
        """
        include = tuple(include)
        if 'approximate' in include:
            return list(self.factors)
        if include == ('exact',):
            return list(self.exact_factors)
        raise ValueError(f"include must be ('exact',) or ('exact', 'approximate'), got {include}")

    def order(self, include=('exact', 'approximate')):
        """Order of the selected group.

        Parameters
        ----------
        include : tuple of str, optional
            ``('exact',)`` or ``('exact', 'approximate')``, as for :meth:`select`.

        Returns
        -------
        int
            The product of the orders of the selected factors.
        """
        return int(math.prod(f.order for f in self.select(include)))

    @property
    def molecule_blocks(self):
        """The one set of exchangeable solvent blocks (empty if none).

        Raises
        ------
        NotImplementedError
            If two solvent moleculetypes have several copies among the generator atoms.
        """
        sets = [s for s in self.molecule_block_sets if len(s) > 1]
        if len(sets) > 1:
            raise NotImplementedError(
                f'{len(sets)} different solvent moleculetypes with several copies among the '
                'generator atoms; the OT pairing handles one set of identical molecule blocks')
        return sets[0] if sets else []

    def name(self, i):
        """The name of a generator atom.

        Parameters
        ----------
        i : int
            Generator-atom index.

        Returns
        -------
        str
            ``'<resname><resnr>.<name>'``, or the index when no names are known.
        """
        return self.atom_names[i] if self.atom_names else str(i)

    def summary(self):
        """A JSON-serializable description of the analysis.

        Returns
        -------
        dict
            Group orders, orbit and factor counts, OT candidates per pass and the factors themselves.
        """
        def describe(f):
            return {'kind': f.kind, 'order': f.order, 'support': list(f.support),
                    'atoms': [self.name(i) for i in f.support], 'label': f.label}

        out = {
            'n_atoms': self.n_atoms,
            'group_order': self.order(('exact', 'approximate')),
            'exact_subgroup_order': self.order(('exact',)),
            'n_nontrivial_orbits': len(self.orbits),
            'n_atoms_in_nontrivial_orbits': int(sum(len(o) for o in self.orbits)),
            'n_factors': len(self.factors),
            'n_exact_factors': len(self.exact_factors),
            'n_candidates_per_ot_pass': int(sum(f.order for f in self.factors if f.enumerated)),
            'n_candidates_per_ot_pass_exact': int(sum(f.order for f in self.exact_factors
                                                      if f.enumerated)),
            'wl_classes_equal_orbits': bool(self.wl_equals_orbits),
            'max_factor_order': self.max_factor_order,
            'factors': [describe(f) for f in self.factors],
            'exact_factors': [describe(f) for f in self.exact_factors],
            'molecule_block_sets': [len(s) for s in self.molecule_block_sets],
        }
        if self.twins_only_min_atoms is not None:
            out['twins_only_min_atoms'] = int(self.twins_only_min_atoms)
            out['twins_only_moleculetypes'] = list(self.twins_only_moleculetypes)
        return out


# ---------------------------------------------------------------------------
# graph machinery
# ---------------------------------------------------------------------------

def _labelled_graph(records, bonds, atoms=None):
    """``networkx.Graph`` with node attribute ``label`` (the force-field identity)."""
    import networkx as nx

    keep = None if atoms is None else {int(a) for a in atoms}
    graph = nx.Graph()
    for record in records:
        if keep is None or record.index in keep:
            graph.add_node(int(record.index), label=record.label, element=int(record.element))
    for i, j in np.asarray(bonds, dtype=int).reshape(-1, 2):
        i, j = int(i), int(j)
        if keep is None or (i in keep and j in keep):
            graph.add_edge(i, j)
    return graph


def _wl_colours(graph, pins=None):
    """1-WL color refinement from the node labels (and optional pin marks) to the stable partition."""
    pins = pins or {}
    init = {n: (repr(graph.nodes[n]['label']), int(pins.get(n, -1))) for n in graph}
    palette = {key: i for i, key in enumerate(sorted(set(init.values())))}
    colours = {n: palette[init[n]] for n in graph}
    n_classes = len(palette)
    while True:
        signature = {n: (colours[n], tuple(sorted(colours[m] for m in graph.neighbors(n))))
                     for n in graph}
        palette = {key: i for i, key in enumerate(sorted(set(signature.values())))}
        new = {n: palette[signature[n]] for n in graph}
        if len(palette) == n_classes:
            return new
        colours, n_classes = new, len(palette)


class _Matcher:
    """VF2 on two marked copies of one colored graph: is there an automorphism with u_k -> v_k?"""

    def __init__(self, graph, colours):
        import networkx as nx

        self.h1, self.h2 = nx.Graph(), nx.Graph()
        for n in graph:
            self.h1.add_node(n, c=(colours[n], 0))
            self.h2.add_node(n, c=(colours[n], 0))
        edges = list(graph.edges())
        self.h1.add_edges_from(edges)
        self.h2.add_edges_from(edges)

    def exists(self, pairs):
        from networkx.algorithms import isomorphism

        saved = []
        for k, (u, v) in enumerate(pairs, start=1):
            saved.append((u, self.h1.nodes[u]['c'], v, self.h2.nodes[v]['c']))
            self.h1.nodes[u]['c'] = (self.h1.nodes[u]['c'][0], k)
            self.h2.nodes[v]['c'] = (self.h2.nodes[v]['c'][0], k)
        try:
            matcher = isomorphism.GraphMatcher(self.h1, self.h2,
                                               node_match=lambda a, b: a['c'] == b['c'])
            return bool(matcher.is_isomorphic())
        finally:
            for u, cu, v, cv in reversed(saved):
                self.h1.nodes[u]['c'] = cu
                self.h2.nodes[v]['c'] = cv


def _orbits(graph, colours=None):
    """All orbits (singletons included) of the label-preserving automorphism group, exactly."""
    colours = dict(colours) if colours is not None else _wl_colours(graph)
    by_colour = {}
    for n in sorted(graph):
        by_colour.setdefault(colours[n], []).append(n)
    matcher, out = None, []
    for members in by_colour.values():
        remaining = list(members)
        while remaining:
            u, rest = remaining[0], remaining[1:]
            orbit, left = [u], []
            if rest:
                matcher = matcher or _Matcher(graph, colours)
                for v in rest:
                    (orbit if matcher.exists([(u, v)]) else left).append(v)
            out.append(sorted(orbit))
            remaining = left
    return sorted(out, key=lambda o: o[0])


def _automorphism_group_order(graph, movable=None):
    """The group order by the orbit-stabilizer chain, without enumerating the group."""
    nodes = sorted(graph) if movable is None else sorted({int(a) for a in movable})
    pins, order = {}, 1
    for u in nodes:
        colours = _wl_colours(graph, pins=pins)
        candidates = [v for v in nodes if v != u and v not in pins and colours[v] == colours[u]]
        if candidates:
            matcher = _Matcher(graph, colours)
            order *= 1 + sum(1 for v in candidates if matcher.exists([(u, v)]))
        pins[u] = len(pins) + 1
    return int(order)


# ---------------------------------------------------------------------------
# interaction lists: exact or approximate
# ---------------------------------------------------------------------------

def _canonical(section, atoms):
    if section in ('bonds', 'pairs', 'constraints'):
        return tuple(sorted(atoms))
    if section in ('angles', 'dihedrals'):
        # i-j-k == k-j-i; i-j-k-l == l-k-j-i for propers and impropers
        return min(tuple(atoms), tuple(reversed(atoms)))
    return tuple(atoms)                     # cmap (phi/psi order matters), settles


class _Interactions:
    """Canonical interaction sets of one molecule, for the exactness check of a permutation."""

    def __init__(self, interactions, atoms):
        keep = {int(a) for a in atoms}
        self.keys, self.by_atom = set(), {}
        for section, entries in interactions.items():
            if section not in _BONDED_SECTIONS:
                continue
            for funct, at, params in entries:
                if not set(at) <= keep:
                    continue
                self.keys.add((section, int(funct), _canonical(section, tuple(at)), tuple(params)))
                for a in set(at):
                    self.by_atom.setdefault(a, []).append((section, int(funct), tuple(at),
                                                           tuple(params)))

    def preserved_by(self, atom_map):
        moved = [a for a, b in atom_map.items() if a != b]
        seen = set()
        for a in moved:
            for section, funct, at, params in self.by_atom.get(a, ()):
                if (section, funct, at, params) in seen:
                    continue
                seen.add((section, funct, at, params))
                image = tuple(atom_map.get(x, x) for x in at)
                if (section, funct, _canonical(section, image), params) not in self.keys:
                    return False
        return True


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

def _union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def _decompose(support, perms):
    """Split an explicit permutation group (positions within `support`) into direct factors."""
    k = len(support)
    perms = sorted({tuple(p) for p in perms})
    find, union = _union_find(k)
    for p in perms:
        for j in range(k):
            union(j, p[j])
    orbit = {}
    for j in range(k):
        orbit.setdefault(find(j), []).append(j)
    blocks = [sorted(o) for o in orbit.values() if len(o) > 1]
    if not blocks:
        return []

    def projection(block):
        return {tuple(p[j] for j in block) for p in perms}

    sizes = [len(projection(b)) for b in blocks]
    block_find, block_union = _union_find(len(blocks))
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            if len(projection(blocks[a] + blocks[b])) < sizes[a] * sizes[b]:
                block_union(a, b)
    merged = {}
    for a, block in enumerate(blocks):
        merged.setdefault(block_find(a), []).extend(block)
    parts = [sorted(v) for v in merged.values()]
    if math.prod(len(projection(p)) for p in parts) != len(perms):
        parts = [sorted(j for block in blocks for j in block)]
    out = []
    for part in sorted(parts, key=lambda p: p[0]):
        position = {j: q for q, j in enumerate(part)}
        elements = {tuple(position[p[j]] for j in part) for p in perms}
        identity = tuple(range(len(part)))
        ordered = [identity] + sorted(e for e in elements if e != identity)
        out.append((tuple(support[j] for j in part), ordered))
    return out


def _orbits_of(support, elements):
    find, union = _union_find(len(support))
    for e in elements:
        for j in range(len(e)):
            union(j, e[j])
    groups = {}
    for j in range(len(support)):
        groups.setdefault(find(j), []).append(support[j])
    return tuple(tuple(sorted(v)) for v in sorted(groups.values(), key=min) if len(v) > 1)


def _cluster_group(graph, colours, cluster, cap):
    """The automorphisms moving only the cluster's atoms, as atom maps; None above `cap`."""
    import networkx as nx
    from networkx.algorithms import isomorphism

    members = set(cluster)
    neighbours = set().union(*(set(graph.neighbors(k)) for k in members)) - members
    nodes = sorted(members | neighbours)
    h = nx.Graph()
    for n in nodes:
        h.add_node(n, c=(colours[n], n if n in neighbours else -1))
    h.add_edges_from(graph.subgraph(nodes).edges())
    matcher = isomorphism.GraphMatcher(h, h, node_match=lambda a, b: a['c'] == b['c'])
    out = []
    for mapping in matcher.isomorphisms_iter():
        out.append({k: int(mapping[k]) for k in sorted(members)})
        if len(out) > cap:
            return None
    return out


def _warn_unenumerated(factor, cap, what):
    if _once.once(('symmetry unenumerated', factor.label, factor.order, cap)):
        print(f'WARNING: symmetry {what} {factor.label} has order {factor.order} > '
              f'max_factor_order {cap}: not enumerated, its orbits fall back to Hungarian '
              'assignment in the OT pairing', flush=True)


def _factors_from_perms(support, perms, exact_perms, interactions, molecule, label):
    factors, exact_factors = [], []
    for sub_support, elements in _decompose(support, perms):
        ok = all(interactions.preserved_by({sub_support[j]: sub_support[e[j]]
                                            for j in range(len(e))}) for e in elements)
        factors.append(Factor(sub_support, tuple(elements), len(elements), bool(ok),
                              _orbits_of(sub_support, elements), molecule, label(sub_support)))
    for sub_support, elements in _decompose(support, exact_perms):
        exact_factors.append(Factor(sub_support, tuple(elements), len(elements), True,
                                    _orbits_of(sub_support, elements), molecule,
                                    label(sub_support)))
    return factors, exact_factors


def _molecule_factors(graph, interactions, cap, molecule, names):
    """(orbits, factors, exact factors, wl_equals_orbits) of one molecule graph."""
    import networkx as nx

    colours = _wl_colours(graph)
    orbits = _orbits(graph, colours)
    classes = {}
    for n in graph:
        classes.setdefault(colours[n], []).append(n)
    wl_equal = sorted(sorted(c) for c in classes.values()) == sorted(orbits)
    nontrivial = [o for o in orbits if len(o) > 1]
    moving = sorted(a for o in nontrivial for a in o)
    if not moving:
        return [], [], [], wl_equal
    components = [sorted(c) for c in nx.connected_components(graph.subgraph(moving))]
    component_of = {a: k for k, c in enumerate(components) for a in c}
    find, union = _union_find(len(components))
    for o in nontrivial:
        for a in o[1:]:
            union(component_of[o[0]], component_of[a])
    clusters = {}
    for k, c in enumerate(components):
        clusters.setdefault(find(k), []).extend(c)

    def label(atoms):
        return ' '.join(names.get(a, str(a)) for a in atoms)

    factors, exact_factors = [], []
    for cluster in sorted((sorted(c) for c in clusters.values()), key=lambda c: c[0]):
        maps = _cluster_group(graph, colours, cluster, cap)
        cluster_orbits = tuple(tuple(o) for o in nontrivial if o[0] in set(cluster))
        if maps is None:
            order = _automorphism_group_order(graph, movable=cluster)
            factor = Factor(tuple(cluster), None, order, None, cluster_orbits, molecule,
                            label(cluster))
            _warn_unenumerated(factor, cap, 'factor')
            factors.append(factor)
            exact_factors.append(factor)
            continue
        support = tuple(cluster)
        position = {a: j for j, a in enumerate(support)}
        perms, exact_perms = [], []
        for mapping in maps:
            p = tuple(position[mapping[a]] for a in support)
            perms.append(p)
            if interactions.preserved_by(mapping):
                exact_perms.append(p)
        more, more_exact = _factors_from_perms(support, perms, exact_perms, interactions,
                                               molecule, label)
        factors += more
        exact_factors += more_exact
    return nontrivial, factors, exact_factors, wl_equal


def _twin_classes(graph):
    """Classes (size >= 2) of false twins: equal node label and equal neighbour set."""
    by_key = {}
    for n in sorted(graph):
        key = (repr(graph.nodes[n]['label']), tuple(sorted(int(m) for m in graph.neighbors(n))))
        by_key.setdefault(key, []).append(int(n))
    return sorted((sorted(c) for c in by_key.values() if len(c) > 1), key=lambda c: c[0])


def _molecule_twin_factors(graph, interactions, cap, molecule, names):
    """The twin-swap subgroup of one molecule, in the format of :func:`_molecule_factors`."""
    classes = _twin_classes(graph)

    def label(atoms):
        return ' '.join(names.get(a, str(a)) for a in atoms)

    factors, exact_factors = [], []
    for cls in classes:
        support = tuple(cls)
        order = math.factorial(len(support))
        if order > cap:
            factor = Factor(support, None, order, None, (support,), molecule, label(support))
            _warn_unenumerated(factor, cap, 'twin class')
            factors.append(factor)
            exact_factors.append(factor)
            continue
        perms = [tuple(p) for p in itertools.permutations(range(len(support)))]
        exact_perms = [p for p in perms if interactions.preserved_by(
            {support[j]: support[p[j]] for j in range(len(p))})]
        more, more_exact = _factors_from_perms(support, perms, exact_perms, interactions,
                                               molecule, label)
        factors += more
        exact_factors += more_exact
    return [list(c) for c in classes], factors, exact_factors, True


def analyse(mt, *, max_factor_order=720, twins_only_min_atoms=None):
    """Symmetry analysis of the generator atoms.

    Parameters
    ----------
    mt : aimmd.cgaimmd.preparation.topology.ModelTopology
        The generator atoms (records, bonds, interactions, molecules).
    max_factor_order : int, optional
        Largest factor that is enumerated; larger ones fall back to Hungarian assignment of their orbits.
    twins_only_min_atoms : int, optional
        Moleculetypes with at least this many atoms get only their twin swaps (None: full analysis).

    Returns
    -------
    SymmetryReport
        Orbits, factors, exact factors and molecule blocks.

    Raises
    ------
    ValueError
        If `twins_only_min_atoms` is given and smaller than 1.
    """
    if twins_only_min_atoms is not None and int(twins_only_min_atoms) < 1:
        raise ValueError(f'twins_only_min_atoms must be None or >= 1, got {twins_only_min_atoms}')
    records = mt.records
    names = {r.index: f'{r.resname}{r.resnr}.{r.name}' for r in records}
    by_index = {r.index: r for r in records}
    all_orbits, all_factors, all_exact = [], [], []
    twins_only, wl_equal, done, block_sets = [], True, {}, {}
    for k, (atoms, moleculetype) in enumerate(zip(mt.molecules, mt.moleculetypes)):
        atoms = list(atoms)
        if moleculetype not in done:
            graph = _labelled_graph(records, mt.bonds, atoms)
            interactions = _Interactions(mt.interactions, atoms)
            if twins_only_min_atoms is not None and len(atoms) >= int(twins_only_min_atoms):
                result = _molecule_twin_factors(graph, interactions, int(max_factor_order), k,
                                                names)
                twins_only.append(moleculetype)
            else:
                result = _molecule_factors(graph, interactions, int(max_factor_order), k, names)
            orbits, factors, exact_factors, wl = result
            done[moleculetype] = (atoms[0], orbits, factors, exact_factors)
            wl_equal &= wl
            all_orbits += orbits
            all_factors += factors
            all_exact += exact_factors
        else:
            first, orbits, factors, exact_factors = done[moleculetype]
            delta = atoms[0] - first
            all_orbits += [[a + delta for a in o] for o in orbits]
            all_factors += [f._shifted(delta, k) for f in factors]
            all_exact += [f._shifted(delta, k) for f in exact_factors]
        if all(by_index[a].is_solvent or by_index[a].is_ion for a in atoms):
            block_sets.setdefault(moleculetype, []).append(atoms)
    return SymmetryReport(
        n_atoms=len(records), orbits=sorted(all_orbits, key=lambda o: o[0]), factors=all_factors,
        exact_factors=all_exact, molecule_block_sets=list(block_sets.values()),
        wl_equals_orbits=bool(wl_equal), max_factor_order=int(max_factor_order),
        atom_names=[names[r.index] for r in records],
        twins_only_min_atoms=None if twins_only_min_atoms is None else int(twins_only_min_atoms),
        twins_only_moleculetypes=twins_only)


# ---------------------------------------------------------------------------
# the OT pairing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OTSpec:
    """The permutations the OT pairing searches over, as plain data."""

    groups: tuple
    """Hungarian groups (atom index tuples)."""

    factors: tuple
    """Enumerated factors as ``(support, elements)`` tuples, elements as positions within the support."""

    molecule_blocks: tuple
    """Whole identical solvent molecules (atom index tuples), assigned as blocks."""

    mode: str
    """The mode of :data:`SYMMETRY_MODES` that produced it."""

    def kwargs(self):
        """Keyword arguments of ``aimmd.cgaimmd.flow_matching.flowmath.ot_pair``.

        Returns
        -------
        dict
            ``{'groups', 'factors', 'molecule_blocks'}``.
        """
        return {'groups': self.groups, 'factors': self.factors,
                'molecule_blocks': self.molecule_blocks}


def check_ot_tokens(tokens, *, groups=(), factors=(), molecule_blocks=(), where='OT pairing',
                    names=None):
    """Check that every OT permutation exchanges only atoms that share an embedding token.

    Parameters
    ----------
    tokens : array_like
        ``(M,)`` embedding tokens of the generator atoms.
    groups : sequence of sequence of int, optional
        Hungarian groups.
    factors : sequence or PermutationFactors, optional
        ``(support, elements)`` factors (or an object with a ``factors`` attribute holding them).
    molecule_blocks : sequence of sequence of int, optional
        Whole identical molecules.
    where : str, optional
        What is checked, for the message.
    names : sequence of str, optional
        Atom names for the message.

    Raises
    ------
    OTTokenMismatch
        If a permutation would exchange atoms with different tokens.

    Notes
    -----
    An OT pairing that swaps two atoms the embedding tells apart teaches the
    velocity field to move atom i to where atom j belongs: the target is then
    not a function of the network's input.
    """
    t = np.asarray(tokens).reshape(-1)

    def nm(i):
        return f'{names[i]} (#{i})' if names else f'#{i}'

    for group in groups:
        group = [int(i) for i in group]
        if any(t[i] != t[group[0]] for i in group):
            raise OTTokenMismatch(
                f'{where}: group {[nm(i) for i in group]} mixes embedding tokens '
                f'{sorted({int(t[i]) for i in group})}; tie the tokens or drop the group')
    items = getattr(factors, 'factors', factors) or ()
    for support, elements in items:
        for e in elements:
            for j, source in enumerate(e):
                a, b = int(support[j]), int(support[source])
                if t[a] != t[b]:
                    raise OTTokenMismatch(
                        f'{where}: a symmetry factor exchanges {nm(a)} and {nm(b)}, which have '
                        f'different embedding tokens ({int(t[a])} vs {int(t[b])}); tie the tokens '
                        "along orbits (tie='orbits') or choose a smaller symmetry mode")
    if len(molecule_blocks):
        reference = [int(t[i]) for i in molecule_blocks[0]]
        for block in molecule_blocks[1:]:
            if [int(t[i]) for i in block] != reference:
                raise OTTokenMismatch(f'{where}: molecule blocks {list(molecule_blocks[0])} and '
                                      f'{list(block)} carry different tokens')


def ot_spec(report, *, mode='topology', tokens=None, groups=()):
    """The OT specification of a symmetry mode.

    Parameters
    ----------
    report : SymmetryReport or None
        The analysis of the generator atoms (may be None for ``'groups'`` and ``'none'``).
    mode : str, optional
        One of :data:`SYMMETRY_MODES`.
    tokens : array_like, optional
        The network's embedding tokens; given, every permutation is checked against them.
    groups : sequence of sequence of int, optional
        The Hungarian groups of ``mode='groups'``.

    Returns
    -------
    OTSpec
        Groups, factors and molecule blocks for ``aimmd.cgaimmd.flow_matching.flowmath.ot_pair``.

    Raises
    ------
    ValueError
        For an unknown mode, `groups` with another mode than ``'groups'``, or a missing report.
    OTTokenMismatch
        If `tokens` are given and a permutation exchanges atoms with different tokens.

    Notes
    -----
    ``'none'`` is rotations only and ``'groups'`` the given groups, both
    without molecule blocks. The topology modes use the enumerated exact (or
    all) factors, the orbits of unenumerated factors as Hungarian groups and
    the report's molecule blocks.
    """
    if mode not in SYMMETRY_MODES:
        raise ValueError(f'symmetry mode must be one of {SYMMETRY_MODES}, got {mode!r}')
    groups = tuple(tuple(int(i) for i in g) for g in groups)
    if groups and mode != 'groups':
        raise ValueError(f"permutation groups are used only with mode='groups', not {mode!r}")
    names = getattr(report, 'atom_names', None)
    if mode == 'none':
        return OTSpec((), (), (), mode)
    if mode == 'groups':
        if tokens is not None:
            check_ot_tokens(tokens, groups=groups, where='permutation groups', names=names)
        return OTSpec(groups, (), (), mode)
    if report is None:
        raise ValueError(f'mode {mode!r} needs a SymmetryReport')
    selected = report.select(('exact',) if mode == 'topology' else ('exact', 'approximate'))
    factors = tuple((tuple(f.support), tuple(tuple(e) for e in f.elements))
                    for f in selected if f.enumerated)
    fallback = tuple(tuple(o) for f in selected if not f.enumerated for o in f.orbits)
    blocks = tuple(tuple(b) for b in report.molecule_blocks)
    if tokens is not None:
        check_ot_tokens(tokens, groups=fallback, factors=factors, molecule_blocks=blocks,
                        where=f'symmetry {mode!r}', names=names)
    return OTSpec(fallback, factors, blocks, mode)
