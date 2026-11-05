"""
aimmd.cgaimmd.preparation.validity
==================================

Validity checks of generated samples: the bond graph and the
relabelling into topology order, chirality with automorphisms, and steric
clashes. :class:`TopologyValidity` is a ``cgaimmd_validity_function``.

A generator produces points in R^{3M}; nothing constrains them to be the right
molecule. A malformed structure does not merely give a bad shot, it gives a
silently bad one: GROMACS integrates a molecule with an inverted stereocenter
or with bonds between the wrong atoms without complaint, and the path enters
the ensemble.

1. **Bond graph and relabelling** (:class:`Canonicalizer`): atoms ``i, j``
   are bonded when ``d_ij < 1.30 (r_i + r_j)`` (covalent radii, Cordero 2008;
   Br 1.20). The edge count must equal the topology's (``'bond_count'``);
   1-WL color refinement from (element, token) colors must reproduce the
   reference color histograms (``'not_isomorphic'``); VF2 restricted to equal
   colors finds the isomorphism under an alarm (``'vf2_timeout'``). A sample
   whose own labelling already has the topology's edges keeps it. The
   isomorphism relabels the sample into topology order: GROMACS applies the
   topology's bonds by index, so a sample in any other labelling would be
   another molecule.
2. **Chirality with automorphisms**: stereocenters are atoms with four
   bonded neighbours in four distinct automorphism orbits; their reference
   signs come from the structure. Centers in one orbit are compared as a
   multiset of signs. Wrong-handed samples are rejected (``'chirality'``);
   with ``reflect_mirror_images`` a sample whose mirror image passes at
   every stereocenter is replaced by that mirror image.
3. **Clashes**: the minimum distance between non-bonded heavy atoms (graph
   distance >= 4, outside GROMACS' ``nrexcl = 3`` exclusions), per element
   pair, against the pair's minimum over unbiased reference MD frames minus a
   margin; without reference frames 0.75 x the Bondi van der Waals radii,
   logged once (``'clash'``). Reference frames MUST be unbiased MD (no
   umbrella, OPES, steered or MBAR-selected frames).

Port of ``cfaimmd.canonicalise``, ``cfaimmd.geometry`` and the
``Canonicaliser`` and clash scoring of the chignolin generator evaluator of
the C-GAIMMD development project. ``networkx`` and MDAnalysis are imported
inside the functions that need them.
"""

import os
import signal
import threading
import time

import numpy as np

from .. import _once
from ..types import ValidityResult
from .descriptors import _BOND_SCALE, _COVALENT_RADII, _covalent_pairs

__all__ = ['TopologyValidity', 'Canonicalizer', 'detect_stereocenters', 'covalent_bond_edges']

#: Bondi van der Waals radii (Å) of the clash fallback
_VDW_RADII = {1: 1.20, 3: 1.82, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 11: 2.27, 12: 1.73,
              14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 19: 2.75, 20: 2.31, 26: 2.00, 29: 1.40,
              30: 1.39, 35: 1.85, 53: 1.98}
_SYMBOLS = {1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 14: 'Si',
            15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu', 30: 'Zn', 35: 'Br',
            53: 'I'}


def covalent_bond_edges(positions, elements, cutoff_scale=_BOND_SCALE):
    """The bond graph of one structure by covalent radii.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(M, 3)`` Å, one structure (no periodic images).
    elements : array_like of int
        ``(M,)`` atomic numbers.
    cutoff_scale : float, optional
        Pairs closer than ``cutoff_scale * (r_cov,i + r_cov,j)`` are bonded.

    Returns
    -------
    numpy.ndarray
        ``(E, 2)`` int pairs ``i < j``, sorted.

    Raises
    ------
    ValueError
        For an element without a tabulated covalent radius.
    """
    edges = _covalent_pairs(positions, elements, cutoff_scale)
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return edges


# ---------------------------------------------------------------------------
# relabelling
# ---------------------------------------------------------------------------

class _VF2Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _VF2Timeout


def _adjacency(n, edges):
    neighbours = [[] for _ in range(n)]
    for i, j in edges:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    return neighbours


def _refine(colours, neighbours):
    """One 1-WL round; Python hashes int tuples deterministically (no PYTHONHASHSEED dependence)."""
    return [hash((colours[i], tuple(sorted(colours[j] for j in neighbours[i]))))
            for i in range(len(colours))]


def _token_codes(elements, tokens):
    elements = np.asarray(elements, dtype=np.int64)
    if tokens is None:
        return [int(z) for z in elements]
    t = np.asarray(tokens, dtype=np.int64)
    _, inverse = np.unique(np.stack([elements, t], 1), axis=0, return_inverse=True)
    return [1000 + int(v) for v in np.asarray(inverse).reshape(-1)]


class Canonicalizer:
    """Match a structure's atoms onto the topology order by bond-graph isomorphism.

    Parameters
    ----------
    elements : array_like of int
        ``(M,)`` atomic numbers in topology order.
    bonds : array_like of int
        ``(n_bonds, 2)`` bonds of the topology.
    tokens : array_like of int, optional
        Embedding tokens: the isomorphism must preserve them as well as the elements.
    index_first : bool, optional
        Keep a sample's own labelling when it already has the topology's edges.
    vf2_timeout : float, optional
        Seconds allowed for the VF2 search of one sample.
    cutoff_scale : float, optional
        Bond rule of :func:`covalent_bond_edges`.

    Raises
    ------
    ValueError
        If `tokens` has another length than `elements`, or an element has no covalent radius.

    Notes
    -----
    Several isomorphisms exist whenever the molecule has symmetries (the three
    hydrogens of a methyl); any one is correct, because the structures they
    give are related by a relabelling of identical atoms. With `tokens` (a
    typed network tells its atoms apart) only relabellings within a token
    class are accepted, so atom ``i`` of the result is the atom the network
    was asked to generate as atom ``i``.

    The alarm uses ``SIGALRM`` (``signal.setitimer``), which exists only in
    the main thread; elsewhere VF2 runs without a timeout (logged once). The
    previous handler is restored after every sample.
    """

    def __init__(self, elements, bonds, *, tokens=None, index_first=True, vf2_timeout=10.0,
                 cutoff_scale=_BOND_SCALE):
        self.elements = np.asarray(elements, dtype=int).reshape(-1)
        self.n = len(self.elements)
        if tokens is not None and len(np.asarray(tokens).reshape(-1)) != self.n:
            raise ValueError(f'{len(np.asarray(tokens).reshape(-1))} tokens for {self.n} atoms')
        missing = sorted({int(z) for z in self.elements if int(z) not in _COVALENT_RADII})
        if missing:
            raise ValueError(f'no covalent radius for elements {missing}')
        self.index_first = bool(index_first)
        self.vf2_timeout = float(vf2_timeout)
        self.cutoff_scale = float(cutoff_scale)
        bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
        self.ref_edges = {(int(min(i, j)), int(max(i, j))) for i, j in bonds}
        self.n_bonds = len(self.ref_edges)
        self._codes = _token_codes(self.elements, tokens)
        neighbours = _adjacency(self.n, sorted(self.ref_edges))
        history = [list(self._codes)]
        while True:                                  # to the stable partition
            refined = _refine(history[-1], neighbours)
            stable = len(set(refined)) == len(set(history[-1]))
            history.append(refined)
            if stable or len(history) > self.n + 2:
                break
        self._n_iter = len(history) - 1
        self._ref_histograms = [sorted(h) for h in history]
        self._ref_colours = history[-1]
        self._ref_graph = None

    def _reference_graph(self):
        if self._ref_graph is None:
            import networkx as nx
            graph = nx.Graph()
            for i in range(self.n):
                graph.add_node(i, c=self._ref_colours[i])
            graph.add_edges_from(sorted(self.ref_edges))
            self._ref_graph = graph
        return self._ref_graph

    def _isomorphic(self, matcher):
        """``matcher.is_isomorphic()`` under the alarm; None on timeout."""
        main = threading.current_thread() is threading.main_thread()
        if not main or not hasattr(signal, 'setitimer') or self.vf2_timeout <= 0:
            if self.vf2_timeout > 0 and _once.once('validity vf2 without alarm'):
                print('WARNING: C-GAIMMD validity: VF2 runs without its timeout outside the main '
                      'thread (SIGALRM is main-thread only)', flush=True)
            return bool(matcher.is_isomorphic())
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        old_delay, _ = signal.setitimer(signal.ITIMER_REAL, self.vf2_timeout)
        start = time.monotonic()
        try:
            return bool(matcher.is_isomorphic())
        except _VF2Timeout:
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
            if old_delay > 0:                        # re-arm a timer that was running before
                signal.setitimer(signal.ITIMER_REAL,
                                 max(old_delay - (time.monotonic() - start), 1e-3))

    def __call__(self, positions):
        """Relabel one structure into topology order.

        Parameters
        ----------
        positions : numpy.ndarray
            ``(M, 3)`` Å, one sample in any labelling.

        Returns
        -------
        tuple
            ``(perm, reason)``: ``perm[i]`` is the sample atom that becomes topology atom ``i``; ``(None, reason)`` without an isomorphism.

        Notes
        -----
        Reasons: ``'non_finite'``, ``'bond_count'``, ``'not_isomorphic'``,
        ``'vf2_timeout'``; ``''`` with a permutation.
        """
        x = np.asarray(positions, dtype=float)
        if x.shape != (self.n, 3) or not np.all(np.isfinite(x)):
            return None, 'non_finite'
        edges = covalent_bond_edges(x, self.elements, self.cutoff_scale)
        if len(edges) != self.n_bonds:
            return None, 'bond_count'
        edge_set = {(int(i), int(j)) for i, j in edges}
        if self.index_first and edge_set == self.ref_edges:
            return np.arange(self.n, dtype=int), ''
        neighbours = _adjacency(self.n, edges)
        colours = list(self._codes)
        if sorted(colours) != self._ref_histograms[0]:
            return None, 'not_isomorphic'
        for k in range(1, self._n_iter + 1):
            colours = _refine(colours, neighbours)
            if sorted(colours) != self._ref_histograms[k]:
                return None, 'not_isomorphic'
        import networkx as nx
        from networkx.algorithms import isomorphism
        graph = nx.Graph()
        for i in range(self.n):
            graph.add_node(i, c=colours[i])
        graph.add_edges_from(edge_set)
        matcher = isomorphism.GraphMatcher(self._reference_graph(), graph,
                                           node_match=lambda a, b: a['c'] == b['c'])
        found = self._isomorphic(matcher)
        if found is None:
            return None, 'vf2_timeout'
        if not found:
            return None, 'not_isomorphic'
        return np.array([matcher.mapping[i] for i in range(self.n)], dtype=int), ''


# ---------------------------------------------------------------------------
# chirality
# ---------------------------------------------------------------------------

_MIRROR = np.array([-1.0, 1.0, 1.0])   # the reflection of reflect_mirror_images: x -> -x (through the yz plane)


def _improper_signs(x, quads):
    """Signs of the improper dihedrals ``quads`` ``(k, 4)`` in structures ``(n, M, 3)``: ``(n, k)``."""
    quads = np.asarray(quads, dtype=int).reshape(-1, 4)
    if not len(quads):
        return np.zeros((len(x), 0), dtype=int)
    p0, p1, p2, p3 = (x[:, quads[:, k]] for k in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.sign(np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))).astype(int)


def _orbit_ids(report, n_atoms):
    ids = list(range(n_atoms))
    for orbit in report.orbits:
        for atom in orbit:
            ids[int(atom)] = int(min(orbit))
    return ids


def detect_stereocenters(model_topology, report, reference_positions):
    """Stereocenters of the generator atoms and their reference signs.

    Parameters
    ----------
    model_topology : aimmd.cgaimmd.preparation.topology.ModelTopology
        The generator atoms.
    report : aimmd.cgaimmd.preparation.symmetry.SymmetryReport
        Their symmetry analysis (orbits).
    reference_positions : numpy.ndarray
        ``(M, 3)`` Å reference structure (whole).

    Returns
    -------
    list of tuple
        ``[((n1, center, n2, n3), expected_sign), ...]``.

    Notes
    -----
    A stereocenter has exactly four bonded neighbours lying in four distinct
    orbits of the labelled-graph automorphism group. The improper orders the
    neighbours by the first atom of their orbits (their own index order when
    each is the first of its orbit) and uses the first three: an automorphism
    carries that order along, so no relabelling the canonicalization may pick
    changes a center's sign. Centers with a planar reference (sign 0) are
    skipped.
    """
    n_atoms = model_topology.n_atoms
    ids = _orbit_ids(report, n_atoms)
    neighbours = _adjacency(n_atoms, np.asarray(model_topology.bonds).reshape(-1, 2))
    x = np.asarray(reference_positions, dtype=float)[None]
    out = []
    for center in range(n_atoms):
        bonded = neighbours[center]
        if len(bonded) != 4 or len({ids[a] for a in bonded}) != 4:
            continue
        ordered = sorted(bonded, key=lambda a: ids[a])
        quad = (int(ordered[0]), int(center), int(ordered[1]), int(ordered[2]))
        sign = int(_improper_signs(x, [quad])[0, 0])
        if sign:
            out.append((quad, sign))
    return out


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------

def _pair_class(a, b):
    x, y = sorted((_SYMBOLS.get(int(a), str(int(a))), _SYMBOLS.get(int(b), str(int(b)))))
    return f'{x}{y}'


class TopologyValidity:
    """``cgaimmd_validity_function``: bond graph, relabelling, chirality and clashes of raw samples.

    Parameters
    ----------
    topology : str
        GROMACS topology of the full system.
    descriptors : aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors
        The generator atoms (and the structure the reference signs come from).
    node_labels : array_like of int, optional
        The network's tokens (``network.atom_tokens_numpy()``); isomorphisms must preserve them.
    bond_cutoff_scale : float, optional
        Bond rule ``d < scale * (r_cov,i + r_cov,j)``.
    stereocenters : str or sequence, optional
        ``'auto'`` (detected), explicit ``[(n1, center, n2, n3), ...]``, or ``()`` (no chirality check).
    clash_reference : sequence of str, optional
        Unbiased MD trajectory files of the full system (generator atoms taken with ``descriptors.select``).
    clash_margin : float, optional
        Å subtracted from the reference minimum of each element pair.
    clash_fallback_scale : float, optional
        Without reference frames, the clash distance is this times the sum of the Bondi radii.
    vf2_timeout : float, optional
        Seconds allowed for the VF2 search of one sample.
    symmetry_max_factor_order : int, optional
        ``max_factor_order`` of the symmetry analysis behind ``stereocenters='auto'``.
    symmetry_twins_only_min_atoms : int, optional
        ``twins_only_min_atoms`` of that analysis (None: the full orbit search; see Notes).
    reflect_mirror_images : bool, optional
        False (default): a sample with inverted stereocenters is rejected (``'chirality'``). True: a sample that
        fails the chirality check but whose mirror image passes it at EVERY stereocenter is replaced by that
        mirror image (reflected through the yz plane, x -> -x) and checked on; ``info[n]['reflected']`` is True. Exact for a generator that is
        equivariant under reflections (E(3), no parity breaking) with a parity-symmetric prior: a sample and its
        mirror image are equally likely outputs of the same request. Partial inversions stay ``'chirality'``.

    Raises
    ------
    ValueError
        If the topology's generator atoms do not match the descriptors (count, solute names), an element has no tabulated radius, or an explicit stereocenter is invalid or planar.
    NotImplementedError
        For virtual sites among the generator atoms.

    Notes
    -----
    The relabelling may apply any automorphism of the graph labelled by
    (element, `node_labels`). With the network's tokens and orbit ties (the
    reference model) that group is the force-field group the stereocenters
    are detected with, and the chirality check is exact; with element-only
    labels it can be larger, and an element-graph automorphism that is no
    force-field symmetry could change a center's sign (none in NANMA or
    chignolin).

    ``stereocenters='auto'`` runs :func:`aimmd.cgaimmd.preparation.symmetry.analyse` at
    construction. Its full orbit search does not finish on highly symmetric
    molecules (the 196-atom calixarene-G4 host not in 300 s), so pass the
    network's ``symmetry_twins_only_min_atoms`` (and
    ``symmetry_max_factor_order``), or ``stereocenters=()`` or an explicit
    list. Twins-only orbits are finer than the true orbits: a center found
    with them may have neighbours that an element-graph automorphism
    exchanges (a ring flip), so the check is then exact only with the
    network's tokens, tied along the same orbits, as `node_labels`: they
    restrict the relabelling to twin swaps.

    The topology's names and residue names of the solute atoms must equal
    those of the first atoms of ``descriptors.structure``, where the
    reference signs and clash minima come from.

    ``settings`` holds every setting and the clash thresholds; every
    per-sample ``info`` dict references it, so the sidecar records it.

    Examples
    --------
    >>> _validity = TopologyValidity('topol.top', _descriptors,          # doctest: +SKIP
    ...                              node_labels=network.atom_tokens_numpy(),
    ...                              clash_reference=['free_md_A.xtc', 'free_md_B.xtc'])
    >>> def cgaimmd_validity_function(positions, context):                  # doctest: +SKIP
    ...     return _validity(positions, context)
    """

    def __init__(self, topology, descriptors, *, node_labels=None, bond_cutoff_scale=_BOND_SCALE,
                 stereocenters='auto', clash_reference=None, clash_margin=0.10,
                 clash_fallback_scale=0.75, vf2_timeout=10.0, symmetry_max_factor_order=720,
                 symmetry_twins_only_min_atoms=None, reflect_mirror_images=False):
        import MDAnalysis as mda

        from .symmetry import analyse
        from .topology import model_topology, topological_distances

        self.topology = os.path.abspath(str(topology))
        self.descriptors = descriptors
        generator_atoms = descriptors.generator_atoms
        cluster = generator_atoms.cluster
        mt = model_topology(self.topology, n_solute=generator_atoms.n_solute,
                            n_waters=generator_atoms.n_waters,
                            water_resname=cluster.water_resname if cluster else 'SOL',
                            atoms_per_water=cluster.atoms_per_water if cluster else 3)
        if mt.n_atoms != descriptors.n_atoms:
            raise ValueError(f'{self.topology} gives {mt.n_atoms} generator atoms, the descriptors '
                             f'{descriptors.n_atoms}')
        with_structure = mda.Universe(descriptors.structure)
        S = generator_atoms.n_solute
        names = with_structure.atoms.names[:S]
        resnames = with_structure.atoms.resnames[:S]
        mismatched = [k for k, record in enumerate(mt.records[:S])
                      if record.name != names[k] or record.resname != resnames[k]]
        if mismatched:
            k = mismatched[0]
            raise ValueError(
                f'{self.topology} and {descriptors.structure} disagree on the solute: atom {k + 1} '
                f'is {mt.records[k].resname} {mt.records[k].name} in the topology, {resnames[k]} '
                f'{names[k]} in the structure ({len(mismatched)} of {S} atoms differ)')
        self.n_atoms = mt.n_atoms
        elements = mt.elements
        if np.any(elements == 0):
            raise NotImplementedError('virtual sites among the generator atoms are not supported by '
                                      'the validity checks')
        missing = sorted({int(z) for z in elements if int(z) not in _VDW_RADII})
        if missing:
            raise ValueError(f'no van der Waals radius for elements {missing}')
        self._elements = elements
        tokens = None if node_labels is None else np.asarray(node_labels, dtype=np.int64)
        self._canonicalizer = Canonicalizer(elements, mt.bonds, tokens=tokens,
                                            vf2_timeout=vf2_timeout,
                                            cutoff_scale=bond_cutoff_scale)
        reference = descriptors.select(with_structure.atoms.positions, with_structure.dimensions)
        # stereocenters
        if isinstance(stereocenters, str):
            if stereocenters != 'auto':
                raise ValueError(f"stereocenters must be 'auto', a list or (), "
                                 f"got {stereocenters!r}")
            report = analyse(mt, max_factor_order=int(symmetry_max_factor_order),
                             twins_only_min_atoms=symmetry_twins_only_min_atoms)
            centers = detect_stereocenters(mt, report, reference)
            ids = _orbit_ids(report, mt.n_atoms)
            group_of = [ids[quad[1]] for quad, _ in centers]
        else:
            quads = [tuple(int(i) for i in quad) for quad in stereocenters]
            for quad in quads:
                if (len(quad) != 4 or len(set(quad)) != 4 or min(quad) < 0
                        or max(quad) >= mt.n_atoms):
                    raise ValueError(f'stereocenter {quad} must be four distinct generator atoms '
                                     f'(0 to {mt.n_atoms - 1})')
            signs = _improper_signs(reference[None], quads)[0] if quads else []
            centers = [(quad, int(sign)) for quad, sign in zip(quads, signs)]
            planar = [quad for quad, sign in centers if sign == 0]
            if planar:
                raise ValueError(f'stereocenters {planar} are planar in the reference structure '
                                 '(sign 0): every sample would be rejected')
            group_of = list(range(len(centers)))
        self.stereocenters = centers
        self._quads = np.array([quad for quad, _ in centers], dtype=int).reshape(-1, 4)
        self._center_groups = []
        for group in sorted(set(group_of)):
            members = [k for k, g in enumerate(group_of) if g == group]
            self._center_groups.append((np.array(members),
                                        sorted(centers[k][1] for k in members)))
        # clashes: non-bonded heavy pairs, by element pair
        distances = topological_distances(mt.n_atoms, mt.bonds, max_distance=4)
        heavy = elements != 1
        i, j = np.triu_indices(mt.n_atoms, 1)
        keep = (distances[i, j] >= 4) & heavy[i] & heavy[j]
        self._pairs = np.stack([i[keep], j[keep]], 1)
        classes = [_pair_class(elements[a], elements[b]) for a, b in self._pairs]
        self._class_names = sorted(set(classes))
        self._pair_class = np.array([self._class_names.index(c) for c in classes], dtype=int)
        self._fallback = not clash_reference
        if self._fallback:
            thresholds = {}
            for (a, b), c in zip(self._pairs, classes):
                thresholds[c] = float(clash_fallback_scale) * (_VDW_RADII[int(elements[a])]
                                                               + _VDW_RADII[int(elements[b])])
        else:
            files = [os.path.abspath(str(f)) for f in clash_reference]
            universe = mda.Universe(descriptors.structure, files)
            minima = np.full(len(self._class_names), np.inf)
            for ts in universe.trajectory:
                selected = descriptors.select(ts.positions, ts.dimensions)
                minima = np.minimum(minima, self._class_minima(selected[None])[0])
            thresholds = {c: float(m) - float(clash_margin)
                          for c, m in zip(self._class_names, minima)}
            clash_reference = files
        self._thresholds = np.array([thresholds[c] for c in self._class_names])
        self.settings = {
            'function': 'aimmd.cgaimmd.preparation.validity.TopologyValidity',
            'topology': self.topology, 'structure': descriptors.structure,
            'node_labels': None if tokens is None else 'given',
            'bond_cutoff_scale': float(bond_cutoff_scale),
            'stereocenters': ('auto' if isinstance(stereocenters, str)
                              else [list(q) for q, _ in centers]),
            'n_stereocenters': len(centers),
            'clash_reference': None if self._fallback else list(clash_reference),
            'clash_margin': float(clash_margin),
            'clash_fallback_scale': float(clash_fallback_scale),
            'clash_thresholds_A': {c: round(float(v), 4) for c, v in thresholds.items()},
            'vf2_timeout': float(vf2_timeout),
            'symmetry_max_factor_order': int(symmetry_max_factor_order),
            'symmetry_twins_only_min_atoms': (None if symmetry_twins_only_min_atoms is None
                                              else int(symmetry_twins_only_min_atoms)),
            'reflect_mirror_images': bool(reflect_mirror_images),
        }

    def _class_minima(self, x):
        """``(n, n_classes)`` minimum distance per element-pair class (inf for none)."""
        out = np.full((len(x), len(self._class_names)), np.inf)
        if not len(self._pairs):
            return out
        d = np.linalg.norm(x[:, self._pairs[:, 0]] - x[:, self._pairs[:, 1]], axis=-1)
        for c in range(len(self._class_names)):
            members = self._pair_class == c
            out[:, c] = d[:, members].min(axis=1)
        return out

    def _handed(self, y):
        """Whether every stereocenter group of ``y`` ``(M, 3)`` has the reference signs."""
        signs = _improper_signs(y[None], self._quads)[0]
        return all(sorted(signs[members].tolist()) == expected for members, expected in self._center_groups)

    def __call__(self, positions, context=None):
        """Check a batch of raw samples.

        Parameters
        ----------
        positions : numpy.ndarray
            ``(n, M, 3)`` (or ``(M, 3)``) Å raw generator output.
        context : aimmd.cgaimmd.types.CandidateContext, optional
            Its ``log`` receives the once-per-process clash fallback message.

        Returns
        -------
        ValidityResult
            Positions relabelled to topology order (unchanged where no isomorphism was found), ok flags, reasons and per-sample info.

        Raises
        ------
        ValueError
            For samples with another atom count than the generator atoms.
        """
        x = np.asarray(positions, dtype=float)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != (self.n_atoms, 3):
            raise ValueError(f'expected (n, {self.n_atoms}, 3) samples, got {x.shape}')
        if self._fallback and _once.once(('validity clash fallback', self.topology)):
            message = (f'C-GAIMMD validity: no clash_reference frames; clash distances are the '
                       f'fallback {self.settings["clash_fallback_scale"]:g} x Bondi radii '
                       f'({self.settings["clash_thresholds_A"]})')
            (context.log if context is not None else print)(message)
        out = x.copy()
        ok = np.zeros(len(x), dtype=bool)
        reason = np.array([''] * len(x), dtype=object)
        info = []
        for n in range(len(x)):
            perm, why = self._canonicalizer(x[n])
            record = {'permutation_is_identity': None, 'min_nonbonded_A': None,
                      'n_stereocenters': len(self.stereocenters), 'reflected': False, 'settings': self.settings}
            info.append(record)
            if perm is None:
                reason[n] = why
                continue
            y = x[n][perm]
            out[n] = y
            record['permutation_is_identity'] = bool(np.array_equal(perm, np.arange(self.n_atoms)))
            minima = self._class_minima(y[None])[0]
            record['min_nonbonded_A'] = float(minima.min()) if len(minima) else None
            if len(self._quads) and not self._handed(y):
                mirrored = y * _MIRROR
                if not (self.settings['reflect_mirror_images'] and self._handed(mirrored)):
                    reason[n] = 'chirality'
                    continue
                y = mirrored                # the full mirror image, reflected back (see reflect_mirror_images)
                out[n] = y
                record['reflected'] = True
            if np.any(minima < self._thresholds):
                reason[n] = 'clash'
                continue
            ok[n] = True
        return ValidityResult(positions=out, ok=ok, reason=reason, info=info)
