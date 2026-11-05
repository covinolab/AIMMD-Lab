"""
aimmd.cgaimmd.preparation.descriptors
=====================================

The generator atoms and their descriptors.

The *generator atoms* are the fixed-size, ordered atom set a generator sees
and generates: the first ``S`` atoms of ``params.topology`` (the solute; the
whole system in vacuum) and, for a *cluster*, ``K`` waters chosen per frame
by ranking rules. Under ``chain_type='cgaimmd'`` the ``descriptors_function``
MUST return their flat Ångström coordinates; :class:`GeneratorDescriptors`
computes them from full-system MD frames, making molecules whole across the
periodic box, and :func:`coordinates_descriptors` is the plain version for
systems that need no whole-making (the toy engine).

Order of the generator atoms: the solute in topology order, then the ``K``
waters in the order the ranking chose them (oxygen first). The waters are a
set: a network used with a cluster MUST be invariant to their order (tokens
tie all waters; the OT pairing permutes whole waters).

Whole-making (``whole=True``): each solute molecule is made whole by a walk
along its bonds from its first atom (exact while every bond is shorter than
half the box; the minimum image against one atom leaves extended peptides
broken). A single molecule stays where its first atom is. Several molecules
are placed in two passes: every molecule after the first is translated by the
lattice vector that brings its center of mass nearest that of the first
molecule, then every molecule (the first included) by the lattice vector that
brings its center of mass nearest the center (the solute's center of mass, or
that of ``center_selection``, after the first pass). Every chosen water is
made whole around its oxygen and translated so that its oxygen is nearest the
center. The cluster is then whole and contiguous in Cartesian space and needs
no periodic images.

Port of ``cfaimmd.solvation.EnvironmentSelector`` and
``cfaimmd.geometry.make_whole`` (the prototype of C-GAIMMD) without the
variable-count environment block (a generator needs a fixed atom count).
Standard library and numpy at import; MDAnalysis inside the constructor.
"""

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ['RankingRule', 'ClusterSelection', 'GeneratorAtoms', 'GeneratorDescriptors',
           'coordinates_descriptors', 'make_whole']

_MODES = ('min_distance', 'com')
_RULE_KEYS = ('reference', 'mode', 'n')

#: covalent radii (Å, Cordero et al. 2008; Br 1.20) of the bond rule d < 1.3 (r_i + r_j)
_COVALENT_RADII = {
    1: 0.31, 3: 1.28, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 11: 1.66, 12: 1.41,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 19: 2.03, 20: 1.76, 26: 1.32, 29: 1.32, 30: 1.22,
    35: 1.20, 53: 1.39,
}
_BOND_SCALE = 1.30


# ---------------------------------------------------------------------------
# the generator atoms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingRule(Mapping):
    """One ranking rule of a :class:`ClusterSelection`, a frozen and hashable read-only mapping.

    ``rule['mode']``, ``dict(rule)`` and ``rule.get('n')`` work as for the
    dict it replaces; :class:`ClusterSelection` turns its ``ranking`` dicts
    into these, which makes it (and :class:`GeneratorAtoms`) hashable.
    """

    reference: Optional[str] = None
    """MDAnalysis selection of solute atoms, or None for the solute heavy atoms."""

    mode: str = 'min_distance'
    """``'min_distance'`` (to the nearest reference atom) or ``'com'`` (to their center of mass)."""

    n: int = -1
    """Waters the rule takes; -1 (the last rule only) takes the rest up to ``K``."""

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f'ranking mode must be one of {_MODES}, got {self.mode!r}')
        object.__setattr__(self, 'n', int(self.n))

    def __getitem__(self, key):
        if key not in _RULE_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(_RULE_KEYS)

    def __len__(self):
        return len(_RULE_KEYS)


@dataclass(frozen=True)
class ClusterSelection:
    """How the ``K`` cluster waters are chosen in a full-system frame."""

    n_waters: int
    """``K >= 1``, the number of waters among the generator atoms."""

    water_resname: str = 'SOL'
    """Residue name of the water model."""

    atoms_per_water: int = 3
    """Atoms per water molecule, oxygen first."""

    ranking: tuple = ()
    """Ranking rules, dicts (or :class:`RankingRule`) with ``'reference'`` (an MDAnalysis selection of solute atoms, or None for the solute heavy atoms), ``'mode'`` and ``'n'``, stored as :class:`RankingRule`; see Notes of :class:`GeneratorDescriptors`."""

    center_selection: Optional[str] = None
    """MDAnalysis selection of solute atoms whose center of mass the cluster is made whole around; None means the solute."""

    def __post_init__(self):
        if int(self.n_waters) < 1:
            raise ValueError(f'n_waters must be >= 1, got {self.n_waters}')
        if int(self.atoms_per_water) < 1:
            raise ValueError(f'atoms_per_water must be >= 1, got {self.atoms_per_water}')
        rules = []
        for rule in self.ranking:
            if not isinstance(rule, RankingRule):
                rule = dict(rule)
                unknown = set(rule) - set(_RULE_KEYS)
                if unknown:
                    raise ValueError(f'unknown ranking keys {sorted(unknown)}')
                rule = RankingRule(rule.get('reference'), rule.get('mode', 'min_distance'),
                                   rule.get('n', -1))
            rules.append(rule)
        for rule in rules[:-1]:
            if rule.n < 1:
                raise ValueError(f'every ranking rule but the last needs n >= 1, got {rule.n}')
        if rules and rules[-1].n != -1:
            raise ValueError(f'the last ranking rule must have n = -1 (the rest), got '
                             f'{rules[-1].n}')
        object.__setattr__(self, 'n_waters', int(self.n_waters))
        object.__setattr__(self, 'atoms_per_water', int(self.atoms_per_water))
        object.__setattr__(self, 'ranking', tuple(rules))


@dataclass(frozen=True)
class GeneratorAtoms:
    """The generator atoms: the first ``S`` atoms of the topology, plus an optional cluster of waters."""

    n_solute: int
    """``S``, the solute, i.e. the first ``S`` atoms of ``params.topology``."""

    cluster: Optional[ClusterSelection] = None
    """The cluster waters, or None (vacuum and solute-only systems)."""

    def __post_init__(self):
        if int(self.n_solute) < 1:
            raise ValueError(f'n_solute must be >= 1, got {self.n_solute}')
        object.__setattr__(self, 'n_solute', int(self.n_solute))

    @property
    def n_waters(self):
        """``K``, or 0 without a cluster."""
        return 0 if self.cluster is None else self.cluster.n_waters


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def make_whole(positions, box=None, reference_atom=0):
    """Bring every atom into the periodic image of one reference atom.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(..., N, 3)`` coordinates.
    box : array_like, optional
        Orthorhombic box lengths ``(3,)`` or MDAnalysis dimensions ``(6,)``, in the units of `positions`.
    reference_atom : int, optional
        The atom whose image the others are brought to.

    Returns
    -------
    numpy.ndarray
        float64 coordinates; `positions` itself when `box` is None or not a valid box.

    Notes
    -----
    GROMACS runs vacuum systems in a large periodic box and writes coordinates
    inside it, so a molecule that drifts across a face is written split in two
    (12.7 % of the NANMA frames of the C-GAIMMD development project). The minimum
    image against one atom is exact for any molecule smaller than half the
    box; :class:`GeneratorDescriptors` walks along bonds instead, which is exact
    for longer molecules too.
    """
    if box is None:
        return positions
    lengths = np.asarray(box, dtype=float).reshape(-1)[:3]
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        return positions
    x = np.asarray(positions, dtype=float)
    reference = x[..., reference_atom: reference_atom + 1, :]
    return x - np.round((x - reference) / lengths) * lengths


def _box_lengths(dimensions):
    """Orthorhombic box lengths (3,) or None (no or invalid box); the box helper of the preparation tools.

    Raises ValueError for a triclinic box (the preparation tools support
    orthorhombic boxes only).
    """
    if dimensions is None:
        return None
    d = np.asarray(dimensions, dtype=float).reshape(-1)
    if d.size < 3 or not np.all(np.isfinite(d[:3])) or np.any(d[:3] <= 0):
        return None
    if d.size >= 6 and not np.allclose(d[3:6], 90.0, atol=1e-3):
        raise ValueError(f'the C-GAIMMD preparation tools support orthorhombic boxes only, got '
                         f'angles {d[3:6]}')
    return d[:3]


def _min_image(v, lengths):
    """Minimum-image vectors in an orthorhombic box (`lengths` None: `v` as it is)."""
    return v if lengths is None else v - lengths * np.round(v / lengths)


def _elements_from_masses(masses):
    """Element per mass, 0 where no element is near (the tolerant mode of the topology's rule)."""
    from .topology import _masses_to_atomic_numbers
    return _masses_to_atomic_numbers(masses, strict=False)


def _covalent_pairs(positions, elements, scale=_BOND_SCALE, lengths=None, default=None):
    """(E, 2) pairs i < j closer than scale * (r_i + r_j); element 0 never bonds."""
    x = np.asarray(positions, dtype=float)
    z = np.asarray(elements, dtype=int)
    radii = np.array([_COVALENT_RADII.get(int(e), default if default is not None else np.nan)
                      if e > 0 else np.nan for e in z])
    if default is None and np.any(np.isnan(radii[z > 0])):
        missing = sorted({int(e) for e, r in zip(z, radii) if e > 0 and np.isnan(r)})
        raise ValueError(f'no covalent radius for elements {missing}')
    out = []
    for start in range(0, len(x), 256):                  # blocks keep memory flat
        stop = min(start + 256, len(x))
        d = np.linalg.norm(_min_image(x[start:stop, None, :] - x[None, :, :], lengths), axis=-1)
        threshold = scale * (radii[start:stop, None] + radii[None, :])
        i, j = np.nonzero((d < threshold) & (d > 0))
        i = i + start
        keep = i < j
        out.append(np.stack([i[keep], j[keep]], 1))
    return np.concatenate(out).astype(int) if out else np.zeros((0, 2), dtype=int)


def _walk_plan(n_atoms, bonds):
    """Connected components as ``(atoms, root, levels)``; levels are breadth-first (child, parent) arrays."""
    from collections import deque

    neighbours = [[] for _ in range(n_atoms)]
    for i, j in np.asarray(bonds, dtype=int).reshape(-1, 2):
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    seen = np.zeros(n_atoms, dtype=bool)
    plans = []
    for root in range(n_atoms):
        if seen[root]:
            continue
        seen[root] = True
        depth, order, by_level = {root: 0}, [root], {}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in sorted(neighbours[u]):
                if not seen[v]:
                    seen[v] = True
                    depth[v] = depth[u] + 1
                    order.append(v)
                    by_level.setdefault(depth[v], []).append((v, u))
                    queue.append(v)
        levels = [(np.array([c for c, _ in pairs]), np.array([p for _, p in pairs]))
                  for _, pairs in sorted(by_level.items())]
        plans.append((np.sort(np.asarray(order)), root, levels))
    return plans


def _com(x, masses):
    total = masses.sum()
    if total <= 0:
        return x.mean(axis=-2)
    return (x * masses[..., :, None]).sum(axis=-2) / total


# ---------------------------------------------------------------------------
# the descriptors
# ---------------------------------------------------------------------------

class GeneratorDescriptors:
    """``descriptors_function`` of C-GAIMMD: full-system frames to flat generator-atom coordinates.

    Parameters
    ----------
    structure : str
        Structure file of the full system (``params.topology``: ``.gro``, ``.pdb``, ``.tpr``, ...).
    generator_atoms : GeneratorAtoms
        The generator atoms.
    whole : bool, optional
        Make molecules whole and place the cluster next to its center (False: coordinates as stored).
    masses : array_like, optional
        ``(N,)`` masses (u) of the system instead of the structure's, e.g. ``aimmd.cgaimmd.preparation.topology.read_masses(top)``.

    Raises
    ------
    ValueError
        If the structure does not fit the generator atoms (too few atoms or waters, references outside the solute) or `masses` does not fit the structure.

    Notes
    -----
    Ranking rules (``ClusterSelection.ranking``, applied in order): each
    takes the ``n`` not-yet-chosen waters whose oxygen is nearest its
    reference, by the minimum distance to the reference atoms
    (``'min_distance'``) or the distance to their center of mass (``'com'``);
    the last rule has ``n = -1`` (the rest up to ``K``). With several
    references (a host cavity and a guest) each region stays represented even
    when another one would claim all ``K`` waters. ``'reference': None`` and
    the default ``ranking=()`` mean the solute heavy atoms (mass above 1.5).
    References and the center selection MUST be solute atoms, because
    :meth:`cluster_radii` and the solvator recompute the rules on the
    generator atoms alone. Distances use the minimum image of each frame's box, ties
    keep the lower water index. The rules are stored as :class:`RankingRule`.

    The solute's molecules are the connected components of the bonds of
    `structure`: its own bonds when the file has them (a ``.tpr``) and they
    cover every solute atom the covalent rule bonds, else the pairs closer
    than 1.3 × the sum of their covalent radii in the structure itself
    (minimum image, elements from the masses); a file with partial bonds (a
    PDB with ``CONECT`` records for a ligand only) thus falls back to the
    covalent rule. Several solute molecules are placed in two passes (see
    the module docstring): next to the first molecule, then next to the
    center. The center is computed on the whole-made solute.

    Masses are those of the structure's MDAnalysis Universe, exact for a
    ``.tpr`` but guessed from atom names otherwise (NANMA's CHARMM carbons
    ``CL``/``CLP`` become chlorine), unless `masses` gives the topology's
    (the masses the minimizer and the solvator use). They enter the centers
    of mass and the elements of the covalent rule, so a ``.tpr`` structure
    or `masses` is recommended whenever ``'com'`` rules or several solute
    molecules are used.

    Called on a trajectory, the descriptors work frame by frame and keep only
    the ``(M, 3)`` generator atoms of each frame (AIMMD passes batches of up to
    4096 full-system frames). Deterministic; creates no CUDA context.

    Examples
    --------
    >>> descriptors = GeneratorDescriptors('conf.gro', GeneratorAtoms(n_solute=22))  # doctest: +SKIP
    >>> def descriptors_function(trajectory):                              # doctest: +SKIP
    ...     return descriptors(trajectory)
    """

    def __init__(self, structure, generator_atoms, *, whole=True, masses=None):
        import MDAnalysis as mda

        if not isinstance(generator_atoms, GeneratorAtoms):
            raise TypeError(f'generator_atoms must be a GeneratorAtoms, got {type(generator_atoms).__name__}')
        self.structure = os.path.abspath(str(structure))
        self.generator_atoms = generator_atoms
        self.whole = bool(whole)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            universe = mda.Universe(self.structure)
        atoms = universe.atoms
        self._n_total = int(atoms.n_atoms)
        S = generator_atoms.n_solute
        if S > self._n_total:
            raise ValueError(f'{self.structure} has {self._n_total} atoms, fewer than '
                             f'n_solute = {S}')
        if masses is None:
            try:
                masses = np.asarray(atoms.masses, dtype=float)
            except Exception:                                     # noqa: BLE001
                masses = np.ones(self._n_total)
        else:
            masses = np.asarray(masses, dtype=float).reshape(-1)
            if masses.shape != (self._n_total,):
                raise ValueError(f'masses has {masses.size} values, {self.structure} '
                                 f'{self._n_total} atoms')
            if not np.all(np.isfinite(masses)) or np.any(masses < 0):
                raise ValueError('masses must be finite and >= 0')
        self._masses = masses
        box = _box_lengths(universe.dimensions)
        solute_positions = np.asarray(atoms.positions[:S], dtype=float)
        bonds = self._solute_bonds(universe, solute_positions, masses[:S], box)
        self._walk = _walk_plan(S, bonds)
        self._heavy = np.flatnonzero(masses[:S] > 1.5)
        self._solute_masses = masses[:S]
        cluster = generator_atoms.cluster
        self._rules, self._center, self._water_atoms = [], None, None
        if cluster is not None:
            water = universe.select_atoms(f'resname {cluster.water_resname}').indices
            apw = cluster.atoms_per_water
            if len(water) == 0 or len(water) % apw:
                raise ValueError(f'resname {cluster.water_resname}: {len(water)} atoms in '
                                 f'{self.structure}, not a multiple of atoms_per_water = {apw}')
            if water.min() < S:
                raise ValueError(f'water atoms among the first n_solute = {S} atoms of '
                                 f'{self.structure}')
            self._water_atoms = water.reshape(-1, apw)
            if not np.all(np.abs(masses[self._water_atoms[:, 0]] - 16.0) < 0.5):
                raise ValueError('the first atom of every water residue must be its oxygen')
            if cluster.n_waters > len(self._water_atoms):
                raise ValueError(f'n_waters = {cluster.n_waters} exceeds the '
                                 f'{len(self._water_atoms)} waters of {self.structure}')
            ranking = cluster.ranking or ({'reference': None, 'mode': 'min_distance', 'n': -1},)
            for rule in ranking:
                reference = rule['reference']
                if reference is None:
                    indices = self._heavy
                else:
                    indices = universe.select_atoms(reference).indices
                    if len(indices) == 0 or indices.max() >= S:
                        raise ValueError(f'ranking reference {reference!r} must select solute '
                                         f'atoms (the first {S})')
                self._rules.append({'indices': np.asarray(indices, dtype=int),
                                    'mode': rule['mode'], 'n': rule['n']})
            if cluster.center_selection is not None:
                center = universe.select_atoms(cluster.center_selection).indices
                if len(center) == 0 or center.max() >= S:
                    raise ValueError(f'center_selection {cluster.center_selection!r} must select '
                                     f'solute atoms (the first {S})')
                self._center = np.asarray(center, dtype=int)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _solute_bonds(universe, positions, masses, box):
        """The file's solute bonds if they cover every atom the covalent rule bonds, else the rule's."""
        S = len(positions)
        covalent = _covalent_pairs(positions, _elements_from_masses(masses), lengths=box,
                                   default=0.8)
        try:
            bonds = np.asarray(universe.bonds.indices, dtype=int).reshape(-1, 2)
        except Exception:                                         # noqa: BLE001
            bonds = None
        if bonds is not None and len(bonds):
            own = bonds[(bonds < S).all(axis=1)]
            if np.isin(np.unique(covalent), np.unique(own)).all():
                return own
        return covalent

    # -- public ---------------------------------------------------------------

    @property
    def flow_atoms(self):
        """The old name of :attr:`generator_atoms`, read for one release (with a ``DeprecationWarning``)."""
        from .. import _legacy
        new = _legacy.ATTRIBUTES[('GeneratorDescriptors', 'flow_atoms')]
        _legacy.warn_renamed('GeneratorDescriptors.flow_atoms', f'GeneratorDescriptors.{new}',
                             key=('attribute', 'GeneratorDescriptors', 'flow_atoms'))
        return getattr(self, new)

    @property
    def n_atoms(self):
        """``M = S + K * atoms_per_water``, the number of generator atoms."""
        cluster = self.generator_atoms.cluster
        return self.generator_atoms.n_solute + (0 if cluster is None
                                           else cluster.n_waters * cluster.atoms_per_water)

    def select(self, positions, dimensions):
        """The generator atoms of one full-system frame.

        Parameters
        ----------
        positions : numpy.ndarray
            ``(N, 3)`` Å, the full frame in topology order.
        dimensions : array_like or None
            The frame's box ``(6,)`` (Å, degrees); None: no periodic images.

        Returns
        -------
        numpy.ndarray
            ``(M, 3)`` float64 Å: the solute made whole, then the cluster waters in ranking order.

        Raises
        ------
        ValueError
            For a frame with another atom count than the structure, or a non-orthorhombic box.
        """
        return self._select_one(np.asarray(positions, dtype=float), dimensions)

    def cluster_radii(self, generator_positions):
        """Per ranking rule, the largest reference distance among the waters that rule chose.

        Parameters
        ----------
        generator_positions : numpy.ndarray
            ``(M, 3)`` Å generator atoms (whole, as :meth:`select` returns them).

        Returns
        -------
        list of float
            One radius (Å) per rule; 0.0 for a rule that chose no water.

        Raises
        ------
        ValueError
            Without a cluster.

        Notes
        -----
        The rules are applied again to the ``K`` waters of the cluster (the
        greedy selection of :meth:`select`), so the radii do not depend on the
        order of the waters: a generated cluster, or one relabelled by
        :class:`~aimmd.cgaimmd.preparation.validity.TopologyValidity`, holds them in any
        order. In ranking order the result is the selection's own radii. In
        the source frame no unchosen water was closer to a rule's reference
        than that rule's radius, which is the solvator's selection consistency
        test.
        """
        if self.generator_atoms.cluster is None:
            raise ValueError('cluster_radii needs generator atoms with a cluster')
        x = np.asarray(generator_positions, dtype=float)
        if x.shape != (self.n_atoms, 3):
            raise ValueError(f'expected ({self.n_atoms}, 3) generator-atom positions, got {x.shape}')
        S, apw = self.generator_atoms.n_solute, self.generator_atoms.cluster.atoms_per_water
        oxygens = x[S::apw][:self.generator_atoms.n_waters]
        distances = self._rule_distances(oxygens, x[:S], None)
        return [float(distances[r][pick].max()) if len(pick) else 0.0
                for r, pick in enumerate(self._rank(distances))]

    def __call__(self, trajectory):
        """AIMMD ``descriptors_function``: a trajectory to ``(n_frames, 3 M)`` float32 Å.

        Parameters
        ----------
        trajectory : iterable of MDAnalysis Timestep, or one Timestep
            The frames (AIMMD passes a reader).

        Returns
        -------
        numpy.ndarray
            ``(n_frames, 3 M)`` float32; ``(0, 3 M)`` for empty input.
        """
        rows = [self._select_one(x, dimensions).reshape(-1).astype(np.float32)
                for x, dimensions in _iter_frames(trajectory)]
        if not rows:
            return np.zeros((0, 3 * self.n_atoms), dtype=np.float32)
        return np.stack(rows)

    # -- internals --------------------------------------------------------------

    def _rule_counts(self):
        """Waters each rule takes (the last takes the rest up to K)."""
        K, counts = self.generator_atoms.n_waters, []
        for rule in self._rules:
            left = K - sum(counts)
            counts.append(left if rule['n'] < 0 else min(rule['n'], left))
        return counts

    def _rank(self, distances):
        """The waters each rule takes, in order: the greedy selection of the ranking rules.

        Rule ``r`` takes the ``_rule_counts()[r]`` waters nearest its reference
        among those no earlier rule took (a stable sort: ties keep the lower
        water index). Returns one index array per rule.
        """
        taken = np.zeros(distances.shape[1], dtype=bool)
        picks = []
        for r, count in enumerate(self._rule_counts()):
            if not count:
                picks.append(np.zeros(0, dtype=int))
                continue
            d = np.where(taken, np.inf, distances[r])
            pick = np.argsort(d, kind='stable')[:count]
            taken[pick] = True
            picks.append(pick)
        return picks

    def _rule_distances(self, oxygens, solute, lengths):
        """``(n_rules, n_waters)`` reference distances of water oxygens (Å)."""
        out = np.empty((len(self._rules), len(oxygens)))
        for r, rule in enumerate(self._rules):
            reference = solute[rule['indices']]
            if rule['mode'] == 'com':
                center = _com(reference, self._solute_masses[rule['indices']])
                out[r] = np.linalg.norm(_min_image(oxygens - center, lengths), axis=-1)
                continue
            d = np.full(len(oxygens), np.inf)
            for start in range(0, len(reference), 64):
                block = reference[start:start + 64]
                dd = np.linalg.norm(_min_image(oxygens[:, None, :] - block[None], lengths),
                                    axis=-1).min(axis=1)
                d = np.minimum(d, dd)
            out[r] = d
        return out

    def _whole_solute(self, solute, lengths):
        """Each molecule whole by the bond walk; molecules placed in the two passes of the module docstring."""
        x = solute.copy()
        for atoms, root, levels in self._walk:
            for children, parents in levels:
                x[children] = x[parents] + _min_image(x[children] - x[parents], lengths)
        masses = self._solute_masses
        if len(self._walk) > 1:
            first = self._walk[0][0]
            anchor = _com(x[first], masses[first])
            for atoms, _, _ in self._walk[1:]:
                com = _com(x[atoms], masses[atoms])
                x[atoms] += _min_image(com - anchor, lengths) - (com - anchor)
        center = self._center_of(x)
        if len(self._walk) > 1:
            for atoms, _, _ in self._walk:
                com = _com(x[atoms], masses[atoms])
                x[atoms] += _min_image(com - center, lengths) - (com - center)
            center = self._center_of(x)
        return x, center

    def _center_of(self, solute):
        """Center of mass of the center atoms of an already whole solute (no minimum image)."""
        if self._center is None:
            return _com(solute, self._solute_masses)
        return _com(solute[self._center], self._solute_masses[self._center])

    def _select_one(self, x, dimensions):
        """``(M, 3)`` float64 generator atoms of one ``(N, 3)`` float64 full-system frame."""
        if x.shape != (self._n_total, 3):
            raise ValueError(f'expected a frame of ({self._n_total}, 3) positions (the atoms of '
                             f'{self.structure}), got {x.shape}')
        S = self.generator_atoms.n_solute
        cluster = self.generator_atoms.cluster
        out = np.empty((self.n_atoms, 3))
        lengths = _box_lengths(dimensions)
        if self.whole and lengths is not None:
            solute, center = self._whole_solute(x[:S], lengths)
        else:
            solute = x[:S]
            center = None                       # waters are placed only when whole
        out[:S] = solute
        if cluster is None:
            return out
        apw = cluster.atoms_per_water
        oxygens = x[self._water_atoms[:, 0]]
        distances = self._rule_distances(oxygens, solute, lengths)
        chosen = [int(i) for pick in self._rank(distances) for i in pick]
        for slot, water in enumerate(chosen):
            block = x[self._water_atoms[water]]
            if self.whole and lengths is not None:
                block = block[0] + _min_image(block - block[0], lengths)
                block = block + (_min_image(block[0] - center, lengths) - (block[0] - center))
            out[S + slot * apw: S + (slot + 1) * apw] = block
        return out


def _iter_frames(trajectory, dtype=float):
    """``(positions (N, 3), dimensions)`` per frame of a trajectory, reader or single Timestep.

    One frame at a time (a copy of each, as a reader reuses its buffer), so
    a batch of full-system frames is never held in memory at once.
    """
    if hasattr(trajectory, 'positions') and np.asarray(trajectory.positions).ndim == 2:
        yield (np.array(trajectory.positions, dtype=dtype),
               getattr(trajectory, 'dimensions', None))
        return
    for ts in trajectory:
        dims = getattr(ts, 'dimensions', None)
        yield (np.array(ts.positions, dtype=dtype),
               None if dims is None else np.array(dims, dtype=float))


def coordinates_descriptors(trajectory):
    """Every atom's coordinates as stored: ``(n_frames, 3 N)`` float32 Å.

    Parameters
    ----------
    trajectory : iterable of MDAnalysis Timestep, or one Timestep
        The frames.

    Returns
    -------
    numpy.ndarray
        ``(n_frames, 3 N)`` float32, ``x0, y0, z0, x1, ...``; empty input gives ``(0, 3 N)``, or ``(0, 0)`` without an atom count (see Notes).

    Notes
    -----
    For systems whose generator atoms are the whole system and need no
    whole-making (the toy engine). Empty input has ``3 N`` columns when it
    knows its atom count (a reader, or an empty slice of one:
    ``trajectory.n_atoms`` or ``trajectory.trajectory.n_atoms``); an empty
    sequence (``[]``) has none to give, so a width check needs a frame.
    """
    rows = [x.reshape(-1) for x, _ in _iter_frames(trajectory, dtype=np.float32)]
    if not rows:
        n_atoms = getattr(trajectory, 'n_atoms', None)
        if n_atoms is None:
            n_atoms = getattr(getattr(trajectory, 'trajectory', None), 'n_atoms', None)
        return np.zeros((0, 3 * int(n_atoms or 0)), dtype=np.float32)
    return np.stack(rows)
