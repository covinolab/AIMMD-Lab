"""
aimmd.cgaimmd.preparation.solvate
=================================

Re-solvation of generated solute-only and cluster candidates with GROMACS:
:class:`GromacsSolvator` is a ``cgaimmd_prepare_function``.

A cluster generator produces the solute and ``K`` waters; AIMMD shoots from
full systems. The solvator puts the candidate back into the run's periodic
system with exactly the composition and atom order of ``params.topology`` and
its ``.top`` (AIMMD runs grompp on them for every shot, and every path of a
run shares one topology):

1. plan (at construction): the topology's solute, water and single-atom ions
   and the box of the structure;
2. place: the solute centroid at the box center;
3. solvate: ``gmx solvate`` around the generator atoms, lowering the vdW scale
   until enough waters fit;
4. selection consistency (clusters): delete every added water that the
   descriptors' own ranking would have chosen instead of a cluster water,
   so the cluster is still what the descriptors select from the solvated
   system;
5. trim: delete random far waters down to the topology's count;
6. ions: ``gmx genion`` replaces bulk waters far from the solute and outside
   an exclusion sphere by the topology's ions;
7. assemble in topology order (the ``K`` cluster waters first);
8. derive the restrained topology (generator atoms, the cluster waters as a
   restrained copy, an optional flat-bottom sphere keeping bulk water out);
9. restrained steepest-descent EM;
10. optional restrained NVT stages (NVT only: ``pcoupl = no`` is forced and
    the box stays the run's);
11. result: the last frame, with the displacement of every generator atom (a
    changed box is a failed candidate).

Frames of the restrained stages are never registered or used as data; the
shooting point gets fresh Maxwell-Boltzmann velocities and the production
topology has no restraints. Preparation cost appears in the sidecar only.

Generalized from the calixarene re-solvation brute-force committor script and
the chignolin commitment-test preparation of the C-GAIMMD development project.
"""

import os
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from ..types import PreparedFrame
from .descriptors import _min_image
from .gromacs import (RESTRAINT_DEFINE, _default_em_mdp, _gromacs_seed, _grompp_command,
                      _gro_records, _mdp_key, _mdp_value, _mdrun_command, _params_grompp, _read_mdp,
                      _with_absolute_path, _work_dir, gmx_executable, mdp_text, parse_em_log,
                      read_trr_last_frame, restraint_mode, run_gmx, sanitize_mdrun, topology_file,
                      write_gro)

__all__ = ['GromacsSolvator', 'NVTStage', 'ExclusionSphere']

#: mdp keys the solvator sets on every NVT stage itself (normalized); overrides may not set them
_NVT_OWN_KEYS = ('nsteps', 'gen_vel', 'gen_seed', 'continuation', 'ld_seed', 'nstxout',
                 'nstvout', 'nstfout')
#: box lengths (Å) and angles (degrees) the last frame may differ from the topology's box
_BOX_TOLERANCE = 1e-3


@dataclass(frozen=True)
class NVTStage:
    """One restrained NVT stage of the solvator."""

    mdp: str
    """Template mdp (the system's NVT or production mdp)."""

    ps: float
    """Length in ps, > 0 (``nsteps = round(ps / dt)`` >= 1; ``dt`` from the overrides, else the template, else 0.001 ps)."""

    overrides: dict = field(default_factory=dict)
    """Extra mdp settings, e.g. ``{'tau_t': 0.1}`` (see the Notes of :class:`GromacsSolvator`)."""

    gen_vel: bool = False
    """Draw velocities (``gen_seed`` from ``context.seed``); set it True on the first stage."""

    def __post_init__(self):
        try:
            ps = float(self.ps)
        except (TypeError, ValueError):
            raise ValueError(f'NVTStage ps must be a number, got {self.ps!r}') from None
        if not np.isfinite(ps) or ps <= 0:
            raise ValueError(f'NVTStage ps must be > 0, got {self.ps!r}')
        for key, value in dict(self.overrides).items():
            name = _mdp_key(key)
            if name in _NVT_OWN_KEYS:
                raise ValueError(f'NVTStage overrides must not set {key!r}: the solvator sets '
                                 f'{", ".join(_NVT_OWN_KEYS)} itself')
            if name == 'pcoupl' and _mdp_value(value).strip().lower() != 'no':
                raise ValueError(f'NVTStage overrides set pcoupl = {value!r}: the restrained '
                                 'stages are NVT (the solvator sets pcoupl = no)')


@dataclass(frozen=True)
class ExclusionSphere:
    """A sphere kept free of ions or bulk water."""

    center: Callable
    """``center(generator_positions) -> (3,)`` Å, e.g. a host's center of mass."""

    radius: object
    """Å, a float or ``radius(generator_positions) -> float``."""

    def resolve(self, generator_positions):
        """``(center (3,), radius)`` for one candidate (Å)."""
        center = np.asarray(self.center(generator_positions), dtype=float).reshape(3)
        radius = self.radius(generator_positions) if callable(self.radius) else self.radius
        return center, float(radius)


def _copy_suffixes():
    """Suffixes of the restrained water copy: ``R``, then ``R1``, ``R2``, ..."""
    yield 'R'
    k = 1
    while True:
        yield f'R{k}'
        k += 1


def _stage_timing(stage):
    """``(template, overrides, dt, nsteps)`` of an NVT stage: normalized keys, dt in ps."""
    template = _read_mdp(stage.mdp)
    user = {_mdp_key(key): value for key, value in stage.overrides.items()}
    dt = float(user['dt']) if 'dt' in user else float(template.get('dt', 0.001))
    return template, user, dt, int(round(float(stage.ps) / dt)) if dt > 0 else 0


def _stage_gen_temp(template, user):
    """The ``gen_temp`` of a ``gen_vel`` stage (overrides, template, first ``ref_t``) or None."""
    gen_temp = user.get('gen_temp', template.get('gen_temp'))
    if gen_temp is None and template.get('ref_t'):
        gen_temp = template['ref_t'].split()[0]
    return gen_temp


class _Failed(Exception):
    """A candidate that cannot be prepared (not a bug)."""


def _describe(value):
    if isinstance(value, ExclusionSphere):
        radius = value.radius if not callable(value.radius) else getattr(
            value.radius, '__qualname__', repr(value.radius))
        return {'center': getattr(value.center, '__qualname__', repr(value.center)),
                'radius': radius}
    if isinstance(value, NVTStage):
        return {'mdp': os.path.abspath(value.mdp), 'ps': float(value.ps),
                'overrides': dict(value.overrides), 'gen_vel': bool(value.gen_vel)}
    return value


def _nearest(points, targets, lengths, block=256):
    """Minimum-image distance of every point to its nearest target (Å)."""
    out = np.full(len(points), np.inf)
    for start in range(0, len(targets), block):
        chunk = targets[start:start + block]
        d = np.linalg.norm(_min_image(points[:, None, :] - chunk[None], lengths), axis=-1)
        out = np.minimum(out, d.min(axis=1))
    return out


class GromacsSolvator:
    """``cgaimmd_prepare_function``: re-solvate a solute-only or cluster candidate with GROMACS.

    Parameters
    ----------
    descriptors : aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors
        The generator atoms (its structure is ``params.topology``).
    topology : str, optional
        GROMACS topology (default ``aimmd.cgaimmd.preparation.gromacs.topology_file(params)`` at the first call).
    water_box : str, optional
        ``gmx solvate -cs`` water box.
    solvate_scales : tuple of float, optional
        ``gmx solvate -scale`` values, tried in order until enough waters fit.
    selection_consistency : bool, optional
        Delete added waters the cluster's ranking would have chosen (clusters only).
    trim_min_distance : float, optional
        Å; only waters farther than this from every generator atom are deleted at random.
    ion_min_solute_distance : float, optional
        Å; ions replace only waters whose oxygen is farther than this from every solute atom.
    ion_exclusion : ExclusionSphere, optional
        No ion inside this sphere.
    ion_rmin_nm : float, optional
        ``gmx genion -rmin`` (nm).
    water_exclusion : ExclusionSphere, optional
        Bulk water oxygens are kept outside this sphere by a flat-bottomed restraint during EM and NVT.
    box_margin : float, optional
        Å every generator atom must keep from every box face.
    moved_atom_tolerance : float, optional
        Å a generator atom may move in solvate or genion output.
    restrain : str, optional
        ``'generator_heavy'``, ``'generator_all'`` or ``'none'``
        (:data:`~aimmd.cgaimmd.preparation.gromacs.RESTRAINT_MODES`). The old names
        ``'flow_heavy'`` and ``'flow_all'`` are read with a ``DeprecationWarning`` for one release.
    restraint_k : float, optional
        Position-restraint force constant (kJ mol^-1 nm^-2).
    flat_bottom_k : float, optional
        Force constant of the water exclusion (kJ mol^-1 nm^-2).
    em_mdp : str, optional
        EM template (default ``aimmd/engines/em.mdp``; solvated systems pass their own).
    em_nsteps : int, optional
        EM steps.
    nvt_stages : sequence of NVTStage, optional
        Restrained NVT stages after the EM.
    mdrun : str, optional
        mdrun command (default ``params.gmx_mdrun``; with a wrapper script, pass it, see Notes); either is passed
        through ``sanitize_mdrun`` (no ``-plumed``, no checkpoint options).
    grompp : str, optional
        grompp command (default ``split_grompp(params.gmx_grompp)[0]``; an index file ``-n`` is kept, see Notes).
    maxwarn : int, optional
        grompp ``-maxwarn``.
    timeout : float, optional
        Seconds per GROMACS call.
    max_displacement : float, optional
        Å; reject the candidate if a generator heavy atom moves more (None: no check).

    Raises
    ------
    ValueError
        For bad settings (NVT stages included), or (with `topology`) a topology that does not match the structure.
    NotImplementedError
        For systems the solvator cannot rebuild (see Notes).

    Notes
    -----
    After the solute, ``[ molecules ]`` may list only the water
    moleculetype and single-atom ions, at most one cation and one anion
    species (the limit of ``genion``); the ``K`` cluster waters are the first
    ``K`` of its first water entry. The topology's atom and residue names
    must equal those of the structure. Atom names in every file written are
    the topology's. The restrained copy of the cluster waters is the
    moleculetype ``<water>R`` (``<water>R1``, ``R2``, ... when the topology
    already defines that name), recorded as ``record['build']['water_copy']``.

    The index file of the grompp command (``-n``) is kept, with its path
    made absolute (relative to ``params.parent`` for ``params.gmx_grompp``,
    to the working directory at construction for `grompp`), for the EM and
    NVT stages, whose system has the run's atoms in its order; the
    ``genion`` input (a system without ions) is prepared without it.

    The default EM mdrun command drops the GPU offloads GROMACS does not
    support for EM, but it cannot see inside a wrapper script (``gmx_mdrun =
    'bash mdrun_gpu.sh'`` that adds ``-pme gpu``): with a wrapper, pass
    `mdrun` explicitly (it is then used for EM and NVT alike).

    The EM and the NVT stages are never biased: ``-plumed`` (and ``-cpi``,
    ``-cpt``, ``-maxh``, ``-deffnm``, ``-noappend``) is removed from the run's
    command and from an explicit `mdrun` alike
    (:func:`~aimmd.cgaimmd.preparation.gromacs.sanitize_mdrun`). A wrapper
    script that adds ``-plumed`` itself is invisible here; it must not.

    GROMACS seeds are derived from ``context.seed`` (32-bit, never 0):
    ``genion -seed`` from the seed itself, ``gen_seed`` and ``ld_seed`` of
    NVT stage ``k`` from ``context.seed + 1 + k``; the trim draws from
    ``context.rng``. Every argument is recorded in ``settings`` and in each
    record.

    NVT stages: ``nsteps = round(ps / dt)`` with ``dt`` from the stage's
    overrides, else its template (else 0.001 ps); the constructor reads every
    template and raises ``ValueError`` for a stage of fewer than one step or
    a ``gen_vel`` stage without ``gen_temp`` or ``ref_t``. The solvator sets ``define``
    (``-DAIMMD_CGAIMMD_POSRES``), ``include`` (``-I<topology dir>``),
    ``nstxout-compressed = 0``, ``pcoupl = no`` and the keys that
    :class:`NVTStage` refuses in its overrides (``nsteps``, ``gen_vel``,
    ``gen_seed``, ``continuation``, ``ld_seed``, ``nstxout``, ``nstvout``,
    ``nstfout``); a ``define`` or ``include`` override is added to the
    solvator's, other overrides replace the template's values. The stages are
    NVT because the run's ``states_function`` may depend on the box; the last
    frame's box is compared with the topology's, and a changed box fails the
    candidate.

    A GROMACS failure of a candidate (non-zero exit, missing or non-finite
    output, too few waters, a stop by ``must_stop``, the timeout) is
    ``ok=False`` with ``reason='prepare_failed: <message>'``; a composition
    that does not add up after assembly is a bug (``RuntimeError``). The work
    directory is removed unless ``cgaimmd_parameters['keep_scratch']``.

    Examples
    --------
    >>> _solvator = GromacsSolvator(_descriptors, em_mdp='em.mdp',                 # doctest: +SKIP
    ...                             nvt_stages=[NVTStage('nvt.mdp', 20.0, gen_vel=True)])
    >>> def cgaimmd_prepare_function(positions, context):                           # doctest: +SKIP
    ...     return _solvator(positions, context)
    """

    def __init__(self, descriptors, *, topology=None, water_box='spc216.gro',
                 solvate_scales=(0.57, 0.55, 0.53, 0.51, 0.49), selection_consistency=True,
                 trim_min_distance=6.0, ion_min_solute_distance=5.0, ion_exclusion=None,
                 ion_rmin_nm=0.6, water_exclusion=None, box_margin=0.5,
                 moved_atom_tolerance=0.01, restrain='generator_heavy', restraint_k=1e4,
                 flat_bottom_k=1e4, em_mdp=None, em_nsteps=50000, nvt_stages=(), mdrun=None,
                 grompp=None, maxwarn=3, timeout=3600., max_displacement=None):
        restrain = restraint_mode(restrain)
        stages = tuple(nvt_stages)
        for stage in stages:
            if not isinstance(stage, NVTStage):
                raise TypeError(f'nvt_stages must hold NVTStage objects, got '
                                f'{type(stage).__name__}')
        for name, sphere in (('ion_exclusion', ion_exclusion),
                             ('water_exclusion', water_exclusion)):
            if sphere is not None and not isinstance(sphere, ExclusionSphere):
                raise TypeError(f'{name} must be an ExclusionSphere or None')
        if not tuple(solvate_scales):
            raise ValueError('solvate_scales must not be empty')
        self.descriptors = descriptors
        self.topology = None if topology is None else os.path.abspath(str(topology))
        self.water_box = str(water_box)
        self.solvate_scales = tuple(float(s) for s in solvate_scales)
        self.selection_consistency = bool(selection_consistency)
        self.trim_min_distance = float(trim_min_distance)
        self.ion_min_solute_distance = float(ion_min_solute_distance)
        self.ion_exclusion = ion_exclusion
        self.ion_rmin_nm = float(ion_rmin_nm)
        self.water_exclusion = water_exclusion
        self.box_margin = float(box_margin)
        self.moved_atom_tolerance = float(moved_atom_tolerance)
        self.restrain = restrain
        self.restraint_k = float(restraint_k)
        self.flat_bottom_k = float(flat_bottom_k)
        self.em_mdp = None if em_mdp is None else os.path.abspath(str(em_mdp))
        self.em_nsteps = int(em_nsteps)
        self.nvt_stages = tuple(NVTStage(os.path.abspath(s.mdp), float(s.ps), dict(s.overrides),
                                         bool(s.gen_vel)) for s in stages)
        self._check_stages()
        self.mdrun = mdrun
        # an index file of an explicit grompp command is relative to the params file's folder
        self.grompp = None if grompp is None else _with_absolute_path(str(grompp), '-n',
                                                                      os.getcwd())
        self.maxwarn = int(maxwarn)
        self.timeout = float(timeout)
        self.max_displacement = None if max_displacement is None else float(max_displacement)
        names = ('topology', 'water_box', 'solvate_scales', 'selection_consistency',
                 'trim_min_distance', 'ion_min_solute_distance', 'ion_exclusion', 'ion_rmin_nm',
                 'water_exclusion', 'box_margin', 'moved_atom_tolerance', 'restrain',
                 'restraint_k', 'flat_bottom_k', 'em_mdp', 'em_nsteps', 'nvt_stages', 'mdrun',
                 'grompp', 'maxwarn', 'timeout', 'max_displacement')
        self.settings = {'function': 'aimmd.cgaimmd.preparation.solvate.GromacsSolvator',
                         'descriptors': {'structure': descriptors.structure,
                                         'n_solute': descriptors.generator_atoms.n_solute,
                                         'n_waters': descriptors.generator_atoms.n_waters}}
        for name in names:
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = [_describe(v) for v in value]
            self.settings[name] = _describe(value)
        self._plans = {}
        if self.topology is not None:
            self._plan(self.topology)

    def _check_stages(self):
        """Configuration errors of the NVT stages raise here, not once per candidate."""
        for k, stage in enumerate(self.nvt_stages):
            if not os.path.isfile(stage.mdp):
                raise ValueError(f'NVT stage {k}: the template {stage.mdp} does not exist')
            template, user, dt, nsteps = _stage_timing(stage)
            if dt <= 0:
                raise ValueError(f'NVT stage {k}: dt = {dt:g} ps')
            if nsteps < 1:
                raise ValueError(f'NVT stage {k}: ps = {stage.ps:g} gives {nsteps} steps of '
                                 f'dt = {dt:g} ps')
            if stage.gen_vel and _stage_gen_temp(template, user) is None:
                raise ValueError(f'NVT stage {k}: gen_vel needs gen_temp or ref_t (in '
                                 f'{stage.mdp} or the overrides)')

    # -- plan -----------------------------------------------------------------

    def _plan(self, topology):
        """The composition of the full system (cached per topology file)."""
        if topology in self._plans:
            return self._plans[topology]
        import MDAnalysis as mda

        from .topology import read_atom_records, read_topology

        generator_atoms = self.descriptors.generator_atoms
        cluster = generator_atoms.cluster
        water_resname = cluster.water_resname if cluster else 'SOL'
        apw = cluster.atoms_per_water if cluster else 3
        S, K = generator_atoms.n_solute, generator_atoms.n_waters
        top = read_topology(topology)
        records = read_atom_records(top, water_resname=water_resname)
        universe = mda.Universe(self.descriptors.structure)
        if len(records) != len(universe.atoms):
            raise ValueError(f'{topology} has {len(records)} atoms, the structure '
                             f'{self.descriptors.structure} {len(universe.atoms)}')
        mismatched = [k for k, (r, name, resname) in enumerate(zip(
            records, universe.atoms.names, universe.atoms.resnames))
            if r.name != name or r.resname != resname]
        if mismatched:
            k = mismatched[0]
            raise ValueError(f'{topology} and {self.descriptors.structure} disagree on atom names: '
                             f'atom {k + 1} is {records[k].resname} {records[k].name} in the '
                             f'topology, {universe.atoms.resnames[k]} {universe.atoms.names[k]} in '
                             f'the structure ({len(mismatched)} atoms differ)')
        dimensions = universe.dimensions
        if dimensions is None or np.any(np.asarray(dimensions[:3]) <= 0):
            raise ValueError(f'{self.descriptors.structure} has no box')
        if not np.allclose(dimensions[3:6], 90.0, atol=1e-3):
            raise NotImplementedError('the solvator supports orthorhombic boxes only')
        layout, solute_entries, position = [], [], 0
        water, cation, anion = None, None, None
        n_water = 0
        ions = {}
        for name, count in top.molecules:
            moleculetype = top.moleculetypes[name]
            size = len(moleculetype.atoms)
            if position < S:
                if position + size * count > S and (S - position) % size:
                    raise ValueError(f'n_solute = {S} cuts moleculetype {name!r}')
                take = min(count, (S - position) // size)
                if take:
                    solute_entries.append((name, take))
                    position += take * size
                count -= take
                if not count:
                    continue
                if position < S:
                    raise ValueError(f'n_solute = {S} cuts the molecules of {topology}')
            atoms = moleculetype.atoms
            is_water = size > 1 and (atoms[0]['resname'] == water_resname
                                     or bool(moleculetype.interactions.get('settles')))
            if is_water:
                if water is not None and water != name:
                    raise NotImplementedError(f'two water moleculetypes ({water}, {name})')
                if cluster is not None and size != apw:
                    raise ValueError(f'{name} has {size} atoms, atoms_per_water = {apw}')
                water, apw = name, size
                layout.append(('water', name, count))
                n_water += count
            elif size == 1 and abs(float(atoms[0]['charge'])) > 1e-6:
                charge = int(round(float(atoms[0]['charge'])))
                species = (name, atoms[0]['name'], charge)
                if charge > 0:
                    if cation is not None and cation != species:
                        raise NotImplementedError(
                            f'two cation species ({cation[0]}, {name}): genion places one')
                    cation = species
                else:
                    if anion is not None and anion != species:
                        raise NotImplementedError(
                            f'two anion species ({anion[0]}, {name}): genion places one')
                    anion = species
                ions[name] = ions.get(name, 0) + count
                layout.append(('ion', name, count))
            else:
                raise NotImplementedError(
                    f'after the solute, [ molecules ] may list only water and single-atom ions, '
                    f'not {name!r}')
        if position != S:
            raise ValueError(f'{topology} has fewer than n_solute = {S} solute atoms')
        if water is None:
            raise ValueError(f'{topology} lists no {water_resname!r} water after the solute')
        first_water = next(count for kind, name, count in layout if kind == 'water')
        if K > first_water:
            raise NotImplementedError(
                f'the first water entry of [ molecules ] has {first_water} < K = {K} molecules')
        water_records = [r for r in records if r.moleculetype == water][:apw]
        copy_suffix = next(suffix for suffix in _copy_suffixes()
                           if f'{water}{suffix}' not in top.moleculetypes)
        masses = np.array([r.mass for r in records])
        plan = SimpleNamespace(
            topology=topology, top_text=Path(topology).read_text(), records=records,
            names=[r.name for r in records], resnames=[r.resname for r in records],
            n_atoms=len(records), box=np.asarray(dimensions[:3], dtype=float),
            dimensions=np.asarray(dimensions, dtype=float), S=S, K=K, apw=apw,
            water_resname=water_records[0].resname,
            solute_entries=solute_entries, layout=layout, water=water, n_water=n_water,
            water_names=[r.name for r in water_records], water_copy_suffix=copy_suffix,
            water_copy=f'{water}{copy_suffix}', cation=cation, anion=anion,
            n_cation=ions.get(cation[0], 0) if cation else 0,
            n_anion=ions.get(anion[0], 0) if anion else 0, masses=masses,
            solute_heavy=np.flatnonzero(masses[:S] > 1.5),
            water_heavy=np.flatnonzero(np.array([r.mass for r in water_records]) > 1.5))
        plan.n_ions = plan.n_cation + plan.n_anion
        # global index of the first atom of every water, in system order
        starts, offset = [], 0
        for name, count in top.molecules:
            size = len(top.moleculetypes[name].atoms)
            if name == water:
                starts.extend(offset + size * k for k in range(count))
            offset += size * count
        plan.water_starts = np.asarray(starts, dtype=int)
        self._plans[topology] = plan
        return plan

    # -- steps that need no GROMACS ---------------------------------------------

    def _consistency_violations(self, generated, oxygens, lengths, radii):
        """Mask of added waters whose oxygen is closer to a rule's reference than its radius."""
        if radii is None or not len(oxygens):
            return np.zeros(len(oxygens), dtype=bool)
        S = self.descriptors.generator_atoms.n_solute
        distances = self.descriptors._rule_distances(oxygens, generated[:S], lengths)
        return np.any(distances < np.asarray(radii)[:, None], axis=0)

    def _replaceable(self, generated, oxygens, lengths):
        """Mask of bulk waters an ion may replace."""
        S = self.descriptors.generator_atoms.n_solute
        ok = _nearest(oxygens, generated[:S], lengths) > self.ion_min_solute_distance
        if self.ion_exclusion is not None:
            center, radius = self.ion_exclusion.resolve(generated)
            ok &= np.linalg.norm(_min_image(oxygens - center, lengths), axis=-1) > radius
        return ok

    @staticmethod
    def _order_for_ions(bulk, replaceable):
        """Bulk waters with the replaceable ones last, and the number of the others before them."""
        ordered = np.concatenate([bulk[~replaceable], bulk[replaceable]])
        return ordered, int((~replaceable).sum())

    def _records(self, plan, n_bulk):
        """Names of cluster + bulk waters (solute first), residue numbers counted up."""
        names = plan.names[:plan.S]
        resnames = plan.resnames[:plan.S]
        resids = [r.resnr for r in plan.records[:plan.S]]
        top_resid = max(resids) if resids else 0
        for k in range(plan.K + n_bulk):
            names += plan.water_names
            resnames += [plan.water_resname] * plan.apw
            resids += [top_resid + 1 + k] * plan.apw
        return SimpleNamespace(names=names, resnames=resnames, resids=resids)

    def _molecules_text(self, plan, n_waters):
        lines = [f'{name:<16s} {count}' for name, count in plan.solute_entries]
        lines.append(f'{plan.water:<16s} {n_waters}')
        text = plan.top_text
        match = list(re.finditer(r'^\s*\[\s*molecules\s*\]', text, flags=re.M | re.I))
        if not match:
            raise ValueError(f'{plan.topology} has no [ molecules ] section')
        return (text[:match[-1].start()]
                + '[ molecules ]\n; written by aimmd.cgaimmd.preparation.solvate\n'
                + '\n'.join(lines) + '\n')

    # -- the call ---------------------------------------------------------------

    def _commands(self, params):
        """The resolved commands: grompp, grompp_genion (no index file), mdrun, mdrun_em, gmx."""
        grompp = self.grompp if self.grompp is not None else _params_grompp(params)
        # an explicit command is sanitized too: candidate preparation is never biased
        mdrun = sanitize_mdrun(self.mdrun if self.mdrun is not None else params.gmx_mdrun)
        return {
            'grompp': f'{_grompp_command(grompp, maxwarn=self.maxwarn)} -maxwarn {self.maxwarn}',
            # the pre-ion system has fewer atoms than the run's: its index groups do not apply
            'grompp_genion': (f'{_grompp_command(grompp, maxwarn=self.maxwarn, index=False)} '
                              f'-maxwarn {self.maxwarn}'),
            'mdrun': _mdrun_command(mdrun),
            'mdrun_em': _mdrun_command(mdrun, minimization=self.mdrun is None),
            'gmx': gmx_executable(params)}

    def __call__(self, positions, context):
        """Build, minimize and equilibrate the full system around one candidate.

        Parameters
        ----------
        positions : numpy.ndarray
            ``(M, 3)`` Å generator atoms of one valid candidate.
        context : aimmd.cgaimmd.types.CandidateContext
            The candidate's preparation context (``rng``/``seed`` are its preparation stream).

        Returns
        -------
        aimmd.cgaimmd.types.PreparedFrame
            The full system in topology order and the box, or ``ok=False`` with the reason.
        """
        start = time.time()
        params = context.params
        plan = self._plan(self.topology or topology_file(params))
        universe = getattr(params, 'universe', None)
        if universe is not None and len(universe.atoms) != plan.n_atoms:
            raise RuntimeError(f'params.universe has {len(universe.atoms)} atoms, the topology '
                               f'{plan.n_atoms}')
        commands = self._commands(params)
        record = {'settings': self.settings, 'build': {}, 'em': None, 'nvt': [],
                  'displacement': None, 'seconds': {},
                  'commands': {**commands, 'topology': plan.topology}}
        generator_input = np.asarray(positions, dtype=float)
        M = self.descriptors.n_atoms
        if generator_input.shape != (M, 3):
            raise ValueError(f'expected ({M}, 3) generator-atom positions, got '
                             f'{generator_input.shape}')

        def failed(message, dimensions=None):
            record['error'] = message
            record['seconds']['total'] = time.time() - start
            first = message.strip().splitlines()[0] if message.strip() else 'failed'
            return PreparedFrame(positions=np.zeros((plan.n_atoms, 3)), dimensions=dimensions,
                                 ok=False, reason=f'prepare_failed: {first}', record=record,
                                 seconds=record['seconds']['total'])

        if not np.all(np.isfinite(generator_input)):
            return failed('non-finite generator-atom positions')
        workdir = _work_dir(context.scratch, 'solvate')
        record['directory'] = str(workdir)
        try:
            return self._build(plan, generator_input, context, workdir, record, start, commands)
        except (_Failed, RuntimeError, OSError) as exc:
            if isinstance(exc, RuntimeError) and getattr(exc, '_aimmd_bug', False):
                raise
            return failed(str(exc), plan.dimensions)
        finally:
            if not (context.settings or {}).get('keep_scratch'):
                shutil.rmtree(workdir, ignore_errors=True)

    def _build(self, plan, generator_input, context, workdir, record, start, commands):
        from .topology import restrained_topology

        grompp, mdrun = commands['grompp'], commands['mdrun']
        mdrun_em, gmx = commands['mdrun_em'], commands['gmx']
        S, K, apw, L = plan.S, plan.K, plan.apw, plan.box
        M = len(generator_input)
        log = workdir / 'gromacs.log'
        timing = record['seconds']
        run = dict(timeout=self.timeout, must_stop=context.must_stop)
        topdir = os.path.dirname(plan.topology)
        em_template = self.em_mdp or _default_em_mdp()
        build = record['build']

        # 2. place
        t = time.time()
        generated = generator_input - generator_input[:S].mean(axis=0) + L / 2.0
        if np.any(generated < self.box_margin) or np.any(generated > L - self.box_margin):
            extent = np.ptp(generated, axis=0)
            raise _Failed(f'does not fit: extent {np.round(extent, 1).tolist()} A in the '
                          f'{np.round(L, 2).tolist()} A box (margin {self.box_margin:g} A)')
        # 3. solvate (and 4. selection consistency)
        cluster_records = self._records(plan, 0)
        write_gro(workdir / 'cluster.gro', cluster_records, generated, plan.dimensions, decimals=6)
        radii = None
        if self.descriptors.generator_atoms.cluster is not None and self.selection_consistency:
            radii = self.descriptors.cluster_radii(generated)
            build['cluster_radii_A'] = [round(r, 4) for r in radii]
        needed = plan.n_water - K + plan.n_ions
        bulk = None
        for scale in self.solvate_scales:
            run_gmx(f'{gmx} solvate -cp cluster.gro -cs {shlex.quote(self.water_box)} '
                    f'-o solvated.gro -scale {scale:g}', workdir, log, **run)
            _, names, solvated, _ = _gro_records(workdir / 'solvated.gro')
            added_names = names[M:]
            if (len(added_names) % apw
                    or added_names != added_names[:apw] * (len(added_names) // apw)):
                raise _Failed(f'the water box {self.water_box} does not have {apw}-atom molecules '
                              f'(added atoms {" ".join(added_names[:2 * apw])} ...)')
            moved = np.abs(solvated[:M] - generated).max()
            if moved > self.moved_atom_tolerance:
                raise _Failed(f'gmx solvate moved a generator atom by {moved:.3f} A')
            added = solvated[M:].reshape(-1, apw, 3)
            violating = self._consistency_violations(generated, added[:, 0], L, radii)
            build.update(solvate_scale=scale, waters_added=int(len(added)),
                         deleted_by_consistency=int(violating.sum()))
            if len(added) - int(violating.sum()) >= needed:
                bulk = added[~violating]
                break
        if bulk is None:
            raise _Failed(f'gmx solvate fits only {build.get("waters_added")} waters '
                          f'({build.get("deleted_by_consistency")} of them inconsistent with the '
                          f'selection), {needed} are needed')
        # 5. trim
        far = np.flatnonzero(_nearest(bulk[:, 0], generated, L) > self.trim_min_distance)
        n_drop = len(bulk) - needed
        if n_drop > len(far):
            raise _Failed(f'cannot delete {n_drop} waters: only {len(far)} are farther than '
                          f'{self.trim_min_distance:g} A from the generator atoms')
        drop = context.rng.choice(far, size=n_drop, replace=False) if n_drop else np.zeros(0, int)
        bulk = np.delete(bulk, drop, axis=0)
        build['deleted_by_trim'] = int(n_drop)
        timing['solvate'] = time.time() - t
        # 6. ions
        t = time.time()
        if plan.n_ions:
            replaceable = self._replaceable(generated, bulk[:, 0], L)
            if int(replaceable.sum()) < plan.n_ions:
                raise _Failed(f'only {int(replaceable.sum())} waters may be replaced by the '
                              f'{plan.n_ions} ions')
            bulk, n_fixed = self._order_for_ions(bulk, replaceable)
            pre = self._records(plan, len(bulk))
            x_pre = np.concatenate([generated, bulk.reshape(-1, 3)])
            write_gro(workdir / 'pre_ion.gro', pre, x_pre, plan.dimensions, decimals=6)
            (workdir / 'pre_ion.top').write_text(self._molecules_text(plan, K + len(bulk)))
            first = M + apw * n_fixed
            indices = np.arange(first, M + apw * len(bulk)) + 1
            with open(workdir / 'ions.ndx', 'w') as handle:
                handle.write('[ replaceable ]\n')
                for k in range(0, len(indices), 15):
                    handle.write(' '.join(str(i) for i in indices[k:k + 15]) + '\n')
            (workdir / 'ions.mdp').write_text(mdp_text(
                em_template, {'include': f'-I{topdir}'},
                header='genion input (aimmd.cgaimmd.preparation.solvate)'))
            run_gmx(f'{commands["grompp_genion"]} -f ions.mdp -c pre_ion.gro -p pre_ion.top '
                    f'-o ions.tpr -po ions_mdout.mdp', workdir, log, **run)
            seed = _gromacs_seed(context.seed)
            options = []
            if plan.n_cation:
                name, atom, charge = plan.cation
                options += [f'-pname {atom} -pq {charge} -np {plan.n_cation}']
            else:
                options += ['-np 0']
            if plan.n_anion:
                name, atom, charge = plan.anion
                options += [f'-nname {atom} -nq {charge} -nn {plan.n_anion}']
            else:
                options += ['-nn 0']
            run_gmx(f'{gmx} genion -s ions.tpr -n ions.ndx -o genion.gro {" ".join(options)} '
                    f'-rmin {self.ion_rmin_nm:g} -seed {seed}', workdir, log, stdin='replaceable\n',
                    **run)
            resnames, names, x_ion, _ = _gro_records(workdir / 'genion.gro')
            moved = np.abs(x_ion[:M] - generated).max() if len(x_ion) >= M else np.inf
            if moved > self.moved_atom_tolerance:
                raise _Failed(f'gmx genion moved a generator atom by {moved:.3f} A')
            rest = np.arange(M, len(x_ion))
            cation_atom = plan.cation[1] if plan.cation else None
            anion_atom = plan.anion[1] if plan.anion else None
            is_cation = np.array([names[i] == cation_atom for i in rest], dtype=bool)
            is_anion = np.array([names[i] == anion_atom for i in rest], dtype=bool)
            cations, anions = x_ion[rest[is_cation]], x_ion[rest[is_anion]]
            waters = x_ion[rest[~(is_cation | is_anion)]]
            if (len(cations) != plan.n_cation or len(anions) != plan.n_anion
                    or len(waters) != apw * (plan.n_water - K)):
                raise _Failed(f'genion gave {len(cations)} cations, {len(anions)} anions and '
                              f'{len(waters) // apw} waters, the topology needs {plan.n_cation}, '
                              f'{plan.n_anion} and {plan.n_water - K}')
            bulk = waters.reshape(-1, apw, 3)
            all_ions = np.concatenate([cations, anions]) if plan.n_ions else np.zeros((0, 3))
            if len(all_ions):
                build['ion_min_solute_distance_A'] = round(float(
                    _nearest(all_ions, generated[:S], L).min()), 3)
            build['ion_seed'] = seed
        else:
            cations = anions = np.zeros((0, 3))
        timing['ions'] = time.time() - t
        # 7. assemble in topology order
        x_full = self._assemble(plan, generated, bulk, cations, anions)
        full = SimpleNamespace(names=plan.names, resnames=plan.resnames,
                               resids=[r.resnr for r in plan.records])
        write_gro(workdir / 'full_initial.gro', full, x_full, plan.dimensions, decimals=6)
        generator_index = np.concatenate([np.arange(S)] + [
            plan.water_starts[k] + np.arange(apw) for k in range(K)]).astype(int)
        # 8. restrained topology
        restrained = []
        if self.restrain != 'none':
            if self.restrain == 'generator_heavy':
                restrained += list(plan.solute_heavy)
                restrained += [int(plan.water_starts[k] + j) for k in range(K)
                               for j in plan.water_heavy]
            else:
                restrained += list(generator_index)
        flat = None
        reference = x_full.copy()
        if self.water_exclusion is not None:
            center, radius = self.water_exclusion.resolve(generated)
            flat = (plan.water_names[0], radius, self.flat_bottom_k)
            reference[plan.water_starts[K:]] = center
            build['water_exclusion'] = {'center_A': np.round(center, 3).tolist(),
                                        'radius_A': round(radius, 4)}
        restraint_args = ''
        if restrained or flat is not None:
            # the cluster waters are always the copy, so that a flat-bottom sphere on the
            # bulk waters never acts on them
            (workdir / 'topol_restrained.top').write_text(restrained_topology(
                plan.topology, restrained, self.restraint_k, define=RESTRAINT_DEFINE,
                flat_bottom=flat, water_copy_suffix=plan.water_copy_suffix,
                n_restrained_waters=K))
            if K:
                build['water_copy'] = plan.water_copy
            write_gro(workdir / 'restraint_reference.gro', full, reference, plan.dimensions,
                      decimals=6)
            top_argument, restraint_args = 'topol_restrained.top', ' -r restraint_reference.gro'
            define = {'define': f'-D{RESTRAINT_DEFINE}'}
        else:
            top_argument, define = plan.topology, {}
        top_argument = shlex.quote(top_argument)
        # 9. restrained EM
        t = time.time()
        (workdir / 'em_restrained.mdp').write_text(mdp_text(em_template, {
            **define, 'nsteps': self.em_nsteps, 'include': f'-I{topdir}'},
            header=('restrained EM of a re-solvated C-GAIMMD candidate '
                    '(aimmd.cgaimmd.preparation.solvate)')))
        run_gmx(f'{grompp} -f em_restrained.mdp -c full_initial.gro{restraint_args} '
                f'-p {top_argument} -o em.tpr -po em_mdout.mdp', workdir, log, **run)
        run_gmx(f'{mdrun_em} -deffnm em -nobackup', workdir, log, **run)
        record['em'] = parse_em_log(workdir / 'em.log')
        timing['em'] = time.time() - t
        last = 'em'
        # 10. restrained NVT stages
        for k, stage in enumerate(self.nvt_stages):
            t = time.time()
            overrides, stage_record = self._nvt_stage(k, bool(define), topdir, context.seed)
            name = f'nvt{k}'
            (workdir / f'{name}.mdp').write_text(mdp_text(stage.mdp, overrides, header=(
                f'restrained NVT stage {k} of a re-solvated C-GAIMMD candidate '
                '(aimmd.cgaimmd.preparation.solvate)')))
            run_gmx(f'{grompp} -f {name}.mdp -c {last}.gro{restraint_args} '
                    f'-t {stage_record["start"]} -p {top_argument} -o {name}.tpr '
                    f'-po {name}_mdout.mdp', workdir, log, **run)
            run_gmx(f'{mdrun} -deffnm {name} -nobackup', workdir, log, **run)
            stage_record['seconds'] = time.time() - t
            record['nvt'].append(stage_record)
            timing[name] = stage_record['seconds']
            last = name
        # 11. result
        x, box = read_trr_last_frame(workdir / f'{last}.trr')
        if x.shape != x_full.shape or not np.all(np.isfinite(x)):
            raise _Failed(f'non-finite or misshapen output of {last}')
        if (not np.allclose(box[:3], L, atol=_BOX_TOLERANCE)
                or not np.allclose(box[3:6], 90.0, atol=_BOX_TOLERANCE)):
            raise _Failed(f'the box changed in {last}: {np.round(box, 4).tolist()}, the topology '
                          f'has {np.round(plan.dimensions, 4).tolist()} (the stages must be NVT)')
        moved = np.linalg.norm(_min_image(x[generator_index] - x_full[generator_index], L), axis=-1)
        heavy_local = np.concatenate([plan.solute_heavy] + [
            S + apw * k + plan.water_heavy for k in range(K)]).astype(int)
        heavy_moved = moved[heavy_local] if len(heavy_local) else moved
        record['displacement'] = {
            'max_A': float(moved.max()), 'heavy_max_A': float(heavy_moved.max()),
            'heavy_rms_A': float(np.sqrt((heavy_moved ** 2).mean()))}
        if self.max_displacement is not None and heavy_moved.max() > self.max_displacement:
            raise _Failed(f'a generator heavy atom moved {heavy_moved.max():.3f} A > '
                          f'max_displacement {self.max_displacement:g} A')
        timing['total'] = time.time() - start
        return PreparedFrame(positions=x, dimensions=plan.dimensions.copy(), ok=True, reason='',
                             record=record, seconds=timing['total'])

    def _nvt_stage(self, k, restrained, topdir, seed):
        """The mdp overrides and the record of NVT stage `k` (step 10 without GROMACS).

        Parameters
        ----------
        k : int
            The stage (0-based).
        restrained : bool
            Whether the restraint define is set.
        topdir : str
            Directory of the topology (``include = -I<topdir>``).
        seed : int
            ``context.seed``.

        Returns
        -------
        tuple
            ``(overrides for mdp_text, stage record)``.
        """
        stage = self.nvt_stages[k]
        template, user, dt, nsteps = _stage_timing(stage)
        last = k == len(self.nvt_stages) - 1
        stage_seed = _gromacs_seed(int(seed) + 1 + k)
        define = ' '.join(v for v in (f'-D{RESTRAINT_DEFINE}' if restrained else '',
                                      _mdp_value(user.pop('define', ''))) if v)
        include = ' '.join(v for v in (f'-I{topdir}', _mdp_value(user.pop('include', ''))) if v)
        overrides = {'include': include, 'nsteps': nsteps, 'nstxout_compressed': 0,
                     'nstxout': nsteps if last else 0, 'nstvout': 0, 'nstfout': 0,
                     'pcoupl': 'no', 'ld_seed': stage_seed}
        if define:
            overrides['define'] = define
        record = {'mdp': stage.mdp, 'ps': stage.ps, 'dt': dt, 'nsteps': nsteps,
                  'ld_seed': stage_seed, 'start': 'em.trr' if k == 0 else f'nvt{k - 1}.cpt'}
        if stage.gen_vel:
            gen_temp = _stage_gen_temp(template, user)      # not None: checked at construction
            overrides.update({'gen_vel': True, 'gen_seed': stage_seed, 'gen_temp': gen_temp,
                              'continuation': False})
            record.update(gen_seed=stage_seed, gen_temp=float(gen_temp))
        else:
            overrides.update({'gen_vel': False, 'continuation': True})
        overrides.update(user)                   # NVTStage refused the keys set above
        return overrides, record

    def _assemble(self, plan, generated, bulk, cations, anions):
        """The full system in topology order: solute, then [ molecules ] with the cluster waters first."""
        S, K, apw = plan.S, plan.K, plan.apw
        waters = np.concatenate([generated[S:].reshape(-1, apw, 3), bulk]) if K else bulk
        pools = {'water': list(waters), plan.cation[0] if plan.cation else None: list(cations),
                 plan.anion[0] if plan.anion else None: list(anions)}
        parts = [generated[:S]]
        for kind, name, count in plan.layout:
            pool = pools['water'] if kind == 'water' else pools[name]
            if len(pool) < count:
                error = RuntimeError(f'assembly: {name} needs {count} molecules, {len(pool)} left '
                                     '(a bug)')
                error._aimmd_bug = True
                raise error
            taken, pools_left = pool[:count], pool[count:]
            if kind == 'water':
                pools['water'] = pools_left
            else:
                pools[name] = pools_left
            parts.append(np.asarray(taken, dtype=float).reshape(-1, 3))
        left = sum(len(v) for v in pools.values())
        x = np.concatenate(parts)
        if left or len(x) != plan.n_atoms:
            error = RuntimeError(f'assembly: {len(x)} atoms and {left} molecules left over for a '
                                 f'{plan.n_atoms}-atom topology (a bug)')
            error._aimmd_bug = True
            raise error
        return x
