"""
aimmd.cgaimmd.preparation.minimize
==================================

Restrained GROMACS energy minimization of generated candidates.

- :class:`GromacsMinimizer` is a ``cgaimmd_minimize_function`` for vacuum and
  solute-only systems: it minimizes the prepared candidate that is shot.
- :func:`polish` minimizes raw generated structures alone (no solvent) before
  their validity is checked, so that the near-contacts of raw samples are not
  read as extra bonds. :class:`aimmd.cgaimmd.preparation.validity.PolishedValidity`
  is the ready-made validity hook built on it.

Why its own grompp/mdrun pipeline and not ``Params.minimize_energy``: the
position restraints must hold the generated geometry (the candidate itself
is the restraint reference, through a derived topology), the run's
``gmx_mdrun`` may carry ``-plumed`` (a bias would bias the minimization) and
checkpoint flags, a worker's ``must_stop`` must be able to interrupt it, a
hook must not mutate ``params``, and a candidate is one in-memory frame, not
a trajectory file. With ``restrain='none'`` the result agrees with
``Params.minimize_energy`` on the same EM mdp (a ``--runslow`` test checks
RMS < 0.01 Å).

The restrained default follows the measurement that unrestrained
minimization rolls barrier structures into the basins (C-GAIMMD development
notebook, 2026-09-24).
"""

import os
import shlex
import shutil
import time

from typing import NamedTuple, Optional

import numpy as np

from ..types import PreparedFrame
from .descriptors import _box_lengths, _min_image
from .gromacs import (RESTRAINT_DEFINE, _default_em_mdp, _grompp_command, _mdrun_command,
                      _params_grompp, _with_absolute_path, _work_dir, mdp_text, parse_em_log,
                      read_trr_last_frame, restraint_mode, run_gmx, sanitize_mdrun, topology_file,
                      write_gro)

__all__ = ['GromacsMinimizer', 'PolishResult', 'polish']


def _abs(path):
    return None if path is None else os.path.abspath(str(path))


def _displacement(new, old, dimensions):
    """Per-atom displacement (Å), minimum image in an orthorhombic box."""
    delta = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    return np.linalg.norm(_min_image(delta, _box_lengths(dimensions)), axis=-1)


class GromacsMinimizer:
    """``cgaimmd_minimize_function``: restrained steepest-descent EM of a candidate with GROMACS.

    Parameters
    ----------
    em_mdp : str, optional
        EM mdp template (default ``aimmd/engines/em.mdp``; vacuum systems pass their own).
    restrain : str, optional
        ``'generator_heavy'`` (the heavy generator atoms), ``'generator_all'`` or ``'none'``
        (:data:`~aimmd.cgaimmd.preparation.gromacs.RESTRAINT_MODES`). The old names
        ``'flow_heavy'`` and ``'flow_all'`` are read with a ``DeprecationWarning`` for one release.
    restraint_k : float, optional
        Position-restraint force constant (kJ mol^-1 nm^-2).
    emtol : float, optional
        Convergence criterion Fmax (kJ mol^-1 nm^-1).
    nsteps : int, optional
        Maximum number of EM steps.
    mdrun : str, optional
        mdrun command (default ``params.gmx_mdrun`` without GPU PME/bonded/update offloads); either is passed through
        ``sanitize_mdrun`` (no ``-plumed``, no checkpoint options; see Notes).
    grompp : str, optional
        grompp command (default ``split_grompp(params.gmx_grompp)[0]``; an index file ``-n`` is kept, see Notes).
    topology : str, optional
        GROMACS topology (default ``aimmd.cgaimmd.preparation.gromacs.topology_file(params)``).
    maxwarn : int, optional
        grompp ``-maxwarn`` (None: whatever the grompp command says).
    timeout : float, optional
        Seconds per GROMACS call.
    max_displacement : float, optional
        Å; reject the candidate if a generator heavy atom moves more.

    Raises
    ------
    ValueError
        For an unknown `restrain`.

    Notes
    -----
    Per call, in ``{context.scratch}/em_<counter>/``: ``start.gro`` and
    ``ref.gro`` (the frame at 6 decimals, names from ``params.universe``),
    ``restrained.top`` (``aimmd.cgaimmd.preparation.topology.restrained_topology``) and
    ``em.mdp`` (the template with ``define = -DAIMMD_CGAIMMD_POSRES``,
    ``emtol``, ``nsteps``, ``nstxout = 0``, ``include = -I<topdir>``), then
    ``<grompp> -nobackup -f em.mdp -c start.gro -r ref.gro -p restrained.top
    -o em.tpr -po em_mdout.mdp`` and ``<mdrun> -deffnm em -nobackup``. The
    result is the last frame of ``em.trr`` (GROMACS writes the final
    minimized frame). ``restrain='none'`` uses the unmodified topology and no
    ``-r``.

    The generator atoms are the first ``M = aimmd.cgaimmd.protocol.n_atoms_of(
    context.network)`` atoms of the frame (vacuum: all; solute only: the
    solute), their heavy atoms those with a topology mass above 1.5.
    Displacements are minimum-image distances in the frame's box.

    The index file of the grompp command (``-n``) is kept, with its path made
    absolute: relative to ``params.parent`` for ``params.gmx_grompp``, to the
    working directory at construction for `grompp`. The frame is the full
    system in the run's atom order, so the run's index groups apply.

    The default mdrun command drops the GPU offloads GROMACS does not support
    for EM, but it cannot see inside a wrapper script (``gmx_mdrun = 'bash
    mdrun_gpu.sh'`` that adds ``-pme gpu``): with a wrapper, pass `mdrun`
    explicitly.

    The minimization is never biased: ``-plumed`` (and ``-cpi``, ``-cpt``,
    ``-maxh``, ``-deffnm``, ``-noappend``) is removed from the run's command
    and from an explicit `mdrun` alike (:func:`~aimmd.cgaimmd.preparation.gromacs.sanitize_mdrun`),
    so a run under a static PLUMED bias can hand its shooting command to the
    minimizer. A wrapper script that adds ``-plumed`` itself is invisible
    here; it must not.

    A GROMACS failure (non-zero exit, a stop by ``must_stop``, the timeout,
    missing or non-finite output) is ``ok=False`` with ``reason=
    'minimize_failed: <message>'``; configuration errors (an unreadable
    topology) raise. The work directory is removed unless
    ``cgaimmd_parameters['keep_scratch']``.

    Examples
    --------
    >>> _minimizer = GromacsMinimizer(em_mdp='em.mdp', restrain='generator_heavy')  # doctest: +SKIP
    >>> def cgaimmd_minimize_function(frame, context):                             # doctest: +SKIP
    ...     return _minimizer(frame, context)
    """

    def __init__(self, *, em_mdp=None, restrain='generator_heavy', restraint_k=1e4, emtol=100.0,
                 nsteps=1000, mdrun=None, grompp=None, topology=None, maxwarn=None, timeout=600.,
                 max_displacement=None):
        self.em_mdp = _abs(em_mdp)
        self.restrain = restraint_mode(restrain)
        self.restraint_k = float(restraint_k)
        self.emtol = float(emtol)
        self.nsteps = int(nsteps)
        self.mdrun = mdrun
        # an index file of an explicit grompp command is relative to the params file's folder
        self.grompp = None if grompp is None else _with_absolute_path(str(grompp), '-n',
                                                                      os.getcwd())
        self.topology = _abs(topology)
        self.maxwarn = None if maxwarn is None else int(maxwarn)
        self.timeout = float(timeout)
        self.max_displacement = None if max_displacement is None else float(max_displacement)
        self.settings = {
            'function': 'aimmd.cgaimmd.preparation.minimize.GromacsMinimizer',
            'em_mdp': self.em_mdp,
            'restrain': self.restrain, 'restraint_k': self.restraint_k, 'emtol': self.emtol,
            'nsteps': self.nsteps, 'mdrun': self.mdrun, 'grompp': self.grompp,
            'topology': self.topology, 'maxwarn': self.maxwarn, 'timeout': self.timeout,
            'max_displacement': self.max_displacement}

    def _commands(self, params):
        """``(grompp, mdrun, topology, em_mdp)`` for a run, resolved."""
        grompp = self.grompp if self.grompp is not None else _params_grompp(params)
        grompp = _grompp_command(grompp, maxwarn=self.maxwarn)
        if self.maxwarn is not None:
            grompp = f'{grompp} -maxwarn {self.maxwarn}'
        # an explicit command is sanitized too: candidate preparation is never biased
        mdrun = sanitize_mdrun(self.mdrun if self.mdrun is not None else params.gmx_mdrun)
        mdrun = _mdrun_command(mdrun, minimization=self.mdrun is None)
        topology = self.topology or topology_file(params)
        return grompp, mdrun, topology, self.em_mdp or _default_em_mdp()

    def __call__(self, frame, context):
        """Minimize one prepared frame.

        Parameters
        ----------
        frame : aimmd.cgaimmd.types.PreparedFrame
            The candidate as a full system (``params.topology`` order).
        context : aimmd.cgaimmd.types.CandidateContext
            The hook context (params, network, scratch, must_stop, settings).

        Returns
        -------
        aimmd.cgaimmd.types.PreparedFrame
            The minimized frame, or the input with ``ok=False`` and the reason.
        """
        from ..protocol import n_atoms_of
        from .topology import read_masses, restrained_topology

        start = time.time()
        params = context.params
        grompp, mdrun, topology, em_mdp = self._commands(params)
        positions = np.asarray(frame.positions, dtype=float)
        n_atoms = len(positions)
        n_generator = n_atoms_of(context.network)
        if n_generator > n_atoms:
            raise ValueError(f'the network has {n_generator} generator atoms, the frame {n_atoms} '
                             'atoms')
        masses = read_masses(topology, expected=n_atoms)
        heavy = np.flatnonzero(masses[:n_generator] > 1.5)
        restrained = {'generator_heavy': heavy, 'generator_all': np.arange(n_generator),
                      'none': np.zeros(0, dtype=int)}[self.restrain]
        record = {'settings': self.settings,
                  'commands': {'grompp': grompp, 'mdrun': mdrun, 'topology': topology,
                               'em_mdp': em_mdp}}

        def failed(message):
            record['error'] = message
            record['seconds'] = time.time() - start
            first = message.strip().splitlines()[0] if message.strip() else 'failed'
            return PreparedFrame(positions=positions, dimensions=frame.dimensions, ok=False,
                                 reason=f'minimize_failed: {first}', record=record,
                                 seconds=record['seconds'])

        if not np.all(np.isfinite(positions)):
            return failed('non-finite input positions')
        workdir = _work_dir(context.scratch, 'em')
        record['directory'] = str(workdir)
        try:
            universe = getattr(params, 'universe', None)
            if universe is None or len(universe.atoms) != n_atoms:
                raise ValueError(f'params.universe must have the {n_atoms} atoms of the frame')
            write_gro(workdir / 'start.gro', universe.atoms, positions, frame.dimensions,
                      decimals=6)
            overrides = {'emtol': self.emtol, 'nsteps': self.nsteps, 'nstxout': 0,
                         'include': f'-I{os.path.dirname(topology)}'}
            restraints = ''
            if len(restrained):
                shutil.copy(workdir / 'start.gro', workdir / 'ref.gro')
                (workdir / 'restrained.top').write_text(
                    restrained_topology(topology, restrained, self.restraint_k,
                                        define=RESTRAINT_DEFINE))
                overrides['define'] = f'-D{RESTRAINT_DEFINE}'
                top_argument, restraints = 'restrained.top', ' -r ref.gro'
            else:
                top_argument = topology
            (workdir / 'em.mdp').write_text(mdp_text(em_mdp, overrides, header=(
                'energy minimization of a C-GAIMMD candidate '
                '(aimmd.cgaimmd.preparation.minimize.GromacsMinimizer)')))
            log = workdir / 'gromacs.log'
            try:
                run_gmx(f'{grompp} -nobackup -f em.mdp -c start.gro{restraints} '
                        f'-p {shlex.quote(str(top_argument))} -o em.tpr -po em_mdout.mdp', workdir,
                        log, timeout=self.timeout, must_stop=context.must_stop)
                run_gmx(f'{mdrun} -deffnm em -nobackup', workdir, log, timeout=self.timeout,
                        must_stop=context.must_stop)
                minimized, _ = read_trr_last_frame(workdir / 'em.trr')
            except (RuntimeError, OSError) as exc:
                return failed(str(exc))
            if minimized.shape != positions.shape or not np.all(np.isfinite(minimized)):
                return failed(f'non-finite or misshapen output {minimized.shape}')
            em = parse_em_log(workdir / 'em.log')
            moved = _displacement(minimized[:n_generator], positions[:n_generator],
                                  frame.dimensions)
            heavy_moved = moved[heavy] if len(heavy) else moved
            record.update(
                epot_before=em['epot_start'], epot_after=em['epot'], fmax=em['fmax'],
                steps=em['steps'], converged=em['converged'],
                max_displacement_A=float(heavy_moved.max()) if len(heavy_moved) else 0.0,
                heavy_rms_displacement_A=(float(np.sqrt((heavy_moved ** 2).mean()))
                                          if len(heavy_moved) else 0.0))
            if (self.max_displacement is not None
                    and record['max_displacement_A'] > self.max_displacement):
                return failed(f'a generator heavy atom moved {record["max_displacement_A"]:.3f} '
                              f'A > max_displacement {self.max_displacement:g} A')
            record['seconds'] = time.time() - start
            return PreparedFrame(positions=minimized, dimensions=frame.dimensions, ok=True,
                                 reason='', record=record, seconds=record['seconds'])
        finally:
            if not (context.settings or {}).get('keep_scratch'):
                shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# polish
# ---------------------------------------------------------------------------

class PolishResult(NamedTuple):
    """The polish of one structure (a tuple ``(positions, ok, reason, record)``)."""

    positions: Optional[np.ndarray]
    """``(M, 3)`` float64 Å, the polished structure at the centroid of the input; None if not `ok`."""

    ok: bool
    """Whether the minimization succeeded."""

    reason: str
    """The minimizer's reason when not `ok` (e.g. ``'minimize_failed: ...'``), else ``''``."""

    record: dict
    """The minimizer's record without its ``'settings'`` (e.g. ``heavy_rms_displacement_A``)."""


def polish(positions, minimizer, network, params, box, scratch, workers=1, seed=0, *,
           must_stop=None, keep_scratch=False):
    """Minimize each generated structure alone, centred in a box.

    The polish of raw generated samples: each structure is minimized on its
    own (no solvent), with its heavy atoms restrained (as the minimizer is
    configured), so that the near-contacts of a raw sample relax before its
    bond graph is read. :class:`~aimmd.cgaimmd.preparation.validity.PolishedValidity`
    wraps it into a ``cgaimmd_validity_function``.

    Parameters
    ----------
    positions : array_like
        ``(n, M, 3)`` Å, the structures.
    minimizer : callable
        ``minimizer(frame, context) -> PreparedFrame``, e.g. a
        :class:`GromacsMinimizer` with the topology of the ``M`` atoms alone
        (its ``topology=``; also ``grompp=`` and ``mdrun=`` unless `params`
        has ``gmx_grompp`` and ``gmx_mdrun``).
    network : object
        What the minimizer reads ``n_atoms`` from (the atoms it restrains are
        the first ``n_atoms``): the network, or any object with ``n_atoms``.
    params : object
        What the minimizer reads as the run's params: ``universe``, an
        MDAnalysis Universe of the ``M`` atoms (their names in the GROMACS
        files), and ``parent``, the folder that relative files of
        ``gmx_grompp`` are found from; plus ``gmx_grompp`` and ``gmx_mdrun``
        when the minimizer has no commands of its own.
    box : array_like
        ``(6,)`` box of the minimization (Å, degrees).
    scratch : str or os.PathLike
        Work folder; each minimization makes a folder of its own in it.
    workers : int, optional
        Minimizations run at the same time (threads; each runs GROMACS).
    seed : int, optional
        Structure ``k`` gets the context seed ``seed + k``.
    must_stop : callable, optional
        ``must_stop() -> bool``, handed to the minimizer (a worker's stop request).
    keep_scratch : bool, optional
        Keep the work folders of the minimizations.

    Returns
    -------
    list of PolishResult
        One per structure, in input order.

    Raises
    ------
    ValueError
        If `box` is not 6 numbers.

    Notes
    -----
    Structure ``k`` is translated so that its centroid is at the centre of
    `box`, minimized, and the first ``M`` atoms of the result are translated
    back to the centroid of the input. The minimizer's context has
    ``path_name='polish<k:06d>'``, ``scratch`` and ``folder`` `scratch`, the
    seed above and a CPU device.

    ``polish(positions, minimizer, network, params, box, scratch, workers,
    seed)`` is the call of the validity hooks in the archived production
    params files, which called it by its old private name
    ``aimmd.flow.validate._polish``; that name still works for one release.

    Examples
    --------
    >>> results = polish(samples, GromacsMinimizer(topology='solute.top', em_mdp='em.mdp',  # doctest: +SKIP
    ...                                            grompp='/path/to/gmx grompp',
    ...                                            mdrun='/path/to/gmx mdrun -ntomp 1'),
    ...                  network, SimpleNamespace(universe=universe, parent='.'),
    ...                  universe.dimensions, 'scratch/polish', workers=4)
    >>> polished = [r.positions for r in results if r.ok]                       # doctest: +SKIP
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    from ..types import CandidateContext

    box = np.asarray(box, dtype=float).reshape(-1)
    if box.shape != (6,):
        raise ValueError(f'the polish box must be the 6 numbers of a box (A, degrees), got {box}')
    stop = must_stop or (lambda: False)

    def one(item):
        k, x = item
        center = x.mean(0)
        frame = PreparedFrame(positions=x - center + box[:3] / 2, dimensions=box.copy())
        context = CandidateContext(
            params=params, network=network, network_info=None, directory=str(scratch),
            folder=str(scratch), path_name=f'polish{k:06d}', scratch=str(scratch),
            rng=np.random.default_rng(seed + k), seed=int(seed + k),
            generator=torch.Generator(), device=torch.device('cpu'), must_stop=stop,
            log=lambda message: None, settings={'keep_scratch': bool(keep_scratch)})
        result = minimizer(frame, context)
        record = dict(getattr(result, 'record', {}) or {})
        record.pop('settings', None)
        if not result.ok:
            return PolishResult(None, False, str(result.reason), record)
        y = np.asarray(result.positions, dtype=float)[:len(x)]
        return PolishResult(y - y.mean(0) + center, True, '', record)

    items = list(enumerate(np.asarray(positions, dtype=float)))
    if int(workers) <= 1:
        return [one(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(one, items))
