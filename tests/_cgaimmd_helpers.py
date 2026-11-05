"""Shared helpers of the C-GAIMMD tests.

Every C-GAIMMD stream tests against these, so that no stream needs another
stream's code to build its fixtures:

- :func:`write_fake_network_file`: ``network{s}.json`` + ``network{s}.h5`` as
  the trainer writes them;
- :func:`make_candidate_context`: a :class:`aimmd.cgaimmd.types.CandidateContext`
  for hook and worker tests;
- :func:`cgaimmd_params_namespace`: a ``Params.placeholder`` copy configured
  for C-GAIMMD (no validation), as ``tests/test_worker_runtime_unit.py`` does
  for RFPS;
- :func:`build_cgaimmd_run_dir`: a run folder with shooting chains, free
  trajectories and their caches;
- :class:`ScriptedShootingPointSource`: a stand-in for the shooting-point
  source of ``aimmd.worker._shoot`` that hands out scripted shooting points;
- :func:`gromacs_has_thread_mpi` and :func:`mdrun_thread_options`: mdrun
  thread options that work with a thread-MPI ``gmx`` and an MPI ``gmx_mpi``.
"""

import datetime
import functools
import hashlib
import io
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.io import atomic_write_bytes, atomic_write_json
from aimmd.cgaimmd.sidecar import SIDECAR_SCHEMA, SIDECAR_SOURCE
from aimmd.cgaimmd.types import CandidateContext, ShootingPoint
from aimmd.params.utils import canonical_cgaimmd_parameters, create_default_values_function
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import state_labels_from_positions, write_trajectory


def flat_coordinates_descriptors(trajectory):
    """Descriptors of C-GAIMMD: ``(n_frames, 3 N)`` float32 Å, x0, y0, z0, x1, ..."""
    if hasattr(trajectory, 'positions') and np.asarray(trajectory.positions).ndim == 2:
        positions = np.asarray(trajectory.positions, dtype=np.float32)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in trajectory], dtype=np.float32)
    return positions.reshape(len(positions), -1)


def x_states_function(trajectory, states='ARB'):
    """States from the x of atom 0: ``x <= -0.5`` first state, ``x >= 0.5`` last, else R."""
    if hasattr(trajectory, 'positions') and np.asarray(trajectory.positions).ndim == 2:
        positions = np.asarray(trajectory.positions, dtype=float)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in trajectory], dtype=float)
    return state_labels_from_positions(positions, states=states)


# ---------------------------------------------------------------------------
# network files
# ---------------------------------------------------------------------------

def write_fake_network_file(directory, network, states='ARB', round=1, n_trained_shots=0, *,
                            metadata=None, write_json=True):
    """Write ``network{states}.h5`` and its ``network{states}.json`` as the trainer does.

    Parameters
    ----------
    directory : str or os.PathLike
        Run directory.
    network : torch.nn.Module
        Whose ``state_dict()`` is saved.
    states : str, optional
        ``params.sorted_states``.
    round, n_trained_shots : int, optional
        The metadata of the round that "produced" the network.
    metadata : dict, optional
        Extra metadata keys (override the defaults).
    write_json : bool, optional
        False writes the ``.h5`` only (a missing-metadata scenario).

    Returns
    -------
    str
        The sha256 of the ``.h5`` bytes. The json
        (``{**network_metadata, 'sha256', 'saved', 'total_steps'}``) is written
        *before* the ``.h5``, as ``save_network_atomically`` does.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(network.state_dict(), buffer)
    data = buffer.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    if write_json:
        record = {'round': round, 'n_trained_shots': n_trained_shots, 'step': 0,
                  'fit': 'tests._cgaimmd_helpers', 'ema_decay': None, 'sha256': sha,
                  'saved': datetime.datetime.now(datetime.timezone.utc).isoformat(
                      timespec='seconds'),
                  'total_steps': 0}
        record.update(metadata or {})
        atomic_write_json(directory / f'network{states}.json', record)
    atomic_write_bytes(directory / f'network{states}.h5', data)
    return sha


# ---------------------------------------------------------------------------
# GROMACS builds
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def gromacs_has_thread_mpi(gmx):
    """Whether the GROMACS executable `gmx` is a thread-MPI build (``gmx --version``).

    An MPI build (``gmx_mpi`` on JUPITER and Goethe) refuses ``mdrun -ntmpi``.
    When the version cannot be read, a thread-MPI build is assumed.
    """
    try:
        out = subprocess.run([str(gmx), '--version'], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    for line in out.splitlines():
        if line.strip().lower().startswith('mpi library:'):
            return line.split(':', 1)[1].strip().lower() == 'thread_mpi'
    return True


def mdrun_thread_options(gmx, ntomp=1):
    """``'-ntmpi 1 -ntomp N'`` for a thread-MPI build of `gmx`, ``'-ntomp N'`` for an MPI build."""
    ntomp = f'-ntomp {int(ntomp)}'
    return f'-ntmpi 1 {ntomp}' if gromacs_has_thread_mpi(gmx) else ntomp


# ---------------------------------------------------------------------------
# shooting-point source
# ---------------------------------------------------------------------------

class ScriptedShootingPointSource:
    """Scripted stand-in for ``aimmd.cgaimmd.source.GenerativeShootingPointSource``.

    Parameters
    ----------
    params, worker, directory
        As the real source receives them (stored, not used).
    points : iterable, optional
        What successive :meth:`next_shooting_point` calls return: ``(N, 3)``
        arrays (wrapped into a :class:`ShootingPoint` with ``dimensions`` from
        ``params.universe`` if it has a box), ``ShootingPoint`` objects (their
        record gets ``path_name``/``chain`` filled in), or None (returned as is,
        as after ``must_stop``). Exhausted: None.
    wait_result : bool, optional
        What :meth:`wait_for_network` returns.

    Attributes
    ----------
    calls : list of (str, dict)
        Every method call with its arguments, in order.
    records : list of dict
        The records handed out with the shooting points.
    counters : dict
        ``{'state_readback': n, ...}`` as the real source counts.
    consecutive_readback : int
        Read-back rejections since the last registered shot.
    watcher, pool
        None (the interface names them; the stand-in has neither).
    """

    def __init__(self, params, worker, directory, *, points=(), wait_result=True):
        self.params = params
        self.worker = worker
        self.directory = directory
        self._points = list(points)
        self.wait_result = wait_result
        self.calls = [('__init__', {'directory': directory})]
        self.records = []
        self.counters = {}
        self.consecutive_readback = 0
        self.watcher = None
        self.pool = None

    def wait_for_network(self):
        self.calls.append(('wait_for_network', {}))
        return self.wait_result

    def _dimensions(self):
        universe = getattr(self.params, 'universe', None)
        dimensions = getattr(universe, 'dimensions', None) if universe is not None else None
        return None if dimensions is None else np.asarray(dimensions, dtype=np.float32)

    def next_shooting_point(self, *, chain, folder, path_name):
        self.calls.append(('next_shooting_point',
                           {'chain': chain, 'folder': folder, 'path_name': path_name}))
        if not self._points:
            return None
        point = self._points.pop(0)
        if point is None:
            return None
        base = {'schema': SIDECAR_SCHEMA, 'source': SIDECAR_SOURCE,
                'note': 'ScriptedShootingPointSource', 'lambda_target': 0.0}
        if isinstance(point, ShootingPoint):
            record = {**base, **point.record}
            positions, dimensions = point.positions, point.dimensions
        else:
            record = dict(base)
            positions, dimensions = point, self._dimensions()
        record.update(path_name=path_name, chain=os.path.basename(os.path.normpath(folder)))
        self.records.append(record)
        return ShootingPoint(np.asarray(positions, dtype=np.float32), dimensions, record)

    def note_glitch(self, folder, exc, record):
        self.calls.append(('note_glitch', {'folder': folder, 'exc': exc, 'record': record}))
        self.counters['state_readback'] = self.counters.get('state_readback', 0) + 1
        self.consecutive_readback += 1

    def note_registered(self):
        self.calls.append(('note_registered', {}))
        self.consecutive_readback = 0

    def called(self, name):
        """The argument dicts of every call of method `name`."""
        return [kwargs for method, kwargs in self.calls if method == name]


# ---------------------------------------------------------------------------
# params and contexts
# ---------------------------------------------------------------------------

def cgaimmd_params_namespace(**overrides):
    """A ``Params.placeholder`` copy set up for C-GAIMMD, without validation.

    Defaults: ``chain_type='cgaimmd'``, ``states='ARB'``, the toy engine,
    ``trajectory_extension='.xtc'``, ``network=TinyFlowNetwork()``, flat
    coordinate descriptors (:func:`flat_coordinates_descriptors`), x-coordinate
    states (:func:`x_states_function`), the default values function of the
    network, ``round_shooting_index=True``, ``gen_temperature=300.0``,
    ``cgaimmd_parameters=None`` and ``cgaimmd_generator=None``.

    `overrides` are assigned through ``__dict__`` (as
    ``tests/test_worker_runtime_unit.py`` does); ``cgaimmd_parameters`` is
    canonicalized first, so ``params.cgaimmd_settings`` works, and a new
    ``network`` without a ``values_function`` gets its default values function.
    """
    params = aimmd.Params.placeholder.copy()
    # built only when needed: TinyFlowNetwork's init draws from the global torch RNG
    network = overrides['network'] if 'network' in overrides else TinyFlowNetwork()
    values = dict(states='ARB', chain_type='cgaimmd', engine='toy', trajectory_extension='.xtc',
                  network=network, descriptors_function=flat_coordinates_descriptors,
                  states_function=x_states_function,
                  values_function=create_default_values_function(network),
                  round_shooting_index=True, gen_temperature=300.0, cgaimmd_parameters=None,
                  cgaimmd_generator=None)
    values.update(overrides)
    values['cgaimmd_parameters'] = canonical_cgaimmd_parameters(values['cgaimmd_parameters'])
    params.__dict__.update(values)
    params.__dict__['_default_values_function'] = 'values_function' not in overrides
    return params


class LogRecorder:
    """A ``log(message)`` callable that keeps the messages."""

    def __init__(self):
        self.messages = []

    def __call__(self, message):
        self.messages.append(str(message))


def make_candidate_context(tmp_path, params=None, network=None, **settings):
    """A :class:`CandidateContext` for hook and worker tests.

    Parameters
    ----------
    tmp_path : os.PathLike
        The run directory is ``tmp_path/'run'``, the folder
        ``run/chainR0`` (created, with ``cgaimmd_scratch/``).
    params : aimmd.Params, optional
        Default: :func:`cgaimmd_params_namespace` with `network` and the
        `settings` as its ``cgaimmd_parameters``.
    network : optional
        Default: ``params.network``.
    **settings
        C-GAIMMD settings (validated like ``cgaimmd_parameters``); the
        context's ``settings`` are ``params.cgaimmd_settings`` updated with them.

    Returns
    -------
    CandidateContext
        With ``rng``/``seed`` 0, a CPU ``torch.Generator`` seeded 0, device CPU,
        ``must_stop`` returning False and ``log`` a :class:`LogRecorder`
        (``context.log.messages``).
    """
    settings = canonical_cgaimmd_parameters(settings) or {}
    if params is None:
        extra = {} if network is None else {'network': network}
        params = cgaimmd_params_namespace(cgaimmd_parameters=settings or None, **extra)
    network = params.network if network is None else network
    directory = Path(tmp_path) / 'run'
    folder = directory / 'chainR0'
    scratch = folder / 'cgaimmd_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator()
    generator.manual_seed(0)
    return CandidateContext(
        params=params, network=network, network_info=None, directory=str(directory),
        folder=str(folder), path_name='path000001', scratch=str(scratch),
        rng=np.random.default_rng(0), seed=0, generator=generator,
        device=torch.device('cpu'), must_stop=lambda: False, log=LogRecorder(),
        settings={**params.cgaimmd_settings, **settings})


# ---------------------------------------------------------------------------
# run folders
# ---------------------------------------------------------------------------

def _write_with_caches(fname, positions, times, states='ARB'):
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    written = write_trajectory(fname.parent, stem=fname.name[:-len(fname.suffix)],
                               ext=fname.suffix, positions=positions, times=times)
    stored = aimmd.Path(written).positions                  # what the file holds (xtc precision)
    save_npy(get_cache_fname(written, 'states'),
             state_labels_from_positions(stored, states=states))
    save_npy(get_cache_fname(written, 'descriptors'),
             np.asarray(stored, dtype=np.float32).reshape(len(stored), -1))
    save_npy(get_cache_fname(written, 'values'), np.asarray(stored[:, 0, 0], dtype=float))
    return written


def _shot_positions(n_atoms, x):
    positions = np.zeros((len(x), n_atoms, 3), dtype=np.float32)
    positions[:, :, 0] = np.asarray(x, dtype=np.float32)[:, None]
    positions[:, :, 1] = 1.5 * np.arange(n_atoms, dtype=np.float32)[None, :]
    return positions


def build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3, n_free=1, n_atoms=1, *,
                          states='ARB', ext='.xtc', n_parts=1):
    """A run folder with C-GAIMMD-style shooting chains and free trajectories.

    Parameters
    ----------
    tmp_path : os.PathLike
        The run is ``tmp_path/'run'``.
    n_chains, n_shots : int, optional
        ``run/chain{R}{k}/path{n:06g}{ext}`` for ``k < n_chains``, ``n <=
        n_shots``: 7-frame shots, alternately first state -> last state and
        last -> first, with one end frame in each state (as registered shots
        have), whose shooting frame (index 3, time 0; the times run
        3, 2, 1, 0, 1, 2, 3 as the shoot worker writes a path, so
        ``Path.find_shooting_index`` finds it) is in R; each shot is shifted a
        little so that no two shooting frames coincide.
    n_free : int, optional
        ``run/free{A}/traj{n:06g}.part0001{ext}`` and the same in the last
        state: 10 frames deep in the state.
    n_atoms : int, optional
        Atoms per frame; atom ``i`` sits at ``y = 1.5 i`` and shares atom 0's
        x, which alone decides the state (x <= -0.5 first state, x >= 0.5
        last, see :func:`x_states_function`).
    n_parts : int, optional
        Part files per free trajectory, ``.part0001`` to ``.part{n_parts:04d}``,
        10 frames each. Every later part starts with the frame (and time) its
        predecessor ended with, as the parts of a restarted ``gmx mdrun`` do,
        so the trajectory has ``9 n_parts + 1`` frames.

    Returns
    -------
    types.SimpleNamespace
        ``run`` (str), ``chains`` (list of lists of :class:`aimmd.Path`, each
        with ``shooting_index``), ``shots`` (flat list), ``free`` (list of
        :class:`aimmd.Path`, the part files joined with
        ``remove_overlapping_frames=True``), ``states``, ``shooting_index``.
        Every file has ``.states``, ``.descriptors`` (flat Å, float32) and
        ``.values`` (x of atom 0) caches.
    """
    run = Path(tmp_path) / 'run'
    run.mkdir(parents=True, exist_ok=True)
    first, r, last = states
    shooting_index = 3
    base = np.array([-1.0, -0.4, -0.2, 0.0, 0.2, 0.4, 1.0])   # one end frame per state
    chains, shots = [], []
    for k in range(n_chains):
        chain = []
        for n in range(1, n_shots + 1):
            shift = 0.01 * ((k * n_shots + n) % 9)            # R frames stay below 0.5
            x = base + shift * (np.abs(base) < 0.5)
            if (k + n) % 2:
                x = x[::-1].copy()
            # as the shoot worker writes a path: the backward branch reversed, so the
            # times run si, ..., 1, 0, 1, ... (Path.find_shooting_index reads them)
            times = np.abs(np.arange(len(x), dtype=float) - shooting_index)
            fname = _write_with_caches(run / f'chain{r}{k}' / f'path{n:06g}{ext}',
                                       _shot_positions(n_atoms, x), times, states=states)
            path = aimmd.Path(fname, shooting_index=shooting_index)
            chain.append(path)
            shots.append(path)
        chains.append(chain)
    free = []
    rng = np.random.default_rng(0)
    for state, center in ((first, -1.5), (last, 1.5)):
        for n in range(1, n_free + 1):
            x = center + 0.1 * rng.standard_normal(10)
            times = np.arange(10, dtype=float)
            fnames = []
            for part in range(1, n_parts + 1):
                if part > 1:                     # a restart repeats the last frame written
                    x = np.concatenate([x[-1:], center + 0.1 * rng.standard_normal(9)])
                    times = times[-1] + np.arange(10, dtype=float)
                fnames.append(_write_with_caches(
                    run / f'free{state}' / f'traj{n:06g}.part{part:04d}{ext}',
                    _shot_positions(n_atoms, x), times, states=states))
            free.append(aimmd.Path(fnames[0]) if n_parts == 1
                        else aimmd.Path(fnames, remove_overlapping_frames=True))
    return SimpleNamespace(run=str(run), chains=chains, shots=shots, free=free, states=states,
                           shooting_index=shooting_index)
