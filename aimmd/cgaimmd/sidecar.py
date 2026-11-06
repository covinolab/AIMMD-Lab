"""
aimmd.cgaimmd.sidecar
=====================

The in-flight record of a C-GAIMMD shot, the per-shot sidecar, their
reconciliation after a restart, the provenance they record, and the per-shot
table of a run.

A generated shooting point leaves no trace in AIMMD's own files: once the two
segments are initialized, nothing says at which target committor it was
generated, by which network and generator, or what was rejected before it.
C-GAIMMD shooting therefore writes

- an **in-flight record** per chain folder while a shot is simulated,
  ``chain{t}{k}/.cgaimmd_shooting_point.json`` (the record) and ``.npy`` (the
  shooting-point coordinates), written just before the segments are
  initialized and cleared after registration;
- a **sidecar** per registered path, ``chain{t}{k}/path{n:06g}{ext}.cgaimmd.json``:
  the record plus what registration adds (the registered name, the path type,
  ``lambda_registered``, ...).

A record is attached only to the path it names or, after AIMMD renumbered the
shot at a restart, to a path whose shooting frame matches its coordinates;
otherwise it is quarantined under ``stale_record`` with ``source='unknown'``.
None of the file names matches ``*back*``, ``*forw*`` or ``path??????{ext}``,
so AIMMD's per-shot clean-up and path loaders never touch them. The
reweighting reads nothing from the sidecars: they exist so that the
shooting-point law can be studied and corrected later.

What a sidecar records about the software and the run (:func:`provenance`):
the aimmd version, its git commit (or the reason why there is none) and a
digest of the package's source files; the versions of Python, numpy, torch,
MDAnalysis, bioemu2 and cuequivariance (read from the installed
distributions, nothing is imported); the network's own provenance (for
BioEmu2: the checkpoint, its Hugging Face snapshot and the parameter
checksum); the generator's method and settings; and the PLUMED bias settings
when the run has a bias (:func:`bias_settings`).

Old names are read for one release (:mod:`aimmd.cgaimmd._legacy`): the
markers ``.flow_shooting_point.{json,npy}`` and the sidecars
``*.xtc.flow.json`` of runs of the old code are found, and their schema,
source and generator labels are translated to the new names, each with one
``DeprecationWarning``. Only new names are written.

:func:`shot_table` and :func:`summarise_shots` turn the sidecars and state
caches of a run folder into one row per shot and a summary, optionally with
collective variables of every shooting point; they only read (no cache, lock
or offsets file is created in the run folder).

Imports only the standard library, numpy and light C-GAIMMD modules at module
level (``aimmd.worker._shoot`` imports it for every C-GAIMMD run).
"""

import datetime
import functools
import glob
import hashlib
import importlib.metadata
import io as _io
import json
import math
import os
import platform
import subprocess
import sys
import types

import numpy as np

from . import _legacy
from .io import atomic_write_bytes, atomic_write_json


SIDECAR_SCHEMA = 'aimmd.cgaimmd.shot/1'
"""Schema tag of in-flight records and sidecars."""

SIDECAR_SOURCE = 'cgaimmd'
"""The ``source`` of a record of a generated shot."""

SIDECAR_SUFFIX = '.cgaimmd.json'
"""The sidecar of ``path{n:06g}{ext}`` is ``path{n:06g}{ext}`` + this suffix."""

IN_FLIGHT_MARKER = '.cgaimmd_shooting_point'
"""The in-flight marker of a chain folder: this stem + ``.json`` (record) and ``.npy`` (coordinates)."""

SP_MATCH_RMS = 0.05
"""RMS in Å (after removing each centroid) below which a registered shooting frame
is the in-flight configuration (xtc stores coordinates to 0.01 Å)."""

SIDECAR_KEYS = (
    'schema', 'source', 'note', 'stale_record', 'path', 'chain', 'created', 'registered',
    'recovered_after_restart', 'lambda_target', 'lambda_raw', 'lambda_final',
    'lambda_registered', 'state_final', 'lambda_law', 'acceptance', 'network', 'generator',
    'pool', 'stage1_rejected', 'candidates', 'validity', 'prepare', 'minimize',
    'cost_seconds', 'seed', 'device', 'aimmd', 'software', 'bias', 'registration')
"""The keys every sidecar holds; ``None`` where not applicable."""

REGISTRATION_KEYS = ('path', 'registered', 'lambda_registered', 'registration',
                     'recovered_after_restart')
"""The keys of :data:`SIDECAR_KEYS` that registration adds (an in-flight record has none of them)."""

SOFTWARE_DISTRIBUTIONS = {
    'numpy': 'numpy', 'torch': 'torch', 'mdanalysis': 'MDAnalysis', 'bioemu2': 'bioemu2',
    'cuequivariance': 'cuequivariance', 'cuequivariance_torch': 'cuequivariance-torch'}
"""Sidecar key -> installed distribution whose version ``software`` records (None if not installed)."""

_RENUMBERED_NOTE = ('registered under a different index than expected; shooting point '
                    'verified by coordinates')


def _now():
    """ISO 8601 UTC time, to the second."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _log(message):
    """Print into the worker log (AIMMD's flushing print, set by ``aimmd._init``)."""
    from .. import _config
    getattr(_config, 'print', print)(message)


# ════════════════════════════════════════════════════════════════════════════
# File names (new names written, old names read)
# ════════════════════════════════════════════════════════════════════════════

def in_flight_fname(folder):
    """The in-flight record of a chain folder: ``f'{folder}/.cgaimmd_shooting_point.json'``."""
    return f'{folder}/{IN_FLIGHT_MARKER}.json'


def in_flight_positions_fname(folder):
    """The in-flight coordinates of a chain folder: ``f'{folder}/.cgaimmd_shooting_point.npy'``."""
    return f'{folder}/{IN_FLIGHT_MARKER}.npy'


def _old_name(new_basename, table):
    """The old file name of `new_basename` in a ``_legacy`` table (old -> new)."""
    for old, new in table.items():
        if new == new_basename:
            return old
    raise KeyError(new_basename)


def _old_in_flight(folder, suffix):
    return f'{folder}/{_old_name(IN_FLIGHT_MARKER + suffix, _legacy.IN_FLIGHT_MARKERS)}'


def _existing_in_flight(folder):
    """``(json, npy)`` of the marker of `folder`: the new one, else the old one, else None.

    An old marker (a run of the old code that stopped mid-shot) is found with
    one ``DeprecationWarning``.
    """
    new = in_flight_fname(folder)
    if os.path.exists(new):
        return new, in_flight_positions_fname(folder)
    old = _old_in_flight(folder, '.json')
    if os.path.exists(old):
        _legacy.translate(_legacy.IN_FLIGHT_MARKERS, os.path.basename(old), 'in-flight marker')
        return old, _old_in_flight(folder, '.npy')
    return None


def sidecar_fname(path_fname):
    """The sidecar of a registered path file: ``f'{path_fname}.cgaimmd.json'``."""
    return f'{path_fname}{SIDECAR_SUFFIX}'


def _old_sidecar_fname(path_fname):
    return f'{path_fname}{_old_name(SIDECAR_SUFFIX, _legacy.SIDECAR_SUFFIXES)}'


def _translated(record):
    """`record` (a dict read from a file) with the old schema, source and generator label renamed.

    Each old value gives one ``DeprecationWarning`` (:func:`aimmd.cgaimmd._legacy.translate`).
    A quarantined ``stale_record`` is translated too. The dict is changed in place and returned.
    """
    if not isinstance(record, dict):
        return record
    for key, table, what in (('schema', _legacy.SIDECAR_SCHEMAS, 'sidecar schema'),
                             ('source', _legacy.SIDECAR_SOURCES, 'sidecar source')):
        if key in record:
            record[key] = _legacy.translate(table, record[key], what)
    generator = record.get('generator')
    if isinstance(generator, dict) and 'method' in generator:
        generator['method'] = _legacy.translate(_legacy.GENERATOR_LABELS, generator['method'],
                                                'sidecar generator method')
    if isinstance(record.get('stale_record'), dict):
        _translated(record['stale_record'])
    return record


# ════════════════════════════════════════════════════════════════════════════
# In-flight records
# ════════════════════════════════════════════════════════════════════════════

def write_in_flight(folder, record, positions):
    """Record the shot about to be simulated in `folder`.

    Parameters
    ----------
    folder : str
        Chain folder.
    record : dict
        The in-flight record with ``'path_name'``; it is not modified.
    positions : array-like
        ``(N, 3)`` shooting-point coordinates in Å (stored as float32).

    Raises
    ------
    ValueError
        If the record has no ``'path_name'`` or `positions` is not ``(N, 3)``.

    Notes
    -----
    ``'path_name'`` is the name the path is expected to be registered under
    (``'path000012'``). The coordinates (``.npy``) are written first, then
    the record (``.json``), both atomically; the stored record adds
    ``'positions_sha256'`` (the sha256 of the ``.npy`` file bytes) and
    ``'created'`` (now, unless the record has a non-empty value). An old-name
    marker of the folder is removed.
    """
    if not record or not record.get('path_name'):
        raise ValueError("the in-flight record needs 'path_name' (the name expected at "
                         "selection, e.g. 'path000012')")
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f'in-flight positions must have shape (N, 3), got {positions.shape}')
    buffer = _io.BytesIO()
    np.save(buffer, positions)
    data = buffer.getvalue()
    stored = dict(record)
    stored['positions_sha256'] = hashlib.sha256(data).hexdigest()
    if not stored.get('created'):
        stored['created'] = _now()
    _remove(_old_in_flight(folder, '.json'), _old_in_flight(folder, '.npy'))
    atomic_write_bytes(in_flight_positions_fname(folder), data)
    atomic_write_json(in_flight_fname(folder), stored)


def read_in_flight(folder):
    """The in-flight record of `folder`.

    Parameters
    ----------
    folder : str
        Chain folder.

    Returns
    -------
    dict or None
        The record; None if there is none.

    Notes
    -----
    The new marker is read, else the old one (``.flow_shooting_point.json``,
    one ``DeprecationWarning``), whose schema, source and generator label are
    translated to the new names. An unreadable marker gives a record with
    ``source='unknown'``, ``path_name=None`` and the error in ``note`` (it
    can then never be attached to a path by name).
    """
    found = _existing_in_flight(folder)
    if found is None:
        return None
    fname = found[0]
    try:
        with open(fname) as handle:
            return _translated(json.load(handle))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return {'schema': SIDECAR_SCHEMA, 'source': 'unknown', 'path_name': None,
                'note': f'unreadable in-flight record {fname}: {exc}'}


def _remove(*fnames):
    for fname in fnames:
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass


def clear_in_flight(folder):
    """Remove the in-flight record and coordinates of `folder`, new and old names (missing files are fine)."""
    _remove(in_flight_fname(folder), in_flight_positions_fname(folder),
            _old_in_flight(folder, '.json'), _old_in_flight(folder, '.npy'))


def shooting_point_matches(path, folder):
    """Whether a registered path's shooting frame is the in-flight configuration of `folder`.

    Parameters
    ----------
    path : aimmd.Path
        A registered path with ``shooting_index`` set.
    folder : str
        Chain folder holding the in-flight ``.npy`` (new or old name).

    Returns
    -------
    bool or None
        True if the frames agree within :data:`SP_MATCH_RMS`; None if it cannot be decided.

    Notes
    -----
    The RMS difference is taken after removing each centroid; another atom
    count gives False; no ``.npy``, no shooting index or unreadable data give
    None.
    """
    found = _existing_in_flight(folder)
    fname = in_flight_positions_fname(folder) if found is None else found[1]
    index = getattr(path, 'shooting_index', None)
    if index is None or not os.path.isfile(fname):
        return None
    try:
        expected = np.load(fname).astype(np.float64)
        actual = np.asarray(path[int(index)].positions, dtype=np.float64)
    except Exception:                                       # noqa: BLE001
        return None
    if expected.shape != actual.shape:
        return False
    expected = expected - expected.mean(axis=0)
    actual = actual - actual.mean(axis=0)
    rms = float(np.sqrt(((expected - actual) ** 2).sum(axis=1).mean()))
    return rms < SP_MATCH_RMS


def unknown_origin_record(folder, path_name):
    """The record of a shot that resumed after a restart without an in-flight record.

    Parameters
    ----------
    folder : str
        Chain folder.
    path_name : str
        The name the resumed shot is expected to be registered under.

    Returns
    -------
    dict
        ``schema``, ``source='resumed_without_cgaimmd_record'``, ``path_name``, ``chain``,
        ``created``, ``note``.

    Notes
    -----
    Such a shot was selected before the run became a C-GAIMMD run (an RFPS
    shot in flight when the run was continued as ``chain_type='cgaimmd'``) or
    its record was lost.
    """
    return {'schema': SIDECAR_SCHEMA, 'source': 'resumed_without_cgaimmd_record',
            'path_name': path_name, 'chain': os.path.basename(os.path.normpath(folder)),
            'created': _now(),
            'note': 'the shot resumed after a restart without an in-flight C-GAIMMD record '
                    '(selected before C-GAIMMD shooting, or the record was lost)'}


# ════════════════════════════════════════════════════════════════════════════
# Sidecars
# ════════════════════════════════════════════════════════════════════════════

def _stem(fname, ext):
    base = os.path.basename(fname)
    if ext and base.endswith(ext):
        return base[:-len(ext)]
    return os.path.splitext(base)[0]


def lambda_registered(path, params):
    """The committor of a registered path's shooting frame, and why it is missing if it is.

    Parameters
    ----------
    path : aimmd.Path
        A registered path (``shooting_index`` set).
    params : aimmd.Params
        Provides the descriptor and value functions (``compute_descriptors_args``,
        ``compute_values_args``, both of ``params.descriptors_series``).

    Returns
    -------
    tuple
        ``(value, reason)``: the finite float value and None, or None and a
        sentence saying why there is no value.

    Notes
    -----
    The shooting frame's descriptor row is computed first where the cache of
    the run's descriptor series lacks it
    (:func:`aimmd.worker.utils.ensure_descriptors`; a registered path normally
    has it, from the segments' pipeline). The value is then computed from
    that row with ``params.values_function`` and **not** cached (the values
    series stays the trainer's). An exception of either step is a reason,
    not an error: the sidecar of a registered shot is always written.
    """
    index = getattr(path, 'shooting_index', None)
    if index is None:
        return None, 'the registered path has no shooting index'
    frame = path[int(index):int(index) + 1]
    try:
        from ..worker.utils import ensure_descriptors
        ensure_descriptors(frame, params)
        function, _, source = params.compute_values_args
        values = frame.compute(function, '', source, return_result=True)
        values = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:                                # noqa: BLE001
        return None, (f'computing the value of the shooting frame raised '
                      f'{type(exc).__name__}: {exc}')
    if not len(values):
        return None, (f'no descriptor row of series {source!r} for the shooting frame '
                      f'{int(index)}, and none could be computed')
    value = float(values[0])
    if not math.isfinite(value):
        return None, f'the value of the shooting frame is not finite ({value})'
    return value, None


def _package_version():
    try:
        from .. import __version__
        return __version__
    except Exception:                                       # noqa: BLE001
        return None


def _shot_sidecar(path, record, *, params, states, target_state, recovered_after_restart):
    fname = path.fname
    folder = os.path.dirname(os.path.abspath(fname))
    ext = getattr(params, 'trajectory_extension', None) or os.path.splitext(fname)[1]
    stem = _stem(fname, ext)
    data = dict.fromkeys(SIDECAR_KEYS)
    if record is None:
        data['source'] = 'unknown'
        data['note'] = 'no in-flight record for this shot'
    elif record.get('path_name') == stem:
        data.update(record)
    elif shooting_point_matches(path, folder):
        data.update(record)
        data['note'] = (f"{record['note']}; {_RENUMBERED_NOTE}" if record.get('note')
                        else _RENUMBERED_NOTE)
    else:
        data['source'] = 'unknown'
        data['note'] = (f"stale in-flight record (expected {record.get('path_name')!r}, "
                        f"registered {stem!r}; shooting point not verified)")
        data['stale_record'] = record
    data['schema'] = SIDECAR_SCHEMA
    data['source'] = data.get('source') or SIDECAR_SOURCE
    data['path'] = os.path.basename(fname)
    data['chain'] = os.path.basename(folder)
    data['registered'] = _now()
    data['recovered_after_restart'] = bool(recovered_after_restart)
    index = getattr(path, 'shooting_index', None)
    value, reason = lambda_registered(path, params)
    data['lambda_registered'] = value
    data['registration'] = {
        'n_frames': int(len(path)),
        'path_type': str(path.type),
        'shooting_index': None if index is None else int(index),
        'complete': bool(path.is_complete(target_state, states)),
        'lambda_registered_missing': reason,
    }
    if reason is not None:
        _log(f'WARNING: no lambda_registered for {fname}: {reason}')
    if not data.get('aimmd'):
        data['aimmd'] = aimmd_provenance()
    if not data.get('software'):
        data['software'] = software_versions()
    if data.get('bias') is None:
        data['bias'] = bias_settings(params)
    out = sidecar_fname(fname)
    atomic_write_json(out, data)
    return out


def write_shot_sidecar(path, record, *, params, states, target_state):
    """Write the sidecar of a freshly registered path.

    Parameters
    ----------
    path : aimmd.Path
        The registered path (renamed by ``register_path``, ``shooting_index`` set).
    record : dict or None
        The in-flight record of the shot.
    params : aimmd.Params
        Provides the descriptor and value functions and ``trajectory_extension``.
    states : str
        ``params.states``.
    target_state : str
        The chain's target state (``states[1]`` for C-GAIMMD chains).

    Returns
    -------
    str
        The sidecar file name, ``f'{path.fname}.cgaimmd.json'``.

    Notes
    -----
    The record is attached if its ``'path_name'`` is the registered name;
    else only if :func:`shooting_point_matches` (with the note "registered
    under a different index than expected; shooting point verified by
    coordinates"); else the sidecar gets ``source='unknown'`` and the record
    under ``'stale_record'``. Without a record the source is ``'unknown'``
    too. Registration adds ``'registered'`` (ISO time), ``'path'``,
    ``'chain'``, ``'registration'`` (``n_frames``, ``path_type``,
    ``shooting_index``, ``complete``, ``lambda_registered_missing``) and
    ``'lambda_registered'`` (:func:`lambda_registered`: the committor of the
    registered shooting frame; when it cannot be had, None, with the reason
    in ``registration['lambda_registered_missing']`` and a WARNING in the
    worker log). A record without ``aimmd``, ``software`` or ``bias`` gets
    them now. Every key of :data:`SIDECAR_KEYS` is present (None where not
    applicable); extra keys of the record are kept. The write is atomic.
    """
    return _shot_sidecar(path, record, params=params, states=states, target_state=target_state,
                         recovered_after_restart=False)


def read_shot_sidecar(path_fname):
    """The sidecar of a registered path file (new or old name).

    Parameters
    ----------
    path_fname : str
        The path file (``chain{t}{k}/path{n:06g}{ext}``), not the sidecar.

    Returns
    -------
    dict or None
        The sidecar's JSON object; None if the path has no sidecar.

    Raises
    ------
    ValueError
        If the sidecar exists but is not valid JSON (``json.JSONDecodeError``).

    Notes
    -----
    ``*.cgaimmd.json`` is read, else the old ``*.flow.json`` of a run of the
    old code, whose schema, source and generator label are translated to the
    new names (one ``DeprecationWarning`` per old value). The file is only
    read.
    """
    for fname, old in ((sidecar_fname(path_fname), False),
                       (_old_sidecar_fname(path_fname), True)):
        try:
            with open(fname) as handle:
                data = json.load(handle)
        except FileNotFoundError:
            continue
        if old:
            _legacy.translate(_legacy.SIDECAR_SUFFIXES,
                              _old_name(SIDECAR_SUFFIX, _legacy.SIDECAR_SUFFIXES),
                              'sidecar suffix')
        return _translated(data)
    return None


def _chain_path(chain, fname):
    """The Path of `chain` stored in `fname` (for its shooting_index), else a fresh one."""
    target = os.path.abspath(fname)
    for path in (chain or []):
        try:
            if os.path.abspath(path.fname) == target:
                return path
        except Exception:                                   # noqa: BLE001
            continue
    from ..path import Path
    return Path(fname)


def reconcile_in_flight(folder, chain, params):
    """Bring the in-flight marker of `folder` in line with the chain when a shooting worker starts.

    Parameters
    ----------
    folder : str
        Chain folder.
    chain : aimmd.PathEnsemble
        The loaded chain (``params.shot_chains(...)``), for a recovered path's shooting index.
    params : aimmd.Params
        Provides ``trajectory_extension``, ``states``, ``check_if_initialized``, the values
        function. ``params.network`` must hold the run's network: the shooting
        worker calls this only after its source has loaded ``network<s>.h5``
        (``wait_for_network``), because a recovered sidecar's
        ``lambda_registered`` is that network's committor of the shooting frame.
        Before the load, ``params.network`` is the network the params file
        builds (untrained), whose value would be recorded as if it were valid.

    Returns
    -------
    str
        ``'none'``, ``'recovered'``, ``'cleared'`` or ``'kept'`` (see the notes).

    Notes
    -----
    The marker is the new one or, for a run of the old code, the old one
    (:func:`read_in_flight`).

    - ``'none'``: no marker.
    - ``'recovered'``: the marker's path (``f'{folder}/{path_name}{ext}'``) is
      registered without a sidecar (a crash between ``register_path`` and the
      sidecar write): the sidecar is written (same attach rule,
      ``recovered_after_restart=True``) and the marker cleared. Its
      ``lambda_registered`` comes from the network loaded at the restart,
      which is newer than the one a normal registration would have used if
      the trainer published a network in between.
    - ``'cleared'``: the marker's path is registered with a sidecar (a crash
      before the marker was cleared), or the segments are not both
      initialized (nothing, or one half, e.g. a crash between the two grompp
      calls): the marker is cleared and the worker selects afresh.
    - ``'kept'``: ``params.check_if_initialized(f'{folder}/back',
      f'{folder}/forw')`` (GROMACS: both ``.tpr``; toy: both segments): the
      shot resumes and keeps its record.
    """
    record = read_in_flight(folder)
    if record is None:
        return 'none'
    path_name = record.get('path_name')
    ext = params.trajectory_extension
    registered = f'{folder}/{path_name}{ext}' if path_name else None
    if registered and os.path.isfile(registered):
        if read_shot_sidecar(registered) is None:
            path = _chain_path(chain, registered)
            states = params.states
            _shot_sidecar(path, record, params=params, states=states, target_state=states[1],
                          recovered_after_restart=True)
            clear_in_flight(folder)
            _log(f'C-GAIMMD: recovered the sidecar of {registered} from the in-flight record '
                 f'(registered before a restart)')
            return 'recovered'
        clear_in_flight(folder)
        _log(f'C-GAIMMD: cleared the in-flight record of {path_name} in {folder}: the path is '
             f'already registered with its sidecar')
        return 'cleared'
    if params.check_if_initialized(f'{folder}/back', f'{folder}/forw'):
        _log(f'C-GAIMMD: resuming the in-flight shot {path_name} in {folder}')
        return 'kept'
    clear_in_flight(folder)
    _log(f'C-GAIMMD: cleared the in-flight record of {path_name} in {folder}: its segments are '
         f'not initialized; selecting a new shooting point')
    return 'cleared'


# ════════════════════════════════════════════════════════════════════════════
# Provenance
# ════════════════════════════════════════════════════════════════════════════

def _package_dir():
    """The directory of the imported ``aimmd`` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(directory, *args):
    """``(stdout, None)`` of ``git <args>`` run in `directory`, or ``(None, why it failed)``."""
    try:
        result = subprocess.run(['git', *args], cwd=directory, capture_output=True, text=True,
                                timeout=10)
    except FileNotFoundError:
        return None, 'the git executable was not found on PATH'
    except subprocess.TimeoutExpired:
        return None, f'git {" ".join(args)} took longer than 10 s'
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'git could not be run ({type(exc).__name__}: {exc})'
    if result.returncode != 0:
        message = (result.stderr or result.stdout or '').strip().splitlines()
        return None, (f'git {" ".join(args)} failed: '
                      f'{message[0] if message else f"exit code {result.returncode}"}')
    return result.stdout, None


def _is_commit(text):
    return len(text) in (40, 64) and set(text) <= set('0123456789abcdef')


def _git_dir(root):
    """The git directory of a checkout `root` (``.git`` folder, or the target of a ``.git`` file)."""
    dot_git = os.path.join(root, '.git')
    if os.path.isdir(dot_git):
        return dot_git
    if os.path.isfile(dot_git):                             # a worktree or submodule
        with open(dot_git) as handle:
            line = handle.read().strip()
        if line.startswith('gitdir:'):
            target = line[len('gitdir:'):].strip()
            return os.path.normpath(os.path.join(root, target))
    return None


def _read_ref(git_dir, ref):
    """The commit of `ref` in `git_dir` (loose ref, then packed-refs, then the common dir)."""
    candidates = [git_dir]
    common = os.path.join(git_dir, 'commondir')
    if os.path.isfile(common):
        with open(common) as handle:
            candidates.append(os.path.normpath(os.path.join(git_dir, handle.read().strip())))
    for directory in candidates:
        loose = os.path.join(directory, ref)
        if os.path.isfile(loose):
            with open(loose) as handle:
                return handle.read().strip()
        packed = os.path.join(directory, 'packed-refs')
        if os.path.isfile(packed):
            with open(packed) as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
    return None


def _commit_from_files(package):
    """``(commit, None)`` read from the ``.git`` files of the checkout that holds `package`.

    Used when the git executable cannot answer (not on PATH, or a repository
    git refuses, e.g. one owned by another user on a cluster file system). It
    only looks in the folder that contains the package folder (a clone of
    AIMMD-Lab), never further up, so that an install inside another
    project's work tree is not given that project's commit.
    """
    root = os.path.dirname(package)
    try:
        git_dir = _git_dir(root)
        if git_dir is None:
            return None, f'{root} has no .git (the package is not a git checkout)'
        with open(os.path.join(git_dir, 'HEAD')) as handle:
            head = handle.read().strip()
        commit = _read_ref(git_dir, head[len('ref:'):].strip()) if head.startswith('ref:') else head
    except OSError as exc:
        return None, f'reading the .git files of {root} failed ({type(exc).__name__}: {exc})'
    if not commit or not _is_commit(commit):
        return None, f'the .git files of {root} name no commit for HEAD {head!r}'
    return commit, None


DISTRIBUTION = 'aimmd-lab'
"""The distribution name of AIMMD-Lab (for the install record of a non-editable install)."""


def _commit_from_install(package):
    """``(commit, None)`` from pip's install record of the imported package, or ``(None, why)``.

    A non-editable install (the package copied into ``site-packages`` or a
    ``--target`` folder) has no ``.git``. When pip installed it from git
    (``pip install git+...@<ref>``), the distribution's ``direct_url.json``
    holds the commit it checked out (``vcs_info.commit_id``). Only the
    distribution whose ``aimmd/__init__.py`` is the imported one is read, so
    that another installed copy is never credited.
    """
    def located(distribution):
        try:
            return os.path.realpath(str(distribution.locate_file('aimmd/__init__.py')))
        except Exception:                                   # noqa: BLE001
            return None

    try:
        candidates = list(importlib.metadata.distributions(name=DISTRIBUTION))
    except Exception as exc:                                # noqa: BLE001
        return None, f'the {DISTRIBUTION} distributions could not be read ({type(exc).__name__})'
    if not candidates:
        return None, f'no {DISTRIBUTION} distribution is installed'
    imported = os.path.realpath(os.path.join(package, '__init__.py'))
    distribution = next((d for d in candidates if located(d) == imported), None)
    if distribution is None:
        return None, (f'no installed {DISTRIBUTION} distribution holds the imported package '
                      f'(an editable install, or a copy of the source)')
    try:
        text = distribution.read_text('direct_url.json')
        record = json.loads(text) if text else None
    except (OSError, ValueError) as exc:
        return None, f"its direct_url.json could not be read ({type(exc).__name__}: {exc})"
    if not isinstance(record, dict):
        return None, (f'{DISTRIBUTION} {distribution.version} was installed without a pip '
                      f'install record (direct_url.json), e.g. from an index or a wheel file')
    url = record.get('url')
    commit = (record.get('vcs_info') or {}).get('commit_id')
    if isinstance(commit, str) and _is_commit(commit):
        return commit, None
    if 'dir_info' in record:
        return None, (f'{DISTRIBUTION} was installed from the folder {url} (not from git): pip '
                      f'records no commit for such an install')
    if 'archive_info' in record:
        return None, f'{DISTRIBUTION} was installed from the archive {url}, which names no commit'
    return None, f'the install record of {DISTRIBUTION} ({url}) names no commit'


@functools.lru_cache(maxsize=1)
def git_provenance():
    """The git commit of the imported ``aimmd`` package, or why there is none.

    Returns
    -------
    dict
        ``{'git_commit', 'git_dirty', 'git_reason', 'git_source'}``: the
        commit (None if unknown); whether the package directory has
        uncommitted changes (None if unknown); None or the reason why the
        commit or the dirty flag is unknown; ``'git'`` (the executable
        answered), ``'git files'`` (read from the ``.git`` files),
        ``'direct_url'`` (pip's install record), ``'build record'`` (the
        record ``setup.py`` writes into a built package) or None.

    Notes
    -----
    1. ``git ls-files --error-unmatch __init__.py`` in the package directory
       says whether a git repository tracks the package; then ``git
       rev-parse HEAD`` gives the commit and ``git status --porcelain
       --untracked-files=all -- . ':(exclude)*.pyc'`` the dirty flag
       (changed, staged and untracked files of the package directory; files
       elsewhere in the repository do not count, and neither does the
       bytecode Python writes into ``__pycache__/``, which AIMMD-Lab does not
       ignore in a committed ``.gitignore``).
    2. When git does not answer (no executable on PATH, a repository it
       refuses, a time-out), the commit is read from the ``.git`` files of
       the folder that holds the package (a clone of AIMMD-Lab); the dirty
       flag is then unknown and ``git_reason`` says why. This is what the
       sidecars of the old code lacked: they recorded ``git_commit: null``
       with no reason (seen on JUPITER).
    3. A non-editable install (the package copied into ``site-packages``) is
       not tracked by any repository. Its commit is then read from pip's
       install record (``direct_url.json``), which names one for an install
       from git (``pip install git+...``; the dirty flag is unknown); an
       install from a folder, an archive or an index names none.
    4. Then the commit is read from the build record that ``setup.py``
       writes into every built package (``aimmd/_build_record.json``: the
       commit of the checkout it was built from and its dirty flag), which
       covers wheels and installs from a folder. Without one the commit is
       None and the reason says why. :func:`aimmd_provenance` then still
       identifies the code by ``source_sha256``.

    Looked up once per process (``functools.lru_cache``).
    """
    package = _package_dir()
    out = {'git_commit': None, 'git_dirty': None, 'git_reason': None, 'git_source': None}
    tracked, why = _git(package, 'ls-files', '--error-unmatch', '__init__.py')
    if tracked is None and 'did not match' in why:
        # git answered: the package is not tracked (an install inside another
        # project's work tree); that project's commit is not ours
        why = (f'the package directory {package} is not tracked by a git repository (an '
               f'installed copy, not a checkout)')
        return _from_install(out, package, why)
    if tracked is not None:
        head, why = _git(package, 'rev-parse', 'HEAD')
        commit = (head or '').strip()
        if head is not None and _is_commit(commit):
            out.update(git_commit=commit, git_source='git')
            status, why = _git(package, 'status', '--porcelain', '--untracked-files=all',
                               '--', '.', ':(exclude)*.pyc')
            if status is None:
                out['git_reason'] = f'the dirty flag is unknown: {why}'
            else:
                out['git_dirty'] = bool(status.strip())
            return out
        why = why or f'git rev-parse HEAD gave {commit!r}'
    commit, file_why = _commit_from_files(package)
    if commit is not None:
        out.update(git_commit=commit, git_source='git files',
                   git_reason=f'read from the .git files, the dirty flag is unknown, because {why}')
        return out
    return _from_install(out, package, f'{why}; {file_why}')


BUILD_RECORD = '_build_record.json'
"""The build record in the package folder of a built AIMMD-Lab (written by ``setup.py``)."""


def _commit_from_build_record(package):
    """``(record, None)`` of the build record of the imported package, or ``(None, why)``.

    ``setup.py`` writes ``aimmd/_build_record.json`` into every built package
    (a wheel, a non-editable install, a source distribution): the commit of
    the git checkout it was built from and whether the package folder had
    uncommitted changes. A checkout or an editable install has none.
    """
    fname = os.path.join(package, BUILD_RECORD)
    if not os.path.isfile(fname):
        return None, f'the package has no build record ({BUILD_RECORD})'
    try:
        with open(fname) as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f'its build record could not be read ({type(exc).__name__}: {exc})'
    if not isinstance(record, dict):
        return None, 'its build record is not a JSON object'
    commit = record.get('git_commit')
    if not (isinstance(commit, str) and _is_commit(commit)):
        return None, (f'its build record names no commit '
                      f'({record.get("git_reason") or "no reason given"})')
    return record, None


def _from_install(out, package, why):
    """`out` with the commit of pip's install record or of the build record, or with the reasons."""
    commit, install_why = _commit_from_install(package)
    if commit is not None:
        out.update(git_commit=commit, git_source='direct_url',
                   git_reason=(f"read from pip's install record (direct_url.json), the dirty "
                               f"flag is unknown, because {why}"))
        return out
    record, record_why = _commit_from_build_record(package)
    if record is not None:
        dirty = record.get('git_dirty')
        out.update(git_commit=record['git_commit'],
                   git_dirty=dirty if isinstance(dirty, bool) else None,
                   git_source='build record',
                   git_reason=(f'read from the build record ({BUILD_RECORD}, built '
                               f'{record.get("built_at")} from a '
                               f'{record.get("built_from") or "checkout"}), because {why}; '
                               f'{install_why}'))
        return out
    out['git_reason'] = f'{why}; {install_why}; {record_why}'
    return out


def aimmd_git_commit():
    """The git commit of the imported ``aimmd`` package (None if unknown; see :func:`git_provenance`)."""
    return git_provenance()['git_commit']


def aimmd_git_dirty():
    """Whether the package directory has uncommitted changes (None if unknown; see :func:`git_provenance`)."""
    return git_provenance()['git_dirty']


@functools.lru_cache(maxsize=1)
def source_sha256():
    """sha256 over the imported ``aimmd`` package's ``.py`` files (relative names and bytes, sorted).

    It identifies the code that ran even where there is no git commit (a copy
    of the source without ``.git``, a non-editable install). Computed once per
    process.
    """
    package = _package_dir()
    digest = hashlib.sha256()
    names = []
    for directory, folders, files in os.walk(package):
        folders[:] = sorted(name for name in folders if name != '__pycache__')
        names.extend(os.path.join(directory, name) for name in files if name.endswith('.py'))
    for name in sorted(names):
        digest.update(os.path.relpath(name, package).replace(os.sep, '/').encode())
        digest.update(b'\0')
        with open(name, 'rb') as handle:
            digest.update(handle.read())
        digest.update(b'\0')
    return digest.hexdigest()


def aimmd_provenance():
    """The sidecar's ``aimmd`` block: version, git commit (or why not) and source digest.

    Returns
    -------
    dict
        ``{'version', 'git_commit', 'git_dirty', 'git_reason', 'git_source', 'source_sha256'}``
        (:func:`git_provenance`, :func:`source_sha256`).
    """
    return {'version': _package_version(), **git_provenance(), 'source_sha256': source_sha256()}


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:                                       # noqa: BLE001
        return None


def software_versions():
    """The sidecar's ``software`` block: Python and the versions of :data:`SOFTWARE_DISTRIBUTIONS`.

    Returns
    -------
    dict
        ``{'python': '3.13.x', 'numpy': ..., 'torch': ..., 'mdanalysis': ..., 'bioemu2': ...,
        'cuequivariance': ..., 'cuequivariance_torch': ...}``; None for a distribution that is
        not installed. Read from the installed distributions' metadata: nothing is imported.
    """
    out = {'python': platform.python_version()}
    for key, distribution in SOFTWARE_DISTRIBUTIONS.items():
        out[key] = _distribution_version(distribution)
    return out


PREPARATION_HOOKS = ('cgaimmd_validity_function', 'cgaimmd_prepare_function',
                     'cgaimmd_minimize_function')
"""The params hooks of candidate preparation: :func:`bias_settings` records whether their MD is
known to run without the bias."""

_UNBIASED_PREPARATION = (('minimize', 'GromacsMinimizer'), ('solvate', 'GromacsSolvator'),
                         ('validity', 'TopologyValidity'))
"""Preparation classes of the package that run no MD under the bias: the GROMACS ones remove
``-plumed`` from their mdrun commands (``sanitize_mdrun``), TopologyValidity runs no MD."""


def _preparation_class(module, name):
    """A class of ``aimmd.cgaimmd.preparation.<module>``, None if that module is not imported
    (then no hook can be an instance of it, and nothing needs importing)."""
    return getattr(sys.modules.get(f'{__package__}.preparation.{module}'), name, None)


def _unbiased_preparation(hook):
    """Whether candidate preparation `hook` is known to run no MD under the PLUMED bias.

    True for None (no hook: :func:`~aimmd.cgaimmd.source.place_in_box`, or no
    minimization), for an object of :data:`_UNBIASED_PREPARATION` or one of
    its bound methods, for a ``PolishedValidity`` whose validity and
    minimizer are such objects, and for the params-file wrapper whose code
    names nothing but one such object (``def cgaimmd_minimize_function(frame,
    context): return MINIMIZER(frame, context)``). False for anything else:
    a hook of the user's own may run mdrun itself, with ``-plumed`` or not.
    """
    if hook is None:
        return True
    if isinstance(hook, types.MethodType):
        hook = hook.__self__
    classes = tuple(cls for cls in (_preparation_class(*entry) for entry in _UNBIASED_PREPARATION)
                    if cls is not None)
    if classes and isinstance(hook, classes):
        return True
    polished = _preparation_class('validity', 'PolishedValidity')
    if polished is not None and isinstance(hook, polished):
        parts = (getattr(hook, 'validity', None), getattr(hook, 'minimizer', None))
        return all(part is not None and _unbiased_preparation(part) for part in parts)
    code = getattr(hook, '__code__', None)
    names = getattr(hook, '__globals__', None)
    if (code is None or names is None or getattr(hook, '__closure__', None)
            or len(code.co_names) != 1
            or any(isinstance(constant, types.CodeType) for constant in code.co_consts)):
        return False
    target = names.get(code.co_names[0])
    return (target is not None and getattr(target, '__code__', None) is None
            and _unbiased_preparation(target))


def _qualified_name(function):
    """``module.qualname`` of a function, or of an object's class."""
    if function is None:
        return None
    return (f"{getattr(function, '__module__', None) or '?'}."
            f"{getattr(function, '__qualname__', None) or type(function).__name__}")


def bias_settings(params):
    """The sidecar's ``bias`` block: the PLUMED bias of the run, None without one.

    Parameters
    ----------
    params : aimmd.Params
        The run's params.

    Returns
    -------
    dict or None
        None when the run records no bias (``record_bias`` False) and its
        ``gmx_mdrun`` has no ``-plumed``. Otherwise ``record_bias``,
        ``bias_source``, ``bias_function`` (its qualified name),
        ``bias_reactive_threshold``, ``plumed_file`` (the ``-plumed`` file of
        ``gmx_mdrun``, absolute; a relative one is resolved against the
        params folder), ``plumed_file_relative`` (True when ``gmx_mdrun``
        names it relative: mdrun then looks for it in each chain folder),
        ``plumed_sha256`` (of that file, None if it cannot be read),
        ``plumed_inputs`` (the files the PLUMED input reads, e.g. a frozen
        OPES state's ``STATE_RFILE``: ``keyword``, ``path``, ``sha256``;
        :func:`aimmd.cgaimmd.plumed.plumed_input_files`),
        ``preparation_unbiased`` and ``preparation_unverified_hooks``.

    Notes
    -----
    ``preparation_unbiased`` is True when every preparation hook of
    :data:`PREPARATION_HOOKS` is known to run its MD without the bias: none
    is set, or each is the package's own preparation (``GromacsMinimizer``
    and ``GromacsSolvator`` remove ``-plumed`` from their mdrun commands,
    :func:`aimmd.cgaimmd.preparation.gromacs.sanitize_mdrun`;
    ``TopologyValidity`` runs no MD; see :func:`_unbiased_preparation`). It is
    None (unknown) when a hook is the user's own code, which may run mdrun
    itself; ``preparation_unverified_hooks`` then maps those hooks' params
    names to their qualified names (empty when the flag is True).
    """
    from .plumed import plumed_setup
    record_bias = bool(getattr(params, 'record_bias', False))
    setup = plumed_setup(params)
    if not record_bias and setup is None:
        return None
    name = _qualified_name(getattr(params, 'bias_function', None))
    unverified = {field: _qualified_name(getattr(params, field, None))
                  for field in PREPARATION_HOOKS
                  if not _unbiased_preparation(getattr(params, field, None))}
    threshold = getattr(params, 'bias_reactive_threshold', None)
    inputs = None
    if setup is not None:
        inputs = [{'keyword': entry['keyword'], 'path': entry['path'] or entry['written'],
                   'sha256': entry['sha256']} for entry in setup['inputs']]
    return {'record_bias': record_bias, 'bias_source': getattr(params, 'bias_source', None),
            'bias_function': name,
            'bias_reactive_threshold': (threshold if threshold is None
                                        or isinstance(threshold, (int, float))
                                        else list(threshold)),
            'plumed_file': None if setup is None else setup['path'],
            'plumed_file_relative': None if setup is None else setup['relative'],
            'plumed_sha256': None if setup is None else setup['sha256'],
            'plumed_inputs': inputs,
            'preparation_unbiased': True if not unverified else None,
            'preparation_unverified_hooks': unverified}


# ════════════════════════════════════════════════════════════════════════════
# The per-shot table of a run (read only)
# ════════════════════════════════════════════════════════════════════════════

SHOT_TABLE_COLUMNS = (
    'shot_index', 'chain', 'path', 'registered', 'n_frames', 'first_state', 'last_state',
    'reactive', 'shooting_index', 'source', 'lambda_target', 'lambda_raw', 'lambda_final',
    'lambda_registered', 'state_final', 'acceptance_rule', 'acceptance_enforced',
    'network_round', 'network_trained_shots', 'network_sha256', 'generator_method',
    'candidates_tried', 'candidates_rejected', 'stage1_rejected', 'seconds_generate',
    'seconds_validity', 'seconds_raw_lambda', 'seconds_prepare', 'seconds_minimize',
    'seconds_final_check', 'seconds_total')
"""The keys of every row of :func:`shot_table` (None where a shot has no value)."""


def _read_states(fname):
    """The cached state labels of a path file, read without locks (None if there are none)."""
    from ..path.utils import get_cache_fname
    try:
        labels = np.load(get_cache_fname(fname, 'states'), allow_pickle=False)
    except (OSError, ValueError):
        return None
    return [str(label) for label in np.asarray(labels).reshape(-1)]


def _mtime_iso(fname):
    stamp = os.path.getmtime(fname)
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).isoformat(
        timespec='seconds')


def _cv_specs(cv):
    """``{name: cache name or callable}`` from None, a cache name, a callable or a dict."""
    if cv is None:
        return {}
    if isinstance(cv, dict):
        specs = dict(cv)
    elif isinstance(cv, str):
        specs = {cv: cv}
    elif callable(cv):
        specs = {getattr(cv, '__name__', 'cv'): cv}
    else:
        raise TypeError(f'cv must be None, a cache name, a callable or a dict of them, got {cv!r}')
    clash = sorted(set(specs) & set(SHOT_TABLE_COLUMNS))
    if clash:
        raise ValueError(f'collective-variable names {clash} are column names of the shot '
                         f'table; name them differently')
    return specs


def _cache_row(fname, name, index):
    """Row `index` of the per-frame cache `name` of `fname`, read without a lock (None if missing)."""
    from ..path.utils import get_cache_fname
    try:
        data = np.load(get_cache_fname(fname, name), mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError):
        return None
    if index >= len(data):
        return None
    return np.asarray(data[index], dtype=float).reshape(1, -1)


def _cv_value(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 1:
        return float(values[0])
    return [float(v) for v in values]


def _shooting_frame_cvs(fname, index, specs, cv_source):
    """The collective variables of frame `index` of `fname` (``{name: value}``)."""
    from .analysis._trajectory import frames_reader, read_frames
    out = {}
    source = None
    for name, spec in specs.items():
        if isinstance(spec, str):
            row = _cache_row(fname, spec, index)
            out[name] = None if row is None else _cv_value(row)
            continue
        if source is None:
            if cv_source == 'reader':
                positions, dimensions, _ = read_frames(fname, [index])
                source = frames_reader(positions, None if dimensions is None else dimensions[0])
            else:
                source = _cache_row(fname, cv_source, index)
                if source is None:
                    source = False
        out[name] = None if source is False else _cv_value(np.asarray(spec(source), dtype=float))
    return out


def shot_table(run_dir, *, states='ARB', trajectory_extension='.xtc', params=None, cv=None,
               cv_source='reader', round_shooting_index=False):
    """One row per registered shot of the reactive chains of a run folder.

    Parameters
    ----------
    run_dir : str or os.PathLike
        The run folder (its ``chain{states[1]}*`` folders are read).
    states : str, optional
        The run's states (``params.states``); ``states[1]`` names the chains,
        ``states[0]`` and ``states[2]`` are the end states.
    trajectory_extension : str, optional
        ``params.trajectory_extension``.
    params : aimmd.Params, optional
        When given, its ``states``, ``trajectory_extension`` and
        ``round_shooting_index`` are used.
    cv : None, str, callable or dict, optional
        Collective variables of each shooting point, added as columns (the
        run progression of a paper figure: which configuration was generated
        at which requested committor, and where it went):

        - a string names a per-frame cache of the run (``'values'``,
          ``'descriptors'``, a named descriptor series, ``'bias'``): its row
          at the shooting frame;
        - a callable gets the shooting frame as `cv_source` gives it and
          returns its value (or values);
        - a dict ``{name: str or callable}`` gives several columns.

        A column holds a float, or a list for several values; None where the
        frame has no value (a missing cache row). A callable written for
        :func:`aimmd.analysis.reweighting.frame_table` works here unchanged.
    cv_source : str, optional
        What the callables get: ``'reader'`` (default: a reader over the one
        shooting frame, the reader type AIMMD hands to states and descriptors
        functions) or the name of a per-frame cache (its row at the shooting
        frame, ``(1, D)``; faster, no trajectory is opened).
    round_shooting_index : bool, optional
        Without `params`: how the shooting frame of a shot without a record
        is found from the times (``Params.round_shooting_index``).

    Returns
    -------
    list of dict
        Rows with the keys of :data:`SHOT_TABLE_COLUMNS` (and one per
        collective variable), sorted by registration time (then chain and
        path), ``shot_index`` counting from 0 in that order.
        ``first_state``/``last_state``/``n_frames`` come from the path's
        states cache (else the sidecar's registration record): a registered
        path holds the backward half reversed, then the forward half, so
        ``first_state`` is where the backward half ended and ``last_state``
        where the forward half ended; ``reactive`` is True when the path runs
        from one end state to the other; ``registered`` is the sidecar's time,
        else the path file's modification time (UTC); ``shooting_index`` is
        the shooting frame in the path file (from the record; for a shot
        without one only with `cv`, found from the times as AIMMD finds it).
        The sidecar columns (new and old sidecars; :func:`read_shot_sidecar`)
        are None for a shot without one (an RFPS shot of a continued run);
        ``candidates_rejected`` and ``stage1_rejected`` are dicts
        ``{reason: count}``.

    Notes
    -----
    Nothing is written: trajectories are opened only for collective
    variables on the reader, with MDAnalysis's low-level XTC/TRR file objects
    (its readers would write an offsets file next to the trajectory); state
    caches and other caches are read with ``numpy.load`` (AIMMD's cache
    readers take file locks, which older filelock versions leave behind as
    files, as in the archived runs); the caller decides where to save the
    table. This is what replaces the per-campaign summary scripts, which
    wrote their CSV into the run folder.
    """
    if params is not None:
        states = params.states
        trajectory_extension = params.trajectory_extension
        round_shooting_index = bool(getattr(params, 'round_shooting_index', False))
    specs = _cv_specs(cv)
    if specs and not isinstance(cv_source, str):
        raise TypeError(f"cv_source must be 'reader' or a cache name, got {cv_source!r}")
    run_dir = os.fspath(run_dir)
    ends = (states[0], states[2])
    rows = []
    for folder in sorted(glob.glob(os.path.join(glob.escape(run_dir), f'chain{states[1]}*'))):
        if not os.path.isdir(folder):
            continue
        pattern = os.path.join(glob.escape(folder), f'path??????{trajectory_extension}')
        for fname in sorted(glob.glob(pattern)):
            record = read_shot_sidecar(fname) or {}
            registration = record.get('registration') or {}
            labels = _read_states(fname)
            first = labels[0] if labels else None
            last = labels[-1] if labels else None
            n_frames = len(labels) if labels else registration.get('n_frames')
            acceptance = record.get('acceptance') or {}
            network = record.get('network') or {}
            generator = record.get('generator') or {}
            candidates = record.get('candidates') or {}
            cost = record.get('cost_seconds') or {}
            row = dict.fromkeys(SHOT_TABLE_COLUMNS)
            row.update(
                chain=os.path.basename(folder), path=os.path.basename(fname),
                registered=record.get('registered') or _mtime_iso(fname), n_frames=n_frames,
                first_state=first, last_state=last,
                reactive=bool(first in ends and last in ends and first != last),
                shooting_index=registration.get('shooting_index'),
                source=record.get('source'), lambda_target=record.get('lambda_target'),
                lambda_raw=record.get('lambda_raw'), lambda_final=record.get('lambda_final'),
                lambda_registered=record.get('lambda_registered'),
                state_final=record.get('state_final'),
                acceptance_rule=acceptance.get('rule'),
                acceptance_enforced=acceptance.get('enforced'),
                network_round=network.get('round'),
                network_trained_shots=network.get('n_trained_shots'),
                network_sha256=network.get('sha256'), generator_method=generator.get('method'),
                candidates_tried=candidates.get('tried'),
                candidates_rejected=dict(candidates.get('rejected') or {}) if record else None,
                stage1_rejected=dict(record.get('stage1_rejected') or {}) if record else None,
                **{f'seconds_{key}': cost.get(key) for key in (
                    'generate', 'validity', 'raw_lambda', 'prepare', 'minimize', 'final_check',
                    'total')})
            if specs:
                if row['shooting_index'] is None:
                    from .analysis._trajectory import (count_frames, read_frames,
                                                       shooting_index_from_times)
                    n = count_frames(fname)
                    _, _, times = read_frames(fname, [0, 1] if n > 1 else [0])
                    row['shooting_index'] = shooting_index_from_times(
                        times, n, nearest=round_shooting_index)
                row.update(_shooting_frame_cvs(fname, int(row['shooting_index']), specs,
                                               cv_source))
            rows.append(row)
    rows.sort(key=lambda row: (row['registered'], row['chain'], row['path']))
    for index, row in enumerate(rows):
        row['shot_index'] = index
    return rows


def summarise_shots(rows, *, frame_interval_ps=None, window=100):
    """Counts and progress of a :func:`shot_table`.

    Parameters
    ----------
    rows : list of dict
        :func:`shot_table` rows (in ``shot_index`` order).
    frame_interval_ps : float, optional
        Time between saved frames; when given, MD time in shots is reported in ns.
    window : int, optional
        Number of consecutive shots per window of the reactive share.

    Returns
    -------
    dict
        ``n_shots``, ``n_reactive``, ``reactive_share`` (None without shots),
        ``per_chain`` (``{chain: {'shots', 'reactive'}}``), ``path_types``
        (``{first + last state: count}``), ``sources`` (``{source: count}``,
        ``'none'`` for shots without a sidecar), ``acceptance_rules``,
        ``candidates_tried`` (sum), ``candidates_rejected`` and
        ``stage1_rejected`` (summed ``{reason: count}``), ``md_frames_in_shots``,
        ``md_ns_in_shots`` (None without `frame_interval_ps`),
        ``cumulative_reactive`` (reactive shots after each shot) and
        ``reactive_share_in_windows`` (``[(first shot, last shot + 1, share)]``
        of consecutive windows of `window` shots).
    """
    window = int(window)
    if window < 1:
        raise ValueError(f'window must be an int >= 1, got {window!r}')

    def count(mapping, key, n=1):
        mapping[key] = mapping.get(key, 0) + n

    per_chain, path_types, sources, rules = {}, {}, {}, {}
    rejected, stage1 = {}, {}
    tried = frames = 0
    cumulative = []
    reactive_total = 0
    for row in rows:
        chain = per_chain.setdefault(row['chain'], {'shots': 0, 'reactive': 0})
        chain['shots'] += 1
        chain['reactive'] += int(bool(row['reactive']))
        reactive_total += int(bool(row['reactive']))
        cumulative.append(reactive_total)
        if row['first_state'] is not None and row['last_state'] is not None:
            count(path_types, f"{row['first_state']}{row['last_state']}")
        count(sources, row['source'] if row['source'] is not None else 'none')
        if row['acceptance_rule'] is not None:
            count(rules, row['acceptance_rule'])
        tried += int(row['candidates_tried'] or 0)
        for reason, n in (row['candidates_rejected'] or {}).items():
            count(rejected, reason, int(n))
        for reason, n in (row['stage1_rejected'] or {}).items():
            count(stage1, reason, int(n))
        frames += int(row['n_frames'] or 0)
    windows = []
    for start in range(0, len(rows), window):
        part = rows[start:start + window]
        windows.append((start, start + len(part),
                        sum(bool(row['reactive']) for row in part) / len(part)))
    return {
        'n_shots': len(rows), 'n_reactive': reactive_total,
        'reactive_share': reactive_total / len(rows) if rows else None,
        'per_chain': per_chain, 'path_types': dict(sorted(path_types.items())),
        'sources': dict(sorted(sources.items())), 'acceptance_rules': dict(sorted(rules.items())),
        'candidates_tried': tried, 'candidates_rejected': dict(sorted(rejected.items())),
        'stage1_rejected': dict(sorted(stage1.items())), 'md_frames_in_shots': frames,
        'md_ns_in_shots': (None if frame_interval_ps is None
                           else frames * float(frame_interval_ps) / 1000.0),
        'cumulative_reactive': cumulative, 'reactive_share_in_windows': windows,
    }
