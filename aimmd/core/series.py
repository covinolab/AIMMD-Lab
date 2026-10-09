"""
aimmd.core.series
=================

Is every trajectory of a run covered by the params' descriptor series?

A params file names the per-trajectory series its descriptors are cached in,
``{trajectory}.{descriptors_series}.npy``. A name computed from the
descriptor settings (a node-table featurizer's ``'descriptors-gn...'``)
changes with them: after such a change no trajectory of a run has a file of
the new series, and AIMMD would featurize every frame again, lazily (the
trainer serially at the start of a round, the workers on every path they
touch). This module finds the gap when a job is created and when a worker
starts, before any MD or training:

- `find_trajectories` lists the trajectories of runs: those with a states
  series ``{trajectory}.states.npy``;
- `series_coverage` tells which of them have no file of the series at all,
  leaving out the initial paths in the ``initial*`` folders (exported and
  featurized again at every launch). A file with zero or missing (short) rows
  counts as there: the compute ledger fills such rows as usual;
- `register_series` records, per series name, whether a job may refill a gap
  (``refill``) and the callable that refills it. The code that names a
  series registers it when it is constructed (the node-table featurizers of
  `aimmd.network.nodetables`), in every process that executes the params
  file. A series that nobody registers is never refilled;
- `check_series_coverage` (the launcher, before it writes or starts a job)
  raises `SeriesCoverageError` for a gap that may not be refilled, and
  announces the refill otherwise;
- `ensure_series_coverage` (every worker, before its task) raises for a gap
  that may not be refilled. Otherwise exactly one process of the run refills
  it, under the lock file `REFILL_LOCK` in the run folder, while every other
  process of the run waits for that lock and starts once the series is
  complete. A waiter that finds the series still incomplete after the
  refiller let go of the lock raises instead of refilling again.

Every line the refill writes to the logs starts with ``'SERIES REFILL:'``.
The default series ``'descriptors'`` is never checked. This module needs
neither torch nor a graph library.
"""

# external
import functools
import json
import os
import socket
import time
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout


#: The historical descriptor series, which is never checked.
DEFAULT_SERIES = 'descriptors'

#: Suffix of the series that makes a trajectory part of a run.
STATES_SUFFIX = '.states.npy'

#: Prefix of the folders of the exported initial paths (``initial{states}``)
#: directly below a run, or below a system folder of a multi-system run.
SEED_PREFIX = 'initial'

#: Lock file in the run folder, held by the process that refills a series.
REFILL_LOCK = '.series-refill.lock'

#: File in the run folder naming the process that refills (for the waiters).
REFILL_INFO = '.series-refill.json'

#: Prefix of every log line of the refill.
LOG_PREFIX = 'SERIES REFILL:'

#: Seconds a waiter blocks on the lock before it checks for a stop request.
WAIT_SECONDS = 5.

#: Seconds between the 'still waiting' lines of a waiter.
WAIT_REPORT_SECONDS = 300.

#: Seconds between the progress lines of the refilling process.
PROGRESS_SECONDS = 60.

# seconds between two attempts to take the lock (it lives on a shared
# filesystem, so do not poll it many times a second)
_POLL_SECONDS = 1.

_print = functools.partial(print, flush=True)


class SeriesCoverageError(RuntimeError):
    """Trajectories of a run have no file of the params' descriptor series,
    and the series may not be (or could not be) refilled."""


# ----------------------------------------------------------------------
# what to do about a gap, per series name

SeriesPolicy = namedtuple('SeriesPolicy', 'series refill refiller')
SeriesPolicy.__doc__ = """What a job does when trajectories lack a series.

Attributes
----------
series : str
    The series name.
refill : bool
    Whether a job refills the series before it starts (True) or stops with
    `SeriesCoverageError` (False).
refiller : callable or None
    ``refiller(coverage, params_file=..., workdir=..., jobs=..., log=...,
    progress=...)`` writes ``{trajectory}.{series}.npy`` for the trajectories
    of ``coverage.missing`` (a `SeriesCoverage`) with ``jobs`` processes,
    reports through ``log`` (one line per call) and calls ``progress(done,
    frames)`` with the trajectories and frames done so far; it returns a dict
    whose ``'route'`` says how it refilled. Optional attributes describe it in
    messages: ``route(coverage, workdir)`` (how it would refill), ``label``
    (what the settings are called, e.g. ``'node-table'``), ``legacy_note``
    (why only ``*.descriptors.npy`` is there) and ``prefill_command(
    params_file, run)`` (the command that fills the series offline).
"""

_POLICIES = {}


def register_series(series, refill=False, refiller=None):
    """Record what a job does when trajectories of its run lack `series`.

    Called by the code that names the series when it is constructed (e.g. a
    `aimmd.network.nodetables.NodeTableFeaturizer`), so in every process that
    executes the params file. The last registration of a name counts.

    Parameters
    ----------
    series : str
        The series name (``params.descriptors_series``).
    refill : bool, default=False
        Whether a job refills the series first (see `SeriesPolicy`).
    refiller : callable, optional
        The callable that refills it (see `SeriesPolicy`).

    Returns
    -------
    SeriesPolicy
    """
    policy = SeriesPolicy(str(series), bool(refill), refiller)
    _POLICIES[policy.series] = policy
    return policy


def series_policy(series):
    """The registered `SeriesPolicy` of `series`; a series that nobody
    registered is never refilled (``refill=False``, no refiller)."""
    return _POLICIES.get(series) or SeriesPolicy(series, False, None)


# ----------------------------------------------------------------------
# the trajectories of a run

def _walk(run, system_ids=None, seeds=True):
    """``(folder, system_id, files)`` of the folders of a run that hold its
    trajectories, in sorted order. With `system_ids` (a multi-system run)
    only ``{run}/{system_id}/...``; without `seeds`, the ``initial*``
    folders directly below the run (or a system folder) are skipped."""
    run = os.path.abspath(run)
    for folder, folders, files in os.walk(run):
        parts = Path(os.path.relpath(folder, run)).parts
        system_id = None
        if system_ids is not None:
            if not parts:                               # the run folder
                folders[:] = sorted(name for name in folders
                                    if name in system_ids)
                continue
            system_id, parts = parts[0], parts[1:]
        if not seeds and not parts:
            folders[:] = [name for name in folders
                          if not name.startswith(SEED_PREFIX)]
        folders.sort()
        yield folder, system_id, files


def _system_ids(system_ids):
    return None if system_ids is None else [str(name) for name in system_ids]


def find_trajectories(runs, system_ids=None, seeds=True):
    """The trajectories of AIMMD runs: those with a states series.

    A trajectory belongs to a run when it has a states series,
    ``{trajectory}.states.npy``: the exported initial paths, the chain
    paths, the halves of shots in flight and the parts of free simulations.

    Parameters
    ----------
    runs : list of str
        Run folders.
    system_ids : list of str, optional
        The systems of a multi-system run: a trajectory belongs to the system
        named by its first folder below the run (``{run}/{system_id}/...``);
        others are left out.
    seeds : bool, default=True
        Include the exported initial paths (the ``initial*`` folders directly
        below the run, or below a system folder).

    Returns
    -------
    list of tuple
        ``(trajectory, system_id)``, absolute paths in sorted order per run;
        ``system_id`` is None without `system_ids`.
    """
    system_ids = _system_ids(system_ids)
    found = []
    for run in runs:
        in_run = []
        for folder, system_id, files in _walk(run, system_ids, seeds):
            for name in files:
                if name.startswith('.') or not name.endswith(STATES_SUFFIX):
                    continue
                trajectory = os.path.join(folder, name[:-len(STATES_SUFFIX)])
                if os.path.isfile(trajectory):
                    in_run.append((trajectory, system_id))
        found += sorted(in_run)
    return found


def _npy_length(fname):
    """Length (first dimension) of an npy file from its header; 0 if it
    cannot be read."""
    try:
        with open(fname, 'rb') as file:
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape = np.lib.format.read_array_header_1_0(file)[0]
            else:
                shape = np.lib.format.read_array_header_2_0(file)[0]
        return int(shape[0]) if shape else 0
    except (OSError, ValueError):
        return 0


def _descriptor_series_of(files):
    """``{file stem: [series]}`` of the descriptor series files among
    `files` (``{stem}.descriptors.npy`` or ``{stem}.descriptors-*.npy``)."""
    index = {}
    for name in files:
        if name.startswith('.') or not name.endswith('.npy'):
            continue
        cut = name.find(f'.{DEFAULT_SERIES}')
        if cut < 1:
            continue
        series = name[cut + 1:-len('.npy')]
        if series == DEFAULT_SERIES or series.startswith(
                f'{DEFAULT_SERIES}-'):
            index.setdefault(name[:cut], []).append(series)
    return index


class SeriesCoverage:
    """Which trajectories of a run have no file of a descriptor series.

    Built by `series_coverage`.

    Attributes
    ----------
    run : str
        The run folder (absolute).
    series : str
        The series checked.
    system_ids : list of str or None
        The systems of a multi-system run.
    trajectories : list of tuple
        ``(trajectory, system_id)`` of every trajectory of the run with a
        states series, the initial paths (``initial*`` folders) left out.
    missing : list of tuple
        Those of `trajectories` without ``{trajectory}.{series}.npy``.
    frames : dict
        ``{trajectory: frames}`` of the `missing` trajectories, the length of
        their states series.
    siblings : dict
        ``{trajectory: [series]}``: the other descriptor series files next to
        each `missing` trajectory (``'descriptors'`` for the legacy series).
    examples : dict
        ``{(series, system_id): file}``: one file of every other descriptor
        series per system, over all `trajectories` (e.g. for its row width).
    """

    def __init__(self, run, series, system_ids, trajectories, missing,
                 frames, siblings, examples):
        self.run = run
        self.series = series
        self.system_ids = system_ids
        self.trajectories = trajectories
        self.missing = missing
        self.frames = frames
        self.siblings = siblings
        self.examples = examples

    def __repr__(self):
        return (f'{type(self).__name__}(run={self.run!r}, '
                f'series={self.series!r}, missing={len(self.missing)} of '
                f'{len(self.trajectories)})')

    @property
    def complete(self):
        """bool: every trajectory has a file of the series."""
        return not self.missing

    @property
    def missing_frames(self):
        """int: frames of the trajectories without the series."""
        return sum(self.frames.values())

    @property
    def others(self):
        """dict: ``{series: trajectories}``, how many of the `missing`
        trajectories have a file of each other ``'descriptors-*'`` series."""
        counts = Counter(series for names in self.siblings.values()
                         for series in set(names) if series != DEFAULT_SERIES)
        return dict(sorted(counts.items(), key=lambda item: (-item[1],
                                                             item[0])))

    @property
    def legacy(self):
        """int: `missing` trajectories with a legacy ``*.descriptors.npy``."""
        return sum(DEFAULT_SERIES in names
                   for names in self.siblings.values())

    def summary(self):
        """'14 of 36 trajectories (412,733 frames) of run ...'"""
        return (f'{len(self.missing)} of {len(self.trajectories)} '
                f'trajectories ({self.missing_frames:,} frames) of run '
                f'{self.run!r}')


def series_coverage(run, series, system_ids=None):
    """Which trajectories of a run have no file of `series`.

    Parameters
    ----------
    run : str
        The run folder (in a multi-system run, the folder above the system
        folders). A folder that does not exist has no trajectories.
    series : str
        The descriptor series (``params.descriptors_series``).
    system_ids : list of str, optional
        The systems of a multi-system run (``{run}/{system_id}/...``).

    Returns
    -------
    SeriesCoverage
        Of the trajectories of the run (`find_trajectories`) without the
        initial paths. A trajectory is missing when it has no file
        ``{trajectory}.{series}.npy``; a file with zero or missing (short)
        rows is not.
    """
    run = os.path.abspath(run)
    system_ids = _system_ids(system_ids)
    trajectories, missing, frames, siblings, examples = [], [], {}, {}, {}
    for folder, system_id, files in _walk(run, system_ids, seeds=False):
        names = set(files)
        index = _descriptor_series_of(files)
        for name in sorted(files):
            if name.startswith('.') or not name.endswith(STATES_SUFFIX):
                continue
            stem = name[:-len(STATES_SUFFIX)]
            trajectory = os.path.join(folder, stem)
            if not os.path.isfile(trajectory):
                continue
            item = (trajectory, system_id)
            trajectories.append(item)
            others = [other for other in index.get(stem, [])
                      if other != series]
            for other in others:
                if other != DEFAULT_SERIES:
                    examples.setdefault(
                        (other, system_id),
                        os.path.join(folder, f'{stem}.{other}.npy'))
            if f'{stem}.{series}.npy' not in names:
                missing.append(item)
                frames[trajectory] = _npy_length(os.path.join(folder, name))
                siblings[trajectory] = sorted(others)
    return SeriesCoverage(run, series, system_ids, trajectories, missing,
                          frames, siblings, examples)


# ----------------------------------------------------------------------
# messages

def _plural(count, word, plural=None):
    return f'{count} {word if count == 1 else plural or word + "s"}'


def _trajectories(count):
    return _plural(count, 'trajectory', 'trajectories')


def missing_series_message(coverage, policy=None, params_file=None,
                           prefix='SERIES CHECK:'):
    """What is missing, what is there instead, and the remedies.

    Parameters
    ----------
    coverage : SeriesCoverage
        An incomplete coverage.
    policy : SeriesPolicy, optional
        The policy of the series (default: the registered one).
    params_file : str, optional
        The params file, for the prefill command.
    prefix : str
        Start of every line.

    Returns
    -------
    str
        One line per item, each starting with `prefix`.
    """
    policy = policy or series_policy(coverage.series)
    refiller = policy.refiller
    label = getattr(refiller, 'label', 'descriptor')
    lines = [f'{coverage.summary()} have no {coverage.series!r} series file '
             f'(the descriptors_series of the params file; the initial paths '
             f'are not counted).']
    others = coverage.others
    if others:
        names = ', '.join(f'{name!r} ({_trajectories(count)})'
                          for name, count in others.items())
        lines.append(f'Next to them is the series {names}: the {label} '
                     f'settings changed since these frames were featurized.')
    if coverage.legacy:
        note = getattr(refiller, 'legacy_note', None) or (
            f'this campaign has not been migrated from the series '
            f'{DEFAULT_SERIES!r}')
        lines.append(f"Next to {_trajectories(coverage.legacy)} of them is "
                     f"the series '*.{DEFAULT_SERIES}.npy': {note}.")
    if not others and not coverage.legacy:
        lines.append('No other descriptor series is next to them: their '
                     'frames were never featurized (or their series files '
                     'were deleted).')
    lines.append('Without the series, AIMMD featurizes all these frames '
                 'again, lazily: the trainer serially at the start of a '
                 'round and the workers on every path they touch, which can '
                 'take hours. Do one of:')
    restore = (f' (those of {", ".join(map(repr, others))})' if others
               else '')
    lines.append(f'  1. restore the {label} settings these frames were '
                 f'featurized with{restore}, if the change was not '
                 f'intended;')
    command = None
    if hasattr(refiller, 'prefill_command'):
        command = refiller.prefill_command(params_file, coverage.run)
    lines.append(f'  2. fill the series before the next job: {command};'
                 if command else
                 '  2. compute the series of these trajectories before the '
                 'next job;')
    if refiller is not None:
        lines.append('  3. construct the featurizer with refill=True: the '
                     'next job then refills the series first (one process '
                     'of the run refills, the others wait).')
    return '\n'.join(f'{prefix} {line}' for line in lines)


def _route(policy, coverage, workdir):
    """How the refiller of `policy` would refill, as text."""
    route = getattr(policy.refiller, 'route', None)
    if route is None:
        return 'its refill callable'
    try:
        return route(coverage, workdir)
    except Exception as error:                          # noqa: BLE001
        return f'a route it could not plan yet ({type(error).__name__}: ' \
               f'{error})'


# ----------------------------------------------------------------------
# checks before a job

def _checked_series(params):
    """The series of `params` to check, or None (the default series, or no
    descriptors at all)."""
    series = getattr(params, 'descriptors_series', None) or DEFAULT_SERIES
    if series == DEFAULT_SERIES:
        return None
    if not getattr(params, 'descriptors_function', None):
        return None
    return str(series)


def run_folder(params, directory):
    """The run folder of a worker or launcher directory and its systems.

    Parameters
    ----------
    params : aimmd.Params
        The params of the run.
    directory : str
        The run folder, or (multi-system run) one of its system folders
        ``{run}/{system_id}``.

    Returns
    -------
    tuple
        ``(run, system_ids)``: the absolute run folder, and the systems of a
        multi-system run (None otherwise).
    """
    directory = os.path.normpath(os.path.abspath(directory))
    system_ids = None
    if getattr(params, 'multi_system', False):
        system_ids = _system_ids(getattr(params, 'system_ids', None) or [])
        if os.path.basename(directory) in system_ids:
            directory = os.path.dirname(directory)
    return directory, system_ids


def _params_file(params):
    path = getattr(params, 'path', None)
    return None if path is None else str(path)


def _workdir(params):
    parent = getattr(params, 'parent', None)
    return os.path.abspath(str(parent) if parent is not None else '.')


def check_series_coverage(params, directory, log=None):
    """The launcher's check: refuse a job that would featurize again.

    Parameters
    ----------
    params : aimmd.Params
        The params of the run.
    directory : str
        The run folder.
    log : callable, optional
        Called with the notice (default: print).

    Returns
    -------
    SeriesCoverage or None
        None for a series that is not checked (the default series).

    Raises
    ------
    SeriesCoverageError
        If trajectories of the run have no file of the series and its
        policy does not refill it.
    """
    log = log or _print
    series = _checked_series(params)
    if series is None:
        return None
    run, system_ids = run_folder(params, directory)
    coverage = series_coverage(run, series, system_ids)
    if coverage.complete:
        return coverage
    policy = series_policy(series)
    if not (policy.refill and policy.refiller is not None):
        raise SeriesCoverageError(
            missing_series_message(coverage, policy, _params_file(params)))
    log(f'{LOG_PREFIX} {coverage.summary()} have no {series!r} series '
        f'file. The featurizer has refill=True: the job first refills them '
        f'in one process, by {_route(policy, coverage, _workdir(params))}, '
        f'while the other processes of the run wait; the files of other '
        f'series stay in place.')
    return coverage


def _cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                              # not on Linux
        return os.cpu_count() or 1


def _me(role):
    me = f'{socket.gethostname()}, pid {os.getpid()}'
    return f'{me}, {role}' if role else me


def _write_info(run, info):
    fname = os.path.join(run, REFILL_INFO)
    temporary = os.path.join(run, f'.{REFILL_INFO}.{os.getpid()}.tmp')
    try:
        with open(temporary, 'w') as file:
            json.dump(info, file)
        os.replace(temporary, fname)
    except OSError:
        pass


def _read_info(run):
    try:
        with open(os.path.join(run, REFILL_INFO)) as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def _remove_info(run):
    try:
        os.remove(os.path.join(run, REFILL_INFO))
    except OSError:
        pass


def _who(info):
    if not info:
        return 'another process of the run'
    who = f'{info.get("host")}, pid {info.get("pid")}'
    return f'{who}, {info["role"]}' if info.get('role') else who


def _log_lines(log, text):
    for line in str(text).splitlines():
        log(line)


def _duration(seconds):
    """'45 s', '12 min' or '2.5 h'."""
    if seconds < 120:
        return f'{seconds:.0f} s'
    if seconds < 7200:
        return f'{seconds / 60:.0f} min'
    return f'{seconds / 3600:.1f} h'


class _Progress:
    """The refiller's progress lines, at most every `PROGRESS_SECONDS`."""

    def __init__(self, log, coverage, started):
        self.log = log
        self.trajectories = len(coverage.missing)
        self.frames = coverage.missing_frames
        self.started = self.last = started

    def __call__(self, done, frames):
        now = time.time()
        if now - self.last < PROGRESS_SECONDS:
            return
        self.last = now
        elapsed = now - self.started
        line = (f'{LOG_PREFIX} progress: {done} of {self.trajectories} '
                f'trajectories, {frames:,} of {self.frames:,} frames, '
                f'{_duration(elapsed)}')
        if 0 < frames < self.frames:
            left = elapsed * (self.frames - frames) / frames
            line += f', about {_duration(left)} left'
        self.log(line)


def ensure_series_coverage(params, directory, role=None, log=None,
                           stop=None, jobs=None):
    """The worker's check, before its task: the series is complete, or is
    refilled by exactly one process of the run while the others wait.

    Parameters
    ----------
    params : aimmd.Params
        The params of the run.
    directory : str
        The worker's folder: the run folder, or a system folder of a
        multi-system run.
    role : str, optional
        What this process is (e.g. ``'shoot run1/chainR0'``), for the logs.
    log : callable, optional
        Called with each log line (default: print).
    stop : callable, optional
        Polled while waiting: when it returns true the wait ends and False
        is returned (e.g. the worker received SIGTERM).
    jobs : int, optional
        Processes of a refill; by default the CPUs this process may run on.

    Returns
    -------
    bool
        True when the series is complete (or not checked), False when
        `stop` ended the wait.

    Raises
    ------
    SeriesCoverageError
        If trajectories lack the series and its policy does not refill it,
        if the refill of this process failed, or if the refill of another
        process ended with the series still incomplete.
    """
    log = log or _print
    series = _checked_series(params)
    if series is None:
        return True
    run, system_ids = run_folder(params, directory)
    coverage = series_coverage(run, series, system_ids)
    if coverage.complete:
        return True
    policy = series_policy(series)
    params_file = _params_file(params)
    if not (policy.refill and policy.refiller is not None):
        message = missing_series_message(coverage, policy, params_file)
        _log_lines(log, message)
        raise SeriesCoverageError(message)

    lock = FileLock(os.path.join(run, REFILL_LOCK))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return _wait(lock, coverage, policy, params_file, log, stop)
    try:
        return _refill(coverage, policy, params, role, log, jobs)
    finally:
        _remove_info(run)
        lock.release()


def _refill(coverage, policy, params, role, log, jobs):
    """Refill the series of a run while holding its refill lock."""
    run, series = coverage.run, coverage.series
    me = _me(role)
    _write_info(run, dict(host=socket.gethostname(), pid=os.getpid(),
                          role=role, series=series, started=time.time()))
    # another process may have refilled between the check and the lock
    coverage = series_coverage(run, series, coverage.system_ids)
    if coverage.complete:
        log(f'{LOG_PREFIX} {series!r} of run {run!r} is complete '
            f'(refilled by another process); starting.')
        return True
    params_file, workdir = _params_file(params), _workdir(params)
    jobs = max(1, int(jobs or _cpus()))
    route = _route(policy, coverage, workdir)
    _write_info(run, dict(host=socket.gethostname(), pid=os.getpid(),
                          role=role, series=series, started=time.time(),
                          route=route, trajectories=len(coverage.missing),
                          frames=coverage.missing_frames))
    log(f'{LOG_PREFIX} {coverage.summary()} have no {series!r} rows; this '
        f'process ({me}) refills them now with {jobs} '
        f'process{"es" if jobs != 1 else ""} by {route}; the other AIMMD '
        f'processes of this run wait until it is done.')
    started = time.time()
    progress = _Progress(log, coverage, started)

    def refill_log(line):
        log(f'{LOG_PREFIX}   {line}')

    try:
        result = policy.refiller(coverage, params_file=params_file,
                                 workdir=workdir, jobs=jobs, log=refill_log,
                                 progress=progress)
    except Exception as error:
        message = (f'{LOG_PREFIX} the refill of {series!r} in this process '
                   f'({me}) failed after {time.time() - started:.1f} s: '
                   f'{type(error).__name__}: {error}')
        after = series_coverage(run, series, coverage.system_ids)
        if not after.complete:
            message += '\n' + missing_series_message(
                after, policy, params_file, prefix=LOG_PREFIX)
        _log_lines(log, message)
        raise SeriesCoverageError(message) from error
    seconds = time.time() - started
    after = series_coverage(run, series, coverage.system_ids)
    if not after.complete:
        message = (f'{LOG_PREFIX} the refill of {series!r} in this process '
                   f'({me}) ended after {seconds:.1f} s, but '
                   f'{_trajectories(len(after.missing))} still have no file '
                   f'of the series (see the lines above).\n'
                   + missing_series_message(after, policy, params_file,
                                            prefix=LOG_PREFIX))
        _log_lines(log, message)
        raise SeriesCoverageError(message)
    if isinstance(result, dict) and result.get('route'):
        route = result['route']
    log(f'{LOG_PREFIX} done: {series!r} of '
        f'{_trajectories(len(coverage.missing))} '
        f'({coverage.missing_frames:,} frames) of run {run!r} refilled by '
        f'{route} in {seconds:.1f} s; the files of the old series were left '
        f'in place.')
    return True


def _wait(lock, coverage, policy, params_file, log, stop):
    """Wait for the process that refills the series, then check it."""
    run, series = coverage.run, coverage.series
    # the refiller names itself right after it took the lock
    info, deadline = _read_info(run), time.time() + 2.
    while info is None and time.time() < deadline:
        time.sleep(.05)
        info = _read_info(run)
    who = _who(info)
    log(f'{LOG_PREFIX} waiting for {who} to refill {series!r} '
        f'({coverage.summary()} lack it) before starting.')
    started = reported = time.time()
    while True:
        if stop is not None and stop():
            log(f'{LOG_PREFIX} stopped while waiting for the refill of '
                f'{series!r}.')
            return False
        try:
            lock.acquire(timeout=WAIT_SECONDS, poll_interval=_POLL_SECONDS)
            break
        except Timeout:
            now = time.time()
            if now - reported >= WAIT_REPORT_SECONDS:
                reported = now
                info = _read_info(run)
                if info:
                    who = _who(info)
                log(f'{LOG_PREFIX} still waiting for {who} to refill '
                    f'{series!r} ({_duration(now - started)} so far).')
    try:
        after = series_coverage(run, series, coverage.system_ids)
    finally:
        lock.release()
    waited = time.time() - started
    if after.complete:
        log(f'{LOG_PREFIX} {series!r} is complete after {_duration(waited)} '
            f'of waiting (refilled by {who}); starting.')
        return True
    message = (f'{LOG_PREFIX} {who} let go of the refill of {series!r} '
               f'after {_duration(waited)}, but '
               f'{_trajectories(len(after.missing))} still have no file of '
               f'the series: the refill failed (see its log). This process '
               f'does not refill again.\n'
               + missing_series_message(after, policy, params_file,
                                        prefix=LOG_PREFIX))
    _log_lines(log, message)
    raise SeriesCoverageError(message)
