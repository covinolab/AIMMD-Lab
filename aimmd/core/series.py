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
- both checks first refuse a params file written for another series than
  its featurizer now computes (`SeriesMismatchError`): the ``paramsN.py`` of
  a job holds the series name of the job's creation, while the featurizer
  is built from the params file as it is now. A params series that is not
  registered, while a registered refill callable claims names like it
  (``claims``) for another series, is such a mismatch;
- `check_series_coverage` (the launcher, before it writes or starts a job)
  raises `SeriesCoverageError` for a gap that may not be refilled, and
  announces the refill otherwise;
- `ensure_series_coverage` (every worker, before its task) raises for a gap
  that may not be refilled. Otherwise exactly one process of the run refills
  it, under the lock file `REFILL_LOCK` in the run folder, while every other
  process of the run waits for that lock and starts once the series is
  complete. A waiter that finds the series still incomplete after the
  refiller let go of the lock raises instead of refilling again, and so does
  a process of the same job that starts later: a failed refill leaves the
  marker `REFILL_FAILED` in the run folder (the series, the job, the
  process, the time and the error), which only a new job (another SLURM job,
  or a launch that started after the failure) disregards; a refill that
  succeeds removes it.

Every line the refill writes to the logs starts with ``'SERIES REFILL:'``:
its own lines, every line of an error, and what is printed (to
``sys.stdout`` or ``sys.stderr``) while the refill callable runs. The
default series ``'descriptors'`` is never checked. This module needs
neither torch nor a graph library.
"""

# external
import contextlib
import functools
import io
import json
import os
import socket
import sys
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

#: File in the run folder recording the last refill that failed (or was
#: stopped), so that the processes of the same job that start later do not
#: refill again.
REFILL_FAILED = '.series-refill-failed.json'

#: Environment variable holding the start (epoch seconds) of a launch
#: without SLURM: `aimmd.Launcher.run` sets it for the processes it starts.
JOB_START = 'AIMMD_JOB_START'

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


class SeriesMismatchError(SeriesCoverageError):
    """The params file of a job names another descriptor series than its
    featurizer computes: it was written for other settings."""


class RefillStopped(Exception):
    """A refill ended early on a stop request: raised by a refill callable
    when the ``stop`` it was passed returns true."""


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
    progress=..., stop=...)`` writes ``{trajectory}.{series}.npy`` for the
    trajectories of ``coverage.missing`` (a `SeriesCoverage`) with ``jobs``
    processes, reports through ``log`` (one line per call) and calls
    ``progress(done, frames)`` with the trajectories and frames done so far;
    it polls ``stop()`` between trajectories (or chunks of frames) and raises
    `RefillStopped` when it returns true, leaving the files it completed. It
    returns a dict whose ``'route'`` says how it refilled. Optional attributes describe it in
    messages: ``route(coverage, workdir)`` (how it would refill), ``label``
    (what the settings are called, e.g. ``'node-table'``), ``legacy_note``
    (why only ``*.descriptors.npy`` is there) and ``prefill_command(
    params_file, run, coverage=..., workdir=...)`` (the command that fills
    the series of the missing trajectories offline, by the route it would
    take). Two
    more find a params file written for other settings: ``claims(name)``
    tells whether a series name is of the kind this code computes (a params
    series of that kind that is not registered is then a mismatch), and
    ``parts`` lists registered series that are parts of this one (the
    systems of a multi-system featurizer), which a mismatch does not name.
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


def _claims(refiller, series):
    claims = getattr(refiller, 'claims', None)
    try:
        return bool(callable(claims) and claims(series))
    except Exception:                                   # noqa: BLE001
        return False


def computed_instead(series):
    """The series that the code of this process computes instead of
    `series`, a series name of the params.

    Parameters
    ----------
    series : str
        The params' descriptor series.

    Returns
    -------
    list of str
        Empty when `series` is registered, or when no registered refill
        callable claims a name like `series`. Otherwise the series whose
        refill callables claim it (in the order of registration), without
        the parts of others (see `SeriesPolicy`).
    """
    if series in _POLICIES:
        return []
    claiming = [policy for policy in _POLICIES.values()
                if _claims(policy.refiller, series)]
    parts = {part for policy in claiming
             for part in (getattr(policy.refiller, 'parts', None) or ())}
    names = [policy.series for policy in claiming]
    return [name for name in names if name not in parts] or names


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
                           prefix='SERIES CHECK:', workdir=None):
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
    workdir : str, optional
        The job's working directory (where a graph cache would be), for the
        prefill command.

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
        command = refiller.prefill_command(params_file, coverage.run,
                                           coverage=coverage, workdir=workdir)
    lines.append(f'  2. fill the series before the next job: {command};'
                 if command else
                 '  2. compute the series of these trajectories before the '
                 'next job;')
    if refiller is not None:
        lines.append('  3. construct the featurizer with refill=True: the '
                     'next job then refills the series first (one process '
                     'of the run refills, the others wait).')
    return '\n'.join(f'{prefix} {line}' for line in lines)


def mismatch_message(series, computed, params_file=None,
                     prefix='SERIES CHECK:'):
    """The error for a params file written for another series.

    Parameters
    ----------
    series : str
        The params' descriptor series (the one of the job's creation).
    computed : list of str
        What the featurizer computes now (`computed_instead`).
    params_file : str, optional
        The params file of the job (``paramsN.py``).
    prefix : str
        Start of every line.

    Returns
    -------
    str
        One line per item, each starting with `prefix`.
    """
    label = getattr(series_policy(computed[0]).refiller, 'label', None)
    featurizer = f'{label} featurizer' if label else 'featurizer'
    now = ' or '.join(map(repr, computed))
    where = (f'The params file this job uses, {params_file!r},' if params_file
             else 'The params file of this job')
    lines = [f'{where} was written for the descriptor series {series!r}, but '
             f'its {featurizer} now computes {now}.',
             f'The settings changed after the job was created: the params '
             f'file that builds the featurizer was edited, or MDAnalysis '
             f'guessed atom types differently on this host. Nothing starts: '
             f'the job would write rows of the new settings into the files '
             f'of {series!r}.',
             f'Regenerate the job: rerun the job-script generator '
             f'(Launcher.create_job), which writes a new params file for '
             f'{now}; the series check then applies as usual (an error, or a '
             f'refill with refill=True, if trajectories lack that series). '
             f'Create the job on the kind of host it runs on.']
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


def _check_computed(params, series, log=None):
    """Raise `SeriesMismatchError` if `params` was written for another
    series than the featurizers of this process compute (logging the lines
    first with `log`)."""
    computed = computed_instead(series)
    if not computed:
        return
    message = mismatch_message(series, computed, _params_file(params))
    if log is not None:
        _log_lines(log, message)
    raise SeriesMismatchError(message)


def run_folder(params, directory):
    """The run folder of a worker or launcher directory and its systems.

    Parameters
    ----------
    params : aimmd.Params
        The params of the run.
    directory : str
        The run folder, or (multi-system run) one of its system folders
        ``{run}/{system_id}``: a folder named like a system that holds the
        exported initial paths (``initial*``) but no system folder, below a
        folder that holds a folder of every system (as the launcher builds
        them). A run folder named like a system is the run.

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
        if _is_system_folder(directory, system_ids):
            directory = os.path.dirname(directory)
    return directory, system_ids


def _is_system_folder(directory, system_ids):
    """Whether `directory` is a system folder of a multi-system run (see
    `run_folder`)."""
    if os.path.basename(directory) not in system_ids:
        return False
    try:
        folders = {name for name in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, name))}
    except OSError:
        return False
    if folders & set(system_ids):           # a run named like a system
        return False
    if not any(name.startswith(SEED_PREFIX) for name in folders):
        return False
    parent = os.path.dirname(directory)
    return all(os.path.isdir(os.path.join(parent, system_id))
               for system_id in system_ids)


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
    SeriesMismatchError
        If the params file was written for another series than its
        featurizer now computes (see `computed_instead`).
    SeriesCoverageError
        If trajectories of the run have no file of the series and its
        policy does not refill it.
    """
    log = log or _print
    series = _checked_series(params)
    if series is None:
        return None
    _check_computed(params, series)
    run, system_ids = run_folder(params, directory)
    coverage = series_coverage(run, series, system_ids)
    if coverage.complete:
        return coverage
    policy = series_policy(series)
    if not (policy.refill and policy.refiller is not None):
        raise SeriesCoverageError(missing_series_message(
            coverage, policy, _params_file(params), workdir=_workdir(params)))
    log(f'{LOG_PREFIX} {coverage.summary()} have no {series!r} series '
        f'file. The featurizer has refill=True: the job first refills them '
        f'in one process, by {_route(policy, coverage, _workdir(params))}, '
        f'while the other processes of the run wait; the files of other '
        f'series stay in place.')
    marker = _read_json(os.path.join(run, REFILL_FAILED))
    if (isinstance(marker, dict) and marker.get('series') == series
            and marker.get('status') == 'failed'):
        log(f'{LOG_PREFIX} the last refill of {series!r} '
            f'({_job_text(marker)}{_who(marker)}, at '
            f'{_clock(marker.get("time"))}) failed: {marker.get("error")}; '
            f'the job tries again.')
    return coverage


def _cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                              # not on Linux
        return os.cpu_count() or 1


def _me(role):
    me = f'{socket.gethostname()}, pid {os.getpid()}'
    return f'{me}, {role}' if role else me


def _write_json(fname, data):
    """Write `data` to `fname` (replacing it at once); no error if the
    folder cannot be written."""
    folder, name = os.path.split(fname)
    temporary = os.path.join(folder, f'.{name}.{os.getpid()}.tmp')
    try:
        with open(temporary, 'w') as file:
            json.dump(data, file)
        os.replace(temporary, fname)
    except OSError:
        _remove(temporary)


def _read_json(fname):
    try:
        with open(fname) as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def _remove(fname):
    try:
        os.remove(fname)
    except OSError:
        pass


def _write_info(run, info):
    _write_json(os.path.join(run, REFILL_INFO), info)


def _read_info(run):
    return _read_json(os.path.join(run, REFILL_INFO))


def _remove_info(run):
    _remove(os.path.join(run, REFILL_INFO))


def _who(info):
    if not info:
        return 'another process of the run'
    who = f'{info.get("host")}, pid {info.get("pid")}'
    return f'{who}, {info["role"]}' if info.get('role') else who


# the record of a failed refill, per job

def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _process_start():
    """When this process started (epoch seconds), or None."""
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:                                   # noqa: BLE001
        return None


def _job():
    """``(job, start)``: the SLURM job of this process (its id, and the
    restart of a requeued job) or None, and when the job started (epoch
    seconds, None if unknown). Without SLURM a job is a launch of
    `aimmd.Launcher.run` (`JOB_START`), or else this process."""
    job = os.environ.get('SLURM_JOB_ID') or None
    if job:
        restart = os.environ.get('SLURM_RESTART_COUNT') or '0'
        if restart != '0':
            job = f'{job}, restart {restart}'
        return job, _float(os.environ.get('SLURM_JOB_START_TIME'))
    start = _float(os.environ.get(JOB_START))
    return None, start if start is not None else _process_start()


def _job_text(marker):
    job = marker.get('job')
    return f'SLURM job {job}: ' if job else ''


def _clock(seconds):
    seconds = _float(seconds)
    if seconds is None:
        return 'an unknown time'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))


def _summary(text, limit=500):
    """`text` on one line, at most `limit` characters."""
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _record_failure(run, series, role, started, status, error):
    """Leave `REFILL_FAILED`: the refill of `series` by this process
    ended with `status` ('failed' or 'stopped') and `error`."""
    _write_json(os.path.join(run, REFILL_FAILED), dict(
        series=series, status=status, job=_job()[0],
        host=socket.gethostname(), pid=os.getpid(), role=role,
        started=started, time=time.time(), error=_summary(error)))


def _failed_in_this_job(run, series):
    """The record of a refill of `series` that failed in the job of this
    process (not one that was stopped, nor one of an earlier job), or
    None."""
    marker = _read_json(os.path.join(run, REFILL_FAILED))
    if (not isinstance(marker, dict) or marker.get('series') != series
            or marker.get('status') != 'failed'):
        return None
    job, start = _job()
    failed = _float(marker.get('time'))
    if marker.get('job') != job or failed is None:
        return None
    if start is not None and failed < start:
        return None                 # before this job started
    return marker


def _log_lines(log, text):
    for line in str(text).splitlines():
        log(line)


def _prefixed(text, prefix=LOG_PREFIX):
    """`text` with every line starting with `prefix` (the lines that do not
    are indented below it)."""
    return '\n'.join(line if line.startswith(prefix)
                     else f'{prefix}   {line}'.rstrip()
                     for line in str(text).splitlines())


class _LineWriter(io.TextIOBase):
    """A text stream that passes every complete line to `emit`."""

    def __init__(self, emit):
        super().__init__()
        self._emit = emit
        self._pending = ''

    def writable(self):
        return True

    def write(self, text):
        *lines, self._pending = (self._pending + str(text)).split('\n')
        for line in lines:
            self._emit(line)
        return len(text)

    def finish(self):
        """Pass on a last line that has no end."""
        if self._pending:
            line, self._pending = self._pending, ''
            self._emit(line)


@contextlib.contextmanager
def _prefixed_output(log):
    """Inside, every line printed to ``sys.stdout`` or ``sys.stderr`` (by
    the refill callable, e.g. the featurizer's ERROR and WARNING lines)
    reaches `log` as a refill line. Yields ``emit(line)``, which passes a
    line to `log` as it is, with the streams of before (`log` may print)."""
    stdout, stderr = sys.stdout, sys.stderr

    def emit(line):
        inside = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        try:
            log(line)
        finally:
            sys.stdout, sys.stderr = inside

    writer = _LineWriter(lambda line: emit(_prefixed(line) or LOG_PREFIX))
    sys.stdout = sys.stderr = writer
    try:
        yield emit
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        writer.finish()


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
        A stop request (e.g. the worker received SIGTERM): polled while
        waiting and before a refill, and passed to the refill callable,
        which polls it between trajectories or chunks of frames. When it
        returns true the wait or the refill ends (the refill leaving the
        files it completed, its lock released and a 'stopped' record that
        does not keep a job from refilling) and False is returned.
    jobs : int, optional
        Processes of a refill; by default the CPUs this process may run on.

    Returns
    -------
    bool
        True when the series is complete (or not checked), False when
        `stop` ended the wait or the refill.

    Raises
    ------
    SeriesMismatchError
        If the params file was written for another series than its
        featurizer now computes (see `computed_instead`), before anything
        else, even when the series of the params file is complete.
    SeriesCoverageError
        If trajectories lack the series and its policy does not refill it,
        if the refill of this process failed, or if the refill of another
        process ended with the series still incomplete.
    """
    log = log or _print
    series = _checked_series(params)
    if series is None:
        return True
    _check_computed(params, series, log)
    run, system_ids = run_folder(params, directory)
    coverage = series_coverage(run, series, system_ids)
    if coverage.complete:
        return True
    policy = series_policy(series)
    params_file = _params_file(params)
    if not (policy.refill and policy.refiller is not None):
        message = missing_series_message(coverage, policy, params_file,
                                         workdir=_workdir(params))
        _log_lines(log, message)
        raise SeriesCoverageError(message)

    lock = FileLock(os.path.join(run, REFILL_LOCK))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return _wait(lock, coverage, policy, params_file, log, stop,
                     _workdir(params))
    try:
        return _refill(coverage, policy, params, role, log, jobs, stop)
    finally:
        _remove_info(run)
        lock.release()


def _refill(coverage, policy, params, role, log, jobs, stop):
    """Refill the series of a run while holding its refill lock."""
    run, series = coverage.run, coverage.series
    me = _me(role)
    # another process may have refilled between the check and the lock (the
    # info file for the waiters comes once this process really refills)
    coverage = series_coverage(run, series, coverage.system_ids)
    if coverage.complete:
        log(f'{LOG_PREFIX} {series!r} of run {run!r} is complete '
            f'(refilled by another process); starting.')
        return True
    params_file, workdir = _params_file(params), _workdir(params)
    failed = _failed_in_this_job(run, series)
    if failed is not None:
        message = (f'{LOG_PREFIX} the refill of {series!r} already failed in '
                   f'this job ({_job_text(failed)}{_who(failed)}, at '
                   f'{_clock(failed.get("time"))}): {failed.get("error")}. '
                   f'This process does not refill again; the next job tries '
                   f'again (to let this job try again, delete '
                   f'{os.path.join(run, REFILL_FAILED)}).\n'
                   + missing_series_message(coverage, policy, params_file,
                                            prefix=LOG_PREFIX,
                                            workdir=workdir))
        _log_lines(log, message)
        raise SeriesCoverageError(message)
    if stop is not None and stop():
        log(f'{LOG_PREFIX} stopped before the refill of {series!r} '
            f'({coverage.summary()} lack it).')
        return False
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
    try:
        with _prefixed_output(log) as emit:
            def refill_log(text):
                for line in str(text).splitlines() or ['']:
                    emit(_prefixed(line) or LOG_PREFIX)

            result = policy.refiller(
                coverage, params_file=params_file, workdir=workdir,
                jobs=jobs, log=refill_log, stop=stop,
                progress=_Progress(emit, coverage, started))
    except RefillStopped as error:
        _record_failure(run, series, role, started, 'stopped',
                        f'{type(error).__name__}: {error}')
        after = series_coverage(run, series, coverage.system_ids)
        log(f'{LOG_PREFIX} the refill of {series!r} in this process ({me}) '
            f'stopped on request after {time.time() - started:.1f} s; '
            f'{_trajectories(len(after.missing))} still lack the series '
            f'(the trajectories it completed keep their files), and the next '
            f'job refills them.')
        return False
    except Exception as error:
        text = f'{type(error).__name__}: {error}'
        _record_failure(run, series, role, started, 'failed', text)
        message = _prefixed(
            f'{LOG_PREFIX} the refill of {series!r} in this process ({me}) '
            f'failed after {time.time() - started:.1f} s: {text}')
        after = series_coverage(run, series, coverage.system_ids)
        if not after.complete:
            message += '\n' + missing_series_message(
                after, policy, params_file, prefix=LOG_PREFIX,
                workdir=workdir)
        _log_lines(log, message)
        raise SeriesCoverageError(message) from error
    seconds = time.time() - started
    after = series_coverage(run, series, coverage.system_ids)
    if not after.complete:
        text = (f'{_trajectories(len(after.missing))} still have no file of '
                f'the series after the refill')
        _record_failure(run, series, role, started, 'failed', text)
        message = (f'{LOG_PREFIX} the refill of {series!r} in this process '
                   f'({me}) ended after {seconds:.1f} s, but '
                   f'{_trajectories(len(after.missing))} still have no file '
                   f'of the series (see the lines above).\n'
                   + missing_series_message(after, policy, params_file,
                                            prefix=LOG_PREFIX,
                                            workdir=workdir))
        _log_lines(log, message)
        raise SeriesCoverageError(message)
    _remove(os.path.join(run, REFILL_FAILED))
    if isinstance(result, dict) and result.get('route'):
        route = result['route']
    log(f'{LOG_PREFIX} done: {series!r} of '
        f'{_trajectories(len(coverage.missing))} '
        f'({coverage.missing_frames:,} frames) of run {run!r} refilled by '
        f'{route} in {seconds:.1f} s; the files of the old series were left '
        f'in place.')
    return True


def _wait(lock, coverage, policy, params_file, log, stop, workdir=None):
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
    failed = _read_json(os.path.join(run, REFILL_FAILED))
    error = (f': {failed.get("error")}' if isinstance(failed, dict)
             and failed.get('series') == series else '')
    message = (f'{LOG_PREFIX} {who} let go of the refill of {series!r} '
               f'after {_duration(waited)}, but '
               f'{_trajectories(len(after.missing))} still have no file of '
               f'the series: the refill failed (see its log){error}. This '
               f'process does not refill again.\n'
               + missing_series_message(after, policy, params_file,
                                        prefix=LOG_PREFIX, workdir=workdir))
    _log_lines(log, message)
    raise SeriesCoverageError(message)
