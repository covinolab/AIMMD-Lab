"""Prefill, repack and verify the node-table series of AIMMD runs.

The work behind ``python -m aimmd.network.nodetables`` (the command line is in
`aimmd.network.nodetables._cli`). A trajectory belongs to a run when it has a
states series ``{trajectory}.states.npy``: the exported initial paths, the
chain paths, the halves of shots in flight and the parts of free simulations.
Its node-table series is ``{trajectory}.{series}.npy``, where ``series`` is
the series name of the featurizer of the params file. Like AIMMD itself
(`aimmd.cache.mda.count_safe_frames`), the tools cover the readable frames of
a trajectory: a last frame cut short (by a job killed while it wrote) is left
out and reported.

- `prefill` computes node-table rows for every frame (with ``only_missing``:
  for the frames whose rows are missing or zero). With a graph cache of the
  coordinate-descriptor mode (``db``) it takes each frame's cached graph
  (decode the frame, compute its graph-cache key, look it up read-only,
  `NodeTableFeaturizer.row_from_graph`), and featurizes the frame from its
  coordinates when the cache has no usable graph for it. ``verify`` frames per
  trajectory are then featurized directly and compared bit for bit.
- `repack` rewrites series files for another row capacity (``n_max``)
  without reading any trajectory (`repack_rows`).
- `verify` reports missing, short and zero rows per trajectory and checks
  sampled rows against a direct featurization.

Writes go to a temporary file ``.{series file}.tmp`` next to the series file,
which replaces the series file (`os.replace`, under the series file's lock,
as `aimmd.cache.npy.save_npy` does) only once it is complete and verified. A
crash leaves the series file as it was; the next run overwrites the temporary
file. The files are plain ``.npy`` files with the 128-byte header that
`aimmd.cache.npy.update_npy` expects. The tools never open the coordinate
series ``*.descriptors.npy``, and they open graph caches read-only and
immutable: they never write a graph cache (run them while no job writes it).

Parallel runs (``jobs > 1``) use spawned processes, each of which imports the
params file once (and checks that its featurizer still has the series being
written); the work is split into chunks of frames. Their output goes where
this process's goes, unless `CHILD_OUTPUT` names a file for it (the refill
of a job passes it on to its log). A report says ``broken_pool`` when the
processes could not start or died.

A job runs `prefill` and `repack` itself, for the trajectories of its run that
lack the series, when the featurizer has ``refill=True`` (see
`aimmd.network.nodetables._refill`).
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import io
import itertools
import multiprocessing
import os
import sqlite3
import sys
import time
import types
import zlib
from collections import namedtuple
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout

from ._featurizer import (MultiSystemNodeTableFeaturizer, NodeTableFeaturizer,
                          repack_rows)
from ...cache.mda import count_safe_frames
from ...core import series as _series

#: Suffix of the series that makes a trajectory part of a run.
STATES_SUFFIX = _series.STATES_SUFFIX

#: Byte offset of the rows in the series files written here (the header
#: size `aimmd.cache.npy.update_npy` relies on).
HEADER_BYTES = 128

#: Default number of frames per prefill task.
DEFAULT_CHUNK_FRAMES = 1000

#: Frames per trajectory that a prefill from a graph cache verifies by
#: default: the cache keys hash coordinates only, so graphs cached with other
#: selections would otherwise pass unnoticed.
DEFAULT_DB_VERIFY = 4

#: Timing categories of the reports (seconds of task time, all processes).
SECONDS = ('plan', 'read', 'hash', 'lookup', 'graph', 'featurize', 'write',
           'verify', 'install')

# frames featurized together (the extract route's misses)
_FEATURIZE_BATCH = 32
# rows per block when copying, scanning or repacking series files
_BLOCK_ROWS = 4096
# seconds to wait for the lock of a series file
_LOCK_TIMEOUT = 60.
# frames listed per file in a report
_MAX_LISTED = 20
# file names in the run folders
_RUN_LOCK = '.nodetables.lock'
_TEMPORARY_SUFFIX = '.tmp'

_Frame = namedtuple('_Frame', 'time positions')
_SeriesInfo = namedtuple('_SeriesInfo', 'n_rows width offset')

_IMPORTS = itertools.count()

#: Environment variable naming a file that the spawned processes write their
#: stdout and stderr to (set by the refill of a job, which passes the lines
#: on to its log).
CHILD_OUTPUT = 'AIMMD_NODETABLES_CHILD_OUTPUT'


class UsageError(ValueError):
    """A command cannot start: bad params file, runs or options."""


# ----------------------------------------------------------------------
# the featurizer of a params file

class ParamsFeaturizer:
    """The node-table featurizer of a params file (see `load_featurizer`).

    Attributes
    ----------
    params_file : str
        Absolute path of the params file.
    name : str
        Name of the featurizer in the params file (e.g. ``'FEATURIZER'``).
    featurizer : NodeTableFeaturizer or MultiSystemNodeTableFeaturizer
        The featurizer.
    pinned : str or None
        ``descriptors_series`` of the params file, if it sets one (checked
        against the featurizer).
    """

    def __init__(self, params_file, name, featurizer, pinned):
        self.params_file = params_file
        self.name = name
        self.featurizer = featurizer
        self.pinned = pinned

    @property
    def series(self):
        """str: the series name of the featurizer."""
        return self.featurizer.series


def load_featurizer(params_file, name=None):
    """Import a params file as a module and take its node-table featurizer.

    The file is executed in its own folder, like `aimmd.Params.load` does,
    but no `aimmd.Params` is built (Params.load featurizes the initial paths
    and saves a params file with this host's paths). A file that sets
    ``GRAPH_INPUT`` (as the template does) to another string than
    ``'nodetables'`` is refused before it runs: in ``'sqlite'`` mode it would
    open (and could create) its graph cache.

    Parameters
    ----------
    params_file : str
        Params file in node-table mode.
    name : str, optional
        Name of the featurizer in the params file. By default
        ``FEATURIZER``, or the only `NodeTableFeaturizer` or
        `MultiSystemNodeTableFeaturizer` the file defines.

    Returns
    -------
    ParamsFeaturizer

    Raises
    ------
    UsageError
        If the file is missing, is not in node-table mode, fails to import
        (e.g. its pinned ``descriptors_series`` does not match the
        featurizer), defines no such featurizer or several, or pins another
        series name.
    """
    params_file = os.path.abspath(params_file)
    if not os.path.isfile(params_file):
        raise UsageError(f'params file {params_file!r} not found')
    graph_input = _graph_input(params_file)
    if graph_input not in (None, 'nodetables'):
        raise UsageError(
            f'{params_file!r} sets GRAPH_INPUT = {graph_input!r}: switch it '
            f"to GRAPH_INPUT = 'nodetables' first. It was not executed (in "
            f"'sqlite' mode it would open its graph cache read-write).")
    try:
        module = _import_module(params_file)
    except Exception as error:
        raise UsageError(f'importing the params file {params_file!r} failed: '
                         f'{type(error).__name__}: {error}') from error
    featurizer, name = _module_featurizer(module, name, params_file)
    pinned = getattr(module, 'descriptors_series', None)
    if pinned is not None:
        try:
            featurizer.check_series(pinned)
        except ValueError as error:
            raise UsageError(str(error)) from error
    return ParamsFeaturizer(params_file, name, featurizer, pinned)


def _graph_input(params_file):
    """The string a params file assigns to ``GRAPH_INPUT`` at module level
    (the last such assignment), read without executing the file; None if it
    assigns none, or something else than a string."""
    try:
        tree = ast.parse(Path(params_file).read_text(), params_file)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None                 # the import reports the error
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, assigned = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, assigned = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == 'GRAPH_INPUT'
               for target in targets):
            value = (assigned.value if isinstance(assigned, ast.Constant)
                     and isinstance(assigned.value, str) else None)
    return value


def _import_module(params_file):
    """Execute a params file in its folder as a fresh module.

    The source is compiled every time (no bytecode cache, which could serve
    an edited file of the same size and modification second).
    """
    folder = os.path.dirname(params_file)
    module = types.ModuleType(f'_aimmd_nodetables_params_{next(_IMPORTS)}')
    module.__file__ = params_file
    source = Path(params_file).read_text()
    cwd = os.getcwd()
    sys.path.insert(0, folder)
    os.chdir(folder)
    try:
        exec(compile(source, params_file, 'exec'), module.__dict__)
    finally:
        os.chdir(cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(folder)
    return module


def _module_featurizer(module, name, params_file):
    """The featurizer of a params module and its name."""
    kinds = (NodeTableFeaturizer, MultiSystemNodeTableFeaturizer)
    if name is not None:
        featurizer = getattr(module, name, None)
        if not isinstance(featurizer, kinds):
            raise UsageError(f'{name!r} of {params_file!r} is not a '
                             f'NodeTableFeaturizer or '
                             f'MultiSystemNodeTableFeaturizer')
        return featurizer, name
    found = {key: value for key, value in vars(module).items()
             if isinstance(value, kinds)}
    if 'FEATURIZER' in found:
        return found['FEATURIZER'], 'FEATURIZER'
    if len(found) == 1:
        (name, featurizer), = found.items()
        return featurizer, name
    if not found:
        raise UsageError(
            f'{params_file!r} defines no NodeTableFeaturizer (or '
            f'MultiSystemNodeTableFeaturizer). Use the params file in '
            f"node-table mode (GRAPH_INPUT = 'nodetables'), which defines "
            f'the featurizer, e.g. FEATURIZER = NodeTableFeaturizer(...)')
    raise UsageError(f'{params_file!r} defines several node-table '
                     f'featurizers ({sorted(found)}): choose one with '
                     f'--featurizer')


def _system(featurizer, system_id):
    """The featurizer of one system (`featurizer` itself in a single-system
    run)."""
    return featurizer if system_id is None else featurizer[system_id]


# ----------------------------------------------------------------------
# runs and files

def find_trajectories(runs, featurizer):
    """The trajectories of AIMMD runs: those with a states series.

    Parameters
    ----------
    runs : list of str
        Run folders.
    featurizer : NodeTableFeaturizer or MultiSystemNodeTableFeaturizer
        In a multi-system run, a trajectory belongs to the system named by
        its first folder below the run (``{run}/{system_id}/...``); others
        are left out.

    Returns
    -------
    list of tuple
        ``(trajectory, system_id)``, absolute paths in sorted order per run;
        ``system_id`` is None for a single-system featurizer.

    See Also
    --------
    aimmd.core.series.find_trajectories : the rule, shared with the check
        of a run's series before a job.
    """
    multi = isinstance(featurizer, MultiSystemNodeTableFeaturizer)
    return _series.find_trajectories(
        runs, featurizer.system_ids if multi else None)


def _series_files(runs, featurizer):
    """``(series file, trajectory, system_id)`` of every series file of the
    featurizer's series in the runs (with or without a trajectory)."""
    multi = isinstance(featurizer, MultiSystemNodeTableFeaturizer)
    suffix = f'.{featurizer.series}.npy'
    found = []
    for run in runs:
        run = os.path.abspath(run)
        in_run = []
        for folder, folders, files in os.walk(run):
            folders.sort()
            system_id = None
            if multi:
                system_id = Path(os.path.relpath(folder, run)).parts[:1]
                system_id = system_id[0] if system_id else None
                if system_id not in featurizer.system_ids:
                    continue
            for name in files:
                if not name.startswith('.') and name.endswith(suffix):
                    in_run.append((os.path.join(folder, name),
                                   os.path.join(folder, name[:-len(suffix)]),
                                   system_id))
        found += sorted(in_run)
    return found


def series_file(trajectory, series):
    """The series file of a trajectory, ``{trajectory}.{series}.npy``."""
    return f'{trajectory}.{series}.npy'


def _lock(fname, timeout=_LOCK_TIMEOUT):
    """The lock of a series file, named as `aimmd.cache.npy` names it."""
    folder, name = os.path.split(fname)
    return FileLock(os.path.join(folder, f'.{name}.lock'), timeout=timeout)


def _temporary(fname):
    """The temporary file a series file is written to before it replaces
    the series file."""
    folder, name = os.path.split(fname)
    return os.path.join(folder, f'.{name}{_TEMPORARY_SUFFIX}')


def _discard(fname):
    with contextlib.suppress(FileNotFoundError):
        os.remove(fname)


def _identity(fname):
    """What changes when a file is replaced or written; None if missing."""
    try:
        stat = os.stat(fname)
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


@contextlib.contextmanager
def _open_reader(trajectory, featurizer):
    """An MDAnalysis reader of a trajectory, checked against the featurizer's
    atom count."""
    from MDAnalysis.coordinates.core import reader
    frames = reader(trajectory)
    try:
        if frames.n_atoms != featurizer.n_atoms:
            raise ValueError(f'{trajectory} has {frames.n_atoms} atoms, the '
                             f'featurizer topology {featurizer.n_atoms}')
        yield frames
    finally:
        frames.close()


def _readable_frames(reader):
    """``(readable, unreadable)``: the number of frames ``0..readable - 1``
    that can be read, as AIMMD counts them, and of the frames after them
    (e.g. a last frame cut short)."""
    try:
        readable = count_safe_frames(reader)
    except RuntimeError:                     # no readable frame
        readable = 0
    return readable, reader.n_frames - readable


def _header(n_rows, width):
    """The npy header of a float32 ``(n_rows, width)`` series file."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {'descr': np.dtype(np.float32).str, 'fortran_order': False,
                 'shape': (int(n_rows), int(width))})
    header = buffer.getvalue()
    if len(header) != HEADER_BYTES:
        raise RuntimeError(f'npy header of {len(header)} bytes, expected '
                           f'{HEADER_BYTES}')
    return header


def _create(fname, n_rows, width):
    """Create a zero-filled float32 ``(n_rows, width)`` npy file."""
    with open(fname, 'wb') as file:
        file.write(_header(n_rows, width))
        file.truncate(HEADER_BYTES + int(n_rows) * int(width) * 4)


def _series_info(file, fname, width=None):
    """Shape and data offset of an open series file.

    Raises
    ------
    ValueError
        If it is not a 2-d, C-order float32 npy file (with rows of `width`
        columns, if given).
    """
    version = np.lib.format.read_magic(file)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
    else:
        raise ValueError(f'{fname}: npy format {version} is not supported')
    if dtype != np.float32 or fortran_order or len(shape) != 2:
        raise ValueError(f'{fname} is not a node-table series: '
                         f'{dtype} {shape}, fortran_order={fortran_order}')
    if width is not None and shape[1] != width:
        raise ValueError(f'{fname} has rows of width {shape[1]}, the '
                         f'featurizer {width} (another n_max or layout)')
    info = _SeriesInfo(int(shape[0]), int(shape[1]), file.tell())
    if os.fstat(file.fileno()).st_size < info.offset + \
            info.n_rows * info.width * 4:
        raise ValueError(f'{fname} is shorter than its header says')
    return info


def _read_block(file, info, start, stop):
    """Rows ``start:stop`` of an open series file."""
    nbytes = (stop - start) * info.width * 4
    data = os.pread(file.fileno(), nbytes, info.offset +
                    start * info.width * 4)
    if len(data) != nbytes:
        raise OSError(f'short read from {file.name}')
    return np.frombuffer(data, dtype=np.float32).reshape(stop - start,
                                                         info.width)


def _read_rows(fname, frames, width):
    """Rows at sorted `frames` of a series file."""
    rows = np.zeros((len(frames), width), dtype=np.float32)
    with open(fname, 'rb') as file:
        info = _series_info(file, fname, width)
        for begin, end in _runs(frames):
            rows[begin:end] = _read_block(file, info, int(frames[begin]),
                                          int(frames[end - 1]) + 1)
    return rows


def _present_rows(file, info, n_frames):
    """Which of the first `n_frames` rows of an open series file are filled
    (not zero), as a boolean array of length `n_frames`."""
    present = np.zeros(n_frames, dtype=bool)
    stop = min(info.n_rows, n_frames)
    for start in range(0, stop, _BLOCK_ROWS):
        end = min(start + _BLOCK_ROWS, stop)
        present[start:end] = _read_block(file, info, start, end)[:, 0] != 0
    return present


def _runs(frames):
    """``(begin, end)`` index ranges of the consecutive runs of sorted
    `frames`."""
    if not len(frames):
        return []
    breaks = (np.flatnonzero(np.diff(frames) != 1) + 1).tolist()
    edges = [0, *breaks, len(frames)]
    return list(zip(edges[:-1], edges[1:]))


def _write_rows(fname, frames, rows):
    """Write `rows` at the sorted `frames` of a series file made by
    `_create`."""
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    rowbytes = rows.shape[1] * 4
    fd = os.open(fname, os.O_WRONLY)
    try:
        for begin, end in _runs(frames):
            data = memoryview(rows[begin:end]).cast('B')
            offset = HEADER_BYTES + int(frames[begin]) * rowbytes
            while len(data):
                written = os.pwrite(fd, data, offset)
                data = data[written:]
                offset += written
    finally:
        os.close(fd)


def _fsync(fname):
    fd = os.open(fname, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _bits(rows):
    return np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32)


def _sample(trajectory, frames, k, seed):
    """Up to `k` of `frames`, drawn reproducibly per trajectory, sorted."""
    frames = np.asarray(frames, dtype=np.int64)
    if k <= 0 or not len(frames):
        return frames[:0]
    if k >= len(frames):
        return frames
    rng = np.random.default_rng([seed, zlib.crc32(trajectory.encode())])
    return np.sort(rng.choice(frames, k, replace=False))


def _featurize_frames(featurizer, trajectory, frames):
    """Rows of the sorted `frames` of a trajectory, featurized directly."""
    with _open_reader(trajectory, featurizer) as reader:
        return featurizer.descriptors_function(reader[frames.tolist()])


def _check_rows(featurizer, trajectory, frames, stored):
    """Frames whose `stored` rows differ (in any bit) from a direct
    featurization."""
    if not len(frames):
        return []
    fresh = _featurize_frames(featurizer, trajectory, frames)
    differ = np.any(_bits(stored) != _bits(fresh), axis=1)
    return [int(frame) for frame in frames[differ]]


@contextlib.contextmanager
def _run_locks(runs):
    """Hold a lock per run folder, so that two writing commands never work
    on one run."""
    held = []
    try:
        for run in runs:
            lock = FileLock(os.path.join(run, _RUN_LOCK), timeout=0)
            try:
                lock.acquire()
            except Timeout:
                raise UsageError(f'another node-table command is running on '
                                 f'{run!r} (it holds {lock.lock_file!r})'
                                 ) from None
            held.append(lock)
        yield
    finally:
        for lock in reversed(held):
            lock.release()


def _run_folders(runs):
    """Absolute, distinct run folders; UsageError if one is missing."""
    folders = []
    for run in runs:
        folder = os.path.abspath(run)
        if not os.path.isdir(folder):
            raise UsageError(f'run folder {run!r} not found')
        if folder not in folders:
            folders.append(folder)
    return folders


# ----------------------------------------------------------------------
# graph caches (extract route)

def _connect(database):
    """A read-only, immutable connection to a graph cache: it never writes
    the database, nor -wal or -shm files next to it."""
    uri = Path(database).resolve().as_uri() + '?mode=ro&immutable=1'
    return sqlite3.connect(uri, uri=True)


def _databases(databases, featurizer):
    """``{system_id or None: path}`` of the graph caches to extract from.

    Each entry of `databases` is a path (every system) or, for a
    multi-system featurizer, ``SYSTEM_ID=PATH``.
    """
    if not databases:
        return {}
    try:
        import torch_geometric  # noqa: F401
    except ImportError as error:
        raise UsageError(f'extracting rows from a graph cache needs the '
                         f'optional graphs extra (torch_geometric) to decode '
                         f'the cached graphs: {error}') from error
    system_ids = getattr(featurizer, 'system_ids', [])
    result = {}
    for value in databases:
        system_id, separator, path = value.partition('=')
        if not separator or system_id not in system_ids:
            system_id, path = None, value
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise UsageError(f'graph cache {value!r} not found')
        try:
            connection = _connect(path)
            try:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                    "name='graphs_cache'").fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise UsageError(f'cannot read the graph cache {path!r}: '
                             f'{error}') from error
        if table is None:
            raise UsageError(f'{path!r} has no graphs_cache table: not a '
                             f'graph cache of aimmd.network.graph_utils')
        result[system_id] = path
    return result


def _database_notes(databases):
    """Warnings about graph caches with a write-ahead log, whose rows an
    immutable reader does not see."""
    notes = []
    for path in sorted(set(databases.values())):
        wal = f'{path}-wal'
        if os.path.exists(wal) and os.path.getsize(wal):
            notes.append(
                f'WARNING: {wal} is not empty: graphs written since the last '
                f'checkpoint are not seen (the cache is read immutable) and '
                f'their frames are featurized instead. Checkpoint the cache '
                f'while no job runs, or accept the slower route for them.')
    return notes


# ----------------------------------------------------------------------
# executing tasks

class _Serial:
    """Runs every task in this process when it is submitted."""

    def __init__(self, featurizer):
        self.featurizer = featurizer

    def submit(self, function, task):
        future = concurrent.futures.Future()
        try:
            future.set_result(function(self.featurizer, task))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, cancel=False):
        pass


class _Parallel:
    """Runs tasks in spawned processes, each with its own featurizer."""

    def __init__(self, jobs, params_file=None, name=None, series=None):
        self.pool = concurrent.futures.ProcessPoolExecutor(
            jobs, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker,
            initargs=(params_file, name, series))

    def submit(self, function, task):
        return self.pool.submit(_run_in_worker, function, task)

    def shutdown(self, cancel=False):
        self.pool.shutdown(wait=True, cancel_futures=cancel)


_WORKER_FEATURIZER = None


def _redirect_output(fname):
    """Send this process's stdout and stderr (file descriptors 1 and 2, so
    also what libraries write) to the end of `fname`."""
    descriptor = os.open(fname, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        for stream, target in ((sys.stdout, 1), (sys.stderr, 2)):
            with contextlib.suppress(Exception):
                stream.flush()
            os.dup2(descriptor, target)
    finally:
        os.close(descriptor)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)


def _initialize_worker(params_file, name, series=None):
    """Import the params file once per worker process; with `series`,
    refuse a featurizer of another series (the file changed meanwhile)."""
    global _WORKER_FEATURIZER
    output = os.environ.get(CHILD_OUTPUT)
    if output:
        _redirect_output(output)
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    try:
        if params_file is not None:
            loaded = load_featurizer(params_file, name)
            if series is not None and loaded.series != series:
                raise UsageError(
                    f'{params_file!r} now gives the series '
                    f'{loaded.series!r}, not {series!r}, which this command '
                    f'writes: the params file changed while the command ran')
            _WORKER_FEATURIZER = loaded.featurizer
    except BaseException as error:
        if output:              # the one line the log needs, before the trace
            print(f'a spawned process could not load the featurizer: '
                  f'{type(error).__name__}: {error}', file=sys.stderr,
                  flush=True)
        raise
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(1)


def _run_in_worker(function, task):
    return function(_WORKER_FEATURIZER, task)


@contextlib.contextmanager
def _executor(jobs, params=None):
    """Serial (``jobs == 1``) or parallel execution; with `params`, tasks get
    its featurizer."""
    if jobs == 1:
        executor = _Serial(params.featurizer if params else None)
    elif params is not None:
        executor = _Parallel(jobs, params.params_file, params.name,
                             params.series)
    else:
        executor = _Parallel(jobs)
    cancel = True
    try:
        yield executor
        cancel = False
    finally:
        executor.shutdown(cancel=cancel)


class _Scheduler:
    """Submits tagged tasks and yields their results in submission order of
    the finished ones."""

    def __init__(self, executor):
        self.executor = executor
        self.pending = {}
        self.order = itertools.count()

    def submit(self, function, task, tag):
        future = self.executor.submit(function, task)
        self.pending[future] = (next(self.order), tag)

    def results(self):
        """Yield ``(tag, result, error)``; tasks may be submitted while
        iterating."""
        while self.pending:
            done, _ = concurrent.futures.wait(
                self.pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: self.pending[f][0]):
                _, tag = self.pending.pop(future)
                error = future.exception()
                yield tag, None if error else future.result(), error


def _stop_requested(stop):
    """Raise `aimmd.core.series.RefillStopped` if `stop` says so."""
    if stop is not None and stop():
        raise _series.RefillStopped('stopped on request')


def _error_text(error):
    if isinstance(error, concurrent.futures.BrokenExecutor):
        return f'a worker process died: {error}'
    return f'{type(error).__name__}: {error}'


def _new_seconds():
    return dict.fromkeys(SECONDS, 0.)


def _add_seconds(total, seconds):
    for key, value in seconds.items():
        total[key] = total.get(key, 0.) + value


# ----------------------------------------------------------------------
# prefill

def _plan_prefill(featurizer, task):
    """Count the frames of a trajectory, find the rows to compute and create
    the temporary series file (with the rows to keep, for only_missing)."""
    started = time.perf_counter()
    featurizer = _system(featurizer, task['system_id'])
    target, temp = task['series_file'], task['temporary']
    with _open_reader(task['trajectory'], featurizer) as reader:
        n_frames, unreadable = _readable_frames(reader)
    plan = dict(frames=n_frames, unreadable_frames=unreadable, kept=0,
                extra_rows=0, identity=_identity(target))
    try:
        if plan['identity'] is not None and task['only_missing']:
            with _lock(target), open(target, 'rb') as file:
                plan['identity'] = _identity(target)
                info = _series_info(file, target, featurizer.width)
                present = _present_rows(file, info, n_frames)
                todo = np.flatnonzero(~present)
                plan.update(kept=int(present.sum()),
                            extra_rows=max(0, info.n_rows - n_frames))
                if len(todo):
                    _create(temp, max(n_frames, info.n_rows), info.width)
                    with open(temp, 'r+b') as out:
                        for start in range(0, info.n_rows, _BLOCK_ROWS):
                            stop = min(start + _BLOCK_ROWS, info.n_rows)
                            out.seek(HEADER_BYTES + start * info.width * 4)
                            out.write(_read_block(file, info, start,
                                                  stop).tobytes())
        else:
            todo = np.arange(n_frames)
            _create(temp, n_frames, featurizer.width)
    except BaseException:
        _discard(temp)
        raise
    plan['todo'] = todo
    plan['seconds'] = {'plan': time.perf_counter() - started}
    return plan


def _featurize_into(featurizer, rows, misses):
    """Featurize `misses`, ``(index into rows, _Frame)``, into `rows`."""
    if misses:
        computed = featurizer.descriptors_function(
            [frame for _, frame in misses])
        rows[[index for index, _ in misses]] = computed


def _prefill_chunk(featurizer, task):
    """Compute the rows of some frames of a trajectory and write them into
    its temporary series file."""
    featurizer = _system(featurizer, task['system_id'])
    frames = task['frames']
    seconds = dict.fromkeys(('read', 'hash', 'lookup', 'graph', 'featurize',
                             'write'), 0.)
    counts = dict(db_hits=0, db_unusable=0, recomputed=0)
    rows = np.zeros((len(frames), featurizer.width), dtype=np.float32)
    extracted = []
    connection = None
    if task['db']:
        from ..graph_utils import _decode, get_stable_hash
        connection = _connect(task['db'])
    clock = time.perf_counter
    misses = []
    n_read = 0
    try:
        with _open_reader(task['trajectory'], featurizer) as reader:
            first, last = int(frames[0]), int(frames[-1])
            if last - first + 1 == len(frames):
                selection = reader[first:last + 1]
            else:
                selection = reader[frames.tolist()]
            before = clock()
            for index, ts in enumerate(selection):
                n_read += 1
                now = clock()
                seconds['read'] += now - before
                if connection is not None:
                    key = get_stable_hash(ts.positions.ravel().copy())
                    hashed = clock()
                    found = connection.execute(
                        'SELECT data FROM graphs_cache WHERE key = ?',
                        (key,)).fetchone()
                    looked_up = clock()
                    seconds['hash'] += hashed - now
                    seconds['lookup'] += looked_up - hashed
                    if found is not None:
                        try:
                            rows[index] = featurizer.row_from_graph(
                                _decode(found[0]))
                            counts['db_hits'] += 1
                            extracted.append(int(frames[index]))
                            found = True
                        except ValueError:
                            # another graph definition, or more than n_max
                            # nodes: featurize (and report) the frame
                            counts['db_unusable'] += 1
                            found = None
                        seconds['graph'] += clock() - looked_up
                    if found:
                        before = clock()
                        continue
                misses.append((index, _Frame(ts.time, ts.positions.copy())))
                if len(misses) == _FEATURIZE_BATCH:
                    start = clock()
                    _featurize_into(featurizer, rows, misses)
                    seconds['featurize'] += clock() - start
                    counts['recomputed'] += len(misses)
                    misses = []
                before = clock()
        if n_read != len(frames):
            # an iteration can end early on a frame it cannot read: never
            # leave its row empty, as if it did not fit the layout
            raise OSError(f'{task["trajectory"]}: read {n_read} of the '
                          f'{len(frames)} frames {first}..{last}')
        start = clock()
        _featurize_into(featurizer, rows, misses)
        seconds['featurize'] += clock() - start
        counts['recomputed'] += len(misses)
    finally:
        if connection is not None:
            connection.close()
    start = clock()
    _write_rows(task['temporary'], frames, rows)
    seconds['write'] = clock() - start
    return dict(counts, computed=len(frames), seconds=seconds,
                empty_rows=int((rows[:, 0] == 0).sum()),
                extracted=np.asarray(extracted, dtype=np.int64))


def _finalize_prefill(featurizer, task):
    """Verify sampled rows and install the temporary series file (or, with
    nothing computed, verify the series file). The rows are drawn from
    ``task['candidates']`` (default: every frame); with ``task['defer']``
    a verified temporary file is kept for the caller to install (status
    ``'verified'``)."""
    featurizer = _system(featurizer, task['system_id'])
    target, temp = task['series_file'], task['temporary']
    seconds = {}
    clock = time.perf_counter
    result = dict(verified=0, verify_mismatches=0, mismatched_frames=[],
                  seconds=seconds)
    keep = False
    try:
        start = clock()
        candidates = task.get('candidates')
        frames = _sample(task['trajectory'],
                         np.arange(task['frames']) if candidates is None
                         else candidates, task['verify'], task['seed'])
        if len(frames):
            if task['install']:
                stored = _read_rows(temp, frames, featurizer.width)
            else:
                with _lock(target):
                    stored = _read_rows(target, frames, featurizer.width)
            mismatched = _check_rows(featurizer, task['trajectory'], frames,
                                     stored)
            result.update(verified=len(frames),
                          verify_mismatches=len(mismatched),
                          mismatched_frames=mismatched[:_MAX_LISTED])
        seconds['verify'] = clock() - start
        if result['verify_mismatches']:
            result['status'] = 'mismatch'
            return result
        if not task['install']:
            result['status'] = 'complete'
            return result
        if task.get('defer'):
            keep = True
            result['status'] = 'verified'
            return result
        start = clock()
        _install(temp, target, task['identity'])
        seconds['install'] = clock() - start
        result['status'] = 'written'
        return result
    finally:
        if not keep:
            _discard(temp)


def _install(temp, target, identity):
    """Replace the series file `target` by its complete temporary file,
    under its lock, unless it changed since it was planned (`identity`)."""
    _fsync(temp)
    with _lock(target):
        if _identity(target) != identity:
            raise RuntimeError(
                f'{target} changed while it was prefilled (is a job '
                f'running on this run?); it was left as it is')
        os.replace(temp, target)


def _file_entry(trajectory, system_id, series):
    return dict(trajectory=trajectory, system_id=system_id,
                series_file=series_file(trajectory, series), status='pending',
                error=None, frames=0, unreadable_frames=0, computed=0, kept=0,
                extra_rows=0,
                db_hits=0, db_unusable=0, recomputed=0, empty_rows=0,
                verified=0, verify_mismatches=0, mismatched_frames=[],
                seconds=_new_seconds())


def _merge(entry, result):
    """Add the counts of a task result to a file entry."""
    for key, value in result.items():
        if key == 'seconds':
            _add_seconds(entry['seconds'], value)
        elif key in ('computed', 'db_hits', 'db_unusable', 'recomputed',
                     'empty_rows'):
            entry[key] += value
        elif key in ('frames', 'unreadable_frames', 'kept', 'extra_rows',
                     'verified', 'verify_mismatches', 'mismatched_frames',
                     'status'):
            entry[key] = value


def prefill(params, runs, db=None, jobs=1, verify=None, only_missing=False,
            chunk_frames=DEFAULT_CHUNK_FRAMES, seed=0, log=print,
            trajectories=None, progress=None, stop=None, strict_db=False):
    """Write the node-table series of every trajectory of AIMMD runs.

    Parameters
    ----------
    params : ParamsFeaturizer
        The featurizer (`load_featurizer`).
    runs : list of str
        Run folders.
    db : list of str, optional
        Graph caches of the coordinate-descriptor mode to extract rows from
        (``PATH``, or ``SYSTEM_ID=PATH`` per system of a multi-system run).
        Without, every row is featurized from the coordinates.
    jobs : int, default=1
        Processes.
    verify : int, optional
        Frames per trajectory to featurize directly and compare bit for bit
        with the rows (a mismatching file is not installed). By default
        `DEFAULT_DB_VERIFY` with `db` (a graph cache of other settings is
        caught: its keys hash coordinates only), else 0.
    only_missing : bool, default=False
        Compute only rows that are missing (beyond the end of the series
        file, or no file) or zero; keep the others. Without it, every row is
        computed and the series file replaced.
    chunk_frames : int
        Frames per task.
    seed : int, default=0
        Seed of the frames drawn for `verify`.
    log : callable, default=print
        Called with each line of the progress report.
    trajectories : list of tuple, optional
        ``(trajectory, system_id)`` of the runs to prefill (default: every
        trajectory of the runs, `find_trajectories`).
    progress : callable, optional
        Called with a trajectory's entry of the report whenever one of its
        tasks has finished.
    stop : callable, optional
        A stop request (the refill of a job), polled before every task
        (chunk of frames) and after every result: when it returns true the
        command ends with `aimmd.core.series.RefillStopped`; the pending
        tasks are cancelled and their temporary files removed, the files
        already written stay.
    strict_db : bool, default=False
        With `db` (the refill of a job): draw the `verify` frames of a
        trajectory from those whose rows came from the graph cache (not
        from every frame), and trust the cache only while it gives no wrong
        row. Files with rows from the cache are installed once every
        trajectory was checked; after a trajectory whose rows from the cache
        mismatch, the trajectories not extracted yet are featurized and
        those with rows from the cache are not installed (status
        ``'failed'``, for the caller to featurize), nor are they when
        processes died (``broken_pool``: some trajectories were not
        checked).

    Returns
    -------
    dict
        The report: settings, ``files`` (one entry per trajectory with its
        ``status``: ``'written'``, ``'complete'`` (nothing to compute),
        ``'mismatch'`` or ``'failed'``, its counts and timings), ``totals``,
        ``seconds``, ``ms_per_frame``, ``wall_seconds``, ``ok`` and
        ``broken_pool`` (the processes could not start or died).

    Raises
    ------
    UsageError
        If a run or graph cache is missing, no trajectory is found, or
        another node-table command runs on a run.
    aimmd.core.series.RefillStopped
        If `stop` asked to stop.
    """
    started = time.perf_counter()
    featurizer = params.featurizer
    runs = _run_folders(runs)
    databases = _databases(db, featurizer)
    if verify is None:
        verify = DEFAULT_DB_VERIFY if databases else 0
    if trajectories is None:
        found = find_trajectories(runs, featurizer)
    else:
        found = sorted((os.path.abspath(trajectory), system_id)
                       for trajectory, system_id in trajectories)
    if not found:
        raise UsageError(f'no trajectories with a states series '
                         f'(*{STATES_SUFFIX}) in {runs}')
    files = [_file_entry(trajectory, system_id, featurizer.series)
             for trajectory, system_id in found]
    report = dict(command='prefill', params=params.params_file,
                  featurizer=params.name, series=featurizer.series,
                  runs=runs, db=sorted(set(databases.values())), jobs=jobs,
                  verify=verify, only_missing=only_missing,
                  chunk_frames=chunk_frames, seed=seed, files=files)
    log(f'prefill {featurizer.series} ({params.name} of '
        f'{params.params_file}): {len(files)} trajectories in '
        f'{len(runs)} run(s), {jobs} process(es), '
        + (f'extracting from {", ".join(report["db"])}' if databases
           else 'featurizing every frame')
        + (', only missing rows' if only_missing else ''))
    for note in _database_notes(databases):
        log(note)

    remaining = {}
    broken = False
    strict = bool(databases) and strict_db
    distrusted = None           # the trajectory whose cached rows mismatched
    deferred = []               # verified, with rows from the cache
    with _run_locks(runs), _executor(jobs, params) as executor:
        scheduler = _Scheduler(executor)

        def submit(function, index, **task):
            _stop_requested(stop)
            entry = files[index]
            task.update(trajectory=entry['trajectory'],
                        system_id=entry['system_id'],
                        series_file=entry['series_file'],
                        temporary=_temporary(entry['series_file']))
            scheduler.submit(function, task, (function, index))

        try:
            for index in range(len(files)):
                submit(_plan_prefill, index, only_missing=only_missing)
            for (function, index), result, error in scheduler.results():
                _stop_requested(stop)
                entry = files[index]
                if error is None:
                    if function is _prefill_chunk:
                        entry.setdefault('_extracted', []).append(
                            result.pop('extracted'))
                    _merge(entry, result)
                else:
                    entry['status'] = 'failed'
                    entry['error'] = entry['error'] or _error_text(error)
                    broken |= isinstance(
                        error, concurrent.futures.BrokenExecutor)
                if function is _plan_prefill and error is None:
                    # one task per chunk of the frames to compute
                    todo = result['todo']
                    remaining[index] = 0
                    for start in range(0, len(todo), chunk_frames):
                        submit(_prefill_chunk, index,
                               frames=todo[start:start + chunk_frames],
                               db=None if distrusted else databases.get(
                                   entry['system_id'], databases.get(None)))
                        remaining[index] += 1
                    # no temporary file when only_missing finds nothing
                    entry['_install'] = bool(len(todo)) or not (
                        only_missing and result['identity'] is not None)
                    entry['_identity'] = result['identity']
                elif function is _prefill_chunk:
                    remaining[index] -= 1
                if function is _finalize_prefill and strict:
                    if (entry['status'] == 'mismatch' and entry['db_hits']
                            and distrusted is None):
                        distrusted = entry['trajectory']
                        log(f'the graph cache gave rows that differ from a '
                            f'direct featurization for '
                            f'{_display(distrusted)}: no other trajectory of '
                            f'this prefill takes rows from it')
                    if entry['status'] == 'verified':
                        deferred.append(index)
                        continue                    # installed at the end
                if (function is _finalize_prefill
                        or remaining.get(index) is None):  # or a failed plan
                    log(_prefill_line(entry))
                elif not remaining[index]:                 # every chunk done
                    if entry['status'] == 'failed':
                        _discard(_temporary(entry['series_file']))
                        log(_prefill_line(entry))
                    else:
                        extracted = entry.pop('_extracted', [])
                        submit(_finalize_prefill, index,
                               frames=entry['frames'],
                               identity=entry['_identity'],
                               install=entry['_install'], verify=verify,
                               seed=seed,
                               candidates=np.sort(np.concatenate(
                                   extracted or [np.zeros(0, np.int64)]))
                               if strict else None,
                               defer=strict and bool(entry['db_hits']))
                if progress is not None:
                    progress(entry)
            for index in deferred:          # strict_db: after every check
                entry = files[index]
                temp = _temporary(entry['series_file'])
                if distrusted is not None:
                    reason = (f'the graph cache gave rows that differ from '
                              f'a direct featurization for '
                              f'{_display(distrusted)}')
                elif broken:
                    # the trajectories of the dead processes were not
                    # checked: their rows from the cache may be wrong
                    reason = ('processes died before every trajectory was '
                              'checked against a direct featurization')
                else:
                    reason = None
                if reason is not None:
                    entry.update(status='failed',
                                 error=f'not installed: {reason}')
                    _discard(temp)
                else:
                    try:
                        _install(temp, entry['series_file'],
                                 entry['_identity'])
                        entry['status'] = 'written'
                    except Exception as error:      # noqa: BLE001
                        entry.update(status='failed',
                                     error=_error_text(error))
                        _discard(temp)
                log(_prefill_line(entry))
                if progress is not None:
                    progress(entry)
        finally:
            for entry in files:
                entry.pop('_identity', None)
                entry.pop('_install', None)
                entry.pop('_extracted', None)
            if any(entry['status'] in ('pending', 'failed', 'verified')
                   for entry in files):
                executor.shutdown(cancel=True)
                for entry in files:
                    if entry['status'] in ('pending', 'failed', 'verified'):
                        _discard(_temporary(entry['series_file']))

    report['wall_seconds'] = time.perf_counter() - started
    report['broken_pool'] = broken
    _prefill_summary(report)
    for line in _prefill_summary_lines(report):
        log(line)
    return report


def _display(path):
    """A path relative to the working directory, if that is shorter."""
    relative = os.path.relpath(path)
    return relative if len(relative) < len(path) else path


def _prefill_line(entry):
    line = (f'{entry["status"]:9s} {_display(entry["trajectory"])}: '
            f'{entry["frames"]} frames')
    if entry['status'] != 'failed' or entry['computed']:
        line += (f', {entry["computed"]} computed ({entry["db_hits"]} from '
                 f'the graph cache, {entry["recomputed"]} featurized)')
    if entry['kept']:
        line += f', {entry["kept"]} kept'
    if entry['unreadable_frames']:
        line += _unreadable_text(entry['unreadable_frames'])
    if entry['verified']:
        line += (f', {entry["verified"]} verified '
                 f'({entry["verify_mismatches"]} mismatched')
        if entry['mismatched_frames']:
            line += f': frames {entry["mismatched_frames"]}'
        line += ')'
    if entry['empty_rows']:
        line += f', {entry["empty_rows"]} EMPTY rows'
    if entry['error']:
        line += f'; ERROR: {entry["error"]}'
    return line


def _unreadable_text(count):
    return (f' (+{count} unreadable frame(s) at the end, left out as AIMMD '
            f'does)')


def _prefill_summary(report):
    files = report['files']
    totals = {key: sum(entry[key] for entry in files)
              for key in ('frames', 'unreadable_frames', 'computed', 'kept',
                          'db_hits', 'db_unusable', 'recomputed',
                          'empty_rows', 'verified', 'verify_mismatches')}
    totals['trajectories'] = len(files)
    for status in ('written', 'complete', 'mismatch', 'failed'):
        totals[status] = sum(entry['status'] == status for entry in files)
    seconds = _new_seconds()
    for entry in files:
        _add_seconds(seconds, entry['seconds'])
    looked_up = totals['db_hits'] + totals['db_unusable']
    if report['db']:
        looked_up = totals['computed']
    per_frame = {
        'read': _ms(seconds['read'], totals['computed']),
        'extract': _ms(seconds['hash'] + seconds['lookup'] + seconds['graph'],
                       looked_up),
        'hash': _ms(seconds['hash'], looked_up),
        'lookup': _ms(seconds['lookup'], looked_up),
        'graph': _ms(seconds['graph'], totals['db_hits'] +
                     totals['db_unusable']),
        'recompute': _ms(seconds['featurize'], totals['recomputed']),
        'write': _ms(seconds['write'], totals['computed']),
        'verify': _ms(seconds['verify'], totals['verified']),
        'task_total': _ms(sum(seconds.values()), totals['computed']),
    }
    report.update(totals=totals, seconds=seconds, ms_per_frame=per_frame)
    report['ok'] = not (totals['failed'] or totals['mismatch'] or
                        totals['empty_rows'])


def _unreadable_note(count):
    return (f'NOTE: {count} frame(s) at the end of trajectories cannot be '
            f'read (cut short, e.g. by a job killed while it wrote). AIMMD '
            f'reads only the frames before them; they get no rows.')


def _ms(seconds, count):
    return 1e3 * seconds / count if count else None


def _format_ms(value):
    return 'n/a' if value is None else f'{value:.2f} ms'


def _prefill_summary_lines(report):
    totals, per_frame = report['totals'], report['ms_per_frame']
    wall = report['wall_seconds']
    lines = [
        f'prefill: {totals["trajectories"]} trajectories, {totals["frames"]} '
        f'frames: {totals["computed"]} computed ({totals["db_hits"]} from the '
        f'graph cache, {totals["recomputed"]} featurized'
        + (f', {totals["db_unusable"]} cached graphs unusable'
           if totals['db_unusable'] else '')
        + f'), {totals["kept"]} kept; {totals["written"]} files written, '
        f'{totals["complete"]} complete, {totals["mismatch"]} mismatched, '
        f'{totals["failed"]} failed; {wall:.1f} s wall with {report["jobs"]} '
        f'process(es)'
        + (f' ({totals["computed"] / wall:.0f} frames/s)' if wall > 0 and
           totals['computed'] else ''),
        f'  per frame (task time): read {_format_ms(per_frame["read"])}, '
        f'extract {_format_ms(per_frame["extract"])} (hash '
        f'{_format_ms(per_frame["hash"])}, lookup '
        f'{_format_ms(per_frame["lookup"])}, graph to row '
        f'{_format_ms(per_frame["graph"])}), featurize '
        f'{_format_ms(per_frame["recompute"])}, write '
        f'{_format_ms(per_frame["write"])}',
    ]
    if report['verify']:
        lines.append(f'  verify: {totals["verified"]} frames featurized '
                     f'directly, {totals["verify_mismatches"]} mismatched')
    if totals['unreadable_frames']:
        lines.append(_unreadable_note(totals['unreadable_frames']))
    if totals['empty_rows']:
        lines.append(
            f'ERROR: {totals["empty_rows"]} row(s) are empty: their frames do '
            f'not fit the layout (more than n_max graph nodes, or atom types '
            f'outside atom_types; see the ERROR lines above). Training stops '
            f'on them. Widen the rows with python -m aimmd.network.nodetables '
            f'repack --n-max N, set n_max=N in the params file (its series '
            f'is then the new one) and run prefill --only-missing.')
    if totals['verify_mismatches']:
        # where the differing rows can come from, by the options of the run
        causes = []
        if report['only_missing']:
            causes.append('rows kept from a series file may be stale or '
                          'corrupted: rerun prefill without --only-missing '
                          'for those runs')
        if report['db']:
            causes.append('the graph cache may come from other settings: '
                          'rerun without --db for those runs')
        lines.append(
            'ERROR: rows differ from a direct featurization; mismatching '
            'files were not installed'
            + (f' ({"; ".join(causes)})' if causes else '') + '.')
    if totals['failed']:
        lines.append(f'ERROR: {totals["failed"]} file(s) failed; their series '
                     f'files were left as they were')
    lines.append('OK' if report['ok'] else 'FAILED')
    return lines


# ----------------------------------------------------------------------
# repack

def _repack_file(featurizer, task):
    """Rewrite one series file for another capacity."""
    start = time.perf_counter()
    source, target = task['source_file'], task['series_file']
    identity = _identity(target)
    if identity is not None and not task['overwrite']:
        return dict(status='exists', rows=0, empty_rows=0,
                    seconds={'write': time.perf_counter() - start})
    temp = _temporary(target)
    width = 2 + 4 * task['n_max']
    empty = 0
    try:
        with _lock(source), open(source, 'rb') as file:
            info = _series_info(file, source, task['source_width'])
            _create(temp, info.n_rows, width)
            with open(temp, 'r+b') as out:
                for begin in range(0, info.n_rows, _BLOCK_ROWS):
                    end = min(begin + _BLOCK_ROWS, info.n_rows)
                    rows = repack_rows(_read_block(file, info, begin, end),
                                       task['n_max'])
                    empty += int((rows[:, 0] == 0).sum())
                    out.seek(HEADER_BYTES + begin * width * 4)
                    out.write(rows.tobytes())
                out.flush()
                os.fsync(out.fileno())
        with _lock(target):
            if _identity(target) != identity:
                raise RuntimeError(f'{target} changed while it was repacked; '
                                   f'it was left as it is')
            os.replace(temp, target)
    finally:
        _discard(temp)
    return dict(status='written', rows=info.n_rows, empty_rows=empty,
                seconds={'write': time.perf_counter() - start})


def _source_n_max(values, featurizer):
    """The ``n_max`` of the existing series, for ``featurizer.with_n_max``.

    `values` is None, an int, or a list of ``M`` (the default of every
    system) and, for a multi-system featurizer, ``SYSTEM_ID=M`` (one
    system). Returns None, an int, or ``{system_id: M}``.
    """
    if values is None or isinstance(values, int):
        return values
    system_ids = getattr(featurizer, 'system_ids', None)
    if isinstance(values, dict):
        if system_ids is None:
            raise UsageError(f'n_max per system {values!r} is for a '
                             f'multi-system featurizer')
        values = [f'{system_id}={n_max}'
                  for system_id, n_max in values.items()]
    default, per_system = None, {}
    for value in values:
        system_id, separator, text = str(value).rpartition('=')
        try:
            n_max = int(text)
        except ValueError:
            n_max = 0
        if n_max < 1:
            raise UsageError(f'--from-n-max {value!r}: the n_max must be a '
                             f'positive integer')
        if not separator:
            if default is not None:
                raise UsageError('--from-n-max: more than one default M')
            default = n_max
        elif system_ids is None:
            raise UsageError(f'--from-n-max {value!r}: SYSTEM_ID=M is for a '
                             f'multi-system featurizer; give M alone')
        elif system_id not in system_ids:
            raise UsageError(f'--from-n-max {value!r}: no system '
                             f'{system_id!r}; known: {system_ids}')
        elif system_id in per_system:
            raise UsageError(f'--from-n-max: system {system_id!r} given '
                             f'twice')
        else:
            per_system[system_id] = n_max
    if not per_system:
        return default
    if default is not None:
        per_system = {**dict.fromkeys(system_ids, default), **per_system}
    return per_system


def repack(params, runs, n_max, from_n_max=None, jobs=1, overwrite=False,
           log=print, trajectories=None, progress=None, stop=None):
    """Rewrite node-table series files for another row capacity.

    Every series file of the source featurizer in the runs is rewritten into
    a series file of ``featurizer.with_n_max(n_max)`` with `repack_rows`. No
    trajectory is read; zero rows (frames that overflowed) stay zero for
    ``prefill --only-missing`` or the trainer's ledger to fill with the wider
    featurizer. The source files are kept.

    Parameters
    ----------
    params : ParamsFeaturizer
        The featurizer (`load_featurizer`): the source layout, or the target
        one when `from_n_max` gives the source's ``n_max``.
    runs : list of str
        Run folders.
    n_max : int or mapping
        Row capacity of the new series: every system of a multi-system
        featurizer, or ``{system_id: n_max}`` per system (the others keep
        theirs).
    from_n_max : int, list or mapping, optional
        Row capacity of the existing series, if the params file already holds
        the new one: an int, or a list of ``M`` (every system not named) and,
        for a multi-system featurizer, ``SYSTEM_ID=M`` (that system), or
        ``{system_id: M}``.
    jobs : int, default=1
        Processes.
    overwrite : bool, default=False
        Replace existing files of the new series; by default they are left as
        they are (status ``'exists'``).
    log : callable, default=print
        Called with each line of the progress report.
    trajectories : list of tuple, optional
        ``(trajectory, system_id)`` whose series files to rewrite (default:
        every series file of the source series in the runs).
    progress : callable, optional
        Called with a file's entry of the report when it is done.
    stop : callable, optional
        A stop request, polled before every file and after every result
        (see `prefill`): the files already written stay.

    Returns
    -------
    dict
        The report: ``source_series``, ``series`` (the new name),
        ``files`` with each ``status`` (``'written'``, ``'exists'`` or
        ``'failed'``), ``totals``, ``ok`` and ``broken_pool``.

    Raises
    ------
    UsageError
        If `from_n_max` is invalid, the source and the new series are the
        same, or the runs hold no source series file.
    aimmd.core.series.RefillStopped
        If `stop` asked to stop.
    """
    started = time.perf_counter()
    featurizer = params.featurizer
    runs = _run_folders(runs)
    from_n_max = _source_n_max(from_n_max, featurizer)
    try:
        # copies of the tool: they register nothing
        source = (featurizer if from_n_max is None
                  else featurizer.with_n_max(from_n_max, _register=False))
        target = source.with_n_max(n_max, _register=False)
    except ValueError as error:
        raise UsageError(str(error)) from error
    if target.series == source.series:
        raise UsageError(
            f'the series {source.series} already has n_max={n_max}: nothing '
            f'to repack. If the params file already holds the new n_max, give '
            f'the n_max of the existing series with --from-n-max')
    found = _series_files(runs, source)
    if trajectories is not None:
        wanted = {(os.path.abspath(trajectory), system_id)
                  for trajectory, system_id in trajectories}
        found = [item for item in found if item[1:] in wanted]
    if not found:
        raise UsageError(f'no {source.series} series files in {runs}')
    files = []
    for source_file, trajectory, system_id in found:
        files.append(dict(trajectory=trajectory, system_id=system_id,
                          source_file=source_file,
                          series_file=series_file(trajectory, target.series),
                          status='pending', error=None, rows=0, empty_rows=0,
                          seconds=_new_seconds()))
    report = dict(command='repack', params=params.params_file,
                  featurizer=params.name, source_series=source.series,
                  series=target.series, n_max=n_max, runs=runs, jobs=jobs,
                  overwrite=overwrite, files=files)
    log(f'repack {source.series} -> {target.series} (n_max={n_max}): '
        f'{len(files)} series files in {len(runs)} run(s), {jobs} '
        f'process(es)')

    broken = False
    with _run_locks(runs), _executor(jobs) as executor:
        scheduler = _Scheduler(executor)
        for index, entry in enumerate(files):
            _stop_requested(stop)
            source_system = _system(source, entry['system_id'])
            scheduler.submit(_repack_file, dict(
                source_file=entry['source_file'],
                series_file=entry['series_file'],
                source_width=source_system.width,
                n_max=_system(target, entry['system_id']).n_max,
                overwrite=overwrite), index)
        for index, result, error in scheduler.results():
            _stop_requested(stop)
            entry = files[index]
            if error is not None:
                entry.update(status='failed', error=_error_text(error))
                broken |= isinstance(error,
                                     concurrent.futures.BrokenExecutor)
            else:
                entry.update(status=result['status'], rows=result['rows'],
                             empty_rows=result['empty_rows'])
                _add_seconds(entry['seconds'], result['seconds'])
            log(f'{entry["status"]:9s} {_display(entry["series_file"])}: '
                f'{entry["rows"]} rows'
                + (f', {entry["empty_rows"]} empty' if entry['empty_rows']
                   else '')
                + (f'; ERROR: {entry["error"]}' if entry['error'] else ''))
            if progress is not None:
                progress(entry)

    totals = {key: sum(entry[key] for entry in files)
              for key in ('rows', 'empty_rows')}
    totals['files'] = len(files)
    for status in ('written', 'exists', 'failed'):
        totals[status] = sum(entry['status'] == status for entry in files)
    report.update(totals=totals, wall_seconds=time.perf_counter() - started,
                  ok=not totals['failed'], broken_pool=broken)
    log(f'repack: {totals["written"]} written, {totals["exists"]} already '
        f'there, {totals["failed"]} failed; {totals["rows"]} rows, '
        f'{totals["empty_rows"]} empty; {report["wall_seconds"]:.1f} s')
    log(f'The params file needs n_max={n_max} for this series (with '
        f'descriptors_series = FEATURIZER.series it is {target.series!r})'
        + ('; then fill the empty rows with prefill --only-missing'
           if totals['empty_rows'] else ''))
    log('OK' if report['ok'] else 'FAILED')
    return report


# ----------------------------------------------------------------------
# verify

def _verify_file(featurizer, task):
    """Completeness of one trajectory's series, and sampled rows checked."""
    start = time.perf_counter()
    featurizer = _system(featurizer, task['system_id'])
    target = task['series_file']
    with _open_reader(task['trajectory'], featurizer) as reader:
        n_frames, unreadable = _readable_frames(reader)
    result = dict(frames=n_frames, unreadable_frames=unreadable, rows=0,
                  missing_rows=n_frames,
                  zero_rows=0, extra_rows=0, verified=0, verify_mismatches=0,
                  mismatched_frames=[], status='missing', error=None)
    if os.path.exists(target):
        try:
            with _lock(target), open(target, 'rb') as file:
                info = _series_info(file, target, featurizer.width)
                present = _present_rows(file, info, n_frames)
                frames = _sample(task['trajectory'], np.flatnonzero(present),
                                 task['sample'], task['seed'])
                stored = _read_rows(target, frames, featurizer.width)
        except ValueError as error:
            result.update(status='layout', error=str(error))
        else:
            n_rows = min(info.n_rows, n_frames)
            result.update(rows=info.n_rows,
                          missing_rows=max(0, n_frames - info.n_rows),
                          zero_rows=int(n_rows - present.sum()),
                          extra_rows=max(0, info.n_rows - n_frames))
            mismatched = _check_rows(featurizer, task['trajectory'], frames,
                                     stored)
            result.update(verified=len(frames),
                          verify_mismatches=len(mismatched),
                          mismatched_frames=mismatched[:_MAX_LISTED])
            if mismatched:
                result['status'] = 'mismatch'
            elif result['missing_rows'] or result['zero_rows']:
                result['status'] = 'incomplete'
            else:
                result['status'] = 'complete'
    result['seconds'] = {'verify': time.perf_counter() - start}
    return result


def verify(params, runs, sample=0, jobs=1, seed=0, log=print):
    """Report how complete the node-table series of AIMMD runs are.

    Parameters
    ----------
    params : ParamsFeaturizer
        The featurizer (`load_featurizer`).
    runs : list of str
        Run folders.
    sample : int, default=0
        Filled rows per trajectory to featurize directly and compare bit for
        bit.
    jobs : int, default=1
        Processes.
    seed : int, default=0
        Seed of the sampled frames.
    log : callable, default=print
        Called with each line of the report.

    Returns
    -------
    dict
        The report: ``files`` with each ``status`` (``'complete'``,
        ``'incomplete'`` (missing or zero rows), ``'missing'`` (no series
        file), ``'layout'`` (not rows of this featurizer), ``'mismatch'`` or
        ``'failed'``) and counts (``frames``, ``rows``, ``missing_rows``,
        ``zero_rows``, ``extra_rows``, ``verified``,
        ``verify_mismatches``), ``totals`` and ``ok`` (every file
        complete).
    """
    started = time.perf_counter()
    featurizer = params.featurizer
    runs = _run_folders(runs)
    found = find_trajectories(runs, featurizer)
    if not found:
        raise UsageError(f'no trajectories with a states series '
                         f'(*{STATES_SUFFIX}) in {runs}')
    files = [dict(trajectory=trajectory, system_id=system_id,
                  series_file=series_file(trajectory, featurizer.series),
                  status='pending', error=None)
             for trajectory, system_id in found]
    report = dict(command='verify', params=params.params_file,
                  featurizer=params.name, series=featurizer.series,
                  runs=runs, jobs=jobs, sample=sample, seed=seed, files=files)
    log(f'verify {featurizer.series} ({params.name} of '
        f'{params.params_file}): {len(files)} trajectories in '
        f'{len(runs)} run(s)')
    with _executor(jobs, params) as executor:
        scheduler = _Scheduler(executor)
        for index, entry in enumerate(files):
            scheduler.submit(_verify_file, dict(
                trajectory=entry['trajectory'], system_id=entry['system_id'],
                series_file=entry['series_file'], sample=sample, seed=seed),
                index)
        for index, result, error in scheduler.results():
            entry = files[index]
            if error is not None:
                entry.update(status='failed', error=_error_text(error))
            else:
                entry.update(result)
            if entry['status'] != 'complete' or entry.get(
                    'unreadable_frames'):
                log(_verify_line(entry))

    keys = ('frames', 'unreadable_frames', 'rows', 'missing_rows',
            'zero_rows', 'extra_rows', 'verified', 'verify_mismatches')
    totals = {key: sum(entry.get(key, 0) for entry in files) for key in keys}
    totals['trajectories'] = len(files)
    for status in ('complete', 'incomplete', 'missing', 'layout', 'mismatch',
                   'failed'):
        totals[status] = sum(entry['status'] == status for entry in files)
    report.update(totals=totals, wall_seconds=time.perf_counter() - started,
                  ok=totals['complete'] == len(files))
    log(f'verify: {totals["complete"]} of {len(files)} complete, '
        f'{totals["incomplete"]} incomplete, {totals["missing"]} without '
        f'series file, {totals["layout"]} of another layout, '
        f'{totals["mismatch"]} mismatched, {totals["failed"]} failed; '
        f'{totals["missing_rows"]} missing and {totals["zero_rows"]} zero '
        f'rows of {totals["frames"]} frames'
        + (f'; {totals["verified"]} rows featurized directly, '
           f'{totals["verify_mismatches"]} mismatched' if sample else ''))
    if totals['unreadable_frames']:
        log(_unreadable_note(totals['unreadable_frames']))
    log('OK' if report['ok'] else 'INCOMPLETE')
    return report


def _verify_line(entry):
    line = f'{entry["status"]:10s} {_display(entry["trajectory"])}'
    if entry['status'] in ('incomplete', 'missing', 'mismatch'):
        line += (f': {entry["frames"]} frames, {entry["rows"]} rows, '
                 f'{entry["missing_rows"]} missing, {entry["zero_rows"]} zero')
    if entry.get('mismatched_frames'):
        line += f', mismatched frames {entry["mismatched_frames"]}'
    if entry.get('unreadable_frames'):
        line += _unreadable_text(entry['unreadable_frames'])
    if entry.get('error'):
        line += f'; {entry["error"]}'
    return line
