"""
aimmd.network.graph_keys_cli
============================

Offline tools for runs that cache graph keys
(``Params.descriptor_cache = 'graphkeys'``, :mod:`aimmd.network.graph_keys`)::

    python -m aimmd.network.graph_keys_cli backfill --run RUN [--run RUN ...]
        [--db DB] [-j N] [--verify-npy K] [--only-missing] [--out-root DIR]
        [--report FILE]

``backfill``
    Writes ``<traj>.graphkeys.npy`` for every trajectory of the runs, with
    keys computed from the trajectory itself -- never from
    ``<traj>.descriptors.npy`` -- and reports which of them the graph cache
    ``DB`` holds a graph for. The keys are the very keys the cache stores
    the graphs under, so a campaign switched to graph keys after a backfill
    rebuilds no graph, and its first rounds need not key the ensemble.
    ``--verify-npy K`` compares ``K`` random keys per trajectory with the
    keys of its old descriptor rows: a check that the trajectories decode on
    this machine exactly as they did when the rows were written.
    ``--only-missing`` computes only the rows a key file lacks (no file,
    short, zero rows), so running it again does nothing.

Run it between jobs, while no worker or trainer uses the runs. It only
reads the cache.

Trajectories of a run
---------------------
Every file with the trajectory extension (``--extension``, default ``.xtc``)
under the run folder that has a ``<traj>.states.npy`` beside it: AIMMD
computes the states of every frame it ingests. That covers the initial paths
in ``initialARB/``, the chain paths ``chain*/path*``, the halves of the shots
in flight ``chain*/back|forw`` and the free parts ``free*/*.part*``. The rows
of the states file are the trajectory's ingested frames. A multi-system run
keeps one graph cache per system: give ``--run <run>/<system_id>`` together
with that system's ``--db``.

Keys
----
``backfill`` decodes the frames with the MDAnalysis reader AIMMD uses and
keys the row ``graph_utils.atom_coordinate_descriptors_function`` computes
(all atoms), the ``descriptors_function`` of graph runs. A run whose
``descriptors_function`` computes other rows (a subset of atoms, say) cannot
be backfilled with this tool: almost none of the keys would be in its
cache, which ``backfill`` reports as a failure. Such a run is keyed by its
own trainer once it runs with graph keys. Frame offsets are built once per
trajectory, by one process, in a private temporary folder: nothing but the
key files is written next to the trajectories.

Exit status
-----------
0 if everything checks out. 1 if ``backfill`` could not key a trajectory, a
key differs from the key of an old descriptor row (``--verify-npy``), or
more than 1 % of the keys have no graph in a non-empty cache. 2 on a usage
error.
"""

# external
import argparse
import json
import multiprocessing
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import (FIRST_COMPLETED, Future, ProcessPoolExecutor,
                                wait)
import numpy as np
from filelock import FileLock
from MDAnalysis.coordinates.core import reader as Reader

# aimmd imports
from .._config import print
from ..cache.mda import count_safe_frames
from ..cache.npy import lock_fname
from ..core.graphkey import KEY_BYTES, graph_key, graph_keys, keys_to_hex
from ..core.graphkey import pad_keys
from ..path.utils import get_cache_fname
from .graph_keys import key_file


__all__ = ['trajectories', 'backfill', 'main', 'MISSING_LIMIT']

#: Fraction of keys without a graph, in a cache that holds graphs, above
#: which ``backfill`` fails.
MISSING_LIMIT = 0.01

#: Frames per decode task: what one process decodes and keys in one go.
_CHUNK = 1024

#: Frames decoded into memory at once inside a task (about 44 MB of
#: production rows).
_DECODE_BATCH = 64

#: The covering-index scan of every key of a graph cache.
_KEY_SCAN = 'SELECT key FROM graphs_cache'


# ---------------------------------------------------------- the run folder --
def trajectories(run, extension='.xtc'):
    """Trajectories a run has ingested: files with a states file beside.

    Parameters
    ----------
    run : str
        Run folder (for a multi-system run, one system's folder).
    extension : str, default '.xtc'
        Trajectory file extension.

    Returns
    -------
    list of str
        Sorted paths, ``run`` joined with the path inside it. Hidden files
        and folders are skipped.
    """
    found = []
    for folder, folders, files in os.walk(run):
        folders[:] = [name for name in folders if not name.startswith('.')]
        for name in files:
            if name.startswith('.') or not name.endswith(extension):
                continue
            fname = os.path.join(folder, name)
            if os.path.isfile(get_cache_fname(fname, 'states')):
                found.append(fname)
    return sorted(found)


def _npy_shape(fname):
    """Shape of a ``.npy`` file, from its header."""
    with open(fname, 'rb') as fh:
        major, _ = np.lib.format.read_magic(fh)
        read = (np.lib.format.read_array_header_1_0 if major == 1
                else np.lib.format.read_array_header_2_0)
        return read(fh)[0]


def _ingested(fname):
    """Frames ``fname`` has ingested: the rows of its states file."""
    return int(_npy_shape(get_cache_fname(fname, 'states'))[0])


def _stored(fname):
    """Key rows stored in a key file.

    Returns
    -------
    tuple
        ``(keys, why)``: an ``(n, 32)`` uint8 array and None, or None and
        ``'no key file'`` / ``'not a key file: ...'``.
    """
    if not os.path.exists(fname):
        return None, 'no key file'
    try:
        return pad_keys(np.load(fname, allow_pickle=False), 0, fname), None
    except Exception as exception:                       # noqa: BLE001
        return None, f'not a key file: {exception}'


# -------------------------------------------------------------- the cache --
def _connect(db):
    """A read-only connection to graph cache ``db``."""
    if not os.path.isfile(db):
        raise FileNotFoundError(f'graph cache {db!r} does not exist')
    uri = f'file:{urllib.parse.quote(os.path.abspath(db))}?mode=ro'
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.execute('PRAGMA busy_timeout=60000')
    return conn


def _scan(conn):
    """Every key of the cache, in one covering-index scan."""
    return {row[0] for row in conn.execute(_KEY_SCAN)}


def _db_keys(db):
    """Every key of graph cache ``db`` (read-only) and the seconds it took."""
    start = time.monotonic()
    conn = _connect(db)
    try:
        keys = _scan(conn)
    finally:
        conn.close()
    return keys, time.monotonic() - start


def _missing(keys, cached):
    """``(missing, checked)``: non-zero key rows without a graph, of all."""
    hexes = keys_to_hex(keys[keys.any(axis=1)])
    return sum(h not in cached for h in hexes), len(hexes)


def _missing_problem(missing, checked, cached, db):
    """The failure when too many keys have no graph, or None."""
    if not cached or not checked or missing <= MISSING_LIMIT * checked:
        return None
    return (f'{missing} of {checked} keys ({100 * missing / checked:.2f} %) '
            f'have no graph in {db}, more than {100 * MISSING_LIMIT:g} %: '
            f'were the graphs built from other rows (descriptors_function, '
            f'trajectory decoder), or is it the cache of another run?')


# ------------------------------------------------------------------ tasks --
class _Inline:
    """An executor that runs each task as it is submitted (``-j 1``)."""

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exception:                     # noqa: BLE001
            future.set_exception(exception)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _pool(jobs):
    """``jobs`` worker processes, or :class:`_Inline` for one."""
    if jobs <= 1:
        return _Inline()
    # forked workers inherit the imported modules; spawned ones would import
    # aimmd (and torch) again, about 5 s each
    context = (multiprocessing.get_context('fork')
               if 'fork' in multiprocessing.get_all_start_methods() else None)
    return ProcessPoolExecutor(max_workers=jobs, mp_context=context)


def _default_rows_function():
    """``graph_utils.atom_coordinate_descriptors_function``.

    Imported on use: graph_utils needs the graph stack (torch_geometric).
    """
    from .graph_utils import atom_coordinate_descriptors_function
    return atom_coordinate_descriptors_function


def _link(scratch, index, fname):
    """A symbolic link to ``fname`` in a private folder of ``scratch``.

    MDAnalysis keeps the frame offsets of a trajectory next to the name it
    opens, so opening the link keeps them -- built once, by the index task
    -- out of the run folder, also when it is read-only.
    """
    folder = os.path.join(scratch, str(index))
    os.makedirs(folder)
    link = os.path.join(folder, os.path.basename(fname))
    os.symlink(os.path.abspath(fname), link)
    return link


def _index_task(link):
    """Readable frames of a trajectory; stores its offsets beside ``link``.

    Returns
    -------
    tuple
        ``(frames, seconds)``.
    """
    start = time.perf_counter()
    reader = Reader(link, refresh_offsets=True)
    try:
        frames = count_safe_frames(reader) if len(reader) else 0
    finally:
        reader.close()
    return frames, time.perf_counter() - start


def _keys_task(link, frames, rows_function):
    """Keys of ``frames`` (sorted, distinct) of a trajectory.

    Returns
    -------
    tuple
        ``(keys, cpu seconds, wall seconds)``.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    reader = Reader(link)                     # offsets from the index task
    try:
        keys = np.empty((len(frames), KEY_BYTES), dtype=np.uint8)
        for begin in range(0, len(frames), _DECODE_BATCH):
            part = frames[begin:begin + _DECODE_BATCH]
            first, last = int(part[0]), int(part[-1])
            batch = (reader[first:last + 1] if last - first + 1 == len(part)
                     else reader[part])
            rows = np.asarray(rows_function(batch))
            if len(rows) != len(part):
                raise RuntimeError(f'{len(rows)} descriptor rows for '
                                   f'{len(part)} frames')
            keys[begin:begin + len(part)] = graph_keys(rows)
    finally:
        reader.close()
    return keys, time.process_time() - cpu, time.perf_counter() - wall


def _npy_task(fname, count, seed):
    """Keys of ``count`` random rows of a descriptor file.

    All-zero rows were never computed and are not keyed.

    Returns
    -------
    tuple
        ``(rows, keys, keyed)``: the sorted row indices, their keys and
        whether each was keyed.
    """
    stored = np.load(fname, mmap_mode='r')
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(stored), min(count, len(stored)),
                                replace=False))
    keys = np.zeros((len(picked), KEY_BYTES), dtype=np.uint8)
    keyed = np.zeros(len(picked), dtype=bool)
    for k, i in enumerate(picked):
        row = stored[i]
        if row.any():
            keys[k] = np.frombuffer(graph_key(row), dtype=np.uint8)
            keyed[k] = True
    return picked, keys, keyed


def _store(target, rows, keys):
    """Write rows of a key file, atomically, under its ``.npy`` lock.

    The stored file is read, padded, given the new rows and written to a
    temporary file that then replaces it, all under the lock that
    ``aimmd.cache.npy`` holds: no reader ever sees half a file, and a crash
    leaves the old one.

    Returns
    -------
    tuple
        ``(keys, written, stale)``: the rows now stored, whether the file
        changed, and how many non-zero rows changed.
    """
    folder = os.path.dirname(target) or '.'
    os.makedirs(folder, exist_ok=True)
    temp = os.path.join(folder, f'temp.{os.path.basename(target)}')
    length = int(rows.max()) + 1 if len(rows) else 0
    with FileLock(lock_fname(target), timeout=60.0):
        old, _ = _stored(target)
        new = np.array(pad_keys(old, length, target), copy=True)
        before = new[rows].copy()
        new[rows] = keys
        stale = int((before.any(axis=1) & (before != keys).any(axis=1)).sum())
        if old is not None and np.array_equal(old, new):
            return old, False, 0
        try:
            np.save(temp, new)
            os.replace(temp, target)
        except BaseException:
            if os.path.exists(temp):
                os.remove(temp)
            raise
    return new, True, stale


# --------------------------------------------------------------- backfill --
def _target(fname, run, out_root):
    """Where the key file of trajectory ``fname`` of ``run`` goes."""
    if out_root is None:
        return key_file(fname)
    name = os.path.basename(os.path.normpath(run))
    return os.path.join(out_root, name, os.path.relpath(key_file(fname), run))


def _names(runs, files):
    """Display name of every trajectory: its path inside its run, prefixed
    with the run when there are several."""
    return {fname: (os.path.relpath(fname, run) if len(runs) == 1
                    else os.path.join(run, os.path.relpath(fname, run)))
            for fname, run in files.items()}


class _Backfill:
    """One backfill: what to key, the tasks in flight, what they found.

    Parameters
    ----------
    runs : list of str
        Run folders.
    out_root : str or None
        Where the key files go (see :func:`backfill`).
    only_missing : bool
        Key only the rows a key file lacks.
    extension : str
        Trajectory file extension.
    """

    def __init__(self, runs, out_root, only_missing, extension):
        owner = {fname: run for run in runs
                 for fname in trajectories(run, extension)}
        self.names = _names(runs, owner)
        self.only_missing = only_missing
        #: per trajectory, the record of the report
        self.files = {}
        #: per trajectory to key: target, stored rows, link, frames to key,
        #: their keys, tasks left
        self.plan = {}
        #: per trajectory, the key rows stored at the end
        self.final = {}
        #: per trajectory, the descriptor spot check (or its exception)
        self.spots = {}
        for fname, run in owner.items():
            record = self.files[fname] = {
                'ingested': 0, 'frames': None, 'keys': None, 'keyed': 0,
                'written': False, 'stale': 0, 'zero': 0, 'unreadable': 0,
                'missing': None, 'spot': None, 'index_s': 0.0, 'cpu_s': 0.0,
                'decode_s': 0.0, 'error': None}
            target = _target(fname, run, out_root)
            try:
                record['ingested'] = _ingested(fname)
            except Exception as exception:              # noqa: BLE001
                record['ingested'] = None
                record['error'] = f'cannot read its states file: {exception!r}'
                continue
            stored, _ = _stored(target)
            ingested = record['ingested']
            if (only_missing and stored is not None and len(stored) >= ingested
                    and stored[:ingested].any(axis=1).all()):
                self.final[fname] = stored             # complete: not opened
            else:
                self.plan[fname] = {'target': target, 'stored': stored}

    def run(self, pool, scratch, rows_function, verify_npy, seed, verbose):
        """Index, key and store every planned trajectory; spot-check.

        Every trajectory is indexed by one task; its frames are then keyed
        in chunks of ``_CHUNK`` by as many tasks, and its key file is
        written as soon as the last one is done.
        """
        pending = {}
        for index, fname in enumerate(self.plan):
            link = self.plan[fname]['link'] = _link(scratch, index, fname)
            pending[pool.submit(_index_task, link)] = ('index', fname, None)
        if verify_npy > 0:
            for index, fname in enumerate(self.files):
                descriptors = get_cache_fname(fname, 'descriptors')
                if os.path.isfile(descriptors):
                    future = pool.submit(_npy_task, descriptors, verify_npy,
                                         [seed, index])
                    pending[future] = ('npy', fname, None)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind, fname, part = pending.pop(future)
                record = self.files[fname]
                try:
                    result = future.result()
                except Exception as exception:             # noqa: BLE001
                    if kind == 'npy':                      # keys still count
                        self.spots[fname] = exception
                    elif record['error'] is None:
                        record['error'] = repr(exception)
                    continue
                if kind == 'npy':
                    self.spots[fname] = result
                elif record['error'] is None:
                    if kind == 'index':
                        tasks = self._indexed(fname, result, rows_function)
                        for task, part in tasks:
                            future = pool.submit(*task)
                            pending[future] = ('keys', fname, part)
                    else:
                        self._keyed(fname, result, part)
                    if not self.plan[fname]['left']:
                        self._write(fname, verbose)

    def _indexed(self, fname, result, rows_function):
        """Plan the key tasks of an indexed trajectory."""
        record, state = self.files[fname], self.plan[fname]
        frames, record['index_s'] = result
        record['frames'] = frames
        record['unreadable'] = max(0, record['ingested'] - frames)
        todo = np.arange(frames)
        if self.only_missing:
            have = pad_keys(state['stored'], frames, fname)
            todo = todo[~have[:frames].any(axis=1)]
        state['todo'] = todo
        state['keys'] = np.zeros((len(todo), KEY_BYTES), dtype=np.uint8)
        tasks = [((_keys_task, state['link'], todo[begin:begin + _CHUNK],
                   rows_function), (begin, len(todo[begin:begin + _CHUNK])))
                 for begin in range(0, len(todo), _CHUNK)]
        state['left'] = len(tasks)
        return tasks

    def _keyed(self, fname, result, part):
        """Take the keys of one chunk."""
        record, state = self.files[fname], self.plan[fname]
        keys, cpu, wall = result
        begin, length = part
        state['keys'][begin:begin + length] = keys
        state['left'] -= 1
        record['cpu_s'] += cpu
        record['decode_s'] += wall

    def _write(self, fname, verbose):
        """Write the key file of a trajectory whose chunks are all keyed."""
        record, state = self.files[fname], self.plan[fname]
        record['keyed'] = len(state['todo'])
        if not record['keyed']:
            self.final[fname] = pad_keys(state['stored'], 0, fname)
        else:
            try:
                self.final[fname], record['written'], record['stale'] = (
                    _store(state['target'], state['todo'], state['keys']))
            except Exception as exception:                 # noqa: BLE001
                record['error'] = repr(exception)
                return
        if verbose:
            print(f'  {self.names[fname]}: {record["frames"]} frames, '
                  f'{record["keyed"]} keyed'
                  f'{", written" if record["written"] else ""}')

    def check(self, db, cached):
        """Fill in the checks of every record; returns the problems and the
        number of keys checked against the cache."""
        problems, checked = [], 0
        for fname, record in self.files.items():
            name = self.names[fname]
            if record['error'] is not None:
                if record['ingested'] != 0:   # nothing ingested: no matter
                    problems.append(f'{name}: {record["error"]}')
                continue
            keys = pad_keys(self.final.get(fname), record['ingested'], fname)
            if fname in self.final:
                record['keys'] = len(self.final[fname])
            ingested = keys[:record['ingested']]
            record['zero'] = int((~ingested.any(axis=1)).sum())
            if record['keyed']:
                record['cpu_ms_per_frame'] = (1e3 * record['cpu_s']
                                              / record['keyed'])
            if record['unreadable']:
                problems.append(f'{name}: {record["unreadable"]} ingested '
                                f'frame(s) beyond the {record["frames"]} it '
                                f'can read')
            if db is not None:
                record['missing'], count = _missing(keys, cached)
                checked += count
            spot = self.spots.get(fname)
            if isinstance(spot, Exception):
                record['spot'] = {'checked': 0, 'mismatched': 0, 'skipped': 0,
                                  'error': repr(spot)}
                problems.append(f'{name}: cannot read its descriptor rows: '
                                f'{spot!r}')
            elif spot is not None:
                record['spot'] = _spot(spot, keys)
                if record['spot']['mismatched']:
                    problems.append(
                        f'{name}: {record["spot"]["mismatched"]} of '
                        f'{record["spot"]["checked"]} keys differ from the '
                        f'keys of its descriptor rows: the trajectory does '
                        f'not decode as it did when they were written')
        return problems, checked


def backfill(runs, db=None, jobs=1, verify_npy=0, out_root=None,
             only_missing=False, extension='.xtc', rows_function=None,
             seed=0, verbose=False):
    """Write the key file of every trajectory of some runs.

    See the module docstring for what it does and when to run it.

    Parameters
    ----------
    runs : list of str
        Run folders.
    db : str, optional
        Graph cache to check the keys against (read only).
    jobs : int, default 1
        Processes. Each trajectory is indexed by one of them and its frames
        keyed in chunks of 1024 by all.
    verify_npy : int, default 0
        Keys per trajectory to compare with its ``<traj>.descriptors.npy``
        rows. Only then is a descriptor file opened.
    out_root : str, optional
        Write the key files under ``<out_root>/<run name>/`` instead of next
        to the trajectories.
    only_missing : bool, default False
        Compute only the rows a key file lacks; a trajectory whose key file
        covers its ingested frames is not opened.
    extension : str, default '.xtc'
        Trajectory file extension.
    rows_function : callable, optional
        ``trajectory -> (n, n_descriptors)`` rows to key; default
        ``graph_utils.atom_coordinate_descriptors_function``.
    seed : int, default 0
        Seed of the rows ``verify_npy`` draws.
    verbose : bool, default False
        Print a line per trajectory as it is done.

    Returns
    -------
    dict
        The report: ``files`` (a record per trajectory), ``totals``,
        ``problems`` and ``ok``. Per trajectory: ``ingested`` frames,
        readable ``frames`` (None if it was not opened), rows in the key
        file (``keys``), rows computed (``keyed``), whether the file was
        ``written``, ``stale`` rows replaced, ``zero`` rows left among the
        ingested frames, ingested frames not readable (``unreadable``), keys
        without a graph (``missing``), the descriptor spot check
        (``spot``), the seconds to index it (``index_s``), the CPU and
        summed wall seconds of decoding and keying it (``cpu_s``,
        ``decode_s``; ``cpu_ms_per_frame``), and ``error``.

    Raises
    ------
    FileNotFoundError
        If ``db`` does not exist.
    ValueError
        If ``out_root`` is given for runs of the same name.
    """
    start = time.monotonic()
    if db is not None and not os.path.isfile(db):
        raise FileNotFoundError(f'graph cache {db!r} does not exist')
    if out_root is not None:
        names = [os.path.basename(os.path.normpath(run)) for run in runs]
        if len(set(names)) != len(names):
            raise ValueError(f'--out-root needs runs with distinct names, '
                             f'got {names}')
    job = _Backfill(runs, out_root, only_missing, extension)
    if job.plan and rows_function is None:
        rows_function = _default_rows_function()     # import before forking
    scratch = tempfile.mkdtemp(prefix='aimmd-graphkeys-')
    pool = _pool(jobs)
    keys_start = time.monotonic()
    try:
        job.run(pool, scratch, rows_function, verify_npy, seed, verbose)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)
    keys_wall = time.monotonic() - keys_start

    cached, scan = (set(), 0.0) if db is None else _db_keys(db)
    problems, checked = job.check(db, cached)
    files = job.files
    totals = _totals(files, ('ingested', 'frames', 'keys', 'keyed', 'stale',
                             'zero', 'unreadable', 'index_s', 'cpu_s',
                             'decode_s'))
    spots = [r['spot'] for r in files.values() if r['spot']]
    totals.update(
        files=len(files),
        written=sum(r['written'] for r in files.values()),
        errors=sum(r['error'] is not None for r in files.values()),
        missing=(None if db is None else
                 sum(r['missing'] or 0 for r in files.values())),
        missing_fraction=None, db_keys=None if db is None else len(cached),
        spot_checked=sum(s['checked'] for s in spots),
        spot_mismatched=sum(s['mismatched'] for s in spots),
        spot_skipped=sum(s['skipped'] for s in spots),
        cpu_ms_per_frame=(1e3 * totals['cpu_s'] / totals['keyed']
                          if totals['keyed'] else None),
        keys_wall_s=keys_wall, db_scan_s=scan, jobs=jobs)
    if db is not None:
        totals['missing_fraction'] = (totals['missing'] / checked if checked
                                      else None)
        problem = _missing_problem(totals['missing'], checked, cached, db)
        if problem:
            problems.append(problem)
    totals['wall_s'] = time.monotonic() - start
    return {'command': 'backfill', 'runs': list(runs), 'db': db,
            'options': {'jobs': jobs, 'verify_npy': verify_npy,
                        'out_root': out_root, 'only_missing': only_missing,
                        'extension': extension, 'seed': seed},
            'names': job.names, 'files': files, 'totals': totals,
            'problems': problems, 'ok': not problems}


def _spot(result, keys):
    """Compare the keys of descriptor rows with the stored keys."""
    picked, npy_keys, keyed = result
    inside = picked < len(keys)
    compared = keyed & inside
    differ = (npy_keys[compared] != keys[picked[compared]]).any(axis=1)
    return {'checked': int(compared.sum()),
            'mismatched': int(differ.sum()),
            'skipped': int((~compared).sum())}


def _totals(files, names):
    """Sums of some fields over the records (None counts as 0)."""
    return {name: sum(r[name] or 0 for r in files.values()) for name in names}


# ------------------------------------------------------------ command line --
def _seconds(value):
    return '-' if value is None else f'{value:.2f} s'


def _show_backfill(report):
    names, totals = report['names'], report['totals']
    width = max([len(n) for n in names.values()] + [10])
    print(f'{"trajectory":<{width}}  {"ingested":>8}  {"frames":>8}  '
          f'{"keyed":>8}  {"written":>7}  {"missing":>7}  {"spot":>7}  '
          f'{"index":>8}  {"ms/frame":>8}')
    for fname, r in report['files'].items():
        spot = (f'{r["spot"]["checked"] - r["spot"]["mismatched"]}/'
                f'{r["spot"]["checked"]}' if r['spot'] else '-')
        cpu = (f'{r["cpu_ms_per_frame"]:.2f}' if r.get('cpu_ms_per_frame')
               else '-')
        print(f'{names[fname]:<{width}}  {r["ingested"]:>8}  '
              f'{"-" if r["frames"] is None else r["frames"]:>8}  '
              f'{r["keyed"]:>8}  {"yes" if r["written"] else "no":>7}  '
              f'{"-" if r["missing"] is None else r["missing"]:>7}  '
              f'{spot:>7}  {_seconds(r["index_s"]):>8}  {cpu:>8}'
              + (f'  ERROR {r["error"]}' if r['error'] else ''))
    cpu = totals['cpu_ms_per_frame']
    print(f'\n{totals["files"]} trajectories, {totals["ingested"]} ingested '
          f'frames, {totals["frames"]} read; {totals["keyed"]} keyed '
          f'({totals["stale"]} stale), {totals["written"]} key files '
          f'written, {totals["errors"]} error(s)')
    print(f'timing: index {totals["index_s"]:.1f} s, decode+hash '
          f'{"-" if cpu is None else f"{cpu:.2f}"} ms/frame CPU '
          f'({totals["cpu_s"]:.1f} s CPU, {totals["decode_s"]:.1f} s in '
          f'tasks), keys {totals["keys_wall_s"]:.1f} s wall with '
          f'{totals["jobs"]} job(s), cache scan {totals["db_scan_s"]:.2f} s; '
          f'total {totals["wall_s"]:.1f} s')
    if totals['missing'] is not None:
        fraction = totals['missing_fraction']
        print(f'graph cache: {totals["db_keys"]} graphs; {totals["missing"]} '
              f'key(s) missing'
              + ('' if fraction is None else f' ({100 * fraction:.3f} %)'))
    if report['options']['verify_npy']:
        print(f'descriptor spot check: {totals["spot_checked"]} key(s) '
              f'compared, {totals["spot_mismatched"]} mismatched, '
              f'{totals["spot_skipped"]} skipped (zero rows)')


def _parser():
    parser = argparse.ArgumentParser(
        prog='python -m aimmd.network.graph_keys_cli',
        description='Offline tools for runs with '
                    "descriptor_cache='graphkeys'. Run them between jobs.")
    commands = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--run', action='append', required=True,
                        metavar='RUN', help='run folder (repeatable); for a '
                        "multi-system run, one system's folder")
    common.add_argument('--extension', default='.xtc',
                        help='trajectory extension (default: .xtc)')
    common.add_argument('--report', metavar='FILE',
                        help='also write the report as JSON')
    command = commands.add_parser(
        'backfill', parents=[common],
        help='write <traj>.graphkeys.npy from the trajectories',
        description='Write <traj>.graphkeys.npy for every trajectory with a '
                    'states file, from the trajectory itself, and check the '
                    'keys against the graph cache (read only).')
    command.add_argument('--db', help='graph cache to check the keys against')
    command.add_argument('-j', '--jobs', type=int, default=1,
                         help='processes (default: 1)')
    command.add_argument('--verify-npy', type=int, default=0, metavar='K',
                         help='compare K random keys per trajectory with its '
                              'old descriptor rows')
    command.add_argument('--out-root', metavar='DIR',
                         help='write the key files under DIR/<run name>/')
    command.add_argument('--only-missing', action='store_true',
                         help='compute only the rows key files lack')
    return parser


def main(argv=None):
    """Run the command line; returns the exit status."""
    args = _parser().parse_args(argv)
    try:
        print(f'backfill: {", ".join(args.run)}, {args.jobs} job(s)')
        report = backfill(args.run, db=args.db, jobs=args.jobs,
                          verify_npy=args.verify_npy, out_root=args.out_root,
                          only_missing=args.only_missing,
                          extension=args.extension, verbose=True)
        _show_backfill(report)
    except (FileNotFoundError, ValueError, ImportError) as exception:
        print(f'error: {exception}', file=sys.stderr)
        return 2
    for problem in report['problems']:
        print(f'!! {problem}')
    if args.report:
        with open(args.report, 'w') as fh:
            json.dump(report, fh, indent=1)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
