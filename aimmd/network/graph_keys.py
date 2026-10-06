"""
aimmd.network.graph_keys
========================

Per-frame graph keys in place of per-frame coordinates.

With ``Params.descriptor_cache = 'graphkeys'`` AIMMD keeps, next to every
trajectory, ``<traj>.graphkeys.npy``: an ``(n_frames, 32)`` uint8 array whose
row ``i`` is the graph-cache key of frame ``i`` -- the very key the graph
cache already stores the frame's graph under (:mod:`aimmd.core.graphkey`). An
all-zero row means "not computed", the rule every per-frame series follows.
No ``<traj>.descriptors.npy`` is read or written: coordinates are decoded
from the trajectory only

- at ingestion, to key new frames and build their graphs
  (:class:`GraphKeysFunction`), and
- to repair a frame whose key row is zero or whose graph is missing
  (:func:`repair`, :func:`call_with_repair`, :func:`verify`).

Everything else -- value passes, shooting-point selection, training --
receives key rows, and ``process_descriptors_pyg`` only looks the graphs up
(:mod:`aimmd.network.graph_lookup`).

Contract for the user functions
-------------------------------
``descriptors_function(trajectory)`` still defines the coordinate rows; it
may, but no longer needs to, build graphs as a side effect.
``descriptor_transform`` and ``values_function`` must accept key rows as well
as coordinate rows, as functions built on ``process_descriptors_pyg`` do, and
``descriptor_transform`` returns one graph per row. Both may take a
``system_id`` keyword (multi-system runs); it is forwarded only to functions
that accept it.

A ``GraphCacheMiss`` never leaves an ingestion or a value pass: a miss is
repaired from the trajectory and the call retried once, and a miss that
survives the retry is evaluated from the decoded coordinates, exactly as
without graph keys. Only errors that make a frame unrecoverable (an
unreadable trajectory, say) propagate.

Environment
-----------
``AIMMD_GRAPHKEYS_VERIFY``  ``0``/``off``/``false``/``no`` skips :func:`verify`
                            (default: on)
"""

# external
import os
import time
from typing import NamedTuple
import numpy as np

# aimmd imports
from .._config import MDA_CACHE, NPY_CACHE, print
from ..cache.npy import update_npy
from ..core.graphkey import (KEY_BYTES, graph_keys, keys_to_hex, hex_to_keys,
                             pad_keys)
from ..core.utils import accepts_system_id
from .graph_lookup import (GraphCacheMiss, graph_overlay, graphs_present,
                           capture_connection)


__all__ = ['SERIES', 'GraphKeysFunction', 'KeyedFunction', 'Repair',
           'repair', 'call_with_repair', 'verify', 'verify_enabled',
           'key_file', 'load_keys', 'frame_refs', 'coordinate_rows',
           'repair_stats', 'reset_repair_stats',
           'GraphCacheMiss', 'graph_overlay', 'graphs_present']

#: Name of the per-frame series: ``<traj>.graphkeys.npy``.
SERIES = 'graphkeys'

#: Frames decoded and rebuilt per step of a repair. Bounds the graphs held in
#: memory at once when :func:`verify` repairs a whole trajectory, and makes
#: progress durable: each step writes its key rows.
_REPAIR_CHUNK = 1024

#: The covering-index scan of every key in a graph cache (:func:`verify`).
_KEY_SCAN = 'SELECT key FROM graphs_cache'

_DISABLED = ('0', 'off', 'none', 'false', 'no')

# counters since the last reset_repair_stats(), for the trainer log
_STATS = {'filled': 0, 'repaired': 0, 'stale': 0, 'retries': 0, 'fallbacks': 0}


def repair_stats():
    """Counters of key repairs in this process, since the last reset.

    Returns
    -------
    dict
        ``filled``: frames whose zero key row was computed before a call
        (also counted in ``repaired``);
        ``repaired``: frames repaired (zero rows and missing graphs);
        ``stale``: repaired frames whose non-zero key was wrong;
        ``retries``: calls retried after a ``GraphCacheMiss``;
        ``fallbacks``: calls evaluated from coordinates after the retry
        missed again.
    """
    return dict(_STATS)


def reset_repair_stats():
    """Set every :func:`repair_stats` counter to zero."""
    for name in _STATS:
        _STATS[name] = 0


def _call(function, data, system_id=None):
    """Call ``function(data)``, with ``system_id`` only if it accepts it."""
    if system_id is not None and accepts_system_id(function):
        return function(data, system_id=system_id)
    return function(data)


def _as_keys(keys):
    return np.ascontiguousarray(keys, dtype=np.uint8).reshape(-1, KEY_BYTES)


# ------------------------------------------------------------ key files --
def key_file(fname):
    """``<fname>.graphkeys.npy``: the key file of trajectory ``fname``."""
    from ..path.utils import get_cache_fname   # aimmd.path imports this module
    return get_cache_fname(fname, SERIES)


def load_keys(fnames, locs):
    """Key rows of frames, read from their trajectories' key files.

    Parameters
    ----------
    fnames : array-like of str
        Trajectory file of each frame.
    locs : array-like of int
        Frame index of each frame in its file.

    Returns
    -------
    numpy.ndarray
        ``(n, 32)`` uint8. Frames without a key file, or beyond its end, get
        zero rows ("not computed").

    Raises
    ------
    RuntimeError
        If a key file is not an ``(n, 32)`` uint8 array.
    """
    fnames = np.asarray(fnames).astype(str)
    locs = np.asarray(locs, dtype=np.int64)
    result = np.zeros((len(fnames), KEY_BYTES), dtype=np.uint8)
    for fname in dict.fromkeys(fnames):
        rows = np.flatnonzero(fnames == fname)
        target = key_file(fname)
        length = int(locs[rows].max()) + 1
        stored = pad_keys(NPY_CACHE.get(target, min_length=length), length,
                          target)
        result[rows] = stored[locs[rows]]
    return result


def frame_refs(paths):
    """Trajectory file and frame index of every frame of some paths.

    Parameters
    ----------
    paths : Path, PathEnsemble or iterable of them
        Nested iterables are walked in order.

    Returns
    -------
    tuple of numpy.ndarray
        ``(fnames, locs)``, one entry per frame, in path order.
    """
    fnames, locs = [], []

    def visit(item):
        if hasattr(item, '_fnames') and hasattr(item, '_first'):   # a Path
            if len(item):
                fnames.append(item.filenames)
                locs.append(item.locs)
        else:
            for sub in item:
                visit(sub)

    visit(paths)
    if not fnames:
        return np.array([], dtype=str), np.array([], dtype=np.int64)
    return (np.concatenate(fnames).astype(str),
            np.concatenate(locs).astype(np.int64))


def _frames(fname, locs):
    """Reader over frames ``locs`` of ``fname`` (random access, any order)."""
    locs = np.asarray(locs, dtype=np.int64)
    reader = MDA_CACHE.get(fname, int(locs.max()) + 1)
    if reader is None:
        raise RuntimeError(f'cannot repair graph keys: {fname!r} could not '
                           f'be opened')
    if len(reader) <= locs.max():
        raise RuntimeError(f'cannot repair graph keys: {fname!r} has '
                           f'{len(reader)} readable frame(s), frame '
                           f'{int(locs.max())} was requested')
    return reader[locs]


def coordinate_rows(fnames, locs, keys_function, system_id=None):
    """Descriptor rows of frames, decoded from their trajectories.

    Each distinct frame is decoded once; the rows come back in the order of
    ``fnames``/``locs``.
    """
    fnames = np.asarray(fnames).astype(str)
    locs = np.asarray(locs, dtype=np.int64)
    result = None
    for fname in dict.fromkeys(fnames):
        rows = np.flatnonzero(fnames == fname)
        frames, inverse = np.unique(locs[rows], return_inverse=True)
        data = keys_function.rows(_frames(fname, frames), system_id)
        if result is None:
            result = np.empty((len(fnames),) + data.shape[1:], dtype=data.dtype)
        result[rows] = data[inverse]
    return result


# ------------------------------------------------------------- functions --
class GraphKeysFunction:
    """Reader -> graph keys, making sure every frame's graph is cached.

    The ``descriptors_function`` of graph-key runs
    (``params.compute_descriptors_args``): it decodes the frames into
    coordinate rows with ``descriptors_function``, keys them, and passes only
    the frames whose graph is not cached yet to ``descriptor_transform``,
    which builds and stores them -- writer role (MD workers) or reader role
    (trainer: memo and pending backlog) exactly as for coordinate rows. A
    store that gives up never raises, as before.

    Parameters
    ----------
    descriptors_function : callable
        ``trajectory -> (n, n_descriptors)`` coordinate rows. Graphs it
        builds as a side effect are recognised and not built again.
    descriptor_transform : callable or None
        ``rows -> graphs`` and ``keys -> graphs``, e.g. a wrapper of
        ``graph_utils.process_descriptors_pyg``. With None, frames are keyed
        and no graph is built or checked.

    Notes
    -----
    Presence is asked of the cache the transform itself reads (and only of
    that one; in a multi-system run, of the one it picks for the
    ``system_id``): calling it on an empty key batch reveals the connection
    (:func:`aimmd.network.graph_lookup.capture_connection`). A transform that
    reads no sqlite cache is asked by key instead.
    """

    def __init__(self, descriptors_function, descriptor_transform):
        self.descriptors_function = descriptors_function
        self.descriptor_transform = descriptor_transform
        self._connections = {}

    def __call__(self, reader, system_id=None):
        """Keys of the frames of ``reader``; their graphs are cached after.

        Returns
        -------
        numpy.ndarray
            ``(n_frames, 32)`` uint8.
        """
        # graphs built inside the block (e.g. by descriptors_function itself)
        # count as present, so a store that gave up is not repeated
        with graph_overlay():
            rows = self.rows(reader, system_id)
            keys = graph_keys(rows)
            if self.descriptor_transform is not None and len(keys):
                missing = self.missing(keys, system_id)
                if missing.any():
                    _call(self.descriptor_transform, rows[missing], system_id)
        return keys

    def rows(self, reader, system_id=None):
        """Coordinate rows of ``reader`` (``descriptors_function``)."""
        rows = np.asarray(_call(self.descriptors_function, reader, system_id))
        if rows.size == 0:      # no frames (some functions return (1, 0))
            return rows.reshape(0, rows.shape[-1] if rows.ndim == 2 else 0)
        if rows.ndim != 2:
            raise ValueError(f'descriptors_function must return one row per '
                             f'frame (2-D), got shape {rows.shape}')
        return rows

    def connection(self, system_id=None):
        """The graph-cache connection the transform reads, or None.

        Found once per ``system_id`` by calling the transform on an empty key
        batch and recording the connection its lookup uses.
        """
        if system_id not in self._connections:
            conn = None
            if self.descriptor_transform is not None:
                with capture_connection() as seen:
                    try:
                        _call(self.descriptor_transform,
                              np.zeros((0, KEY_BYTES), dtype=np.uint8),
                              system_id)
                    except Exception:           # noqa: BLE001 - asked by key
                        pass
                conn = seen[0] if seen else None
            self._connections[system_id] = conn
        return self._connections[system_id]

    def missing(self, keys, system_id=None):
        """Which key rows have no cached graph.

        Parameters
        ----------
        keys : array-like
            ``(n, 32)`` uint8 key rows.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` bool: True for all-zero rows and for keys whose graph
            is in neither the open overlays nor the cache.
        """
        keys = _as_keys(keys)
        absent = ~keys.any(axis=1)
        known = ~absent
        if self.descriptor_transform is None or not known.any():
            return absent
        conn = self.connection(system_id)
        if conn is not None:
            absent[known] = ~graphs_present(keys[known], conn)
            return absent
        return absent | self._ask(keys, system_id)

    def _ask(self, keys, system_id):
        """Missing keys, found by calling the transform on key rows.

        A transform may stop at its first missing sub-batch, so the keys it
        reports are set aside and the rest asked again, until it answers.
        """
        hexes = keys_to_hex(keys)
        remaining = list(dict.fromkeys(
            h for h, row in zip(hexes, keys) if row.any()))
        missing = set()
        while remaining:
            try:
                _call(self.descriptor_transform, hex_to_keys(remaining),
                      system_id)
                break
            except GraphCacheMiss as miss:
                reported = set(keys_to_hex(miss.keys)).intersection(remaining)
                if not reported:        # a miss we cannot place: all of them
                    reported = set(remaining)
                missing |= reported
                remaining = [h for h in remaining if h not in reported]
        return np.array([h in missing for h in hexes], dtype=bool)

    def build(self, reader, system_id=None):
        """Keys and graphs of the frames of ``reader``.

        Every graph is built, or loaded if it is cached, and stored if it was
        built (as by ``descriptor_transform`` on coordinate rows).

        Returns
        -------
        tuple
            ``(keys, graphs)``: ``(n_frames, 32)`` uint8 keys and
            ``{hex key: graph}``.
        """
        with graph_overlay() as collected:
            rows = self.rows(reader, system_id)
            keys = graph_keys(rows)
            if self.descriptor_transform is None:
                return keys, {}
            hexes = keys_to_hex(keys)
            todo = [i for i, h in enumerate(hexes) if h not in collected]
            if todo:
                graphs = _call(self.descriptor_transform, rows[todo], system_id)
                # a transform that does not hand its graphs to the overlay:
                # take its own output, one graph per row
                try:
                    if len(graphs) == len(todo):
                        for i, graph in zip(todo, graphs):
                            collected.setdefault(hexes[i], graph)
                except TypeError:
                    pass
        return keys, {h: collected[h] for h in hexes if h in collected}


class KeyedFunction:
    """A function of key rows that knows how to repair them.

    ``params.compute_values_args`` wraps ``values_function`` in this for
    graph-key runs. Calls are forwarded unchanged (``system_id`` only if the
    function accepts it); ``keys_function`` is what ``Path.compute`` and
    ``fit`` use to repair a batch after a ``GraphCacheMiss``
    (:func:`call_with_repair`).

    Parameters
    ----------
    function : callable
        The wrapped function, e.g. ``values_function``.
    keys_function : GraphKeysFunction
        The run's key function.
    """

    def __init__(self, function, keys_function):
        self.function = function
        self.keys_function = keys_function
        self.__name__ = getattr(function, '__name__', type(self).__name__)
        self.__doc__ = getattr(function, '__doc__', None)

    def __call__(self, data, system_id=None, **kwargs):
        if system_id is not None and accepts_system_id(self.function):
            return self.function(data, system_id=system_id, **kwargs)
        return self.function(data, **kwargs)

    def __repr__(self):
        return f'KeyedFunction({self.function!r})'


# ---------------------------------------------------------------- repair --
class Repair(NamedTuple):
    """Result of :func:`repair`."""

    #: The key batch, with the repaired rows replaced.
    keys: np.ndarray
    #: ``{hex key: graph}`` of the repaired frames (empty if not kept).
    graphs: dict
    #: Distinct frames repaired.
    repaired: int
    #: Of those, frames whose non-zero key differed from the recomputed one.
    stale: int


def repair(keys, fnames, locs, keys_function, system_id=None, need=None,
           keep_graphs=True):
    """Rebuild the frames of a key batch that need it, from the trajectory.

    A row needs repair if it is all-zero (never computed) or its graph is not
    cached. Its frame is decoded (random access, each distinct frame once),
    its graph built -- or loaded, if the recomputed key has one -- and its key
    recomputed. Key rows that change (zero or stale) are written back to
    ``<traj>.graphkeys.npy`` under the file lock, one row per distinct frame,
    and the key file is evicted from ``NPY_CACHE``.

    Parameters
    ----------
    keys : array-like
        ``(n, 32)`` uint8 key rows of the batch.
    fnames : array-like of str
        Trajectory file of each row.
    locs : array-like of int
        Frame index of each row in its file.
    keys_function : GraphKeysFunction
        Decodes, keys and builds.
    system_id : hashable, optional
        Forwarded to the user functions that accept it.
    need : array-like of bool, optional
        Rows to repair. Default: ``keys_function.missing(keys)``.
    keep_graphs : bool, default True
        Return the graphs (for the retry's overlay). :func:`verify` repairs
        whole trajectories and does not keep them.

    Returns
    -------
    Repair
        ``(keys, graphs, repaired, stale)``.

    Raises
    ------
    RuntimeError
        If a trajectory cannot be read up to a requested frame.
    """
    keys = np.array(_as_keys(keys), copy=True)
    fnames = np.asarray(fnames).astype(str)
    locs = np.asarray(locs, dtype=np.int64)
    if not len(fnames) == len(locs) == len(keys):
        raise ValueError(f'{len(keys)} key rows, {len(fnames)} file names '
                         f'and {len(locs)} frame indices do not match')
    if need is None:
        need = keys_function.missing(keys, system_id)
    need = np.asarray(need, dtype=bool)

    graphs, repaired, stale = {}, 0, 0
    for fname in dict.fromkeys(fnames[need]):
        rows = np.flatnonzero(need & (fnames == fname))
        frames, first, inverse = np.unique(
            locs[rows], return_index=True, return_inverse=True)
        old = keys[rows[first]]
        new = np.empty_like(old)
        target = key_file(fname)
        for start in range(0, len(frames), _REPAIR_CHUNK):
            part = slice(start, start + _REPAIR_CHUNK)
            built_keys, built = keys_function.build(
                _frames(fname, frames[part]), system_id)
            if len(built_keys) != len(frames[part]):
                raise RuntimeError(
                    f'cannot repair graph keys of {fname!r}: '
                    f'descriptors_function returned {len(built_keys)} rows '
                    f'for {len(frames[part])} frames')
            new[part] = built_keys
            if keep_graphs:
                graphs.update(built)
            changed = (old[part] != new[part]).any(axis=1)
            if changed.any():
                update_npy(target, new[part][changed], frames[part][changed])
                NPY_CACHE.remove(target)
        stale += int((old.any(axis=1) & (old != new).any(axis=1)).sum())
        repaired += len(frames)
        keys[rows] = new[inverse]
    _STATS['repaired'] += repaired
    _STATS['stale'] += stale
    return Repair(keys, graphs, repaired, stale)


def call_with_repair(function, keys, fnames, locs, keys_function,
                     system_id=None):
    """Evaluate a function of key rows, repairing missing graphs.

    1. Zero rows (never computed) are filled from the trajectory first.
    2. ``function(keys)`` is called.
    3. On a ``GraphCacheMiss`` every frame of the batch that misses (not
       only the ones reported: user code may stop at its first missing
       sub-batch) is repaired, and the call retried ONCE, with the graphs
       just built served from an overlay -- so a store that gave up, or an
       evicted memo, cannot make the retry miss.
    4. If it still misses, ``function`` is evaluated on the decoded
       coordinate rows instead (the code path of runs without graph keys,
       identical values), with a warning.

    A ``GraphCacheMiss`` therefore never propagates.

    Parameters
    ----------
    function : callable
        Function of key rows, e.g. ``values_function`` or
        ``descriptor_transform``; it must also accept coordinate rows (step
        4).
    keys : array-like
        ``(n, 32)`` uint8 key rows.
    fnames : array-like of str
        Trajectory file of each row.
    locs : array-like of int
        Frame index of each row in its file.
    keys_function : GraphKeysFunction
        The run's key function.
    system_id : hashable, optional
        Forwarded to the functions that accept it.

    Returns
    -------
    object
        What ``function`` returns.
    """
    keys = _as_keys(keys)
    fnames = np.asarray(fnames).astype(str)
    locs = np.asarray(locs, dtype=np.int64)
    graphs = {}
    zero = ~keys.any(axis=1)
    if zero.any():
        filled = repair(keys, fnames, locs, keys_function, system_id,
                        need=zero)
        keys, graphs = filled.keys, filled.graphs
        _STATS['filled'] += filled.repaired

    with graph_overlay(graphs) as overlay:
        try:
            return _call(function, keys, system_id)
        except GraphCacheMiss as miss:
            reported = miss.keys
        need = keys_function.missing(keys, system_id)
        need |= np.isin(keys_to_hex(keys), keys_to_hex(reported))
        fixed = repair(keys, fnames, locs, keys_function, system_id, need=need)
        keys = fixed.keys
        overlay.update(fixed.graphs)
        _STATS['retries'] += 1
        print(f'... graph keys: repaired {fixed.repaired} frame(s) after a '
              f'graph-cache miss ({len(reported)} reported, {fixed.stale} '
              f'stale key(s))')
        try:
            return _call(function, keys, system_id)
        except GraphCacheMiss as miss:
            still = len(miss.keys)

    _STATS['fallbacks'] += 1
    print(f'!! graph keys: {still} graph(s) still missing after a repair; '
          f'evaluating {len(keys)} frame(s) from trajectory coordinates')
    rows = coordinate_rows(fnames, locs, keys_function, system_id)
    return _call(function, rows, system_id)


# ---------------------------------------------------------------- verify --
def verify_enabled():
    """False if ``AIMMD_GRAPHKEYS_VERIFY`` switches :func:`verify` off."""
    value = os.environ.get('AIMMD_GRAPHKEYS_VERIFY', '1')
    return value.strip().lower() not in _DISABLED


def _absent(keys, keys_function, system_id):
    """Rows of non-zero keys whose graph is not cached, in bulk.

    With a sqlite cache: one covering-index scan of its keys and a set
    membership test, then the few keys not found are asked again of memo,
    pending backlog, replica and database (rows written since the scan).
    """
    hexes = keys_to_hex(keys)
    nonzero = keys.any(axis=1)
    distinct = list(dict.fromkeys(h for h, ok in zip(hexes, nonzero) if ok))
    conn = keys_function.connection(system_id)
    if conn is None:
        missing = keys_function.missing(hex_to_keys(distinct), system_id)
        lost = {h for h, m in zip(distinct, missing) if m}
    else:
        cached = {row[0] for row in conn.execute(_KEY_SCAN)}
        candidates = [h for h in distinct if h not in cached]
        if candidates:
            present = graphs_present(hex_to_keys(candidates), conn)
            lost = {h for h, p in zip(candidates, present) if not p}
        else:
            lost = set()
    return np.array([ok and h in lost for h, ok in zip(hexes, nonzero)],
                    dtype=bool)


def verify(paths, keys_function, system_id=None, repair_missing=True):
    """Make sure every frame of some paths has a key and a cached graph.

    Meant for the trainer, after the round-start key ledger and before fit
    and the value passes: it moves the repairs those would otherwise do
    batch by batch into one bulk step. Graphs built in the reader role go to
    the memo and the pending backlog and are flushed with the round's
    backlog, as usual.

    Parameters
    ----------
    paths : Path, PathEnsemble or iterable of them
        The frames to check.
    keys_function : GraphKeysFunction
        The run's key function (``params.graphkeys_function``).
    system_id : hashable, optional
        The system whose cache to check (multi-system runs).
    repair_missing : bool, default True
        Repair zero rows and missing graphs; otherwise only count them.

    Returns
    -------
    dict or None
        None if ``AIMMD_GRAPHKEYS_VERIFY`` switches the check off. Otherwise
        ``frames`` (checked), ``zero`` (rows never computed), ``missing``
        (non-zero keys without a cached graph), ``repaired`` (distinct
        frames repaired), ``stale`` (repaired frames whose key was wrong)
        and ``seconds``.
    """
    if not verify_enabled():
        return None
    start = time.monotonic()
    fnames, locs = frame_refs(paths)
    keys = load_keys(fnames, locs)
    zero = ~keys.any(axis=1)
    absent = (_absent(keys, keys_function, system_id) if len(keys)
              else np.zeros(0, dtype=bool))
    counts = {'frames': len(keys), 'zero': int(zero.sum()),
              'missing': int(absent.sum()), 'repaired': 0, 'stale': 0}
    need = zero | absent
    if repair_missing and need.any():
        fixed = repair(keys, fnames, locs, keys_function, system_id,
                       need=need, keep_graphs=False)
        counts['repaired'] = fixed.repaired
        counts['stale'] = fixed.stale
    counts['seconds'] = round(time.monotonic() - start, 3)
    return counts
