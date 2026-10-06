"""
aimmd.network.graph_lookup
==========================

Looking graphs up by key: the part of the graph cache that needs no graph stack.

With ``Params.descriptor_cache = 'graphkeys'`` the network consumers
(``descriptor_transform``, ``values_function``) receive graph keys -- an
``(n, 32)`` uint8 array, row ``i`` the 32-byte key of frame ``i``
(:mod:`aimmd.core.graphkey`) -- instead of coordinate rows, and only look the
graphs up. :func:`aimmd.network.graph_utils.process_descriptors_pyg`
dispatches such a batch to :func:`lookup_graphs`.

A lookup never builds a graph: a key without a cached graph, or an all-zero
row (a frame whose key was never computed), raises :class:`GraphCacheMiss`
carrying the missing keys. The AIMMD caller maps them back to frames,
rebuilds the graphs from the trajectory and retries
(:func:`aimmd.network.graph_keys.call_with_repair`).

Lookup order
------------
``overlay -> memo -> pending backlog -> /dev/shm replica -> database``:

- the **overlay** (:func:`graph_overlay`) holds graphs this process has just
  built, for the duration of a block. A repair hands its graphs to the retry
  through it, so they are served even when their store gave up after lock
  contention (writer role: nothing reaches the memo) or the memo has evicted
  them. The coordinate path of ``process_descriptors_pyg`` also adds every
  graph it returns to the open overlays;
- the per-connection **memo** and **pending** backlog (the trainer's reader
  role keeps its new graphs there until the once-per-round flush) and the
  **replica** are those of :mod:`aimmd.network.shm_cache`; the database is the
  connection itself.

Only the connection that is passed is asked. In a multi-system run every
system has its own cache, and a key found in another system's cache must not
count.

Everything here is process-local and needs only numpy, sqlite3 and the
standard library.
"""

# external
import sqlite3
from contextlib import contextmanager
import numpy as np

# aimmd imports
from . import shm_cache
from ..core.graphkey import KEY_BYTES, keys_to_hex


__all__ = ['GraphCacheMiss', 'graph_overlay', 'overlay_get', 'collect_graphs',
           'capture_connection', 'graphs_present', 'cache_watermark',
           'stored_before', 'lookup_graphs']

#: Keys per ``IN (...)`` query: well under SQLite's bound-parameter limit (999
#: before 3.32), and large enough that a 4096-frame batch is a few statements.
_SQL_CHUNK = 500

# open overlays (innermost last) and open connection captures
_OVERLAYS = []
_CAPTURES = []


class GraphCacheMiss(LookupError):
    """Graph keys were looked up whose graphs are not in the graph cache.

    Raised by :func:`lookup_graphs` (and so by ``process_descriptors_pyg`` on
    key rows). Carries the missing keys, not row positions, because user code
    may split a batch into sub-batches: the caller maps the keys back to its
    frames. An all-zero key stands for rows that were never computed.

    Attributes
    ----------
    keys : numpy.ndarray
        ``(m, 32)`` uint8: each missing key once, in order of first
        appearance in the batch.
    """

    def __init__(self, keys):
        self.keys = np.ascontiguousarray(
            keys, dtype=np.uint8).reshape(-1, KEY_BYTES)
        n_zero = int((~self.keys.any(axis=1)).sum())
        message = f'{len(self.keys)} graph(s) missing from the graph cache'
        if n_zero:
            message += ' (including frames whose key was never computed)'
        super().__init__(message)

    def __reduce__(self):
        return (type(self), (self.keys,))


# ---------------------------------------------------------------- overlay --
def _remove(stack, item):
    """Remove ``item`` from ``stack`` by identity (equal dicts are not one)."""
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is item:
            del stack[i]
            return


@contextmanager
def graph_overlay(graphs=None):
    """Serve, and collect, graphs by key for the duration of a block.

    Inside the block every key lookup (:func:`lookup_graphs`,
    :func:`graphs_present`) finds the graphs of ``graphs`` first, and the
    coordinate path of ``process_descriptors_pyg`` adds every graph it
    returns to the overlay.

    This is what keeps a graph that was just built from being lost: a repair
    builds the graphs of a batch and the retry looks them up by key, which
    would miss whenever the store did not land -- a writer gives up after
    ``AIMMD_STORE_RETRY_SECONDS`` of lock contention and nothing reaches its
    memo, and a reader's memo may already have evicted them. Today's
    coordinate path returns the graphs it built in either case, and so must
    the key path.

    Parameters
    ----------
    graphs : dict, optional
        ``{hex key: graph}`` to serve. It is copied; the yielded dict is the
        live overlay.

    Yields
    ------
    dict
        ``{hex key: graph}``: ``graphs`` plus everything collected so far.

    Examples
    --------
    >>> with graph_overlay(repaired_graphs):
    ...     values = values_function(keys)
    """
    overlay = dict(graphs) if graphs else {}
    _OVERLAYS.append(overlay)
    try:
        yield overlay
    finally:
        _remove(_OVERLAYS, overlay)


def overlay_get(hex_key):
    """Graph of ``hex_key`` in the open overlays (innermost first), or None."""
    for overlay in reversed(_OVERLAYS):
        graph = overlay.get(hex_key)
        if graph is not None:
            return graph
    return None


def collect_graphs(hex_keys, graphs):
    """Add graphs to every open overlay; a no-op when none is open.

    Called by the coordinate path of ``process_descriptors_pyg`` with every
    graph it returns, built or loaded. A transform that does not go through
    ``process_descriptors_pyg`` may call it as well.
    """
    if not _OVERLAYS:
        return
    pairs = [(h, g) for h, g in zip(hex_keys, graphs) if g is not None]
    for overlay in _OVERLAYS:
        overlay.update(pairs)


@contextmanager
def capture_connection():
    """Record the cache connections that key lookups use inside a block.

    AIMMD core does not know which cache a ``descriptor_transform`` reads
    from (in a multi-system run it picks one per ``system_id``). Calling the
    transform on an empty key batch inside this block reveals it, so that
    presence can then be asked of that cache alone.

    Yields
    ------
    list
        Connections passed to :func:`lookup_graphs` in the block, in order.
    """
    seen = []
    _CAPTURES.append(seen)
    try:
        yield seen
    finally:
        _remove(_CAPTURES, seen)


# ----------------------------------------------------------------- lookup --
def _as_keys(keys):
    return np.ascontiguousarray(keys, dtype=np.uint8).reshape(-1, KEY_BYTES)


def _replica(conn):
    """The connection's /dev/shm replica, staging it on first use.

    The same rule as ``graph_utils.load_from_sqlite``: a trainer arms its
    caches once per round and the copy happens on the first lookup.
    """
    replica = getattr(conn, '_aimmd_replica', None)
    if replica is None and getattr(conn, '_aimmd_stage_pending', False):
        shm_cache.stage_cache(conn)
        replica = getattr(conn, '_aimmd_replica', None)
    return replica


def _select(db, hex_keys, columns):
    """``SELECT <columns> ... WHERE key IN (...)``, in chunks."""
    rows = []
    for start in range(0, len(hex_keys), _SQL_CHUNK):
        chunk = hex_keys[start:start + _SQL_CHUNK]
        marks = ','.join('?' * len(chunk))
        rows.extend(db.execute(
            f'SELECT {columns} FROM graphs_cache WHERE key IN ({marks})',
            chunk).fetchall())
    return rows


def _from_replica(conn, hex_keys, columns):
    """Rows of ``hex_keys`` found in the replica; detaches a broken one."""
    replica = _replica(conn)
    if replica is None or not hex_keys:
        return None
    try:
        return _select(replica, hex_keys, columns)
    except sqlite3.Error as exc:
        shm_cache.detach(conn, reason=str(exc))
        return None


def graphs_present(keys, conn):
    """Whether the graph of each key is in one graph cache.

    Asks the open overlays, then ``conn``'s memo, pending backlog, replica
    and database, without decoding anything (batched ``IN`` queries on the
    primary key). Only ``conn`` is asked.

    Parameters
    ----------
    keys : array-like
        ``(n, 32)`` uint8 graph keys.
    conn : sqlite3.Connection
        The cache of the caller's system (as returned by
        ``graph_utils.init_db``).

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool. All-zero rows are never present.
    """
    keys = _as_keys(keys)
    present = np.zeros(len(keys), dtype=bool)
    hexes = keys_to_hex(keys)
    rows_of = {}
    for i in np.flatnonzero(keys.any(axis=1)):
        rows_of.setdefault(hexes[i], []).append(i)

    memo = getattr(conn, '_aimmd_memo', None)
    pending = getattr(conn, '_aimmd_pending', None) or {}
    found = {h for h in rows_of
             if overlay_get(h) is not None
             or (memo is not None and memo.get(h) is not None)
             or h in pending}
    rest = [h for h in rows_of if h not in found]
    replica_rows = _from_replica(conn, rest, 'key')
    if replica_rows:
        found.update(row[0] for row in replica_rows)
        rest = [h for h in rest if h not in found]
    if rest:
        found.update(row[0] for row in _select(conn, rest, 'key'))
    for h in found:
        present[rows_of[h]] = True
    return present


def cache_watermark(conn):
    """Highest rowid of a graph cache's database: 0 if it holds no graph.

    The table only grows (``INSERT`` of keys that were looked up and
    missed), so the rows at or below a watermark are the graphs the
    database held when it was taken (see :func:`stored_before`).
    """
    try:
        row = conn.execute('SELECT MAX(rowid) FROM graphs_cache').fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0)


def stored_before(keys, conn, watermark):
    """Whether each key's graph was in the database at a watermark.

    Parameters
    ----------
    keys : array-like
        ``(n, 32)`` uint8 graph keys.
    conn : sqlite3.Connection
        The graph cache.
    watermark : int
        A :func:`cache_watermark` taken earlier.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool: True for keys stored at a rowid up to ``watermark``.
        Only the database is asked (graphs stored since, the memo, the
        pending backlog and the overlays do not count); zero rows never
        count.
    """
    keys = _as_keys(keys)
    hexes = keys_to_hex(keys)
    distinct = list(dict.fromkeys(
        h for h, row in zip(hexes, keys) if row.any()))
    found = set()
    for start in range(0, len(distinct), _SQL_CHUNK):
        chunk = distinct[start:start + _SQL_CHUNK]
        marks = ','.join('?' * len(chunk))
        found.update(row[0] for row in conn.execute(
            f'SELECT key FROM graphs_cache WHERE rowid <= ? '
            f'AND key IN ({marks})', [int(watermark)] + chunk))
    return np.array([h in found for h in hexes], dtype=bool)


def lookup_graphs(keys, conn, decode):
    """Graphs of a batch of keys, looked up and never built.

    Parameters
    ----------
    keys : array-like
        ``(n, 32)`` uint8 graph keys.
    conn : sqlite3.Connection
        The graph cache to read.
    decode : callable
        Turns a stored blob into a graph (``graph_utils._decode``).

    Returns
    -------
    list
        One graph per row, in row order. Every row gets its own decoded
        graph (as a memo or database hit does); an overlay hit is the very
        object the overlay holds.

    Raises
    ------
    GraphCacheMiss
        If a row is all-zero or a key has no graph, with each missing key
        once, in order of first appearance. Nothing is decoded then.

    Notes
    -----
    Replica and database hits are mirrored into the memo and counted in
    ``shm_cache.replica_stats()``, as ``graph_utils.load_from_sqlite`` does.
    """
    keys = _as_keys(keys)
    for seen in _CAPTURES:
        seen.append(conn)
    hexes = keys_to_hex(keys)
    nonzero = keys.any(axis=1)
    distinct = list(dict.fromkeys(h for h, ok in zip(hexes, nonzero) if ok))

    objects, blobs = {}, {}
    memo = getattr(conn, '_aimmd_memo', None)
    pending = getattr(conn, '_aimmd_pending', None) or {}
    for h in distinct:
        graph = overlay_get(h)
        if graph is not None:
            objects[h] = graph
            continue
        blob = memo.get(h) if memo is not None else None
        if blob is not None:
            shm_cache._STATS['memo_hits'] += 1
            blobs[h] = blob
            continue
        blob = pending.get(h)
        if blob is not None:
            blobs[h] = blob

    rest = [h for h in distinct if h not in objects and h not in blobs]
    replica_rows = _from_replica(conn, rest, 'key, data')
    if replica_rows is not None:
        shm_cache._STATS['hits'] += len(replica_rows)
        shm_cache._STATS['misses'] += len(rest) - len(replica_rows)
        for h, blob in replica_rows:
            blobs[h] = blob
            if memo is not None:
                memo.put(h, blob)
        rest = [h for h in rest if h not in blobs]
    if rest:
        for h, blob in _select(conn, rest, 'key, data'):
            blobs[h] = blob
            if memo is not None:
                memo.put(h, blob)

    missing = list(dict.fromkeys(
        h for h, ok in zip(hexes, nonzero)
        if not ok or (h not in objects and h not in blobs)))
    if missing:
        raise GraphCacheMiss(np.stack([
            np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in missing]))
    return [objects[h] if h in objects else decode(blobs[h]) for h in hexes]
