"""Node-local ``/dev/shm`` read-replicas (and an in-process memo) for the graph cache.

Why
---
The trainer spends most of its wall clock re-scoring the whole reactive ensemble
every round, and ~98% of each re-scored frame is a single sqlite point lookup
against ``graphs_cache.sqlite`` on shared scratch.  That lookup's latency scales
with the size of the cache file, because the file never stays resident:

    cache size   cold lookup   same file in /dev/shm
      0.25 GB       0.039 ms          0.007 ms
      8.04 GB      19.292 ms          0.048 ms
     22.96 GB      24.597 ms          0.011 ms

i.e. ~2300x at production size, and flat in size.  Holding the cache in RAM
therefore removes the dominant cost of the phase.

Why a replica and not simply "put the cache in tmpfs"
-----------------------------------------------------
In a multi-system campaign the MD writers are spread over many nodes while the
trainer is a single process.  The *shared* cache cannot move to tmpfs -- writers
on the other nodes would lose it.  What can move is a trainer-side, node-local,
**read-only** replica, refreshed once per training cycle.

Why that is safe
----------------
Graphs are content-addressed and immutable: the key is
``sha256(pickle.dumps(descriptor_row))`` (``graph_utils.get_stable_hash``), the
only write is ``INSERT OR REPLACE``, and there is no ``UPDATE`` and no ``DELETE``
anywhere in ``graph_utils``.  For a fixed ``(cutoff, selections, atom_types,
universe)`` the key -> value map is a pure function, so **a replica can never be
wrong, only incomplete.**  A miss simply falls through to the real database.

(The one way to break that is to change ``CUTOFF`` or the selections mid-campaign,
which already invalidates the real cache today; the replica inherits the problem
rather than creating it.)

Lookup order is ``memo -> replica -> real DB -> recompute``, uniform for every
role.  Writers never stage anything, so ``_aimmd_replica`` is ``None`` there and
the extra cost is one ``getattr`` (~100 ns).

Safety
------
``/dev/shm`` is tmpfs backed by node RAM and is shared with the MPI runtime's
shared-memory transport, so filling it breaks unrelated tasks on the node.  Every
stage is preceded by a free-space check that keeps a reserve, and total usage is
capped.  Because ``MemAvailable`` discounts tmpfs 1:1, staging also shrinks the
memory the npy cache believes it may use -- so ``stage_cache`` decrements
``NPY_CACHE.max_size`` by what it stages and ``cleanup_replicas`` restores it,
keeping the node-level memory budget exact rather than approximate.

Everything here is best-effort: **no function in this module raises.** On any
failure it warns once and falls back to the behaviour of the unmodified code.

Environment
-----------
``AIMMD_SHM_DIR``          tmpfs directory; ``''``/``0``/``off``/``none`` disables the layer
``AIMMD_SHM_RESERVE``      bytes of tmpfs never to consume (default max(4 GiB, 10%))
``AIMMD_SHM_MAX_BYTES``    ceiling on total AIMMD replica bytes (default 50% of tmpfs)
``AIMMD_SHM_MAX_AGE``      seconds before an orphaned replica dir is reaped (default 3 d)
``AIMMD_GRAPH_MEMO_BYTES`` per-connection blob memo budget (default 64 MiB, 0 disables)
"""
import atexit
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import time
import weakref

from .._config import print


__all__ = ['CacheConnection', 'BlobMemo', 'register', 'registered_connections',
           'shm_root', 'replica_path', 'free_bytes', 'reserve_bytes',
           'budget_bytes', 'stage_cache', 'stage_replicas', 'refresh_replicas',
           'cleanup_replicas', 'replica_stats', 'detach',
           'set_reader_role', 'reader_role']


_GIB = 1024 ** 3
#: Hard wall-clock ceiling for staging ONE cache into tmpfs. The old
#: ``backup(pages=-1)`` had no ceiling and could block for the whole SLURM
#: allocation (6-12 h of silent trainer idle observed in production); this bounds
#: it, and on the ceiling staging degrades to "no replica, read the real DB"
#: rather than hang. Overridable with ``AIMMD_STAGE_DEADLINE``.
_STAGE_DEADLINE_SECONDS = float(os.environ.get('AIMMD_STAGE_DEADLINE', 300.0))
#: Sequential copy chunk; matches JUPITER's GPFS block size (8 MiB).
_COPY_CHUNK = 8 * 1024 * 1024
#: Busy timeout for the short-lived checkpoint/pin connection used while staging.
_STAGE_PIN_BUSY_S = 30.0
#: Consecutive staging failures after which a cache stops being re-armed for the
#: rest of the process. Without this a cache that cannot be staged is retried on
#: every lookup, turning one blown deadline into a stall before each one.
_STAGE_MAX_ATTEMPTS = int(os.environ.get('AIMMD_STAGE_MAX_ATTEMPTS', 3))
_DISABLED = ('', '0', 'off', 'none', 'false', 'no')
_DIR_PREFIX = 'aimmd-cache-u'

# Live cache connections, so a trainer can find every cache without needing a
# reference from `params` -- `Params.load` only copies declared dataclass fields,
# so a params-module global such as `SPEC` never reaches the params object.
# A WeakSet (not a dict) because nothing here should keep a connection alive:
# `Params.load` may run more than once per process.
_REGISTRY = weakref.WeakSet()

_STATS = {'hits': 0, 'misses': 0, 'staged_bytes': 0, 'memo_hits': 0}
_OWNED = {}          # replica path -> bytes, only what THIS process staged
_ATEXIT_ARMED = False
_WARNED = set()
_NPY_BUDGET_RETURNED = 0
#: True in a process that only *reads* the shared graph cache (the trainer).
#: Set structurally by the trainer entry points, never inferred.
_READER_ROLE = False


def set_reader_role(enabled=True):
    """Declare this process a cache *reader* (the trainer).

    A reader still creates graphs it needs, but keeps them in its memo and its
    own tmpfs replica instead of writing them to the shared database. That takes
    the trainer out of contention for SQLite's single, unfair write lock, which
    it was losing to ~35 MD writers for 300 s at a time -- long enough to be
    indistinguishable from a hang. Nothing is lost: the writers cache those same
    graphs when they reach the frames, and the cache is content-addressed, so a
    graph computed twice is byte-identical.

    Set ``AIMMD_TRAINER_WRITES_CACHE=1`` to restore the old behaviour (useful if
    a trainer ever runs with no MD writers to populate the cache for it).
    """
    global _READER_ROLE
    if os.environ.get('AIMMD_TRAINER_WRITES_CACHE', '') not in _DISABLED:
        return                       # explicitly opted out of the reader role
    _READER_ROLE = bool(enabled)


def reader_role():
    """True if this process must not write to the shared graph cache."""
    return _READER_ROLE


def _warn_once(tag, msg):
    """Warn at most once per tag, so a per-frame failure cannot spam the log."""
    if tag not in _WARNED:
        _WARNED.add(tag)
        print(f'!! shm_cache: {msg}')


def _env_bytes(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        _warn_once(f'env-{name}', f'{name}={raw!r} is not a number, using default')
        return default


class CacheConnection(sqlite3.Connection):
    """A ``sqlite3.Connection`` that can carry a ``/dev/shm`` read-replica.

    Subclassed purely so per-connection state can be attached: a plain
    ``sqlite3.Connection`` has no ``__dict__`` and cannot be weak-referenced.
    ``isinstance(c, sqlite3.Connection)`` still holds and ``.backup()`` still
    works, so every existing annotation and caller is unaffected.  Measured
    attribute-lookup overhead: ~100 ns, against a 24 ms cold sqlite read.
    """

    _aimmd_db_path = None
    _aimmd_replica = None
    _aimmd_replica_path = None
    _aimmd_stage_pending = False
    _aimmd_stage_failures = 0
    _aimmd_replica_frozen = False   # out of tmpfs: stops top-up AND write-through
    _aimmd_topup_broken = False     # source unreadable: stops top-up only
    _aimmd_watermark = 0
    _aimmd_memo = None


class BlobMemo:
    """Byte-budgeted FIFO memo of *compressed* graph blobs, keyed by hash.

    Stores the blob rather than the decoded ``Data`` for two reasons: it is ~8x
    smaller (7.5 KiB vs ~58 KiB for a production graph), and every hit returns a
    freshly unpickled object, so object-mutation semantics are **identical** to
    going to sqlite.  That is what makes the memo provably transparent rather
    than probably transparent.

    It exists because writers featurise every frame twice -- once during
    trajectory extension (which discards the graphs and only populates the
    cache) and again at shooting-point selection -- so the second lookup of each
    frame is a guaranteed hit that today costs a full sqlite read.
    """

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self.total = 0
        self._d = {}

    def get(self, key):
        return self._d.get(key)

    def put(self, key, blob):
        if self.max_bytes <= 0 or key in self._d:
            return
        size = len(blob)
        if size > self.max_bytes:        # never evict everything for one blob
            return
        while self._d and self.total + size > self.max_bytes:
            _, dropped = self._d.popitem()        # dict is insertion-ordered
            self.total -= len(dropped)
        self._d[key] = blob
        self.total += size

    def clear(self):
        self._d.clear()
        self.total = 0

    def __len__(self):
        return len(self._d)


# --------------------------------------------------------------- registry --
def register(conn):
    """Record a cache connection so a trainer can later stage it."""
    try:
        _REGISTRY.add(conn)
        budget = _env_bytes('AIMMD_GRAPH_MEMO_BYTES', 64 * 1024 * 1024)
        if budget > 0 and getattr(conn, '_aimmd_memo', None) is None:
            conn._aimmd_memo = BlobMemo(budget)
    except TypeError:
        pass          # a plain sqlite3.Connection: no replica, no memo, no harm


def registered_connections():
    """Live registered connections, ordered by database path (deterministic)."""
    out = [c for c in _REGISTRY if getattr(c, '_aimmd_db_path', None)]
    return sorted(out, key=lambda c: c._aimmd_db_path)


# ------------------------------------------------------------ shm layout --
def shm_root(shm_dir=None):
    """Per-user, per-job replica directory, or ``None`` if the layer is off."""
    base = shm_dir if shm_dir is not None else os.environ.get('AIMMD_SHM_DIR', '/dev/shm')
    if base is None or str(base).strip().lower() in _DISABLED:
        return None
    if not os.path.isdir(base):
        _warn_once('nodir', f'{base} is not a directory; replicas disabled')
        return None
    job = os.environ.get('SLURM_JOB_ID') or f'pid{os.getpid()}'
    return os.path.join(base, f'{_DIR_PREFIX}{os.getuid()}-j{job}')


def replica_path(db_path, shm_dir=None):
    """Replica filename for ``db_path``, or ``None`` if the layer is off.

    The path hash is what keeps the caches of different systems apart, and also
    two runs whose caches happen to share a basename in different directories.
    """
    root = shm_root(shm_dir)
    if root is None:
        return None
    real = os.path.realpath(db_path)
    tag = hashlib.sha256(real.encode()).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(real))[0][:40]
    stem = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in stem)
    return os.path.join(root, f'{stem}.{tag}.sqlite')


def free_bytes(shm_dir=None):
    root = shm_root(shm_dir)
    if root is None:
        return 0
    try:
        st = os.statvfs(os.path.dirname(root))
        return st.f_bavail * st.f_frsize
    except OSError as exc:
        _warn_once('statvfs', f'cannot statvfs the tmpfs ({exc}); replicas disabled')
        return 0


def _total_bytes(shm_dir=None):
    root = shm_root(shm_dir)
    if root is None:
        return 0
    try:
        st = os.statvfs(os.path.dirname(root))
        return st.f_blocks * st.f_frsize
    except OSError:
        return 0


def reserve_bytes(shm_dir=None):
    """Tmpfs we must never consume: the MPI transport and others also live here."""
    return _env_bytes('AIMMD_SHM_RESERVE',
                      max(4 * _GIB, int(0.10 * _total_bytes(shm_dir))))


def budget_bytes(shm_dir=None):
    """Ceiling on total AIMMD replica bytes on this node."""
    return _env_bytes('AIMMD_SHM_MAX_BYTES', int(0.50 * _total_bytes(shm_dir)))


def _fits(need, shm_dir=None):
    """Is there room for ``need`` more bytes, keeping the reserve and the cap?"""
    if free_bytes(shm_dir) - need < reserve_bytes(shm_dir):
        return False, 'would eat into the tmpfs reserve'
    if sum(_OWNED.values()) + need > budget_bytes(shm_dir):
        return False, 'would exceed AIMMD_SHM_MAX_BYTES'
    return True, ''


# ------------------------------------------------------- npy budget swap --
def _take_npy_budget(nbytes):
    """Shrink the npy cache budget by what we just moved into tmpfs.

    ``NpyReaderCache.max_size`` is half of *available* memory, fixed at import --
    which for the trainer happens before any staging.  Since ``MemAvailable``
    discounts tmpfs 1:1, staging silently invalidates that budget by exactly the
    staged amount.  Hand the bytes over so the node-level total stays honest.
    """
    global _NPY_BUDGET_RETURNED
    try:
        from .. import _config
        cache = getattr(_config, 'NPY_CACHE', None)
        if cache is None:
            return
        cache.max_size = max(_GIB, int(cache.max_size) - int(nbytes))
        _NPY_BUDGET_RETURNED += int(nbytes)
    except Exception:
        pass


def _return_npy_budget():
    global _NPY_BUDGET_RETURNED
    if not _NPY_BUDGET_RETURNED:
        return
    try:
        from .. import _config
        cache = getattr(_config, 'NPY_CACHE', None)
        if cache is not None:
            cache.max_size = int(cache.max_size) + _NPY_BUDGET_RETURNED
    except Exception:
        pass
    _NPY_BUDGET_RETURNED = 0
#: True in a process that only *reads* the shared graph cache (the trainer).
#: Set structurally by the trainer entry points, never inferred.
_READER_ROLE = False


# ---------------------------------------------------------------- reaping --
def _owner_file(root):
    return os.path.join(root, f'owner.{os.getpid()}.json')


def _write_owner(root):
    try:
        with open(_owner_file(root), 'w') as fh:
            json.dump({'pid': os.getpid(), 'host': os.uname().nodename,
                       'job': os.environ.get('SLURM_JOB_ID'),
                       'created': time.time(),
                       'replicas': sorted(_OWNED)}, fh)
    except OSError:
        pass


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _reap_stale(shm_dir=None):
    """Delete replica directories whose owning process is gone.

    SLURM does not clean ``/dev/shm``, so a SIGKILL or a node crash would
    otherwise strand tens of GB of RAM until reboot.
    """
    root = shm_root(shm_dir)
    if root is None:
        return
    parent = os.path.dirname(root)
    keep = os.path.basename(root)
    max_age = _env_bytes('AIMMD_SHM_MAX_AGE', 3 * 24 * 3600)
    try:
        entries = os.listdir(parent)
    except OSError:
        return
    for name in entries:
        if not name.startswith(f'{_DIR_PREFIX}{os.getuid()}-') or name == keep:
            continue
        d = os.path.join(parent, name)
        try:
            owners = [f for f in os.listdir(d) if f.startswith('owner.')]
            alive = False
            for f in owners:
                try:
                    pid = int(f.split('.')[1])
                except (IndexError, ValueError):
                    continue
                if _pid_alive(pid):
                    alive = True
                    break
            stale_age = (time.time() - os.path.getmtime(d)) > max_age
            if alive and not stale_age:
                continue
            size = sum(os.path.getsize(os.path.join(d, f))
                       for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f)))
            shutil.rmtree(d, ignore_errors=True)
            print(f'shm_cache: reaped orphaned {d} ({size / 1e9:.1f} GB)')
        except OSError:
            continue


# ---------------------------------------------------------------- staging --
def _recount_owned(dst):
    """Record a replica's true tmpfs footprint: main file plus its WAL.

    The old ``backup()`` produced no ``-wal``, so counting the main file alone
    was exact. A copy can bring one across (the biggest production cache carried
    2.79 GB of it), and that is real tmpfs -- invisible otherwise to both
    ``AIMMD_SHM_MAX_BYTES`` and the npy-budget handover, i.e. node memory
    over-commit in the unsafe direction.
    """
    total = 0
    for suffix in ('', '-wal'):
        try:
            total += os.path.getsize(dst + suffix)
        except OSError:
            pass
    _OWNED[dst] = total
    _STATS['staged_bytes'] = sum(_OWNED.values())
    return total


def _stage_failed(conn):
    """Disarm a cache after a failed staging attempt.

    Clearing ``_aimmd_stage_pending`` is what stops the read path retrying the
    whole copy on the very next lookup; the failure counter is what stops
    ``stage_replicas`` re-arming it every round forever.
    """
    conn._aimmd_stage_pending = False
    conn._aimmd_stage_failures = getattr(conn, '_aimmd_stage_failures', 0) + 1


def _snapshot_copy(conn, db_path, partial, deadline_s):
    """Bounded, hang-proof snapshot of a live WAL cache into ``partial``.

    Replaces ``conn.backup(dest, pages=-1)``. That call defeats the
    restart-livelock of a chunked backup, but only by holding one WAL
    read-snapshot for the entire, uninterruptible, single C-level copy -- and
    that read-mark blocks WAL checkpointing. On a multi-GB, write-hot cache the
    WAL then grows without bound while writers append, per-page reads on a
    parallel filesystem collapse, and the one copy never returns: 6-12 h of
    silent trainer idle were observed in production, and ``Connection.interrupt``
    is ignored by ``sqlite3_backup_step`` so it cannot even be aborted.

    Instead:
      1. ``PRAGMA wal_checkpoint(TRUNCATE)`` folds the WAL into the main file. A
         checkpoint RETURNS (busy or done) -- it never waits unboundedly -- so
         this step cannot hang; a partial checkpoint just leaves the main file
         slightly behind, which is fine (see below).
      2. Pin a short read snapshot on a throwaway connection, held only for the
         seconds of the copy (not the hours a ``pages=-1`` step would). If the
         TRUNCATE succeeded the pin takes read-mark 0, which blocks *all*
         backfill and freezes the main file outright. If it returned busy, the
         pin caps backfill at its own mark -- a checkpointer CAN still rewrite
         main-file pages under the copy, but only pages that also have a frame
         in the WAL we copy next, and the WAL wins on read, so replay repairs
         the tearing. **This is why the ``-wal`` must be copied, and copied
         after the main file: removing or reordering it introduces real
         corruption.** The pin also blocks WAL *reset*, which is what keeps
         those repair frames from being truncated away mid-copy.
      3. Plain SEQUENTIAL file copy of the main db (+ the now-tiny WAL) with a
         wall-clock deadline enforced in our own loop -- sequential I/O plays to
         a parallel FS's strength, and a deadline is trivially enforceable
         because the loop is ours.

    Correctness rests on the cache being content-addressed and append-only: the
    copy is at worst slightly stale ("incomplete, never wrong"), and
    :func:`refresh_replicas` tops it up. Raises ``TimeoutError`` on the deadline.
    """
    t0 = time.monotonic()
    # 1. bounded checkpoint -- a partial result is acceptable, so swallow errors.
    try:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    except sqlite3.Error:
        pass
    # 2. pin a read snapshot so a checkpoint cannot move main-file pages mid-copy
    # A busy TRUNCATE can burn the source connection's whole busy_timeout, so
    # check the clock before committing to the copy rather than only inside it.
    if time.monotonic() - t0 > deadline_s:
        raise TimeoutError(f'checkpoint alone exceeded the {deadline_s:.0f}s deadline')
    pin = sqlite3.connect(db_path, timeout=_STAGE_PIN_BUSY_S)
    try:
        pin.execute('BEGIN')
        pin.execute('SELECT 1 FROM graphs_cache LIMIT 1').fetchone()
        # 3. sequential copy under the deadline
        for suffix in ('', '-wal'):
            src = db_path + suffix
            if not os.path.exists(src):
                continue
            with open(src, 'rb') as fi, open(partial + suffix, 'wb') as fo:
                while True:
                    if time.monotonic() - t0 > deadline_s:
                        raise TimeoutError(
                            f'snapshot exceeded {deadline_s:.0f}s deadline')
                    chunk = fi.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    fo.write(chunk)
    finally:
        try:
            pin.rollback()
        except sqlite3.Error:
            pass
        pin.close()


def stage_cache(conn, shm_dir=None, force=False):
    """Copy one cache into tmpfs and attach the replica to ``conn``.

    The copy is a bounded checkpoint-then-sequential-copy (:func:`_snapshot_copy`)
    with a hard wall-clock deadline, chosen so staging can never block the
    trainer -- see that function for why the previous ``backup(pages=-1)`` could
    hang for the whole allocation on a large, write-hot cache.

    Returns the replica path, or ``None`` if it was skipped (including on a blown
    staging deadline, in which case the trainer simply reads the real database).
    Never raises.
    """
    global _ATEXIT_ARMED
    db_path = getattr(conn, '_aimmd_db_path', None)
    if db_path is None:
        return None
    if getattr(conn, '_aimmd_replica', None) is not None and not force:
        return getattr(conn, '_aimmd_replica_path', None)
    dst = replica_path(db_path, shm_dir)
    if dst is None:
        return None

    try:
        need = os.path.getsize(db_path)
        wal = db_path + '-wal'
        if os.path.exists(wal):
            need += os.path.getsize(wal)
        need = int(need * 1.05)
    except OSError as exc:
        _warn_once('size', f'cannot size {db_path} ({exc}); not staging')
        _stage_failed(conn)
        return None

    # the copy lands in a .partial beside any previous replica before the swap
    ok, why = _fits(need * 2 if os.path.exists(dst) else need, shm_dir)
    if not ok:
        _warn_once(f'space-{dst}',
                   f'{why}; {os.path.basename(db_path)} stays on disk '
                   f'({need / 1e9:.1f} GB needed, {free_bytes(shm_dir) / 1e9:.1f} GB free)')
        _stage_failed(conn)
        return None

    root = os.path.dirname(dst)
    partial = f'{dst}.partial.{os.getpid()}'
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
        t0 = time.time()
        _snapshot_copy(conn, db_path, partial, _STAGE_DEADLINE_SECONDS)
        os.replace(partial, dst)                    # atomic on tmpfs
        # Move the (now tiny) WAL alongside, or clear any stale one, so the
        # replica opens against a matched pair.
        if os.path.exists(partial + '-wal'):
            os.replace(partial + '-wal', dst + '-wal')
        else:
            try:
                os.remove(dst + '-wal')
            except OSError:
                pass
        size = os.path.getsize(dst)
        dt = time.time() - t0
    except (sqlite3.Error, OSError, TimeoutError) as exc:
        for leftover in (partial, partial + '-wal', partial + '-shm',
                         dst + '-wal'):
            try:
                os.remove(leftover)
            except OSError:
                pass
        _warn_once(f'stage-{dst}',
                   f'could not stage {os.path.basename(db_path)} within '
                   f'{_STAGE_DEADLINE_SECONDS:.0f}s ({exc}); '
                   f'continuing on the real database')
        _stage_failed(conn)
        return None

    try:
        # NB opened read-WRITE: `mode=ro` would be a file-level open that
        # PRAGMA query_only=OFF cannot lift, which would silently disable the
        # top-up and write-through paths. query_only gives the same protection
        # and can be toggled for those two writers.
        replica = sqlite3.connect(dst, check_same_thread=False)
        replica.execute('PRAGMA journal_mode=WAL')
        replica.execute('PRAGMA synchronous=OFF')
        replica.execute('PRAGMA query_only=ON')
        watermark = replica.execute(
            'SELECT COALESCE(MAX(rowid), 0) FROM graphs_cache').fetchone()[0]
    except sqlite3.Error as exc:
        _warn_once(f'open-{dst}', f'staged but could not open {dst} ({exc})')
        try:
            os.remove(dst)
        except OSError:
            pass
        _stage_failed(conn)
        return None

    old = _OWNED.pop(dst, 0)
    size = _recount_owned(dst)          # main + any WAL the copy brought across
    _take_npy_budget(size - old)
    conn._aimmd_stage_failures = 0
    conn._aimmd_replica = replica
    conn._aimmd_replica_path = dst
    conn._aimmd_watermark = watermark
    conn._aimmd_stage_pending = False
    conn._aimmd_replica_frozen = False
    _write_owner(root)
    if not _ATEXIT_ARMED:
        atexit.register(cleanup_replicas)
        _ATEXIT_ARMED = True
    print(f'shm_cache: staged {os.path.basename(db_path)} -> {dst} '
          f'({size / 1e9:.2f} GB, {size / 1e6 / max(dt, 1e-9):.0f} MB/s, '
          f'{watermark:,} rows)')
    return dst


def stage_replicas(shm_dir=None, lazy=True, verbose=True):
    """Trainer entry point: make every registered cache available in tmpfs.

    Call once per training cycle.  By default this only *arms* each connection
    and the copy happens on its first lookup, because with
    ``multi_system_share_network=False`` the launcher runs one trainer per system
    while every trainer still has all systems' connections registered -- eager
    staging would try to replicate every cache in every trainer.  Lazy staging
    means a trainer replicates only the caches it actually reads.
    """
    conns = registered_connections()
    if not conns or shm_root(shm_dir) is None:
        return {}
    _reap_stale(shm_dir)
    out = {}
    for conn in conns:
        if getattr(conn, '_aimmd_replica', None) is not None:
            conn._aimmd_replica_frozen = False       # a new cycle may have room
            conn._aimmd_topup_broken = False         # and the source may be back
            out[conn._aimmd_db_path] = conn._aimmd_replica_path
        elif lazy:
            # Do not re-arm a cache that has already failed repeatedly: the read
            # path retries whenever this is set, so an unstageable cache would
            # otherwise pay a staging attempt before every lookup.
            if getattr(conn, '_aimmd_stage_failures', 0) < _STAGE_MAX_ATTEMPTS:
                conn._aimmd_stage_pending = True
        else:
            path = stage_cache(conn, shm_dir=shm_dir)
            if path:
                out[conn._aimmd_db_path] = path
    if verbose and out:
        print(f'shm_cache: {len(out)} replica(s) live, '
              f'{sum(_OWNED.values()) / 1e9:.1f} GB in tmpfs')
    return out


def _open_source(replica, db_path):
    """ATTACH the real database as ``src``, without URI interpretation.

    ``ATTACH DATABASE 'file:<path>?mode=ro'`` only honours the URI when
    SQLite's *global* ``SQLITE_USE_URI`` is enabled; the per-connection
    ``uri=True`` open flag does not extend to ATTACH. On builds where it is off
    -- JUPITER's is -- the whole URI is taken as a literal filename and the
    attach fails with "unable to open database", which froze every replica at
    its staging row count for an entire production run.

    A bound parameter is used rather than an f-string so that paths containing
    quotes, ``?`` or ``#`` cannot be misread. Read-only-ness is not needed:
    ``src`` is only ever SELECTed from.
    """
    replica.execute('ATTACH DATABASE ? AS src', (db_path,))


def write_through(conn, keys, blobs):
    """Mirror freshly written graphs straight into the replica.

    Independent of the top-up: this needs no access to the real database, so a
    broken source must not disable it. Only a *space* freeze does, because
    tmpfs is one shared resource.

    Returns True if the rows reached the replica.
    """
    replica = getattr(conn, '_aimmd_replica', None)
    if replica is None or getattr(conn, '_aimmd_replica_frozen', False):
        return False
    ok, why = _fits(sum(len(b) for b in blobs))
    if not ok:
        conn._aimmd_replica_frozen = True
        _warn_once(f'wt-{conn._aimmd_replica_path}',
                   f'{why}; replica frozen -- reads still served from it, new '
                   f'graphs go to the real database only')
        return False
    try:
        replica.execute('PRAGMA query_only=OFF')
        replica.executemany(
            'INSERT OR REPLACE INTO graphs_cache (key, data) VALUES (?, ?)',
            zip(keys, blobs))
        replica.commit()
        replica.execute('PRAGMA query_only=ON')
        return True
    except sqlite3.Error as exc:
        conn._aimmd_replica_frozen = True
        _warn_once(f'wt-err-{conn._aimmd_replica_path}',
                   f'write-through failed ({exc}); replica frozen')
        return False


def refresh_replicas(shm_dir=None, verbose=False):
    """Top up every staged replica with the rows added since it was staged.

    Correct because the table is append-only and content-addressed: no existing
    key's value can change, and ``INSERT OR REPLACE`` only ever assigns a
    *higher* rowid, so nothing new can appear below the watermark.  Costs a
    fraction of a second where a full restage costs tens of seconds, which is
    why the trainer can afford to call it again just before the re-score.
    """
    added = {}
    for conn in registered_connections():
        replica = getattr(conn, '_aimmd_replica', None)
        if (replica is None
                or getattr(conn, '_aimmd_replica_frozen', False)
                or getattr(conn, '_aimmd_topup_broken', False)):
            continue
        db_path = conn._aimmd_db_path
        dst = conn._aimmd_replica_path
        try:
            hi = conn.execute(
                'SELECT COALESCE(MAX(rowid), 0) FROM graphs_cache').fetchone()[0]
        except sqlite3.Error:
            continue
        low = getattr(conn, '_aimmd_watermark', 0)
        if hi <= low:
            continue
        try:
            grow = int((hi - low) * (os.path.getsize(dst) / max(low, 1)) * 1.1)
        except OSError:
            grow = 0
        ok, why = _fits(max(grow, 0), shm_dir)
        if not ok:
            conn._aimmd_replica_frozen = True
            _warn_once(f'freeze-{dst}',
                       f'{why}; {os.path.basename(db_path)} replica frozen at '
                       f'{low:,} rows -- reads still served from it, new graphs '
                       f'go to the real database only')
            continue
        try:
            replica.execute('PRAGMA query_only=OFF')
            replica.commit()                    # ATTACH fails inside a txn
            _open_source(replica, db_path)
            try:
                replica.execute(
                    'INSERT OR IGNORE INTO graphs_cache '
                    'SELECT key, data FROM src.graphs_cache '
                    'WHERE rowid > ? AND rowid <= ?', (low, hi))
                replica.commit()
            finally:
                replica.execute('DETACH DATABASE src')
            replica.execute('PRAGMA query_only=ON')
            conn._aimmd_watermark = hi
            added[db_path] = hi - low
            try:
                _recount_owned(dst)
            except OSError:
                pass
        except sqlite3.Error as exc:
            # NOT a space freeze: write-through needs no source access and must
            # keep running, or the replica cannot grow at all.
            conn._aimmd_topup_broken = True
            try:
                replica.execute('DETACH DATABASE src')
            except sqlite3.Error:
                pass
            try:
                replica.execute('PRAGMA query_only=ON')
            except sqlite3.Error:
                pass
            _warn_once(f'topup-{dst}',
                       f'top-up failed ({exc}); replica still readable and '
                       f'still accepting new graphs, but no longer catching up '
                       f'with rows written by other processes')
    if verbose and added:
        print('shm_cache: topped up ' +
              ', '.join(f'{os.path.basename(k)} +{v:,}' for k, v in added.items()))
    return added


def detach(conn, reason=''):
    """Stop using a connection's replica for the rest of the process."""
    replica = getattr(conn, '_aimmd_replica', None)
    if replica is None:
        return
    try:
        replica.close()
    except sqlite3.Error:
        pass
    conn._aimmd_replica = None
    conn._aimmd_stage_pending = False
    if reason:
        _warn_once(f'detach-{conn._aimmd_replica_path}',
                   f'detached replica for {os.path.basename(conn._aimmd_db_path)}: '
                   f'{reason}; falling back to the real database')


def cleanup_replicas():
    """Close replica connections and delete only the files this process staged.

    A sibling trainer of the same job on the same node shares the directory, so
    the directory itself is only removed once it is empty.
    """
    for conn in list(_REGISTRY):
        replica = getattr(conn, '_aimmd_replica', None)
        if replica is not None:
            try:
                replica.close()
            except sqlite3.Error:
                pass
            conn._aimmd_replica = None
    freed = 0
    root = None
    for path, size in list(_OWNED.items()):
        root = root or os.path.dirname(path)
        try:
            os.remove(path)
            freed += size
        except OSError:
            pass
        for suffix in ('-wal', '-shm'):
            try:
                os.remove(path + suffix)
            except OSError:
                pass
        _OWNED.pop(path, None)
    _return_npy_budget()
    _STATS['staged_bytes'] = 0
    if root:
        try:
            os.remove(_owner_file(root))
        except OSError:
            pass
        try:
            os.rmdir(root)                # only if no sibling still owns files
        except OSError:
            pass
    if freed:
        print(f'shm_cache: released {freed / 1e9:.1f} GB of tmpfs')


def replica_stats():
    """Counters for confirming in production that replicas are actually used."""
    return dict(_STATS)
