"""Tests for the /dev/shm graph-cache replica layer (aimmd.network.shm_cache).

These run in the default suite: shm_cache imports only the standard library, so
unlike graph_utils it needs no GNN stack. Every test points the layer at
``tmp_path`` instead of the real ``/dev/shm``, so nothing here depends on tmpfs
being present, writable, or empty.

The cache fixtures are plain sqlite tables holding pickled bytes, which is all
the layer ever sees -- it never decodes a graph.
"""
import os
import pickle
import sqlite3

import pytest

from aimmd.network import shm_cache


# --------------------------------------------------------------- fixtures --
def _make_cache(path, payloads):
    """A stand-in graph cache: same schema and PRAGMAs as init_db."""
    conn = sqlite3.connect(str(path), factory=shm_cache.CacheConnection)
    conn.execute('CREATE TABLE IF NOT EXISTS graphs_cache'
                 '(key TEXT PRIMARY KEY, data BLOB)')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executemany('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                     [(k, pickle.dumps(v)) for k, v in payloads.items()])
    conn.commit()
    conn._aimmd_db_path = os.path.abspath(str(path))
    shm_cache.register(conn)
    return conn


def _get(conn, key):
    """The read path, mirroring graph_utils.load_from_sqlite's ordering."""
    memo = getattr(conn, '_aimmd_memo', None)
    if memo is not None:
        blob = memo.get(key)
        if blob is not None:
            return pickle.loads(blob)
    replica = getattr(conn, '_aimmd_replica', None)
    if replica is None and getattr(conn, '_aimmd_stage_pending', False):
        shm_cache.stage_cache(conn)
        replica = getattr(conn, '_aimmd_replica', None)
    if replica is not None:
        try:
            row = replica.execute(
                'SELECT data FROM graphs_cache WHERE key = ?', (key,)).fetchone()
        except sqlite3.Error as exc:
            shm_cache.detach(conn, reason=str(exc))
            row = None
        if row is not None:
            if memo is not None:
                memo.put(key, row[0])
            return pickle.loads(row[0])
    row = conn.execute(
        'SELECT data FROM graphs_cache WHERE key = ?', (key,)).fetchone()
    if row is None:
        return None
    if memo is not None:
        memo.put(key, row[0])
    return pickle.loads(row[0])


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point the layer at tmp_path and leave no global state between tests."""
    shm = tmp_path / 'shm'
    shm.mkdir()
    monkeypatch.setenv('AIMMD_SHM_DIR', str(shm))
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    shm_cache._REGISTRY.clear()
    shm_cache._OWNED.clear()
    shm_cache._WARNED.clear()
    shm_cache._NPY_BUDGET_RETURNED = 0
    for k in shm_cache._STATS:
        shm_cache._STATS[k] = 0
    yield shm
    shm_cache.cleanup_replicas()


# ------------------------------------------------------------ read paths --
def test_replica_hit_serves_from_shm(tmp_path):
    """A staged key comes from the replica, without touching the real database."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert shm_cache.stage_cache(conn) is not None

    calls = []
    real_execute = conn.execute
    conn.execute = lambda *a, **k: (calls.append(a), real_execute(*a, **k))[1]
    conn._aimmd_memo = None                      # isolate the replica from the memo

    assert _get(conn, 'a') == 1
    assert calls == [], 'the real database was queried despite a replica hit'


def test_replica_miss_falls_through_to_real_db(tmp_path):
    """A key added after staging is still found -- via the real database."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('b', pickle.dumps(2)))
    conn.commit()

    assert _get(conn, 'b') == 2, 'a stale-replica miss must not escape as None'
    assert _get(conn, 'zz') is None, 'a genuinely absent key must still be None'


def test_stale_replica_miss_does_not_recompute(tmp_path):
    """The requirement that makes the whole scheme safe.

    If a miss against a stale replica escaped as None, the caller would rebuild a
    graph that already exists -- silently, and for every frame.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    for key in ('b', 'c', 'd'):
        conn.execute('INSERT INTO graphs_cache VALUES (?,?)',
                     (key, pickle.dumps(key)))
    conn.commit()

    rebuilt = [k for k in ('a', 'b', 'c', 'd') if _get(conn, k) is None]
    assert rebuilt == [], f'would have recomputed {rebuilt}'


def test_worker_process_without_staging_is_unaffected(tmp_path):
    """35 of 36 tasks never stage; their behaviour must be bit-identical."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1, 'b': 2})
    conn._aimmd_memo = None
    assert getattr(conn, '_aimmd_replica', None) is None
    assert _get(conn, 'a') == 1 and _get(conn, 'b') == 2
    assert _get(conn, 'nope') is None
    assert shm_cache.replica_stats()['hits'] == 0


# -------------------------------------------------------------- staging ---
def test_stage_cache_creates_replica_and_attaches(tmp_path):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1, 'b': 2, 'c': 3})
    path = shm_cache.stage_cache(conn)
    assert path and os.path.exists(path)
    assert conn._aimmd_replica is not None
    n = conn._aimmd_replica.execute(
        'SELECT count(*) FROM graphs_cache').fetchone()[0]
    assert n == 3
    assert conn._aimmd_watermark == 3


def test_stage_refused_when_space_is_short(tmp_path, monkeypatch):
    """Short on tmpfs: warn, skip, never raise."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    monkeypatch.setattr(shm_cache, 'free_bytes', lambda *a, **k: 1)
    assert shm_cache.stage_cache(conn) is None
    assert getattr(conn, '_aimmd_replica', None) is None
    assert _get(conn, 'a') == 1, 'must still work off the real database'


def test_stage_refused_when_over_budget(tmp_path, monkeypatch):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    monkeypatch.setenv('AIMMD_SHM_MAX_BYTES', '1')
    assert shm_cache.stage_cache(conn) is None


def test_lazy_staging_only_stages_touched_caches(tmp_path):
    """With share=False every trainer registers all systems but reads only one.

    Eager staging would replicate every cache in every trainer; lazy staging
    replicates only what is actually read.
    """
    conns = [_make_cache(tmp_path / f'c{i}.sqlite', {'a': i}) for i in range(3)]
    shm_cache.stage_replicas(lazy=True)
    root = shm_cache.shm_root()
    assert not os.path.isdir(root) or not [
        f for f in os.listdir(root) if f.endswith('.sqlite')]

    _get(conns[1], 'a')
    staged = [f for f in os.listdir(root) if f.endswith('.sqlite')]
    assert len(staged) == 1, f'expected 1 replica, got {staged}'


def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv('AIMMD_SHM_DIR', 'off')
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert shm_cache.shm_root() is None
    assert shm_cache.stage_replicas() == {}
    assert shm_cache.stage_cache(conn) is None
    assert _get(conn, 'a') == 1


def test_missing_shm_dir_warns_once_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv('AIMMD_SHM_DIR', str(tmp_path / 'does-not-exist'))
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert shm_cache.stage_cache(conn) is None
    assert shm_cache.stage_cache(conn) is None
    assert sum(1 for w in shm_cache._WARNED if w == 'nodir') == 1
    assert _get(conn, 'a') == 1


def test_corrupt_replica_detaches_and_falls_back(tmp_path):
    """A replica that starts erroring is abandoned, not allowed to fail reads.

    Corrupting the file on disk is not enough to trigger this: the open replica
    connection keeps serving from its own page cache. What the production path
    actually guards against is the read *raising*, so that is what is simulated.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None

    class Exploding:                       # sqlite3.Connection.execute is read-only
        def execute(self, *a, **k):
            raise sqlite3.DatabaseError('database disk image is malformed')

        def close(self):
            pass
    conn._aimmd_replica = Exploding()

    assert _get(conn, 'a') == 1, 'must fall through to the real database'
    assert getattr(conn, '_aimmd_replica', None) is None, 'must detach'
    assert _get(conn, 'a') == 1, 'and stay working afterwards'


# ------------------------------------------------------------- freezing ---
def test_out_of_space_freezes_writes_but_keeps_reads(tmp_path, monkeypatch):
    """Requirement: stop accumulating, but keep what is already staged useful."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('b', pickle.dumps(2)))
    conn.commit()

    monkeypatch.setattr(shm_cache, 'free_bytes', lambda *a, **k: 1)
    shm_cache.refresh_replicas()
    assert conn._aimmd_replica_frozen

    assert _get(conn, 'a') == 1, 'already-staged keys must still be served'
    assert _get(conn, 'b') == 2, 'new keys must still resolve via the real DB'


def test_freeze_clears_when_space_returns(tmp_path, monkeypatch):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_replica_frozen = True
    shm_cache.stage_replicas()
    assert not conn._aimmd_replica_frozen


# ------------------------------------------------------------- refresh ----
def test_incremental_refresh_picks_up_new_rows(tmp_path):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    for k in ('b', 'c'):
        conn.execute('INSERT INTO graphs_cache VALUES (?,?)', (k, pickle.dumps(k)))
    conn.commit()

    added = shm_cache.refresh_replicas()
    assert added.get(conn._aimmd_db_path) == 2
    n = conn._aimmd_replica.execute(
        'SELECT count(*) FROM graphs_cache').fetchone()[0]
    assert n == 3


def test_incremental_refresh_after_insert_or_replace(tmp_path):
    """INSERT OR REPLACE moves a row to a HIGHER rowid, so a watermark is safe."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1, 'b': 2})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    conn.execute('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                 ('a', pickle.dumps(99)))
    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('c', pickle.dumps(3)))
    conn.commit()

    shm_cache.refresh_replicas()
    rows = dict(conn._aimmd_replica.execute('SELECT key, data FROM graphs_cache'))
    assert set(rows) == {'a', 'b', 'c'}, 'no duplicate keys after a REPLACE'
    assert _get(conn, 'c') == 3


# -------------------------------------------------------------- naming ----
def test_multi_cache_naming_isolation(tmp_path):
    """Five systems, and two caches sharing a basename in different folders."""
    d1, d2 = tmp_path / 'G2', tmp_path / 'G3'
    d1.mkdir(); d2.mkdir()
    c1 = _make_cache(d1 / 'graphs_cache.sqlite', {'k': 'from-G2'})
    c2 = _make_cache(d2 / 'graphs_cache.sqlite', {'k': 'from-G3'})
    p1, p2 = shm_cache.stage_cache(c1), shm_cache.stage_cache(c2)
    assert p1 != p2
    c1._aimmd_memo = c2._aimmd_memo = None
    assert _get(c1, 'k') == 'from-G2'
    assert _get(c2, 'k') == 'from-G3'


def test_replica_path_isolated_by_uid_and_job(tmp_path, monkeypatch):
    db = tmp_path / 'c.sqlite'
    monkeypatch.setenv('SLURM_JOB_ID', '111')
    a = shm_cache.replica_path(str(db))
    monkeypatch.setenv('SLURM_JOB_ID', '222')
    b = shm_cache.replica_path(str(db))
    assert a != b
    assert f'-u{os.getuid()}-' in a


# ------------------------------------------------------------- cleanup ----
def test_cleanup_on_normal_exit(tmp_path):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    path = shm_cache.stage_cache(conn)
    root = os.path.dirname(path)
    shm_cache.cleanup_replicas()
    assert not os.path.exists(path)
    assert not os.path.isdir(root)
    shm_cache.cleanup_replicas()          # idempotent


def test_cleanup_leaves_sibling_replicas_alone(tmp_path):
    """A sibling trainer of the same job shares the directory."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    path = shm_cache.stage_cache(conn)
    root = os.path.dirname(path)
    sibling = os.path.join(root, 'someone-elses.sqlite')
    open(sibling, 'wb').write(b'x')

    shm_cache.cleanup_replicas()
    assert not os.path.exists(path)
    assert os.path.exists(sibling), 'clobbered a replica we do not own'
    assert os.path.isdir(root), 'removed a directory that is still in use'


def test_abnormal_exit_leftovers_are_reaped(tmp_path, monkeypatch):
    """SIGKILL leaves tens of GB behind; SLURM does not clean /dev/shm."""
    shm = tmp_path / 'shm'
    dead = shm / f'{shm_cache._DIR_PREFIX}{os.getuid()}-j999999'
    dead.mkdir()
    (dead / 'stale.sqlite').write_bytes(b'x' * 100)
    import json
    (dead / 'owner.2147480000.json').write_text(json.dumps({'pid': 2147480000}))

    live = shm / f'{shm_cache._DIR_PREFIX}{os.getuid()}-j888888'
    live.mkdir()
    (live / f'owner.{os.getpid()}.json').write_text(
        json.dumps({'pid': os.getpid()}))

    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_replicas(lazy=False)

    assert not dead.exists(), 'orphan of a dead pid was not reaped'
    assert live.exists(), 'reaped a directory whose owner is still alive'


def test_age_based_reaping(tmp_path, monkeypatch):
    shm = tmp_path / 'shm'
    old = shm / f'{shm_cache._DIR_PREFIX}{os.getuid()}-j777777'
    old.mkdir()
    (old / 'stale.sqlite').write_bytes(b'x')
    import json
    (old / f'owner.{os.getpid()}.json').write_text(
        json.dumps({'pid': os.getpid()}))          # alive, but ancient
    os.utime(old, (0, 0))

    monkeypatch.setenv('AIMMD_SHM_MAX_AGE', '1')
    _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_replicas(lazy=False)
    assert not old.exists()


def test_registry_does_not_leak_connections(tmp_path):
    """A plain dict registry would pin every connection Params.load ever made."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert len(shm_cache.registered_connections()) == 1
    del conn
    import gc
    gc.collect()
    assert len(shm_cache.registered_connections()) == 0


def test_staging_hands_back_npy_budget(tmp_path):
    """tmpfs is RAM, and MemAvailable discounts it 1:1.

    NpyReaderCache.max_size is fixed at import, which for the trainer happens
    before any staging -- so without this the trainer would authorise a cache
    budget that no longer exists.
    """
    from aimmd import _config
    cache = getattr(_config, 'NPY_CACHE', None)
    if cache is None:
        pytest.skip('NPY_CACHE not initialised')
    before = cache.max_size
    conn = _make_cache(tmp_path / 'c.sqlite', {str(i): i for i in range(200)})
    path = shm_cache.stage_cache(conn)
    assert cache.max_size == before - os.path.getsize(path)
    shm_cache.cleanup_replicas()
    assert cache.max_size == before


# ---------------------------------------------------------------- memo ----
def test_memo_serves_second_lookup_without_sqlite(tmp_path):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert _get(conn, 'a') == 1                  # primes the memo
    calls = []
    real = conn.execute
    conn.execute = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    assert _get(conn, 'a') == 1
    assert calls == [], 'memo hit still queried sqlite'


def test_memo_returns_equal_but_distinct_objects(tmp_path):
    """Why the memo holds blobs, not decoded objects.

    Every hit unpickles afresh, so a caller mutating what it got back cannot
    corrupt the next caller -- exactly as when the value comes from sqlite.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': [1, 2, 3]})
    first, second = _get(conn, 'a'), _get(conn, 'a')
    assert first == second
    assert first is not second


def test_memo_evicts_on_byte_budget(tmp_path, monkeypatch):
    memo = shm_cache.BlobMemo(max_bytes=300)
    for i in range(10):
        memo.put(f'k{i}', b'x' * 100)
    assert memo.total <= 300
    assert len(memo) == 3


def test_memo_eviction_falls_through_not_recomputes(tmp_path):
    conn = _make_cache(tmp_path / 'c.sqlite', {str(i): i for i in range(50)})
    conn._aimmd_memo = shm_cache.BlobMemo(max_bytes=200)
    for i in range(50):
        assert _get(conn, str(i)) == i
    for i in range(50):
        assert _get(conn, str(i)) == i, 'an evicted key must fall through, not vanish'


def test_memo_is_per_connection(tmp_path):
    c1 = _make_cache(tmp_path / 'a.sqlite', {'k': 'from-a'})
    c2 = _make_cache(tmp_path / 'b.sqlite', {'k': 'from-b'})
    assert _get(c1, 'k') == 'from-a'
    assert _get(c2, 'k') == 'from-b', 'memos cross-served between caches'


def test_memo_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv('AIMMD_GRAPH_MEMO_BYTES', '0')
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    assert getattr(conn, '_aimmd_memo', None) is None
    assert _get(conn, 'a') == 1


# ------------------------------------------- top-up source open (regress) --
def test_refresh_opens_source_without_uri_interpretation(tmp_path):
    """The top-up must not route the source path through SQLite URI parsing.

    Regression test for the production failure on JUPITER, where
    ``ATTACH DATABASE 'file:<path>?mode=ro'`` raised "unable to open database"
    and froze every replica at its staging row count for the whole run. ATTACH
    only honours a ``file:`` URI when SQLite's *global* ``SQLITE_USE_URI`` is
    on -- the per-connection open flag does not cover it -- so the URI form is
    unusable here.

    A ``#`` in the path makes the bug deterministic everywhere: under URI
    parsing everything from ``#`` on is a fragment, so the source resolves to a
    different, non-existent file. A plain path has no such reading.
    """
    conn = _make_cache(tmp_path / 'we#ird.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('b', pickle.dumps(2)))
    conn.commit()

    added = shm_cache.refresh_replicas()

    assert added.get(conn._aimmd_db_path) == 1, 'top-up must find the source'
    assert not getattr(conn, '_aimmd_topup_broken', False)
    n = conn._aimmd_replica.execute(
        'SELECT count(*) FROM graphs_cache').fetchone()[0]
    assert n == 2, 'the new row must be present in the replica itself'


def test_topup_failure_does_not_disable_write_through(tmp_path, monkeypatch):
    """A broken top-up must not take write-through down with it.

    These are independent mechanisms: the top-up copies rows *from* the real
    database, write-through pushes the trainer's own new graphs *into* the
    replica and needs no source access at all. In production a single ATTACH
    failure disabled both, so the replica stayed empty instead of at least
    accumulating what the trainer created.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None

    def _boom(*a, **k):
        raise sqlite3.OperationalError('unable to open database')
    monkeypatch.setattr(shm_cache, '_open_source', _boom)

    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('b', pickle.dumps(2)))
    conn.commit()
    shm_cache.refresh_replicas()
    assert getattr(conn, '_aimmd_topup_broken', False), 'top-up must be marked'
    assert not conn._aimmd_replica_frozen, 'that is not a space problem'

    assert shm_cache.write_through(conn, ['z'], [pickle.dumps(26)]) is True
    n = conn._aimmd_replica.execute(
        "SELECT count(*) FROM graphs_cache WHERE key='z'").fetchone()[0]
    assert n == 1, 'write-through must still reach the replica'


def test_out_of_space_still_blocks_write_through(tmp_path, monkeypatch):
    """The space freeze is shared on purpose -- tmpfs is one resource."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    monkeypatch.setattr(shm_cache, 'free_bytes', lambda *a, **k: 1)

    assert shm_cache.write_through(conn, ['z'], [pickle.dumps(26)]) is False
    assert conn._aimmd_replica_frozen
    assert _get(conn, 'a') == 1, 'reads must survive the freeze'


def test_topup_breakage_is_retried_next_cycle(tmp_path, monkeypatch):
    """A transient source failure must not latch for the process lifetime."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    shm_cache.stage_cache(conn)
    conn._aimmd_memo = None
    conn._aimmd_topup_broken = True

    shm_cache.stage_replicas()
    assert not getattr(conn, '_aimmd_topup_broken', False)

    conn.execute('INSERT INTO graphs_cache VALUES (?,?)', ('b', pickle.dumps(2)))
    conn.commit()
    assert shm_cache.refresh_replicas().get(conn._aimmd_db_path) == 1


# ------------------------------------------- staging must never be able to hang --
def test_stage_cache_aborts_on_deadline_and_falls_back(tmp_path, monkeypatch):
    """Staging must be bounded by a wall-clock deadline.

    The production hang was `conn.backup(dest, pages=-1)`: it holds one WAL
    read-snapshot for the entire uninterruptible copy, which blocks WAL
    checkpointing, so under concurrent writers the WAL grows without bound and
    the single copy never returns -- 6-12 h of silent trainer idle. The
    replacement must instead have a deadline it cannot exceed, and on the
    deadline it must degrade to "no replica, read the real DB" (return None,
    attach nothing, leave no partial files) rather than block.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {chr(97 + i): i for i in range(6)})
    monkeypatch.setattr(shm_cache, '_STAGE_DEADLINE_SECONDS', 0.0, raising=True)

    result = shm_cache.stage_cache(conn)

    assert result is None, 'a blown deadline must fall back, not stage'
    assert getattr(conn, '_aimmd_replica', None) is None
    # no half-written replica left behind anywhere under the shm root
    root = shm_cache.shm_root()
    leftovers = []
    for dirpath, _dirs, files in os.walk(root):
        leftovers += [f for f in files if '.partial' in f or f.endswith('.sqlite')]
    assert leftovers == [], f'staging left files behind: {leftovers}'


def test_stage_cache_completes_under_concurrent_writers(tmp_path):
    """A cache written continuously during staging must still stage, bounded.

    This is the scenario that hung in production (writers appending while the
    trainer copies). The replica must come out queryable and contain at least
    the rows present when staging began.
    """
    import threading

    path = tmp_path / 'c.sqlite'
    conn = _make_cache(path, {f'k{i}': i for i in range(200)})

    stop = threading.Event()

    def writer():
        w = sqlite3.connect(str(path), timeout=5.0)
        w.execute('PRAGMA busy_timeout=5000')
        i = 0
        while not stop.is_set():
            try:
                w.execute('INSERT OR REPLACE INTO graphs_cache VALUES (?,?)',
                          (f'w{i}', pickle.dumps(i)))
                w.commit()
                i += 1
            except sqlite3.OperationalError:
                pass
        w.close()

    threads = [threading.Thread(target=writer) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        t0 = __import__('time').monotonic()
        result = shm_cache.stage_cache(conn)
        elapsed = __import__('time').monotonic() - t0
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)

    assert result is not None and os.path.exists(result)
    assert elapsed < 60, f'staging took {elapsed:.1f}s under writers (should be seconds)'
    n = conn._aimmd_replica.execute(
        'SELECT count(*) FROM graphs_cache').fetchone()[0]
    assert n >= 200, f'replica has {n} rows, lost data present at stage start'


def test_stage_cache_never_raises_on_copy_failure(tmp_path, monkeypatch):
    """Any staging failure must degrade to None, never propagate."""
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})

    def boom(*a, **k):
        raise OSError('simulated tmpfs failure')
    monkeypatch.setattr(shm_cache, '_snapshot_copy', boom)

    result = shm_cache.stage_cache(conn)   # must not raise
    assert result is None
    assert getattr(conn, '_aimmd_replica', None) is None


def test_blown_deadline_does_not_retry_on_every_lookup(tmp_path, monkeypatch):
    """A failed stage must not re-arm itself for the very next lookup.

    `_aimmd_stage_pending` was cleared only on success, and graph_utils'
    load path retries stage_cache whenever it is still set. So one blown
    deadline turned into a fresh full staging attempt -- up to another whole
    deadline -- before EVERY subsequent cache lookup: the unbounded hang this
    module exists to prevent, merely chunked, and hidden because _warn_once
    suppresses the repeats. Bound it instead: give up the pending flag on
    failure, and stop re-arming after a few consecutive failures.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {chr(97 + i): i for i in range(6)})
    calls = {'n': 0}
    real = shm_cache._snapshot_copy

    def slow_fail(*a, **k):
        calls['n'] += 1
        raise TimeoutError('simulated blown deadline')
    monkeypatch.setattr(shm_cache, '_snapshot_copy', slow_fail)

    # arm it the way the trainer does: stage_replicas marks it pending, the
    # first cache lookup then triggers the copy
    shm_cache.stage_replicas()
    assert getattr(conn, '_aimmd_stage_pending', False) is True
    assert shm_cache.stage_cache(conn) is None
    assert calls['n'] == 1
    # the load path only retries while the pending flag is set
    assert getattr(conn, '_aimmd_stage_pending', False) is False, (
        'a failed stage left itself armed; the next lookup would restage')

    # and re-arming across rounds must be bounded, not forever
    for _ in range(10):
        shm_cache.stage_replicas()
        if getattr(conn, '_aimmd_stage_pending', False):
            shm_cache.stage_cache(conn)
    assert calls['n'] <= shm_cache._STAGE_MAX_ATTEMPTS, (
        f'{calls["n"]} staging attempts; must stop after '
        f'{shm_cache._STAGE_MAX_ATTEMPTS} consecutive failures')

    monkeypatch.setattr(shm_cache, '_snapshot_copy', real)


def test_owned_accounting_includes_the_replica_wal(tmp_path, monkeypatch):
    """tmpfs accounting must count the replica's -wal, not just the main file.

    The old backup() produced no -wal, so main-only accounting was exact. The
    copy can bring one across (up to 2.79 GB for the biggest production cache),
    and it is real tmpfs -- invisible to both AIMMD_SHM_MAX_BYTES and the npy
    budget handover, i.e. node memory over-commit in the unsafe direction.
    """
    conn = _make_cache(tmp_path / 'c.sqlite', {chr(97 + i): i for i in range(6)})
    dst = shm_cache.stage_cache(conn)
    assert dst is not None

    # simulate a replica that carries a WAL (busy TRUNCATE leaves one behind)
    with open(dst + '-wal', 'wb') as fh:
        fh.write(b'\0' * (3 * 1024 * 1024))

    shm_cache._recount_owned(dst)
    on_disk = os.path.getsize(dst) + os.path.getsize(dst + '-wal')
    assert shm_cache._OWNED[dst] == on_disk, (
        f'accounted {shm_cache._OWNED[dst]} but {on_disk} bytes are in tmpfs')


# ------------------------------------------------ reader role (the trainer) --
def test_reader_role_defaults_off_and_toggles():
    """Writers must be unaffected; only the trainer opts in."""
    assert shm_cache.reader_role() is False
    try:
        shm_cache.set_reader_role()
        assert shm_cache.reader_role() is True
    finally:
        shm_cache.set_reader_role(False)
    assert shm_cache.reader_role() is False
