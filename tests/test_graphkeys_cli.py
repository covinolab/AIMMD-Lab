"""The offline graph-key tools (aimmd.network.graph_keys_cli).

``backfill`` keys every trajectory a run has ingested from the trajectory
itself, ``verify`` reports, and ``gc`` deletes graphs no key file
references. They work on a toy run folder with the production layout:
initialARB, a chain path, the in-flight halves of a chain and the parts
of a free simulation, each with its states file; a stray trajectory without
one; and a toy graph cache in a real sqlite file.

The default-suite tests key frames with the toy descriptors function of
the test helpers, which is what
``graph_utils.atom_coordinate_descriptors_function`` computes (flattened
float32 coordinates) without needing torch_geometric. The test marked
``graph`` (``--rungraph``) runs the command line itself, with the default,
``atom_coordinate_descriptors_function``.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from MDAnalysis.coordinates.core import reader as Reader

from aimmd.cache.npy import save_npy
from aimmd.core.graphkey import graph_keys, keys_to_hex
from aimmd.network import graph_keys_cli as cli
from aimmd.path.utils import get_cache_fname
from tests._helpers_graphkeys import (SqliteToyCache, descriptors_function,
                                      forbid_descriptor_files)
from tests._helpers_unit import write_trajectory


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (folder, stem, frames, ingested frames); ingested None: no states file
LAYOUT = [
    ('initialARB', 'initial', 6, 6),
    ('chainR0', 'path000001', 9, 9),
    ('chainR0', 'back', 4, 4),
    ('chainR0', 'forw', 7, 5),           # 2 frames written, not ingested yet
    ('freeA', 'traj000001.part0000', 1, 1),
    ('freeA', 'traj000001.part0001', 80, 80),
    ('chainR0', 'stray', 3, None),       # not a trajectory of the run
]
#: Readable frames of the trajectories of the run (forw: 7, 5 ingested)
FRAMES = sum(n for *_, n, k in LAYOUT if k is not None)       # 107


def _write_run(root, seed=0):
    """The toy run folder; returns ``{name: trajectory}``."""
    run = os.path.join(root, 'run')
    trajs = {}
    for k, (folder, stem, n, ingested) in enumerate(LAYOUT):
        os.makedirs(os.path.join(run, folder), exist_ok=True)
        rng = np.random.default_rng(seed + k)
        positions = rng.uniform(0, 30, (n, 10, 3)).astype(np.float32)
        fname = write_trajectory(os.path.join(run, folder), stem=stem,
                                 positions=positions)
        if ingested is not None:
            save_npy(get_cache_fname(fname, 'states'),
                     np.array(['R'] * ingested))
        trajs[f'{folder}/{stem}'] = fname
    return run, trajs


def _rows(fname, scratch):
    """Coordinate rows of every frame, decoded from a copy of ``fname`` (so
    that MDAnalysis writes no offsets file into the run)."""
    os.makedirs(scratch, exist_ok=True)
    copy = os.path.join(scratch, os.path.basename(fname))
    shutil.copyfile(fname, copy)
    reader = Reader(copy)
    try:
        return descriptors_function(reader[:])
    finally:
        reader.close()


@pytest.fixture
def toy(tmp_path):
    """A toy run, its expected keys and rows, and a graph cache holding the
    graph of every ingested frame."""
    run, trajs = _write_run(str(tmp_path))
    rows = {name: _rows(fname, str(tmp_path / 'ref'))
            for name, fname in trajs.items()}
    keys = {name: graph_keys(r) for name, r in rows.items()}
    cache = SqliteToyCache(tmp_path / 'graphs_cache.sqlite')
    for name, (*_, ingested) in zip(trajs, LAYOUT):
        if ingested is not None:
            cache.add(rows[name])
    cache.conn.close()
    return dict(run=run, trajs=trajs, rows=rows, keys=keys,
                db=str(tmp_path / 'graphs_cache.sqlite'), tmp=tmp_path)


def _key_file(fname):
    return get_cache_fname(fname, 'graphkeys')


@pytest.fixture(autouse=True)
def _toy_rows(monkeypatch):
    """Key frames with the toy descriptors function: no torch_geometric."""
    monkeypatch.setattr(cli, '_default_rows_function',
                        lambda: descriptors_function)


def _backfill(toy, **kwargs):
    kwargs.setdefault('db', toy['db'])
    return cli.backfill([toy['run']], **kwargs)


def _digest(fname):
    with open(fname, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree(root):
    """``{relative path: sha256}`` of every file under ``root`` (lock files
    aside: whether a released lock file stays depends on filelock)."""
    result = {}
    for folder, _, files in os.walk(root):
        for name in files:
            if name.endswith('.lock'):
                continue
            fname = os.path.join(folder, name)
            result[os.path.relpath(fname, root)] = _digest(fname)
    return result


def _ingested(toy):
    return {name: fname for name, fname in toy['trajs'].items()
            if not name.endswith('stray')}


# --------------------------------------------------------------- backfill --
def test_trajectories_are_the_files_with_states(toy):
    found = cli.trajectories(toy['run'])
    assert found == sorted(_ingested(toy).values())


def test_backfill_keys_every_ingested_trajectory(toy):
    report = _backfill(toy)
    assert report['ok'], report['problems']
    for name, fname in _ingested(toy).items():
        stored = np.load(_key_file(fname))
        assert stored.dtype == np.uint8 and stored.shape[1] == 32
        assert np.array_equal(stored, toy['keys'][name]), name
        record = report['files'][fname]
        assert record['frames'] == record['keyed'] == len(stored)
        assert record['written'] and record['missing'] == 0
    # the stray trajectory has no states file: not ingested, not keyed
    assert not os.path.exists(_key_file(toy['trajs']['chainR0/stray']))
    totals = report['totals']
    assert totals['files'] == 6
    assert totals['frames'] == totals['keyed'] == FRAMES
    assert totals['ingested'] == FRAMES - 2
    assert totals['missing'] == 0
    assert totals['db_keys'] == FRAMES
    for name in ('index_s', 'cpu_s', 'cpu_ms_per_frame', 'keys_wall_s',
                 'db_scan_s', 'wall_s'):
        assert totals[name] >= 0, name


def test_key_files_are_npy_with_the_header_update_npy_expects(toy):
    _backfill(toy)
    fname = _key_file(toy['trajs']['freeA/traj000001.part0001'])
    with open(fname, 'rb') as fh:
        header = fh.read(128)
    assert header[:8] == b'\x93NUMPY\x01\x00'
    assert header.endswith(b'\n') and b"'descr': '|u1'" in header
    assert os.path.getsize(fname) == 128 + 80 * 32


def test_backfill_in_parallel_writes_the_same_files(toy, monkeypatch):
    _backfill(toy, jobs=1)
    serial = _tree(toy['run'])
    for fname in _ingested(toy).values():
        os.remove(_key_file(fname))
    monkeypatch.setattr(cli, '_CHUNK', 7)           # several tasks per file
    report = _backfill(toy, jobs=3)
    assert report['ok'], report['problems']
    assert report['totals']['jobs'] == 3
    assert _tree(toy['run']) == serial


def test_backfill_writes_no_offsets_file_into_the_run(toy):
    before = set(_tree(toy['run']))
    _backfill(toy, jobs=2)
    added = set(_tree(toy['run'])) - before
    assert added == {os.path.relpath(_key_file(f), toy['run'])
                     for f in _ingested(toy).values()}
    assert not any('offsets' in name for _, _, files in os.walk(toy['run'])
                   for name in files)


def test_only_missing_is_idempotent(toy):
    _backfill(toy)
    tree = _tree(toy['run'])
    stamps = {f: os.stat(_key_file(f)).st_mtime_ns
              for f in _ingested(toy).values()}
    report = _backfill(toy, only_missing=True)
    assert report['ok'], report['problems']
    assert report['totals']['keyed'] == 0
    assert report['totals']['written'] == 0
    assert _tree(toy['run']) == tree
    assert stamps == {f: os.stat(_key_file(f)).st_mtime_ns
                      for f in _ingested(toy).values()}


def test_only_missing_computes_exactly_the_missing_rows(toy):
    _backfill(toy)
    trajs = toy['trajs']
    part = _key_file(trajs['freeA/traj000001.part0001'])
    keys = np.load(part)
    keys[[3, 40]] = 0                                     # zero rows
    np.save(part, keys)
    path = _key_file(trajs['chainR0/path000001'])
    np.save(path, np.load(path)[:4])                      # short
    os.remove(_key_file(trajs['initialARB/initial']))     # missing

    report = _backfill(toy, only_missing=True)
    assert report['ok'], report['problems']
    keyed = {f: r['keyed'] for f, r in report['files'].items() if r['keyed']}
    assert keyed == {trajs['freeA/traj000001.part0001']: 2,
                     trajs['chainR0/path000001']: 5,
                     trajs['initialARB/initial']: 6}
    for name, fname in _ingested(toy).items():
        assert np.array_equal(np.load(_key_file(fname)), toy['keys'][name])


def test_backfill_replaces_stale_and_malformed_key_files(toy):
    _backfill(toy)
    trajs = toy['trajs']
    back = _key_file(trajs['chainR0/back'])
    keys = np.load(back)
    keys[1] = 7                                            # stale row
    np.save(back, keys)
    np.save(_key_file(trajs['initialARB/initial']), np.zeros((6, 3)))
    report = _backfill(toy)
    assert report['ok'], report['problems']
    assert report['files'][trajs['chainR0/back']]['stale'] == 1
    for name, fname in _ingested(toy).items():
        assert np.array_equal(np.load(_key_file(fname)), toy['keys'][name])


def test_a_crash_mid_write_leaves_no_partial_file(toy, monkeypatch):
    _backfill(toy)
    target = _key_file(toy['trajs']['freeA/traj000001.part0001'])
    keys = np.load(target)
    keys[5] = 0
    np.save(target, keys)
    before = _digest(target)
    real_save = np.save

    def crash(fname, array, *args, **kwargs):
        if str(fname).endswith(os.path.basename(target)):
            with open(fname, 'wb') as fh:              # half a file, then die
                fh.write(b'\x93NUMPY')
            raise OSError('disk full')
        return real_save(fname, array, *args, **kwargs)

    monkeypatch.setattr(np, 'save', crash)
    report = _backfill(toy, only_missing=True)
    assert not report['ok']
    assert 'disk full' in report['files'][toy['trajs'][
        'freeA/traj000001.part0001']]['error']
    assert _digest(target) == before                      # untouched
    monkeypatch.setattr(np, 'save', real_save)
    assert _backfill(toy, only_missing=True)['ok']
    assert np.array_equal(np.load(target),
                          toy['keys']['freeA/traj000001.part0001'])


def test_backfill_never_writes_the_db_or_opens_descriptors(toy, monkeypatch):
    for name in ('chainR0/path000001', 'freeA/traj000001.part0001'):
        save_npy(get_cache_fname(toy['trajs'][name], 'descriptors'),
                 toy['rows'][name])
    folder = os.path.dirname(toy['db'])
    before = {f: _digest(os.path.join(folder, f)) for f in os.listdir(folder)
              if f.startswith('graphs_cache')}
    with forbid_descriptor_files(monkeypatch) as opened:
        report = _backfill(toy, jobs=2)
    assert report['ok'], report['problems']
    assert opened == []
    after = {f: _digest(os.path.join(folder, f)) for f in os.listdir(folder)
             if f.startswith('graphs_cache')}
    assert after == before
    # only --verify-npy reads them
    with forbid_descriptor_files(monkeypatch) as opened:
        _backfill(toy, verify_npy=3)
    assert opened


def test_verify_npy_compares_keys_with_the_old_descriptor_rows(toy):
    names = ('chainR0/path000001', 'freeA/traj000001.part0001')
    for name in names:
        save_npy(get_cache_fname(toy['trajs'][name], 'descriptors'),
                 toy['rows'][name])
    report = _backfill(toy, verify_npy=5)
    assert report['ok'], report['problems']
    for name in names:
        spot = report['files'][toy['trajs'][name]]['spot']
        assert spot == {'checked': 5, 'mismatched': 0, 'skipped': 0}
    assert report['files'][toy['trajs']['chainR0/back']]['spot'] is None
    assert report['totals']['spot_checked'] == 10


def test_verify_npy_detects_a_corrupted_descriptor_row(toy):
    fname = toy['trajs']['chainR0/path000001']
    rows = toy['rows']['chainR0/path000001'].copy()
    rows[4, 7] += 1e-3
    save_npy(get_cache_fname(fname, 'descriptors'), rows)
    report = _backfill(toy, verify_npy=100)                 # every row
    assert not report['ok']
    assert report['files'][fname]['spot'] == {
        'checked': 9, 'mismatched': 1, 'skipped': 0}
    assert report['totals']['spot_mismatched'] == 1
    assert any('descriptor' in p for p in report['problems'])
    assert cli.main(['backfill', '--run', toy['run'],
                     '--verify-npy', '100']) != 0


def test_zero_descriptor_rows_are_skipped_in_the_spot_check(toy):
    fname = toy['trajs']['chainR0/back']
    rows = toy['rows']['chainR0/back'].copy()
    rows[2] = 0                                   # never computed in npy mode
    save_npy(get_cache_fname(fname, 'descriptors'), rows)
    report = _backfill(toy, verify_npy=100)
    assert report['ok'], report['problems']
    assert report['files'][fname]['spot'] == {
        'checked': 3, 'mismatched': 0, 'skipped': 1}


def _drop_graphs(db, keys):
    import sqlite3
    conn = sqlite3.connect(db)
    conn.executemany('DELETE FROM graphs_cache WHERE key = ?',
                     [(h,) for h in keys_to_hex(keys)])
    conn.commit()
    conn.close()


def test_backfill_reports_keys_without_a_graph(toy):
    _drop_graphs(toy['db'], toy['keys']['freeA/traj000001.part0001'][[0]])
    report = _backfill(toy)
    assert report['ok'], report['problems']              # 1 of 107: < 1 %
    assert report['totals']['missing'] == 1
    assert report['files'][toy['trajs'][
        'freeA/traj000001.part0001']]['missing'] == 1

    _drop_graphs(toy['db'], toy['keys']['chainR0/path000001'][:3])
    report = _backfill(toy)
    assert report['totals']['missing'] == 4
    assert not report['ok']
    assert any('no graph' in p for p in report['problems'])


def test_an_empty_or_absent_db_reports_no_failure(toy, tmp_path):
    report = cli.backfill([toy['run']], db=None)
    assert report['ok'] and report['totals']['missing'] is None
    empty = SqliteToyCache(tmp_path / 'empty.sqlite')
    empty.conn.close()
    report = cli.backfill([toy['run']], db=str(tmp_path / 'empty.sqlite'))
    assert report['ok'], report['problems']
    assert report['totals']['missing'] == FRAMES


def test_a_missing_db_file_is_an_error(toy, tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.backfill([toy['run']], db=str(tmp_path / 'nope.sqlite'))
    assert not os.path.exists(tmp_path / 'nope.sqlite')


def test_out_root_writes_the_key_files_elsewhere(toy, tmp_path):
    before = _tree(toy['run'])
    out = tmp_path / 'out'
    report = _backfill(toy, out_root=str(out))
    assert report['ok'], report['problems']
    assert _tree(toy['run']) == before
    for name, fname in _ingested(toy).items():
        rel = os.path.relpath(_key_file(fname), os.path.dirname(toy['run']))
        assert np.array_equal(np.load(out / rel), toy['keys'][name])


def test_an_unreadable_trajectory_is_reported_and_the_rest_keyed(toy):
    fname = toy['trajs']['chainR0/back']
    with open(fname, 'wb') as fh:
        fh.write(b'not an xtc file')
    report = _backfill(toy)
    assert not report['ok']
    assert report['files'][fname]['error']
    assert not os.path.exists(_key_file(fname))
    part = toy['trajs']['freeA/traj000001.part0001']
    assert np.array_equal(np.load(_key_file(part)),
                          toy['keys']['freeA/traj000001.part0001'])


def test_a_trajectory_with_nothing_ingested_blocks_nothing(toy):
    fname = os.path.join(toy['run'], 'freeA', 'traj000001.part0002.xtc')
    open(fname, 'wb').close()                  # a part MD has not written to
    save_npy(get_cache_fname(fname, 'states'), np.array([], dtype='<U1'))
    report = _backfill(toy)
    assert report['ok'], report['problems']
    assert report['files'][fname]['ingested'] == 0
    assert cli.verify([toy['run']], toy['db'])['ok']
    assert cli.gc([toy['run']], toy['db'])['ok']


def test_main_writes_the_report(toy, tmp_path):
    out = tmp_path / 'report.json'
    code = cli.main(['backfill', '--run', toy['run'], '--db', toy['db'],
                     '-j', '2', '--report', str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report['command'] == 'backfill'
    assert report['totals']['frames'] == FRAMES


@pytest.mark.graph
def test_the_command_line_keys_with_atom_coordinate_descriptors(toy):
    graph_utils = pytest.importorskip('aimmd.network.graph_utils')
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES='')
    out = subprocess.run(
        [sys.executable, '-W', 'ignore', '-m', 'aimmd.network.graph_keys_cli',
         'backfill', '--run', toy['run'], '--db', toy['db'], '-j', '2',
         '--verify-npy', '2'],
        env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    for name, fname in _ingested(toy).items():
        copy = os.path.join(toy['tmp'], 'ref', os.path.basename(fname))
        reader = Reader(copy)
        rows = graph_utils.atom_coordinate_descriptors_function(reader[:])
        reader.close()
        assert np.array_equal(np.load(_key_file(fname)), graph_keys(rows))
    assert 'missing' in out.stdout


# ----------------------------------------------------------------- verify --
def test_verify_passes_after_a_backfill(toy):
    _backfill(toy)
    report = cli.verify([toy['run']], toy['db'])
    assert report['ok'], report['problems']
    assert report['totals']['ingested'] == FRAMES - 2
    assert report['totals']['incomplete'] == 0
    assert report['totals']['missing'] == 0
    assert cli.main(['verify', '--run', toy['run'], '--db', toy['db']]) == 0


def test_verify_reports_missing_keys_and_graphs(toy):
    _backfill(toy)
    trajs = toy['trajs']
    os.remove(_key_file(trajs['initialARB/initial']))           # no key file
    part = _key_file(trajs['freeA/traj000001.part0001'])
    keys = np.load(part)
    keys[[2, 9]] = 0                                            # zero rows
    np.save(part, keys)
    path = _key_file(trajs['chainR0/path000001'])
    np.save(path, np.load(path)[:7])                            # short
    _drop_graphs(toy['db'], toy['keys']['chainR0/back'][:2])    # lost graphs
    tree, db = _tree(toy['run']), _digest(toy['db'])

    report = cli.verify([toy['run']], toy['db'])
    assert not report['ok']
    files = report['files']
    assert files[trajs['initialARB/initial']]['status'] == 'no key file'
    assert files[trajs['freeA/traj000001.part0001']]['status'] == 'zero rows'
    assert files[trajs['freeA/traj000001.part0001']]['zero'] == 2
    assert files[trajs['chainR0/path000001']]['status'] == 'short'
    assert files[trajs['chainR0/path000001']]['short'] == 2
    assert files[trajs['chainR0/back']]['missing'] == 2
    assert files[trajs['chainR0/forw']]['status'] == 'ok'   # 7 keys for 5
    assert report['totals']['incomplete'] == 3
    assert report['totals']['missing'] == 2
    assert _tree(toy['run']) == tree and _digest(toy['db']) == db
    assert cli.main(['verify', '--run', toy['run'], '--db', toy['db']]) == 1


# --------------------------------------------------------------------- gc --
def _sql(db, statement, *args):
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(statement, *args).fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


def _cached(db):
    return {key for (key,) in _sql(db, 'SELECT key FROM graphs_cache')}


def _add_unreferenced(db, count=5, seed=1):
    """Graphs of frames no run has: ``{hex key: blob}``."""
    import pickle
    rng = np.random.default_rng(seed)
    rows = rng.uniform(0, 30, (count, 30)).astype(np.float32)
    extra = {h: pickle.dumps(('graph', float(r.sum())))
             for h, r in zip(keys_to_hex(graph_keys(rows)), rows)}
    for h, blob in extra.items():
        _sql(db, 'INSERT INTO graphs_cache VALUES (?, ?)', (h, blob))
    return extra


@pytest.fixture
def wal(toy):
    """The toy cache in WAL mode, as production caches are."""
    assert _sql(toy['db'], 'PRAGMA journal_mode=WAL') == [('wal',)]
    return toy


@pytest.mark.parametrize('damage', ['no key file', 'zero rows', 'short'])
def test_gc_refuses_while_a_key_file_is_incomplete(wal, damage):
    toy = wal
    _backfill(toy)
    _add_unreferenced(toy['db'])
    fname = toy['trajs']['chainR0/path000001']
    if damage == 'no key file':
        os.remove(_key_file(fname))
    elif damage == 'zero rows':
        keys = np.load(_key_file(fname))
        keys[3] = 0
        np.save(_key_file(fname), keys)
    else:
        np.save(_key_file(fname), np.load(_key_file(fname))[:8])
    before = _cached(toy['db'])
    for apply in (False, True):
        report = cli.gc([toy['run']], toy['db'], apply=apply)
        assert report['refused'] and not report['ok']
        assert report['incomplete'] == {fname: damage}
    assert cli.main(['gc', '--run', toy['run'], '--db', toy['db'],
                     '--apply']) == 1
    assert _cached(toy['db']) == before


def test_gc_refuses_without_trajectories(toy, tmp_path):
    os.makedirs(tmp_path / 'empty')
    report = cli.gc([str(tmp_path / 'empty')], toy['db'], apply=True)
    assert report['refused'] and not report['ok']
    assert _cached(toy['db'])


def test_gc_dry_run_counts_and_changes_nothing(wal):
    toy = wal
    _backfill(toy)
    extra = _add_unreferenced(toy['db'])
    _drop_graphs(toy['db'], toy['keys']['chainR0/back'][:1])
    _sql(toy['db'], 'PRAGMA wal_checkpoint(TRUNCATE)')
    before = _digest(toy['db'])
    report = cli.gc([toy['run']], toy['db'])
    assert report['ok'] and not report['refused'] and not report['applied']
    assert report['referenced'] == FRAMES
    assert report['db_rows'] == FRAMES - 1 + 5
    assert report['unreferenced'] == 5
    assert report['unreferenced_bytes'] == sum(map(len, extra.values()))
    assert report['missing'] == 1
    assert report['deleted'] == 0
    assert _digest(toy['db']) == before
    assert len(_cached(toy['db'])) == FRAMES - 1 + 5


def test_gc_apply_deletes_exactly_the_unreferenced_rows(wal):
    toy = wal
    _backfill(toy)
    extra = _add_unreferenced(toy['db'])
    # a key file without a states file beside it still references its graphs
    stray = toy['trajs']['chainR0/stray']
    save_npy(_key_file(stray), toy['keys']['chainR0/stray'])
    cache = SqliteToyCache(toy['db'])
    cache.add(toy['rows']['chainR0/stray'])
    cache.conn.close()
    before = _cached(toy['db'])
    assert set(extra) < before

    report = cli.gc([toy['run']], toy['db'], apply=True, vacuum=True)
    assert report['ok'] and report['applied'] and report['vacuumed']
    assert report['deleted'] == report['unreferenced'] == 5
    assert _cached(toy['db']) == before - set(extra)
    assert set(keys_to_hex(toy['keys']['chainR0/stray'])) <= _cached(toy['db'])
    assert report['db_bytes_after'] <= report['db_bytes_before']
    again = cli.gc([toy['run']], toy['db'], apply=True)
    assert again['ok'] and again['unreferenced'] == again['deleted'] == 0


def test_gc_vacuums_only_with_apply(toy):
    with pytest.raises(ValueError):
        cli.gc([toy['run']], toy['db'], vacuum=True)
    assert cli.main(['gc', '--run', toy['run'], '--db', toy['db'],
                     '--vacuum']) == 2
