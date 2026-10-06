"""``prefill --db``: node-table rows from the graphs of an old graph cache.

A campaign in coordinate-descriptor mode has cached the graph of every frame
in a global sqlite graph cache, under the hash of the frame's coordinates.
Prefilling from it (decode the frame, compute the key, look the graph up,
`NodeTableFeaturizer.row_from_graph`) is several times faster than
featurizing. These tests build such a cache for a toy run with
`process_descriptors_pyg`, as ingestion did, and check that

- the extract route and the recompute route give the same rows, bit for bit,
  serially and in parallel, with frames missing from the cache featurized;
- the cache is never written (not even -wal or -shm files next to it) and the
  stale ``*.descriptors.npy`` is never opened;
- a wrong graph in the cache is caught by ``--verify`` (exit status 1, the
  file is not installed), and graphs of another definition are not used.

Needs torch_geometric (``--rungraph``).
"""
import hashlib
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from aimmd.network.nodetables import _cli, _tool
from tests._helpers_unit import forbid_opening
from tests._nodetables_run import (LAYOUT, bits, expected_rows,
                                   hidden_temporaries, load_report,
                                   make_campaign, series_file, trajectories)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION)

pytestmark = pytest.mark.graph

WORKTREE = Path(__file__).resolve().parents[1]
N_FRAMES = sum(n for n, _ in LAYOUT.values())


def _coordinates(campaign, trajectory):
    """Coordinate descriptors of a trajectory, as ingestion computed them."""
    from aimmd.network import graph_utils
    universe = mda.Universe(str(Path(campaign.folder) / 'toy.gro'),
                            trajectory)
    return graph_utils.atom_coordinate_descriptors_function(
        universe.trajectory)


def _key(row):
    from aimmd.network import graph_utils
    return graph_utils.get_stable_hash(row)


def _build_cache(campaign, folder,
                 environment_selection=ENVIRONMENT_SELECTION):
    """The graph cache of the campaign's frames, as ingestion wrote it."""
    from aimmd.network import graph_utils
    folder.mkdir()
    database = folder / 'graphs_cache.sqlite'
    universe = mda.Universe(str(Path(campaign.folder) / 'toy.gro'),
                            to_guess=['types', 'bonds'])
    connection = graph_utils.init_db(str(database))
    for trajectory in trajectories(campaign.run):
        graph_utils.process_descriptors_pyg(
            _coordinates(campaign, trajectory), mdanalysis_universe=universe,
            system_selection=SYSTEM_SELECTION,
            environment_selection=environment_selection, cutoff=CUTOFF,
            conn=connection, atom_types=ATOM_TYPES)
    graph_utils.flush_pending_writes(connection)
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    for suffix in ('-wal', '-shm'):
        if os.path.exists(f'{database}{suffix}'):
            os.remove(f'{database}{suffix}')
    with sqlite3.connect(database) as check:
        assert check.execute('SELECT COUNT(*) FROM graphs_cache'
                             ).fetchone()[0] == N_FRAMES
    check.close()
    return str(database)


@pytest.fixture
def cached(tmp_path):
    """A toy campaign and the graph cache its frames went through."""
    pytest.importorskip('torch_geometric')
    campaign = make_campaign(tmp_path / 'campaign')
    return campaign, _build_cache(campaign, tmp_path / 'cache')


def _edit_cache(database, statement, *parameters):
    """Change the cache; it stays in WAL mode, as AIMMD's caches are."""
    connection = sqlite3.connect(database)
    connection.execute(statement, parameters)
    connection.commit()
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    for suffix in ('-wal', '-shm'):
        assert not os.path.exists(f'{database}{suffix}')


def _state(database):
    """Content and metadata of the graph cache and its folder."""
    folder = Path(database).parent
    return (hashlib.sha256(Path(database).read_bytes()).hexdigest(),
            os.stat(database).st_mtime_ns, sorted(os.listdir(folder)))


def _python(*args):
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), CUDA_VISIBLE_DEVICES='')
    return subprocess.run([sys.executable, *args], env=env,
                          capture_output=True, text=True, timeout=600)


def test_the_extract_and_recompute_routes_agree_bitwise(cached, tmp_path):
    campaign, database = cached
    series = campaign.featurizer.series
    # three frames the cache does not have
    forw = trajectories(campaign.run)[4]                  # chainR0/forw.xtc
    for row in _coordinates(campaign, forw)[:3]:
        _edit_cache(database, 'DELETE FROM graphs_cache WHERE key = ?',
                    _key(row))
    before = _state(database)

    report = tmp_path / 'extract.json'
    with pytest.MonkeyPatch.context() as patch:
        opened = forbid_opening(patch, '.descriptors.npy')
        assert _cli.main(['prefill', '--params', campaign.params, '--run',
                          campaign.run, '--db', database, '--verify', '2',
                          '--report', str(report)]) == 0
    assert opened == []
    assert _state(database) == before                    # never written

    result = load_report(report)
    assert result['db'] == [database]
    totals = result['totals']
    assert totals['db_hits'] == N_FRAMES - 3
    assert totals['recomputed'] == 3 and totals['db_unusable'] == 0
    assert result['by_trajectory'][forw]['recomputed'] == 3
    assert result['ms_per_frame']['extract'] > 0
    extracted = {}
    for trajectory in trajectories(campaign.run):
        fname = series_file(trajectory, series)
        extracted[trajectory] = Path(fname).read_bytes()
        assert np.array_equal(bits(np.load(fname)), bits(
            expected_rows(campaign.featurizer, trajectory)))

    # the recompute route writes the same bytes
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run]) == 0
    for trajectory in trajectories(campaign.run):
        assert Path(series_file(trajectory, series)).read_bytes() == \
            extracted[trajectory]


def test_parallel_extraction_from_a_read_only_cache_folder(cached, tmp_path):
    campaign, database = cached
    folder = Path(database).parent
    before = _state(database)
    report = tmp_path / 'parallel.json'
    os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
    try:
        run = _python('-m', 'aimmd.network.nodetables', 'prefill',
                      '--params', campaign.params, '--run', campaign.run,
                      '--db', database, '-j', '2', '--chunk-frames', '3',
                      '--report', str(report))
    finally:
        os.chmod(folder, stat.S_IRWXU)
    assert run.returncode == 0, run.stdout + run.stderr
    assert _state(database) == before
    result = load_report(report)
    assert result['jobs'] == 2 and result['totals']['db_hits'] == N_FRAMES
    for trajectory in trajectories(campaign.run):
        stored = np.load(series_file(trajectory, campaign.featurizer.series))
        assert np.array_equal(bits(stored), bits(
            expected_rows(campaign.featurizer, trajectory)))
    assert hidden_temporaries(campaign.run) == []


def test_verify_catches_a_wrong_graph_in_the_cache(cached, tmp_path):
    campaign, database = cached
    path = trajectories(campaign.run)[1]                  # 9 frames
    rows = _coordinates(campaign, path)
    # frame 5 gets the graph of frame 2
    connection = sqlite3.connect(database)
    blob, = connection.execute('SELECT data FROM graphs_cache WHERE key = ?',
                               (_key(rows[2]),)).fetchone()
    connection.close()
    _edit_cache(database, 'UPDATE graphs_cache SET data = ? WHERE key = ?',
                blob, _key(rows[5]))

    report = tmp_path / 'verify.json'
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run, '--db', database, '--verify', '100',
                      '--report', str(report)]) == 1

    result = load_report(report)
    item = result['by_trajectory'][path]
    assert item['status'] == 'mismatch' and item['mismatched_frames'] == [5]
    assert not os.path.exists(series_file(path, campaign.featurizer.series))
    assert result['totals']['mismatch'] == 1
    assert result['totals']['written'] == len(LAYOUT) - 1
    assert hidden_temporaries(campaign.run) == []


def test_a_cache_of_other_settings_is_caught_without_verify(cached,
                                                          tmp_path):
    """Cache keys hash coordinates only, and a graph of another environment
    selection is a valid node table. With --db, prefill therefore verifies
    a few frames per file by default: such a cache is caught (exit status
    1, nothing installed) without --verify."""
    campaign, _ = cached
    other = _build_cache(campaign, tmp_path / 'other',
                         ENVIRONMENT_SELECTION.replace('5.0', '6.0'))
    report = tmp_path / 'report.json'

    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run, '--db', other, '--report',
                      str(report)]) == 1

    result = load_report(report)
    assert result['verify'] == _tool.DEFAULT_DB_VERIFY > 0
    assert result['totals']['db_hits'] == N_FRAMES
    assert result['totals']['mismatch'] == len(LAYOUT)
    assert not any(os.path.exists(series_file(t, campaign.featurizer.series))
                   for t in trajectories(campaign.run))
    # an explicit --verify 0 still skips the check
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run, '--db', other, '--verify', '0']) == 0


def test_graphs_of_another_definition_are_featurized_instead(cached,
                                                             tmp_path):
    from aimmd.network import graph_utils
    campaign, database = cached
    path = trajectories(campaign.run)[2]
    key = _key(_coordinates(campaign, path)[0])
    connection = sqlite3.connect(database)
    blob, = connection.execute('SELECT data FROM graphs_cache WHERE key = ?',
                               (key,)).fetchone()
    connection.close()
    graph = graph_utils._decode(blob)
    import torch
    node_attrs = graph['node_attrs']     # a 7-type table instead of 6
    graph['node_attrs'] = torch.cat(
        [node_attrs, torch.zeros(len(node_attrs), 1)], dim=1)
    _edit_cache(database, 'UPDATE graphs_cache SET data = ? WHERE key = ?',
                graph_utils._encode(graph, 'lz4'), key)

    report = tmp_path / 'report.json'
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run, '--db', database, '--report',
                      str(report)]) == 0
    item = load_report(report)['by_trajectory'][path]
    assert item['db_unusable'] == 1 and item['recomputed'] == 1
    stored = np.load(series_file(path, campaign.featurizer.series))
    assert np.array_equal(bits(stored),
                          bits(expected_rows(campaign.featurizer, path)))


def test_a_write_ahead_log_is_reported(cached, capsys):
    campaign, database = cached
    Path(f'{database}-wal').write_bytes(b'\0' * 64)
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run, '--db', database]) == 0
    assert f'WARNING: {database}-wal is not empty' in capsys.readouterr().out


def test_a_missing_or_foreign_cache_is_refused(cached, tmp_path, capsys):
    campaign, database = cached
    argv = ['prefill', '--params', campaign.params, '--run', campaign.run]
    assert _cli.main(argv + ['--db', str(tmp_path / 'none.sqlite')]) == 2
    other = tmp_path / 'other.sqlite'
    sqlite3.connect(other).execute('CREATE TABLE x (y)').connection.close()
    assert _cli.main(argv + ['--db', str(other)]) == 2
    assert 'no graphs_cache table' in capsys.readouterr().err
    assert hidden_temporaries(campaign.run) == []
