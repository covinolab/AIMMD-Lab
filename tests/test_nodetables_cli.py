"""``python -m aimmd.network.nodetables``: prefill, repack and verify.

Switching a running campaign to node tables needs the node-table series of
every trajectory of its runs before the workers restart (otherwise each
resumed worker featurizes its whole in-flight half again). These tests run
the command-line tool on a toy run folder (initial path, chain paths, the
halves of a shot in flight, free-simulation parts) through the recompute
route, which needs only numpy and MDAnalysis:

- prefill writes, for every trajectory with a states series, rows equal to
  featurizing it directly, as plain npy files, and never opens the stale
  ``*.descriptors.npy``;
- ``--only-missing`` fills only missing, short and zero rows and is
  idempotent; a crash, even a SIGKILL, mid-write leaves no partial series;
- ``--verify K`` and the ``verify`` command detect a corrupted row (exit
  status 1), and ``verify`` reports missing, short and zero rows;
- repack rewrites the rows into a wider layout without reading a trajectory,
  exactly, and a following prefill fills the frames that overflowed.

The extract route (rows from the graphs of an old graph cache) is in
test_nodetables_cli_extract.py (``--rungraph``).
"""
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from MDAnalysis.coordinates.core import reader

from aimmd.network.nodetables import MultiSystemNodeTableFeaturizer
from aimmd.network.nodetables import _cli, _tool
from aimmd.network.nodetables._featurizer import repack_rows
from tests._helpers_unit import forbid_opening
from tests._nodetables_run import (LAYOUT, UNTRACKED, bits, expected_rows,
                                   hidden_temporaries, load_report,
                                   make_campaign, npy_bytes, series_file,
                                   toy_featurizer, trajectories, write_params,
                                   write_run)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, reference_nodes,
                                   toy_universe)

WORKTREE = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(tmp_path):
    return make_campaign(tmp_path / 'campaign')


def _prefill(campaign, *options, report=None):
    argv = ['prefill', '--params', campaign.params, '--run', campaign.run,
            *options]
    if report:
        argv += ['--report', str(report)]
    return _cli.main(argv)


def _snapshot(paths):
    return {path: (os.stat(path).st_mtime_ns, Path(path).read_bytes())
            for path in paths}


def _check_complete(campaign, featurizer=None):
    featurizer = featurizer or campaign.featurizer
    for trajectory in trajectories(campaign.run):
        stored = np.load(series_file(trajectory, featurizer.series))
        assert np.array_equal(bits(stored),
                              bits(expected_rows(featurizer, trajectory)))


def _python(*args, **kwargs):
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), CUDA_VISIBLE_DEVICES='')
    return subprocess.run([sys.executable, *args], env=env,
                          capture_output=True, text=True, timeout=600,
                          **kwargs)


# ----------------------------------------------------------------------
# prefill

def test_prefill_writes_the_series_of_every_run_trajectory(campaign,
                                                           tmp_path, capsys):
    report = tmp_path / 'report.json'
    assert _prefill(campaign, report=report) == 0

    series = campaign.featurizer.series
    for trajectory in trajectories(campaign.run):
        fname = series_file(trajectory, series)
        rows = expected_rows(campaign.featurizer, trajectory)
        assert rows[:, 0].all()
        # a plain npy file, as np.save (and update_npy) write them
        assert Path(fname).read_bytes() == npy_bytes(rows)
    assert not os.path.exists(series_file(Path(campaign.run) / UNTRACKED,
                                          series))
    assert hidden_temporaries(campaign.run) == []

    result = load_report(report)
    assert result['command'] == 'prefill' and result['ok']
    assert result['series'] == series
    assert sorted(result['by_trajectory']) == sorted(
        trajectories(campaign.run))
    for trajectory, (n_frames, _) in zip(trajectories(campaign.run),
                                         LAYOUT.values()):
        item = result['by_trajectory'][trajectory]
        assert item['status'] == 'written'
        assert item['frames'] == item['computed'] == item['recomputed'] \
            == n_frames
        assert item['db_hits'] == 0 and item['empty_rows'] == 0
    totals = result['totals']
    assert totals['frames'] == sum(n for n, _ in LAYOUT.values())
    assert set(result['seconds']) >= {'read', 'featurize', 'write'}
    assert result['ms_per_frame']['recompute'] > 0
    out = capsys.readouterr().out
    assert series in out and 'written' in out


def test_prefill_never_opens_the_stale_coordinate_series(campaign):
    with pytest.MonkeyPatch.context() as patch:
        opened = forbid_opening(patch, '.descriptors.npy')
        assert _prefill(campaign, '--verify', '3') == 0
    assert opened == []
    _check_complete(campaign)


def test_prefill_only_missing_is_idempotent(campaign, tmp_path):
    assert _prefill(campaign) == 0
    series = campaign.featurizer.series
    files = [series_file(t, series) for t in trajectories(campaign.run)]
    before = _snapshot(files)

    report = tmp_path / 'again.json'
    assert _prefill(campaign, '--only-missing', report=report) == 0

    assert _snapshot(files) == before
    result = load_report(report)
    assert {item['status'] for item in result['files']} == {'complete'}
    assert result['totals']['computed'] == 0


def test_prefill_only_missing_fills_missing_short_and_zero_rows(campaign,
                                                                tmp_path):
    assert _prefill(campaign) == 0
    series = campaign.featurizer.series
    deleted, short, holed, *rest = trajectories(campaign.run)
    os.remove(series_file(deleted, series))
    rows = np.load(series_file(short, series))
    np.save(series_file(short, series), rows[:2])
    rows = np.load(series_file(holed, series))
    rows[[1, 3]] = 0
    np.save(series_file(holed, series), rows)
    untouched = _snapshot([series_file(t, series) for t in rest])

    report = tmp_path / 'fill.json'
    assert _prefill(campaign, '--only-missing', report=report) == 0

    _check_complete(campaign)
    assert _snapshot([series_file(t, series) for t in rest]) == untouched
    result = load_report(report)['by_trajectory']
    assert result[deleted]['computed'] == LAYOUT['initialARB/initial.xtc'][0]
    assert result[short]['computed'] == LAYOUT['chainR0/path000001.xtc'][0] - 2
    assert result[short]['kept'] == 2
    assert result[holed]['computed'] == 2
    assert {result[t]['status'] for t in rest} == {'complete'}


def test_prefill_overwrites_rows_without_only_missing(campaign):
    assert _prefill(campaign) == 0
    trajectory = trajectories(campaign.run)[0]
    fname = series_file(trajectory, campaign.featurizer.series)
    rows = np.load(fname)
    rows[0, 2] += 1.0
    np.save(fname, rows)
    assert _prefill(campaign) == 0
    _check_complete(campaign)


@pytest.mark.parametrize('existing', [False, True])
def test_a_crash_mid_write_leaves_no_partial_file(campaign, monkeypatch,
                                                  tmp_path, existing):
    series = campaign.featurizer.series
    trajectory = trajectories(campaign.run)[1]           # 9 frames
    fname = series_file(trajectory, series)
    if existing:                     # a short series, kept on failure
        rows = expected_rows(campaign.featurizer, trajectory)[:3]
        np.save(fname, rows)
        before = Path(fname).read_bytes()
    write_rows = _tool._write_rows
    calls = []

    def failing(temp, frames, rows):
        if Path(temp).name.startswith(f'.{Path(trajectory).name}.'):
            calls.append(len(frames))
            if len(calls) == 2:
                raise OSError('disk full (test)')
        return write_rows(temp, frames, rows)

    monkeypatch.setattr(_tool, '_write_rows', failing)
    report = tmp_path / 'crash.json'
    options = ['--chunk-frames', '2'] + (['--only-missing'] if existing
                                         else [])
    assert _prefill(campaign, *options, report=report) == 1

    assert len(calls) >= 2
    if existing:
        assert Path(fname).read_bytes() == before
    else:
        assert not os.path.exists(fname)
    assert hidden_temporaries(campaign.run) == []
    result = load_report(report)
    assert result['by_trajectory'][trajectory]['status'] == 'failed'
    assert 'disk full (test)' in result['by_trajectory'][trajectory]['error']
    others = [t for t in trajectories(campaign.run) if t != trajectory]
    assert all(result['by_trajectory'][t]['status'] == 'written'
               for t in others)

    monkeypatch.setattr(_tool, '_write_rows', write_rows)
    assert _prefill(campaign, '--only-missing') == 0
    _check_complete(campaign)


DRIVER = '''
import os, signal, sys
from aimmd.network.nodetables import _cli, _tool

write_rows = _tool._write_rows
calls = []


def killing(temp, frames, rows):
    calls.append(1)
    if len(calls) == 2:
        os.kill(os.getpid(), signal.SIGKILL)
    return write_rows(temp, frames, rows)


_tool._write_rows = killing
sys.exit(_cli.main(sys.argv[1:]))
'''


def test_a_killed_prefill_leaves_no_partial_file_and_reruns_in_parallel(
        campaign, tmp_path):
    series = campaign.featurizer.series
    argv = ['prefill', '--params', campaign.params, '--run', campaign.run,
            '--chunk-frames', '4']

    killed = _python('-c', DRIVER, *argv)
    assert killed.returncode == -signal.SIGKILL, killed.stderr

    # killed in the second chunk of the first trajectory (chainR0/back.xtc,
    # 5 frames): its first chunk went to a temporary file, never to a series
    assert hidden_temporaries(campaign.run)
    assert all(not os.path.exists(series_file(t, series))
               for t in trajectories(campaign.run))

    report = tmp_path / 'rerun.json'
    rerun = _python('-m', 'aimmd.network.nodetables', *argv, '-j', '2',
                    '--verify', '2', '--report', str(report))
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    _check_complete(campaign)
    assert hidden_temporaries(campaign.run) == []
    assert load_report(report)['jobs'] == 2


def test_prefill_verify_detects_a_corrupted_row(campaign, tmp_path, capsys):
    assert _prefill(campaign) == 0
    series = campaign.featurizer.series
    trajectory = trajectories(campaign.run)[2]
    fname = series_file(trajectory, series)
    rows = np.load(fname)
    rows.view(np.uint32)[4, 2 + 3 * 2] ^= 1         # one bit of node 2's x
    np.save(fname, rows)
    corrupted = Path(fname).read_bytes()
    capsys.readouterr()

    report = tmp_path / 'verify.json'
    assert _prefill(campaign, '--only-missing', '--verify', '100',
                    report=report) == 1

    assert Path(fname).read_bytes() == corrupted     # reported, not changed
    # the hint names the remedy for kept rows, not the graph cache
    error, = [line for line in capsys.readouterr().out.splitlines()
              if line.startswith('ERROR: rows differ')]
    assert 'without --only-missing' in error and '--db' not in error
    result = load_report(report)
    assert not result['ok']
    item = result['by_trajectory'][trajectory]
    assert item['status'] == 'mismatch' and item['mismatched_frames'] == [4]
    assert item['verified'] == LAYOUT['chainR0/path000002.xtc'][0]
    assert result['totals']['verify_mismatches'] == 1
    others = [t for t in trajectories(campaign.run) if t != trajectory]
    assert all(result['by_trajectory'][t]['status'] == 'complete'
               and result['by_trajectory'][t]['verify_mismatches'] == 0
               for t in others)

    report = tmp_path / 'check.json'
    assert _cli.main(['verify', '--params', campaign.params,
                      '--run', campaign.run, '--sample', '100',
                      '--report', str(report)]) == 1
    item = load_report(report)['by_trajectory'][trajectory]
    assert item['status'] == 'mismatch' and item['mismatched_frames'] == [4]

    # the remedy it names: a prefill that computes every row repairs it
    assert _prefill(campaign, '--verify', '100') == 0
    _check_complete(campaign)


def test_prefill_reports_frames_that_overflow_n_max(tmp_path, capsys):
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    counts = _all_node_counts(campaign)
    assert (counts > 40).any() and (counts <= 40).any()
    report = tmp_path / 'report.json'

    assert _prefill(campaign, report=report) == 1

    result = load_report(report)
    assert result['totals']['empty_rows'] == int((counts > 40).sum())
    assert not result['ok']
    out = capsys.readouterr().out
    assert 'repack --n-max' in out


def _truncate_last_frame(trajectory):
    """Cut the last frame short, as a job killed while mdrun writes does."""
    size = os.path.getsize(trajectory)
    with open(trajectory, 'r+b') as file:
        file.truncate(size - 200)


def test_a_truncated_last_frame_is_left_out_as_aimmd_does(campaign, tmp_path,
                                                          capsys):
    """AIMMD reads only the readable frames of a trajectory (MDA_CACHE,
    count_safe_frames); prefill and verify cover the same frames and report
    the cut one, instead of an empty row blamed on n_max."""
    from aimmd._config import MDA_CACHE
    victim = trajectories(campaign.run)[3]               # chainR0/back.xtc
    _truncate_last_frame(victim)
    n_frames = LAYOUT['chainR0/back.xtc'][0] - 1
    assert len(MDA_CACHE.get(victim)) == n_frames
    MDA_CACHE.clear()
    report = tmp_path / 'report.json'

    assert _prefill(campaign, '--verify', '9', report=report) == 0

    rows = np.load(series_file(victim, campaign.featurizer.series))
    expected = expected_rows(campaign.featurizer, victim)
    assert len(rows) == len(expected) == n_frames
    assert np.array_equal(bits(rows), bits(expected))
    item = load_report(report)['by_trajectory'][victim]
    assert item['status'] == 'written' and item['frames'] == n_frames
    assert item['unreadable_frames'] == 1 and item['empty_rows'] == 0
    out = capsys.readouterr().out
    assert 'unreadable' in out and 'repack' not in out

    assert _prefill(campaign, '--only-missing') == 0
    report = tmp_path / 'verify.json'
    assert _cli.main(['verify', '--params', campaign.params, '--run',
                      campaign.run, '--sample', '9', '--report',
                      str(report)]) == 0
    item = load_report(report)['by_trajectory'][victim]
    assert item['status'] == 'complete' and item['unreadable_frames'] == 1


def test_a_chunk_with_unreadable_frames_fails(campaign):
    """Frames that cannot be read never become empty rows."""
    victim = trajectories(campaign.run)[3]
    _truncate_last_frame(victim)
    temp = str(tmp_series := Path(campaign.folder) / 'chunk.npy')
    _tool._create(temp, 5, campaign.featurizer.width)
    task = dict(trajectory=victim, system_id=None, frames=np.arange(5),
                db=None, temporary=str(tmp_series))
    with pytest.raises(OSError, match='4 of the 5 frames'):
        _tool._prefill_chunk(campaign.featurizer, task)


# ----------------------------------------------------------------------
# verify

def test_verify_reports_missing_short_and_zero_rows(campaign, tmp_path):
    assert _prefill(campaign) == 0
    series = campaign.featurizer.series
    deleted, short, holed, wide, *rest = trajectories(campaign.run)
    os.remove(series_file(deleted, series))
    rows = np.load(series_file(short, series))
    np.save(series_file(short, series), rows[:2])
    rows = np.load(series_file(holed, series))
    rows[[1, 3]] = 0
    np.save(series_file(holed, series), rows)
    np.save(series_file(wide, series), np.zeros((4, 10), np.float32))

    report = tmp_path / 'verify.json'
    with pytest.MonkeyPatch.context() as patch:
        opened = forbid_opening(patch, '.descriptors.npy')
        status = _cli.main(['verify', '--params', campaign.params,
                            '--run', campaign.run, '--report', str(report)])
    assert status == 1 and opened == []

    result = load_report(report)
    assert result['command'] == 'verify' and not result['ok']
    items = result['by_trajectory']
    assert items[deleted]['status'] == 'missing'
    assert items[deleted]['missing_rows'] == \
        LAYOUT['initialARB/initial.xtc'][0]
    assert items[short]['status'] == 'incomplete'
    assert items[short]['missing_rows'] == \
        LAYOUT['chainR0/path000001.xtc'][0] - 2
    assert items[holed]['status'] == 'incomplete'
    assert items[holed]['zero_rows'] == 2 and items[holed]['missing_rows'] == 0
    assert items[wide]['status'] == 'layout'
    assert all(items[t]['status'] == 'complete' for t in rest)
    # rows of another layout are missing rows too
    assert items[wide]['missing_rows'] == LAYOUT['chainR0/back.xtc'][0]
    assert result['totals']['missing_rows'] == \
        LAYOUT['initialARB/initial.xtc'][0] + \
        LAYOUT['chainR0/path000001.xtc'][0] - 2 + \
        LAYOUT['chainR0/back.xtc'][0]


def test_verify_passes_on_a_complete_run(campaign, tmp_path):
    assert _prefill(campaign) == 0
    report = tmp_path / 'verify.json'
    assert _cli.main(['verify', '--params', campaign.params, '--run',
                      campaign.run, '--sample', '3', '--report',
                      str(report)]) == 0
    result = load_report(report)
    assert result['ok']
    assert {item['status'] for item in result['files']} == {'complete'}
    assert result['totals']['verified'] == sum(
        min(3, n) for n, _ in LAYOUT.values())


# ----------------------------------------------------------------------
# repack

def _all_node_counts(campaign):
    universe = toy_universe()
    counts = []
    for trajectory in trajectories(campaign.run):
        frames = reader(trajectory)
        counts += [len(reference_nodes(universe, ts.positions)[1])
                   for ts in frames]
        frames.close()
    return np.array(counts)


def test_repack_is_exact_and_reads_no_trajectory(tmp_path, monkeypatch):
    narrow_n_max = 40
    campaign = make_campaign(tmp_path / 'campaign', n_max=narrow_n_max)
    counts = _all_node_counts(campaign)
    assert _prefill(campaign) == 1                     # overflowing frames
    narrow = campaign.featurizer
    wide_n_max = int(counts.max()) + 4
    wide = toy_featurizer(campaign.folder, wide_n_max)

    def no_trajectories(*args, **kwargs):
        raise AssertionError('repack read a trajectory')

    monkeypatch.setattr(_tool, '_open_reader', no_trajectories)
    report = tmp_path / 'repack.json'
    assert _cli.main(['repack', '--params', campaign.params, '--run',
                      campaign.run, '--n-max', str(wide_n_max),
                      '--report', str(report)]) == 0
    monkeypatch.undo()

    result = load_report(report)
    assert result['source_series'] == narrow.series
    assert result['series'] == wide.series
    empty = 0
    for trajectory in trajectories(campaign.run):
        old = np.load(series_file(trajectory, narrow.series))
        new = np.load(series_file(trajectory, wide.series))
        assert Path(series_file(trajectory, wide.series)).read_bytes() == \
            npy_bytes(repack_rows(old, wide_n_max))
        expected = expected_rows(wide, trajectory)
        fits = old[:, 0] != 0
        assert np.array_equal(bits(new[fits]), bits(expected[fits]))
        assert not new[~fits].any()
        empty += int((~fits).sum())
        assert result['by_trajectory'][trajectory]['status'] == 'written'
    assert empty == int((counts > narrow_n_max).sum()) > 0
    assert result['totals']['empty_rows'] == empty
    assert hidden_temporaries(campaign.run) == []

    # with the wider featurizer pinned, prefill fills only the empty rows
    write_params(campaign.folder, wide_n_max)
    report = tmp_path / 'fill.json'
    assert _prefill(campaign, '--only-missing', report=report) == 0
    _check_complete(campaign, wide)
    assert load_report(report)['totals']['computed'] == empty


def test_repack_from_an_older_n_max_and_existing_targets(tmp_path):
    campaign = make_campaign(tmp_path / 'campaign', n_max=56)
    assert _prefill(campaign) == 0
    old = campaign.featurizer
    # the params file already holds the new capacity
    write_params(campaign.folder, 64)
    new = toy_featurizer(campaign.folder, 64)
    argv = ['repack', '--params', campaign.params, '--run', campaign.run,
            '--n-max', '64']
    assert _cli.main(argv) == 2                        # nothing to repack
    assert _cli.main(argv + ['--from-n-max', '56']) == 0
    _check_complete(campaign, new)

    target = series_file(trajectories(campaign.run)[0], new.series)
    rows = np.load(target)
    rows[0, 2] += 1
    np.save(target, rows)
    report = tmp_path / 'again.json'
    assert _cli.main(argv + ['--from-n-max', '56', '--report',
                             str(report)]) == 0
    assert load_report(report)['by_trajectory'][
        trajectories(campaign.run)[0]]['status'] == 'exists'
    assert np.array_equal(np.load(target), rows)
    assert _cli.main(argv + ['--from-n-max', '56', '--overwrite']) == 0
    _check_complete(campaign, new)
    assert os.path.exists(series_file(trajectories(campaign.run)[0],
                                      old.series))


# ----------------------------------------------------------------------
# params and runs

def test_a_params_file_without_featurizer_is_refused(campaign, tmp_path,
                                                     capsys):
    params = Path(campaign.folder) / 'legacy.py'
    params.write_text("descriptors_function = None\n")
    assert _cli.main(['prefill', '--params', str(params), '--run',
                      campaign.run]) == 2
    err = capsys.readouterr().err
    assert 'NodeTableFeaturizer' in err and "GRAPH_INPUT" in err


def test_a_params_file_in_sqlite_mode_is_not_executed(campaign, capsys):
    """A params file in 'sqlite' mode opens its graph cache when it runs
    (init_db: read-write, may create the file or -wal and -shm files): the
    tools refuse it before executing anything."""
    params = Path(campaign.folder) / 'legacy.py'
    marker = Path(campaign.folder) / 'graphs_cache.sqlite'
    params.write_text(
        "GRAPH_INPUT = 'sqlite'                 # 'nodetables' | 'sqlite'\n"
        "if GRAPH_INPUT == 'sqlite':\n"
        f"    open({str(marker)!r}, 'w').close()   # stands for init_db\n")
    for command in ('prefill', 'verify', 'repack'):
        argv = [command, '--params', str(params), '--run', campaign.run]
        if command == 'repack':
            argv += ['--n-max', '1024']
        assert _cli.main(argv) == 2
        err = capsys.readouterr().err
        assert "GRAPH_INPUT = 'sqlite'" in err and "'nodetables'" in err
    assert not marker.exists()

    # a GRAPH_INPUT that is not a plain string is left to the import
    params.write_text("import os\n"
                      "GRAPH_INPUT = os.environ.get('MODE', 'sqlite')\n")
    assert _cli.main(['verify', '--params', str(params), '--run',
                      campaign.run]) == 2
    assert 'defines no NodeTableFeaturizer' in capsys.readouterr().err


def test_a_mismatched_pinned_series_is_refused(campaign, capsys):
    params = write_params(campaign.folder, series='descriptors-gn0000000000',
                          name='pinned.py')
    assert _cli.main(['prefill', '--params', params, '--run',
                      campaign.run]) == 2
    assert 'descriptors-gn0000000000' in capsys.readouterr().err
    assert not any(Path(campaign.run).rglob('*descriptors-gn*'))


def test_a_missing_run_folder_is_refused(campaign, capsys):
    assert _cli.main(['prefill', '--params', campaign.params, '--run',
                      campaign.run + '-nothere']) == 2
    assert 'nothere' in capsys.readouterr().err


def test_relative_paths_are_resolved_before_the_params_import(campaign,
                                                              monkeypatch):
    monkeypatch.chdir(Path(campaign.folder).parent)
    params = os.path.relpath(campaign.params)
    run = os.path.relpath(campaign.run)
    assert _cli.main(['prefill', '--params', params, '--run', run]) == 0
    _check_complete(campaign)


MULTI_PARAMS = '''
import MDAnalysis as mda
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      NodeTableFeaturizer)


def _featurizer(n_max):
    return NodeTableFeaturizer(
        mda.Universe('toy.gro', to_guess=['types', 'bonds']),
        {system!r}, {environment!r}, {atom_types!r}, cutoff={cutoff!r},
        n_max=n_max)


FEATURIZERS = MultiSystemNodeTableFeaturizer({{
    'lig1': _featurizer({n_max_1}), 'lig2': _featurizer({n_max_2})}})
descriptors_series = FEATURIZERS.series
'''


def _write_multi_params(params, n_max_1=64, n_max_2=80):
    """The multi-system params file: two systems with their own n_max."""
    Path(params).write_text(MULTI_PARAMS.format(
        system=SYSTEM_SELECTION, environment=ENVIRONMENT_SELECTION,
        atom_types=ATOM_TYPES, cutoff=CUTOFF, n_max_1=n_max_1,
        n_max_2=n_max_2))
    return MultiSystemNodeTableFeaturizer({
        'lig1': toy_featurizer(Path(params).parent, n_max_1),
        'lig2': toy_featurizer(Path(params).parent, n_max_2)})


def test_prefill_and_verify_a_multi_system_run(tmp_path):
    campaign = make_campaign(tmp_path / 'campaign')
    folder = Path(campaign.folder)
    params = folder / 'multi.py'
    featurizers = _write_multi_params(params)
    run = folder / 'multi_run'
    layout = dict(list(LAYOUT.items())[:3])
    write_run(run / 'lig1', layout)
    write_run(run / 'lig2', layout, untracked=None)

    assert _cli.main(['prefill', '--params', str(params), '--run',
                      str(run)]) == 0

    for system_id in ('lig1', 'lig2'):
        for trajectory in trajectories(run / system_id, layout):
            stored = np.load(series_file(trajectory, featurizers.series))
            assert stored.shape[1] == featurizers[system_id].width
            assert np.array_equal(bits(stored), bits(expected_rows(
                featurizers[system_id], trajectory)))
    assert _cli.main(['verify', '--params', str(params), '--run', str(run),
                      '--sample', '2']) == 0


def test_repack_a_multi_system_run_from_its_n_max_per_system(tmp_path):
    """Each system has its own n_max. Once the params file holds the new
    capacity, --from-n-max names the old one per system (SYSTEM_ID=M; a
    plain M is the default of the systems not named)."""
    campaign = make_campaign(tmp_path / 'campaign')
    folder = Path(campaign.folder)
    params = folder / 'multi.py'
    old = _write_multi_params(params, 64, 80)
    run = folder / 'multi_run'
    layout = dict(list(LAYOUT.items())[:3])
    write_run(run / 'lig1', layout)
    write_run(run / 'lig2', layout, untracked=None)
    assert _cli.main(['prefill', '--params', str(params), '--run',
                      str(run)]) == 0
    new = _write_multi_params(params, 96, 96)       # the new capacity
    argv = ['repack', '--params', str(params), '--run', str(run),
            '--n-max', '96']

    # one value for every system: no such series (lig2 had 80)
    assert _cli.main(argv + ['--from-n-max', '64']) == 2
    report = tmp_path / 'repack.json'
    assert _cli.main(argv + ['--from-n-max', 'lig1=64', '--from-n-max',
                             'lig2=80', '--report', str(report)]) == 0

    result = load_report(report)
    assert (result['source_series'], result['series']) == (old.series,
                                                            new.series)
    for system_id in ('lig1', 'lig2'):
        for trajectory in trajectories(run / system_id, layout):
            rows = np.load(series_file(trajectory, old.series))
            assert Path(series_file(trajectory, new.series)).read_bytes() \
                == npy_bytes(repack_rows(rows, 96))
    assert _cli.main(['verify', '--params', str(params), '--run', str(run),
                      '--sample', '2']) == 0
    # a default and an override name the same source series
    report = tmp_path / 'again.json'
    assert _cli.main(argv + ['--from-n-max', '80', '--from-n-max',
                             'lig1=64', '--report', str(report)]) == 0
    assert load_report(report)['source_series'] == old.series
    # an unknown system, or SYSTEM_ID=M for a single system, is refused
    assert _cli.main(argv + ['--from-n-max', 'lig3=64']) == 2
    assert _cli.main(['repack', '--params', campaign.params, '--run',
                      campaign.run, '--n-max', '96', '--from-n-max',
                      'lig1=64']) == 2


def test_source_n_max_refuses_repeated_or_bad_values():
    featurizers = SimpleNamespace(system_ids=['lig1', 'lig2'])
    assert _tool._source_n_max(['64'], featurizers) == 64
    assert _tool._source_n_max(['lig2=80'], featurizers) == {'lig2': 80}
    for values in (['64', '80'], ['lig1=64', 'lig1=80'], ['lig1=x'],
                   ['0']):
        with pytest.raises(_tool.UsageError):
            _tool._source_n_max(values, featurizers)


def test_find_trajectories_follows_the_states_series(campaign):
    found = _tool.find_trajectories([campaign.run], campaign.featurizer)
    assert [trajectory for trajectory, _ in found] == sorted(
        trajectories(campaign.run))
    assert {system_id for _, system_id in found} == {None}


def test_an_edited_params_file_is_read_again(campaign):
    # same size, same second: a bytecode cache would serve the old file
    write_params(campaign.folder, 56)
    assert _tool.load_featurizer(campaign.params).featurizer.n_max == 56
    write_params(campaign.folder, 64)
    loaded = _tool.load_featurizer(campaign.params)
    assert loaded.featurizer.n_max == 64
    assert loaded.pinned == loaded.series
    assert not (Path(campaign.folder) / '__pycache__').exists()
