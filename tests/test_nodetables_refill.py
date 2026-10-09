"""Refilling a node-table series that trajectories of a run lack.

The series name of a node-table featurizer is computed from its settings, so
a changed selection, ``n_max``, atom-type table or topology starts a new
series that no trajectory has. With ``refill=True`` the next job refills it
before any MD or training, in exactly one process of the run while the
others wait (`aimmd.core.series.ensure_series_coverage`); with the default
``refill=False`` it stops with an error naming the remedies. These tests
check that

- ``refill`` does not change the series name, and every featurizer
  registers its series with its flag and a `NodeTableRefiller`, but not the
  copies of ``with_n_max`` (plan, route, repack);
- the refill takes the cheapest correct route: an ``n_max``-only change is
  repacked without opening a trajectory (zero rows stay zero), other
  settings are featurized again, and an unmigrated graph-cache campaign is
  extracted from its graph cache (``--rungraph``), falling back to
  featurizing when the cache fails the check; multi-system runs are
  repacked per system; parallel refills rebuild the featurizer from the
  params file in spawned processes;
- only the trajectories without the series are written: files of the
  series that exist, the initial paths and the old series stay as they are;
- across processes, one refills while the others wait and start after it,
  with the log lines of both sides, and a failed refill stops the waiters
  with the error instead of a second refill.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.core import series
from aimmd.core.series import (ensure_series_coverage, series_coverage,
                               series_policy)
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      _tool)
from aimmd.network.nodetables._featurizer import repack_rows
from aimmd.network.nodetables._refill import NodeTableRefiller
from tests._nodetables_run import (LAYOUT, bits, expected_rows, make_campaign,
                                   series_file, toy_featurizer, trajectories,
                                   write_run)
from tests._nodetables_toy import ENVIRONMENT_SELECTION, write_toy_gro, \
    toy_frames
from tests._series_refill import (make_full_campaign, remove_legacy_series,
                                  run_trajectories)

WORKTREE = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).resolve().parent / '_series_refill.py'
OTHER_ENVIRONMENT = ENVIRONMENT_SELECTION.replace('5.0', '4.5')


def _quiet(line):
    pass


def _prefill(featurizer, run):
    """The series of `featurizer` for every trajectory of the run."""
    report = _tool.prefill(_tool.ParamsFeaturizer(None, 'F', featurizer, None),
                           [run], log=_quiet)
    assert report['totals']['failed'] == 0
    return report


def _params(folder, featurizer, system_ids=None):
    """What the series check reads from aimmd.Params."""
    return SimpleNamespace(
        descriptors_series=featurizer.series,
        descriptors_function=featurizer.descriptors_function,
        multi_system=system_ids is not None, system_ids=system_ids,
        path=Path(folder) / 'params.py', parent=Path(folder))


def _snapshot(paths):
    return {path: (os.stat(path).st_mtime_ns, Path(path).read_bytes())
            for path in paths}


# ----------------------------------------------------------------------
# the flag

def test_refill_does_not_change_the_series_name(tmp_path):
    write_toy_gro(tmp_path / 'toy.gro', toy_frames(1, seed=10)[0])
    plain = toy_featurizer(tmp_path)
    refilling = toy_featurizer(tmp_path, refill=True)

    assert (plain.refill, refilling.refill) == (False, True)
    assert refilling.series == plain.series
    assert refilling.spec() == plain.spec()
    assert 'refill' not in plain.spec()
    wider = refilling.with_n_max(1024)
    assert wider.refill and wider.series == plain.with_n_max(1024).series

    systems = {'lig1': plain, 'lig2': toy_featurizer(tmp_path, 96)}
    multi = MultiSystemNodeTableFeaturizer(systems)
    multi_refilling = MultiSystemNodeTableFeaturizer(systems, refill=True)
    assert (multi.refill, multi_refilling.refill) == (False, True)
    assert multi_refilling.series == multi.series
    assert multi_refilling.with_n_max(128).refill


def test_every_featurizer_registers_its_series(tmp_path):
    write_toy_gro(tmp_path / 'toy.gro', toy_frames(1, seed=10)[0])
    featurizer = toy_featurizer(tmp_path, refill=True)
    policy = series_policy(featurizer.series)
    assert policy.refill is True
    assert isinstance(policy.refiller, NodeTableRefiller)
    assert policy.refiller.featurizer is featurizer
    # the featurizer built last counts (the params file of the process)
    toy_featurizer(tmp_path)
    assert series_policy(featurizer.series).refill is False

    multi = MultiSystemNodeTableFeaturizer(
        {'lig1': toy_featurizer(tmp_path, 64)}, refill=True)
    assert series_policy(multi.series).refill is True
    assert series_policy(multi.series).refiller.featurizer is multi


def test_the_copies_of_the_tools_register_nothing(tmp_path):
    # with_n_max (plan and route, also run by the launcher, and repack) must
    # not register: the old series would take the current refill flag, and
    # after a repack the current series would be refilled by a copy
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    old = campaign.featurizer
    _prefill(old, campaign.run)
    new = toy_featurizer(campaign.folder, 64, refill=True)
    multi = MultiSystemNodeTableFeaturizer({'lig1': new}, refill=True)
    registered = dict(series._POLICIES)

    assert new.with_n_max(40).series == old.series
    assert new.with_n_max(1024).series not in series._POLICIES
    assert multi.with_n_max(1024).series not in series._POLICIES
    params = _params(campaign.folder, new)
    series.check_series_coverage(params, campaign.run, log=_quiet)
    _tool.repack(_tool.ParamsFeaturizer(None, 'F', old, None),
                 [campaign.run], 48, log=_quiet)
    assert dict(series._POLICIES) == registered
    assert ensure_series_coverage(params, campaign.run, jobs=1, log=_quiet)

    assert dict(series._POLICIES) == registered
    assert series_policy(new.series).refiller.featurizer is new
    assert series_policy(old.series).refill is False


def test_the_copies_of_the_tools_never_hide_a_stale_params_file(tmp_path):
    # a worker builds only the params' featurizer (n_max 64), while its
    # paramsN.py still names the series of n_max 40, which the route's copy
    # (with_n_max(40)) computes: still a mismatch
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    old = campaign.featurizer
    _prefill(old, campaign.run)
    new = toy_featurizer(campaign.folder, 64, refill=True)
    del series._POLICIES[old.series]
    coverage = series_coverage(campaign.run, new.series)
    assert 'repacking' in series_policy(new.series).refiller.route(
        coverage, campaign.folder)
    lines = []
    with pytest.raises(series.SeriesMismatchError) as info:
        ensure_series_coverage(_params(campaign.folder, old), campaign.run,
                               log=lines.append)
    assert (f'was written for the descriptor series {old.series!r}, but its '
            f'node-table featurizer now computes {new.series!r}'
            in str(info.value))
    assert lines == str(info.value).splitlines()

    # multi-system: the series of the whole run is named, not the systems'
    multi = MultiSystemNodeTableFeaturizer({
        'lig1': new, 'lig2': toy_featurizer(campaign.folder, 80)})
    stale = _params(campaign.folder, multi.with_n_max(96), ['lig1', 'lig2'])
    with pytest.raises(series.SeriesMismatchError) as info:
        series.check_series_coverage(stale, tmp_path / 'multi_run')
    assert f'now computes {multi.series!r}' in str(info.value)
    assert new.series not in str(info.value)


# ----------------------------------------------------------------------
# routes

def test_an_n_max_change_is_repacked_without_reading_trajectories(
        tmp_path, monkeypatch):
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    old = campaign.featurizer
    _prefill(old, campaign.run)
    new = toy_featurizer(campaign.folder, 64, refill=True)
    initial = trajectories(campaign.run)[0]
    assert '/initialARB/' in initial

    def no_trajectories(*args, **kwargs):
        raise AssertionError('the refill opened a trajectory')

    monkeypatch.setattr(_tool, '_open_reader', no_trajectories)
    lines = []
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=lines.append)
    monkeypatch.undo()

    assert (f'by repacking the rows of {old.series} (n_max 40 -> 64; no '
            f'trajectory is read)') in lines[0]
    empty = 0
    for trajectory in run_trajectories(campaign.run):
        stored = np.load(series_file(trajectory, old.series))
        rows = np.load(series_file(trajectory, new.series))
        assert np.array_equal(bits(rows), bits(repack_rows(stored, 64)))
        fits = stored[:, 0] != 0
        expected = expected_rows(new, trajectory)
        assert np.array_equal(bits(rows[fits]), bits(expected[fits]))
        assert not rows[~fits].any()        # zero rows stay zero
        empty += int((~fits).sum())
    assert empty > 0
    # neither the initial path nor the old series is touched
    assert not os.path.exists(series_file(initial, new.series))
    assert os.path.exists(series_file(initial, old.series))
    assert series_coverage(campaign.run, new.series).complete


def test_a_settings_change_is_featurized_and_nothing_else_is_written(
        tmp_path):
    campaign = make_campaign(tmp_path / 'campaign')
    old = campaign.featurizer
    _prefill(old, campaign.run)
    new = toy_featurizer(campaign.folder, environment=OTHER_ENVIRONMENT,
                         refill=True)
    assert new.series != old.series
    # a trajectory whose series file exists (rows not filled yet) is not
    # missing and keeps its file
    present = run_trajectories(campaign.run)[0]
    np.save(series_file(present, new.series),
            np.zeros((2, new.width), dtype=np.float32))
    untouched = _snapshot([series_file(present, new.series)]
                          + [series_file(trajectory, old.series)
                             for trajectory in trajectories(campaign.run)])
    lines = []
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=lines.append)

    assert 'by featurizing every frame of the trajectories' in lines[0]
    assert lines[0].startswith('SERIES REFILL: 5 of 6 trajectories (')
    assert lines[-1].startswith(f'SERIES REFILL: done: {new.series!r} of 5 '
                                f'trajectories')
    assert 'the files of the old series were left in place' in lines[-1]
    for trajectory in run_trajectories(campaign.run)[1:]:
        assert np.array_equal(
            bits(np.load(series_file(trajectory, new.series))),
            bits(expected_rows(new, trajectory)))
    assert _snapshot(untouched) == untouched


def test_a_multi_system_run_is_repacked_per_system(tmp_path):
    campaign = make_campaign(tmp_path / 'campaign')
    folder = campaign.folder
    run = Path(folder) / 'multi_run'
    layout = dict(list(LAYOUT.items())[:4])
    write_run(run / 'lig1', layout)
    write_run(run / 'lig2', layout, untracked=None)
    old = MultiSystemNodeTableFeaturizer({
        'lig1': toy_featurizer(folder, 64), 'lig2': toy_featurizer(folder, 80)})
    _prefill(old, run)
    new = MultiSystemNodeTableFeaturizer({
        'lig1': toy_featurizer(folder, 96), 'lig2': toy_featurizer(folder, 96)},
        refill=True)
    lines = []
    assert ensure_series_coverage(
        _params(folder, new, ['lig1', 'lig2']), run / 'lig2', jobs=1,
        log=lines.append)

    assert (f'by repacking the rows of {old.series} (n_max lig1 64 -> 96, '
            f'lig2 80 -> 96;') in lines[0]
    for system_id in ('lig1', 'lig2'):
        for trajectory in run_trajectories(run / system_id, layout):
            rows = np.load(series_file(trajectory, old.series))
            assert np.array_equal(
                bits(np.load(series_file(trajectory, new.series))),
                bits(repack_rows(rows, 96)))


def test_a_parallel_refill_rebuilds_the_featurizer_from_the_params_file(
        tmp_path, monkeypatch):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True,
                                  environment=OTHER_ENVIRONMENT)
    monkeypatch.chdir(campaign.folder)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-none')
    params = aimmd.Params.load('params.py')
    lines = []
    assert ensure_series_coverage(params, 'run1', jobs=2, log=lines.append)

    assert 'refills them now with 2 processes by featurizing' in lines[0]
    assert not any('in this process only' in line for line in lines)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-none'
    featurizer = toy_featurizer(campaign.folder, 64,
                                environment=OTHER_ENVIRONMENT)
    assert featurizer.series == params.descriptors_series
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, featurizer.series))),
            bits(expected_rows(featurizer, trajectory)))


def _graph_cache(campaign, tmp_path, environment=ENVIRONMENT_SELECTION):
    """The graph cache of the campaign's frames, in the campaign folder."""
    from tests.test_nodetables_cli_extract import _build_cache
    database = _build_cache(campaign, tmp_path / 'cache', environment)
    target = Path(campaign.folder) / 'graphs_cache.sqlite'
    shutil.copy(database, target)
    return str(target)


def _spy_prefill(monkeypatch):
    calls = []
    prefill = _tool.prefill

    def spy(*args, **kwargs):
        calls.append(kwargs.get('db'))
        return prefill(*args, **kwargs)

    monkeypatch.setattr(_tool, 'prefill', spy)
    return calls


@pytest.mark.graph
def test_an_unmigrated_campaign_is_extracted_from_its_graph_cache(
        tmp_path, monkeypatch):
    pytest.importorskip('torch_geometric')
    campaign = make_campaign(tmp_path / 'campaign')  # stale *.descriptors.npy
    database = _graph_cache(campaign, tmp_path)
    new = toy_featurizer(campaign.folder, refill=True)
    calls = _spy_prefill(monkeypatch)
    lines = []
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=lines.append)

    assert f'by extracting the rows from the graph cache {database}' in \
        lines[0]
    assert calls == [[database]]
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, new.series))),
            bits(expected_rows(new, trajectory)))


@pytest.mark.graph
def test_a_graph_cache_of_other_settings_falls_back_to_featurizing(
        tmp_path, monkeypatch):
    pytest.importorskip('torch_geometric')
    campaign = make_campaign(tmp_path / 'campaign')
    _graph_cache(campaign, tmp_path, ENVIRONMENT_SELECTION.replace('5.0',
                                                                   '6.0'))
    new = toy_featurizer(campaign.folder, refill=True)
    calls = _spy_prefill(monkeypatch)
    lines = []
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=lines.append)

    assert len(calls) == 2 and calls[0] and calls[1] is None
    assert any('failed the check against a direct featurization' in line
               for line in lines)
    assert 'featurized instead' in lines[-1]
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, new.series))),
            bits(expected_rows(new, trajectory)))


# ----------------------------------------------------------------------
# one process refills, the others wait

def _processes(tmp_path, campaign, mode, n=3):
    """Run `n` processes of the run (tests/_series_refill.py) on the params
    file of the workers; returns their exit statuses, logs and the refill
    counter."""
    folder = campaign.folder
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params_file = str(aimmd.Params.load('params.py').path)
    finally:
        os.chdir(cwd)
    ready = tmp_path / 'ready'
    ready.mkdir()
    counter = tmp_path / 'refills.txt'
    logs = [tmp_path / f'process{i}.log' for i in range(n)]
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1')
    processes = [subprocess.Popen(
        [sys.executable, str(HELPER), params_file, campaign.run, str(log),
         str(counter), str(ready), str(n), mode], env=env)
        for log in logs]
    codes = [process.wait(timeout=600) for process in processes]
    texts = [log.read_text() if log.exists() else '' for log in logs]
    events = counter.read_text().split('\n') if counter.exists() else []
    return codes, texts, [event.split() for event in events if event]


def test_one_process_refills_while_the_others_wait(tmp_path):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True)
    codes, logs, events = _processes(tmp_path, campaign, 'ok')

    assert codes == [0, 0, 0], logs
    assert [event[0] for event in events] == ['start', 'end']
    refiller = events[0][1]
    end = float(events[1][2])
    refilling = [log for log in logs if 'refills them now' in log]
    assert len(refilling) == 1
    assert f'pid {refiller}, test process {refiller})' in refilling[0]
    assert 'SERIES REFILL: done:' in refilling[0]
    waiting = [log for log in logs if log is not refilling[0]]
    for log in waiting:
        assert f'SERIES REFILL: waiting for ' in log
        assert f'pid {refiller}, test process {refiller} to refill' in log
        assert 'is complete after' in log
        started = float(log.split('STARTED ')[1].split()[0])
        assert started >= end
    assert any('SERIES REFILL: still waiting for ' in log for log in waiting)
    for log in logs:
        assert all(line.startswith('SERIES REFILL:')
                   for line in log.splitlines()[:-1])
    featurizer = toy_featurizer(campaign.folder, 64)
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, featurizer.series))),
            bits(expected_rows(featurizer, trajectory)))
    assert not os.path.exists(Path(campaign.run) / series.REFILL_LOCK)


def test_a_failed_refill_stops_the_waiters_without_a_second_refill(
        tmp_path):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True)
    remove_legacy_series(campaign.run)
    codes, logs, events = _processes(tmp_path, campaign, 'fail')

    assert codes == [3, 3, 3], logs
    assert [event[0] for event in events] == ['start']
    refiller = events[0][1]
    failed = [log for log in logs if 'refills them now' in log]
    assert len(failed) == 1
    assert ('failed after' in failed[0]
            and 'the refill failed on purpose' in failed[0])
    for log in logs:
        if log is failed[0]:
            continue
        assert f'pid {refiller}, test process {refiller} let go of the ' \
               f'refill' in log
        assert 'This process does not refill again.' in log
        assert 'construct the featurizer with refill=True' in log
        assert 'refills them now' not in log
    for log in logs:
        assert all(line.startswith('SERIES REFILL:')
                   for line in log.splitlines()[:-1])
    coverage = series_coverage(campaign.run,
                               toy_featurizer(campaign.folder, 64).series)
    assert len(coverage.missing) == len(coverage.trajectories) == 6
    assert not os.path.exists(Path(campaign.run) / series.REFILL_LOCK)
