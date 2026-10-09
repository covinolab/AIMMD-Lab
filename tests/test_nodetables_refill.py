"""Refilling a node-table series that trajectories of a run lack.

The series name of a node-table featurizer is computed from its settings, so
a changed selection, ``n_max``, atom-type table or topology starts a new
series that no trajectory has. With ``refill=True`` the next job refills it
before any MD or training, in exactly one process of the run while the
others wait (`aimmd.core.series.ensure_series_coverage`); with the default
``refill=False`` it stops with an error naming the remedies. These tests
check that

- ``refill`` does not change the series name, and every featurizer
  registers its series with its flag and a `NodeTableRefiller` (also one
  built with ``with_n_max``), but not the copies that the tools make
  (plan, route, repack);
- the error suggests the command of the route, which runs as it is, also
  for a params file whose featurizer is not named FEATURIZER;
- the refill takes the cheapest correct route: an ``n_max``-only change is
  repacked without opening a trajectory (zero rows stay zero), other
  settings are featurized again, and an unmigrated graph-cache campaign is
  extracted from its graph cache (``--rungraph``), checked on frames that
  came from the cache, falling back to featurizing when the cache fails the
  check, and then for every trajectory that took rows from it, and when its
  processes died before every trajectory was checked; multi-system runs are
  repacked per system; parallel refills rebuild the featurizer from the
  params file in spawned processes;
- only the trajectories without the series are written: files of the
  series that exist, the initial paths and the old series stay as they are;
- a stop request ends a refill between chunks of frames, without leaving
  temporary files, and the next check refills the rest;
- what the spawned processes of a parallel refill print reaches the log
  with the prefix, and processes that cannot load the params file (it
  needs a GPU, which they do not see) leave their error in the log and the
  refill continues in the process itself;
- only the spawned processes see no GPU: the refilling process keeps its
  environment throughout (its own featurization included) on every route;
- across processes, one refills while the others wait and start after it,
  with the log lines of both sides, and a failed refill stops the waiters
  with the error instead of a second refill, and also the processes of the
  same job that start after it (a new job tries again).
"""
import concurrent.futures
import contextlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.core import series
from aimmd.core.series import (ensure_series_coverage, series_coverage,
                               series_policy)
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      _cli, _tool)
from aimmd.network.nodetables._featurizer import repack_rows
from aimmd.network.nodetables._refill import NodeTableRefiller
from tests._nodetables_run import (LAYOUT, bits, expected_rows, make_campaign,
                                   series_file, toy_featurizer, trajectories,
                                   write_run)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames,
                                   write_toy_gro)
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

    # a featurizer that a params file builds with with_n_max, with the flag
    # it keeps
    wider = toy_featurizer(tmp_path, 40, refill=True).with_n_max(96)
    assert series_policy(wider.series).refill is True
    assert series_policy(wider.series).refiller.featurizer is wider
    multi_wider = multi.with_n_max(128)
    assert series_policy(multi_wider.series).refill is True
    assert series_policy(multi_wider.series).refiller.featurizer is \
        multi_wider


def test_the_copies_of_the_tools_register_nothing(tmp_path):
    # the copies that plan and route (also run by the launcher) and repack
    # make with with_n_max must not register: the old series would take the
    # current refill flag, and after a repack the current series would be
    # refilled by a copy
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    old = campaign.featurizer
    _prefill(old, campaign.run)
    new = toy_featurizer(campaign.folder, 64, refill=True)
    multi = MultiSystemNodeTableFeaturizer({'lig1': new}, refill=True)
    registered = dict(series._POLICIES)

    assert new.with_n_max(40, _register=False).series == old.series
    assert multi.with_n_max(1024, _register=False).series not in \
        series._POLICIES
    assert dict(series._POLICIES) == registered
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
    # the params file's module-level wrapper (which hides the featurizer)
    stale = _params(campaign.folder, old)
    stale.descriptors_function = lambda trajectory: \
        new.descriptors_function(trajectory)
    lines = []
    with pytest.raises(series.SeriesMismatchError) as info:
        ensure_series_coverage(stale, campaign.run, log=lines.append)
    assert (f'was written for the descriptor series {old.series!r}, but its '
            f'node-table featurizer now computes {new.series!r}'
            in str(info.value))
    assert lines == str(info.value).splitlines()

    # multi-system: the series of the whole run is named, not the systems'
    multi = MultiSystemNodeTableFeaturizer({
        'lig1': new, 'lig2': toy_featurizer(campaign.folder, 80)})
    stale = _params(campaign.folder, multi, ['lig1', 'lig2'])
    stale.descriptors_series = multi.with_n_max(96, _register=False).series
    stale.descriptors_function = lambda trajectory, system_id: \
        multi.descriptors_function(trajectory, system_id)
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


def test_the_error_names_the_command_of_the_route(tmp_path):
    campaign = make_campaign(tmp_path / 'campaign', n_max=40)
    run = campaign.run
    _prefill(campaign.featurizer, run)
    prefill = (f'python -m aimmd.network.nodetables prefill --params P.py '
               f'--run {run} --only-missing')
    # only n_max changed: repack
    new = toy_featurizer(campaign.folder, 64)
    coverage = series_coverage(run, new.series)
    command = series_policy(new.series).refiller.prefill_command(
        'P.py', run, coverage, campaign.folder)
    assert command == (
        f'python -m aimmd.network.nodetables repack --params P.py --run '
        f'{run} --n-max 64 --from-n-max 40 (no trajectory is read), then '
        f'{prefill} (the rows that did not fit)')
    with pytest.raises(series.SeriesCoverageError) as info:
        series.check_series_coverage(_params(campaign.folder, new), run)
    params_file = str(Path(campaign.folder) / 'params.py')
    assert (f"2. fill the series before the next job: "
            f"{command.replace('P.py', params_file)};") in str(info.value)
    # other settings: prefill the missing trajectories
    other = toy_featurizer(campaign.folder, environment=OTHER_ENVIRONMENT)
    refiller = series_policy(other.series).refiller
    coverage = series_coverage(run, other.series)
    assert refiller.prefill_command('P.py', run, coverage, campaign.folder) \
        == f"{prefill} (or the campaign's prefill_nodetables.sh)"
    # ... from the graph cache of an unmigrated campaign
    if importlib.util.find_spec('torch_geometric') is not None:
        database = Path(campaign.folder) / 'graphs_cache.sqlite'
        database.write_bytes(b'')
        assert refiller.prefill_command(
            'P.py', run, coverage, campaign.folder) == (
            f"{prefill} --db {database} (or the campaign's "
            f"prefill_nodetables.sh)")


MULTI_PARAMS = """
import MDAnalysis as mda
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      NodeTableFeaturizer)

f1 = NodeTableFeaturizer(
    mda.Universe('toy.gro', to_guess=['types', 'bonds']),
    {system!r}, {environment!r}, {atom_types!r}, cutoff={cutoff!r}, n_max=96)
f2 = NodeTableFeaturizer(
    mda.Universe('toy.gro', to_guess=['types', 'bonds']),
    {system!r}, {environment!r}, {atom_types!r}, cutoff={cutoff!r}, n_max=96)
FEATURIZERS = MultiSystemNodeTableFeaturizer({{'lig1': f1, 'lig2': f2}})
descriptors_series = FEATURIZERS.series
"""


def test_the_suggested_commands_name_a_featurizer_not_named_featurizer(
        tmp_path, monkeypatch):
    # the multi-system params file of the docs, with its featurizers at
    # module level: the tools cannot choose one without --featurizer
    campaign = make_campaign(tmp_path / 'campaign')
    folder = Path(campaign.folder)
    run = folder / 'multi_run'
    layout = dict(list(LAYOUT.items())[:4])
    write_run(run / 'lig1', layout)
    write_run(run / 'lig2', layout, untracked=None)
    _prefill(MultiSystemNodeTableFeaturizer({
        'lig1': toy_featurizer(folder, 64),
        'lig2': toy_featurizer(folder, 80)}), run)
    params = folder / 'multi_params.py'
    params.write_text(MULTI_PARAMS.format(
        system=SYSTEM_SELECTION, environment=ENVIRONMENT_SELECTION,
        atom_types=ATOM_TYPES, cutoff=CUTOFF))
    monkeypatch.chdir(folder)
    spec = importlib.util.spec_from_file_location('multi_params', params)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'multi_params', module)
    spec.loader.exec_module(module)        # as Params.load imports it
    series_name = module.FEATURIZERS.series
    coverage = series_coverage(run, series_name, ['lig1', 'lig2'])

    command = series_policy(series_name).refiller.prefill_command(
        str(params), str(run), coverage, str(folder))

    commands = [part.split(' (')[0] for part in command.split(', then ')]
    assert [part.split()[3] for part in commands] == ['repack', 'prefill']
    for part in commands:
        argv = shlex.split(part)
        assert argv[:3] == ['python', '-m', 'aimmd.network.nodetables']
        assert argv[argv.index('--featurizer') + 1] == 'FEATURIZERS'
        assert _cli.main(argv[3:]) == 0
    assert series_coverage(run, series_name, ['lig1', 'lig2']).complete


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
    coverage = series_coverage(run, new.series, ['lig1', 'lig2'])
    assert series_policy(new.series).refiller.prefill_command(
        'P.py', str(run), coverage, folder) == (
        f'python -m aimmd.network.nodetables repack --params P.py --run '
        f'{run} --n-max 96 --from-n-max lig1=64 --from-n-max lig2=80 (no '
        f'trajectory is read), then python -m aimmd.network.nodetables '
        f'prefill --params P.py --run {run} --only-missing (the rows that '
        f'did not fit)')
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


def test_a_stop_request_ends_the_refill_between_chunks(tmp_path):
    campaign = make_campaign(tmp_path / 'campaign')
    new = toy_featurizer(campaign.folder, environment=OTHER_ENVIRONMENT,
                         refill=True)
    featurize = new.descriptors_function
    chunks = []

    def counted(frames):
        chunks.append(len(frames))
        return featurize(frames)

    new.descriptors_function = counted      # what the refill featurizes with
    lines = []
    assert ensure_series_coverage(
        _params(campaign.folder, new), campaign.run, jobs=1,
        log=lines.append, stop=lambda: len(chunks) >= 2) is False

    assert len(chunks) == 2                 # of 6 trajectories
    assert 'stopped on request after' in lines[-1]
    assert not list(Path(campaign.run).rglob('*.tmp'))
    assert not os.path.exists(Path(campaign.run) / series.REFILL_LOCK)
    assert not series_coverage(campaign.run, new.series).complete
    # the next start refills the rest
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=_quiet)
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, new.series))),
            bits(expected_rows(new, trajectory)))


def _child_params(campaign, statement):
    """Put `statement` at the top of the params file: it runs in the
    spawned processes of a refill only (they see no GPU)."""
    params = Path(campaign.params)
    params.write_text(f"import os\nif os.environ.get('CUDA_VISIBLE_DEVICES')"
                      f" == '':\n    {statement}\n" + params.read_text())


def test_what_the_processes_of_a_refill_print_reaches_the_log(
        tmp_path, monkeypatch, capfd):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True,
                                  environment=OTHER_ENVIRONMENT)
    _child_params(campaign, "print('a refill process imported the params "
                            "file', flush=True)")
    monkeypatch.chdir(campaign.folder)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-none')
    params = aimmd.Params.load('params.py')
    capfd.readouterr()
    lines = []
    assert ensure_series_coverage(params, 'run1', jobs=2, log=lines.append)

    assert 'refills them now with 2 processes by featurizing' in lines[0]
    assert all(line.startswith('SERIES REFILL:') for line in lines)
    assert ('SERIES REFILL:   a refill process imported the params file'
            in lines)
    out = capfd.readouterr()
    assert 'a refill process' not in out.out + out.err
    assert not os.environ.get(_tool.CHILD_OUTPUT)


def test_processes_that_cannot_load_the_params_fall_back_to_this_one(
        tmp_path, monkeypatch, capfd):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True,
                                  environment=OTHER_ENVIRONMENT)
    _child_params(campaign, "raise RuntimeError('this params file needs a "
                            "GPU')")
    monkeypatch.chdir(campaign.folder)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-none')
    params = aimmd.Params.load('params.py')
    capfd.readouterr()
    lines = []
    assert ensure_series_coverage(params, 'run1', jobs=2, log=lines.append)

    assert all(line.startswith('SERIES REFILL:') for line in lines)
    text = '\n'.join(lines)
    assert 'RuntimeError: this params file needs a GPU' in text
    assert 'refilling in this process instead' in text
    assert lines[-1].startswith('SERIES REFILL: done:')
    out = capfd.readouterr()
    assert 'needs a GPU' not in out.out + out.err
    featurizer = toy_featurizer(campaign.folder, 64,
                                environment=OTHER_ENVIRONMENT)
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, featurizer.series))),
            bits(expected_rows(featurizer, trajectory)))


def test_only_the_spawned_processes_of_a_refill_see_no_gpu(tmp_path,
                                                          monkeypatch):
    # in this process, while it refills in one process, in two and in its
    # own after the two could not start, the GPUs stay visible (its own
    # featurization and every log line) and the environment is unchanged
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-none')
    child = {'one': None,
             'two': "print('a refill process sees no GPU', flush=True)",
             'fallback': "raise RuntimeError('this params file needs a "
                         "GPU')"}
    for route, statement in child.items():
        campaign = make_full_campaign(tmp_path / route, refill=True,
                                      environment=OTHER_ENVIRONMENT)
        if statement:
            _child_params(campaign, statement)
        monkeypatch.chdir(campaign.folder)
        params = aimmd.Params.load('params.py')
        featurizer = series_policy(params.descriptors_series).refiller \
            .featurizer
        featurize = featurizer.descriptors_function
        featurized, logged, lines = [], [], []

        def spy(frames):
            featurized.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
            return featurize(frames)

        def log(line):
            logged.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
            lines.append(line)

        featurizer.descriptors_function = spy
        environment = dict(os.environ)
        assert ensure_series_coverage(params, 'run1',
                                      jobs=1 if route == 'one' else 2,
                                      log=log)

        assert dict(os.environ) == environment, route
        assert set(logged) == {'GPU-none'}, route
        assert set(featurized) == ({'GPU-none'} if route != 'two'
                                   else set()), route
        text = '\n'.join(lines)
        assert ('a refill process sees no GPU' in text) == (route == 'two')
        assert ('refilling in this process instead' in text) == (
            route == 'fallback')
        assert lines[-1].startswith('SERIES REFILL: done:'), route


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


@pytest.mark.graph
def test_the_extract_route_checks_extracted_rows_and_distrusts_a_bad_cache(
        tmp_path, monkeypatch):
    # back.xtc: one frame from the cache, a graph of other settings, which a
    # check of frames drawn from all frames misses; path000001.xtc: every
    # frame from the cache, one of other settings that its own check misses.
    # Once the cache gave a wrong row, nothing else is taken from it.
    pytest.importorskip('torch_geometric')
    import sqlite3
    from tests.test_nodetables_cli_extract import (_build_cache,
                                                   _coordinates, _key)
    campaign = make_campaign(tmp_path / 'campaign')  # stale *.descriptors.npy
    new = toy_featurizer(campaign.folder, refill=True)
    database = _graph_cache(campaign, tmp_path)
    bad = _build_cache(campaign, tmp_path / 'bad',
                       ENVIRONMENT_SELECTION.replace('5.0', '6.0'))
    connection = sqlite3.connect(bad)
    wrong = dict(connection.execute('SELECT key, data FROM graphs_cache'))
    connection.close()

    def unchecked(trajectory, n_frames):
        """A frame that a check of frames drawn from all frames skips."""
        drawn = _tool._sample(trajectory, np.arange(n_frames),
                              _tool.DEFAULT_DB_VERIFY, 0)
        return sorted(set(range(n_frames)) - set(drawn.tolist()))[0]

    run = Path(campaign.run)
    first, second = str(run / 'chainR0/back.xtc'), str(
        run / 'chainR0/path000001.xtc')
    connection = sqlite3.connect(database)
    for trajectory in trajectories(campaign.run):
        keys = [_key(row) for row in _coordinates(campaign, trajectory)]
        if trajectory in (first, second):
            frame = unchecked(trajectory, len(keys))
            connection.execute('UPDATE graphs_cache SET data = ? WHERE key = ?',
                               (wrong[keys[frame]], keys[frame]))
            if trajectory == first:
                keys = keys[:frame] + keys[frame + 1:]
            else:
                keys = []
        connection.executemany('DELETE FROM graphs_cache WHERE key = ?',
                               [(key,) for key in keys])
    connection.commit()
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    monkeypatch.chdir(campaign.folder)
    lines = []
    assert ensure_series_coverage(_params(campaign.folder, new),
                                  campaign.run, jobs=1, log=lines.append)

    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, new.series))),
            bits(expected_rows(new, trajectory))), trajectory
    text = '\n'.join(lines)
    assert 'mismatch  run1/chainR0/back.xtc: 5 frames, 5 computed (1 from ' \
        'the graph cache, 4 featurized), 1 verified (1 mismatched' in text
    assert ('the graph cache gave rows that differ from a direct '
            'featurization for run1/chainR0/back.xtc: no other trajectory '
            'of this prefill takes rows from it') in text
    assert ('failed    run1/chainR0/path000001.xtc: 9 frames, 9 computed (9 '
            'from the graph cache, 0 featurized), 4 verified (0 '
            'mismatched); ERROR: not installed: the graph cache gave rows '
            'that differ from a direct featurization for '
            'run1/chainR0/back.xtc') in text
    assert '2 trajectories failed the check' in text


@pytest.mark.graph
def test_no_rows_from_the_graph_cache_are_installed_when_processes_died(
        tmp_path, monkeypatch):
    # strict_db installs rows from the cache once every trajectory was
    # checked; the trajectories of processes that died never were, so
    # nothing from the cache is installed (the refill reruns the call in
    # its own process, which extracts and checks them all again)
    pytest.importorskip('torch_geometric')
    campaign = make_campaign(tmp_path / 'campaign')  # stale *.descriptors.npy
    database = _graph_cache(campaign, tmp_path)
    new = toy_featurizer(campaign.folder, refill=True)
    dead = str(Path(campaign.run) / 'chainR0' / 'back.xtc')

    class Dying(_tool._Serial):
        """Runs the tasks here, but the chunks of `dead` find the
        processes dead."""

        def submit(self, function, task):
            if (task['trajectory'] == dead
                    and function is _tool._prefill_chunk):
                future = concurrent.futures.Future()
                future.set_exception(BrokenProcessPool(
                    'a process terminated abruptly'))
                return future
            return super().submit(function, task)

    @contextlib.contextmanager
    def executor(jobs, params=None):
        yield Dying(params.featurizer)

    monkeypatch.setattr(_tool, '_executor', executor)
    report = _tool.prefill(
        _tool.ParamsFeaturizer(None, 'F', new, None), [campaign.run],
        db=[database], jobs=2, only_missing=True, log=_quiet,
        trajectories=[(trajectory, None) for trajectory
                      in run_trajectories(campaign.run)], strict_db=True)

    assert report['broken_pool']
    entries = {entry['trajectory']: entry for entry in report['files']}
    assert 'a worker process died' in entries.pop(dead)['error']
    for trajectory, entry in entries.items():
        assert entry['db_hits'] and entry['verify_mismatches'] == 0
        assert entry['status'] == 'failed', trajectory
        assert 'not installed' in entry['error']
    for trajectory in run_trajectories(campaign.run):
        assert not os.path.exists(series_file(trajectory, new.series))
    assert not list(Path(campaign.run).rglob('*.tmp'))


# ----------------------------------------------------------------------
# one process refills, the others wait

def _processes(tmp_path, campaign, mode, n=3, job='1001', batch=''):
    """Run `n` processes of the run (tests/_series_refill.py) on the params
    file of the workers, as processes of the SLURM job `job`; returns their
    exit statuses, logs and the refill counter (of every batch)."""
    folder = campaign.folder
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        params_file = str(aimmd.Params.load('params.py').path)
    finally:
        os.chdir(cwd)
    ready = tmp_path / f'ready{batch}'
    ready.mkdir()
    counter = tmp_path / 'refills.txt'
    logs = [tmp_path / f'process{batch}{i}.log' for i in range(n)]
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), CUDA_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1', SLURM_JOB_ID=job)
    env.pop('SLURM_RESTART_COUNT', None)
    env.pop('SLURM_JOB_START_TIME', None)
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
    assert os.path.exists(Path(campaign.run) / series.REFILL_FAILED)

    # processes of the same job that start after the failure (a late
    # trainer, the workers of a sweep job script with a plain 'wait'): no
    # second refill
    codes, logs, events = _processes(tmp_path, campaign, 'fail', n=2,
                                     batch='late')
    assert codes == [3, 3], logs
    assert [event[0] for event in events] == ['start']
    # the first to take the lock finds the record; the other may wait for it
    refused = [log for log in logs if 'already failed in this job' in log]
    assert refused
    assert 'already failed in this job (SLURM job 1001: ' in refused[0]
    assert f'pid {refiller}, test process {refiller}, at ' in refused[0]
    for log in logs:
        assert 'does not refill again' in log
        assert 'the refill failed on purpose' in log
        assert 'refills them now' not in log
        assert all(line.startswith('SERIES REFILL:')
                   for line in log.splitlines()[:-1])

    # the next job tries again, and its refill removes the marker
    codes, logs, events = _processes(tmp_path, campaign, 'ok', n=1,
                                     job='1002', batch='next')
    assert codes == [0], logs
    assert [event[0] for event in events] == ['start', 'start', 'end']
    assert not os.path.exists(Path(campaign.run) / series.REFILL_FAILED)
