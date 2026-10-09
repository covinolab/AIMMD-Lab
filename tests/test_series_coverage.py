"""Is every trajectory of a run covered by the params' descriptor series?

`aimmd.core.series` checks, before a job does any work, that every
trajectory of a run (every file with a states series ``*.states.npy``, the
exported initial paths in ``initial*`` left out) has a file of the params'
``descriptors_series``. A computed series name (a node-table featurizer's)
changes with the settings, and AIMMD would otherwise featurize every frame
again, lazily. These tests use placeholder trajectory files: the check never
reads a trajectory, only lists files and reads the length of the states
series. They cover

- the coverage: a new run and a complete one pass; zero or short rows are
  not missing; the initial paths are left out; multi-system runs; what is
  next to the trajectories that lack the series (another series, the legacy
  ``*.descriptors.npy``);
- the registry of series policies (``refill`` and the refill callable): an
  unknown series is never refilled, the last registration counts;
- the launcher's check (an error naming the other series or the legacy
  series and the remedies, or the notice of a refill), and the worker's,
  which refills in one process with a refill callable and runs the
  callable once.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aimmd.core import series
from aimmd.core.series import (SeriesCoverageError, check_series_coverage,
                               ensure_series_coverage, find_trajectories,
                               register_series, series_coverage,
                               series_policy)

SERIES = 'descriptors-gn0123456789'
OTHER = 'descriptors-gnabcdefabcd'

# trajectory of a run -> frames (the length of its states series)
LAYOUT = {
    'initialARB/initial.xtc': 4,
    'chainR0/path000001.xtc': 9,
    'chainR0/path000002.xtc': 7,
    'chainR0/back.xtc': 5,
    'chainR0/forw.xtc': 3,
    'freeA/traj000001.part0000.xtc': 8,
}
RUN_TRAJECTORIES = [name for name in LAYOUT if not name.startswith('initial')]


def _write(run, layout=LAYOUT, series=(SERIES,), rows=None):
    """Placeholder trajectories with a states series and, for each name in
    `series`, a series file of `rows` rows (default: one per frame)."""
    for name, n_frames in layout.items():
        trajectory = Path(run) / name
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_bytes(b'not read')
        np.save(f'{trajectory}.states.npy', np.full(n_frames, 'R'))
        for name_ in series:
            n_rows = n_frames if rows is None else rows
            np.save(f'{trajectory}.{name_}.npy',
                    np.ones((n_rows, 6), dtype=np.float32))
    return str(run)


def _params(folder, series_name=SERIES, multi_system=False, system_ids=None):
    """What the checks read from aimmd.Params."""
    return SimpleNamespace(
        descriptors_series=series_name,
        descriptors_function=lambda trajectory: None,
        multi_system=multi_system, system_ids=system_ids,
        path=Path(folder) / 'params1.py', parent=Path(folder))


@pytest.fixture(autouse=True)
def _registry():
    """Every test starts without registered series."""
    saved = dict(series._POLICIES)
    series._POLICIES.clear()
    yield
    series._POLICIES.clear()
    series._POLICIES.update(saved)


# ----------------------------------------------------------------------
# coverage

def test_a_new_run_is_complete(tmp_path):
    for run in (tmp_path / 'missing', tmp_path / 'empty'):
        if run.name == 'empty':
            run.mkdir()
        coverage = series_coverage(run, SERIES)
        assert coverage.complete
        assert coverage.trajectories == coverage.missing == []
        check_series_coverage(_params(tmp_path), run)
        assert ensure_series_coverage(_params(tmp_path), run)


def test_a_complete_run_is_complete(tmp_path):
    run = _write(tmp_path / 'run1')
    coverage = series_coverage(run, SERIES)
    assert coverage.complete
    assert sorted(trajectory for trajectory, _ in coverage.trajectories) == \
        sorted(str(tmp_path / 'run1' / name) for name in RUN_TRAJECTORIES)


def test_zero_and_short_rows_are_not_missing(tmp_path):
    run = _write(tmp_path / 'run1', rows=0)
    np.save(tmp_path / 'run1' / f'chainR0/back.xtc.{SERIES}.npy',
            np.zeros((2, 6), dtype=np.float32))
    assert series_coverage(run, SERIES).complete


def test_missing_files_frames_and_what_is_next_to_them(tmp_path):
    run = _write(tmp_path / 'run1', series=(SERIES, OTHER, 'descriptors'))
    for name in ('chainR0/path000002.xtc', 'freeA/traj000001.part0000.xtc'):
        os.remove(tmp_path / 'run1' / f'{name}.{SERIES}.npy')
    os.remove(tmp_path / 'run1' / f'chainR0/path000002.xtc.{OTHER}.npy')

    coverage = series_coverage(run, SERIES)

    assert not coverage.complete
    assert coverage.missing == sorted(
        (str(tmp_path / 'run1' / name), None)
        for name in ('chainR0/path000002.xtc',
                     'freeA/traj000001.part0000.xtc'))
    assert coverage.missing_frames == 7 + 8
    assert coverage.others == {OTHER: 1}
    assert coverage.legacy == 2
    assert coverage.summary().startswith('2 of 5 trajectories (15 frames)')
    # one file per other series, for its layout
    assert set(coverage.examples) == {(OTHER, None)}


def test_the_initial_paths_are_left_out(tmp_path):
    run = _write(tmp_path / 'run1')
    os.remove(tmp_path / 'run1' / f'initialARB/initial.xtc.{SERIES}.npy')
    # a chain folder whose name only contains 'initial' is part of the run
    _write(tmp_path / 'run1', {'chain_initialR0/path000001.xtc': 2},
           series=())

    coverage = series_coverage(run, SERIES)

    assert [Path(trajectory).parent.name for trajectory, _ in
            coverage.missing] == ['chain_initialR0']
    assert str(tmp_path / 'run1' / 'initialARB' / 'initial.xtc') not in {
        trajectory for trajectory, _ in coverage.trajectories}
    # the trajectories of the tools still include them
    assert (str(tmp_path / 'run1' / 'initialARB' / 'initial.xtc'), None) in \
        find_trajectories([run])
    assert (str(tmp_path / 'run1' / 'initialARB' / 'initial.xtc'), None) \
        not in find_trajectories([run], seeds=False)


def test_a_trajectory_needs_its_file_and_states(tmp_path):
    run = _write(tmp_path / 'run1')
    os.remove(tmp_path / 'run1' / 'chainR0' / 'forw.xtc')     # states only
    (tmp_path / 'run1' / 'chainR0' / 'temp.xtc').write_bytes(b'no states')
    names = {Path(trajectory).name for trajectory, _ in
             series_coverage(run, SERIES).trajectories}
    assert 'forw.xtc' not in names and 'temp.xtc' not in names


def test_a_multi_system_run(tmp_path):
    run = tmp_path / 'run1'
    _write(run / 'lig1')
    _write(run / 'lig2')
    _write(run / 'lig3')                                 # not a system
    _write(run / 'lig3', {'chainR0/path000009.xtc': 2}, series=())
    os.remove(run / 'lig2' / f'chainR0/path000001.xtc.{SERIES}.npy')
    os.remove(run / 'lig2' / f'initialARB/initial.xtc.{SERIES}.npy')

    coverage = series_coverage(run, SERIES, ['lig1', 'lig2'])

    assert coverage.missing == [
        (str(run / 'lig2' / 'chainR0' / 'path000001.xtc'), 'lig2')]
    assert len(coverage.trajectories) == 2 * len(RUN_TRAJECTORIES)
    assert {system_id for _, system_id in coverage.trajectories} == {
        'lig1', 'lig2'}
    # a worker of a system folder checks the whole run
    params = _params(tmp_path, multi_system=True,
                     system_ids=['lig1', 'lig2'])
    assert series.run_folder(params, run / 'lig1') == (
        str(run), ['lig1', 'lig2'])
    assert series.run_folder(params, run) == (str(run), ['lig1', 'lig2'])
    with pytest.raises(SeriesCoverageError):
        ensure_series_coverage(params, run / 'lig1', log=lambda line: None)


# ----------------------------------------------------------------------
# policies

def test_an_unknown_series_is_never_refilled():
    policy = series_policy('descriptors-unknown')
    assert (policy.series, policy.refill, policy.refiller) == (
        'descriptors-unknown', False, None)


def test_the_last_registration_counts():
    def refiller(*args, **kwargs):
        pass
    register_series(SERIES, refill=True, refiller=refiller)
    assert series_policy(SERIES) == (SERIES, True, refiller)
    register_series(SERIES, refill=False, refiller=refiller)
    assert series_policy(SERIES).refill is False


def test_the_default_series_is_not_checked(tmp_path):
    run = _write(tmp_path / 'run1', series=())
    for params in (_params(tmp_path, 'descriptors'),
                   SimpleNamespace(parent=tmp_path),
                   SimpleNamespace(descriptors_series=SERIES,
                                   descriptors_function=None)):
        assert check_series_coverage(params, run) is None
        assert ensure_series_coverage(params, run)


# ----------------------------------------------------------------------
# the checks of the launcher and of the workers

def _run_with_gap(tmp_path, *, other=False, legacy=False):
    names = (SERIES,) + ((OTHER,) if other else ()) + (
        ('descriptors',) if legacy else ())
    run = _write(tmp_path / 'run1', series=names)
    for name in RUN_TRAJECTORIES[:3]:
        os.remove(tmp_path / 'run1' / f'{name}.{SERIES}.npy')
    return run


def test_without_refill_the_launcher_names_other_series_and_remedies(
        tmp_path):
    run = _run_with_gap(tmp_path, other=True)
    with pytest.raises(SeriesCoverageError) as info:
        check_series_coverage(_params(tmp_path), run)
    message = str(info.value)
    assert f'3 of 5 trajectories (21 frames) of run {run!r}' in message
    assert f"have no {SERIES!r} series file" in message
    assert f"{OTHER!r} (3 trajectories)" in message
    assert 'settings changed since these frames were featurized' in message
    assert 'not been migrated' not in message
    assert '1. restore the' in message
    assert '2. compute the series of these trajectories' in message
    # an unknown series has no refill callable to offer
    assert 'refill=True' not in message
    assert all(line.startswith('SERIES CHECK:')
               for line in message.splitlines())


def test_without_refill_the_messages_name_legacy_series_and_remedies(
        tmp_path):
    class Refiller:
        label = 'node-table'
        legacy_note = 'not migrated from the graph-cache input'

        def prefill_command(self, params_file, run):
            return f'prefill --params {params_file} --run {run}'

        def __call__(self, *args, **kwargs):
            raise AssertionError('refill=False must not refill')

    register_series(SERIES, refill=False, refiller=Refiller())
    run = _run_with_gap(tmp_path, legacy=True)
    lines = []
    with pytest.raises(SeriesCoverageError) as info:
        ensure_series_coverage(_params(tmp_path), run, log=lines.append)
    message = str(info.value)
    assert lines == message.splitlines()
    assert ("'*.descriptors.npy': not migrated from the graph-cache input"
            in message)
    assert 'node-table settings changed' not in message
    assert (f'2. fill the series before the next job: prefill --params '
            f'{tmp_path / "params1.py"} --run {run}' in message)
    assert '3. construct the featurizer with refill=True' in message


def test_with_refill_the_launcher_announces_the_refill(tmp_path, capsys):
    class Refiller:
        def route(self, coverage, workdir):
            return f'the route for {len(coverage.missing)} in {workdir}'

        def __call__(self, *args, **kwargs):
            raise AssertionError('the launcher must not refill')

    register_series(SERIES, refill=True, refiller=Refiller())
    run = _run_with_gap(tmp_path)
    coverage = check_series_coverage(_params(tmp_path), run)
    assert len(coverage.missing) == 3
    out = capsys.readouterr().out
    assert out.startswith('SERIES REFILL: 3 of 5 trajectories (21 frames)')
    assert f'by the route for 3 in {tmp_path}' in out


def test_with_refill_a_worker_refills_once_and_logs_it(tmp_path):
    calls = []

    def refiller(coverage, params_file, workdir, jobs, log, progress):
        calls.append((params_file, workdir, jobs))
        for trajectory, _ in coverage.missing:
            np.save(f'{trajectory}.{SERIES}.npy',
                    np.ones((1, 6), dtype=np.float32))
        log('a line of the refill')
        progress(len(coverage.missing), coverage.missing_frames)
        return dict(route='the test route')

    register_series(SERIES, refill=True, refiller=refiller)
    run = _run_with_gap(tmp_path)
    lines = []
    assert ensure_series_coverage(_params(tmp_path), run, role='shoot R0',
                                  log=lines.append, jobs=3)
    assert calls == [(str(tmp_path / 'params1.py'), str(tmp_path), 3)]
    assert series_coverage(run, SERIES).complete
    assert all(line.startswith('SERIES REFILL:') for line in lines)
    assert 'refills them now with 3 processes by its refill callable' in \
        lines[0]
    assert ', shoot R0)' in lines[0]
    assert lines[-1].startswith(f'SERIES REFILL: done: {SERIES!r} of 3 '
                                f'trajectories (21 frames)')
    assert 'refilled by the test route' in lines[-1]
    assert not os.path.exists(Path(run) / series.REFILL_LOCK)
    assert not os.path.exists(Path(run) / series.REFILL_INFO)
    # complete now: a second start does not refill
    assert ensure_series_coverage(_params(tmp_path), run, log=lines.append)
    assert len(calls) == 1


def test_a_refill_that_leaves_trajectories_missing_raises(tmp_path):
    register_series(SERIES, refill=True,
                    refiller=lambda coverage, **kwargs: None)
    run = _run_with_gap(tmp_path)
    lines = []
    with pytest.raises(SeriesCoverageError) as info:
        ensure_series_coverage(_params(tmp_path), run, log=lines.append)
    assert '3 trajectories still have no file of the series' in \
        str(info.value)
    assert all(line.startswith('SERIES REFILL:') for line in lines)
    assert not os.path.exists(Path(run) / series.REFILL_LOCK)
