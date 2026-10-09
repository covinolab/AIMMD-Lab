"""The series check where jobs start: the launcher and every worker.

A run whose trajectories lack the params' node-table series (here: an older
``n_max``, or a campaign that still has the coordinate series of the
graph-cache input) must not start silently featurizing every frame again.
These tests load a toy params file in node-table mode with aimmd.Params and
check that

- `Launcher.create_job` refuses to write the job (and `Launcher.run` to
  start) with ``refill=False``, naming the series that is there and the
  remedies, and with ``refill=True`` announces the refill and writes the job;
- a worker raises before its task (no MD) with ``refill=False``, and with
  ``refill=True`` refills the series before its task starts;
- a complete run starts as before;
- a worker whose params file (``paramsN.py``) was written before the params
  file that builds the featurizer was edited stops with a mismatch error
  naming both series, also when the series it names is complete;
- `Launcher.run` marks the start of its launch for its processes (a failed
  refill of an earlier launch does not stop them).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.core import series
from aimmd.core.series import SeriesCoverageError, series_coverage
from aimmd.network.nodetables import _tool
from tests._nodetables_run import bits, expected_rows, series_file, \
    toy_featurizer
from tests._nodetables_toy import ENVIRONMENT_SELECTION
from tests._series_refill import (make_full_campaign, remove_legacy_series,
                                  run_trajectories)

WORKTREE = Path(__file__).resolve().parents[1]

# a worker of the job: a fresh process that loads the job's params file
WORKER = '''
import sys
import aimmd
from aimmd.core import series
params = aimmd.Params(sys.argv[1], initial_paths=None, save=False)
try:
    series.ensure_series_coverage(params, 'run1', log=print)
except series.SeriesMismatchError:
    sys.exit(3)
print('STARTED')
'''


@pytest.fixture
def no_hardware(monkeypatch):
    monkeypatch.setattr('aimmd.launcher._helpers.get_num_cpus', lambda: 4)
    monkeypatch.setattr('aimmd.launcher._helpers.get_num_gpus', lambda: 0)
    monkeypatch.setattr(series, '_cpus', lambda: 1)


def _older_n_max(campaign):
    """The run holds the series of n_max=56 (and no legacy series); the
    params file now says n_max=64."""
    old = toy_featurizer(campaign.folder, 56)
    _tool.prefill(_tool.ParamsFeaturizer(None, 'old', old, None),
                  [campaign.run], log=lambda line: None)
    remove_legacy_series(campaign.run)
    return old


def test_create_job_without_refill_refuses_before_writing_the_job(
        tmp_path, monkeypatch, no_hardware):
    campaign = make_full_campaign(tmp_path / 'campaign')
    old = _older_n_max(campaign)
    monkeypatch.chdir(campaign.folder)
    params = aimmd.Params.load('params.py')
    launcher = aimmd.Launcher(params, 'run1')

    with pytest.raises(SeriesCoverageError) as info:
        launcher.create_job('job.sh', n=1, nframes=10)
    assert not os.path.exists('job.sh')
    with pytest.raises(SeriesCoverageError):
        launcher.run(1, nframes=10, walltime=5)
    assert not os.path.exists('run1/chainR0/pool.log')     # nothing built

    message = str(info.value)
    run = os.path.abspath('run1')
    assert (f"6 of 6 trajectories (36 frames) of run {run!r} have no "
            f"{params.descriptors_series!r} series file") in message
    assert (f"Next to them is the series {old.series!r} (6 trajectories): "
            f"the node-table settings changed since these frames were "
            f"featurized.") in message
    assert 'not been migrated' not in message
    assert '1. restore the node-table settings' in message
    assert (f'2. fill the series before the next job: python -m '
            f'aimmd.network.nodetables prefill --params '
            f'{os.path.abspath("params.py")} --run {run} (or the '
            f"campaign's prefill_nodetables.sh)") in message
    assert '3. construct the featurizer with refill=True' in message


def test_create_job_with_refill_announces_it_and_writes_the_job(
        tmp_path, monkeypatch, capsys, no_hardware):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True)
    old = _older_n_max(campaign)
    monkeypatch.chdir(campaign.folder)
    params = aimmd.Params.load('params.py')
    capsys.readouterr()

    aimmd.Launcher(params, 'run1').create_job('job.sh', n=1, nframes=10)

    assert os.path.exists('job.sh')
    notice, = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith('SERIES REFILL:')]
    assert notice.startswith('SERIES REFILL: 6 of 6 trajectories (36 '
                             'frames) of run')
    assert (f'the job first refills them in one process, by repacking the '
            f'rows of {old.series} (n_max 56 -> 64; no trajectory is '
            f'read)') in notice
    # the launcher only announces it
    assert not series_coverage('run1', params.descriptors_series).complete


def test_a_worker_without_refill_stops_before_any_md(tmp_path, monkeypatch):
    campaign = make_full_campaign(tmp_path / 'campaign')  # legacy series
    monkeypatch.chdir(campaign.folder)
    params = aimmd.Params.load('params.py')
    tasks = []
    for task in ('_shoot', '_free', '_train'):
        monkeypatch.setattr(aimmd.Worker, task,
                            lambda self, *args, task=task, **kwargs:
                            tasks.append(task))
    worker = aimmd.Worker(params, 'run1', log_file='stdout')

    for start in (lambda: worker.shoot('R', 0),
                  lambda: worker.free('A', 0),
                  lambda: worker.train(1)):
        with pytest.raises(SeriesCoverageError) as info:
            start()
    assert tasks == []
    message = str(info.value)
    assert ("Next to 6 trajectories of them is the series "
            "'*.descriptors.npy': this campaign has not been migrated from "
            "the graph-cache ('sqlite') input.") in message
    assert 'settings changed' not in message
    assert os.getcwd() == campaign.folder


def test_a_worker_with_refill_refills_before_its_task(tmp_path, monkeypatch,
                                                      capsys):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True)
    monkeypatch.chdir(campaign.folder)
    monkeypatch.setattr(series, '_cpus', lambda: 1)
    params = aimmd.Params.load('params.py')
    seen = []

    def free(self, *args, **kwargs):
        seen.append(series_coverage('run1', params.descriptors_series)
                    .complete)

    monkeypatch.setattr(aimmd.Worker, '_free', free)
    capsys.readouterr()
    aimmd.Worker(params, 'run1', log_file='stdout').free('A', 0)

    assert seen == [True]
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith('SERIES REFILL:')]
    assert 'refills them now with 1 process by featurizing' in lines[0]
    assert ', free run1/freeA (worker 0))' in lines[0]
    assert lines[-1].startswith('SERIES REFILL: done:')
    featurizer = toy_featurizer(campaign.folder, 64)
    for trajectory in run_trajectories(campaign.run):
        assert np.array_equal(
            bits(np.load(series_file(trajectory, featurizer.series))),
            bits(expected_rows(featurizer, trajectory)))


def test_a_complete_run_starts_as_before(tmp_path, monkeypatch, capsys,
                                         no_hardware):
    campaign = make_full_campaign(tmp_path / 'campaign')
    featurizer = toy_featurizer(campaign.folder, 64)
    _tool.prefill(_tool.ParamsFeaturizer(None, 'F', featurizer, None),
                  [campaign.run], log=lambda line: None)
    monkeypatch.chdir(campaign.folder)
    params = aimmd.Params.load('params.py')
    tasks = []
    monkeypatch.setattr(aimmd.Worker, '_shoot',
                        lambda self, *args, **kwargs: tasks.append(args))
    capsys.readouterr()

    aimmd.Worker(params, 'run1', log_file='stdout').shoot('R', 0)
    aimmd.Launcher(params, 'run1').create_job('job.sh', n=1, nframes=10)

    assert len(tasks) == 1 and os.path.exists('job.sh')
    assert 'SERIES' not in capsys.readouterr().out


def test_a_job_written_before_the_params_file_changed_stops(tmp_path,
                                                            monkeypatch):
    campaign = make_full_campaign(tmp_path / 'campaign', refill=True)
    remove_legacy_series(campaign.run)
    monkeypatch.chdir(campaign.folder)
    params = aimmd.Params.load('params.py')        # writes params1.py
    old = toy_featurizer(campaign.folder, 64)
    assert params.descriptors_series == old.series
    _tool.prefill(_tool.ParamsFeaturizer(None, 'F', old, None),
                  [campaign.run], log=lambda line: None)
    # params.py is edited after the job script was written
    other = ENVIRONMENT_SELECTION.replace('5.0', '4.5')
    source = Path('params.py').read_text()
    assert source.count(ENVIRONMENT_SELECTION) == 1
    Path('params.py').write_text(source.replace(ENVIRONMENT_SELECTION,
                                                other))
    new = toy_featurizer(campaign.folder, 64, environment=other)
    files = sorted(Path(campaign.run).rglob(f'*.{old.series}.npy'))
    before = {fname: fname.read_bytes() for fname in files}

    result = subprocess.run(
        [sys.executable, '-c', WORKER, str(params.path)],
        cwd=campaign.folder, capture_output=True, text=True, timeout=600,
        env=dict(os.environ, PYTHONPATH=str(WORKTREE),
                 CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))

    assert result.returncode == 3, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines()
             if line.startswith('SERIES')]
    assert lines and all(line.startswith('SERIES CHECK:') for line in lines)
    text = '\n'.join(lines)
    assert (f"The params file this job uses, {str(params.path)!r}, was "
            f"written for the descriptor series {old.series!r}, but its "
            f"node-table featurizer now computes {new.series!r}") in text
    assert 'rerun the job-script generator (Launcher.create_job)' in text
    assert 'STARTED' not in result.stdout
    assert {fname: fname.read_bytes() for fname in files} == before
    assert not list(Path(campaign.run).rglob(f'*.{new.series}.npy'))


def test_a_launch_marks_its_start_for_its_processes(tmp_path, monkeypatch,
                                                    no_hardware):
    # a failed refill of this launch stops its later processes; one of an
    # earlier launch does not (aimmd.core.series.JOB_START)
    from aimmd.execute.processes import ProcessExecutor
    campaign = make_full_campaign(tmp_path / 'campaign')
    _tool.prefill(_tool.ParamsFeaturizer(
        None, 'F', toy_featurizer(campaign.folder, 64), None),
        [campaign.run], log=lambda line: None)
    monkeypatch.chdir(campaign.folder)
    monkeypatch.setenv(series.JOB_START, '0')     # restored after the test
    monkeypatch.delenv(series.JOB_START)
    params = aimmd.Params.load('params.py')
    seen = []
    monkeypatch.setattr(ProcessExecutor, 'run', lambda self, *args, **kw:
                        seen.append(os.environ.get(series.JOB_START)))
    monkeypatch.setattr(ProcessExecutor, 'clear', lambda self, *args, **kw:
                        None)
    before = time.time()
    aimmd.Launcher(params, 'run1').run(1, nframes=10, walltime=0.1)
    assert len(seen) == 1 and before <= float(seen[0]) <= time.time()
