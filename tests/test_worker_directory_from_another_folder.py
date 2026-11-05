"""Shoot and free workers find their run when started outside the params folder.

``Worker.run`` changes into the params folder (``params.parent``) before a
task and stores the run directory relative to it in ``self._directory``;
``self.directory`` keeps the name the worker was given, which is relative to
the folder the worker was started from. Inside a task only ``_directory``
points at the run. The trainer always used it; the shoot and free workers
used ``self.directory`` to load the chains and to wait for the network and
the bins. Started from another folder with a relative run directory, a shoot
worker then loaded empty chains and registered its next shot as
``path000001`` again, over the existing file, and a free worker waited for a
network in a folder that does not exist.

The tests below give the worker a run directory relative to its start folder
and change into the params folder exactly as ``Worker.run`` does.
"""
import os

import numpy as np

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker._free import WorkerFree
from aimmd.worker._shoot import WorkerShoot
from aimmd.worker._simulate import WorkerSimulate
from tests._helpers_unit import build_params_file, build_path


TRANSITION = np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32)


def start_outside_the_params_folder(monkeypatch, tmp_path):
    """Return (params folder, run directory as given, as Worker.run makes it).

    The worker is started in `tmp_path` with the run directory
    ``project/run``; the params file lives in ``project``.
    """
    project = tmp_path / 'project'
    (project / 'run').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)                     # the start folder
    directory = os.path.join('project', 'run')      # as the worker was given it
    _directory = os.path.relpath(directory, project)  # as Worker.run computes it
    monkeypatch.chdir(project)                      # Worker.run: into params.parent
    return project, directory, _directory


class TinyShootWorker(WorkerShoot):
    """Concrete host for `_shoot` that stops once its chain is loaded."""

    def __init__(self, params, initial_paths, directory, _directory):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = directory
        self._directory = _directory
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


class TinyFreeWorker(WorkerFree, WorkerSimulate):
    """Concrete host for `_free`."""

    def __init__(self, params, initial_paths, directory, _directory):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = directory
        self._directory = _directory
        self.must_stop = False
        self.termination_signal = 0
        self.termination_timeout = 1.0
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def test_shoot_worker_loads_its_chain_when_started_elsewhere(monkeypatch, tmp_path):
    project, directory, _directory = start_outside_the_params_folder(
        monkeypatch, tmp_path)
    (project / 'run' / 'chainR0').mkdir()
    existing = build_path(project / 'run' / 'chainR0', stem='path000001',
                          positions=TRANSITION)
    initial = build_path(tmp_path, stem='initial', positions=TRANSITION)
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        chain_type='rfps',
        at_least_one_transition_in_pool=False,
        nbins=1,                       # no network/bins gate
        selection_pool_size=1,
        free_overriding_states='',
        engine='toy',
        trajectory_extension='.xtc',
    )
    worker = TinyShootWorker(params, PathEnsemble(initial), directory, _directory)

    loaded = []

    def record_and_stop(pool, size, chain, *args, **kwargs):
        loaded.append(chain)
        worker.must_stop = True        # leave `_shoot` before the main loop

    monkeypatch.setattr('aimmd.worker._shoot.update_selection_pool',
                        record_and_stop)
    worker._shoot(target_state='R', k=0, sweep=False)

    assert [len(chain) for chain in loaded] == [1], (
        'the existing chain was not found: the next shot would be registered '
        'as path000001 again')
    assert os.path.samefile(loaded[0].fnames[0], existing.fname)
    assert worker.total_steps == 1
    # the progress bar still names the run as the worker was given it
    assert worker._location == f'{directory}/chainR0'


def test_free_worker_waits_for_the_network_of_its_run(monkeypatch, tmp_path):
    project, directory, _directory = start_outside_the_params_folder(
        monkeypatch, tmp_path)
    initial = build_path(project, stem='initial', positions=TRANSITION)
    params = aimmd.Params.load(build_params_file(project, initial.fname),
                               save=False)
    params.update(nbins=2, save=False)
    worker = TinyFreeWorker(params, PathEnsemble(initial), directory, _directory)

    asked = []

    def update_network(directory, *args, **kwargs):
        asked.append(directory)
        worker.termination_signal = 1  # leave `_free` after the first try
        raise RuntimeError('no network yet')

    params.__dict__['update_network'] = update_network
    worker._free(target_state='A', k=0, total=1, wait=True)

    assert asked, 'the free worker never asked for the network'
    assert os.path.isdir(asked[0]) and os.path.samefile(
        asked[0], project / 'run'), (
        f'the free worker waits for a network in {asked[0]!r}, '
        f'not in its run directory')
