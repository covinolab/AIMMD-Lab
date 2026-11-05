"""``Worker.run`` publishes its task in ``aimmd._config.WORKER_TASK``.

Code that may only claim per-task resources (such as a GPU) inside a worker
task, and not while a params file is loaded, reads the task from
``aimmd._config.WORKER_TASK`` at call time. ``Worker.run`` sets it next to
``self.task`` and resets it to None in its ``finally``, also when the task
raises.
"""

from types import SimpleNamespace

import pytest

from aimmd import _config
from aimmd.worker._run import WorkerRun


def task_read_at_call_time():
    """How a consumer reads the task: from the module, at call time."""
    return getattr(_config, 'WORKER_TASK', None)


class TaskRecordingWorker(WorkerRun):
    """A ``WorkerRun`` whose tasks record the published task (as test_worker_runtime_unit)."""

    def __init__(self, tmp_path, fail=False):
        self.directory = str(tmp_path / 'run')
        self._directory = self.directory
        self.params = SimpleNamespace(parent=str(tmp_path))
        self.log_file = self.original_stdout = 'stdout'
        self.localid = 0
        self.fail = fail
        self.seen = []

    def _bind_resources(self):
        self.seen.append(('bind', _config.WORKER_TASK))

    def _update_stop_condition(self, **kwargs):
        pass

    def _terminate_operations(self):
        self.seen.append(('terminate', _config.WORKER_TASK))

    def _reset_stop_condition(self):
        pass

    def _task(self, name):
        self.seen.append((name, _config.WORKER_TASK, task_read_at_call_time()))
        if self.fail:
            raise RuntimeError(f'{name} failed')
        return name

    def _shoot(self, *args, **kwargs):
        return self._task('shoot')

    def _free(self, *args, **kwargs):
        return self._task('free')

    def _train(self, *args, **kwargs):
        return self._task('train')

    def _kinetics_convergence(self, *args, **kwargs):
        return self._task('kinetics_convergence')


@pytest.mark.parametrize('task', ['shoot', 'free', 'train', 'kinetics_convergence'])
def test_worker_run_sets_task(tmp_path, task):
    """Set during the task (also while resources are bound), None after it."""
    worker = TaskRecordingWorker(tmp_path)
    assert _config.WORKER_TASK is None
    assert worker.run(task) == task
    assert (task, task, task) in worker.seen
    assert ('bind', task) in worker.seen          # _bind_resources runs inside the task
    assert _config.WORKER_TASK is None
    assert task_read_at_call_time() is None


def test_worker_run_resets_task_after_an_exception(tmp_path):
    worker = TaskRecordingWorker(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='train failed'):
        worker.run('train')
    assert ('train', 'train', 'train') in worker.seen
    assert _config.WORKER_TASK is None


def test_worker_run_resets_task_for_an_unknown_task(tmp_path):
    worker = TaskRecordingWorker(tmp_path)
    with pytest.raises(TypeError):
        worker.run('bogus')
    assert _config.WORKER_TASK is None


def test_consecutive_tasks_in_one_process(tmp_path):
    """One process may run several tasks (the in-process toy tests): each is published."""
    worker = TaskRecordingWorker(tmp_path)
    worker.run('train')
    worker.run('shoot')
    tasks = [entry[1] for entry in worker.seen if entry[0] in ('train', 'shoot')]
    assert tasks == ['train', 'shoot']
    assert _config.WORKER_TASK is None
