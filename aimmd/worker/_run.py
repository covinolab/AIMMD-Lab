"""
aimmd.worker._run
================

Task dispatch and execution wrapper for AIMMD workers.

This module defines :class:`WorkerRun`, a mixin providing the high-level
:meth:`~WorkerRun.run` method used by worker entry points (e.g. CLI scripts or
scheduler-launched processes).

The :meth:`WorkerRun.run` wrapper is responsible for the common worker runtime
procedure:

- record the selected task name for reporting,
- switch the current working directory to the parameters' parent directory
  (ensuring all relative paths resolve consistently),
- print a short startup message,
- bind CPU/GPU resources for the given worker ``localid``,
- clear global reader caches that must not leak across tasks,
- configure stop conditions (walltime, max steps, max frames),
- reset per-task counters (start time, total steps, total frames),
- dispatch to the corresponding task implementation method.

Concrete worker classes are expected to implement the task methods:

- :meth:`_shoot`
- :meth:`_free`
- :meth:`_train`

They are also expected to provide helper methods and attributes used here,
typically via :class:`~aimmd.worker._helpers.WorkerHelpers` and
:class:`~aimmd.worker._properties.WorkerProperties`.

Directory handling
------------------
Workers always execute tasks from within ``params.parent`` (the directory that
owns the parameters file / working directory). The attribute :attr:`_directory`
is updated to be relative to ``params.parent`` so that downstream components
can build paths consistently.

Cache handling
--------------
The global caches :data:`aimmd._config.MDA_CACHE` and :data:`aimmd._config.NPY_CACHE`
are cleared before each task to avoid reusing readers/arrays created under a
different working directory or task context.

Error handling
--------------
Exceptions are re-raised after optionally printing a traceback to the *original*
stdout when the worker has redirected output to a log file, making scheduler
logs easier to inspect.

Cleanup is always executed in the ``finally`` block:

- restore the previous working directory,
- restore :attr:`_directory`,
- run termination cleanup operations (e.g., close logs),
- reset stop-condition thresholds.

Notes
-----
This wrapper intentionally does not catch termination signals itself. Instead,
signal handlers are expected to set :attr:`termination_signal`, and the task
methods should periodically query :attr:`must_stop` (or equivalent) to exit
cleanly.
"""

# external
import os
import time
import traceback
from abc import ABC

# aimmd imports
from .._config import MDA_CACHE, NPY_CACHE, print
from ..core.utils import now


class WorkerRun(ABC):
    """
    Mixin implementing the top-level worker task runner.

    The concrete worker is expected to provide:

    - configuration and logging attributes (e.g., :attr:`directory`,
      :attr:`params`, :attr:`log_file`, :attr:`original_stdout`),
    - helper methods (e.g., :meth:`_bind_resources`,
      :meth:`_update_stop_condition`, :meth:`_terminate_operations`,
      :meth:`_reset_stop_condition`),
    - task implementations (e.g., :meth:`_shoot`, :meth:`_free`, :meth:`_train`).
    """

    def run(self, task, *args, **kwargs):
        """
        Run a single worker task.

        Parameters
        ----------
        task : str
            Task identifier. Supported values are:

            - ``'shoot'``: dispatches to :meth:`_shoot`
            - ``'free'``: dispatches to :meth:`_free`
            - ``'train'``: dispatches to :meth:`_train`

        *args
            Positional arguments forwarded to the selected task method.
        **kwargs
            Keyword arguments forwarded to the selected task method. This method
            also consumes stop-condition keys via :meth:`_update_stop_condition`
            (typically ``nsteps``, ``nframes``, ``walltime``), removing them from
            ``kwargs`` before dispatch.

        Returns
        -------
        object
            The return value of the selected task method.

        Raises
        ------
        TypeError
            If ``task`` does not match a supported task name.
        Exception
            Re-raises any exception thrown by task execution after optional
            traceback reporting.

        Side Effects
        ------------
        - Changes the current working directory to :attr:`params.parent` for the
          duration of the task.
        - Clears global caches :data:`~aimmd._config.MDA_CACHE` and
          :data:`~aimmd._config.NPY_CACHE`.
        - Resets per-task counters (:attr:`_t0`, :attr:`_total_steps`,
          :attr:`_total_frames`).
        - Ensures cleanup in a ``finally`` block even on errors.
        """
        # initialize
        self.task = task

        try:
            # Always run from the parameters' directory.
            cwd = os.getcwd()
            self._directory = os.path.relpath(
                self.directory, self.params.parent)
            # Directory relative to params' folder.
            os.chdir(self.params.parent)

            # report
            if self.log_file == self.original_stdout:
                print(f"Press Control+C to interrupt.")
            else:
                print(f"Starting: worker{self.localid}, {task} {now()}")

            # bind resources
            self._bind_resources()

            # clear caches
            MDA_CACHE.clear()
            NPY_CACHE.clear()

            # update stop condition and remove from kwargs
            self._update_stop_condition(**kwargs)

            # reset time, total steps, total frames
            self._t0 = time.time()
            self._total_steps = 0
            self._total_frames = 0

            # execute task
            if task == 'shoot':
                return self._shoot(*args, **kwargs)
            if task == 'free':
                return self._free(*args, **kwargs)
            if task == 'train':
                return self._train(*args, **kwargs)
            raise TypeError(f'{task} not implemented in Worker.run')

        except Exception as exception:
            if self.log_file != self.original_stdout:
                traceback.print_exc(file=self.original_stdout)
            raise exception

        finally:
            os.chdir(cwd)  # back to main folder
            self._directory = self.directory
            self._terminate_operations()
            self._reset_stop_condition()
