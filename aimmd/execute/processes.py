"""
aimmd.execute.processes
=======================

Process-based task executor implementation.

This module defines `ProcessExecutor`, a concrete implementation of
:class:`aimmd.execute.base.TaskExecutor` backed by
`multiprocessing.Process` objects (spawn context).

The executor provides:
- Process construction via the global `ctx` spawn context
- Graceful termination (terminate)
- Forced termination (also terminate; Python processes do not have a
  distinct "kill" API in the standard library)
- Resource cleanup via `join`

Notes
-----
- This executor relies on `multiprocessing.get_context('spawn')` from
  `aimmd.execute.base` (imported as `ctx`).
- The `_closed` state is backend-specific and uses a private attribute
  of `multiprocessing.Process`. This is practical but not part of the
  public API; behavior may vary across Python versions.
"""

# external
import os
import signal

# aimmd imports
from .base import TaskExecutor, ctx


class ProcessExecutor(TaskExecutor):
    """
    Task executor using `multiprocessing.Process`.

    Instances of this executor manage a list of processes created with a
    spawn start method (safer across platforms and with CUDA/GPU contexts).

    Attributes
    ----------
    __task_name__ : str
        Human-readable label for the backend task type.
    """

    __task_name__ = 'Process'

    def _initialize(self, target):
        """
        Create a new process for the given target.

        Parameters
        ----------
        target : callable
            Callable to run in the child process. In practice, this will be a
            wrapper produced by `TaskExecutor._build`, which calls
            `aimmd.execute.utils.target_wrapper`.

        Returns
        -------
        multiprocessing.Process
            A not-yet-started process object.
        """
        return ctx.Process(target=target)

    def _terminate(self, localid):
        """
        Request graceful termination of a process.

        Parameters
        ----------
        localid : int
            Index of the task to terminate.

        Notes
        -----
        `multiprocessing.Process.terminate()` sends a termination signal
        (on POSIX typically SIGTERM) or uses platform-specific mechanisms.
        """
        task = self._tasks[localid]
        if task and task.is_alive():
            task.terminate()

    def _kill(self, localid):
        """
        Forcefully kill a process.

        Parameters
        ----------
        localid : int
            Index of the task to kill.

        Notes
        -----
        The standard library provides `.terminate()` but not a separate
        "kill" method (unlike subprocess). Here we alias "kill" to
        terminate to keep a unified interface across backends.
        """
        task = self._tasks[localid]
        if task and task.is_alive():
            task.terminate()

    def _close(self, localid):
        """
        Join a process briefly to allow cleanup.

        Parameters
        ----------
        localid : int
            Index of the task to close.

        Notes
        -----
        A small timeout is used to avoid blocking forever in pathological cases.
        """
        task = self._tasks[localid]
        if task:
            task.join(timeout=0.1)  # small timeout to avoid blocking forever

    def _closed(self, localid):
        """
        Determine whether the underlying process is marked as closed.

        Parameters
        ----------
        localid : int
            Index of the task to query.

        Returns
        -------
        bool
            True if the task exists and the backend considers it closed.

        Notes
        -----
        This uses the private attribute `_closed` of multiprocessing.Process.
        It is effective in practice but not guaranteed by the public API.
        """
        if self._tasks[localid] is None:
            return False
        return self._tasks[localid]._closed
