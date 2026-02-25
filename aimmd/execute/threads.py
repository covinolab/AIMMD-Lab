"""
aimmd.execute.threads
=====================

Thread-based task executor implementation.

This module defines `ThreadExecutor`, a concrete implementation of
:class:`aimmd.execute.base.TaskExecutor` backed by `threading.Thread`.

Threads are created as daemon threads, meaning:
- they will not prevent Python from exiting if only daemon threads remain,
- they are appropriate for background helper tasks,
- but they are *not* suitable when you require guaranteed cleanup/finalization.

Notes
-----
- Python does not provide a safe, general mechanism to "terminate" threads.
  Therefore, `TaskExecutor.stop()` relies on backend hooks; for threads,
  `_terminate`/`_kill` are intentionally not implemented here. Consumers
  should design thread targets to exit cooperatively (e.g., by checking a
  shared stop condition/event).
"""

# external
import threading

# aimmd imports
from .base import TaskExecutor


class ThreadExecutor(TaskExecutor):
    """
    Task executor using `threading.Thread`.

    Attributes
    ----------
    __task_name__ : str
        Human-readable label for the backend task type.
    """

    __task_name__ = 'Thread'

    def _initialize(self, target):
        """
        Create a new daemon thread for the given target.

        Parameters
        ----------
        target : callable
            Callable executed by the thread. In practice, this is a wrapper
            created by `TaskExecutor._build` that calls `target_wrapper`.

        Returns
        -------
        threading.Thread
            A not-yet-started daemon thread.
        """
        # Daemon thread: does not block interpreter shutdown.
        return threading.Thread(target=target, daemon=True)
