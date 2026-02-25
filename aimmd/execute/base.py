"""
aimmd.execute.base
==================

Core abstraction layer for task execution in AIMMD.

This module defines the abstract class `TaskExecutor`, which provides
a unified interface for managing multiple concurrent tasks.

A *task* may be either:
- a multiprocessing.Process (see ProcessExecutor),
- or a threading.Thread (see ThreadExecutor).

Design philosophy
-----------------
- Uniform lifecycle management of tasks.
- Allow restartability.
- Support both parallel and sequential execution.
- Defer backend-specific details (terminate, kill, join, etc.)
  to concrete implementations.

Important
---------
Concrete subclasses must implement:

- `_initialize(self, target)`
- `_terminate(self, localid)`
- `_kill(self, localid)`
- `_close(self, localid)`
- optionally `_closed(self, localid)`
"""

# external
import time
import numpy as np
import functools
import traceback
import multiprocessing
from abc import ABC, abstractmethod

# aimmd imports
from .utils import target_wrapper
from ..core.utils import now

# Always use spawn for safe multiprocessing in AIMMD
ctx = multiprocessing.get_context('spawn')


class TaskExecutor(ABC):
    """
    Abstract base class for managing concurrent tasks.

    This class provides:
    - Task registration
    - Lifecycle control (start, stop, kill, reset)
    - Parallel or sequential execution
    - Restartability

    Subclasses specify the execution backend
    (e.g., process-based or thread-based).

    Notes
    -----
    Tasks are indexed locally via `localid`, corresponding to the order
    in which they were added.
    """

    def __init__(self):
        """
        Initialize an empty TaskExecutor.
        """
        self._tasks = []
        self._names = []
        self._targets = []
        self._args = []
        self._kwargs = []

    def __len__(self):
        """
        Return the number of registered tasks.

        Returns
        -------
        int
            Number of tasks currently managed.
        """
        return len(self._tasks)

    def __iter__(self):
        """
        Iterate over task objects.

        Returns
        -------
        iterator
            Iterator over internal task objects.
        """
        return iter(self._tasks)

    def __getitem__(self, key):
        """
        Get task(s) by index or selection.

        Parameters
        ----------
        key : int, slice, list-like, or None
            Task selector.

        Returns
        -------
        object or list
            Selected task(s).
        """
        if key is None:
            return list(self._tasks)
        if hasattr(key, '__len__'):
            return [self[key] for key in np.arange(len(self))[key].flatten()]
        return self._tasks[key]

    def __repr__(self):
        """
        Return human-readable representation of executor and tasks.
        """
        text = [f'{self.__class__.__name__} with {len(self)} '
                f'task{"s" if len(self) != 1 else ""}']
        if len(self):
            text.append(':')
        for i in range(len(self)):
            text.append(f'\n... {self._names[i]}')
        return ''.join(text)

    @abstractmethod
    def _initialize(self, target):
        """
        Backend-specific task creation.

        Parameters
        ----------
        target : callable
            Target function to run.

        Returns
        -------
        object
            Backend task object (Process or Thread).
        """
        pass

    def _alive(self, localid):
        """
        Check if a task is alive.

        Parameters
        ----------
        localid : int
            Index of task.

        Returns
        -------
        bool
        """
        if self._tasks[localid] is None:
            return False
        if self._closed(localid):
            return False
        return self._tasks[localid].is_alive()

    def _terminate(self, localid):
        """
        Gracefully terminate task.

        Must be implemented by subclass.
        """
        pass

    def _kill(self, localid):
        """
        Forcefully kill task.

        Must be implemented by subclass.
        """
        pass

    def _close(self, localid):
        """
        Release backend resources.

        Must be implemented by subclass.
        """
        pass

    def _closed(self, localid):
        """
        Whether backend considers task closed.

        Returns
        -------
        bool
        """
        return False

    def _reset(self, localid, timeout=20.):
        """
        Replace an existing task with a pristine one.

        Parameters
        ----------
        localid : int
        timeout : float, optional
            Graceful shutdown timeout (seconds).
        """
        self.stop(localid, timeout=timeout)
        self._close(localid)
        self._tasks[localid] = None

    def _build(self, localid):
        """
        Construct backend task wrapper.

        Returns
        -------
        object
            Backend task object.
        """
        target = self._targets[localid]
        args = self._args[localid]
        kwargs = self._kwargs[localid]
        name = self._names[localid]

        target = functools.partial(
            target_wrapper, target, name, *args, **kwargs
        )

        self._tasks[localid] = self._initialize(target)
        return self._tasks[localid]

    @property
    def tasks(self):
        """List of backend task objects."""
        return list(self._tasks)

    @property
    def targets(self):
        """List of registered target callables."""
        return list(self._targets)

    @property
    def args(self):
        """List of positional arguments per task."""
        return list(self._args)

    @property
    def kwargs(self):
        """List of keyword arguments per task."""
        return list(self._kwargs)

    @property
    def alive(self):
        """
        Boolean array indicating running tasks.
        """
        return np.array(
            [self._alive(i) for i in range(len(self))],
            dtype=bool
        )

    def add(self, target, *args, name='', **kwargs):
        """
        Register a new task.

        Parameters
        ----------
        target : callable
            Function to execute.
        *args
            Positional arguments passed to target.
        name : str, optional
            Human-readable name. Auto-generated if empty.
        **kwargs
            Keyword arguments passed to target.
        """
        if not name:
            name = [f'{target.__name__}(']
            for arg in args:
                name.append(f'{arg}, ')
            for key, arg in kwargs.items():
                name.append(f'{key}={arg}, ')
            name[-1] = name[-1].rstrip(', ')
            name = ''.join(name + [')'])

        self._tasks.append(None)
        self._names.append(name)
        self._targets.append(target)
        self._args.append(args)
        self._kwargs.append(kwargs)

    def sequential(self, key=None):
        """
        Run selected tasks sequentially in current process.

        Parameters
        ----------
        key : selection, optional
            Task selector.
        """
        localids = np.arange(len(self))[key].flatten()
        for localid in localids:
            target_wrapper(
                self._targets[localid],
                self._names[localid],
                *self._args[localid],
                **self._kwargs[localid]
            )

    def run(self, key=None, parallel=True, timeout=20.):
        """
        Execute selected tasks.

        Parameters
        ----------
        key : selection, optional
            Task selector.
        parallel : bool, optional
            Whether to run tasks in parallel.
        timeout : float, optional
            Restart timeout.
        """
        localids = np.arange(len(self))[key].flatten()

        if not len(localids):
            return

        if not parallel or len(localids) == 1:
            return self.sequential(localids)

        for localid in localids:
            self._reset(localid)
            task = self._build(localid)
            task.start()

    def stop(self, key=None, timeout=20.):
        """
        Stop selected tasks.

        Parameters
        ----------
        key : selection, optional
        timeout : float, optional
            Graceful shutdown time (seconds).
        """
        localids = list(np.arange(len(self))[key].flatten())

        t0 = time.time()
        terminating = set()

        # Graceful termination loop
        while time.time() - t0 < timeout and len(localids):
            for localid in localids[:]:
                if self._alive(localid):
                    if localid not in terminating:
                        self._terminate(localid)
                        terminating.add(localid)
                else:
                    localids.remove(localid)

        # Forced termination
        for localid in localids:
            if self._alive(localid):
                self._kill(localid)
                time.sleep(timeout)

        # Final check
        for localid in localids:
            if self._alive(localid):
                print(f'Warning: some processes are still alive {now()}')
                break

    def clear(self, timeout=20.):
        """
        Stop and remove all tasks.

        Parameters
        ----------
        timeout : float, optional
            Graceful shutdown timeout.
        """
        self.stop(self.alive, timeout=timeout)

        for localid in range(len(self)):
            self._close(localid)

        self._tasks = []
        self._names = []
        self._targets = []
        self._args = []
        self._kwargs = []
