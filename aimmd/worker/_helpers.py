"""
aimmd.worker._helpers
====================

Helper mixins for AIMMD worker processes.

This module defines :class:`WorkerHelpers`, a lightweight mixin used by worker
implementations to provide consistent initialization, stop-condition handling,
signal handling, logging setup, and optional CPU/GPU resource binding.

The worker "shell" is intentionally small and predictable:

- initialize from a :class:`~aimmd.params.Params` instance or from a parameter
  file path,
- optionally bind CPU/GPU resources for a given ``localid``,
- handle termination requests via POSIX signals (SIGTERM, SIGINT),
- track simple stop conditions (walltime, maximum steps, maximum frames),
- route logging through the :attr:`log_file` interface (implemented by the
  concrete worker).

Notes
-----
- Signal handlers are registered in :meth:`WorkerHelpers._init`. This requires
  initialization to occur in the **main thread** of the worker process (the
  standard case for multiprocessing workers).
- The methods in this module are internal helpers (underscore-prefixed) and are
  expected to be used by worker subclasses.
"""

# external
import os
import sys
import time
import signal
from abc import ABC
from math import inf

# aimmd imports
from ..params import Params
from ..resources import bind_resources
from ..core.utils import now, remove


class WorkerHelpers(ABC):

    def _init(self, params, directory='.', localid=0,
              cpus_per_task='skip', gpus_per_task='skip',
              log_file='stdout', walltime=inf,
              nsteps=inf, nframes=inf,
              termination_timeout=60.):
        """
        Initialize a worker process.

        The worker process runs one AIMMD task at a time (free simulation,
        shooting simulation, NN training), typically under a batch scheduler
        where CPU/GPU resources are allocated per worker.

        This initializer:

        1) stores core worker attributes (parameters, directory, local ID,
           stop-condition thresholds),
        2) installs SIGTERM/SIGINT handlers that record a pending termination
           request in :attr:`termination_signal`,
        3) wires logging through :attr:`log_file` (usually a property on the
           concrete worker that redirects stdout/stderr and/or opens a file).

        Parameters
        ----------
        params : str or aimmd.params.Params
            Either an already-instantiated :class:`~aimmd.params.Params`, or a
            path to a parameters file that can be loaded into Params.
        directory : str, optional
            Working directory for the worker. Stored in both :attr:`directory`
            and :attr:`_directory`. Default is ``'.'``.
        localid : int, optional
            Local worker index, used for deterministic resource binding.
            Default is ``0``.
        cpus_per_task : {'skip', 'share', 'all'} or int, optional
            CPU allocation policy for this worker:

            - ``'skip'``: do not explicitly bind; only report availability.
            - ``'share'``: divide available CPUs among workers.
            - ``'all'``: bind all available CPUs to this worker.
            - int: bind exactly this many CPUs (policy interpreted by
              :func:`~aimmd.resources.bind_resources`).

            Default is ``'skip'``.
        gpus_per_task : {'skip', 'share', 'all'} or int, optional
            GPU allocation policy, analogous to ``cpus_per_task``. Default is
            ``'skip'``.
        log_file : {'stdout'} or str or file-like, optional
            Logging target. Assigned via :attr:`log_file`, which is expected to
            be implemented by the concrete worker class. If ``'stdout'``,
            output is left on the original stdout/stderr. Default is
            ``'stdout'``.
        walltime : float, optional
            Maximum allowed walltime in seconds for the current task before the
            worker should stop. Default is ``inf`` (no limit).
        nsteps : float, optional
            Maximum number of steps before stopping. Default is ``inf``.
        nframes : float, optional
            Maximum number of frames before stopping. Default is ``inf``.
        termination_timeout : float, optional
            Grace period (seconds) used by higher-level logic to allow a task
            to terminate cleanly after a stop/termination request. This mixin
            stores the value but does not enforce it directly. Default is
            ``60.`` seconds.

        Returns
        -------
        None

        See Also
        --------
        aimmd.resources.bind_resources
            Implements the CPU/GPU binding policy based on ``localid`` and the
            per-task configuration.
        """
        if isinstance(params, Params):
            self.params = params
        else:
            self.params = Params(params, initial_paths=None, save=False)

        self.directory = self._directory = directory
        self.localid = int(localid)
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        self.walltime = float(walltime)
        self.nsteps = float(nsteps)
        self.nframes = float(nframes)
        self.termination_timeout = float(termination_timeout)

        self.task = 'worker'  # for reporting
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._log_file = None
        self.t0 = inf  # when worker started
        self.termination_signal = None

        # Register signal handlers for all future tasks.
        # This assumes the worker installs handlers in its main thread.
        signal.signal(signal.SIGTERM, self._terminate_handler)
        signal.signal(signal.SIGINT, self._terminate_handler)

        # Assign log file through the public interface (typically a property).
        self.log_file = log_file

    def _terminate_handler(self, signum=None, frame=None):
        """
        Record a pending termination request.

        This is a minimal signal handler: it stores the received signal number
        in :attr:`termination_signal` and returns immediately.

        Parameters
        ----------
        signum : int, optional
            Signal number (e.g., ``signal.SIGTERM`` or ``signal.SIGINT``).
        frame : frame, optional
            Current stack frame at the time the signal was received (unused).

        Returns
        -------
        None

        Notes
        -----
        Avoid heavy I/O or long-running cleanup here: signal handlers should be
        fast and deterministic. Cleanup should happen in the worker control
        logic after observing :attr:`termination_signal`.
        """
        # acknowledge signal
        #print(f'\n"{self.task}" worker received termination signal '
        #      f'{signum} {now()}')
        self.termination_signal = signum

    def _terminate_operations(self):
        """
        Perform post-termination cleanup steps.

        This method executes cleanup that is safe outside the signal handler:

        - closes the current log file (by assigning :attr:`log_file` to ``None``;
          actual behavior is defined by the concrete worker),
        - resets :attr:`termination_signal`.

        Returns
        -------
        None
        """
        self.log_file = None
        self.termination_signal = None

    def _bind_resources(self):
        """
        Bind CPU/GPU resources for this worker.

        This is a thin wrapper around :func:`~aimmd.resources.bind_resources`,
        passing ``localid`` and the configured CPU/GPU allocation policies.

        Returns
        -------
        object
            Whatever :func:`~aimmd.resources.bind_resources` returns (typically
            a description of selected/bound resources).
        """
        return bind_resources(self.localid,
                              self.cpus_per_task,
                              self.gpus_per_task)

    def _reset_stop_condition(self):
        """
        Reset stop-condition thresholds to "no limit".

        Sets :attr:`nsteps`, :attr:`nframes`, :attr:`walltime`, and :attr:`t0`
        to ``inf``, disabling stop conditions until updated.

        Returns
        -------
        None
        """
        self.nsteps = inf
        self.nframes = inf
        self.walltime = inf
        self.t0 = inf

    def _update_stop_condition(self, **kwargs):
        """
        Update stop-condition thresholds and start the walltime clock.

        Sets :attr:`t0` to the current time and updates any of the supported
        thresholds provided in ``kwargs``:

        - ``nsteps``
        - ``nframes``
        - ``walltime``

        Values are cast to ``float`` before assignment. Recognized keys are
        removed from ``kwargs`` via ``pop``.

        Parameters
        ----------
        **kwargs
            Stop-condition overrides and potentially other options used by the
            caller. This method consumes the keys listed above, leaving any
            unrelated keys in ``kwargs``.

        Returns
        -------
        None
        """
        self.t0 = time.time()
        for name in ('nsteps', 'nframes', 'walltime'):
            if name in kwargs:
                setattr(self, name, float(kwargs.pop(name)))
