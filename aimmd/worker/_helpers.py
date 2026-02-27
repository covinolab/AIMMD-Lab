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
from tqdm.auto import tqdm

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

        # will then potentially be different at execution time
        # directory is user provided
        # _directory is the directory with respect to the params file location
        # _folder is where simulations actually run (if any)
        self.directory = self._folder = self._directory = directory
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

        # for reporting (will create progress bars)
        self._total_steps = None
        self._total_frames = None

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

        - closes progress bars (if present, by assigning :attr:`total_nsteps`
        and :attr:`total_nframes` to ``None``);
        - closes the current log file (by assigning :attr:`log_file` to ``None``;
          actual behavior is defined by the concrete worker),
        - resets :attr:`termination_signal`.

        Returns
        -------
        None
        """
        self.total_nsteps = None
        self.total_nframes = None
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
    
    def _set_progress_bar(self, pbar, n, unit="steps", offset=0):
           """Create, update, or close a progress reporter (terminal + Jupyter-safe).
    
        This helper centralizes progress reporting for long-running loops that track
        a monotonically increasing counter `n` (e.g., integrated steps, emitted frames,
        completed trajectories). The function supports three operations:
    
        1) **Create** a new progress bar if `pbar is None` and `n` is not None.
        2) **Update** an existing progress bar to reflect the new absolute progress `n`.
        3) **Close** the progress bar if `n is None`.
    
        Backend selection
        -----------------
        Progress rendering depends on whether the output stream behaves like a real
        terminal (TTY):
    
        - **TTY output** (typical CLI runs): use the standard terminal tqdm backend
          which updates bars in place using carriage return and cursor control.
        - **Non-TTY output** (Jupyter, captured output, redirected logs): terminal
          cursor control is not supported and would degrade into "one bar line per
          update". In that case this function uses the *notebook* tqdm backend
          (`tqdm.notebook.tqdm`), which renders a widget-like bar that updates in place.
    
        Important: for the notebook backend, `file=` must not be overridden; display
        is handled by the notebook frontend.
    
        Parameters
        ----------
        pbar : tqdm instance or None
            Existing progress bar object (returned by a previous call), or None to
            create a new one.
    
        n : int or None
            Absolute progress value. If `n` is an integer, the bar is created/updated
            to match `n`. If `n` is None, the progress bar is closed.
    
            This function treats `n` as an absolute counter, not a delta. The delta
            is computed internally as `dn = n - pbar.n`.
    
        unit : str, optional
            Unit label displayed by tqdm (e.g., "step", "frame", "traj"). The string
            is stripped to avoid formatting artifacts such as "s/ trajs" caused by
            leading spaces.
    
        offset : int, optional
            Additional position offset used to place multiple bars on separate lines.
            The actual tqdm `position` is computed as `self.localid * 2 + offset`,
            which reserves multiple lines per worker (e.g., one for trajectories and
            one for frames). Considered only when printing on terminal.
    
        Returns
        -------
        pbar : tqdm instance or None
            A progress bar object (terminal or notebook backend) when `n` is not None.
            Returns None when closing (`n is None`).
    
        Notes
        -----
        - This function intentionally keeps the progress bar state outside the object
          (via the `pbar` argument) to allow the caller to manage multiple bars.
        - The total is set to `int(self.nsteps)` when finite; otherwise tqdm is used
          in "unknown total" mode (`total=None`).
        - In non-TTY environments, the notebook backend requires Jupyter widget support
          (commonly provided by `ipywidgets`). If widget support is missing, the
          notebook backend may fall back to a text representation.
    
        """
        
        # 1. Detect TTY status (False in Pytest/Notebooks)
        tty = getattr(self.original_stdout, "isatty", lambda: False)()
    
        # 2. CLOSE: If n is None, shut down the bar
        if n is None:
            if pbar is not None:
                pbar.close()
            return None
    
        # 3. CREATE: If no pbar exists, initialize one
        if pbar is None:
            total = int(self.nsteps) if self.nsteps < inf else None
            
            # Base settings safe for all backends
            kwargs = {
                "desc": str(self._folder),
                "unit": unit,
                "initial": int(n),
                "total": total,
                "leave": True,
            }
    
            if tty:
                # Add terminal-specific positioning and stream routing
                kwargs.update({
                    "position": (self.localid * 2 + offset),
                    "dynamic_ncols": True,
                    "file": self.original_stdout,
                })
            else:
                # In non-TTY (Pytest/Jupyter), we let tqdm.auto manage the 'file' 
                # and 'display' logic to avoid the 'disp' AttributeError.
                pass
    
            return tqdm(**kwargs)
    
        # 4. UPDATE: Calculate delta and advance
        # tqdm.update() requires a relative increment (dn)
        dn = int(n) - int(pbar.n)
        if dn:
            pbar.update(dn)
    
        return pbar
