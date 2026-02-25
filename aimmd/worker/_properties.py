"""
aimmd.worker._properties
=======================

Derived properties for AIMMD worker classes.

This module defines :class:`WorkerProperties`, a mixin that provides:

- a :attr:`log_file` property controlling where the worker prints its output
  (by redirecting ``sys.stdout`` / ``sys.stderr``),
- a :attr:`must_stop` property implementing the worker's stop-condition checks,
- an :attr:`initial_paths` convenience property to locate (or lazily create)
  initial paths in the working directory.

The mixin assumes it is used together with other worker mixins that define
runtime state such as:

- :attr:`directory`, :attr:`_directory`
- :attr:`params` (with :attr:`~aimmd.params.Params.sorted_states`)
- :attr:`termination_signal`
- :attr:`walltime`, :attr:`nsteps`, :attr:`nframes`
- :attr:`original_stdout`, :attr:`original_stderr`
- :attr:`_t0`, :attr:`_total_frames`, :attr:`_total_steps`

Notes
-----
- :attr:`log_file` mutates global interpreter state by reassigning
  ``sys.stdout`` and ``sys.stderr``. This is appropriate for isolated worker
  processes, but should be used with care in interactive contexts.
- The stop-condition checks in :attr:`must_stop` are *read-only*: they only set
  :attr:`termination_signal` as a side effect when a threshold is exceeded.
"""

# external
import sys
import time
from abc import ABC
from math import inf

# aimmd imports
from ..pathensemble import PathEnsemble


class WorkerProperties(ABC):
    """
    Property mixin for worker instances.

    This mixin provides computed properties and small convenience accessors.
    It does not define the full worker lifecycle; it expects a concrete worker
    class (or other mixins) to manage counters, timestamps, and task execution.
    """

    @property
    def log_file(self):
        """
        Current logging target.

        The worker uses this property to route all print output by redirecting
        ``sys.stdout`` and ``sys.stderr``:

        - If a file-like object is provided, it becomes the new stdout/stderr.
        - If a string is provided, it is treated as a filename relative to
          :attr:`directory` and opened in append mode (line-buffered).
        - If ``'stdout'`` or ``None`` is provided, stdout/stderr are restored
          to their original streams captured at initialization.

        Returns
        -------
        object
            The current log target stored in :attr:`_log_file`. This is either
            a file-like object or the original stdout stream.
        """
        return self._log_file

    @log_file.setter
    def log_file(self, log_file):
        """
        Set the logging target and update stdout/stderr accordingly.

        Parameters
        ----------
        log_file : {'stdout'} or str or file-like or None
            Logging destination:

            - ``None`` or ``'stdout'`` restores the original stdout/stderr.
            - str opens ``f'{directory}/{log_file}'`` in append mode.
            - file-like redirects stdout/stderr to that object.

        Returns
        -------
        None

        Notes
        -----
        - If stdout has previously been redirected to a file by this property,
          the file handle is closed when changing to a different target.
        - The file is opened with ``buffering=1`` (line buffering) to make logs
          visible promptly.
        """
        if log_file is None:
            log_file = self.original_stdout
        if log_file == 'stdout':
            log_file = self.original_stdout
        if log_file == self._log_file:
            return

        # Close previously redirected stdout (if it differs from the original).
        if self.original_stdout != sys.stdout:
            sys.stdout.close()

        # Restore original streams before potentially reassigning again.
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Convert a filename into an open file handle.
        if isinstance(log_file, str):
            log_file = open(f'{self.directory}/{log_file}', 'a+', buffering=1)

        # Redirect output if a target is provided.
        if log_file:
            sys.stdout = log_file
            sys.stderr = sys.stdout

        self._log_file = log_file

    @property
    def must_stop(self):
        """
        Whether the current task should stop.

        The worker stops if:

        - a termination signal was received (SIGTERM or SIGINT),
        - walltime exceeded: ``time.time() - _t0 >= walltime``,
        - frame limit exceeded: ``_total_frames >= nframes``,
        - step limit exceeded: ``_total_steps >= nsteps``.

        If a stop-condition threshold is exceeded, this property sets
        :attr:`termination_signal` to ``2`` (used as a SIGINT-like marker)
        and returns ``True``.

        Returns
        -------
        bool
            ``True`` if the worker should stop, otherwise ``False``.

        Side Effects
        ------------
        termination_signal
            Set to ``2`` when a stop-condition threshold triggers and no
            termination signal was previously recorded.

        Notes
        -----
        This property assumes that the concrete worker maintains the counters
        :attr:`_total_frames`, :attr:`_total_steps` and the task start time
        :attr:`_t0` (seconds since epoch).
        """
        if self.termination_signal:
            return True

        if (time.time() - self._t0 >= self.walltime or
            self._total_frames >= self.nframes or
            self._total_steps >= self.nsteps):
            self.termination_signal = 2  # sigint
            return True

        return False

    @property
    def initial_paths(self):
        """
        Initial paths available to the worker.

        The worker expects initial paths to live under a folder named according
        to the sorted end states, e.g. ``initial('A','B')``. This property first
        attempts to load existing paths from ``{directory}/initial{states}/*``.
        If none are found, it lazily initializes them via :class:`~aimmd.launcher.Launcher`
        in :attr:`_directory` and tries again.

        Returns
        -------
        aimmd.pathensemble.PathEnsemble
            Ensemble containing all initial paths located (or created) for the
            current end-state set.

        Notes
        -----
        - The state tuple is obtained from :attr:`params.sorted_states` and is
          embedded directly into the folder name.
        - The lazy creation branch imports :class:`~aimmd.launcher.Launcher`
          locally to reduce import-time coupling and avoid circular imports.
        """
        # retrieve initial path in directory
        states = self.params.sorted_states
        initial_paths = PathEnsemble(f'{self.directory}/initial{states}/*')
        if not initial_paths:
            from ..launcher import Launcher
            # need to initialize first
            launcher = Launcher(self.params, self._directory)
            launcher._update(n=0)
            launcher._build()
            initial_paths = PathEnsemble(f'{self._directory}/initial{states}/*')
        return initial_paths
