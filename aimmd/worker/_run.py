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
from .._config import MDA_CACHE, NPY_CACHE, print, require_gromacs
from ..core.utils import now, accepts_system_id


def _system_id_binder(function, system_id):
    """Wrap a user data function so it is always called for one fixed system.

    The returned function takes only the data argument (so callers/`Path.compute`
    won't try to pass ``system_id`` again) and forwards the bound ``system_id``
    to the wrapped function.
    """
    def bound(data):
        return function(data, system_id=system_id)
    return bound


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
            - ``'kinetics_convergence'``: dispatches to :meth:`_kinetics_convergence`

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

        # Shooting / free MD with the GROMACS engine invokes `gmx`; fail fast
        # with the same error import used to raise. Training and analysis, and
        # toy-engine sampling, do not require GROMACS. (Placed before the try so
        # the finally block, which references `cwd`, is not entered.)
        if task in ('shoot', 'free') and getattr(self.params, 'engine', None) == 'gromacs':
            require_gromacs()

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

            # multi-system: bind this worker's system_id into the user data
            # functions so every downstream compute call featurizes / classifies
            # for the correct system (no per-call-site threading needed).
            self._bind_system_id()

            # clear caches
            MDA_CACHE.clear()
            NPY_CACHE.clear()

            # update stop condition and remove from kwargs
            self._update_stop_condition(**kwargs)

            # reset time, total steps, total frames progress bars
            self._t0 = time.time()
            self.total_steps = None
            self.total_frames = None
          
            # execute task
            if task == 'shoot':
                return self._shoot(*args, **kwargs)
            if task == 'free':
                return self._free(*args, **kwargs)
            if task == 'train':
                return self._train(*args, **kwargs)
            if task == 'kinetics_convergence':
                return self._kinetics_convergence(*args, **kwargs)
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

    def _bind_system_id(self):
        """Bind this worker's system to the params data functions (multi-system).

        In a multi-system run each worker operates in a per-system subfolder
        ``<run>/<system_id>/``. The ``system_id`` is the subfolder name. We wrap
        the params' ``states_function`` / ``descriptors_function`` /
        ``values_function`` (those that accept a ``system_id`` keyword) so they
        are always evaluated for THIS worker's system — every shoot/free/train
        compute call then uses the right per-system state cutoffs and graph
        featurization without threading ``system_id`` through each call site.

        No-op for single-system runs and for the shared trainer (whose directory
        is the run root, not a system subfolder); the shared trainer dispatches
        per system explicitly.
        """
        params = self.params
        if not getattr(params, 'multi_system', False):
            return
        system_id = os.path.basename(os.path.normpath(self._directory))
        system_ids = list(params.system_ids or [])
        if system_id not in system_ids:
            return
        self._system_id = system_id
        for name in ('states_function', 'descriptors_function',
                     'values_function', 'descriptor_transform',
                     'bias_function'):
            function = getattr(params, name, None)
            if function is not None and accepts_system_id(function):
                params.__dict__[name] = _system_id_binder(function, system_id)
        # Resolve per-system engine configuration for THIS worker's system, so
        # the GROMACS/toy engine (grompp, masses, mdp) uses the right system's
        # files. Per-worker, single process -> safe to specialize in place.
        sidx = system_ids.index(system_id)
        params.__dict__['_universe'] = params.universe_of(system_id)
        for name in ('topology', 'gmx_mdp', 'gmx_grompp', 'gmx_mdrun'):
            value = getattr(params, name, None)
            if isinstance(value, (list, tuple)):
                params.__dict__[name] = value[sidx]
