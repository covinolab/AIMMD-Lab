"""
aimmd.launcher._run
==================

Local multi-process execution for :class:`aimmd.launcher.Launcher`.

This module defines:

- :func:`run_task`, a thin top-level wrapper that constructs a
  :class:`~aimmd.worker.Worker` and dispatches a single task, and
- :class:`LauncherRun`, a mixin implementing :meth:`LauncherRun.run`, which
  launches multiple AIMMD workers locally using a process executor.

Why `run_task` lives here
-------------------------
`run_task` is defined at module top-level because multiprocessing (especially
with the *spawn* start method) requires the target callable to be picklable.
Defining the callable inside a method would make it non-picklable in many
contexts, hence the explicit module-level definition.

Execution model
---------------
:meth:`LauncherRun.run` performs the following steps:

1) Update launcher configuration (number of workers per role, stop conditions,
   and resource policies) via :meth:`LauncherHelpers._update`.

2) Build the concrete process plan and create run/task folders via
   :meth:`LauncherBuild._build`.

3) Register all worker processes with :class:`~aimmd.execute.processes.ProcessExecutor`.

4) Start all processes, then monitor them until:
   - the launcher walltime budget is exceeded, or
   - any worker exits (successfully or with error).

The monitoring policy is intentionally "fail-fast": as soon as a worker exits,
the launcher begins shutdown. If the exiting worker has a non-zero exit code,
the launcher raises a ``RuntimeError`` to signal failure of the overall run.

Finally, the launcher clears all remaining processes, using the configured
``termination_timeout`` as a grace period.

Notes
-----
- This module imports ``multiprocessing`` but does not use it directly in the
  current code; it is preserved as part of the original file.
- The input parameter ``walltime`` in :meth:`LauncherRun.run` is both passed to
  workers (as a stop condition) and used locally as the maximum monitoring time
  for the launcher loop.
"""

# external
import time
import multiprocessing
from abc import ABC
from math import inf

# aimmd imports
from ..worker import Worker
from ..core.utils import now


def run_task(params_file, directory,
             localid, cpus_per_task, gpus_per_task,
             log_file, walltime, nsteps, nframes,
             termination_timeout, task, *args, **kwargs):
    """
    Construct a Worker and run a single task.

    This function is used as the target callable for multiprocessing workers and
    therefore must be defined at module scope.

    Parameters
    ----------
    params_file : str
        Path to the Params file (or other Worker-compatible params argument).
    directory : str
        Working directory for this run.
    localid : int
        Local worker identifier (used for resource binding).
    cpus_per_task : str or int
        CPU binding policy or explicit CPU count passed to the Worker.
    gpus_per_task : str or int
        GPU binding policy or explicit GPU count passed to the Worker.
    log_file : str or file-like
        Log target passed to the Worker (often a per-worker log file).
    walltime : float
        Walltime stop condition passed to the Worker.
    nsteps : float
        Step budget stop condition passed to the Worker.
    nframes : float
        Frame budget stop condition passed to the Worker.
    termination_timeout : float
        Grace period for termination passed to the Worker.
    task : str
        Worker task name (e.g., ``'shoot'``, ``'free'``, ``'train'``).
    *args, **kwargs
        Additional task arguments passed through to ``Worker.run``.

    Returns
    -------
    None
    """
    Worker(params_file, directory,
           localid, cpus_per_task, gpus_per_task,
           log_file, walltime, nsteps, nframes,
           termination_timeout).run(task, *args, **kwargs)


class LauncherRun(ABC):
    def run(self, n=1, n1=0, n2=0,
            reactive_region_mode='chain',
            state1_mode='free', state2_mode='free',
            nsteps=inf, nframes=inf, nrounds=None, walltime=inf,
            cpus_per_task='share', gpus_per_task='share'):
        """
        Launch AIMMD runs locally by spawning multiple worker processes.

        Parameters
        ----------
        n : int or iterable of int, optional
            Number of replicas dedicated to reactive-region sampling for each
            run. Default is ``1``.
        n1 : int or iterable of int, optional
            Number of replicas dedicated to sampling in/around the initial end
            state (``params.states[0]``). Default is ``0``.
        n2 : int or iterable of int, optional
            Number of replicas dedicated to sampling in/around the final end
            state (``params.states[2]``). Default is ``0``.
        reactive_region_mode : {'chain', 'free', 'sweep'} or iterable, optional
            Mode for reactive-region replicas:
            
            - ``'chain'``: committor-guided shooting chain (TPS/RFPS-like),
            - ``'sweep'``: sweep shooting for brute-force committor validation,
            - ``'free'``: free simulations in place of shooting.
            
            Default is ``'chain'``.
        state1_mode : {'free', 'shoot'} or iterable, optional
            Mode for state-1 replicas. Default is ``'free'``.
        state2_mode : {'free', 'shoot'} or iterable, optional
            Mode for state-2 replicas. Default is ``'free'``.
        nsteps : float or iterable of float, optional
            Maximum number of simulated independent trajectories (worker stop
            condition). Default is ``inf``. Attention! If "train" runs, then
            nsteps refers to the total number of steps across the shooting
            simulations only.
            Otherwise, it refers to the number of steps of each single worker
            in the launcher run. The first worker reaching nsteps stops all
            the others.
        nframes : float or iterable of float, optional
            Maximum number of simulated frames (worker stop condition). Default
            is ``inf``. Attention! If "train" runs, then
            nframes refers to the total number of frames across all workers.
            Otherwise, it refers to the number of nframes of each single worker
            in the launcher run. The first worker reaching nsteps stops all
            the others.
        nrounds : float or iterable of float, optional
            If `None` and new simulations are requested, add a new process that
            trains the model and computes selection bins and densities 
            indefinitely. If `None` and no new simulations are requested, just
            does one round before exiting. If != 0, the process does training
            rounds up until reaching `nrounds`, from that point on it just
            updates selection bins and densities.
            Forced to zero when `reactive_region_mode = 'sweep'`.
        walltime : float, optional
            Maximum walltime for the launcher monitoring loop (seconds). Also
            passed to workers as a stop condition when training is not enabled.
            Default is ``inf``.
        cpus_per_task : {'share', 'all', 'skip'} or int, optional
            CPU allocation policy per worker. Default is ``'share'``.
        gpus_per_task : {'share', 'all', 'skip'} or int, optional
            GPU allocation policy per worker. Default is ``'share'``.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If any worker exits with a non-zero exit code.
        Exception
            Re-raises unexpected exceptions that occur during process startup or
            monitoring.

        Notes
        -----
        The monitoring policy is "stop all as soon as any other stops". This is
        consistent with AIMMD workflows where workers share on-disk state and
        should not keep running after another worker has stopped or failed.
        """
        # update run settings
        self._update(n, n1, n2,
                     reactive_region_mode, state1_mode, state2_mode,
                     nsteps, nframes, nrounds, walltime,
                     cpus_per_task, gpus_per_task)

        # initialize processes, create folders
        for args, description in zip(*self._build()):
            self._processes.add(run_task, *args, name=description)

        # start processes
        try:
            self._processes.run(timeout=self.termination_timeout)

            # wait for completion within walltime
            # stop all as soon as any other stops
            t0 = time.time()
            must_stop = not self._processes.alive.all()
            while time.time() - t0 < walltime:
                if must_stop:
                    break
                for process in self._processes:
                    exitcode = process.exitcode
                    if exitcode is None:
                        continue
                    must_stop = True
                    if exitcode:
                        raise RuntimeError('launcher run failed')
                    break
                time.sleep(.01)  # avoid freezing

        # catch exceptions
        except Exception as exception:
            raise exception

        # clear all processes
        finally:
            self._processes.clear(timeout=self.termination_timeout)
