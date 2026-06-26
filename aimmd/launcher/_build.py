"""
aimmd.launcher._build
====================

Build the execution plan for :class:`aimmd.launcher.Launcher`.

This module defines :class:`LauncherBuild`, a mixin implementing
:meth:`~LauncherBuild._build`, which materializes the launcher configuration into
a concrete list of worker invocations.

The output of :meth:`_build` is designed to be consumed by a process executor or
a scheduler wrapper. It returns:

- ``args``: a list of positional argument tuples to construct and run workers,
- ``descriptions``: human-readable labels for each planned process.

The build step also prepares the working directory structure for each run by:

1) creating the main run directory (if missing),
2) exporting initial paths into ``initial{sorted_states}``,
3) exporting per-path cached arrays (e.g., states/descriptors) as `.npy` files,
4) creating task folders for free simulations, shooting chains, sweep shooting,
   and trainers,
5) creating per-worker log files when multiple processes are launched.

Process identifiers
-------------------
The build step maintains a ``process_identifiers`` list to prevent spawning
multiple processes that would write into the same output location. This is a
pragmatic guard against conflicting writers, which would corrupt trajectories,
caches, or logs.

Stop conditions
---------------
Each worker is passed stop-condition budgets as positional arguments:

- ``walltime`` (seconds),
- ``nsteps`` (total shooting chain size),
- ``nframes`` (frames).

If training is enabled for a run (``nrounds > 0``), sampling workers are given
infinite budgets and the trainer controls the overall workflow. If no training
is enabled but sampling workers are requested, the budgets come from the
launcher configuration. If neither training nor sampling is requested for a run,
the run is skipped.

Sweep mode
----------
For reactive-region mode ``'sweep'``, the build step creates ``sweep{t}{k}``
folders (rather than ``chain{t}{k}``) and passes ``sweep=True`` to the shooting
task. In AIMMD, sweep shooting is typically used for brute-force committor
validation (repeated shooting from a deterministic set of frames).

Notes
-----
- This method writes trajectories and `.npy` arrays. It is therefore an
  *imperative* setup step and should be considered mutating with respect to the
  filesystem.
- The code uses ``print`` but does not import it in this module. This is part of
  the original code and is preserved here; ensure that the calling context
  provides ``print`` in scope or that this module is imported where ``print`` is
  defined (e.g., via :mod:`aimmd._config`).
"""

# external
import os
import numpy as np
from abc import ABC
from math import inf
from pathlib import Path as PosixPath
from itertools import islice

# aimmd imports
from ..cache.npy import save_npy
from ..core.utils import remove, unique_path
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble


class LauncherBuild(ABC):

    def _build(self):
        """
        Build worker argument tuples and process descriptions to start all runs
        invoked by `self`.
            
        Returns
        -------
        tuple
            ``(args, descriptions)`` where:

            - ``args`` is a list of tuples. Each tuple contains the positional
              arguments required to construct a :class:`aimmd.worker.Worker`
              and immediately dispatch a task via ``Worker.run``.
            - ``descriptions`` is a list of strings with the same length as
              ``args`` describing each planned process for logging/UI use.

        Raises
        ------
        RuntimeError
            If no processes are requested, or if two planned processes would
            initialize/overwrite the same output location (conflicting process
            identifiers).
        TypeError
            If any run's Params lacks initial paths.

        Side Effects
        ------------
        - Creates directories for each run and for each task type.
        - Clears and rewrites ``initial{sorted_states}`` contents for each run.
        - Writes initial trajectories and their cached attribute arrays.

        Notes
        -----
        ``termination_timeout`` passed to workers is reduced by 1 second to
        ensure the worker has a chance to terminate gracefully before the
        launcher-level timeout budget is exhausted.
        """
        args = []
        descriptions = []
        process_identifiers = []  # keep track of what you will initialize
        # to avoid conflicting processes running together
        termination_timeout = max(self.termination_timeout - 1., 0.0)
        num_processes = sum(self._num_processes)
        if num_processes == 0:
            raise RuntimeError('no processes to run')

        # initialize worker id
        i = 0

        for run_id in range(len(self)):
            params = self._params[run_id]
            initial_paths = params.initial_paths
            if not initial_paths:
                raise TypeError("'initial_paths' missing in aimmd.Params")
            a, r, b = params.states
            sorted_states = params.sorted_states
            params_path = os.path.relpath(params.save())
            directory = self._directories[run_id] or '.'
            n = self._n[run_id]
            n1 = self._n1[run_id]
            n2 = self._n2[run_id]
            reactive_region_mode = self._reactive_region_mode[run_id]
            state1_mode = self._state1_mode[run_id]
            state2_mode = self._state2_mode[run_id]
            nchains_per_worker = self._nchains_per_worker[run_id]
            nsteps = self._nsteps[run_id]
            nframes = self._nframes[run_id]
            nrounds = self._nrounds[run_id]
            walltime = self._walltime
            if nrounds:
                conditions = (inf, inf, inf)
            elif n + n1 + n2:
                conditions = (walltime, nsteps, nframes)
            else:  # not running anything here
                continue

            # main directory
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f'+++ created {directory!r}')

            # process initial paths
            folder = f'{directory}/initial{sorted_states}'
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f'+++ created {folder!r}')
            remove(f'{folder}/*')
            for path in params.initial_paths:
                old = path.fname
                fname = unique_path(f'{folder}/{PosixPath(old).name}')
                path.write(fname)
                print(f'+++ saved {str(fname)!r} (from: {old!r})')
                for attribute, series in islice(
                    path.__dict__.items(), 6, None):
                    name = get_cache_fname(fname, attribute)
                    save_npy(name, series)
                    print(f'+++ saved {name!r}')

            # free simulations
            for t, m in zip([r, a, b], [n, n1, n2]):
                if not m:
                    continue
                if ((t == a and state1_mode == 'shoot') or
                    (t == b and state2_mode == 'shoot')):
                    continue
                if t == r and reactive_region_mode != 'free':
                    continue

                # folders
                folder = f'free{t}'
                dfolder = f'{directory}/{folder}'
                if not os.path.exists(dfolder):
                    os.makedirs(dfolder)
                    print(f'+++ created {dfolder}')
                for k in range(m):
                    process_identifier = f'{dfolder}{k}'
                    if process_identifier in process_identifiers:
                        raise RuntimeError(f"can't initialize process {i} "
                              f"(free {k}) in {dfolder!r} more than once")
                    process_identifiers.append(process_identifier)

                    localid = i % self._ntasks_per_node
                    if num_processes > 1:
                        log_file = f'{folder}/worker{k}.log'
                    else:
                        log_file = 'stdout'
                    args.append(
                        (params_path, directory, localid,
                         self._cpus_per_task, self._gpus_per_task,
                         log_file, *conditions, termination_timeout,
                         'free', t, k, m, nrounds > 0 or n > 0))
                    descriptions.append(f'"{directory}" {folder} (worker{k})')
                    i += 1  # advance the task id

            # shooting simulations
            for t, m in zip([r, a, b], [n, n1, n2]):
                if t == r and reactive_region_mode == 'free':
                    continue
                if ((t == a and state1_mode != 'shoot') or
                    (t == b and state2_mode != 'shoot')):
                    continue
                sweep = (t == r and reactive_region_mode == 'sweep')
                for k in range(m * nchains_per_worker):
                    if sweep:
                        folder = f'sweep{t}{k}'
                    else:
                        folder = f'chain{t}{k}'
                    dfolder = f'{directory}/{folder}'

                    # check that you can initialize the process
                    process_identifier = dfolder
                    if process_identifier in process_identifiers:
                        raise RuntimeError(f"can't initialize process {i} "
                              f"(shoot) in {dfolder!r} more than once")
                    process_identifiers.append(process_identifier)
                    if not os.path.exists(dfolder):
                        os.makedirs(dfolder)
                        print(f'+++ created {dfolder}')
                    
                    # for now, just pick the first chain associated to worker
                    if k % nchains_per_worker:
                        continue

                    localid = i % self._ntasks_per_node
                    if num_processes > 1:
                        log_file = f'{folder}/worker.log'
                    else:
                        log_file = 'stdout'
                    args.append(
                        (params_path, directory, localid,
                        self._cpus_per_task, self._gpus_per_task,
                        log_file, *conditions, termination_timeout,
                        'shoot', t, k, sweep, nchains_per_worker))
                    descriptions.append(f'"{directory}" {folder}')
                    i += 1  # advance the task id

            # trainer
            if nrounds:
                localid = i % self._ntasks_per_node
                process_identifier = f'{directory}{sorted_states}'
                if process_identifier in process_identifiers:
                   raise RuntimeError(f"can't initialize process {i} (train "
                          f"{sorted_states}) in {directory!r} more than once")
                process_identifiers.append(process_identifier)
                if num_processes > 1:
                    keep_running = True
                    log_file = f'train{sorted_states}.log'
                else:
                    keep_running = False
                    log_file = 'stdout'
                args.append(
                    (params_path, directory, localid,
                     self._cpus_per_task, self._gpus_per_task,
                     log_file, walltime, nsteps, nframes,
                     termination_timeout, 'train', nrounds, keep_running))
                descriptions.append(f'"{directory}" {sorted_states} trainer')
                i += 1  # advance the task id

        # combine
        return args, descriptions
