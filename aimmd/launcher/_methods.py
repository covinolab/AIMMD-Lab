"""
aimmd.launcher._methods
======================

High-level launcher methods.

This module defines :class:`LauncherMethods`, a mixin implementing user-facing
operations for :class:`aimmd.launcher.Launcher`. In particular, it provides:

- dynamic modification of the run list (``add`` / ``pop``),
- generation of a SLURM job script that spawns AIMMD workers via ``srun``
  (``create_job``).

Job-script behavior
-------------------
The SLURM script produced by :meth:`create_job` is intentionally simple: it
launches all planned worker processes in the background, then terminates the
entire SLURM allocation as soon as *any* worker exits (successfully or not).

This behavior is implemented using:

- ``wait -n`` to wait for the first background process to finish,
- ``scancel ${SLURM_JOB_ID}`` to request cancellation of the allocation,
- ``wait`` to reap remaining background processes locally.

This "kill-on-first-exit" behavior is useful for AIMMD workflows where workers
are tightly coupled by shared on-disk state: if one worker stops (because of a
stop condition, error, or external termination), it is typically desirable to
stop the whole job and restart cleanly rather than let other workers continue
blindly.

Thus, the launcher calls :meth:`LauncherHelpers._update` with ``walltime=inf``
for worker-side budgets, as the batch job walltime is handled by SLURM.

Resource policy and overrides
-----------------------------
The SLURM header stored in ``params.slurm_header`` may already contain resource
directives. :meth:`create_job` inspects it and applies the following precedence:

- ``--ntasks-per-node``: if present in the header, it is used; otherwise the
  provided ``ntasks_per_node`` is injected.
- ``--cpus-per-task``: if present, it is used; otherwise ``cpus_per_task`` is
  injected.
- GPU resources:
  - if ``--gpus-per-task`` is present, it is used;
  - if ``--gres=gpu:<N>`` is present, it is converted to per-task GPUs by
    dividing by ``ntasks_per_node``;
  - otherwise, the provided ``gpus_per_task`` is injected (as a ``--gres`` line
    allocating ``gpus_per_task * ntasks_per_node`` GPUs per node).

Resource flags precedence
-------------------------
:meth:`create_job` derives the effective per-task CPU/GPU allocation and tasks
per node by combining:

1) input arguments (``cpus_per_task``, ``gpus_per_task``, ``ntasks_per_node``),
2) overrides found in ``self.params[0].slurm_header``.

If the relevant SBATCH directives are found in the header, they override the
method inputs. Otherwise, directives are appended to the header.

Binding control
---------------
If ``skip_binding=True`` (default), worker-side explicit binding is disabled by
passing the string ``'skip'`` for ``cpus_per_task`` / ``gpus_per_task`` to the
worker. This allows the scheduler/runtime to manage placement without the worker
pinning resources itself.

Notes
-----
- ``create_job`` uses :data:`aimmd._config.PYTHON` and :data:`aimmd._config.WORKER`
  as default executable paths in the generated script.
- This module intentionally does not execute jobs; it only writes scripts.
"""

# external
import math
from abc import ABC
from math import inf
from numpy import bool_

# aimmd imports
from .._config import PYTHON, WORKER


class LauncherMethods(ABC):

    def add(self, params, directories):
        """
        Append one or more runs to the launcher.

        Parameters
        ----------
        params : str or aimmd.params.Params or iterable
            Params specification(s) for the run(s). See
            :meth:`LauncherHelpers._init` for accepted forms.
        directories : str or iterable of str
            Working directory (or directories) for the new run(s).

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If, after appending, directories are not unique.

        Notes
        -----
        The launcher is updated in place and :meth:`_update` is called to refresh
        derived fields (process counts, resource allocation, etc.).
        """
        from . import Launcher
        instance = Launcher(params, directories)
        self._params.extend(instance.params)
        self._directories.extend(instance.directories)
        if len(set(self._directories)) < len(self._directories):
            raise TypeError('All AIMMD run directories must be different')
        self._update()

    def pop(self, i):
        """
        Remove the i-th run from the launcher.

        Parameters
        ----------
        i : int
            Index of the run to remove.

        Returns
        -------
        None

        Notes
        -----
        This method mutates internal lists but does not call :meth:`_update`.
        Callers that rely on derived fields should call :meth:`_update`
        afterwards.
        """
        self._params.pop(i)
        self._directories.pop(i)

    def create_job(self, filename, n=1, n1=0, n2=0,
                   reactive_region_mode='chain',
                   state1_mode='free', state2_mode='free',
                   nchains_per_worker=1, nsteps=inf, nframes=inf, nrounds=inf,
                   walltime=24*3600, cpus_per_task=1, gpus_per_task=0,
                   ntasks_per_node=1, skip_binding=True):
        """
        Create a SLURM job script that launches AIMMD workers via ``srun``.

        The script produced by this method:

        - writes the SLURM header derived from ``params.slurm_header`` and from
          the provided defaults (if missing),
        - sets shell job control (``set -m``),
        - launches each planned worker as a background process using
          ``srun --exclusive ... &``,
        - stops the full job allocation as soon as the first worker exits.

        Parameters
        ----------
        filename : str
            Destination path of the job script.
        n : int or iterable of int, optional
            Number of replicas dedicated to reactive-region sampling. The exact
            meaning depends on ``reactive_region_mode``:

            - ``'chain'``: committor-guided shooting chain (TPS/RFPS-like),
            - ``'sweep'``: sweep shooting for committor validation,
            - ``'free'``: free simulations instead of shooting.

            Default is ``1``.
        n1 : int or iterable of int, optional
            Number of replicas dedicated to sampling around the initial end
            state (``params.states[0]``). Default is ``0``.
        n2 : int or iterable of int, optional
            Number of replicas dedicated to sampling around the final end state
            (``params.states[2]``). Default is ``0``.
        reactive_region_mode : {'chain', 'free', 'sweep'} or iterable, optional
            Mode for reactive-region replicas. Default is ``'chain'``.
        state1_mode : {'free', 'shoot'} or iterable, optional
            Mode for state-1 replicas. Default is ``'free'``.
        state2_mode : {'free', 'shoot'} or iterable, optional
            Mode for state-2 replicas. Default is ``'free'``.
        nchains_per_worker : int, optional, default = 1
            If > 1, the worker will manage more than one chain. A higher value
            of `nchains_per_worker` tends to regularize the training set and thus
            improve performance. If running only one shooting worker,
            `nchains_per_worker=10` is recommended.
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
            Job walltime in seconds, written as ``#SBATCH --time=HH:MM:SS``.
            Default is ``24*3600`` (24 hours).
        cpus_per_task : int, optional
            CPUs per worker task. May be overridden by an SBATCH directive
            present in ``params.slurm_header``. Default is ``1``.
        gpus_per_task : int, optional
            GPUs per worker task. May be overridden by an SBATCH directive
            present in ``params.slurm_header``. Default is ``0``.
        ntasks_per_node : int, optional
            Tasks per node. May be overridden by an SBATCH directive present in
            ``params.slurm_header``. Default is ``1``.
        skip_binding : bool, optional
            If ``True``, pass ``'skip'`` to the worker for CPU/GPU binding so
            that workers do not explicitly bind resources. Default is ``True``.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If ``filename`` cannot be written.
        TypeError
            If the launcher cannot be configured with the requested modes.

        Notes
        -----
        - The effective number of nodes is computed as:

          ``ceil(total_processes / ntasks_per_node)``

          and written as ``#SBATCH --nodes=...``.
        - The script uses ``srun --exclusive --nodes=1 --ntasks=1`` for each
          worker to ensure processes do not share a task allocation.
        """
        # retrieve run information: slurm header
        slurm_header = self.params[0].slurm_header + ''

        # retrieve run information: ntasks per node
        default_ntasks_per_node = int(ntasks_per_node)
        ntasks_per_node = None
        for fields in slurm_header.split():
            if 'ntasks-per-node' in fields:
                ntasks_per_node = int(fields.split('=')[-1])
        if not ntasks_per_node:
            ntasks_per_node = default_ntasks_per_node
            slurm_header += \
                f'\n#SBATCH --ntasks-per-node={ntasks_per_node}'

        # retrieve run information: cpus per task
        default_cpus_per_task = int(cpus_per_task)
        cpus_per_task = None
        for fields in slurm_header.split():
            if 'cpus-per-task' in fields:
                cpus_per_task = int(fields.split('=')[-1])
        if not cpus_per_task:
            cpus_per_task = default_cpus_per_task
            slurm_header += \
                f'\n#SBATCH --cpus-per-task={cpus_per_task}'

        # retrieve run information: gpus per task
        default_gpus_per_task = int(gpus_per_task)
        gpus_per_task = None
        for fields in slurm_header.split():
            if 'gres=gpu:' in fields:
                gpus_per_task = int(fields.split(':')[-1]) // ntasks_per_node
            if 'gpus-per-task' in fields:
                gpus_per_task = int(fields.split('=')[-1])
        if not gpus_per_task:
            gpus_per_task = default_gpus_per_task
            if gpus_per_task:
                slurm_header += \
                    f'\n#SBATCH --gres=gpu:{gpus_per_task * ntasks_per_node}'

        # update info
        self._update(n, n1, n2,
                     reactive_region_mode,
                     state1_mode, state2_mode,
                     nchains_per_worker, nsteps, nframes, nrounds,
                     inf,  # infinite walltime
                     cpus_per_task if not skip_binding else 'skip',
                     gpus_per_task if not skip_binding else 'skip',
                     ntasks_per_node)

        # number of nodes
        nodes = math.ceil(sum(self._num_processes) / self._ntasks_per_node)
        slurm_header += f'\n#SBATCH --nodes={nodes}'

        # time information
        walltime = int(walltime)
        hours = walltime // 3600
        minutes = (walltime - hours * 3600) // 60
        seconds = walltime - hours * 3600 - minutes * 60
        slurm_header += \
            f'\n#SBATCH --time={hours:02g}:{minutes:02g}:{seconds:02g}'

        # write job script
        with open(filename, 'w') as file:

            # slurm header
            file.write('#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params[0].name}\n')
            file.write(f'{slurm_header}\n\n')

            # default names
            file.write('# default names\n')
            file.write(f'PYTHON="{PYTHON}"\n')
            file.write(f'WORKER="{WORKER}"\n\n')

            # enable job control
            file.write('# enable job control\n')
            file.write('set -m\n')

            # launch commands
            file.write('\n# workers')
            for i, (args, description) in enumerate(zip(*self._build())):
                file.write(f'\n# {description}\n')
                args = ' '.join([f'"{arg}"'
                                 if not isinstance(arg, (bool, bool_)) else
                                 '"True"' if arg else '""' for arg in args])
                file.write(f'srun --exclusive --nodes=1 --ntasks=1 '
                           f'--cpus-per-task={cpus_per_task} '
                           f'--gpus-per-task={gpus_per_task} \\\n')
                file.write(f'  "${{PYTHON}}" "${{WORKER}}" {args} &\n')

            # wait until any process exits
            file.write('\n# wait until any process exits\n')
            file.write('wait -n\n')
            file.write('scancel ${SLURM_JOB_ID}\n')
            file.write('wait\n')
