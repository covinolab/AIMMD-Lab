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

Environment, run-folder guard and preflight
-------------------------------------------
Between the SLURM header and the workers, the script can hold three more
blocks, in this order:

- ``preamble``: lines written as they are (``module load ...``, ``source
  .../activate``, ``export HF_HUB_OFFLINE=1``). Workers started by ``srun``
  inherit the environment these lines set.
- the run-folder guard (``guard_run_folders=True``, the default): a job
  claims each of its run folders by writing its SLURM job id to
  ``<run>/aimmd_job.txt``, and does not start while another job that
  claimed one of them is still running (``squeue``). Two jobs in one run
  folder would write the same chains, caches and network files.
- the preflight (``preflight=True``): ``python -m aimmd.cgaimmd.preflight``
  on the params file of every C-GAIMMD run before any worker starts; the job
  stops there when a check fails (:mod:`aimmd.cgaimmd.preflight`).

Notes
-----
- ``create_job`` uses :data:`aimmd._config.PYTHON` and :data:`aimmd._config.WORKER`
  as default executable paths in the generated script.
- This module intentionally does not execute jobs; it only writes scripts.
"""

# external
import math
import shlex
from abc import ABC
from math import inf
from numpy import bool_

# aimmd imports
from .._config import PYTHON, WORKER, require_gromacs


RUN_FOLDER_RECORD = 'aimmd_job.txt'
"""The file in a run folder that names the SLURM job using it (see ``guard_run_folders``)."""

PREFLIGHT_MODULE = 'aimmd.cgaimmd.preflight'
"""The module a job script runs as its preflight (``python -m``)."""

# A job claims each run folder by writing its job id into it; another job
# that claimed the folder and is still running stops this one. The re-read
# after two seconds settles two jobs that start at the same moment: the last
# writer keeps the folder, the other stops.
_GUARD_FUNCTION = r"""claim_run_folder() {
    local folder="$1" record="$1/%(record)s" other=""
    mkdir -p "${folder}"
    if [[ -f "${record}" ]]; then
        other=$(head -n 1 "${record}" | cut -d ' ' -f 1)
    fi
    if [[ -n "${other}" && "${other}" != "${SLURM_JOB_ID}" ]] && \
            squeue -h -j "${other}" -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED -o %%i \
            2>/dev/null | grep -qx "${other}"; then
        echo "job ${other} is still running in the run folder ${folder}:" \
             "this job (${SLURM_JOB_ID}) does not start" >&2
        exit 1
    fi
    echo "${SLURM_JOB_ID} $(hostname) $(date -Iseconds)" > "${record}"
}
confirm_run_folder() {
    local folder="$1" owner=""
    owner=$(head -n 1 "${folder}/%(record)s" | cut -d ' ' -f 1)
    if [[ "${owner}" != "${SLURM_JOB_ID}" ]]; then
        echo "job ${owner} claimed the run folder ${folder} at the same time:" \
             "this job (${SLURM_JOB_ID}) does not start" >&2
        exit 1
    fi
}
if ! command -v squeue > /dev/null; then
    echo "warning: squeue not found; the run-folder guard cannot see other jobs" >&2
fi
""" % {'record': RUN_FOLDER_RECORD}


def _preamble_lines(preamble):
    """The lines of `preamble` (None, a str or an iterable of str)."""
    if preamble is None:
        return []
    if isinstance(preamble, str):
        lines = preamble.splitlines()
    else:
        lines = []
        for item in preamble:
            if not isinstance(item, str):
                raise TypeError(f'preamble must be a str or an iterable of str, got an item '
                                f'{item!r}')
            lines.extend(item.splitlines())
    return [line.rstrip() for line in lines]


def _preflight_arguments(preflight):
    """None (no preflight) or the extra command-line arguments of the preflight."""
    if preflight is None or preflight is False:
        return None
    if preflight is True:
        return []
    if isinstance(preflight, str):
        return shlex.split(preflight)
    raise TypeError(f"preflight must be False, True or a str of extra arguments for "
                    f"'python -m {PREFLIGHT_MODULE}', got {preflight!r}")


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

    def _preflight_runs(self, built_args):
        """``[(params file, run folder)]`` of the C-GAIMMD runs that start, for the preflight.

        The params file is the one the run's workers are started with (the
        first element of their argument tuples, see :meth:`_build`).
        """
        runs = []
        for run_id, params in enumerate(self.params):
            if getattr(params, 'chain_type', None) != 'cgaimmd' \
                    or not self._runs_anything(run_id):
                continue
            folder = self._directories[run_id] or '.'
            if getattr(params, 'multi_system', False):
                raise ValueError(f'preflight: the run {folder!r} is a multi-system run; the '
                                 f'C-GAIMMD preflight checks single systems')
            prefix = folder.rstrip('/') + '/'
            params_path = next(args[0] for args in built_args
                               if args[1] == folder or str(args[1]).startswith(prefix))
            runs.append((params_path, folder))
        if not runs:
            raise ValueError("preflight: no run of this job has chain_type='cgaimmd'; the "
                             "preflight checks C-GAIMMD params files")
        return runs

    def create_job(self, filename, n=1, n1=0, n2=0,
                   reactive_region_mode='chain',
                   state1_mode='free', state2_mode='free',
                   nsteps=inf, nframes=inf, nrounds=inf,
                   walltime=24*3600, cpus_per_task=1, gpus_per_task=0,
                   ntasks_per_node=1, skip_binding=True,
                   nchains_per_worker=1, *, preamble=None, preflight=False,
                   guard_run_folders=True):
        """
        Create a SLURM job script that launches AIMMD workers via ``srun``.

        The script produced by this method:

        - writes the SLURM header derived from ``params.slurm_header`` and from
          the provided defaults (if missing),
        - writes the ``preamble`` lines (the job's environment), if any,
        - claims the run folders for this job (``guard_run_folders``),
        - runs the C-GAIMMD preflight of every C-GAIMMD run (``preflight``),
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
            Mode for state-2 replicas. Default is ``'free'``. With
            ``chain_type='cgaimmd'``, ``'shoot'`` (for either state) needs a
            transition in every initial path, which internal shooting
            chains extract; per-basin initial files need ``'free'``.
        nchains_per_worker : int, optional, default = 1
            If > 1, the worker will manage more than one chain. A higher value
            of `nchains_per_worker` tends to regularize the training set and
            thus improve performance. If running only one shooting worker,
            `nchains_per_worker=10` is recommended.
        nsteps : float or iterable of float, optional
            Maximum number of simulated independent trajectories (worker stop
            condition). Default is ``inf``. Attention! If "train" runs, then
            nsteps refers to the total number of steps across the shooting
            simulations only.
            Otherwise, it refers to the number of steps of each single worker
            in the launcher run. The first worker reaching nsteps stops all
            the others.
            Exception -- ``reactive_region_mode='sweep'``: ``nsteps`` is each
            sweep worker's *share* of a global target, so the campaign runs
            until the combined committed-shot count over all ``n`` sweep workers
            reaches ``n * nsteps`` (the per-worker step/frame caps are disabled
            and the job waits for all workers rather than scancel-ing on the
            first to finish). This keeps the intended total exact even when
            workers are uneven, and lets a finished campaign be extended by
            raising ``nsteps`` and resubmitting.
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
        preamble : str or iterable of str, optional, keyword only
            Shell lines written after the SLURM header, before anything else
            runs: the job's environment, e.g. ``['module load
            GROMACS/2025.4-CUDA-13', 'source /path/to/venv/bin/activate',
            'export HF_HUB_OFFLINE=1']``. Written as they are. Default: none.
        preflight : bool or str, optional, keyword only
            ``True``: before any worker starts, run ``"${PYTHON}" -m
            aimmd.cgaimmd.preflight <params file> --run <run folder>`` for
            every run with ``chain_type='cgaimmd'`` (one generated candidate
            through the whole preparation, the GROMACS executables, the
            PLUMED bias); the job stops when a check fails, before compute
            time is spent. A str adds arguments to that command (e.g.
            ``'--candidates 2'`` or ``'--no-generate'``). Default ``False``.
        guard_run_folders : bool, optional, keyword only
            ``True`` (default): the job writes its SLURM job id into
            ``<run folder>/aimmd_job.txt`` for each of its run folders, and
            stops before any worker starts when another job that wrote one of
            these files is still running (``squeue``), or when two jobs claim
            a folder at the same moment (the later claim wins). Two jobs in
            one run folder would write the same chains, caches and network
            files. ``False`` writes no guard.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If ``filename`` cannot be written.
        TypeError
            If the launcher cannot be configured with the requested modes, or
            for a `preamble` or `preflight` of the wrong type.
        ValueError
            For ``preflight`` when no run of the job has
            ``chain_type='cgaimmd'``, or a multi-system C-GAIMMD run (the
            preflight checks single systems).
        aimmd.core.series.SeriesCoverageError
            If trajectories of a run have no file of the params' descriptor
            series and its featurizer does not refill it (no job is
            written); with ``refill=True`` the refill is announced instead.

        Notes
        -----
        - The effective number of nodes is computed as:

          ``ceil(total_processes / ntasks_per_node)``

          and written as ``#SBATCH --nodes=...``.
        - The script uses ``srun --exclusive --nodes=1 --ntasks=1`` for each
          worker to ensure processes do not share a task allocation.
        - Paths in the script (params files, run folders) are relative to
          the folder ``create_job`` runs in; submit the job from there (SLURM
          starts a batch script in the submission folder).
        """
        preamble_lines = _preamble_lines(preamble)
        preflight_arguments = _preflight_arguments(preflight)

        # A GROMACS-engine job script only makes sense if `gmx` is available to
        # the workers it launches; require it at generation time. Toy-engine job
        # scripts need no GROMACS.
        if any(getattr(p, 'engine', None) == 'gromacs' for p in self.params):
            require_gromacs()

        # every trajectory of the runs needs the params' descriptor series:
        # refuse the job (or announce its refill) before it is written
        self._check_series()

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

        # the worker commands (this also prepares the run folders); the
        # preflight needs their params files
        built_args, built_descriptions = self._build()
        preflight_runs = (self._preflight_runs(built_args)
                          if preflight_arguments is not None else [])
        run_folders = []
        for run_id in range(len(self)):
            folder = self._directories[run_id] or '.'
            if self._runs_anything(run_id) and folder not in run_folders:
                run_folders.append(folder)

        # write job script
        with open(filename, 'w') as file:

            # slurm header
            file.write('#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params[0].name}\n')
            file.write(f'{slurm_header}\n\n')

            # the job's environment
            if preamble_lines:
                file.write('# environment\n')
                file.write('\n'.join(preamble_lines) + '\n\n')

            # default names
            file.write('# default names\n')
            file.write(f'PYTHON="{PYTHON}"\n')
            file.write(f'WORKER="{WORKER}"\n\n')

            # one job per run folder
            if guard_run_folders and run_folders:
                file.write('# one job per run folder: claim the run folders, or stop if another\n'
                           '# job that is still running claimed one of them\n')
                file.write(_GUARD_FUNCTION)
                for folder in run_folders:
                    file.write(f'claim_run_folder {shlex.quote(folder)}\n')
                file.write('sleep 2\n')
                for folder in run_folders:
                    file.write(f'confirm_run_folder {shlex.quote(folder)}\n')
                file.write('\n')

            # preflight of the C-GAIMMD runs
            if preflight_runs:
                file.write('# preflight: check every C-GAIMMD run before any worker starts\n')
                for params_path, folder in preflight_runs:
                    command = ['-m', PREFLIGHT_MODULE, params_path, '--run', folder,
                               *preflight_arguments]
                    file.write('"${PYTHON}" ' + ' '.join(shlex.quote(word) for word in command)
                               + ' || {\n')
                    file.write(f'    echo "preflight of {params_path} failed: no worker '
                               f'started" >&2\n')
                    file.write('    exit 1\n')
                    file.write('}\n')
                file.write('\n')

            # enable job control
            file.write('# enable job control\n')
            file.write('set -m\n')

            # launch commands
            file.write('\n# workers')
            for i, (args, description) in enumerate(
                    zip(built_args, built_descriptions)):
                file.write(f'\n# {description}\n')
                args = ' '.join([f'"{arg}"'
                                 if not isinstance(arg, (bool, bool_)) else
                                 '"True"' if arg else '""' for arg in args])
                file.write(f'srun --exclusive --nodes=1 --ntasks=1 '
                           f'--cpus-per-task={cpus_per_task} '
                           f'--gpus-per-task={gpus_per_task} \\\n')
                file.write(f'  "${{PYTHON}}" "${{WORKER}}" {args} &\n')

            # Termination policy depends on whether a trainer coordinates the
            # workers (task name lives at the fixed index 10 of each worker's
            # argument tuple). With a trainer, the workers are interdependent
            # and the run ends when the trainer finishes: the first process to
            # exit triggers `scancel` to tear everything down. Without one
            # (e.g. a committor sweep, which forces nrounds=0), the workers are
            # independent and each exits cleanly on its own stop condition --
            # the global sweep target or walltime -- so we simply wait for ALL
            # of them. This also means one worker crashing no longer scancels
            # the whole sweep.
            has_trainer = any(len(a) > 10 and a[10] == 'train'
                              for a in built_args)
            if has_trainer:
                # wait until any process exits, then cancel the rest
                file.write('\n# wait until any process exits\n')
                file.write('wait -n\n')
                file.write('scancel ${SLURM_JOB_ID}\n')
                file.write('wait\n')
            else:
                # independent workers: let every one finish on its own
                file.write('\n# wait for all workers to finish\n')
                file.write('wait\n')
