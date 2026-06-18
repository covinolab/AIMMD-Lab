"""
aimmd.launcher._helpers
======================

Helper mixin for :class:`aimmd.launcher.Launcher`.

This module defines :class:`LauncherHelpers`, which provides the initialization
and configuration logic used by AIMMD launchers. A launcher coordinates one or
more *runs* (potentially with different parameter sets and working directories)
and decides how many worker processes should be spawned for each run, and with
which per-process resource allocations.

In contrast to :class:`aimmd.worker.Worker`, which executes a single task in a
single process, the launcher is responsible for:

- validating and normalizing user inputs (params and directories),
- expanding scalar configuration values to per-run arrays,
- computing the number of worker processes required per run,
- computing per-process CPU/GPU allocations from the available resources,
- installing process-wide signal handlers to terminate all spawned tasks.

Concepts
--------
Run
    A single AIMMD working directory + parameter set pair. A launcher may manage
    multiple runs at once (e.g., multiple replicas or multiple systems).

Process budget
    For each run, the launcher computes how many processes are needed for:
    - reactive-region sampling (``n``),
    - state-1 sampling (``n1``),
    - state-2 sampling (``n2``),
    - training (``nrounds``, > 0 value adds a training worker).

Resource sharing
    ``cpus_per_task`` and ``gpus_per_task`` can be specified as policies:

    - ``'skip'``: do not bind; only report availability.
    - ``'share'``: divide available resources across all processes the launcher
      intends to run (across all runs combined).
    - ``'all'``: assign all available resources to each process.
    - int: explicit number of resources per process.

Signal behavior
---------------
The launcher installs SIGINT/SIGTERM handlers that clear all managed processes.
On SIGINT it raises ``KeyboardInterrupt``; on SIGTERM it exits with the
conventional code ``128 + SIGTERM``.

Notes
-----
- The helper `_terminate_handler` references ``sys`` but this module does not
  import it. This is part of the original code and is preserved here; ensure
  that the concrete launcher module imports ``sys`` or that this handler is not
  invoked in contexts where ``sys`` is unavailable.
"""

# external
import sys
import numpy as np
import signal
from abc import ABC
from math import inf
from collections.abc import Iterable

# aimmd imports
from ._run import run_task
from ..params import Params
from .._config import print
from ..resources import get_num_cpus, get_num_gpus
from ..core.utils import now
from ..execute.processes import ProcessExecutor


class LauncherHelpers(ABC):

    def _init(self, params, directories, termination_timeout=60.):
        """
        Initialize a launcher instance.

        Parameters
        ----------
        params : str or aimmd.params.Params or iterable of these
            Parameter specification(s) for the run(s). Each element may be:

            - a path to a saved Params file (string), loaded via
              :meth:`aimmd.params.Params.load`, or
            - an already instantiated :class:`~aimmd.params.Params`.

            If a single string/Params is given, it is treated as a one-element
            list (one run).
        directories : str or iterable of str
            Working directory (or directories) in which to run simulations. Each
            directory is stored as a string and is interpreted relative to the
            current working directory by higher-level launcher logic.

            If a single string is given, it is treated as a one-element list.
        termination_timeout : float, optional
            Grace period (seconds) for terminating worker processes, after which
            they may be killed by the executor. Stored as
            :attr:`termination_timeout`. Default is ``60.``.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If a params entry cannot be loaded, if any params object has no
            initial paths, or if ``directories`` cannot be converted into a list
            of strings.

        Notes
        -----
        - All Params are forced to have a saved file on disk: if ``params[i].path``
          is not a file, ``params[i].save()`` is called.
        - If the number of params and directories differs, the shorter list is
          extended by repeating its last element so that the lengths match.
        - Signal handlers for SIGTERM and SIGINT are installed at initialization.
        """
        # process params
        if isinstance(params, (str, Params)):
            params = [params]
        else:
            params = list(params)

        # convert to Params object and check
        for i in range(len(params)):
            if isinstance(params[i], str):
                try:
                    params[i] = Params.load(params[i])
                except:
                    raise TypeError(f'failed loading params {params[i]!r}')
            if not params[i].initial_paths:
                raise TypeError(f'{i}-th input params have no initial paths')
            # params need a saved file
            if not params[i].path.is_file():
                params[i].save()

        # process directories
        if type(directories) is str:
            directories = [directories]
        else:
            try:
                directories = [str(directory) for directory in directories]
            except:
                raise TypeError(f'{directories!r} must be either a str '
                                f' or a list of strings')

        # extend directories to params or viceversa
        delta = len(params) - len(directories)
        directories.extend(directories[-1:] * delta)
        params.extend(params[-1:] * -delta)

        # initialize processes
        self._processes = ProcessExecutor()

        # populate fields
        self._params = params
        self._directories = directories
        self._update()  # initialize
        self.termination_timeout = termination_timeout

        # register signal handlers (for all future tasks)
        signal.signal(signal.SIGTERM, self._terminate_handler)
        signal.signal(signal.SIGINT, self._terminate_handler)

    def _terminate_handler(self, signum=None, frame=None):
        """
        Terminate all managed processes in response to a signal.

        Parameters
        ----------
        signum : int, optional
            Received signal number.
        frame : frame, optional
            Current stack frame (unused).

        Returns
        -------
        None

        Side Effects
        ------------
        - Clears all processes tracked by :attr:`_processes`.
        - On SIGINT raises ``KeyboardInterrupt``.
        - On SIGTERM exits with code ``128 + SIGTERM``.
        - On any other signal exits with code ``0``.

        Notes
        -----
        The concrete launcher is expected to provide a meaningful ``__repr__``
        so that the printed message identifies the launcher instance.
        """
        print(f'\n[{self}] received termination signal {signum} {now()}')
        self._processes.clear()

        # standard behavior
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        elif signum == signal.SIGTERM:
            sys.exit(128 + signal.SIGTERM)
        else:
            sys.exit(0)

    def _process_input(self, name, value, dtype=None):
        """
        Normalize a configuration input to a per-run array.

        Parameters
        ----------
        name : str
            Name of the input (used in error messages).
        value : object
            Input value. Supported forms are:
            - scalar of type ``dtype`` (replicated across runs),
            - iterable of values (must have length ``len(self)``),
            - any other scalar (replicated and cast to ``dtype``).
        dtype : type, optional
            Desired dtype for the resulting NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of length ``len(self)`` containing one value per run.

        Raises
        ------
        TypeError
            If an iterable is provided and its length does not match the number
            of runs.
        """
        if isinstance(value, dtype):
            value = np.repeat(value, len(self))
        elif isinstance(value, Iterable):
            value = np.asarray(value, dtype=dtype)
            if len(value) != len(self):
                raise TypeError(f"{name}'s length must be the same as "
                                f"number of runs ({len(self)})")
        else:
            value = np.repeat(value, len(self)).astype(dtype)
        return value

    def _update(self,
                n=1, n1=0, n2=0,
                reactive_region_mode='chain',
                state1_mode='free',
                state2_mode='free',
                nsteps=inf,
                nframes=inf,
                nrounds=None,
                walltime=inf,
                cpus_per_task='share',
                gpus_per_task='share',
                ntasks_per_node=None):
        """
        Update the launch plan and derived resource assignments.

        This method converts the provided configuration into per-run arrays,
        validates mode selections, computes how many worker processes will be
        launched per run, and determines CPU/GPU allocations per process.

        Parameters
        ----------
        n : int or iterable of int, optional
            Number of processes allocated to sampling in the reactive region for
            each run. Default is ``1``.
        n1 : int or iterable of int, optional
            Number of processes allocated to sampling associated with state 1.
            Default is ``0``.
        n2 : int or iterable of int, optional
            Number of processes allocated to sampling associated with state 2.
            Default is ``0``.
        reactive_region_mode : {'chain', 'free', 'sweep'} or iterable, optional
            Sampling mode used for the reactive region for each run. Default is
            ``'chain'``.
        state1_mode : {'free', 'shoot'} or iterable, optional
            Mode used for state 1 processes for each run. Default is ``'free'``.
        state2_mode : {'free', 'shoot'} or iterable, optional
            Mode used for state 2 processes for each run. Default is ``'free'``.
        nsteps : float or iterable of float, optional
            Maximum number of simulated independent trajectories (worker stop
            condition). Default is ``inf``. Attention! If "train" runs, then
            nsteps refers to the total number of steps across all workers.
            Otherwise, it refers to the number of steps of each single worker
            in the launcher run. The first worker reaching nsteps stops all
            the others.
        nframes : float or iterable of float, optional
            Maximum number of simulated frames (worker stop condition). Default
            is ``inf``. Attention! If "train" runs, then
            nframes refers to the total number of frames across all workers.
            Otherwise, it refers to the number of nframes of each single worker
            in the launcher run. The first worker reaching nsteps stops all
            the others..
        nrounds : float or iterable of float, optional
            If `None` and new simulations are requested, add a new process that
            trains the model and computes selection bins and densities 
            indefinitely. If `None` and no new simulations are requested, just
            does one round before exiting. If != 0, the process does training
            rounds up until reaching `nrounds`, from that point on it just
            updates selection bins and densities.
            Forced to zero when `reactive_region_mode = 'sweep'`.
        walltime : float, optional
            Walltime limit in seconds (shared scalar across runs). Default is
            ``inf``.
        cpus_per_task : {'skip', 'share', 'all'} or int, optional
            CPU allocation policy per process (see module docstring). Default is
            ``'share'``.
        gpus_per_task : {'skip', 'share', 'all'} or int, optional
            GPU allocation policy per process (see module docstring). Default is
            ``'share'``.
        ntasks_per_node : int, optional
            If provided, explicitly set the number of tasks per node. If not,
            it defaults to the total number of processes computed across all
            runs.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any of the mode entries contain unsupported values.
        """
        
        # process input
        n = self._process_input('n', n, int)
        n1 = self._process_input('n1', n1, int)
        n2 = self._process_input('n2', n2, int)
        reactive_region_mode = self._process_input(
            'reactive_region_mode', reactive_region_mode, str)
        state1_mode = self._process_input('state1_mode', state1_mode, str)
        state2_mode = self._process_input('state2_mode', state2_mode, str)
        nsteps = self._process_input('nsteps', nsteps, float)
        nframes = self._process_input('nframes', nframes, float)
        nrounds = self._process_input('nrounds', nrounds, object)
        walltime = float(walltime)

        # process nrounds according to specification
        for i in range(len(self)):
            # force to zero if mode is sweep
            if reactive_region_mode[i] == 'sweep':
                nrounds[i] = 0
            if nrounds[i] is None:
                if n[i] + n1[i] + n2[i]:
                    nrounds[i] = inf
                else:
                    nrounds[i] = 1
        nrounds = nrounds.astype(float)
        
        # modes check
        for mode in reactive_region_mode:
            if mode not in ('chain', 'free', 'sweep'):
                raise TypeError("'reactive_region_mode' must be either "
                                f"'chain', 'free', or 'sweep', got {mode!r}")
        for mode in state1_mode:
            if mode not in ('free', 'shoot'):
                raise TypeError("'state1_mode' must be either "
                                f"'free' or 'shoot', got {mode!r}")
        for mode in state2_mode:
            if mode not in ('free', 'shoot'):
                raise TypeError("'state2_mode' must be either "
                                f"'free' or 'shoot', got {mode!r}")

        # assign fields
        self._n = n
        self._n1 = n1
        self._n2 = n2
        self._reactive_region_mode = reactive_region_mode
        self._state1_mode = state1_mode
        self._state2_mode = state2_mode
        self._nsteps = nsteps
        self._nframes = nframes
        self._nrounds = nrounds
        self._walltime = walltime

        # get number of processes. In multi-system runs the per-run worker
        # counts (n/n1/n2) apply PER SYSTEM and there is one trainer per system
        # (separate networks) or a single shared trainer (shared network).
        per_run_workers = self._n1 + self._n2 + self._n
        has_trainer = self._nrounds.astype(bool).astype(int)
        num_processes = []
        for run_id in range(len(self)):
            params = self._params[run_id]
            if getattr(params, 'multi_system', False):
                n_systems = max(1, len(params.system_ids))
                n_trainers = (1 if params.multi_system_share_network
                              else n_systems)
                num_processes.append(int(per_run_workers[run_id]) * n_systems
                                     + n_trainers * int(has_trainer[run_id]))
            else:
                num_processes.append(int(per_run_workers[run_id])
                                     + int(has_trainer[run_id]))
        self._num_processes = np.array(num_processes)
        total_num_processes = sum(self._num_processes)

        # determine number of CPUs per task
        self._cpus_per_task = cpus_per_task
        if cpus_per_task != 'skip':
            num_cpus_avail = get_num_cpus()
            if cpus_per_task == 'share':
                self._cpus_per_task = max(1,
                    num_cpus_avail // total_num_processes)
            elif cpus_per_task == 'all':
                self._cpus_per_task = num_cpus_avail
            else:
                self._cpus_per_task = int(cpus_per_task)

        # determine number of GPUs per task
        self._gpus_per_task = gpus_per_task
        if gpus_per_task != 'skip':
            num_gpus_avail = get_num_gpus()
            if gpus_per_task == 'share':
                self._gpus_per_task = max(int(num_gpus_avail > 0),
                    num_gpus_avail // total_num_processes)
            elif gpus_per_task == 'all':
                self._gpus_per_task = num_gpus_avail
            else:
                self._gpus_per_task = int(gpus_per_task)

        if ntasks_per_node:
            self._ntasks_per_node = int(ntasks_per_node)
        else:
            self._ntasks_per_node = total_num_processes
