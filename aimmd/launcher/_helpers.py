"""
...
"""

# external
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

# launcher helpers
class LauncherHelpers(ABC):
    
    def _init(self, params, directories, termination_timeout=60.):
        """
        params: either a string or an `aimmd.Params` instance, or a list
                thereof
        directory: path relative to working directory where to run simulations
        termination_timeout: float, default 20
            Grace time for terminating processes, after which they are killed
        
        All parameters for the run can be updated before (re)launching
        a simulation.
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
        self._update() # initialize
        self.termination_timeout = termination_timeout
        
        # register signal handlers (for all future tasks)
        signal.signal(signal.SIGTERM, self._terminate_handler)
        signal.signal(signal.SIGINT, self._terminate_handler)

    def _terminate_handler(self, signum=None, frame=None):        
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
        """make it a list as long as self"""
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
                nrounds=inf,
                walltime=inf,
                cpus_per_task='share',
                gpus_per_task='share',
                ntasks_per_node=None):
        """populate all fields required to run/create job"""
        
        # process input
        n = self._process_input('n', n, int)
        n1 = self._process_input('n1', n1, int)
        n2 = self._process_input('n2', n2, int)
        reactive = self._process_input(
            'reactive_region_mode', reactive_region_mode, str)
        state1_mode = self._process_input('state1_mode', state1_mode, str)
        state2_mode = self._process_input('state2_mode', state2_mode, str)
        nsteps = self._process_input('nsteps', nsteps, float)
        nframes = self._process_input('nframes', nframes, float)
        nrounds = self._process_input('nrounds', nrounds, float)
        walltime = float(walltime)

        # check and assign
        for mode in reactive:
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
        self._n = n
        self._n1 = n1
        self._n2 = n2
        self._reactive_region_mode = reactive
        self._state1_mode = state1_mode
        self._state2_mode = state2_mode
        self._nsteps = nsteps
        self._nframes = nframes
        self._nrounds = nrounds
        self._walltime = walltime
        
        # get number of processes
        self._num_processes = (self._n1 + self._n2 + self._n +
                               self._nrounds.astype(bool))
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
            if self._cpus_per_task * total_num_processes > num_cpus_avail:
                print(f'Warning: oversubscribing CPUs '
                      f'({num_cpus_avail} available on this machine)')
        
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
            if self._gpus_per_task * total_num_processes > num_gpus_avail:
                print(f'Warning: oversubscribing GPUs '
                      f'({num_gpus_avail} available on this machine)')
        
        if ntasks_per_node:
            self._ntasks_per_node = int(ntasks_per_node)
        else:
            self._ntasks_per_node = total_num_processes
