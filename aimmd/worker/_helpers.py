"""
...
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
        Worker process responsible for running independent AIMMD tasks
        (simulations, training, or management) on allocated CPUs/GPUs.
        
        Parameters
        ----------
        params : str or aimmd.core.Params
            Path to the parameters file or Params object.
        directory : str, optional
            Working directory for the worker, by default '.'.
        localid : int, optional
            Local ID of the worker (used for resource allocation), by default 0.
        cpus_per_task: str or int, defaut 'skip'
            Number of CPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all (explicitly bind resources)
            if 'skip': just report available resources, do not explicitly bind
        gpus_per_task: str or int, default 'skip'
            Number of GPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all (explicitly bind resources)
            if 'skip': just report available resources, do not explicitly bind
        
        Returns
        -------
        None
        """
        
        # populate attributes
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

        # defaults
        self.task = 'worker'  # for reporting
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._log_file = None        
        self.t0 = inf  # when worker started
        self.termination_signal = None
        
        # register signal handlers (for all future tasks)
        # this works because worker is always in the main thread
        signal.signal(signal.SIGTERM, self._terminate_handler)
        signal.signal(signal.SIGINT, self._terminate_handler)
        
        # assign log file
        self.log_file = log_file
    
    def _terminate_handler(self, signum=None, frame=None):
        """Gracefully terminate the worker and its subprocess."""
        # acknowledge signal
        #print(f'\n"{self.task}" worker received termination signal '
        #      f'{signum} {now()}')
        self.termination_signal = signum

    def _terminate_operations(self):
        
        # close log file if open
        self.log_file = None
        
        # reset termination signal
        self.termination_signal = None
    
    def _bind_resources(self):
        return bind_resources(self.localid,
                              self.cpus_per_task,
                              self.gpus_per_task)

    def _reset_stop_condition(self):
        self.nsteps = inf
        self.nframes = inf
        self.walltime = inf
        self.t0 = inf
    
    def _update_stop_condition(self, **kwargs):
        self.t0 = time.time()
        for name in ('nsteps', 'nframes', 'walltime'):
            if name in kwargs:
                setattr(self, name, float(kwargs.pop(name)))
