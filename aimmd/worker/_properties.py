"""
...
"""

# external
import sys
import time
from abc import ABC
from math import inf

# aimmd imports
from ..pathensemble import PathEnsemble

# properties of Worker class
class WorkerProperties(ABC):

    @property
    def log_file(self):
        return self._log_file
    
    @log_file.setter
    def log_file(self, log_file):
        if log_file is None:
            log_file = self.original_stdout
        if log_file == 'stdout':
            log_file = self.original_stdout
        if log_file == self._log_file:
            return
        if self.original_stdout != sys.stdout:
            sys.stdout.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if isinstance(log_file, str):
            log_file = open(f'{self.directory}/{log_file}', 'a+', buffering=1)
        if log_file:
            sys.stdout = log_file
            sys.stderr = sys.stdout
        self._log_file = log_file

    @property
    def must_stop(self):
        
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
