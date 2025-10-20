import os
import time
import sys
import psutil
import signal
import subprocess
import MDAnalysis as mda
from aimmd.sampling.train import train
from aimmd.sampling.manage import manage
from aimmd.sampling.simulate import simulate
from aimmd.core import Params
from aimmd.core.utils import now, remove

inf = float('inf')

class Worker:
    
    def __init__(self, params, directory='.',
                 localid=0, cpus_per_task=1, gpus_per_task=0,
                 termination_timeout=20.):
        """
        Worker process responsible for running independent AIMMD tasks
        (simulations, training, or management) on allocated CPUs/GPUs.
        """
        
        self.directory = directory
        if not isinstance(params, Params):
            params = Params.load(params)
        self.params = params
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.task = 'worker'  # for reporting
        self.is_process = False   # =belonging to a launcher run execution
        self.termination_signal = None
        self.termination_timeout = termination_timeout
        self.__log_file = None
        self.cleanup = []  # files to delete after termination
        
        # determine local id
        self.localid = int(os.getenv("SLURM_LOCALID", f"{localid}"))
        
        # CPU binding
        cpus_per_task = int(os.getenv(
            "SLURM_CPUS_PER_TASK", f"{cpus_per_task}"))
        os.environ["OMP_NUM_THREADS"] = str(cpus_per_task)
        os.environ["MKL_NUM_THREADS"] = str(cpus_per_task)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
        try:
            start = self.localid * cpus_per_task
            cpus = list(range(start, start + cpus_per_task))
            psutil.Process().cpu_affinity(cpus)
        except Exception as exception:
            print(f"[Warning] Could not set CPU affinity: {exception}")
            cpus = []
        
        # GPU binding
        start = self.localid * gpus_per_task
        gpus = ",".join([f"{i}" for i in range(start, start + gpus_per_task)])
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", gpus if gpus else None)
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        
        # report resource allocation
        print(f"CPU ids: {','.join(map(str, cpus))}")
        print(f"GPU ids: {gpus}")
        
        self.cpus = cpus
        self.gpus = gpus
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        
        # register signal handlers (for all future tasks)
        signal.signal(signal.SIGTERM, self.terminate_handler)
        signal.signal(signal.SIGINT, self.terminate_handler)
    
    @property
    def log_file(self):
        return self.__log_file
    
    @log_file.setter
    def log_file(self, log_file):
        if log_file == self.__log_file:
            return
        if self.original_stdout != sys.stdout:
            sys.stdout.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if log_file:
            sys.stdout = open(f'{self.directory}/{log_file}', 'a+')
            sys.stderr = sys.stdout
        self.__log_file = log_file
    
    def terminate_handler(self, signum=None, frame=None):
        """Gracefully terminate the worker and its subprocess."""
        
        # acknowledge signal
        print(f'\n"{self.task}" worker received termination signal '
              f'{signum} ({now()})')
        self.termination_signal = signum
    
    def terminate_operations(self):
        # delete what needs to
        for fname in self.cleanup:
            remove(fname)
        self.cleanup = []
        
        # close log file if open
        self.log_file = None
        
        # exit only if worker is a child process or not keyboardinterrupt
        if self.is_process or self.termination_signal != 2:
            self.termination_signal = None
            sys.exit(0)
        
        # reset termination signal in any case
        self.termination_signal = None
    
    
    def run(self, task, *args):
        
        # initialize
        self.termination_signal = None
        self.task = task
        
        try:
            # task execution
            if task == 'train':
                return train(self, *args)
            if task == 'manage':
                return manage(self, *args)
            if task == 'simulate':
                return simulate(self, *args)
            
            # not implemented
            raise TypeError(f'Task {task} not implented for AIMMD worker')
        
        finally:
            self.terminate_operations()
    
    def train(self, log_file=None, verbose=False, nrounds=inf, walltime=inf):
        
        # preprocessing exclusive to this function
        os.system(f'rm -f {self.directory}/initial_paths/*')
        self.params.save_initial_paths(f'{self.directory}/initial_paths')
        
        # just call "run"
        return self.run('train', log_file, verbose, nrounds, walltime)
    
    def manage(self, n, nA, nB, eA=0, eB=0, log_file=None,
               nsteps=inf, nframes=inf, walltime=inf):
        return self.run('manage', n, nA, nB, eA, eB,
                        log_file, nsteps, nframes, walltime)
    
    def simulate(self, run_file, log_file=None,
                 noappend=False, walltime=inf):
        return self.run('simulate', run_file, log_file, noappend, walltime)

if __name__ == '__main__':
    Worker(*sys.argv[1:3]).run(*sys.argv[3:])
