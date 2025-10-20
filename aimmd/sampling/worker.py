import os
import sys
import time
import torch
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

        Parameters
        ----------
        params : str or aimmd.core.Params
            Path to the parameters file or Params object.
        directory : str, optional
            Working directory for the worker, by default '.'.
        localid : int, optional
            Local ID of the worker (used for resource allocation), by default 0.
        cpus_per_task : int, optional
            Number of CPUs allocated per task, by default 1.
        gpus_per_task : int, optional
            Number of GPUs allocated per task, by default 0.

        Returns
        -------
        None
        """
        
        self.directory = directory
        if not isinstance(params, Params):
            params = Params.load(params)
        self.params = params
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.task = 'worker'  # for reporting
        self.termination_signal = None
        self.termination_timeout = termination_timeout
        self.__log_file = None
        self.cleanup = []  # files to delete after termination
        
        # determine local id, if not provided
        # self.localid = int(os.getenv("SLURM_LOCALID", f"{localid}"))
        # I am disabling this automatic SLURM detection, because it creates issues where
        # counting doesn't start at zero, and we have doubling of some worker ids.
        # Instead, enforce that this is provided externally in the run script.
        self.localid = int(localid)
        self.cpus_per_task = int(cpus_per_task)
        self.gpus_per_task = int(gpus_per_task)

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

    def bind_resources(self):
        print(f'Worker\'s resources info')
        print(f'------------------------')
        print(f'LocalID {self.localid}')
        
        # CPU binding
        cpus_per_task = int(os.getenv(
            "SLURM_CPUS_PER_TASK", f"{self.cpus_per_task}"))
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
        
        # find available gpus, using torch to avoid extra dependency, on cuda or ROCm
        num_gpus_avail = torch.cuda.device_count()
        
        # check if requested resources are available
        if self.gpus_per_task > 0 and num_gpus_avail == 0:
            raise RuntimeError(f"No GPUs available but {self.gpus_per_task} requested")
        if self.gpus_per_task > num_gpus_avail:
            raise RuntimeError(f"Only {num_gpus_avail} GPUs available but "
                               f"{self.gpus_per_task} requested per task.")

        # GPU binding
        if self.gpus_per_task > 0:
            start = self.localid * self.gpus_per_task
            gpus = ",".join([f"{i%num_gpus_avail}" for i in range(
                start, start + self.gpus_per_task)])
            if gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus
                # for NVIDIA GPUs, and also ROCm picked up by torch
                os.environ["GPU_DEVICE_ORDINAL"] = gpus
                # for ROCm GPUs, Gromacs will use OpenCL
        
        # notify the user if this worker is oversubscribing a GPU
        if (self.localid > 0 and
            self.gpus_per_task > 0 and
            num_gpus_avail <= (self.localid * self.gpus_per_task)):
            print(f"[Note] Worker may be oversubscribing GPUs\n"
                  f"  Available GPUs: {num_gpus_avail}\n"
                  f"  GPUs per task: {self.gpus_per_task}")
        
        # report resource allocation
        print(f"CPU ids: {','.join(map(str, cpus)) if len(cpus) else 'all'}")
        if self.gpus_per_task > 0:
            print(f"GPU ids: {gpus}")
        else:
            print(f"No GPUs allocated")
        print(f'------------------------\n')
    
    def terminate_handler(self, signum=None, frame=None):
        """Gracefully terminate the worker and its subprocess."""
        
        # acknowledge signal
        print(f'\n"{self.task}" worker received termination signal '
              f'{signum} ({now()})')
        self.termination_signal = signum
    
    def terminate_operations(self):
        # delete what needs to
        while len(self.cleanup):
            remove(self.cleanup.pop())
        
        # close log file if open
        self.log_file = None
        
        # reset termination signal
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
    Worker(*sys.argv[1:6]).run(*sys.argv[6:])
