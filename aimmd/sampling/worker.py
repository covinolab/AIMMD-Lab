import os
import numpy as np
import sys
import psutil
import signal
import subprocess
from .train import train
from .manage import manage
from .simulate import simulate
from ..core import Params
from ..core.utils import now

class Worker:
    
    def __init__(self, params, directory='.',
                 localid=0, cpus_per_task=1, gpus_per_task=0):
        """
        Worker process responsible for running independent AIMMD tasks
        (simulations, training, or management) on allocated CPUs/GPUs.
        """

        self.directory = directory
        if not isinstance(params, Params):
            params = Params.load(f'{directory}/{params}')
        self.params = params
        self.process = None
        self.log = None
        
        # determine local id
        self.localid = int(os.getenv("SLURM_LOCALID", f"{localid}"))
        
        # CPU binding
        cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", f"{cpus_per_task}"))
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
        print(f"[Worker {self.localid}] CPU ids: {','.join(map(str, cpus))}")
        print(f"[Worker {self.localid}] GPU ids: {gpus}")
        
        self.cpus = cpus
        self.gpus = gpus
        self.cpus_per_task = cpus_per_task
        self.gpus_per_task = gpus_per_task
        
        # register signal handlers (for all future tasks)
        signal.signal(signal.SIGTERM, self.terminate_handler)
        signal.signal(signal.SIGINT, self.terminate_handler)  # (s, f)
    
    def terminate_handler(self, signum=None, frame=None, report=True, exit=False):
        """Gracefully terminate the worker and its subprocess."""
        
        # report
        if report:
            if signum:
                msg = f"Received signal {signum}, terminating process."
            else:
                msg = f"Terminating process."
            self.report(msg)
        
        # end current process
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.report("Process did not exit in time, killing...")
                self.process.kill()
            except Exception as exception:
                self.report(f"Exception while killing process: {exception}")
        
        # unbind process
        self.process = None
        
        # close and unbind log
        if self.log:
            self.log.close()
            self.log = None
        
        # exit if required
        if exit:
            sys.exit(0)
    
    def report(self, msg, preamble=True):
        if preamble:
            msg = f"[Worker {self.localid}] ({now()}) " + msg
        print(msg)
        if self.log and not self.log.closed:
            self.log.write(msg + "\n")
            self.log.flush()
    
    def run(self, task, *args):
        if task == 'train':
            return train(self, *args)
        if task == 'manage':
            return manage(self, *args)
        if task == 'simulate':
            return simulate(self, *args)
        raise TypeError(f'Task {task} not implented for AIMMD worker')
    
    def train(self, log_file=None, verbose=False):
        return train(self, log_file, verbose)
    
    def manage(self, n, nA, nB, eA, eB,
           log_file=None, nsteps=int(1e6), nframes=np.inf):
        return manage(self, n, nA, nB, eA, eB,
                      log_file, nsteps, nframes)
    
    def simulate(self, run_file, log_file=None, noappend=False):
        return simulate(self, run_file, log_file, noappend)

if __name__ == '__main__':
    Worker(*sys.argv[1:3]).run(*sys.argv[3:])
