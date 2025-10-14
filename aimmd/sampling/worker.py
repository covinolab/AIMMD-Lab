import os
import pty
import sys
import psutil
import signal
import subprocess
from ..core import Params
from ..core.utils import get_current_simulation, now

class Worker:
    
    def __init__(self, params, localid=0, cpus_per_task=1, gpus_per_task=0):
        """
        Worker process responsible for running independent AIMMD tasks
        (simulations, training, or management) on allocated CPUs/GPUs.
        """
        
        if not isinstance(params, Params):
            params = Params.load(params)
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
    
    def terminate_handler(self, signum=None, frame=None, exit=False):
        """Gracefully terminate the worker and its subprocess."""
        
        # report
        msg = (f"[Worker {self.localid}] Received signal "
               f"{signum}, terminating ({now})...")
        print(msg)
        if self.log:
            self.log.write(msg + "\n")
            self.log.flush()
        
        # end current process
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        
        # close log
        if self.log:
            self.log.close()
        
        # exit if required
        if exit:
            sys.exit(0)
    
    def simulate(self, run_file, log_file=None, append=False):
        """
        Continuously run simulations as directed by the run file.
        append: bool, if True: do not create new part (Gromacs' -noappend).
        """
        
        self.log = open(log_file, "a+") if log_file else None
        print(f"[Worker {self.localid}] Starting simulation loop...")
        print(f"Press Control+C to interrupt.")
        
        try:
            
            # run continuously
            while True:
                # what to simulate
                fname = get_current_simulation(run_file)
                if not fname:
                    continue  # no job assigned yet
                
                # create command
                command = self.params.mdrun.split() + [
                    "-deffnm", fname,
                    "-cpo", f"{fname}.cpt",
                    "-cpi", f"{fname}.cpt",
                    "-cpt", ".1"]
                if not self.append:
                    command.append("-noappend")
                        
                # open pseudo-terminal to capture real-time stdout
                master_fd, slave_fd = pty.openpty()
                
                # run command
                process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy(),
                    close_fds=True)
                os.close(slave_fd)
                
                with os.fdopen(master_fd) as stdout:
                    for line in iter(stdout.readline, ''):
                        if get_current_simulation(run_file) != fname:
                            msg =(f"[Worker {self.localid}] Simulation changed — terminating.")
                            self.process.terminate()
                            break
                        
                        print(line, end="")
                        if self.log:
                            self.log.write(line)
                            self.log.flush()
                
                if self.process.poll() is None:
                    self.process.wait()
                self.process = None
        
        except SystemExit:
            self.terminate_handler()
        
        except KeyBoardInterrupt:
            self.terminate_handler(exit=False)
        
        finally:
            self.terminate_handler()
    
    def train(self):
        """Placeholder for ML training task."""
        print(f"[Worker {self.localid}] Training task not yet implemented.")

    def manage(self):
        """Placeholder for management/supervisory task."""
        print(f"[Worker {self.localid}] Management task not yet implemented.")

if __name__ == '__main__':
    Worker(Params.load('temp.py')).simulate(
        sys.argv[1], sys.argv[2], len(sys.argv) > 3)
