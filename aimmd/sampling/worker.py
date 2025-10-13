import os
import sys
import psutil
import signal
import selectors
import subprocess
from ..core import Params
from ..core.utils import get_current_simulation

def limit_resources(localid=0, cpus_per_task=1, gpus_per_task=0):
    """Restrict this process to the allocated CPUs and GPUs."""
    
    # local id 
    localid = int(os.getenv("SLURM_LOCALID", f'{localid}'))
    
    # CPU binding
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK", f'{cpus_per_task}')
    if cpus_per_task:
        try:
            cpus_per_task = int(cpus_per_task)
            start = localid * cpus_per_task
            cpus = list(range(start, start + cpus_per_task))
            psutil.Process().cpu_affinity(cpus)
        except Exception as exception:
            print(f"Warning: could not set CPU affinity ({exception})")
            cpus = 'all'
    else:
        cpus = 'all'
    
    # GPU binding
    start = localid * gpus_per_task
    gpus = ','.join([f'{gpu_id}' for gpu_id in range(
        localid * gpus_per_task, start + gpus_per_task)])
    gpus = os.getenv("CUDA_VISIBLE_DEVICES", gpus if gpus else None)
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if cpus != 'all':
        cpus = ','.join([f'{cpu_id}' for cpu_id in cpus])
    print(f'Local id: {localid}')
    print(f'CPU ids : {cpus}')
    print(f'CUDA ids: {gpus}')


def run(params, run_file, log_file, append,
        localid=0, cpus_per_task=1, gpus_per_task=0):
    """
    Run and log in real-time on the allocated resources.
    
    params: Params class or dill file with saved params.
    run_file: File indicating what to simulate.
    log_file: path to log file.
    append: Append to existing simulations or start a new part.
    localid, cpus_per_task, gpus_per_task: resource allocation.
    """
    
    if type(params) != Params:
        params = Params.load(params)
    
    limit_resources(localid, cpus_per_task, gpus_per_task)
    
    log = open(log_file, "a+") if log_file else None
    process = None  # current subprocess
    
    def terminate_handler(signum, frame):
        print(f"Received signal {signum}, terminating...")
        if log:
            log.write(f"Received signal {signum}, terminating...")
        if process and process.poll() is None:
            process.terminate()
        sys.exit(0)
    
    # register signal handlers
    signal.signal(signal.SIGTERM, terminate_handler)
    signal.signal(signal.SIGINT, terminate_handler)
    
    try:
        while True:
            # get current simulation
            fname = get_current_simulation(run_file)
            if not fname:
                continue  # nothing to run
            
            # simulation command
            command = params.mdrun.split() + [
                "-deffnm", fname,
                "-cpo", f"{fname}.cpt",
                "-cpi", f"{fname}.cpt",
                "-cpt", ".1"]
            if not append:
                command.append('-noappend')
            
            # start subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy())
            
            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            
            try:
                while True:
                    # read available output
                    for key, _ in sel.select(timeout=0.1):
                        line = key.fileobj.readline()
                        if not line:  # EOF
                            continue
            
                        print(line, end="")  # print to terminal
            
                        if log:
                            log.write(line)
                            log.flush()
            
                    # stop if simulation file changed
                    if get_current_simulation(run_file) != fname:
                        process.terminate()
                        break
            
                    # exit if process finished
                    if process.poll() is not None:
                        break
            
            finally:
                sel.unregister(process.stdout)
                process.stdout.close()
                process = None
    
    except KeyboardInterrupt:
        print(f"[Worker {localid}] Exiting main loop due to interrupt.")
    
    finally:
        if log:
            log.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: worker.py "
              f"<params_file> <run_file> <log_file> <append> [optional]")
        print(run.__doc__)
        sys.exit(1)
    
    run(*sys.argv[1:])
