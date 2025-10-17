#WIP for now it works if class is called in the same location as params' working directory

import os
import sys
import time
import signal
import multiprocessing
from math import ceil
from pathlib import Path
from .worker import Worker
from ..core.params import Params

inf = float('inf')

# worker path
PYTHON = sys.executable
WORKER = os.path.join(os.path.dirname(
  os.path.abspath(__file__)), "worker.py")

# multiprocessing context: spwan
ctx = multiprocessing.get_context('spawn')

def _run_task(params_file, directory,
             localid, cpus_per_task, gpus_per_task,
             task, *args):
    worker = Worker(params_file, directory,
                    localid, cpus_per_task, gpus_per_task)
    if task != 'simulate':
        worker.is_process = True  # so to handle termination correctly
    return worker.run(task, *args)


class Launcher:
    
    def __init__(self, params, directory):
        """
        directory: where simulations carried
        params: python file with params or Params
        
        All parameters for the run can be updated before (re)launching a simulation.
        """
        if not isinstance(params, Params):
            params = Params.load(params)
        self.params = params
        self.directory = directory
        
        # params need a file
        if not params.path.is_file():
            i = 0
            while os.path.exists(
              fname := f'{params.path}/params{str(i) if i else ""}.py'):
                i += 1
            with open(fname, 'w') as file:
                file.write(f'{params}')  # already good
            print(f'Written {fname} for params')
            params.path = Path(fname).resolve()
        
        # create folder structure (keep existing data)
        for i, folder in enumerate([self.directory,
                                 f'{self.directory}/initial_paths',
                                 f'{self.directory}/equilibriumA',
                                 f'{self.directory}/equilibriumB',
                                 f'{self.directory}/extendA',
                                 f'{self.directory}/extendB']):
            if not os.path.exists(folder):
                os.system(f'mkdir {folder}')
                print(f'+++ created {folder}')
            if i > 1:
                os.system(f'touch {folder}/indicted_trajectories.log')
        
        # save initial paths
        os.system(f'rm -f {self.directory}/initial_paths/*')
        self.params.save_initial_paths(f'{self.directory}/initial_paths')
    
    def run(self, n, nA, nB, eA=0, eB=0,
            nsteps=inf, nframes=inf, walltime=inf,
            cpus_per_task=1, gpus_per_task=1):
        """
        n: number of replicas dedicated to shooting simulations
           (creates folders if not existing)
        nA: number of replicas dedicated to free simulations around A
        nB: number of replicas dedicated to free simulations around B
        eA: number of replicas dedicated to extending transitions reaching A
        eA: number of replicas dedicated to extending transitions reaching B
        nsteps: default inf, maximum number of shooting simulations
        nframes: default inf, maximum number of simulated frames,
                 has priority over nsteps
        walltime: default inf, maximum number of simulation time,
                  has priority over nframes and nsteps
        cpus_per_task
        gpus_per_task (if present)
        """
        
        processes = []
        
        # simulators
        total = nA + nB + eA + eB + n
        for i in range(total):
            localid = len(processes)
            if i < total - n:
                noappend = True
            else:
                noappend = False
            processes.append(ctx.Process(target=_run_task, args=(
                self.params.path, self.directory,
                localid, cpus_per_task, gpus_per_task,
                'simulate', f'worker{localid}.run', f'worker{localid}.log',
                noappend)))
        
        # trainer (sharing the same localid as manager)
        localid = len(processes)
        processes.append(ctx.Process(target=_run_task, args=(
            self.params.path, self.directory,
            localid, cpus_per_task, gpus_per_task,
            'train', 'trainer.log')))
        
        # manager (sharing the same localid as trainer)
        processes.append(ctx.Process(target=_run_task, args=(
            self.params.path, self.directory,
            localid, cpus_per_task, gpus_per_task,
            'manage', n, nA, nB, eA, eB,
            'manager.log', nsteps, nframes)))
        
        # function to terminate all workers
        def terminate_all(signum=None, frame=None, timeout=5, exit=True):
            for process in processes:
                if process.is_alive():
                    process.terminate()  # sends SIGTERM
            
            # wait
            t0 = time.time()
            while time.time() - t0 < timeout:
                completed = True
                for process in processes:
                    if process.is_alive():
                        completed = False
                if completed:
                    break
            
            # force termination
            if not completed:
                for process in processes:
                    if process.is_alive():
                        process.kill()
            
            if exit:
                sys.exit(0)
        
        # register signal handlers in the main process
        signal.signal(signal.SIGINT, terminate_all)
        signal.signal(signal.SIGTERM, terminate_all)
        
        # start all processes
        for process in processes:
            process.start()
        
        # wait for completion with wall-time
        t0 = time.time()
        
        try:
            while True:
                all_done = True
                for process in processes:
                    if process.exitcode:
                        # one process terminated
                        print(f"Worker {process.pid} terminated "
                              f"(exitcode={process.exitcode}), terminating all")
                        terminate_all(exit=False)
                    if process.is_alive():
                        all_done = False
                
                if all_done:
                    break
                
                # check wall time
                if time.time() - t0 > walltime:
                    print(f"Wall time {walltime} exceeded, terminating all")
                    terminate_all(exit=False)
                
                time.sleep(1)
        
        finally:
            # ensure all processes are cleaned up
            for process in processes:
                if process.is_alive():
                    process.terminate()
    
    def create_job(self, filename, n, nA, nB, eA=0, eB=0,
                   nsteps=inf, nframes=inf):
        """
        Returns a slurm script in `filename` that can be launched by cluster.
        Walltime's default is in slurm header!
        n: number of replicas dedicated to shooting simulations
           (creates folders if not existing)
        nA: number of replicas dedicated to free simulations around A
        nB: number of replicas dedicated to free simulations around B
        eA: number of replicas dedicated to extending transitions reaching A
        eA: number of replicas dedicated to extending transitions reaching B
        """
        
        # retrieve run information
        gpu = False
        ntasks_per_node = 1
        for fields in self.params.slurm_header.split():
            if ('gpu' in fields and
                '=0' not in fields and
                ':0' not in fields):
                gpu = True
            if 'ntasks-per-node' in fields:
                ntasks_per_node = int(fields.split('=')[-1])
            if 'cpus-per-task' in fields:
                cpus_per_task = int(fields.split('=')[-1])
        nodes = ceil((1 + n +  # trainer/worker, shooting
                      nA + nB +  # free A and B
                      eA + eB)  # extension A and B
                      / ntasks_per_node)
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params.name}\n')
            file.write(f'#SBATCH --nodes={nodes}\n')
            file.write(f'{self.params.slurm_header}\n\n')
            
            # remove completed information and which to run
            file.write(f'rm -f {self.directory}/completed.flag\n')
            file.write(f'rm -f {self.directory}/*.run\n\n')
            
            # srun initialization
            file.write(f"srun --cpus-per-task={cpus_per_task} "
                            f"--cpu-bind=cores bash -c '\n\n")
            file.write(f'  # update task variables\n')
            file.write(f'  export i=$SLURM_PROCID\n')
            file.write(f'  export li=$SLURM_LOCALID\n')
            if gpu:
                file.write(f'  export CUDA_VISIBLE_DEVICES=$li\n')
            
            file.write(f'\n  # default names\n')
            file.write(f'  PYTHON="{PYTHON}"\n')
            file.write(f'  WORKER="{WORKER}"\n')
            file.write(f'  PARAMS="{self.params.path}"\n')
            file.write(f'\ncase $i in\n')
            
            # equilibrium workers
            i = -1
            for i in range(nA + nB):
                file.write(f'{i})\n')
                if i < nA:
                    state = 'A'
                    j = i
                else:
                    state = 'B'
                    j = i - nA
                file.write(f'  # worker {i} (equilibrium {state}{j})\n')
                file.write('  "${PYTHON}" "${WORKER}" "${PARAMS}" '
                           f'"{self.directory}" simulate '
                           f'worker{i}.run worker{i}.log noappend\n')
                file.write(f'  ;;\n')
            
            # extension workers
            begin = i + 1
            for i in range(begin, begin + eA + eB):
                j = i - begin
                if j < eA:
                    state = 'A'
                else:
                    state = 'B'
                    j -= eA
                file.write(f'{i})\n')
                file.write(f'  # worker {i} (extension {state}{j})\n')
                file.write('  "${PYTHON}" "${WORKER}" "${PARAMS}" '
                           f'"{self.directory}" simulate '
                           f'worker{i}.run worker{i}.log noappend\n')
                file.write(f'  ;;\n')
            
            # shooting workers
            begin = i + 1
            for i in range(begin, begin + n):
                j = i - begin
                file.write(f'{i})\n')
                file.write(f'  # worker {i} (shooting {j})\n')
                file.write('  "${PYTHON}" "${WORKER}" "${PARAMS}" '
                           f'"{self.directory}" simulate '
                           f'worker{i}.run worker{i}.log &\n')
                file.write(f'  ;;\n')
            
            # trainer
            file.write(f'{i + 1})\n')
            file.write(f'  # trainer\n')
            file.write('  "${PYTHON}" "${WORKER}" "${PARAMS}" '
                       f'"{self.directory}" train '
                       f'trainer.log &\n')
            file.write(f'  trainer_pid=$!\n\n')

            # manager
            file.write(f'  # manager\n')
            file.write('  "${PYTHON}" "${WORKER}" "${PARAMS}" '
                       f'"{self.directory}" manage '
                       f'{n} {nA} {nB} {eA} {eB} '
                       f'manager.log {nsteps} {nframes} &\n')
            file.write(f'  manager_pid=$!\n\n')
            
            # handle task termination
            file.write(f'  # handle task termination\n')
            file.write(f'  while kill -0 $trainer_pid 2>/dev/null'
                       f' || kill -0 $manager_pid 2>/dev/null; do\n')
            file.write(f'    wait -n\n')
            file.write(f'    rm -f {self.directory}/*.run\n')
            file.write( '    scancel ${SLURM_JOB_ID}\n')
            file.write(f'  done\n  ;;\n')

            # end
            file.write(f'*)\n  echo "[Worker $i] No task assigned."\n  ;;\n')
            file.write(f'esac\n\'\n')
