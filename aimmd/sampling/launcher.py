import os
import sys
import time
import numpy as np
import signal
import multiprocessing
from math import ceil
from pathlib import Path
from .worker import Worker
from .resources import get_available_cpus
from ..core.utils import now
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
    try:
        Worker(params_file, directory,
               localid, cpus_per_task, gpus_per_task).run(task, *args)
    except Exception as exception:
        print(f'[Error] {exception}')
        return 1
    return 0


class Launcher:
    
    def __init__(self, params, directory,
                 termination_timeout=20.):
        """
        directory: where simulations carried
        params: python file with params or Params
        
        All parameters for the run can be updated before (re)launching
        a simulation.
        """
        if not isinstance(params, Params):
            params = Params.load(params)
        self.params = params
        self.directory = directory
        self.processes = Processes()  # in "run"
        self.termination_signal = None
        self.termination_timeout = termination_timeout
        
        # params need a written file
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
        
        # register signal handlers (for all future tasks)
        signal.signal(signal.SIGTERM, self.terminate_handler)
        signal.signal(signal.SIGINT, self.terminate_handler)
    
    def terminate_handler(self, signum=None, frame=None):        
        print(f'\nLauncher received termination signal {signum} ({now()})')
        self.termination_signal = signum
    
    def terminate_operations(self):
        self.processes.clean(self.termination_timeout)
        
        # not keyboard interruption
        if self.termination_signal != 2:
            raise SystemExit(0)
        
        # keyboard interruption: reset
        self.termination_signal = None
    
    def run(self, n, nA, nB, eA=0, eB=0,
            nsteps=inf, nframes=inf, walltime=inf,
            cpus_per_task=None, gpus_per_task=0,
            termination_timeout=20.):
        """
        Launch the simulation locally, spawning multiple processes.
        
        Parameters
        ----------
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
        cpus_per_task: default None, number of CPUs to allocate per task
            if None or 0: equally distribute available resources among workers
        gpus_per_task: default 1, number of GPUs to allocate per task

        Returns
        -------
        None
        """
        
        self.processes.clean()
        
        # total number of processes: simulators and workers
        num_processes = nA + nB + eA + eB + n + 1
        
        # determine number of CPUs per task
        if not cpus_per_task:
            num_cpus_avail = len(get_available_cpus())
            cpus_per_task = max(1, round(num_cpus_avail // num_processes))
        
        try:
            # simulators
            for i in range(num_processes - 1):
                localid = len(self.processes)
                if i < num_processes - 1 - n:
                    noappend = True
                else:
                    noappend = False
                self.processes.launch(self.params.path, self.directory,
                                      localid, cpus_per_task, gpus_per_task,
                                      'simulate', f'worker{localid}',
                                      f'worker{localid}.log', noappend)
            
            # trainer (sharing the same localid as manager)
            localid = len(self.processes)
            self.processes.launch(self.params.path, self.directory,
                                  localid, cpus_per_task, gpus_per_task,
                                  'train', 'trainer.log')
            
            # manager (sharing the same localid as trainer)
            self.processes.launch(self.params.path, self.directory,
                                  localid, cpus_per_task, gpus_per_task,
                                  'manage', n, nA, nB, eA, eB,
                                  'manager.log', nsteps, nframes)
            
            # wait for completion with walltime
            t0 = time.time()
            while time.time() - t0 < walltime and np.all(self.processes.alive):
                continue
        
        # safe termination
        finally:
            self.termination_signal = 2  # KeyboardInterrupt
            self.terminate_operations()
    
    @property
    def job_cleanup(self):
        """Bash script for cleanup when scanceling SLURM job"""
        return f'''# custom operation on scancel: full cleanup
function cleanup {{
    echo "Job is being canceled, doing cleanup"
    touch {self.directory}/.terminate
    sleep {int(self.termination_timeout)}
}}
trap cleanup SIGTERM'''
    
    @property
    def job_stop_condition(self):
        """Bash script for stop condition in SLURM job"""
        return f'''stop_condition() {{
        local pids=("$@")
        
        while true; do
            # exit if .terminate file exists
            if [[ -f "{self.directory}/.terminate" ]]; then
                break
            fi
            
            # exit if any PID in the list has terminated
            for pid in "${{pids[@]}}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break 2  # break both loops
                fi
            done
            
            # exit if exceeded (WALLTIME - {int(self.termination_timeout)}) s)
            current_time=$(date +%s)
            elapsed=$((current_time - START_TIME))
            if (( elapsed > WALLTIME - {int(self.termination_timeout)} )); then
                break
            fi
            
            sleep 1
        done
        
        # create terminate (if not existing already)
        touch "{self.directory}/.terminate"
        
        # send termination signal to all PIDs
        for pid in "${{pids[@]}}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
        
        # wait for all to exit cleanly
        wait "${{pids[@]}}" 2>/dev/null
    }}'''
    
    def create_job(self, filename, n, nA, nB, eA=0, eB=0,
                   nsteps=inf, nframes=inf,
                   cpus_per_task=1, gpus_per_task=0,
                   ntasks_per_node=1, walltime=24*3600):
        """
        Returns a slurm script in `filename` that can be launched by cluster.
        Walltime's default is in slurm header!
        
        Parameters
        ----------
        filename: name of the job script to create
        n: number of replicas dedicated to shooting simulations
           (creates folders if not existing)
        nA: number of replicas dedicated to free simulations around A
        nB: number of replicas dedicated to free simulations around B
        eA: number of replicas dedicated to extending transitions reaching A
        eB: number of replicas dedicated to extending transitions reaching B
        nsteps: default inf, maximum number of shooting simulations
        nframes: default inf, maximum number of simulated frames,
        cpus_per_task: default 1, number of CPUs to allocate per task,
            may be overridden by params.slurm_header
        gpus_per_task: default 1, number of GPUs to allocate per task,
            may be overridden by params.slurm_header
        ntasks_per_node: default 1, number of tasks per node
            may be overridden by params.slurm_header
        walltime: default 24*3600 s (24h) job simulation time
        """
        
        # retrieve run information: slurm header
        slurm_header = self.params.slurm_header + ''
        
        # retrieve run information: ntasks per node
        default_ntasks_per_node = ntasks_per_node
        ntasks_per_node = None
        for fields in slurm_header.split():
            if 'ntasks-per-node' in fields:
                ntasks_per_node = int(fields.split('=')[-1])
        if not ntasks_per_node:
            ntasks_per_node = default_ntasks_per_node
            slurm_header += \
                f'\n#SBATCH --ntasks-per-node={ntasks_per_node}'
        
        # retrieve run information: cpus per task
        default_cpus_per_task = cpus_per_task
        cpus_per_task = None
        for fields in slurm_header.split():
            if 'cpus-per-task' in fields:
                cpus_per_task = int(fields.split('=')[-1])
        if not cpus_per_task:
            cpus_per_task = default_cpus_per_task
            slurm_header += \
                f'\n#SBATCH --cpus-per-task={cpus_per_task}'
        
        # retrieve run information: gpus per task
        default_gpus_per_task = gpus_per_task
        gpus_per_task = None
        for fields in slurm_header.split():
            if 'gres=gpu:' in fields:
                gpus_per_task = int(fields.split(':')[-1]) // ntasks_per_node
            if 'gpus-per-task' in fields:
                gpus_per_task = int(fields.split('=')[-1])
        if not gpus_per_task:
            gpus_per_task = default_gpus_per_task
            if gpus_per_task:
                slurm_header += \
                    f'\n#SBATCH --gres:gpu={gpus_per_task * ntasks_per_node}'
        
        # number of nodes
        nodes = ceil((1 + n +  # trainer/worker, shooting
                      nA + nB +  # free A and B
                      eA + eB)  # extension A and B
                      / ntasks_per_node)
        slurm_header += f'\n#SBATCH --nodes={nodes}'
        
        # time information
        walltime = int(walltime)
        hours = walltime // 3600
        minutes = (walltime - hours * 3600) // 60
        seconds = walltime - hours * 3600 - minutes * 60
        slurm_header += f'\n#SBATCH --time={hours:02g}:{minutes:02g}:{seconds:02g}'
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params.name}\n')
            file.write(f'{slurm_header}\n')
            file.write(f"rm -f {self.directory}/.terminate\n\n")
            file.write(f"{self.job_cleanup}\n\n")
            
            # srun call
            file.write(f'# srun call\n')
            file.write(f"srun --cpus-per-task={cpus_per_task} "
                            f"--cpu-bind=cores bash -c '\n")
            file.write(f'  export i=$SLURM_PROCID\n\n')
            
            # default names
            file.write(f'  # default names\n')
            file.write(f'  PYTHON="{PYTHON}"\n')
            file.write(f'  WORKER="{WORKER}"\n')
            file.write(f'  PARAMS="{self.params.path}"\n\n')
            
            # stop condition
            file.write(f'  # setup stop condition\n')
            file.write(f"  START_TIME=$(date +%s)\n")
            file.write(f"  WALLTIME={walltime}\n\n")
            file.write(f"  {self.job_stop_condition}\n\n")
            
            # cases
            file.write(f'  # srun rank by rank\n')
            file.write(f'  case $i in\n')
            def _case(i, description, noappend=False):
                file.write(f'\n  {i})  # worker {i} ({description})\n')
                file.write(f'    "${{PYTHON}}" "${{WORKER}}" "${{PARAMS}}" '
                           f'{self.directory} {i % ntasks_per_node} '
                           f'{cpus_per_task} {gpus_per_task} '
                           f'simulate worker{i} worker{i}.log'
                           f'{" noappend" if noappend else ""} &\n')
                file.write(f'    pid=$!\n')
                file.write(f'    stop_condition $pid\n')
                file.write(f'  ;;\n')
            
            # equilibrium workers
            i = -1
            for i in range(nA + nB):
                if i < nA:
                    state = 'A'
                    j = i
                else:
                    state = 'B'
                    j = i - nA
                _case(i, f'equilibrium {state}{j}', True)
            
            # extension workers
            begin = i + 1
            for i in range(begin, begin + eA + eB):
                j = i - begin
                if j < eA:
                    state = 'A'
                else:
                    state = 'B'
                    j -= eA
                _case(i, f'extension {state}{j}', True)
            
            # shooting workers
            begin = i + 1
            for i in range(begin, begin + n):
                j = i - begin
                _case(i, f'shooting {j}', False)
            
            # last rank
            file.write(f'\n  {i + 1})  # trainer and manager\n')
            file.write(f'    pids=()\n')
            
            # trainer
            file.write('    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                       f'"{self.directory}" {i + 1} '
                       f'{cpus_per_task} {gpus_per_task} '
                       f'train trainer.log &\n')
            file.write(f'    pids+=($!)\n')
            
            # manager
            file.write('    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                       f'"{self.directory}" {(i + 1) % ntasks_per_node} '
                       f'{cpus_per_task} {gpus_per_task} '
                       f'manage {n} {nA} {nB} {eA} {eB} '
                       f'manager.log {nsteps} {nframes} &\n')
            file.write(f'    pids+=($!)\n')
            
            # monitor
            file.write(f'    stop_condition "${{pids[@]}}"\n')
            file.write(f'    ;;\n')
            
            # end cases with possible idle processes...
            file.write(f'\n  *)\n')
            file.write(f'    echo "[Worker $i] No task assigned."\n')
            file.write(f'    ;;\n')
            file.write(f'  esac\n\'\n')


class Processes:
    """Used by `Launcher` to manage the processes spawned by the
    `Launcher.run` method."""
    
    def __init__(self):
        self.__list = []
    
    def __len__(self):
        return len(self.__list)
    
    def __iter__(self):
        return iter(self.__list)
    
    def __getitem__(self, key):
        return self.__list[key]
    
    @property
    def list(self):
        return self.__list
    
    @property
    def alive(self):
        return ~self.closed
    
    @property
    def closed(self):
        result = np.repeat(False, len(self))
        for i, process in enumerate(self):
            if process._closed or not process.is_alive():
                result[i] = True
                if not process._closed:
                    print(f'[Worker {process.pid}] terminated '
                          f'with exit code {process.exitcode}')
                    process.join()
                    process.close()
        return result
    
    def launch(self, *args):
        process = ctx.Process(target=_run_task, args=args)
        self.__list.append(process)
        process.start()
        print(f'[Worker {process.pid}] args: '
              f'{" ".join([str(arg) for arg in args])}')
    
    def clean(self, timeout=20.):
        
        # graceful termination
        t0 = time.time()
        terminating = set()
        while time.time() - t0 < timeout and np.any(self.alive):
            for i in np.where(self.alive)[0]:
                if i in terminating:
                    continue
                terminating.add(i)
                try:
                    os.kill(self[i].pid, signal.SIGINT)
                except:
                    pass
        
        # forced termination
        for i in np.where(self.alive)[0]:
            try:
                self[i].kill()
                self[i].join(timeout=1.)
            except:
                pass
        
        # final report
        self.closed
        
        # pristine state
        self.__list = []
