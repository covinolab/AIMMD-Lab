import os
import sys
import time
import numpy as np
import signal
import multiprocessing
from math import ceil
from pathlib import Path
from .worker import Worker
from .resources import get_num_cpus, get_num_gpus
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
              termination_timeout, task, *args):
    try:
        Worker(params_file, directory,
               localid, cpus_per_task, gpus_per_task,
               termination_timeout).run(task, *args)
    except Exception as exception:
        print(f'[Error] {exception}')
        return 1
    return 0


class Processes:
    """Used by `Launcher` and `LaunchersCollection` to manage the
    processes spawned by the `run` method."""
    
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


class Launcher:

    processes = Processes()
    termination_signal = None
    
    def __init__(self, params, directory,
                 termination_timeout=20.):
        """
        directory: where simulations carried
        params: python file with params or Params
        
        All parameters for the run can be updated before (re)launching
        a simulation.

        params: either a string or an `aimmd.Params` instance
        directory: path relative to working directory where to run simulations
        termination_timeout: float, default 20
            Grace time for terminating processes, after which they are killed
        """
        if not isinstance(params, Params):
            params = Params.load(params)
        self.params = params
        self.directory = directory
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
            cpus_per_task='share', gpus_per_task='share'):
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
        cpus_per_task: str or int, defaut 'share'
            Number of CPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all
        gpus_per_task: str or int, default 'share'
            Number of GPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all

        Returns
        -------
        None
        """
        
        self.processes.clean()
        
        # total number of processes: simulators and trainer+manager
        num_processes = nA + nB + eA + eB + n + 1
        
        # determine number of CPUs per task
        if cpus_per_task == 'share':
            num_cpus_avail = get_num_cpus()
            cpus_per_task = max(1, num_cpus_avail // num_processes)
        
        # determine number of GPUs per task
        if gpus_per_task == 'share':
            num_gpus_avail = get_num_gpus()
            gpus_per_task = max(int(num_gpus_avail > 0),
                                num_gpus_avail // num_processes)
        
        # workers' termination timeout
        termination_timeout = max(0, self.termination_timeout - 1.)
        
        try:
            # simulators
            localid = 0
            for i in range(num_processes - 1):
                if i < num_processes - 1 - n:
                    noappend = True
                else:
                    noappend = False
                self.processes.launch(
                    self.params.path, self.directory,
                    localid, cpus_per_task, gpus_per_task,
                    termination_timeout,
                    'simulate', f'worker{i}',
                    f'worker{i}.log', noappend)
                localid += 1
            
            # trainer (sharing the same localid as manager)
            self.processes.launch(self.params.path, self.directory,
                                  localid, cpus_per_task, gpus_per_task,
                                  termination_timeout,
                                  'train', 'trainer.log')
            
            # manager (sharing the same localid as trainer)
            self.processes.launch(self.params.path, self.directory,
                                  localid, cpus_per_task, gpus_per_task,
                                  termination_timeout,
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
                   ntasks_per_node=1, skip_binding=False,
                   walltime=24*3600):
        """
        Returns a slurm script in `filename` that can be launched by cluster.
        
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
        nframes: default inf, maximum number of simulated frames
        cpus_per_task: int, defaut 1
            Number of CPUs to allocate per task
            if --cpus-per-task is present in params.slurm_header, the
            corresponding value overrides this input argument
        gpus_per_task: int, default 0
            Number of GPUs to allocate per task
            if --gpus-per-task or --gres=gpu is present in params.slurm_header,
            the corresponding value overrides this input argument
        skip_binding: bool, default False
            If True, do not explicitly bind resources ('skip' option)
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
                    f'\n#SBATCH --gres=gpu:{gpus_per_task * ntasks_per_node}'
        
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
        slurm_header += \
            f'\n#SBATCH --time={hours:02g}:{minutes:02g}:{seconds:02g}'

        # workers' cpus_per_task and gpus_per_task
        if skip_binding:
            cpus_per_task = 'skip'
            gpus_per_task = 'skip'
        
        # workers' termination timeout
        termination_timeout = max(0, self.termination_timeout - 1.)
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params.name}\n')
            file.write(f'{slurm_header}\n\n')
            file.write(f"rm -f {self.directory}/.terminate\n\n")
            
            # srun call
            file.write(f'# srun call\n')
            file.write(f"srun --cpus-per-task={cpus_per_task} "
                            f"--cpu-bind=cores bash -c '\n\n")
            
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
            file.write(f'  case $SLURM_PROCID in\n')
            def _case(i, description, noappend=False):
                file.write(f'\n  {i})  # worker {i} ({description})\n')
                file.write(
                    f'    "${{PYTHON}}" "${{WORKER}}" "${{PARAMS}}" '
                    f'"{self.directory}" {i % ntasks_per_node} '
                    f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
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
                _case(i, f'free {state}{j}', True)
            
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
            file.write(
                '    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                f'"{self.directory}" {(i + 1) % ntasks_per_node} '
                f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
                f'train trainer.log &\n')
            file.write(f'    pids+=($!)\n')
            
            # manager
            file.write(
                '    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                f'"{self.directory}" {(i + 1) % ntasks_per_node} '
                f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
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


class LaunchersCollection(Launcher):
    """Many launchers together, so that you can "run" them
    together both on a workstation or on a slurm job script."""
    
    def __init__(self, *launchers):
        
        # check input
        if not len(launchers):
            raise TypeError('Need at least one "aimmd.sampling.Launcher"')
        for i, launcher in enumerate(launchers):
            if type(launcher) is not Launcher:
                raise TypeError(f'The {i + 1}-th input argument {launcher}'
                                f' is not an aimmd.sampling.Launcher')
        
        # assign
        self.__launchers = launchers
        
        # termination timeout from the first launcher
        self.termination_timeout = self[0].termination_timeout
        
        # same handlers as original launchers
        signal.signal(signal.SIGTERM, self.terminate_handler)
        signal.signal(signal.SIGINT, self.terminate_handler)
    
    def __len__(self):
        return len(self.__launchers)
    
    def __iter__(self):
        return iter(self.__launchers)
    
    def __getitem__(self, key):
        return self.__launchers[key]
    
    @property
    def launchers(self):
        return self.__launchers
    
    @launchers.setter
    def launchers(self, launchers):
        self.__init__(*launchers)

    @property
    def directory(self):
        """Leading directory, just for placing .terminate in SLURM job"""
        return self[0].directory
    
    def _process_input(self, name, value):
        """make it a list as long as self"""
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            if len(value) != len(self):
                raise TypeError(f"{name}'s length must be the same as "
                                f"number of launchers {len(self)}")
        else:
            value = [value] * len(self)
        return value
    
    def run(self, n, nA, nB, eA=0, eB=0,
            nsteps=inf, nframes=inf, walltime=inf,
            cpus_per_task=0, gpus_per_task=0):
        """
        Launch the simulation locally, spawning multiple processes.
        
        Parameters
        ----------
        n: list, for each launcher in instance: number of replicas
           dedicated to shooting simulations (creates folders if not existing)
        nA: list, for each launcher in instance: number of replicas
             dedicated to free simulations around A
        nB: list, for each launcher in instance: number of replicas
            dedicated to free simulations around B
        eA: list, for each launcher in instance: number of replicas
            dedicated to extending transitions reaching A
        eB: list, for each launcher in instance: number of replicas
            dedicated to extending transitions reaching B
        nsteps: default inf, maximum number of shooting simulations
                if number: applies to each launcher in instance
                if list with as many elements as launchers: set different
                nsteps for each launcher
        nframes: default inf, maximum number of simulated frames,
                 has priority over nsteps
                 if number: applies to each launcher in instance
                 if list with as many elements as launchers: set different
                 nframes for each launcher
        walltime: default inf, maximum number of simulation time,
                  has priority over nframes and nsteps
        cpus_per_task: str or int, defaut 'share'
            Number of CPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all
        gpus_per_task: str or int, default 'share'
            Number of GPUs to allocate per task
            if 'share': equally distribute available resources among workers
            if 'all': each worker takes them all
        
        Returns
        -------
        None
        """
        
        self.processes.clean()
        
        # process input
        n = self._process_input('n', n)
        nA = self._process_input('nA', nA)
        nB = self._process_input('nB', nB)
        eA = self._process_input('eA', eA)
        eB = self._process_input('eB', eB)
        nsteps = self._process_input('nsteps', nsteps)
        nframes = self._process_input('nframes', nframes)
        
        # total number of processes: simulators and trainer+manager
        num_processes = \
            sum(nA) + sum(nB) + sum(eA) + sum(eB) + sum(n) + len(self)
        
        # determine number of CPUs per task
        if cpus_per_task == 'share':
            num_cpus_avail = get_num_cpus()
            cpus_per_task = max(1, num_cpus_avail // num_processes)
        
        # determine number of GPUs per task
        if gpus_per_task == 'share':
            num_gpus_avail = get_num_gpus()
            gpus_per_task = max(int(num_gpus_avail > 0),
                                num_gpus_avail // num_processes)
        
        # workers' termination timeout
        termination_timeout = max(0, self.termination_timeout - 1.)
        
        try:
            # in each iteration of this for loop,
            # we fall back to the original launcher.run
            localid = 0
            for launcher, n, nA, nB, eA, eB, nsteps, nframes in zip(
                self,     n, nA, nB, eA, eB, nsteps, nframes):
                num_processes = nA + nB + eA + eB + n + 1
                
                # simulators
                for i in range(num_processes - 1):
                    if i < num_processes - 1 - n:
                        noappend = True
                    else:
                        noappend = False
                    self.processes.launch(
                        launcher.params.path, launcher.directory,
                        localid, cpus_per_task, gpus_per_task,
                        termination_timeout,
                        'simulate', f'worker{i}',
                        f'worker{i}.log', noappend)
                    localid += 1
                
                # trainer (sharing the same localid as manager)
                self.processes.launch(launcher.params.path, launcher.directory,
                                      localid, cpus_per_task, gpus_per_task,
                                      termination_timeout,
                                      'train', 'trainer.log')
                
                # manager (sharing the same localid as trainer)
                self.processes.launch(launcher.params.path, launcher.directory,
                                      localid, cpus_per_task, gpus_per_task,
                                      termination_timeout,
                                      'manage', n, nA, nB, eA, eB,
                                      'manager.log', nsteps, nframes)
                localid += 1
            
            # wait for completion with walltime
            t0 = time.time()
            while time.time() - t0 < walltime and np.all(self.processes.alive):
                continue
        
        # safe termination
        finally:
            self.termination_signal = 2  # KeyboardInterrupt
            self.terminate_operations()
    
    def create_job(self, filename, n, nA, nB, eA=0, eB=0,
                   nsteps=inf, nframes=inf,
                   cpus_per_task=1, gpus_per_task=0,
                   ntasks_per_node=1, skip_binding=False,
                   walltime=24*3600):
        
        """
        Returns a slurm script in `filename` that can be launched by cluster.
        
        Parameters
        ----------
        filename: name of the job script to create
        n: list, for each launcher in instance: number of replicas
           dedicated to shooting simulations (creates folders if not existing)
        nA: list, for each launcher in instance: number of replicas
             dedicated to free simulations around A
        nB: list, for each launcher in instance: number of replicas
            dedicated to free simulations around B
        eA: list, for each launcher in instance: number of replicas
            dedicated to extending transitions reaching A
        eB: list, for each launcher in instance: number of replicas
            dedicated to extending transitions reaching B
        nsteps: default inf, maximum number of shooting simulations
                if number: applies to each launcher in instance
                if list with as many elements as launchers: set different
                nsteps for each launcher
        nframes: default inf, maximum number of simulated frames,
                 has priority over nsteps
                 if number: applies to each launcher in instance
                 if list with as many elements as launchers: set different
                 nframes for each launcher
        nsteps: default inf, maximum number of shooting simulations
        nframes: default inf, maximum number of simulated frames,
        cpus_per_task: int, defaut 1
            Number of CPUs to allocate per task
            if --cpus-per-task is present in params.slurm_header, the
            corresponding value overrides this input argument
        gpus_per_task: int, default 0
            Number of GPUs to allocate per task
            if --gpus-per-task or --gres=gpu is present in params.slurm_header,
            the corresponding value overrides this input argument
        skip_binding: bool, default False
            If True, do not explicitly bind resources ('skip' option)
        ntasks_per_node: default 1, number of tasks per node
            may be overridden by params.slurm_header
        walltime: default 24*3600 s (24h) job simulation time
        """
        
        # process input
        n = self._process_input('n', n)
        nA = self._process_input('nA', nA)
        nB = self._process_input('nB', nB)
        eA = self._process_input('eA', eA)
        eB = self._process_input('eB', eB)
        nsteps = self._process_input('nsteps', nsteps)
        nframes = self._process_input('nframes', nframes)
        
        # retrieve run information: slurm header
        slurm_header = self[0].params.slurm_header + ''
        
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
                    f'\n#SBATCH --gres=gpu:{gpus_per_task * ntasks_per_node}'
        
        # number of nodes
        nodes = ceil((len(self) + sum(n) +  # trainer/worker, shooting
                      sum(nA) + sum(nB) +  # free A and B
                      sum(eA) + sum(eB))  # extension A and B
                      / ntasks_per_node)
        slurm_header += f'\n#SBATCH --nodes={nodes}'
        
        # time information
        walltime = int(walltime)
        hours = walltime // 3600
        minutes = (walltime - hours * 3600) // 60
        seconds = walltime - hours * 3600 - minutes * 60
        slurm_header += \
            f'\n#SBATCH --time={hours:02g}:{minutes:02g}:{seconds:02g}'
        
        # workers' cpus_per_task and gpus_per_task
        if skip_binding:
            cpus_per_task = 'skip'
            gpus_per_task = 'skip'
        
        # workers' termination timeout
        termination_timeout = max(0, self.termination_timeout - 1.)
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self[0].params.name}\n')
            file.write(f'{slurm_header}\n\n')
            file.write(f"rm -f {self.directory}/.terminate\n\n")
            
            # srun call
            file.write(f'# srun call\n')
            file.write(f"srun --cpus-per-task={cpus_per_task} "
                            f"--cpu-bind=cores bash -c '\n\n")
            
            # default names
            file.write(f'  # default names\n')
            file.write(f'  PYTHON="{PYTHON}"\n')
            file.write(f'  WORKER="{WORKER}"\n\n')
            
            # stop condition
            file.write(f'  # setup stop condition\n')
            file.write(f"  START_TIME=$(date +%s)\n")
            file.write(f"  WALLTIME={walltime}\n\n")
            file.write(f"  {self.job_stop_condition}\n\n")
            
            # cases
            file.write(f'  # srun rank by rank\n')
            file.write(f'  case $SLURM_PROCID in\n')
            def _case(localid, i, launcher, description, noappend=False):
                file.write(f'\n  {localid})  '
                           f'# {launcher.directory} worker {i} '
                           f'({description})\n')
                file.write(f'    PARAMS="{launcher.params.path}"\n')
                file.write(
                    f'    "${{PYTHON}}" "${{WORKER}}" "${{PARAMS}}" '
                    f'"{launcher.directory}" {localid % ntasks_per_node} '
                    f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
                    f'simulate worker{i} worker{i}.log'
                    f'{" noappend" if noappend else ""} &\n')
                file.write(f'    pid=$!\n')
                file.write(f'    stop_condition $pid\n')
                file.write(f'  ;;\n')

            # in each iteration of this for loop,
            # we fall back to the original launcher.create_job
            localid = 0
            for launcher, n, nA, nB, eA, eB, nsteps, nframes in zip(
                self,     n, nA, nB, eA, eB, nsteps, nframes):
                
                # equilibrium workers
                i = -1
                for i in range(nA + nB):
                    if i < nA:
                        state = 'A'
                        j = i
                    else:
                        state = 'B'
                        j = i - nA
                    _case(localid, i, launcher, f'free {state}{j}', True)
                    localid += 1
                
                # extension workers
                begin = i + 1
                for i in range(begin, begin + eA + eB):
                    j = i - begin
                    if j < eA:
                        state = 'A'
                    else:
                        state = 'B'
                        j -= eA
                    _case(localid, i, launcher, f'extension {state}{j}', True)
                    localid += 1
                
                # shooting workers
                begin = i + 1
                for i in range(begin, begin + n):
                    j = i - begin
                    _case(localid, i, launcher, f'shooting {j}', False)
                    localid += 1
                
                # last rank
                file.write(f'\n  {localid})  '
                           f'# {launcher.directory} trainer and manager\n')
                file.write(f'    pids=()\n')
                file.write(f'    PARAMS="{launcher.params.path}"\n')
                
                # trainer
                file.write(
                    '    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                    f'"{launcher.directory}" {localid % ntasks_per_node} '
                    f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
                    f'train trainer.log &\n')
                file.write(f'    pids+=($!)\n')
                
                # manager
                file.write(
                    '    "${PYTHON}" "${WORKER}" "${PARAMS}" '
                    f'"{launcher.directory}" {localid % ntasks_per_node} '
                    f'{cpus_per_task} {gpus_per_task} {termination_timeout} '
                    f'manage {n} {nA} {nB} {eA} {eB} '
                    f'manager.log {nsteps} {nframes} &\n')
                file.write(f'    pids+=($!)\n')
                
                # monitor
                file.write(f'    stop_condition "${{pids[@]}}"\n')
                file.write(f'    ;;\n')
                
                # update localid
                localid += 1
            
            # end cases with possible idle processes...
            file.write(f'\n  *)\n')
            file.write(f'    echo "[Worker $i] No task assigned."\n')
            file.write(f'    ;;\n')
            file.write(f'  esac\n\'\n')
