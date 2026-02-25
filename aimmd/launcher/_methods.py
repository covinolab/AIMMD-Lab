"""
...
"""

# external
import math
from abc import ABC
from math import inf
from numpy import bool_

# aimmd imports
from .._config import PYTHON, WORKER

# launcher methods functions
class LauncherMethods(ABC):
    
    def add(self, params, directories):
        from . import Launcher
        instance = Launcher(params, directories)
        self._params.extend(instance.params)
        self._directories.extend(instance.directories)
        if len(set(self._directories)) < len(self._directories):
            raise TypeError('All AIMMD run directories must be different')
        self._update()
    
    def pop(self, i):
        self._params.pop(i)
        self._directories.pop(i) 

    def create_job(self, filename, n=1, n1=0, n2=0,
                   reactive_region_mode='chain',
                   state1_mode='free', state2_mode='free',
                   nsteps=inf, nframes=inf, nrounds=inf, walltime=24*3600,
                   cpus_per_task=1, gpus_per_task=0,
                   ntasks_per_node=1, skip_binding=True):
        """
        Returns a slurm script in `filename` that can be launched by cluster.
        
        Parameters
        ----------
        filename: name of the job script to create
        n: default 1, number of replicas dedicated to shooting simulations
        n1: default 0, number of replicas dedicated to free simulations around
            the initial state (specified by self.params.states[0])
        n2: default 0, number of replicas dedicated to free simulations around
            the final state (specified by self.params.states[2])
        reactive_region_mode: default 'chain'; if 'chain': create a Markov
            chain (either TPS or RFPS algorithm); if 'sweep': shoot in order
            from the initial trajectories' configuration, repeat when done
            (useful for directly measuring committor values); if 'free': run
            free simulations instead
        state1_mode: default 'free'; if 'free': run standard free simulations
            in/around the initial state; if 'shoot': shoot in the state
            (use a single bin)
        state2_mode: default 'free'; if 'free': run standard free simulations
            in/around the final state; if 'shoot': shoot in the state
            (use a single bin)
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
        skip_binding: bool, default True
            If True, do not explicitly bind resources ('skip' option)
        ntasks_per_node: default 1, number of tasks per node
            may be overridden by params.slurm_header
        walltime: default 24*3600 s (24h) job simulation time
        """
        
        # retrieve run information: slurm header
        slurm_header = self.params[0].slurm_header + ''
        
        # retrieve run information: ntasks per node
        default_ntasks_per_node = int(ntasks_per_node)
        ntasks_per_node = None
        for fields in slurm_header.split():
            if 'ntasks-per-node' in fields:
                ntasks_per_node = int(fields.split('=')[-1])
        if not ntasks_per_node:
            ntasks_per_node = default_ntasks_per_node
            slurm_header += \
                f'\n#SBATCH --ntasks-per-node={ntasks_per_node}'
        
        # retrieve run information: cpus per task
        default_cpus_per_task = int(cpus_per_task)
        cpus_per_task = None
        for fields in slurm_header.split():
            if 'cpus-per-task' in fields:
                cpus_per_task = int(fields.split('=')[-1])
        if not cpus_per_task:
            cpus_per_task = default_cpus_per_task
            slurm_header += \
                f'\n#SBATCH --cpus-per-task={cpus_per_task}'
        
        # retrieve run information: gpus per task
        default_gpus_per_task = int(gpus_per_task)
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
        
        # update info
        self._update(n, n1, n2,
                     reactive_region_mode,
                     state1_mode, state2_mode,
                     nsteps, nframes, nrounds, inf,  # infinite walltime
                     cpus_per_task if not skip_binding else 'skip',
                     gpus_per_task if not skip_binding else 'skip',
                     ntasks_per_node)
        
        # number of nodes
        nodes = math.ceil(sum(self._num_processes) / self._ntasks_per_node)
        slurm_header += f'\n#SBATCH --nodes={nodes}'
        
        # time information
        walltime = int(walltime)
        hours = walltime // 3600
        minutes = (walltime - hours * 3600) // 60
        seconds = walltime - hours * 3600 - minutes * 60
        slurm_header += \
            f'\n#SBATCH --time={hours:02g}:{minutes:02g}:{seconds:02g}'
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write('#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params[0].name}\n')
            file.write(f'{slurm_header}\n\n')
                        
            # default names
            file.write('# default names\n')
            file.write(f'PYTHON="{PYTHON}"\n')
            file.write(f'WORKER="{WORKER}"\n\n')
            
            # enable job control
            file.write('# enable job control\n')
            file.write('set -m\n')

            # launch commands
            file.write('\n# workers')
            for i, (args, description) in enumerate(zip(*self._build())):
                file.write(f'# {description}\n')
                args = ' '.join([f'"{arg}"'
                                 if not isinstance(arg, (bool, bool_)) else
                                 '"True"' if arg else '""' for arg in args])
                file.write(f'srun --exclusive --ntasks=1 '
                           f'--cpus-per-task={cpus_per_task} '
                           f'--gpus-per-task={gpus_per_task} \\\n')
                file.write(f'  "${{PYTHON}}" "${{WORKER}}" {args} &\n')

            # wait until any process exits
            file.write('\n# wait until any process exits\n')
            file.write('wait -n\n')
            file.write('scancel ${SLURM_JOB_ID}\n')
            file.write('wait\n')
