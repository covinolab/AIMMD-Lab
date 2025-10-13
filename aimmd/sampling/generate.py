import os, sys, argparse, subprocess, time
from ..core import Params

class Launcher:
    
    def __init__(self, params, directory, n, nA, nB, eA=0, eB=0,
                 nsteps=np.inf, nframes=np.inf, walltime=np.inf):
        """
        directory: where simulations carried
        params: python file with params
        n: number of replicas dedicated to shooting simulations
        nA: number of replicas dedicated to free simulations around A
        nB: number of replicas dedicated to free simulations around B
        eA: number of replicas dedicated to extending transitions reaching A
        eA: number of replicas dedicated to extending transitions reaching B
        nsteps: default inf, maximum number of shooting simulations
        nframes: default inf, maximum number of simulated frames, has priority over nsteps
        walltime: default inf, maximum number of simulation time, has priority over nframes and nsteps
        
        All parameters for the run can be updated before (re)launching a simulation.
        """
        if type(params) is Params:
            self.params = params
        else:
            self.params = Params.update(params)
        self.directory = directory
        self.nsteps = nsteps
        self.n = n
        self.nA = nA
        self.nB = nB
        self.eA = eA
        self.eB = eB
        
        # create folder structure (keep existing data)
        os.system(f'mkdir {self.directory}')
        
        # save params to directory
        self.params.save(f'{self.directory}/params.dill')
        
        # folders for shooting simulations
        for worker_id in range(n):
            os.system(f'mkdir {self.directory}/shots{worker_id}')
        
        # folders for free simulations
        os.system(f'mkdir {self.directory}/equilibriumA')
        os.system(f'touch {self.directory}/equilibriumA/'
                  f'indicted_trajectories.log')
        os.system(f'mkdir {self.directory}/equilibriumB')
        os.system(f'touch {self.directory}/equilibriumB/'
                  f'indicted_trajectories.log')
        
        # folders for extension simulations
        if eA:
            file.write(f'mkdir {self.directory}/extendA')
            file.write(f'touch {self.directory}/extendA/'
                       f'indicted_trajectories.log')
        if eB:
            file.write(f'mkdir {self.directory}/extendB')
            file.write(f'touch {self.directory}/extendB/'
                       f'indicted_trajectories.log')
        
        # remove existing xtc and trajectory files
        for filename in os.listdir(self.directory):
            if filename.endswith('.xtc') or filename.endswith('.trr'):
                print(f'Removing {self.directory}/{filename}')
                os.system(f'rm {self.directory}/*{filename}*')
        
        # save initial paths
        filenames = [path.filename for path in self.params.initial_paths]
        for i, path in enumerate(self.params.initial_paths):
            filename = filenames[i]
            
            # avoid duplicates 
            if filename in filenames[:i]:
                filename = (f'{".".join(filenames[i].split(".")[:-1])}'
                            f'-2.{filenames[i].split(".")[-1]}')
                filenames[i] = filename
            
            # report
            print(f'Writing {self.directory}/{filename}')
            
            # actual save: get n_atoms
            n_atoms = len(path[0].positions)
            
            # create an empty Universe with n_atoms (no topology required)
            universe = mda.Universe.empty(n_atoms, trajectory=True)
            
            # write positions
            with mda.Writer(f'{self.directory}/{filename}', n_atoms) as writer:
                for frame in path:
                    universe.atoms.positions = frame.positions
                    writer.write(universe)
    
    def create_job(self, filename):
        """
        Returns a slurm script that can be launched by cluster.
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
        nodes = int(np.ceil((1 + self.n +  # trainer/worker, shooting
                             self.nA + self.nB +  # free A and B
                             self.eA + self.eB)  # extension A and B
                            / ntasks_per_node))
        
        # write job script
        with open(filename, 'w') as file:
            
            # slurm header
            file.write(f'#!/bin/bash -x\n')
            file.write(f'#SBATCH --job-name={self.params.name}\n')
            file.write(f'#SBATCH --nodes={nodes}\n')
            file.write(f'{self.params.slurm_header}\n\n')
            
            # python executable
            file.write(f'PYTHON={PYTHON}\n\n')
            
            # remove completed information and which to run
            file.write(f'rm {self.directory}/completed.flag\n')
            file.write(f'rm {self.directory}/*.run\n\n')
            
            # srun initialization
            file.write(f"srun --cpus-per-task={cpus_per_task} --cpu-bind=cores "
                       f"bash -c '\n\n")
            file.write(f'  # update task variables\n')
            file.write(f'  export i=\$SLURM_PROCID\n')
            file.write(f'  export li=\$SLURM_LOCALID\n')
            if gpu:
                file.write(f'  export CUDA_VISIBLE_DEVICES=$li\n')
            file.write(f'\n')
            
            # equilibrium workers
            i = -1
            for i in range(self.nA + self.nB):
                if i < self.nA:
                    state = 'A'
                    j = i
                else:
                    state = 'B'
                    j = i - self.nA
                file.write(f'  # worker {i} (equilibrium {state}{j})\n')
                file.write(f'  if [ "$i" -eq {i} ]; then\n')
                file.write(f'    bash worker.sh "{self.directory}/worker{i}.run" '
                           f'"{self.params.mdrun} -noappend" >> {self.directory}/worker{i}.log 2>&1\n')
                file.write(f'  fi\n\n')
            
            # extension workers
            begin = i + 1
            for i in range(begin, begin + self.eA + self.eB):
                j = i - begin
                if j < self.eA:
                    state = 'A'
                else:
                    state = 'B'
                    j -= self.eA
                file.write(f'  # worker {i} (extension {state}{j})\n')
                file.write(f'  if [ "$i" -eq {i} ]; then\n')
                file.write(f'    \$PYTHON worker.py\n')  # noappend option here
                file.write(f'  fi\n\n')
            
            # shooting workers
            begin = i + 1
            for i in range(begin, begin + self.n):
                j = i - begin
                file.write(f'  # worker {i} (shooting {j})\n')
                file.write(f'  if [ "$i" -eq {i} ]; then\n')
                file.write(f'    bash worker.sh "{self.directory}/worker{i}.run" '
                           f'"{self.params.mdrun}" >> {self.directory}/worker{i}.log 2>&1\n')
                file.write(f'  fi\n\n')
            
            # trainer and manager
            file.write(f'  # trainer and manager\n')
            file.write(f'  if [ "$i" -eq {i + 1} ]; then\n')
            file.write(f'    $PYTHON trainer.py "{self.directory}" >> '
                      f'{self.directory}/trainer.log 2>&1 &\n')
            file.write(f'    trainer_pid=$!\n\n')
            file.write(f'    $PYTHON manager.py "{self.directory}" {self.nsteps} '
                      f'{self.n} {self.nA} {self.nB} {self.eA} {self.eB} >> '
                      f'{self.directory}/manager.log 2>&1 &\n')
            file.write(f'    manager_pid=$!\n\n')
            
            # handle task termination
            file.write(f'    # handle task termination\n')
            file.write(f'    while kill -0 $trainer_pid 2>/dev/null'
                       f' || kill -0 $manager_pid 2>/dev/null; do\n')
            file.write(f'      wait -n\n')
            file.write(f'      rm {self.directory}/*.run\n')
            file.write( '      scancel ${SLURM_JOB_ID}\n')
            file.write(f'    done\n')
            file.write(f'  fi\n\'\n')
