import os, sys, argparse, subprocess, time
from ..core import Params

class Launcher:
    
    def __init__(self, params, directory, n, nA, nB, eA, eB,
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
        self.params = Params().update(params)
        self.directory = directory
        self.nsteps = nsteps
        self.n = n
        self.nA = nA
        self.nB = nB
        self.eA = eA
        self.eB = eB
        
        # create folder structure (keep existing data)
        os.system(f'mkdir {directory}')
        
        # save directory
        self.params.save(directory)
        
        # create folder structure (keep existing data)
        for worker_id in range(n):
            os.system(f'mkdir {directory}/shots{worker_id}')
        if nA:
            os.system(f'mkdir {directory}/equilibriumA')
            os.system(f'touch {directory}/equilibriumA/'
                      f'indicted_trajectories.log')
        if nB:
            os.system(f'mkdir {directory}/equilibriumB')
            os.system(f'touch {directory}/equilibriumB/'
                      f'indicted_trajectories.log')
        if eA:
            file.write(f'mkdir {directory}/extendA')
            file.write(f'touch {directory}/extendA/'
                       f'indicted_trajectories.log')
        if eB:
            file.write(f'mkdir {directory}/extendB')
            file.write(f'touch {directory}/extendB/'
                       f'indicted_trajectories.log')

        # remove completed information (needed?)
        file.write(f'rm {directory}/completed.flag')
        file.write(f'rm {directory}/*.run')
    
    def create_job(self, filename):
        """
        Returns a slurm script that can be launched by cluster.
        """
                
        # retrieve run information
        gpu = False
        ntasks_per_node = 1
        for fields in self.params.slurm_options.split():
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
            
            # header
            
            # srun command
            file.write(f"srun --cpus-per-task={cpus_per_task} --cpu-bind=cores "
                  f"bash -c '\n")
            file.write(f'  # update task variables')
            file.write(f'  export i=\$SLURM_PROCID')
            file.write(f'  export li=\$SLURM_LOCALID')
            if gpu:
                file.write(f'  export CUDA_VISIBLE_DEVICES=\$li')
            
            # equilibrium workers
            i = -1
            for i in range(nA + nB):
                if i < nA:
                    state = 'A'
                    j = i
                else:
                    state = 'B'
                    j = i - nA
                file.write(f'  # worker {i} (equilibrium {state}{j})')
                file.write(f'  if [ "\$i" -eq {i} ]; then')
                file.write(f'    bash worker.sh "{directory}/worker{i}.run" '
                           f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1',
                      fname, logfile)
                file.write('  fi\n')
            
            # extension workers
            begin = i + 1
            for i in range(begin, begin + eA + eB):
                j = i - begin
                if j < eA:
                    state = 'A'
                else:
                    state = 'B'
                    j -= eA
                file.write(f'  # worker {i} (extension {state}{j})')
                file.write(f'  if [ "\$i" -eq {i} ]; then')
                file.write(f'    bash worker.sh "{directory}/worker{i}.run" '
                           f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1',
                      fname, logfile)
                file.write('  fi\n')
            
            # shooting workers
            begin = i + 1
            for i in range(begin, begin + n):
                j = i - begin
                file.write(f'  # worker {i} (shooting {j})')
                file.write(f'  if [ "\$i" -eq {i} ]; then')
                file.write(f'    bash worker.sh "{directory}/worker{i}.run" '
                           f'"{mdrun}" >> {directory}/worker{i}.log 2>&1',
                      fname, logfile)
                file.write('  fi\n')
            
            # trainer and manager
            file.write(f'  # trainer and manager')
            file.write(f'  if [ "\$i" -eq {i + 1} ]; then')
            file.write(f'    {PYTHON} trainer.py "{directory}" "{params}" >> '
                      f'{directory}/trainer.log 2>&1 &')
            file.write(f'    trainer_pid=\$!')
            file.write('')
            file.write(f'    {PYTHON} manager.py "{directory}" {nsteps} '
                      f'{n} {nA} {nB} {eA} {eB} "{params}" >> '
                      f'{directory}/manager.log 2>&1 &')
            file.write(f'    manager_pid=\$!')
            file.write('')
            
            # handle task termination
            file.write(f'    # handle task termination')
            file.write(f'    while kill -0 \$trainer_pid 2>/dev/null'
                       f' || kill -0 \$manager_pid 2>/dev/null; do')
            file.write(f'      wait -n')
            file.write(f'      rm {directory}/*.run')
            file.write( '      scancel \${SLURM_JOB_ID}')
            file.write(f'    done')
            file.write('  fi')
    
    def run(self, dependency):
        """
        Run on workstation. Detaches with nohup so that you can exit python.
        dependency: wait for jobid to complete
        """
        pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AIMMD launcher')
    parser.add_argument('directory', type=str,
        help='project directory')
    parser.add_argument('nsteps', type=int,
        help='target 2-way-shooting excursions in the transition region')
    parser.add_argument('n', type=int,
        help='number of tasks dedicated to 2-way shooting')
    parser.add_argument('nA', type=int,
        help='number of tasks dedicated to equilibrium simulations in A')
    parser.add_argument('nB', type=int,
        help='number of tasks dedicated to equilibrium simulations in B')
    parser.add_argument('-eA', '--extend_A', type=int, default=0,
        help='number of tasks dedicated to extending transitin ending in A')
    parser.add_argument('-eB', '--extend_B', type=int, default=0,
        help='number of tasks dedicated to extending transitin ending in B')
    parser.add_argument('-p', '--params', type=str, default='params.py',
        help='aimmd run parameters (will override defaults)')
    parser.add_argument('-s', '--slurm', action='store_true')
    parser.add_argument('-d', '--dependency', type=str, default='',
        help='wait for job to terminate before starting')
    args = parser.parse_args()
    directory = args.directory
    nsteps = args.nsteps
    n = args.n
    nA = args.nA
    nB = args.nB
    eA = args.extend_A
    eB = args.extend_B
    params = args.params
    slurm = args.slurm
    dependency = args.dependency

    # TODO put generate backend here as a bonus
    
    PYTHON = sys.executable
    command = (f'{PYTHON} generate_backend.py '
               f'"{directory}" {nsteps} {n} {nA} {nB} {eA} {eB} '
               f'"{params}" {slurm} "{dependency}"')
