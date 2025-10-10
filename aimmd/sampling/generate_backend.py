import os, sys
import importlib
import numpy as np
from time import sleep
from textwrap import wrap
from datetime import datetime

###############################################################################
### INITIALIZATION ############################################################
###############################################################################

# parse input
directory = sys.argv[1]
nsteps = int(sys.argv[2])
n = int(sys.argv[3])
nA = int(sys.argv[4])
nB = int(sys.argv[5])
eA = int(sys.argv[6])
eB = int(sys.argv[7])
params = str(sys.argv[8])
slurm = str(sys.argv[9]) == 'True'
dependency = str(sys.argv[10])
logfile = f'{directory}/manager.log'

# process attributes and params
PYTHON = sys.executable
aimmd_run_params = getattr(importlib.import_module(
    params.split('.py')[0]), 'aimmd_run_params')
mdrun = aimmd_run_params['mdrun']

# process dependency
if dependency:
    if slurm and dependency.isdigit():
        dependency = f' -d afterany:{dependency}'
    elif dependency.isdigit():  # PID
        dependency = int(dependency)
        while True:
            try:
                os.kill(dependency, 0)  # Check if the process is running
                sleep(1)
            except OSError:
                break  # Process has finished
    else:  # folder
        flag = f'{dependency}/completed.flag'
        write(f'waiting for {flag}')
        while not os.path.exists(flag):
            sleep(1.)

# function: write
def write(text, *paths, wrap_text=False):
    if wrap_text:
        text = "\n".join(wrap(text, 80,
            break_long_words=False, replace_whitespace=False))
    #text = text.replace("'", "\\'") # replace single quotes with escape char
    text = text.replace('"', '\\"') # replace double quotes with escape char
    os.system(f'echo "{text}"')
    for path in paths:
        os.system(f'echo "{text}" >> {path}')

# function: job submission
if slurm:
    def submit_job(command):
        global jobids
        ID = os.popen(command).read()
        jobids.append(ID.split()[-1])
else:
    def submit_job(command):
        global jobids
        write(f'{command}\n', logfile)
        ID = str(int(os.popen(f'nohup {command} & echo $!').read()))
        jobids.append(ID.split()[-1])

# load and copy params
t0 = str(datetime.now())[:19]
write(f'''
######################## AIMMD RUN {t0} ########################
''', logfile)

write(f'''
AIMMD run arguments ___________________________________________________________

directory = {directory}
nsteps = {nsteps}
n = {n}
nA = {nA}
nB = {nB}
params = {params}
slurm = {slurm}
dependency = {dependency}''', logfile)

write(f'''
AIMMD run parameters "{params}" {"_" * (56 - len(params))}

{open(params).read()}
''', logfile)

###############################################################################
### JOB SUBMISSION ############################################################
###############################################################################

if slurm:  # on cluster
    slurm_options = aimmd_run_params['slurm_options']
    
    # extract run info from options
    gpu = False
    ntasks_per_node = 1
    for arguments in slurm_options.split():
        if ('gpu' in arguments and
            '=0' not in arguments and
            ':0' not in arguments):
            gpu = True
        if 'ntasks-per-node' in arguments:
            ntasks_per_node = int(arguments.split('=')[-1])
        if 'cpus-per-task' in arguments:
            cpus_per_task = int(arguments.split('=')[-1])
    nodes = int(np.ceil((n + nA + nB + eA + eB + 1) / ntasks_per_node))
    
    # fill slurm submission arguments
    fname = f'{directory}_job.sh'
    if os.path.exists(fname):
        os.remove(fname)
    write(f'SLURM job "{fname}" {"_" * (67 - len(fname))}', logfile)
    write(f'''#!/bin/bash -x
#SBATCH --job-name=AIMMD_{directory}
#SBATCH --nodes={nodes}
{slurm_options}
''', fname, logfile)
    
    # directories handling
    for worker_id in range(n):
        write(f'mkdir {directory}/shots{worker_id}', fname, logfile)
    if nA:
        write(f'mkdir {directory}/equilibriumA', fname, logfile)
        write(f'touch {directory}/equilibriumA/'
              f'indicted_trajectories.log', fname, logfile)
    if nB:
        write(f'mkdir {directory}/equilibriumB', fname, logfile)
        write(f'touch {directory}/equilibriumB/'
              f'indicted_trajectories.log', fname, logfile)
    if eA:
        write(f'mkdir {directory}/extendA', fname, logfile)
        write(f'touch {directory}/extendA/'
              f'indicted_trajectories.log', fname, logfile)
    if eB:
        write(f'mkdir {directory}/extendB', fname, logfile)
        write(f'touch {directory}/extendB/'
              f'indicted_trajectories.log', fname, logfile)
    write(f'rm {directory}/completed.flag', fname, logfile)
    write(f'rm {directory}/*.run', fname, logfile)
    write('', fname, logfile)
    
    # srun command
    write(f"srun --cpus-per-task={cpus_per_task} --cpu-bind=cores "
          f"bash -c '\n", fname, logfile)
    write(f'  # update task variables', fname, logfile)
    write(f'  export i=\$SLURM_PROCID', fname, logfile)
    write(f'  export li=\$SLURM_LOCALID', fname, logfile)
    if gpu:
        write(f'  export CUDA_VISIBLE_DEVICES=\$li', fname, logfile)
    write('', fname, logfile)
    
    # equilibrium workers
    i = -1
    for i in range(nA + nB):
        if i < nA:
            state = 'A'
            j = i
        else:
            state = 'B'
            j = i - nA
        write(f'  # worker {i} (equilibrium {state}{j})', fname, logfile)
        write(f'  if [ "\$i" -eq {i} ]; then', fname, logfile)
        write(f'    bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1',
              fname, logfile)
        write('  fi\n', fname, logfile)
    
    # extension workers
    begin = i + 1
    for i in range(begin, begin + eA + eB):
        j = i - begin
        if j < eA:
            state = 'A'
        else:
            state = 'B'
            j -= eA
        write(f'  # worker {i} (extension {state}{j})', fname, logfile)
        write(f'  if [ "\$i" -eq {i} ]; then', fname, logfile)
        write(f'    bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1',
              fname, logfile)
        write('  fi\n', fname, logfile)
    
    # shooting workers
    begin = i + 1
    for i in range(begin, begin + n):
        j = i - begin
        write(f'  # worker {i} (shooting {j})', fname, logfile)
        write(f'  if [ "\$i" -eq {i} ]; then', fname, logfile)
        write(f'    bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun}" >> {directory}/worker{i}.log 2>&1',
              fname, logfile)
        write('  fi\n', fname, logfile)
    
    # trainer and manager
    write(f'  # trainer and manager', fname, logfile)
    write(f'  if [ "\$i" -eq {i + 1} ]; then', fname, logfile)
    write(f'    {PYTHON} trainer.py "{directory}" "{params}" >> '
              f'{directory}/trainer.log 2>&1 &', fname, logfile)
    write(f'    trainer_pid=\$!', fname, logfile)
    write('', fname, logfile)
    write(f'    {PYTHON} manager.py "{directory}" {nsteps} '
              f'{n} {nA} {nB} {eA} {eB} "{params}" >> '
              f'{directory}/manager.log 2>&1 &', fname, logfile)
    write(f'    manager_pid=\$!', fname, logfile)
    write('', fname, logfile)
    
    # handle task termination
    write(f'    # handle task termination', fname, logfile)
    write(f'    while kill -0 \$trainer_pid 2>/dev/null'
               f' || kill -0 \$manager_pid 2>/dev/null; do', fname, logfile)
    write(f'      wait -n', fname, logfile)
    write(f'      rm {directory}/*.run', fname, logfile)
    write( '      scancel \${SLURM_JOB_ID}', fname, logfile)
    write(f'    done', fname, logfile)
    write('  fi', fname, logfile)
    
    # end
    write("'\n", fname, logfile)
    write(f'{"#"*80}', logfile)
    write('DONE! Please submit:')
    write(f'sbatch {fname}{dependency}')
    write(f'{"#"*80}')

else:  # on workstation
    jobids = []
    write(f'Shell commands {"_" * 66}\n', logfile)

    def execute(command):
        write(command, logfile)
        os.system(command)
    
    # directories handling
    for worker_id in range(n):
        execute(f'mkdir {directory}/shots{worker_id}')
    if nA:
        execute(f'mkdir {directory}/equilibriumA')
        execute(f'touch {directory}/equilibriumA/indicted_trajectories.log')
    if nB:
        execute(f'mkdir {directory}/equilibriumB')
        execute(f'touch {directory}/equilibriumB/indicted_trajectories.log')
    if eA:
        execute(f'mkdir {directory}/extendA')
        execute(f'touch {directory}/extendA/indicted_trajectories.log')
    if eB:
        execute(f'mkdir {directory}/extendB')
        execute(f'touch {directory}/extendB/indicted_trajectories.log')
    execute(f'rm {directory}/completed.flag')
    execute(f'rm {directory}/*.run')
    write('', logfile)
    
    # equilibrium workers
    i = -1
    for i in range(nA + nB):
        if i < nA:
            state = 'A'
            j = i
        else:
            state = 'B'
            j = i - nA
        write(f'# worker {i} (equilibrium {state}{j})', logfile)
        submit_job(f'bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1')
    
    # extension workers
    begin = i + 1
    for i in range(begin, begin + eA + eB):
        j = i - begin
        if j < nA:
            state = 'A'
        else:
            state = 'B'
            j -= eA
        write(f'# worker {i} (extension {state}{j})', logfile)
        submit_job(f'bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun} -noappend" >> {directory}/worker{i}.log 2>&1')
    

    # shooting workers
    begin = i + 1
    for i in range(begin, begin + n):
        j = i - begin
        write(f'# worker {i} (shooting {j})', logfile)
        submit_job(f'bash worker.sh "{directory}/worker{i}.run" '
                   f'"{mdrun}" >> {directory}/worker{i}.log 2>&1')
    
    # trainer
    write(f'# trainer', logfile)
    submit_job(f'{PYTHON} trainer.py "{directory}" "{params}" >> '
               f'{directory}/trainer.log 2>&1')
    
    # manager
    write(f'# manager', logfile)
    submit_job(f'{PYTHON} manager.py "{directory}" {nsteps} '
               f'{n} {nA} {nB} {eA} {eB} "{params}" >> '
               f'{directory}/manager.log 2>&1')
    
    # write jobids
    write(f'{"#"*80}', logfile)
    write(f'PIDS {" ".join(jobids)}', logfile, wrap_text=True)
    write(f'{"#"*80}', logfile)
    
    # handle task termination
    command= f'''while true; do
  if ! kill -0 {jobids[-1]} 2>/dev/null; then
    break
  fi
  if ! kill -0 {jobids[-2]} 2>/dev/null; then
    break
  fi
  sleep 1
done

rm {directory}/*.run'''
    for jobid in jobids:
        command += f'\nkill {jobid}'
    os.system(command)
