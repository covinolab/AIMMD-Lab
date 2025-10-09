"""
This script tests virtually all function of AIMMD sampling/analysis
on a molecular model defined in the "retinal" folder.
It runs with GROMACS in about five minutes.
However, there is still the polishing left to do, and compatibility
on different machines is not guaranteed.
"""

import os, sys
from time import sleep
from time import time as current_time

FOLDER = 'retinal'
PYTHON = 'python3'  # will call generate.py

# Attention! Gromacs options come from "retinal/params.py"

print(f'Changing working directory to {FOLDER}')
os.chdir(FOLDER)

seconds = 12  # of equilibration

print('Copying scripts from "src" folder (workaround)')
os.system(f'cp ../../aimmd/core/generate.py .')
os.system(f'cp ../../aimmd/core/generate_backend.py .')
os.system(f'cp ../../aimmd/core/utils.py .')
os.system(f'cp ../../aimmd/core/pathensemble.py .')
os.system(f'cp ../../aimmd/core/manager.py .')
os.system(f'cp ../../aimmd/core/trainer.py .')
os.system(f'cp ../../aimmd/core/worker.sh .')

print('Cleaning directories')
os.system('rm -rf equilibrium; mkdir equilibrium')
os.system('rm -rf run1; mkdir run1')

print('\nImporting libraries')
from params import *

states_function = aimmd_run_params['states_function']
descriptors_function = aimmd_run_params['descriptors_function']
values_function = aimmd_run_params['values_function']
topology = aimmd_run_params['topology']
mdrun_parameters = aimmd_run_params['mdrun_parameters']
random_velocities = aimmd_run_params['random_velocities']
grompp = aimmd_run_params['grompp']
mdrun = aimmd_run_params['mdrun']
trajectory_extension = aimmd_run_params['trajectory_extension']
os.system(f'cp initial{trajectory_extension} run1')

# better visualization
np.set_printoptions(threshold=20)

print(f'\nEquilibrating system for {seconds} seconds')
code = os.system(f'{grompp} -f {mdrun_parameters} '
          f'-c {topology} -r {topology} -o equilibrium/run.tpr')
if code:
    raise RuntimeError('Error during tpr generation')
code = os.system(f'{mdrun} -deffnm equilibrium/run -maxh {seconds/3600}')
if code:
    raise RuntimeError('Error during simulation')

print('\nInitializing equilibrium path ensemble', end='')
equilibrium = PathEnsemble(
    topology=f'../{topology}', directory='equilibrium',
    states_function=states_function,
    descriptors_function=descriptors_function,
    values_function=values_function)
print(':', equilibrium)

print('\nLoading trajectory from "equilibrium" folder')
nframes, time = equilibrium.append(
    f'run{trajectory_extension}', verbose=True)
print(f'Loaded {nframes} frames, time: {time}')
if not nframes:
    raise RuntimeError('Could not load the trajectory file')
print('Obtained', equilibrium)

print('\nSplitting path ensemble')
equilibrium.split()
npaths = len(equilibrium)
print('Obtained', equilibrium)

print('\nSaving path ensemble')
equilibrium.save('equilibrium/pathensemble.h5', directory='.')

print('(Re)loading path ensemble')
# need to reassign functions (no saved in memory)
equilibrium = PathEnsemble(
    states_function=states_function,
    descriptors_function=descriptors_function,
    values_function=values_function)
equilibrium.load('equilibrium/pathensemble.h5')

# report and check that everything goes well
inA_indices = np.where((equilibrium.initial_states == 'R') *
                       (equilibrium.internal_states == 'A') *
                       (equilibrium.final_states == 'R'))[0]
inB_indices = np.where((equilibrium.initial_states == 'R') *
                       (equilibrium.internal_states == 'B') *
                       (equilibrium.final_states == 'R'))[0]
ARA_indices = np.where((equilibrium.initial_states == 'A') *
                       (equilibrium.internal_states == 'R') *
                       (equilibrium.final_states == 'A'))[0]
BRB_indices = np.where((equilibrium.initial_states == 'B') *
                       (equilibrium.internal_states == 'R') *
                       (equilibrium.final_states == 'B'))[0]
TPs_indices = np.where(equilibrium.are_transitions)[0]

print('\nStatistics')
print('Number of paths  ', len(equilibrium))
print('Number of frames ', equilibrium.nframes)
print('Lengths          ', equilibrium.lengths)
print('Initial states   ', equilibrium.initial_states)
print('Shooting states  ', equilibrium.shooting_states)
print('Internal states  ', equilibrium.internal_states)
print('Final states     ', equilibrium.final_states)
print('ARA excursions   ', ARA_indices)
print('In A segments    ', inA_indices)
print('BRB excursions   ', BRB_indices)
print('In B segments    ', inB_indices)
print('Transitions      ', TPs_indices)

print('\nPlotting time series of selected paths (descriptors/values space)')
npaths = 5

if len(TPs_indices) and len(equilibrium) > TPs_indices[0] + 2:
    i = TPs_indices[-1] - 2
else:
    i = max(0, len(equilibrium) - 3)

old_label = ''
for i in range(i, min(len(equilibrium), i + npaths)):
    descriptors = cv(equilibrium.path(i))
    times = equilibrium.times(i)[0]
    if i in ARA_indices:
        color = 'tomato'
        label = f'ARA'
    elif i in inA_indices:
        color = 'firebrick'
        label = f'inA'
    elif i in TPs_indices:
        color = 'black'
        label = f'TP'
    elif i in BRB_indices:
        color = 'dodgerblue'
        label = f'BRB'
    elif i in inB_indices:
        color = 'blue'
        label = f'inB'
    if old_label == label:
        raise RuntimeError(f'Paths {old_label} ({i}) and {label} ({i}) '
                           f'should be of different kind.')
    old_label = label
    plt.figure(1)
    plt.plot(times[0], descriptors[0], 'o', color='black', zorder=100)
    plt.plot(times, descriptors, color=color, label=label)

plt.figure(1)
plt.grid()
plt.xlabel('Time [dt]')
plt.ylabel('CV')
plt.legend()
plt.savefig('equilibrium_partial_time_series.pdf')

print('\nReweighting (the weights should be constants)')
wA, *_ = equilibrium.reweight('A')
wB, *_ = equilibrium.reweight('B')
print('Weights:', wA + wB)

print('\nProjecting the "from A" and "from B" equilibrium ensembles')
bins = np.linspace(np.min(equilibrium.frame_descriptors) - 1e-15,
                   np.max(equilibrium.frame_descriptors) + 1e-15, 11)
equilibrium.weights = wA
rhoA = equilibrium.project(bins, f=lambda descriptors:descriptors[:, 0])
equilibrium.weights = wB
rhoB = equilibrium.project(bins, f=lambda descriptors:descriptors[:, 0])
print(f'Bins    {bins}  (CV\'s space)')
print(f'Centers {(bins[:-1] + bins[1:]) / 2}')
print(f'From A  {rhoA}')
print(f'From B  {rhoB}')

print('\nInitializing AIMMD: one simulator at a time')
print('Attention! On a HPC cluster you can run many simulations '
      'in parallel, with the --slurm option of "generate.py", '
      'without the risk of freezing your machine...')

for i, command in enumerate([
    f'{PYTHON} generate.py run1 100 0 1 0 -p params.py',
    f'{PYTHON} generate.py run1 100 0 0 1 -p params.py',
    f'{PYTHON} generate.py run1 100 1 0 0 -p params.py']):
    
    process = subprocess.Popen(command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.stdout.read().decode().strip()
    error = process.stderr.read().decode().strip()
    if error:
        raise RuntimeError(error)
    
    pids = [int(pid) for pid in 
            stdout.split('PIDS')[1].split('\n')[0].split()]
    print(f'\nLaunched {command}, process PIDs: {pids}')
    if i < 2:
        print(f'Running for at most one minute...')
    else:
        print(f'Running for at most two minutes...')
    print(f'(You can inspect {FOLDER}/run1\'s log files for details.)')
    t0 = current_time()
    while current_time() - t0 < (60 if i < 2 else 120):
        # Check if process is still running and if simulation output exists
        if np.any([not psutil.pid_exists(pid) for pid in pids]) \
        and 'completed.txt' not in os.listdir('run1'):
            print('Warning: AIMMD process terminated unexpectedly')
            break
        if 'completed.txt' in os.listdir('run1'):
            print('Completed successfully')
            break
        sleep(1)
    
    if 'completed.txt' not in os.listdir('run1'):
        print('Killing processes')
        for pid in pids:
            os.system(f'kill {pid}')

print('\nRunning manager individually for 30 seconds')
manager_cmd = f'{PYTHON} manager.py run1 100 1 1 1 0 0 params.py'
process = subprocess.Popen(manager_cmd, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    stdout, stderr = process.communicate(timeout=30)
    if stderr.decode().strip():
        raise RuntimeError(f'Manager error:\n{stderr.decode().strip()}')
except subprocess.TimeoutExpired:
    print('Manager timed out after 30 seconds, killing...')
    process.kill()

print('All fine')

print('\nRunning trainer individually for 30 seconds')
trainer_cmd = f'{PYTHON} trainer.py run1 params.py'
process = subprocess.Popen(trainer_cmd, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    stdout, stderr = process.communicate(timeout=30)
    if stderr.decode().strip():
        raise RuntimeError(f'Trainer error:\n{stderr.decode().strip()}')
except subprocess.TimeoutExpired:
    print('Trainer timed out after 30 seconds, killing...')
    process.kill()
print('All fine')

print('\nChecking if workers simulated...')
file = f'run1/equilibriumA/traj000001.part0001{trajectory_extension}'
if not os.path.exists(file) or os.path.getsize(file) <= 68:
    raise RuntimeError('run1/worker0 failed')
print('All fine') 

print('\nLoading AIMMD path ensemble')
pathensemble, added_nframes = update_pathensemble('run1', topology,
    states_function, descriptors_function, values_function,
    add_missing_paths=False, add_missing_frames=False, verbose=True)
if not len(pathensemble):
    raise RuntimeError('AIMMD path ensemble still empty!')

print('\nLoading the NN weights and updating the path ensemble values')
network.load_state_dict(torch.load('run1/network.h5'))
pathensemble.update_values()
equilibrium.update_values()

# report and check that everything goes well
inA_indices = np.where((pathensemble.initial_states == 'R') *
                       (pathensemble.internal_states == 'A') *
                       (pathensemble.final_states == 'R'))[0]
inB_indices = np.where((pathensemble.initial_states == 'R') *
                       (pathensemble.internal_states == 'B') *
                       (pathensemble.final_states == 'R'))[0]
ARA_indices = np.where((pathensemble.initial_states == 'A') *
                       (pathensemble.internal_states == 'R') *
                       (pathensemble.final_states == 'A'))[0]
BRB_indices = np.where((pathensemble.initial_states == 'B') *
                       (pathensemble.internal_states == 'R') *
                       (pathensemble.final_states == 'B'))[0]
TPs_indices = np.where(pathensemble.are_transitions)[0]

print('\nStatistics')
print('Number of paths  ', len(pathensemble))
print('Number of frames ', pathensemble.nframes)
print('Lengths          ', pathensemble.lengths)
print('Initial states   ', pathensemble.initial_states)
print('Shooting states  ', pathensemble.shooting_states)
print('Internal states  ', pathensemble.internal_states)
print('Final states     ', pathensemble.final_states)
print('ARA excursions   ', ARA_indices)
print('In A segments    ', inA_indices)
print('BRB excursions   ', BRB_indices)
print('In B segments    ', inB_indices)
print('Transitions      ', TPs_indices)

print('\nPlotting time series of selected paths (descriptors/values space)')

if len(TPs_indices) and len(pathensemble) > TPs_indices[0] + 2:
    i = TPs_indices[-1] - 2
else:
    i = max(0, len(pathensemble) - 3)

for i in range(i, min(len(pathensemble), i + npaths)):
    if i >= len(pathensemble) - 1:
        break
    descriptors = cv(pathensemble.path(i))
    values = expit(pathensemble.values(i)[0])  # committor
    times = np.arange(len(values))
    shooting_index = pathensemble.shooting_indices[i]
    if i in ARA_indices:
        color = 'tomato'
        label = f'ARA'
    elif i in inA_indices:
        color = 'firebrick'
        label = f'inA'
    elif i in TPs_indices:
        color = 'black'
        label = f'TP'
    elif i in BRB_indices:
        color = 'dodgerblue'
        label = f'BRB'
    elif i in inB_indices:
        color = 'blue'
        label = f'inB'
    
    plt.figure(2)
    plt.plot(times[shooting_index],
             descriptors[shooting_index], 'o', color='black', zorder=100)
    plt.plot(times, descriptors, color=color, label=label)
    
    plt.figure(3)
    plt.plot(times[shooting_index],
             values[shooting_index], 'o', color='black', zorder=100)
    plt.plot(times, values, color=color, label=label)

plt.figure(2)
plt.grid()
plt.xlabel('Time [dt]')
plt.ylabel('x coordinate')
plt.legend()
plt.savefig('aimmd_descriptors_partial_time_series.pdf')

plt.figure(3)
plt.grid()
plt.xlabel('Time [dt]')
plt.ylabel('Estimated committor')
plt.legend()
plt.savefig('aimmd_values_partial_time_series.pdf')

print('\nReweighting and estimating the rates')
wA, *_ = pathensemble.reweight('A')
wB, *_ = pathensemble.reweight('B')
print('Weights:', wA + wB)
kAB = np.nan_to_num(1 / np.sum(wA * pathensemble.internal_lengths))
kBA = np.nan_to_num(1 / np.sum(wB * pathensemble.internal_lengths))
print(f'kAB estimate: {kAB:.3e} [1/dt]')
print(f'kBA estimate: {kBA:.3e} [1/dt]')

print('\nProjecting the "from A" and "from B" ensembles')
bins = np.linspace(np.min(pathensemble.frame_values) - 1e-15,
                   np.max(pathensemble.frame_values) + 1e-15, 11)
pathensemble.weights = wA
rhoA = pathensemble.project(bins)
pathensemble.weights = wB
rhoB = pathensemble.project(bins)

print(f'Bins    {bins}  (logit committor)')
print(f'Centers {(bins[:-1] + bins[1:]) / 2}')
print(f'From A  {rhoA}  (AIMMD reweighted)')

wA, *_ = equilibrium.reweight('A')
equilibrium.weights = wA
rhoA = equilibrium.project(bins)

print(f'        {rhoA}  (equilibrium)')

print(f'From B  {rhoB}  (AIMMD reweighted)')

wB, *_ = equilibrium.reweight('B')
equilibrium.weights = wB
rhoB = equilibrium.project(bins)

print(f'        {rhoB}  (equilibrium)')

print(f'\nTesting trajectory file-based projections')
pathensemble.project([0, 1, 2], f=cv, frames=True)
print('Done!')
