""" Test script for the 1D toy model example of AIMMD. """
import pytest

def test_toy_1d():
    """
    This tests virtually all function of AIMMD sampling/analysis
    on a 1D toy model defined in the "toy_1d" folder.
    It runs in under five minutes.
    However, there is still the polishing left to do, and compatibility
    on different machines is not guaranteed.
    """

    import os, sys
    from time import sleep
    from time import time as current_time
    from aimmd.core.utils import PathEnsemble, update_pathensemble
    import numpy as np
    from tqdm import tqdm
    import mdtraj as md
    import matplotlib.pyplot as plt
    import subprocess
    import psutil
    import torch
    from scipy.special import expit  # logistic sigmoid

    # run either with pytest from above, or as script in main folder
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = current_dir if current_dir.endswith('tests') else current_dir + '/tests'

    FOLDER = tests_dir+ '/toy_1d'
    PYTHON = 'python3'  # will call generate.py

    nframes = 50000  # will run the exploratory eq simulation for that long

    print(f'Changing working directory to {FOLDER}')
    os.chdir(FOLDER)
    sys.path.append('.')

    print('Copying scripts from "sampling" folder (workaround)')
    os.system(f'cp ../../aimmd/sampling/generate.py .')
    os.system(f'cp ../../aimmd/sampling/generate_backend.py .')
    os.system(f'cp ../../aimmd/sampling/manager.py .')
    os.system(f'cp ../../aimmd/sampling/trainer.py .')
    os.system(f'cp ../../aimmd/sampling/worker.sh .')

    print('Cleaning directories')
    os.system('rm -rf equilibrium; mkdir equilibrium')
    os.system('rm -rf run1; mkdir run1')

    print('\nImporting libraries')
    from params import aimmd_run_params, network
    from integrator import run

    states_function = aimmd_run_params['states_function']
    descriptors_function = aimmd_run_params['descriptors_function']
    values_function = aimmd_run_params['values_function']
    topology = aimmd_run_params['topology']

    # better visualization
    np.set_printoptions(threshold=20)

    print(f'\nEquilibrating 1D system for {nframes} frames')
    traj = np.zeros(nframes)
    for i in tqdm(range(len(traj)), position=0):
        traj[i] = run(traj[i - 1])

    print('Saving in "equilibrium" folder (test relative dependencies)')
    traj_md = md.load(topology)
    traj_md.xyz = np.zeros((len(traj), 1, 3))
    traj_md.xyz[:, 0, 0] = traj / 10  # nm
    traj_md.time = np.arange(len(traj))
    traj_md.save('equilibrium/run.xtc')

    print('\nEquilibrium path ensemble initialization', end='')
    equilibrium = PathEnsemble(
        topology=f'../{topology}', directory='equilibrium',
        states_function=states_function,
        descriptors_function=descriptors_function,
        values_function=values_function)
    print(':', equilibrium)

    print('\nLoading trajectory from "equilibrium" folder')
    nframes, time = equilibrium.append('run.xtc', verbose=True)
    print(f'Loaded {nframes} frames, time: {time}')
    if not nframes:
        raise RuntimeError('Could not load the trajectory file')
    print('Obtained', equilibrium)

    print('\nSplitting path ensemble')
    equilibrium.split()
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

    if len(TPs_indices) and len(equilibrium) > TPs_indices[0] + npaths // 2:
        i = TPs_indices[-1] - npaths // 2
    else:
        i = max(1, len(equilibrium) - npaths)

    old_label = ''
    for i in range(i, min(len(equilibrium) - 1, i + npaths)):
        descriptors = equilibrium.descriptors(i)[0][:, 0]
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
            raise RuntimeError(f'Paths {old_label} ({i - 1}) and {label} ({i}) '
                            f'should be of different kind.')
        old_label = label
        plt.figure(1)
        plt.plot(times[0], descriptors[0], 'o', color='black', zorder=100)
        plt.plot(times, descriptors, color=color, label=label)

    plt.figure(1)
    plt.grid()
    plt.xlabel('Time [dt]')
    plt.ylabel('x coordinate')
    plt.legend()
    plt.savefig('equilibrium_descriptors_partial_time_series.pdf')

    print(f'\nExtracting central path of the plotted ones ({i})', end='')
    if i - 2 > 0:
        i = i - 2
    else:
        i = i // 2
    path = equilibrium.path(i)
    print(':', path)
    print('Saving the path as a separate trajectory file')
    path.write('run1/initial.xtc')
    print('Adding it to the "initial path" ensemble', end='')
    initial_path = equilibrium[:0]
    initial_path.add_path('../run1/initial.xtc')
    print(':', initial_path)

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
    print(f'Bins    {bins}  (1st descriptor\'s space)')
    print(f'Centers {(bins[:-1] + bins[1:]) / 2}')
    print(f'From A  {rhoA}')
    print(f'From B  {rhoB}')

    print('\nInitializing AIMMD: three simulators, one manager, one trainer')
    command = f'{PYTHON} generate.py run1 100 1 1 1 -p params.py'
    process = subprocess.Popen(command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.stdout.read().decode().strip()
    error = process.stderr.read().decode().strip()
    if error:
        raise RuntimeError(error)

    pids = [int(pid) for pid in 
            stdout.split('PIDS')[1].split('\n')[0].split()]
    print(f'\nAIMMD process PIDs: {pids}')
    print(f'Running for at most one minute...')
    print(f'(You can inspect {FOLDER}/run1\'s log files for details.)')
    t0 = current_time()
    while current_time() - t0 < 60:
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
    file = 'run1/equilibriumA/traj000001.part0001.xtc'
    if not os.path.exists(file) or os.path.getsize(file) <= 68:
        file = 'run1/equilibriumA/traj000001.part0002.xtc'
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
    if len(TPs_indices) and len(pathensemble) > TPs_indices[0] + npaths // 2:
        i = TPs_indices[-1] - npaths // 2
    else:
        i = max(0, len(pathensemble) - npaths)

    old_label = ''
    for i in range(i, min(i + npaths, len(pathensemble))):
        descriptors = pathensemble.descriptors(i)[0][:, 0]
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
        if old_label == label:
            raise RuntimeError(f'Paths {old_label} ({i}) and {label} ({i}) '
                            f'should be of different kind.')
        old_label = label
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
    bins = np.linspace(np.min(equilibrium.frame_values) - 1e-15,
                    np.max(equilibrium.frame_values) + 1e-15, 11)
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
    pathensemble.project([0, 1, 2],
        f=lambda traj: np.array([frame.positions[0, 0] for frame in traj]),
                        frames=True)
    
    # test for basic scientific correctness by checking if the rates are as expected, ie within one order of magnitude
    assert 5e-5 < kAB < 5e-3, f'kAB={kAB} out of expected range'

    print('Done!')

if __name__ == '__main__':
    test_toy_1d()