"""
This script tests virtually all function of AIMMD sampling/analysis
on a molecular model defined in the "retinal" folder.
It runs with GROMACS in under five minutes.
However, there is still the polishing left to do, and compatibility
on different machines is not guaranteed.
"""

import os
import sys
import time
import aimmd
import torch
from aimmd import PathEnsemble
from aimmd.core.utils import update_pathensemble
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(threshold=20)  # better visualization

if __name__ == '__main__':
    
    FOLDER = 'retinal'
    topology = 'run.gro'
    seconds = 12  # of equilibration
    
    # Attention! Gromacs options come from "retinal/params.py"
    
    print(f'Changing working directory to {FOLDER}')
    os.chdir(FOLDER)
    sys.path.append('.')
    
    print('Cleaning directories')
    os.system('rm -rf equilibrium; mkdir equilibrium')
    os.system('rm -rf run1')
    
    print('\nImporting parameters')
    from params import cv
    params = aimmd.Params.update('params.py')
    topology = params.topology
    grompp = params.grompp
    mdp = params.gmx_run_mdp
    states_function = params.states_function
    descriptors_function = params.descriptors_function
    values_function = params.values_function
    network = params.network
    mdrun = params.mdrun
    trajectory_extension = params.trajectory_extension
    
    # no max time option here
    mdrun = mdrun.split()
    for i, field in enumerate(mdrun):
        if field == '-maxh':
            mdrun[i] = ''
            mdrun[i + 1] = ''
            break
    mdrun = ' '.join(mdrun)
    
    print(f'\nEquilibrating system for {seconds} seconds')
    code = os.system(f'{grompp} -f {mdp} '
              f'-c {topology} -r {topology} '
              f'-o equilibrium/run.tpr')
    if code:
        raise RuntimeError('Error during tpr generation')
    code = os.system(f'{mdrun} -deffnm equilibrium/run '
                     f'-maxh {seconds/3600}')
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
            raise RuntimeError(f'Paths {old_label} ({i - 1}) and {label} ({i}) '
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
    
    print('\nInitializing AIMMD: three simulators, one manager, one trainer')
    launcher = aimmd.Launcher(params, 'run1', 1, 1, 1)
    
    print('Testing slurm job creation')
    launcher.create_job('job.sh')
    # can run it e.g. on cluster

    # running one at a time
    t0 = time.time()
    Worker(params, 'run1').train(walltime=10)
    if time.time() - t0 < 10:
        print('TRAINER FAILED')
    
    t0 = time.time()
    Worker(params, 'run1').manage(1, 1, 1, 0, 0, walltime=10)
    if time.time() - t0 < 10:
        print('MANAGER FAILED')
    
    t0 = time.time()
    Worker(params, 'run1').simulate('worker0.run', append=True, walltime=10)
    if time.time() - t0 < 10:
        print('FREE A FAILED')

    t0 = time.time()
    Worker(params, 'run1').simulate('worker1.run', append=True, walltime=10)
    if time.time() - t0 < 10:
        print('FREE B FAILED')

    # worker, trainer, and equilibrium shooting together
    launcher = aimmd.Launcher(params, 'run1', 1, 0, 0)
    t0 = time.time()
    launcher.run(walltime=60)
    if time.time() - t0 < 60:
        print('SHOOTING FAILED')
    
    print('\nChecking if workers simulated...')
    file = f'run1/equilibriumA/traj000001.part0001{trajectory_extension}'
    if not os.path.exists(file) or os.path.getsize(file) <= 68:
        raise RuntimeError('run1/worker0 failed')
    print('All fine')

    print('\nTest appending')
    t0 = time.time()
    Worker(params, 'run1').simulate('worker0.run', append=True, walltime=10)
    if time.time() - t0 < 10:
        print('FREE A FAILED')
    
    print('\nChecking if workers simulated...')
    file = f'run1/equilibrium/traj000001.part0002{trajectory_extension}'
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
    
    for i in range(i, min(len(pathensemble), i + npaths)):
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
