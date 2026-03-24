"""
This is a test of the whole AIMMD workflow on a toy 1D system, which is simple enough to run in a few minutes on a CPU. It is not meant to be a test of the accuracy of the method, but rather of the functionality of the code. It can be run with pytest or as a script in the main folder. It assumes that Gromacs is installed and available in the PATH, and that the "retinal/params.py" file is present and correctly configured for this test.
"""
import pytest

def test_toy_1d():

    import os
    import sys
    import time
    import aimmd
    import torch
    import numpy as np
        
    np.set_printoptions(threshold=20)  # better visualization
    
    # run either with pytest from above, or as script in main folder
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = current_dir if current_dir.endswith('tests') else current_dir + '/tests'

    FOLDER = tests_dir + '/toy_1d'
    
    # Attention! Gromacs options come from "retinal/params.py"
    
    print(f'Changing working directory to {FOLDER}')
    os.chdir(FOLDER)
    sys.path.append('.')
    
    print('Cleaning directories')
    os.system('rm -rf equilibrium; mkdir equilibrium')
    os.system('rm -rf run1 run2')
    
    print('\nImporting parameters')
    params = aimmd.Params.load('params.py')
    #params.check_engine(timeout=2)
    states_function = params.states_function
    descriptors_function = params.descriptors_function
    values_function = params.values_function
    network = params.network
    trajectory_extension = params.trajectory_extension
    
    # running one at a time
    print('Running one task at a time')
    t0 = time.time()
    aimmd.Worker(params, 'run1', walltime=120).train(nrounds=1)
    if time.time() - t0 > 110:
        raise RuntimeError(
            'Training takes too much time, you may want to skip this test')
    aimmd.Worker(params, 'run1', nframes=100, walltime=120).free(0, 0, 1)
    aimmd.Worker(params, 'run1', nsteps=5, walltime=120).shoot(1, 0)

    print('\nManaging two chains with the same worker')
    aimmd.Worker(params, 'run1', walltime=60).shoot(1, 0, nchains_per_task=2)
    
    print('\nInitializing AIMMD')
    launcher = aimmd.Launcher('params.py', 'run1')
    
    print('Testing slurm job creation')
    launcher.create_job('job.sh', n=1, n1=1, n2=1, nchains_per_task=3, walltime=3600)
    # can run it e.g. on cluster

    # together
    print('Running with launcher (one chain per task)')
    launcher.run(1, 1, 1, nsteps=5, walltime=60)
    launcher.run(1, 1, 1, nsteps=12, walltime=60)
    
    print('Running with launcher (three chains per task)')
    launcher.run(1, 1, 1, nsteps=20, walltime=60, nchains_per_task=3)
    
    print('\nLoading AIMMD path ensemble')
    pathensemble = params.pathensemble('run1')
    n_paths = len(pathensemble)
    if not len(pathensemble):
        raise RuntimeError('AIMMD path ensemble still empty!')
    
    print('\nLoading the NN weights and updating the path ensemble values')
    network.load_state_dict(torch.load('run1/networkARB.h5'))
    pathensemble.compute(*params.compute_values_args, verbose=True)
    
    # report and check that everything goes well
    i = pathensemble.initial('states')
    m = pathensemble.middle('states')
    f = pathensemble.final('states')
    s = pathensemble.shooting('states')
    inA_indices = np.flatnonzero((i == 'R') * (m == 'A') * (f == 'R'))
    inB_indices = np.flatnonzero((i == 'R') * (m == 'B') * (f == 'R'))
    ARA_indices = np.flatnonzero((i == 'A') * (m == 'R') * (f == 'A'))
    BRB_indices = np.flatnonzero((i == 'B') * (m == 'R') * (f == 'B'))
    TPs_indices = np.flatnonzero(pathensemble.are_transitions())
    are_shot = pathensemble.types('...R')
    
    print('\nStatistics')
    print('Number of paths  ', len(pathensemble))
    print('Number of shots  ', np.sum(are_shot))
    print('Number of frames ', sum(pathensemble.n_frames))
    print('Lengths          ', pathensemble.n_frames)
    print('Initial states   ', i)
    print('Shooting states  ', s)
    print('Middle states    ', m)
    print('Final states     ', f)
    print('ARA excursions   ', ARA_indices)
    print('In A segments    ', inA_indices)
    print('BRB excursions   ', BRB_indices)
    print('In B segments    ', inB_indices)
    print('Transitions      ', TPs_indices)
    
    print(f'\nTesting trajectory file-based projections')
    cv = lambda traj: np.array([ts.positions.copy() for ts in traj])
    pathensemble = pathensemble[pathensemble.are_excursions()]
    projection = pathensemble.project(
        [[0, 1, 2, 3, 4], [0, 1], [0, 1]],
        function=cv, source='reader', verbose=True)
    # since our params.toy_mdrun function is such that x==y==z,
    # and A has x in (1, 2), the projection should be zero everywhere
    assert projection.max() == 0
    projection = pathensemble.project(
        [[0, 1, 2, 3, 4], [1, 2], [1, 2]],
        function=cv, source='reader')
    assert projection.sum() == pathensemble.n_frames.sum()
    print('Done!')

if __name__ == '__main__':
    test_toy_1d()
