"""
This script tests virtually all function of AIMMD sampling/analysis
on a molecular model defined in the "retinal" folder.
It runs with GROMACS in under five minutes.
However, there is still the polishing left to do, and compatibility
on different machines is not guaranteed.
"""
import pytest

@pytest.mark.slow
def test_retinal():

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

    FOLDER = tests_dir + '/retinal'
    
    # Attention! Gromacs options come from "retinal/params.py"
    
    print(f'Changing working directory to {FOLDER}')
    os.chdir(FOLDER)
    sys.path.append('.')
    
    print('Cleaning directories')
    os.system('rm -rf equilibrium; mkdir equilibrium')
    os.system('rm -rf run1 run2')
    
    print('\nImporting parameters')
    from params import cv
    params = aimmd.Params.load('params.py')
    params.check_engine()
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
    
    print('\nInitializing AIMMD')
    launcher = aimmd.Launcher('params.py', 'run1')
    
    print('Testing slurm job creation')
    launcher.create_job('job.sh', n=1, n1=1, n2=1, walltime=3600)
    # can run it e.g. on cluster

    # together
    print('Running with launcher')
    launcher.run(1, 1, 1, nsteps=10, walltime=120)
    
    print('\nLoading AIMMD path ensemble')
    pathensemble = params.pathensemble('run1')
    n_paths = len(pathensemble)
    if not len(pathensemble):
        raise RuntimeError('AIMMD path ensemble still empty!')
    
    print('\nLoading the NN weights and updating the path ensemble values')
    network.load_state_dict(torch.load('run1/networkARB.h5'))
    pathensemble.compute(params.values_function,
                         'values', 'descriptors', verbose=True)
        
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
    print(pathensemble.project([0, 1, 2, 3, 4], function=cv, source='reader'))
    print('Done!')

if __name__ == '__main__':
    test_retinal()
