"""
Conducts an AIMMD run on a 1D toy model defined in the "toy_1d" folder. 
"""


def test_toy_1d():
    
    import os
    import time
    import aimmd
    import numpy as np
    import mdtraj as md

    def check_file(filename, min_size=68):
        if (not os.path.exists(filename)
            or os.path.getsize(filename) <= min_size):
            raise RuntimeError(f'"{file}" not found')
        
    def check_time(walltime, name=''):
        nonlocal t0
        if time.time() - t0 < walltime:
            raise RuntimeError(
                f'{name}{" " if name else ""}'
                f'did not run for more than {walltime} seconds')
    
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = cwd if cwd.endswith('tests') else cwd + '/tests'
    FOLDER = tests_dir + '/toy_1d'
    
    try:  # setup
        os.chdir(FOLDER)
        initial_path = md.load('run.gro')
        initial_path.xyz = np.linspace([[0.,0.,0.]], [[1.,1.,1.]])
        initial_path.time = np.arange(len(initial_path))
        initial_path.save('initial.xtc')
        params = aimmd.Params('params.py')
        
        print('Test 1: creating launcher')
        launcher = aimmd.Launcher(params, 'run1')
        
        print('Test 2: creating slurm job script')
        launcher.create_job('job.sh', n=1, nA=1, nB=1,
                            ntasks_per_node=4, cpus_per_task=8,
                            gpus_per_task=1, walltime=3600)
        
        print('Test 3: individual worker (train)')
        worker = aimmd.Worker(params, 'run1')
        worker.train(nrounds=2, walltime=60)
        check_file('run1/network.h5')
        
        print('Test 4: individual worker (manager)')
        t0 = time.time()
        worker.manage(n=1, nA=1, nB=1, walltime=5)
        check_time(5, 'manager')
        
        print('Test 5: individual worker (simulator)')
        t0 = time.time()
        worker.simulate('worker0', noappend=True, walltime=10)
        check_time(10, 'simulator')
        check_file('run1/equilibriumA/traj000001.part0002.xtc')
        
        print('Test 6: AIMMD run sharing resources')
        t0 = time.time()
        launcher.run(1, 1, 1, cpus_per_task='share', gpus_per_task='share',
                     nsteps=10, walltime=60)
        if len(aimmd.PathEnsemble().load('run1/shots0/chain.h5')) < 10:
            check_time(60, 'AIMMD')
            if time.time() - t0 < 61:
                raise RuntimeError('not reached 10 steps')
        check_file('run1/equilibriumB/traj000001.part0002.xtc')
        
        print('Test 7: AIMMD run with "all" resources (append)')
        t0 = time.time()
        launcher.run(1, 1, 1, cpus_per_task='all', gpus_per_task='all',
                     walltime=20)
        check_time(20, 'AIMMD')
        
        print('Test 8: AIMMD run with limited resources (append)')
        t0 = time.time()
        launcher.run(1, 1, 1, cpus_per_task=1, gpus_per_task=0,
                     walltime=20)
        check_time(20, 'AIMMD')
        
        print('Test 9: two AIMMD runs with LaunchersCollection '
              '(append + new)')
        with open('params1.py') as file:
            print(file.read())  # DEBUG
        launchers = aimmd.LaunchersCollection(launcher,
            aimmd.Launcher('params1.py', 'run2'))
        t0 = time.time()
        launchers.run(1, [0, 1], 1, nsteps=[float('inf'), 10], walltime=60)
        if len(aimmd.PathEnsemble().load('run2/shots0/chain.h5')) < 10:
            check_time(60, 'AIMMD')
            if time.time() - t0 < 61:
                raise RuntimeError('not reached 10 steps')
        check_file('run2/equilibriumA/traj000001.part0002.xtc')
        check_file('run2/equilibriumB/traj000001.part0002.xtc')
    
    finally:  # cleanup
        os.chdir(FOLDER)
        os.system(f'rm -rf run1 run2 initial.xtc params1.py job.sh')
    
if __name__ == '__main__':
    test_toy_1d()
