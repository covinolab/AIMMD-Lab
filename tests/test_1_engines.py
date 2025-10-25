"""
This script tests that the simulators' engines work correctly,
after initializing while loading params.
"""

def test_engines():
    
    import os
    import aimmd
    import numpy as np
    import mdtraj as md
    
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = cwd if cwd.endswith('tests') else cwd + '/tests'
    FOLDER1 = tests_dir + '/toy_1d'
    FOLDER2 = tests_dir + '/retinal'
    
    try:
        
        print('Test 1: check toy engine (toy_1d)')
        os.chdir(FOLDER1)
        initial_path = md.load('run.gro')
        initial_path.xyz = np.linspace([[0.,0.,0.]], [[1.,1.,1.]])
        initial_path.time = np.arange(len(initial_path))
        initial_path.save('initial.xtc')
        params = aimmd.Params('params.py')
        params._check_engine()
        
        print('Test 2: check gromacs engine (retinal)')
        os.chdir(FOLDER2)
        params = aimmd.Params('params.py')
        params._check_engine()
        
        print('Test 3: check gromacs engine from a different folder')
        os.chdir('..')
        params._check_engine()
    
    finally:  # cleanup
        os.system(f'rm -rf {FOLDER1}/params1.py {FOLDER1}/initial.xtc')
        os.system(f'rm -rf {FOLDER2}/params1.py')
        os.chdir(cwd)


if __name__ == '__main__':
    test_engines()
