"""
This script tests the functionalities of the Params class,
with special attention to relative import and modules creation.
"""

def test_params():
    
    import os
    import aimmd
    from aimmd.core.utils import absolute_path
    
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = cwd if cwd.endswith('tests') else cwd + '/tests'
    FOLDER = tests_dir + '/retinal'
    
    try:  # setup
        os.chdir(FOLDER)
        os.system('mkdir run1 run2')
        os.system('cp params.py run1')
        os.system('cp initial.trr run1')
        
        print('Test 1: loading "params.py" into "params" object')
        params = aimmd.Params('params.py')
        if not os.path.exists('params1.py'):
            raise RuntimeError('params1.py not created')
        
        print('Test 2: loading "params.py" into "params2" results in "params"'
              ' and "params2" being equal')
        params2 = aimmd.Params('params.py')
        assert params == params2
        
        print('Test 3: loading "params1.py" into "params2" returns the same '
              '"params2" as before')
        params2 = aimmd.Params(params.path)
        assert params == params2
        if os.path.exists('params3.py'):
            raise RuntimeError('params3.py should not exist')
        
        print('Test 4: loading "params1.py" into "params2" while in the '
              '"run1" folder returns the same "params2" as before')
        os.chdir('run1')
        params2 = aimmd.Params(params.path)
        assert params == params2
        if os.path.exists('../params3.py'):
            raise RuntimeError('params3.py should not exist')
        
        print('Test 5: loading "params.py" into "params2" while in the '
              '"run1" folder creates "params3.py" in the "retinal" folder, '
              'and returns the same "params2" as before')
        params2 = aimmd.Params('../params.py')
        assert params == params2
        if not os.path.exists('../params3.py'):
            raise RuntimeError(
                'params3.py should exist in the "retinal" folder')
        
        # back to "retinal"
        os.chdir('..')
            
        print('Test 6: loading "params1.py" file into "params3" with '
              'additional  fields (fit function and initial paths)')
        
        # as if you defined "identity" directly on terminal
        identity = lambda x: 1
        identity.__module__ = '__main__'
        identity.__source__ = 'lambda x: 1'
        
        params3 = aimmd.Params(
            params.path, fit=identity,
            initial_paths=['initial.trr', 'run1/initial.trr'])
        if not os.path.exists('params4.py'):
            raise RuntimeError('params4.py not created')
        
        if params3.fit.__module__ != \
            absolute_path('params4.py').rstrip('.py'):
            raise RuntimeError(f'new fit not assigned to params4, '
                               f'{params3.fit.__module__} instead')
        
        initial_paths = params3.initial_paths
        if len(initial_paths) != 2:
            raise RuntimeError('there should be two initial paths now')
        
        if params3.initial_paths[1].filename != absolute_path(
            'run1/initial.trr'):
            raise RuntimeError(f'second initial path\'s location '
                               f'should be run1/initial.trr, '
                               f'{params3.initial_paths[1].filename} instead')
        
        print('Test 7: updating "params3" with "params3.py" leaves "params3"'
              ' unchanged')
        params3 = params3.load('params4.py')
        if os.path.exists('params5.py'):
            raise RuntimeError('params5.py should not exist')
        
        assert params3.fit(10) == 1
        
        print('Test 8: udating "params2" with "params2.py" leaves "params2" '
              'unchanged')
        params2 = params2.load('params2.py')
        assert params == params2
        if os.path.exists('params5.py'):
            raise RuntimeError('params5.py should not exist')
        
        print('Test 9: udating "params2" with the different fields in '
              '"params3" results in "params2" being equivalent to "params3"')
        params2 = params2.update(
            fit=identity, initial_paths=['initial.trr', 'run1/initial.trr'])
        assert params2 == params3
        if not os.path.exists('params5.py'):
            raise RuntimeError('params5.py not created')
        
        print('Test 10: trying to assign a bad states function throws an '
              'error and leaves the corresponding field unchanged')
        try:
            params.update(states_function=identity)
            raise RuntimeError(
                'bad states function did not result in failure')
        except TypeError:
            if params.states_function == identity:
                raise RuntimeError('states function should not be identity')
            pass  # success
        
        print('Test 11: trying to assign a bad descriptors_function throws '
              'an error and leaves the corresponding field unchanged')
        try:
            params.descriptors_function = identity
            raise RuntimeError(
                'bad descriptors function did not result in failure')
        except TypeError:
            if params.descriptors_function == identity:
                raise RuntimeError(
                    'descriptors function should not be identity')
            pass  # success
        
        print('Test 12: trying to assign a bad values function throws an '
              'error and leaves the corresponding field unchanged')
        try:
            params.values_function = identity
            raise RuntimeError(
                'bad values function did not result in failure')
        except TypeError:
            if params.values_function == identity:
                raise RuntimeError('values function should not be identity')
            pass  # success
        
        print('Test 13: trying to assign bad initial_paths throws an error '
              'and leaves the corresponding field unchanged')
        try:
            params.initial_paths = 'run.gro'
            raise RuntimeError('bad initial paths did not result in failure')
        except TypeError:
            if len(initial_paths[0]) == 1:
                raise RuntimeError('"params.initial_paths" changed')
            pass  # success
        
        print('Test 14: saving initial paths to the "run2" folder, going to '
              'that folder, and reloading one saved path as initial path for '
              '"params4"')
        params3.save_initial_paths('run2')
        os.chdir('run2')
        params4 = aimmd.Params('../params.py', initial_paths='initial.trr')
        if not os.path.exists('../params6.py'):
            raise RuntimeError('params6.py not created')
        
        assert params4.initial_paths[0].filename == absolute_path(
            'initial.trr')
        
        print('Test 15: checking "params4" initial paths and states function '
              'after moving to a different folder ("run1")')
        os.chdir('../run1')
        params4._check_initial_paths_and_states_function()
        
        print('Test 16: checking that "params4" cannot be updated with a '
              'file in a different folder than "retinal"')
        try:
            params4 = params4.load('params.py')
            raise RuntimeError('"params4" was updated with "run1/params.py"')
        except TypeError:
            pass
    
    finally:  # cleanup
        os.chdir(FOLDER)
        os.system('rm -rf params?.py run1 run2')
        os.chdir(cwd)


if __name__ == '__main__':
    test_params()
