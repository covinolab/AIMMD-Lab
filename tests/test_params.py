"""
This script tests the functionalities of the Params class,
with special attention to relative import and modules creation.
"""

def test_params():
    
    import os
    import aimmd
    import numpy as np
    import torch
    from pathlib import PosixPath
    from aimmd.network import Rescalable
            
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = cwd if cwd.endswith('tests') else cwd + '/tests'
    
    for FOLDER, initial in zip(
        [tests_dir + '/toy_1d', tests_dir + '/retinal'],
        ['initial.xtc', 'initial.trr']):
        
        # relative path
        folder = FOLDER.split('/')[-1]
        
        try:  # setup
            os.chdir(FOLDER)
            os.system('mkdir run1 run2')
            os.system('cp params.py run1')
            os.system(f'cp {initial} run1')
            
            print('Test 1: loading "params.py" into "params" object')
            params = aimmd.Params('params.py')
            if not os.path.exists('params1.py'):
                raise RuntimeError('params1.py not created')
            
            print('Test 2: loading "params.py" into "params2" results in '
                  '"params" and "params2" being equal')
            params2 = aimmd.Params('params.py')
            assert params == params2
            if os.path.exists('params2.py'):
                raise RuntimeError('params2.py should not exist')
            
            print('Test 3: loading "params1.py" into "params2" returns the '
                  'same "params2" as before')
            params2 = aimmd.Params(params.path)
            assert params == params2
            if os.path.exists('params2.py'):
                raise RuntimeError('params2.py should not exist')
            
            print('Test 4: loading "params1.py" into "params2" while in the '
                  '"run1" folder returns the same "params2" as before')
            os.chdir('run1')
            params2 = aimmd.Params(params.path)
            assert params == params2
            if os.path.exists('../params2.py'):
                raise RuntimeError('params2.py should not exist')
            
            print('Test 5: loading "params.py" into "params2" while in the '
                  '"run1" folder returns the same "params2" as before')
            params2 = aimmd.Params('../params.py')
            assert params == params2
            if os.path.exists('../params2.py'):
                raise RuntimeError('../params2.py should not exist')
            
            # back to "folder"
            os.chdir('..')
            
            print('Test 6: loading "params1.py" file into "params3" with '
                  'additional  fields (fit function and initial paths)')
            
            # as if you defined "identity" directly on terminal
            def fit(params, pathensemble, verbose, worker):
                return 1

            def identity(x):
                return 1
            
            params3 = aimmd.Params(
                params.path, fit=fit,
                initial_paths=[initial, f'run1/{initial}'])
            if not os.path.exists('params2.py'):
                raise RuntimeError('params2.py not created')
            
            initial_paths = params3.initial_paths
            if len(initial_paths) != 2:
                raise RuntimeError('there should be two initial paths now')

            if params3.initial_paths[1].fname != f'run1/{initial}':
                raise RuntimeError(
                    f'second initial path\'s location '
                    f'should be run1/{initial}, got'
                    f'{params3.initial_paths[1].fname} instead')
            
            print('Test 7: updating "params3" with "params2.py" leaves '
                  '"params3" unchanged')
            #params3 = params3.load('params2.py')
            #if os.path.exists('params3.py'):
            #    raise RuntimeError('params3.py should not exist')
            
            #assert params3.fit(10) == 1
            
            print('Test 8: udating "params2" with "params1.py" leaves '
                  '"params2" unchanged')
            params2 = params2.load('params1.py')
            assert params == params2
            if os.path.exists('params3.py'):
                raise RuntimeError('params3.py should not exist')
                        
            print('Test 9: udating "params2" with the different fields in '
                  '"params3" results in "params2" being equivalent to '
                  '"params3"')
            #params2 = params2.update(
            #    fit=identity, initial_paths=[initial, f'run1/{initial}'])
            #if os.path.exists('params3.py'):
            #    raise RuntimeError('params3.py should not exist')

            print('Test 10: udating "params2" with its identical parameters '
                  'leaves params2 unchanged')
            #params2 = params2.update(network=params2.network)
            #if os.path.exists('params3.py'):
            #    raise RuntimeError('params3.py should not exist')
            
            print('Test 11: trying to assign a bad states function throws an '
                  'error and leaves the corresponding field unchanged')
            try:
                params.update(states_function=identity)
                raise RuntimeError(
                    'bad states function did not result in failure')
            except TypeError:
                if params.states_function == identity:
                    raise RuntimeError('states function should not be identity')
                pass  # success
            
            print('Test 12: trying to assign a bad descriptors_function throws '
                  'an error and leaves the corresponding field unchanged')
            try:
                params.descriptors_function = identity
                raise RuntimeError(
                    'bad descriptors function did not result in failure')
            except Exception:
                if params.descriptors_function == identity:
                    raise RuntimeError(
                        'descriptors function should not be identity')
                pass  # success
            
            print('Test 13: trying to assign a bad values function throws an '
                  'error and leaves the corresponding field unchanged')
            try:
                params.values_function = identity
                raise RuntimeError(
                    'bad values function did not result in failure')
            except TypeError:
                if params.values_function == identity:
                    raise RuntimeError('values function should not be identity')
                pass  # success

            # THIS IS NOT A THING ANYMORE, it just throws a warning, error
            # if running with reactive_region_mode = 'shoot'
            #
            # print('Test 14: trying to assign bad initial_paths throws an error '
            #       'and leaves the corresponding field unchanged')
            # try:
            #     params.initial_paths = 'run.gro'
            #     raise RuntimeError('bad initial paths did not result in failure')
            # except TypeError:
            #     if len(initial_paths[0]) == 1:
            #         raise RuntimeError('"params.initial_paths" changed')
            #     pass  # success
            
            print('Test 15: creating a new "Network" class with the required '
                  'methods missing, initializing it on __main__, and then '
                  'trying to update "params2" with the object results in an '
                  'error')
            class NewNetwork:  # without required methods
                pass
            
            network = NewNetwork()
            
            # as if on terminal
            NewNetwork.__module__ = '__main__'
            network.__module__ = '__main__'
            
            try:
                params2.update(network=network)
                raise RuntimeError('"params2" could be updated')
            except TypeError:
                pass
            
            print('Test 16: creating a new "Network" class, initializing it '
                  'on __main__ with a different name than network, and then '
                  'updating "params2" with "network" changes "params2" and '
                  'generates the file "params3.py" where the network is '
                  '"network" (regardless of the original variable name)')
            class NewNetwork(Rescalable):  # with required methods
                def parameters(self):
                    return iter([torch.zeros(1)])
            
            new_network = NewNetwork()
            
            # as if on terminal
            NewNetwork.__module__ = '__main__'
            NewNetwork.__source__ = ('class NewNetwork:\n'
                                     ' def parameters(self):\n'
                                     '  return iter([torch.zeros(1)])')
            new_network.__module__ = '__main__'
            
            params2.update(network=new_network)
            if not os.path.exists('params3.py'):
                raise RuntimeError('params3.py not created')

            print('Test 17: reloading "params3.py" now gives an equivalent '
                  'object to "params2"')
            #assert aimmd.Params('params3.py') == params2
            
            print('Test 18: check initial paths properties')
            pass
            
            print('Test 19: check initial paths properties')
            pass
            
            print('Test 20: checking that "params3" cannot be updated with a '
                  f'file in a different folder than "{folder}"')
            os.chdir('run1')
            os.system('cp ../params.py .')
            try:
                params3 = params3.load('params.py')
                raise RuntimeError('"params4" was updated with "run1/params.py"')
            except TypeError:
                pass
            
            print('Test 21: checking that "params3" cannot be saved in a '
                  f'different folder than "{folder}"')
            try:
                params3.save('../../../params.py')
                raise RuntimeError('"params4" was saved in "../../../"')
            except TypeError:
                pass
        
        finally:  # cleanup
            os.chdir(FOLDER)
            os.system('rm -rf params?.py run1 run2')
            os.chdir(cwd)


if __name__ == '__main__':
    test_params()
