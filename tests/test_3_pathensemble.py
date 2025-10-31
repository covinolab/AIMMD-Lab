"""" Tests relating to pathensemble module."""

def test_only_update_pathensemble_when_network_changes():
    """ This test checks new functionality in pathensemble, where the values are only updated when
    the network changes for those entries that have already calculated values.
    """

    import aimmd
    from aimmd.core.pathensemble import PathEnsemble
    import os
    from time import time

    # run either with pytest from above, or as script in main folder
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tests_dir = current_dir if current_dir.endswith('tests') else current_dir + '/tests'

    # go into retinal folder to be able to load params properly
    FOLDER = tests_dir + '/retinal'
    os.chdir(FOLDER)

    params = aimmd.Params.load('params.py')

    # make a test path ensemble based on the retinal case
    pe = PathEnsemble(topology='run.gro',
                      states_function=params.states_function,
                      descriptors_function=params.descriptors_function,
                      values_function=params.values_function)
    pe.append('initial.trr')

    assert pe.lengths[0] == 202, "Path Ensemble can't initialise and load trajectory properly."

    # Path ensemble initialisation already assigns values. Change the network for testing now.
    params.network.reset_parameters()

    # Time the evaluation of the values function on this path ensemble
    start_time = time()
    pe.update_values()
    first_duration = time() - start_time

    # Time the evaluation of the values function on this path ensemble
    start_time = time()
    pe.update_values()
    second_duration = time() - start_time

    assert second_duration < first_duration / 3, f"Path Ensemble values update did not speed up enough on second call, {second_duration} vs {first_duration}."

    # Clean up
    os.chdir(FOLDER)
    os.system('rm -rf params?.py')
    os.chdir(cwd)

if __name__ == '__main__':
    test_only_update_pathensemble_when_network_changes()