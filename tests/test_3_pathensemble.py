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


def test_sparse_value_updates_and_bins():
    """ Test the sparse value update functionality, which should speed up determining bins by
    only calling the value function on a certain maximum number of frames. The remainder should
    be filled with zeros, to avoid situations where frame values from previous network evaluations
    interfere with the current network evaluation bins.
    """

    import aimmd
    from aimmd.core.pathensemble import PathEnsemble
    from aimmd.core.utils import get_bins
    import os
    import pytest
    import numpy as np

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

    pe.update_values()

    # make sure densities work properly after full update
    bins = get_bins(pe, states=True)
    densities = pe.project(bins)
    assert np.sum(densities) == 132, "Path Ensemble project did not work properly after full update."

    # Check that no values are zero after full update
    assert not any(pe.frame_values == 0), "Path Ensemble has zero values after full update."

    # update sparsely with clash, this should raise an error
    with pytest.raises(ValueError):
        pe.update_values(sparse_update_max_frames=10, check_for_network_change=True)
    
    pe.update_values(sparse_update_max_frames=10, check_for_network_change=False)

    # Check that all but 10 values are zero after sparse update
    num_zeros = sum(pe.frame_values == 0)
    zeros_indices = pe.frame_values == 0
    assert num_zeros == len(pe.frames()) - 10, f"Path Ensemble sparse update did not result in correct sparse update, got {num_zeros} zeros where {len(pe.frames()) - 10} were expected."

    # Check that different frames are set to zero after another sparse update
    previous_zeros_indices = zeros_indices.copy()
    pe.update_values(sparse_update_max_frames=10, check_for_network_change=False)
    new_zeros_indices = pe.frame_values == 0

    assert not all(previous_zeros_indices == new_zeros_indices), "Path Ensemble sparse update did not change zeroed frames on subsequent call."

    bins = get_bins(pe, cutoff_min=0.5)

    assert len(bins) > 1, "Path Ensemble sparse update did not allow binning to proceed correctly."
    assert np.max(bins) == np.max(pe.frame_values[pe.frame_values != 0]), "Did not get correct max bin value."
    assert np.min(bins) == np.min(pe.frame_values[pe.frame_values != 0]), "Did not get correct min bin value."

    # Now let's make sure the densities calculation in the trainer works properly with sparse updates
    bins = get_bins(pe, states=True)
    densities = pe.project(bins)
    
    # check we are disregarding zero values in pe.project
    assert np.sum(densities) <= 10, "Path Ensemble project did not disregard zero values properly."

    # Clean up
    os.chdir(FOLDER)
    os.system('rm -rf params?.py')
    os.chdir(cwd)

if __name__ == '__main__':
    #test_only_update_pathensemble_when_network_changes()
    test_sparse_value_updates_and_bins()