# base params file

# dummy functions in case not provided by the user

from tqdm import tqdm

def states_function(trajectory, verbose=False):
    """
    Input: MDAnalysis trajectory, or
           pathensemble.MDATrajectory object, or
           list of MDAnalysis trajectory frames.
    
    Output: <U1 array, same length as the input trajectory,
            containing the state of each trajectory fame
            ("A", "B", or "R" in between, and others).
            
            The output of this function will be cached in the
            "PathEnsemble" objects during an AIMMD run. In this
            way, you can load just the .h5 pathensembles instead
            of the original trajectory files for your analysis.
    """
    return np.array(['R'
        for frame in tqdm(trajectory, position=0, disable=not verbose)])


def descriptors_function(trajectory, verbose=False):
    """
    Input: MDAnalysis trajectory, or
           pathensemble.MDATrajectory object, or
           list of MDAnalysis trajectory frames.
    
    Output: 2D float array, length as the input trajectory,
            containing a representation of each trajectory frame.
    """
    return np.array([frame.positions.copy()
        for frame in tqdm(trajectory, position=0, disable=not verbose)])

def values_function(descriptors, verbose=False):
    return np.repeat(0., len(descriptors))


