"""
...
"""

# external
import numpy as np
from math import inf
from pathlib import PosixPath
from collections.abc import Iterable
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.coordinates.timestep import Timestep

# aimmd imports
from ..path import Path
from ..path.utils import get_fnames

# match patterns
def match_patterns(types, *patterns):
    # initialize matches
    matches = np.zeros(len(types), dtype=bool)
    
    if not len(matches):
        return matches
    
    # process types
    types = np.array(list(map(list, types))).T
    
    for pattern in patterns:
        current = np.ones(len(matches), dtype=bool)
        
        # standardized form
        pattern = (pattern + '...')[:4]
        
        # conditions
        for i in range(4):
            state1 = pattern[i]
            if state1 == '.':
                continue  # nothing to do
            current[types[i] != state1] = False

        # add to previous
        matches |= current

    # return
    return matches


def get_paths(*instances, initialize=True, **kwargs):
    """kwargs are passed to path initialization"""
    from . import PathEnsemble
    from ..path import Path
    result = []
    for instance in instances:
        if isinstance(instance, Path):
            result.append(Path(instance, **kwargs))
        elif isinstance(instance, PathEnsemble):
            result += [Path(path, **kwargs) for path in instance._paths]
        elif isinstance(instance, (str, PosixPath)):
            fnames = get_fnames(str(instance))
            if initialize:
                for fname in fnames:
                    path = Path(fname, **kwargs)
                    if path:
                        result.append(path)
            else:
                result += fnames
        elif isinstance(instance, Iterable):
            for inst in instance:
                result += get_paths(inst, initialize=initialize, **kwargs)
    return result


def assemble_pathensemble(*paths_or_pathensembles):
    from . import PathEnsemble
    pathensemble = PathEnsemble()
    for item in paths_or_pathensembles:
        if isinstance(item, Path):
            pathensemble._paths += item.split()._paths
        elif isinstance(item, PathEnsemble):
            pathensemble._paths += item._paths
        elif isinstance(item, Iterable):
            pathensemble += assemble_pathensemble(*item)
    return pathensemble
