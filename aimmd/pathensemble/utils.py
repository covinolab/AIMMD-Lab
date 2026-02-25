"""
aimmd.pathensemble.utils
=======================

Small utilities for :mod:`aimmd.pathensemble`.

This module provides helper functions used by the PathEnsemble implementation
and its mixins. The utilities here focus on two tasks:

1) Pattern matching on Path type labels.
2) Normalization of heterogeneous inputs into a flat list of Path objects
   (or, optionally, just trajectory filenames), and assembly of a PathEnsemble.

Path type patterns
------------------
AIMMD represents the categorical class of a path with a short type string.
Throughout :mod:`aimmd.pathensemble`, this is treated as a 4-character code.
Each character is a state label or a special marker.

`match_patterns` implements a simple matching language:

- pattern length is standardized to 4 by padding with '.' and truncation:
  ``(pattern + '...')[:4]``
- '.' is a wildcard that matches any character position
- any other character must match exactly

Multiple patterns are OR-ed together.

Path normalization
------------------
`get_paths` is used as a normalization entry point to accept common input forms:

- a Path instance,
- a PathEnsemble instance,
- a filename or glob pattern (resolved via :func:`aimmd.path.utils.get_fnames`),
- nested iterables mixing any of the above.

The function returns a flat list. By default it returns fully initialized
:class:`aimmd.path.Path` objects. With ``initialize=False`` it returns only
the resolved filenames (strings).

Ensemble assembly
-----------------
`assemble_pathensemble` merges multiple Path / PathEnsemble / iterable inputs
into a single PathEnsemble. When a Path is provided, it is first split into its
segments and the resulting sub-paths are appended.

Notes
-----
- This module does not perform I/O directly. Any file discovery is delegated to
  :func:`aimmd.path.utils.get_fnames`.
- The imports from MDAnalysis in this file are currently unused by the shown
  implementation but are kept as-is.
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
    """
    Match Path type strings against one or more patterns.

    Parameters
    ----------
    types : array-like of str
        Iterable of per-path type strings. Each type is interpreted as an
        iterable of characters. Only the first 4 characters are relevant.

    *patterns : str
        One or more pattern strings. Each pattern is standardized to 4
        characters with:

            (pattern + '...')[:4]

        A dot '.' is a wildcard matching any character in that position.
        Any other character must match exactly.

    Returns
    -------
    numpy.ndarray, dtype=bool
        Boolean mask of length ``len(types)``. An entry is True if the
        corresponding type matches at least one pattern.

    Notes
    -----
    Matching is performed position-by-position on the 4-character codes.
    Multiple patterns are combined with logical OR.
    """
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
    """
    Normalize heterogeneous inputs into a flat list of Path objects.

    Parameters
    ----------
    *instances : object
        Any mix of:
        - :class:`aimmd.path.Path`
        - :class:`aimmd.pathensemble.PathEnsemble`
        - str or :class:`pathlib.PosixPath` (filename or glob pattern)
        - iterable of any of the above (nested arbitrarily)

    initialize : bool, default=True
        Controls how filename inputs are handled:

        - True:
            Resolve filename(s) via :func:`aimmd.path.utils.get_fnames`,
            initialize :class:`aimmd.path.Path` objects for each filename,
            and include only truthy paths (``if path:``).

        - False:
            Resolve filename(s) and return the filenames (strings) directly,
            without initializing Path objects.

    **kwargs
        Keyword arguments forwarded to Path initialization:

            Path(fname, **kwargs)
            Path(existing_path, **kwargs)

        This is used to copy/initialize paths with consistent settings.

    Returns
    -------
    list
        Flat list of Path objects if `initialize=True`, otherwise a flat list
        of filenames (strings).

    Notes
    -----
    - When a Path instance is provided, it is copied via ``Path(instance, **kwargs)``.
    - When a PathEnsemble is provided, each stored path is copied similarly.
    - Nested iterables are handled recursively.
    """
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
    """
    Assemble a single PathEnsemble from multiple inputs.

    Parameters
    ----------
    *paths_or_pathensembles : object
        Any mix of:
        - :class:`aimmd.path.Path`
        - :class:`aimmd.pathensemble.PathEnsemble`
        - iterable of the above (nested)

    Returns
    -------
    PathEnsemble
        New PathEnsemble whose ``_paths`` is the concatenation of all provided
        paths, in the traversal order.

    Notes
    -----
    - When a Path instance is provided, it is first split with ``item.split()``
      and the resulting sub-paths are appended. This ensures that a multi-segment
      Path contributes its component segments as separate entries.
    - When a PathEnsemble is provided, its internal ``_paths`` list is appended
      directly (no copying).
    - When an iterable is provided, it is recursively assembled and added via
      ``pathensemble += assemble_pathensemble(*item)``.
    """
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
