"""
aimmd.pathensemble.utils
=======================

Small utilities for :mod:`aimmd.pathensemble`.

This module provides helper functions used by the PathEnsemble implementation
and its mixins. The utilities here focus on three tasks:

1) Pattern matching on Path type labels.
2) Normalization of heterogeneous inputs into a flat list of Path objects
   (or, optionally, just trajectory filenames), and assembly of a PathEnsemble.
3) Processing the results from `PathPosition` methods for evaluating
   `PathEnsemblePositions` methods.
4) Projecting a batch of values from pathensemble.

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

PathPosition methods processing
-------------------------------

`process_pathpositions_result` combinse the output of a `PathPosition` method
called on each Path in `self` to yield the output the associated
`PathEnsemblePosition` method.

- In most cases, it returns an NumPy array.
- If requesting "frames", it returns a MDAnalysis Reader.
- If requesting "self", it returns a PathEnsemble object.

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
from ..core.utils import memory_reader_from_timesteps
from ..path.utils import get_fnames
from ..path.chainreader import ChainReader


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

        This is used to copy/initialize paths with consistent settings.
        Only when calling on fname, not on a preexisting path!
    
    Returns
    -------
    list
        Flat list of Path objects if `initialize=True`, otherwise a flat list
        of filenames (strings).
    
    Notes
    -----
    - When a Path instance is provided, it is copied.
    - When a PathEnsemble is provided, each stored path is copied similarly.
    - Nested iterables are handled recursively.
    """
    from . import PathEnsemble
    from ..path import Path
    result = []
    for instance in instances:
        if isinstance(instance, Path):
            result.append(instance.copy())
        elif isinstance(instance, PathEnsemble):
            result += [path.copy() for path in instance._paths]
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


def process_path_position_result(result, attribute):
    """

    Take the list of results from a `PathPosition` methods and process it
    depending on the requested attribute.
    
    Used to combine the output of a `PathPosition` method called on each
    Path in `self` inside the associated `PathEnsemblePosition` position
    method.

    Parameters
    ----------

    result : list
        The `PathPosition` method evaluated on each Path in `self`

    attribute : str
        The 'attribute' parameter in the  `PathPosition` method 

    Returns
    -------
    array | MDAnalysis.core.trajectory.Reader | PathEnsemble
    
    Notes
    -----
    - If requesting "frames", it returns a MDAnalysis Reader.
    - If requesting "self", it returns a PathEnsemble object.
    - In all other cases, it returns an NumPy array.
    """
    if attribute == 'frames':
        return result
    if attribute == 'reader':
        return memory_reader_from_timesteps(result)
    if attribute == 'self':
        from . import PathEnsemble
        return PathEnsemble(result)
    if not len(result):
        if attribute == 'states':
            return np.zeros(0, dtype='<U1')
        else:
            return np.zeros(0, dtype=np.float32)
    return np.array(result)


def project_batch(bins, function, source,
                   batch_input, batch_weight):
    """
    Compute one histogram contribution from the currently buffered batch.

    Parameters
    ----------
    bins : list[array_like]
        Bin edges for each projected dimension (as required by
        ``np.histogramdd``).
    function : callable
        Transformation applied to the batch data before binning.
        It must accept either:
        - a concatenated NumPy array (if ``source != 'reader'``), or
        - a ChainReader (if ``source == 'reader'``),
        and return an array-like object with one row per frame.
    source : str
        Source stream name used to decide how to assemble batch data.
    batch_input : list
        Buffered per-frame data chunks (arrays or timesteps/readers).
    batch_weight : list
        Buffered per-frame weight chunks. Each element must be 1D and
        match the length of the corresponding element in `batch_input`.

    Returns
    -------
    numpy.ndarray
        Histogram counts for this batch, with shape
        ``(len(bins[0])-1, len(bins[1])-1, ...)``.

    Notes
    -----
    - The output of `function` is coerced to ``np.asarray`` and reshaped
      to ``(n_frames, -1)`` to match ``np.histogramdd`` input format.
    - Only the histogram counts are returned (index 0 of histogramdd).
    """
    if source == 'reader':
        data = ChainReader(*batch_input)
    else:
        data = np.concatenate(batch_input, axis=0)
    weights = np.concatenate(batch_weight)
    data = np.asarray(function(data))
    data = data.reshape((len(data), -1))
    return np.histogramdd(data, bins, density=False, weights=weights)[0]
