"""
aimmd.path.utils
===============

Utility functions for the :mod:`aimmd.path` package.

The functions in this module are small, dependency-light helpers used by the
Path implementation and its mixins.

Main utilities
--------------
get_fnames
    Normalize filename inputs (strings, patterns, lists, txt/log lists) into a
    flat list of existing trajectory filenames.
get_last_time_and_dt
    Robustly obtain the last time and an estimate of the time step from a
    reader-like object.
get_cache_fname
    Map `(trajectory_fname, attribute)` pairs to the corresponding per-file
    `.npy` cache filename.
split
    Split a state string/array into contiguous segments where the internal state
    label changes (used to identify excursions and stopping conditions).
compute_batch
    Execute `function` on a concatenated batch and optionally update per-file
    `.npy` caches for the computed target.

Notes
-----
- Cache file updating is delegated to :func:`aimmd.cache.npy.update_npy`.
- `compute_batch` uses :class:`aimmd.path.chainreader.ChainReader` when the
  source is a reader-like stream, to avoid copying timesteps.
"""

# external
import os
import numpy as np
from glob import glob
from numbers import Number
from pathlib import PosixPath
from collections.abc import Iterable

# aimmd imports
from ..cache.npy import update_npy
from .chainreader import ChainReader

# utils function
def get_fnames(*patterns):
    """Expand one or more filename specifications into a flat list of filenames.

    Parameters
    ----------
    *patterns : str or pathlib.Path or Iterable
        One or more filename specifications. Each element may be:
        - a concrete filename,
        - a glob pattern (containing '*' or '?'),
        - a .txt/.log file listing filenames (whitespace-separated),
        - an iterable of any of the above.

    Returns
    -------
    list[str]
        Flat list of resolved filenames (strings).
    """
    result = []
    for pattern in patterns:
        if isinstance(pattern, (str, PosixPath)):
            pattern = str(pattern)
            if '*' in pattern or '?' in pattern:
                result += sorted(glob(pattern))
            elif pattern.endswith('txt') or pattern.endswith('log'):
                path = PosixPath(pattern)
                fnames = path.read_text().split() if path.exists() else []
                result += [str(path.parent / f) for f in fnames if f]
            else:
                result.append(pattern)
        elif isinstance(pattern, Iterable):
            for patt in pattern:
                result += get_fnames(patt)
    return result


def get_last_time_and_dt(reader, last_index):
    """Return the last time value and an estimate of the time step.

    Parameters
    ----------
    reader : object
        Reader-like sequence supporting `__getitem__` and returning either numeric
        times or objects with a `.time` attribute.
    last_index : int
        Index of the last frame to query.

    Returns
    -------
    tuple[float, float]
        `(t_last, dt)` where `t_last` is the time of `reader[last_index]` and `dt`
        is estimated from the previous frame if available (otherwise 1.0).
    """
    """Last time, dt"""
    t1 = reader[last_index]
    if not isinstance(t1, Number):
        t1 = t1.time
        if last_index:
            dt = t1 - reader[last_index - 1].time
        else:
            dt = 1.
    elif last_index:
        dt = t1 - reader[last_index - 1]
    else:
        dt = 1.
    return t1, dt


def get_cache_fname(fname, attribute):
    """Return the `.npy` cache filename for a given trajectory file and attribute.

    Parameters
    ----------
    fname : str
        Trajectory filename.
    attribute : str
        Attribute name (e.g. 'states', 'values', 'positions').

    Returns
    -------
    str
        Cache filename of the form `f"{fname}.{attribute}.npy"`.
    """
    """From fname to cache fname"""
    return f'{fname}.{attribute}.npy'


def split(states):
    """Split a state sequence into contiguous segments.

    Parameters
    ----------
    states : array-like
        Sequence of single-character state labels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        `(start, stop)` arrays, where each pair `(start[i], stop[i])` identifies a
        contiguous segment. The segmentation ignores the first and last frame when
        detecting changes (by design of the original algorithm).
    """
    states = np.asarray(states).astype('S1').view(np.uint8)
    diffs = np.diff(states[1:-1])
    # detect empty char-to-state crossings:
    # those which the absolute difference in uint8 space is > 60
    # diffs[(diffs > 60) * (diffs < 196)] = 0
    start = [0]
    stop = []
    for i in np.flatnonzero(diffs) + 1:
        start.append(i)
        stop.append(i + 2)
    stop.append(len(states))
    return np.array(start), np.array(stop)


def compute_batch(function,
                  batch_input, batch_targets,
                  source_is_reader, return_result=False,
                  system_id=None):
    """Compute a batch and optionally update per-file cache targets.

    Parameters
    ----------
    function : callable
        Function applied to the batch input. Must return an array-like result with
        one output per input frame.
    batch_input : list
        List of batch chunks. If `source_is_reader` is True, each element is a
        reader-like slice; otherwise each is a numpy array chunk.
    batch_targets : list[tuple[str, array-like]]
        List of `(cache_fname, locs)` pairs describing where to write the computed
        results for each chunk.
    source_is_reader : bool
        If True, treat the input as reader-like and wrap it in `ChainReader`.
    return_result : bool, optional
        If True, return the computed array; otherwise return the number of computed
        frames.
    system_id : hashable or None, optional
        Multi-system identifier. When not None, it is passed to `function` as a
        ``system_id=`` keyword (so one function can dispatch per system); when
        None, `function` is called with the data argument only — the
        single-system convention. The caller is responsible for passing None
        when `function` does not accept the keyword (see
        `aimmd.core.utils.accepts_system_id`).

    Returns
    -------
    numpy.ndarray or int
        Computed result array if `return_result` else the number of computed frames.
    """

    # compute batch
    if source_is_reader:
        data = ChainReader(*batch_input)
    else:
        data = np.concatenate(batch_input, axis=0)
    if system_id is None:
        result = np.asarray(function(data))
    else:
        result = np.asarray(function(data, system_id=system_id))
    
    # update targets
    begin = 0
    for fname, locs in batch_targets:
        end = begin + len(locs)
        update_npy(fname, result[begin:end], locs)
        begin = end
    
    # return
    if return_result:
        return result
    return len(result)
