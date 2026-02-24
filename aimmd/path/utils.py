"""
...
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
    """From fname to cache fname"""
    return f'{fname}.{attribute}.npy'


def split(states):
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
                  source_is_reader, return_result=False):
    
    # compute batch
    if source_is_reader:
        data = ChainReader(*batch_input)
    else:
        data = np.concatenate(batch_input, axis=0)
    result = np.asarray(function(data))
    
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
