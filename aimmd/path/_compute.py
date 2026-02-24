"""
...
"""

# external
import os
import time
import numpy as np
import itertools
from abc import ABC
from math import inf
from tqdm import tqdm

# aimmd imports
from .utils import get_cache_fname, compute_batch
from .._config import NPY_CACHE

# compute methods for path
class PathCompute(ABC):
    """special methods for compute"""
    
    def compute(self, function=lambda x:[x.time for x in x],
                target='', source='reader', conditions={},
                overwrite=False, mtime=None, batch_size=4096,
                return_result=False, raise_if_error=False,
                verbose=False, worker=None):

        # process "return_result"
        return_result = return_result or not target
        
        # in case no computation is done
        if function is None:
            if return_result:
                return None
            return 0
        
        # compute in batches
        batch_input = []
        batch_targets = []  # info for batch
        current_size = 0
        result = [] if return_result else 0
        source = source or 'reader'
        source_is_reader = source == 'reader'
        targ_fnames = set()
        
        # loop over fnames
        progress = tqdm(total=len(self), disable=not verbose)
        for k, fname in enumerate(self._fnames):
                        
            if getattr(worker, 'termination_signal', False):
                if return_result:
                    return np.array(result)
                return result
            
            # initialize mask
            locs = self._extract(k, 'locs')
            mask = np.ones(len(locs), dtype=bool)
                        
            # get target and check mtime
            if target:
                targ_fname = get_cache_fname(fname, target)
                targ_fnames.add(targ_fname)
                if not overwrite:
                    old = NPY_CACHE.get(targ_fname, min_length=locs[-1])
                    # will remove later
                    if old is not None:
                        # update mask: do not compute where old is not "0"
                        if (mtime is None or
                            os.path.getmtime(targ_fname) >= mtime):
                            keepers = locs < len(old)
                            old = old[locs[keepers]]
                            if len(old.shape) > 1:
                                mask[keepers] &= ~old.any(axis=1)
                            elif target != 'states':
                                mask[keepers] &= ~old.astype(bool)
                            else:
                                mask[keepers] &= old == ''
                                                
            # apply conditions (if possible)
            for reference, condition in conditions.items():
                try:
                    mask *= condition(self._extract(k, reference)).flatten()
                except Exception as exception:
                    if raise_if_error:
                        raise exception
                        
            if verbose:
                progress.update(~mask.sum())
            
            # how much do you need to compute?
            if not mask.any():
                continue
            
            # different treatment to optimize loading speed
            if source in ('reader', 'frames', 'positions', 'times',
                          'coordinates', 'velocities', 'dimensions'):
                input_data = self._extract(
                    k, source, mask, raise_if_missing=raise_if_error)
                mask = np.flatnonzero(mask)
            else:
                try:
                    input_data = self._extract(
                        k, source, raise_if_missing=raise_if_error)
                except Exception as exception:
                    if raise_if_error:
                        raise exception
                mask = np.flatnonzero(mask[:len(input_data)])
                input_data = input_data[mask]
            
            remaining = mask.size
            if not remaining:
                continue
            
            # get locations: where you actually update the values
            locs = locs[mask]
            
            # remove from cache (will reload if needed)
            if target:
                NPY_CACHE.remove(targ_fname)
            
            # populate batches, compute every batch_size elements
            current = 0
            while remaining and (
                not getattr(worker, 'termination_signal', False)):
                delta = min(batch_size - current_size, remaining)
                current_slice = slice(current, current + delta, 1)
                batch_input.append(input_data[current_slice])
                if target:
                    batch_targets.append((targ_fname, locs[current_slice]))
                current += delta
                current_size += delta
                remaining -= delta
                if current_size >= batch_size:
                    this = compute_batch(function,
                                         batch_input, batch_targets,
                                         source_is_reader, return_result)
                    if return_result:
                        result.extend(this)
                    else:
                        result += this
                    if verbose:
                        progress.update(current_size)
                    current_size = 0
                    batch_input = []
                    batch_targets = []
        
        # last computation
        if current_size and not getattr(worker, 'termination_signal', False):
            this = compute_batch(function,
                                 batch_input, batch_targets,
                                 source_is_reader, return_result)
            if return_result:
                result.extend(this)
            else:
                result += this
        progress.update(progress.total - progress.n)
        progress.close()
        
        # set mtime of fname's target
        if mtime is not None:
            atime = time.time()
            for targ_fname in targ_fnames:
                os.utime(targ_fname, (atime, mtime))
        
        # return
        if return_result:
            return np.array(result)
        return result
