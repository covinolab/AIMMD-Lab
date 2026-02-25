"""
aimmd.path._compute
==================

Batch computation and caching utilities for :class:`aimmd.path.Path`.

This module defines :class:`~aimmd.path._compute.PathCompute`, a mixin that
implements the high-level `Path.compute(...)` method.

Core responsibilities
---------------------
- Evaluate a user-supplied `function` on per-frame data extracted from a Path.
- Optionally write computed results into per-trajectory `.npy` cache files
  (one file per `(trajectory file, target)` pair).
- Support incremental computation:
  - skip frames that already have cached target values,
  - optionally skip frames based on `conditions`.
- Support chunked evaluation (`batch_size`) to reduce Python overhead and
  control memory usage.

Terminology
-----------
- **source**:
  Name of the time series used as input to `function`.
  Common sources include:
  - 'reader' (MDAnalysis reader slice; function is called per-timestep),
  - 'frames' (list of timesteps),
  - 'positions', 'coordinates', 'velocities', 'times', 'dimensions',
  - any other cached attribute stored as `.npy` (loaded via NPY_CACHE).

- **target**:
  Name of the time series that will be written to cache (e.g. 'states',
  'values', 'descriptors', ...). If empty, no cache file is written and the
  computed values are returned.

- **conditions**:
  Mapping `{reference_name: predicate}` where `predicate(series)` returns a
  boolean mask. Masks are applied per file-segment `k` and combined to decide
  which frames actually need computing.

Caching model
-------------
Cache files are per-trajectory-file (per `fname`) and per target. The cache file
name is produced by `get_cache_fname(fname, target)`.

Skipping behavior (important)
-----------------------------
If `target` is provided and `overwrite=False`, this method attempts to load
existing cached target values and skips frames that are already filled.

The "filled" test depends on the target type:
- If cached array is multidimensional: frame is considered filled if any
  element is non-zero (`~old.any(axis=1)`).
- If 1D and target != 'states': frame is considered filled if non-zero/True.
- If target == 'states': frame is considered filled if the string is not ''.

This convention allows sparse/incremental population of caches.

Notes on termination
--------------------
If `worker` is provided and has a truthy attribute `termination_signal`,
computation stops early and returns the partial result.

See also
--------
aimmd.path._extract.PathExtract._extract
    Low-level per-file extraction used by `compute`.
aimmd.path.utils.compute_batch
    Batch executor responsible for calling `function` and writing to cache.
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
        """
        Compute a time series on this Path, optionally caching the results.

        Parameters
        ----------
        function : callable or None, optional
            Computation function. Its expected calling convention depends on
            `source`:
            - If `source == 'reader'`, `compute_batch` will feed a reader slice
              (or timesteps) and `function` typically iterates over timesteps.
            - Otherwise, `function` is applied to numpy-like batches of the
              extracted source series.
            If `function is None`, no computation is performed and the method
            returns either `None` (if returning results) or 0 (if only updating).
        target : str, optional
            Name of the cache target series to write.
            - If non-empty: results are written to a `.npy` cache file per input
              trajectory file.
            - If empty: nothing is written and results are returned.
        source : str, optional
            Input series name used to generate inputs for `function`.
            Common values: 'reader', 'frames', 'positions', 'coordinates',
            'velocities', 'times', 'dimensions', or the name of an existing cached
            attribute.
        conditions : dict, optional
            Mapping `{reference_name: predicate}`. For each file-segment `k`,
            the method attempts to load `reference_name` via `_extract(k, ...)`
            and applies `predicate(...)` to obtain a boolean mask. Masks are
            multiplied together.
        overwrite : bool, optional
            If False (default), try to skip frames already present in the target
            cache file (if it exists and is recent enough).
        mtime : float or None, optional
            If provided, cached target files with modification time >= `mtime`
            are considered valid for skipping. At the end, the method sets the
            mtime of all touched target cache files to this value (via `os.utime`).
        batch_size : int, optional
            Maximum number of frames passed to `compute_batch` at once.
            Controls memory usage and amortizes Python overhead.
        return_result : bool, optional
            If True, return computed values as a numpy array.
            If False, return the number of computed frames (or another scalar
            summary returned by `compute_batch`).
            This is forced to True if `target` is empty.
        raise_if_error : bool, optional
            If True, propagate errors encountered when extracting condition
            references or sources. If False, missing references simply do not
            affect the mask (conditions are best-effort).
        verbose : bool, optional
            If True, show a tqdm progress bar indicating how many frames are
            processed/skipped/computed.
        worker : object, optional
            Optional worker/controller object. If it has a truthy attribute
            `termination_signal`, the method returns early.

        Returns
        -------
        numpy.ndarray or int
            - If `return_result` is True: numpy array of computed results
              (concatenated over all files and selected frames).
            - Otherwise: integer-like count accumulated from `compute_batch`.

        Notes
        -----
        - This method operates per underlying trajectory file (`self._fnames`),
          applying per-file masks and updating per-file cache targets.
        - When `target` is provided, this method removes the relevant entry from
          `NPY_CACHE` before writing to ensure subsequent reads see updates.
        """

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
                        
            # Cooperative termination: return partial results.
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
                # Frames skipped by mask are "already done" for the purpose of progress.
                progress.update((~mask).sum())
            
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
