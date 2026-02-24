"""
...
"""

# external
import os
import time
import numpy as np
import torch
import random
from math import inf
from pathlib import PosixPath

# aimmd imports
from ..path import Path
from .._config import NPY_CACHE, MDA_CACHE, print
from ..cache.npy import save_npy
from ..core.utils import now, process_state
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble
from ..execute.utils import execute_command
from ..network.utils import rescale
from ..analysis.utils import bin_centers
from ..pathensemble.utils import match_patterns, assemble_pathensemble

# functions
def update_selection_pool(pool, size, chain,
                          initial_paths=None,
                          at_least_one=''):
    """at least one: type"""
    
    # update with last ok path (weight != 0)
    if chain and (path := chain.path) and path.fname != pool.fname:
        # pool element that produced new path was already removed
        # thus we can just safely append to current pool
        pool.append(path)
        
    # replicate up to half size
    missing = max((size + 1) // 2 - len(pool), 0)
    if missing and initial_paths is not None:
        length = len(initial_paths)
        while missing:
            missing -= 1
            pool._paths.insert(0, initial_paths._paths[missing % length])
    
    # remove selected element in the chain or first one
    while len(pool) > size:
        pool.pop(0)

    # re-add transition if required
    if at_least_one and chain and (
        not pool.extract(at_least_one, at_least_one[::-1])):
        for path in chain._paths[::-1]:
            if path.type[:3] in (at_least_one, at_least_one[::-1]):
                pool._paths = [path] + pool._paths
                break
    
    return pool


def rescale_bins(bins, knots, values):
    """in place"""
    if not len(bins):
        return
    i = 0
    j = len(bins)
    a, b = bins[[i, j - 1]]
    if j - 1:
        if a == -inf and not np.isinf(bins[+1]):
            a = bins[+1]
            i = 1
        if b == +inf and not np.isinf(bins[-2]):
            b = bins[-2]
            j -= 1
    a, b = rescale([a, b], knots, values)
    bins[i:j] = np.linspace(a, b, j - i)


def register_path(path, chain, eneconv=None):
    """From two way shooting"""
    
    length = len(path)
    
    # get file names
    back_fname = path._fnames[0]
    back_fname = PosixPath(back_fname)
    folder = back_fname.parent
    ext = back_fname.suffix
    name = f'path{len(chain) + 1:06g}'
    temp_fname = f'{folder}/.{name}{ext}'
    fname = f'{folder}/{name}{ext}'
    fname_states = get_cache_fname(fname, 'states')
    fname_descr = get_cache_fname(fname, 'descriptors')
    
    # backward and forward
    forw_fname = f'{folder}/back{ext}'
    forw_fname = f'{folder}/forw{ext}'
    back_fname_states = get_cache_fname(back_fname, 'states')
    forw_fname_states = get_cache_fname(forw_fname, 'states')
    back_fname_descr = get_cache_fname(back_fname, 'descriptors')
    forw_fname_descr = get_cache_fname(forw_fname, 'descriptors')
    
    # only back
    if path.n_files == 1:

        # save states time series (delete cached)
        states = NPY_CACHE.pop(back_fname_states)[path.locs]
        save_npy(fname_states, states)
        
        # save descriptors time series (if existing)
        descriptors = NPY_CACHE.pop(back_fname_descr)
        if descriptors is not None:
            descriptors = descriptors[path.locs]
            save_npy(fname_descr, descriptors)
    
    else:  # backward and forward
        
        # get frame indices in backward and forward trajectories
        back_indices = path._extract(0, 'locs')
        forw_indices = path._extract(1, 'locs')
        
        # save states time series (delete cached)
        states = np.concatenate([
            NPY_CACHE.pop(back_fname_states)[back_indices],
            NPY_CACHE.pop(forw_fname_states)[forw_indices]])
        save_npy(fname_states, states)
                
        # save descriptors time series (if existing)
        descriptors = NPY_CACHE.pop(forw_fname_descr)
        if descriptors is not None:
            descriptors = np.concatenate([
                NPY_CACHE.pop(back_fname_descr)[back_indices],
                descriptors[forw_indices]])
            save_npy(fname_descr, descriptors)
        
        # merge energies (if existing)
        if eneconv:
            back_edr = f'{folder}/back.edr'
            forw_edr = f'{folder}/forw.edr'
            fname_edr = f'{folder}/{name}.edr'
            command = f'{eneconv} -f {back_edr} {forw_edr} -o {fname_edr}'
            execute_command(command, log_file='')
            print(f'+++ created {fname_edr}')
    
    # save/load trajectory through temp file
    # in this way, you minimize potential disruptions
    # from sudden interruptions
    # also: modify in place
    path.write(temp_fname, overwrite=True)
    os.rename(temp_fname, fname)
    path._fnames = [fname]
    path._first = [0]
    path._last = [length - 1]
    
    # remove cache (if existing) - important
    MDA_CACHE.pop(back_fname)
    MDA_CACHE.pop(forw_fname)
    
    # add to chain
    chain._paths.append(path)
    
    # report
    print(f'+++ added {fname} ({path.type[:3]}) '
          f'with {len(path)} frames {now()}')


def select_shooting_point(pool, params, folder,
                          chain=None,
                          free_trajectories=[],
                          target_state=1):
    """Slightly different algorithm (N computation).
    Target as before."""
    
    # params
    lorentzian = params.lorentzian
    states = params.states
    t = process_state(target_state, states)
    states = params.sorted_states
    
    # easy situation: internal shooting
    if t != states[1]:
        path = pool.pop()
        index = np.random.choice(path.internal('indices'))
        k, loc = path._get_local_loc(index)
        print(f'=== selecting frame {path._fnames[k]}, {loc}')
        return path[index]
    
    # next params
    free_overriding_states = params.free_overriding_states
    if free_overriding_states == 'all':
        free_overriding_states = '.'
    overriding_types = [f'{s}{t}' for s in free_overriding_states]
    overriding_attempts = params.free_overriding_attempts
    overriding_rate = params.free_overriding_recovery_rate
    compute_values_args = params.compute_values_args
    adjust_selection = params.adjust_selection_in_bins
    pool_size = params.selection_pool_size
    len_ext = len(params.trajectory_extension)
    nbins = params.nbins
    
    # process chain
    if chain is not None:
        chain = chain[chain.accepted]
    else:
        chain = PathEnsemble()

    # overriding configurations
    candidate_paths = PathEnsemble()
    for trajectory in free_trajectories:
        candidate_paths += trajectory.split().extract(*overriding_types)
    overriding = overriding_unique = candidate_paths.sample(
        overriding_attempts)
    if overriding_attempts:
        print(f'*** shortlisted {len(overriding)} overriding frames '
              f'from {len(candidate_paths)} candidates')
        overriding_unique = PathEnsemble(overriding).merge()
    
    # clear cache to avoid picking the wrong values
    NPY_CACHE.clear()
    
    # network parameters, bins, densities
    if nbins > 1:
        params.update_network(f'{folder}/..')
        bins, densities = params.load_bins_and_densities(f'{folder}/..')
        
        # compute only where there are no values (yet)
        n1 = pool.compute(*compute_values_args,
                          raise_if_error=True)
        n2 = overriding_unique.compute(*compute_values_args,
                                       raise_if_error=True)
        print(f'*** updated {n1 + n2} frame values ({n1} from pool)')
        # need to compute overriding values already here to be sure that
        # both pool and overriding values are evaluated on the same NN model
        # this is because I am not re-evaluating pre-existing values

    else:
        bins = np.array([-inf, +inf])
        densities = np.array([1.])
    
    # immediately get all values
    # (such that there is a lower risk of desync)
    pool_values = pool.values
    pool_shooting_values = pool.shooting('values')
    overriding_values = overriding.values
    populations = np.histogram(chain.shooting('values'), bins)[0]
    # shooting values have all been computed at this point

    report, histograms = pool.report(bins=bins, values=pool_values)
    print(f'Selection pool\n{report}')
    if nbins > 1:
        print(f'*** current pool shooting interfaces: {pool_shooting_values}')
    
    # prune if nothing
    norms = histograms.sum(axis=1)
    keepers = norms.astype(bool)
    if keepers.any() and not keepers.all():
        pool._paths = list(pool.paths[keepers])
        histograms = histograms[keepers]
        print(f'xxx removed {np.flatnonzero(~keepers)} paths from pool')
        
    # normalize histograms, average in "combined" histogram
    histograms /= norms[:, None]
    combined = histograms.mean(axis=0)
    
    # choose path
    pool_index = np.random.choice(len(pool))
    path = pool[pool_index]
    values = pool_values[pool_index]
    indices = path.internal('indices')
    locs = path.internal('locs')
    fname = path.fname
    print(f'=== selecting path {fname!r}')
    
    # assign selection probabilities
    histogram = histograms[pool_index]

    # report
    print(f'*** bins        {bins}')
    print(f'*** densities   {densities}')
    print(f'*** populations {populations}')

    # corrections
    densities *= populations + 1.
    if lorentzian < inf:
        centers = bin_centers(bins)
        densities *= 1 / (centers ** 2 + lorentzian ** 2)
        
    # weights
    mask = histogram > 0
    if mask.any():
        bin_weights = np.zeros(len(histogram))
        bin_weights[mask] = 1 / densities[mask]
        bin_weights /= bin_weights.sum()
        print(f'*** sel weights {bin_weights}')
        if adjust_selection:
            bin_weights[mask] *= combined[mask] / histogram[mask]
            bin_weights /= bin_weights.sum()
            print(f'   after adjust {bin_weights}')

        # select bin
        k = np.random.choice(len(bin_weights), p=bin_weights)
        print(f'=== selecting bin {k}: {bins[k:k+2]}')

        # select shooting point among candidates in bin
        candidates = np.flatnonzero(np.digitize(values, bins) - 1 == k)
        i = np.random.choice(candidates)
        
    else: # special situation: FFS-like
        i = np.argmin(np.abs(values))
        print(f'!!! outside bins range')
        k = None
    
    # get shooting point and report info
    shooting_point = path[indices[i]]
    loc = locs[i]
    if nbins > 1:
        value = values[i]
        print(f'=== selecting frame {loc} (value: {value:.3f})')
    else:
        print(f'=== selecting frame {loc}')
    
    # save params for TPS
    if params.chain_type == 'tps':
        torch.save(params.network.state_dict(),
                   f'{folder}/network{states}.h5')
        save_npy(f'{folder}/bins{states}.npy', bins)
        save_npy(f'{folder}/densities{states}.npy', densities)
        
    if overriding_types and k is not None:
        if np.digitize(pool_shooting_values[pool_index], bins) - 1 == k:
            if np.random.random() > overriding_rate:
                print(f'*** skipped overriding because the old SP is in the '
                      f'same bin (rec. rate = {overriding_rate})')
                overriding = None
            else:
                # on a very rare occasion: still override
                print(f'*** rescued overriding with '
                      f'rec. rate = {overriding_rate}')
    
    if overriding:
        candidates = np.flatnonzero(np.digitize(
            overriding_values, bins) - 1 == k)
        if len(candidates):
            i = np.random.choice(candidates)
            path = overriding
            loc = overriding.locs[i]
            name = overriding.filenames[i]
            shooting_point = path[i]  # ALL
            if nbins > 1:
                value = overriding_values[i]
                print(f'=== overriding with {name}, {loc} '
                      f'(value: {value:.3f})')
            else:
                print(f'=== overriding with {name}, {loc}')
        else:
            print(f'*** no overriding candidates in bin {k} '
                  f'({len(overriding)} frames in total)')
    
    # remove from pool
    if len(pool) >= pool_size:
        print(f'xxx removed {fname} from pool')
        pool.pop(pool_index)
        
    print(f'Shooting initialization completed {now()}\n')
    return shooting_point


def accept_or_reject_last_path(chain, params):
    """Executed when doing TPS."""

    # retrieve states info
    states = params.states

    current = chain[-1]
    current.weight = 0.
    leading = chain.path

    # if "current" is not a transition: reject
    if not chain[-1:].types(states, states[::-1])[0]:
        if leading:
            leading.weight += 1.
        print(f'=== acceptance probability: {0:.3f}')
        print('*** rejected')
        return

    # if "current" is the first sampled transition: accept
    if leading is None:
        current.weight = 1.
        print(f'=== acceptance probability: {1:.3f}')
        print('*** accepted')
        return
    
    # get model, bins, densities at the time of shooting point selection
    folder = PosixPath(chain.fname).parent
    params.update_network(folder)
    bins, densities = params.load_bins_and_densities(folder)
    
    # get bin weights at the time of selection
    # (population, lorentzian corrections were already incorporated
    #  in f'{folder}/density.npy')
    bin_weights = 1 / densities
    
    # get (internal) values
    source = 'descriptors' if params.descriptors_function else 'reader'
    batch_size = params.network_batch_size
    current_values = current[1:-1].compute(
        params.values_function, source=source, batch_size=batch_size)
    leading_values = leading[1:-1].compute(
        params.values_function, source=source, batch_size=batch_size)

    # get bins
    current_bin_indices = np.digitize(current_values, bins) - 1
    leading_bin_indices = np.digitize(leading_values, bins) - 1

    # selection biases
    current_selection_biases = bin_weights[current_bin_indices]
    leading_selection_biases = bin_weights[leading_bin_indices]
    current_selection_biases /= current_selection_biases.sum()
    leading_selection_biases /= leading_selection_biases.sum()

    # of shooting point
    current_shooting_point_bias = current_selection_biases[
        current.shooting_index - 1]
    leading_shooting_point_bias = leading_selection_biases[
        leading.shooting_index - 1]
    
    # compute acceptance probability
    acceptance = current_shooting_point_bias / leading_shooting_point_bias
    print(f'=== acceptance probability: {acceptance:.3f}')

    # finally run acceptance/rejection
    if np.random.random() < acceptance:
        current.weight = 1.
        print(f'*** accepted')
    else:
        leading.weight += 1.
        print(f'*** rejected')

def update_pathensemble(worker, *compute_args, **compute_kwargs):
    compute_kwargs.pop('worker', None)
    pathensemble = assemble_pathensemble(
        worker._shot_chains,
        worker._free_trajectories)
    n_frames = pathensemble.n_frames.sum()
    old_frames = getattr(worker, '_nframes', 0)
    if n_frames > old_frames and (
        compute_args or compute_kwargs):
        pathensemble.compute(*compute_args, **compute_kwargs)
    worker._nframes = n_frames
    return pathensemble, n_frames - old_frames
