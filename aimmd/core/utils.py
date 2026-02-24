"""
...
"""

# external
import os
import time
import numpy as np
import bisect
from glob import glob
from numbers import Integral
from pathlib import PosixPath
from scipy.special import logit, expit

# functions
def get_local_index(i, offsets, clip=False):

    # nothing
    if not len(offsets):
        if not clip:
            raise IndexError(i)
        return 0, 0

    # before
    if i <= 0:
        if not clip and i < 0:
            raise IndexError(i)
        return 0, 0

    # after
    if i >= offsets[-1]:
        if not clip:
            raise IndexError(i)
        return offsets[-1] - 1
    
    # in between
    k = bisect.bisect_right(offsets[:-1], i)
    return k, i - offsets[k - 1] if k else i        


def convert_seconds(seconds):
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    seconds = max(seconds - .5, 0.)
    return f'{days} days, {hours:02g}:{minutes:02g}:{seconds:02.0f}'


def now():
    return f'({time.ctime()})'


def cycle(lst, n):
    """Change list first element, all the others will follow"""
    n %= len(lst)
    return lst[n:] + lst[:n]


def concatenate(arrays, **kwargs):
    arrays = [array for array in arrays if len(array)]
    if not len(arrays):
        return np.array([])
    return np.concatenate(arrays, axis=0, **kwargs)


def merge_ranges(ranges):
    ranges = sorted(ranges)
    out = []
    for b, e in ranges:
        if out and b <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((b, e))
    return out


def remove(*patterns, except_for=[], verbose=True):
    """Remove file(s) matched by a pattern (supports wildcards)."""

    if isinstance(except_for, str):
        except_for = [except_for]
    
    for pattern in patterns:
        
        # expand wildcard patterns
        if '*' in pattern or '?' in pattern or '[' in pattern:
            filenames = glob(pattern)
        else:
            filenames = [pattern]
        
        # remove one by one
        for filename in filenames:
            if filename in except_for:
                continue
            try:
                os.remove(filename)
                if verbose:
                    print(f'--- removed {filename}')
            except FileNotFoundError:
                if verbose:
                    print(f'--- {filename} not found')
            except IsADirectoryError:
                if verbose:
                    print(f'--- skipping directory {filename}')


def unique_path(path, extension=None):
    """
    gives unique name
    forces extension "extension"
    """
    path = PosixPath(path)
    if extension is not None and path.suffix != extension:
        path = path.with_name(f'{path.name}{extension}')
    
    n = 0
    while path.exists():
        
        # find n
        if not n:
            stem = path.stem
            i = len(stem)
            while stem[i - 1].isnumeric() and i:
                i -= 1
            stem = stem[:i]
            n = stem[i:]
            n = int(n) if n else 0
        
        # update n and path
        n += 1
        path = path.with_stem(f'{stem}{n}')
    return path


def randomize_velocities(masses, T):
    """
    masses : array-like of masses in a.m.u.
    T : temperature in K
    returns: array of velocities in angstroms/ps
    """
    
    # special case
    if T <= 0:
        return np.zeros((len(masses), 3))
    
    # Boltzmann constant in right unit
    kB = 0.0083144621  # kJ/mol/K
    
    # conversion
    masses = np.asarray(masses)
    std = np.sqrt(kB * T / masses) * 10  # A/ps
    
    # actual sampling
    velocities = np.random.normal(scale=std[:, None], size=(len(masses), 3))
    
    # center of mass removal
    velocities -= np.average(velocities, axis=0, weights=masses)
    return velocities


def replace_in_cache(cache, old_name, new_name, prefixes=['']):
    """only if existing; replaces also cache"""
    for fname in prefixes:
        old_fname = f'{fname}{old_name}'
        new_fname = f'{fname}{new_name}'
        try:
            os.replace(old_fname, new_fname)
        except:
            continue
        if old_fname in cache._cache:
            cache._cache[new_fname] = cache._cache.pop(old_fname)


def memory_reader_from_timesteps(*list_of_timesteps):
    """Attention! Time info lost."""
    
    # copy
    positions = []
    velocities = []
    dimensions = []
    dt = 1.
    
    def update_with(loc):
        nonlocal dt
        positions.append(loc.positions.copy())
        if hasattr(loc, 'velocities'):
            velocities.append(loc.velocities.copy())
        else:
            velocities.append(np.zeros((loc.n_atoms, 3)))
        dimensions.append(loc.dimensions.copy())
        dt = loc.dt
    
    def recurse(obj):
        # base case
        if isinstance(obj, Timestep):
            update_with(obj)
            return
        
        # recursive case
        try:
            for child in obj:
                recurse(child)
        except TypeError:
            raise TypeError(f'{obj!r} is not an (iterable of) Timestep')
    
    for obj in list_of_timesteps:
        recurse(obj)
    
    # (default) shape
    if len(positions):
        shape = (len(positions), *positions[0].shape)
    else:
        raise TypeError('no timesteps')

    # turn to array
    positions = np.array(positions).reshape(shape)
    velocities = np.array(velocities).reshape(shape)
    dimensions = np.array(dimensions).reshape((len(positions), 6))

    # create reader
    return MemoryReader(
        positions,
        velocities=velocities,
        dimensions=dimensions,
        dt=dt)


def process_state(state, allowed_states='ARB'):
    if isinstance(state, str) and state.isnumeric():
        state = int(state)
    if isinstance(state, Integral):
        return allowed_states[state]
    state = str(state).upper()
    if state not in allowed_states:
        raise TypeError(f'{state} is not a valid state ({allowed_states})')
    return state


def extend_array(instance, min_length):
    if len(instance) >= min_length:
        return instance
    result = np.zeros((min_length, *instance.shape[1:]),
                      dtype=instance.dtype)
    result[:len(instance)] = instance
    result.flags.writeable = False
    return result


def extract_folder_and_name(fname):
    split_fname = fname.split('/')
    return '/'.join(split_fname[:-1]) or '.', split_fname[-1]


def guess_masses(atoms):
    """atoms: atomsgroup
    martini: all get 72 by default, better than underestimating masses
    """
    # assign martini beads
    for atom in atoms[:50]:
        if atom.name.startswith(('BB', 'SC', 'PO4')):
            return np.full(len(atoms), 72.0)
    return atoms.masses
