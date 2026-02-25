"""
aimmd.core.utils
================

General-purpose utility functions used across AIMMD.

This module contains lightweight helper functions that operate independently
of high-level AIMMD classes. Many of these utilities are related to:

- handling trajectory blocks and global/local frame indexing,
- manipulating filesystem paths and cached files,
- merging index ranges,
- zero-padding arrays,
- generating velocities for molecular simulations,
- parsing state labels,
- assembling in-memory trajectory readers.

Design philosophy
-----------------
- Keep utilities independent and easily testable.
- Avoid heavy AIMMD imports.
- Prefer pure functions.
- Make I/O side effects explicit.

Important concept: trajectory "blocks"
--------------------------------------
Several AIMMD workflows represent long trajectories as a concatenation of
multiple blocks (trajectory chunks). Functions such as `get_local_index`
translate between:

- global frame indices (as if all blocks were concatenated),
- local frame indices within a specific block.

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


# -----------------------------------------------------------------------------
# Indexing utilities
# -----------------------------------------------------------------------------

def get_local_index(i, offsets, clip=False):
    """
    Map a global frame index into a (block_index, local_index) within
    trajectory/blocks.

    In AIMMD, Paths are a concatenation of multiple **blocks** (trajectory
    chunks). Each block contains a contiguous piece of a trajectory.

    `offsets` specifies the cumulative end positions of these blocks in the
    global (concatenated) indexing scheme.

    Equivalently, PathEnsembles are a concatenation of multiple Paths, where
    `offsets` follows the same logic.

    Parameters
    ----------
    i : int
        Global frame index into the conceptual concatenation of all blocks.
    offsets : sequence of int
        Monotonically increasing cumulative end offsets.

        If block lengths are [L0, L1, L2], then:

            offsets = [L0, L0+L1, L0+L1+L2]

        Example
        -------
        Blocks [3, 5, 2] → offsets [3, 8, 10]

        Mapping:
        - i ∈ [0,1,2]   → block 0, local i
        - i ∈ [3..7]    → block 1, local i-3
        - i ∈ [8..9]    → block 2, local i-8

    clip : bool, default False
        - If False: raise IndexError for out-of-range i.
        - If True: clip i to nearest valid position.

    Returns
    -------
    tuple
        (block_index, local_index) in normal cases.

        NOTE: In the special clip=True and i ≥ offsets[-1] case,
        returns a single int (offsets[-1] - 1) to preserve legacy behavior.

    Notes
    -----
    Uses bisect for O(log n_blocks) block lookup.
    """

    # No blocks available
    if not len(offsets):
        if not clip:
            raise IndexError(i)
        return 0, 0

    # Index before first frame
    if i <= 0:
        if not clip and i < 0:
            raise IndexError(i)
        return 0, 0

    # Index beyond last frame
    if i >= offsets[-1]:
        if not clip:
            raise IndexError(i)
        return offsets[-1] - 1

    # Binary search to locate correct block
    k = bisect.bisect_right(offsets[:-1], i)

    # Convert to local index by subtracting previous offset
    return k, i - offsets[k - 1] if k else i


# -----------------------------------------------------------------------------
# Time formatting utilities
# -----------------------------------------------------------------------------

def convert_seconds(seconds):
    """
    Convert seconds into a formatted time string.

    Returns
    -------
    str
        Format: '<days> days, HH:MM:SS'
    """
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    # small bias avoids rounding 59.9999 → 60
    seconds = max(seconds - .5, 0.)

    return f'{days} days, {hours:02g}:{minutes:02g}:{seconds:02.0f}'


def now():
    """Return current local time as formatted string."""
    return f'({time.ctime()})'


# -----------------------------------------------------------------------------
# List and array manipulation
# -----------------------------------------------------------------------------

def cycle(lst, n):
    """
    Cyclically rotate a list by n positions.
    """
    n %= len(lst)
    return lst[n:] + lst[:n]


def concatenate(arrays, **kwargs):
    """
    Concatenate non-empty arrays along axis 0.
    """
    arrays = [array for array in arrays if len(array)]
    if not len(arrays):
        return np.array([])
    return np.concatenate(arrays, axis=0, **kwargs)


def merge_ranges(ranges):
    """
    Merge overlapping or adjacent (begin, end) intervals.
    """
    ranges = sorted(ranges)
    out = []

    for b, e in ranges:
        if out and b <= out[-1][1]:
            # Extend overlapping region
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((b, e))

    return out


# -----------------------------------------------------------------------------
# Filesystem utilities
# -----------------------------------------------------------------------------

def remove(*patterns, except_for=[], verbose=True):
    """
    Remove file(s) matching patterns (wildcards supported).
    """

    if isinstance(except_for, str):
        except_for = [except_for]

    for pattern in patterns:

        # Expand glob patterns
        if '*' in pattern or '?' in pattern or '[' in pattern:
            filenames = glob(pattern)
        else:
            filenames = [pattern]

        for filename in filenames:

            # Skip protected files
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
    Return a non-existing path by appending/incrementing numeric suffix.
    """

    path = PosixPath(path)

    if extension is not None and path.suffix != extension:
        path = path.with_name(f'{path.name}{extension}')

    n = 0

    while path.exists():

        if not n:
            stem = path.stem
            i = len(stem)

            # detect trailing integer
            while i and stem[i - 1].isnumeric():
                i -= 1

            base = stem[:i]
            num = stem[i:]
            n = int(num) if num else 0

        n += 1
        path = path.with_stem(f'{base}{n}')

    return path


# -----------------------------------------------------------------------------
# Simulation utilities
# -----------------------------------------------------------------------------

def randomize_velocities(masses, T):
    """
    Sample Maxwell-Boltzmann distributed velocities.

    Returns velocities in Å/ps and removes center-of-mass motion.
    """

    if T <= 0:
        return np.zeros((len(masses), 3))

    kB = 0.0083144621  # kJ/mol/K

    masses = np.asarray(masses)
    std = np.sqrt(kB * T / masses) * 10  # Å/ps conversion

    velocities = np.random.normal(scale=std[:, None], size=(len(masses), 3))

    # Remove center-of-mass drift
    velocities -= np.average(velocities, axis=0, weights=masses)

    return velocities


def replace_in_cache(cache, old_name, new_name, prefixes=['']):
    """
    Replace file name and maintain cache consistency.
    """

    for fname in prefixes:

        old_fname = f'{fname}{old_name}'
        new_fname = f'{fname}{new_name}'

        try:
            os.replace(old_fname, new_fname)
        except:
            continue

        # Update internal cache mapping if present
        if old_fname in cache._cache:
            cache._cache[new_fname] = cache._cache.pop(old_fname)


# -----------------------------------------------------------------------------
# Trajectory assembly utilities
# -----------------------------------------------------------------------------

def memory_reader_from_timesteps(*list_of_timesteps):
    """
    Construct an in-memory trajectory reader from Timestep objects.

    NOTE: Time metadata is not fully preserved.
    """

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
        if isinstance(obj, Timestep):
            update_with(obj)
            return

        try:
            for child in obj:
                recurse(child)
        except TypeError:
            raise TypeError(f'{obj!r} is not an (iterable of) Timestep')

    for obj in list_of_timesteps:
        recurse(obj)

    if not len(positions):
        raise TypeError('no timesteps')

    shape = (len(positions), *positions[0].shape)

    positions = np.array(positions).reshape(shape)
    velocities = np.array(velocities).reshape(shape)
    dimensions = np.array(dimensions).reshape((len(positions), 6))

    return MemoryReader(
        positions,
        velocities=velocities,
        dimensions=dimensions,
        dt=dt
    )


# -----------------------------------------------------------------------------
# State and array utilities
# -----------------------------------------------------------------------------

def process_state(state, allowed_states='ARB'):
    """
    Normalize state to single-character code.
    """

    if isinstance(state, str) and state.isnumeric():
        state = int(state)

    if isinstance(state, Integral):
        return allowed_states[state]

    state = str(state).upper()

    if state not in allowed_states:
        raise TypeError(f'{state} is not a valid state ({allowed_states})')

    return state


def extend_array(instance, min_length):
    """
    Extend array along axis 0 with zero padding (read-only result).
    """

    if len(instance) >= min_length:
        return instance

    result = np.zeros((min_length, *instance.shape[1:]),
                      dtype=instance.dtype)

    result[:len(instance)] = instance
    result.flags.writeable = False

    return result


def extract_folder_and_name(fname):
    """
    Split POSIX path into (folder, name).
    """

    split_fname = fname.split('/')
    return '/'.join(split_fname[:-1]) or '.', split_fname[-1]


def guess_masses(atoms):
    """
    Heuristically guess masses for Martini coarse-grained systems.

    If typical Martini bead names are detected, assign 72 a.m.u.
    """

    for atom in atoms[:50]:
        if atom.name.startswith(('BB', 'SC', 'PO4')):
            return np.full(len(atoms), 72.0)

    return atoms.masses
