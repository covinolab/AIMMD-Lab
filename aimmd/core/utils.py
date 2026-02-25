"""
aimmd.utils
===========

General-purpose helper functions used across AIMMD.

This module intentionally collects small, dependency-light utilities that are
reused across the package. These functions are designed to:

- stay independent of high-level AIMMD classes,
- be standalone and easy to test,
- avoid heavy imports of AIMMD internals,
- keep side effects explicit and localized.

The utilities here cover:
- indexing helpers for block-structured data,
- lightweight time formatting for logs,
- common list/array operations,
- filesystem convenience utilities,
- small simulation helpers (e.g., random velocity initialization),
- cache-related filename replacement,
- MDAnalysis trajectory helpers.

Design notes
------------
- Functions are generally pure unless their purpose is I/O (e.g., `remove`,
  `replace_in_cache`) or they explicitly sample randomness (e.g.,
  `randomize_velocities`).
- For performance and portability, implementations avoid complex dependencies.
- Some helpers assume conventions used throughout AIMMD (e.g., "offsets" as
  cumulative block boundaries).

Contents
--------
Indexing
- get_local_index: map a global index to (block_index, local_index) using offsets.

Time formatting
- now: current time string for logs.

List/array helpers
- cycle: rotate a list by n.
- concatenate: concatenate non-empty arrays.
- merge_ranges: merge overlapping integer ranges.

Filesystem helpers
- remove: remove files matched by patterns (supports wildcards).
- unique_path: generate a non-existing filesystem path by incrementing a suffix.

Simulation helpers
- randomize_velocities: Maxwell-Boltzmann sampling + COM removal.

Cache utilities
- replace_in_cache: rename files and keep cache keys consistent.

Trajectory helpers (MDAnalysis)
- memory_reader_from_timesteps: pack timesteps into a MemoryReader (time info caveat).
- process_state: normalize a state label (e.g., 'A', 'B', 'R') or numeric indices.
- extend_array: pad an array to a minimum length (immutable result).
- extract_folder_and_name: split a path into (folder, basename).
- guess_masses: heuristic mass assignment for Martini beads.

MDAnalysis dependency note
-------------------------
Some functions in the last section refer to `Timestep` and `MemoryReader` but do
not import them in this module. This is intentional in the current code layout:
the caller environment is expected to provide these names (typically via
MDAnalysis imports). If you want this module to be self-contained, add explicit
imports from MDAnalysis (but that would be a code change).
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

# -----------------------------------------------------------------------------------
# Time formatting
# -----------------------------------------------------------------------------------

def now():
    """
    Current local time formatted for log messages.

    Returns
    -------
    str
        Time formatted as ``"(Mon Feb 25 13:37:00 2026)"`` using `time.ctime()`.

    Notes
    -----
    - The output format is intentionally stable and human-readable, suitable for
      log files and HPC stdout.
    - Timezone and locale are inherited from the running environment.
    """
    # `time.ctime()` returns a locale-dependent but commonly readable format.
    return f'({time.ctime()})'


# -----------------------------------------------------------------------------------
# Indexing
# -----------------------------------------------------------------------------------

def get_local_index(i, offsets, clip=False):
    """
    Convert a global index into ``(block_index, local_index)`` given cumulative offsets.

    Parameters
    ----------
    i : int
        Global index to map.
    offsets : array-like of int
        Cumulative offsets defining block boundaries.

        Convention used by AIMMD:
        - `offsets[k]` is the starting global index of block `k`,
        - `offsets[-1]` is the first index *after* the last valid element
          (i.e., a total length).
    clip : bool, optional
        If False (default), raise IndexError when `i` is outside bounds.
        If True, clip to the closest valid index.

    Returns
    -------
    tuple[int, int]
        ``(block_index, local_index)``, where `local_index` is relative to the
        start of the selected block.

    Raises
    ------
    IndexError
        If `clip` is False and `i` is outside valid bounds.

    Notes
    -----
    This helper is used for block-structured trajectory storage, where a large
    logical index space is composed of multiple blocks (e.g., trajectory chunks).
    The function uses `bisect_right` to find the first block boundary strictly
    greater than `i`.
    """

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


# -----------------------------------------------------------------------------------
# List/array helpers
# -----------------------------------------------------------------------------------

def cycle(lst, n):
    """
    Rotate a list by `n` positions.

    Parameters
    ----------
    lst : list
        Input list.
    n : int
        Rotation offset. Positive values rotate left, negative rotate right.

    Returns
    -------
    list
        Rotated list.

    Notes
    -----
    - This is a convenience wrapper around slicing.
    - `n` is reduced modulo `len(lst)`.
    """
    """Change list first element, all the others will follow"""
    n %= len(lst)
    return lst[n:] + lst[:n]


def concatenate(arrays, **kwargs):
    """
    Concatenate arrays along axis 0, skipping empty arrays.

    Parameters
    ----------
    arrays : iterable of array-like
        Input arrays. Elements with ``len(array) == 0`` are ignored.
    **kwargs
        Passed to `numpy.concatenate`.

    Returns
    -------
    np.ndarray
        Concatenated array. If all inputs are empty, returns ``np.array([])``.

    Notes
    -----
    This function is useful when building datasets incrementally, where some
    components may be empty (e.g., no samples collected for a state).
    """
    arrays = [array for array in arrays if len(array)]
    if not len(arrays):
        return np.array([])
    return np.concatenate(arrays, axis=0, **kwargs)


def extend_array(instance, min_length):
    """
    Extend an array to at least `min_length` along axis 0 by zero-padding.

    Parameters
    ----------
    instance : np.ndarray
        Original array (assumed at least 1D).
    min_length : int
        Target minimum length.

    Returns
    -------
    np.ndarray
        If `instance` is already long enough, it is returned unchanged.
        Otherwise, a new zero-padded array is returned.

    Notes
    -----
    - The returned array is marked read-only (``flags.writeable = False``).
      This supports cache-like usage where consumers must not mutate the array.
    - Padding is applied only along axis 0; all trailing dimensions are preserved.
    """
    if len(instance) >= min_length:
        return instance
    result = np.zeros((min_length, *instance.shape[1:]),
                      dtype=instance.dtype)
    result[:len(instance)] = instance
    result.flags.writeable = False
    return result


def merge_ranges(ranges):
    """
    Merge overlapping integer ranges.

    Parameters
    ----------
    ranges : iterable of tuple(int, int)
        Input ranges as ``(begin, end)`` tuples. Ranges are merged when the next
        begin is ``<=`` the current end.

    Returns
    -------
    list[tuple(int, int)]
        Sorted and merged ranges.

    Notes
    -----
    - This is a generic interval merge helper; it does not enforce a specific
      inclusive/exclusive convention beyond the overlap test.
    - The input is sorted, so the function is O(n log n) due to sorting.
    """
    ranges = sorted(ranges)
    out = []
    for b, e in ranges:
        if out and b <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((b, e))
    return out


# -----------------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------------

def remove(*patterns, except_for=[], verbose=True):
    """
    Remove files matched by one or more patterns.

    Parameters
    ----------
    *patterns : str
        Filenames or glob-style patterns (wildcards supported).
        Examples: ``"foo.txt"``, ``"*.npy"``, ``"temp/[0-9]*.xtc"``.
    except_for : str or list[str], optional
        Filenames to skip even if matched.
    verbose : bool, optional
        If True, print status messages to stdout.

    Notes
    -----
    - Directories are skipped (`IsADirectoryError`).
    - Missing files are ignored (`FileNotFoundError`).
    - Patterns are expanded using `glob.glob` when they contain wildcard tokens.

    Side effects
    ------------
    Deletes files from disk.
    """
    # except_for must be a list
    if isinstance(except_for, str):
        except_for = [except_for]
    
    for pattern in patterns:
        
        # Expand wildcard patterns only when needed.
        if '*' in pattern or '?' in pattern or '[' in pattern:
            filenames = glob(pattern)
        else:
            filenames = [pattern]
        
        # Remove one by one, skipping exceptions and directories.
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
    Generate a unique filesystem path by incrementing a numeric suffix.

    Parameters
    ----------
    path : str or pathlib.Path
        Candidate path.
    extension : str, optional
        If provided, force this suffix (e.g., ``".xtc"``). If `path` does not
        already have this suffix, the extension is appended to the filename.

    Returns
    -------
    pathlib.Path
        A path that does not exist on disk.

    Notes
    -----
    - If `path` already exists, a numeric suffix is appended/updated:
      ``file -> file1 -> file2 -> ...``.
    - If the stem already ends with digits, they are treated as the initial
      counter and incremented.
    """
    path = PosixPath(path)
    if extension is not None and path.suffix != extension:
        path = path.with_name(f'{path.name}{extension}')
    
    n = 0
    while path.exists():
        
        # Find initial numeric suffix (if any) once.
        if not n:
            stem = path.stem
            i = len(stem)
            while stem[i - 1].isnumeric() and i:
                i -= 1
            stem = stem[:i]
            n = stem[i:]
            n = int(n) if n else 0
        
        # Update counter and path stem.
        n += 1
        path = path.with_stem(f'{stem}{n}')
    return path


# -----------------------------------------------------------------------------------
# Cache utilities
# -----------------------------------------------------------------------------------

def replace_in_cache(cache, old_name, new_name, prefixes=['']):
    """
    Rename file(s) on disk and update an in-memory cache key if present.

    Parameters
    ----------
    cache : object
        Cache instance exposing a dict-like attribute `cache._cache` mapping
        filenames to cached objects.
    old_name : str
        Old filename (relative to each prefix).
    new_name : str
        New filename (relative to each prefix).
    prefixes : list[str], optional
        Prefix strings prepended to the names. Typical usage is to provide
        directory prefixes like ``["run1/", "run2/"]`` or ``[""]`` (default).

    Notes
    -----
    - Each prefix is attempted independently.
    - If `os.replace` fails for a prefix (missing file, permission, etc.), that
      prefix is silently skipped.
    - If the old filename exists in the cache, the key is moved to the new name.

    Side effects
    ------------
    Renames files on disk via `os.replace`.
    """
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


# -----------------------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------------------

def randomize_velocities(masses, T):
    """
    Sample velocities from a Maxwell–Boltzmann distribution and remove COM motion.

    Parameters
    ----------
    masses : array-like
        Particle masses in atomic mass units (a.m.u.).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Velocities of shape ``(n, 3)``, in Angstrom/ps.

    Notes
    -----
    - Uses :math:`k_B = 0.0083144621` kJ/mol/K (gas constant in kJ/mol/K form).
    - The per-particle standard deviation is:

      ``std_i = sqrt(kB * T / m_i) * 10``

      where the factor 10 converts the internal unit scaling used by this code
      (A/ps) consistently with typical MD conventions in this project.
    - After sampling, the mass-weighted center-of-mass velocity is subtracted.

    Determinism
    -----------
    This function uses NumPy's global RNG. Control determinism by setting
    `np.random.seed(...)` in the caller.
    """
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


# -----------------------------------------------------------------------------------
# Trajectory helpers (MDAnalysis)
# -----------------------------------------------------------------------------------

def memory_reader_from_timesteps(*list_of_timesteps):
    """
    Build an MDAnalysis `MemoryReader` from one or more timesteps (or iterables).

    Parameters
    ----------
    *list_of_timesteps : Timestep or iterable
        Any combination of:
        - a single MDAnalysis `Timestep`,
        - an iterable of timesteps,
        - nested iterables of timesteps.

    Returns
    -------
    MemoryReader
        An in-memory trajectory-like reader containing positions, velocities,
        and dimensions.

    Notes
    -----
    - Time metadata is not preserved (as stated by the original code).
    - Missing velocities are filled with zeros for each timestep.
    - `dt` is taken from the last visited timestep.
    - This function expects `Timestep` and `MemoryReader` to be available
      in the module namespace (not imported here).
    """
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
    """
    Normalize a state label.

    Parameters
    ----------
    state : str or int
        One of:
        - a label in `allowed_states` (case-insensitive), e.g. ``'A'``, ``'B'``, ``'R'``;
        - a numeric index selecting from `allowed_states`, e.g. ``0 -> 'A'``.
        - a numeric string (e.g. ``"1"``) is interpreted as an integer index.
    allowed_states : str, optional
        Allowed state symbols in positional order.

    Returns
    -------
    str
        Normalized single-character state label.

    Raises
    ------
    TypeError
        If the provided state is not valid.

    Notes
    -----
    This helper makes state handling uniform across:
    - user-facing interfaces (strings),
    - internal indexing (integers),
    - serialized configurations (numeric strings).
    """
    if isinstance(state, str) and state.isnumeric():
        state = int(state)
    if isinstance(state, Integral):
        return allowed_states[state]
    state = str(state).upper()
    if state not in allowed_states:
        raise TypeError(f'{state} is not a valid state ({allowed_states})')
    return state


def extract_folder_and_name(fname):
    """
    Split a path string into ``(folder, name)``.

    Parameters
    ----------
    fname : str
        A filesystem path.

    Returns
    -------
    tuple[str, str]
        ``(folder, name)`` where:
        - `name` is the basename (final path component),
        - `folder` is the parent directory, or ``"."`` if not present.

    Notes
    -----
    This helper exists because several cache utilities build auxiliary files
    next to the target file (locks, temp files, offsets), and need a robust
    way to form sibling paths.
    """
    split_fname = fname.split('/')
    return '/'.join(split_fname[:-1]) or '.', split_fname[-1]


def guess_masses(atoms):
    """
    Heuristic mass assignment for an MDAnalysis AtomGroup.

    Parameters
    ----------
    atoms : AtomGroup
        MDAnalysis `AtomGroup`.

    Returns
    -------
    np.ndarray
        Per-atom masses.

    Notes
    -----
    - Martini heuristic:
      If any of the first ~50 atoms has a name starting with ``BB``, ``SC``,
      or ``PO4``, this function assumes a Martini-like coarse-grained system and
      returns a uniform mass of 72.0 for all beads.
    - Otherwise, returns `atoms.masses` as provided by the topology.

    Rationale
    ---------
    In several workflows it is preferable to assign a reasonable default mass
    rather than propagate missing/zero masses (which can destabilize velocity
    initialization and COM removal).
    """
    """atoms: atomsgroup
    martini: all get 72 by default, better than underestimating masses
    """
    for atom in atoms[:50]:
        if atom.name.startswith(('BB', 'SC', 'PO4')):
            # assign martini beads
            return np.full(len(atoms), 72.0)
    return atoms.masses
