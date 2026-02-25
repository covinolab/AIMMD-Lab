"""
aimmd.cache.npy
===============

Safe `.npy` file utilities and a lightweight `.npy` reader cache.

This module has two responsibilities:

1) Safe on-disk storage of NumPy arrays
   - `save_npy`: write an array atomically (temp file + replace), protected by a lock.
   - `load_npy`: read an array under the same lock.
   - `update_npy`: update selected rows of an existing `.npy` file in place,
     also protected by a lock.

2) Faster repeated reads of `.npy` files
   - `NpyReaderCache`: a small in-memory cache for arrays loaded from `.npy`
     files. It avoids repeated `np.load` on the same file.

Locking model
-------------
All operations use a per-file lock located next to the `.npy` file:

- Data: ``<folder>/<name>.npy``
- Lock: ``<folder>/.<name>.lock``

The lock prevents readers from loading a file while it is being written or
updated.

Atomic full writes (`save_npy`)
------------------------------
`save_npy` writes to a temporary file in the same directory and then replaces
the target file using `os.replace`. On typical POSIX filesystems, this makes
the write appear atomic to readers.

In-place updates (`update_npy`)
-------------------------------
`update_npy` is optimized for the case where you want to overwrite only some
rows (axis 0) without loading the full array into memory.

Important constraints (by design)
---------------------------------
`update_npy` is intentionally narrow and assumes:

- The `.npy` header is treated as a fixed 128-byte region.
- Array data are assumed to start at byte offset 128.
- Only *simple* dtypes are supported (no structured dtypes).
- Trailing dimensions (shape[1:]) must match between file and update data.

If these constraints are violated, `update_npy` raises a RuntimeError.

Rationale
---------
AIMMD commonly produces large arrays that are:
- appended/updated incrementally by one process,
- read frequently by others for monitoring or downstream computation.

This module provides robust semantics for that workflow.
"""

# external
import os
import numpy as np
import psutil
from numbers import Integral
from filelock import FileLock

# aimmd imports
from .base import AbstractCache
from ..core.utils import extract_folder_and_name, extend_array


def save_npy(fname, array):
    """
    Save an array to a `.npy` file safely (lock + atomic replace).

    Parameters
    ----------
    fname : str
        Target `.npy` filename.
    array : numpy.ndarray
        Array to write.

    Side Effects
    ------------
    - Creates/uses a lock file next to `fname`.
    - Writes a temporary hidden file next to `fname`.
    - Replaces `fname` atomically via `os.replace`.

    Notes
    -----
    This function is intended for complete rewrites of a file. For sparse
    row updates, use :func:`update_npy`.
    """
    folder, name = extract_folder_and_name(fname)
    temp = f'{folder}/.{name}'
    lock = f'{folder}/.{name}.lock'
    with FileLock(lock):
        np.save(temp, array)
        os.replace(temp, fname)


def load_npy(fname, timeout=5.):
    """
    Load an array from a `.npy` file safely (under a lock).

    Parameters
    ----------
    fname : str
        `.npy` file to read.
    timeout : float, default 5.0
        Maximum time (seconds) to wait for the file lock.

    Returns
    -------
    numpy.ndarray or None
        Loaded array, or None if:
        - the file does not exist,
        - the lock cannot be acquired,
        - the file cannot be loaded for any reason.

    Notes
    -----
    This function is intentionally permissive and returns None on failure.
    """
    if not os.path.exists(fname):
        return None
    try:
        folder, name = extract_folder_and_name(fname)
        lock = f'{folder}/.{name}.lock'
        with FileLock(lock, timeout=timeout):
            return np.load(fname)
    except:
        return None


def update_npy(fname, data, indices):
    """
    Update selected rows (axis 0) of a `.npy` file in place.

    Parameters
    ----------
    fname : str
        Target `.npy` file to update.
    data : array-like
        New values to write.
        - If `indices` is a single integer, `data` represents one row.
        - Otherwise, `data` must provide one row per index.
    indices : int or array-like of int
        Row indices to update along axis 0.

    Raises
    ------
    RuntimeError
        If any of the following occur:
        - structured dtype is provided (only simple dtypes supported),
        - dtype does not match the dtype already stored in the file,
        - trailing shape (shape[1:]) does not match the file.

    Side Effects
    ------------
    - Creates/uses a lock file next to `fname`.
    - Creates the file if it does not exist (by materializing a zero-filled array).
    - May grow the file if `max(indices)` exceeds the current length.
    - Writes updated rows directly into the underlying file.

    Notes
    -----
    This function assumes a specific `.npy` layout:
    - the header is read/written as the first 128 bytes,
    - data are assumed to start at byte offset 128.

    This matches AIMMD's controlled usage but is not a general `.npy` editor.
    """
    """Please document very cool function
    Works also with indices integral, then data will be added another d."""
    
    # process and get info
    if isinstance(indices, Integral):
        data = [data]
    data = np.atleast_1d(data)
    indices = np.asarray(indices).flatten()
    min_size = int(indices.max()) + 1
    data_shape = data.shape
    data_dtype = data.dtype
    data_descr = data_dtype.descr
    if len(data_descr) > 1:
        raise RuntimeError(f'only simple arrays allowed')
    
    # create
    if not os.path.exists(fname):
        new_shape = (min_size,) + data_shape[1:]
        result = np.zeros(new_shape, dtype=data_dtype)
        result[indices] = data
        save_npy(fname, result)
    
    # update in place
    # get row size
    rowsize = data.itemsize
    if len(data_shape) >= 1:
        rowsize *= np.prod(data_shape[1:])
    rowsize = int(rowsize)
    
    # go thrugh file
    folder, name = extract_folder_and_name(fname)
    with FileLock(f'{folder}/.{name}.lock'):
        with open(fname, "r+b") as file:
            header = file.read(128)
            
            # check descr
            descr_begin = header.find(b"'descr': ") + 9
            descr_end = header.find(b", 'fortran")
            descr = header[descr_begin:descr_end]
            data_descr = f"'{data_descr[0][1]}'".encode()
            if descr != data_descr:
                descr = descr.decode('latin1')
                data_descr = data_descr.decode('latin1')
                raise RuntimeError(f'compute result must have '
                   f'descr {descr}, got {str(data_descr)} '
                   f'instead; consider deleting {fname!r} first')
            
            # get shape
            shape_begin = header.find(b"'shape': (") + 9
            shape_end = shape_begin + header[shape_begin:].find(b'),') + 1
            shape = header[shape_begin + 1:shape_end - 1].decode('latin1')
            shape = tuple([int(s) for s in shape.split(',') if s.strip()])
            
            if shape[1:] != data_shape[1:]:
                raise RuntimeError(f'compute result must have '
                   f'shape {(-1, ) + shape[1:]}, got {data_shape} '
                   f'instead; consider deleting {fname!r} first')
            
            # update shape to final size
            new_size = max(int(min_size), int(shape[0]))
            
            # resize
            if shape[0] != new_size:
                file.truncate(128 + new_size * rowsize)
            
            # write rows frame by frame
            for i, rowdata in zip(indices, data):
                file.seek(128 + i * rowsize)
                file.write(rowdata.tobytes())
            
            # write header for last (more robust with np.load)
            if shape[0] != new_size:
                new_shape = (new_size,) + shape[1:]
                header = (header[:shape_begin] +
                          str(new_shape).encode('latin1') +
                          header[shape_end:])[:127] + b"\n"
                file.seek(0)
                file.write(header)
            
            file.flush()
            os.fsync(file.fileno())


class NpyReaderCache(AbstractCache):
    """
    In-memory cache for arrays loaded from `.npy` files.

    Behavior
    --------
    - Loads arrays using :func:`load_npy`.
    - Marks arrays as read-only before returning/caching them.
    - Uses the base cache eviction mechanism when the heuristic size budget is exceeded.

    Attributes
    ----------
    max_size : int
        Heuristic cache budget, set to available system memory at import time.

    Notes
    -----
    - The base class uses `sys.getsizeof`, which does not fully account for
      NumPy buffer memory. This cache budget is therefore approximate.
    - This cache is process-local (not shared across processes).
    """
    
    max_size = int(psutil.virtual_memory().available)

    def _open(self, fname):
        """
        Load `fname` and return a read-only NumPy array.

        Parameters
        ----------
        fname : str
            `.npy` filename.

        Returns
        -------
        numpy.ndarray
            Loaded array (read-only).

        Raises
        ------
        TypeError
            If the file cannot be loaded (missing file, lock timeout, corruption).
        """
        result = load_npy(fname)
        if result is None:
            raise TypeError(f'could not open {fname!r}')
        result.flags.writeable = False
        return result

    def _extend(self, instance, min_length):
        """
        Optionally extend/pad an array to satisfy a minimum length.

        Parameters
        ----------
        instance : numpy.ndarray
            Cached array.
        min_length : int
            Minimum required length along axis 0.

        Returns
        -------
        numpy.ndarray
            Extended array (read-only), or the original array if already long enough.
        """
        return extend(instance, min_length)
