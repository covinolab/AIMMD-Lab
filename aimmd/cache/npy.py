"""
aimmd.cache.npy
===============

Utilities for safe `.npy` persistence and a NumPy-array reader cache.

This module serves two related needs in AIMMD:

1) Robust persistence of NumPy arrays to `.npy` files in workflows where files
   may be read repeatedly and may also be written/updated incrementally.

2) Fast repeated access to `.npy` arrays via a size-limited, in-memory cache
   (:class:`NpyReaderCache`), created during package initialization (see
   :mod:`aimmd._init`) and exposed via :mod:`aimmd._config`.

Locking and atomicity
---------------------
All file operations are protected by a per-file lock located next to the data
file, using the naming convention:

- Data file: ``<folder>/<name>.npy``
- Lock file: ``<folder>/.<name>.lock``

Full writes use a temporary file + atomic replace:

1) Write to: ``<folder>/.<name>``
2) Replace:  ``os.replace(temp, fname)``

In-place updates: `update_npy`
------------------------------
`update_npy` updates selected indices along axis 0 without loading the full
array into memory. This is tailored to AIMMD usage and relies on assumptions:

- `.npy` header is treated as a fixed 128-byte region.
- Data begin at byte offset 128.
- Only simple (non-structured) dtypes are supported.

These constraints are preserved as-is.
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
    Safely save a NumPy array to `fname` using a lock + atomic replace.

    Parameters
    ----------
    fname : str
        Target `.npy` filename.
    array : numpy.ndarray
        Array to save.

    Side Effects
    ------------
    - Creates/uses lock file: ``.<name>.lock`` in the same folder.
    - Writes a temporary hidden file ``.<name>`` in the same folder.
    - Atomically replaces `fname` with the temporary file content.

    Notes
    -----
    Atomicity is provided by `os.replace`, which is atomic on POSIX filesystems
    when source and destination are on the same filesystem.
    """
    folder, name = extract_folder_and_name(fname)
    temp = f'{folder}/.{name}'
    lock = f'{folder}/.{name}.lock'
    with FileLock(lock):
        np.save(temp, array)
        os.replace(temp, fname)


def load_npy(fname, timeout=5.):
    """
    Safely load a NumPy array from `fname` under a per-file lock.

    Parameters
    ----------
    fname : str
        `.npy` file to load.
    timeout : float, default 5.0
        Seconds to wait for acquiring the lock before failing.

    Returns
    -------
    numpy.ndarray or None
        Loaded array, or None if:
        - the file does not exist, or
        - the lock cannot be acquired within `timeout`, or
        - `np.load` fails for any reason.

    Notes
    -----
    This function is intentionally permissive and catches all exceptions,
    returning None on failure (preserved behavior).
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
    Update selected axis-0 entries of a `.npy` file in place.

    Parameters
    ----------
    fname : str
        Target `.npy` file.
    data : array-like
        Data to write at `indices`.

        If `indices` is an integer, `data` is treated as a single-row update and
        wrapped in a list prior to `np.atleast_1d`.
    indices : int or array-like of int
        Indices along axis 0 to update.

    Raises
    ------
    RuntimeError
        - If `data.dtype` is structured (`len(dtype.descr) > 1`),
        - If dtype does not match the dtype stored in the file,
        - If trailing dimensions do not match the file's stored trailing shape.

    Side Effects
    ------------
    - Creates/uses lock file: ``.<name>.lock``.
    - May truncate/extend the `.npy` file to accommodate a larger axis-0 size.
    - Performs in-place writes to the underlying file data region.

    Notes
    -----
    This implementation is intentionally narrow and assumes:
    - header is read/written in the first 128 bytes,
    - data start at byte offset 128,
    - row size is computed as `itemsize * prod(shape[1:])`,
    - only simple dtypes are supported.

    These assumptions are preserved as they match AIMMD's controlled outputs.
    """
    """Please document very cool function
    Works also with indices integral, then data will be added another d."""
    
    # Normalize single-index update into a length-1 batch.
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
    
    # Create file if missing (full materialization).
    if not os.path.exists(fname):
        new_shape = (min_size,) + data_shape[1:]
        result = np.zeros(new_shape, dtype=data_dtype)
        result[indices] = data
        save_npy(fname, result)
    
    # Compute row byte size for in-place writes.
    rowsize = data.itemsize
    if len(data_shape) >= 1:
        rowsize *= np.prod(data_shape[1:])
    rowsize = int(rowsize)
    
    # Locked in-place update.
    folder, name = extract_folder_and_name(fname)
    with FileLock(f'{folder}/.{name}.lock'):
        with open(fname, "r+b") as file:
            header = file.read(128)
            
            # Check dtype descriptor in header.
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
            
            # Parse shape from header.
            shape_begin = header.find(b"'shape': (") + 9
            shape_end = shape_begin + header[shape_begin:].find(b'),') + 1
            shape = header[shape_begin + 1:shape_end - 1].decode('latin1')
            shape = tuple([int(s) for s in shape.split(',') if s.strip()])
            
            # Validate trailing dimensions.
            if shape[1:] != data_shape[1:]:
                raise RuntimeError(f'compute result must have '
                   f'shape {(-1, ) + shape[1:]}, got {data_shape} '
                   f'instead; consider deleting {fname!r} first')
            
            # Determine new axis-0 size after update.
            new_size = max(int(min_size), int(shape[0]))
            
            # Resize file if needed.
            if shape[0] != new_size:
                file.truncate(128 + new_size * rowsize)
            
            # Write each row.
            for i, rowdata in zip(indices, data):
                file.seek(128 + i * rowsize)
                file.write(rowdata.tobytes())
            
            # Rewrite header with updated shape (for robust np.load).
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
    Cache for arrays stored in `.npy` files.

    Behavior
    --------
    - Loads arrays via :func:`load_npy`.
    - Raises `TypeError` if the file cannot be loaded (preserved behavior).
    - Marks loaded arrays as read-only (`flags.writeable = False`).

    Attributes
    ----------
    max_size : int
        Heuristic cache budget set to currently available system memory at
        import time via `psutil.virtual_memory().available`.

    Notes
    -----
    The base cache accounts memory via `sys.getsizeof`, which can undercount
    NumPy memory usage. This behavior is preserved.
    """
    
    max_size = int(psutil.virtual_memory().available)

    def _open(self, fname):
        """
        Load a `.npy` file and return it as a read-only NumPy array.

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
            If loading fails (missing file, lock contention, corrupted file, etc.).
        """
        result = load_npy(fname)
        if result is None:
            raise TypeError(f'could not open {fname!r}')
        result.flags.writeable = False
        return result

    def _extend(self, instance, min_length):
        """
        Extend/pad a cached array to satisfy a minimum length.

        Parameters
        ----------
        instance : numpy.ndarray
            Cached array.
        min_length : int
            Desired minimum length along axis 0.

        Returns
        -------
        numpy.ndarray
            Extended array.

        Notes
        -----
        The current implementation delegates to `extend(instance, min_length)`.
        This symbol resolution is preserved exactly.
        """
        return extend(instance, min_length)
   
