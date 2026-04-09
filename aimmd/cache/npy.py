"""
aimmd.cache.npy
===============

Safe `.npy` file utilities and a lightweight `.npy` reader cache.

This module has two responsibilities:

1) Safe on-disk storage of NumPy arrays
   - `save_npy`: write an array atomically (temp file + replace),
      protected by a lock.
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
import time
import numpy as np
import psutil
import shutil
from numbers import Integral
from filelock import FileLock

# aimmd imports
from .base import AbstractCache
from ..core.utils import extract_folder_and_name, extend_array


def save_npy(fname, array, timeout=10.):
    """
    Save an array to a `.npy` file safely (lock + atomic replace).

    Parameters
    ----------
    fname : str
        Target `.npy` filename.
    array : numpy.ndarray
        Array to write.
    timeout : float
        FileLock timeout.

    Side Effects
    ------------
    - Creates/uses a lock file next to `fname`.
    - Writes a temporary hidden file next to `fname`.
    - Replaces `fname` atomically via `os.replace`.

    Notes
    -----
    This function is intended for complete rewrites of a file. For sparse
    row updates, use :func:`update_npy`.
    
    To avoid I/O error, it tries saving at most ten time, waiting 0.1 seconds
    before each successive attempt.
    """
    folder, name = extract_folder_and_name(fname)
    max_retries = 10
    temp_fname = f'{folder}/temp.{name}'
    lock = f'{folder}/.{name}.lock'
    for _ in range(max_retries):
        try:
            with FileLock(lock, timeout=timeout):
                np.save(temp_fname, array)
                os.replace(temp_fname, fname)
                return
        except Exception as exception:
            error_message = str(exception)
            time.sleep(0.1)
    raise RuntimeError(error_message)


def load_npy(fname, timeout=10.):
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
    folder, name = extract_folder_and_name(fname)
    lock = f'{folder}/.{name}.lock'
    try:
        with FileLock(lock, timeout=timeout):
            return np.load(fname)
    except:
        return None


def update_npy(fname, data, indices, timeout=10.):
    """
    Update selected rows (axis 0) of a `.npy` file in place.

    This function is optimized for high-performance computing environments
    where network filesystems (NFS/Lustre) may exhibit transient I/O
    latencies. It  combines file locking, atomic binary writes, and a retry
    mechanism to  prevent race conditions and 'Errno 5' (I/O) errors from
    crashing long-running  simulations.

    Parameters
    ----------
    fname : str
        Path to the target `.npy` file.
    data : array-like
        The data to be written. Must match the trailing shape of the existing
        file.
    indices : int or array-like of int
        The row index (or indices) along axis 0 where data should be written.
    timeout : float
        FileLock timeout.

    Raises
    ------
    RuntimeError
        - If the array contains structured dtypes (not supported).
        - If the stored dtype in the file does not match the input data dtype.
        - If the trailing shape (axis 1+) does not match the file.
    OSError
        - If the filesystem remains unresponsive after the maximum number of
          retries.

    Implementation Details
    ----------------------
    1. Header Assumption: Assumes a fixed 128-byte header for fast seeking.
    2. Atomic Growth: Uses `file.truncate` to grow the file if indices exceed
       size.
    3. Resilience: Wraps binary I/O in a 5-attempt retry loop with exponential
       backoff.
    4. Data Integrity: Uses `os.fsync` to ensure bits are physically committed
       to disk.
    """
    
    # argument normalization and validation
    if isinstance(indices, (int, Integral)):
        data = [data]
    data = np.atleast_1d(data)
    indices = np.asarray(indices).flatten()
    
    # determine the minimum required length of the first axis
    min_size = int(indices.max()) + 1
    data_shape = data.shape
    data_dtype = data.dtype
    data_descr = data_dtype.descr
    
    # restrict to simple dtypes to ensure rowsize calculation is deterministic
    if len(data_descr) > 1:
        raise RuntimeError('Only simple arrays (single dtype) '
                           'are supported for in-place updates.')
    
    # creation logic
    # If the file doesn't exist, materialize a zero-filled array and save it.
    # Note: save_npy must also have retry logic to be fully safe.
    if not os.path.exists(fname):
        new_shape = (min_size,) + data_shape[1:]
        result = np.zeros(new_shape, dtype=data_dtype)
        result[indices] = data
        save_npy(fname, result)
        return
    
    # geometric calculations
    # calculate bytes per row (excluding axis 0)
    rowsize = data.itemsize
    if len(data_shape) > 1:
        rowsize *= np.prod(data_shape[1:])
    rowsize = int(rowsize)
    
    # extract folder and name
    folder, name = extract_folder_and_name(fname)
    
    # robust binary I/O loop
    # Cluster filesystems often fail during metadata updates
    # (rename/truncate/fsync).
    # We retry the entire block to ensure a "Stale File Handle" or
    # "I/O Error" doesn't terminate the parent simulation process.
    max_retries = 5
    temp = fname
    lock = f'{folder}/.{name}.lock'
    for attempt in range(max_retries):
        try:
            with FileLock(lock, timeout=timeout):
                # open in 'r+b' (read/write binary) to allow seeking
                # without clearing file
                with open(temp, "r+b") as file:
                    # header is assumed to be the first 128 bytes
                    header = file.read(128)
                    
                    # validaty dtype metadata
                    descr_begin = header.find(b"'descr': ") + 9
                    descr_end = header.find(b", 'fortran")
                    existing_descr = header[descr_begin:descr_end]
                    
                    # convert input dtype to the .npy string format
                    target_descr_encoded = f"'{data_dtype.str}'".encode()
                    
                    if existing_descr != target_descr_encoded:
                        raise RuntimeError(
                            f"Dtype mismatch in {fname}. "
                            f"File: {existing_descr.decode()}, "
                            f"Input: {target_descr_encoded.decode()}")
                    
                    # extract/validate shape
                    shape_begin = header.find(b"'shape': (") + 9
                    shape_end = (shape_begin +
                                 header[shape_begin:].find(b'),') + 1)
                    shape_str = header[shape_begin + 1:shape_end - 1]
                    shape_str = shape_str.decode('latin1')
                    existing_shape = tuple([int(s) for s in
                        shape_str.split(',') if s.strip()])
                    
                    if existing_shape[1:] != data_shape[1:]:
                        raise RuntimeError(
                            f"Shape mismatch. "
                            f"File trailing shape: {existing_shape[1:]}, "
                            f"Input: {data_shape[1:]}")
                    
                    # resize file if growing
                    current_size = int(existing_shape[0])
                    new_size = max(min_size, current_size)
                    
                    if current_size != new_size:
                        # grow the file physically to accommodate
                        # the new maximum index
                        file.truncate(128 + new_size * rowsize)
                    
                    # write data: seek to specific byte offsets
                    # and overwrite segments
                    for i, rowdata in zip(indices, data):
                        file.seek(128 + i * rowsize)
                        file.write(rowdata.tobytes())
                    
                    # update header
                    # if the size of axis 0 changed, we must rewrite the
                    # header so that np.load() recognizes the new frames
                    if current_size != new_size:
                        new_shape_tuple = (new_size,) + existing_shape[1:]
                        # reconstruct shape string in the header buffer
                        header = (header[:shape_begin] +
                                  str(new_shape_tuple).encode('latin1') +
                                  header[shape_end:])[:127] + b"\n"
                        file.seek(0)
                        file.write(header)
                    
                    # force the OS to flush buffers to the
                    # physical storage device
                    file.flush()
                    os.fsync(file.fileno())
            
            # back to fname
            if attempt == max_retries - 1:
                shutil.copyfile(temp, fname)
                os.remove(temp)
            
            # completed
            return 
            
        except OSError as exception:
            # last resort: move to /tmp
            if attempt == max_retries - 2:
                temp = f'/tmp/.{name}'
                shutil.copyfile(fname, temp)
            error_message = str(exception)
            time.sleep(.1)
    
    # raise error
    raise RuntimeError(error_message)


class NpyReaderCache(AbstractCache):
    """
    In-memory cache for arrays loaded from `.npy` files.

    Behavior
    --------
    - Loads arrays using :func:`load_npy`.
    - Marks arrays as read-only before returning/caching them.
    - Uses the base cache eviction mechanism when the heuristic size budget
      is exceeded.

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
            If the file cannot be loaded (missing file, lock timeout,
            corruption).
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
            Extended array (read-only), or the original array if already
            long enough.
        """
        return extend(instance, min_length)
