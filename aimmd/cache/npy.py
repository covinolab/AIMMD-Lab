"""
aimmd.cache.npy
===============

Utilities for safe `.npy` persistence and a NumPy-array reader cache.

This module serves two related needs in AIMMD:

1) **Robust persistence** of NumPy arrays to `.npy` files in workflows where
   files may be read repeatedly and may also be written/updated incrementally.

2) **Fast repeated access** to `.npy` arrays via a size-limited, in-memory cache
   (:class:`NpyReaderCache`), which is created during package initialization
   (see :mod:`aimmd._init`) and exposed via :mod:`aimmd._config`.

File locking and atomicity
--------------------------
All file operations are protected by a per-file lock located *next to* the data
file, using the naming convention:

- For data file:      ``<folder>/<name>.npy``
- Lock file:          ``<folder>/.<name>.lock``

The lock prevents concurrent readers and writers from corrupting state or seeing
partial writes.

Writing a full file is done in two steps:

1) Write to a hidden temporary file: ``<folder>/.<name>``
2) Atomically replace the target: ``os.replace(temp, fname)``

This pattern makes the operation robust against interruption: readers either see
the old complete file or the new complete file, but not a partially written one.

In-place updates: `update_npy`
------------------------------
:func:`update_npy` updates *selected indices along axis 0* of an existing `.npy`
file **without loading the full array into memory**. This is useful when a
large array is built incrementally.

However, the implementation is intentionally narrow and makes strong assumptions:

- The `.npy` header is treated as a fixed 128-byte region.
- Data are assumed to start at byte offset 128.
- Updates operate on a contiguous "row" defined as one slice along the first
  dimension (axis 0) with size:

    ``rowsize = itemsize * prod(shape[1:])``

- Only "simple" dtypes are supported:
  structured/compound dtypes are rejected via ``len(dtype.descr) > 1``.

This is a pragmatic implementation tailored to AIMMD's usage patterns; it is
**not** a general-purpose `.npy` editor.

Cache behavior: `NpyReaderCache`
--------------------------------
:class:`NpyReaderCache` loads arrays via :func:`load_npy` and:

- raises ``TypeError`` if the file cannot be loaded (current behavior),
- marks loaded arrays as read-only (`result.flags.writeable = False`) so cached
  arrays are not mutated accidentally.

`max_size` is set to currently available system memory at import time via
``psutil.virtual_memory().available`` (heuristic budget).

Notes
-----
- The module imports `extend_array` but does not call it in this file. This is
  preserved.
- `_extend` currently calls `extend(...)` (symbol resolution and behavior are
  preserved exactly).
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

    Side effects
    ------------
    - Creates/uses lock file: ``.<name>.lock`` in the same folder.
    - Writes a temporary hidden file ``.<name>`` in the same folder.
    - Atomically replaces `fname` with the temporary file content.

    Notes
    -----
    Atomicity is provided by `os.replace`, which is atomic on POSIX filesystems
    when source and destination are on the same filesystem (ensured here by
    writing the temp file into the same directory).
    """
    folder, name = extract_folder_and_name(fname)
    # Hidden temp file next to the target (same filesystem → atomic replace works)
    temp = f'{folder}/.{name}'
    # Per-file lock living next to the data file
    lock = f'{folder}/.{name}.lock'
    with FileLock(lock):
        # np.save writes a complete .npy file to the temporary location
        np.save(temp, array)
        # Atomic swap into place (readers either see old or new file)
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
    returning None on failure. Error reporting is handled by call sites when
    needed.
    """
    if not os.path.exists(fname):
        # Missing file is not an error in this API; caller can decide what to do
        return None
    try:
        folder, name = extract_folder_and_name(fname)
        lock = f'{folder}/.{name}.lock'
        with FileLock(lock, timeout=timeout):
            # Default np.load behavior (allow_pickle=False by default in modern NumPy)
            return np.load(fname)
    except:
        # Preserve current permissive behavior
        return None


# in-place update helper
def update_npy(fname, data, indices):
    """
    Update selected axis-0 entries of a `.npy` file *in place*.

    This function writes `data` into `fname` at `indices` along the first
    dimension (axis 0) without loading the entire array into memory.

    Parameters
    ----------
    fname : str
        Target `.npy` file.
    data : array-like
        Data to write at `indices`.

        - If `indices` is an integer, it is treated as a single-row update and
          `data` is wrapped in a list before converting to `np.atleast_1d`.
        - Otherwise `data` is coerced via `np.atleast_1d(data)`.

        The resulting `data` is expected to align with a selection along axis 0
        of the underlying stored array.
    indices : int or array-like of int
        Indices along axis 0 to update.

    Behavior
    --------
    **Case A: File does not exist**
    - Allocate a new array of zeros with shape:
        `(max(indices)+1,) + data.shape[1:]`
    - Fill `result[indices] = data`
    - Save using :func:`save_npy` (lock + atomic replace)

    **Case B: File exists**
    - Acquire lock ``.<name>.lock``.
    - Open file in binary update mode ``r+b``.
    - Read the first 128 bytes as `header`.
    - Validate dtype (`'descr'`) matches `data.dtype`.
    - Parse `'shape'` and validate trailing dimensions match `data.shape[1:]`.
    - Grow the file if `max(indices)+1` exceeds current axis-0 size.
    - For each `(i, rowdata)`:
        - seek to byte offset `128 + i * rowsize`
        - write `rowdata.tobytes()`
    - If resized, rewrite header `'shape'` field to reflect new axis-0 length.
    - Flush and `fsync` to force persistence.

    Constraints / assumptions
    -------------------------
    - The `.npy` header is treated as fixed length: 128 bytes.
    - Data start at offset 128.
    - Storage is compatible with row-wise writes computed by:
        `rowsize = itemsize * prod(shape[1:])`.
    - Structured dtypes are not supported (`len(dtype.descr) > 1` raises).

    Raises
    ------
    RuntimeError
        - if dtype differs from stored dtype,
        - if trailing shape differs from stored shape,
        - if dtype is structured (not supported).

    Notes
    -----
    This is a specialized utility for AIMMD-controlled `.npy` outputs. It is not
    intended to edit arbitrary `.npy` files produced elsewhere.
    """
    """Please document very cool function
    Works also with indices integral, then data will be added another d."""

    # Normalize single-index updates into a length-1 batch update
    if isinstance(indices, Integral):
        data = [data]

    # Coerce to at least 1D array (so `.shape` and iteration are well-defined)
    data = np.atleast_1d(data)

    # Flatten indices to a 1D integer array
    indices = np.asarray(indices).flatten()

    # The array must be at least this long along axis 0 to include max index
    min_size = int(indices.max()) + 1

    data_shape = data.shape
    data_dtype = data.dtype
    data_descr = data_dtype.descr

    # Reject structured/compound dtypes (multiple fields)
    if len(data_descr) > 1:
        raise RuntimeError(f'only simple arrays allowed')

    # Case A: create new file
    if not os.path.exists(fname):
        # Create a new array with axis-0 length sufficient for the requested indices
        new_shape = (min_size,) + data_shape[1:]
        result = np.zeros(new_shape, dtype=data_dtype)
        result[indices] = data
        save_npy(fname, result)

    # Case B: in-place update
    # Compute bytes per "row" (one index along axis 0)
    rowsize = data.itemsize
    if len(data_shape) >= 1:
        rowsize *= np.prod(data_shape[1:])
    rowsize = int(rowsize)

    folder, name = extract_folder_and_name(fname)
    with FileLock(f'{folder}/.{name}.lock'):
        with open(fname, "r+b") as file:
            # Read fixed-size header region
            header = file.read(128)

            # Validate dtype descriptor
            descr_begin = header.find(b"'descr': ") + 9
            descr_end = header.find(b", 'fortran")
            descr = header[descr_begin:descr_end]
            data_descr = f"'{data_descr[0][1]}'".encode()

            if descr != data_descr:
                # Decode for a readable error message
                descr = descr.decode('latin1')
                data_descr = data_descr.decode('latin1')
                raise RuntimeError(f'compute result must have '
                   f'descr {descr}, got {str(data_descr)} '
                   f'instead; consider deleting {fname!r} first')

            # Parse and validate shape
            shape_begin = header.find(b"'shape': (") + 9
            shape_end = shape_begin + header[shape_begin:].find(b'),') + 1
            shape = header[shape_begin + 1:shape_end - 1].decode('latin1')
            shape = tuple([int(s) for s in shape.split(',') if s.strip()])

            # Trailing dimensions must match the incoming data trailing dimensions
            if shape[1:] != data_shape[1:]:
                raise RuntimeError(f'compute result must have '
                   f'shape {(-1, ) + shape[1:]}, got {data_shape} '
                   f'instead; consider deleting {fname!r} first')

            # Decide final axis-0 size after update
            new_size = max(int(min_size), int(shape[0]))

            # Grow file if needed (header + new_size rows)
            if shape[0] != new_size:
                file.truncate(128 + new_size * rowsize)

            # Write row bytes for each requested index
            for i, rowdata in zip(indices, data):
                file.seek(128 + i * rowsize)
                file.write(rowdata.tobytes())

            # If resized, rewrite header shape for compatibility with np.load
            if shape[0] != new_size:
                new_shape = (new_size,) + shape[1:]
                header = (header[:shape_begin] +
                          str(new_shape).encode('latin1') +
                          header[shape_end:])[:127] + b"\n"
                file.seek(0)
                file.write(header)

            # Ensure data and header are persisted
            file.flush()
            os.fsync(file.fileno())


# Cache class
class NpyReaderCache(AbstractCache):
    """
    Cache for arrays stored in `.npy` files.

    Purpose
    -------
    Provide fast repeated access to disk-backed NumPy arrays while preventing
    accidental mutation of cached instances.

    Behavior
    --------
    - `_open(fname)` loads the array with :func:`load_npy`.
    - If loading fails (returns None), `_open` raises `TypeError`.
    - Successfully loaded arrays are marked read-only:
        `result.flags.writeable = False`

    Memory budget
    -------------
    `max_size` is set to the system's available memory at import time
    (via `psutil.virtual_memory().available`). This is a heuristic budget.

    Notes
    -----
    The base cache accounts size via `sys.getsizeof`, which may not reflect the
    true memory cost of NumPy arrays (data buffer size is not always included).
    """
    max_size = int(psutil.virtual_memory().available)

    def _open(self, fname):
        """
        Load an array from a `.npy` file and return it as read-only.

        Parameters
        ----------
        fname : str
            `.npy` filename.

        Returns
        -------
        numpy.ndarray
            Loaded array, with `writeable=False`.

        Raises
        ------
        TypeError
            If the array could not be loaded (file missing, lock timeout,
            corrupted file, etc.).

        Notes
        -----
        This method is intentionally strict (raises) so that the cache caller
        can distinguish "not loadable" from "empty array".
        """
        result = load_npy(fname)
        if result is None:
            raise TypeError(f'could not open {fname!r}')
        # Enforce immutability of cached arrays to prevent side effects
        result.flags.writeable = False
        return result

    def _extend(self, instance, min_length):
        """
        Extend/pad a cached array to at least `min_length`.

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
        Current implementation delegates to `extend(instance, min_length)`.
        This symbol is expected to exist in the runtime environment (as in the
        original codebase). This behavior is preserved exactly.
        """
        return extend(instance, min_length)
