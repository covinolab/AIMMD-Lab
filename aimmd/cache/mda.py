"""
aimmd.cache.mda
===============

MDAnalysis reader cache for robust trajectory opening.

AIMMD frequently needs to open trajectories repeatedly, potentially while they
are being written by other processes. Some MDAnalysis formats maintain
auxiliary offset and lock files (e.g., XTC/TRR offsets), which can become stale
and lead to hangs or errors.

This module mitigates that by:
- removing MDAnalysis-generated offset/lock files before opening,
- opening with `refresh_offsets=True`,
- detecting how many frames are safely readable,
- returning only the safe prefix of the trajectory.

Integration
-----------
This cache is constructed during :func:`aimmd._init.initialize` and stored as
``aimmd._config.MDA_CACHE`` for global reuse.

Notes
-----
This cache is process-local. It does not coordinate between processes beyond
filesystem-level effects (offset/lock file removal).
"""

# external
import os
import time
from MDAnalysis.coordinates.core import reader as Reader

# aimmd imports
from .base import AbstractCache
from ..core.utils import extract_folder_and_name


def remove_offset_files(fname):
    """
    Remove MDAnalysis offset and lock files for a given trajectory file.

    Parameters
    ----------
    fname : str
        Trajectory file path.

    Side Effects
    ------------
    Deletes auxiliary files next to the trajectory (if present), typically:
    - `.<name>_offsets.npz`
    - `.<name>_offsets.lock`

    Notes
    -----
    The removal is performed in a loop until both files are absent. This
    preserves the current behavior under concurrent creation/deletion.
    """
    # Avoid MDAnalysis loading to be stuck due to offsets.
    folder, name = extract_folder_and_name(fname)
    fname1 = f'{folder}/.{name}_offsets.npz'
    fname2 = f'{folder}/.{name}_offsets.lock'
    while os.path.exists(fname1) or os.path.exists(fname2):
        if os.path.exists(fname1):
            os.remove(fname1)
        if os.path.exists(fname2):
            os.remove(fname2)


def count_safe_frames(reader):
    """
    Determine how many frames are safely readable from an MDAnalysis reader.

    Parameters
    ----------
    reader : object
        An opened MDAnalysis reader supporting `__len__` and random access
        via `reader[i]`.

    Returns
    -------
    int
        The largest `n` such that frames `[0 .. n-1]` are readable.

    Raises
    ------
    RuntimeError
        If no readable frame can be found.

    Notes
    -----
    Algorithm (preserved):
    - Start from `len(reader)`.
    - Try to access the last frame.
    - On `(StopIteration, EOFError, OSError)`, decrement and retry.
    """
    n_frames = len(reader)
    while n_frames:
        try:
            reader[n_frames - 1]
            return n_frames
        except (StopIteration, EOFError, OSError):
            n_frames -= 1
    if not n_frames:
        raise RuntimeError("No readable frames found")


class MDAReaderCache(AbstractCache):
    """
    Cache for MDAnalysis readers.

    Behavior
    --------
    - Validates file exists and extension is supported.
    - Removes stale offset/lock files.
    - Retries opening multiple times.
    - Returns a reader sliced to only safe frames.

    Attributes
    ----------
    max_size : int
        Size budget (currently a placeholder heuristic).

    Notes
    -----
    `max_size` is not a byte-accurate budget for readers; it is used only to
    limit how many objects remain in the cache (heuristically).
    """

    max_size = 48 * 500  # ~500 Readers open

    def _open(self, fname, ntries=10):
        """
        Open a trajectory robustly and return a reader containing only safe frames.

        Parameters
        ----------
        fname : str
            Trajectory/topology filename.
        ntries : int, default 10
            Number of retry attempts.

        Returns
        -------
        object
            An MDAnalysis reader (potentially sliced) containing only readable frames.

        Raises
        ------
        FileNotFoundError
            If `fname` does not exist.
        TypeError
            If the file extension is not supported.
        RuntimeError
            If opening fails after all retries.

        Notes
        -----
        The reader is created with `refresh_offsets=True` to force MDAnalysis to
        rebuild offsets after auxiliary files are removed.
        """
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname!r} does not exist")
        if not fname.endswith(('.trr', '.xtc', '.gro', '.pdb', '.dcd')):
            raise TypeError(f"{fname!r} extension not supported")

        # Keep last exception message for diagnostics.
        exception = ''
        for _ in range(ntries):
            try:
                # Defensive cleanup for MDAnalysis-generated offset/lock files.
                remove_offset_files(fname)
                # Open reader with refreshed offsets.
                reader = Reader(fname, refresh_offsets=True)
                # Determine safe readable prefix.
                n_safe = count_safe_frames(reader)
                return reader[:n_safe]
            except Exception as exception:
                exception = str(exception)
                # Backoff to tolerate concurrent writes/renames.
                time.sleep(1.0)

        raise RuntimeError(f"Could not open {fname!r} safely ({exception})")

    def _close(self, instance):
        """
        Close a cached MDAnalysis reader instance.

        Parameters
        ----------
        instance : object
            Reader to close.

        Side Effects
        ------------
        Releases file handles associated with the reader.
        """
        instance.close()
