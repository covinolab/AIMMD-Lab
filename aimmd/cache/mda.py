"""
aimmd.cache.mda
===============

MDAnalysis reader cache for robust trajectory opening.

AIMMD frequently needs to open trajectories repeatedly (for analysis,
evaluation, or sampling). Some MDAnalysis trajectory formats maintain auxiliary
offset and lock files (e.g. XTC/TRR offsets), which can become stale and lead to
hangs or errors. This module mitigates that by:

- removing MDAnalysis-generated offset/lock files before opening,
- opening with `refresh_offsets=True`,
- detecting how many frames are safely readable,
- returning only the safe prefix of the trajectory.

Integration
-----------
This cache is constructed during :func:`aimmd._init.initialize` and stored as
``aimmd._config.MDA_CACHE`` for global reuse. :contentReference[oaicite:9]{index=9}
"""

# external
import os
import time
from MDAnalysis.coordinates.core import reader as Reader

# aimmd imports
from .base import AbstractCache
from ..core.utils import extract_folder_and_name

# auxiliary functions
def remove_offset_files(fname):
    """
    Remove MDAnalysis offset and lock files for a given trajectory file.

    Parameters
    ----------
    fname : str
        Trajectory file path.

    Notes
    -----
    Offset/lock files are typically named:
    - `.<name>_offsets.npz`
    - `.<name>_offsets.lock`

    This function loops until both files are absent, to handle scenarios where
    a file is recreated between checks.
    """
    """Avoid MDAnalysis loading to be stuck due to offsets"""
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
    Count how many frames are safely readable sequentially.

    Parameters
    ----------
    reader : MDAnalysis reader
        Opened reader supporting `__len__` and frame indexing.

    Returns
    -------
    int
        The largest `n` such that frames `[0..n-1]` are readable.

    Notes
    -----
    Some trajectories may have a corrupt tail (e.g. incomplete write). This
    function probes from the end backwards until a readable last frame is found.
    """
    """
    Count how many frames are safely readable sequentially.
    Stops at EOF or OSError.
    """
    n_frames = len(reader)
    while n_frames:
        try:
            # Probe last frame at current assumed length
            reader[n_frames - 1]
            return n_frames
        except (StopIteration, EOFError, OSError):
            # Step back on read failure
            n_frames -= 1
    if not n_frames:
        raise RuntimeError("No readable frames found")


class MDAReaderCache(AbstractCache):
    """
    Cache for MDAnalysis readers.

    Behavior
    --------
    - validates the file exists and extension is supported
    - removes stale offset/lock files
    - retries opening multiple times
    - returns a reader sliced to only safe frames

    Attributes
    ----------
    max_size : int
        Size budget (currently a placeholder heuristic).
    """

    """
    Open trajectories robustly and return an MDAnalysis Trajectory object
    containing only fully readable frames.
    """

    max_size = 48 * 1000  # placeholder for caching logic

    @staticmethod
    def _open(fname, ntries=10):
        """
        Open a trajectory file robustly.

        Parameters
        ----------
        fname : str
            Trajectory/topology filename.
        ntries : int, default 10
            Number of retry attempts.

        Returns
        -------
        MDAnalysis reader
            A reader (potentially sliced) containing only readable frames.

        Raises
        ------
        FileNotFoundError
            If `fname` does not exist.
        TypeError
            If the file extension is not supported.
        RuntimeError
            If opening fails after all retries.
        """
        # validate file
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname!r} does not exist")
        if not fname.endswith(('.trr', '.xtc', '.gro', '.pdb', '.dcd')):
            raise TypeError(f"{fname!r} extension not supported")

        # retry loop in case of temporary issues
        exception = ''
        for _ in range(ntries):
            try:
                # remove old offset files (robustness)
                remove_offset_files(fname)
                reader = Reader(fname, refresh_offsets=True)
                n_safe = count_safe_frames(reader)
                # slice only safe frames
                return reader[:n_safe]
            except Exception as exception:
                # Keep the most recent error message for final report
                exception = str(exception)
                time.sleep(1.0)
        
        raise RuntimeError(f"Could not open {fname!r} safely ({exception})")

    @staticmethod
    def _close(instance):
        """
        Close a cached MDAnalysis reader instance.

        Parameters
        ----------
        instance : MDAnalysis reader
            Reader to close.
        """
        instance.close()
