"""
...
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
    Stops at EOF or OSError.
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
    Open trajectories robustly and return an MDAnalysis Trajectory object
    containing only fully readable frames.
    """

    max_size = 48 * 1000  # placeholder for caching logic

    def _open(self, fname):
        # validate file
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname!r} does not exist")
        if not fname.endswith(('.trr', '.xtc', '.gro', '.pdb', '.dcd')):
            raise TypeError(f"{fname!r} extension not supported")

        # retry loop in case of temporary issues
        exception = ''
        for _ in range(10):
            try:
                # remove old offset files (robustness)
                remove_offset_files(fname)
                reader = Reader(fname, refresh_offsets=True)
                n_safe = count_safe_frames(reader)
                # slice only safe frames
                return reader[:n_safe]
            except Exception as exception:
                exception = str(exception)
                time.sleep(1.0)
        
        raise RuntimeError(f"Could not open {fname!r} safely ({exception})")

    def _close(instance):
        instance.close()
