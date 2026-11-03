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

Unchanged files are not reopened
--------------------------------
While ``mdrun`` runs, the worker polls its trajectory about ten times a
second (the stop condition of
:meth:`aimmd.worker._simulate.WorkerSimulate._simulate`, called by
:func:`aimmd.execute.utils.execute_command`), and each poll asks
:meth:`aimmd.path.Path.extend` for up to ``trajectory_update_batch_size``
new frames, i.e. for a reader that many frames longer than the frames the
path already holds. The cached reader is almost never that long, so every
poll used to reopen the file, and every reopen reads every frame header of
the whole trajectory again (twice when its last frame is still being
written), competing with ``mdrun`` for the CPU, although a new frame arrives
only every second or so: the longer the trajectory, the slower ``mdrun``.

:meth:`MDAReaderCache.get` therefore returns the cached reader it loaded
itself, however short, as long as the file is unchanged since that reader
was opened: reopening an unchanged file gives a reader of the same safe
frames, so only the redundant rescans disappear. A file counts as unchanged
while its :func:`file_signature` (device, inode, size, modification and
status-change times) is the one taken just before the reader was opened.
Appending, truncating, replacing or touching the file changes it, and the
next poll then reopens the file exactly as before. Whenever the signature
cannot be taken or compared, :meth:`MDAReaderCache.get` does what
:meth:`aimmd.cache.base.AbstractCache.get` does, and it never opens a file
that :meth:`MDAReaderCache._open` would refuse by its name.

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

# the extensions MDAReaderCache._open opens (it refuses any other name)
TRAJECTORY_EXTENSIONS = ('.trr', '.xtc', '.gro', '.pdb', '.dcd')


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


def file_signature(fname):
    """
    Identify the current version of a file without reading its contents.

    Parameters
    ----------
    fname : str
        File path.

    Returns
    -------
    tuple
        ``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` of the file.

    Raises
    ------
    OSError
        If the file cannot be opened (e.g., it does not exist).

    Notes
    -----
    - The status is taken from a freshly opened descriptor (``os.open``,
      ``os.fstat``, ``os.close``), not from the path: on NFS-like file
      systems opening a file revalidates its cached attributes
      (close-to-open consistency), so the signature describes the file that
      a reopen would read.
    - Writing to a file changes its size or its modification time, and
      always its status-change time, which no user call can set back. A
      file replaced by another one has another inode: the reader cached for
      the old file keeps it open, so its inode number cannot be reused. Two
      versions of the same file can share a signature only if both were
      written within one tick of the file system clock and have the same
      size, which an appending ``mdrun``, a checkpoint rollback (which
      truncates) and AIMMD's replacements of a file never produce.
    """
    fd = os.open(fname, os.O_RDONLY)
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class MDAReaderCache(AbstractCache):
    """
    Cache for MDAnalysis readers.

    Behavior
    --------
    - Validates file exists and extension is supported.
    - Removes stale offset/lock files.
    - Retries opening multiple times.
    - Returns a reader sliced to only safe frames.
    - Does not reopen a file that is unchanged since :meth:`get` loaded its
      cached reader, however short that reader (see :meth:`get`).

    Attributes
    ----------
    max_size : int
        Size budget (currently a placeholder heuristic).
    _signatures : dict
        ``fname -> (signature, reader)``: the :func:`file_signature` of the
        file taken just before :meth:`get` loaded `reader`, the reader cached
        for it. An entry leaves with its reader.

    Notes
    -----
    `max_size` is not a byte-accurate budget for readers; it is used only to
    limit how many objects remain in the cache (heuristically).
    """

    max_size = 48 * 500  # ~500 Readers open

    def __init__(self):
        """
        Initialize an empty reader cache without file signatures.
        """
        super().__init__()
        # fname -> (signature, reader): the file just before reader was opened
        self._signatures = {}

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

    def _trajectory_signature(self, fname):
        """
        The :func:`file_signature` of `fname`, or None.

        None if the signature cannot be taken, and for any name that
        :meth:`_open` refuses by its extension, which is then not opened
        here either: a glob of a folder hands the ``.npy`` caches beside a
        trajectory to this cache too.
        """
        if not (isinstance(fname, str)
                and fname.endswith(TRAJECTORY_EXTENSIONS)):
            return None
        try:
            return file_signature(fname)
        except Exception:  # missing file, ...
            return None

    def get(self, fname, min_length=0, extend=False):
        """
        Retrieve a reader from cache, reopening the file only if it changed.

        As :meth:`aimmd.cache.base.AbstractCache.get`, except that a cached
        reader shorter than `min_length` is returned as it is, without
        reopening the file, if this method loaded it and the file is
        unchanged since: a reopen would give a reader of the same safe
        frames.

        Parameters
        ----------
        fname : str
            File path used as cache key.
        min_length : int, default 0
            Requested minimum length.
        extend : bool, default False
            If True, run :meth:`_extend` after retrieval.

        Returns
        -------
        object or None
            Cached/loaded reader, or None if open/load failed.

        Notes
        -----
        - A reader that is long enough is returned as before, without
          looking at the file.
        - Otherwise the file's signature is taken. If it is the one stored
          with the cached reader, that reader is returned. If not, the base
          class reloads the file as before, and the reader it loads gets the
          signature, taken just before the file was opened: a file that
          changes while it is being opened is reopened at the next request.
        - Only readers loaded here get a signature. A reader loaded by a
          direct :meth:`load` call (the toy engine reads one in another
          thread) is reloaded at the first request it is too short for, as
          before, and so never handed out for such a request.
        - A missing, unsized or inaccessible reader, a file whose signature
          cannot be taken, a name that is not a trajectory's: the base class
          behavior. :meth:`peek` is not affected.
        """
        instance = self._cache.get(fname, None)
        try:
            length = len(instance)  # TypeError if missing or unsized
            enough = length >= min_length
        except Exception:  # missing, unsized, inaccessible: reloaded
            length, enough = None, False
        if enough:
            return super().get(fname, min_length, extend)

        signature = self._trajectory_signature(fname)
        entry = self._signatures.get(fname)
        if (length is not None and signature is not None
                and entry is not None and entry[1] is instance
                and entry[0] == signature):
            if extend:
                instance = self._extend(instance, min_length)
            return instance

        # (re)load as before; a reader loaded now gets the signature
        result = super().get(fname, min_length, extend)
        loaded = self._cache.get(fname, None)
        if (signature is not None and loaded is not None
                and loaded is not instance):
            self._signatures[fname] = (signature, loaded)
        return result

    def load(self, fname):
        """
        As :meth:`aimmd.cache.base.AbstractCache.load`.

        A successful load replaces the cached reader and its signature (see
        :meth:`put`); the new reader has none, unless :meth:`get` loaded it.
        A failed load keeps the cached reader, as the base class does, but
        drops its signature: that reader is then reopened whenever it is too
        short.
        """
        instance = super().load(fname)
        if instance is None:
            self._signatures.pop(fname, None)
        return instance

    def pop(self, fname=None):
        """
        As :meth:`aimmd.cache.base.AbstractCache.pop`, dropping the signature
        of the popped reader.
        """
        if isinstance(fname, str):
            self._signatures.pop(fname, None)
        elif fname is None and self._cache:
            # the base class pops the oldest entry
            self._signatures.pop(next(iter(self._cache)), None)
        return super().pop(fname)

    def _discard(self, fname):
        """
        As :meth:`aimmd.cache.base.AbstractCache._discard`, dropping the
        signature of the evicted reader.
        """
        self._signatures.pop(fname, None)
        super()._discard(fname)

    def clear(self):
        """
        As :meth:`aimmd.cache.base.AbstractCache.clear`, dropping all
        signatures.
        """
        super().clear()
        self._signatures.clear()
