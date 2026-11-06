"""How the analysis tools read a run folder: without writing next to its files,
and without the frames of biased simulations unless the caller asks for them.

Private helpers of :mod:`aimmd.analysis` and :mod:`aimmd.cgaimmd.analysis`.

Reading without writing
-----------------------
AIMMD's readers keep two kinds of hidden files next to the data they read:

- a file lock next to every cache file (``.<cache>.lock``), taken by the
  cache reader of :mod:`aimmd.cache.npy` for every read; in a folder the
  reader cannot write to (an archive, a read-only mount, files without write
  permission), the lock fails and the reader returns nothing, so the cache
  looks missing;
- MDAnalysis's frame index of every XTC and TRR file
  (``.<trajectory>_offsets.npz`` and its lock), which the reader cache of
  :mod:`aimmd.cache.mda` deletes and rebuilds whenever it opens a trajectory.

:func:`read_only_access` switches AIMMD's two global reader caches, for the
duration of a ``with`` block, to readers that write neither: cache files are
read with :func:`numpy.load` without the lock, and trajectories are opened
with readers that build their frame index in memory. What is read is the
same; only the hidden files are not touched. The lock only matters while a
run is being written: the workers replace whole cache files atomically, but
the trainer's value pass updates rows of a cache in place, so a read during
that pass can mix old and new values of one file. Numbers to report come from
a finished run (or a copy of one).

Run folders
-----------
The analysis tools never write into a run folder: the workers, the trainer
and a later restart own it. :func:`enclosing_run_folder` recognises one by
the subfolders AIMMD keeps there (shooting chains ``chainR0``, free
simulations ``freeA``, sweep chains ``sweepR0``), so an output path inside a
run folder is refused even when the caller does not name the run.

Frames of biased simulations
----------------------------
A bias changes which configurations a simulation visits and how it moves
between them. Frames of biased simulations (umbrella sampling, metadynamics,
OPES, steered pulls) are therefore not used as training data, as committor
references or for input scalers unless the caller decides so explicitly
(:func:`refuse_biased_frames`). A run is recognised as biased by
``params.record_bias``; a bias that the run does not record cannot be seen
here.
"""

import contextlib
import functools
import os
import re
import time

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# frames of biased simulations
# ════════════════════════════════════════════════════════════════════════════

def refuse_biased_frames(params, use, allow_biased_frames=False):
    """Refuse a run that records a bias as the source of `use`, unless the caller allows it.

    Parameters
    ----------
    params : aimmd.Params
        The run's params.
    use : str
        What the frames would become, for the message (``'training data'``,
        ``'a committor reference'``, ``'an input scaler'``).
    allow_biased_frames : bool, optional
        True: the caller has decided to use the frames anyway.

    Raises
    ------
    ValueError
        If ``params.record_bias`` is set and `allow_biased_frames` is False.
    """
    if allow_biased_frames or not bool(getattr(params, 'record_bias', False)):
        return
    raise ValueError(
        f'the run records a bias (params.record_bias): its simulations ran under the bias, and '
        f'frames of biased simulations are not used as {use} unless that is decided '
        f'explicitly. Pass allow_biased_frames=True to use them anyway (the docstring says when '
        f'that can be justified).')


# ════════════════════════════════════════════════════════════════════════════
# reading without writing next to the data
# ════════════════════════════════════════════════════════════════════════════

_XDR_EXTENSIONS = ('.xtc', '.trr')

_INDEX_FREE_READERS = {}
"""``{extension: reader class}``: MDAnalysis's XTC and TRR readers that keep their frame index in memory."""


def _index_free_reader(extension):
    """The MDAnalysis reader class of `extension` that never stores its frame index on disk."""
    reader = _INDEX_FREE_READERS.get(extension)
    if reader is None:
        from MDAnalysis.coordinates.TRR import TRRReader
        from MDAnalysis.coordinates.XTC import XTCReader
        base = XTCReader if extension == '.xtc' else TRRReader

        # no `format` attribute of its own: MDAnalysis registers only classes that define one,
        # so this class never replaces the reader MDAnalysis uses elsewhere
        class IndexInMemory(base):
            """Reads as its base class; the frame index stays in memory."""

            def _read_offsets(self, store=False):
                return super()._read_offsets(store=False)

        IndexInMemory.__name__ = IndexInMemory.__qualname__ = f'{base.__name__}IndexInMemory'
        reader = _INDEX_FREE_READERS[extension] = IndexInMemory
    return reader


def _open_trajectory(fname, ntries=10):
    """:meth:`aimmd.cache.mda.MDAReaderCache._open` without touching the frame-index files.

    The same reader of the same safe frames, with the same retries; the frame
    index is built from the file (as ``refresh_offsets=True`` does) and kept in
    memory.
    """
    from MDAnalysis.coordinates.core import reader as open_reader
    from ..cache.mda import TRAJECTORY_EXTENSIONS, count_safe_frames
    if not os.path.exists(fname):
        raise FileNotFoundError(f'{fname!r} does not exist')
    if not fname.endswith(TRAJECTORY_EXTENSIONS):
        raise TypeError(f'{fname!r} extension not supported')
    extension = os.path.splitext(fname)[1]
    exception = ''
    for _ in range(ntries):
        try:
            if extension in _XDR_EXTENSIONS:
                reader = open_reader(fname, format=_index_free_reader(extension),
                                     refresh_offsets=True)
            else:
                reader = open_reader(fname, refresh_offsets=True)
            return reader[:count_safe_frames(reader)]
        except Exception as error:                   # as the reader cache: retry a file in flux
            exception = str(error)
            time.sleep(1.0)
    raise RuntimeError(f'Could not open {fname!r} safely ({exception})')


def _open_array(fname):
    """:meth:`aimmd.cache.npy.NpyReaderCache._open` without the file lock (read-only array)."""
    try:
        result = np.load(fname, allow_pickle=False)
    except Exception as error:
        raise TypeError(f'could not open {fname!r} ({error})') from None
    result.flags.writeable = False
    return result


@contextlib.contextmanager
def read_only_access():
    """AIMMD's global readers, inside the block, read without writing next to the data.

    The cache reader (:data:`aimmd._config.NPY_CACHE`) loads cache files
    without its file lock, and the trajectory reader cache
    (:data:`aimmd._config.MDA_CACHE`) opens trajectories without deleting or
    writing MDAnalysis's frame-index files. Arrays and readers opened inside
    the block stay in the caches afterwards, as any other; they write nothing
    later either. Nested blocks are allowed (the outermost one restores).
    Not thread safe: meant for an analysis process, not for a running worker.
    """
    from .. import _config
    patched = []
    for cache, opener in ((getattr(_config, 'NPY_CACHE', None), _open_array),
                          (getattr(_config, 'MDA_CACHE', None), _open_trajectory)):
        if cache is not None and '_open' not in vars(cache):
            cache._open = opener
            patched.append(cache)
    try:
        yield
    finally:
        for cache in patched:
            vars(cache).pop('_open', None)


def without_writing(function):
    """Decorator: run `function` inside :func:`read_only_access`."""
    @functools.wraps(function)
    def reading(*args, **kwargs):
        with read_only_access():
            return function(*args, **kwargs)
    return reading


# ════════════════════════════════════════════════════════════════════════════
# run folders
# ════════════════════════════════════════════════════════════════════════════

_RUN_SUBFOLDER = re.compile(r'^(?:chain[A-Za-z]\d+|free[A-Za-z]|sweep[A-Za-z]\d+)$')
"""Names of the subfolders AIMMD keeps in a run folder: shooting chains ``chain{state}{k}``,
free simulations ``free{state}`` and sweep chains ``sweep{state}{k}``."""


def looks_like_run_folder(directory):
    """Whether `directory` holds AIMMD's run subfolders (``chainR0``, ``freeA``, ``sweepR0``, ...)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    return any(_RUN_SUBFOLDER.match(name) and os.path.isdir(os.path.join(directory, name))
               for name in names)


def enclosing_run_folder(target):
    """The innermost folder that holds `target` (or is `target`) and :func:`looks_like_run_folder`.

    Returns None when no folder on the way up to the root looks like a run folder.
    """
    path = os.path.realpath(os.path.abspath(os.fspath(target)))
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    while True:
        if looks_like_run_folder(directory):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent
