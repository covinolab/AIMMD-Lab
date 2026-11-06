"""Reading frames of trajectory files without leaving files next to them.

Private helpers of the analysis modules. MDAnalysis's readers of XTC and TRR
files keep a frame index next to the trajectory (``.<name>_offsets.npz`` and a
lock file), which would write into a run folder. Here those two formats are
read frame by frame with MDAnalysis's low-level file objects, which keep
nothing on disk; the other formats AIMMD accepts (``.gro``, ``.pdb``,
``.dcd``) are read with MDAnalysis's readers, which write no index for them.
"""

import os

import numpy as np

XDR_FORMATS = ('.xtc', '.trr')
"""The formats read with the low-level file objects."""


def _xdr_file(fname):
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile, XTCFile
    return (XTCFile if str(fname).lower().endswith('.xtc') else TRRFile)(str(fname), 'r')


def _dimensions(box_nm):
    from MDAnalysis.lib.mdamath import triclinic_box
    box = 10.0 * np.asarray(box_nm, dtype=np.float64)
    if not np.any(box):
        return None
    return np.asarray(triclinic_box(*box), dtype=np.float32)


def read_frames(fname, indices=None):
    """Positions (Å), boxes and times of some frames of a trajectory file.

    Parameters
    ----------
    fname : str or os.PathLike
        The trajectory.
    indices : sequence of int, optional
        The frames (0-based; default every frame). Negative indices count from the end.

    Returns
    -------
    tuple
        ``(positions (k, N, 3) float32 Å, dimensions (k, 6) float32 or None,
        times (k,) float64 ps)`` in the order of `indices`; ``dimensions`` is
        None when the file has no box.

    Raises
    ------
    IndexError
        If an index is beyond the last frame.
    """
    fname = os.fspath(fname)
    wanted = None if indices is None else [int(i) for i in indices]
    kept = {}
    if fname.lower().endswith(XDR_FORMATS):
        with _xdr_file(fname) as handle:
            n = len(handle) if wanted is None or min(wanted, default=0) < 0 else None
            if wanted is not None and n is not None:
                wanted = [i + n if i < 0 else i for i in wanted]
            last = (n - 1 if wanted is None else max(wanted, default=-1))
            needed = None if wanted is None else set(wanted)
            count = 0
            for count, frame in enumerate(handle, start=1):
                i = count - 1
                if needed is None or i in needed:
                    x = getattr(frame, 'x', None)
                    if getattr(frame, 'has_x', True) is False or x is None:
                        x = np.full((handle.n_atoms, 3), np.nan)
                    kept[i] = (10.0 * np.asarray(x, dtype=np.float32), _dimensions(frame.box),
                               float(frame.time))
                if last is not None and i >= last:
                    break
            n = count if n is None else n
    else:
        from MDAnalysis.coordinates.core import reader
        handle = reader(fname)
        try:
            n = handle.n_frames
            if wanted is not None:
                wanted = [i + n if i < 0 else i for i in wanted]
            needed = None if wanted is None else set(wanted)
            for ts in handle:
                if needed is None or ts.frame in needed:
                    kept[ts.frame] = (np.array(ts.positions, dtype=np.float32),
                                      None if ts.dimensions is None
                                      else np.asarray(ts.dimensions, dtype=np.float32),
                                      float(ts.time))
        finally:
            handle.close()
    if wanted is None:
        wanted = list(range(len(kept)))
    for i in wanted:
        if i not in kept:
            raise IndexError(f'frame {i} of {fname!r}, which has fewer frames')
    positions = (np.stack([kept[i][0] for i in wanted]) if wanted
                 else np.zeros((0, 0, 3), dtype=np.float32))
    picked = [kept[i][1] for i in wanted]
    dimensions = (None if not picked or any(b is None for b in picked)
                  else np.stack(picked).astype(np.float32))
    return positions, dimensions, np.array([kept[i][2] for i in wanted], dtype=float)


def count_frames(fname):
    """The number of frames of a trajectory file (nothing written next to it)."""
    fname = os.fspath(fname)
    if fname.lower().endswith(XDR_FORMATS):
        with _xdr_file(fname) as handle:
            return int(len(handle))
    from MDAnalysis.coordinates.core import reader
    handle = reader(fname)
    try:
        return int(handle.n_frames)
    finally:
        handle.close()


def frames_reader(positions, dimensions=None):
    """A reader over in-memory frames, the reader type AIMMD hands to user functions.

    Parameters
    ----------
    positions : array_like
        ``(k, N, 3)`` Å.
    dimensions : array_like, optional
        ``(6,)`` or ``(k, 6)`` box.

    Returns
    -------
    aimmd.path.chainreader.ChainReader
        A chain reader over one MDAnalysis ``MemoryReader``, as
        ``Path.compute`` gives a states, descriptors or collective-variable
        function.
    """
    from MDAnalysis.coordinates.memory import MemoryReader

    from ...path.chainreader import ChainReader
    x = np.asarray(positions, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    box = None if dimensions is None else np.asarray(dimensions, dtype=np.float32)
    return ChainReader(MemoryReader(x, dimensions=box))


def shooting_index_from_times(times, n_frames, nearest=False):
    """The shooting frame of a two-way path from its first two times, as ``aimmd.Path`` finds it.

    The backward half is stored reversed, so the times run down to 0 at the
    shooting frame: ``index = -t0 / (t1 - t0)`` (truncated, or rounded with
    `nearest`, as ``Params.round_shooting_index`` selects), clamped to the
    last frame; 0 when the times increase from the first frame.
    """
    if n_frames < 2:
        return 0
    t0, t1 = round(float(times[0]), 6), round(float(times[1]), 6)
    dt = t1 - t0
    if dt >= 0:
        return 0
    index = -t0 / dt
    index = int(round(index)) if nearest else int(index)
    return min(index, n_frames - 1)
