"""
aimmd.worker.rollback
=====================

Roll a trajectory and its per-frame caches back to the checkpoint ``mdrun``
resumes from.

Why
---
AIMMD caches per-frame results beside each trajectory file, one
``<trajectory>.<series>.npy`` per series (``states``, descriptor series,
``values``, ``new``, ``bias``, ...), and takes row ``i`` of such a file for
the result of frame ``i`` whenever the row exists and is filled. It never
checks which frame a row was computed from. That holds as long as frame ``i``
of a file never changes, and GROMACS changes it whenever it resumes from a
checkpoint that is older than the last frame AIMMD has read:

- appending (shooting halves): ``mdrun -cpi`` truncates ``<deffnm>.xtc`` to
  the length the checkpoint recorded and simulates the frames after it again,
  under the same frame indices;
- ``-noappend`` (free runs): ``mdrun`` writes part ``s + 1`` from a checkpoint
  of part ``s``, over any part file of that name (``-nobackup``), while the
  later parts of the abandoned continuation stay on disk.

A checkpoint is a few frames behind its trajectory when a job died before it
could write its last checkpoint, and arbitrarily far behind when it was
replaced by an older copy, for example by syncing a downloaded copy of a run
folder back to the cluster.

What
----
Each time the worker is about to resume ``mdrun`` from ``<deffnm>.cpt``,
:func:`rollback_to_checkpoint` first makes the files agree with the
checkpoint:

- appending: the trajectory is cut to the length the checkpoint recorded,
  exactly as ``mdrun`` is about to cut it, every per-frame cache of it is cut
  to the frames that are left, and so is the PLUMED ``<deffnm>_COLVAR`` of a
  file-mode bias;
- ``-noappend``: every part numbered above the checkpoint's part is moved,
  with all its files (trajectory, caches, PLUMED ``_COLVAR`` slice, log,
  energies), into a new folder ``rolled_back_<UTC time>/`` beside it, so that
  nothing is deleted and no AIMMD glob sees them any more; the rows of those
  parts are moved out of the cumulative PLUMED ``COLVAR`` too;
- the worker's in-memory Path and this process' reader and array caches
  forget what was rolled back.

There is no threshold and no opt-out: the checkpoint decides how far back the
run goes. The worker log gets one entry per rolled-back trajectory, starting
with ``!!! CHECKPOINT ROLLBACK``, that says how far it went; when there is
nothing to roll back it gets one short ``...`` line. If the checkpoint cannot
be read, a loud ``!!! CHECKPOINT`` warning says so and nothing is rolled back.

A trajectory the worker finds complete is not resumed, so it is not rolled
back either.
"""

# external
import os
import re
import sys
import glob
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone

# aimmd imports
from .._config import print
from ..cache import forget
from ..cache.npy import truncate_npy
from ..engines.gromacs import (CheckpointError, read_checkpoint, gmx_command,
                               checksum_matches, format_ps)
from ..params._methods import (_part_row_ranges, _colvar_rowcount, _cut_colvar,
                               _append_colvar)

# first words of a rollback's log entry
ROLLBACK = '!!! CHECKPOINT ROLLBACK'

# sidecars named like a per-frame cache that do not have one row per frame
NOT_PER_FRAME = ('sweep_frame',)

_PART = re.compile(r'^(?P<base>.+)\.part(?P<part>\d{4})$')


@dataclass
class Rollback:
    """
    What :func:`rollback_to_checkpoint` found and did.

    Parameters
    ----------
    checkpoint : aimmd.engines.gromacs.Checkpoint or None
        The checkpoint ``mdrun`` resumes from (None if it could not be read).
    message : str
        The entry written to the worker log.
    kept : dict
        ``{trajectory file: frames kept}`` for every file that lost frames; a
        part moved aside keeps 0. Frames of an in-memory Path beyond these are
        gone (see :func:`truncate_path`).
    moved_to : str or None
        Folder the later parts of a ``-noappend`` run were moved into.
    """
    checkpoint: object = None
    message: str = ''
    kept: dict = field(default_factory=dict)
    moved_to: str = None

    @property
    def rolled_back(self):
        """Whether anything was cut or moved."""
        return self.message.startswith(ROLLBACK)


def frame_caches(trajectory):
    """
    The per-frame caches of a trajectory file.

    Parameters
    ----------
    trajectory : str
        Trajectory file, e.g. ``'run1/chainR0/back.xtc'``.

    Returns
    -------
    list of (str, str)
        ``(series, cache file)`` for every ``<trajectory>.<series>.npy`` on
        disk (states, descriptor series, values, new, bias, ...), sorted by
        name. Sidecars that do not hold one row per frame (``sweep_frame``)
        are left out.
    """
    trajectory = str(trajectory)
    result = []
    for fname in sorted(glob.glob(f'{glob.escape(trajectory)}.*.npy')):
        series = fname[len(trajectory) + 1:-len('.npy')]
        if series and series not in NOT_PER_FRAME:
            result.append((series, fname))
    return result


def truncate_path(path, kept):
    """
    Remove from `path`, in place, the first frame that was rolled back and
    every frame after it.

    Parameters
    ----------
    path : aimmd.path.Path
        A Path built by extending it from the files (the worker's).
    kept : dict
        ``{trajectory file: frames kept}`` (see :attr:`Rollback.kept`); a frame
        of one of these files at a local index at or beyond its count is gone.

    Returns
    -------
    int
        Number of frames removed.
    """
    kept = {os.path.abspath(str(fname)): int(n) for fname, n in kept.items()}
    position = 0
    for fname, first, last in zip(path._fnames, path._first, path._last):
        keep = kept.get(os.path.abspath(str(fname)))
        if keep is not None:
            if first <= last and last >= keep:
                return path.truncate(position + max(keep - first, 0))
            if first > last and first >= keep:
                return path.truncate(position)
        position += abs(last - first) + 1
    return 0


def _xdr_file(fname):
    """The libmdaxdr class that reads trajectory `fname`."""
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile, TRRFile
    readers = {'.xtc': XTCFile, '.trr': TRRFile}
    reader = readers.get(os.path.splitext(str(fname))[1].lower())
    if reader is None:
        raise TypeError(f'cannot roll back {fname!r}: AIMMD rolls back .xtc '
                        f'and .trr trajectories only')
    return reader


def _layout(fname, offset=None):
    """
    Frames of an XDR trajectory, read in one pass over its frame headers.

    Parameters
    ----------
    fname : str
        ``.xtc`` or ``.trr`` file.
    offset : int or None
        A byte offset; the result then says which frames lie before it.

    Returns
    -------
    dict
        ``n``: number of frames that can be read (an incomplete last frame
        does not count); ``n_before``: frames that start before `offset`
        (``n`` if `offset` is None); ``t_first`` and ``t_last``: times of
        frames ``0`` and ``n - 1``; ``t_after``: time of the first frame at or
        after `offset`; ``t_before``: time of the last frame before it (times
        in ps, None when there is no such frame).
    """
    reader = _xdr_file(fname)
    result = dict(n=0, n_before=0, t_first=None, t_last=None, t_after=None,
                  t_before=None)
    if not os.path.getsize(fname):
        return result

    def time(file, index):
        try:
            file.seek(index)
            return float(file.read().time)
        except Exception:
            return None

    try:
        file = reader(str(fname), 'r')
    except Exception:
        return result
    with file:
        try:
            starts = [int(start) for start in file.offsets]
        except Exception:  # not even one complete frame header
            return result
        n = len(starts)
        t_last = None
        while n and (t_last := time(file, n - 1)) is None:
            n -= 1
        if n:
            result.update(n=n, t_last=t_last, t_first=time(file, 0))
        n_before = n if offset is None else min(bisect_left(starts, offset), n)
        result['n_before'] = n_before
        if n_before < n:
            result['t_after'] = time(file, n_before)
        if n_before:
            result['t_before'] = time(file, n_before - 1)
    return result


def _frames(n):
    """'1 frame', 'N frames'."""
    return f'{n} frame{"s" if n != 1 else ""}'


def _span(first, last, checkpoint):
    """'t = a to b ps; X ns of simulated time past the checkpoint', as known."""
    span = []
    if first is not None and last is not None and first != last:
        span.append(f't = {format_ps(first)} to {format_ps(last)} ps')
    elif first is not None or last is not None:
        span.append(f't = {format_ps(last if first is None else first)} ps')
    if last is not None:
        past = last - checkpoint.time
        past = f'{past / 1000:.3f} ns' if abs(past) >= 1000 else f'{past:.1f} ps'
        span.append(f'{past} of simulated time past the checkpoint')
    return '; '.join(span)


def _refusal(fname, record):
    """
    Why ``mdrun`` will not append to output file `fname`, or None.

    GROMACS appends only if every output file the checkpoint lists is there,
    holds at least the recorded number of bytes, and has the recorded MD5
    checksum of the bytes before them (``record``, see
    :func:`aimmd.engines.gromacs.checksum_matches`).
    """
    if not os.path.exists(fname):
        return f'{fname} is missing'
    size = os.path.getsize(fname)
    if record.offset > size:
        return (f'it accounts for {record.offset} bytes of {fname}, which has '
                f'{size}')
    if checksum_matches(fname, record) is False:
        return (f'{fname} does not match it (the MD5 checksum of its '
                f'{record.checksum_size} bytes before byte {record.offset} '
                f'differs)')
    return None


def rollback_appended(trajectory, checkpoint, bias_file=False):
    """
    Cut a trajectory that ``mdrun`` appends to, and its caches, back to a
    checkpoint.

    ``mdrun -cpi`` appends only if each output file the checkpoint lists
    (log, trajectory, energies) holds at least the bytes the checkpoint
    recorded and their MD5 checksum matches; it then truncates the files to
    those lengths. This function checks the same, cuts the trajectory as
    ``mdrun`` is about to, and cuts every per-frame cache to the frames left
    (also a cache longer than the trajectory: its extra rows belong to frames
    that are gone).

    Parameters
    ----------
    trajectory : str
        The trajectory, e.g. ``'run1/chainR0/back.xtc'``.
    checkpoint : aimmd.engines.gromacs.Checkpoint
        The checkpoint ``mdrun`` resumes from.
    bias_file : bool, optional
        Whether PLUMED writes the bias to files (``record_bias=True`` and
        ``bias_source='file'``); the half's ``<deffnm>_COLVAR``, which holds
        one row per frame, is then cut too, after the rows of a session whose
        job was killed (still in the folder's ``COLVAR``) were appended to it.

    Returns
    -------
    Rollback
        Nothing is changed when the checkpoint lists no such output file, or
        GROMACS will refuse to append (an output file missing, shorter than
        recorded, or with another checksum); the message then says so.
    """
    trajectory = str(trajectory)
    name = os.path.basename(trajectory)
    cpt = os.path.basename(checkpoint.fname)
    resumes = f'{cpt} ({checkpoint.describe()})'
    output = checkpoint.output(trajectory)
    if output is None:
        return Rollback(checkpoint, f'!!! CHECKPOINT {resumes} lists no output '
                                    f'file {name}: nothing rolled back')
    folder = os.path.dirname(trajectory) or '.'
    for record in checkpoint.outputs:
        problem = _refusal(os.path.join(folder, record.name), record)
        if problem:
            return Rollback(checkpoint, (
                f'!!! CHECKPOINT {resumes}: {problem}, so GROMACS will refuse '
                f'to append; nothing rolled back'))
    size = os.path.getsize(trajectory)
    offset = output.offset

    # GROMACS records lengths at frame boundaries: the frames that start
    # before the offset are the ones mdrun keeps
    layout = _layout(trajectory, offset)
    n_frames, n_keep = layout['n'], layout['n_before']
    if size > offset:
        os.truncate(trajectory, offset)
    caches = []
    for series, cache in frame_caches(trajectory):
        old = truncate_npy(cache, n_keep)
        if old is not None and old > n_keep:
            caches.append(f'{series} ({old})')
    colvar = None
    if bias_file:
        fname = f'{os.path.splitext(trajectory)[0]}_COLVAR'
        session = os.path.join(os.path.dirname(trajectory) or '.', 'COLVAR')
        if os.path.exists(session):
            # the rows of a session whose job was killed before AIMMD could
            # append them (see aimmd.params._methods.run_simulation)
            _append_colvar(session, fname)
        if os.path.exists(fname):
            rows = _cut_colvar(fname, n_keep)
            if rows > n_keep:
                colvar = f'{os.path.basename(fname)} cut from {rows} to {n_keep} rows'

    if size == offset and not caches and colvar is None:
        return Rollback(checkpoint, (
            f'... {trajectory} ends at {resumes} ({n_keep} frames): nothing to '
            f'roll back'))
    forget(trajectory)

    entry = [f'{ROLLBACK} {trajectory}: mdrun resumes from {resumes}']
    if n_frames > n_keep:
        span = _span(layout['t_after'], layout['t_last'], checkpoint)
        entry.append(f'dropped the {_frames(n_frames - n_keep)} written after it'
                     + (f' ({span})' if span else '')
                     + f', {name} cut from {n_frames} to {n_keep} frames')
    elif size > offset:
        entry.append(f'cut {size - offset} bytes of an incomplete frame from '
                     f'{name}, which keeps {n_keep} frames')
    else:
        entry.append(f'{name} ends at the checkpoint ({n_keep} frames)')
    if caches:
        entry.append(f'caches cut to {n_keep} rows: {", ".join(caches)}')
    if colvar:
        entry.append(colvar)
    return Rollback(checkpoint, '; '.join(entry), kept={trajectory: n_keep})


def _new_folder(parent, stem='rolled_back'):
    """Create and return ``<parent>/<stem>_<UTC time>`` (made unique)."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = os.path.join(parent, f'{stem}_{stamp}')
    candidate, k = folder, 1
    while True:
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            k += 1
            candidate = f'{folder}_{k}'


def _owns_cumulative_colvar(deffnm, ext):
    """
    Whether the cumulative PLUMED COLVAR of the folder belongs to `deffnm`.

    PLUMED appends to one ``COLVAR`` per folder for as long as a free
    trajectory is resumed, and starts it afresh for each new trajectory, so it
    belongs to the newest trajectory in the folder (as in
    :func:`aimmd.pathensemble.bias_utils.derive_bias_from_cumulative_colvar`).
    """
    folder, base = os.path.split(deffnm)
    present = set()
    for fname in glob.glob(os.path.join(glob.escape(folder or '.'),
                                        f'*.part????{ext}')):
        match = _PART.match(os.path.basename(fname)[:-len(ext)])
        if match:
            present.add(match.group('base'))
    return bool(present) and base == max(present)


def _rollback_plumed(deffnm, ext, part, folder):
    """
    Move the PLUMED rows of rolled-back parts out of the folder's COLVAR files.

    Called once the parts after `part` are gone from beside `deffnm`. PLUMED
    writes one cumulative ``COLVAR`` per folder, which the free worker slices
    into per-part ``_COLVAR`` files after each ``mdrun``
    (:func:`aimmd.params._methods._split_cumulative_colvar`). It then keeps the
    rows of the parts still there (as their slices or frame counts account for
    them, see :func:`aimmd.params._methods._part_row_ranges`); the rows after
    them go to ``<folder>/COLVAR``. PLUMED appends the resumed part's rows
    after the kept ones, so its slice starts at the right row. Per-part
    ``COLVAR.partNNNN`` files that some PLUMED builds write instead, and that
    were not renamed yet, are moved for parts after `part`.

    Returns
    -------
    str or None
        What was done, for the log; None if nothing was changed.
    """
    directory, base = os.path.split(deffnm)
    directory = directory or '.'
    cumulative = os.path.join(directory, 'COLVAR')
    leftovers = [fname for fname in glob.glob(os.path.join(
                     glob.escape(directory), 'COLVAR.part[0-9][0-9][0-9][0-9]'))
                 if int(fname[-4:]) > part]
    if not (os.path.exists(cumulative) or leftovers):
        return None
    if not _owns_cumulative_colvar(deffnm, ext):
        return 'PLUMED COLVAR left alone (it belongs to a newer trajectory)'
    done = []
    for fname in sorted(leftovers):
        os.rename(fname, os.path.join(folder, os.path.basename(fname)))
    if leftovers:
        done.append(f'moved {len(leftovers)} COLVAR.partNNNN file'
                    f'{"s" if len(leftovers) != 1 else ""}')
    if os.path.exists(cumulative):
        ranges = _part_row_ranges(directory, base, ext, sys.maxsize)
        keep = max((stop for _, stop, _ in ranges.values()), default=0)
        rows = _colvar_rowcount(cumulative)
        if rows > keep:
            _cut_colvar(cumulative, keep, dropped=os.path.join(folder, 'COLVAR'))
            done.append(f'cumulative COLVAR cut from {rows} to {keep} rows (the '
                        f'others moved too)')
    return '; '.join(done) or None


def rollback_parts(deffnm, ext, checkpoint, bias_file=False):
    """
    Move the parts of a ``-noappend`` run that come after a checkpoint aside.

    ``mdrun -noappend`` resuming from a checkpoint of part ``s`` writes part
    ``s + 1``. Every part numbered above ``s`` belongs to a continuation that
    is being abandoned: all its files (``<deffnm>.partNNNN*``: trajectory,
    caches, PLUMED ``_COLVAR`` slice, log, energies) are moved into a new
    folder ``rolled_back_<UTC time>/`` beside them. The frames of part ``s``
    written after the checkpoint stay: the resumed part ``s + 1`` starts at the
    checkpoint time and supersedes them (see :meth:`aimmd.path.Path.extend`).

    Parameters
    ----------
    deffnm : str
        The run's ``-deffnm``, e.g. ``'run1/freeA/traj000001'``.
    ext : str
        Trajectory extension, e.g. ``'.xtc'``.
    checkpoint : aimmd.engines.gromacs.Checkpoint
        The checkpoint ``mdrun`` resumes from.
    bias_file : bool, optional
        Whether PLUMED writes the bias to files (``record_bias=True`` and
        ``bias_source='file'``): the rows of the moved parts are then moved
        out of the cumulative ``COLVAR`` as well.

    Returns
    -------
    Rollback
    """
    deffnm = str(deffnm)
    cpt = os.path.basename(checkpoint.fname)
    resumes = f'{cpt} ({checkpoint.describe()})'
    later = {}
    for fname in glob.glob(f'{glob.escape(deffnm)}.part[0-9][0-9][0-9][0-9]*'):
        number = int(fname[len(deffnm) + 5:len(deffnm) + 9])
        if number > checkpoint.part:
            later.setdefault(number, []).append(fname)
    if not later:
        return Rollback(checkpoint, (
            f'... {deffnm} resumes from {resumes} with part '
            f'{checkpoint.part + 1:04d}: nothing to roll back'))

    trajectories, n_frames, t_first, t_last = [], 0, None, None
    for number in sorted(later):
        fname = f'{deffnm}.part{number:04d}{ext}'
        if fname in later[number]:
            trajectories.append(fname)
            layout = _layout(fname)
            n_frames += layout['n']
            if layout['n']:
                if t_first is None:
                    t_first = layout['t_first']
                t_last = layout['t_last']

    folder = _new_folder(os.path.dirname(deffnm) or '.')
    files = 0
    for number in sorted(later):
        for fname in sorted(later[number]):
            os.rename(fname, os.path.join(folder, os.path.basename(fname)))
            files += 1
    forget(*trajectories)
    colvar = (_rollback_plumed(deffnm, ext, checkpoint.part, folder)
              if bias_file else None)

    numbers = sorted(later)
    parts = (f'part {numbers[0]:04d}' if len(numbers) == 1 else
             f'parts {numbers[0]:04d}-{numbers[-1]:04d}')
    entry = [f'{ROLLBACK} {deffnm}: mdrun resumes from {resumes} with part '
             f'{checkpoint.part + 1:04d}',
             f'moved the {files} files of {parts} ({_frames(n_frames)}'
             + (f'; {span}' if (span := _span(t_first, t_last, checkpoint))
                else '')
             + f') to {folder}']
    if colvar:
        entry.append(colvar)
    return Rollback(checkpoint, '; '.join(entry),
                    kept={fname: 0 for fname in trajectories}, moved_to=folder)


def rollback_to_checkpoint(deffnm, trajectory=None, params=None,
                           noappend=False):
    """
    Roll a run back to the checkpoint ``mdrun`` is about to resume from.

    Called by :meth:`aimmd.worker._simulate.WorkerSimulate._simulate` right
    before it resumes ``mdrun`` from ``<deffnm>.cpt``. Reads the checkpoint
    with ``gmx dump -cp`` (see :mod:`aimmd.engines.gromacs`), then
    :func:`rollback_appended` (shooting halves) or :func:`rollback_parts`
    (``-noappend`` free runs), writes the result to the log, and cuts the
    worker's in-memory Path to the frames that are left.

    Parameters
    ----------
    deffnm : str
        The run's ``-deffnm``; ``<deffnm>.cpt`` must exist.
    trajectory : aimmd.path.Path or None, optional
        The Path the worker extends from these files; frames that were rolled
        back are removed from it, in place.
    params : aimmd.Params or None, optional
        Gives the trajectory extension, the GROMACS command (from
        ``gmx_grompp``, see :func:`aimmd.engines.gromacs.gmx_command`) and
        whether the bias is recorded from PLUMED files.
    noappend : bool, optional
        True for ``-noappend`` runs (free simulations).

    Returns
    -------
    Rollback or None
        None if the checkpoint could not be read: then nothing is rolled back,
        and the log says so loudly.
    """
    ext = getattr(params, 'trajectory_extension', None) or '.xtc'
    cpt = f'{deffnm}.cpt'
    try:
        checkpoint = read_checkpoint(cpt, gmx_command(params))
    except CheckpointError as exception:
        print(f'!!! CHECKPOINT {cpt} could not be read, so nothing was rolled '
              f'back to it; if it is older than the trajectory, caches keep '
              f'rows of frames mdrun now simulates again: {exception}')
        return None
    bias_file = (bool(getattr(params, 'record_bias', False))
                 and getattr(params, 'bias_source', '') == 'file')
    if noappend:
        result = rollback_parts(deffnm, ext, checkpoint, bias_file)
    else:
        result = rollback_appended(f'{deffnm}{ext}', checkpoint, bias_file)
    print(result.message)
    if trajectory is not None and result.kept:
        truncate_path(trajectory, result.kept)
    return result
