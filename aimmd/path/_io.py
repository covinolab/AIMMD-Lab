"""
aimmd.path._io
==============

Incremental I/O and trajectory growth utilities for :class:`aimmd.path.Path`.

This module defines :class:`PathIO`, a mixin providing two mutating operations:

- :meth:`PathIO.extend`  
  Incrementally grows an existing Path by
  1) optionally extending the **current last segment** (same filename),
  2) then appending frames from additional trajectory files.

- :meth:`PathIO.write`  
  Exports frames from a Path into a new trajectory file via MDAnalysis writers.

Overview of Path storage
------------------------
A Path is represented as an ordered sequence of *segments*:

- ``_fnames`` : list[str]  
  One trajectory filename per segment.
- ``_first`` / ``_last`` : list[int]  
  Local frame indices in each file (inclusive bounds).

Segments may be forward or backward in local index ordering. However,
``extend`` only supports extending **forward-running segments**.

Incremental growth semantics (extend)
-------------------------------------
The behavior of :meth:`PathIO.extend` is precise and state-dependent:

1) The method expands ``fnames`` using
   :func:`aimmd.path.utils.get_fnames`, producing an ordered list of
   candidate trajectory files.

2) The current last segment is extended **only if**:
   - the Path is non-empty,
   - the effective ``skip`` resolves to 0,
   - and the last filename ``self._fnames[-1]`` appears in the expanded list.

   In this case:
   - Frames are appended starting from ``self._last[-1] + 1``.
   - The filename is removed from the remaining list of files.
   - Extension stops early if the frame budget ``nframes`` is exhausted.

3) Remaining files in the expanded list are appended sequentially as new
   segments until the frame budget is exhausted.

Overlap removal
---------------
If ``remove_overlapping_frames=True`` and time information is available,
the method performs **time-based overlap correction**:

- The time of the first candidate frame in the new file (rounded to 3 decimals)
  is compared to the time of the last appended frame.
- The implementation may rewind *multiple frames* from the previous segment
  until ``t_last < t_new``.
- Rewinding increases the remaining frame budget accordingly.
- If the rewind affects the original final segment, the “start” index used
  for post-processing is adjusted.

This is not limited to removing a single frame.

Concurrent file access
----------------------
This module is designed for AIMMD workflows where trajectory files may be
written and read concurrently. Therefore:

- Reader access is handled through the global ``MDA_CACHE``.
- Some operations are wrapped in broad ``try/except`` blocks to tolerate
  transient I/O errors.

Special cases
-------------
- If ``MDA_CACHE.get(fname, min_length)`` returns None when trying to extend
  the last segment, the last segment is removed entirely and the method
  returns early.
- ``frames_left`` in the return value refers to the last reader touched.
  When the method stops due to frame-budget exhaustion in the file loop,
  it may return ``-1`` to indicate that future availability is unknown.

Scope
-----
This file defines only the mixin; the concrete :class:`Path` class is
assembled elsewhere in :mod:`aimmd.path`.
"""

# external imports
import os
import numpy as np
from abc import ABC
from math import inf
from glob import glob
from MDAnalysis import Universe, Writer

# aimmd imports
from .utils import get_fnames, get_last_time_and_dt, get_cache_fname
from .._config import MDA_CACHE, DEFAULT_DIMENSIONS

# Path's input/output methods
class PathIO(ABC):
    
    def extend(self, fnames,
               nframes=inf, skip=0,
               remove_overlapping_frames=True,
               pipeline=()):
        """
        Extend the current Path by appending frames available on disk.
    
        This method is designed for **incrementally growing trajectories** that may be
        written concurrently. It tries to append up to ``nframes`` new frames by:
    
        (A) optionally extending the Path's **current last segment** (same filename), then
        (B) appending additional trajectory files from ``fnames`` as new segments.
    
        Crucially, step (A) only happens when **both** of the following are true:
    
        - after normalization, ``skip == 0`` (see the ``skip`` parameter semantics), and
        - the Path is non-empty and its last filename ``self._fnames[-1]`` appears in the
          expanded filename list derived from ``fnames``.
    
        Parameters
        ----------
        fnames : str | pathlib.Path | Sequence[str | pathlib.Path] | None
            One or more trajectory filenames and/or glob patterns. Expansion and ordering
            are handled by :func:`aimmd.path.utils.get_fnames`.
    
            The *expanded list* is the authoritative sequence of candidate files. If the
            expanded list is empty, the method returns ``(0, 0)`` and does not modify the Path.
    
            If the expanded list includes the current last segment file, that file is
            *consumed* by the “extend last segment” step and removed from the list of files
            to append afterwards.
        nframes : int | float, default=math.inf
            Maximum number of frames to append in total. Internally treated as a remaining
            budget ``frames_to_add``. If ``nframes`` is 0 or falsy, the method returns ``(0, 0)``.
        skip : int | None, default=0
            Skip count applied before appending new frames.
    
            Semantics in this implementation:
    
            1) ``skip`` is normalized via:
               - ``skip = 0`` if ``skip is None``,
               - otherwise ``skip = max(int(skip), 0)``.
            2) Then it is converted to “extra skip beyond the current Path length” via:
               ``skip = max(skip - start, 0)``, where ``start = len(self)`` before any changes.
    
            Consequences:
            - If ``skip < len(self)``, then the effective skip becomes 0 (it does **not**
              skip within existing frames).
            - If ``skip > len(self)``, the extra skip is applied to the first new input file(s)
              by advancing the local starting index for those files.
        remove_overlapping_frames : bool, default=True
            If True and a time reference from the previous appended frame is available,
            attempt to remove overlap between the last appended segment and the next segment.
    
            **Important:** this implementation may remove *multiple* frames from the end of
            the previously appended segment by repeatedly decrementing ``self._last[-1]``
            until ``t1 < t0``, where:
    
            - ``t1`` is the time of the last frame currently in the Path (rounded to 3 decimals),
            - ``t0`` is the time of the first candidate frame in the next file at local index
              ``skip`` (also rounded to 3 decimals),
            - ``dt`` is taken from :func:`aimmd.path.utils.get_last_time_and_dt` on the previous
              reader.
    
            Each rewind increases ``frames_to_add`` by 1 (to keep the total append budget
            consistent). If the rewind affected the original last segment that existed before
            calling this method, ``start`` is decremented so the pipeline recomputes one extra
            frame.
        pipeline : Iterable[tuple] | tuple, default=()
            Optional post-processing pipeline applied after extension.
    
            If non-empty, the code constructs:
    
            - ``path = self[start:]`` if ``start`` is non-zero,
            - otherwise ``path = self``.
    
            Then for each element ``args`` in ``pipeline``, it executes:
            ``path.compute(*args)``.
    
            Therefore each pipeline element must be a tuple of positional arguments compatible
            with ``Path.compute``.
            The pipeline is executed regardless of whether frames were added (the code checks
            only whether ``pipeline`` is truthy), but the slice ``self[start:]`` may be empty.
    
        Returns
        -------
        added_frames : int
            Total number of frames appended across all updated/appended segments.
        frames_left : int
            Remaining frames available in the **last reader touched** after extension:
    
            - For the last file processed, ``frames_left`` is computed as
              ``len(reader) - end_index`` (or ``len(reader) - skip - delta`` in the new-file loop).
            - When the method stops because ``frames_to_add <= 0`` inside the new-file loop,
              it sets ``frames_left = -1`` as a sentinel meaning “unknown (next file)”.
    
        Raises
        ------
        RuntimeError
            If the method attempts to extend the current last segment but the segment is
            running backwards, detected by ``self._first[-1] >= self._last[-1] + 1``.
    
        Side effects
        ------------
        - Mutates ``self._fnames``, ``self._first``, ``self._last``.
        - May shorten the last segment (overlap rewind).
        - Special case: if ``MDA_CACHE.get(fname, min_length)`` returns None when trying to
          extend the current last file, the method removes that last segment entirely and
          returns early with ``(added_frames, -1)``.
    
        Notes
        -----
        - The method assumes that "continuing" a file means appending local indices starting
          at ``self._last[-1] + 1``.
        - All time comparisons used for overlap removal are rounded to 3 decimals.
        - Several operations are wrapped in broad ``try/except`` blocks to tolerate transient
          I/O failures during concurrent writes.
        """
        
        if skip is None:
            skip = 0
        else:
            skip = max(skip, 0)
        frames_to_add = nframes or inf 
        if not frames_to_add:
            return 0, 0
        
        # get fnames
        new_fnames = get_fnames(fnames)
        if not new_fnames:
            return 0, 0

        # initialize
        frames_left = 0
        added_frames = 0
        length = 0  # of current reader
        start = len(self)  # where you start computing from
        skip = max(skip - start, 0)
        t1 = None
                
        # update current last file in path
        n_files = len(self._fnames)
        if n_files and not skip and (fname := self._fnames[-1]) in new_fnames:
            end = self._last[-1] + 1
            if self._first[-1] >= end:
                raise RuntimeError("can't extend path running backwards")
            new_fnames = new_fnames[new_fnames.index(fname) + 1:]
            
            # min length that you want to add
            min_length = end + frames_to_add
            
            # how many new frames? (get them through reader)
            reader = MDA_CACHE.get(fname, min_length)
            if reader is None:  # special case: shrink the path instead
                self._fnames.pop(-1)
                self._last.pop(-1)
                added_frames = self._first.pop(-1) - end
                return added_frames, -1
            length = len(reader)
            
            # calculate progress
            frames_left = length - end
            delta = min(frames_left, frames_to_add)
            frames_left -= delta
            frames_to_add -= delta
            added_frames += delta
            
            # update
            self._last[-1] += delta

            # get time info
            try:  # try/except to prevent i/o errors with concurrent access
                t1, dt = get_last_time_and_dt(reader, self._last[-1])
            except:
                new_frames = []

        length = 0
        
        # update new files
        for fname in new_fnames:
            if frames_to_add <= 0:
                frames_left = -1  # the next path (you never know)
                break
            
            # get reader or times
            min_length = skip + frames_to_add
            reader = MDA_CACHE.get(fname, min_length)  
            if reader is None:  # special case: skip
                continue
            length = len(reader)
            
            # no frames to add
            if length <= skip:
                continue
            
            # remove overlapping frames
            if remove_overlapping_frames and t1 is not None:
                
                # current time
                try:  # try/except to prevent i/o errors with concurrent access
                    t0 = round(reader[skip].time, 3)
                except:
                    break
                
                # rewind previous trajectory up until before t0
                while True:
                    if self._last[-1] < 0:
                        self._fnames.pop(-1)
                        self._first.pop(-1)
                        self._last.pop(-1)
                        break
                    if t1 < t0:
                        break
                    t1 = round(t1 - dt, 3)
                    self._last[-1] -= 1
                    frames_to_add += 1
                    if len(self._fnames) == n_files:
                        start -= 1  # compute one frame more
            
            # no frames to add
            if length <= skip:
                continue
            
            # calculate progress
            frames_left = length - skip
            delta = min(frames_left, frames_to_add)
            frames_left -= delta
            frames_to_add -= delta
            added_frames += delta
            
            # update
            self._fnames.append(fname)
            self._first.append(skip)
            end = skip + delta
            self._last.append(end - 1)
            skip = max(skip - length, 0)
            
            # get time info
            try:  # try/except to prevent i/o errors with concurrent access
                t1, dt = get_last_time_and_dt(reader, self._last[-1])
            except:
                break
        
        # compute from "start": only the new frames
        if pipeline:
            path = self[start:] if start else self
            for args in pipeline:
                path.compute(*args)
        
        # return
        return added_frames, frames_left

    def write(self, filename, key=None, atoms=None,
              t0=None, overwrite=False, return_writer=False):
        """
        Write frames from the Path to a trajectory file.

        Frames are written through an MDAnalysis :class:`~MDAnalysis.coordinates.base.Writer`.
        The source is either:

        - file-backed: ``self.reader`` if the required arrays are not fully in memory, or
        - in-memory: the cached arrays stored on the Path (``times``, ``positions``,
          ``velocities``, ``dimensions``).

        Parameters
        ----------
        filename : str | pathlib.Path
            Output trajectory filename. The output format is inferred by MDAnalysis from
            the file extension.
        key : int | slice | numpy.ndarray | None, default=None
            Frame selector in global Path indexing. Internally converted by
            ``np.arange(len(self))[key]`` and flattened. If None, all frames are written.
        atoms : int | slice | numpy.ndarray | None, default=None
            Atom selector applied as ``np.arange(self.n_atoms)[atoms]`` and flattened.
            If None, all atoms are written.
        t0 : float | None, default=None
            Optional time origin override. If provided, the written time is set to
            ``t0 + dt * i`` where ``i`` is the write counter (0..n_frames-1) and ``dt`` is
            inferred from the first two selected frames (or set to 1.0 for a single frame).
            If None, uses the times from the source frames.
        overwrite : bool, default=False
            If False, refuse to overwrite an existing file.
        return_writer : bool, default=False
            If True, return the writer object (left open). If False, close it and return
            the list of written times.

        Returns
        -------
        writer_or_times : MDAnalysis Writer | list[float]
            - If ``return_writer=True``: the MDAnalysis writer instance.
            - Otherwise: a list of times corresponding to the written frames.

        Raises
        ------
        TypeError
            If ``overwrite`` is False and ``filename`` already exists.
        RuntimeError
            If the selection yields no frames.

        Notes
        -----
        - A temporary in-memory universe is created via
          ``Universe.empty(n_atoms, trajectory=True)``.
        - If a source frame has not dimensions, ``aimmd._config.DEFAULT_DIMENSIONS`` is written.
        """
        
        # not overwriting
        if not overwrite and os.path.exists(filename):
            raise TypeError(f'{filename!r} exists, cannot write')

        # in memory stuff
        times_in_memory = self.in_memory('times')
        positions_in_memory = self.in_memory('positions')
        velocities_in_memory = self.in_memory('velocities')
        dimensions_in_memory = self.in_memory('dimensions')
        if not (times_in_memory and positions_in_memory and
                velocities_in_memory and dimensions_in_memory):
            reader = self.reader
        else:
            reader = np.arange(len(self))  # mockup
        
        if key is not None:
            key = np.arange(len(self))[key].flatten()
            reader = reader[key]
        else:
            key = range(len(self))
        n_frames = len(reader)

        if not n_frames:
            raise RuntimeError("no frames to write")
        
        # get universe and timestep
        indices = np.arange(self.n_atoms)[atoms].flatten()
        n_atoms = len(indices)
        universe = Universe.empty(n_atoms, trajectory=True)
        ts = universe.trajectory.ts
        
        # get time info
        if n_frames >= 2:
            if times_in_memory:
                t1, t2 = self.__dict__['times'][key[:2]]
            else:
                t1 = reader[0].time
                t2 = reader[1].time
            dt = abs(t2 - t1) or 1.0
        else:
            if times_in_memory:
                t1 = self.__dict__['times'][key[0]]
            else:
                t1 = reader[0].time
            dt = 1.0

        # get direction
        directions = np.ones(n_frames)
        directions[:-1][np.diff(key) < 0] = -1.0
        
        # initialize times list
        times = []
        dt = 1.0
        
        # open writer
        writer = Writer(str(filename), len(indices))
        try:
                        
            # iterate over frames
            for i, frame in enumerate(reader):
                # get info

                # time and dt
                if times_in_memory:
                    t2 = self.__dict__['times'][key[i]]
                else:
                    t2 = frame.time
                if i == 1:
                    dt = abs(t2 - t1)
                t1 = t2
                if t0 is not None:
                    t = t0 + dt * i
                else:
                    t = t1

                # positions
                if positions_in_memory:
                    positions = self.__dict__['positions'][key[i]]
                else:
                    positions = frame.positions[indices].copy()

                # velocities (right direction)
                if velocities_in_memory:
                    velocities = self.__dict__['velocities'][key[i]]
                else:
                    velocities = frame._velocities
                    if velocities.any():
                        velocities = velocities[indices] * directions[i]
                    else:
                        velocities = np.zeros((n_atoms, 3))

                # dimensions
                if dimensions_in_memory:
                    dimensions = self.__dict__['dimensions'][key[i]]
                else:
                    dim = frame.dimensions
                    if dim is None:
                        dimensions = DEFAULT_DIMENSIONS
                    else:
                        dimensions = dim
                
                # update and write frame
                ts.positions = positions
                ts.velocities = velocities * directions[i]
                ts.dimensions = dimensions
                ts.time = t
                writer.write(universe)
                times.append(t)
            
            # return writer or written times
            if return_writer:
                return writer
            writer.close()
            return times
        
        # clear in case of exception
        except Exception as exception:
            writer.close()
            raise exception
