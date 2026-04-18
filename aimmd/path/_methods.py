"""
aimmd.path._methods
===================

High-level convenience methods for :class:`aimmd.path.Path`.

This module defines :class:`PathMethods`, a mixin implementing operations that
interpret a Path in terms of state sequences and trajectory semantics rather than
raw segment bookkeeping.

Main groups of helpers
----------------------
Classification (based on ``self.type``)
    - :meth:`is_transition`, :meth:`is_excursion`, :meth:`is_internal`,
      :meth:`is_complete`.

Online-growth stop analysis
    - :meth:`check_stop` scans the discrete state labels and returns the first
      index where trajectory growth should stop (excluded region, disallowed
      state, or too-long block).

Representation conversion
    - :meth:`to_memory` caches the full trajectory in arrays on the Path.
    - :meth:`from_files` drops cached arrays and reverts to file-backed reading.

Utilities
    - :meth:`split` partitions the Path into contiguous blocks of non-empty state
      labels.
    - :meth:`partial`, :meth:`copy`, :meth:`sample` provide extraction/copying.

Notes
-----
- This module assumes the concrete Path provides private helpers such as
  ``_extract``, ``_get_local_index``, ``reader``, and the lazily computed attribute
  interface ``self.states`` / ``self.type``.
- Docstrings in this module describe the API contract; the actual semantics of
  "internal/backward/forward" ranges are defined by the Path core implementation.
"""

# external
import os
import re
import numpy as np
from abc import ABC
from math import inf
from tqdm import tqdm
from numbers import Integral
from itertools import islice

# aimmd imports
from ._get import PathGet
from .utils import split
from .._config import DEFAULT_DIMENSIONS
from ..core.utils import process_state

# path methods
class PathMethods(ABC):

    def is_complete(self, target_state='R', states='ARB'):
        """Return whether the path is complete with respect to a target state.

        Parameters
        ----------
        target_state : str, default='R'
            Target one-letter state label to test for. The label is normalized by
            :func:`aimmd.core.utils.process_state` using the ``states`` alphabet.
        states : str, default='ARB'
            Three-character alphabet defining (initial, middle/reactive, final) state
            labels.

        Returns
        -------
        bool
            True if ``self.type`` indicates that the target state has been reached in
            an allowed/complete pattern, otherwise False.
        """
        t = process_state(target_state, states)
        i, r, f = states

        # reactive
        if t == r:
            return self.type[:3] in (f'{i}{r}{f}', f'{f}{r}{i}',
                                     f'{i}{r}{i}', f'{f}{r}{f}')

        # internal
        path_type = self.type
        if path_type[0] in f'.{t}' or path_type[2] == f'.{t}':
            return False
        if path_type[1] != t:
            return False
        return True

    def is_transition(self, states='ARB'):
        """Return True if the path is a direct A<->B transition.

        Parameters
        ----------
        states : str, default='ARB'
            Three-character alphabet defining the two end states and the middle state.

        Returns
        -------
        bool
            True if ``self.type[:3]`` equals ``states`` or its reverse.
        """
        path_type = self.type[:3]
        return path_type in (states, states[::-1])

    def is_excursion(self, states='ARB'):
        """Return True if the path leaves an end state and visits the middle state.

        Parameters
        ----------
        states : str, default='ARB'
            Three-character alphabet defining the end states and middle state.

        Returns
        -------
        bool
            True if the initial state is an end state (A or B) and the middle state is
            the reactive/middle label (R).
        """
        return (self.initial('states') in (states[0], states[-1]) and
                self.middle('states') == states[1])

    def is_internal(self, states='ARB'):
        """Return True if the path is classified as internal by its middle state.

        Parameters
        ----------
        states : str, default='ARB'
            Three-character alphabet defining the end states and middle state.

        Returns
        -------
        bool
            True if ``self.middle('states')`` is one of the end-state labels.
        """
        i, m, f = self.type[:3]
        return self.middle('states') in (states[0], states[2])

    def check_stop(self, allowed_states='', max_length=inf,
                   check_first_frame=True):
        """Determine whether growth should stop by scanning the state sequence.

        Parameters
        ----------
        allowed_states : str, default=''
            Allowed one-letter state labels. If empty or ``'all'``, no restriction is
            enforced. If non-empty and ``check_first_frame=True``, the first frame must
            be in ``allowed_states`` or a RuntimeError is raised.
        max_length : int | float, default=math.inf
            Maximum allowed length (in frames) of a contiguous block as returned by
            :func:`aimmd.path.utils.split`.
        check_first_frame : bool, default=True
            Whether to enforce that the first frame is in ``allowed_states`` when
            ``allowed_states`` is provided.

        Returns
        -------
        stop_index : int | None
            Global frame index (0-based) where the violating block begins, or None if
            no stop condition is met.
        nframes : int
            Number of usable frames after trimming trailing empty state labels.
        last_state : str
            A state label associated with the detected condition (or the final observed
            state if no condition is met).
        block_length : int
            Length in frames of the relevant contiguous block.

        Notes
        -----
        This method temporarily disables ``_exclude_from`` to query the full ``states``
        array, then restores the original value.
        """
        if not len(self):
            return None, 0, '', 0

        # get info: states
        exclude_from = self._exclude_from
        try:
            self._exclude_from = -1
            states = self.states
        finally:
            self._exclude_from = exclude_from
        if not states[0]:
            return None, 0, '', 0

        # restrict states to where they are useful
        nframes = len(states)
        while nframes > 1 and states[nframes - 1] == '':
            nframes -= 1
        states = states[:nframes]

        # get info: lengths
        start, stop = split(states)
        lengths = stop - start
        final_states = states[stop - 1]
        n_split_paths = len(lengths)

        # get info: states
        if allowed_states == 'all':
            allowed_states = ''

        # cold start
        if allowed_states:
            if check_first_frame and states[0] not in allowed_states:
                raise RuntimeError(
                    f'{self.fnames[0]}, {self.first[0]} in state {states[0]}, '
                    f'should be in {allowed_states}; consider deleting the '
                    f'trajectory file to allow AIMMD to recreate it')
            elif nframes == 1:  # nothing to do
                return None, 1, states[0], 1
            if stop[0] > 1 and states[1] not in allowed_states:
                return (0, nframes, states[1],
                        2 if states[0] != states[1] else lengths[-1])

        # condition 1: excluded
        i1 = n_split_paths
        if self._exclude_from >= 0:
            condition = start > self._exclude_from
            if (where := np.flatnonzero(condition)).size:
                i1 = where[0]

        # condition 2: bad state
        i2 = n_split_paths
        if allowed_states:
            condition = np.logical_and.reduce(
                [final_states != s for s in allowed_states])
            if (where := np.flatnonzero(condition)).size:
                i2 = where[0]

        # condition 3: max length
        i3 = n_split_paths
        condition = lengths > max_length
        if (where := np.flatnonzero(condition)).size:
            i3 = where[0]

        # which was first?
        i = min(min(i1, i2), i3)

        # return
        if i < n_split_paths:
            last_state = states[max(start[i] + 1, stop[i] - 1)]
            return start[i], nframes, last_state, lengths[i]
        return None, nframes, final_states[-1], lengths[-1]

    def partial(self, attribute='self', key=None):
        """Extract one or more file segments from the Path.

        Parameters
        ----------
        attribute : str, default='self'
            What to extract from each selected segment. The special value ``'self'``
            returns a Path object; other values are delegated to the private extraction
            helper ``_extract``.
        key : int | slice | numpy.ndarray | None, default=None
            Segment selector over file indices (0..``self.n_files-1``). If not an
            integer, the selection is expanded via ``np.arange(self.n_files)[key]`` and
            a list is returned.

        Returns
        -------
        object | list[object]
            Extracted object(s) for the selected segment(s).
        """
        if not isinstance(key, Integral):
            return [self._extract(k, attribute)
                    for k in np.arange(self.n_files)[key].flatten()]
        return self._extract(key, attribute)

    def in_memory(self, attribute=None):
        """Return whether cached in-memory arrays are present.

        Parameters
        ----------
        attribute : str | None, default=None
            - If None (or one of ``'self'``, ``'reader'``, ``'frames'``): check for the
              full in-memory representation (times/positions/velocities/dimensions).
            - Otherwise: check whether that attribute key exists in ``self.__dict__``.

        Returns
        -------
        bool
        """
        if attribute in (None, 'self', 'reader', 'frames'):
            return ('times' in self.__dict__ and
                    'positions' in self.__dict__ and
                    'velocities' in self.__dict__ and
                    'dimensions' in self.__dict__)
        return attribute in self.__dict__

    def split(self, return_start_stop=False, states=None):
        """Split the path into contiguous blocks of non-empty state labels.

        Parameters
        ----------
        return_start_stop : bool, default=False
            Currently unused (kept for API compatibility).
        states : numpy.ndarray | Sequence[str] | None, default=None
            Optional external state array. If not provided, the method attempts to use
            ``self.states``; on failure it uses an array of empty labels.

        Returns
        -------
        aimmd.pathensemble.PathEnsemble
            Ensemble whose paths are slices corresponding to blocks returned by
            :func:`aimmd.path.utils.split`.
        """
        from ..pathensemble import PathEnsemble

        try:
            states = self.states
        except:
            states = np.full(len(self), '')

        result = PathEnsemble()
        if len(states):
            result._paths = [self[start:stop]
                             for start, stop in zip(*split(states))]
        return result

    def to_memory(self):
        """Cache the full trajectory in memory on this Path.

        Returns
        -------
        self
            The same Path instance, mutated in-place. After completion, the Path has
            cached arrays: ``positions``, ``velocities``, ``dimensions``, and ``times``.

        Notes
        -----
        - Missing velocities are replaced by zeros.
        - Missing box dimensions are replaced by ``DEFAULT_DIMENSIONS``.
        """
        positions = []
        velocities = []
        dimensions = []
        times = []
        for frame in self.reader:
            positions.append(frame.positions.copy())
            vel = frame._velocities.copy()
            if vel.size:
                velocities.append(vel)
            else:
                velocities.append(positions[-1] * 0.)
            dim = frame.dimensions
            if dim is None:
                dimensions.append(DEFAULT_DIMENSIONS)
            else:
                dimensions.append(dim.copy())
            times.append(frame.time)
        self.positions = positions
        self.velocities = velocities
        self.dimensions = dimensions
        self.times = times
        return self

    def from_files(self):
        """Drop cached in-memory arrays and revert to file-backed reading.

        Returns
        -------
        self
            The same Path instance, mutated in-place.
        """
        self.positions = None
        self.velocities = None
        self.dimensions = None
        self.times = None
        return self

    def update_exclude_from(self, log_fname):
        """Update ``_exclude_from`` by parsing an external log file.

        Parameters
        ----------
        log_fname : str | pathlib.Path
            Log file path. The method scans whitespace-separated tokens per line. If the
            basename of token 0 matches ``self.fname``, token 1 (if present) is parsed
            as an integer exclude-from index; otherwise exclude-from is set to 0.

        Returns
        -------
        None
        """
        if os.path.exists(log_fname):
            self._exclude_from = -1
            with open(log_fname) as file:
                for line in file:
                    fields = line.split()
                    if not fields:
                        continue
                    if split(r'[\\/]', fields[0])[-1] in self.fname:
                        if len(fields) == 1:
                            self._exclude_from = 0
                        else:
                            self._exclude_from = int(fields[1])
                        return

    def copy(self):
        """Return a shallow copy of the Path.

        Returns
        -------
        aimmd.path.Path
            New Path instance with duplicated segment lists and copied cached arrays.

        Notes
        -----
        This method copies values stored in ``self.__dict__`` starting after the standard
        internal fields, using ``value.copy()``.
        """
        from . import Path
        result = object.__new__(Path)        
        result._fnames = self._fnames[:]
        result._first = self._first[:]
        result._last = self._last[:]
        result._weight = self._weight
        result._exclude_from = self._exclude_from
        result._shooting_index = self._shooting_index
        for attribute, value in islice(
            self.__dict__.items(), 6, None):
            result.__dict__[attribute] = value.copy()
        return result
    
    def find_shooting_index(self):
        """
        Return the frame index corresponding to the shooting point.
    
        The shooting point is inferred purely from the trajectory time stamps.
        In AIMMD paths, frames before the shooting point are typically stored in
        reverse time order, while frames after the shooting point are stored in
        forward time order. This means that the sign of the time increment between
        the first two frames tells us whether the path starts by moving forward or
        backward away from the shooting point:
    
        - ``dt >= 0``: the trajectory already starts at the shooting point, so the
          shooting index is ``0``.
        - ``dt < 0``: the first frames belong to the backward branch, so the
          shooting point lies later in the path and can be reconstructed from the
          time grid.
    
        The method assumes that the trajectory times are approximately equally
        spaced around the shooting point and that the shooting frame corresponds to
        time zero.
    
        Returns
        -------
        int
            Index of the shooting frame.
    
        Notes
        -----
        The computation uses times rounded to 6 decimal places to suppress tiny
        floating-point noise from trajectory readers.
    
        For paths with fewer than two frames, the only sensible answer is ``0``.
        """
    
        # Degenerate case: with 0 or 1 frame there is no time direction to inspect.
        # In that situation we conventionally treat the only available frame
        # as the shooting point.
        if len(self) < 2:
            return 0
    
        # Read the first two time values, since they are sufficient to determine
        # whether the stored path initially moves forward or backward in time.
        #
        # Prefer the cached in-memory array if it already exists:
        # - avoids advancing / reopening the trajectory reader,
        # - is faster,
        # - keeps the logic independent of reader side effects.
        if 'times' in self.__dict__:  # value is in memory
            t0, t1 = np.round(self.__dict__['times'][:2], 6)
    
        # Otherwise, pull only the first two times directly from the reader.
        # We do not build the full time array, because only two values are needed.
        else:
            for i, ts in enumerate(self.reader):
                if i == 0:
                    # Time of the first stored frame.
                    t0 = round(ts.time, 6)
                else:
                    # Time of the second stored frame; once obtained, we can stop.
                    t1 = round(ts.time, 6)
                    break
    
        # Effective timestep between the first two stored frames.
        # Its sign encodes the ordering of the beginning of the path:
        # - positive or zero: forward in time
        # - negative: backward in time
        dt = t1 - t0
    
        # If time increases from the first to the second frame, then the first frame
        # is already the shooting frame. This is the simple "forward branch first"
        # layout.
        if dt >= 0:
            return 0
    
        # If dt < 0, the path begins on the backward branch.
        #
        # Example:
        #   times = [3, 2, 1, 0, 1, 2]
        #   t0 = 3, dt = -1
        #   shooting index = 3
        #
        # Since the shooting frame is defined by time 0, and times are assumed to
        # follow a regular spacing dt, its index can be reconstructed from:
        #
        #   t(index) = t0 + index * dt = 0
        #   index = -t0 / dt
        #
        # Because dt is negative here, the result is positive. Must cast to int.
        #
        # Finally, clamp the value to len(self) - 1 so that small numerical or
        # formatting inconsistencies cannot produce an out-of-bounds index.
        return min(int(-t0 / dt), len(self) - 1)
    
    def shooting_result(self, states='ARB'):
        """Return a 2-element outcome count derived from the 3-letter path type.

        Parameters
        ----------
        states : str, default='ARB'
            Three-character alphabet defining (A, R, B).

        Returns
        -------
        numpy.ndarray
            Array of shape (2,) with counts for reaching A (index 0) and B (index 1),
            conditional on the middle letter being R.
        """
        a, r, b = states
        states = self.type
        result = np.zeros(2)
        if states[1] != r:
            return result
        result[0] += states[0] == a
        result[0] += states[2] == a
        result[1] += states[0] == b
        result[1] += states[2] == b
        return result

    def sample(self, n_samples, source='values', vmin=None, vmax=None):
        """Randomly sample internal frames and return them as a
        new single-frame-segment Path.

        Parameters
        ----------
        n_samples : int
            Number of frames to sample. If 0 or the Path is empty, an empty Path is
            returned.
        source  : limit to source vmin to vmax

        Returns
        -------
        aimmd.path.Path
            A new Path with ``n_samples`` segments, each containing exactly one frame.
        """
        from . import Path
        result = Path()
        if not n_samples or not len(self):
            return result
        fnames = []
        first = []
        last = []
        indices = self.internal('indices')
        
        # restrict between vmin and vmax
        if vmin is not None or vmax is not None:
            values = self.internal(source)
            mask = np.ones(len(values), dtype=bool)
            if vmin is not None:
                mask &= values >= vmin
            if vmax is not None:
                mask &= values < vmax
            indices = indices[mask]
        
        # sample
        for i in np.random.choice(indices, n_samples):
            k, i = self._get_local_index(i)
            fnames.append(self._fnames[k])
            first.append(i)
            last.append(i)
        result._fnames = fnames
        result._first = first
        result._last = last
        return result
    
    get = PathGet._get
    """Alias for PathGet._get."""
