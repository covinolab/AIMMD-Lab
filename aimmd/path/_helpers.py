"""
aimmd.path._helpers
===================

Internal helper mixin for :class:`aimmd.path.Path`.

This module defines :class:`~aimmd.path._helpers.PathHelpers`, a mixin that
implements low-level utilities used across the Path implementation:

- initialization from filenames or from an existing Path,
- mapping global Path indices to per-file indices/locations,
- helper routines used by convenience accessors (initial/final/middle/etc.),
- helper for selecting index ranges (all/internal/backward/forward),
- retrieving a single "position" entry for various attributes.

Design notes
------------
- This mixin does not perform I/O directly (except via other mixins).
- It assumes that `self._fnames`, `self._first`, `self._last` describe the
  per-file segments (possibly reversed) and are already consistent.
- Most helpers are internal and are not part of the public API; they exist to
  keep the public-facing methods short and to centralize indexing logic.
"""

# external imports
import os
import time
import numpy as np
from abc import ABC
from math import inf, nan
from collections.abc import Iterable

# aimmd imports
from ..core.utils import get_local_index

# Path's helper methods
class PathHelpers(ABC):

    def _init(self, fnames=[],
              start=None,
              stop=None,
              remove_overlapping_frames=False,
              pipeline=(),
              weight=1.0,
              exclude_from=-1,
              shooting_index=0):
        """Initialize a :class:`~aimmd.path.Path` instance.

        This method is used as the concrete `Path.__init__` via aliasing.
        It supports initializing from filenames/patterns or by copying/slicing
        an existing `Path`.

        Parameters
        ----------
        fnames : str or pathlib.Path or Iterable or aimmd.path.Path, optional
            Either (i) a filename/pattern, or an iterable of them, or (ii) an
            existing `Path` instance to copy/slice.
        start : int or None, optional
            Global start index for slicing when initializing from another Path,
            or for skipping initial frames when initializing from files.
        stop : int or None, optional
            Global stop index for slicing, or maximum number of frames.
        remove_overlapping_frames : bool, optional
            If True, attempt to remove overlap between consecutive files based
            on time stamps when extending.
        pipeline : tuple, optional
            Optional compute pipeline. Each element is an argument tuple passed
            as `path.compute(*args)` after initialization.
        weight : float, optional
            Statistical weight of this path.
        exclude_from : int, optional
            If >= 0, mark the path as rejected from this frame onward.
        shooting_index : int or any, optional
            Shooting point index in global Path coordinates. If not an integer,
            the property setter "finds" it (frame with time = 0 or first frame).

        Notes
        -----
        - `exclude_from` affects the public `states` view (frames after it may be
          treated as invalid) but does not physically truncate the Path.
        """
        
        if isinstance(fnames, Iterable):
            
            # initialize core file/segment bookkeeping
            self._fnames = []
            self._first = []
            self._last = []
    
            # normalize start/stop for file-based initialization
            if start is None:
                start = 0
            else:
                start = max(0, start)
            if stop is None:
                stop = inf
            nframes = stop - start
            self.extend(fnames, nframes, start,
                        remove_overlapping_frames, pipeline)

        else:  # just update with the path
            from . import Path
            path = fnames
            if not isinstance(path, Path):
                raise TypeError(f'input to Path is either a string, '
                                f'a list of strings, or another path, '
                                f'got {path!r}')
            # copy dict from sliced Path
            self.__dict__.update(path[start:stop].__dict__)
            # optionally compute/cached requested series on the source path
            for args in pipeline:
                path.compute(*args)
        
        # assign user-facing attributes via validated setters (in PathProperties)
        self.weight = weight
        self.exclude_from = exclude_from
        self.shooting_index = shooting_index
    
    def _get_local_loc(self, i, clip=False):
        """Return `(k, loc)` where `k` is the file index and `loc` is the
        absolute frame index inside that file.

        Parameters
        ----------
        i : int
            Global frame index in Path coordinates.
        clip : bool, optional
            If True, clip out-of-range indices into valid bounds.

        Returns
        -------
        tuple[int, int]
            `(k, loc)` where `loc` is the file-local absolute location.
        """
        k, i = self._get_local_index(i, clip=clip)
        start, last = self._first[k], self._last[k]
        step = 1 if start <= last else -1
        return k, start + i * step
    
    def _get_local_index(self, i, clip=False):
        """Map a global Path index into `(file_index, index_within_file_segment)`.

        Parameters
        ----------
        i : int
            Global Path index.
        clip : bool, optional
            Forwarded to :func:`aimmd.core.utils.get_local_index`.

        Returns
        -------
        tuple[int, int]
            `(k, i_local)` where `k` is the file index and `i_local` counts
            frames along the Path segment for that file.
        """
        return get_local_index(i, self.offsets, clip=clip)
    
    def _extreme(self, attribute, operation, where, source='values'):
        """Return `attribute` at the frame where `source` is extreme.

        Parameters
        ----------
        attribute : str
            Attribute to return (e.g. 'indices', 'times', 'states', 'values', ...).
        operation : callable
            Numpy arg-extreme function (e.g. `np.argmin`, `np.argmax`).
        where : {'all', 'internal', 'backward', 'forward'}
            Path region selector.
        source : str, optional
            Attribute used to locate the extreme.

        Returns
        -------
        object
            The value of `attribute` at the selected frame.
            If `attribute == 'self'`: returns a `aimmd.path.Path`.
        """
        start, stop = self._range(where)
        values = self._get(source, start, stop)
        if attribute == source:
            series = values
        else:
            series = self._get(attribute, start, stop)
        i = operation(values)
        if attribute == 'self':
            return series[i:i + 1]  # a path
        return series[i]

    def _range(self, where):
        """Resolve region selectors into `(start, stop)` bounds.

        Parameters
        ----------
        where : {'all', 'internal', 'backward', 'forward'}
            Region definition.

        Returns
        -------
        tuple[int|None, int|None]
            Slice bounds suitable for `self._get(attribute, start, stop)`.

        Notes
        -----
        - 'internal' excludes endpoint frames depending on the path type.
        - 'backward'/'forward' further restrict 'internal' relative to the
          shooting point index.
        """
        start = None
        stop = None
        
        # all
        if where == 'all':
            return start, stop

        # internal
        if where == 'internal':
            if not (abcd := self.type):
                return start, len(self)
            if abcd[0] != abcd[1]:
                start = 1
            if abcd[1] != abcd[2]:
                stop = len(self) - 1
            return start, stop
        
        # backward as a modification of "internal"
        if where == 'backward':
            shooting_index = self._shooting_index
            start, stop = self._range('internal')
            stop = min(stop or len(self), shooting_index + 1)
            return start, stop
        
        # forward as a modification of "internal"
        if where == 'forward':
            shooting_index = self._shooting_index
            start, stop = self._range('internal')
            start = max(start or 0, shooting_index)
            return start, stop
        
        raise ValueError(
            f'"where" must be one of ("all", "internal", "backward", '
            f'"forward"), got {where!r} instead')

    def _position(self, i, attribute='indices'):
        """Return a single element of a Path series at index `i`.

        This is the low-level backend used by :class:`PathPositions`.

        Parameters
        ----------
        i : int
            Frame index (global Path coordinates; negative indices supported).
        attribute : str, optional
            Series name ('indices', 'locs', 'reader', 'frames', 'states', etc.).

        Returns
        -------
        object
            Value at the requested index. For unknown series, returns a default:
            - '' for 'states'
            - NaN for numeric series (best-effort)
            If `attribute == 'self'`: returns a `aimmd.path.Path`.

        Notes
        -----
        If `attribute == 'states'` and the Path is rejected (`exclude_from >= 0`),
        frames at/after `exclude_from` are represented as '.' for single-item
        access (consistent with `PathProperties.type` conventions).
        """
        if attribute == 'indices':
            return i
        if attribute == 'locs':
            return self.locs[i]
        if attribute in ('reader', 'frames'):
            return self[i]
        if attribute == 'self':
            return self[i:i + 1]  # a path
        if (attribute == 'states' and
            self._exclude_from >= 0 and
            i >= self._exclude_from):
            return '.'
        if attribute in self.__dict__:
            result = self.__dict__[attribute][i]
            return result
        if i == -1:
            k = -1
            i = self.lengths[-1] - 1
        else:
            if i < 0:
                i += len(self)
            if self.n_files == 1:
                k = 0
            else:
                k, i = self._get_local_index(i)
        if attribute == 'filenames':
            return self._fnames[k]
        try:
            return self._extract(k, attribute)[i]
        except:
            if attribute == 'states':
                return ''
            return nan
