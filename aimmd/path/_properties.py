"""
aimmd.path._properties
======================

Derived properties for :class:`aimmd.path.Path`.

This module defines :class:`~aimmd.path._properties.PathProperties`, a mixin
providing small computed properties and a few validated setters.

The properties here are intentionally lightweight and free of heavy I/O. More
substantial data access (loading from readers, caches, etc.) is handled by other
mixins such as :class:`~aimmd.path._get.PathGet` and
:class:`~aimmd.path._extract.PathExtract`.

Key concepts
------------
- A Path may span multiple files; `_fnames`, `_first`, `_last` store per-file
  segments.
- Segments may be forward or backward depending on whether `first <= last`.
- `exclude_from >= 0` marks a rejected path; some downstream logic treats such
  paths specially.

Notes
-----
- Many properties return read-only numpy arrays (writeable flag disabled).
- `shooting_index` setter supports non-integral values by delegating to
  `find_shooting_index`.
"""

# external imports
import numpy as np
from abc import ABC
from numbers import Integral

# aimmd imports
from .._config import MDA_CACHE

# Path's properties
class PathProperties(ABC):

    @property
    def fnames(self):
        """Per-file trajectory filenames composing this Path."""
        return np.array(self._fnames, dtype=str)

    @property
    def first(self):
        """Per-file first frame index (absolute index within each file)."""
        return np.array(self._first, dtype=int)

    @property
    def last(self):
        """Per-file last frame index (absolute index within each file)."""
        return np.array(self._last, dtype=int)

    @property
    def weight(self):
        """Statistical weight associated with this Path."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = float(weight)
    
    @property
    def exclude_from(self):
        """Rejection marker.

        Returns
        -------
        int
            If < 0, the path is accepted.
            If >= 0, frames at/after this index are considered invalid/rejected.
        """
        return self._exclude_from
    
    @exclude_from.setter
    def exclude_from(self, exclude_from):
        # Clip to [-1, len(self)] to keep consistent invariants.
        self._exclude_from = min(max(-1, int(exclude_from)), len(self))
    
    @property
    def shooting_index(self):
        """Index of the shooting point in global Path coordinates."""
        return self._shooting_index
    
    @shooting_index.setter
    def shooting_index(self, shooting_index):
        # Non-integral inputs trigger inference (e.g. 'find').
        if not isinstance(shooting_index, Integral):
            self._shooting_index = self.find_shooting_index()
            return
        self._shooting_index = min(max(0, shooting_index), len(self) - 1)
    
    @property
    def fname(self):
        """Last ("active") trajectory filename, or '' for an empty Path."""
        if not self.n_files:
            return ''
        return self._fnames[-1]
    
    @property
    def lengths(self):
        """Per-file segment lengths (number of frames in each file segment)."""
        return np.abs(self.first - self.last) + 1
    
    @property
    def offsets(self):
        """Cumulative sum of lengths, used for global→local index mapping."""
        return np.cumsum(self.lengths)
    
    @property
    def dt(self):
        """Estimate time step between frames.

        Notes
        -----
        - If times are in memory, uses the first two times.
        - Otherwise uses `middle('times') - initial('times')`.
        """
        if len(self) <= 1:
            return 1.
        if 'times' in self.__dict__:
            return np.diff(self.__dict__['times'][:2])[0]
        return self.middle('times') - self.initial('times')
    
    @property
    def indices(self):
        """Global Path indices (0..len(self)-1)."""
        return np.arange(len(self))
    
    @property
    def locs(self):
        """Absolute file-local locations corresponding to each Path frame.

        Returns
        -------
        numpy.ndarray
            Concatenation of per-file `range(first, last+step, step)`.
        """
        if not len(self):
            return np.array([], dtype=int)
        result = []
        for start, last in zip(self._first, self._last):
            step = 1 if start <= last else -1
            stop = last + step
            result.append(range(start, stop, step))
        result = np.concatenate(result, dtype=int)
        result.flags.writeable = False
        return result
    
    @property
    def filenames(self):
        """Per-frame filename array aligned with `locs`."""
        return np.repeat(self._fnames, self.lengths).astype(str)

    @property
    def accepted(self):
        """Whether the path is accepted (`exclude_from < 0`)."""
        return self._exclude_from < 0
    
    @accepted.setter
    def accepted(self, accepted):
        if accepted:
            self._exclude_from = -1
        elif self._exclude_from >= 0:
            self._exclude_from = 0
    
    @property
    def type(self):
        """Compact 4-character summary of the path.

        Convention
        ----------
        Returns a string of length 4 representing:
        - initial state label,
        - "middle" state label,
        - final state label,
        - shooting state label.

        Rejected/empty paths return '....'.
        """
        if not self.accepted:
            return '....'
        n_files = self.n_files
        if not n_files:
            return '....'
        if n_files == 1:
            states = self.states
            if len(states) > 1:
                return ((states[0] or '.') +
                        (states[1] or '.') +
                        (states[-1] or '.') +
                        (states[self._shooting_index] or '.'))
            return (states[0] or '.') * 4
        try:
            return ((self.initial('states') or '.') +
                    (self.middle('states') or '.') +
                    (self.final('states') or '.') +
                    (self.shooting('states') or '.'))
        except:
            return '....'
    
    @property
    def n_atoms(self):
        """Number of atoms per frame, inferred from the first trajectory file."""
        if self.n_files:
            return MDA_CACHE.get(self._fnames[0])[0].n_atoms
        return 0
    
    @property
    def n_files(self):
        """Number of underlying trajectory files contributing to the Path."""
        return len(self._fnames)
    
    @property
    def n_frames(self):
        """Number of *internal* frames."""
        return len(self.internal('indices'))
