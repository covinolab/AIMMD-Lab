"""
aimmd.path._magic
=================

Python protocol methods for :class:`aimmd.path.Path`.

This module defines :class:`PathMagic`, a mixin that makes Path instances behave
like lightweight, sliceable sequences of frames while also enforcing AIMMD's
attribute-caching rules.

Provided behavior
-----------------
- **Lazy attribute resolution** via ``__getattr__``: unknown public attributes are
  computed through ``self._get(attribute)`` (implemented by the Path getter mixin).
- **Validated assignment** via ``__setattr__`` for common cached arrays such as
  ``positions``, ``velocities``, ``dimensions``, ``times``, and ``states``.
- Sequence protocol:
  - ``len(path)`` counts frames across all stored segments.
  - ``path[i]`` returns a single frame (file-backed or in-memory).
  - ``path[a:b:step]`` returns a new Path view (step must be +1 or -1).
- ``path1 + path2`` concatenates segment lists, with a boundary merge when possible.

Design notes
------------
- File-backed access uses the global MDAnalysis reader cache ``aimmd._config.MDA_CACHE``.
- Property assignments are routed through :class:`aimmd.path._properties.PathProperties`.
- This file does not define the concrete Path class; it is mixed into it in
  :mod:`aimmd.path`.

Important
---------
This module adds *no new data model* to Path. It only defines protocol methods.
All heavy lifting (reader construction, local/global index mapping, derived array
construction) is delegated to the core Path implementation.
"""

# external imports
import numpy as np
from abc import ABC
from numbers import Integral
from itertools import islice
from MDAnalysis.coordinates.memory import MemoryReader

# aimmd imports
from .._config import MDA_CACHE
from ._properties import PathProperties

# Path's magic methods
class PathMagic(ABC):

    def __getattr__(self, attribute):
        """Fallback attribute access for lazily computed Path attributes.

        Parameters
        ----------
        attribute : str
            Attribute name being requested.

        Returns
        -------
        object
            The value returned by ``self._get(attribute)``.

        Raises
        ------
        AttributeError
            If ``attribute`` is private (starts with ``'_'``), conflicts with
            PathEnsemble-like names (``'paths'``/``'weights'``), or cannot be computed.

        Notes
        -----
        ``__getattr__`` is called only if normal attribute lookup fails. This method
        delegates to the Path getter mixin; it does not compute values itself.
        """
        if attribute in ('paths', 'weights') or attribute.startswith('_'):
            raise AttributeError(f'Path instance has no {attribute!r}')
        try:
            return self._get(attribute)
        except Exception as exception:
            raise AttributeError(f'{exception}')
    
    def __setattr__(self, attribute, value):
        """Validated assignment for cached arrays and properties.

        Parameters
        ----------
        attribute : str
            Attribute name being set.
        value : object
            Value to assign. For supported cached arrays, the value is converted to a
            NumPy array and validated for dtype/shape (see Notes).

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If attempting to set a read-only Path property, or if the provided value
            has an incompatible shape.

        Notes
        -----
        Special handling implemented by this method:

        - Names starting with ``'_'`` are stored directly in ``self.__dict__``.
        - If ``attribute`` corresponds to a :class:`PathProperties` property, the
          property's setter is used (and assignment is rejected if no setter exists).
        - Assigning ``None`` removes the cached array from ``self.__dict__``.
        - Cached array conventions enforced here:

          ``states`` (and ``true_states``)
              dtype ``'<U1'``, shape ``(len(self),)``.
          ``times``
              float dtype, shape ``(len(self),)``.
          ``positions`` / ``velocities``
              dtype ``float32``, shape ``(len(self), self.n_atoms, 3)``.
          ``dimensions``
              dtype ``float32``, shape ``(len(self), 6)``.
        """

        # private attributes
        if attribute[0] == '_':
            self.__dict__[attribute] = value
            return

        # property
        path_property = getattr(PathProperties, attribute, None)
        if isinstance(path_property, property):
            if path_property.fset is None:
                raise AttributeError(
                    f"can't set aimmd.Path property {attribute!r}")
            # dispatch to the property setter
            path_property.fset(self, value)
            return
      
        # reader and frames
        if attribute == 'reader':
            raise AttributeError(f"can't set aimmd.Path 'reader'")
        
        if attribute == 'frames':
            raise AttributeError(f"can't set aimmd.Path 'frames'")
    
        # true states -> states
        if attribute == 'true_states':
            attribute = 'states'
        
        # remove from dict
        if value is None:
            if attribute in self.__dict__:
                self.__dict__.pop(attribute)
            return
        
        # process (general)
        value = np.asarray(value)
        shape = value.shape
        target_shape = None
        
        # process states
        if attribute == 'states':
            value = value.astype('<U1')
            target_shape = (len(self),)
            
        # memory
        elif attribute == 'times':
            value = value.astype(float)
            target_shape = (len(self),)
            
        elif attribute in ('positions', 'velocities'):
            value = value.astype(np.float32)  # single precision
            target_shape = (len(self), self.n_atoms, 3)
            
        elif attribute == 'dimensions':
            value = value.astype(np.float32)  # single precision
            target_shape = (len(self), 6)
        
        elif shape[0] != len(self):
            raise AttributeError(
                f'{attribute!r} must have length'
                f'{len(self)}, got {shape[0]} instead')
        
        if target_shape is not None and shape != target_shape:
            raise AttributeError(
                f'{attribute!r} must have shape'
                f'{target_shape}, got {shape} instead')
                
        # assign
        self.__dict__[attribute] = value
        
    def __len__(self):
        """Return the total number of frames in the Path.

        Returns
        -------
        int
            Total frame count across all segments. For each segment, the contribution
            is ``abs(last - first) + 1``.
        """
        if not self._fnames:
            return 0
        length = 0
        for first, last in zip(self._first, self._last):
            length += abs(last - first) + 1
        return length

    def __repr__(self):
        """Return a short human-readable representation.

        Returns
        -------
        str
            Representation of the form ``'Path with <n> frames'``.
        """
        return f'Path with {len(self)} frames'
    
    def __getitem__(self, key):
        """Index or slice the Path.

        Parameters
        ----------
        key : int | slice
            - If an integer index is provided, return a single frame.
            - If a slice is provided, return a new Path view.

        Returns
        -------
        MDAnalysis Timestep-like | aimmd.path.Path
            - For integer indexing: a frame object.
              * If the Path is in-memory, this is produced via
                :class:`MDAnalysis.coordinates.memory.MemoryReader`.
              * Otherwise, the frame is taken from the cached reader in ``MDA_CACHE``.
            - For slicing: a new Path instance whose segment lists are adjusted.

        Raises
        ------
        TypeError
            If ``key`` is neither an integer nor a slice, or if a slice step is not
            +1 or -1.

        Notes
        -----
        Slicing adjusts internal metadata (``_exclude_from`` and ``_shooting_index``)
        to preserve their meaning in the returned view.
        """
        if isinstance(key, Integral):
            if self.in_memory():
                result = MemoryReader(
                    self.positions[key].copy()[None],
                    velocities=self.velocities[key].copy()[None],
                    dimensions=self.dimensions[key].copy()[None])[0]
                result.time = self.times[key]
                return result
            if key < 0:
                key += len(self)
            k, i = self._get_local_loc(key)
            return MDA_CACHE.get(self._fnames[k])[i]
        
        # get indices
        if not isinstance(key, slice):
            raise TypeError(
                f'path indices must be integers or slices, not {type(key)}')
        start, stop, step = key.indices(len(self))
        if abs(step) != 1:
            raise TypeError(f'path slice indices must have '
                            f'step=+1 or step=-1, got {step}')
        
        from . import Path

        # special case: empty path
        if start == stop:
            return Path(exclude_from=self._exclude_from)

        # find limits
        k_start, i_first = self._get_local_loc(start)
        k_last, i_last = self._get_local_loc(stop - step)
        k_step = step
        k_stop = k_last + k_step
        if k_stop < 0:
            k_stop = None
        
        # will fill
        fnames = self._fnames[k_start:k_stop:k_step]
        first = self._first[k_start:k_stop:k_step]
        last = self._last[k_start:k_stop:k_step]
        
        # swap
        if step < 0:
            first, last = last, first
        
        # adjust first and last
        first[0] = i_first
        last[-1] = i_last
        
        # assign
        result = object.__new__(Path)
        result._fnames = fnames
        result._first = first
        result._last = last
        result._weight = self.weight
        
        # process exclude from
        exclude_from = -1
        if self._exclude_from >= 0:
            if step < 0:
                exclude_from = 0
            else:
                exclude_from = max(0, self._exclude_from - start)
                if exclude_from > len(result):
                    exclude_from = -1
        result._exclude_from = exclude_from
        
        # process shooting index
        if step > 0:
            if self._shooting_index <= start:
                result._shooting_index = 0
            elif self._shooting_index >= stop - 1:
                result._shooting_index = abs(stop - start) - 1
            else:
                result._shooting_index = (self._shooting_index - start)
        elif self._shooting_index <= stop + 1:
            result._shooting_index = abs(stop - start) - 1
        elif self._shooting_index >= start:
            result._shooting_index = 0
        else:
            result._shooting_index = (
                len(self) - self._shooting_index - 1 - stop + 1)
                
        # attributes
        for attribute, value in islice(
            self.__dict__.items(), 6, None):
            result.__dict__[attribute] = value[key]
        
        return result

    def __eq__(self, other):
        """Elementwise equality on cached arrays.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            True if both objects have identical ``__dict__`` keys and all associated
            arrays are equal under ``numpy.array_equal``.

        Notes
        -----
        This comparison is based on cached in-memory attributes, not on file segments.
        """
        if len(self.__dict__) != len(other.__dict__):
            return False
        for attribute in self.__dict__:
            if attribute not in other.__dict__:
                return False
            if not np.array_equal(
                self.__dict__[attribute],
                other.__dict__[attribute]):
                return False
        return True
    
    def __add__(self, other):
        """Concatenate two Paths.

        Parameters
        ----------
        other : aimmd.path.Path
            The Path to append.

        Returns
        -------
        aimmd.path.Path
            A new Path consisting of the segments from ``self`` followed by those from
            ``other``. If the last segment of ``self`` and the first segment of
            ``other`` refer to the same file and their local indices are contiguous,
            the boundary is merged into a single segment.

        Raises
        ------
        TypeError
            If ``other`` is not an :class:`aimmd.path.Path`.

        Notes
        -----
        - The resulting weight is the sum of the two path weights.
        - The shooting index is taken from the first operand.
        """
        from . import Path
        if not isinstance(other, Path):
            raise TypeError(f'can only add aimmd.Path instance to {self!r}')
        if not other._fnames:
            return self[:]
        if not self._fnames:
            return other[:]
        
        result = object.__new__(Path)

        # can you merge?
        if (self._fnames[-1] == other._fnames[0] and
            abs(self._last[-1] - other._first[0]) == 1 and
           ((self._first[-1] <= self._last[-1] and
             other._first[-1] <= other._last[-1]) or
            (self._first[-1] > self._last[-1] and
             other._first[-1] > other._last[-1]))):
            result._fnames = self._fnames + other._fnames[1:]
            result._first = self._first + other._first[1:]
            result._last = self._last[:-1] + other._last
        else:  # just join
            result._fnames = self._fnames + other._fnames
            result._first = self._first + other._first
            result._last = self._last + other._last
            
        # all the rest
        result._shooting_index = self._shooting_index
        result._weight = self._weight + other._weight
        if (exclude_from := self._exclude_from) < 0:
            if other._exclude_from >= 0:
                exclude_from = len(self)
            else:
                exclude_from = -1
        result._exclude_from = exclude_from
        return result

    def __radd__(self, other):
        """Right-addition helper for ``sum``.

        Parameters
        ----------
        other : object
            If ``other == 0``, treat it as the identity element.

        Returns
        -------
        aimmd.path.Path
            ``self`` if ``other == 0``, else ``self + other``.
        """
        if other == 0:
            return self
        return self.__add__(other)
