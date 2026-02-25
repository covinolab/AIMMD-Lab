"""
aimmd.pathensemble._properties
==============================

Properties and array-like views for :class:`aimmd.pathensemble.PathEnsemble`.

This module provides two small building blocks used throughout the
`pathensemble` package:

- :class:`PathProperties`
  An :class:`aimmd.core.base.AbstractArray` implementation that exposes a
  Path attribute as an array-like view over an ensemble. It supports both
  reading and writing and mirrors NumPy indexing semantics.

- :class:`PathEnsembleProperties`
  A mixin that defines common convenience properties for a PathEnsemble,
  including views for per-path weights/acceptance flags and a few derived
  counters.

Storage model
-------------
A PathEnsemble stores its members in:

- ``_paths`` : list[aimmd.path.Path]
  Ordered list of Path objects.

Each Path is expected to expose the attributes referenced here, in particular:
``weight``, ``accepted``, ``exclude_from``, ``shooting_index``,
``_fnames``, ``files``, ``type`` and ``fname``.

Execution model
---------------
`PathProperties` never copies Path objects. It resolves values on demand:

- integer indexing reads or writes a single Path attribute;
- slice/array indexing maps indices through ``np.arange(len(ensemble))[key]``
  and returns a NumPy array (or assigns values element-wise).

For setters, scalar (or single-element) inputs are broadcast to match the
number of selected paths.

Notes
-----
- ``accepted`` is implemented as a derived view over ``exclude_from``.
  Setting ``accepted`` writes ``exclude_from = -accepted.astype(int)``,
  matching the Path convention used in AIMMD: negative values indicate
  "not excluded".
- The :meth:`PathEnsembleProperties.path` property returns the most recent
  "usable" Path by scanning from the end and checking ``_weight`` and
  ``_exclude_from``. This is used to define a representative ``fname``.
"""

# external
import numpy as np
from abc import ABC
from numbers import Integral

# aimmd imports
from .utils import get_paths
from ..core.base import AbstractArray


# path properties class
class PathProperties(AbstractArray):
    """
    Array-like view of a Path attribute over a PathEnsemble.

    Parameters
    ----------
    pathensemble : PathEnsemble
        Owner ensemble providing ``_paths`` and ``__len__``.
    attribute : str
        Name of the attribute to expose on each Path.
    dtype : type, default=float
        dtype used when returning arrays and when coercing assigned values.

    Notes
    -----
    - Integer keys access a single Path attribute.
    - Non-integer keys are interpreted as NumPy-style indexing applied to
      ``np.arange(len(pathensemble))``.
    - Setting with a scalar (or a single-element array) broadcasts that value
      to all selected paths.
    """

    def __init__(self, pathensemble, attribute, dtype=float):
        self.pathensemble = pathensemble
        self.attribute = attribute
        self.dtype = dtype
    
    def __getitem__(self, key):
        """
        Retrieve one or more attribute values.

        Parameters
        ----------
        key : int or slice or array-like
            Indexing key. If `key` is an integer, a single value is returned.
            Otherwise, the key is applied to ``np.arange(len(pathensemble))``
            and the selected values are returned as a NumPy array.

        Returns
        -------
        object or numpy.ndarray
            Single attribute value (int key) or an array of values.
        """
        if isinstance(key, Integral):
            path = self.pathensemble._paths[key]
            return getattr(path, self.attribute)
        return np.array([
            getattr(self.pathensemble._paths[i], self.attribute)
            for i in np.arange(len(self.pathensemble))[key]],
                        dtype=self.dtype)
    
    def _array(self):
        """
        Materialize the full attribute vector.

        Returns
        -------
        numpy.ndarray
            Array of length ``len(pathensemble)`` containing the attribute
            for every path.
        """
        return np.array([
            getattr(path, self.attribute) for path in
            self.pathensemble._paths])
    
    def __setitem__(self, key, values):
        """
        Assign one or more attribute values.

        Parameters
        ----------
        key : int or slice or array-like
            Selection of paths to modify.
        values : scalar or array-like
            Values to assign. Scalars (and single-element arrays) are broadcast
            to the number of selected paths. Otherwise, the first dimension must
            match the number of selected indices.

        Notes
        -----
        Assignment is performed path-by-path via ``setattr``.
        """
        if isinstance(key, Integral):
            path = self.pathensemble._paths[key]
            setattr(path, self.attribute, values)
            return
        
        key = np.arange(len(self.pathensemble))[key]
        values = np.asarray(values, dtype=self.dtype)
        
        # broadcast scalar or single-element
        if values.ndim == 0:
            values = np.full(key.shape, values)
        elif values.shape[0] == 1 and key.shape[0] != 1:
            values = np.full(key.shape, values.item())
        
        # assign
        for i, value in zip(key, values):
            path = self.pathensemble._paths[i]
            setattr(path, self.attribute, value)


# path ensemble properties
class PathEnsembleProperties(ABC):
    """
    Convenience properties for PathEnsemble-like classes.

    This mixin expects:
    - ``self._paths`` storing Path objects,
    - ``__len__`` implemented by the host class,
    - Path attributes referenced below (weight/accepted/etc.).

    The majority of setters are implemented via :class:`PathProperties`
    to keep indexing/broadcasting behavior consistent across attributes.
    """

    @property
    def path(self):
        """
        Representative Path for the ensemble.

        The last Path in ``self._paths`` is scanned backwards and the first
        Path satisfying:

        - ``path._weight`` is truthy
        - ``path._exclude_from < 0``

        is returned. If no path matches, returns None.

        Notes
        -----
        This is primarily used to provide a stable ``fname`` even when the
        most recent paths are excluded or not yet weighted.
        """
        # last ok path
        for path in self._paths[::-1]:
            if path._weight and path._exclude_from < 0:
                return path

    @property
    def fname(self):
        """
        Filename of the representative Path.

        Returns
        -------
        str
            ``path.fname`` if a representative path exists, otherwise ''.
        """
        if path := self.path:
            return path.fname
        return ''

    @property
    def n_paths(self):
        """Number of paths in the ensemble (alias for ``len(self)``)."""
        return len(self)

    @property
    def paths(self):
        """
        NumPy object array view of ``self._paths``.

        Returns
        -------
        numpy.ndarray
            Object array containing Path objects.
        """
        return np.fromiter(self._paths, dtype=object)

    @paths.setter
    def paths(self, paths):
        """
        Set ``self._paths`` from a heterogeneous input.

        Parameters
        ----------
        paths : object
            Accepted by :func:`aimmd.pathensemble.utils.get_paths` which
            normalizes to a list of Path objects.
        """
        self._paths = get_paths(paths)

    @property
    def offsets(self):
        """
        Cumulative frame offsets across paths.

        Returns
        -------
        numpy.ndarray
            ``np.cumsum([len(path) for path in self.paths])``.
        """
        return np.cumsum([len(path) for path in self.paths])

    @property
    def weights(self):
        """
        Array-like view of per-path weights. See `Path.weight`.

        Returns
        -------
        PathProperties
            View over the Path attribute ``weight``.
        """
        return PathProperties(self, 'weight', float)

    @property
    def accepted(self):
        """
        Array-like view of per-path acceptance flags. See `Path.accepted`.

        Returns
        -------
        PathProperties
            View over the Path attribute ``accepted``.

        Notes
        -----
        Setting this property does not write ``accepted`` directly. Instead it
        writes ``exclude_from`` (see setter below).
        """
        return PathProperties(self, 'accepted', bool)

    @property
    def exclude_from(self):
        """
        Array-like view of `exclude_from` for every path. See `Path.exclude_from`.
        
        Returns
        -------
        PathProperties
            View over the Path attribute ``exclude_from``.
        """
        return PathProperties(self, 'exclude_from', int)

    @property
    def true_states(self):
        """
        Cached 'true_states' array for the ensemble.  See `Path.true_states`.

        Returns
        -------
        object
            Delegated to the host class via ``self._get('true_states')``.
        """
        return self._get('true_states')

    @property
    def shooting_indices(self):
        """
        Array-like view of per-path shooting indices.

        Returns
        -------
        PathProperties
            View over the Path attribute ``shooting_index``.
        """
        return PathProperties(self, 'shooting_index', int)

    @property
    def fnames(self):
        """
        Concatenated trajectory filenames of all paths.

        Returns
        -------
        numpy.ndarray
            Concatenation of each path's internal ``_fnames`` list.
            Empty array if there are no paths.
        """
        if not len(self._paths):
            return np.array([], dtype=str)
        return np.concatenate([path._fnames for path in self._paths])

    @property
    def n_files(self):
        """
        Total number of trajectory files across all paths.

        Returns
        -------
        int
            Sum of ``len(path.files)`` over all paths.
        """
        return sum([len(path.files) for path in self._paths])

    @weights.setter
    def weights(self, weights):
        """
        Set per-path weights.

        Parameters
        ----------
        weights : array-like
            Values assigned to Path attribute ``weight``.
        """
        PathProperties(self, 'weight', float)[:] = weights

    @accepted.setter
    def accepted(self, accepted):
        """
        Set per-path acceptance flags.

        Parameters
        ----------
        accepted : array-like of bool
            Acceptance flags.

        Notes
        -----
        Acceptance is implemented via the exclusion convention:
        ``exclude_from = -accepted.astype(int)``.
        """
        exclude_from = - np.asarray(accepted).astype(int)
        PathProperties(self, 'exclude_from', int)[:] = exclude_from

    @exclude_from.setter
    def exclude_from(self, exclude_from):
        """
        Set per-path exclusion flags.

        Parameters
        ----------
        exclude_from : array-like of int
            Values assigned to Path attribute ``exclude_from``.
        """
        PathProperties(self, 'exclude_from', int)[:] = exclude_from

    @shooting_indices.setter
    def shooting_indices(self, shooting_indices):
        """
        Set per-path shooting indices.

        Parameters
        ----------
        shooting_indices : array-like of int
            Values assigned to Path attribute ``shooting_index``.
        """
        PathProperties(self, 'shooting_index', int)[:] = shooting_indices

    @property
    def lengths(self):
        """
        Path lengths as stored.

        Returns
        -------
        numpy.ndarray
            ``np.array([len(path) for path in self._paths])``.
        """
        return np.array([len(path) for path in self._paths])

    @property
    def n_frames(self):
        """
        Effective number of frames per path, excluding boundary-state frames.

        This property corrects the stored length of each path by removing
        boundary frames that correspond to end states, matching the ensemble
        convention used elsewhere in AIMMD.

        For a path of length <= 1, the stored length is returned unchanged.

        For longer paths, the correction depends on the Path state triplet
        ``path.type``:

        - If ``states[0] != states[1]`` then the first frame is treated as an
          end-state boundary and is excluded from the effective count.
        - If ``states[1] != states[2]`` then the last frame is treated as an
          end-state boundary and is excluded from the effective count.

        Returns
        -------
        numpy.ndarray
            Integer array of effective frame counts.
        """
        n_frames = []
        for path in self._paths:
            length = len(path)
            if length <= 1:
                n_frames.append(length)
            else:
                states = path.type
                if states[0] != states[1]:
                    length -= 1
                if states[1] != states[2]:
                    length -= 1
                n_frames.append(length)
        return np.array(n_frames, dtype=int)
