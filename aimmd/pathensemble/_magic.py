"""
aimmd.pathensemble._magic
========================

"Magic" methods for :class:`~aimmd.pathensemble.PathEnsemble`.

This mixin makes ``PathEnsemble`` behave like a Python container while keeping
ensemble-specific semantics:

- list-like access: ``len(ensemble)``, iteration, indexing, slicing,
- concatenation: ``ensemble + other``, ``sum([...])`` support,
- vectorized attribute access: ``ensemble.values`` returns a list of per-path
  values extracted over the default range (see :meth:`PathEnsembleHelpers._get`),
- controlled attribute setting: setting non-private attributes is restricted to
  protect derived properties and avoid ambiguous "broadcast" assignments.
"""

# external
import numpy as np
import bisect
from abc import ABC
from numbers import Integral
from collections.abc import Iterable

# aimmd imports
from .utils import get_paths
from ..path import Path
from ._properties import PathEnsembleProperties
from ..core.utils import get_local_index


class PathEnsembleMagic(ABC):
    
    def __getitem__(self, i):
        """
        Index or slice the ensemble.

        Parameters
        ----------
        i : int or slice or array-like
            If an integer, return the corresponding :class:`~aimmd.path.Path`.
            Otherwise, build a new :class:`~aimmd.pathensemble.PathEnsemble`
            containing the selected paths.

        Returns
        -------
        Path or PathEnsemble
            A single path for integer indexing, otherwise a new ensemble.
        """
        if isinstance(i, Integral):
            return self._paths[i]
        from . import PathEnsemble
        result = object.__new__(PathEnsemble)
        result._paths = list(self.paths[i].flatten())
        return result

    def __setitem__(self, key, value):
        """
        Replace a single path in the ensemble.

        Parameters
        ----------
        key : int
            Index to replace.
        value
            Any path-like input accepted by :func:`~aimmd.pathensemble.utils.get_paths`.

        Raises
        ------
        TypeError
            If ``value`` does not contain any path.
        """
        paths = get_paths(value)
        if not len(paths):
            raise TypeError('no paths found')
        self._paths[key] = paths[0]

    def __iter__(self):
        """Iterate over member paths."""
        return self._paths.__iter__()

    def __len__(self):
        """Number of member paths."""
        return len(self._paths)

    def __repr__(self):
        """Readable summary string."""
        return f'PathEnsemble with {len(self)} paths'

    def __eq__(self, other):
        """
        Equality by path identity/ordering.

        Two ensembles are considered equal if ``np.array_equal(self.paths, other.paths)``.
        """
        return np.array_equal(self.paths, other.paths)

    def __add__(self, other):
        """
        Concatenate with another ensemble or append a single path.

        Parameters
        ----------
        other : PathEnsemble or Path or iterable
            - If ``PathEnsemble``: concatenate path lists.
            - If ``Path``: append.
            - If an iterable: accepted only in a very narrow way as implemented.

        Returns
        -------
        PathEnsemble
            A new ensemble (not in-place).
        """
        from . import PathEnsemble
        if isinstance(other, PathEnsemble):
            paths = self._paths + other._paths
        elif isinstance(other, Path):
            paths = self._paths + [other]
        elif not isinstance(other, Iterable) or len(other):
            raise TypeError(f'Cannot add {other!r} to {self}.')
        else:
            paths = self._paths
        from . import PathEnsemble
        result = object.__new__(PathEnsemble)
        result._paths = paths
        return result

    def __radd__(self, other):
        """
        Support ``sum([...])`` patterns.

        ``sum`` starts with 0; ``0 + ensemble`` should return the ensemble.
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __getattr__(self, attribute):
        """
        Vectorized access to per-path attributes.

        Any attribute name that is not a protected name is interpreted as an
        internal request to extract that attribute over each path's default range
        (see :meth:`aimmd.pathensemble._helpers.PathEnsembleHelpers._get`).

        Raises
        ------
        AttributeError
            If the attribute is protected (private or special-cased).
        """
        """Internal concatenated attributes."""
        if (attribute in ('first', 'last', 'weight', 'cache') or
            attribute.startswith('_')):
            raise AttributeError(f"can't get {attribute!r}")
        return self._get(attribute)

    def __setattr__(self, attribute, value):
        """
        Controlled attribute assignment.

        - Private attributes (starting with ``_``) are set directly.
        - If ``attribute`` corresponds to a property on
          :class:`~aimmd.pathensemble._properties.PathEnsembleProperties`, then
          its setter is used (if available).
        - Setting a non-private attribute to ``None`` deletes that attribute from
          each underlying path.
        - Any other assignment is refused to prevent ambiguous broadcasting.

        Raises
        ------
        AttributeError
            For forbidden assignments.
        """
        # private attributes
        if attribute[0] == '_':
            self.__dict__[attribute] = value
            return

        # property
        path_property = getattr(PathEnsembleProperties, attribute, None)
        if isinstance(path_property, property):
            if path_property.fset is None:
                raise AttributeError(
                    f"can't set aimmd.Path property {attribute!r}")
            # dispatch to the property setter
            path_property.fset(self, value)
            return

        # delete
        if value is None:
            for i, path in enumerate(self._paths):
                setattr(path, attribute, None)
            return

        # raise
        raise AttributeError(f"can't set {attribute!r} to aimmd.PathEnsemble"
                             ", must do that path-by-path")

    def __bool__(self):
        """
        Truthiness based on whether `self` has any paths with frames.

        Returns
        -------
        bool
            ``True`` if at least one path has ``len(path) > 0``.
        """
        for path in self._paths:
            if len(path):
                return True
        return False
