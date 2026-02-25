"""
aimmd.pathensemble._magic
========================

Magic methods for :class:`aimmd.pathensemble.PathEnsemble`.

This mixin gives `PathEnsemble` list-like behavior:

- truthiness (``bool(ensemble)``),
- indexing/slicing (``ensemble[i]`` and ``ensemble[mask]``),
- iteration and length,
- addition/concatenation (``ensemble + other``),
- attribute forwarding for bulk access.

Attribute forwarding
--------------------
For attributes not explicitly defined on `PathEnsemble`, `__getattr__` forwards
to :meth:`aimmd.pathensemble._helpers.PathEnsembleHelpers._get` to retrieve
per-path data, *unless* the attribute is private or one of a small set of
reserved names.

Important implementation note
-----------------------------
This file contains **two** definitions of ``__bool__``. In Python, the second
definition overrides the first. Therefore, the effective truthiness check is:

- True if **any** contained path has non-zero length,
- False otherwise.

The earlier (stricter) definition remains in the source but is unused.
"""

# list-like dunder methods for PathEnsemble

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


class PathEnsembleMagic(ABC):
    """
    Mixin implementing Python "dunder" methods for PathEnsemble.

    The underlying storage is ``self._paths``.
    """

    def __getitem__(self, i):
        """
        Index or slice the ensemble.

        Parameters
        ----------
        i : int or slice or array-like
            - If `int`: return the corresponding Path.
            - Otherwise: treat as a NumPy-like index (slice, mask, list of indices)
              and return a *new* PathEnsemble containing the selected paths.

        Returns
        -------
        aimmd.path.Path or aimmd.pathensemble.PathEnsemble
        """
        if isinstance(i, Integral):
            return self._paths[i]
        from . import PathEnsemble

        # construct without calling __init__ to avoid re-normalizing paths
        result = object.__new__(PathEnsemble)
        result._paths = list(self.paths[i].flatten())
        return result

    def __setitem__(self, key, value):
        """
        Replace a single path in the ensemble.

        Parameters
        ----------
        key : int or slice
            Index to replace.
        value : Path or PathEnsemble or iterable
            The replacement; normalized via :func:`get_paths`.

        Raises
        ------
        TypeError
            If `value` yields no paths.
        """
        paths = get_paths(value)
        if not len(paths):
            raise TypeError("no paths found")
        self._paths[key] = paths[0]

    def __iter__(self):
        """Iterate over contained paths."""
        return self._paths.__iter__()

    def __len__(self):
        """Number of paths in the ensemble."""
        return len(self._paths)

    def __repr__(self):
        """Human-readable representation."""
        return f"PathEnsemble with {len(self)} paths"

    def __eq__(self, other):
        """
        Equality check based on path object array equality.

        Notes
        -----
        This uses :func:`numpy.array_equal` on the `paths` property. It compares
        object identity/equality as provided by the Path objects.
        """
        return np.array_equal(self.paths, other.paths)

    def __add__(self, other):
        """
        Concatenate with another ensemble or a single Path.

        Parameters
        ----------
        other : PathEnsemble or Path or iterable
            - If PathEnsemble: concatenate path lists.
            - If Path: append it.
            - If iterable: behavior depends on its truthiness/length (see code).

        Returns
        -------
        PathEnsemble
            New ensemble instance with combined paths.
        """
        from . import PathEnsemble
        if isinstance(other, PathEnsemble):
            paths = self._paths + other._paths
        elif isinstance(other, Path):
            paths = self._paths + [other]
        elif not isinstance(other, Iterable) or len(other):
            raise TypeError(f"Cannot add {other!r} to {self}.")
        else:
            paths = self._paths
        from . import PathEnsemble
        result = object.__new__(PathEnsemble)
        result._paths = paths
        return result

    def __radd__(self, other):
        """
        Support `sum([...])` patterns.

        Convention: `sum` starts with 0, so `0 + ensemble` should return ensemble.
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __getattr__(self, attribute):
        """
        Forward unknown attribute access to bulk per-path retrieval.

        Parameters
        ----------
        attribute : str
            Attribute name requested by user code.

        Returns
        -------
        list
            One entry per path: ``path._get(attribute, *path._range(where))``
            using the default `where='internal'` logic of `_get`.

        Raises
        ------
        AttributeError
            For reserved attributes, private attributes, or if the attribute is
            not meant to be bulk-fetched.
        """
        # protect internal fields and some reserved names
        if (attribute in ("first", "last", "weight", "cache") or attribute.startswith("_")):
            raise AttributeError(f"can't get {attribute!r}")
        return self._get(attribute)

    def __setattr__(self, attribute, value):
        """
        Attribute assignment policy.

        - Private attributes (starting with `_`) are assigned normally.
        - If `attribute` is a read-only property on PathEnsembleProperties, raise.
        - If `value` is None: delete the attribute on each contained path
          (by setting it to None).
        - Otherwise: disallow setting arbitrary attributes on the ensemble
          (must set per-path).

        This prevents ambiguous bulk mutation of `Path` fields.
        """
        # private attributes bypass the policy checks
        if attribute[0] == "_":
            self.__dict__[attribute] = value
            return

        # if this is a known property on PathEnsembleProperties, use its setter
        path_property = getattr(PathEnsembleProperties, attribute, None)
        if isinstance(path_property, property):
            if path_property.fset is None:
                raise AttributeError(f"can't set aimmd.Path property {attribute!r}")
            path_property.fset(self, value)
            return

        # "delete" semantics for bulk removal
        if value is None:
            for i, path in enumerate(self._paths):
                setattr(path, attribute, None)
            return

        # disallow ambiguous bulk set
        raise AttributeError(
            f"can't set {attribute!r} to aimmd.PathEnsemble" ", must do that path-by-path"
        )

    def __bool__(self):
        """
        Truthiness of the ensemble.

        Returns
        -------
        bool
            True if at least one contained path has non-zero length; False if all
            paths are empty or ensemble is empty.
        """
        for path in self._paths:
            if len(path):
                return True
        return False
