"""
aimmd.pathensemble._positions
=============================

Position selectors for :class:`aimmd.pathensemble.PathEnsemble`.

This module defines :class:`PathEnsemblePositions`, a mixin that broadcasts
the position-selection API of :class:`aimmd.path.Path` over an ensemble
stored in ``self._paths``.

The ensemble does not define new semantics. Each method delegates to the
corresponding Path method and only normalizes the container type of the
returned collection.

Storage model
-------------
A PathEnsemble stores:

- ``_paths`` : list[Path]
  Ordered list of Path objects. The order is preserved in all returns.

Return modes
------------
Many selectors accept an ``attribute`` argument which is passed through to
the underlying Path method. At the ensemble level, results are normalized
as follows:

- ``attribute == 'frames'``
  Return the raw list of per-path frame objects.

- ``attribute == 'reader'``
  Assemble the per-path frames into an in-memory MDAnalysis reader using
  :func:`aimmd.core.utils.memory_reader_from_timesteps`.

- otherwise
  Convert the per-path results to a NumPy array.

Extrema selectors
-----------------
For extrema methods (``min*`` / ``max*``), ``source`` specifies the stream
used to locate the extremum along a path (typically ``'values'``). The
returned ``attribute`` is evaluated at that extremum location, as defined
by the Path implementation.

Forward/backward portions
-------------------------
Path objects in AIMMD are conceptually split at the shooting point:

- backward portion
    Frames from the shooting point back to the initial end, inclusive of
    the shooting frame.

- forward portion
    Frames from the shooting point to the final end, inclusive of the
    shooting frame.

This mixin does not reimplement the split. It delegates to Path for the
exact indexing rules, but the intended meaning is "up to / starting from
the shooting point, with the shooting point included in both portions".

Notes
-----
- No mutation.
- No caching.
- All selection semantics are defined by Path.
"""

# external
import numpy as np
from abc import ABC

# aimmd imports
from .utils import process_path_position_result
from ..core.utils import memory_reader_from_timesteps


# position methods for path ensemble class
class PathEnsemblePositions(ABC):
    """
    Ensemble-level wrappers around Path positional selectors.

    This mixin assumes the host class defines ``self._paths``.
    """

    def initial(self, attribute):
        """
        Initial-point selector for the ensemble.

        Calls ``path.initial(attribute)`` for each path and returns one result
        per path. Container normalization follows the module-level policy.
        """
        result = [path.initial(attribute) for path in self._paths]
        return process_path_position_result(result, attribute)


    def shooting(self, attribute):
        """
        Shooting-point selector for the ensemble.

        Calls ``path.shooting(attribute)`` for each path and returns one result
        per path. Container normalization follows the module-level policy.
        """
        result = [path.shooting(attribute) for path in self._paths]
        return process_path_position_result(result, attribute)

    def final(self, attribute):
        """
        Final-point selector for the ensemble.

        Calls ``path.final(attribute)`` for each path and returns one result
        per path. Container normalization follows the module-level policy.
        """
        result = [path.final(attribute) for path in self._paths]
        return process_path_position_result(result, attribute)

    def middle(self, attribute):
        """
        Middle-point selector for the ensemble.

        Calls ``path.middle(attribute)`` for each path. The definition of
        "middle" is delegated entirely to Path.
        """
        result = [path.middle(attribute) for path in self._paths]
        return process_path_position_result(result, attribute)

    def min(self, attribute, source='values'):
        """
        Return `attribute` at the minimum of `source`, one per path.

        For each path, this delegates to ``path.min(attribute, source)``.
        The Path method defines how the minimum is located (based on `source`)
        and what is returned at that location (based on `attribute`).

        Typical usage is:
        - `source` selects the scalar stream minimized along the path
          (often `'values'`).
        - `attribute` selects what is returned at the argmin location
          (e.g. frames, positions, descriptors, values).

        Container normalization follows the module-level policy.
        """
        result = [path.min(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def max(self, attribute, source='values'):
        """
        Return `attribute` at the maximum of `source`, one per path.

        Delegates to ``path.max(attribute, source)`` for each path. See
        :meth:`min` for the meaning of `attribute` and `source`.
        """
        result = [path.max(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def min_backward(self, attribute, source='values'):
        """
        Return `attribute` at the backward-portion minimum of `source`,
        per path.

        The backward portion is the part of the path running from the shooting
        point back to the initial end, with the shooting frame included. The
        exact slicing and indexing are defined by Path.

        Delegates to ``path.min_backward(attribute, source)``.
        """
        result = [path.min_backward(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def max_backward(self, attribute, source='values'):
        """
        Return `attribute` at the backward-portion maximum of `source`,
        per path.

        See :meth:`min_backward` for the meaning of the backward portion.

        Delegates to ``path.max_backward(attribute, source)``.
        """
        result = [path.max_backward(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def min_forward(self, attribute, source='values'):
        """
        Return `attribute` at the forward-portion minimum of `source`,
        per path.

        The forward portion is the part of the path running from the shooting
        point to the final end, with the shooting frame included. The exact
        slicing and indexing are defined by Path.

        Delegates to ``path.min_forward(attribute, source)``.
        """
        result = [path.min_forward(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def max_forward(self, attribute, source='values'):
        """
        Return `attribute` at the forward-portion maximum of `source`,
        per path.

        See :meth:`min_forward` for the meaning of the forward portion.

        Delegates to ``path.max_forward(attribute, source)``.
        """
        result = [path.max_forward(attribute, source) for path in self._paths]
        return process_path_position_result(result, attribute)

    def backward(self, attribute):
        """
        Return the backward portion for each path in a PathEnsemble object.

        The backward portion is the segment from the shooting point back to
        the initial end, with the shooting frame included. This method calls
        ``path.backward(attribute)`` for each path and returns the list of
        per-path results.
        """
        result = [path.backward(attribute) for path in self._paths]
        if attribute == 'self':
            result = PathEnsemble(result)
        return result

    def forward(self, attribute):
        """
        Return the forward portion for each path in a PathEnsemble object.

        The forward portion is the segment from the shooting point to the
        final end, with the shooting frame included. This method calls
        ``path.forward(attribute)`` for each path and returns the list of
        per-path results.
        """
        result = [path.forward(attribute) for path in self._paths]
        if attribute == 'self':
            result = PathEnsemble(result)
        return result
        
    def all(self, attribute):
        """
        Return a copy of `self`.
        """
        result = [path.all(attribute) for path in self._paths]
        if attribute == 'self':
            result = PathEnsemble(result)
        return result
        
    def internal(self, attribute):
        """
        Return the internal portion for each path in a PathEnsemble object.

        Calls ``path.internal(attribute)`` for each path and returns the list
        of per-path results. The definition of "internal" is delegated to Path.
        """
        result = [path.internal(attribute) for path in self._paths]
        if attribute == 'self':
            result = PathEnsemble(result)
        return result 
