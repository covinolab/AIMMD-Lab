"""
aimmd.path._positions
=====================

Positional accessors for :class:`aimmd.path.Path`.

This mixin provides small convenience wrappers to retrieve a Path attribute at
specific *locations* (initial/final/middle/shooting) or over specific *ranges*
(backward/forward/internal). It also provides helpers to return an attribute at
the location where another quantity (``source``) attains its minimum/maximum.

The implementation delegates all indexing logic to private Path helpers:

- ``_position(index, attribute)``
- ``_get(attribute, start=None, stop=None)``
- ``_range(kind)``
- ``_extreme(attribute, reducer, kind, source)``

No heavy computation is done here; this is an API-layer convenience mixin.

Notes
-----
- Locations are defined in **global Path indexing** (0..len(self)-1).
- “backward”, “forward”, and “internal” ranges are defined by the core Path
  implementation (via ``_range``) and depend on the shooting point index and/or
  path conventions used by AIMMD. E.g., backward is up until (and included)
  the shooting point, while forward is from the shooting point on.
"""

# external
import numpy as np
from abc import ABC

# position methods for path class
class PathPositions(ABC):

    def initial(self, attribute):
        """
        Return an attribute at the initial (first) frame.

        Parameters
        ----------
        attribute : str
            Name of the attribute to retrieve. Typical values are
            ``'positions'``, ``'velocities'``, ``'dimensions'``, ``'times'``,
            ``'states'``, ``'values'``, or any other key supported by the Path
            getter logic.

        Returns
        -------
        object
            The value of ``attribute`` at global index 0.

        Notes
        -----
        This is equivalent to ``self._position(0, attribute)``.
        """
        return self._position(0, attribute)

    def final(self, attribute):
        """
        Return an attribute at the final (last) frame.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve (see :meth:`initial`).

        Returns
        -------
        object
            The value of ``attribute`` at global index ``-1``.
        """
        return self._position(-1, attribute)

    def middle(self, attribute):
        """
        Return an attribute at the middle frame.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

            Special case:
            - if ``attribute == 'indices'``, this method returns the integer global
              index chosen as “middle” by the Path (see Notes).

        Returns
        -------
        object
            - If ``attribute == 'indices'``: an ``int`` global index.
            - Otherwise: the value of ``attribute`` at that middle index.

        Notes
        -----
        The “middle index” used by this code is **not** ``len(self)//2``.
        The implementation is:

        ``min(len(self), 1)``

        This returns:
        - 0 for an empty path (though this method is typically not called then),
        - 1 for any non-empty path of length >= 1.

        This behavior is inherited from the existing code and should be treated
        as an API contract for this package version.
        """
        if attribute == 'indices':
            return min(len(self), 1)
        return self._position(self.middle('indices'), attribute)

    def shooting(self, attribute):
        """
        Return an attribute at the shooting frame.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

        Returns
        -------
        object
            The value of ``attribute`` at global index ``self._shooting_index``.
        """
        return self._position(self._shooting_index, attribute)

    def all(self, attribute):
        """
        Return the full attribute array over the entire path.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

        Returns
        -------
        object
            The complete attribute as returned by ``self._get(attribute)``.
            In most cases this is a NumPy array of length ``len(self)``.
        """
        return self._get(attribute)

    def backward(self, attribute):
        """
        Return an attribute restricted to the “backward” range.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

        Returns
        -------
        object
            The attribute slice corresponding to ``self._range('backward')``.
        """
        return self._get(attribute, *self._range('backward'))

    def forward(self, attribute):
        """
        Return an attribute restricted to the “forward” range.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

        Returns
        -------
        object
            The attribute slice corresponding to ``self._range('forward')``.
        """
        return self._get(attribute, *self._range('forward'))

    def internal(self, attribute):
        """
        Return an attribute restricted to the “internal” range.

        Parameters
        ----------
        attribute : str
            Attribute name to retrieve.

        Returns
        -------
        object
            The attribute slice corresponding to ``self._range('internal')``.
        """
        return self._get(attribute, *self._range('internal'))

    def min(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is minimal.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location (e.g. ``'positions'``,
            ``'times'``, ``'states'``, ``'indices'``).
        source : str, default='values'
            Attribute whose numeric values are used to locate the minimum. This is
            commonly ``'values'`` (e.g. a collective variable) but may be any Path
            attribute understood by the underlying ``_extreme`` implementation.

        Returns
        -------
        object
            The value of ``attribute`` evaluated at the frame index (within the
            internal segment) where ``source`` attains its minimum.

        Notes
        -----
        - The minimum is computed over the **internal** range only.
        - This method does **not** return the minimum value of ``source``; it returns
          ``attribute`` at the argmin of ``source``.
        - Delegates to:
          ``self._extreme(attribute, np.argmin, 'internal', source)``.
        """
        return self._extreme(attribute, np.argmin, 'internal', source)

    def max(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is maximal.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location.
        source : str, default='values'
            Attribute used to locate the maximum.

        Returns
        -------
        object
            The value of ``attribute`` at the argmax location of ``source`` within
            the internal segment.

        Notes
        -----
        Delegates to:
        ``self._extreme(attribute, np.argmax, 'internal', source)``.
        """
        return self._extreme(attribute, np.argmax, 'internal', source)

    def min_backward(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is minimal in backward range.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location.
        source : str, default='values'
            Attribute used to locate the minimum.

        Returns
        -------
        object
            The value of ``attribute`` at the argmin location of ``source`` within
            the backward segment.

        Notes
        -----
        Delegates to:
        ``self._extreme(attribute, np.argmin, 'backward', source)``.
        """
        return self._extreme(attribute, np.argmin, 'backward', source)

    def max_backward(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is maximal in backward range.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location.
        source : str, default='values'
            Attribute used to locate the maximum.

        Returns
        -------
        object
            The value of ``attribute`` at the argmax location of ``source`` within
            the backward segment.
        """
        return self._extreme(attribute, np.argmax, 'backward', source)

    def min_forward(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is minimal in forward range.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location.
        source : str, default='values'
            Attribute used to locate the minimum.

        Returns
        -------
        object
            The value of ``attribute`` at the argmin location of ``source`` within
            the forward segment.
        """
        return self._extreme(attribute, np.argmin, 'forward', source)

    def max_forward(self, attribute, source='values'):
        """
        Return ``attribute`` at the location where ``source`` is maximal in forward range.

        Parameters
        ----------
        attribute : str
            Attribute to return at the extremum location.
        source : str, default='values'
            Attribute used to locate the maximum.

        Returns
        -------
        object
            The value of ``attribute`` at the argmax location of ``source`` within
            the forward segment.
        """
        return self._extreme(attribute, np.argmax, 'forward', source)
