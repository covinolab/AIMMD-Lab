"""
aimmd.pathensemble._helpers
==========================

Internal helper mixin for :class:`aimmd.pathensemble.PathEnsemble`.

This module contains the minimal initialization and internal bulk-get utilities
shared by several mixins.

Key responsibilities
--------------------
- Construct ``self._paths`` from heterogeneous inputs using
  :func:`aimmd.pathensemble.utils.get_paths`.
- Provide a small internal dispatcher for retrieving per-path computed data
  via each path's internal API.
- Provide the global-to-local index mapping used by operations that address the
  ensemble as if it were a single concatenated sequence of frames.

Storage
-------
`PathEnsemble` stores its content in:

- ``self._paths`` : list[aimmd.path.Path]
"""

# external
import numpy as np
from abc import ABC
from tqdm import tqdm

# aimmd imports
from .utils import get_paths
from ..core.utils import get_local_index


class PathEnsembleHelpers(ABC):
    """
    Helper mixin implementing initialization and internal bulk access.

    This mixin is not intended to be used standalone; it is composed into
    :class:`aimmd.pathensemble.PathEnsemble`.
    """

    def _init(self, *paths, find_shooting_indices=False, pipeline=()):
        """
        Initialize the ensemble by normalizing all inputs into `Path` objects.

        Parameters
        ----------
        *paths
            Any of the accepted inputs for :func:`aimmd.pathensemble.utils.get_paths`.
        find_shooting_indices : bool, default=False
            If True, pass ``shooting_index='find'`` into `Path` initialization.
        pipeline : tuple, default=()
            Optional pipeline forwarded to `Path` initialization.

        Notes
        -----
        - This method is aliased as `PathEnsemble.__init__`.
        - The initialization here is intentionally minimal: it only builds
          ``self._paths`` and leaves everything else to derived properties/methods.
        """

        # hash: normalize kwargs passed into Path(...) construction
        path_kwargs = {}
        if pipeline:
            path_kwargs["pipeline"] = tuple(pipeline)
        if find_shooting_indices:
            path_kwargs["shooting_index"] = "find"

        # hash: build the underlying list of Path objects
        self._paths = get_paths(paths, initialize=True, **path_kwargs)

    def _get(self, attribute, where="internal", verbose=False):
        """
        Internal bulk getter that delegates to each path.

        Parameters
        ----------
        attribute : str
            Name of the per-path attribute to fetch through the `Path` internal API.
            This is forwarded to ``path._get(attribute, *path._range(where))``.
        where : str, default="internal"
            Range spec forwarded to each path's ``_range(where)`` method. Typical
            values include (depending on `Path` implementation): "internal",
            "all", "forward", "backward", etc.
        verbose : bool, default=False
            If True, display a progress bar over the paths.

        Returns
        -------
        list
            One element per path, each being the corresponding path's `_get(...)`
            result.

        Notes
        -----
        This is an internal routine used by `PathEnsemble.__getattr__` to provide
        convenience "vectorized" attribute access. Many public properties return
        arrays instead; prefer them when you want stable NumPy types.
        """
        return [
            path._get(attribute, *path._range(where))
            for path in tqdm(self._paths, total=len(self), position=0, disable=not verbose)
        ]

    def _get_local_index(self, i, clip=False):
        """
        Map a global frame index into (path_index, local_index).

        Parameters
        ----------
        i : int
            Global index into the concatenation of all frames in the ensemble.
        clip : bool, default=False
            If True, out-of-range indices are clipped to valid bounds by
            :func:`aimmd.core.utils.get_local_index`.

        Returns
        -------
        tuple[int, int]
            ``(k, j)`` where `k` is the path index in ``self._paths`` and `j`
            is the local frame index within that path.

        See also
        --------
        :func:`aimmd.core.utils.get_local_index`
        :property:`aimmd.pathensemble.PathEnsembleProperties.offsets`
        """
        return get_local_index(i, self.offsets, clip=clip)
