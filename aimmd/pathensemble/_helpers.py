"""
aimmd.pathensemble._helpers
==========================

Low-level helper methods for :class:`~aimmd.pathensemble.PathEnsemble`.

This module contains small, internal utilities that are shared across multiple
mixins:

- initialization from flexible inputs (paths, ensembles, filenames, iterables),
- internal attribute extraction from the stored paths,
- conversion of global frame indices into (path_index, local_index).

The methods here are intentionally small and are not meant to be the primary
user-facing API; instead, they underpin higher-level operations in
:mod:`aimmd.pathensemble._methods`, :mod:`aimmd.pathensemble._project`, and
:mod:`aimmd.pathensemble._reweight`.
"""

# external
import numpy as np
from abc import ABC
from tqdm import tqdm

# aimmd imports
from .utils import get_paths
from ..core.utils import get_local_index


class PathEnsembleHelpers(ABC):
    def _init(self, *paths, find_shooting_indices=False, pipeline=()):
        """
        Initialize a path ensemble from flexible path-like inputs.

        Parameters
        ----------
        *paths
            One or more of:

            - :class:`~aimmd.path.Path` instances,
            - other :class:`~aimmd.pathensemble.PathEnsemble` instances,
            - strings / :class:`pathlib.Path` pointing to path filenames,
            - iterables of any of the above.

            Inputs are normalized via :func:`~aimmd.pathensemble.utils.get_paths`
            and stored as a list in ``self._paths``.

        find_shooting_indices : bool, optional
            If ``True``, request path initialization to determine the shooting
            index (by passing ``shooting_index='find'`` to the path constructor).
            This is useful if paths were created from files and the shooting
            point is not explicitly stored.

        pipeline : tuple, optional
            Optional compute pipeline forwarded to path initialization. The
            pipeline is stored on each created :class:`~aimmd.path.Path` (via
            ``Path(..., pipeline=...)``) and can be used later for on-demand
            computation.

        Notes
        -----
        This method is aliased as :meth:`aimmd.pathensemble.PathEnsemble.__init__`.
        """

        # process kwargs
        path_kwargs = {}
        if pipeline:
            path_kwargs['pipeline'] = tuple(pipeline)
        if find_shooting_indices:
            path_kwargs['shooting_index'] = 'find'

        # get it
        self._paths = get_paths(paths, initialize=True, **path_kwargs)

    def _get(self, attribute, where='internal', verbose=False):
        """
        Extract and concatenate a per-path attribute over a path range.

        This is the backend used by :meth:`PathEnsembleMagic.__getattr__` to
        expose "vectorized" access to path attributes at the ensemble level.

        Parameters
        ----------
        attribute : str
            Name of the attribute to extract from each path via
            ``path._get(attribute, start, stop)``.
        where : str, optional
            Range selector forwarded to :meth:`aimmd.path.Path._range`. Typical
            values are ``'internal'``, ``'all'``, ``'forward'``, ``'backward'``.
        verbose : bool, optional
            If ``True``, show a progress bar.

        Returns
        -------
        list
            A list with one element per path, each element being the result of
            ``path._get(attribute, start, stop)`` for the chosen range.
        """
        return [path._get(attribute, *path._range(where))
                for path in tqdm(self._paths, total=len(self),
                                 position=0, disable=not verbose)]

    def _get_local_index(self, i, clip=False):
        """
        Convert a global frame index into a (path_index, local_index) pair.

        Parameters
        ----------
        i : int
            Global index in the *concatenated* ensemble view of frames.
        clip : bool, optional
            Forwarded to :func:`aimmd.core.utils.get_local_index`. If ``True``,
            out-of-range indices are clipped to the valid interval instead of
            raising.

        Returns
        -------
        tuple
            ``(k, j)`` where ``k`` is the path index in ``self._paths`` and ``j``
            is the local index within that path.
        """
        return get_local_index(i, self.offsets, clip=clip)
