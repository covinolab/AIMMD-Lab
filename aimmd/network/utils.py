"""
aimmd.network.utils
==================

Utility helpers for AIMMD network components.

This module provides:

- :class:`PlaceholderNetwork`: a minimal `torch.nn.Module` used as a safe default
  when no trained network has been configured.
- :data:`placeholder`: a module-level instance of :class:`PlaceholderNetwork`
  with an additional ``__source__`` attribute used for provenance/logging.
- :func:`extract_indices_and_series`: flatten per-path frame indices (and optional
  per-frame series) into contiguous NumPy arrays, while constructing boolean masks
  that label the backward and forward portions of each path relative to its
  shooting point.

Design constraints
------------------
- The placeholder network is intentionally **stateless**: it serializes to an
  empty state dict and ignores state loading.
- `extract_indices_and_series` is written against a *Path-like* interface and
  only relies on the attributes/methods it explicitly touches.

"""

# external
import numpy as np
import torch
from torch import nn

# aimmd
from .._config import print


class PlaceholderNetwork(nn.Module):
    """Stateless default network used as a placeholder.

    This class exists so that components expecting a `torch.nn.Module` can run
    even when the user has not provided a real model.

    Behavior
    --------
    - Forward pass returns only the first channel/feature: ``x[:, :1]``.
    - The model is **stateless** with respect to serialization:
      :meth:`state_dict` returns ``{}`` and :meth:`load_state_dict` does nothing.

    Notes
    -----
    - A dummy parameter is registered to ensure the module has a parameter tensor
      that anchors device/dtype when code expects parameters to exist (e.g. for
      `.to(device)` semantics).

    """

    def __init__(self):
        super().__init__()
        # dummy parameter: anchors device/dtype, not used for computation
        self._dummy = torch.nn.Parameter(torch.empty(0))

    def forward(self, x):
        """Compute placeholder output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Expected shape is typically ``(n_samples, n_features)``
            so that slicing ``x[:, :1]`` yields a tensor of shape ``(n_samples, 1)``.
            The method does not validate rank explicitly; it relies on PyTorch
            slicing semantics.

        Returns
        -------
        y : torch.Tensor
            A view of the first column of `x`, i.e. ``x[:, :1]``. For a 2D input
            this has shape ``(n_samples, 1)`` and shares storage with `x`.

        """
        # keep only the first output channel
        return x[:, :1]

    def state_dict(self, *args, **kwargs):
        """Return an empty state dictionary.

        Parameters
        ----------
        *args, **kwargs
            Accepted for API compatibility with `torch.nn.Module.state_dict`
            but ignored.

        Returns
        -------
        state : dict
            Always an empty dict ``{}``.

        """
        # placeholder is intentionally stateless
        return {}

    def load_state_dict(self, state_dict, strict=True):
        """Load a state dictionary (no-op).

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            Ignored.
        strict : bool, optional
            Ignored. Present for compatibility with PyTorch's API.

        Returns
        -------
        None

        """
        # accept any "state" without effect
        pass


# module-level instance used as the default "network"
placeholder = PlaceholderNetwork()
# provenance marker used by other components for reproducible import strings
placeholder.__source__ = 'from aimmd.network import placeholder as network'

# helpers
def extract_indices_and_series(paths, key, *names):
    """Extract and concatenate indices and per-frame series across paths.

    The function iterates over a selected subset of `paths` and produces flat
    arrays suitable for downstream vectorized analysis/plotting. For each path,
    it:

    1) obtains the internal frame indices via ``path.internal('indices')``;
    2) builds two boolean masks (backward/forward) in the local index space:
       - backward: frames from path start through the shooting frame (inclusive);
       - forward: frames from the shooting frame through the end (inclusive),
         but only if the path actually transitions (see below);
    3) loads each requested series `name` via ``path.get`` over the global index
       window defined by the first/last internal index;
    4) appends indices, masks, and series to global outputs.

    A path is skipped entirely if loading any requested series fails or if a
    loaded series does not match the number of internal indices.

    Path transition rule
    --------------------
    Forward frames are only marked if ``path.type[2] != path.type[1]``. This is
    interpreted as “the path ends in a different state than it started” (i.e.
    it is “going somewhere”); otherwise the forward mask for that path remains
    all False.

    Parameters
    ----------
    paths : Sequence[Path-like]
        Sequence of Path-like objects.

        Each `path` is expected to provide the following attributes/methods
        (as used by the current implementation):

        - `path.type` :
            Indexable object where elements 1 and 2 encode start/end states.
        - `path.internal('indices')` -> array-like[int] :
            Internal frame indices used as the reference axis for series alignment.
        - `path.shooting_index` : int
            Global frame index of the shooting point.
        - `path.indices` : array-like[int]
            Path-global indices; `path.indices[0]` is used as the first global index.
        - `path.get(name, start=int, stop=int, raise_if_missing=bool)` :
            Returns a per-frame series aligned to the global indices in
            ``[start, stop)``. The return value must have length equal to
            ``len(path.internal('indices'))``.

    key : None or array-like
        Path selector.

        - If `None`, all paths are processed in their natural order (equivalent
          to selecting ``range(len(paths))``).
        - Otherwise, `key` is used to index ``np.arange(len(paths))`` and then
          flattened. This supports boolean masks, integer index arrays, slices,
          etc., as long as NumPy advanced indexing accepts it.

    *names : str
        Names of additional per-frame series to extract for each path. For each
        `name`, the function calls:

        ``path.get(name, start=path_indices[0], stop=path_indices[-1] + 1, raise_if_missing=True)``

        and requires that the returned object has length equal to the number of
        internal indices.

    Returns
    -------
    indices : numpy.ndarray, dtype=int
        Concatenated internal indices from all successfully processed paths.

    back_mask : numpy.ndarray, dtype=bool
        Boolean mask aligned with `indices`. True for frames belonging to the
        backward segment (start through shooting, inclusive).

    forw_mask : numpy.ndarray, dtype=bool
        Boolean mask aligned with `indices`. True for frames belonging to the
        forward segment (shooting through end) **only** when the path transitions
        according to ``path.type[2] != path.type[1]``; otherwise False.

    *series : numpy.ndarray
        For each `name` in `names`, a NumPy array concatenating the corresponding
        per-frame series across successfully processed paths, aligned with `indices`.
        The dtype is inferred by `np.array(...)` from the appended values.

    n_selected : int
        Number of selected paths after applying `key` (i.e. ``len(key)`` after
        normalization). This counts *selected* candidates, not how many were
        successfully processed (some may be skipped due to I/O/validation errors).

    Notes
    -----
    - The function is intentionally tolerant to missing/failed series loads: it
      simply skips problematic paths.

    """
    # normalize path selection into an iterable of integer indices
    if key is None:
        key = range(len(paths))
    else:
        key = np.arange(len(paths))[key].flatten()

    # accumulators (extended per-path; converted to arrays at the end)
    indices_out = []
    series_out = [[] for name in names]
    back_out = []
    forw_out = []

    # iterate over selected paths
    for k in key:
        path = paths[k]

        # path metadata and internal reference indices
        path_type = path.type
        path_indices = path.internal('indices')

        # local masks for this path (aligned with path_indices)
        path_back = np.zeros(len(path_indices), dtype=bool)
        path_forw = np.zeros(len(path_indices), dtype=bool)

        # compute local shooting index relative to the first global index
        si = path.shooting_index - path_indices[0]
        
        # backward: include shooting point
        path_back[:si + 1] = True

        # forward: only if path transitions to a different end state
        if path_type[2] != path_type[1]:
            # only if the path is actually going somewhere...
            path_forw[si:] = True

        # load requested series; if any fail, skip this path
        path_series = []
        try:  # avoid i/o problems: check wether you can load everything
            for name in names:
                series = path.get(
                    name,
                    start=path_indices[0],
                    stop=path_indices[-1] + 1,
                    raise_if_missing=True,
                )
                # ensure strict alignment to internal indices
                assert len(series) == len(path_indices)
                path_series.append(series)
        except Exception as exception:
            # print(exception)
            continue

        # commit indices + masks
        indices_out.extend(path_indices)
        back_out.extend(path_back)
        forw_out.extend(path_forw)

        # commit each series in the same order as `names`
        for i, series in enumerate(path_series):
            series_out[i].extend(series)

    # finalize arrays (dtype enforced for indices/masks; inferred for series)
    return (
        np.array(indices_out, dtype=int),
        np.array(back_out, dtype=bool),
        np.array(forw_out, dtype=bool),
        *[np.array(series) for series in series_out],
        len(key),
    )
