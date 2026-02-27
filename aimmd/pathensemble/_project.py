"""
aimmd.pathensemble._project
===========================

Histogram-based projection utilities for :class:`aimmd.pathensemble.PathEnsemble`.

This module defines :class:`PathEnsembleProject`, a mixin that implements an
ensemble-level *projection* operator:

- gather per-frame data from a set of paths,
- optionally transform it with a user function,
- accumulate weighted counts into an N-dimensional histogram (``np.histogramdd``),
  with either the PathEnsemble or the user-defined weights.

The implementation is optimized for large ensembles by processing data in
fixed-size batches. This avoids concatenating all frames into a single array.

What “project” means here
-------------------------
Given a per-frame data stream ``x`` (for example, CV values, descriptors,
or coordinates), ``project`` computes:

    H[b0, b1, ...] = Σ_frame w(frame) * 1[ x(frame) in bin(b0,b1,...) ]

where binning is defined by ``bins`` and weights are derived from:

- ensemble weights (``self.weights``), optionally overridden by the caller,
- multiplicity induced by selecting paths via ``key`` (counts of repeats),
- optional value-range filtering (``vmin`` / ``vmax``) applied per frame.

Data sources
------------
Per-frame input is obtained by calling the private Path extractor
``path._extract(k, source)`` for each trajectory segment ``k``.

Two common cases are supported:

- ``source != 'reader'``:
    Per-segment data are arrays and are concatenated before being passed
    through ``function``.

- ``source == 'reader'``:
    Per-segment data are timesteps/readers and are wrapped with
    :class:`aimmd.path.chainreader.ChainReader` so that ``function`` sees a
    single reader-like object for the current batch.

Portions of a path
------------------
The ``where`` argument restricts which frames of each path contribute:

- ``'all'``:
    Use the entire stored trajectory.

- ``'forward'`` and ``'backward'``:
    Use only the portion from the shooting point to an end state.
    The shooting frame is included. Concretely, on the segment containing the
    shooting point:
      - forward keeps indices ``[shooting_i, ...]``
      - backward keeps indices ``[..., shooting_i]`` (implemented as stop
        at ``shooting_i + 1`` to include it)
    Segments strictly “on the other side” of the shooting segment are skipped.

- other values (commonly ``'internal'`` in this code base):
    Use the portion excluding boundary frames that correspond to the end states,
    if the path changes state at the beginning/end. This is implemented via
    the ``start`` / ``stop`` corrections based on ``path.type``.

The exact semantics of state boundaries and shooting indices are delegated to
Path. This module only applies consistent slicing rules across segments.

Notes
-----
- The public API returns a histogram of raw counts (``density=False``).
- If after filtering there are no paths, the returned histogram is all zeros.
- Errors extracting per-segment data are silently skipped (``try/except``),
  because not all sources may be available for all paths/segments.
"""

# external
import numpy as np
from abc import ABC
from math import inf
from tqdm import tqdm
from collections.abc import Iterable

# aimmd imports
from .utils import project_batch
from ..path.chainreader import ChainReader


# project
class PathEnsembleProject(ABC):
    """
    Projection operator for a PathEnsemble.

    This mixin assumes the host class provides:
    - ``__len__`` and ``__getitem__`` compatible with the `key` selection below
    - ``self._paths`` : list[Path]
    - ``self.weights`` : array-like of per-path weights
    - ``self.accepted`` : boolean array-like marking accepted paths
    """
    
    def project(self, bins=[-inf, +inf],
                key=None, weights=None,
                function=lambda x:x, source='values',
                where='internal', values_source='values',
                vmin=None, vmax=None,
                batch_size=4096, verbose=False):
        """
        Project per-frame data from an ensemble into an N-dimensional histogram.

        Parameters
        ----------
        bins : array_like or list[array_like], default=[-inf, +inf]
            Bin edges for histogramming. If a single 1D iterable is provided,
            it is treated as one-dimensional bins. If a list of iterables is
            provided, each element defines the bin edges for one dimension.
        key : object, optional
            Selection applied as ``np.arange(len(self))[key]`` to choose which
            paths contribute. Repeated indices are allowed and are accounted
            for by multiplying weights by their multiplicity.
        weights : array_like, optional
            Per-path weights overriding ``self.weights[indices]``. If not
            provided, path ensemble weights are used (`self.weights[key]`).
        function : callable, default=lambda x: x
            Transformation applied to the raw per-frame data before binning.
            It must produce an array-like object with one row per frame.
        source : str, default='values'
            Name of the per-frame stream extracted from Path via
            ``path._extract(k, source)``. If ``source == 'reader'``, per-segment
            inputs are wrapped with :class:`ChainReader` before calling
            ``function``.
        where : str, default='internal'
            Portion of each path to include. The slicing rules are described in
            the module docstring. Typical values are 'all', 'internal',
            'forward', 'backward'.
        values_source : str, default='values'
            Stream used to compute the value-range mask when ``vmin``/``vmax``
            are provided. If it equals ``source``, the already-extracted data
            are reused; otherwise the mask is computed from an independent
            extraction.
        vmin, vmax : float, optional
            If provided, apply an inclusion mask per frame:

            - both provided: keep frames with ``vmin <= values < vmax``
            - only vmin:     keep frames with ``values >= vmin`` (vmax = +inf)
            - only vmax:     keep frames with ``values < vmax`` (vmin = -inf)

            The mask is applied to the weights (frames outside range get weight 0).
        batch_size : int, default=4096
            Maximum number of frames buffered before computing a partial
            histogram contribution. Larger values reduce overhead but increase
            memory usage.
        verbose : bool, default=False
            If True, show a tqdm progress bar. The progress bar tracks the
            number of frames processed (including frames skipped due to slicing).

        Returns
        -------
        numpy.ndarray
            N-dimensional histogram of weighted counts.

        Notes
        -----
        Path selection and weighting:
        - Paths are selected via `key`, then unique indices and their counts
          (multiplicity) are computed.
        - Only accepted paths are kept (``self.accepted``).
        - Paths with NaN weights or zero weights are dropped.
        - The final per-path weight is multiplied by its multiplicity.

        Missing data handling:
        - If ``path._extract(k, source)`` fails for a segment, that segment
          is skipped.
        """
        
        # process bins
        if isinstance(bins, Iterable):
            if not isinstance(bins[0], Iterable):
                bins = [bins]
        result = np.zeros([len(b) - 1 for b in bins])

        # get paths
        paths = np.arange(len(self))[key].flatten()
        
        # nothing
        if not len(paths):
            return result

        # get unique paths and weights
        indices, counts = np.unique(paths, return_counts=True)
        weights = weights or self.weights[indices]
        keepers = self.accepted[indices] & ~np.isnan(weights) & (weights != 0)
        indices = indices[keepers]
        counts = counts[keepers]
        weights = weights[keepers]
        weights *= counts
        
        # process vmin and vmax
        if vmin is None and vmax is None:
            pass
        elif vmin is None and vmax is not None:
            vmin = -inf
        elif vmax is None and vmin is not None:
            vmax = +inf
        
        # compute in batches 
        batch_input = []
        batch_weight = []
        current_size = 0
        lengths = np.array([len(path) for path in self._paths]).astype(int)
        progress = tqdm(total=lengths.sum(), disable=not verbose)
        for i, weight in zip(indices, weights):
            path = self._paths[i]
            states = path.type
            if where in ('forward', 'backward'):
                shooting_k, shooting_i = path._get_local_index(
                    path.shooting('indices'))
            n_files = path.n_files
            for k in range(n_files):
                if (where == 'forward' and shooting_k > k or
                    where == 'backward' and shooting_k < k):
                    if verbose:
                        progress.update(lengths[i])
                    continue
                try:
                    # data not present: hence cannot project
                    # that's because data weren't computed on the whole source
                    input_data = path._extract(k, source)
                except:
                    continue
                input_data_length = len(input_data)
                if k == 0 and where != 'all' and states[0] != states[1]:
                    start = 1
                else:
                    start = 0
                if (k == n_files - 1 and where != 'all' and
                    states[1] != states[2]):
                    stop = input_data_length - 1
                else:
                    stop = input_data_length
                if where == 'forward' and shooting_k == k:
                    start = max(start, shooting_i)
                if where == 'backward' and shooting_k == k:
                    stop = min(stop, shooting_i + 1)
                if start or stop < input_data_length:
                    input_data = input_data[start:stop]
                    if verbose:
                        progress.update(start + input_data_length - stop)
                if vmin is not None:
                    if source == values_source:
                        values = input_data
                    else:
                        values = path._extract(k, values_source)
                remaining = len(input_data)
                current = 0
                while remaining:
                    delta = min(batch_size - current_size, remaining)
                    batch_input.append(input_data[current:current + delta])
                    if vmin is not None:
                        if vmax == inf:
                            batch_weight.append(weight * (values >= vmin))
                        elif vmin == -inf:
                            batch_weight.append(weight * (values < vmax))
                        else:
                            batch_weight.append(
                                weight * (values >= vmin) * (values < vmax))
                    else:
                        batch_weight.append([weight] * delta)
                    current += delta
                    current_size += delta
                    remaining -= delta
                    if current_size >= batch_size:
                        result += project_batch(
                            bins, function, source, batch_input, batch_weight)
                        if verbose:
                            progress.update(current_size)
                        current_size = 0
                        batch_input = []
                        batch_weight = []
        
        # last computation and return
        if current_size:
            result += project_batch(
                bins, function, source, batch_input, batch_weight)
        progress.update(progress.total - progress.n)
        progress.close()
        return result
