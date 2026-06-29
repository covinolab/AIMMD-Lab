"""
aimmd.pathensemble._methods
==========================

Convenience methods for :class:`aimmd.pathensemble.PathEnsemble`.

This mixin provides list-like mutators and common analyses operating across
all paths in the ensemble:

- appending/removing paths,
- filtering by *type* patterns (transition/excursion/internal),
- extracting frames across the ensemble,
- copying/splitting,
- merging/concatenating all paths into a single :class:`aimmd.path.Path`,
- sampling frames according to path ensemble weights from selected regions,
- producing aggregated shooting results over sweeps.

Terminology
-----------
`Path.type` is used extensively here. In AIMMD it appears to be a 4-character
string describing the relationship between:
- the path start state,
- the path "middle" states (those inside the path excluded the margins) 
- the path end state,
- and path shooting point's state,
where '.' indicates an unspecified state. See `PathEnsembleProperties.type`
for details.

Filtering uses :func:`aimmd.pathensemble.utils.match_patterns`.
"""

# bulk convenience methods over the stored path list

# external
import numpy as np
from abc import ABC
from tqdm import tqdm
from numbers import Integral

# aimmd imports
from .utils import match_patterns
from ..path import Path
from ..path.utils import read_sweep_frame
from ._helpers import PathEnsembleHelpers
from ..core.utils import merge_ranges


class PathEnsembleMethods(ABC):
    """
    Mixin implementing common list-like and analysis methods for PathEnsemble.
    """

    def append(self, path):
        """
        Append a single Path.

        Parameters
        ----------
        path : aimmd.path.Path
            Path instance to append.
        """
        self._paths.append(path)

    def extend(self, paths):
        """
        Extend the ensemble with multiple paths.

        Parameters
        ----------
        paths : iterable of aimmd.path.Path
            Paths to append.
        """
        self._paths.extend(paths)

    def are_complete(self, states="ARB"):
        """
        Vectorized wrapper around `Path.is_complete`.

        Parameters
        ----------
        states : str, default="ARB"
            State alphabet / specification forwarded to Path.

        Returns
        -------
        numpy.ndarray, dtype=bool
            Boolean array of shape (n_paths,).
        """
        return np.array([path.is_complete(states) for path in self])

    def are_transitions(self, states="ARB"):
        """Vectorized wrapper around `Path.is_transition`."""
        return np.array([path.is_transition(states) for path in self])

    def are_excursions(self, states="ARB"):
        """Vectorized wrapper around `Path.is_excursion`."""
        return np.array([path.is_excursion(states) for path in self])

    def are_internal(self, states="ARB"):
        """Vectorized wrapper around `Path.is_internal`."""
        return np.array([path.is_internal(states) for path in self])

    def extract(self, *types):
        """
        Extract a sub-ensemble matching one or more type patterns.

        Parameters
        ----------
        *types : str
            Patterns matched against `Path.type` using `match_patterns`.
            If no types are provided, returns an empty `PathEnsemble()`.

        Returns
        -------
        PathEnsemble
            Selected paths as a new ensemble.
        """
        if not len(types):
            from . import PathEnsemble
            return PathEnsemble()
        return self[self.types(*types)]

    def subsample(self, caps, states='ARB', rng=None):
        """Return a randomly down-sampled sub-ensemble with per-category caps.

        This bounds the cost of an expensive per-round value pass (and the bins /
        reweighting that consume it) by keeping only a capped, uniformly drawn
        subset of paths in each category, while leaving the *full* ensemble for
        training. Selection is uniform within each category so the reweighting
        per-category renormalisation stays consistent; in-state-only paths carry
        zero reweight, so dropping them never biases the rate estimate.

        Categories use the 4-character ``Path.type`` (initial, middle, final,
        shooting state). With ``states='ARB'`` (a=A, r=R, b=B):

        - *shot* excursions  : middle == r and shooting == r,
        - *free* excursions  : middle == r and shooting != r,
        - *in-state* paths   : middle != r (grouped by the initial state).

        Parameters
        ----------
        caps : dict or None
            Validated ``Params.subsample_caps`` for this system. Recognised keys:

            - ``'shot'`` / ``'free'`` : max PATHS kept *per direction-type*
              (``oXdX``, i.e. each of AA/AB/BA/BB capped independently),
            - ``'in_state'`` : max FRAMES kept per state.

            A missing key leaves that category uncapped. ``None``/empty returns
            ``self`` unchanged.
        states : str, default 'ARB'
            Three-character state string ``(A, R, B)``.
        rng : numpy.random.Generator or None
            Source of randomness. A fresh default generator is used if ``None``.

        Returns
        -------
        PathEnsemble
            A new ensemble (a real slice of ``self``). Returns ``self`` if no cap
            applies or the ensemble is empty.
        """
        if not caps:
            return self
        n = len(self._paths)
        if n == 0:
            return self
        if rng is None:
            rng = np.random.default_rng()

        a, r, b = states[0], states[1], states[2]
        codes = self.types().view('U1').reshape(n, 4)   # init, mid, final, shoot
        init, mid, final, shoot = codes[:, 0], codes[:, 1], codes[:, 2], codes[:, 3]
        n_frames = self.n_frames

        keep = np.ones(n, dtype=bool)

        def cap_paths(mask, cap):
            idx = np.flatnonzero(mask)
            if cap is not None and len(idx) > cap:
                drop = rng.choice(idx, size=len(idx) - cap, replace=False)
                keep[drop] = False

        excursion = (mid == r)
        is_shot = excursion & (shoot == r)
        is_free = excursion & (shoot != r)

        shot_cap = caps.get('shot')
        free_cap = caps.get('free')
        for origin in (a, b):
            for dest in (a, b):
                cap_paths(is_shot & (init == origin) & (final == dest), shot_cap)
                cap_paths(is_free & (init == origin) & (final == dest), free_cap)

        in_cap = caps.get('in_state')
        if in_cap is not None:
            in_state = ~excursion & ((init == a) | (init == b))
            for state in (a, b):
                idx = np.flatnonzero(in_state & (init == state))
                if len(idx) == 0:
                    continue
                order = rng.permutation(idx)
                # keep paths whose cumulative frame total (before this path) is
                # still below the budget -> always keeps >= 1 path per state.
                prefix = np.cumsum(n_frames[order]) - n_frames[order]
                keep[order[prefix >= in_cap]] = False

        return self[keep]

    def pop(self, i=None):
        """
        Remove and return a path (like list.pop).

        Parameters
        ----------
        i : int or None, default=None
            If None, pop the last element.

        Returns
        -------
        aimmd.path.Path
        """
        return self._paths.pop(i if i is not None else -1)

    def remove(self, path):
        """Remove a specific path (like list.remove)."""
        self._paths.remove(path)

    def index(self, path):
        """Return the index of a specific path (like list.index)."""
        return self._paths.index(path)

    def frame(self, i):
        """
        Return a single frame addressed by a global index.

        Parameters
        ----------
        i : int
            Index into the concatenation of all path frames (using `n_frames`).

        Returns
        -------
        Any
            The underlying representation of a frame as returned by `Path.__getitem__`.

        Notes
        -----
        Negative indices count from the end of the concatenated frame sequence.
        """
        if i < 0:
            i += self.n_frames
        k, i = self._get_local_index(i)
        return self.paths[k][i]

    def types(self, *patterns):
        """
        Return or filter the array of path type strings.

        Parameters
        ----------
        *patterns : str
            If provided, return a boolean mask selecting types that match any of
            these patterns (see :func:`match_patterns`).

        Returns
        -------
        numpy.ndarray
            If no patterns: array of dtype '<U...' with `Path.type` per path.
            If patterns: boolean mask of shape (n_paths,).
        """
        types = np.array([path.type for path in self._paths])
        if not len(patterns):
            return types
        return match_patterns(types, *patterns)

    def to_memory(self):
        """
        Force all paths to load their data into memory.

        Delegates to `Path.to_memory`.
        """
        for path in self._paths:
            path.to_memory()

    def from_files(self):
        """
        Force all paths to read their data from files (drop in-memory state).

        Delegates to `Path.from_files`.
        """
        for path in self._paths:
            path.from_files()

    def shooting_results(self, states="ARB", sweep_size=0):
        """
        Aggregate shooting results over sweeps.

        Parameters
        ----------
        states : str, default="ARB"
            State alphabet forwarded to `Path.shooting_result`.
        sweep_size : int, default=0
            Size of the sweep block. If <= 0, defaults to `len(self)`.

        Returns
        -------
        numpy.ndarray, shape (sweep_size, 2)
            For each sweep point, sums the (n_to_A, n_to_B)-like counters returned
            by each path's `shooting_result(states)`.

        Notes
        -----
        Each path is assigned to the sweep point (validation frame) it was shot
        from: the frame index tagged on the trajectory by
        :func:`aimmd.path.utils.write_sweep_frame`, if present. Shots written by
        an older code path carry no tag and fall back to positional assignment
        `i % sweep_size` -- which is exactly the frame the old strictly-sequential
        sweep shot, so legacy data is attributed correctly without modification.
        """
        if sweep_size <= 0:
            sweep_size = len(self)

        results = np.zeros((sweep_size, 2))
        for i, path in enumerate(self._paths):
            frame = read_sweep_frame(path.fname) if path.fname else None
            index = i if frame is None else frame
            results[index % sweep_size] += path.shooting_result(states)
        return results

    def copy(self):
        """
        Deep-ish copy of the ensemble: copies each Path.

        Returns
        -------
        PathEnsemble
            New ensemble with `path.copy()` for each stored path.
        """
        copied_paths = [path.copy() for path in self._paths]
        from . import PathEnsemble

        result = PathEnsemble()
        result._paths = copied_paths
        return result

    def split(self, verbose=False):
        """
        Split each path and flatten the resulting pieces into one ensemble.

        Parameters
        ----------
        verbose : bool, default=False
            If True, show a progress bar.

        Returns
        -------
        PathEnsemble
            New ensemble where each input path has been replaced by the pieces
            produced by `Path.split()`.

        Notes
        -----
        This expects `Path.split()` to return a `PathEnsemble`-like object with
        a `_paths` attribute; this matches the current AIMMD design.
        """
        split_paths = []
        for path in tqdm(self._paths, disable=not verbose, position=0):
            split_paths.extend(path.split()._paths)
        from . import PathEnsemble

        result = PathEnsemble()
        result._paths = split_paths
        return result

    def in_memory(self, attribute="reader"):
        """
        Query whether each path currently holds a given attribute in memory.

        Parameters
        ----------
        attribute : str, default="reader"
            Passed to `Path.in_memory(attribute)`.

        Returns
        -------
        numpy.ndarray, dtype=bool
        """
        return np.array([path.in_memory(attribute) for path in self._paths])

    def join(self):
        """
        Merge all paths into a single `Path` composed of concatenated file
        ranges.
        
        Returns
        -------
        aimmd.path.Path
            A new Path with `_fnames`, `_first`, `_last` representing the
            concatenation of all trajectory segments across all paths.
            In-memory data will not be copied.

        Warnings
        --------
        - Only the first path's `shooting_index` is propagated (if any paths exist).
        - Exclusion logic is simplified: if not all paths are accepted, the merged
          path uses `exclude_from = 0` (see code comment).
        """

        fnames = []
        first = []
        last = []
        for path in self._paths:
            fnames.extend(path._fnames)
            first.extend(path._first)
            last.extend(path._last)
        
        # simplify acceptance/exclusion propagation
        if not self.accepted.all():
            exclude_from = 0
        else:
            exclude_from = -1
        
        # build a Path instance without calling its initializer
        result = object.__new__(Path)
        result._fnames = fnames
        result._first = first
        result._last = last
        result._weight = 1.
        result._exclude_from = exclude_from
        if len(self):
            result._shooting_index = self._paths[0]._shooting_index
        else:
            result._shooting_index = 0
        return result
        

    def merge(self):
        """
        Merge all paths into a single `Path` composed of merged file ranges.

        Returns
        -------
        aimmd.path.Path
            A new Path with `_fnames`, `_first`, `_last` representing the union of
            all trajectory segments across all paths.

        Warnings
        --------
        - Only the first path's `shooting_index` is propagated (if any paths exist).
        - Backward segments are converted to forward ranges.
        - In-memory data is lost because a new Path object is built manually.
        - Exclusion logic is simplified: if not all paths are accepted, the merged
          path uses `exclude_from = 0` (see code comment).

        Notes
        -----
        This method uses :func:`aimmd.core.utils.merge_ranges` per filename to merge
        local index ranges into a minimal set of intervals.
        """
        ranges = {}
        for path in self._paths:
            for fname, first, last in zip(path._fnames, path._first, path._last):
                if fname not in ranges:
                    ranges[fname] = []
                if first <= last:
                    start = first
                    stop = last + 1
                else:
                    start = last
                    stop = first + 1
                ranges[fname].append((start, stop))

        fnames = []
        first = []
        last = []
        for fname, fname_ranges in ranges.items():
            start, stop = np.array(merge_ranges(fname_ranges)).T
            fnames.extend([fname] * len(start))
            first.extend(start)
            last.extend(stop - 1)  # they are all growing

        # simplify acceptance/exclusion propagation
        if not self.accepted.all():
            exclude_from = 0
        else:
            exclude_from = -1

        # build a Path instance without calling its initializer
        result = object.__new__(Path)
        result._fnames = fnames
        result._first = first
        result._last = last
        result._exclude_from = exclude_from
        if len(self):
            result._shooting_index = self._paths[0]._shooting_index
        else:
            result._shooting_index = 0
        return result

    def compute(self, *args, **kwargs):
        """
        Compute quantities on the merged path.

        This is a thin wrapper around:

        ``self.merge().compute(*args, **kwargs)``

        Notes
        -----
        The comment in the original code suggests borrowing documentation from
        `Path.compute`. For user-facing docs, consult `aimmd.path.Path.compute`.
        """
        return self.merge().compute(*args, **kwargs)

    def sample(self, n_samples, weights=None,
               source='values', vmin=None, vmax=None):
        """
        Sample individual frames from the ensemble and return them as a Path,
        according to the path weights.

        Parameters
        ----------
        n_samples : int
            Number of frames to sample (with replacement).
        weights : array_like, optional
            Per-path weights. If not provided, path ensemble weights
            are used (`self.weights`).
        source : str, default='values'
            Considered for getting frames between vmin and vmax
        vmin : float, default=None
            If specified, do not get frames with value below vmin
        vmax : float, default=None
            If specified, do not get frames with value above vmax
        
        Returns
        -------
        aimmd.path.Path
            A Path whose segments each correspond to a single selected frame
            (each segment uses the original filename and a one-frame range).

        Notes
        -----
        This routine constructs a new Path manually (no initialization). The
        resulting Path uses per-frame ranges: `_first[i] == _last[i]`.
        """
        from ..path import Path

        result = Path()
        if not n_samples or not len(self):
            return result

        # process weights
        if isinstance(weights, Integral):
            weights = np.ones(len(self))  # all weights scaled the same
        elif weights is None:
            weights = self.weights

        # build a "flat" index over all eligible frames across all paths
        paths = []
        indices = []
        p = []  # selection probabilities
        for path, weight in zip(self._paths, weights):
            # restrict between vmin and vmax
            this = path.internal('indices')
            if vmin is not None or vmax is not None:
                values = path.internal(source)
                mask = np.ones(len(values), dtype=bool)
                if vmin is not None:
                    mask &= values >= vmin
                if vmax is not None:
                    mask &= values < vmax
                this = this[mask]
            indices.extend(this)
            paths.extend([path] * len(this))
            p.extend([weight] * len(this))
        
        if not indices:
            return result

        # process selection probabilities
        p = np.array(p)
        if not (norm := p.sum()):
            raise RuntimeError('cannot sample: each frame has nonzero weight')
        p /= norm

        fnames = []
        first = []
        last = []
        for i in np.random.choice(len(indices), n_samples, p=p):
            path, i = paths[i], indices[i]
            k, i = path._get_local_loc(i)
            fnames.append(path._fnames[k])
            first.append(i)
            last.append(i)

        result._fnames = fnames
        result._first = first
        result._last = last
        return result

    # expose the helper bulk getter as a public-ish method name
    get = PathEnsembleHelpers._get
