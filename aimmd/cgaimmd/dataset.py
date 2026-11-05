"""
aimmd.cgaimmd.dataset
=====================

The training data of one round of a C-GAIMMD fit: shooting points with their
two-way outcomes, and the frames of the path ensemble with their states and
density weights.

Every C-GAIMMD fit reads its data through :func:`extract_dataset`: the
flow-matching fit of PaiNN networks and the sequential fit of BioEmu2
networks. The rows are the descriptor rows of the run's descriptor series,
``{trajectory}.{series}.npy``, which for C-GAIMMD are the flat Angstrom
coordinates of the network's atoms, ``(n_frames, 3 M)``. The series is a
parameter: the default ``'descriptors'`` reads the caches of every run
written so far; a fit passes ``params.descriptors_series`` so that a run
with a series of its own (:mod:`aimmd.core.series`) is read from that series.

- :class:`TrainingDataset`: the shooting points and frames of a round. Every
  frame carries the trajectory it belongs to and its frame index along that
  trajectory, with the part files of one trajectory joined;
- :func:`extract_dataset`: builds it from a path ensemble (what AIMMD's
  trainer passes to ``fit``), reusing the rows of the previous round for
  files that did not change;
- :func:`path_weights`: AIMMD's density weight of every path, through its
  reweighting;
- :func:`resolve_exclusions`: the ``exclude_frames`` setting of a fit (a
  mapping or a JSON file) with absolute file names;
- :func:`time_averaged_labels`: committor labels averaged over a time window
  along each trajectory (an option of the fits, off by default).

>>> import numpy as np
>>> from aimmd.cgaimmd.dataset import time_averaged_labels
>>> labels = np.array([-2.0, 0.0, 2.0])
>>> np.round(time_averaged_labels(labels, [0, 0, 0], [0.0, 1.0, 2.0], 1.0), 4)
array([-0.802,  0.   ,  0.802])

Ported from ``aimmd.network.flow_fit`` of the flow branch
(``FlowDataset``, ``extract_flow_dataset``, ``path_weights`` and the private
``_resolve_exclusions``).
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..path.utils import get_cache_fname


DEFAULT_SERIES = 'descriptors'
"""The descriptor series read when a fit names none (the caches of every run so far)."""


# --------------------------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------------------------

@dataclass
class TrainingDataset:
    """Training data of one round: shooting points and weighted frames.

    Descriptor rows are kept as stored (dtype and bytes of the descriptor
    caches), so that graph-cache keys match those of the trainer's value
    pass; ``sp_x`` and ``fr_x`` are their ``(n, M, 3)`` views. State codes:
    1 = ``sorted_states[0]``, 2 = ``sorted_states[2]``, 0 = the reactive
    region.

    A trajectory is a set of files that appear together in one path of the
    ensemble: AIMMD stores a free trajectory in part files
    (``traj000001.part0001.xtc``, ``.part0002``, ...) and its trainer passes
    pieces of it that cross from one part into the next, so the part files of
    one free trajectory form one trajectory; a shot path and every other
    file that no path joins to another is a trajectory on its own.
    """

    sp_rows: np.ndarray
    """``(n_sp, 3 M)`` descriptor rows of the shooting frames."""
    sp_r_first: np.ndarray
    """``(n_sp,)`` float32 two-way outcomes ending in ``sorted_states[0]``."""
    sp_r_last: np.ndarray
    """``(n_sp,)`` float32 two-way outcomes ending in ``sorted_states[2]``."""
    sp_group: np.ndarray
    """``(n_sp,)`` int64 reactive-chain number of every shooting point (-1 outside ``chain{r}{k}``)."""
    sp_fname: list
    """File of every shooting point."""
    fr_rows: np.ndarray
    """``(n_fr, 3 M)`` descriptor rows of every frame with positive weight (strided per file)."""
    fr_state: np.ndarray
    """``(n_fr,)`` int8 state codes."""
    fr_w: np.ndarray
    """``(n_fr,)`` float64 density weights (sums over the paths holding a frame, times the stride)."""
    fr_group: np.ndarray
    """``(n_fr,)`` int64 chain number ``k`` of frames of reactive chains ``chain{r}{k}``, else -1."""
    fr_fname: np.ndarray
    """``(n_fr,)`` object array, the file of every frame."""
    fr_loc: np.ndarray
    """``(n_fr,)`` int64 file-local frame index of every frame."""
    fr_trajectory: np.ndarray = None
    """``(n_fr,)`` int64 trajectory of every frame: an index into :attr:`trajectories`."""
    fr_frame: np.ndarray = None
    """``(n_fr,)`` int64 frame index along the trajectory (part files joined, the frame a
    restarted part repeats counted once); the first frame of the trajectory that a path of the
    ensemble holds is 0. Times the frame interval it is the time along the trajectory."""
    trajectories: list = field(default_factory=list)
    """The files of every trajectory, in the order of the trajectory (tuples of file names)."""
    notes: dict = field(default_factory=dict)
    """Counts and settings of the extraction."""
    _file_cache: dict = field(default_factory=dict, repr=False, compare=False)
    """Reuse bookkeeping for the next extraction (frame counts, indices and offsets per file, never rows)."""

    def __post_init__(self):
        # a dataset built by hand without trajectories: every file is a trajectory of its own
        if self.fr_trajectory is None:
            names, inverse = np.unique(np.asarray(self.fr_fname, dtype=str), return_inverse=True)
            self.fr_trajectory = inverse.astype(np.int64).reshape(-1)
            if not self.trajectories:
                self.trajectories = [(str(name),) for name in names]
        if self.fr_frame is None:
            self.fr_frame = np.asarray(self.fr_loc, dtype=np.int64).copy()

    @property
    def sp_x(self):
        """``(n_sp, M, 3)`` view of :attr:`sp_rows`."""
        return self.sp_rows.reshape(len(self.sp_rows), -1, 3)

    @property
    def fr_x(self):
        """``(n_fr, M, 3)`` view of :attr:`fr_rows`."""
        return self.fr_rows.reshape(len(self.fr_rows), -1, 3)

    def summary(self):
        """Counts of shooting points, frames per state and outcomes, and the notes.

        Returns
        -------
        dict
            JSON-serializable summary (``n_sp``, ``n_frames``, ``frames_by_state``, ``sp_outcomes`` and the notes).
        """
        first, last = self.sp_r_first, self.sp_r_last
        return {'n_sp': int(len(self.sp_rows)), 'n_frames': int(len(self.fr_rows)),
                'frames_by_state': {k: int((self.fr_state == k).sum()) for k in (0, 1, 2)},
                'sp_outcomes': {'(2,0)': int(((first == 2) & (last == 0)).sum()),
                                '(1,1)': int(((first == 1) & (last == 1)).sum()),
                                '(0,2)': int(((first == 0) & (last == 2)).sum())},
                **self.notes}


# --------------------------------------------------------------------------------------------
# files, chains and trajectories
# --------------------------------------------------------------------------------------------

_CHAIN_FOLDER = re.compile(r'^chain(\D+)(\d+)$')


def _chain_folder(fname):
    """``(t, k)`` of a file whose own directory is the chain folder ``chain{t}{k}``, else None."""
    match = _CHAIN_FOLDER.match(os.path.basename(os.path.dirname(os.path.abspath(str(fname)))))
    return (match.group(1), int(match.group(2))) if match else None


def _file_key(fname):
    return os.path.normpath(os.path.abspath(str(fname)))


def is_initial_path(path):
    """Whether every file of `path` lives in an ``initial*`` folder (the exported initial paths).

    Parameters
    ----------
    path : aimmd.Path
        A path of the ensemble.

    Returns
    -------
    bool
    """
    return all(os.path.basename(os.path.dirname(str(f))).startswith('initial')
               for f in path.fnames)


class _Trajectories:
    """Joins files into trajectories, with the frame index of every file's frame 0.

    A union-find over files: :meth:`join` records that a path holds file `b`
    right after (or ahead of) file `a`, so that the frame index along the
    trajectory of local frame ``loc`` of `b` is ``shift(b) - shift(a)`` plus
    that of `a`'s local frame ``loc``. Constraints that contradict an earlier
    one are counted and ignored.
    """

    def __init__(self):
        self.parent = {}        # file -> parent file
        self.shift = {}         # file -> frame index of its local frame 0 minus its parent's
        self.first = {}         # file -> smallest local index a path holds
        self.conflicts = 0

    def add(self, fname, first_loc):
        if fname not in self.parent:
            self.parent[fname], self.shift[fname], self.first[fname] = fname, 0, int(first_loc)
        else:
            self.first[fname] = min(self.first[fname], int(first_loc))

    def find(self, fname):
        """``(root, shift of fname's frame 0 relative to the root's frame 0)``."""
        chain = []
        while self.parent[fname] != fname:
            chain.append(fname)
            fname = self.parent[fname]
        root, total = fname, 0
        for member in reversed(chain):                  # compress: point every member at the root
            total += self.shift[member]
            self.parent[member], self.shift[member] = root, total
        return root, (self.shift[chain[0]] if chain else 0)

    def join(self, a, b, delta):
        """Record ``shift(b) - shift(a) == delta``."""
        root_a, shift_a = self.find(a)
        root_b, shift_b = self.find(b)
        if root_a == root_b:
            if shift_b - shift_a != delta:
                self.conflicts += 1
            return
        self.parent[root_b], self.shift[root_b] = root_a, shift_a + delta - shift_b

    def add_path(self, parts):
        """Add the files of one path, ``[(fname, file-local indices), ...]`` in path order.

        Files of a path with several files are joined when every file's piece
        runs forward in time (or every piece backward: the path then runs
        through the trajectory backwards).
        """
        parts = [(fname, part) for fname, part in parts if len(part)]
        for fname, part in parts:
            self.add(fname, part.min())
        if len(parts) < 2:
            return
        steps = [np.diff(part) for _, part in parts]
        if all((step == 1).all() for step in steps):
            ordered = parts
        elif all((step == -1).all() for step in steps):
            ordered = [(fname, part[::-1]) for fname, part in reversed(parts)]
        else:
            return
        start = 0                                       # frame index of each piece's first frame
        first_fname, first_part = ordered[0]
        for fname, part in ordered:
            # frame index of local frame loc of fname: start + loc - part[0], relative to the
            # first piece, whose local frame first_part[0] has index 0
            self.join(first_fname, fname, start - int(part[0]) + int(first_part[0]))
            start += len(part)

    def resolve(self):
        """``{file: (root, frame index of its local frame 0)}``, every trajectory starting at 0."""
        found = {fname: self.find(fname) for fname in self.parent}
        start = {}
        for fname, (root, shift) in found.items():
            start[root] = min(start.get(root, shift + self.first[fname]), shift + self.first[fname])
        return {fname: (root, shift - start[root]) for fname, (root, shift) in found.items()}


# --------------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------------

def _width_hint():
    from .values import MIGRATE_HINT
    return MIGRATE_HINT


def _open_caches(fname, n_atoms, series):
    """Descriptor (memory-mapped) and state caches of a trajectory, with retries on a file being replaced."""
    for attempt in range(4):
        try:
            rows = np.load(get_cache_fname(fname, series), mmap_mode='r')
            states = np.load(get_cache_fname(fname, 'states'), mmap_mode='r')
            break
        except FileNotFoundError:
            raise
        except Exception:                                    # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(0.25)
    if rows.ndim != 2 or rows.shape[1] != 3 * n_atoms:
        raise ValueError(f'{fname}: the cache of the descriptor series {series!r} has shape '
                         f'{rows.shape}, expected width {3 * n_atoms} (3 x {n_atoms} atoms of the '
                         f'network); {_width_hint()}')
    return rows, states


def _check_series(series):
    from ..params.utils import check_descriptors_series
    return check_descriptors_series(series)


def extract_dataset(pathensemble, *, states, n_atoms, series=DEFAULT_SERIES, weights=None,
                    free_stride=5, shot_stride=1, shot_stride_min_frames=0, exclude_frames=None,
                    previous=None):
    """The training data of a path ensemble: shooting points and weighted frames.

    Parameters
    ----------
    pathensemble : aimmd.PathEnsemble
        Shot chains, free trajectories and margins (what the trainer passes).
    states : str
        ``params.sorted_states``.
    n_atoms : int
        Number ``M`` of the network's atoms (the descriptor width is ``3 M``).
    series : str, optional
        The descriptor series to read, ``{trajectory}.{series}.npy``: a fit
        passes ``params.descriptors_series``. The default ``'descriptors'``
        reads the caches of runs without a series of their own.
    weights : array-like, optional
        Per-path density weights aligned with `pathensemble` (:func:`path_weights`); None weighs every path 1.
    free_stride : int, optional
        Frame stride inside files outside chain folders (weights multiplied by it).
    shot_stride : int, optional
        Frame stride inside shooting-path files.
    shot_stride_min_frames : int, optional
        Apply `shot_stride` only to shot files longer than this (0: to all).
    exclude_frames : mapping, optional
        ``{absolute file name: set of local frame indices | 'all'}``: frames never used, also as
        shooting points (:func:`resolve_exclusions` builds it from a fit's setting).
    previous : TrainingDataset, optional
        The dataset of the previous round: files whose caches kept their frame counts are not
        read again (only when it was extracted from the same series).

    Returns
    -------
    TrainingDataset
        The shooting points and frames.

    Raises
    ------
    ValueError
        If `series` is not a descriptor series name, or if a cache of the
        series does not have width ``3 M`` (with the hint how to give the
        run a series of its own).

    Notes
    -----
    A frame's weight is the sum of the weights of the paths that contain it
    (de-duplicated per file through ``path.fnames`` and ``path.locs``);
    frames of weight 0 are dropped. A path is a shooting point if its single
    file lives in a chain folder ``chain{t}{k}``, its middle state is the
    reactive one, it has a shooting index and its shooting result sums to at
    least 2; its row is the descriptor row of the shooting frame. Groups
    (the unit of the held-out split) are the chain numbers ``k`` of the
    reactive chains ``chain{r}{k}``; frames of free trajectories, margins and
    internal-shooting chains (``chain{t}{k}`` with ``t`` an end state) have
    group -1 and are never held out. Frames past the end of a cache (a free
    part being written) are skipped and counted. Files whose caches are
    missing are skipped and counted. With two or more atoms an all-zero
    descriptor row (every atom at the origin: a hole a descriptor pass left,
    since AIMMD's caches are zero-filled) is no structure: those frames and
    shooting points are skipped and counted (``zero_rows``,
    ``zero_row_shooting_points``), and their files are read again in every
    round. With one atom the origin is a position like any other.

    Files that one path of the ensemble holds together form one trajectory
    (:class:`TrainingDataset`); the frame index along it is set by the order
    of the files in that path, so the frame a restarted part file repeats,
    which AIMMD's paths leave out, is counted once. Strides, exclusions and
    holes remove frames but never shift the frame index of the others.

    The dataset holds one copy of its rows: the reuse bookkeeping of
    `previous` (``_file_cache``) keeps only frame indices and offsets into
    the previous dataset's arrays, and the rows are gathered from there, so
    that an extraction holds at most the previous and the new rows.
    """
    t0 = time.time()
    series = _check_series(series)
    s0, r, s2 = states[0], states[1], states[2]
    paths = list(pathensemble)
    n_paths = len(paths)
    weights = np.ones(n_paths) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(weights) != n_paths:
        raise ValueError(f'{len(weights)} weights for {n_paths} paths')
    exclude = {_file_key(k): ('all' if v == 'all' else {int(i) for i in v})
               for k, v in (exclude_frames or {}).items()}
    files = {}
    shooting = []
    joined = _Trajectories()
    n_excluded_frames, weight_excluded, n_excluded_sp = 0, 0.0, 0

    def new_entry(fname):
        folder = _chain_folder(fname)
        return {'locs': [], 'w': [], 'shot': folder is not None,
                'group': folder[1] if folder is not None and folder[0] == r else -1, 'n_file': 0}

    for i, path in enumerate(paths):
        fnames = [str(f) for f in path.fnames]
        locs = np.asarray(path.locs, dtype=np.int64)
        if len(fnames) == 1:
            parts = [(fnames[0], locs)]
        else:
            offsets = np.concatenate([[0], np.cumsum(path.lengths)])
            parts = [(fnames[k], locs[offsets[k]:offsets[k + 1]]) for k in range(len(fnames))]
        joined.add_path(parts)
        w = float(weights[i])
        for fname, part in parts:
            entry = files.get(fname)
            if entry is None:
                entry = files[fname] = new_entry(fname)
            if len(part):
                entry['n_file'] = max(entry['n_file'], int(part.max()) + 1)
            if w > 0 and len(part):
                excluded = exclude.get(_file_key(fname))
                if excluded is not None:
                    drop = (np.ones(len(part), dtype=bool) if excluded == 'all'
                            else np.isin(part, sorted(excluded)))
                    n_excluded_frames += int(drop.sum())
                    weight_excluded += w * int(drop.sum())
                    part = part[~drop]
                entry['locs'].append(part)
                entry['w'].append(np.full(len(part), w))
        folder = _chain_folder(fnames[0]) if len(fnames) == 1 else None
        if folder is None or path.shooting_index is None:
            continue
        try:
            middle = path.type[1]
        except Exception:                                   # noqa: BLE001
            continue
        if middle != r:
            continue
        result = np.asarray(path.shooting_result(states), dtype=np.float64)
        si = int(path.shooting_index)
        if result.sum() < 2 or not 0 <= si < len(locs):
            continue
        loc = int(locs[si])
        excluded = exclude.get(_file_key(fnames[0]))
        if excluded is not None and (excluded == 'all' or loc in excluded):
            n_excluded_sp += 1
            continue
        group = folder[1] if folder[0] == r else -1
        shooting.append((fnames[0], loc, float(result[0]), float(result[1]), group))

    by_file_sp = {}
    for fname, loc, first, last, group in sorted(shooting):
        by_file_sp.setdefault(fname, []).append((loc, first, last, group))
    along = joined.resolve()
    trajectory_ids = {}                 # root file -> trajectory id, in the order of the frames
    # rows are reused only from a dataset of the same series
    old_cache = (getattr(previous, '_file_cache', {}) if previous is not None
                 and previous.notes.get('series', DEFAULT_SERIES) == series else {})
    file_cache = {}
    fr = {'rows': [], 'state': [], 'w': [], 'group': [], 'fname': [], 'loc': [],
          'trajectory': [], 'frame': []}
    sp = {'rows': [], 'first': [], 'last': [], 'group': [], 'fname': []}
    fr_files = []                       # (fname, idx) of every file that contributed frames
    n_missing = n_beyond = n_reused = n_used = n_zero_rows = n_zero_sp = 0
    for fname in sorted(set(files) | set(by_file_sp)):
        entry = files.get(fname) or new_entry(fname)
        if entry['locs']:
            all_locs = np.concatenate(entry['locs'])
            idx, inverse = np.unique(all_locs, return_inverse=True)
            w = np.bincount(inverse, weights=np.concatenate(entry['w']), minlength=len(idx))
            keep = w > 0
            idx, w = idx[keep], w[keep]
        else:
            idx, w = np.zeros(0, dtype=np.int64), np.zeros(0)
        is_shot = entry['shot']
        stride = shot_stride if is_shot else free_stride
        if is_shot and shot_stride_min_frames and entry['n_file'] <= shot_stride_min_frames:
            stride = 1
        if stride > 1 and len(idx):
            keep = idx % stride == 0
            idx, w = idx[keep], w[keep] * stride
        sp_entries = by_file_sp.get(fname, [])
        sp_locs = np.asarray([e[0] for e in sp_entries], dtype=np.int64)
        if not len(idx) and not len(sp_locs):
            continue
        try:
            rows_all, states_all = _open_caches(fname, n_atoms, series)
        except FileNotFoundError:
            n_missing += 1
            continue
        counts = (int(rows_all.shape[0]), int(states_all.shape[0]))
        n_ok = min(counts)
        n_beyond += int((idx >= n_ok).sum()) + int((sp_locs >= n_ok).sum())
        ok = idx < n_ok
        idx, w = idx[ok], w[ok]
        sp_entries = [e for e in sp_entries if e[0] < n_ok]
        sp_locs = sp_locs[sp_locs < n_ok]
        if not len(idx) and not len(sp_locs):
            continue
        old = old_cache.get(fname)
        # a file with all-zero rows (holes of an interrupted descriptor pass) is read again in
        # every round, so that a hole filled since is used
        reuse = (old is not None and old['counts'] == counts and old['n_zero'] == 0
                 and np.isin(idx, old['fr_locs']).all()
                 and np.isin(sp_locs, np.union1d(old['fr_locs'], old['sp_locs'])).all())
        n_zero = 0
        if reuse:
            pos = old['fr_start'] + np.searchsorted(old['fr_locs'], idx)
            rows, codes = previous.fr_rows[pos], previous.fr_state[pos]

            def sp_row(loc, old=old):
                k = np.searchsorted(old['fr_locs'], loc)
                if k < len(old['fr_locs']) and old['fr_locs'][k] == loc:
                    return previous.fr_rows[old['fr_start'] + k]
                return previous.sp_rows[old['sp_pos'][int(np.flatnonzero(old['sp_locs']
                                                                          == loc)[0])]]
            n_reused += 1
        else:
            rows = np.array(rows_all[idx])
            labels = np.asarray(states_all[idx])
            codes = np.where(labels == s0, 1, np.where(labels == s2, 2, 0)).astype(np.int8)
            zero = ~rows.any(axis=1) if n_atoms > 1 else np.zeros(len(rows), dtype=bool)
            if zero.any():
                n_zero = int(zero.sum())
                rows, codes, idx, w = rows[~zero], codes[~zero], idx[~zero], w[~zero]

            def sp_row(loc, rows_all=rows_all):
                return np.array(rows_all[loc])
        del rows_all, states_all
        n_used += 1
        if len(idx):
            root, frame0 = along.get(fname, (fname, 0))
            trajectory = trajectory_ids.setdefault(root, len(trajectory_ids))
            fr['rows'].append(rows)
            fr['state'].append(codes)
            fr['w'].append(w)
            fr['group'].append(np.full(len(idx), entry['group'], dtype=np.int64))
            fr['fname'].append(np.full(len(idx), fname, dtype=object))
            fr['loc'].append(idx.astype(np.int64))
            fr['trajectory'].append(np.full(len(idx), trajectory, dtype=np.int64))
            fr['frame'].append(frame0 + idx.astype(np.int64))
            fr_files.append((fname, idx))
        sp_pos, sp_kept, n_zero_sp_file = [], [], 0
        for loc, first, last, group in sp_entries:
            row = sp_row(loc)
            if n_atoms > 1 and not row.any():                 # a hole: no structure
                n_zero_sp_file += 1
                continue
            sp_pos.append(len(sp['rows']))
            sp_kept.append(loc)
            sp['rows'].append(row)
            sp['first'].append(first)
            sp['last'].append(last)
            sp['group'].append(group)
            sp['fname'].append(fname)
        n_zero_rows += n_zero
        n_zero_sp += n_zero_sp_file
        file_cache[fname] = {'counts': counts, 'fr_locs': idx.astype(np.int64), 'fr_start': 0,
                             'sp_locs': np.asarray(sp_kept, dtype=np.int64),
                             'sp_pos': np.asarray(sp_pos, dtype=np.int64),
                             'n_zero': n_zero + n_zero_sp_file}

    width = 3 * int(n_atoms)
    # one copy of the frame rows: move the per-file blocks into one preallocated array
    n_fr = int(sum(len(b) for b in fr['rows']))
    dtype = np.result_type(*[b.dtype for b in fr['rows']]) if fr['rows'] else np.float32
    fr_rows = np.empty((n_fr, width), dtype=dtype)
    start = 0
    for k, (fname, idx) in enumerate(fr_files):
        n = len(idx)
        fr_rows[start:start + n] = fr['rows'][k]
        fr['rows'][k] = None                                  # freed as the array fills
        file_cache[fname]['fr_start'] = start
        start += n
    members = {}
    for fname, (root, frame0) in along.items():
        if root in trajectory_ids:
            members.setdefault(root, []).append((frame0 + joined.first[fname], fname))
    trajectories = [tuple(fname for _, fname in sorted(members.get(root, [(0, root)])))
                    for root in trajectory_ids]

    def joined_array(name, dtype):
        return np.concatenate(fr[name]) if fr[name] else np.zeros(0, dtype)

    data = TrainingDataset(
        sp_rows=np.stack(sp['rows']) if sp['rows'] else np.zeros((0, width), np.float32),
        sp_r_first=np.asarray(sp['first'], np.float32),
        sp_r_last=np.asarray(sp['last'], np.float32),
        sp_group=np.asarray(sp['group'], np.int64),
        sp_fname=list(sp['fname']),
        fr_rows=fr_rows,
        fr_state=joined_array('state', np.int8),
        fr_w=joined_array('w', np.float64),
        fr_group=joined_array('group', np.int64),
        fr_fname=joined_array('fname', object),
        fr_loc=joined_array('loc', np.int64),
        fr_trajectory=joined_array('trajectory', np.int64),
        fr_frame=joined_array('frame', np.int64),
        trajectories=trajectories,
        notes={'n_paths': n_paths, 'n_files': n_used, 'files_missing': n_missing,
               'frames_beyond_cache': n_beyond, 'reused_files': n_reused,
               'free_stride': int(free_stride), 'shot_stride': int(shot_stride),
               'shot_stride_min_frames': int(shot_stride_min_frames),
               'excluded_files': len(exclude), 'excluded_path_frames': n_excluded_frames,
               'excluded_density_weight': float(weight_excluded),
               'excluded_shooting_points': n_excluded_sp,
               'zero_rows': n_zero_rows, 'zero_row_shooting_points': n_zero_sp,
               'series': series, 'n_trajectories': len(trajectories),
               'trajectory_conflicts': joined.conflicts,
               'extract_seconds': round(time.time() - t0, 3)},
        _file_cache=file_cache)
    return data


def path_weights(pathensemble, states, reweight_parameters=None):
    """AIMMD's density weight of every path: the reweighting weights of both directions, summed.

    Parameters
    ----------
    pathensemble : aimmd.PathEnsemble
        The ensemble (values in the ``values`` caches).
    states : str
        ``params.sorted_states``.
    reweight_parameters : dict, optional
        Keyword arguments of ``PathEnsemble.reweight`` (the trainer's, with its ``sp_cutoff_*``).

    Returns
    -------
    numpy.ndarray
        ``reweight(states)[0] + reweight(states[::-1])[0]``; non-finite entries are 0.
    """
    parameters = dict(reweight_parameters or {})
    w1 = np.asarray(pathensemble.reweight(states, **parameters, source='values')[0],
                    dtype=np.float64)
    w2 = np.asarray(pathensemble.reweight(states[::-1], **parameters, source='values')[0],
                    dtype=np.float64)
    w = w1 + w2
    w[~np.isfinite(w)] = 0.0
    return w


def resolve_exclusions(spec, base, json_base):
    """The ``exclude_frames`` setting of a fit as ``{absolute file: set of frames | 'all'}``.

    Parameters
    ----------
    spec : mapping or str or os.PathLike or None
        ``{file name: list of file-local frame indices | 'all'}``, or a JSON
        file holding that mapping (on its own or under the key ``'exclude'``).
    base : str or os.PathLike
        The directory relative file names of the mapping refer to (the run directory).
    json_base : str or os.PathLike
        The directory a relative JSON file name refers to (the params file's directory).

    Returns
    -------
    dict or None
        ``{normalized absolute file name: set of int | 'all'}`` (the
        ``exclude_frames`` argument of :func:`extract_dataset`); None for
        ``spec=None``.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    """
    if spec is None:
        return None
    if isinstance(spec, (str, os.PathLike)):
        fname = os.fspath(spec)
        if not os.path.isabs(fname):
            fname = os.path.join(os.fspath(json_base), fname)
        with open(fname) as handle:
            spec = json.load(handle)
        spec = spec.get('exclude', spec)
    out = {}
    for name, value in spec.items():
        name = os.fspath(name)
        key = _file_key(name if os.path.isabs(name) else os.path.join(os.fspath(base), name))
        out[key] = 'all' if value == 'all' else {int(k) for k in value}
    return out


# --------------------------------------------------------------------------------------------
# labels averaged over time
# --------------------------------------------------------------------------------------------

_GATHER_ELEMENTS = 1 << 22
"""Largest number of (frame, neighbour) pairs gathered at once by :func:`time_averaged_labels`."""


def _windowed_logsumexp(values, lo, hi):
    """``log(sum(exp(values[lo[i]:hi[i]])))`` for every i (every window non-empty)."""
    out = np.empty(len(lo))
    width = int((hi - lo).max())
    rows = max(1, _GATHER_ELEMENTS // width)
    offsets = np.arange(width)
    for a in range(0, len(lo), rows):
        index = lo[a:a + rows, None] + offsets[None, :]
        inside = index < hi[a:a + rows, None]
        gathered = np.where(inside, values[np.minimum(index, len(values) - 1)], -np.inf)
        top = gathered.max(axis=1)
        out[a:a + rows] = top + np.log(np.exp(gathered - top[:, None]).sum(axis=1))
    return out


def time_averaged_labels(labels, trajectory, time, window, *, include=None):
    """Committor labels averaged over a time window along each trajectory.

    The label of every included frame becomes the logit of the mean
    committor probability ``sigmoid(label)`` of the included frames of the
    same trajectory whose time lies within ``+- window`` of its own (itself
    included). Frames that are not included keep their label and do not
    enter any mean. The mean is taken in log space,
    ``log(sum p) - log(sum (1 - p))``, so that labels of any size stay
    finite and nothing is clipped.

    This is the time-averaged labels option of the C-GAIMMD fits
    (``label_time_window_ps``, off by default): with a
    :class:`TrainingDataset` ``data``, ``trajectory=data.fr_trajectory``,
    ``time=data.fr_frame * frame_interval_ps`` and
    ``include=data.fr_state == 0`` (the reactive frames; in-state frames
    keep their own labels).

    Parameters
    ----------
    labels : array-like
        ``(n,)`` logit committors.
    trajectory : array-like
        ``(n,)`` trajectory of every frame (any comparable ids).
    time : array-like
        ``(n,)`` time of every frame along its trajectory, in the unit of `window`.
    window : float or None
        Half-width of the window; None returns the labels unchanged.
    include : array-like of bool, optional
        ``(n,)`` the frames that are averaged and enter the means; default
        every frame. Frames with a non-finite label never are.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` the averaged labels, of the dtype of `labels` (float64 for
        integer labels). For ``window=None`` a copy of `labels`.

    Raises
    ------
    ValueError
        If `window` is negative or not finite, the arrays are not of one
        length, or a time is not finite.
    """
    labels = np.asarray(labels)
    if window is None:
        return labels.copy()
    window = float(window)
    if not (math.isfinite(window) and window >= 0):
        raise ValueError(f'window must be a finite number >= 0 or None, got {window!r}')
    if labels.ndim != 1:
        raise ValueError(f'labels must be one-dimensional, got shape {labels.shape}')
    n = len(labels)
    trajectory = np.asarray(trajectory).reshape(-1)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    include = np.ones(n, dtype=bool) if include is None else np.asarray(include, bool).reshape(-1)
    for name, array in (('trajectory', trajectory), ('time', time), ('include', include)):
        if len(array) != n:
            raise ValueError(f'{name} has {len(array)} entries for {n} labels')
    use = include & np.isfinite(labels)
    out = labels.astype(labels.dtype if np.issubdtype(labels.dtype, np.floating) else np.float64)
    members = np.flatnonzero(use)
    if not len(members):
        return out
    if not np.isfinite(time[members]).all():
        raise ValueError('every averaged frame needs a finite time')
    members = members[np.lexsort((time[members], trajectory[members]))]
    owner, when = trajectory[members], time[members]
    logits = labels[members].astype(np.float64)
    lo = np.empty(len(members), np.int64)
    hi = np.empty(len(members), np.int64)
    starts = np.flatnonzero(np.concatenate([[True], owner[1:] != owner[:-1]]))
    for a, b in zip(starts, np.append(starts[1:], len(members))):
        times = when[a:b]
        lo[a:b] = a + np.searchsorted(times, times - window, side='left')
        hi[a:b] = a + np.searchsorted(times, times + window, side='right')
    log_p = -np.logaddexp(0.0, -logits)                 # log sigmoid(label)
    log_q = -np.logaddexp(0.0, logits)                  # log (1 - sigmoid(label))
    out[members] = _windowed_logsumexp(log_p, lo, hi) - _windowed_logsumexp(log_q, lo, hi)
    return out
