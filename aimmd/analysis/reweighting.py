"""
aimmd.analysis.reweighting
==========================

Rates, equilibrium weights, populations and free energies of a finished (or
running) AIMMD run folder, computed offline with the trainer's own
reweighting.

Why this module exists
----------------------
The trainer reweights the whole path ensemble after every training round and
prints two rate estimates, but it saves neither the per-path weights nor any
per-frame weight: ``densities{states}.npy`` holds the reweighted density on
the committor bins only. A free-energy surface along a physical collective
variable (a dihedral angle, a contact fraction, a distance), the populations
of the states, or the rates at a fraction of the data all need the
reweighting again. :func:`reweight_run` repeats it exactly as the trainer does
it (the same :func:`aimmd.analysis.utils.compute_bins`, the same shooting-value
cutoffs from :func:`aimmd.worker.utils.reweight_parameters_with_bins`, the same
:meth:`aimmd.PathEnsemble.reweight` in both directions); it calls that code, it
does not copy it, so any fix to AIMMD's reweighting applies here as well.

What is computed
----------------
- **Rates.** ``k12 = 1 / sum_i(w1_i * L_i)`` and ``k21 = 1 / sum_i(w2_i * L_i)``,
  where ``w1`` and ``w2`` are the reweighting weights of the paths in the two
  directions (from ``sorted_states[0]`` to ``sorted_states[2]`` and back) and
  ``L_i`` the number of frames a path counts
  (:attr:`aimmd.PathEnsemble.n_frames`). The unit is one per frame interval
  of the trajectories (``1/dt``); divide by the frame interval to get a rate
  per unit of time. The intuition: the weighted ensemble stands in for one
  long equilibrium trajectory; ``sum_i w1_i L_i`` is the mean time that
  trajectory spends with state 1 as the last end state it visited before it
  reaches state 2 (the mean first-passage time), and the rate is its inverse.
- **Equilibrium weights.** Every counted frame of path ``i`` carries the
  weight ``w1_i + w2_i``: summed over frames, these weights describe how often
  each configuration occurs in equilibrium. Histograms of any per-frame
  quantity with these weights give populations and free energies.
- **Free energies.** ``F(x) = -ln p(x)`` in units of kT, with ``p`` the
  weighted histogram of a collective variable ``x`` (:func:`free_energy`).
- **Convergence.** The same reweighting on the first fraction of the data
  (``fractions=``): the shots registered first and the same share of every
  free-simulation stream, with the committor values held fixed (the final
  network, or the network given). This shows how far an estimate still moves
  with the amount of data; it is not a re-run of the training, which
  :meth:`aimmd.Worker.kinetics_convergence` does.

Assumptions and valid estimators
--------------------------------
The rates and weights are those of AIMMD's main reweighting. In outline: the
shots and the free-simulation excursions are ordered by the committor values
(logits) of their frames; for every committor level the method estimates the
probability that an excursion that leaves a state reaches that level, from
how many of the excursions that reached one level went on to the next (a
product over levels, as a survival curve is built); every excursion is then
weighted so that the weighted ensemble has these crossing probabilities, and
the rates follow from the weighted time spent with each state as the last one
visited. This rests on the following.

- **The dynamics in the reactive region are those of the system.** Every
  half of a shot and every free-simulation segment is a piece of unbiased
  dynamics from its first frame on. With a static bias inside the states
  (``record_bias``, below) the bias must vanish in the reactive region; the
  trainer checks this every round.
- **Progress along the committor has no memory.** The chance that an
  excursion at one level goes on to a higher one is taken to depend on the
  level only, not on how fast or from where it got there. This holds for
  diffusive dynamics saved at frame intervals that are long compared with the
  memory of the velocities and short compared with the time to move by about
  one logit. Dynamics that cross the barrier within a few frames (a light
  molecule in vacuum, little friction) violate it: the crossing probability,
  and with it the rate out of the state that is left, then comes out too low;
  frames that are far apart bias it the other way. When at least
  ``free_threshold`` (``params.reweight_parameters``, 20 by default)
  free-simulation excursions from a state reached the other state, the
  crossing probability out of that state comes from the free simulations
  alone, and the rate out of it is in effect their direct count, which does
  not rest on this assumption.
- **Every committor level between the states is crossed** by enough shots or
  free excursions.
- **Within each committor level, the shooting points are distributed as in
  equilibrium.** How many shooting points sit at each level is free (the
  reweighting corrects for it; this is why RFPS and C-GAIMMD may choose their
  levels), but among the configurations that share a committor value the
  shooting points must be distributed as in equilibrium. RFPS satisfies this
  by construction. For C-GAIMMD it is what the generator is trained toward
  and what the tools of :mod:`aimmd.cgaimmd.analysis` test (adherence,
  variety, structural match, shooting validation); where a generator departs
  from it, the weights, free energies and rates carry an error that the
  reweighting itself cannot see.
- **The numbers depend on the committor values used.** The levels are those
  of a network. Cached values (each frame scored by the network of the round
  that scored it) and recomputed values (every frame scored by one network)
  give different numbers, and another network gives different numbers again;
  :attr:`RunReweighting.values` and :attr:`RunReweighting.network` record
  which were used.

Exact path importance sampling and path-ensemble MBAR (the multistate Bennett
acceptance ratio applied to paths) need no memoryless progress along the
committor, but they need the probability with which each shooting point was
proposed. A C-GAIMMD generator's probability density is not evaluated during a
run, and the preparation of a generated structure (restrained minimisation,
solvation, a short relaxation) changes it further, so these estimators cannot
be applied to C-GAIMMD shots; this module provides neither. The sidecar of
every shot keeps what is known about how its shooting point was made (the
requested committor, the generator and its settings, the network round).

The rates counted in the free simulations
(:func:`aimmd.analysis.free_md.free_md_rates`) need no committor at all, but
need free simulations that restart from the equilibrium inside each state.

Static bias
-----------
A run with ``params.record_bias`` applies a bias only inside the metastable
states (accelerated AIMMD; see :mod:`aimmd.pathensemble.bias_utils`). Then:

- the rates are also computed with AIMMD's own bias-reweighted estimator,
  :func:`aimmd.pathensemble.bias_utils.bias_reweighted_rates`, which multiplies
  every frame's time by ``exp(bias)`` (the bias in kT): ``k = 1 / sum_i(w_i L_i
  gamma_i)`` with ``gamma_i`` the mean of ``exp(bias)`` over the path's counted
  frames. These are the "bias-reweighted" rates the trainer prints; the plain
  ``k12``/``k21`` are then the rates of the accelerated dynamics;
- every frame's equilibrium weight is multiplied by ``exp(bias)`` at that
  frame, so that populations and free energies describe the unbiased system
  (without it the biased wells look shallower than they are);
- free-simulation time in :func:`aimmd.analysis.free_md.free_md_rates` is the
  boosted time ``sum exp(bias) dt``.

The bias is read from the per-frame caches the trainer wrote
(``<trajectory>.bias.npy``); this module never computes or writes them.

Reading only
------------
Nothing here writes into the run folder: values are recomputed in memory,
results are returned as objects, and their ``save`` methods refuse a target
inside the run folder. Reading goes through AIMMD's own path and cache
classes, switched for the duration of each call to readers that leave no
file behind: the hidden file locks of the caches (``.<cache>.lock``) are not
taken, and MDAnalysis's frame index of each trajectory
(``.<trajectory>_offsets.npz``) is kept in memory instead of being deleted
and rebuilt next to the trajectory. A run folder on a read-only file system
or in an archive is therefore read as it is, and gives the same numbers as a
writable one. Analyse a finished run (or a copy): while the trainer's value
pass is updating a cache, a read without the lock can mix old and new values
of that file.

Values written after the trainer's last pass
--------------------------------------------
The trainer computes committor values for the frames that exist when its
round starts. Frames written later (free simulations keep running, shots keep
being registered) have no cached value, and AIMMD reads a missing value as 0,
a logit of 0, i.e. a committor of one half: an excursion into those frames
would look like progress to the transition state. With ``values='cached'`` and
``until_values=True`` (the default) every free trajectory is therefore cut at
its first reactive frame without a cached value, and shots with such frames
are left out. Frames in the states need no value (the value pass skips them,
and a part file without reactive frames has no values file at all).

Frames without a state
----------------------
The workers compute the state of every frame they write, but a trajectory can
hold frames beyond its states cache (the last frames written after the
worker's final state pass). AIMMD reads their state as ``''``, and the
trainer's reweighting counts them like any frame, as a state of their own.
:func:`reweight_run` keeps them too, so that it reproduces the trainer;
:attr:`RunReweighting.frames_without_state` says how many there are,
:func:`populations` reports their share as ``'unknown'``, and a warning is
given when they are more than :data:`UNKNOWN_STATE_SHARE` of the counted
frames: the rates of such a run rest partly on frames whose state is not
known.
"""

# external
import contextlib
import io
import json
import os
import warnings
from dataclasses import dataclass, field
from math import inf, nan
from typing import Optional

import numpy as np

from ._data_access import without_writing


__all__ = ['RunReweighting', 'reweight_run', 'FrameTable', 'frame_table', 'populations',
           'FreeEnergy', 'free_energy', 'free_energy_difference', 'free_energy_convergence',
           'trainer_rate_estimates', 'cache_rows', 'check_output_outside', 'VALUES_MODES',
           'FRACTION_BY', 'UNKNOWN_STATE_SHARE']


VALUES_MODES = ('cached', 'recompute')
"""Where :func:`reweight_run` takes the committor values from."""

FRACTION_BY = ('shots', 'time')
"""How :func:`reweight_run` picks the first fraction of the shots."""

UNKNOWN_STATE_SHARE = 0.01
"""Share of the counted frames without a known state above which :func:`reweight_run` warns."""

_range = range
"""The builtin, which the ``range`` keywords of the free-energy functions shadow."""

READER_SOURCES = ('reader', 'frames', 'positions', 'times', 'coordinates',
                  'velocities', 'dimensions')
"""Per-frame data that AIMMD reads from the trajectory file, not from a cache."""


# ════════════════════════════════════════════════════════════════════════════
# small helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_params(params):
    """A :class:`aimmd.Params` from an object or a params file (loaded without saving)."""
    from ..params import Params
    if isinstance(params, Params):
        return params
    return Params(os.path.abspath(str(params)), initial_paths=None, save=False)


def _inside(target, folder):
    """Whether `target` resolves to `folder` or to something inside it."""
    target = os.path.realpath(os.path.abspath(str(target)))
    folder = os.path.realpath(os.path.abspath(str(folder)))
    return target == folder or target.startswith(folder.rstrip(os.sep) + os.sep)


def check_output_outside(target, run_dir=None):
    """Refuse an output file inside a run folder.

    Parameters
    ----------
    target : path-like
        The file (or folder) the caller wants to write.
    run_dir : path-like, optional
        The run folder the results come from. Whether it is given or not, a
        target inside any folder that holds AIMMD's run subfolders (shooting
        chains, free simulations, sweep chains) is refused as well.

    Raises
    ------
    ValueError
        If `target` lies inside `run_dir` or inside another run folder. The
        analysis tools never write into a run folder: the workers, the
        trainer and a later restart own it.
    """
    from ._data_access import enclosing_run_folder
    if run_dir is not None and _inside(target, run_dir):
        raise ValueError(f'{target!r} is inside the run folder {run_dir!r}: analysis results '
                         f'go to a folder of their own')
    enclosing = enclosing_run_folder(target)
    if enclosing is not None:
        raise ValueError(f'{target!r} is inside the run folder {enclosing!r} (it holds AIMMD '
                         f'chain, free-simulation or sweep folders): analysis results go to a '
                         f'folder of their own')


def _cache_length(fname, name):
    """Rows of the per-frame cache `name` of trajectory file `fname` (0 if there is none).

    Reads the header only (memory map), without AIMMD's lock: the caches are
    replaced atomically, so the header always belongs to a complete file.
    """
    from ..path.utils import get_cache_fname
    try:
        return int(len(np.load(get_cache_fname(fname, name), mmap_mode='r',
                               allow_pickle=False)))
    except (OSError, ValueError):
        return 0


def _load_cache(fname, name):
    """The per-frame cache `name` of `fname`, or None (read without a lock, see :func:`_cache_length`)."""
    from ..path.utils import get_cache_fname
    try:
        return np.load(get_cache_fname(fname, name), allow_pickle=False)
    except (OSError, ValueError):
        return None


def _frames_with_rows(path, reactive, name):
    """Leading frames of `path` whose reactive frames all have a row in cache `name`.

    Mirrors what the trainer can have seen: it computes values for the
    reactive frames only and zero-fills the others. Frames beyond the states
    cache have no state yet; the trainer keeps them (AIMMD reads their state
    as ``''``), so they do not end the count either.
    """
    n_keep = 0
    for k in range(path.n_files):
        fname = path._fnames[k]
        locs = np.asarray(path._extract(k, 'locs'), dtype=int)
        states = _load_cache(fname, 'states')
        n_states = 0 if states is None else len(states)
        n_rows = _cache_length(fname, name)
        inside = locs < n_states
        labels = np.full(len(locs), '', dtype='<U1')
        if n_states:
            labels[inside] = np.asarray(states).reshape(-1)[locs[inside]]
        missing = (labels == reactive) & (locs >= n_rows)
        if missing.any():
            return n_keep + int(np.argmax(missing))
        n_keep += len(locs)
    return n_keep


def _last_visited(labels, end_states):
    """Last end state visited at every frame of a time-ordered label sequence ('' before the first)."""
    out = np.full(len(labels), '', dtype='<U1')
    current = ''
    for i, label in enumerate(labels):
        if label in end_states:
            current = label
        out[i] = current
    return out


def _capture(function, *args, verbose=False, **kwargs):
    """Call `function`, capturing what it prints; returns ``(result, text)``."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    text = buffer.getvalue()
    if verbose and text:
        print(text, end='', flush=True)
    return result, text


def _jsonable(value):
    """`value` with numpy scalars and arrays turned into plain Python objects."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('inf' if value > 0 else '-inf')
    return value


# ════════════════════════════════════════════════════════════════════════════
# the reweighting of a run
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RunReweighting:
    """The trainer's reweighting of one run folder (one system), repeated offline.

    Rates are in one per frame interval (``1/dt``); ``k12`` goes from
    ``states[0]`` to ``states[2]``, ``k21`` back.
    """

    states: str
    """The sorted states, e.g. ``'ARB'``: first end state, reactive region, last end state."""

    run_dir: str
    """The run folder (of this system, for a multi-system run)."""

    system_id: Optional[str]
    """The system of a multi-system run, None otherwise."""

    pathensemble: object
    """The reweighted :class:`aimmd.PathEnsemble`: the shots first, then the free trajectories,
    each split into its segments (the trainer's order)."""

    weights_forward: np.ndarray
    """``w1``: per-path weights of the direction ``states[0] -> states[2]``."""

    weights_backward: np.ndarray
    """``w2``: per-path weights of the direction ``states[2] -> states[0]``."""

    n_frames: np.ndarray
    """Frames each path counts (:attr:`aimmd.PathEnsemble.n_frames`)."""

    is_shot: np.ndarray
    """True for the shots, False for the free-trajectory segments."""

    origin: list
    """Per path: the chain folder of a shot (``'chainR0'``) or the first file of the free
    trajectory a segment comes from."""

    bins: np.ndarray
    """The adaptive committor bins (logits), as the trainer computes them."""

    reweight_parameters: dict
    """The keywords of :meth:`aimmd.PathEnsemble.reweight`, with the shooting-value cutoffs."""

    k12: float
    """Rate from ``states[0]`` to ``states[2]`` (1/frame), ``1 / sum(w1 * n_frames)``."""

    k21: float
    """Rate from ``states[2]`` to ``states[0]`` (1/frame), ``1 / sum(w2 * n_frames)``."""

    densities: Optional[np.ndarray]
    """The reweighted density of the excursions on :attr:`bins` (what the trainer saves as
    ``densities{states}.npy``), None without adaptive bins."""

    values: str
    """``'cached'`` (the values the trainer stored) or ``'recompute'``."""

    network: Optional[str]
    """The network file the values were recomputed with (None for cached values)."""

    bias_applied: bool
    """Whether the recorded bias is applied (rates through the bias-reweighted estimator,
    ``exp(bias)`` frame weights)."""

    k12_bias_reweighted: Optional[float] = None
    """Bias-reweighted rate ``1 / sum(w1 * n_frames * gamma1)``, None without the bias."""

    k21_bias_reweighted: Optional[float] = None
    """Bias-reweighted rate of the other direction, None without the bias."""

    gamma_forward: Optional[np.ndarray] = None
    """Per-path ``gamma1``: mean ``exp(bias)`` over the counted frames (weighted paths of w1)."""

    gamma_backward: Optional[np.ndarray] = None
    """Per-path ``gamma2`` (weighted paths of w2)."""

    report: str = ''
    """What AIMMD's bias checks printed (reactive bias, zero point, cache coverage, seeds)."""

    frames_without_state: int = 0
    """Counted frames whose state is not known (beyond their file's states cache); the
    trainer counts them too (see the module docstring)."""

    left_out: dict = field(default_factory=dict)
    """Data left out because it has no values yet: ``free_frames`` (per free trajectory, frames
    cut off its end), ``shots`` (the files of the shots left out)."""

    free_restart_source: Optional[str] = None
    """``params.free_restart_source``: how the free simulations were restarted."""

    fraction: Optional[float] = None
    """For a subset: the fraction of the data it holds (None for all data)."""

    fraction_by: Optional[str] = None
    """For a subset: ``'shots'`` or ``'time'``."""

    registered_until: Optional[float] = None
    """For a subset: the registration time (seconds since the epoch) of its last shot."""

    subsets: list = field(default_factory=list)
    """The reweightings of the first fractions of the data (``fractions=`` of
    :func:`reweight_run`), in the order asked for."""

    @property
    def path_weights(self):
        """``w1 + w2`` per path, non-finite entries set to 0: the equilibrium weight of each of
        the path's counted frames (before the bias factor)."""
        weights = np.asarray(self.weights_forward, dtype=float) + np.asarray(
            self.weights_backward, dtype=float)
        weights[~np.isfinite(weights)] = 0.0
        return weights

    @property
    def n_shots(self):
        """Number of shots in the reweighted ensemble."""
        return int(np.count_nonzero(self.is_shot))

    @property
    def shot_frames(self):
        """Frames counted in the shots."""
        return int(np.sum(np.asarray(self.n_frames)[self.is_shot]))

    @property
    def free_frames(self):
        """Frames counted in the free trajectories."""
        return int(np.sum(np.asarray(self.n_frames)[~self.is_shot]))

    def rate(self, initial, final, *, bias_reweighted=None):
        """The rate from state `initial` to state `final` (1/frame).

        Parameters
        ----------
        initial, final : str
            The two end states, e.g. ``'A'`` and ``'B'``.
        bias_reweighted : bool, optional
            The bias-reweighted rate (default: when the bias is applied).
        """
        if bias_reweighted is None:
            bias_reweighted = self.bias_applied
        first, _, last = self.states
        if (initial, final) == (first, last):
            return self.k12_bias_reweighted if bias_reweighted else self.k12
        if (initial, final) == (last, first):
            return self.k21_bias_reweighted if bias_reweighted else self.k21
        raise ValueError(f'the end states are {first!r} and {last!r}, got {initial!r} -> '
                         f'{final!r}')

    def frame_table(self, **kwargs):
        """:func:`frame_table` of this reweighting."""
        return frame_table(self, **kwargs)

    def populations(self):
        """:func:`populations` of this reweighting (from a frame table without collective variables)."""
        table = frame_table(self)
        return populations(table.state, table.last_visited, table.weight, states=self.states)

    def convergence(self):
        """One row per subset and one for all data: how the rates move with the amount of data.

        Returns
        -------
        list of dict
            ``fraction`` (1.0 for all data), ``fraction_by``, ``n_shots``, ``shot_frames``,
            ``free_frames``, ``registered_until``, ``k12``, ``k21``,
            ``k12_bias_reweighted``, ``k21_bias_reweighted``.
        """
        rows = []
        for item in list(self.subsets) + [self]:
            rows.append({'fraction': 1.0 if item.fraction is None else item.fraction,
                         'fraction_by': item.fraction_by, 'n_shots': item.n_shots,
                         'shot_frames': item.shot_frames, 'free_frames': item.free_frames,
                         'registered_until': item.registered_until,
                         'k12': item.k12, 'k21': item.k21,
                         'k12_bias_reweighted': item.k12_bias_reweighted,
                         'k21_bias_reweighted': item.k21_bias_reweighted})
        return rows

    def summary(self):
        """The scalar results as a JSON-able dict (no per-path arrays)."""
        out = {'states': self.states, 'run_dir': self.run_dir, 'system_id': self.system_id,
               'values': self.values, 'network': self.network,
               'bias_applied': self.bias_applied, 'k12': self.k12, 'k21': self.k21,
               'k12_bias_reweighted': self.k12_bias_reweighted,
               'k21_bias_reweighted': self.k21_bias_reweighted,
               'unit': '1/frame', 'n_paths': int(len(self.n_frames)),
               'n_shots': self.n_shots, 'shot_frames': self.shot_frames,
               'free_frames': self.free_frames, 'bins': self.bins,
               'densities': self.densities,
               'reweight_parameters': {k: v for k, v in self.reweight_parameters.items()},
               'n_nonfinite_weights': int(np.count_nonzero(
                   ~np.isfinite(self.weights_forward)) + np.count_nonzero(
                   ~np.isfinite(self.weights_backward))),
               'left_out': self.left_out, 'frames_without_state': self.frames_without_state,
               'free_restart_source': self.free_restart_source,
               'fraction': self.fraction, 'fraction_by': self.fraction_by,
               'registered_until': self.registered_until,
               'subsets': [subset.summary() for subset in self.subsets]}
        return _jsonable(out)

    def save(self, prefix):
        """Write ``<prefix>.json`` (:meth:`summary`) and ``<prefix>.npz`` (the per-path arrays).

        Parameters
        ----------
        prefix : path-like
            Output name without extension, outside the run folder (the folder is created).

        Returns
        -------
        tuple of str
            The two files written.
        """
        prefix = os.path.abspath(str(prefix))
        check_output_outside(prefix, self.run_dir)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        with open(prefix + '.json', 'w') as handle:
            json.dump(self.summary(), handle, indent=1)
        np.savez_compressed(prefix + '.npz', weights_forward=self.weights_forward,
                            weights_backward=self.weights_backward, n_frames=self.n_frames,
                            is_shot=self.is_shot, origin=np.asarray(self.origin, dtype=str),
                            bins=np.asarray(self.bins, dtype=float))
        return prefix + '.json', prefix + '.npz'


def _system_directory(params, run_dir, system_id):
    """The folder of `system_id` in a multi-system run (`run_dir` itself when it is that folder)."""
    candidate = os.path.join(run_dir, str(system_id))
    return candidate if os.path.isdir(candidate) else run_dir


def _attach_values(paths, params, reactive, system_id, batch_size):
    """Compute the committor values of the reactive frames of `paths` in memory.

    The same call the trainer's value pass makes (``params.compute_values_args``:
    the values function on the descriptors series, or on the coordinates
    without descriptors), stored as each path's in-memory ``values`` array
    (zero for the frames in the states, as the trainer's caches). Nothing is
    written. Slices of a path (its segments) carry the array along.
    """
    from ..core.utils import accepts_system_id
    function, _, source = params.compute_values_args
    pass_system_id = system_id is not None and accepts_system_id(function)
    for path in paths:
        labels = np.asarray(path._get('states'))
        mask = labels == reactive
        values = np.zeros(len(path), dtype=float)
        if mask.any():
            rows = np.asarray(path._get(source))[mask]
            out = []
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                result = (function(chunk, system_id=system_id) if pass_system_id
                          else function(chunk))
                out.append(np.asarray(result, dtype=float).reshape(-1))
            values[mask] = np.concatenate(out)
        path.values = values


def _assemble(chains, free):
    """The trainer's ensemble (``assemble_pathensemble(chains, free)``) with each path's origin."""
    from ..pathensemble import PathEnsemble
    from ..pathensemble.utils import assemble_pathensemble
    shots, origin = [], []
    for chain in chains:
        for path in chain._paths:
            shots.append(path)
            origin.append(os.path.basename(os.path.dirname(str(path.fnames[0]))))
    pieces = []
    for trajectory in free:
        piece = assemble_pathensemble(trajectory)
        pieces.append(piece)
        origin.extend([str(trajectory.fnames[0])] * len(piece))
    ensemble = PathEnsemble()
    ensemble._paths = list(shots) + [path for piece in pieces for path in piece._paths]
    is_shot = np.zeros(len(ensemble._paths), dtype=bool)
    is_shot[:len(shots)] = True
    return ensemble, is_shot, origin


def _reweight(params, run_dir, system_id, chains, free, *, values, network, bias_applied,
              verbose):
    """Reweight one ensemble exactly as the trainer does; returns a :class:`RunReweighting`."""
    from .utils import compute_bins
    from ..worker.utils import reweight_parameters_with_bins
    states = params.sorted_states
    reactive = states[1]
    ensemble, is_shot, origin = _assemble(chains, free)
    if not len(ensemble):
        raise ValueError(f'no paths in {run_dir!r}: nothing to reweight')
    bins = compute_bins(ensemble, params.nbins, cutoff_max=params.cutoff_max,
                        cutoff_min=params.cutoff_min, find_extremes_with='free',
                        source='values', states=states,
                        terminal_bin_extension=params.terminal_bin_extension)
    reweight_parameters = reweight_parameters_with_bins(params.reweight_parameters, bins)
    w1 = np.asarray(ensemble.reweight(states, **reweight_parameters, source='values')[0])
    w2 = np.asarray(ensemble.reweight(states[::-1], **reweight_parameters, source='values')[0])
    lengths = ensemble.n_frames
    # the trainer's expressions ("bonus track: estimate rates"), unchanged
    k12 = np.sum(w1 * lengths)
    k12 = 1 / k12 if k12 else nan
    k21 = np.sum(w2 * lengths)
    k21 = 1 / k21 if k21 else nan

    densities = None
    if len(bins):
        # the trainer's projection: (w1 + w2) on the excursions, internal frames
        saved = np.array(ensemble.weights[:], dtype=float)
        try:
            ensemble.weights = (w1 + w2) * ensemble.types(f'.{reactive}..')
            densities = ensemble.project(bins, source='values')
        finally:
            ensemble.weights = saved
        densities[densities == 0.] = 1e-15
        densities /= densities.sum()

    starts, stops = ensemble.frame_windows
    unknown = 0
    for path, start, stop in zip(ensemble._paths, starts, stops):
        unknown += int(np.count_nonzero(np.asarray(path._get('states'))[start:stop] == ''))
    result = RunReweighting(
        frames_without_state=unknown,
        states=states, run_dir=run_dir, system_id=system_id, pathensemble=ensemble,
        weights_forward=w1, weights_backward=w2, n_frames=np.asarray(lengths),
        is_shot=is_shot, origin=origin, bins=np.asarray(bins, dtype=float),
        reweight_parameters=reweight_parameters, k12=float(k12), k21=float(k21),
        densities=densities, values=values, network=network, bias_applied=bias_applied,
        free_restart_source=getattr(params, 'free_restart_source', None))

    if bias_applied:
        from ..pathensemble.bias_utils import bias_reweighted_rates
        threshold = (params.bias_reactive_threshold_of(system_id) if system_id is not None
                     else params.bias_reactive_threshold)
        label = f'[system {system_id!r}] ' if system_id is not None else ''
        (k12_rw, k21_rw, gamma1, gamma2), text = _capture(
            bias_reweighted_rates, ensemble, w1, w2, lengths=lengths, states=states,
            reactive_threshold=threshold, label=label, verbose=verbose)
        result.k12_bias_reweighted = float(k12_rw)
        result.k21_bias_reweighted = float(k21_rw)
        result.gamma_forward = np.asarray(gamma1)
        result.gamma_backward = np.asarray(gamma2)
        result.report = text
    return result


def _shot_time(path):
    """Registration time of a shot (seconds since the epoch).

    The time its C-GAIMMD sidecar records (``registered``, to the second) when
    there is one, else the modification time of its trajectory file.
    """
    import datetime
    fname = str(path.fnames[0])
    try:
        from ..cgaimmd.sidecar import read_shot_sidecar
        record = read_shot_sidecar(fname) or {}
        registered = record.get('registered')
        if registered:
            return datetime.datetime.fromisoformat(str(registered)).timestamp()
    except (ValueError, TypeError, OSError):
        pass
    return os.path.getmtime(fname)


TIED_REGISTRATION_SHARE = 0.5
"""Share of shots with a registration time shared by another shot above which the
registration order is reported as unreliable."""


def _registration_order(chains):
    """``[(time, rank, k, j)]`` of every shot ``j`` of chain ``k``, in registration order.

    Within a chain the shots are registered in the order of their path numbers,
    so a shot's time is at least that of the shot before it in its chain
    (``rank`` is its place there). Shots registered at the same time (a run
    copied without its file times, or several chains within one second) are
    ordered by their place in their chain, so the chains interleave instead of
    entering one after the other; a warning is given when more than
    :data:`TIED_REGISTRATION_SHARE` of the shots tie.
    """
    order = []
    for k, chain in enumerate(chains):
        latest = -inf
        for j, path in enumerate(chain._paths):
            latest = max(latest, _shot_time(path))
            order.append((latest, j, k, j))
    order.sort()
    times = np.array([entry[0] for entry in order])
    if len(times) > 1:
        _, counts = np.unique(times, return_counts=True)
        tied = int(counts[counts > 1].sum())
        if tied > TIED_REGISTRATION_SHARE * len(times):
            warnings.warn(
                f'{tied} of {len(times)} shots share their registration time with another shot '
                f'(files copied without their modification times?); among them the order '
                f'follows the path numbers, chain by chain in turn', UserWarning, stacklevel=4)
    return order


def _subset(chains, free, fraction, by, order=None):
    """The first `fraction` of the data: the shots registered first, and the same share of every
    free-simulation folder's frames (its trajectories in file order, assuming they ran at a
    constant speed). Returns ``(chains, free, registered_until)``."""
    from ..pathensemble import PathEnsemble
    shots = [(time, k, j) for time, _, k, j in
             (_registration_order(chains) if order is None else order)]
    if by == 'shots':
        keep = shots[:int(round(fraction * len(shots)))]
    else:
        if shots:
            first, last = shots[0][0], shots[-1][0]
            limit = first + fraction * (last - first)
            keep = [entry for entry in shots if entry[0] <= limit]
        else:
            keep = []
    kept = {(k, j) for _, k, j in keep}
    chains_subset = []
    for k, chain in enumerate(chains):
        ensemble = PathEnsemble()
        ensemble._paths = [path for j, path in enumerate(chain._paths) if (k, j) in kept]
        chains_subset.append(ensemble)
    by_folder = {}
    for trajectory in free:
        by_folder.setdefault(os.path.dirname(str(trajectory.fnames[0])), []).append(trajectory)
    free_subset = []
    for folder in sorted(by_folder):
        trajectories = sorted(by_folder[folder], key=lambda t: str(t.fnames[0]))
        budget = int(round(fraction * sum(len(t) for t in trajectories)))
        for trajectory in trajectories:
            if budget < 2:                     # a path needs two frames
                break
            free_subset.append(trajectory if len(trajectory) <= budget
                               else trajectory[:budget])
            budget -= len(trajectory)
    registered_until = keep[-1][0] if keep else None
    return chains_subset, free_subset, registered_until


@without_writing
def reweight_run(params, run_dir, *, values='cached', network=None, bias='auto',
                 until_values=True, fractions=(), fraction_by='shots', system_id=None,
                 batch_size=None, verbose=False):
    """Repeat the trainer's reweighting of a run folder offline.

    Parameters
    ----------
    params : aimmd.Params or path-like
        The run's params (an object, or the params file, loaded without saving a copy).
    run_dir : path-like
        The run folder (read only).
    values : {'cached', 'recompute'}, optional
        ``'cached'`` (default): the committor values the trainer stored, which
        reproduces the rates it printed after its last round. ``'recompute'``:
        every reactive frame is scored again, in memory, with `network`; this is
        the cleaner choice for the final analysis of a finished run, because
        then every frame has a value from the same network, and the only choice
        for a run with ``params.subsample_caps`` (see the Notes).
    network : path-like, optional
        For ``values='recompute'``: the network file to load into
        ``params.network`` (default: the run's ``network{states}.h5``).
        ``params.network`` is changed in place.
    bias : {'auto', True, False}, optional
        Apply the recorded static bias (see the module docstring). ``'auto'``
        (default) applies it when ``params.record_bias`` is set.
    until_values : bool, optional
        With cached values: cut every free trajectory at its first reactive
        frame without a value, and leave out shots with such frames (default
        True; see the module docstring). With recomputed values the same is
        done for frames without a row of the descriptors the values are
        computed from.
    fractions : sequence of float, optional
        Also reweight the first fractions of the data (values in ``(0, 1]``);
        the results are in :attr:`RunReweighting.subsets`.
    fraction_by : {'shots', 'time'}, optional
        ``'shots'`` (default): a subset holds the first ``fraction`` of the
        shots in registration order. ``'time'``: the shots registered within
        the first ``fraction`` of the time between the first and the last
        registration. Either way every free-simulation folder contributes the
        same share of its frames. The registration time of a shot is the one
        its C-GAIMMD sidecar records, else the modification time of its file
        (so copy a run with its file times, e.g. ``cp -p`` or ``rsync -a``);
        within a chain the path numbers give the order.
    system_id : str, optional
        A multi-system run: the system to reweight. None reweights every
        system and returns a dict.
    batch_size : int, optional
        Frames per call of the values function (default ``params.network_batch_size``).
    verbose : bool, optional
        Print what AIMMD's bias checks report (it is in :attr:`RunReweighting.report` anyway).

    Returns
    -------
    RunReweighting or dict
        One :class:`RunReweighting`; for a multi-system run without
        `system_id`, ``{system_id: RunReweighting}``.

    Warns
    -----
    UserWarning
        When more than :data:`UNKNOWN_STATE_SHARE` of the counted frames have
        no known state (see the module docstring).

    Raises
    ------
    ValueError
        For an unknown `values`, `bias` or `fraction_by`, a fraction outside
        ``(0, 1]``, a TPS run (its paths have no equilibrium weights), a run
        folder without paths, ``bias=True`` without a recorded bias, or
        ``values='cached'`` in a run with active ``params.subsample_caps``.

    Notes
    -----
    The reproduction of the trainer's printed rates is exact when the run
    folder holds the data of the trainer's last reweighting: the trainer's
    eval ensemble is the whole ensemble unless ``params.subsample_caps`` is
    set. With caps the trainer computes committor values for a random
    subsample of the frames only and prints no rate, because that subsample
    does not represent the ensemble; the cached values then cover only part
    of the data, so ``values='cached'`` is refused and ``values='recompute'``
    scores every frame of the whole ensemble. With
    ``params.rescale_committor`` the cached values are the rescaled ones,
    which the trainer's printed rates predate.
    """
    if values not in VALUES_MODES:
        raise ValueError(f'values must be one of {VALUES_MODES}, got {values!r}')
    if bias not in ('auto', True, False):
        raise ValueError(f"bias must be 'auto', True or False, got {bias!r}")
    if fraction_by not in FRACTION_BY:
        raise ValueError(f'fraction_by must be one of {FRACTION_BY}, got {fraction_by!r}')
    fractions = [float(f) for f in fractions]
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f'fractions must lie in (0, 1], got {fraction}')
    params = _load_params(params)
    run_dir = os.path.abspath(str(run_dir))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'no run folder {run_dir!r}')
    if params.chain_type == 'tps':
        raise ValueError("a TPS run (chain_type='tps') samples the transition path ensemble "
                         "only: its paths have no equilibrium weights to reweight")
    record_bias = bool(getattr(params, 'record_bias', False))
    if bias is True and not record_bias:
        raise ValueError('bias=True, but the run records no bias (params.record_bias is False)')
    bias_applied = record_bias if bias == 'auto' else bool(bias)

    if getattr(params, 'multi_system', False) and system_id is None:
        return {sid: reweight_run(params, run_dir, values=values, network=network, bias=bias,
                                  until_values=until_values, fractions=fractions,
                                  fraction_by=fraction_by, system_id=sid,
                                  batch_size=batch_size, verbose=verbose)
                for sid in params.system_ids}
    directory = (_system_directory(params, run_dir, system_id)
                 if getattr(params, 'multi_system', False) else run_dir)
    if not getattr(params, 'multi_system', False):
        system_id = None
    caps_active = getattr(params, 'subsample_caps_active', None)
    if values == 'cached' and callable(caps_active) and caps_active(system_id):
        raise ValueError(
            "params.subsample_caps is set: the trainer computed committor values for a random "
            "subsample of the frames only, so the cached values do not cover the ensemble and "
            "rates from them would be biased; use values='recompute', which scores every "
            "reactive frame of the whole ensemble with the run's network")

    states = params.sorted_states
    reactive = states[1]
    chains = params.shot_chains(directory, None)
    free = params.free_trajectories(directory)

    network_file = None
    if values == 'recompute':
        network_file = os.path.abspath(str(network)) if network is not None else \
            params._network_fname(directory)
        params.update_network(network_file, timeout=0)
        source = params.compute_values_args[2]
        rows_of = None if source in READER_SOURCES else source
    else:
        rows_of = 'values'

    left_out = {'free_frames': {}, 'shots': []}
    if until_values and rows_of is not None:
        cut = []
        for trajectory in free:
            n_keep = _frames_with_rows(trajectory, reactive, rows_of)
            if n_keep < len(trajectory):
                left_out['free_frames'][str(trajectory.fnames[0])] = len(trajectory) - n_keep
                if n_keep < 2:
                    continue
                trajectory = trajectory[:n_keep]
            cut.append(trajectory)
        free = cut
        from ..pathensemble import PathEnsemble
        kept_chains = []
        for chain in chains:
            ensemble = PathEnsemble()
            for path in chain._paths:
                if _frames_with_rows(path, reactive, rows_of) < len(path):
                    left_out['shots'].append(str(path.fnames[0]))
                else:
                    ensemble._paths.append(path)
            kept_chains.append(ensemble)
        chains = kept_chains

    if values == 'recompute':
        batch = int(batch_size or getattr(params, 'network_batch_size', None) or 4096)
        _attach_values([path for chain in chains for path in chain._paths] + list(free),
                       params, reactive, system_id, batch)

    result = _reweight(params, directory, system_id, chains, free, values=values,
                       network=network_file, bias_applied=bias_applied, verbose=verbose)
    result.left_out = left_out
    order = _registration_order(chains) if fractions else None
    for fraction in fractions:
        chains_subset, free_subset, registered_until = _subset(chains, free, fraction,
                                                               fraction_by, order)
        subset = _reweight(params, directory, system_id, chains_subset, free_subset,
                           values=values, network=network_file, bias_applied=bias_applied,
                           verbose=verbose)
        subset.fraction = fraction
        subset.fraction_by = fraction_by
        subset.registered_until = registered_until
        result.subsets.append(subset)
    shot_times = ([entry[0] for entry in order] if order is not None else
                  [_shot_time(path) for chain in chains for path in chain._paths])
    result.registered_until = max(shot_times) if shot_times else None
    counted = int(np.sum(result.n_frames))
    if counted and result.frames_without_state > UNKNOWN_STATE_SHARE * counted:
        label = f'system {system_id!r}: ' if system_id is not None else ''
        warnings.warn(
            f'{label}{result.frames_without_state} of the {counted} counted frames '
            f'({100 * result.frames_without_state / counted:.1f} %) have no known state (written '
            f'after the last state pass of their worker); the trainer counts them, and so does '
            f'reweight_run to reproduce it, so the rates rest partly on frames whose state is '
            f'not known', UserWarning, stacklevel=3)
    return result


# ════════════════════════════════════════════════════════════════════════════
# per-frame table
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameTable:
    """One row per counted frame of a reweighted ensemble.

    A frame appears once per path that counts it; the paths of one free
    trajectory partition it, so its frames appear once. Shots and free
    trajectories are different files, so frames are not shared between them.
    """

    states: str
    """The sorted states, e.g. ``'ARB'``."""

    path: np.ndarray
    """Index of the frame's path in the reweighted ensemble."""

    file: np.ndarray
    """Index of the frame's trajectory file in :attr:`files`."""

    frame: np.ndarray
    """Index of the frame in its trajectory file."""

    files: list
    """The trajectory files."""

    state: np.ndarray
    """State label of the frame ('' when unknown)."""

    last_visited: np.ndarray
    """The end state the path visited last, at this frame ('' before the first visit)."""

    is_shot: np.ndarray
    """Whether the frame belongs to a shot (else to a free trajectory)."""

    path_weight: np.ndarray
    """``w1 + w2`` of the frame's path."""

    bias: np.ndarray
    """The recorded bias at the frame (kT); 0 when the bias is not applied, and 0 for a frame
    without a row in its bias cache (see :attr:`bias_missing`)."""

    weight: np.ndarray
    """Equilibrium weight of the frame: ``path_weight * exp(bias)``."""

    value: np.ndarray
    """The committor value (logit toward ``states[2]``) the reweighting used; 0 in the states."""

    cv: Optional[np.ndarray] = None
    """``(n_frames, n_cv)`` collective variables, None if none were asked for."""

    cv_names: tuple = ()
    """Names of the :attr:`cv` columns."""

    bias_applied: bool = False
    """Whether :attr:`weight` includes ``exp(bias)``."""

    run_dir: Optional[str] = None
    """The run folder the table comes from (:meth:`save` refuses to write there)."""

    bias_missing: Optional[np.ndarray] = None
    """With the bias applied: True for the frames without a row in their bias cache (their bias
    reads as 0, so their weight lacks the factor ``exp(bias)``); None otherwise."""

    def __len__(self):
        return int(len(self.weight))

    @property
    def frames_without_bias(self):
        """The number of frames without a bias row (0 when the bias is not applied)."""
        return 0 if self.bias_missing is None else int(np.count_nonzero(self.bias_missing))

    def column(self, name):
        """The collective variable `name` (or a column index)."""
        if self.cv is None:
            raise ValueError('the table has no collective variables')
        index = self.cv_names.index(name) if isinstance(name, str) else int(name)
        return self.cv[:, index]

    def select(self, mask):
        """The rows where `mask` is True, as a new table."""
        mask = np.asarray(mask)
        out = {}
        for name in ('path', 'file', 'frame', 'state', 'last_visited', 'is_shot',
                     'path_weight', 'bias', 'weight', 'value'):
            out[name] = getattr(self, name)[mask]
        return FrameTable(states=self.states, files=self.files,
                          cv=None if self.cv is None else self.cv[mask], cv_names=self.cv_names,
                          bias_applied=self.bias_applied, run_dir=self.run_dir,
                          bias_missing=(None if self.bias_missing is None
                                        else self.bias_missing[mask]), **out)

    def populations(self):
        """:func:`populations` of the table."""
        return populations(self.state, self.last_visited, self.weight, states=self.states)

    def save(self, fname):
        """Write the table to `fname` (``.npz``, outside the run folder); returns the file name.

        The arrays keep their names, except :attr:`file`, stored as ``file_index``.
        """
        fname = os.path.abspath(str(fname))
        check_output_outside(fname, self.run_dir)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        arrays = {name: getattr(self, name) for name in (
            'path', 'frame', 'state', 'last_visited', 'is_shot', 'path_weight', 'bias',
            'weight', 'value')}
        arrays['file_index'] = self.file        # ('file' is numpy's name for the target)
        if self.cv is not None:
            arrays['cv'] = self.cv
        if self.bias_missing is not None:
            arrays['bias_missing'] = self.bias_missing
        arrays.update(files=np.asarray(self.files, dtype=str),
                      cv_names=np.asarray(self.cv_names, dtype=str),
                      states=np.asarray(self.states), bias_applied=self.bias_applied)
        np.savez_compressed(fname, **arrays)
        return fname if fname.endswith('.npz') else fname + '.npz'


def _normalise_cv(cv):
    """``{name: spec}`` from None, a cache name, a callable or a dict of them."""
    if cv is None:
        return {}
    if isinstance(cv, dict):
        return dict(cv)
    if isinstance(cv, str):
        return {cv: cv}
    if callable(cv):
        return {getattr(cv, '__name__', 'cv'): cv}
    raise TypeError(f'cv must be None, a cache name, a callable or a dict of them, got {cv!r}')


def _file_columns(fname, cv, cv_source, cache):
    """``(n_frames_in_file, n_cv)`` collective variables of one trajectory file, computed once.

    Callables get the whole file at once: a reader (``cv_source='reader'``)
    or the rows of the per-frame cache `cv_source`; a string names a cache
    whose rows are used as they are.
    """
    if fname in cache:
        return cache[fname]
    from ..path import Path
    whole = Path(fname)
    source_data = None
    columns = []
    for spec in cv.values():
        if isinstance(spec, str):
            data = np.asarray(whole._get(spec), dtype=float)
        else:
            if source_data is None:
                source_data = (whole._get('reader') if cv_source == 'reader'
                               else np.asarray(whole._get(cv_source)))
            data = np.asarray(spec(source_data), dtype=float)
        data = data.reshape(len(data), -1)
        if len(data) != len(whole):
            raise ValueError(f'the collective variable gave {len(data)} rows for the '
                             f'{len(whole)} frames of {fname!r}')
        columns.append(data)
    out = np.concatenate(columns, axis=1)
    cache[fname] = out
    return out


@without_writing
def frame_table(reweighting, *, cv=None, cv_source='reader', include_zero_weight=False,
                cv_cache=None):
    """One row per counted frame of a reweighted ensemble: weight, state, history, value, CVs.

    Parameters
    ----------
    reweighting : RunReweighting
        The result of :func:`reweight_run` (or one of its subsets).
    cv : None, str, callable or dict, optional
        Collective variables to add as columns:

        - a string names a per-frame cache of the run (``'values'``,
          ``'descriptors'``, ``'bias'``, a named descriptors series); its rows
          become columns (all of them for a multi-column cache);
        - a callable gets one whole trajectory file at a time, as the data of
          `cv_source`, and returns one value (or one row) per frame;
        - a dict ``{name: str or callable}`` gives several columns.
    cv_source : str, optional
        What the callables get: ``'reader'`` (default; an MDAnalysis
        reader over the file's frames, as ``states_function`` gets it) or the
        name of a per-frame cache (e.g. ``'descriptors'``; much faster than
        reading the trajectory).
    include_zero_weight : bool, optional
        Keep the frames of paths with zero weight too (default False).
    cv_cache : dict, optional
        ``{file: columns}`` filled with the collective variables of every file
        read; pass the same dict to several calls (subsets of one run) to
        compute each file once.

    Returns
    -------
    FrameTable

    Warns
    -----
    UserWarning
        With the bias applied, when frames in the end states have no row in
        their bias cache: their bias reads as 0, so their equilibrium weight
        lacks the factor ``exp(bias)`` and the populations and free energies
        come out wrong. :attr:`FrameTable.bias_missing` marks every frame
        without a row (in the reactive region the bias is zero anyway).

    Notes
    -----
    The frames of a path are those its count includes
    (:attr:`aimmd.PathEnsemble.frame_windows`): the boundary frames that a
    segment shares with its neighbour are left out, so that the frames of a
    free trajectory appear once, and the frames the rates count are the frames
    the weights cover. The state history (:attr:`FrameTable.last_visited`) is
    taken along the whole path before the window is cut.
    """
    cv = _normalise_cv(cv)
    cache = {} if cv_cache is None else cv_cache
    ensemble = reweighting.pathensemble
    states = reweighting.states
    end_states = (states[0], states[2])
    path_weights = reweighting.path_weights
    starts, stops = ensemble.frame_windows
    file_index, file_names = {}, []
    columns = {name: [] for name in ('path', 'file', 'frame', 'state', 'last_visited',
                                     'is_shot', 'path_weight')}
    in_memory_values = []          # (row offset, values) of paths with recomputed values
    n_rows = 0
    for i, path in enumerate(ensemble._paths):
        weight = float(path_weights[i])
        start, stop = int(starts[i]), int(stops[i])
        if stop <= start or (weight == 0.0 and not include_zero_weight):
            continue
        labels = np.asarray(path._get('states')).astype('<U1')
        history = _last_visited(labels, end_states)
        files, frames = [], []
        for k in range(path.n_files):
            locs = np.asarray(path._extract(k, 'locs'), dtype=int)
            fname = path._fnames[k]
            index = file_index.get(fname)
            if index is None:
                index = file_index[fname] = len(file_names)
                file_names.append(fname)
            files.append(np.full(len(locs), index, dtype=int))
            frames.append(locs)
        n = stop - start
        columns['path'].append(np.full(n, i, dtype=int))
        columns['file'].append(np.concatenate(files)[start:stop])
        columns['frame'].append(np.concatenate(frames)[start:stop])
        columns['state'].append(labels[start:stop])
        columns['last_visited'].append(history[start:stop])
        columns['is_shot'].append(np.full(n, bool(reweighting.is_shot[i])))
        columns['path_weight'].append(np.full(n, weight))
        if 'values' in path.__dict__:
            in_memory_values.append((n_rows, np.asarray(path._get('values'),
                                                        dtype=float)[start:stop]))
        n_rows += n
    empty = {'path': int, 'file': int, 'frame': int, 'state': '<U1', 'last_visited': '<U1',
             'is_shot': bool, 'path_weight': float}
    out = {name: (np.concatenate(parts) if parts else np.zeros(0, dtype=empty[name]))
           for name, parts in columns.items()}
    files, frames = out['file'], out['frame']
    # per-frame caches, read once per file (AIMMD's convention: a missing row reads as 0)
    out['value'] = cache_rows(file_names, files, frames, 'values')[0]
    for offset, values in in_memory_values:
        out['value'][offset:offset + len(values)] = values
    bias_missing = None
    if reweighting.bias_applied:
        out['bias'], bias_missing = cache_rows(file_names, files, frames, 'bias')
        in_states = bias_missing & np.isin(out['state'], end_states)
        if in_states.any():
            warnings.warn(
                f'{int(in_states.sum())} frames in the states ({int(bias_missing.sum())} in all) '
                f'have no row in their bias cache: their bias reads as 0, so their weight lacks '
                f'exp(bias) and populations and free energies are off. The trainer writes the '
                f'bias caches; let it pass over these files first.', UserWarning, stacklevel=3)
    else:
        out['bias'] = np.zeros(len(files))
    weight = out['path_weight'] * np.exp(out['bias'])
    cv_array, cv_names = None, ()
    if cv:
        cv_array = None
        for f in np.unique(files):
            where = files == f
            data = _file_columns(file_names[f], cv, cv_source, cache)
            if cv_array is None:
                cv_array = np.empty((len(files), data.shape[1]), dtype=float)
            cv_array[where] = data[frames[where]]
        if cv_array is None:
            cv_array = np.zeros((0, len(cv)), dtype=float)
        names = list(cv)
        if cv_array.shape[1] == len(names):
            cv_names = tuple(names)
        else:
            cv_names = tuple(f'cv{j}' for j in range(cv_array.shape[1]))
    return FrameTable(states=states, files=file_names, weight=weight, cv=cv_array,
                      cv_names=cv_names, bias_applied=reweighting.bias_applied,
                      run_dir=reweighting.run_dir, bias_missing=bias_missing, **out)


def cache_rows(file_names, files, frames, name):
    """The first column of the per-frame cache `name` at every ``(file, frame)`` of a table.

    Parameters
    ----------
    file_names : list of str
        Trajectory files.
    files, frames : array-like of int
        Per row: the index into `file_names`, and the frame in that file.
    name : str
        The cache (``'values'``, ``'bias'``, ...).

    Returns
    -------
    tuple of numpy arrays
        ``(rows, missing)``: the values (0 where the cache has no row, as AIMMD
        reads a missing row), and which rows had none. Each file's cache is
        read once, without AIMMD's lock (the caches are replaced atomically).
    """
    files = np.asarray(files, dtype=int)
    frames = np.asarray(frames, dtype=int)
    out = np.zeros(len(files), dtype=float)
    missing = np.ones(len(files), dtype=bool)
    for f in np.unique(files):
        data = _load_cache(file_names[f], name)
        if data is None or not len(data):
            continue
        data = np.asarray(data, dtype=float).reshape(len(data), -1)[:, 0]
        where = np.flatnonzero(files == f)
        ok = frames[where] < len(data)
        out[where[ok]] = data[frames[where][ok]]
        missing[where[ok]] = False
    return out, missing


def populations(state, last_visited, weights, *, states='ARB'):
    """Equilibrium populations of the states and of the two state histories.

    Parameters
    ----------
    state, last_visited : array-like of str
        Per frame: the state label, and the end state visited last ('' before
        the first visit).
    weights : array-like of float
        Per frame equilibrium weights (:attr:`FrameTable.weight`).
    states : str, optional
        The sorted states, e.g. ``'ARB'``.

    Returns
    -------
    dict
        ``{'state': {label: share}, 'last_visited': {label: share, 'none': share},
        'total_weight': float}``. ``state`` gives the share of the weight in
        each region (the probability of finding the system there): the three
        of `states`, any other label the states function gives, and
        ``'unknown'`` for frames without a state; the shares sum to 1.
        ``last_visited`` gives the share whose last end state was each one (the
        "history" populations, whose ratio is about the ratio of the rates in
        a two-state system: ``P(last first) / P(last last) = k21 / k12``).
        Shares are 0 when the total weight is not positive.
    """
    state = np.asarray(state)
    last_visited = np.asarray(last_visited)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    def share(mask):
        return float(np.sum(weights[mask]) / total) if total > 0 else 0.0
    labels = list(states) + sorted(set(str(x) for x in np.unique(state)) - set(states) - {''})
    shares = {label: share(state == label) for label in labels}
    if np.any(state == ''):
        shares['unknown'] = share(state == '')
    return {'state': shares,
            'last_visited': {**{label: share(last_visited == label)
                                for label in (states[0], states[2])},
                             'none': share(last_visited == '')},
            'total_weight': total}


# ════════════════════════════════════════════════════════════════════════════
# free energies
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FreeEnergy:
    """A free-energy profile or surface on a regular grid, in units of kT."""

    free_energy: np.ndarray
    """``-ln p`` per bin, shifted so that the lowest bin is 0; ``inf`` where there is no data
    (or fewer frames than ``min_frames``)."""

    probability: np.ndarray
    """The weighted share of the frames in each bin (sums to 1 over the bins)."""

    n_frames: np.ndarray
    """Frames per bin, unweighted: how much data each bin rests on."""

    edges: tuple
    """Bin edges, one array per dimension."""

    names: tuple = ()
    """Names of the dimensions."""

    @property
    def centers(self):
        """Bin centres, one array per dimension."""
        return tuple(0.5 * (e[1:] + e[:-1]) for e in self.edges)


def free_energy(data, weights=None, *, columns=None, bins=50, range=None, names=None,
                min_frames=1):
    """Free energy ``F = -ln p`` (kT) of one or more collective variables from weighted frames.

    Parameters
    ----------
    data : FrameTable or array-like
        A :class:`FrameTable` with collective variables (its :attr:`FrameTable.weight`
        is used), or an ``(n,)`` / ``(n, d)`` array of values.
    weights : array-like, optional
        Per-frame weights for an array `data` (default: all 1, a plain histogram).
    columns : sequence of str or int, optional
        For a table: the collective variables to use (default: all).
    bins : int, sequence or list of arrays, optional
        As for :func:`numpy.histogramdd` (default 50 per dimension).
    range : sequence of (min, max), optional
        As for :func:`numpy.histogramdd`.
    names : sequence of str, optional
        Names of the dimensions (default: the table's column names).
    min_frames : int, optional
        Bins with fewer frames get ``F = inf`` (default 1: any data).

    Returns
    -------
    FreeEnergy

    Notes
    -----
    Frames with a non-finite value or weight, or a weight of zero, are left
    out. The surface is shifted so that its lowest populated bin is 0; only
    differences of ``F`` carry meaning. The probability is per bin, not a
    density: with bins of different sizes, divide by the bin volume first.
    """
    if isinstance(data, FrameTable):
        if data.cv is None:
            raise ValueError('the frame table has no collective variables (frame_table(cv=...))')
        if columns is None:
            index = list(_range(data.cv.shape[1]))
        else:
            index = [data.cv_names.index(c) if isinstance(c, str) else int(c) for c in columns]
        values = data.cv[:, index]
        weights = data.weight
        if names is None and data.cv_names:
            names = tuple(data.cv_names[j] for j in index)
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if columns is not None:
            values = values[:, list(columns)]
        weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(values):
        raise ValueError(f'{len(weights)} weights for {len(values)} frames')
    keep = np.all(np.isfinite(values), axis=1) & np.isfinite(weights) & (weights != 0)
    if np.any(weights[keep] < 0):
        raise ValueError('negative frame weights')
    histogram, edges = np.histogramdd(values[keep], bins=bins, range=range,
                                      weights=weights[keep])
    counts, _ = np.histogramdd(values[keep], bins=edges)
    total = histogram.sum()
    probability = histogram / total if total > 0 else np.zeros_like(histogram)
    with np.errstate(divide='ignore'):
        energy = -np.log(probability)
    energy[(probability <= 0) | (counts < min_frames)] = inf
    finite = np.isfinite(energy)
    if finite.any():
        energy[finite] -= energy[finite].min()
    return FreeEnergy(free_energy=energy, probability=probability, n_frames=counts,
                      edges=tuple(edges), names=tuple(names or ()))


def free_energy_difference(estimate, reference):
    """How far a free-energy surface is from a reference on the same grid.

    Returns
    -------
    dict
        ``rms`` (kT): the root-mean-square difference over the bins where both
        have data, weighted by the reference probability, after removing the
        weighted mean difference (free energies are defined up to a
        constant); ``shift`` (kT): that mean difference; ``coverage``: the
        share of the reference probability in bins where the estimate has data
        (1 when the estimate reaches every populated bin).
    """
    a, b = np.asarray(estimate.free_energy), np.asarray(reference.free_energy)
    if a.shape != b.shape:
        raise ValueError(f'different grids: {a.shape} and {b.shape}')
    p = np.asarray(reference.probability, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    weight = p[both]
    if not weight.sum() > 0:
        return {'rms': nan, 'shift': nan, 'coverage': 0.0}
    difference = a[both] - b[both]
    shift = float(np.sum(weight * difference) / weight.sum())
    rms = float(np.sqrt(np.sum(weight * (difference - shift) ** 2) / weight.sum()))
    coverage = float(weight.sum() / p[np.isfinite(b)].sum())
    return {'rms': rms, 'shift': shift, 'coverage': coverage}


def free_energy_convergence(reweighting, cv, *, cv_source='reader', columns=None, bins=50,
                            range=None, min_frames=1):
    """Free-energy surfaces of the data subsets of a reweighting, on the grid of all data.

    Parameters
    ----------
    reweighting : RunReweighting
        A result of :func:`reweight_run` with ``fractions=``.
    cv, cv_source : optional
        As for :func:`frame_table`; each trajectory file is read once for all subsets.
    columns, bins, range, min_frames : optional
        As for :func:`free_energy`; the grid comes from all the data.

    Returns
    -------
    dict
        ``full`` (:class:`FreeEnergy` of all the data) and ``subsets``: per
        subset ``fraction``, ``n_shots``, ``free_energy`` and ``difference``
        (:func:`free_energy_difference` from the full-data surface).
    """
    cache = {}
    table = frame_table(reweighting, cv=cv, cv_source=cv_source, cv_cache=cache)
    full = free_energy(table, columns=columns, bins=bins, range=range, min_frames=min_frames)
    out = []
    for subset in reweighting.subsets:
        sub_table = frame_table(subset, cv=cv, cv_source=cv_source, cv_cache=cache)
        surface = free_energy(sub_table, columns=columns, bins=list(full.edges),
                              min_frames=min_frames)
        out.append({'fraction': subset.fraction, 'fraction_by': subset.fraction_by,
                    'n_shots': subset.n_shots, 'free_energy': surface,
                    'difference': free_energy_difference(surface, full)})
    return {'full': full, 'subsets': out}


# ════════════════════════════════════════════════════════════════════════════
# the trainer's printed rates
# ════════════════════════════════════════════════════════════════════════════

_TIME_FORMAT = '%a %b %d %H:%M:%S %Y'


def _parse_time(text):
    import datetime
    try:
        return datetime.datetime.strptime(text.strip(), _TIME_FORMAT).timestamp()
    except ValueError:
        return nan


def trainer_rate_estimates(log_fname):
    """Every rate estimate a trainer printed, with the amount of data and the time it had.

    Parameters
    ----------
    log_fname : path-like
        A trainer log (``train*.log``).

    Returns
    -------
    dict of numpy arrays
        One entry per printed estimate (per system in a multi-system log):

        - ``system``: the system id ('' in a single-system log);
        - ``round``: the training round the estimate follows (0 before the first);
        - ``n_shots``: the number of shots the trainer had loaded (the count of its
          last progress line, or of its "loaded ... shot path(s)" line);
        - ``n_frames``: the frames the estimate counts;
        - ``time``: when the reweighting started (seconds since the epoch, from the log);
        - ``k12``, ``k21``: the estimates (1/frame);
        - ``k12_bias_reweighted``, ``k21_bias_reweighted``: the bias-reweighted ones (nan
          when not printed).

    Notes
    -----
    These are the trainer's online estimates, each with the network of its
    round, computed by the AIMMD version that ran the trainer: if AIMMD's
    reweighting has been corrected since, they are wrong. The rates to report
    come from :func:`reweight_run` with the current code (see "Assumptions
    and valid estimators" in the module docstring for what they rest on). For
    rates at a fraction of the data with the final network, use
    :func:`reweight_run` with ``fractions=``. Against the
    estimates of :func:`aimmd.analysis.utils.extract_rate_estimates_from_log_file`,
    this parser adds the number of shots and the time, and keeps the raw and
    bias-reweighted estimates aligned.
    """
    import re
    keys = ('system', 'round', 'n_shots', 'n_frames', 'time', 'k12', 'k21',
            'k12_bias_reweighted', 'k21_bias_reweighted')
    rows = []
    state = {'round': 0, 'n_shots': {}, 'time': nan}
    current = {}
    progress = re.compile(r'(\d+)(?:/\d+)?(?: paths)? \[[^\]\r\n]*paths')
    loaded = re.compile(r"\[system '([^']*)'\] loaded (\d+) shot path")
    # (a progress bar of the value pass can share the line: search, do not match)
    label = re.compile(r"(?:\[system '([^']*)'\] )?\b(k12|k21) (estimate|bias-reweighted):"
                       r"\s*([^\s\[]+)")
    frames_line = re.compile(r'(\d+) frames \(excluded margins\)')
    stamp = re.compile(r'\((\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\)')
    training = re.compile(r'Training the network \(round (\d+)')

    def finish(system):
        row = current.pop(system, None)
        if row is not None:
            rows.append(row)

    with open(log_fname, errors='replace') as handle:
        for line in handle:
            match = training.search(line)
            if match:
                state['round'] = int(match.group(1))
            for match in progress.finditer(line):
                state['n_shots'][''] = int(match.group(1))
            match = loaded.search(line)
            if match:
                state['n_shots'][match.group(1)] = int(match.group(2))
            if ('Reweighting the full path ensemble' in line or 'adaptation bins' in line):
                match = stamp.search(line)
                if match:
                    state['time'] = _parse_time(match.group(1))
            match = label.search(line)
            if match:
                system = match.group(1) or ''
                which, kind = match.group(2), match.group(3)
                try:
                    value = float(match.group(4))
                except ValueError:
                    value = nan
                if kind == 'estimate':
                    if which == 'k12':
                        finish(system)
                        current[system] = {
                            'system': system, 'round': state['round'],
                            'n_shots': state['n_shots'].get(system, state['n_shots'].get('', -1)),
                            'n_frames': -1, 'time': state['time'], 'k12': value, 'k21': nan,
                            'k12_bias_reweighted': nan, 'k21_bias_reweighted': nan}
                    elif system in current:
                        current[system]['k21'] = value
                elif system in current:
                    current[system][f'{which}_bias_reweighted'] = value
                continue
            match = frames_line.search(line)
            if match and '' in current:
                current['']['n_frames'] = int(match.group(1))
    for system in list(current):
        finish(system)
    out = {}
    for key in keys:
        column = [row[key] for row in rows]
        out[key] = (np.asarray(column, dtype=str) if key == 'system'
                    else np.asarray(column, dtype=float if key in (
                        'time', 'k12', 'k21', 'k12_bias_reweighted', 'k21_bias_reweighted')
                        else int))
    return out
