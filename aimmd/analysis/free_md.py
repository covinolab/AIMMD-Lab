"""
aimmd.analysis.free_md
======================

What the free simulations of an AIMMD run say on their own, without the
reweighting and without the network: direct rate estimates from the
transitions they made, and a committor reference from where each of their
reactive frames went next.

Rates from transition counts
----------------------------
A free simulation that starts in state A spends some time with A as the last
end state it visited ("on the A side"), then perhaps reaches B, after which it
is on the B side, and so on. If it made ``n`` transitions from A to B during a
total time ``T`` on the A side, the rate estimate is ``k_AB = n / T``. The
transitions of a memoryless process with rate ``k`` in time ``T`` are Poisson
distributed with mean ``k T``, so the exact (Garwood) confidence interval of
``k`` follows from the chi-squared distribution: it is wide when ``n`` is small
and has a finite upper limit even for ``n = 0`` ("no transition in this much
time"). This estimate is model-free and is the sanity check of the
reweighted rates: a reweighted rate far above the upper limit of the free
simulations' rate, at comparable sampling, points to a problem with the
reweighting inputs.

It holds only if the free simulations sample the equilibrium dynamics. With
``free_restart_source='crossing'`` the workers restart a simulation that
reached the other state from a frame close to the boundary it crossed; the
time on each side then starts at the boundary instead of deep in the basin,
and the counted rate is too fast. :func:`free_md_rates` warns in that case.

With a static bias inside the states (``record_bias``), time on each side is
the boosted time ``sum exp(bias) dt``: every biased frame stands for
``exp(bias)`` frames of the unbiased dynamics.

A committor reference from free simulations
-------------------------------------------
Every reactive frame of an unbiased free simulation has a known forward
outcome: the end state the simulation reaches next. Averaged over many frames
at the same configuration this outcome is the committor (its probability of
reaching the last state first), so the outcomes are a reference for a
committor model that costs no extra simulation. Two cautions:

- the frames of one excursion (consecutive reactive frames between two state
  visits) share their outcome, so they are not independent;
  :attr:`FreeMDReference.groups` labels the excursions for grouped statistics;
- the forward outcome estimates the committor only when the velocities at the
  frame are distributed as in equilibrium, which holds for equilibrium free
  dynamics but only approximately just after a restart with fresh velocities.

The free simulations of a run that records a bias ran under that bias, and
frames of biased simulations are not used as a committor reference unless the
caller decides so (:func:`free_md_reference` refuses such a run unless
``allow_biased_frames=True``; its notes say when that can be justified).

Frames after the last state visit of a trajectory have no outcome yet
(censored) and are left out.
"""

# external
import os
import warnings
from dataclasses import dataclass, field
from math import nan
from typing import Optional

import numpy as np

from ._data_access import without_writing
from .reweighting import cache_rows


__all__ = ['poisson_rate_interval', 'transition_counts', 'free_md_rates', 'forward_outcomes',
           'FreeMDReference', 'free_md_reference', 'free_md_calibration']


# ════════════════════════════════════════════════════════════════════════════
# pure functions
# ════════════════════════════════════════════════════════════════════════════

def poisson_rate_interval(n_events, time, confidence=0.95):
    """Exact confidence interval of a Poisson rate from ``n_events`` in ``time``.

    Parameters
    ----------
    n_events : int
        Observed events (transitions).
    time : float
        Total observation time (any unit; the rate comes in its inverse).
    confidence : float, optional
        Two-sided confidence level (default 0.95).

    Returns
    -------
    tuple of float
        ``(rate, lower, upper)``: ``n / T`` and the Garwood limits
        ``chi2.ppf(a/2, 2n) / 2T`` and ``chi2.ppf(1 - a/2, 2n + 2) / 2T`` with
        ``a = 1 - confidence``. The lower limit is 0 for ``n = 0``. All nan for a
        time that is not positive.
    """
    from scipy.stats import chi2
    n = int(n_events)
    if n < 0:
        raise ValueError(f'a negative number of events: {n}')
    if not 0 < confidence < 1:
        raise ValueError(f'confidence must lie in (0, 1), got {confidence}')
    time = float(time)
    if not time > 0:
        return nan, nan, nan
    alpha = 1.0 - confidence
    lower = 0.0 if n == 0 else float(chi2.ppf(alpha / 2, 2 * n)) / (2 * time)
    upper = float(chi2.ppf(1 - alpha / 2, 2 * n + 2)) / (2 * time)
    return n / time, lower, upper


def transition_counts(labels, *, states='ARB', frame_time=None):
    """Transitions between the two end states and the time spent on each side, in one trajectory.

    Parameters
    ----------
    labels : array-like of str
        State labels of consecutive frames of one trajectory.
    states : str, optional
        The sorted states; ``states[0]`` and ``states[2]`` are the end states.
    frame_time : array-like of float, optional
        The time each frame stands for (default 1 per frame; ``exp(bias)``
        for a biased run).

    Returns
    -------
    dict
        ``{'time': {s: T_s}, 'transitions': {(s, o): n_so}, 'frames': {s: n_s}}``
        for the two end states ``s`` and ``o``: ``T_s`` sums the frame times
        of the frames whose last visited end state is ``s`` (frames before the
        first visit belong to no side), ``n_so`` counts the arrivals in ``o``
        on the ``s`` side, ``n_s`` the frames on the ``s`` side.
    """
    labels = np.asarray(labels).astype('<U1')
    first, last = states[0], states[2]
    weights = (np.ones(len(labels)) if frame_time is None
               else np.asarray(frame_time, dtype=float))
    if len(weights) != len(labels):
        raise ValueError(f'{len(weights)} frame times for {len(labels)} frames')
    side = np.full(len(labels), '', dtype='<U1')
    current = ''
    transitions = {(first, last): 0, (last, first): 0}
    for i, label in enumerate(labels):
        if label in (first, last):
            if current and label != current:
                transitions[(current, label)] += 1
            current = label
        side[i] = current
    # the arrival frame belongs to the new side: time on the s side ends one frame before
    time = {s: float(np.sum(weights[side == s])) for s in (first, last)}
    frames = {s: int(np.count_nonzero(side == s)) for s in (first, last)}
    return {'time': time, 'transitions': transitions, 'frames': frames}


def forward_outcomes(labels, *, states='ARB'):
    """The end state each frame of a trajectory reaches next, and its excursion.

    Parameters
    ----------
    labels : array-like of str
        State labels of consecutive frames of one trajectory.
    states : str, optional
        The sorted states.

    Returns
    -------
    tuple of numpy arrays
        ``(outcome, excursion)``. ``outcome``: per frame, 0 when the next end
        state reached (at this frame or later) is ``states[0]``, 1 when it is
        ``states[2]``, -1 when the trajectory ends first (censored). Frames in
        an end state have that state as their outcome. ``excursion``: per frame,
        the index of the run of consecutive non-end-state frames it belongs to
        (0, 1, ... in order), -1 for frames in an end state.
    """
    labels = np.asarray(labels).astype('<U1')
    first, last = states[0], states[2]
    n = len(labels)
    outcome = np.full(n, -1, dtype=int)
    upcoming = -1
    for i in range(n - 1, -1, -1):
        if labels[i] == first:
            upcoming = 0
        elif labels[i] == last:
            upcoming = 1
        outcome[i] = upcoming
    in_state = (labels == first) | (labels == last)
    excursion = np.full(n, -1, dtype=int)
    starts = np.flatnonzero(~in_state & np.concatenate([[True], in_state[:-1]]))
    for index, start in enumerate(starts):
        stop = start
        while stop < n and not in_state[stop]:
            stop += 1
        excursion[start:stop] = index
    return outcome, excursion


# ════════════════════════════════════════════════════════════════════════════
# run folders
# ════════════════════════════════════════════════════════════════════════════

def _known_labels(path):
    """The state labels of a free trajectory up to its first frame without a known state."""
    labels = np.asarray(path._get('states')).astype('<U1')
    unknown = np.flatnonzero(labels == '')
    return labels if not len(unknown) else labels[:unknown[0]]


def _locations(path, file_index, file_names):
    """Per frame of `path`: the index of its file in `file_names` (extended in place) and its
    frame in that file. Returns ``(file_names, file, frame)``."""
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
    if not files:
        return file_names, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return file_names, np.concatenate(files), np.concatenate(frames)


def _free_trajectories(params, run_dir, system_id):
    from .reweighting import _load_params, _system_directory
    params = _load_params(params)
    run_dir = os.path.abspath(str(run_dir))
    if getattr(params, 'multi_system', False):
        if system_id is None:
            raise ValueError('a multi-system run: name the system (system_id=)')
        run_dir = _system_directory(params, run_dir, system_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'no run folder {run_dir!r}')
    return params, run_dir, params.free_trajectories(run_dir)


def _warn_restart_source(source):
    if source == 'crossing':
        warnings.warn(
            "params.free_restart_source is 'crossing': free simulations that reached the other "
            "state were restarted from a frame near the boundary they crossed, so they are not "
            "equilibrium dynamics and their transition counts overestimate the rates",
            UserWarning, stacklevel=3)


@without_writing
def free_md_rates(params, run_dir, *, bias='auto', confidence=0.95, system_id=None):
    """Rates counted directly in the free simulations of a run, with exact Poisson limits.

    Parameters
    ----------
    params : aimmd.Params or path-like
        The run's params.
    run_dir : path-like
        The run folder (read only).
    bias : {'auto', True, False}, optional
        Count boosted time ``sum exp(bias)`` per frame (``'auto'``: when
        ``params.record_bias`` is set). The bias comes from the trainer's
        caches; frames without one count with ``exp(0) = 1``.
    confidence : float, optional
        Level of the two-sided interval (default 0.95).
    system_id : str, optional
        The system of a multi-system run.

    Returns
    -------
    dict
        Per direction (key ``'AB'`` for ``states[0] -> states[2]``, ``'BA'``
        back): ``transitions``, ``time`` (frames, boosted when the bias is
        applied), ``frames`` (frames on the starting side), ``rate``,
        ``lower``, ``upper`` (1/frame). Also ``confidence``,
        ``bias_applied``, ``n_trajectories``, ``frames_without_state``
        (frames at the ends of trajectories whose state is not known yet,
        left out), ``frames_without_bias`` and ``free_restart_source``.

    Warns
    -----
    UserWarning
        When ``params.free_restart_source`` is ``'crossing'`` (see the module docstring).
    """
    params, run_dir, free = _free_trajectories(params, run_dir, system_id)
    if bias not in ('auto', True, False):
        raise ValueError(f"bias must be 'auto', True or False, got {bias!r}")
    record_bias = bool(getattr(params, 'record_bias', False))
    if bias is True and not record_bias:
        raise ValueError('bias=True, but the run records no bias (params.record_bias is False)')
    bias_applied = record_bias if bias == 'auto' else bool(bias)
    states = params.sorted_states
    first, last = states[0], states[2]
    source = getattr(params, 'free_restart_source', None)
    _warn_restart_source(source)
    time = {first: 0.0, last: 0.0}
    frames = {first: 0, last: 0}
    transitions = {(first, last): 0, (last, first): 0}
    unknown = 0
    without_bias = 0
    for trajectory in free:
        labels = _known_labels(trajectory)
        unknown += len(trajectory) - len(labels)
        frame_time = None
        if bias_applied:
            names, files, frames_in_file = _locations(trajectory, {}, [])
            values, missing = cache_rows(names, files[:len(labels)],
                                          frames_in_file[:len(labels)], 'bias')
            without_bias += int(missing.sum())
            frame_time = np.exp(values)
        counts = transition_counts(labels, states=states, frame_time=frame_time)
        for s in (first, last):
            time[s] += counts['time'][s]
            frames[s] += counts['frames'][s]
        for key in transitions:
            transitions[key] += counts['transitions'][key]
    out = {'confidence': confidence, 'bias_applied': bias_applied,
           'n_trajectories': len(free), 'frames_without_state': int(unknown),
           'frames_without_bias': int(without_bias), 'free_restart_source': source,
           'unit': '1/frame'}
    for s, o in ((first, last), (last, first)):
        rate, lower, upper = poisson_rate_interval(transitions[(s, o)], time[s], confidence)
        out[s + o] = {'transitions': transitions[(s, o)], 'time': time[s], 'frames': frames[s],
                      'rate': rate, 'lower': lower, 'upper': upper}
    return out


@dataclass
class FreeMDReference:
    """Reactive frames of free simulations with their forward outcomes (a committor reference)."""

    name: str
    """A short name for reports."""

    files: list
    """The trajectory files."""

    file: np.ndarray
    """Per frame: index into :attr:`files`."""

    frame: np.ndarray
    """Per frame: index in its file."""

    trajectory: np.ndarray
    """Per frame: index of its free trajectory."""

    n_first: np.ndarray
    """Per frame: 1 when the next end state reached is ``states[0]``, else 0."""

    n_last: np.ndarray
    """Per frame: 1 when the next end state reached is ``states[2]``, else 0."""

    groups: np.ndarray
    """Per frame: the excursion it belongs to (unique over the whole reference); frames of
    one excursion share their outcome."""

    values: np.ndarray
    """Per frame: the cached committor value (logit) of the run's network, nan where there
    is none."""

    states: str = 'ARB'
    """The sorted states."""

    descriptors: Optional[np.ndarray] = None
    """Per frame: the row of the run's descriptors series (when asked for)."""

    notes: dict = field(default_factory=dict)
    """Provenance: run folder, ``free_restart_source``, censored frames, stride."""

    kind: str = 'free_md_forward'
    """What the counts are: one forward outcome per frame of free dynamics."""

    def __len__(self):
        return int(len(self.n_last))


@without_writing
def free_md_reference(params, run_dir, *, stride=1, descriptors=False, system_id=None,
                      name='free_md', allow_biased_frames=False):
    """The reactive frames of a run's free simulations with the end state each reached next.

    Parameters
    ----------
    params : aimmd.Params or path-like
        The run's params.
    run_dir : path-like
        The run folder (read only).
    stride : int, optional
        Keep every ``stride``-th reactive frame of each trajectory (default 1: all).
    descriptors : bool, optional
        Also load each frame's row of the run's descriptors series (default False).
    system_id : str, optional
        The system of a multi-system run.
    name : str, optional
        :attr:`FreeMDReference.name`.
    allow_biased_frames : bool, optional
        Use the free simulations of a run that records a bias (default False:
        such a run is refused; see the Notes).

    Returns
    -------
    FreeMDReference

    Raises
    ------
    ValueError
        For a run that records a bias (``params.record_bias``), unless
        `allow_biased_frames`.

    Warns
    -----
    UserWarning
        When ``params.free_restart_source`` is ``'crossing'``: restarts near the
        boundary make the forward outcomes of the frames right after them
        unreliable as committor estimates.

    Notes
    -----
    Frames of biased simulations are not used as a committor reference by
    default. What can justify an exception: in a run with a static bias
    inside the states, the outcomes still estimate the committor of the
    unbiased system as long as the bias is zero in the reactive region (the
    trainer checks this every round), because the outcome of a reactive frame
    is decided by the dynamics from that frame until the next entry into a
    state, which the bias does not touch. The reference's
    ``notes['record_bias']`` says whether the run recorded a bias.
    """
    from ._data_access import refuse_biased_frames
    from .reweighting import _load_cache
    params, run_dir, free = _free_trajectories(params, run_dir, system_id)
    refuse_biased_frames(params, 'a committor reference', allow_biased_frames)
    stride = int(stride)
    if stride < 1:
        raise ValueError(f'stride must be a positive integer, got {stride}')
    states = params.sorted_states
    reactive = states[1]
    source = getattr(params, 'free_restart_source', None)
    _warn_restart_source(source)
    series = getattr(params, 'descriptors_series', 'descriptors') or 'descriptors'
    file_index, file_names = {}, []
    columns = {key: [] for key in ('file', 'frame', 'trajectory', 'outcome', 'groups',
                                   'values')}
    rows = []
    censored = 0
    excursion_offset = 0
    for t, trajectory in enumerate(free):
        labels = _known_labels(trajectory)
        n = len(labels)
        outcome, excursion = forward_outcomes(labels, states=states)
        _, files, frames = _locations(trajectory, file_index, file_names)
        files, frames = files[:n], frames[:n]
        reactive_frames = np.flatnonzero(labels == reactive)
        censored += int(np.count_nonzero(outcome[reactive_frames] < 0))
        keep = reactive_frames[outcome[reactive_frames] >= 0][::stride]
        values, missing = cache_rows(file_names, files[keep], frames[keep], 'values')
        values[missing] = np.nan
        if descriptors:
            block = None
            for f in np.unique(files[keep]):
                data = _load_cache(file_names[f], series)
                if data is None:
                    raise FileNotFoundError(f'no {series!r} cache for {file_names[f]!r}')
                data = data.reshape(len(data), -1)
                where = files[keep] == f
                if block is None:
                    block = np.full((len(keep), data.shape[1]), np.nan)
                locs = frames[keep][where]
                ok = locs < len(data)
                block[np.flatnonzero(where)[ok]] = data[locs[ok]]
            if block is not None:
                rows.append(block)
        columns['file'].append(files[keep])
        columns['frame'].append(frames[keep])
        columns['trajectory'].append(np.full(len(keep), t, dtype=int))
        columns['outcome'].append(outcome[keep])
        columns['groups'].append(excursion[keep] + excursion_offset)
        columns['values'].append(values)
        excursion_offset += int(excursion.max() + 1) if len(excursion) else 0
    joined = {key: (np.concatenate(parts) if parts else np.zeros(0, dtype=float if key ==
                                                                  'values' else int))
              for key, parts in columns.items()}
    descriptor_rows = None
    if descriptors:
        descriptor_rows = (np.concatenate(rows, axis=0) if rows
                           else np.zeros((0, 0), dtype=float))
    outcome = joined['outcome']
    return FreeMDReference(
        name=name, files=file_names, file=joined['file'], frame=joined['frame'],
        trajectory=joined['trajectory'], n_first=(outcome == 0).astype(int),
        n_last=(outcome == 1).astype(int), groups=joined['groups'], values=joined['values'],
        states=states, descriptors=descriptor_rows,
        notes={'run_dir': run_dir, 'free_restart_source': source,
               'censored_frames': censored, 'stride': stride, 'descriptors_series': series,
               'n_trajectories': len(free),
               'record_bias': bool(getattr(params, 'record_bias', False))})


def free_md_calibration(reference, logits=None, *, bins=None):
    """Predicted against observed outcomes of a free-simulation reference, per committor bin.

    Parameters
    ----------
    reference : FreeMDReference
        From :func:`free_md_reference`.
    logits : array-like, optional
        Committor logits (toward ``states[2]``) of the reference frames
        (default: :attr:`FreeMDReference.values`, the run's cached values).
    bins : array-like, optional
        Bin edges on the logit scale (default: 12 edges from -6 to 6, with
        open outer bins).

    Returns
    -------
    list of dict
        Per bin: ``low``, ``high`` (logit edges), ``frames``, ``excursions``
        (distinct excursions with frames in the bin), ``predicted`` (the sum
        of the predicted committors: the number of frames expected to reach
        ``states[2]`` first), ``observed`` (frames that did), and
        ``observed_excursions`` (excursions with frames in the bin that
        reached ``states[2]``). A calibrated model has ``predicted`` close to
        ``observed``; because frames of one excursion share their outcome,
        judge the difference on the scale of the number of excursions, not of
        frames.
    """
    logits = reference.values if logits is None else np.asarray(logits, dtype=float)
    if len(logits) != len(reference):
        raise ValueError(f'{len(logits)} logits for {len(reference)} frames')
    if bins is None:
        bins = np.concatenate([[-np.inf], np.linspace(-6, 6, 12), [np.inf]])
    bins = np.asarray(bins, dtype=float)
    finite = np.isfinite(logits)
    p = np.zeros(len(logits))
    p[finite] = 1.0 / (1.0 + np.exp(-logits[finite]))
    which = np.digitize(logits, bins) - 1
    out = []
    for b in range(len(bins) - 1):
        mask = finite & (which == b)
        groups = reference.groups[mask]
        last = reference.n_last[mask] > 0
        out.append({'low': float(bins[b]), 'high': float(bins[b + 1]),
                    'frames': int(mask.sum()), 'excursions': int(len(np.unique(groups))),
                    'predicted': float(p[mask].sum()), 'observed': int(last.sum()),
                    'observed_excursions': int(len(np.unique(groups[last])))})
    return out
