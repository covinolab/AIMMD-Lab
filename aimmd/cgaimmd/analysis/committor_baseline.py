"""
aimmd.cgaimmd.analysis.committor_baseline
=========================================

A plain committor model to compare a C-GAIMMD model with: a multilayer
perceptron on the distances between selected atoms, trained by AIMMD's own fit
on a run's own shooting data, after the run.

Why a baseline
--------------
A C-GAIMMD model learns its committor from the shots it generated. Whether
that committor is good is best judged next to an ordinary AIMMD committor
model trained on the *same* shots: if both score alike on a brute-force set,
the generative part costs nothing in committor accuracy. The baseline here is
deliberately simple and system-independent:

- **inputs**: the distances between all pairs of a set of atoms (the heavy
  atoms of a small molecule, or the binding pocket and the ligand of a
  protein–ligand complex; a full protein–ligand matrix has over a million
  pairs, so the atom selection is an argument). Distances do not change when
  the molecule is moved or rotated, so the model needs no alignment;
- **scaling**: each distance is shifted and scaled by its mean and standard
  deviation over the frames of the run's free simulations
  (:func:`run_distance_scaler`). The scaler is stored in the network, so it
  travels with the saved weights;
- **the network**: a stack of linear layers with SiLU activations
  (:class:`DistanceMLP`), an AIMMD network
  (:class:`aimmd.network.Rescalable`) whose ``forward`` gives the committor
  logit toward the last end state;
- **the data and the fit**: the run's own descriptor caches, which hold the
  flat Cartesian coordinates of the atoms (as in every C-GAIMMD run), and
  AIMMD's ordinary fit (:func:`aimmd.network.fit.fit`) on the run's shots,
  as the trainer would run it (:func:`train_on_run`). Nothing is written into
  the run folder; the caller saves the trained weights where it wants.

Runs with a bias
----------------
A run that records a bias (``params.record_bias``, e.g. a static OPES bias
inside the states) ran its free simulations and shots under that bias. A bias
changes which configurations are visited and how the system moves between
them, so frames of biased simulations do not become training data or an input
scaler unless the caller decides so: :func:`run_distance_scaler` and
:func:`train_on_run` refuse such a run unless ``allow_biased_frames=True``.
What can justify it: a static bias that is zero in the reactive region (the
trainer checks this every round) leaves the dynamics there untouched, so the
outcome of every shot is that of the unbiased system and AIMMD's own trainer
trains on the same shots; the scaler then weights every frame by ``exp(bias)``
so that it describes the unbiased system.

Score it on a brute-force reference with
:func:`aimmd.cgaimmd.analysis.validate.network_committor` (on the coordinates
the run's descriptors hold) or
:func:`~aimmd.cgaimmd.analysis.validate.params_committor` with
``descriptor_transform=None`` (on whole configurations, through the run's
descriptors function), next to the C-GAIMMD model.

>>> from aimmd.cgaimmd.analysis import committor_baseline as baseline
>>> heavy = [i for i, z in enumerate(elements) if z > 1]                 # doctest: +SKIP
>>> scaler = baseline.run_distance_scaler(params, run_dir, heavy)       # doctest: +SKIP
>>> mlp = baseline.DistanceMLP(heavy, n_atoms=len(elements), mean=scaler['mean'],  # doctest: +SKIP
...                            std=scaler['std'])
>>> result = baseline.train_on_run(params, run_dir, mlp, epochs=500)    # doctest: +SKIP
>>> torch.save(mlp.state_dict(), 'analysis/distance_mlp.pt')            # doctest: +SKIP
"""

import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ...network.rescalable import Rescalable

__all__ = ['DistanceMLP', 'pair_distances', 'distance_scaler', 'run_distance_scaler',
           'train_on_run']

MIN_STD = 1e-3
"""Smallest standard deviation of a scaled distance (Å): a pair that never moves is not blown up."""


def _pairs(atom_indices):
    idx = np.asarray(atom_indices, dtype=int).reshape(-1)
    if len(idx) < 2:
        raise ValueError('pair distances need at least two atoms')
    if len(np.unique(idx)) != len(idx):
        raise ValueError('the atom selection lists an atom twice')
    first, second = np.triu_indices(len(idx), 1)
    return idx, idx[first], idx[second]


def pair_distances(coordinates, atom_indices):
    """The distances between all pairs of the selected atoms.

    Parameters
    ----------
    coordinates : array_like
        ``(n, M, 3)`` or flat ``(n, 3 M)`` coordinates in Å.
    atom_indices : array_like of int
        The selected atoms (indices into the ``M`` atoms).

    Returns
    -------
    numpy.ndarray
        ``(n, k (k - 1) / 2)`` float64 distances for ``k`` selected atoms,
        pairs ``i < j`` in the order of `atom_indices` (the order of
        :class:`DistanceMLP`).
    """
    _, i, j = _pairs(atom_indices)
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim == 2:
        x = x.reshape(len(x), -1, 3)
    return np.linalg.norm(x[:, i] - x[:, j], axis=-1)


def distance_scaler(coordinates, atom_indices, *, weights=None, min_std=MIN_STD):
    """Mean and standard deviation of every pair distance over a set of frames.

    Parameters
    ----------
    coordinates : array_like
        ``(n, M, 3)`` or ``(n, 3 M)`` Å frames.
    atom_indices : array_like of int
        The selected atoms.
    weights : array_like, optional
        ``(n,)`` frame weights (e.g. ``exp(bias)`` of the frames of a
        simulation under a bias, so that the scale describes the unbiased
        system).
    min_std : float, optional
        Floor of the standard deviation.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, std)``, one entry per pair.
    """
    d = pair_distances(coordinates, atom_indices)
    if not len(d):
        raise ValueError('no frames to scale the distances with')
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (len(d),) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError('weights must be one non-negative value per frame, not all zero')
    w = w / w.sum()
    mean = w @ d
    std = np.sqrt(np.maximum(w @ (d - mean) ** 2, 0.0))
    return mean, np.maximum(std, float(min_std))


class DistanceMLP(Rescalable):
    """Committor logit from the scaled distances between selected atoms.

    ``logit = MLP((d_ij(x) - mean_ij) / std_ij)`` over the pairs ``i < j`` of
    `atom_indices`, toward the last end state (``sorted_states[2]``).

    Parameters
    ----------
    atom_indices : array_like of int
        The selected atoms (indices into the ``n_atoms`` atoms of the input).
    n_atoms : int
        Atoms of an input frame (the input is ``(B, 3 n_atoms)`` or ``(B, n_atoms, 3)`` Å).
    hidden : sequence of int, optional
        Widths of the hidden layers (default 128, 128, 64).
    mean, std : array_like, optional
        The scaler (:func:`distance_scaler`, :func:`run_distance_scaler`);
        default 0 and 1 (no scaling).

    Notes
    -----
    The selection and the scaler are buffers: they are saved with the
    weights (``state_dict``) and moved with the network. ``reset_parameters``
    re-initialises the linear layers in place and keeps the scaler, as AIMMD's
    fit expects (it resets the network at the start of every round).
    """

    def __init__(self, atom_indices, n_atoms, hidden=(128, 128, 64), mean=None, std=None):
        super().__init__()
        idx, i, j = _pairs(atom_indices)
        self.n_atoms = int(n_atoms)
        if idx.max() >= self.n_atoms or idx.min() < 0:
            raise ValueError(f'atom indices must lie in [0, {self.n_atoms}), got {idx.min()} to '
                             f'{idx.max()}')
        self.hidden_widths = tuple(int(h) for h in hidden)
        n_pairs = len(i)
        self.register_buffer('atom_indices', torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer('pair_first', torch.as_tensor(i, dtype=torch.long))
        self.register_buffer('pair_second', torch.as_tensor(j, dtype=torch.long))
        self.register_buffer('mean', torch.as_tensor(
            np.zeros(n_pairs) if mean is None else np.asarray(mean, dtype=float).reshape(-1),
            dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(
            np.ones(n_pairs) if std is None else np.asarray(std, dtype=float).reshape(-1),
            dtype=torch.float32))
        if self.mean.shape != (n_pairs,) or self.std.shape != (n_pairs,):
            raise ValueError(f'the scaler must have {n_pairs} entries (one per pair), got '
                             f'{tuple(self.mean.shape)} and {tuple(self.std.shape)}')
        layers = []
        width = n_pairs
        for h in self.hidden_widths:
            layers += [nn.Linear(width, h), nn.SiLU()]
            width = h
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def features(self, x):
        """``(B, P)`` scaled pair distances of a ``(B, 3 n_atoms)`` or ``(B, n_atoms, 3)`` input."""
        x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.mean.device)
        x = x.reshape(len(x), self.n_atoms, 3)
        d = torch.linalg.norm(x[:, self.pair_first] - x[:, self.pair_second], dim=-1)
        return (d - self.mean) / self.std

    def forward(self, x):
        """``(B, 1)`` committor logits."""
        return self.output(self.layers(self.features(x)))

    def reset_parameters(self):
        """Re-initialise every linear layer in place (the scaler is kept)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def settings(self):
        """The architecture, as the run card records a network's settings."""
        return {'n_atoms': self.n_atoms, 'n_selected_atoms': int(len(self.atom_indices)),
                'n_pairs': int(len(self.pair_first)), 'hidden': list(self.hidden_widths),
                'activation': 'SiLU'}


def run_distance_scaler(params, run_dir, atom_indices, *, max_frames=20000, bias='auto',
                        series=None, min_std=MIN_STD, allow_biased_frames=False):
    """The distance scaler from the free simulations of a run.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (its free trajectories, descriptor series and bias settings).
    run_dir : str or os.PathLike
        The run folder (read only).
    atom_indices : array_like of int
        The selected atoms (indices into the atoms of a descriptor row, which
        must hold flat Cartesian coordinates).
    max_frames : int, optional
        At most this many frames, taken evenly over all free-simulation
        frames.
    bias : {'auto', True, False}, optional
        For a run with a bias (and `allow_biased_frames`): weight the frames
        by ``exp(bias)`` from the run's bias caches (``'auto'``: when the run
        records a bias), so that the scale is that of the unbiased system.
    series : str, optional
        The descriptor series (default ``params.descriptors_series``).
    min_std : float, optional
        Floor of the standard deviation.
    allow_biased_frames : bool, optional
        Use the free simulations of a run that records a bias (default False:
        such a run is refused; see "Runs with a bias" in the module docstring).

    Returns
    -------
    dict
        ``mean``, ``std`` (one per pair), ``n_frames``, ``files``,
        ``bias_applied``, ``series``.

    Raises
    ------
    ValueError
        For a run that records a bias, unless `allow_biased_frames`.

    Notes
    -----
    Only the free simulations are used: they sample the equilibrium
    distribution (in the states and, rarely, the reactive region), while
    shots and initial paths are selected configurations. Frames are read from
    the descriptor caches with numpy (nothing written into the run folder).
    Frames that the bias cache does not cover get weight 1.
    """
    from ...analysis._data_access import refuse_biased_frames
    from ...path.utils import get_cache_fname
    refuse_biased_frames(params, 'an input scaler', allow_biased_frames)
    series = series or getattr(params, 'descriptors_series', None) or 'descriptors'
    record_bias = bool(getattr(params, 'record_bias', False))
    applied = record_bias if bias == 'auto' else bool(bias)
    free = params.free_trajectories(os.fspath(run_dir))
    files = []
    for trajectory in free:
        for fname in trajectory.fnames:
            if fname not in files:
                files.append(str(fname))
    rows, weights = [], []
    for fname in files:
        try:
            data = np.load(get_cache_fname(fname, series), allow_pickle=False)
        except (OSError, ValueError):
            continue
        data = np.asarray(data, dtype=np.float64).reshape(len(data), -1)
        w = np.ones(len(data))
        if applied:
            try:
                b = np.asarray(np.load(get_cache_fname(fname, 'bias'), allow_pickle=False),
                               dtype=float).reshape(-1)
                n = min(len(b), len(w))
                w[:n] = np.exp(b[:n])
            except (OSError, ValueError):
                pass
        rows.append(data)
        weights.append(w)
    if not rows:
        raise FileNotFoundError(f'no {series!r} caches of free simulations in {run_dir!r}')
    rows = np.concatenate(rows)
    weights = np.concatenate(weights)
    if len(rows) > int(max_frames):
        pick = np.unique(np.linspace(0, len(rows) - 1, int(max_frames)).round().astype(int))
        rows, weights = rows[pick], weights[pick]
    if rows.shape[1] % 3:
        raise ValueError(f'the {series!r} rows of width {rows.shape[1]} are not flat coordinates')
    mean, std = distance_scaler(rows, atom_indices, weights=weights, min_std=min_std)
    return {'mean': mean, 'std': std, 'n_frames': int(len(rows)), 'files': files,
            'bias_applied': applied, 'series': series}


def train_on_run(params, run_dir, network, *, chains=None, descriptor_transform=None,
                 allow_biased_frames=False, **fit_kwargs):
    """Train a network on the shooting data of a run with AIMMD's own fit, after the run.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (its states, descriptors function and series).
    run_dir : str or os.PathLike
        The run folder (read only).
    network : torch.nn.Module
        The network to train (e.g. a :class:`DistanceMLP`); trained in place,
        from ``network.reset_parameters()``.
    chains : sequence of int, optional
        The reactive shooting chains to train on (default: all; leave some out
        to hold them out).
    descriptor_transform : callable, optional
        Applied to the descriptor rows before the network (default None: the
        rows themselves, the flat coordinates a :class:`DistanceMLP` reads).
    allow_biased_frames : bool, optional
        Train on a run that records a bias (default False: such a run is
        refused; see "Runs with a bias" in the module docstring).
    **fit_kwargs
        Settings of :func:`aimmd.network.fit.fit` (``epochs``, ``lr``,
        ``batch_size``, ``augment``, early stopping, ...).

    Returns
    -------
    dict
        ``losses`` (one per epoch), ``n_shots``, ``n_free_trajectories``,
        ``chains``, ``fit_settings``.

    Raises
    ------
    ValueError
        For a run that records a bias, unless `allow_biased_frames`.

    Notes
    -----
    The path ensemble is the trainer's (the shots of the chains and the free
    simulations, which hold the end-state frames the fit needs); AIMMD's fit
    builds its training set from the shooting outcomes (and, with
    ``augment``, from the free simulations), reading the run's descriptor
    caches. Only `network` changes; save its weights where the analysis keeps
    its results.
    """
    from ...analysis._data_access import refuse_biased_frames
    from ...network.fit import fit
    from ...pathensemble.utils import assemble_pathensemble
    refuse_biased_frames(params, 'training data', allow_biased_frames)
    run_dir = os.fspath(run_dir)
    shot_chains = params.shot_chains(run_dir, None, None if chains is None else list(chains))
    trajectories = params.free_trajectories(run_dir)
    ensemble = assemble_pathensemble(shot_chains, trajectories)
    view = SimpleNamespace(network=network, sorted_states=params.sorted_states,
                           descriptors_function=params.descriptors_function,
                           descriptors_series=getattr(params, 'descriptors_series', None)
                           or 'descriptors',
                           descriptor_transform=descriptor_transform)
    losses, *_ = fit(view, ensemble, **fit_kwargs)
    n_shots = sum(len(chain) for chain in shot_chains)
    return {'losses': [float(v) for v in losses], 'n_shots': int(n_shots),
            'n_free_trajectories': len(trajectories),
            'chains': None if chains is None else [int(c) for c in chains],
            'fit_settings': {key: value for key, value in fit_kwargs.items()
                             if isinstance(value, (bool, int, float, str, type(None)))}}
