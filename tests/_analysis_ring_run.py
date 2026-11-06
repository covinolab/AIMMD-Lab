"""A synthetic AIMMD run folder with a known answer, and the real trainer on it.

The system is one particle on a ring, ``x`` in ``[0, 10)`` Å, with the
potential (in kT)

    U(x) = h (1 - cos(2 pi (x - 1) / 5)) / 2,   h = 1.5,

two wells at ``x = 1`` (state A, ``x < 2``) and ``x = 6`` (state B,
``5 <= x < 7``) and two barriers of 1.5 kT at ``x = 3.5`` and ``x = 8.5``: two
reaction channels through the reactive region R. Its equilibrium distribution
is ``exp(-U)``, so the free energy along ``x`` is ``U`` up to a constant: a
known answer for the reweighting.

The dynamics are Metropolis Monte Carlo (one move of standard deviation 0.5 Å
per frame), written frame by frame as AIMMD's workers write them: free
simulations in ``free{A,B}/traj000001.part####.xtc`` (part files that repeat
the previous part's last frame), two-way shots in ``chainR0/path######.xtc``
(the backward half reversed, then the forward half; the times run ``si, ...,
1, 0, 1, ...`` so that AIMMD finds the shooting frame), shooting points drawn
from the reactive frames of the free simulations. Every file gets its states
cache; descriptors and committor values are left to the trainer.

With ``biased=True`` the dynamics feel an extra potential inside A,

    V(x) = (1 + cos(pi (x - 1))) / 2   for x < 2   (0 elsewhere),

which is 1 kT at the bottom of the well and 0 at its edges, like a static
OPES bias that fills the well; the params record it (``record_bias``,
``bias_source='reader'``), so the trainer writes the bias caches, checks
that the bias is zero in R and prints bias-reweighted rates.

The params file's network is the exact committor of free diffusion on the
ring scaled by a parameter, ``logit = s (2 q(x) - 1)`` with ``q`` linear across
each channel; its fit sets ``s = 4`` and returns, so a trainer round costs
nothing but the value pass and the reweighting.

:func:`train_once` runs the trainer in this process for one round and
records, through wrappers that change nothing, the numbers it computed: the
weights of its last reweighting, and the result of its bias-reweighted rate
estimate.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np

HEIGHT = 1.5
"""Barrier height of the ring potential (kT)."""

STEP = 0.5
"""Standard deviation of a Monte Carlo move per frame (Å)."""

RING_PARAMS = '''
import numpy as np
import torch
from aimmd.network.rescalable import Rescalable

engine = 'toy'
initial_paths = 'initial.xtc'
toy_slowdown = 0.0
states = 'ARB'
free_restart_source = 'equilibrium'


def toy_mdrun(ts):
    ts.positions = (ts.positions + 0.5 * np.random.normal()) % 10


def ring_states(x):
    x = np.asarray(x, dtype=float) % 10
    out = np.full(len(x), 'R', dtype='<U1')
    out[x < 2] = 'A'
    out[(x >= 5) & (x < 7)] = 'B'
    return out


def states_function(trajectory):
    return ring_states([frame.positions[0, 0] for frame in trajectory])


def descriptors_function(trajectory):
    return np.array([[frame.positions[0, 0]] for frame in trajectory], dtype=np.float32)


class Network(Rescalable):
    """logit = scale (2 q - 1), q the committor of free diffusion across each channel."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)[:, :1] % 10
        q = torch.where(x < 6, (x - 2) / 3, (10 - x) / 3).clamp(0, 1)
        return self.scale * (2 * q - 1)

    def reset_parameters(self):
        with torch.no_grad():
            self.scale.fill_(1.0)


network = Network()


def fit(params, pathensemble, verbose=False, worker=None):
    """No learning: the scale of the analytic committor is set to 4."""
    with torch.no_grad():
        params.network.scale.fill_(4.0)
    return [0.0], {'network_metadata': {'fit': 'fixed scale', 'round': 1, 'scale': 4.0}}
'''

BIAS_PARAMS = '''
record_bias = True
bias_source = 'reader'
bias_reactive_threshold = 0.5


def bias_function(trajectory):
    x = np.array([frame.positions[0, 0] for frame in trajectory], dtype=float) % 10
    return np.where(x < 2, (1 + np.cos(np.pi * (x - 1))) / 2, 0.0)
'''


def ring_states(x):
    """The states of the ring: A ``x < 2``, B ``5 <= x < 7``, R elsewhere."""
    x = np.asarray(x, dtype=float) % 10
    out = np.full(len(x), 'R', dtype='<U1')
    out[x < 2] = 'A'
    out[(x >= 5) & (x < 7)] = 'B'
    return out


def potential(x):
    """U(x) in kT."""
    return HEIGHT * (1 - np.cos(2 * np.pi * (np.asarray(x, dtype=float) - 1) / 5)) / 2


def bias(x):
    """V(x) in kT: the fill of well A (zero outside A)."""
    x = np.asarray(x, dtype=float) % 10
    return np.where(x < 2, (1 + np.cos(np.pi * (x - 1))) / 2, 0.0)


def expected_free_energy(edges, *, with_bias=False):
    """``-ln`` of the integral of ``exp(-U)`` (or ``exp(-U - V)``) over each bin, minimum 0."""
    grid = np.linspace(0.0, 10.0, 200001)[:-1]
    energy = potential(grid) + (bias(grid) if with_bias else 0.0)
    weight = np.exp(-energy)
    edges = np.asarray(edges, dtype=float)
    which = np.digitize(grid, edges) - 1
    p = np.array([weight[which == b].sum() for b in range(len(edges) - 1)])
    p /= p.sum()
    f = -np.log(p)
    return f - f.min()


def metropolis(rng, x0, n, *, biased, stop=None, max_frames=20000):
    """``n`` frames of Metropolis dynamics from ``x0`` (or until ``stop(x)`` after the first)."""
    def energy(x):
        return float(potential(x) + (bias(x) if biased else 0.0))
    n = int(n) if stop is None else int(max_frames)
    xs = np.empty(n)
    x, e = float(x0), energy(x0)
    for i in range(n):
        xs[i] = x
        if stop is not None and i and stop(x):
            return xs[:i + 1]
        y = (x + STEP * rng.standard_normal()) % 10
        ey = energy(y)
        if ey <= e or rng.random() < np.exp(e - ey):
            x, e = y, ey
    return xs


def write_frames(fname, x, times):
    """An xtc of the one-atom ring (``x == y == z``) and its states cache."""
    import MDAnalysis as mda
    import aimmd
    from aimmd.cache.npy import save_npy
    from aimmd.path.utils import get_cache_fname
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe.empty(1, trajectory=True)
    with mda.Writer(str(fname), n_atoms=1) as writer:
        for xi, ti in zip(x, times):
            universe.trajectory.ts.positions = np.array([[xi, xi, xi]], dtype=np.float32)
            universe.trajectory.ts.time = float(ti)
            universe.trajectory.ts.dimensions = np.array([20, 20, 20, 90, 90, 90],
                                                         dtype=np.float32)
            writer.write(universe.atoms)
    stored = aimmd.Path(str(fname)).positions[:, 0, 0]          # at xtc precision
    save_npy(get_cache_fname(str(fname), 'states'), ring_states(stored))
    return str(fname)


def build_ring_run(folder, *, seed=0, n_free=6000, part=2000, n_shots=16, biased=False):
    """The params file, ``initial.xtc`` and ``run/`` (free simulations and shots) in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    (folder / 'params.py').write_text(RING_PARAMS + (BIAS_PARAMS if biased else ''))
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 6.0, 21)                    # an initial transition A -> B
    write_frames(folder / 'initial.xtc', x, np.arange(len(x), dtype=float))
    run = folder / 'run'
    pool = []
    for state, x0 in (('A', 1.0), ('B', 6.0)):
        x = metropolis(rng, x0, n_free, biased=biased)
        pool.extend(x[ring_states(x) == 'R'])
        start, k = 0, 1
        while start < n_free - 1:
            stop = min(start + part, n_free)
            write_frames(run / f'free{state}' / f'traj000001.part{k:04d}.xtc', x[start:stop],
                         np.arange(start, stop, dtype=float))
            start = stop - 1
            k += 1
    in_a_state = lambda y: ring_states([y])[0] != 'R'                # noqa: E731
    for n in range(1, n_shots + 1):
        shooting_point = pool[rng.integers(len(pool))]
        backward = metropolis(rng, shooting_point, None, biased=biased, stop=in_a_state)
        forward = metropolis(rng, shooting_point, None, biased=biased, stop=in_a_state)
        xs = np.concatenate([backward[::-1], forward[1:]])
        shooting_index = len(backward) - 1
        write_frames(run / 'chainR0' / f'path{n:06d}.xtc', xs,
                     np.abs(np.arange(len(xs), dtype=float) - shooting_index))
    return folder


def train_once(folder):
    """One trainer round in this process, in `folder`; what it computed and printed.

    Returns
    -------
    dict
        ``k12``, ``k21``: the trainer's rate estimates, recomputed exactly
        from the weights of its last reweighting (``1 / sum(w * n_frames)``,
        the trainer's own expression); ``bias``: the return value of its
        ``bias_reweighted_rates`` call (None without a bias); ``out``: what it
        printed.
    """
    import signal
    import aimmd
    from aimmd.pathensemble import PathEnsemble
    from aimmd.pathensemble import bias_utils
    calls, bias_calls = [], []
    reweight = PathEnsemble.reweight
    bias_reweighted_rates = bias_utils.bias_reweighted_rates

    def reweight_spy(self, *args, **kwargs):
        result = reweight(self, *args, **kwargs)
        calls.append((np.asarray(self.n_frames).copy(), np.asarray(result[0]).copy()))
        return result

    def bias_spy(*args, **kwargs):
        result = bias_reweighted_rates(*args, **kwargs)
        bias_calls.append(result)
        return result

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    cwd = os.getcwd()
    buffer = io.StringIO()
    try:
        os.chdir(folder)
        PathEnsemble.reweight = reweight_spy
        bias_utils.bias_reweighted_rates = bias_spy
        params = aimmd.Params.load('params.py', save=False)
        with contextlib.redirect_stdout(buffer):
            aimmd.Worker(params, 'run', walltime=600).train(nrounds=1)
    finally:
        PathEnsemble.reweight = reweight
        bias_utils.bias_reweighted_rates = bias_reweighted_rates
        os.chdir(cwd)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    (n1, w1), (n2, w2) = calls[-2:]
    k12, k21 = np.sum(w1 * n1), np.sum(w2 * n2)
    return {'k12': 1 / k12 if k12 else np.nan, 'k21': 1 / k21 if k21 else np.nan,
            'bias': bias_calls[-1] if bias_calls else None, 'out': buffer.getvalue()}


def snapshot(folder):
    """``{relative path: (size, mtime_ns)}`` of every file under `folder`.

    Two kinds of hidden files are left out, because AIMMD's reading machinery
    writes them, not the code under test: the file locks of its caches
    (``.<cache>.lock``), and MDAnalysis's frame index of a trajectory
    (``.<trajectory>_offsets.npz``), which AIMMD's reader cache deletes and
    rebuilds whenever it opens a trajectory (``aimmd.cache.mda``).
    """
    out = {}
    for directory, _, files in os.walk(folder):
        for name in files:
            if name.endswith('.lock') or (name.startswith('.') and name.endswith('_offsets.npz')):
                continue
            fname = os.path.join(directory, name)
            stat = os.stat(fname)
            out[os.path.relpath(fname, folder)] = (stat.st_size, stat.st_mtime_ns)
    return out
