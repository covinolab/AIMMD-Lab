"""
aimmd.cgaimmd.lambda_law
========================

Target committor levels of C-GAIMMD shooting.

Every refill of a C-GAIMMD shooting worker's candidate pool starts by drawing
``cgaimmd_parameters['n_targets']`` target levels :math:`\\lambda^*` (logit
committors toward ``params.sorted_states[2]``) at which the generator is
asked for configurations. Where the levels come from is the *law*:

- ``'uniform'`` (the default): uniform on the lambda window;
- ``'bins'``: AIMMD's own shooting-point law without a selection pool: an
  adaptive bin of the trainer (restricted to the window) with weight
  inversely proportional to its selection density, with the Lorentzian and
  density adjustments of RFPS selection
  (:func:`aimmd.worker.utils.select_shooting_point`, with the shared
  density step taken from the same helpers), then uniform inside the bin;
- ``params.cgaimmd_lambda_sampler``: a function of the params file, which
  replaces the law.

The *window* (``cgaimmd_parameters['lambda_window']``) is a ``(lo, hi)`` pair or
``'bins'``, the trainer's shooting-value cutoffs, and is always clipped to
the network's conditioning range ``[-L, L]`` (:func:`resolve_window`).
Before the trainer has written its first bins, both ``'bins'`` settings fall
back to uniform draws, so C-GAIMMD workers never wait for bins.

The law is a choice over lambda only: generated shots are book-kept and
reweighted exactly like RFPS shots, whatever the law. The per-shot sidecar
records the law and its inputs, so that it can be studied later.

numpy and light AIMMD modules only at import; the density helpers of
:mod:`aimmd.worker.utils` and the path classes are imported where they are
used. The default window is ``CGAIMMD_PARAMETER_DEFAULTS['lambda_window']``
(:mod:`aimmd.params.utils`); this module keeps no default of its own.
"""

import math
from numbers import Integral

import numpy as np

from ..params.utils import CGAIMMD_PARAMETER_DEFAULTS
from ._once import once
from .protocol import lambda_clip_of


LAWS = ('uniform', 'bins')
"""The values of ``cgaimmd_parameters['lambda_law']``."""

BINS_LAW_AIMMD_ADJUSTMENTS = True
"""Whether the ``'bins'`` law applies RFPS's Lorentzian and density adjustments.

This is the one place where the choice can be flipped: True applies RFPS's
adjustments (the AIMMD-consistent law), False gives the plain inverse
projected density. Production used the default law ``'uniform'``, which does
not read this flag.
"""

EMPTY_DENSITY = 1e-12
"""Projected bin densities at or below this value count as empty (the trainer stores 1e-15)."""


def _positive(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{what} must be a finite float > 0, got {value!r}') from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{what} must be a finite float > 0, got {value!r}')
    return number


def _as_bins(bins):
    """float64 edges, or None when there are no usable bins (None, empty, one edge)."""
    if bins is None:
        return None
    edges = np.asarray(bins, dtype=float).reshape(-1)
    return edges if len(edges) >= 2 else None


def resolve_window(settings, L, bins, *, reweight_parameters=None):
    """The lambda window of C-GAIMMD shooting.

    Parameters
    ----------
    settings : mapping
        C-GAIMMD settings (``params.cgaimmd_settings``); ``settings['lambda_window']`` is read.
    L : float
        The conditioning range (:func:`aimmd.cgaimmd.protocol.lambda_clip_of`).
    bins : array-like or None
        The trainer's current adaptive bin edges; None (or empty) before it has written any.
    reweight_parameters : dict, optional
        ``params.reweight_parameters``: its ``sp_cutoff_min``/``sp_cutoff_max`` win, as in the trainer.

    Returns
    -------
    tuple
        ``((lo, hi), record)`` with ``lo < hi`` floats and a JSON-serializable record.

    Raises
    ------
    ValueError
        If the window is empty after clipping to ``[-L, L]``, or `L` is not a finite float > 0.

    Notes
    -----
    A ``(lo, hi)`` setting is clipped to ``[-L, L]``. ``'bins'`` is the
    trainer's shooting-value cutoffs ``(sp_cutoff_min, sp_cutoff_max)``,
    computed by the trainer's own helper
    :func:`aimmd.worker.utils.reweight_parameters_with_bins` (``bins[0]``, or
    ``bins[1]`` when ``bins[0]`` is ``-inf``; mirrored at the top; user
    values win), clipped to ``[-L, L]``; without bins it is ``[-L, L]``,
    also when `reweight_parameters` sets ``sp_cutoff_min``/``sp_cutoff_max``,
    although the trainer applies those without bins.
    The trainer's reweighting treats shots whose shooting value lies outside
    those cutoffs as free excursions, so ``'bins'`` keeps the targets inside
    them.

    The record is ``{'setting', 'window', 'source', 'lambda_clip'}`` with
    ``setting`` the setting (``'bins'`` or a list), ``window`` ``[lo, hi]``
    and ``source`` one of ``'setting'``, ``'bins'`` and ``'bins (no bins
    yet)'``.
    """
    L = _positive(L, 'the lambda clip L')
    setting = settings.get('lambda_window', CGAIMMD_PARAMETER_DEFAULTS['lambda_window'])
    if isinstance(setting, str):
        if setting != 'bins':
            raise ValueError(f"lambda_window must be a (lo, hi) pair or 'bins', got {setting!r}")
        edges = _as_bins(bins)
        if edges is None:
            lo, hi, source = -L, L, 'bins (no bins yet)'
        else:
            from ..worker.utils import reweight_parameters_with_bins
            cutoffs = reweight_parameters_with_bins(dict(reweight_parameters or {}), edges)
            lo, hi = float(cutoffs['sp_cutoff_min']), float(cutoffs['sp_cutoff_max'])
            source = 'bins'
        recorded = 'bins'
    else:
        lo, hi = (float(value) for value in setting)
        source = 'setting'
        recorded = [lo, hi]
    lo, hi = max(lo, -L), min(hi, L)
    if not lo < hi:
        raise ValueError(f'the lambda window {recorded!r} ({source}) is empty after clipping to '
                         f'[-L, L] = [{-L}, {L}]')
    return (lo, hi), {'setting': recorded, 'window': [lo, hi], 'source': source,
                      'lambda_clip': L}


def finite_window_bins(bins, window):
    """The adaptive bins restricted to a finite window.

    Parameters
    ----------
    bins : array-like
        Bin edges (the outer ones may be infinite).
    window : tuple of float
        ``(lo, hi)``, finite.

    Returns
    -------
    tuple of numpy.ndarray
        ``(low, high, index)``: clipped edges of the bins of finite positive width, and their indices.

    Notes
    -----
    Every edge is clipped to ``[lo, hi]``, so the terminal ``+-inf`` bins
    become finite; bins that lie outside the window (zero width after
    clipping) or have non-finite edges are dropped.
    """
    edges = np.asarray(bins, dtype=float).reshape(-1)
    if len(edges) < 2:
        empty = np.zeros(0)
        return empty, empty.copy(), np.zeros(0, dtype=int)
    lo, hi = float(window[0]), float(window[1])
    low = np.clip(edges[:-1], lo, hi)
    high = np.clip(edges[1:], lo, hi)
    with np.errstate(invalid='ignore'):
        keep = np.isfinite(low) & np.isfinite(high) & (high - low > 0)
    index = np.flatnonzero(keep)
    return low[keep], high[keep], index


def _strict_json(value):
    """`value` with every non-finite float replaced by the string ``'inf'``, ``'-inf'`` or ``'nan'``.

    Parameters
    ----------
    value : object
        A JSON-serializable record (dicts, lists and tuples are converted recursively).

    Returns
    -------
    object
        A copy that ``json.dumps(..., allow_nan=False)`` accepts; ``float`` reads the strings back.

    Notes
    -----
    Python's ``json`` writes non-finite floats as ``Infinity`` and ``NaN``,
    which strict JSON readers (``jq``, JavaScript) reject. The records of
    the ``'bins'`` law hold infinite outer bin edges, and ``lorentzian`` and
    ``density_adjustment`` are ``inf`` by default.
    """
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')
    return value


def _function_name(function):
    module = getattr(function, '__module__', None) or '?'
    name = getattr(function, '__qualname__', None) or getattr(function, '__name__', None)
    return f'{module}.{name or type(function).__name__}'


def _callable_targets(sampler, n, context, lo, hi):
    """Call cgaimmd_lambda_sampler and check its output (shape (n,), finite, inside [lo, hi])."""
    raw = sampler(n, context)
    try:
        values = np.array(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'cgaimmd_lambda_sampler returned {type(raw).__name__}, not an array of '
                         f'floats: {exc}') from None
    if values.shape != (n,):
        raise ValueError(f'cgaimmd_lambda_sampler must return shape ({n},), got {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('cgaimmd_lambda_sampler returned values that are not finite')
    if n and (values.min() < lo or values.max() > hi):
        raise ValueError(f'cgaimmd_lambda_sampler returned values outside the lambda window '
                         f'[{lo}, {hi}] (range {values.min()} .. {values.max()})')
    return values


def _chain_ensemble(chain):
    from ..pathensemble import PathEnsemble
    if chain is None:
        return PathEnsemble()
    if not isinstance(chain, PathEnsemble):
        chain = PathEnsemble(list(chain))
    return chain


def _populations(chain, bins, density_adjustment):
    """Histogram of the chain's last `density_adjustment` shooting values (select_shooting_point)."""
    chain = _chain_ensemble(chain)
    if len(chain):
        chain = chain[chain.accepted]
    values = np.asarray(chain.shooting('values'), dtype=float).reshape(-1) if len(chain) \
        else np.zeros(0)
    if 0 < density_adjustment < math.inf:
        values = values[-int(density_adjustment):]
    return np.histogram(values, bins)[0]


def _shared_populations(context, shooting_chains, bins):
    """Histogram of the shooting points the shared density adjustment counts.

    The same frames and the same values as RFPS selection
    (:func:`aimmd.worker.utils.select_shooting_point`): the registered
    shooting points of `shooting_chains` (featurized where their descriptor
    rows are missing) and frame 0 of every backward half in flight (only
    read), through :func:`aimmd.worker.utils.collect_shared_shooting_points`
    and :func:`aimmd.worker.utils.shared_shooting_populations`.
    """
    from ..worker import utils as worker_utils
    params = context.params
    registered, shared = worker_utils.collect_shared_shooting_points(
        [_chain_ensemble(chain) for chain in shooting_chains], context.folder,
        params.states[1], params.trajectory_extension)
    return worker_utils.shared_shooting_populations(registered, shared, params, bins)


def _uniform(n, rng, lo, hi):
    return lo + (hi - lo) * rng.random(n)


def _bins_law(n, context, lo, hi, base, chain, shooting_chains):
    """The 'bins' law; falls back to uniform draws without usable bins."""
    rng = context.rng
    edges = _as_bins(context.bins)
    if edges is None or context.densities is None:
        if once('cgaimmd lambda_law bins: no bins yet'):
            context.log(f"C-GAIMMD lambda law 'bins': no bins yet; lambda* is drawn uniformly on "
                        f"[{lo:.3f}, {hi:.3f}] until the trainer writes its bins")
        return _uniform(n, rng, lo, hi), {'name': 'bins (uniform: no bins yet)', **base}
    loaded = np.asarray(context.densities, dtype=float).reshape(-1)
    if len(loaded) != len(edges) - 1:
        raise ValueError(f'{len(loaded)} densities for {len(edges) - 1} bins')
    low, high, index = finite_window_bins(edges, (lo, hi))
    if not len(index):
        if once('cgaimmd lambda_law bins: no bins in the window'):
            context.log(f"C-GAIMMD lambda law 'bins': no bin overlaps the window [{lo:.3f}, "
                        f"{hi:.3f}]; lambda* is drawn uniformly on it")
        return _uniform(n, rng, lo, hi), {'name': 'bins (uniform: no bins in the window)', **base}

    # empty bins get the smallest non-empty density of the window, then normalization
    with np.errstate(invalid='ignore'):
        filled = loaded > EMPTY_DENSITY
    in_window = loaded[index][filled[index]]
    floor = float(in_window.min()) if len(in_window) else 1.0
    densities = np.where(filled, loaded, floor)
    densities = densities / densities.sum()

    # RFPS's adjustments (select_shooting_point), through the same helpers
    params = context.params
    lorentzian = float(params.lorentzian)
    density_adjustment = max(params.density_adjustment, 0)
    shared_density_adjustment = bool(params.shared_density_adjustment)
    if params.nbins <= 1:                    # no density adjustment in this case (RFPS)
        density_adjustment = 0
        shared_density_adjustment = False
    if not BINS_LAW_AIMMD_ADJUSTMENTS:
        lorentzian, density_adjustment, shared_density_adjustment = math.inf, 0, False
    from ..worker import utils as worker_utils
    if lorentzian < math.inf:
        densities = worker_utils.apply_lorentzian(densities, edges, lorentzian)
    if density_adjustment:
        populations = _populations(chain, edges, density_adjustment)
        shared = (_shared_populations(context, shooting_chains, edges)
                  if shared_density_adjustment else np.zeros_like(populations))
        densities = worker_utils.apply_density_adjustment(
            densities, populations, shared, np.ones(len(densities)), density_adjustment)
    else:
        densities = densities / densities.sum()

    selected = densities[index]
    if not (np.isfinite(selected).all() and (selected > 0).all()):
        raise ValueError(f"the 'bins' law got non-positive selection densities {selected} "
                         f"(lorentzian={lorentzian})")
    weights = 1.0 / selected
    weights = weights / weights.sum()
    k = rng.choice(len(weights), size=n, p=weights)
    targets = np.clip(low[k] + (high[k] - low[k]) * rng.random(n), lo, hi)
    record = {'name': 'bins', **base, 'bins': edges.tolist(), 'densities': loaded.tolist(),
              'adjusted_densities': densities.tolist(), 'lorentzian': lorentzian,
              'density_adjustment': float(density_adjustment)}
    return targets, record


def draw_targets(n, context, *, chain=None, shooting_chains=()):
    """Draw the target committor levels of one pool refill.

    Parameters
    ----------
    n : int
        Number of targets (``cgaimmd_parameters['n_targets']``), >= 0.
    context : aimmd.cgaimmd.types.CandidateContext
        The refill's context (``rng`` its law stream; ``bins``/``densities`` as loaded before it).
    chain : aimmd.pathensemble.PathEnsemble, optional
        The chain being extended (the ``'bins'`` law's density adjustment).
    shooting_chains : sequence of PathEnsemble, optional
        The chains the worker manages (``shared_density_adjustment``).

    Returns
    -------
    tuple
        ``(targets, record)``: ``(n,)`` float64 logits inside the window and the law record (strict JSON).

    Raises
    ------
    ValueError
        For an unknown law, an empty window, or a bad ``cgaimmd_lambda_sampler`` output (see Notes).

    Notes
    -----
    ``params.cgaimmd_lambda_sampler``, when set, replaces the law: it is called
    as ``cgaimmd_lambda_sampler(n, context)`` (``context.rng`` is the refill's
    law stream) and its output must have the shape ``(n,)``, be finite and
    lie inside the window. Otherwise
    ``settings['lambda_law']`` decides (:data:`LAWS`):

    - ``'uniform'``: ``lo + (hi - lo) U(0, 1)``;
    - ``'bins'``: over the trainer's bins restricted to the window
      (:func:`finite_window_bins`), bin ``k`` with weight ``1 / density_k``,
      then uniform inside it. The densities are the projected densities of
      the trainer (``context.densities``); empty bins (at most
      :data:`EMPTY_DENSITY`) get the smallest non-empty density of the bins
      in the window, and then, as in RFPS selection,
      :func:`aimmd.worker.utils.apply_lorentzian` (``params.lorentzian``) and
      :func:`aimmd.worker.utils.apply_density_adjustment` with the histogram
      of the last ``params.density_adjustment`` shooting values of `chain`
      and, with ``params.shared_density_adjustment``, of the shooting points
      of `shooting_chains` and of frame 0 of every ``chain{t}*/back{ext}``
      of the run (both off for ``params.nbins <= 1``;
      :data:`BINS_LAW_AIMMD_ADJUSTMENTS` switches all of them off). Without
      bins, or when no bin overlaps the window, the draws are uniform on the
      window (logged once).

    The window is :func:`resolve_window` with the network's ``L``
    (:func:`aimmd.cgaimmd.protocol.lambda_clip_of`) and
    ``params.reweight_parameters``. The record is ``{'name', 'window',
    'window_source', ...}``: ``'uniform'`` adds nothing, ``'callable'`` the
    ``'function'`` (``'<module>.<qualname>'``), ``'bins'`` the ``'bins'``,
    the loaded ``'densities'``, the ``'adjusted_densities'`` of every bin
    and the ``'lorentzian'`` and ``'density_adjustment'`` used; the
    fallbacks are named ``'bins (uniform: no bins yet)'`` and ``'bins
    (uniform: no bins in the window)'``. Non-finite floats of the record
    (the infinite outer edges, a ``lorentzian`` or ``density_adjustment``
    of ``inf``) are the strings ``'inf'``/``'-inf'``,
    so that the sidecar is strict JSON.

    A C-GAIMMD worker's one pool serves all its chains (``nchains_per_worker >
    1``), while `chain` is the chain whose selection triggered the refill:
    candidates later served to other chains carry that chain's density
    adjustment (an efficiency matter only; the law is not part of the
    reweighting).
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError(f'n must be an int >= 0, got {n!r}')
    n = int(n)
    params = context.params
    settings = context.settings
    L = lambda_clip_of(context.network, settings)
    (lo, hi), window = resolve_window(settings, L, context.bins,
                                      reweight_parameters=getattr(params, 'reweight_parameters',
                                                                  None))
    base = {'window': [lo, hi], 'window_source': window['source']}
    sampler = getattr(params, 'cgaimmd_lambda_sampler', None)
    if sampler is not None:
        targets = _callable_targets(sampler, n, context, lo, hi)
        record = {'name': 'callable', 'function': _function_name(sampler), **base}
    else:
        law = settings.get('lambda_law', CGAIMMD_PARAMETER_DEFAULTS['lambda_law'])
        if law not in LAWS:
            raise ValueError(f'unknown lambda_law {law!r}; expected one of {LAWS}')
        if law == 'uniform':
            targets, record = _uniform(n, context.rng, lo, hi), {'name': 'uniform', **base}
        else:
            targets, record = _bins_law(n, context, lo, hi, base, chain, shooting_chains)
    return targets, _strict_json(record)
