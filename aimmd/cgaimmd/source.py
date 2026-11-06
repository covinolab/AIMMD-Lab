"""
aimmd.cgaimmd.source
====================

The shooting-point source of ``chain_type='cgaimmd'``.

With ``chain_type='cgaimmd'`` every reactive-region shooting point is
generated at a drawn target committor, checked and prepared, and then shot
exactly as an RFPS shooting point. This module is to C-GAIMMD what
:func:`aimmd.worker.utils.select_shooting_point` is to RFPS: it returns a
shooting point, and the loop stays in ``aimmd.worker._shoot``.

:class:`GenerativeShootingPointSource` works in two stages:

1. **Refill** (stage 1, a batch): draw ``n_targets`` target levels
   (:mod:`aimmd.cgaimmd.lambda_law`), generate ``candidates_per_target``
   candidates per target through the run's generator
   (``make_generator(params.cgaimmd_generator, network)``,
   :mod:`aimmd.cgaimmd.generation`), drop non-finite samples, apply
   ``params.cgaimmd_validity_function``, score the survivors with the
   committor (the *raw* lambda; ``raw_lambda_tolerance`` prefilters on it once
   the acceptance rule is enforced) and put them into the pool
   (:mod:`aimmd.cgaimmd.pool`).
2. **One candidate** (stage 2, per pop, :func:`prepare_candidate`): place it
   in the box (:func:`place_in_box`) or prepare it with
   ``params.cgaimmd_prepare_function`` (e.g. re-solvation), minimize it with
   ``params.cgaimmd_minimize_function``, and run the :func:`final_check`
   (state and committor of the frame handed to grompp) until one passes.

The source knows nothing about how a generator works: it hands it ``(B,)``
target logits and gets ``(B, M, 3)`` candidates in Ångström. It slices a
refill into batches of the generator's ``chunk`` setting (the whole refill
for a generator without one), checks ``must_stop`` between batches and halves
the batch after running out of CUDA memory; the batch size it used is written
to every sidecar next to the generator's method and settings.

The pool resets whenever the network file changes (checked before every
pop; the discarded candidates count as ``'pool_reset'``), with one exception
(the PI's answer of 2 Oct): a refill whose network was replaced while it was
generated serves :data:`STALE_REFILL_POPS` pop(s) first, so that a refill
that takes longer than a training round still yields shots. The source never
raises for want of a candidate: after ``max_refills`` refills without a
shooting point it warns and waits ``wait_seconds``, and it returns None as
soon as the worker must stop. Hooks report bad candidates (``ok=False``);
exceptions from hooks, the generator, the network or AIMMD propagate and
stop the worker.

numpy and light C-GAIMMD modules at import; torch, MDAnalysis and the path
classes are imported where they are used.
"""

import os
import time
from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional

import numpy as np

from ..core.utils import now
from ..params.utils import CGAIMMD_PARAMETER_DEFAULTS
from ._once import once
from .devices import device_for
from .generation import check_generator, make_generator
from .lambda_law import _function_name, draw_targets, resolve_window
from .pool import (Candidate, CandidatePool, NetworkWatcher, base_seed, preparation_stream,
                   refill_seed, refill_streams, stream_seed)
from .protocol import (_values_input, check_network_numerics, check_network_protocol,
                       coordinate_committor, lambda_clip_of, n_atoms_of, network_device)
from .sidecar import (SIDECAR_SCHEMA, SIDECAR_SOURCE, _log, aimmd_provenance, bias_settings,
                      software_versions)
from .sidecar import _now as _utc_now                    # ISO 8601 UTC (core.utils.now is local)
from .types import CandidateContext, PreparedFrame, ShootingPoint, ValidityResult
from .values import descriptor_width_error


READBACK_FAILURE_IS_REJECTION = True
"""Whether a generated shooting point that fails the read-back state check is a candidate rejection.

AIMMD checks that frame 0 of the new segments (``back.xtc``) is in the
shooting point's state and raises ``RuntimeError('... should be in R ...')``
otherwise; unless ``retry_with_state_definition_glitches`` is set, the worker
stops. C-GAIMMD places candidates at the basin rims on purpose, so under
``chain_type='cgaimmd'`` that failure (step-0 constraints, xtc compression)
is routine: with True the worker's glitch branch runs whatever the flag says,
the candidate is discarded (``state_readback`` in the source's counters) and
the next one is taken from the pool. This is the PI's answer of 2 Oct
("discard on read-back failure"), kept in this one place; False restores the
flag's rule for C-GAIMMD chains.
"""

WINDOW_AFTER_WARNING_KEY = 'window_after_trained_shots default'
"""The ``aimmd.cgaimmd._once`` key of the worker's warning about a default ``window_after_trained_shots``.

The Params checks print the same line under their own key (``'params:
window_after_trained_shots default'``), before ``Worker.run`` redirects the
output; with a key of its own the worker prints it into its ``worker.log`` as
well (once per process each).
"""

ACCEPTANCE_RULES = ('target', 'window')
"""The values of ``cgaimmd_parameters['acceptance']`` (the rule the final check enforces)."""

NUMERICS_FRAMES = 4
"""Initial-path frames the start-up numerical checks of the network run on."""

POLL_SECONDS = 1.0
"""Seconds between polls while waiting (for the network file, or after too many rejections)."""

STALE_REFILL_POPS = 1
"""Pops a refill may serve after its network was replaced while it was generated.

The PI's answer of 2 Oct: such a refill serves at least one pop (the shot's
sidecar says ``pool.stale_refill_pop`` and names the refill's network,
``pool.refill_network_sha256``), and then the pool resets as before. 0 keeps
the PI's earlier ruling strictly (every refill outdated before its first pop
is discarded unused, so a refill slower than a training round never shoots).
"""

_STOPPED = object()


def _sleep(seconds):
    """``time.sleep`` (a module function, so that tests can replace the waits)."""
    time.sleep(seconds)


def _code(reason, default):
    """The reason code of a rejection: the text before the first colon, `default` if empty."""
    text = str(reason or '').split(':', 1)[0].strip()
    return text or default


def _format_counts(counts):
    if not counts:
        return 'none'
    return ', '.join(f'{reason}: {count}' for reason, count in sorted(counts.items()))


def _add(counts, reason, n=1):
    counts[reason] = counts.get(reason, 0) + n


def _is_out_of_memory(exc):
    import torch
    oom = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def _empty_cuda_cache():
    import torch
    if torch.cuda.is_initialized():                        # never initializes CUDA
        torch.cuda.empty_cache()


def _seeded_generator(device, seed):
    """A ``torch.Generator`` on `device`, ``manual_seed(seed)``."""
    import torch
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _move_network(params, network, role):
    """Put `network` on the device of `role` (inside a worker task) and return its device."""
    device = device_for(role, params)
    if device is not None:
        network.to(device)
    return network_device(network)


def _refills_after(record):
    """A resumed path's refill counter: one past its record's ``seed.refill_index`` (else 0).

    Only a refill made for the record's own path counts (``seed.refill_path``): a
    candidate the pool carried over from an earlier path's refill means that the path
    made no refill of its own yet.
    """
    seed = record.get('seed')
    if not isinstance(seed, dict):
        return 0
    if 'refill_path' in seed and seed['refill_path'] != record.get('path_name'):
        return 0
    index = seed.get('refill_index')
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        return 0
    return int(index) + 1


def _window_text(window, record):
    """The resolved window as the start-up line prints it."""
    lo, hi = window
    if record['source'] == 'setting':
        return f'{lo}, {hi}'
    if record['source'] == 'bins':
        return f'bins ({lo}, {hi})'
    return f'bins (no bins yet: {lo}, {hi})'


def _checked_frame(frame, hook):
    if not isinstance(frame, PreparedFrame):
        raise TypeError(f'{hook} must return an aimmd.cgaimmd.types.PreparedFrame, got '
                        f'{type(frame).__name__}')
    return frame


def _box_of(universe):
    """The universe's box as a float64 (6,) array, or None without a (non-degenerate) box."""
    if universe is None:
        return None
    dimensions = getattr(universe, 'dimensions', None)
    if dimensions is None:
        return None
    box = np.asarray(dimensions, dtype=np.float64).reshape(-1)
    if box.shape != (6,) or not np.all(np.isfinite(box)) or not np.all(box[:3] > 0):
        return None
    return box


def _decimals(setting, params):
    if setting == 'auto':
        return 2 if getattr(params, 'trajectory_extension', None) == '.xtc' else None
    return None if setting is None else int(setting)


def _reader(positions, dimensions):
    """A one-frame ChainReader over an MDAnalysis MemoryReader (what Path.compute hands on)."""
    from MDAnalysis.coordinates.memory import MemoryReader
    from ..path.chainreader import ChainReader
    return ChainReader(MemoryReader(positions[None], dimensions=dimensions))


def _batch_size(settings):
    """The generator's ``chunk`` setting if it is an int >= 1, else None (one batch per refill)."""
    chunk = settings.get('chunk') if isinstance(settings, dict) else None
    if isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1:
        return None
    return int(chunk)


def build_checked_generator(params):
    """The run's generator, built from ``params.cgaimmd_generator`` and checked strictly.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (``network``, ``cgaimmd_generator``).

    Returns
    -------
    aimmd.cgaimmd.generation.Generator

    Raises
    ------
    TypeError
        If the network does not follow the protocol
        (:func:`~aimmd.cgaimmd.protocol.check_network_protocol`), the
        generator cannot be built (:func:`~aimmd.cgaimmd.generation.make_generator`)
        or cannot use the network (:func:`~aimmd.cgaimmd.generation.check_generator`);
        the message names ``params.cgaimmd_generator``. This holds for params
        read through old names too (``Params.load`` only warns for those).

    Notes
    -----
    What a shooting worker checks at start-up, and what
    :mod:`aimmd.cgaimmd.preflight` checks before a job.
    """
    network = params.network
    check_network_protocol(network)
    spec = getattr(params, 'cgaimmd_generator', None)
    try:
        generator = make_generator(spec, network)
    except (TypeError, ValueError, ImportError) as exc:
        raise TypeError(f"chain_type='cgaimmd': the generator of params.cgaimmd_generator = "
                        f"{spec!r} cannot be built for {type(network).__name__}: "
                        f"{exc}") from exc
    problems = check_generator(generator, network)
    if problems:
        raise TypeError(f"chain_type='cgaimmd': the generator {generator.method!r} "
                        f"(params.cgaimmd_generator = {spec!r}) cannot use "
                        f"{type(network).__name__}:\n  - " + '\n  - '.join(problems))
    return generator


def check_start_up_numerics(params, generator, n_atoms, initial_paths):
    """The descriptor width and the network numerics, as a shooting worker checks them at start-up.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (``descriptors_function``, ``network``, ``descriptor_transform``).
    generator : aimmd.cgaimmd.generation.Generator
        The run's generator (one tiny generation goes through it).
    n_atoms : int
        ``M``, the atoms the network generates.
    initial_paths : sequence of aimmd.path.Path
        The run's initial paths; the first :data:`NUMERICS_FRAMES` frames of
        the first one are featurized and evaluated.

    Returns
    -------
    dict
        The report of :func:`~aimmd.cgaimmd.protocol.check_network_numerics`.

    Raises
    ------
    TypeError
        Without a ``descriptors_function``.
    ValueError
        If the descriptors are not ``3 M`` wide (with the migration hint), or
        from the numerics check.
    """
    m = int(n_atoms)
    function = getattr(params, 'descriptors_function', None)
    if function is None:
        raise TypeError("chain_type='cgaimmd' needs a descriptors_function returning the "
                        "flat Angstrom coordinates of the network's atoms, e.g. "
                        "GeneratorDescriptors (aimmd.cgaimmd.preparation.descriptors)")
    where = 'no initial path'
    descriptors = np.zeros((0, 3 * m), dtype=np.float32)
    if initial_paths and len(initial_paths[0]):
        first = initial_paths[0]
        frames = first[:min(len(first), NUMERICS_FRAMES)]
        descriptors = np.asarray(frames.compute(function, '', 'reader', return_result=True),
                                 dtype=np.float32)
        descriptors = descriptors.reshape(len(descriptors), -1)
        where = str(getattr(first, 'fname', first))
    if descriptors.shape[1] != 3 * m:
        raise descriptor_width_error(descriptors.shape[1], m,
                                     f'the descriptors_function values of {where}')
    coordinates = descriptors.reshape(len(descriptors), m, 3)
    forward_input = None
    if len(descriptors):
        forward_input = _values_input(descriptors, network_device(params.network),
                                      getattr(params, 'descriptor_transform', None))
    return check_network_numerics(params.network, coordinates, forward_input,
                                  generation=generator)


def check_generator_atoms(params, n_atoms):
    """Refuse a network whose atom count is not that of the run's generator atoms.

    Parameters
    ----------
    params : aimmd.Params
        The run's params; their ``descriptors_function`` is looked at.
    n_atoms : int
        ``M``, the atoms the network generates.

    Raises
    ------
    ValueError
        If the descriptors function is (or wraps, see
        :func:`aimmd.cgaimmd.preparation.descriptors.find_generator_descriptors`)
        a :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
        (or another object with ``generator_atoms`` and ``n_atoms``) whose
        ``n_atoms`` is not `n_atoms`. Other descriptors functions are checked
        by the width of their rows only (at start-up, on the initial paths).
    """
    function = getattr(params, 'descriptors_function', None)
    if function is None:
        return
    descriptors = function
    if not hasattr(function, 'generator_atoms'):
        from .preparation.descriptors import find_generator_descriptors
        descriptors = find_generator_descriptors(function)
        if descriptors is None:
            return
    m = getattr(descriptors, 'n_atoms', None)
    if isinstance(m, Integral) and int(m) != int(n_atoms):
        raise ValueError(
            f'the network generates {n_atoms} atoms but the descriptors function '
            f'({type(descriptors).__name__}) selects {int(m)} generator atoms '
            f'(GeneratorAtoms); the network (for BioEmu2: its system folder) and the '
            f'descriptors must describe the same atoms')


def place_in_box(positions, params, placement):
    """Turn the generator atoms of a candidate into the full system without a prepare function.

    Parameters
    ----------
    positions : array-like
        ``(M, 3)`` Å, the candidate's generator atoms, which must be the whole system (``M == N``).
    params : aimmd.Params
        ``params.universe`` gives ``N`` and the box.
    placement : {'box_center', 'none'}
        ``cgaimmd_parameters['placement']``.

    Returns
    -------
    aimmd.cgaimmd.types.PreparedFrame
        Float64 Å positions, the universe's box or None, and the record (keys in the Notes).

    Raises
    ------
    ValueError
        If ``M`` differs from the universe's atom count, or for an unknown placement.
    TypeError
        For ``'box_center'`` without a universe box or with a non-orthorhombic one.

    Notes
    -----
    ``'box_center'`` translates the centroid of the atoms to the center of the
    box, ``dimensions[:3] / 2``: the generator's frame of reference is its own
    (translation-free for the reference model), the box is the run's.
    ``'none'`` keeps the coordinates as generated (generators that work in
    absolute coordinates, such as the toy network) and uses the universe's box
    or None. The record is ``{'placement', 'box', 'translation'}``
    (``translation`` None for ``'none'``).
    """
    started = time.perf_counter()
    positions = np.array(positions, dtype=np.float64)
    universe = getattr(params, 'universe', None)
    if universe is not None:
        n_atoms = universe.atoms.n_atoms
        if len(positions) != n_atoms:
            raise ValueError(f'generator atoms (M={len(positions)}) differ from the system '
                             f'(N={n_atoms}): set cgaimmd_prepare_function, e.g. '
                             f'aimmd.cgaimmd.preparation.GromacsSolvator')
    box = _box_of(universe)
    if placement == 'box_center':
        if box is None:
            raise TypeError("placement 'box_center' needs params.universe with a box (from "
                            "params.topology); use placement 'none' for a box-less system")
        if not np.allclose(box[3:], 90.0):
            raise TypeError(f"placement 'box_center' supports orthorhombic boxes only, got the "
                            f"angles {box[3:].tolist()}")
        translation = box[:3] / 2.0 - positions.mean(axis=0)
        positions = positions + translation
        record = {'placement': 'box_center', 'box': box.tolist(),
                  'translation': translation.tolist()}
    elif placement == 'none':
        record = {'placement': 'none', 'box': None if box is None else box.tolist(),
                  'translation': None}
    else:
        raise ValueError(f"placement must be 'box_center' or 'none', got {placement!r}")
    return PreparedFrame(positions=positions, dimensions=None if box is None else box.copy(),
                         ok=True, reason='', record=record,
                         seconds=time.perf_counter() - started)


def final_check(frame, context, lambda_target):
    """Check a prepared frame: in the reactive region, and at the right committor.

    Parameters
    ----------
    frame : aimmd.cgaimmd.types.PreparedFrame
        The full system as it will be handed to grompp.
    context : aimmd.cgaimmd.types.CandidateContext
        Provides ``params``, ``settings``, ``network``, ``network_info`` and ``bins``.
    lambda_target : float
        The candidate's target level.

    Returns
    -------
    dict
        The check's record (its keys are listed in the Notes).

    Raises
    ------
    ValueError
        If ``settings['acceptance']`` is not one of :data:`ACCEPTANCE_RULES`.

    Notes
    -----
    The record's keys are ``ok``, ``reason`` (``''`` when ok), ``state``,
    ``lambda``, ``rule``, ``enforced``, ``tolerance``, ``window``,
    ``window_after_trained_shots``, ``n_trained_shots``, ``decimals`` and
    ``seconds``.

    1. ``x = frame.positions.astype(float32)``, rounded to
       ``final_check_round`` decimals (``'auto'``: 2 for ``.xtc``, the
       precision frame 0 of ``back.xtc`` is stored at), inside a one-frame
       :class:`aimmd.path.chainreader.ChainReader` over an MDAnalysis
       ``MemoryReader``, the reader type ``Path.compute`` hands to user
       functions.
    2. ``params.states_function`` of it must be ``states[1]``; otherwise the
       reason is ``'state_<X>'``.
    3. ``lambda = params.values_function(params.descriptors_function(reader))``
       (the values function puts the network on its device); a non-finite
       value is ``'lambda_nonfinite'``.
    4. The rule: with ``n_trained_shots`` (``context.network_info``, 0 without
       one) below ``window_after_trained_shots`` any R frame is accepted
       (``'bootstrap_any_R'``); from there on ``'target'`` (``|lambda -
       lambda_target| <= lambda_tolerance``, the default: the PI's decision of
       29 Sep, enforced in production) or ``'window'`` (lambda inside the
       resolved window, :func:`aimmd.cgaimmd.lambda_law.resolve_window`),
       else ``'lambda_window'``.

    The check reshapes the law of the shooting points (it is not a measure of
    the generator, whose adherence is measured without it). ``rule``,
    ``enforced``, ``tolerance`` and ``window`` are filled for every outcome;
    ``lambda`` is None when the committor was not reached or is not finite.
    An unknown ``acceptance`` raises before anything is evaluated, also in the
    bootstrap phase (``canonical_cgaimmd_parameters`` rejects it in params;
    this function takes any ``context.settings``).
    """
    started = time.perf_counter()
    params = context.params
    settings = context.settings
    if settings['acceptance'] not in ACCEPTANCE_RULES:
        raise ValueError(f"unknown acceptance {settings['acceptance']!r}; expected one of "
                         f"{ACCEPTANCE_RULES}")
    info = context.network_info
    n_trained = 0 if info is None or info.n_trained_shots is None else int(info.n_trained_shots)
    after = int(settings['window_after_trained_shots'])
    enforced = n_trained >= after
    rule = settings['acceptance'] if enforced else 'bootstrap_any_R'
    L = lambda_clip_of(context.network, settings)
    (lo, hi), _ = resolve_window(settings, L, context.bins,
                                 reweight_parameters=getattr(params, 'reweight_parameters', None))
    tolerance = float(settings['lambda_tolerance'])
    decimals = _decimals(settings['final_check_round'], params)
    result = {'ok': False, 'reason': '', 'state': None, 'lambda': None, 'rule': rule,
              'enforced': bool(enforced), 'tolerance': tolerance, 'window': [lo, hi],
              'window_after_trained_shots': after, 'n_trained_shots': n_trained,
              'decimals': decimals, 'seconds': 0.0}

    x = np.asarray(frame.positions, dtype=np.float32)
    if decimals is not None:
        x = np.round(x, decimals).astype(np.float32)
    dimensions = None if frame.dimensions is None else np.asarray(frame.dimensions,
                                                                  dtype=np.float32)
    state = str(np.asarray(params.states_function(_reader(x, dimensions))).reshape(-1)[0])
    result['state'] = state
    if state != params.states[1]:
        result['reason'] = f'state_{state}'
    else:
        descriptors = params.descriptors_function(_reader(x, dimensions))
        value = float(np.asarray(params.values_function(descriptors), dtype=float).reshape(-1)[0])
        if not np.isfinite(value):
            result['reason'] = 'lambda_nonfinite'
        else:
            result['lambda'] = value
            if rule == 'bootstrap_any_R':
                passed = True
            elif rule == 'target':
                passed = abs(value - float(lambda_target)) <= tolerance
            else:                                           # 'window'
                passed = lo <= value <= hi
            if passed:
                result['ok'] = True
            else:
                result['reason'] = 'lambda_window'
    result['seconds'] = time.perf_counter() - started
    return result


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: one candidate
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PreparedCandidate:
    """What :func:`prepare_candidate` did with one candidate."""

    ok: bool
    """Whether the frame passed the final check (and is the shooting point)."""

    reason: str
    """``''`` when ok; else the rejection code (``'prepare_failed'``, ``'state_B'``,
    ``'lambda_window'``, ...), or ``'stopped'`` when :attr:`stopped`."""

    stopped: bool
    """True if ``must_stop`` became True during a hook: the frame was neither checked nor rejected."""

    frame: Optional[PreparedFrame]
    """The last frame produced (the one checked, when the check ran); None if no hook returned one."""

    prepare: dict
    """The preparation's record: ``function``, ``placement``, ``ok``, ``reason``, ``seconds``, ``record``."""

    minimize: dict
    """The minimization's record (``function`` None without a minimize function)."""

    check: Optional[dict]
    """The :func:`final_check` record, None if the check was not reached."""

    seconds: dict = field(default_factory=dict)
    """``{'prepare', 'minimize', 'final_check'}`` wall times."""

    values_device: Optional[str] = None
    """The network's device during the final check (None if the check was not reached)."""


def prepare_candidate(positions, lambda_target, context):
    """Stage 2 of C-GAIMMD for one generated candidate: prepare, minimize, final check.

    Parameters
    ----------
    positions : array-like
        ``(M, 3)`` Å generator atoms (after the validity function, if any).
    lambda_target : float
        The target logit committor the candidate was generated at.
    context : aimmd.cgaimmd.types.CandidateContext
        The candidate's context: ``params`` (its ``cgaimmd_prepare_function``,
        ``cgaimmd_minimize_function``, states, descriptor and values
        functions), ``settings``, ``network`` and ``network_info`` (the final
        check's rule), ``scratch``, ``rng``/``seed`` (the candidate's
        preparation stream), ``must_stop``. :func:`candidate_context` builds
        one outside a worker.

    Returns
    -------
    PreparedCandidate
        What happened; :attr:`PreparedCandidate.frame` is the shooting point
        when :attr:`PreparedCandidate.ok`.

    Raises
    ------
    TypeError
        If a hook returns something other than a
        :class:`~aimmd.cgaimmd.types.PreparedFrame`, or for the placement
        errors of :func:`place_in_box`.

    Notes
    -----
    This is exactly what the shooting worker does with every popped candidate
    (:class:`GenerativeShootingPointSource`), made public so that validation
    tools can prepare generated structures the way a run would:

    1. ``params.cgaimmd_prepare_function(positions, context)``, or
       :func:`place_in_box` with ``settings['placement']`` without one; a
       frame with ``ok=False`` is rejected with its reason code (the text
       before the first colon, else ``'prepare_failed'``);
    2. ``params.cgaimmd_minimize_function(frame, context)`` if set (reason
       code default ``'minimize_failed'``);
    3. the network is put on the values device (inside a worker task) and
       :func:`final_check` decides.

    ``context.scratch`` is created when a hook is set. ``must_stop`` is
    checked after each hook: a frame returned after it became True is
    neither checked nor counted as a rejection (``stopped``).
    """
    params = context.params
    settings = context.settings
    prepare_function = getattr(params, 'cgaimmd_prepare_function', None)
    minimize_function = getattr(params, 'cgaimmd_minimize_function', None)
    if prepare_function is not None or minimize_function is not None:
        os.makedirs(context.scratch, exist_ok=True)
    seconds = {'prepare': 0.0, 'minimize': 0.0, 'final_check': 0.0}
    minimize = {'function': None, 'ok': None, 'reason': None, 'seconds': None, 'record': None}

    def outcome(ok, reason, frame, check=None, stopped=False, device=None):
        return PreparedCandidate(ok=ok, reason=reason, stopped=stopped, frame=frame,
                                 prepare=prepare, minimize=minimize, check=check,
                                 seconds=seconds, values_device=device)

    clock = time.perf_counter()
    if prepare_function is not None:
        frame = _checked_frame(prepare_function(np.array(positions), context),
                               'cgaimmd_prepare_function')
        prepare = {'function': _function_name(prepare_function), 'placement': None}
    else:
        frame = place_in_box(positions, params, settings['placement'])
        prepare = {'function': 'place_in_box', 'placement': settings['placement']}
    seconds['prepare'] = time.perf_counter() - clock
    prepare.update(ok=bool(frame.ok), reason=str(frame.reason or ''), seconds=seconds['prepare'],
                   record=frame.record)
    if context.must_stop():
        return outcome(False, 'stopped', frame, stopped=True)
    if not frame.ok:
        return outcome(False, _code(frame.reason, 'prepare_failed'), frame)

    if minimize_function is not None:
        clock = time.perf_counter()
        frame = _checked_frame(minimize_function(frame, context), 'cgaimmd_minimize_function')
        seconds['minimize'] = time.perf_counter() - clock
        minimize = {'function': _function_name(minimize_function), 'ok': bool(frame.ok),
                    'reason': str(frame.reason or ''), 'seconds': seconds['minimize'],
                    'record': frame.record}
        if context.must_stop():
            return outcome(False, 'stopped', frame, stopped=True)
        if not frame.ok:
            return outcome(False, _code(frame.reason, 'minimize_failed'), frame)

    device = str(_move_network(params, context.network, 'values'))
    check = final_check(frame, context, lambda_target)
    seconds['final_check'] = check['seconds']
    return outcome(bool(check['ok']), '' if check['ok'] else check['reason'], frame, check,
                   device=device)


def candidate_context(params, folder, *, path_name='candidate', seed=0, network=None,
                      network_info=None, directory=None, must_stop=None, log=None,
                      bins=None, densities=None):
    """A :class:`~aimmd.cgaimmd.types.CandidateContext` for :func:`prepare_candidate` outside a worker.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (``chain_type='cgaimmd'``).
    folder : str
        A work folder; the hooks' scratch folder is ``f'{folder}/cgaimmd_scratch'``.
    path_name : str, optional
        The name the hooks see.
    seed : int, optional
        The seed of the candidate's random stream (``rng`` is ``numpy.random.default_rng(seed)``).
    network, network_info : optional
        Default: ``params.network``, None (the final check then uses the
        bootstrap rule unless ``window_after_trained_shots`` is 0; pass a
        :class:`~aimmd.cgaimmd.types.NetworkInfo` to apply the trained rule).
    directory : str, optional
        The run directory (default: `folder`'s parent).
    must_stop, log : callable, optional
        Default: never stop; AIMMD's print.
    bins, densities : numpy.ndarray, optional
        The trainer's bins (``lambda_window='bins'``).

    Returns
    -------
    aimmd.cgaimmd.types.CandidateContext
        On the network's device, with a CPU ``torch.Generator`` seeded with `seed`.
    """
    network = params.network if network is None else network
    device = network_device(network)
    folder = str(folder)
    return CandidateContext(
        params=params, network=network, network_info=network_info,
        directory=directory if directory is not None else os.path.dirname(folder) or '.',
        folder=folder, path_name=path_name, scratch=f'{folder}/cgaimmd_scratch',
        rng=np.random.default_rng(int(seed)), seed=int(seed),
        generator=_seeded_generator(device, seed), device=device,
        must_stop=must_stop if must_stop is not None else (lambda: False),
        log=log if log is not None else _log, settings=dict(params.cgaimmd_settings),
        bins=bins, densities=densities)


# ════════════════════════════════════════════════════════════════════════════
# The source
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class _PathState:
    """What one path's selection has spent so far (across the calls of one path)."""

    refills: int = 0
    tried: int = 0
    rejected: dict = field(default_factory=dict)
    prepare: float = 0.0
    minimize: float = 0.0
    final_check: float = 0.0


class GenerativeShootingPointSource:
    """Generated shooting points for the reactive-region chains of a C-GAIMMD run.

    Parameters
    ----------
    params : aimmd.Params
        The run's params (``chain_type='cgaimmd'``).
    worker : aimmd.Worker
        The shooting worker (``must_stop``, ``initial_paths``, ``_location``).
    directory : str
        ``worker._directory``: the run directory, relative to ``params.parent`` (the cwd of workers).

    Attributes
    ----------
    watcher : aimmd.cgaimmd.pool.NetworkWatcher
        Loads ``network{s}.h5`` into ``params.network``.
    pool : aimmd.cgaimmd.pool.CandidatePool
        The stage-1 survivors of the loaded network.
    counters : dict
        Cumulative ``{reason: count}`` of the rejections of both stages and of ``'pool_reset'`` discards.
    consecutive_readback : int
        Read-back rejections (:meth:`note_glitch`) since the last registered shot or read-back wait.
    params : aimmd.Params
        The run's params.
    worker : aimmd.Worker
        The shooting worker.
    directory : str
        The run directory (``worker._directory``).
    settings : dict or None
        ``params.cgaimmd_settings`` as read at start-up (None before :meth:`wait_for_network`).
    generator : aimmd.cgaimmd.generation.Generator or None
        ``make_generator(params.cgaimmd_generator, params.network)``, built and checked at start-up.
    n_atoms : int or None
        ``M``, the network's atom count (start-up).
    base_seed : int or None
        The base seed of candidate generation (:func:`aimmd.cgaimmd.pool.base_seed`; start-up).
    numerics : dict or None
        The report of :func:`aimmd.cgaimmd.protocol.check_network_numerics` at start-up.
    provenance : dict or None
        What every sidecar records about the software, the network and the
        bias (``aimmd``, ``software``, ``bias``, ``network``; start-up).

    Notes
    -----
    Nothing is loaded, built or moved to a device at construction.
    :meth:`wait_for_network` blocks until the trainer's network loads and runs
    the start-up checks; :meth:`next_shooting_point` then returns one
    shooting point per call. ``aimmd.worker._shoot`` calls :meth:`note_glitch`
    when AIMMD's read-back state check rejects a generated shooting point and
    :meth:`note_registered` after a C-GAIMMD shot is registered.

    The pool resets when the network file changes, checked before every
    pop, except that a refill whose network was replaced while it was
    generated (before its first pop) serves :data:`STALE_REFILL_POPS` pops
    first, its *stale pops*, whose final check uses the new network.

    Devices are resolved at every use (:func:`aimmd.cgaimmd.devices.device_for`):
    the network is moved to the generation device for a refill and to the
    values device for the final check; outside a worker task it stays where it
    is.
    """

    def __init__(self, params, worker, directory):
        self.params = params
        self.worker = worker
        self.directory = directory
        self.watcher = NetworkWatcher(params, directory)
        self.pool = CandidatePool()
        self.counters = {}
        self.consecutive_readback = 0
        self.settings = None
        self.generator = None
        self.numerics = None
        self.n_atoms = None
        self.base_seed = None
        self.provenance = None
        self._generator_settings = None
        self._chunk = None
        self._n_refills = 0
        self._refills = {}                 # refill -> what its candidates' records need (one)
        self._paths = {}                   # (chain folder name, path name) -> _PathState
        self._current = {}                 # chain folder name -> path being selected
        self._chains = {}                  # chain folder -> the chain handed with it
        self._bins = None
        self._densities = None
        self._pops_since_refill = 0        # pops served from the pool since its last refill
        self._stale = None                 # a deferred reset: pops left, the replaced network

    # -- small helpers -------------------------------------------------------------

    def _must_stop(self):
        value = getattr(self.worker, 'must_stop', False)
        if callable(value):
            value = value()
        return bool(value)

    def _wait(self):
        """Wait ``wait_seconds`` in polls of at most 1 s; False if the worker must stop."""
        remaining = float(self.settings['wait_seconds'])
        while remaining > 0:
            if self._must_stop():
                return False
            step = min(POLL_SECONDS, remaining)
            _sleep(step)
            remaining -= step
        return not self._must_stop()

    def _device(self, role):
        device = device_for(role, self.params)
        return network_device(self.params.network) if device is None else device

    def _move(self, role):
        """Put the network on the role's device (inside a worker task) and return its device."""
        return _move_network(self.params, self.params.network, role)

    def _location(self, folder):
        return getattr(self.worker, '_location', None) or folder

    def _count(self, reason, n=1):
        _add(self.counters, reason, n)

    def _path_state(self, folder, path_name):
        name = os.path.basename(os.path.normpath(folder))
        # a read-back rejection filed before the folder's first selection without a path name
        # (an unreadable in-flight marker, note_glitch) belongs to the path selected now
        pending = self._paths.pop((name, None), None)
        for key in [key for key in self._paths if key[0] == name and key[1] != path_name]:
            del self._paths[key]                           # a chain selects one path at a time
        self._current[name] = path_name
        if pending is not None:
            self._paths.setdefault((name, path_name), pending)
        return self._paths.setdefault((name, path_name), _PathState())

    def _context(self, folder, path_name, rng, generator, device, seed=None):
        return CandidateContext(
            params=self.params, network=self.params.network, network_info=self.watcher.info,
            directory=self.directory, folder=folder, path_name=path_name,
            scratch=f'{folder}/cgaimmd_scratch', rng=rng,
            seed=stream_seed(rng) if seed is None else int(seed), generator=generator,
            device=device, must_stop=self._must_stop, log=_log, settings=self.settings,
            bins=self._bins, densities=self._densities)

    # -- start-up ------------------------------------------------------------------

    def wait_for_network(self):
        """Block until ``network{s}.h5`` loads into ``params.network``, then run the start-up checks.

        Returns
        -------
        bool
            True once the network is loaded and checked; False if the worker must stop first.

        Raises
        ------
        TypeError
            If the network breaks the protocol, the generator cannot be built
            or cannot use the network, ``L`` cannot be resolved, or placement
            ``'box_center'`` has no orthorhombic box.
        ValueError
            If the descriptors are not the atoms' flat coordinates, the atom
            counts of the network and the descriptors differ, the numerics
            fail or the window is empty.

        Notes
        -----
        The watcher polls every second; ``Waiting for network<s>.h5 (C-GAIMMD
        shooting) <now>`` is printed once. Start-up: the settings, the base
        seed and ``M``; the generator, built with
        :func:`~aimmd.cgaimmd.generation.make_generator` from
        ``params.cgaimmd_generator`` and checked strictly
        (:func:`~aimmd.cgaimmd.protocol.check_network_protocol`, then
        :func:`~aimmd.cgaimmd.generation.check_generator`; for params read
        through old names ``Params.load`` only warns, a shooting worker does
        not); ``M`` against the generator atoms of a
        :class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`
        descriptors function; the WARNING about R frames of the initial paths
        (they seed free simulations only); the WARNING about a default
        ``window_after_trained_shots`` (once per process,
        :data:`WINDOW_AFTER_WARNING_KEY`); the descriptor width and
        :func:`~aimmd.cgaimmd.protocol.check_network_numerics` (one tiny
        generation through the generator) on the first :data:`NUMERICS_FRAMES`
        frames of the first initial path (the report is kept in ``numerics``);
        the lambda window (:func:`~aimmd.cgaimmd.lambda_law.resolve_window`,
        with the trainer's bins if they exist) and, without
        ``cgaimmd_prepare_function``, the placement (:func:`place_in_box` of a
        zero frame), so that these configuration errors stop the worker
        before its first refill; the provenance of the sidecars; the start-up
        line.
        """
        announced = False
        while True:
            self.watcher.refresh()
            if self.watcher.loaded:
                break
            if self._must_stop():
                return False
            if not announced:
                _log(f'Waiting for network{self.params.sorted_states}.h5 (C-GAIMMD shooting) '
                     f'{now()}')
                announced = True
            _sleep(POLL_SECONDS)
        self._start_up()
        return True

    def _build_generator(self):
        """The run's generator, built and checked strictly against the network."""
        return build_checked_generator(self.params)

    def _check_generator_atoms(self):
        """``M`` must equal the generator atoms of a GeneratorDescriptors descriptors function."""
        check_generator_atoms(self.params, self.n_atoms)

    def _start_up(self):
        params = self.params
        network = params.network
        self.settings = settings = dict(params.cgaimmd_settings)
        self.generator = self._build_generator()
        self._generator_settings = dict(self.generator.settings())
        self._chunk = _batch_size(self._generator_settings)
        self.base_seed = base_seed(settings, self.directory)
        self.n_atoms = n_atoms_of(network)
        self._check_generator_atoms()
        L = lambda_clip_of(network, settings)              # TypeError if L cannot be resolved
        initial_paths = list(self.worker.initial_paths)
        self._warn_r_frames(initial_paths)
        user = getattr(params, 'cgaimmd_parameters', None) or {}
        if 'window_after_trained_shots' not in user and once(WINDOW_AFTER_WARNING_KEY):
            _log(f"WARNING: window_after_trained_shots is at its default "
                 f"{CGAIMMD_PARAMETER_DEFAULTS['window_after_trained_shots']}; this is highly "
                 f"system dependent (it was 8000 for the NANMA test system), set it explicitly")
        values_device = self._move('values')
        network.eval()
        self.numerics = self._check_numerics(initial_paths)
        window = _window_text(*self._check_configuration(L))
        generation_device = self._device('generation')
        self.provenance = self._provenance()
        info = self.watcher.info
        sampler = getattr(params, 'cgaimmd_lambda_sampler', None)
        law = f'callable {_function_name(sampler)}' if sampler is not None else settings['lambda_law']
        _log(f'C-GAIMMD shooting: {type(network).__name__} n_atoms={self.n_atoms}, '
             f'generator {self.generator.method} {self._generator_settings}, devices '
             f'generation={generation_device} values={values_device}, network '
             f'{info.sha256[:8]} (round {info.round}, {self.watcher.n_trained_shots()} trained '
             f'shots), law {law}, window {window}, acceptance {settings["acceptance"]} after '
             f'{settings["window_after_trained_shots"]} trained shots')

    def _provenance(self):
        """The provenance blocks of every sidecar (computed once per worker)."""
        network = self.params.network
        member = getattr(network, 'provenance', None)
        network_provenance = None
        if callable(member):
            network_provenance = member()
        return {'aimmd': aimmd_provenance(), 'software': software_versions(),
                'bias': bias_settings(self.params), 'network': network_provenance}

    def _check_configuration(self, L):
        """Fail at start-up, not after the first refill: the window, and the placement.

        Returns the resolved window and its record (:func:`aimmd.cgaimmd.lambda_law.resolve_window`).
        """
        params = self.params
        settings = self.settings
        self._load_bins()
        window, record = resolve_window(
            settings, L, self._bins, reweight_parameters=getattr(params, 'reweight_parameters', None))
        if getattr(params, 'cgaimmd_prepare_function', None) is None:
            place_in_box(np.zeros((self.n_atoms, 3)), params, settings['placement'])
        return window, record

    def _path_states(self, path):
        try:
            states = np.asarray(path.states).astype(str)
            if len(states) == len(path):
                return states
        except Exception:                                   # noqa: BLE001
            pass
        return np.asarray(path.compute(self.params.states_function, '', 'reader',
                                       return_result=True)).astype(str)

    def _warn_r_frames(self, initial_paths):
        r = self.params.states[1]
        n, files = 0, []
        for path in initial_paths:
            k = int(np.sum(self._path_states(path) == r))
            if k:
                n += k
                files.append(str(getattr(path, 'fname', path)))
        if n:
            _log(f'WARNING: C-GAIMMD does not shoot from initial paths: their {n} frames in state '
                 f'{r} ({", ".join(files)}) are not used; initial paths seed free simulations '
                 f'only.')

    def _check_numerics(self, initial_paths):
        return check_start_up_numerics(self.params, self.generator, self.n_atoms, initial_paths)

    # -- the loop ------------------------------------------------------------------

    def next_shooting_point(self, *, chain, folder, path_name):
        """Generate, check and prepare candidates until one passes the final check.

        Parameters
        ----------
        chain : aimmd.pathensemble.PathEnsemble
            The chain being extended (the ``'bins'`` law's density adjustment).
        folder : str
            The chain folder (``worker.folder``).
        path_name : str
            The name the path is expected to be registered under, e.g. ``'path000012'``.

        Returns
        -------
        aimmd.cgaimmd.types.ShootingPoint or None
            The shooting point with its in-flight record; None if the worker must stop.

        Raises
        ------
        RuntimeError
            If :meth:`wait_for_network` has not succeeded yet.

        Notes
        -----
        Before every pop the watcher is refreshed and a new network resets the
        pool (its unused candidates count as ``'pool_reset'`` in
        :attr:`counters` and in the WARNING below), unless the pool holds a
        refill that has served no pop yet: then the reset waits until that
        refill has served :data:`STALE_REFILL_POPS` pops (stale pops). An
        empty pool is refilled (stage 1); after ``max_refills`` refills
        without a shooting point the source warns and waits ``wait_seconds``
        (then tries again, forever). A popped candidate goes through
        :func:`prepare_candidate`; the first that passes is returned.
        ``must_stop`` is checked before every pop, after each preparation
        and minimization hook, between generation batches and during waits.
        The candidates tried and rejected for a path are counted across calls
        for the same path (a read-back rejection makes ``_shoot`` ask again).
        """
        if self.settings is None:
            raise RuntimeError('GenerativeShootingPointSource.wait_for_network() must succeed '
                               'before next_shooting_point()')
        started = time.perf_counter()
        settings = self.settings
        self._chains[folder] = chain
        state = self._path_state(folder, path_name)
        max_refills = int(settings['max_refills'])
        refills_without_shot = 0
        since_wait = {}
        generated_since_wait = 0
        while True:
            if self._must_stop():
                return None
            self._check_network(since_wait)
            if not len(self.pool):
                if refills_without_shot >= max_refills:
                    _log(f'WARNING: no acceptable C-GAIMMD shooting point after '
                         f'{refills_without_shot} pool refills ({generated_since_wait} '
                         f'candidates: {_format_counts(since_wait)}); waiting '
                         f'{float(settings["wait_seconds"]):g} s')
                    if not self._wait():
                        return None
                    refills_without_shot, since_wait, generated_since_wait = 0, {}, 0
                    continue
                generated = self._refill(chain, folder, path_name, state, since_wait)
                if generated is None:
                    return None
                generated_since_wait += generated
                refills_without_shot += 1
                continue
            candidate = self.pool.pop()
            self._pops_since_refill += 1
            stale = self._stale is not None
            if stale:
                self._stale['pops_left'] -= 1
            point = self._stage2(candidate, folder, path_name, state, since_wait, started,
                                 stale=stale)
            if point is not None:
                return point

    def _check_network(self, since_wait):
        """Refresh the watcher; a new network resets the pool, its candidates count as 'pool_reset'.

        A refill that has served no pop when its network is replaced keeps the
        pool for :data:`STALE_REFILL_POPS` pops; the reset follows at the
        first check after them (or at once if the network changes again).
        """
        old = self.watcher.info
        if not self.watcher.refresh():
            stale = self._stale
            if stale is not None and (stale['pops_left'] <= 0 or not len(self.pool)):
                self._reset_pool(since_wait, stale['replaced'])
            return
        if (self._stale is None and STALE_REFILL_POPS > 0 and len(self.pool)
                and self._pops_since_refill == 0):
            self._stale = {'pops_left': int(STALE_REFILL_POPS), 'replaced': old}
            new = self.watcher.info
            pops = int(STALE_REFILL_POPS)
            refill = max(self._refills) if self._refills else None
            _log(f'C-GAIMMD pool: network changed {old.sha256[:8] if old else None} -> '
                 f'{new.sha256[:8]} (round {new.round}, {self.watcher.n_trained_shots()} trained '
                 f'shots) while refill {refill} was generated; its {len(self.pool)} candidates '
                 f'serve {pops} pop{"" if pops == 1 else "s"} before the pool resets')
            return
        self._reset_pool(since_wait, old)

    def _reset_pool(self, since_wait, old):
        """Reset the pool after a network change (`old`: the replaced network's info), logged."""
        self._stale = None
        discarded = self.pool.reset('network changed')
        self._refills.clear()
        if discarded:                      # the WARNING of an empty refill then says why
            _add(since_wait, 'pool_reset', discarded)
            self._count('pool_reset', discarded)
        new = self.watcher.info
        _log(f'C-GAIMMD pool reset: network changed {old.sha256[:8] if old else None} -> '
             f'{new.sha256[:8]} (round {new.round}, {self.watcher.n_trained_shots()} trained '
             f'shots); {discarded} unused candidates discarded')

    def _load_bins(self):
        """The trainer's bins and densities (timeout 0); None while they do not exist or do not match."""
        self._bins = self._densities = None
        states = self.params.sorted_states
        names = (f'{self.directory}/bins{states}.npy', f'{self.directory}/densities{states}.npy')
        if not all(os.path.exists(name) for name in names):
            return
        try:
            bins, densities = self.params.load_bins_and_densities(self.directory, timeout=0,
                                                                  raise_if_failure=True)
        except Exception:                                   # noqa: BLE001
            return
        bins = np.asarray(bins, dtype=float).reshape(-1)
        densities = np.asarray(densities, dtype=float).reshape(-1)
        if len(bins) >= 2 and len(densities) == len(bins) - 1:
            self._bins, self._densities = bins, densities

    # -- stage 1 -------------------------------------------------------------------

    def _generate(self, lam, device, generator, seed):
        """``(samples, batch)``: (n, M, 3) float64 Å, None after running out of memory, or _STOPPED.

        The refill is generated in batches of ``self._chunk`` targets (the whole
        refill when the generator has no ``chunk`` setting), each through
        ``self.generator.generate``; the torch generator is seeded with `seed`
        before the first batch of every attempt, so that a retry after running
        out of memory draws what a refill at the smaller batch size draws.
        """
        import torch
        settings = self.settings
        network = self.params.network
        n = len(lam)
        if n == 0:
            return np.zeros((0, self.n_atoms, 3)), self._chunk
        lam = torch.as_tensor(lam, dtype=torch.float32, device=device)
        halvings = int(settings['generation_oom_halvings'])
        while True:
            batch = self._chunk or n
            generator.manual_seed(int(seed))
            try:
                out = []
                for start in range(0, n, batch):
                    if start and self._must_stop():
                        return _STOPPED, batch
                    with torch.no_grad():
                        x = self.generator.generate(network, lam[start:start + batch],
                                                    generator=generator)
                    out.append(torch.as_tensor(x).detach().to('cpu', torch.float64).numpy())
            except Exception as exc:                        # noqa: BLE001
                if not _is_out_of_memory(exc):
                    raise
                _empty_cuda_cache()
                if halvings <= 0 or batch <= 1:
                    return None, batch
                smaller = max(batch // 2, 1)
                _log(f'C-GAIMMD generation ran out of CUDA memory (batch {batch}); retrying '
                     f'with batch {smaller}')
                self._chunk = smaller
                halvings -= 1
                continue
            samples = np.concatenate(out)
            if samples.shape != (n, self.n_atoms, 3):
                raise ValueError(f'the generator {self.generator.method!r} returned shape '
                                 f'{samples.shape}, expected ({n}, {self.n_atoms}, 3)')
            return samples, batch

    def _validity(self, function, positions, context):
        """Apply cgaimmd_validity_function; its (positions, ok, reasons, info), shape-checked."""
        n = len(positions)
        result = function(positions.copy(), context)
        if not isinstance(result, ValidityResult):
            raise TypeError(f'cgaimmd_validity_function must return an aimmd.cgaimmd.types.'
                            f'ValidityResult, got {type(result).__name__}')
        relabelled = np.asarray(result.positions, dtype=np.float64)
        ok = np.asarray(result.ok, dtype=bool).reshape(-1)
        reason = np.asarray(result.reason, dtype=object).reshape(-1)
        info = result.info
        if (relabelled.shape != positions.shape or ok.shape != (n,) or reason.shape != (n,)
                or (info is not None and len(info) != n)):
            raise ValueError(f'cgaimmd_validity_function returned positions {relabelled.shape}, '
                             f'ok {ok.shape}, reason {reason.shape} and info of length '
                             f'{None if info is None else len(info)} for {n} samples of shape '
                             f'{positions.shape}')
        return relabelled, ok, reason, info

    def _refill(self, chain, folder, path_name, state, since_wait):
        """Stage 1: the number of candidates generated, or None if the worker must stop."""
        import torch
        params = self.params
        network = params.network
        settings = self.settings
        started = time.perf_counter()
        index = state.refills
        state.refills += 1
        self._n_refills += 1
        refill = self._n_refills
        seed = refill_seed(self.base_seed, folder, path_name, index)
        n_targets = int(settings['n_targets'])
        per_target = int(settings['candidates_per_target'])
        n = n_targets * per_target
        law_rng, pop_rng, validity_rng, prep_seeds = refill_streams(seed, n)
        self._load_bins()
        # the law first: the 'bins' law's shared density adjustment and a cgaimmd_lambda_sampler
        # may call params.values_function, which puts the network on the values device
        device = self._device('generation')
        targets, law_record = draw_targets(
            n_targets,
            self._context(folder, path_name, law_rng, _seeded_generator(device, seed), device),
            chain=chain, shooting_chains=list(self._chains.values()))
        lam = np.repeat(np.asarray(targets, dtype=float), per_target)
        device = self._move('generation')
        network.eval()
        generator = _seeded_generator(device, seed)

        seconds = {'generate': 0.0, 'validity': 0.0, 'raw_lambda': 0.0}
        clock = time.perf_counter()
        samples, batch = self._generate(lam, device, generator, seed)
        seconds['generate'] = time.perf_counter() - clock
        if samples is _STOPPED:
            return None
        reasons = np.full(n, '', dtype=object)
        infos = [None] * n
        lambda_raw = np.full(n, np.nan)
        if samples is None:
            reasons[:] = 'generation_oom'
            positions = np.zeros((n, self.n_atoms, 3))
        else:
            positions = samples
            reasons[~np.isfinite(samples).reshape(n, -1).all(axis=1)] = 'non_finite'

        function = getattr(params, 'cgaimmd_validity_function', None)
        alive = np.flatnonzero(reasons == '')
        if function is not None and len(alive):
            clock = time.perf_counter()
            context = self._context(folder, path_name, validity_rng, generator, device)
            relabelled, ok, reason, info = self._validity(function, positions[alive], context)
            positions[alive] = relabelled
            reasons[alive[~ok]] = [_code(text, 'invalid') for text in reason[~ok]]
            if info is not None:
                for j, i in enumerate(alive):
                    infos[i] = info[j]
            seconds['validity'] = time.perf_counter() - clock

        alive = np.flatnonzero(reasons == '')
        if len(alive):
            clock = time.perf_counter()
            self._move('generation')                        # the validity hook may have moved it
            with torch.no_grad():
                values = coordinate_committor(
                    network, positions[alive],
                    descriptor_transform=getattr(params, 'descriptor_transform', None),
                    batch_size=getattr(params, 'network_batch_size', None))
            lambda_raw[alive] = values.detach().to('cpu', torch.float64).numpy()
            seconds['raw_lambda'] = time.perf_counter() - clock
            reasons[alive[~np.isfinite(lambda_raw[alive])]] = 'lambda_nonfinite'
            tolerance = settings['raw_lambda_tolerance']
            # the prefilter follows the final check's rule: off in the bootstrap phase
            enforced = (self.watcher.n_trained_shots()
                        >= int(settings['window_after_trained_shots']))
            if tolerance is not None and enforced:
                alive = np.flatnonzero(reasons == '')
                far = np.abs(lambda_raw[alive] - lam[alive]) > float(tolerance)
                reasons[alive[far]] = 'raw_lambda'

        survivors = np.flatnonzero(reasons == '')
        candidates = [Candidate(positions=positions[i].copy(), lambda_target=float(lam[i]),
                                lambda_raw=float(lambda_raw[i]), refill=refill, index=int(i),
                                refill_seed=seed, prep_seed=prep_seeds[i],
                                validity_info=infos[i], law_record=law_record)
                      for i in survivors]
        sha = self.watcher.info.sha256
        self.pool.extend(candidates, sha, pop_rng)
        self._pops_since_refill = 0
        rejected = {}
        for reason in reasons:
            if reason:
                _add(rejected, reason)
                _add(since_wait, reason)
                self._count(reason)
        # refills happen only when the pool is empty, so it holds one refill's candidates: keep
        # only that refill's bookkeeping (nothing for a refill without survivors)
        self._refills = {} if not candidates else {refill: {
            'path': path_name, 'index': index, 'size': n, 'n_targets': n_targets,
            'candidates_per_target': per_target, 'passed': len(survivors),
            'rejected': dict(sorted(rejected.items())), 'seconds': seconds, 'batch_size': batch,
            'device': str(device), 'network_sha256': sha}}
        _log(f'C-GAIMMD pool refill {refill} for {self._location(folder)}/{path_name}: {n} '
             f'candidates for {n_targets} targets (network {sha[:8]}, {law_record["name"]}); '
             f'{len(survivors)} passed stage 1 ({_format_counts(rejected)}) in '
             f'{time.perf_counter() - started:.2f} s')
        return n

    # -- stage 2 -------------------------------------------------------------------

    def _reject(self, state, since_wait, reason):
        _add(state.rejected, reason)
        _add(since_wait, reason)
        self._count(reason)

    def _stage2(self, candidate, folder, path_name, state, since_wait, started, stale=False):
        """:func:`prepare_candidate` of one popped candidate; its ShootingPoint or None.

        A frame a hook returns after ``must_stop`` became True (a GROMACS call
        it stopped, or one that finished just then) is neither counted as a
        rejection nor checked: None, and the loop returns. `stale`: the
        candidate is a stale pop, recorded in the sidecar.
        """
        refill = self._refills.get(candidate.refill, {})
        state.tried += 1
        device = self._device('generation')
        context = self._context(folder, path_name,
                                preparation_stream(candidate.refill_seed, candidate.index),
                                _seeded_generator(device, candidate.prep_seed), device,
                                seed=candidate.prep_seed)
        outcome = prepare_candidate(candidate.positions, candidate.lambda_target, context)
        state.prepare += outcome.seconds['prepare']
        state.minimize += outcome.seconds['minimize']
        state.final_check += outcome.seconds['final_check']
        if outcome.stopped:
            return None
        if not outcome.ok:
            self._reject(state, since_wait, outcome.reason)
            return None
        check = outcome.check
        record = self._record(candidate, refill, check, outcome.prepare, outcome.minimize, state,
                              folder, path_name, outcome.values_device, started, stale)
        info = self.watcher.info
        _log(f'C-GAIMMD shooting point for {self._location(folder)}/{path_name}: '
             f'lambda*={candidate.lambda_target:.3f} lambda={check["lambda"]:.3f} '
             f'({check["rule"]}), {state.tried} candidates tried '
             f'({_format_counts(state.rejected)}), network {info.sha256[:8]}, '
             f'{record["cost_seconds"]["total"]:.2f} s')
        frame = outcome.frame
        dimensions = None if frame.dimensions is None else np.asarray(frame.dimensions,
                                                                      dtype=np.float32)
        return ShootingPoint(positions=np.asarray(frame.positions, dtype=np.float32),
                             dimensions=dimensions, record=record)

    def _record(self, candidate, refill, check, prepare, minimize, state, folder, path_name,
                values_device, started, stale=False):
        """The in-flight record, without the keys registration adds."""
        params = self.params
        info = self.watcher.info
        provenance = self.provenance or {}
        validity = getattr(params, 'cgaimmd_validity_function', None)
        refill_seconds = refill.get('seconds', {})
        return {
            'schema': SIDECAR_SCHEMA,
            'source': SIDECAR_SOURCE,
            'note': None,
            'stale_record': None,
            'path_name': path_name,
            'chain': os.path.basename(os.path.normpath(folder)),
            'created': _utc_now(),
            'lambda_target': float(candidate.lambda_target),
            'lambda_raw': float(candidate.lambda_raw),
            'lambda_final': check['lambda'],
            'state_final': check['state'],
            'lambda_law': dict(candidate.law_record),
            'acceptance': {key: check[key] for key in (
                'rule', 'enforced', 'tolerance', 'window', 'window_after_trained_shots',
                'n_trained_shots')},
            'network': {'file': info.file, 'sha256': info.sha256, 'round': info.round,
                        'n_trained_shots': info.n_trained_shots,
                        'metadata_matched': info.metadata_matched,
                        'provenance': provenance.get('network')},
            'generator': {'method': self.generator.method,
                          'settings': dict(self._generator_settings),
                          'batch_size': refill.get('batch_size')},
            'pool': {'refill': candidate.refill, 'candidate': candidate.index,
                     'refill_size': refill.get('size'), 'n_targets': refill.get('n_targets'),
                     'candidates_per_target': refill.get('candidates_per_target'),
                     'passed_stage1': refill.get('passed'),
                     'pool_left_after_pop': len(self.pool), 'reset_count': self.pool.reset_count,
                     'stale_refill_pop': bool(stale),
                     'refill_network_sha256': refill.get('network_sha256')},
            'stage1_rejected': dict(refill.get('rejected', {})),
            'candidates': {'tried': state.tried, 'rejected': dict(sorted(state.rejected.items()))},
            'validity': {'function': None if validity is None else _function_name(validity),
                         'info': candidate.validity_info},
            'prepare': prepare,
            'minimize': minimize,
            'cost_seconds': {'generate': refill_seconds.get('generate', 0.0),
                             'validity': refill_seconds.get('validity', 0.0),
                             'raw_lambda': refill_seconds.get('raw_lambda', 0.0),
                             'prepare': state.prepare, 'minimize': state.minimize,
                             'final_check': state.final_check,
                             'total': time.perf_counter() - started},
            'seed': {'base': self.base_seed, 'refill': candidate.refill_seed,
                     'refill_path': refill.get('path'), 'refill_index': refill.get('index'),
                     'candidate_index': candidate.index, 'prep': candidate.prep_seed},
            'device': {'generation': refill.get('device'), 'values': str(values_device)},
            'aimmd': provenance.get('aimmd'),
            'software': provenance.get('software'),
            'bias': provenance.get('bias'),
        }

    # -- read-back rejections and registration ----------------------------------------

    def note_glitch(self, folder, exc, record):
        """Count a generated shooting point that AIMMD's read-back state check rejected.

        Parameters
        ----------
        folder : str
            The chain folder.
        exc : Exception
            The ``RuntimeError`` of the check (``'... should be in R'``).
        record : dict or None
            The shot's in-flight record (None: the path this folder is selecting).

        Raises
        ------
        RuntimeError
            If :meth:`wait_for_network` has not succeeded yet.

        Notes
        -----
        ``'state_readback'`` is counted in the stage-2 counters of the path
        (the next shooting point's sidecar reports it) and in
        :attr:`counters`; ``... C-GAIMMD: candidate discarded, next one from
        the pool`` is logged. After ``max_refills`` consecutive read-back
        rejections (reset by :meth:`note_registered` and by the wait) the
        source warns and waits ``wait_seconds``. The pool is not reset: the
        network did not change.

        A shot selected by an earlier process (resumed after a restart) also
        counts as tried, and when its refill was made for that path
        (``seed.refill_path``) the path's refill counter continues after the
        record's ``seed.refill_index``, so that the restarted worker does not
        regenerate the candidate that just failed. A record without a
        ``path_name`` (an unreadable in-flight marker) is counted for the
        next path the folder selects.
        """
        if self.settings is None:
            raise RuntimeError('GenerativeShootingPointSource.wait_for_network() must succeed '
                               'before note_glitch()')
        settings = self.settings
        name = os.path.basename(os.path.normpath(folder))
        record = record if isinstance(record, dict) else {}
        path_name = record.get('path_name') or self._current.get(name)
        state = self._paths.get((name, path_name))
        if state is None:                  # selected by an earlier process (a resumed shot)
            state = self._paths[(name, path_name)] = _PathState(
                tried=1, refills=_refills_after(record))
            self._current[name] = path_name
        _add(state.rejected, 'state_readback')
        self._count('state_readback')
        _log('... C-GAIMMD: candidate discarded, next one from the pool')
        self.consecutive_readback += 1
        if self.consecutive_readback >= int(settings['max_refills']):
            _log(f'WARNING: {self.consecutive_readback} consecutive C-GAIMMD shooting points '
                 f'failed the read-back state check (states_function on back.xtc frame 0); '
                 f'waiting {float(settings["wait_seconds"]):g} s')
            self._wait()
            self.consecutive_readback = 0

    def note_registered(self):
        """A C-GAIMMD shot was registered: reset the consecutive read-back counter."""
        self.consecutive_readback = 0


def make_cgaimmd_source(params, worker, directory):
    """The shooting-point source of a C-GAIMMD chain.

    Parameters
    ----------
    params : aimmd.Params
        The worker's params (``chain_type='cgaimmd'``).
    worker : aimmd.Worker
        The shooting worker (for ``must_stop``, ``initial_paths``, ``_location``).
    directory : str
        ``worker._directory``: the run directory relative to ``params.parent``, where the
        trainer writes the network.

    Returns
    -------
    GenerativeShootingPointSource

    Notes
    -----
    ``aimmd.worker._shoot`` calls it through this module at the time of use,
    so that RFPS and TPS runs import no C-GAIMMD code and tests can replace it
    (``tests._cgaimmd_helpers.ScriptedShootingPointSource``).
    """
    return GenerativeShootingPointSource(params, worker, directory)
