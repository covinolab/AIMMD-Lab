"""
aimmd.cgaimmd.protocol
======================

The duck-typed protocol of C-GAIMMD networks and the helpers that apply the
fallbacks of its optional members.

A C-GAIMMD network is the committor model of the run: its ``forward`` is
AIMMD's contract, unchanged (the logit committor toward
``params.sorted_states[2]``, shape ``(B, 1)``), and it carries the usual
torch module members. The protocol covers **the committor only**. How
shooting points are generated is a separate choice, a *generator*
(:mod:`aimmd.cgaimmd.generation`), picked by name through
``params.cgaimmd_generator`` or by the network's ``default_generator()``.
Every generation method checks for itself what it needs from the network
(for example the flow-matching ODE needs ``velocity`` and ``sample_prior``,
:class:`aimmd.cgaimmd.flow_matching.sampler.ODEGenerator`), so a network never
has to fake the members of a method it does not use.

There is no base class to inherit from: any object with the members below is
a C-GAIMMD network. C-GAIMMD code never calls an optional member directly; it
goes through the helper named in brackets, which applies the fallback.

Coordinates that cross this interface are in Ångström, ``(B, M, 3)`` with
``M`` the number of generated atoms.

torch is imported inside the functions, so importing this module is light.
"""

import inspect
import math
import sys
from collections.abc import Mapping
from numbers import Integral

import numpy as np


REQUIRED_MEMBERS = ('forward', 'state_dict', 'load_state_dict', 'parameters', 'to', 'eval',
                    'train')
"""Callable members every C-GAIMMD network must have (what AIMMD requires of any network).

``n_atoms`` is required as well (see :func:`check_network_protocol` for its
one fallback, kept for the networks of the old code and removed with the old
names).
"""

OPTIONAL_MEMBERS = ('committor', 'lambda_clip', 'reset_parameters', 'default_generator',
                    'provenance')
"""Members a C-GAIMMD network may have; each has a documented fallback."""

_MISSING = object()


def _bind(member, args, kwargs):
    """None if ``member(*args, **kwargs)`` binds (or cannot be inspected), else the error."""
    try:
        signature = inspect.signature(member)
    except (TypeError, ValueError):                        # builtins: cannot tell
        return None
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        return str(exc)
    return None


def _problems(network):
    """Every protocol violation of `network`, as a list of messages."""
    problems = []
    for name in REQUIRED_MEMBERS:
        try:
            member = getattr(network, name, None)
        except Exception as exc:                            # noqa: BLE001
            problems.append(f'{name}: reading it raised {type(exc).__name__}: {exc}')
            continue
        if member is None or not callable(member):
            problems.append(f'required member {name!r} is missing or not callable')
    if _forward_not_implemented(network):
        problems.append('forward is not implemented (the torch.nn.Module subclass does not '
                        'override forward)')

    n_atoms = getattr(network, 'n_atoms', _MISSING)
    if n_atoms is _MISSING or n_atoms is None:
        if not callable(getattr(network, 'sample_prior', None)):
            problems.append('required attribute n_atoms is missing (an int >= 1: the number of '
                            'atoms the network generates)')
    elif isinstance(n_atoms, bool) or not isinstance(n_atoms, Integral) or n_atoms < 1:
        problems.append(f'n_atoms must be an int >= 1, got {n_atoms!r}')

    optional = {
        'committor': (('x',), 'committor(x)'),
        'reset_parameters': ((), 'reset_parameters()'),
        'default_generator': ((), 'default_generator()'),
        'provenance': ((), 'provenance()'),
    }
    for name, (args, expected) in optional.items():
        member = getattr(network, name, None)
        if member is None:
            continue
        if not callable(member):
            problems.append(f'optional member {name!r} is set but not callable')
            continue
        error = _bind(member, args, {})
        if error is not None:
            problems.append(f'{name} must accept {expected} ({error})')

    lambda_clip = getattr(network, 'lambda_clip', _MISSING)
    if lambda_clip is not _MISSING and lambda_clip is not None:
        try:
            _positive_float(lambda_clip, 'network.lambda_clip')
        except TypeError as exc:
            problems.append(str(exc))
    return problems


def _forward_not_implemented(network):
    """True for a ``torch.nn.Module`` whose ``forward`` is still ``torch.nn.Module.forward``.

    torch is not imported here: a network can only be a Module if torch is
    already in ``sys.modules``.
    """
    module = getattr(getattr(sys.modules.get('torch'), 'nn', None), 'Module', None)
    if not isinstance(module, type) or not isinstance(network, module):
        return False
    unimplemented = getattr(module, 'forward', None)
    return unimplemented is not None and getattr(type(network), 'forward', None) is unimplemented


def is_cgaimmd_network(network):
    """Whether `network` is a C-GAIMMD network that names its own generation method.

    Parameters
    ----------
    network : object
        The candidate (usually ``params.network``).

    Returns
    -------
    bool
        True if :func:`check_network_protocol` passes **and** the network has
        a callable ``default_generator``; False otherwise (also for None).

    Notes
    -----
    Used to choose the default values function under every chain type: a
    C-GAIMMD network gets :class:`aimmd.cgaimmd.values.ProtocolValuesFunction`
    also on an RFPS or TPS run, so that post-training it needs no
    ``values_function``. ``default_generator`` is what sets such a network
    apart from an ordinary AIMMD committor network that happens to have the
    protocol members. A network without it can still be used under
    ``chain_type='cgaimmd'`` with an explicit ``params.cgaimmd_generator``
    (and gets the protocol values function there because of the chain type).
    A network whose members cannot even be inspected is not a C-GAIMMD
    network (no exception).
    """
    if network is None:
        return False
    try:
        return (not _problems(network)
                and callable(getattr(network, 'default_generator', None)))
    except Exception:                                       # noqa: BLE001
        return False


def check_network_protocol(network):
    """Check the static protocol of a C-GAIMMD network.

    Required members (all callable):

    - ``forward(inp) -> Tensor (B, 1)``: the logit committor toward
      ``sorted_states[2]`` of the input the values function builds (dense
      ``(B, 3 M)`` Å, or the graph dict of AIMMD's graph pipeline);
      differentiable in the positions.
    - ``state_dict()``, ``load_state_dict(sd)``, ``parameters()``: tensors
      only, loadable with ``torch.load(..., weights_only=True)``.
    - ``to(device)``, ``eval()``, ``train(mode)``: as ``torch.nn.Module``.

    Required attribute:

    - ``n_atoms``: int ``M`` >= 1, the number of atoms the network generates
      [:func:`n_atoms_of`]. Its one fallback, kept for the flow-matching
      networks of the old code and removed together with the old names
      (:mod:`aimmd.cgaimmd._legacy`): a network with ``sample_prior`` may
      leave it out, and ``M`` is then read from the shape of a prior sample.
      New networks set ``n_atoms``.

    Optional members and the fallback C-GAIMMD code uses without them (the
    helper of this module that applies it in brackets):

    - ``committor(x) -> Tensor (B,)``: fast path for coordinates not on disk;
      must equal ``forward`` [:func:`coordinate_committor`: ``forward`` of the
      values-function input of the flat coordinates].
    - ``lambda_clip``: float ``L`` > 0, the conditioning range
      [:func:`lambda_clip_of`: ``cgaimmd_parameters['lambda_clip']``, then
      required].
    - ``reset_parameters()``: re-initializes the weights; used by fits that
      train from reset every round, which construct a new network without it.
    - ``default_generator()``: the generation method used when
      ``params.cgaimmd_generator`` is None, as a method name, a settings dict
      with ``'method'`` or a generator object
      (:func:`aimmd.cgaimmd.generation.make_generator`). Without it,
      ``cgaimmd_generator`` must be set.
    - ``provenance()``: a JSON-serializable dict of what the network's
      results depend on beyond its weights (for example a frozen pretrained
      model: its checkpoint, the snapshot it came from, a parameter
      checksum); the shooting worker calls it once at start-up and writes it
      into every sidecar (``network.provenance``). Without it the sidecar
      records None there.

    Generation members (``velocity``, ``sample_prior``, ``generate``, ...)
    are not part of this protocol: each generation method checks the members
    it needs (:meth:`aimmd.cgaimmd.generation.Generator.check`).

    Parameters
    ----------
    network : object
        The candidate C-GAIMMD network (usually ``params.network``).

    Raises
    ------
    TypeError
        Listing every problem at once.

    Notes
    -----
    Checks: the required members exist and are callable; a
    ``torch.nn.Module`` overrides ``forward``; ``n_atoms`` is an int >= 1
    (or absent on a network with ``sample_prior``); present optional members
    bind their arguments (``committor``: one; ``reset_parameters``,
    ``default_generator`` and ``provenance``: none); ``lambda_clip`` converts to a finite float
    > 0. ``**kwargs`` satisfies any keyword.
    """
    problems = _problems(network) if network is not None else ['network is None']
    if problems:
        raise TypeError(f'{type(network).__name__} does not implement the C-GAIMMD network '
                        f'protocol (aimmd.cgaimmd.protocol):\n  - ' + '\n  - '.join(problems))


def network_device(network):
    """The device of the network's first parameter, CPU if it has none.

    Callers put inputs there *at call time* (the network may have been moved
    since the last call) and get outputs there.
    """
    import torch
    try:
        first = next(iter(network.parameters()))
    except StopIteration:
        return torch.device('cpu')
    return first.device


def _positive_float(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f'{what} must be a finite float > 0, got {value!r}') from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise TypeError(f'{what} must be a finite float > 0, got {value!r}')
    return number


def n_atoms_of(network):
    """The number ``M`` of atoms `network` generates.

    Parameters
    ----------
    network : object
        A C-GAIMMD network.

    Returns
    -------
    int
        ``network.n_atoms`` when present, else ``network.sample_prior(None, 1).shape[1]``
        (the fallback of flow-matching networks).

    Raises
    ------
    TypeError
        If ``network.n_atoms`` is present but not an int >= 1, or the network
        has neither ``n_atoms`` nor ``sample_prior``.

    Notes
    -----
    The fallback draws under ``torch.no_grad`` with a private
    ``torch.Generator`` (seed 0) on the network's device, so the global torch
    random stream is not consumed; exceptions of ``sample_prior`` propagate
    (:func:`check_network_numerics` reports them as ``ValueError``).
    """
    value = getattr(network, 'n_atoms', None)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise TypeError(f'n_atoms must be an int >= 1, got {value!r}')
        return int(value)
    if not callable(getattr(network, 'sample_prior', None)):
        raise TypeError(f'{type(network).__name__} has no n_atoms (an int >= 1: the number '
                        f'of atoms the network generates)')
    import torch
    generator = torch.Generator(device=network_device(network))
    generator.manual_seed(0)
    with torch.no_grad():
        sample = network.sample_prior(None, 1, generator=generator)
    return int(sample.shape[1])


def lambda_clip_of(network, settings=None):
    """The conditioning range ``L`` of `network`.

    Parameters
    ----------
    network : object
        A C-GAIMMD network.
    settings : mapping or None, optional
        C-GAIMMD settings (``params.cgaimmd_settings``), used when the network has no
        ``lambda_clip``.

    Returns
    -------
    float
        ``network.lambda_clip`` if present (it wins), else ``settings['lambda_clip']``.

    Raises
    ------
    TypeError
        If neither gives a value, a value is not a finite float > 0, or the two disagree.

    Notes
    -----
    Two given values disagree when they differ by more than 1e-6 relative
    (float32 buffers are compared with that tolerance).
    """
    own = getattr(network, 'lambda_clip', None)
    setting = None if settings is None else settings.get('lambda_clip')
    if own is not None:
        value = _positive_float(own, 'network.lambda_clip')
        if setting is not None:
            other = _positive_float(setting, "cgaimmd_parameters['lambda_clip']")
            if not math.isclose(value, other, rel_tol=1e-6, abs_tol=0.0):
                raise TypeError(f"cgaimmd_parameters['lambda_clip'] = {setting!r} differs from "
                                f"the network's lambda_clip = {value!r}; drop the setting "
                                f"(the network's value is used) or make them equal")
        return value
    if setting is None:
        raise TypeError(f"{type(network).__name__} has no lambda_clip attribute; set "
                        f"cgaimmd_parameters['lambda_clip'] (the conditioning range L > 0)")
    return _positive_float(setting, "cgaimmd_parameters['lambda_clip']")


def _call_forward(network, inp):
    """``network(inp)`` for callables (torch modules run their hooks), else ``forward``."""
    return network(inp) if callable(network) else network.forward(inp)


def _batch_dict_on(batch, device):
    """A batch dict with its tensors moved to `device` (other values unchanged)."""
    import torch
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


def _values_input(rows, device, descriptor_transform):
    """The input AIMMD's values function builds from descriptor rows.

    Dense mode (no transform): the rows as a flattened float32 tensor. With a
    transform, its output: a batch dict (``Mapping``, e.g. the node-table
    featurizer's ``batch_dict``) is used as it is with its tensors moved to
    `device`; a list of PyG ``Data`` is batched with
    ``Batch.from_data_list(...).to(device).to_dict()``; anything else is
    turned into a flattened float32 tensor.
    """
    import torch
    if descriptor_transform is None:
        x = torch.as_tensor(rows, dtype=torch.float32, device=device)
        return torch.flatten(x, start_dim=1)
    data = descriptor_transform(rows)
    if isinstance(data, Mapping):                          # graph mode: a ready batch dict
        return _batch_dict_on(data, device)
    if isinstance(data, list):                            # graph mode: a list of PyG Data
        from torch_geometric.data import Batch
        return Batch.from_data_list(data).to(device).to_dict()
    x = torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)
    return torch.flatten(x, start_dim=1)


def coordinate_committor(network, x, *, descriptor_transform=None, batch_size=None):
    """Committor logits of coordinates that are not on disk.

    Parameters
    ----------
    network : object
        A C-GAIMMD network.
    x : numpy.ndarray or torch.Tensor
        ``(B, M, 3)`` coordinates in Å (fresh candidates, in-memory frames).
    descriptor_transform : callable, optional
        ``params.descriptor_transform``, used only by the fallback; None: dense mode.
    batch_size : int, optional
        Evaluate in chunks of this many frames (None: one call).

    Returns
    -------
    torch.Tensor
        ``(B,)`` float32 logits on :func:`network_device`.

    Notes
    -----
    ``network.committor(x)`` when present; else ``forward`` of the input the
    values function would build from ``x.reshape(B, 3 M)`` (the flat
    descriptors of C-GAIMMD): the dense float32 tensor without a
    `descriptor_transform`; the transform's batch dict, or
    ``Batch.from_data_list(...).to(device).to_dict()`` when it returns a list
    of PyG ``Data`` (graph mode); otherwise the transform's output as a
    flattened float32 tensor.

    Autograd is as the caller's grad mode allows: scoring callers wrap the call
    in ``torch.no_grad()``; training callers get gradients through
    ``committor`` and the dense fallback (not through a graph transform, which
    works on numpy rows).

    An empty batch (``B = 0``, e.g. a refill without survivors) returns a
    ``(0,)`` tensor without calling the network.
    """
    import torch
    device = network_device(network)
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError(f'batch_size must be an int >= 1, got {batch_size!r}')
        batch_size = int(batch_size)
    n = len(x)
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if batch_size is None or batch_size >= n:
        chunks = [(0, n)]
    else:
        chunks = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    committor = getattr(network, 'committor', None)
    out = []
    for start, stop in chunks:
        part = x[start:stop]
        if callable(committor):
            values = committor(torch.as_tensor(part, dtype=torch.float32, device=device))
        else:
            if descriptor_transform is None:
                rows = part                    # _values_input: float32 tensor, flattened
            else:
                rows = part.detach().cpu().numpy() if torch.is_tensor(part) else np.asarray(part)
                rows = rows.reshape(rows.shape[0], int(np.prod(rows.shape[1:])))
            values = _call_forward(network, _values_input(rows, device, descriptor_transform))
            values = values[:, 0]
        out.append(values.reshape(-1).to(torch.float32))
    return out[0] if len(out) == 1 else torch.cat(out)


def check_network_numerics(network, coordinates, forward_input=None, *, generation=None,
                           atol=1e-4, rtol=1e-4, n_generate=2):
    """Numerical start-up checks of a C-GAIMMD network and its generator.

    Parameters
    ----------
    network : object
        A C-GAIMMD network that passed :func:`check_network_protocol`.
    coordinates : array-like
        ``(B, M, 3)`` Å frames (a few MD frames).
    forward_input : optional
        The values-function input of the same frames (dense tensor or array, or a graph dict).
    generation : Generator, str or dict, optional
        The generator to draw through (:mod:`aimmd.cgaimmd.generation`): a
        generator object, or a method name or settings dict that
        :func:`~aimmd.cgaimmd.generation.make_generator` builds for `network`.
        None (default): no generation check. The start-up check of a shooting
        worker passes the generator it shoots with.
    atol, rtol : float, optional
        Tolerance of ``|committor - forward| <= atol + rtol |forward|``.
    n_generate : int, optional
        Number of candidates drawn through `generation`.

    Returns
    -------
    dict
        The measured numbers (keys listed in the notes).

    Raises
    ------
    ValueError
        Listing every problem found (see the notes).

    Notes
    -----
    Checks: ``forward(forward_input)`` is ``(B, 1)`` finite; a ``committor``
    returns ``(B,)`` finite and equals ``forward``; one tiny call
    ``generation.generate(network, lam)`` with ``lam`` evenly spaced in
    ``[-1, 1]`` returns ``(n_generate, M, 3)`` finite Å; every tensor is on
    the network's device. A problem is a wrong shape, a non-finite value, a
    tensor off :func:`network_device`, a ``committor`` that differs from
    ``forward``, or an exception raised by the network or the generator
    (reported with its message). The generation call runs only when the
    committor checks found no problem.

    With ``forward_input=None`` the ``forward`` call and the comparison are
    skipped; ``committor`` (if present) is still checked on `coordinates`.
    The network runs in eval mode under ``torch.no_grad`` and its training
    flag is restored; random draws use a private generator (seed 0), so the
    global random stream is untouched. An exception while finding the
    network's device or ``n_atoms`` (a flow-matching network without
    ``n_atoms`` draws from its prior) is reported the same way, as the only
    problem; so is a `generation` spec that cannot be built.

    The returned dict holds ``n_atoms``, ``n_frames``, ``device``,
    ``forward_range``, ``committor_max_abs_diff`` (None when not compared),
    ``generation`` (``{'method', 'settings'}``, None without a generation
    check), ``sample_shape`` and ``sample_abs_max``.
    """
    import torch

    def failed(problems):
        return ValueError(f'{type(network).__name__} failed the C-GAIMMD network checks:\n  - '
                          + '\n  - '.join(problems))

    try:
        device = network_device(network)
    except Exception as exc:                                # noqa: BLE001
        raise failed([f'parameters() raised {type(exc).__name__}: {exc}']) from exc
    try:
        n_atoms = n_atoms_of(network)
    except Exception as exc:                                # noqa: BLE001
        what = ('n_atoms_of (sample_prior(None, 1))' if getattr(network, 'n_atoms', None) is None
                else 'n_atoms_of (network.n_atoms)')
        raise failed([f'{what} raised {type(exc).__name__}: {exc}']) from exc
    if generation is not None:
        from .generation import is_generator, make_generator
        if not is_generator(generation):
            try:
                generation = make_generator(generation, network)
            except Exception as exc:                        # noqa: BLE001
                raise failed([f'the generator {generation!r} could not be built: '
                              f'{type(exc).__name__}: {exc}']) from exc
    problems = []
    coords = torch.as_tensor(np.asarray(coordinates) if not torch.is_tensor(coordinates)
                             else coordinates, dtype=torch.float32, device=device)
    if coords.ndim != 3 or coords.shape[1:] != (n_atoms, 3):
        raise ValueError(f'coordinates must have shape (B, {n_atoms}, 3) for a network with '
                         f'n_atoms={n_atoms}, got {tuple(coords.shape)}')
    n_frames = int(coords.shape[0])
    report = {'n_atoms': n_atoms, 'n_frames': n_frames, 'device': str(device),
              'forward_range': None, 'committor_max_abs_diff': None, 'generation': None,
              'sample_shape': None, 'sample_abs_max': None}

    raised = object()                   # what `attempt` returns when the call raised

    def inspect_tensor(what, value, shape):
        """Shape, then device, then finiteness; returns the tensor if all hold."""
        if value is raised:             # already reported by `attempt`
            return None
        if not torch.is_tensor(value):
            problems.append(f'{what} returned {type(value).__name__}, expected a tensor')
            return None
        if tuple(value.shape) != tuple(shape):
            problems.append(f'{what} returned shape {tuple(value.shape)}, expected '
                            f'{tuple(shape)}')
            return None
        if value.device != device:
            problems.append(f'{what} returned a tensor on device {value.device}, expected the '
                            f'network device {device}')
            return None
        if not bool(torch.isfinite(value).all()):
            problems.append(f'{what} returned non-finite values')
            return None
        return value

    def attempt(what, call):
        try:
            return call()
        except Exception as exc:                            # noqa: BLE001
            problems.append(f'{what} raised {type(exc).__name__}: {exc}')
            return raised

    was_training = getattr(network, 'training', None)
    try:
        attempt('eval()', network.eval)
        with torch.no_grad():
            # 1. forward and committor
            forward = None
            if forward_input is not None:
                inp = forward_input
                if isinstance(inp, Mapping):
                    inp = _batch_dict_on(inp, device)
                else:
                    inp = torch.as_tensor(np.asarray(inp) if not torch.is_tensor(inp) else inp,
                                          dtype=torch.float32, device=device)
                    inp = torch.flatten(inp, start_dim=1)
                out = attempt('forward(forward_input)', lambda: _call_forward(network, inp))
                forward = inspect_tensor('forward(forward_input)', out, (n_frames, 1))
                if forward is not None:
                    report['forward_range'] = [float(forward.min()), float(forward.max())]
            committor = getattr(network, 'committor', None)
            if callable(committor):
                out = attempt('committor(coordinates)', lambda: committor(coords))
                values = inspect_tensor('committor(coordinates)', out, (n_frames,))
                if values is not None and forward is not None:
                    diff = (values - forward[:, 0]).abs()
                    report['committor_max_abs_diff'] = float(diff.max()) if n_frames else 0.0
                    bound = atol + rtol * forward[:, 0].abs()
                    if bool((diff > bound).any()):
                        problems.append(
                            f'committor differs from forward on the same frames by up to '
                            f'{float(diff.max()):.3g} (tolerance atol={atol}, rtol={rtol})')

            # 2. one tiny generation through the generator the caller uses
            if generation is not None and not problems:
                what = f'the {generation.method!r} generator'
                settings = attempt(f'{what}: settings()', generation.settings)
                report['generation'] = {'method': generation.method,
                                        'settings': None if settings is raised else settings}
                random = torch.Generator(device=device)
                random.manual_seed(0)
                lam = torch.linspace(-1.0, 1.0, int(n_generate), device=device)
                x = attempt(f'{what}: generate(network, lam)',
                            lambda: generation.generate(network, lam, generator=random))
                x = inspect_tensor(f'{what}: generate(network, lam)', x,
                                   (int(n_generate), n_atoms, 3))
                if x is not None:
                    report['sample_shape'] = list(x.shape)
                    report['sample_abs_max'] = float(x.abs().max()) if len(x) else 0.0
    finally:
        if was_training is not None:
            attempt('train(mode)', lambda: network.train(bool(was_training)))
    if problems:
        raise failed(problems)
    return report
