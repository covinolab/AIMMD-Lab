"""
aimmd.cgaimmd.generation
========================

How C-GAIMMD generates candidate shooting points: one interface for every
generation method.

A *generator* turns target committors into candidate configurations with a
C-GAIMMD network. The method is picked by name, so that BioEmu2's diffusion
sampler, the flow-matching ODE and later methods are interchangeable:

``'ode'``
    :class:`aimmd.cgaimmd.flow_matching.sampler.ODEGenerator`: integrates the
    network's flow-matching velocity field from its prior (PaiNN networks,
    the toy network).
``'bioemu2_edm'``
    BioEmu2's EDM sampler with the committor adapters
    (``aimmd.cgaimmd.bioemu2``, which needs the ``bioemu2`` extra).

A generator is any object with the members of :class:`Generator`. The network
protocol (:mod:`aimmd.cgaimmd.protocol`) covers the committor only; what a
method needs from the network is checked by the generator itself
(:meth:`Generator.check`), so a network never fakes the members of a method it
does not use.

Which generator a run uses is set by ``params.cgaimmd_generator``: a method
name, a settings dict ``{'method': ..., **settings}``, a generator object, or
None, in which case the network's ``default_generator()`` decides.
:func:`make_generator` builds it. New methods are added with
:func:`register_generator`:

>>> from aimmd.cgaimmd.generation import register_generator, make_generator
>>> class Shifted:
...     method = 'shifted_prior'
...     def __init__(self, shift=1.0):
...         self.shift = float(shift)
...     def check(self, network):
...         return [] if callable(getattr(network, 'sample_prior', None)) else [
...             'shifted_prior needs sample_prior(lam, n, *, generator=None)']
...     def generate(self, network, lam, *, generator=None):
...         return network.sample_prior(lam, len(lam), generator=generator) + self.shift
...     def settings(self):
...         return {'shift': self.shift}
>>> register_generator('shifted_prior', Shifted)          # doctest: +SKIP
>>> make_generator({'method': 'shifted_prior', 'shift': 2.0}).settings()  # doctest: +SKIP
{'shift': 2.0}

The registry is per process. A generator registered in a module that the
params file imports (or in the params file itself) is registered in every
worker that loads the params file; loading the params file again registers
the new definition of the same class without complaint. The built-in methods
are registered lazily: their name is known from the start and their module is
imported on first use.

Standard library only at import time.
"""

import importlib
import json
import re
import threading
from typing import Protocol, runtime_checkable


GENERATOR_MEMBERS = ('check', 'generate', 'settings')
"""Callable members of every generator (besides the string attribute ``method``)."""


@runtime_checkable
class Generator(Protocol):
    """The interface of a generation method (structural: nothing to inherit from).

    Attributes
    ----------
    method : str
        The registry name of the method; written to every sidecar.

    Notes
    -----
    - ``check(network) -> list[str]``: what this method needs from `network`
      and does not find, one message per problem; ``[]`` if the network can be
      used. It must not raise for a network that lacks something, and it is
      cheap (no generation).
    - ``generate(network, lam, *, generator=None) -> torch.Tensor``: `lam` is a
      ``(B,)`` tensor of target logit committors on the network's device;
      returns ``(B, M, 3)`` coordinates in Ångström on the network's device,
      one candidate per entry of `lam`, without autograd history. `generator`
      is the ``torch.Generator`` of the random stream (on the network's
      device). How the batch is chunked is the generator's own setting.
    - ``settings() -> dict``: the generator's settings, JSON-serializable;
      written to every sidecar together with ``method``, and
      ``make_generator({'method': method, **settings()})`` builds an equal
      generator.

    ``isinstance(obj, Generator)`` only tests that the members exist;
    :func:`is_generator` also tests their types and refuses classes.
    """

    method: str

    def check(self, network) -> list:
        """What this method needs from `network` and does not find ([] if usable)."""

    def generate(self, network, lam, *, generator=None):
        """``(B,)`` target logit committors -> ``(B, M, 3)`` candidates in Ångström."""

    def settings(self) -> dict:
        """The JSON-serializable settings of this generator."""


_LAZY = {
    'ode': ('aimmd.cgaimmd.flow_matching.sampler', 'ODEGenerator', None),
    'bioemu2_edm': ('aimmd.cgaimmd.bioemu2.sampling', 'BioEmu2Generator', 'bioemu2'),
}
"""Built-in methods registered on first use: method -> (module, factory name, pip extra).

The module also registers its method when it is imported; the entry lets
:func:`make_generator` find the factory without relying on that. The pip
extra is None for methods of the core install.
"""

_FACTORIES = {}
_LOCK = threading.RLock()
_METHOD_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_.-]*')


def register_generator(method, factory, *, replace=False):
    """Make a generation method available by name.

    Parameters
    ----------
    method : str
        The name (letters, digits, ``'_'``, ``'.'``, ``'-'``; starting with a
        letter). It is what ``params.cgaimmd_generator`` names, and the
        generators the factory builds should carry it as their ``method``.
    factory : callable
        ``factory(**settings) -> Generator``; usually the generator class.
        It raises ``TypeError`` or ``ValueError`` for settings it does not
        accept.
    replace : bool, optional
        Allow replacing another factory already registered under `method`.

    Returns
    -------
    callable
        `factory`.

    Raises
    ------
    ValueError
        For an invalid name, or a name already registered with another
        factory (unless `replace`).
    TypeError
        If `factory` is not callable.

    Notes
    -----
    Registering the same factory again does nothing, and a new definition of
    the registered factory (same name, defined in the same file) replaces it
    without `replace`: ``Params.load`` executes the params file and the
    modules next to it again on every load, and ``importlib.reload`` does the
    same for a module, so their ``register_generator`` calls run again with a
    class that is new but means the same.
    """
    if not (isinstance(method, str) and _METHOD_NAME.fullmatch(method)):
        raise ValueError(f'a generation method name must be a str of letters, digits, '
                         f"'_', '.' or '-' starting with a letter, got {method!r}")
    if not callable(factory):
        raise TypeError(f'the factory of generation method {method!r} must be callable, '
                        f'got {type(factory).__name__}')
    with _LOCK:
        known = _FACTORIES.get(method)
        if known is not None and not replace and not _same_definition(known, factory):
            raise ValueError(f'generation method {method!r} is already registered '
                             f'({_describe(known)}); pass replace=True to replace it')
        _FACTORIES[method] = factory
    return factory


def _register_builtin(method, factory):
    """Register a built-in method when its module is imported.

    Unlike :func:`register_generator`, it never raises and never replaces a
    factory of another definition: a user who replaced the built-in method
    before its module was first imported keeps the replacement, and the
    module still imports. A new definition of the built-in factory itself
    (``importlib.reload``) replaces the old one.
    """
    with _LOCK:
        known = _FACTORIES.get(method)
        if known is None or _same_definition(known, factory):
            _FACTORIES[method] = factory


def _definition(factory):
    """What defines `factory`: (its module, without a folder; its qualified name).

    ``Params.load`` renames the params file and its local modules after
    executing them (``params`` becomes ``/run/folder/params``), so the same
    class has another ``__module__`` on the next load; the folder is dropped
    as ``Params.save`` drops it.
    """
    module = getattr(factory, '__module__', None)
    if isinstance(module, str):
        module = re.split(r'[\\/]', module)[-1]
    return module, getattr(factory, '__qualname__', None)


def _same_definition(known, factory):
    """Whether `factory` is `known` or a new definition of it (see :func:`register_generator`).

    A factory defined inside a function (``<locals>`` in its name) is only
    the same as itself.
    """
    if known is factory:
        return True
    module, name = _definition(factory)
    return (module is not None and isinstance(name, str) and '<locals>' not in name
            and (module, name) == _definition(known))


def unregister_generator(method):
    """Forget a registered method (for tests and notebooks).

    A built-in method stays known and is registered again on its next use.
    An unknown name is ignored.
    """
    with _LOCK:
        _FACTORIES.pop(method, None)


def available_generators():
    """The names of every known generation method, sorted.

    Returns
    -------
    list of str
        The registered methods and the built-in ones that are registered on
        first use (whose module may need a pip extra, e.g. ``'bioemu2_edm'``).
    """
    with _LOCK:
        return sorted({*_FACTORIES, *_LAZY})


def is_known_generator(method):
    """Whether `method` names a registered or a built-in generation method (imports nothing)."""
    with _LOCK:
        return isinstance(method, str) and (method in _FACTORIES or method in _LAZY)


def is_generator(value):
    """Whether `value` has the members of :class:`Generator` (a str ``method``, callable rest).

    A class is not a generator, even when it defines the members: its
    instances are.
    """
    if isinstance(value, type):
        return False
    try:
        return (isinstance(getattr(value, 'method', None), str)
                and all(callable(getattr(value, name, None)) for name in GENERATOR_MEMBERS))
    except Exception:                                       # noqa: BLE001
        return False


def _is_generator_class(value):
    """Whether `value` is a class whose instances would be generators (a likely slip)."""
    return (isinstance(value, type) and isinstance(getattr(value, 'method', None), str)
            and all(callable(getattr(value, name, None)) for name in GENERATOR_MEMBERS))


def _class_given(value):
    """The TypeError for a generator class given where an instance belongs."""
    return TypeError(f'{value.__name__} is a generator class; give an instance, e.g. '
                     f'{value.__name__}(), or its method name {value.method!r} once it is '
                     f'registered')


def _describe(factory):
    module = getattr(factory, '__module__', None)
    name = getattr(factory, '__qualname__', None) or type(factory).__name__
    return f'{module}.{name}' if module else name


def _factory(method):
    """The factory of `method`, importing the module of a built-in method on first use."""
    with _LOCK:
        factory = _FACTORIES.get(method)
    if factory is not None:
        return factory
    lazy = _LAZY.get(method)
    if lazy is None:
        raise ValueError(f'unknown generation method {method!r}; known methods are '
                         f'{available_generators()} (add one with '
                         f'aimmd.cgaimmd.generation.register_generator)')
    module, name, extra = lazy
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        missing = getattr(exc, 'name', None) or ''
        if missing.split('.')[0] == 'aimmd':
            # aimmd's own module is missing or broken: no extra would help
            need = (f'the module {missing} is not part of this AIMMD-Lab installation'
                    + (f" (the method also needs the {extra!r} extra)" if extra else ''))
        elif extra:
            need = f"it needs the {extra!r} extra (pip install 'aimmd-lab[{extra}]')"
        else:
            need = 'it needs a complete AIMMD-Lab installation'
        raise ImportError(f'generation method {method!r} is provided by the module {module}, '
                          f'which could not be imported ({type(exc).__name__}: {exc}); '
                          f'{need}') from exc
    factory = getattr(imported, name, None)
    if factory is None:
        raise RuntimeError(f'the module {module} has no {name}, the factory of the generation '
                           f'method {method!r}')
    with _LOCK:
        return _FACTORIES.setdefault(method, factory)


def _build(method, settings):
    """``factory(**settings)`` of `method`, with the method named in errors."""
    factory = _factory(method)
    try:
        generator = factory(**settings)
    except (TypeError, ValueError) as exc:
        kind = TypeError if isinstance(exc, TypeError) else ValueError
        raise kind(f'generation method {method!r} with settings {settings!r}: '
                   f'{exc}') from exc
    if not is_generator(generator):
        raise TypeError(f'the factory of generation method {method!r} ({_describe(factory)}) '
                        f'returned {type(generator).__name__}, which lacks the generator '
                        f'members (method, check, generate, settings)')
    # sidecars record generator.method, and make_generator rebuilds from it
    if generator.method != method:
        raise TypeError(f'the factory of generation method {method!r} ({_describe(factory)}) '
                        f'built a generator whose method is {generator.method!r}; a factory '
                        f'must build generators of the method it is registered under')
    return generator


def _default_spec(network):
    """``network.default_generator()``, or a TypeError naming ``params.cgaimmd_generator``."""
    if network is None:
        raise TypeError('no generation method given: set params.cgaimmd_generator to a method '
                        f'name or settings dict (known methods: {available_generators()}), '
                        'or pass the network whose default_generator() decides')
    default = getattr(network, 'default_generator', None)
    if not callable(default):
        raise TypeError(f'{type(network).__name__} has no default_generator(), so '
                        f'params.cgaimmd_generator must name the generation method, as a '
                        f"name or as a settings dict {{'method': name, ...}} (known methods: "
                        f'{available_generators()}; each one checks that it can use the '
                        f'network)')
    spec = default()
    if spec is None:
        raise TypeError(f'{type(network).__name__}.default_generator() returned None, so '
                        f'params.cgaimmd_generator must name the generation method '
                        f'(known methods: {available_generators()})')
    if isinstance(spec, dict) and 'method' not in spec:
        raise TypeError(f'{type(network).__name__}.default_generator() returned a dict '
                        f"without 'method': {spec!r}")
    return spec


def make_generator(spec, network=None):
    """Build the generator of a ``params.cgaimmd_generator`` value.

    Parameters
    ----------
    spec : str, dict, Generator or None
        - a method name, e.g. ``'ode'``: the method with its default settings;
        - a dict ``{'method': name, **settings}``: the method with these
          settings;
        - a dict without ``'method'``: settings for the network's default
          method, overriding the settings of ``network.default_generator()``
          (this is what the old ``flow_parameters['generate_chunk']`` becomes);
        - a generator object (see :class:`Generator`): returned as it is;
        - None: ``network.default_generator()`` decides.
    network : object, optional
        The C-GAIMMD network; needed when `spec` is None or a dict without
        ``'method'``. The generator is not checked against it here: use
        :func:`check_generator`.

    Returns
    -------
    Generator

    Raises
    ------
    TypeError
        For a spec of another type (a generator class instead of an
        instance, too), or when the network has to decide and has no
        ``default_generator()`` (the message names
        ``params.cgaimmd_generator``); for a factory that builds generators
        of another method than its registered name; also for settings the
        method rejects as ``TypeError`` (with the method named).
    ValueError
        For an unknown method name, or settings the method rejects as
        ``ValueError`` (with the method named).
    ImportError
        When the module of a built-in method cannot be imported (e.g.
        ``'bioemu2_edm'`` without the bioemu2 extra); the message names the
        module and the extra.
    """
    if is_generator(spec):
        return spec
    if _is_generator_class(spec):
        raise _class_given(spec)
    if spec is None:
        return make_generator(_default_spec(network))
    if isinstance(spec, str):
        return _build(spec, {})
    if isinstance(spec, dict):
        settings = dict(spec)
        method = settings.pop('method', None)
        if method is None:
            default = _default_spec(network)
            if is_generator(default):
                if not is_known_generator(default.method):
                    raise TypeError(
                        f'{type(network).__name__}.default_generator() returned a generator '
                        f'of the unregistered method {default.method!r}, so the settings '
                        f'{settings!r} cannot be applied to it; register the method '
                        f'(aimmd.cgaimmd.generation.register_generator) or give the whole '
                        f"spec {{'method': ..., **settings}}")
                base = {'method': default.method, **default.settings()}
            elif isinstance(default, str):
                base = {'method': default}
            else:
                base = dict(default)
            return make_generator({**base, **settings})
        if not isinstance(method, str):
            raise TypeError(f"the 'method' of a generator settings dict must be a str, "
                            f'got {method!r}')
        return _build(method, settings)
    raise TypeError(f'a generator spec must be a method name, a settings dict, a generator '
                    f'object or None, got {type(spec).__name__}: {spec!r}')


def check_generator(generator, network):
    """Every problem of using `generator` with `network`, as a list of messages.

    Parameters
    ----------
    generator : object
        A generator (built with :func:`make_generator`).
    network : object
        The C-GAIMMD network it will generate with.

    Returns
    -------
    list of str
        Empty if the generator has the :class:`Generator` members, its
        ``settings()`` is a JSON-serializable dict and ``generator.check(network)``
        reports nothing; else one message per problem (an exception raised by
        ``check`` or ``settings`` is reported, not raised).
    """
    if not is_generator(generator):
        missing = [name for name in ('method',) + GENERATOR_MEMBERS
                   if not (isinstance(getattr(generator, name, None), str) if name == 'method'
                           else callable(getattr(generator, name, None)))]
        return [f'{type(generator).__name__} is not a generator: '
                f'{", ".join(missing)} missing or of the wrong type']
    problems = []
    label = f'generator {generator.method!r}'
    try:
        settings = generator.settings()
        if not isinstance(settings, dict):
            problems.append(f'{label}: settings() returned {type(settings).__name__}, not a dict')
        else:
            json.dumps(settings)
    except Exception as exc:                                # noqa: BLE001
        problems.append(f'{label}: settings() is not a JSON-serializable dict '
                        f'({type(exc).__name__}: {exc})')
    try:
        found = generator.check(network)
    except Exception as exc:                                # noqa: BLE001
        problems.append(f'{label}: check(network) raised {type(exc).__name__}: {exc}')
    else:
        if not isinstance(found, (list, tuple)):
            problems.append(f'{label}: check(network) returned {type(found).__name__}, '
                            f'not a list of messages')
        else:
            problems.extend(f'{label}: {message}' for message in found)
    return problems


def canonical_generator_spec(value):
    """Validate a ``params.cgaimmd_generator`` value without building the generator.

    Parameters
    ----------
    value : str, dict, Generator or None
        See :func:`make_generator`.

    Returns
    -------
    str, dict, Generator or None
        `value` (a dict is copied, its lists turned into tuples).

    Raises
    ------
    TypeError
        For a value of another type, a method name that is neither registered
        nor built in, a dict whose ``'method'`` is not such a name, or
        settings that are not JSON-serializable (they are written into the
        params file and every sidecar).

    Notes
    -----
    Methods that are registered already, or built in and part of the core
    install (``'ode'``), are also built once here, so that a wrong setting is
    an error when the params are loaded. A method whose module needs a pip
    extra is not imported: its settings are checked when the generator is
    built (at the start-up check of a shooting worker).
    """
    if value is None or is_generator(value):
        return value
    if _is_generator_class(value):
        raise _class_given(value)
    if isinstance(value, str):
        if not is_known_generator(value):
            raise TypeError(f'unknown generation method {value!r}; known methods are '
                            f'{available_generators()} (add one with '
                            f'aimmd.cgaimmd.generation.register_generator)')
        _probe(value, {})
        return value
    if isinstance(value, dict):
        method = value.get('method')
        if method is not None and not is_known_generator(method):
            raise TypeError(f"unknown generation method {method!r} in 'cgaimmd_generator'; "
                            f'known methods are {available_generators()}')
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f'the keys of a generator settings dict must be str, got '
                            f'{list(value)!r}')
        out = {key: tuple(item) if isinstance(item, list) else item
               for key, item in value.items()}
        try:
            json.dumps(out)
        except (TypeError, ValueError) as exc:
            raise TypeError(f'generator settings must be JSON-serializable, got {value!r} '
                            f'({exc})') from exc
        if method is not None:
            _probe(method, {key: item for key, item in out.items() if key != 'method'})
        return out
    raise TypeError(f"'cgaimmd_generator' must be a method name, a settings dict, a generator "
                    f'object or None, got {type(value).__name__}: {value!r}')


def _probe(method, settings):
    """Build `method` once when that imports no optional extra (see canonical_generator_spec)."""
    with _LOCK:
        registered = method in _FACTORIES
    lazy = _LAZY.get(method)
    if not registered and (lazy is None or lazy[2] is not None):
        return
    try:
        _build(method, settings)
    except (TypeError, ValueError) as exc:
        raise TypeError(str(exc)) from exc
