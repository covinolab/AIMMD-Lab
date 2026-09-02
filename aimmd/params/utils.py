"""
aimmd.params.utils
=================

Utilities supporting `aimmd.params.Params`.

This module provides helpers used during parameter validation, saving, and
serialization of callable objects.

Provided utilities
------------------

update_source(instance, name)
    Attach a `__source__` string to a function/class instance so that
    `Params.__str__` / `Params.save` can reproduce or import it.

create_default_values_function(network, descriptor_transform=None)
    Build a default `values_function` that:
    - accepts numpy-like input (coordinates/descriptors),
    - optionally applies `descriptor_transform`,
    - evaluates `network` in inference mode,
    - returns a 1D numpy array of outputs.

Notes
-----
These helpers are intentionally lightweight and do not depend on higher-level
AIMMD components.
"""

# external
import re
import numpy as np
import torch
from re import split
from types import FunctionType as Function
from inspect import getsource

# aimmd imports
from ..core.utils import accepts_system_id


def update_source(instance, name):
    """
    Attach a reproducible source/import snippet to a callable-like object.

    This function ensures `instance` has a `__source__` attribute, which is
    used by `Params.__str__` and `Params.save()` to serialize functions and
    classes into a Python parameters file.

    Parameters
    ----------
    instance : object
        A function or class instance intended to be stored in `Params`.
        If `instance` is a function, this captures its source code (if defined
        in `__main__`) or generates an import line (if defined in a module).
        If `instance` is not a function, this generates an import line for its
        class (assumes the class is defined in an importable module).
    name : str
        The attribute name under which this object is stored in `Params`.
        This name is used when generating import aliases or lambda assignments.

    Returns
    -------
    None

    Notes
    -----
    - If `instance` already has `__source__`, nothing is done.

    - For local module handling, module paths are reduced to their basename
      by splitting on "/". This matches the loader’s “local module name”
      remapping strategy implemented in `ParamsIO.load`.
    - Lambda functions are handled by rewriting their source into an assignment:
      `name = lambda ...`.
    """

    cls = instance.__class__

    # In order to import functions and classes correctly, `Params.__str__`
    # needs a stable representation for each callable/class.

    # Not a function: cannot be retrieved from main.
    # The object's *class* must be importable and defined in the same file too.
    if not isinstance(instance, Function):
        # When "local" import, omit path
        module = split(r'[\\/]', cls.__module__)[-1]
        instance.__source__ = f'from {module} import {name}\n'

    # Function defined not in "__main__": represent by import statement.
    elif instance.__module__ != '__main__':
        if hasattr(instance, '__source__'):
            return
        module = split(r'[\\/]', instance.__module__)[-1]
        if instance.__name__ in (name, '<lambda>'):
            instance.__source__ = f'from {module} import {name}\n'
        else:
            instance.__source__ = (f'from {module} import '
                                   f'{instance.__name__} as {name}\n')
    
    # Function defined in "__main__": check name is correct.
    elif instance.__name__ not in (name, '<lambda>'):
        raise TypeError(f'please rename function '
                        f'{instance.__name__!r} to {name!r}')
    
    # Function defined in "__main__": embed its full source code.
    else:
        instance.__source__ = getsource(instance).lstrip()  
        
        # Process lambda functions: represent as an assignment to preserve name.
        if (not instance.__source__.startswith('def ')
            and 'lambda' in instance.__source__):
            instance.__source__ = (f'{name} = lambda' +
                instance.__source__.split('lambda', 1)[1])


def create_default_values_function(network, descriptor_transform=None):
    """
    Create the default `values_function` used by AIMMD params given a neural
    network object. The `network` object will always remain the same, but you
    can update its weights after training the desired reaction coordinate.

    The returned function:

    - accepts numpy-like input (coordinates or descriptors),
    - optionally applies `descriptor_transform`,
    - converts to a torch tensor on the network's device/dtype,
    - flattens per-frame inputs to 2D (batch, features),
    - evaluates the network in inference mode,
    - returns a 1D numpy array.

    Parameters
    ----------
    network : torch.nn.Module
        Neural network whose output defines the “values” (e.g., logit committor).
        The function uses the device and dtype of `network.parameters()`.
    descriptor_transform : callable, optional
        Transform applied to input data *before* conversion to torch.
        If None, identity is used.

    Returns
    -------
    callable
        A function `values_function(input_data)` returning a 1D numpy array.

    Notes
    -----
    - Uses `torch.inference_mode()` for performance and to disable autograd.
    - Calls `network.eval()` to disable dropout and freeze batchnorm behavior.
    - Uses `torch.flatten(start_dim=1)` to ensure a (N, F) shaped input.
    """
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    if descriptor_transform is None:
        descriptor_transform = lambda x: x
    transform_takes_system_id = accepts_system_id(descriptor_transform)

    def values_function(input_data, system_id=None):
        """
        Compute values from numpy-like input using the provided network.

        Parameters
        ----------
        input_data : array-like
            Input frames. Typically shape (N, ...) where N is the number of frames.
            The function will flatten each frame into a 1D feature vector.
        system_id : hashable or None, optional
            Multi-system identifier. When provided and the `descriptor_transform`
            accepts a `system_id` keyword, it is forwarded so the transform can
            featurize the correct system into the shared network's input space.
            Ignored (with the data argument only) otherwise — single-system
            behaviour is unchanged.

        Returns
        -------
        numpy.ndarray
            1D array of length N containing network outputs.

        Notes
        -----
        The network output is converted to CPU and flattened with `.ravel()`.
        """
        network.eval()  # disables dropout, batchnorm training behavior
        with torch.inference_mode():  # faster than no_grad()
            if system_id is not None and transform_takes_system_id:
                input_data = descriptor_transform(
                    np.asarray(input_data), system_id=system_id)
            else:
                input_data = descriptor_transform(np.asarray(input_data))
            input_data = torch.tensor(
                input_data, dtype=dtype, device=device)
            input_data = torch.flatten(input_data, start_dim=1)
            output = network(input_data)
        return output.cpu().numpy().ravel()

    return values_function


# ════════════════════════════════════════════════════════════════════════════
# Free-simulation seeding: where the first trajectory starts, and where the
# later ones restart from
# ════════════════════════════════════════════════════════════════════════════

SEEDING_POSITION_ALIASES = {'boundary': 1.0, 'middle': 0.5, 'deepest': 0.0}
"""Named values of ``params.free_seeding_position``.

The frames of the initial path that belong to the target state form one
contiguous run — the one the transition departs from. The position is a
fraction over that run, ordered *far side first*, so the meaning is the same
for a state at the start of the path and for one at its end even though the
file order is mirrored.

``'boundary'`` (1.0)
    The frame adjacent to the reactive region. This is the historical
    behaviour and the default.
``'middle'`` (0.5)
    The middle of the run.
``'deepest'`` (0.0)
    The frame furthest from the reactive region, i.e. as deep into the state
    as the initial path reaches.
"""

SEEDING_POSITION_RANDOM = 'random'
"""``params.free_seeding_position`` value drawing uniformly over the run."""

FREE_RESTART_SOURCES = ('crossing', 'seed', 'basin', 'equilibrium',
                        'transitions')
"""Accepted values of ``params.free_restart_source``.

``'crossing'``
    The last frame the previous free trajectory spent in the target state, i.e.
    the configuration it escaped from. This is the historical behaviour and the
    default. That frame lies *on* the state boundary, so every first passage
    starts from the boundary-entry distribution.
``'seed'``
    The same frame the first seeding used, re-derived from the initial path
    through ``params.free_seeding_position``.
``'basin'``
    A frame drawn uniformly from the in-state frames of the accumulated free
    trajectories, i.e. from the *biased* equilibrium inside the state — the
    occupancy measure the Tiwary-Parrinello boosted clock assumes.
``'equilibrium'``
    The same pool, drawn with probability proportional to ``exp(bias)``, i.e.
    from the **unbiased** (Boltzmann) equilibrium inside the state. With no
    recorded bias every weight is 1 and this coincides with ``'basin'``, which
    is correct: an unbiased trajectory already samples Boltzmann.
``'transitions'``
    The end frames of a randomly sampled AIMMD transition path. This is what
    the deprecated ``restart_free_simulations_with_transitions`` selected, and
    like that flag it also replaces the *first* seed, falling back to a
    randomly chosen initial path while no transition has been sampled yet.
"""

FREE_RESTART_IN_STATE_SOURCES = ('seed', 'basin', 'equilibrium')
"""The sources that draw from inside the state, and so are meaningless for R."""


def _state_keyed_spec(value, field, states, parse_one, empty_default):
    """
    Split a per-state params value into ``(default, per_state)``.

    A scalar applies to every state. A mapping keyed by state letter applies
    only to the states it names; the others keep `empty_default`.

    Parameters
    ----------
    value : object
        The user's value: a scalar accepted by `parse_one`, or a mapping from
        state letter to such a scalar. None and ``''`` mean "unset".
    field : str
        Field name, quoted in error messages.
    states : str or None
        ``params.states``. When given, every key must occur in it.
    parse_one : callable
        ``parse_one(scalar, where) -> canonical scalar``.
    empty_default : object
        The canonical value an unset field resolves to.

    Returns
    -------
    default : object
    per_state : dict
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return empty_default, {}

    if isinstance(value, dict):
        per_state = {}
        for key, item in value.items():
            letter = str(key).strip().upper()
            if len(letter) != 1 or not letter.isalpha():
                raise TypeError(
                    f'{field!r}: {key!r} is not a single state letter')
            if states is not None and letter not in states:
                raise TypeError(
                    f'{field!r}: state {letter!r} is not one of '
                    f'params.states = {states!r}')
            if letter in per_state:
                raise TypeError(
                    f'{field!r} assigns state {letter!r} twice')
            per_state[letter] = parse_one(item, f'{field!r}[{letter!r}]')
        return empty_default, per_state

    return parse_one(value, f'{field!r}'), {}


def _one_seeding_position(value, where):
    """Canonicalise a single `free_seeding_position` scalar."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in SEEDING_POSITION_ALIASES:
            return SEEDING_POSITION_ALIASES[text]
        if text == SEEDING_POSITION_RANDOM:
            return SEEDING_POSITION_RANDOM
        try:
            number = float(text)
        except ValueError:
            names = ', '.join(repr(n) for n in SEEDING_POSITION_ALIASES)
            raise TypeError(
                f'{where}: unknown seeding position {value!r}; expected a '
                f'fraction in [0, 1], one of {names}, or '
                f'{SEEDING_POSITION_RANDOM!r}') from None
    elif isinstance(value, bool):
        # bools are ints in Python and would silently mean 0.0 / 1.0
        raise TypeError(f'{where}: seeding position must be a number or a '
                        f'name, got {value!r}')
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TypeError(f'{where}: seeding position must be a number or a '
                            f'name, got {value!r}') from None

    if not 0.0 <= number <= 1.0:
        raise TypeError(f'{where}: seeding position {number!r} is outside '
                        f'[0, 1]; 0.0 is the frame furthest from the reactive '
                        f'region and 1.0 the one adjacent to it')
    return float(number)


def _one_restart_source(value, where):
    """Canonicalise a single `free_restart_source` scalar."""
    source = str(value).strip().lower()
    if source not in FREE_RESTART_SOURCES:
        raise TypeError(
            f'{where}: unknown restart source {value!r}; expected one of '
            f'{", ".join(repr(s) for s in FREE_RESTART_SOURCES)}')
    return source


def parse_seeding_position(value, states=None, field='free_seeding_position'):
    """
    Parse a ``params.free_seeding_position`` value.

    Returns
    -------
    default : float or str
        Position for states not named explicitly; 1.0 when unset.
    per_state : dict
        ``{state letter: position}`` for the states that were named.

    Raises
    ------
    TypeError
        On an unknown name, a fraction outside [0, 1], a key that is not a
        single state letter, or a state that is not in `states`.
    """
    return _state_keyed_spec(value, field, states, _one_seeding_position,
                             SEEDING_POSITION_ALIASES['boundary'])


def parse_restart_source(value, states=None, field='free_restart_source'):
    """
    Parse a ``params.free_restart_source`` value.

    Returns
    -------
    default : str
        Source for states not named explicitly; ``'crossing'`` when unset.
    per_state : dict
        ``{state letter: source}`` for the states that were named.
    """
    return _state_keyed_spec(value, field, states, _one_restart_source,
                             'crossing')


def _canonical(value, states, field, parser, aliases=None):
    """Normalise a per-state params value so it round-trips through `save`."""
    default, per_state = parser(value, states=states, field=field)
    if aliases:
        inverse = {v: k for k, v in aliases.items()}
        default = inverse.get(default, default)
        per_state = {k: inverse.get(v, v) for k, v in per_state.items()}
    if per_state:
        return {letter: per_state[letter] for letter in sorted(per_state)}
    return default


def canonical_seeding_position(value, states=None,
                               field='free_seeding_position'):
    """Validate a ``free_seeding_position`` value and normalise it.

    Named positions keep their names, fractions stay numbers, and a per-state
    mapping comes back with upper-case keys in sorted order, so the value
    round-trips through :meth:`aimmd.Params.save` unchanged.
    """
    return _canonical(value, states, field, parse_seeding_position,
                      SEEDING_POSITION_ALIASES)


def canonical_restart_source(value, states=None, field='free_restart_source'):
    """Validate a ``free_restart_source`` value and normalise it."""
    return _canonical(value, states, field, parse_restart_source)


def legacy_transitions_replacement(value):
    """The ``free_restart_source`` value equivalent to the deprecated flag.

    ``restart_free_simulations_with_transitions`` was a state-selector string:
    ``'all'`` for every free simulation, otherwise the letters of the states it
    applied to. Returns ``''`` for an empty (inactive) flag, the bare source
    ``'transitions'`` for ``'all'``, and a per-state mapping otherwise.
    """
    text = str(value or '').replace(' ', '')
    if not text:
        return ''
    if text.lower() == 'all':
        return 'transitions'
    return {letter: 'transitions' for letter in text.upper()}
