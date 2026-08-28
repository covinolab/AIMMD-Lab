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
# Free-simulation restart source
# ════════════════════════════════════════════════════════════════════════════

FREE_RESTART_MODES = ('crossing', 'transitions', 'basin', 'equilibrium')
"""Accepted values of ``params.restart_free_simulations_from``.

``'crossing'``
    The last frame the previous free trajectory spent in the target state, i.e.
    the configuration it escaped from. This is the historical behaviour and the
    default. That frame lies *on* the state boundary, so every first passage
    starts from the boundary-entry distribution.
``'transitions'``
    The end frames of a randomly sampled AIMMD transition path. This is what the
    deprecated ``restart_free_simulations_with_transitions`` selected.
``'basin'``
    A frame drawn uniformly from the in-state frames of the accumulated free
    trajectories, i.e. from the *biased* equilibrium inside the state — the
    occupancy measure the Tiwary-Parrinello boosted clock assumes.
``'equilibrium'``
    The same pool, drawn with probability proportional to ``exp(bias)``, i.e.
    from the **unbiased** (Boltzmann) equilibrium inside the state. With no
    recorded bias every weight is 1 and this coincides with ``'basin'``, which
    is correct: an unbiased trajectory already samples Boltzmann.
"""

FREE_RESTART_IN_BASIN_MODES = ('basin', 'equilibrium')
"""The modes that draw from inside the state, and so are meaningless for R."""


def parse_free_restart_from(value, states=None, field='restart_free_simulations_from'):
    """
    Parse a ``restart_free_simulations_from`` specification.

    Grammar (entries separated by whitespace and/or commas)::

        spec  := entry [entry ...]
        entry := mode                 # the default for every state
               | LETTERS ':' mode     # for the named states only

    At most one bare ``mode`` may appear; it becomes the default for states not
    named explicitly. An empty specification means ``'crossing'``.

    Parameters
    ----------
    value : str
        The specification.
    states : str, optional
        ``params.states``. When given, every named state letter must occur in
        it, and the in-basin modes are refused for the reactive state
        ``states[1]``.
    field : str, optional
        Field name to quote in error messages.

    Returns
    -------
    default : str
        Mode for states not named explicitly.
    per_state : dict
        ``{state letter: mode}`` for the states that were named.

    Raises
    ------
    TypeError
        On an unknown mode, a malformed entry, a repeated state, more than one
        bare mode, an unknown state letter, or an in-basin mode on the reactive
        state.
    """
    text = re.sub(r'\s*:\s*', ':', str(value or '').replace(',', ' ')).strip()
    default = 'crossing'
    per_state = {}
    seen_bare = False

    def _mode(token, where):
        mode = token.strip().lower()
        if mode not in FREE_RESTART_MODES:
            raise TypeError(
                f'{field!r}: unknown restart source {token.strip()!r} in '
                f'{where}; expected one of '
                f'{", ".join(repr(m) for m in FREE_RESTART_MODES)}')
        return mode

    for entry in text.split():
        if ':' not in entry:
            if seen_bare:
                raise TypeError(
                    f'{field!r} = {value!r} names more than one default restart '
                    f'source; give at most one bare mode, and qualify the rest '
                    f'as e.g. \'A:equilibrium\'')
            default = _mode(entry, f'{field!r} = {value!r}')
            seen_bare = True
            continue

        parts = entry.split(':')
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise TypeError(
                f'{field!r}: malformed entry {entry!r} in {value!r}; expected '
                f'\'STATES:mode\', e.g. \'A:equilibrium\'')
        letters = parts[0].strip().upper()
        if not letters.isalpha():
            raise TypeError(
                f'{field!r}: {parts[0].strip()!r} in {entry!r} is not a state '
                f'letter list')
        mode = _mode(parts[1], f'entry {entry!r}')
        for letter in letters:
            if letter in per_state:
                raise TypeError(
                    f'{field!r} = {value!r} assigns state {letter!r} twice')
            if states is not None and letter not in states:
                raise TypeError(
                    f'{field!r}: state {letter!r} is not one of '
                    f'params.states = {states!r}')
            if (states is not None and len(states) >= 2
                    and letter == states[1]
                    and mode in FREE_RESTART_IN_BASIN_MODES):
                raise TypeError(
                    f'{field!r}: restart source {mode!r} is not defined for the '
                    f'reactive state {letter!r} - "inside the state" is the '
                    f'barrier region there, where no equilibrium restart '
                    f'distribution exists. Use \'crossing\' or \'transitions\'.')
            per_state[letter] = mode

    return default, per_state


def canonical_free_restart_from(value, states=None,
                                field='restart_free_simulations_from'):
    """Validate a ``restart_free_simulations_from`` value and normalise it.

    Returns the canonical spelling — states upper case, modes lower case,
    single-space separated, the bare default (if any) first — so that the value
    round-trips through :meth:`aimmd.Params.save` unchanged.
    """
    default, per_state = parse_free_restart_from(value, states=states,
                                                 field=field)
    parts = [] if default == 'crossing' and per_state else [default]
    parts += [f'{letter}:{per_state[letter]}' for letter in sorted(per_state)]
    return ' '.join(parts) if parts else 'crossing'


def legacy_free_restart_replacement(value):
    """The ``restart_free_simulations_from`` value equivalent to a legacy flag.

    ``restart_free_simulations_with_transitions`` was a state-selector string:
    ``'all'`` for every free simulation, otherwise the letters of the states it
    applied to. Returns ``''`` for an empty (inactive) flag.
    """
    text = str(value or '').replace(' ', '')
    if not text:
        return ''
    if text.lower() == 'all':
        return 'transitions'
    return f'{text.upper()}:transitions'
