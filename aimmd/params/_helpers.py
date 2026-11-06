"""
aimmd.params._helpers
====================

Helper mixin for :class:`aimmd.params.Params`.

This module defines :class:`~aimmd.params._helpers.ParamsHelpers`, which provides
the *actual constructor* and the *type/consistency enforcement* logic for Params.

Key responsibilities
--------------------

- `_init`: constructor logic (supports loading from a params file, applying
  overrides, and populating defaults).
- `_setattr`: validated assignment for dataclass fields (type checking,
  callable handling, special-case parsing).
- `_process_and_check`: post-assignment validation that depends on multiple
  fields and/or existing data (e.g., consistency of initial paths, states,
  binning constraints, values function definition).

Notes
-----
- This is a mixin class; it is not intended to be instantiated directly.
- Most validation errors are surfaced as `TypeError` with a short explanation.
- The callable serialization mechanism relies on `update_source` to attach
  a `__source__` attribute to functions/classes for reproducible saving.
"""

# external
import os
import sys
import json
import numpy as np
import functools
import inspect
import warnings
from abc import ABC
from math import inf
from re import split as _split
from types import MethodType as Method, FunctionType as Function
from typing import List, Callable
from numbers import Integral
from pathlib import Path as PosixPath
from MDAnalysis import Universe
from collections.abc import Iterable

# aimmd imports
from .utils import (update_source, create_default_values_function,
                    canonical_restart_source, canonical_seeding_position,
                    legacy_transitions_replacement, check_descriptors_series,
                    canonical_cgaimmd_parameters, CGAIMMD_PARAMETER_DEFAULTS,
                    COMMITTOR_ACCEPTANCE_RULES,
                    SEEDING_POSITION_ALIASES, SEEDING_POSITION_RANDOM)
from ..core.utils import accepts_system_id
from ..path import Path
from ..pathensemble import PathEnsemble
from ..network.rescalable import Rescalable as RescalableNetwork


_SUBSAMPLE_CAP_KEYS = ('shot', 'free', 'in_state')

CGAIMMD_HOOK_FIELDS = ('cgaimmd_validity_function', 'cgaimmd_prepare_function',
                       'cgaimmd_minimize_function', 'cgaimmd_lambda_sampler')
"""The function-valued fields of C-GAIMMD, each None or a plain function."""

_CGAIMMD_RULE_FIELDS = frozenset({
    'chain_type', 'network', 'cgaimmd_parameters', 'cgaimmd_generator', 'initial_paths',
    'descriptors_function', 'descriptors_series', 'descriptor_transform', 'gen_temperature',
    'rescale_committor', 'multi_system', 'initial_network', 'cgaimmd_prepare_function',
    'topology', 'round_shooting_index', 'states', 'states_function',
    'free_seeding_position'})
"""The fields the rules of ``chain_type='cgaimmd'`` read.

Assigning one of them under ``chain_type='cgaimmd'`` runs the rules again
(``states``, ``states_function`` and ``free_seeding_position`` for the
initial-path rules).
"""

_ABSENT = object()
"""Marks a field that had no value before an assignment (restored as absent)."""

_LEGACY_FIELDS = frozenset({'flow_parameters', 'flow_validity_function',
                            'flow_prepare_function', 'flow_minimize_function',
                            'flow_lambda_sampler'})
"""Old names of C-GAIMMD fields, read with a warning for one release
(``aimmd.cgaimmd._legacy.PARAMS_FIELDS``, kept here so that ordinary params
import no C-GAIMMD code)."""


def _may_be_cgaimmd_network(network):
    """Cheap pre-test of ``is_cgaimmd_network`` that imports no C-GAIMMD code.

    A network without a callable ``default_generator`` is not a C-GAIMMD
    network in the sense of ``is_cgaimmd_network``, so RFPS and TPS params
    with ordinary networks never import ``aimmd.cgaimmd``.
    """
    try:
        return callable(getattr(network, 'default_generator', None))
    except Exception:                                           # noqa: BLE001
        return False


def _is_cgaimmd_network(network):
    """``aimmd.cgaimmd.protocol.is_cgaimmd_network``, imported only when it can be True."""
    if not _may_be_cgaimmd_network(network):
        return False
    from ..cgaimmd.protocol import is_cgaimmd_network
    return is_cgaimmd_network(network)


def _generator_source(value, name):
    """Give a generator object of `cgaimmd_generator` the ``__source__`` that saves it.

    The workers load the params from the saved file, so the object must be
    rebuilt from what is written there:

    - a generator of a registered (or built-in) method is saved as its
      settings dict ``{'method': ..., **settings()}``, from which
      ``make_generator`` builds an equal one (the generator contract);
    - any other generator is saved as ``from <module> import <name>``, which
      works only when the module of its class (the params file, or a module
      next to it) binds `name` to this very object;
    - anything else is refused here, not in every worker.
    """
    from ..cgaimmd.generation import is_known_generator
    if is_known_generator(value.method):
        spec = {'method': value.method, **value.settings()}
        try:
            json.dumps(spec)
        except (TypeError, ValueError) as exc:
            raise TypeError(f'{name}: settings() of {type(value).__name__} must be '
                            f'JSON-serializable to be saved, got {spec!r} ({exc})') from exc
        value.__source__ = f'{name} = {spec!r}\n'
        return
    module = sys.modules.get(type(value).__module__)
    if getattr(module, name, None) is not value:
        raise TypeError(
            f'{name}: a {type(value).__name__} object of the unregistered method '
            f'{value.method!r} cannot be saved for the workers. Register the method '
            f'(aimmd.cgaimmd.generation.register_generator, in the params file or a '
            f'module it imports) and give its name or settings dict, or define the '
            f"class in the params file and assign '{name} = {type(value).__name__}(...)' "
            f'there')
    update_source(value, name)


def _check_caps_dict(value):
    """Validate one ``subsample_caps`` dict (see ``Params.subsample_caps``).

    Allowed keys are ``'shot'``, ``'free'`` and ``'in_state'``; each value must
    be a positive integer (a per-category cap) or ``None`` (uncapped). Returns a
    shallow copy with ints coerced to plain ``int``.
    """
    if not isinstance(value, dict):
        raise TypeError("'subsample_caps' must be a dict (or a list of dicts / "
                        f"None per system), got {type(value).__name__}")
    out = {}
    for key, cap in value.items():
        if key not in _SUBSAMPLE_CAP_KEYS:
            raise TypeError(f"unknown subsample_caps key {key!r}; allowed keys "
                            f"are {list(_SUBSAMPLE_CAP_KEYS)}")
        if cap is None:
            out[key] = None
            continue
        if isinstance(cap, bool) or not isinstance(cap, Integral) or cap <= 0:
            raise TypeError(f"subsample_caps[{key!r}] must be a positive int or "
                            f"None, got {cap!r}")
        out[key] = int(cap)
    return out


def _store_seed_descriptors(path, series, descriptors):
    """Attach the descriptors of an initial path as its `series` array.

    Descriptor arrays of other series (from an earlier `descriptors_series`)
    are dropped first: the launcher exports every array of an initial path
    to ``initial{states}``, and only the current series is read.
    """
    for name in [name for name in path.__dict__
                 if name == 'descriptors' or name.startswith('descriptors-')]:
        del path.__dict__[name]
    setattr(path, series, descriptors)


def _requires_wrapper(instance):
    """Whether callables bound to `instance` must be refused: objects such as
    node-table featurizers set ``_params_requires_wrapper = True``, since
    Params would drop them (it stores functions, not objects)."""
    try:
        return getattr(instance, '_params_requires_wrapper', False) is True
    except Exception:
        return False


def _wrapper_example(name, function, callable_):
    """The module-level wrapper `name` that calls ``INSTANCE.function`` with
    the parameters of `callable_` (an example for the error messages)."""
    try:
        parameters = list(inspect.signature(callable_).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    arguments = []
    for parameter in parameters:
        if parameter.kind is parameter.VAR_POSITIONAL:
            arguments.append(f'*{parameter.name}')
        elif parameter.kind is parameter.VAR_KEYWORD:
            arguments.append(f'**{parameter.name}')
        elif parameter.kind is parameter.KEYWORD_ONLY:
            arguments.append(f'{parameter.name}={parameter.name}')
        else:
            arguments.append(parameter.name)
    signature = ', '.join(str(parameter) for parameter in parameters)
    return (f'    def {name}({signature}):\n'
            f'        return INSTANCE.{function}({", ".join(arguments)})\n')


def _bound_method_message(name, method):
    """Why the bound `method` cannot be the callable field `name`, and the
    module-level wrapper to write instead."""
    owner = type(method.__self__).__name__
    function = method.__func__.__name__
    return (f'{name!r} is the bound method {owner}.{function}. Params stores '
            f'functions, not objects, and would lose the {owner} instance. '
            f'Define a module-level wrapper in the params file instead, e.g.\n'
            f'\n'
            f'{_wrapper_example(name, function, method)}'
            f'\n'
            f'where INSTANCE is your {owner}.')


def _partial_owner(partial):
    """``(owner, function name)`` if the functools.partial `partial` (nested
    or not) binds an object that `_requires_wrapper`, through a bound method
    or an argument; None otherwise."""
    owner = None
    function = partial
    while isinstance(function, functools.partial):
        bound = [*function.args, *function.keywords.values()]
        if isinstance(function.func, Method):
            bound.append(function.func.__self__)
        if owner is None:
            owner = next((item for item in bound if _requires_wrapper(item)),
                         None)
        function = function.func
    if owner is None:
        return None
    if isinstance(function, Method):
        function = function.__func__
    return owner, getattr(function, '__name__', 'method')


def _partial_message(name, partial):
    """Why the functools.partial `partial` over an object that
    `_requires_wrapper` cannot be the callable field `name`, and the
    module-level wrapper to write instead."""
    owner, function = _partial_owner(partial)
    owner = (owner if isinstance(owner, type) else type(owner)).__name__
    return (f'{name!r} is a functools.partial of {owner}.{function}. Params '
            f'stores functions, not objects: it would save the partial as '
            f"'from functools import {name}', which no worker can import, "
            f'and lose the {owner} instance. Define a module-level wrapper in '
            f'the params file instead, e.g.\n'
            f'\n'
            f'{_wrapper_example(name, function, partial)}'
            f'\n'
            f'where INSTANCE is your {owner}.')


# params' helpers
class ParamsHelpers(ABC):
    def _init(self, *args, **kwargs):
        """
        Initialize a Params instance.

        This initializer supports two usage patterns:

        1) Load from file (preferred for reproducibility):

           - If `args[0]` is a valid path to an existing file, it is treated as
             a params file and loaded via `Params.load(...)`.

        2) No file provided:

           - If no valid file is provided, defaults are loaded from `'params.py'`
             (if present), otherwise initialization proceeds using defaults and
             provided keyword arguments.

        Parameters
        ----------
        *args
            If `args[0]` is a path-like string and exists, it is used as the
            params filename and removed from `args` before passing on.
            Remaining positional args are interpreted by `Params.load` as
            positional overrides in dataclass field order.
        **kwargs
            Field overrides applied with higher priority than file values.

            Special behavior:

            - If `filename` is in `kwargs`, it is treated as a params file
              (handled upstream in `ParamsIO.load`).

        Returns
        -------
        aimmd.params.Params
            The initialized object (assigned in-place).

        Notes
        -----
        This function populates `self.__dict__` directly from a loaded instance’s
        `__dict__` so the dataclass fields appear initialized in the correct
        order (and so that post-init checks can run coherently).
        """
        
        # determine "filename": the file where to take the parameters from
        if not (args or kwargs):
            filename = 'params.py'
        elif args and os.path.exists(args[0]):
            filename = args[0]
            args = args[1:]
        else:
            filename = kwargs.pop('filename', None)
        
        # create separate instance
        # NOTE: importing here avoids circular imports
        from . import Params
        instance = Params.load(filename, *args, **kwargs)
        
        # update state dict with that of instance
        self.__dict__.update(instance.__dict__)

    def _setattr(self, name, value, process_and_check=True):
        """
        Assign a parameter field with validation.

        Parameters
        ----------
        name : str
            Name of the field or a hidden attribute (leading underscore).
        value : object
            Value to assign.
        process_and_check : bool, optional
            If True, run `_process_and_check([name])` after assignment and
            revert on failure. This is disabled internally during bulk loading
            so that mutually-dependent fields can be set first and validated
            together.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the field does not exist, if the type is invalid, or if
            post-validation fails.

        Notes
        -----
        Many fields require special handling:

        - callable fields: unwrap bound methods, attach `__source__`. Bound
          methods of objects with ``_params_requires_wrapper`` (node-table
          featurizers), and functools.partial objects binding such an object,
          are refused: they need module-level wrappers.
        - `states`: normalized to uppercase letters and validated.
        - `topology`: attempts to build an MDAnalysis Universe to cache masses.
        - `initial_paths`: immediately coerced into a PathEnsemble.
        """
        # old names of C-GAIMMD fields: assigned to the new fields (one
        # DeprecationWarning each, see aimmd.cgaimmd._legacy), together: if
        # one of them is refused, none is assigned
        if name in _LEGACY_FIELDS:
            from ..cgaimmd import _legacy
            new_values = _legacy.params_field(name, value)
            backups = {new_name: self.__dict__.get(new_name, _ABSENT)
                       for new_name in (*new_values, '_read_old_names')}
            try:
                for new_name, new_value in new_values.items():
                    self._setattr(new_name, new_value, process_and_check=False)
                self.__dict__['_read_old_names'] = True
                if process_and_check:
                    self._process_and_check(list(new_values))
            except Exception as exception:
                for new_name, backup in backups.items():
                    if backup is _ABSENT:
                        self.__dict__.pop(new_name, None)
                    else:
                        self.__dict__[new_name] = backup
                if isinstance(exception, NotImplementedError):
                    raise
                raise TypeError(f'can\'t update {name!r} with {value!r}: '
                                f'{exception}') from exception
            return

        # assign fields
        if name in self.__dataclass_fields__:
            expected_type = self.__dataclass_fields__[name].type

            # functions
            if expected_type is Callable:
                # These callables may be None (meaning “use defaults”).
                if value is None and name in (
                    'descriptors_function', 'descriptor_transform',
                    'values_function', 'toy_mdrun',
                    'bias_function') + CGAIMMD_HOOK_FIELDS:
                    pass

                # C-GAIMMD hooks: plain functions only. Params.save embeds or
                # imports a function correctly, but writes a broken import
                # line for a callable object of an importable class.
                elif name in CGAIMMD_HOOK_FIELDS and not isinstance(
                        value, Function):
                    raise TypeError(
                        f"{name!r} must be a function defined in the params "
                        f"file (wrap a callable object in 'def {name}(...)')")

                elif not callable(value):
                    raise TypeError(f"'values_function' must be callable, "
                                    f'got {type(value).__name__}: {value}')
                
                # Bound methods are converted to underlying function objects.
                # That drops the instance: objects whose methods need it (e.g.
                # a NodeTableFeaturizer) are refused, the params file must
                # wrap them in module-level functions.
                elif isinstance(value, Method):
                    if _requires_wrapper(value.__self__):
                        raise TypeError(_bound_method_message(name, value))
                    value = value.__func__
                    update_source(value, name)

                # So are partials binding such an object: they would be saved
                # as an import from functools, which the workers cannot load.
                elif (isinstance(value, functools.partial)
                      and _partial_owner(value) is not None):
                    raise TypeError(_partial_message(name, value))

                else:
                    update_source(value, name)

                # Special validation for the `fit` callable signature.
                fit_args = 'params', 'pathensemble', 'verbose', 'worker'
                if name == 'fit':
                    args = tuple(inspect.signature(value).parameters)
                    if args != fit_args:
                        raise TypeError(f"'fit' must have args {fit_args} "
                                        f"in order, got {args} instead")

            # states
            elif name == 'states':
                value = str(value).upper().replace(' ','')
                if not (value.isalpha() and len(set(value)) == 3):
                    raise TypeError(
                        f'{name!r} must be three distinct upper alpha chars')

            # state selector strings (set of letters or 'all')
            elif name in ('extra_bins', 'terminal_bin_extension'):
                value = str(value).replace(' ','')
                if value.lower() != 'all':
                    value = value.upper()
                if value and not value.isalpha():
                    raise TypeError(
                        f'{name!r} must be distinct upper alpha chars')
            
            # free-simulation seeding: where the FIRST trajectory of a state
            # starts inside the state, as a fraction over the state's run of
            # initial-path frames. A scalar, or a mapping keyed by state.
            elif name == 'free_seeding_position':
                value = canonical_seeding_position(
                    value, states=getattr(self, 'states', None))

            # free-simulation reseeding: where every LATER trajectory of a
            # state restarts from. A scalar, or a mapping keyed by state.
            elif name == 'free_restart_source':
                value = canonical_restart_source(
                    value, states=getattr(self, 'states', None))

            # DEPRECATED state selector, superseded by the field above
            elif name == 'restart_free_simulations_with_transitions':
                value = str(value).replace(' ', '')
                if value.lower() != 'all':
                    value = value.upper()
                if value and not value.isalpha():
                    raise TypeError(
                        f'{name!r} must be distinct upper alpha chars, or '
                        f"'all'")
                if value:
                    replacement = legacy_transitions_replacement(value)
                    message = (
                        f"'restart_free_simulations_with_transitions' is "
                        f"deprecated; use "
                        f"free_restart_source = {replacement!r} "
                        f"instead. The new field also offers the 'seed', "
                        f"'basin' and 'equilibrium' restart sources, which "
                        f"draw from inside the state rather than from its "
                        f"boundary, and pairs with 'free_seeding_position' "
                        f"which places the first seed.")
                    warnings.warn(message, DeprecationWarning, stacklevel=4)
                    # warnings are shown once per process and this one has been
                    # missed before; the log line cannot be filtered away
                    print(f'Warning: {message}')

            # topology (update universe)
            elif name == 'topology':
                if isinstance(value, (list, tuple)):
                    # multi-system: one topology per system. Per-system
                    # universes are built in `_process_and_check` (the
                    # system_ids may not be known yet at this point).
                    value = [str(v) for v in value]
                    self.__dict__['_universe'] = None
                else:
                    # Cache an MDAnalysis Universe if possible (used by masses()).
                    try:
                        universe = Universe(value)
                    except:
                        universe = None
                    self.__dict__['_universe'] = universe

            # engine command / mdp fields: a single string, or (multi-system)
            # a list with one entry per system. Resolved per-worker at run time.
            elif name in ('gmx_mdp', 'gmx_grompp', 'gmx_mdrun', 'gmx_eneconv',
                          'gmx_genvel'):
                if isinstance(value, (list, tuple)):
                    value = [str(v) for v in value]
                elif not isinstance(value, str):
                    raise TypeError(f'{name!r} must be a str or a list of '
                                    f'strings, got {type(value).__name__}')

            # multi-system per-system labels
            elif name == 'system_ids':
                if value is None:
                    value = []
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f'{name!r} must be a list of system '
                                    f'labels, got {type(value).__name__}')
                value = [str(v) for v in value]

            # multi-system shared graph atom-type table
            elif name == 'atom_types':
                if value is not None:
                    if not isinstance(value, (list, tuple)):
                        raise TypeError(f'{name!r} must be a list of atom-type '
                                        f'strings or None, got '
                                        f'{type(value).__name__}')
                    value = [str(v) for v in value]

            # reactive-bias threshold: scalar (all systems) or per-system list
            elif name == 'bias_reactive_threshold':
                if isinstance(value, (list, tuple)):
                    value = [float(v) for v in value]
                else:
                    value = float(value)

            # optional value-pass subsampling caps: dict (all systems) or a
            # list of dicts/None (one per system)
            elif name == 'subsample_caps':
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        value = [None if v is None else _check_caps_dict(v)
                                 for v in value]
                    else:
                        value = _check_caps_dict(value)

            # C-GAIMMD settings: validated and normalized; only the keys the
            # user gave are stored (Params.cgaimmd_settings adds defaults)
            elif name == 'cgaimmd_parameters':
                value = canonical_cgaimmd_parameters(value)

            # C-GAIMMD generation method: a name, a settings dict, a generator
            # object or None (aimmd.cgaimmd.generation)
            elif name == 'cgaimmd_generator':
                if value is not None:
                    from ..cgaimmd.generation import (canonical_generator_spec,
                                                      is_generator)
                    value = canonical_generator_spec(value)
                    if is_generator(value):
                        _generator_source(value, name)

            # initial paths (converted in real-time)
            elif name == 'initial_paths':
                # Multi-system runs pass one *group* of initial paths per
                # system (a list whose elements are themselves lists or
                # PathEnsembles). Detect that structurally so the behaviour
                # does not depend on field-assignment order.
                grouped = (isinstance(value, (list, tuple)) and len(value) > 0
                           and all(isinstance(v, (list, tuple, PathEnsemble))
                                   for v in value))
                if grouped:
                    # one PathEnsemble per system; cross-checks (atom counts,
                    # states, ...) run per system in `_process_and_check`.
                    if process_and_check:
                        self.__dict__['_reload_initial_paths'] = True
                    value = [group if isinstance(group, PathEnsemble)
                             else PathEnsemble(group) for group in value]
                else:
                    # single-system (unchanged): `_reload_initial_paths`
                    # controls whether we must reload from disk to recompute
                    # states on demand (avoids stale precomputed arrays).
                    if process_and_check:  # otw bug in reloading paths
                        self.__dict__['_reload_initial_paths'] = \
                            not isinstance(value, (Path, PathEnsemble))
                    initial_paths = PathEnsemble(value)
                    if value and not initial_paths:
                        raise TypeError(
                            f'could not get any initial paths from {value!r}')
                    value = initial_paths

            # name of the descriptors cache series (part of a file name)
            elif name == 'descriptors_series':
                value = check_descriptors_series(value)

            # engine
            elif name == 'engine':
                value = value.lower()
                if value not in ['gromacs', 'toy']:
                    raise TypeError(f'{name} must be either "gromacs" or '
                                    f'"toy", got {value}')

            # network
            elif name == 'network':
                # check that methods required at AIMMD runtime are present
                for attribute in ('forward', 'state_dict',
                                  'load_state_dict', 'parameters'):
                    if (not hasattr(value, attribute) or
                        not callable(getattr(value, attribute))):
                        raise TypeError(
                            f'{name} must have method {attribute!r}')
                update_source(value, name)

            # chain type
            elif name == 'chain_type':
                value = value.lower()
                if value.startswith('tp'):
                    value = 'tps'
                elif value.startswith('rf'):
                    value = 'rfps'
                elif value.startswith('cg'):
                    value = 'cgaimmd'
                elif value.startswith('fl'):
                    # the old name, read for one release
                    from ..cgaimmd import _legacy
                    value = _legacy.chain_type(value)
                    self.__dict__['_read_old_names'] = True
                else:
                    raise TypeError(f"{name!r} must be 'tps', 'rfps' or "
                                    f"'cgaimmd', got {value!r}")

            # pretrained initial network: a path (relative to the params file)
            elif name == 'initial_network':
                if value is not None:
                    if not isinstance(value, (str, os.PathLike)):
                        raise TypeError(
                            f"'initial_network' must be a path (str) or "
                            f"None, got {type(value).__name__}: {value!r}")
                    value = os.fspath(value)

            # list of strings
            elif expected_type is List[str]:
                if type(value) is not list:
                    raise TypeError(f'{name} must be list of strings, '
                                    f'got {type(value).__name__}')
                elif np.any([type(element) is not str for element in value]):
                    raise TypeError(f'{name!r} must be list of strings, '
                                    f'at least one of its elements is not')
            
            # free overriding bins
            elif name == 'free_overriding_bins':
                if isinstance(value, Iterable):
                    value = [int(v) for v in np.array(value, dtype=int)]
                elif isinstance(value, Integral):
                    value = [value]
                elif value is None:
                    pass
                else:
                    raise TypeError(f'{name!r} must index bins, got {value!r}')
            
            # density adjustment True -> inf, False -> 0
            elif name == 'density_adjustment':
                if isinstance(value, bool):
                    value = inf if value else 0
                elif value < inf:
                    value = round(value)
            
            # path
            elif name == 'path':
                if not PosixPath(value).exists():
                    raise TypeError(f'Source path {value!r} does not exist.')

            # all the rest
            elif not isinstance(value, expected_type):
                raise TypeError(f'{name} must be {expected_type}, '
                                f'got {type(value).__name__}')

        elif not name.startswith('_'):
            raise TypeError(f'{name!r} is not a valid Params dataclass field '
                            f'or a hidden parameter')

        # assign
        backup = self.__dict__.get(name, 'absent')
        self.__dict__[name] = value

        # special checks (after assignment)
        if process_and_check:
            try:
                self._process_and_check([name])
            except Exception as exception:  # go back in case of error
                if not isinstance(backup, str) or backup != 'absent':
                    self.__dict__[name] = backup
                # an unsupported combination keeps its type, so that it is the
                # same on assignment and on load (e.g. cgaimmd + multi_system)
                if isinstance(exception, NotImplementedError):
                    raise
                raise TypeError(
                    f'can\'t update {name!r} with {value!r}: '
                    f'{exception}') from exception

    def _process_and_check(self, fields=[]):
        """
        Post-assignment validation and derived-field updates.

        Parameters
        ----------
        fields : list of str, optional
            Names of fields that were assigned since the last check. This is
            used to minimize work (e.g., only recompute default values function
            if the network or transform changed).

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If consistency requirements fail, for example:
            - subsample_caps combined with a rate-like output (record_bias,
              rescale_committor),
            - rescaling requested but network does not implement Rescalable,
            - chain type incompatible with selection pool size,
            - invalid nbins/extra_bins combination,
            - initial paths not transitions or incompatible with topology.

        Notes
        -----
        This method may:
        - overwrite `values_function` if it is defaulted and the network changes,
        - reload initial paths from disk to recompute states if requested,
        - compute and attach descriptors/values to initial paths.
        """

        # subsample caps exclude rate-like outputs: each category is capped on
        # its own while the reweighting pools them, so rates and crossing
        # probabilities computed on the capped value-pass subsample are biased
        # (G2 test case: k_AB x26). They need the full path ensemble.
        if self.subsample_caps_active():
            wanted = [name for name in ('record_bias', 'rescale_committor')
                      if getattr(self, name, False)]
            if wanted:
                raise TypeError(
                    f"subsample_caps cannot be combined with "
                    f"{' or '.join(wanted)}: rates and crossing probabilities "
                    "computed on the capped value-pass subsample are biased, "
                    "they need the full path ensemble. Use subsample_caps only "
                    "for mechanism exploration, or set subsample_caps=None.")

        # check network
        if self.rescale_committor:
            if not isinstance(self.network, RescalableNetwork):
                raise TypeError(f"params' network {self.network} "
                                "is not aimmd.network.Rescalable")

        # check chain type
        if self.chain_type == 'tps' and self.selection_pool_size > 1:
            raise TypeError(f"when chain_type is 'tps', "
                            f"selection pool size must be 1")

        # C-GAIMMD: the rules of every chain type (initial_network, hooks,
        # C-GAIMMD networks), then those of chain_type='cgaimmd' that need
        # no initial paths
        self._check_cgaimmd_fields(fields)
        cgaimmd_checks = self._cgaimmd_rules_apply(fields)
        if cgaimmd_checks:
            if self.__dict__.get('_read_old_names', False):
                self._complete_old_generator()
            self._check_cgaimmd(fields)

        # check nbins
        if 'nbins' in fields or 'terminal_bin_extension' in fields:
            if self.nbins > 0:
                pass
            elif self.nbins == 0 and (
              self.terminal_bin_extension == 'all' or
              (self.states[+0] in self.terminal_bin_extension or
               self.states[-1] in self.terminal_bin_extension)):
                pass
            else:
                states = f'{self.states[+0]}{self.states[-1]}'
                raise TypeError(f'`nbins` must be > 0, or >= 0 when '
                                f'`extra_bins in ({states!r}, \'all\')`')
        
        # free-simulation restart source: re-validate against the final
        # `states` (field assignment order is the params file's, so `states`
        # may have arrived after the switch), and refuse a config that sets
        # both the switch and its deprecated predecessor.
        if (not fields
                or 'states' in fields
                or 'free_seeding_position' in fields
                or 'free_restart_source' in fields
                or 'restart_free_simulations_with_transitions' in fields):
            self.__dict__['free_seeding_position'] = \
                canonical_seeding_position(
                    self.free_seeding_position, states=self.states)
            self.__dict__['free_restart_source'] = \
                canonical_restart_source(
                    self.free_restart_source, states=self.states)
            self._free_restart_spec()   # raises TypeError on a conflict

        # check free_overriding_bins
        if 'free_overriding_bins' in fields:
            try:
                np.arange(self.nbins)[self.free_overriding_bins]
            except Exception as exception:
                raise TypeError(
                  f"can't determine free overriding bins when "
                  f"free_overriding_bins = {self.free_overriding_bins!r}, "
                  f"nbins = {self.nbins}: {exception}")
        
        # redefine values function (the protocol values function for a
        # C-GAIMMD network or chain type, see _make_default_values_function)
        default_values_function = self._default_values_function
        if (self.values_function is None or
           ('network' in fields or 'descriptor_transform' in fields or
            self._default_values_function_is_stale(fields)) and
            default_values_function):
            default_values_function = True
            self.__dict__['values_function'] = \
                self._make_default_values_function()
            check_values = True

        # no need: there is no default values function
        elif 'values_function' in fields:
            default_values_function = False

        # reassign default values function
        self.__dict__['_default_values_function'] = default_values_function

        # multi-system: build per-system universes and validate the per-system
        # groups of initial paths, then return (the single-system loop below
        # assumes one topology / one universe / one states_function).
        if getattr(self, 'multi_system', False):
            self._process_and_check_multi_system(fields)
            self._check_masses_from_topology(fields)
            return
        self._check_masses_from_topology(fields)

        # initial paths-related checks
        initial_paths = getattr(self, 'initial_paths', None)
        if not initial_paths:
            return
        check_paths = 'initial_paths' in fields
        check_states = ('states' in fields or
                        'states_function' in fields or
                        check_paths)
        check_descrs = (self.descriptors_function is not None and
                       ('descriptors_function' in fields or
                        'descriptors_series' in fields or check_paths))
        if self.descriptors_function is None:
            check_values = 'values_function' in fields or check_paths
        else:
            check_values = ('values_function' in fields or
                            'descriptors_function' in fields or
                            'descriptors_series' in fields or
                            check_paths)

        # go through paths
        for i, path in enumerate(initial_paths):
            if check_paths:
                # Ensure topology (if available) matches path atom count.
                try:
                    n_atoms = path.reader.n_atoms
                except:
                    n_atoms = path.reader.trajectory.n_atoms
                universe = self._universe
                if universe and n_atoms != len(universe.atoms):
                    raise TypeError(
                        f'{i}-th initial path {path.fname!r} has '
                        f'{n_atoms} atoms, while the topology file '
                        f'{self.topology!r} has {len(universe.atoms)} atoms')

            if check_states:
                # reload path if required
                if self._reload_initial_paths:
                    path = Path(path.fname)
                # recompute states (only if not manually overwritten)
                if 'states' not in path.__dict__:
                    path.states = path.compute(self.states_function)
                # NB the loop below replaces `path` with its transition
                # block, which starts AT the state boundary. The deep-basin
                # frames every `free_seeding_position` other than 'boundary'
                # needs are read back from `path.fname` on demand, by
                # `untrimmed_initial_paths()`.
                # throw a warning if no transition is found
                transition_found = False 
                for split_path in (split_paths := path.split()):
                    if split_path.type[:3] in (self.states, self.states[::-1]):
                        path = split_path
                        transition_found = True
                        break
                if not transition_found and self.chain_type == 'cgaimmd':
                    # end-state frames are valid C-GAIMMD initial paths: they
                    # seed the free simulations
                    print(f'C-GAIMMD: initial path {path.fname!r} has no '
                          f'{self.states!r} transition; it seeds free '
                          f'simulations only')
                elif not transition_found:
                    types = ", ".join([t[:3] for t in split_paths.types()])
                    print(f'Warning: the initial trajectory {path.fname!r} '
                        f'has no {self.states!r} transitions '
                        f'(path types: {types}), taking it as it is.\n'
                         'Perhaps you are using it for brute-force shooting?')
            
            if check_descrs:
                # Cache descriptors on the Path object, as the series array
                # (exported next to the initial path by the launcher).
                _store_seed_descriptors(
                    path, self.descriptors_series,
                    path.compute(self.descriptors_function))

            if cgaimmd_checks:
                # before the values check below, whose protocol values
                # function would raise its own (less specific) width error
                self._check_cgaimmd_descriptor_width(path, i)

            if check_values:
                # Ensure values_function returns exactly one value per frame.
                if self.descriptors_function is not None:
                    source = getattr(path, self.descriptors_series)[:1]
                else:
                    source = path.coordinates[:1]
                assert self.values_function(source).shape == (1,)

            # reassign path
            self.initial_paths[i] = path

        # C-GAIMMD: the rules that need the initial paths
        if cgaimmd_checks:
            self._check_cgaimmd_initial_paths()

    def _check_masses_from_topology(self, fields=[]):
        """
        With ``masses_from_topology``, read the topology masses now.

        Raises TypeError (at load or assignment rather than at the first shot)
        when the GROMACS topology grompp uses is missing, unreadable or has
        another atom count than ``topology``. Runs when all fields are
        processed or one it depends on was assigned; a no-op for the default
        ``masses_from_topology = False`` and for engines other than GROMACS,
        which keep the guessed masses (so that the default of the field can
        be flipped without touching toy-engine runs, spec K.2).
        """
        if not getattr(self, 'masses_from_topology', False):
            return
        if getattr(self, 'engine', 'gromacs') != 'gromacs':
            return
        if fields and not {'masses_from_topology', 'topology', 'gmx_grompp',
                           'engine', 'multi_system', 'system_ids'} & set(fields):
            return
        if getattr(self, 'multi_system', False):
            for system_id in self.system_ids:
                universe = self.universe_of(system_id)
                if universe is not None:
                    self._topology_masses(
                        len(universe.atoms),
                        self._per_system_value(self.gmx_grompp, system_id),
                        strict=True)
            return
        if self._universe is None:
            raise TypeError(f'masses_from_topology = True needs a structure file '
                            f'topology that MDAnalysis can read, got {self.topology!r}')
        self._topology_masses(len(self._universe.atoms), self.gmx_grompp, strict=True)

    # ------------------------------------------------------------------
    # C-GAIMMD (chain_type='cgaimmd')
    # ------------------------------------------------------------------

    def _cgaimmd_rules_apply(self, fields=[]):
        """Whether the ``chain_type='cgaimmd'`` rules run for this check.

        They run under ``chain_type='cgaimmd'`` when all fields are processed
        or one they read was assigned (``_CGAIMMD_RULE_FIELDS``).
        """
        return (getattr(self, 'chain_type', None) == 'cgaimmd'
                and (not fields or bool(_CGAIMMD_RULE_FIELDS & set(fields))))

    def _check_network_class_local(self):
        """The class of a C-GAIMMD network must not be an aimmd class.

        ``Params.save`` writes ``from <module of the class> import network``,
        which re-imports the network only when that module is the params file
        itself; the Launcher regenerates the params file for its workers, so
        this holds under every chain type.
        """
        module = type(self.network).__module__
        if module == 'aimmd' or module.startswith('aimmd.'):
            # the module name as update_source writes it (a local module
            # loaded from a path is reduced to its basename)
            written = _split(r'[\\/]', module)[-1]
            raise TypeError(
                f"define the network's class in the params file, e.g. "
                f"'class Network({type(self.network).__name__}): pass': "
                f"Params.save writes 'from {written} import network'")

    def _check_cgaimmd_fields(self, fields=[]):
        """The C-GAIMMD rules of every chain type.

        ``initial_network`` with ``multi_system`` is not supported and its
        file must exist (relative to the params file); C-GAIMMD hooks under
        another chain type are warned about; a C-GAIMMD network's class must
        be local whatever the chain type.
        """
        fields = set(fields)
        every = not fields
        initial_network = getattr(self, 'initial_network', None)
        if initial_network is not None and (
                every or fields & {'initial_network', 'multi_system'}):
            if getattr(self, 'multi_system', False):
                raise NotImplementedError(
                    'initial_network is not supported with multi_system')
            fname = self._initial_network_fname()
            if not os.path.isfile(fname):
                raise TypeError(
                    f'initial_network {initial_network!r} does not exist '
                    f'(looked for {fname!r}; a relative path is relative to '
                    f'the folder of the params file)')
        chain_type = getattr(self, 'chain_type', 'rfps')
        if chain_type != 'cgaimmd' and (
                every or fields & {'chain_type', *CGAIMMD_HOOK_FIELDS}):
            if any(getattr(self, hook, None) is not None
                   for hook in CGAIMMD_HOOK_FIELDS):
                print(f'Warning: cgaimmd_* hooks are set but chain_type is '
                      f'{chain_type!r}; they are ignored')
        if (chain_type != 'cgaimmd' and (every or 'network' in fields)
                and _is_cgaimmd_network(self.network)):
            self._check_network_class_local()

    def _complete_old_generator(self):
        """``cgaimmd_generator`` of params read through old names: the old worker's method.

        A network of the old code has no ``default_generator()``; the old
        worker used the network's own ``generate`` if it had one, else the ODE
        (``aimmd.cgaimmd._legacy.generator_spec``). The completed spec is
        stored, so that it is what ``Params.save`` writes.
        """
        from ..cgaimmd import _legacy
        spec = self.cgaimmd_generator
        new = _legacy.generator_spec(spec, self.network)
        if new is not spec:
            from ..cgaimmd.generation import canonical_generator_spec
            self.__dict__['cgaimmd_generator'] = canonical_generator_spec(new)

    def _check_cgaimmd(self, fields=[]):
        """The rules of ``chain_type='cgaimmd'`` that need no initial paths.

        1. the network implements the C-GAIMMD network protocol;
        2. the generator (``cgaimmd_generator``, else the network's default)
           can be built and can use the network. For params read through old
           names (a run of the old code) this is a warning: they load for
           analysis, and a shooting worker checks its generator at start-up;
        3. a single system;
        4. a ``descriptors_function`` (the flat Angstrom coordinates);
        5. no ``rescale_committor``;
        6. ``gen_temperature > 0``;
        7. a conditioning range ``lambda_clip``;
        8. no ``placement='none'`` with a prepare function;
        9. the network's class is local;
        10. warnings: no ``initial_network``, ``round_shooting_index=False``;
            under a committor acceptance rule (``'target'``, ``'window'``)
            ``window_after_trained_shots`` at its default; under the default
            ``'transition_region'`` the settings of the committor rules
            (``lambda_tolerance``, ``window_after_trained_shots``) given,
            since they have no effect there.
        """
        from ..cgaimmd.protocol import check_network_protocol, lambda_clip_of
        from ..cgaimmd.generation import make_generator, check_generator
        from ..cgaimmd._once import once
        network = self.network
        check_network_protocol(network)                                  # 1
        old_run = self.__dict__.get('_read_old_names', False)             # 2
        try:
            generator = make_generator(self.cgaimmd_generator, network)
            problems = check_generator(generator, network)
        except Exception as exception:
            if not old_run:
                raise
            problems = [f'{type(exception).__name__}: {exception}']
        if problems:
            message = (f"chain_type='cgaimmd': the generator cannot use "
                       f"{type(network).__name__}:\n  - " + '\n  - '.join(problems))
            if not old_run:
                raise TypeError(message)
            if once(f'params: old run generator {message}'):
                print(f'Warning: {message}\nThese params were read through old '
                      f'names (a run of the old code): they load for analysis, '
                      f'but no shooting worker can generate with them.')
        if getattr(self, 'multi_system', False):                          # 3
            raise NotImplementedError(
                "chain_type='cgaimmd' supports single systems only")
        if self.descriptors_function is None:                             # 4
            raise TypeError(
                "chain_type='cgaimmd' needs a descriptors_function returning "
                "the flat Angstrom coordinates of the network's atoms, e.g. "
                "GeneratorDescriptors (aimmd.cgaimmd.preparation.descriptors)")
        if self.rescale_committor:                                        # 5
            raise TypeError(
                "rescale_committor is not supported with chain_type="
                "'cgaimmd': it rescales the network's output in place and "
                "breaks the state of the generator fit")
        if not self.gen_temperature > 0:                                  # 6
            raise TypeError('generated shooting points carry no velocities; '
                            'set gen_temperature > 0')
        settings = self.cgaimmd_settings
        lambda_clip_of(network, settings)                                 # 7
        if (settings['placement'] == 'none'                               # 8
                and self.cgaimmd_prepare_function is not None):
            raise TypeError(
                "cgaimmd_parameters['placement'] = 'none' cannot be combined "
                "with cgaimmd_prepare_function: the placement is used only "
                "without a prepare function (drop one of the two)")
        self._check_network_class_local()                                 # 9
        if getattr(self, 'initial_network', None) is None and once(      # 10
                'params: cgaimmd without initial_network'):
            print("Warning: chain_type='cgaimmd' without initial_network: "
                  "the generator starts from params.network's constructed "
                  "weights")
        if not self.round_shooting_index and once(
                'params: cgaimmd with round_shooting_index=False'):
            print("WARNING: chain_type='cgaimmd' with round_shooting_index="
                  "False: reloaded shots may take the frame before t = 0 as "
                  "their shooting point (set round_shooting_index = True)")
        # a once-key of its own: the worker start-up prints the same line
        # into its worker log, after this one went to the launcher's output
        # before Worker.run redirected it
        user_keys = self.cgaimmd_parameters or {}
        rule = settings['acceptance']
        if rule in COMMITTOR_ACCEPTANCE_RULES:
            if ('window_after_trained_shots' not in user_keys
                    and once('params: window_after_trained_shots default')):
                default = CGAIMMD_PARAMETER_DEFAULTS['window_after_trained_shots']
                print(f'WARNING: window_after_trained_shots is at its default '
                      f'{default}; this is highly system dependent (it was 8000 '
                      f'for the NANMA test system), set it explicitly')
        else:
            unused = [key for key in ('lambda_tolerance', 'window_after_trained_shots')
                      if key in user_keys]
            if unused and once('params: committor rule settings without a committor rule'):
                default = (' (the default)' if 'acceptance' not in user_keys else '')
                print(f"WARNING: cgaimmd_parameters sets "
                      f"{', '.join(repr(key) for key in unused)}, which apply only "
                      f"with acceptance 'target' or 'window'; with acceptance "
                      f"{rule!r}{default} every candidate in the transition region "
                      f"is shot, whatever its committor")

    def _check_cgaimmd_descriptor_width(self, path, i=0):
        """Rule 11: descriptors of width ``3 * n_atoms_of(network)``.

        The descriptors are those of the run's ``descriptors_series`` that the
        initial-path check stored on `path`, else computed for its first
        frame.
        """
        from ..cgaimmd.protocol import n_atoms_of
        series = getattr(self, 'descriptors_series', 'descriptors')
        descriptors = path.__dict__.get(series)
        if descriptors is None:
            descriptors = path[:1].compute(self.descriptors_function)
        descriptors = np.asarray(descriptors)
        width = (int(np.prod(descriptors.shape[1:])) if descriptors.ndim > 1
                 else descriptors.size)
        n_atoms = n_atoms_of(self.network)
        if width != 3 * n_atoms:
            raise TypeError(
                f"chain_type='cgaimmd': the descriptors_function returns "
                f"descriptors of width {width} but the network generates "
                f"{n_atoms} atoms and needs 3 x {n_atoms} = {3 * n_atoms} (the "
                f"flat Angstrom coordinates of the network's atoms, x0, y0, "
                f"z0, x1, ...; series {series!r} of the {i}-th initial path "
                f"{path.fname!r})")

    def _check_cgaimmd_initial_paths(self):
        """Rules 12-14 of ``chain_type='cgaimmd'`` (the initial paths).

        12. at least one initial frame in each end state (they seed the free
            simulations);
        13. without a transition in the initial paths, the free simulations
            are seeded at the state boundary;
        14. without a prepare function, the network's atoms are the system.
        """
        from ..cgaimmd.protocol import n_atoms_of
        from ..worker.utils import transition_block
        states = self.states
        initial_paths = self.initial_paths
        labels = []
        for path in initial_paths:
            path_states = (path.states if 'states' in path.__dict__
                           else path.compute(self.states_function))
            labels.append(np.asarray(path_states).astype(str).ravel())
        labels = np.concatenate(labels) if labels else np.zeros(0, str)
        missing = [state for state in (states[0], states[2])
                   if not np.any(labels == state)]
        if missing:                                                       # 12
            raise TypeError(
                f"chain_type='cgaimmd' needs initial paths with at least one "
                f"frame in {states[0]!r} and one in {states[2]!r} (they seed "
                f"the free simulations); none in {', '.join(map(repr, missing))}")
        if all(transition_block(path, states) is None                    # 13
               for path in initial_paths):
            boundary = SEEDING_POSITION_ALIASES['boundary']
            for state in (states[0], states[2]):
                position = self.free_seeding_position_for(state)
                if (position == SEEDING_POSITION_RANDOM
                        or isinstance(position, str)
                        or float(position) != boundary):
                    raise TypeError(
                        f'free_seeding_position {position!r} for state '
                        f'{state!r} needs an initial path with a {states!r} '
                        f'transition (the other positions place the seed '
                        f'inside its transition block); the initial paths '
                        f'hold end-state frames only, so use the default '
                        f"'boundary'")
        if self.cgaimmd_prepare_function is None:                         # 14
            n_atoms = n_atoms_of(self.network)
            if self._universe is not None:
                n_system = len(self._universe.atoms)
            else:
                first = initial_paths[0]
                try:
                    n_system = first.reader.n_atoms
                except Exception:                              # noqa: BLE001
                    n_system = first.reader.trajectory.n_atoms
            if n_atoms != n_system:
                raise TypeError(
                    f'the network generates {n_atoms} atoms but the system '
                    f'has {n_system}: set cgaimmd_prepare_function, which '
                    f'turns the generated atoms into the full system')

    def _uses_protocol_values_function(self):
        """Whether the default values function is the protocol values function.

        It is under ``chain_type='cgaimmd'`` (whose rules require a protocol
        network), and under every chain type for a C-GAIMMD network
        (``aimmd.cgaimmd.protocol.is_cgaimmd_network``), so that
        post-training one on an RFPS or TPS run needs no values function.
        """
        return (getattr(self, 'chain_type', None) == 'cgaimmd'
                or _is_cgaimmd_network(self.network))

    def _make_default_values_function(self):
        """The default values function of ``self.network``.

        ``aimmd.cgaimmd.values.protocol_values_function`` when
        ``_uses_protocol_values_function``; AIMMD's
        ``create_default_values_function`` for every other network
        (unchanged).
        """
        if self._uses_protocol_values_function():
            from ..cgaimmd.values import protocol_values_function
            return protocol_values_function(self.network, self.descriptor_transform,
                                            params=self)
        return create_default_values_function(self.network,
                                              self.descriptor_transform)

    def _default_values_function_is_stale(self, fields=[]):
        """Whether the default values function must be built again.

        A protocol values function reads its device settings from the params
        it was built for, so a copy (``Params.copy``) gets its own; and an
        assignment of ``chain_type`` can switch between the protocol values
        function and AIMMD's.
        """
        module = sys.modules.get('aimmd.cgaimmd.values')
        function = self.__dict__.get('values_function')
        is_protocol = (module is not None
                       and isinstance(function, module.ProtocolValuesFunction))
        if is_protocol and function.params is not self:
            return True
        return ('chain_type' in fields
                and is_protocol != self._uses_protocol_values_function())

    def _process_and_check_multi_system(self, fields=[]):
        """
        Multi-system counterpart of the initial-path checks in
        `_process_and_check`.

        Builds one MDAnalysis Universe per system (cached in ``_universes``,
        keyed by ``system_id``) and validates each system's *group* of initial
        paths against that system's own topology and ``states_function``
        (called with ``system_id=``). Called only when ``multi_system`` is True.

        Multi-system data model
        ------------------------
        - ``topology``      : list of topology files, one per system.
        - ``system_ids``    : list of labels (defaults to ``'0','1',...``); they
          name the per-system subfolders ``<run>/<system_id>/``.
        - ``initial_paths`` : list of PathEnsembles, one group per system,
          parallel to ``system_ids``.
        """
        # topology must be a list in multi-system mode
        topology = self.topology
        if not isinstance(topology, (list, tuple)):
            raise TypeError("when multi_system=True, 'topology' must be a list "
                            "(one topology file per system), got "
                            f"{type(topology).__name__}")
        n_systems = len(topology)

        # system_ids: default to '0', '1', ... and validate uniqueness/length
        system_ids = (list(self.system_ids) if self.system_ids
                      else [str(i) for i in range(n_systems)])
        if len(system_ids) != n_systems:
            raise TypeError(f"len(system_ids)={len(system_ids)} must equal the "
                            f"number of topologies ({n_systems})")
        if len(set(system_ids)) != n_systems:
            raise TypeError(f"system_ids must be unique, got {system_ids}")
        self.__dict__['system_ids'] = system_ids

        # per-system list-valued fields: if given as a list, it must have one
        # entry per system (a scalar/dict is broadcast to all systems).
        for fname in ('bias_reactive_threshold', 'subsample_caps'):
            fval = getattr(self, fname, None)
            if isinstance(fval, (list, tuple)) and len(fval) != n_systems:
                raise TypeError(f"{fname!r} given as a list of length "
                                f"{len(fval)} but there are {n_systems} systems")

        # build (and cache) one universe per system
        universes = {}
        for sid, topo in zip(system_ids, topology):
            try:
                universes[sid] = Universe(topo)
            except Exception:
                universes[sid] = None
        self.__dict__['_universes'] = universes
        self.__dict__['_universe'] = None

        # initial paths: one group (PathEnsemble) per system
        initial_paths = getattr(self, 'initial_paths', None)
        if not initial_paths:
            return
        if isinstance(initial_paths, PathEnsemble):
            raise TypeError("when multi_system=True, 'initial_paths' must be a "
                            "list with one group of paths per system, got a "
                            "single flat PathEnsemble")
        if len(initial_paths) != n_systems:
            raise TypeError(f"'initial_paths' has {len(initial_paths)} groups "
                            f"but there are {n_systems} systems")

        check_paths = 'initial_paths' in fields
        check_states = ('states' in fields or 'states_function' in fields
                        or check_paths)
        check_descrs = (self.descriptors_function is not None and
                        ('descriptors_function' in fields or
                         'descriptors_series' in fields or check_paths))
        if self.descriptors_function is None:
            check_values = 'values_function' in fields or check_paths
        else:
            check_values = ('values_function' in fields or
                            'descriptors_function' in fields or
                            'descriptors_series' in fields or check_paths)
        values_takes_sid = accepts_system_id(self.values_function)

        # go through each system's group of paths
        for s, sid in enumerate(system_ids):
            universe = universes[sid]
            for i, path in enumerate(initial_paths[s]):
                if check_paths:
                    try:
                        n_atoms = path.reader.n_atoms
                    except:
                        n_atoms = path.reader.trajectory.n_atoms
                    if universe and n_atoms != len(universe.atoms):
                        raise TypeError(
                            f"system {sid!r}: {i}-th initial path "
                            f"{path.fname!r} has {n_atoms} atoms, while its "
                            f"topology {topology[s]!r} has "
                            f"{len(universe.atoms)} atoms")

                if check_states:
                    if self._reload_initial_paths:
                        path = Path(path.fname)
                    if 'states' not in path.__dict__:
                        path.states = path.compute(self.states_function,
                                                   system_id=sid)
                    transition_found = False
                    for split_path in (split_paths := path.split()):
                        if split_path.type[:3] in (self.states,
                                                   self.states[::-1]):
                            path = split_path
                            transition_found = True
                            break
                    if not transition_found:
                        types = ", ".join([t[:3] for t in split_paths.types()])
                        print(f"Warning: system {sid!r} initial trajectory "
                              f"{path.fname!r} has no {self.states!r} "
                              f"transitions (path types: {types}), taking it "
                              f"as it is.")

                if check_descrs:
                    _store_seed_descriptors(
                        path, self.descriptors_series,
                        path.compute(self.descriptors_function,
                                     system_id=sid))

                if check_values:
                    if self.descriptors_function is not None:
                        source = getattr(path, self.descriptors_series)[:1]
                    else:
                        source = path.coordinates[:1]
                    if values_takes_sid:
                        assert self.values_function(
                            source, system_id=sid).shape == (1,)
                    else:
                        assert self.values_function(source).shape == (1,)

                self.initial_paths[s][i] = path
