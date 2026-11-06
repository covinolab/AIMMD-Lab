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

check_descriptors_series(value)
    Validate the name of the descriptors cache series.

gromacs_topology_file, gromacs_include_dir, read_topology_masses, masses_disagreement
    The masses of the GROMACS topology grompp uses (``masses_from_topology``).

CGAIMMD_PARAMETER_DEFAULTS, canonical_cgaimmd_parameters(value)
    Defaults and validation of ``params.cgaimmd_parameters`` (the shooting
    loop and device settings of C-GAIMMD); ``Params.cgaimmd_settings`` merges
    the two. ``DEFAULT_LAMBDA_WINDOW`` is the default target window,
    ``CGAIMMD_ACCEPTANCE_RULES`` the rules of the final check and
    ``COMMITTOR_ACCEPTANCE_RULES`` those of them that filter by the committor.

is_device_spec(value)
    Whether a value is a device setting of ``cgaimmd_parameters``
    (``'auto'``, ``'cpu'``, ``'cuda'``, ``'cuda:N'``); shared with
    ``aimmd.cgaimmd.devices.resolve_device``.

Notes
-----
These helpers are intentionally lightweight and do not depend on higher-level
AIMMD components.
"""

# external
import os
import re
import math
import shlex
import shutil
import warnings
import numpy as np
import torch
from glob import glob
from re import split
from types import FunctionType as Function
from inspect import getsource
from numbers import Integral, Real

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


def resolve_path_names(value):
    """
    Absolute file names behind an ``initial_paths``-style value.

    Accepts what `PathEnsemble` accepts as a source of names: a single name, a
    glob pattern, an iterable of either (including the nested per-system lists a
    multi-system run uses), or already-built `Path` objects. NO readers are
    opened - this resolves names only.

    Relative names and patterns are resolved against the CURRENT working
    directory, so call it while chdir'd wherever those names are meant to be
    interpreted. `Params.load` chdirs into the params file's folder before it
    executes the file (`aimmd/params/_io.py`), so calling it from there inherits
    the loader's own resolution rule rather than reimplementing it: a bare name,
    a ``'../'`` name, an absolute name and a glob pattern all behave exactly as
    they do for the field itself.

    Returns
    -------
    list of str
        Absolute file names, glob patterns expanded and sorted, duplicates
        dropped in first-seen order. A name matching nothing is kept as its
        resolved absolute form rather than silently vanishing, so whatever tries
        to read it later can say which file is missing.
    """
    out = []

    def walk(item):
        if item is None:
            return
        if hasattr(item, 'fname'):          # a Path (or anything path-like)
            item = item.fname
        if isinstance(item, (str, os.PathLike)):
            text = os.fspath(item)
            matches = sorted(glob(text))
            for name in (matches or [text]):
                name = os.path.abspath(name)
                if name not in out:
                    out.append(name)
            return
        try:
            children = list(item)           # PathEnsemble, list, tuple, ...
        except TypeError:
            return
        for child in children:
            walk(child)

    walk(value)
    return out


# 'descriptors', or 'descriptors-' and a non-empty, file-name-safe suffix
_DESCRIPTORS_SERIES = re.compile(r'descriptors(-[A-Za-z0-9_.-]+)?')


def check_descriptors_series(value):
    """
    Validate a ``descriptors_series`` value.

    Parameters
    ----------
    value : object
        Candidate series name.

    Returns
    -------
    str
        `value`, unchanged.

    Raises
    ------
    ValueError
        Unless `value` is ``'descriptors'`` or ``'descriptors-'`` followed by
        letters, digits, ``'_'``, ``'.'`` or ``'-'``. The name becomes part of
        the cache file name ``{trajectory}.{descriptors_series}.npy``, and the
        ``'descriptors'`` prefix keeps it apart from the other series
        (states, values, bias, ...).
    """
    if not (isinstance(value, str) and _DESCRIPTORS_SERIES.fullmatch(value)):
        raise ValueError(
            f"descriptors_series must be 'descriptors' or 'descriptors-' "
            f"followed by letters, digits, '_', '.' or '-', got {value!r}")
    return value


# ════════════════════════════════════════════════════════════════════════════
# Masses of the GROMACS topology (params.masses_from_topology)
# ════════════════════════════════════════════════════════════════════════════

def gromacs_topology_file(gmx_grompp, parent='.'):
    """The topology file grompp reads for a ``gmx_grompp`` command.

    Parameters
    ----------
    gmx_grompp : str
        ``params.gmx_grompp``.
    parent : str or os.PathLike, optional
        The folder grompp runs in (``params.parent``); relative names are joined to it.

    Returns
    -------
    str
        The file after ``-p`` (the last one if repeated), else ``'topol.top'`` (grompp's default).
    """
    try:
        tokens = shlex.split(str(gmx_grompp))
    except ValueError:
        tokens = str(gmx_grompp).split()
    name = 'topol.top'
    for i, token in enumerate(tokens[:-1]):
        if token == '-p':
            name = tokens[i + 1]
    return os.path.join(os.fspath(parent), name)


def _share_top_of(executable):
    """``<prefix>/share/gromacs/top`` of a GROMACS executable ``<prefix>/bin/gmx``, else None."""
    if not executable:
        return None
    found = shutil.which(executable)
    if found is None:
        return None
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(found)))
    return os.path.join(prefix, 'share', 'gromacs', 'top')


def gromacs_include_dir(gmx_grompp=None):
    """GROMACS' force-field directory, where ``#include "<ff>.ff/..."`` lines are found.

    Parameters
    ----------
    gmx_grompp : str, optional
        ``params.gmx_grompp``; its first word is the GROMACS executable.

    Returns
    -------
    str or None
        The first existing directory among the candidates (see the notes); None if none exists.

    Notes
    -----
    Candidates, in order: the entries of ``GMXLIB``, ``$GMXDATA/top`` (set by
    GROMACS' ``GMXRC``), ``<prefix>/share/gromacs/top`` of the executable of
    `gmx_grompp` and of the executable AIMMD resolved at import
    (``aimmd._config.GROMACS``).
    """
    candidates = []
    for variable, suffix in (('GMXLIB', ''), ('GMXDATA', 'top')):
        for entry in os.environ.get(variable, '').split(os.pathsep):
            if entry:
                candidates.append(os.path.join(entry, suffix) if suffix else entry)
    executables = []
    if gmx_grompp:
        try:
            words = shlex.split(str(gmx_grompp))
        except ValueError:
            words = str(gmx_grompp).split()
        if words:
            executables.append(words[0])
    from .. import _config
    executables.append(getattr(_config, 'GROMACS', None))
    candidates.extend(_share_top_of(executable) for executable in executables)
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.normpath(candidate)
    return None


def read_topology_masses(top, *, include_dir=None):
    """The per-atom masses of the system of a GROMACS topology.

    Parameters
    ----------
    top : str or os.PathLike
        A ``.top`` file.
    include_dir : str, optional
        Where force-field includes are looked up (:func:`gromacs_include_dir`); None: MDAnalysis' default.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` float64 masses in amu, in system order (``[ molecules ]`` expanded).

    Raises
    ------
    ValueError
        If the file cannot be parsed, or a mass is missing, not finite or not positive.

    Notes
    -----
    Read with MDAnalysis' GROMACS ``ITPParser`` (``infer_system=True``): the
    mass column of ``[ atoms ]``, else the mass of the atom type. Local
    includes are resolved next to the including file, force-field includes
    in `include_dir`.
    """
    from MDAnalysis.topology.ITPParser import ITPParser
    kwargs = {'infer_system': True}
    if include_dir:
        kwargs['include_dir'] = str(include_dir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with ITPParser(os.fspath(top)) as parser:
                topology = parser.parse(**kwargs)
        masses = np.asarray(topology.masses.values, dtype=float)
    except Exception as exception:                          # noqa: BLE001
        raise ValueError(f'could not read the masses of {os.fspath(top)!r}: '
                         f'{type(exception).__name__}: {exception}') from exception
    if not len(masses) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError(f'{os.fspath(top)!r} gives {len(masses)} masses, not all of them '
                         f'finite and positive')
    return masses


def masses_disagreement(names, guessed, topology, *, rtol=0.01, atol=0.01, n_examples=5):
    """Describe where guessed and topology masses differ, or return None.

    Parameters
    ----------
    names : sequence of str
        Atom names, for the message.
    guessed, topology : array-like
        ``(N,)`` masses in amu.
    rtol, atol : float, optional
        Tolerances of ``numpy.isclose`` (force fields round masses differently).
    n_examples : int, optional
        How many distinct ``name: guessed vs topology`` pairs the message lists.

    Returns
    -------
    str or None
        ``'<k> atoms (CL: 35.45 vs 12.011, ...)'``; None when they agree.
    """
    guessed = np.asarray(guessed, dtype=float)
    topology = np.asarray(topology, dtype=float)
    differ = ~np.isclose(guessed, topology, rtol=rtol, atol=atol)
    if not differ.any():
        return None
    examples = []
    for i in np.flatnonzero(differ):
        example = f'{names[i]}: {guessed[i]:g} vs {topology[i]:g}'
        if example not in examples:
            examples.append(example)
        if len(examples) == n_examples:
            break
    return f'{int(differ.sum())} atoms ({", ".join(examples)})'


# ════════════════════════════════════════════════════════════════════════════
# C-GAIMMD settings (params.cgaimmd_parameters)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_LAMBDA_WINDOW = 'bins'
"""The default of ``cgaimmd_parameters['lambda_window']``.

``'bins'``: the targets are drawn inside the trainer's shooting-value cutoffs
(the inner finite bin edges; the whole conditioning range ``[-L, L]`` until
bins exist), so that no generated shot lands where the reweighting counts it
as a free excursion. Kept in this one place so that the decision can be
flipped; a flip also rewrites the ``lambda_window`` sentence of the
``cgaimmd_parameters`` description (``aimmd/params/_fields.py``), which names
the default.
"""

CGAIMMD_ACCEPTANCE_RULES = ('transition_region', 'target', 'window')
"""The values of ``cgaimmd_parameters['acceptance']``, the rule of the final check.

``'transition_region'`` (the default) shoots every prepared candidate that the
MD state function puts in the transition region and whose committor is finite;
it never filters by the network's committor. ``'target'`` and ``'window'``
filter by it once the network has been trained on
``window_after_trained_shots`` shooting points (see
:func:`aimmd.cgaimmd.source.final_check`).
"""

COMMITTOR_ACCEPTANCE_RULES = ('target', 'window')
"""The acceptance rules that filter candidates by the network's committor.

Only under these do ``lambda_tolerance``, ``window_after_trained_shots`` and
``raw_lambda_tolerance`` apply.
"""

DEFAULT_ACCEPTANCE = 'transition_region'
"""The default of ``cgaimmd_parameters['acceptance']``: no committor filter.

A filter on the network's own committor shapes the law of the shooting points
by the network being trained on them, so it is off unless a params file asks
for ``'target'`` or ``'window'``. Old params files (the field name
``flow_parameters``) that set no rule ran under ``'target'`` and are read with
it (:func:`aimmd.cgaimmd._legacy.params_field`).
"""

CGAIMMD_PARAMETER_DEFAULTS = {
    'lambda_law': 'uniform',
    'lambda_window': DEFAULT_LAMBDA_WINDOW,
    'lambda_clip': None,
    'acceptance': DEFAULT_ACCEPTANCE,
    'lambda_tolerance': 1.0,
    'window_after_trained_shots': 1000,
    'raw_lambda_tolerance': None,
    'n_targets': 64,
    'candidates_per_target': 2,
    'placement': 'box_center',
    'final_check_round': 'auto',
    'generation_device': 'auto',
    'values_device': 'auto',
    'training_device': 'auto',
    'max_refills': 20,
    'wait_seconds': 60.0,
    'generation_oom_halvings': 5,
    'generation_seed': None,
    'keep_scratch': False,
}
"""Defaults of every ``params.cgaimmd_parameters`` key.

A plain dict (it deep-copies, pickles and serializes like any settings
dict) that no code modifies: ``Params.cgaimmd_settings`` returns a new
``{**CGAIMMD_PARAMETER_DEFAULTS, **cgaimmd_parameters}``, and code that keeps
or changes settings copies it first (``dict(CGAIMMD_PARAMETER_DEFAULTS)``).
The stored ``cgaimmd_parameters`` holds only the keys the user gave. See the
``cgaimmd_parameters`` field description for the meaning of each key. How
candidates are generated is not set here but by ``params.cgaimmd_generator``
(:mod:`aimmd.cgaimmd.generation`).
"""

_DEVICE_SPEC = re.compile(r'(auto|cpu|cuda(:[0-9]+)?)')


def is_device_spec(value):
    """Whether `value` is a device setting of ``cgaimmd_parameters``.

    Parameters
    ----------
    value : object
        A candidate for ``generation_device``, ``values_device`` or ``training_device``.

    Returns
    -------
    bool
        True for exactly ``'auto'``, ``'cpu'``, ``'cuda'`` and ``'cuda:N'`` (``N >= 0``); False otherwise.

    Notes
    -----
    Only the syntax is checked (no CUDA query, no normalization of case or
    spaces). :func:`aimmd.cgaimmd.devices.resolve_device` turns an accepted
    string into a ``torch.device`` at use, inside a worker task.
    """
    return isinstance(value, str) and _DEVICE_SPEC.fullmatch(value) is not None


def _setting_int(key, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be an int >= '
                        f'{minimum}, got {value!r}')
    if value < minimum:
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be >= {minimum}, '
                        f'got {value!r}')
    return int(value)


def _setting_float(key, value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be a number, '
                        f'got {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be finite, '
                        f'got {value!r}')
    if positive and not value > 0:
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be > 0, '
                        f'got {value!r}')
    if nonnegative and not value >= 0:
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be >= 0, '
                        f'got {value!r}')
    return value


def _setting_choice(key, value, choices):
    if not (isinstance(value, str) and value in choices):
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be one of '
                        f'{", ".join(repr(c) for c in choices)}, '
                        f'got {value!r}')
    return value


def _setting_optional(check):
    def wrapped(key, value):
        return None if value is None else check(key, value)
    return wrapped


def _setting_window(key, value):
    if isinstance(value, str):
        if value != 'bins':
            raise TypeError(f"cgaimmd_parameters[{key!r}] must be a (lo, hi) "
                            f"pair or 'bins', got {value!r}")
        return value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"cgaimmd_parameters[{key!r}] must be a (lo, hi) pair "
                        f"of finite floats or 'bins', got {value!r}")
    lo, hi = (_setting_float(key, v) for v in value)
    if not lo < hi:
        raise TypeError(f'cgaimmd_parameters[{key!r}] needs lo < hi, '
                        f'got {value!r}')
    return (lo, hi)


def _setting_device(key, value):
    if not is_device_spec(value):
        raise TypeError(f"cgaimmd_parameters[{key!r}] must be 'auto', 'cpu', "
                        f"'cuda' or 'cuda:N', got {value!r}")
    return value


def _setting_rounding(key, value):
    if value is None:
        return None
    if isinstance(value, str):
        if value != 'auto':
            raise TypeError(f"cgaimmd_parameters[{key!r}] must be 'auto', None "
                            f"or an int >= 0 (decimals), got {value!r}")
        return value
    return _setting_int(key, value, 0)


def _setting_bool(key, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f'cgaimmd_parameters[{key!r}] must be True or False, '
                        f'got {value!r}')
    return bool(value)


_CGAIMMD_PARAMETER_CHECKS = {
    'lambda_law': lambda k, v: _setting_choice(k, v, ('uniform', 'bins')),
    'lambda_window': _setting_window,
    'lambda_clip': _setting_optional(
        lambda k, v: _setting_float(k, v, positive=True)),
    'acceptance': lambda k, v: _setting_choice(k, v, CGAIMMD_ACCEPTANCE_RULES),
    'lambda_tolerance': lambda k, v: _setting_float(k, v, positive=True),
    'window_after_trained_shots': lambda k, v: _setting_int(k, v, 0),
    'raw_lambda_tolerance': _setting_optional(
        lambda k, v: _setting_float(k, v, positive=True)),
    'n_targets': lambda k, v: _setting_int(k, v, 1),
    'candidates_per_target': lambda k, v: _setting_int(k, v, 1),
    'placement': lambda k, v: _setting_choice(k, v, ('box_center', 'none')),
    'final_check_round': _setting_rounding,
    'generation_device': _setting_device,
    'values_device': _setting_device,
    'training_device': _setting_device,
    'max_refills': lambda k, v: _setting_int(k, v, 1),
    'wait_seconds': lambda k, v: _setting_float(k, v, nonnegative=True),
    'generation_oom_halvings': lambda k, v: _setting_int(k, v, 0),
    'generation_seed': _setting_optional(lambda k, v: _setting_int(k, v, 0)),
    'keep_scratch': _setting_bool,
}


def canonical_cgaimmd_parameters(value):
    """Validate a ``params.cgaimmd_parameters`` value and normalize it.

    Parameters
    ----------
    value : dict or None
        The user's settings. Every key must be one of
        :data:`CGAIMMD_PARAMETER_DEFAULTS`.

    Returns
    -------
    dict or None
        ``None`` for ``None``; otherwise a new dict with **only the keys that
        were given**, in the given order, with lists turned into tuples and
        numbers into plain ``int``/``float`` (so the value round-trips through
        :meth:`aimmd.Params.save` without numpy reprs). Defaults are not
        filled in here; see ``Params.cgaimmd_settings``.

    Raises
    ------
    TypeError
        If `value` is not a dict, or for an unknown key, a wrong type or an
        out-of-range value; the message names the key. The ODE settings of
        the old ``flow_parameters`` (``ode_sampler``, ``ode_steps``,
        ``generate_chunk``) are unknown keys here: they belong to
        ``cgaimmd_generator``, and only the old field name is translated.
        Also if ``raw_lambda_tolerance`` is set while the acceptance rule
        (given or the default) does not filter by the committor: the
        prefilter is a committor filter and would contradict the rule.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"'cgaimmd_parameters' must be a dict or None, got "
                        f"{type(value).__name__}: {value!r}")
    out = {}
    for key, item in value.items():
        check = _CGAIMMD_PARAMETER_CHECKS.get(key)
        if check is None:
            hint = ''
            if key in ('ode_sampler', 'ode_steps', 'generate_chunk'):
                hint = (" (how candidates are generated is set by "
                        "cgaimmd_generator, e.g. cgaimmd_generator = "
                        "{'method': 'ode', 'sampler': 'midpoint', "
                        "'n_steps': 16, 'chunk': 128})")
            raise TypeError(
                f'unknown cgaimmd_parameters key {key!r}; allowed keys are '
                f'{list(CGAIMMD_PARAMETER_DEFAULTS)}{hint}')
        out[key] = check(key, item)
    rule = out.get('acceptance', CGAIMMD_PARAMETER_DEFAULTS['acceptance'])
    if out.get('raw_lambda_tolerance') is not None and rule not in COMMITTOR_ACCEPTANCE_RULES:
        raise TypeError(
            f"cgaimmd_parameters['raw_lambda_tolerance'] filters candidates by the "
            f"network's committor; it applies only with acceptance "
            f"{' or '.join(repr(name) for name in COMMITTOR_ACCEPTANCE_RULES)}, and the "
            f"acceptance is {rule!r}"
            f"{' (the default)' if 'acceptance' not in out else ''}: drop "
            f"raw_lambda_tolerance or set the acceptance rule explicitly")
    return out
