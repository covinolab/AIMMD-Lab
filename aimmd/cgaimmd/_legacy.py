"""
aimmd.cgaimmd._legacy
=====================

The old names of C-GAIMMD, read for one release.

C-GAIMMD was developed as "flow shooting" (F-AIMMD) in the package
``aimmd.flow``. The production runs of that time wrote the old names into their
params files (``chain_type = 'flow'``, the ``flow_*`` fields, imports from
``aimmd.flow``), sidecars (``*.xtc.flow.json``) and in-flight markers. So that
those runs can still be loaded, resumed and analysed with the new tools, the
old names are **read** for one release and never written:

- every old name in use emits exactly one ``DeprecationWarning`` per process
  (and one log line) that names the new spelling; this includes the
  restraint modes of the preparation tools (``restrain='flow_heavy'``), the
  private polish of the archived params (``aimmd.flow.validate._polish``,
  now the public ``aimmd.cgaimmd.preparation.minimize.polish``) and the
  attributes of :data:`ATTRIBUTES`;
- this module holds every old -> new table, and the compatibility modules
  ``aimmd.flow`` and ``aimmd.network.{flow_fit,joint_painn}`` only resolve
  names through these tables;
- code that reads files with old names (sidecars, markers, restraint modes)
  translates them through :func:`translate` with the tables below.

A name of the tables whose new module is not part of this installation (yet)
raises an ``ImportError`` that names the new module, and a name that was
dropped raises an ``ImportError`` with the reason. Any other name a
compatibility module cannot resolve raises ``AttributeError``, so that
``hasattr`` and other introspection keep working on them.

Standard library only.
"""

import importlib
import sys
import threading
import types
import warnings


# ════════════════════════════════════════════════════════════════════════════
# Params
# ════════════════════════════════════════════════════════════════════════════

PARAMS_FIELDS = {
    'flow_parameters': 'cgaimmd_parameters',
    'flow_validity_function': 'cgaimmd_validity_function',
    'flow_prepare_function': 'cgaimmd_prepare_function',
    'flow_minimize_function': 'cgaimmd_minimize_function',
    'flow_lambda_sampler': 'cgaimmd_lambda_sampler',
}
"""Old params field -> new params field."""

GENERATOR_SETTINGS = {
    'ode_sampler': 'sampler',
    'ode_steps': 'n_steps',
    'generate_chunk': 'chunk',
}
"""Keys of the old ``flow_parameters`` -> settings of ``cgaimmd_generator``.

``ode_sampler`` and ``ode_steps`` configured the flow-matching ODE, so they
become ``{'method': 'ode', ...}``. ``generate_chunk`` alone was the chunk of
whatever generated (also a network's own ``generate``), so it becomes a
settings dict without ``'method'``, which overrides the chunk of the network's
default generation method.
"""

_ODE_SETTINGS = ('ode_sampler', 'ode_steps')

CHAIN_TYPES = {'flow': 'cgaimmd'}
"""Old ``chain_type`` -> new (any value starting with ``'fl'`` was stored as ``'flow'``)."""


# ════════════════════════════════════════════════════════════════════════════
# Names on disk and in hook objects
# ════════════════════════════════════════════════════════════════════════════

RESTRAINT_MODES = {'flow_heavy': 'generator_heavy', 'flow_all': 'generator_all'}
"""Old ``restrain=`` strings of the GROMACS preparation tools -> new."""

SIDECAR_SUFFIXES = {'.flow.json': '.cgaimmd.json'}
"""Per-shot sidecar ``chain*/path*.xtc<suffix>``: old -> new."""

SIDECAR_SCHEMAS = {'aimmd.flow.shot/1': 'aimmd.cgaimmd.shot/1'}
"""The ``schema`` of a per-shot sidecar: old -> new."""

SIDECAR_SOURCES = {'flow': 'cgaimmd'}
"""The ``source`` field of a per-shot sidecar: old -> new."""

GENERATOR_LABELS = {'network.generate': 'bioemu2_edm', 'aimmd sampler': 'ode',
                    'aimmd.flow.sampler.generate': 'ode'}
"""How an old run generated -> the generation method name.

Old sidecars (``generator.method``) and validation reports
(``settings['generate']``) recorded *how* the network was asked to generate:
through its own ``generate`` (``'network.generate'``; only the BioEmu2
networks had one) or through AIMMD's ODE sampler (``'aimmd sampler'`` in
sidecars, ``'aimmd.flow.sampler.generate'`` in validation reports).
"""

VALIDATION_SCHEMAS = {'aimmd.flow.validate/1': 'aimmd.cgaimmd.validate/1'}
"""The ``schema`` of a validation report: old -> new."""

IN_FLIGHT_MARKERS = {'.flow_shooting_point.json': '.cgaimmd_shooting_point.json',
                     '.flow_shooting_point.npy': '.cgaimmd_shooting_point.npy'}
"""In-flight marker files of a chain folder: old -> new."""

SCRATCH_FOLDERS = {'flow_scratch': 'cgaimmd_scratch'}
"""Scratch folder of a chain: old -> new (scratch only; nothing reads the old one)."""

ATTRIBUTES = {('GeneratorDescriptors', 'flow_atoms'): 'generator_atoms'}
"""(class, old attribute) -> new attribute; the class keeps the old one as a property that warns."""


# ════════════════════════════════════════════════════════════════════════════
# Modules and their objects
# ════════════════════════════════════════════════════════════════════════════

MODULES = {
    'aimmd.flow': 'aimmd.cgaimmd',
    **{f'aimmd.flow.{name}': f'aimmd.cgaimmd.{name}'
       for name in ('_once', 'types', 'devices', 'io', 'protocol', 'values', 'lambda_law',
                    'pool', 'sidecar')},
    'aimmd.flow.worker': 'aimmd.cgaimmd.source',
    'aimmd.flow.posttrain': 'aimmd.cgaimmd.restart',
    **{f'aimmd.flow.{name}': f'aimmd.cgaimmd.flow_matching.{name}'
       for name in ('sampler', 'toy', 'flowmath', 'prior', 'embedding', 'graphs', 'pretrain')},
    **{f'aimmd.flow.{name}': f'aimmd.cgaimmd.preparation.{name}'
       for name in ('gromacs', 'minimize', 'solvate', 'topology', 'descriptors', 'validity',
                    'symmetry')},
    **{f'aimmd.flow.{name}': f'aimmd.cgaimmd.analysis.{name}'
       for name in ('validate', '_validate_plots')},
    'aimmd.flow.harmonic': None,
    'aimmd.network.flow_fit': 'aimmd.cgaimmd.flow_matching.fit',
    'aimmd.network.joint_painn': 'aimmd.cgaimmd.flow_matching.painn',
}
"""Old module -> the new module its names are looked up in (None: dropped).

A name of an old module is looked up under the same name in the new module,
unless :data:`NAMES` says otherwise.
"""

NAMES = {
    'aimmd.flow.types': {
        'FlowContext': ('aimmd.cgaimmd.types', 'CandidateContext')},
    'aimmd.flow.descriptors': {
        'FlowAtoms': ('aimmd.cgaimmd.preparation.descriptors', 'GeneratorAtoms'),
        'FlowDescriptors': ('aimmd.cgaimmd.preparation.descriptors', 'GeneratorDescriptors')},
    'aimmd.flow.values': {
        'FlowValuesFunction': ('aimmd.cgaimmd.values', 'ProtocolValuesFunction'),
        'flow_values_function': ('aimmd.cgaimmd.values', 'protocol_values_function')},
    'aimmd.flow.protocol': {
        'is_flow_network': ('aimmd.cgaimmd.protocol', 'is_cgaimmd_network'),
        'to_flow': ('aimmd.cgaimmd.flow_matching.sampler', 'to_flow'),
        'from_flow': ('aimmd.cgaimmd.flow_matching.sampler', 'from_flow'),
        'pair': ('aimmd.cgaimmd.flow_matching.sampler', 'pair')},
    'aimmd.flow.worker': {
        'FlowShootingPointSource': ('aimmd.cgaimmd.source', 'GenerativeShootingPointSource'),
        'make_flow_source': ('aimmd.cgaimmd.source', 'make_cgaimmd_source'),
        'FLOW_READBACK_FAILURE_IS_REJECTION': ('aimmd.cgaimmd.source',
                                               'READBACK_FAILURE_IS_REJECTION')},
    'aimmd.flow.posttrain': {
        'check_flow_restart': ('aimmd.cgaimmd.restart', 'check_cgaimmd_restart'),
        'TPS_RESTART_AS_FLOW': ('aimmd.cgaimmd.restart', 'TPS_RESTART_AS_CGAIMMD')},
    'aimmd.flow.validate': {
        '_polish': ('aimmd.cgaimmd.preparation.minimize', 'polish')},
    'aimmd.network.flow_fit': {
        'FlowDataset': ('aimmd.cgaimmd.dataset', 'TrainingDataset'),
        'extract_flow_dataset': ('aimmd.cgaimmd.dataset', 'extract_dataset'),
        'path_weights': ('aimmd.cgaimmd.dataset', 'path_weights'),
        '_resolve_exclusions': ('aimmd.cgaimmd.dataset', 'resolve_exclusions'),
        'flow_fit': ('aimmd.cgaimmd.flow_matching.fit', 'flow_matching_fit')},
}
"""Old module -> {old name: (new module, new name)} for the names that moved or were renamed."""

DROPPED_MODULES = {
    'aimmd.flow.harmonic': 'the harmonic prior is not part of C-GAIMMD; use the Gaussian '
                           'prior (aimmd.cgaimmd.flow_matching.prior) with topology features',
}
"""Old modules without a replacement -> why."""

DROPPED_NAMES = {
    ('aimmd.flow.harmonic', 'HarmonicPrior'):
        'the harmonic prior is not part of C-GAIMMD; use the Gaussian prior '
        '(aimmd.cgaimmd.flow_matching.prior) with topology features',
    ('aimmd.flow.prior', 'MixturePrior'):
        'the mixture prior is not part of C-GAIMMD; use the Gaussian prior',
    ('aimmd.flow.validate', 'cfaimmd_architecture'):
        'the converters of prototype (cfaimmd) checkpoints are not part of C-GAIMMD',
    ('aimmd.flow.validate', 'joint_painn_from_cfaimmd'):
        'the converters of prototype (cfaimmd) checkpoints are not part of C-GAIMMD',
    ('aimmd.network.joint_painn', 'convert_cfaimmd_state_dict'):
        'the converters of prototype (cfaimmd) checkpoints are not part of C-GAIMMD',
    ('aimmd.flow.sampler', 'sample'):
        'generation goes through aimmd.cgaimmd.generation: '
        'make_generator(params.cgaimmd_generator, network).generate(network, lam)',
    ('aimmd.flow.protocol', 'has_generate'):
        'generate is not a member of C-GAIMMD networks; generation goes through '
        'aimmd.cgaimmd.generation',
}
"""(old module, old name) without a replacement -> why."""


# ════════════════════════════════════════════════════════════════════════════
# Warnings
# ════════════════════════════════════════════════════════════════════════════

_WARNED = set()
_LOCK = threading.Lock()


def warn_renamed(old, new, *, key=None, stacklevel=3):
    """Emit the one ``DeprecationWarning`` (and log line) of an old name.

    Parameters
    ----------
    old, new : str
        The old and the new spelling, as they appear in the message.
    key : hashable, optional
        What counts as "the same old name" (default: `old`); each key warns
        once per process.
    stacklevel : int, optional
        Passed to :func:`warnings.warn`.

    Returns
    -------
    bool
        True if the warning was emitted now, False if it was emitted before.
    """
    key = old if key is None else key
    with _LOCK:
        if key in _WARNED:
            return False
        _WARNED.add(key)
    message = (f'{old} is deprecated; use {new} (the old names of C-GAIMMD are read for '
               f'one release)')
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    # a DeprecationWarning is hidden by default outside __main__; the log line is not
    from .. import _config
    getattr(_config, 'print', print)(f'Warning: {message}')
    return True


def _reset():
    """Forget which old names were warned about (test isolation; not for production code)."""
    with _LOCK:
        _WARNED.clear()


def translate(table, value, what):
    """The new spelling of `value` according to `table`, warning once if it was old.

    Parameters
    ----------
    table : dict
        One of the old -> new tables of this module.
    value : object
        The value read (a string from a params file, a sidecar, a file name part).
    what : str
        What the value is, for the message (e.g. ``'restrain'``).

    Returns
    -------
    object
        ``table[value]`` for an old value (after one ``DeprecationWarning``),
        else `value` unchanged.
    """
    try:
        new = table.get(value)
    except TypeError:                                       # unhashable: never an old name
        return value
    if new is None:
        return value
    warn_renamed(f'{what} {value!r}', f'{what} {new!r}', key=(what, value), stacklevel=4)
    return new


def chain_type(value):
    """``'cgaimmd'`` for an old ``chain_type`` (starting with ``'fl'``), with one warning."""
    warn_renamed(f'chain_type {value!r}', "chain_type 'cgaimmd'", key=('chain_type', 'flow'),
                 stacklevel=5)
    return CHAIN_TYPES['flow']


def params_field(name, value):
    """The new params fields of an old field `name` assigned `value`, with one warning.

    Parameters
    ----------
    name : str
        A key of :data:`PARAMS_FIELDS`.
    value : object
        The value given for it.

    Returns
    -------
    dict
        ``{new field: value}``. For ``flow_parameters``, the keys of
        :data:`GENERATOR_SETTINGS` are moved to ``cgaimmd_generator``:
        ``{'method': 'ode', 'sampler': ..., 'n_steps': ..., 'chunk': ...}``
        when an ODE setting is among them, else ``{'chunk': ...}`` (settings
        of the network's default method), listed before
        ``cgaimmd_parameters``.
    """
    new = PARAMS_FIELDS[name]
    out = {}
    note = ''
    if name == 'flow_parameters' and isinstance(value, dict):
        moved = [key for key in value if key in GENERATOR_SETTINGS]
        if moved:
            settings = {GENERATOR_SETTINGS[key]: value[key] for key in moved}
            if any(key in _ODE_SETTINGS for key in moved):
                settings = {'method': 'ode', **settings}
            out['cgaimmd_generator'] = settings
            note = (f", and cgaimmd_generator = {settings!r} for its keys "
                    f"{', '.join(repr(key) for key in moved)}")
        value = {key: item for key, item in value.items() if key not in GENERATOR_SETTINGS}
    out[new] = value
    warn_renamed(f'the params field {name!r}', f'{new!r}{note}', key=('field', name),
                 stacklevel=5)
    return out


def both_names_advice(name, value):
    """What to do when the old field `name` (set to `value`) and its new name are both set.

    For ``flow_parameters`` it says which keys go to ``cgaimmd_generator``
    (under their new names) and which to ``cgaimmd_parameters``: deleting
    the old field alone would drop the loop settings it holds.
    """
    new = PARAMS_FIELDS[name]
    if name != 'flow_parameters' or not isinstance(value, dict):
        return f'keep {new!r} only'
    moved = [key for key in value if key in GENERATOR_SETTINGS]
    rest = [key for key in value if key not in GENERATOR_SETTINGS]
    parts = []
    if moved:
        parts.append(f"its keys {', '.join(repr(key) for key in moved)} into "
                     f"'cgaimmd_generator' (as "
                     f"{', '.join(repr(GENERATOR_SETTINGS[key]) for key in moved)})")
    if rest:
        parts.append(f"its keys {', '.join(repr(key) for key in rest)} into "
                     f"'cgaimmd_parameters'")
    if not parts:
        return f'delete {name!r}'
    return f"move {' and '.join(parts)}, then delete {name!r}"


def generator_spec(spec, network):
    """The ``cgaimmd_generator`` of an old run, with the method its worker chose.

    The old worker had no generation method to pick: it called the network's
    own ``generate`` when the network had one (only the BioEmu2 networks did;
    the ODE settings were passed to it, and it used its own) and AIMMD's ODE
    sampler otherwise. A network of that time has no ``default_generator()``,
    so a spec that leaves the method to the network (None, or the settings
    dict that ``flow_parameters['generate_chunk']`` became) would leave the
    run without a method. This completes it the way the old worker chose
    (:data:`GENERATOR_LABELS`), and never gives ``'ode'`` to a network with
    its own ``generate``: its velocity field may be a stand-in that the old
    worker never integrated.

    Parameters
    ----------
    spec : str, dict, generator object or None
        The ``cgaimmd_generator`` of params read through old names.
    network : object
        Their network.

    Returns
    -------
    object
        `spec` itself when the network has a ``default_generator()``, when
        the spec names a method other than ``'ode'``, or names ``'ode'`` for a
        network without its own ``generate``. Else a new settings dict:
        ``{'method': 'bioemu2_edm', 'chunk': ...}`` for a network with its
        own ``generate`` (other settings dropped: that ``generate`` ignored
        them), ``{'method': 'ode', **settings}`` for any other network. One
        ``DeprecationWarning`` per process and result names it.
    """
    if callable(getattr(network, 'default_generator', None)):
        return spec
    own_generate = callable(getattr(network, 'generate', None))
    if spec is None or (isinstance(spec, str) and spec == 'ode'):
        settings, named = {}, spec is not None
    elif isinstance(spec, dict) and spec.get('method', 'ode') == 'ode':
        settings = {key: item for key, item in spec.items() if key != 'method'}
        named = 'method' in spec
    else:
        return spec
    if not own_generate:
        if named:                                           # 'ode', as the old worker used
            return spec
        new = {'method': GENERATOR_LABELS['aimmd sampler'], **settings}
        how = "AIMMD's ODE sampler"
    else:
        ignored = [key for key in settings if key != 'chunk']
        new = {'method': GENERATOR_LABELS['network.generate'],
               **{key: item for key, item in settings.items() if key == 'chunk'}}
        how = 'its own generate()'
        if ignored:
            how += f", which ignored {', '.join(repr(key) for key in ignored)}"
    warn_renamed(f'letting the old worker pick the generation method of '
                 f'{type(network).__name__} (it used {how})', f'cgaimmd_generator = {new!r}',
                 key=('generator', type(network).__name__, repr(new)), stacklevel=4)
    return new


# ════════════════════════════════════════════════════════════════════════════
# Compatibility modules
# ════════════════════════════════════════════════════════════════════════════

def _resolve(old_module, name):
    """The object an old module's `name` stands for (one warning per old name).

    Only the names listed in :data:`NAMES` and :data:`DROPPED_NAMES` raise
    ``ImportError`` when they cannot be resolved; for any other name the error
    is an ``AttributeError`` (``hasattr`` asks compatibility modules for
    arbitrary names, e.g. code that walks ``sys.modules``).
    """
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(f'module {old_module!r} has no attribute {name!r}')
    reason = DROPPED_NAMES.get((old_module, name))
    if reason is not None:
        raise ImportError(f'{old_module}.{name} has no replacement: {reason}')
    reason = DROPPED_MODULES.get(old_module)
    if reason is not None:
        raise AttributeError(f'module {old_module!r} has no attribute {name!r} ({old_module} '
                             f'has no replacement: {reason})')
    listed = name in NAMES.get(old_module, {})
    new_module, new_name = NAMES.get(old_module, {}).get(name, (MODULES[old_module], name))
    try:
        module = importlib.import_module(new_module)
    except ModuleNotFoundError as exc:
        missing = exc.name is not None and (new_module == exc.name
                                            or new_module.startswith(exc.name + '.'))
        if not missing:
            raise
        message = (f'{old_module}.{name} would be {new_module}.{new_name}, but {new_module} '
                   f'is not available in this installation ({exc})')
        if not listed:
            raise AttributeError(f'module {old_module!r} has no attribute {name!r}; '
                                 f'{message}') from None
        raise ImportError(message) from exc
    try:
        value = getattr(module, new_name)
    except AttributeError:
        raise AttributeError(f'module {old_module!r} has no attribute {name!r} (looked for '
                             f'{new_module}.{new_name})') from None
    warn_renamed(f'{old_module}.{name}', f'{new_module}.{new_name}',
                 key=('name', old_module, name), stacklevel=4)
    return value


def module_getattr(old_module):
    """A module ``__getattr__`` that resolves the names of `old_module` through the tables."""
    def __getattr__(name):
        return _resolve(old_module, name)
    return __getattr__


def _old_names(old_module):
    return sorted(NAMES.get(old_module, {}))


class _OldModule(types.ModuleType):
    """A compatibility module of :data:`MODULES` whose names resolve through the tables."""

    def __getattr__(self, name):
        return _resolve(self.__name__, name)

    def __dir__(self):
        return sorted({*super().__dir__(), *_old_names(self.__name__)})


def install_package_shim(package):
    """Give the old package `package` (``aimmd.flow``) its compatibility submodules.

    Every old submodule of :data:`MODULES` under ``package.__name__`` becomes a
    module in ``sys.modules`` and an attribute of `package`, so that
    ``import aimmd.flow.types`` and ``from aimmd.flow import validate`` work.
    Their names resolve on use, through :func:`_resolve`.
    """
    prefix = package.__name__ + '.'
    for old in MODULES:
        if not old.startswith(prefix) or '.' in old[len(prefix):]:
            continue
        module = sys.modules.get(old)
        if not isinstance(module, _OldModule):
            module = _OldModule(old, f'Compatibility module of {old}: see aimmd.cgaimmd._legacy.')
            module.__package__ = package.__name__
            sys.modules[old] = module
        setattr(package, old[len(prefix):], module)
