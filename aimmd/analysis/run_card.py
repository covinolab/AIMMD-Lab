"""
aimmd.analysis.run_card
=======================

Every setting of an AIMMD run in one record, for the parameter tables of a
paper's supplement and for comparing two runs.

A run's settings are spread over several places: the params file (states,
engine, network, fit function, sampling and reweighting settings), the
C-GAIMMD settings and the generator (for ``chain_type='cgaimmd'``), the
metadata the fit stores next to each network (``network{states}.json``: fit
name, keywords, seed, round), the static bias and its PLUMED files, and the
versions of the software. :func:`run_card` collects them into one plain,
JSON-serializable dict; :func:`run_card_rows` flattens it into
``(section, setting, value)`` rows that go straight into a table, with LaTeX
escaping on request.

Callables are recorded by their qualified name (``module.name``), a network
by its class, its number of parameters and its printed architecture, numbers
and strings as they are. Nothing is written: save the card where you want it
(``json.dump``), never into the run folder.
"""

# external
import dataclasses
import hashlib
import json
import os
from math import isfinite

import numpy as np


__all__ = ['RUN_CARD_SCHEMA', 'run_card', 'run_card_rows', 'describe_value']


RUN_CARD_SCHEMA = 'aimmd.analysis.run_card/1'
"""Identifies the layout of the dict :func:`run_card` returns."""

_MAX_SEQUENCE = 64
"""Longer lists and arrays are summarised (length, first and last items)."""


def _qualified_name(value):
    module = getattr(value, '__module__', None) or '?'
    name = getattr(value, '__qualname__', None) or type(value).__qualname__
    return f'{module}.{name}'


def describe_value(value):
    """A JSON-able description of one setting.

    - None, booleans, integers, finite floats and strings as they are
      (non-finite floats as the strings ``'inf'``, ``'-inf'``, ``'nan'``);
    - lists, tuples and dicts element by element (long sequences summarised);
    - numpy scalars and small arrays as numbers and lists;
    - a torch module as ``{'class', 'n_parameters'}``;
    - a function, method or class as its qualified name (``module.name``);
    - any other object as ``{'class': ..., 'repr': ...}``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return describe_value(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        if value.size <= _MAX_SEQUENCE:
            return describe_value(value.tolist())
        return {'array': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): describe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value) if not isinstance(value, (set, frozenset)) else sorted(value, key=str)
        if len(items) <= _MAX_SEQUENCE:
            return [describe_value(item) for item in items]
        return {'length': len(items), 'first': describe_value(items[:3]),
                'last': describe_value(items[-3:])}
    try:
        import torch
        if isinstance(value, torch.nn.Module):
            return {'class': _qualified_name(type(value)),
                    'n_parameters': int(sum(p.numel() for p in value.parameters()))}
    except ImportError:                                     # pragma: no cover
        pass
    if isinstance(value, type) or callable(value):
        return _qualified_name(value)
    text = repr(value)
    return {'class': _qualified_name(type(value)),
            'repr': text if len(text) <= 200 else text[:197] + '...'}


def _sha256(fname):
    try:
        with open(fname, 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _default_of(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        try:
            return field.default_factory()
        except Exception:                                   # noqa: BLE001
            return dataclasses.MISSING
    return dataclasses.MISSING


def _equal(a, b):
    try:
        result = a == b
        if isinstance(result, np.ndarray):
            return bool(result.all())
        return bool(result)
    except Exception:                                       # noqa: BLE001
        return False


def _params_section(params):
    """Every Params field, and the fields that differ from their defaults."""
    from ..params import Params
    values, non_default = {}, []
    for field in dataclasses.fields(Params):
        value = getattr(params, field.name, None)
        values[field.name] = describe_value(value)
        default = _default_of(field)
        if default is dataclasses.MISSING or not _equal(value, default):
            non_default.append(field.name)
    return values, non_default


def _network_section(params, run_dir):
    network = getattr(params, 'network', None)
    out = describe_value(network) if network is not None else None
    if isinstance(out, dict):
        out = dict(out)
        out['architecture'] = str(network)
        for member in ('settings', 'provenance'):
            function = getattr(network, member, None)
            if callable(function):
                try:
                    out[member] = describe_value(function())
                except Exception as error:                  # noqa: BLE001
                    out[member] = f'not available: {error}'
    record, fname = None, None
    if run_dir is not None:
        from ..worker.utils import network_metadata_fname
        fname = params._network_fname(run_dir)
        metadata = network_metadata_fname(fname)
        if os.path.isfile(metadata):
            try:
                with open(metadata) as handle:
                    record = json.load(handle)
            except (OSError, ValueError) as error:
                record = {'error': f'could not read {metadata}: {error}'}
    files = None
    if fname is not None:
        files = {'network': fname if os.path.isfile(fname) else None,
                 'network_sha256': _sha256(fname) if os.path.isfile(fname) else None}
    initial = getattr(params, 'initial_network', None)
    return out, record, files, initial


def _cgaimmd_section(params):
    """The C-GAIMMD settings with defaults, the generator and the hooks (None for other chains)."""
    spec = getattr(params, 'cgaimmd_generator', None)
    if getattr(params, 'chain_type', None) != 'cgaimmd' and spec is None:
        return None
    from ..cgaimmd.generation import canonical_generator_spec, make_generator
    out = {'settings': describe_value(params.cgaimmd_settings)}
    generator = {'spec': None, 'method': None, 'settings': None, 'error': None}
    try:
        canonical = canonical_generator_spec(spec)
        generator['spec'] = describe_value(canonical)
    except Exception as error:                              # noqa: BLE001
        generator['error'] = f'{type(error).__name__}: {error}'
    try:
        built = make_generator(spec, params.network)
        generator['method'] = getattr(built, 'method', None)
        generator['settings'] = describe_value(built.settings())
    except Exception as error:                              # noqa: BLE001
        generator['error'] = f'{type(error).__name__}: {error}'
    out['generator'] = generator
    out['hooks'] = {name: describe_value(getattr(params, name, None)) for name in (
        'cgaimmd_validity_function', 'cgaimmd_prepare_function', 'cgaimmd_minimize_function',
        'cgaimmd_lambda_sampler')}
    return out


def _engine_section(params):
    engine = getattr(params, 'engine', None)
    out = {'engine': engine}
    names = (('gmx_mdp', 'gmx_grompp', 'gmx_mdrun', 'gmx_eneconv') if engine == 'gromacs'
             else ('toy_mdrun', 'toy_slowdown'))
    for name in ('topology', 'trajectory_extension') + names:
        out[name] = describe_value(getattr(params, name, None))
    for name in ('gmx_mdp', 'topology'):
        fname = _resolved(params, getattr(params, name, None))
        if fname is not None:
            out[f'{name}_sha256'] = _sha256(fname)
    return out


def _resolved(params, fname):
    """`fname` as an existing file (relative names against the params folder), or None."""
    if not isinstance(fname, (str, os.PathLike)):
        return None
    fname = os.path.expanduser(os.fspath(fname))
    if not os.path.isabs(fname):
        fname = os.path.join(os.fspath(getattr(params, 'parent', None) or os.getcwd()), fname)
    return fname if os.path.isfile(fname) else None


def run_card(params, run_dir=None):
    """Every setting of a run in one JSON-serializable dict.

    Parameters
    ----------
    params : aimmd.Params or path-like
        The run's params (an object, or the params file, loaded without saving a copy).
    run_dir : path-like, optional
        The run folder (read only): adds the metadata the fit saved next to the
        run's network (``network{states}.json``) and the network file's sha256.

    Returns
    -------
    dict
        - ``schema``: :data:`RUN_CARD_SCHEMA`;
        - ``params_file`` and ``params_sha256``: the params file and its digest;
        - ``params``: every :class:`aimmd.Params` field (:func:`describe_value`);
        - ``non_default``: the fields that differ from their defaults;
        - ``states``: the sorted states; ``chain_type``;
        - ``engine``: the engine and its executables and files;
        - ``network``: class, number of parameters, printed architecture, and
          the network's own ``settings()`` / ``provenance()`` when it has them;
          ``network_files``: the run's network file and digest;
          ``initial_network``: the params' start network;
        - ``fit``: the fit function and the metadata of the run's last trained
          round (fit name, keywords, seed, round, trained shots), None without
          a run folder or a ``network{states}.json``;
        - ``cgaimmd``: the C-GAIMMD settings with their defaults, the generator
          (method and settings) and the preparation hooks; None for other
          chain types;
        - ``bias``: the static bias and its PLUMED files
          (:func:`aimmd.cgaimmd.sidecar.bias_settings`), None without a bias;
        - ``aimmd`` and ``software``: package version, commit and the versions of
          Python, numpy, torch and the optional packages.
    """
    from .reweighting import _load_params
    from ..cgaimmd.sidecar import aimmd_provenance, bias_settings, software_versions
    params = _load_params(params)
    run_dir = None if run_dir is None else os.path.abspath(str(run_dir))
    values, non_default = _params_section(params)
    network, record, files, initial = _network_section(params, run_dir)
    params_file = getattr(params, 'path', None)
    params_file = os.fspath(params_file) if params_file else None
    try:
        bias = bias_settings(params)
    except Exception as error:                              # noqa: BLE001
        bias = {'error': f'{type(error).__name__}: {error}'}
    return {
        'schema': RUN_CARD_SCHEMA,
        'params_file': params_file,
        'params_sha256': (_sha256(params_file) if params_file and os.path.isfile(params_file)
                          else None),
        'run_dir': run_dir,
        'states': getattr(params, 'sorted_states', None),
        'chain_type': getattr(params, 'chain_type', None),
        'params': values,
        'non_default': non_default,
        'engine': _engine_section(params),
        'network': network,
        'network_files': files,
        'initial_network': describe_value(initial),
        'fit': {'function': describe_value(getattr(params, 'fit', None)),
                'last_round': describe_value(record)},
        'cgaimmd': _cgaimmd_section(params),
        'bias': describe_value(bias),
        'aimmd': describe_value(aimmd_provenance()),
        'software': describe_value(software_versions()),
    }


_LATEX = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
          '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
          '^': r'\textasciicircum{}'}


def _latex(text):
    return ''.join(_LATEX.get(character, character) for character in text)


def _format(value):
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def run_card_rows(card, *, sections=None, latex=False, skip_none=False):
    """Flatten a run card into ``(section, setting, value)`` rows of strings.

    Parameters
    ----------
    card : dict
        From :func:`run_card`.
    sections : sequence of str, optional
        The top-level keys to include (default: all), in that order.
    latex : bool, optional
        Escape the LaTeX special characters (``_``, ``&``, ``%``, ...) so that the
        rows can go into a ``tabular`` as they are.
    skip_none : bool, optional
        Leave out settings whose value is None.

    Returns
    -------
    list of tuple
        Nested dicts give dotted setting names (``generator.settings.n_steps``);
        lists and other values are written as text (lists as JSON).
    """
    rows = []

    def walk(section, prefix, value):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                walk(section, f'{prefix}.{key}' if prefix else str(key), item)
            return
        if value is None and skip_none:
            return
        rows.append((section, prefix, _format(value)))

    for section in (sections or list(card)):
        value = card.get(section)
        if isinstance(value, dict):
            walk(section, '', value)
        elif not (value is None and skip_none):
            rows.append((section, '', _format(value)))
    if latex:
        rows = [tuple(_latex(text) for text in row) for row in rows]
    return rows
