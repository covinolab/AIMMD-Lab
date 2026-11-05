"""The old names of C-GAIMMD, read for one release (``aimmd.cgaimmd._legacy``).

The production runs of the "flow shooting" era wrote ``chain_type = 'flow'``,
the ``flow_*`` fields and imports from ``aimmd.flow`` into their params files.
They are read with exactly one ``DeprecationWarning`` per old name that names
the new spelling, and only the new names are ever written.
"""

import importlib
import importlib.util
import os
import textwrap
import warnings

import numpy as np
import pytest

import aimmd
from aimmd.cgaimmd import _legacy
from aimmd.params._helpers import _LEGACY_FIELDS
from tests._helpers_unit import write_trajectory


PARAMS_HEAD = '''\
import numpy as np
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork

engine = 'toy'
states = 'ARB'
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
round_shooting_index = True
gen_temperature = 300.0
initial_network = None

def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01

def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))

def descriptors_function(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)

class Network(TinyFlowNetwork):
    pass

network = Network()
'''

NEW = "chain_type = 'cgaimmd'\ncgaimmd_parameters = {'placement': 'none', 'lambda_clip': 10.0, " \
      "'window_after_trained_shots': 5}\n"
OLD = "chain_type = 'flow'\nflow_parameters = {'placement': 'none', 'lambda_clip': 10.0, " \
      "'window_after_trained_shots': 5}\n"


def write_params(folder, extra, name='params.py'):
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    for kind, xs in (('A', [-1.0, -1.1, -0.9]), ('B', [1.0, 1.1, 0.9])):
        positions = np.zeros((len(xs), 1, 3), dtype=np.float32)
        positions[:, 0, 0] = xs
        write_trajectory(folder, stem=f'seeds_{kind}', positions=positions)
    fname = os.path.join(folder, name)
    with open(fname, 'w') as file:
        file.write(PARAMS_HEAD + '\n' + textwrap.dedent(extra))
    return fname


def deprecations(record):
    return [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]


def test_the_params_tables_agree():
    """Params keeps its own copy of the old field names (`import aimmd` imports no C-GAIMMD code)."""
    assert _LEGACY_FIELDS == set(_legacy.PARAMS_FIELDS)
    fields = aimmd.Params.__dataclass_fields__
    for old, new in _legacy.PARAMS_FIELDS.items():
        assert new in fields and old not in fields


# ---------------------------------------------------------------------------
# params fields and chain_type
# ---------------------------------------------------------------------------

HOOK_VALUES = {
    'flow_validity_function': 'validity',
    'flow_prepare_function': 'prepare',
    'flow_minimize_function': 'minimize',
    'flow_lambda_sampler': 'sampler',
}


@pytest.mark.parametrize('old', sorted(_legacy.PARAMS_FIELDS))
def test_old_field_assigned(tmp_path, old, recwarn):
    params = aimmd.Params.load(write_params(tmp_path, NEW), save=False)
    new = _legacy.PARAMS_FIELDS[old]
    if old == 'flow_parameters':
        value = {'n_targets': 8, 'lambda_clip': 10.0, 'placement': 'none'}
    else:
        def value(*args):
            return HOOK_VALUES[old]
        if old == 'flow_prepare_function':
            params.cgaimmd_parameters = {'lambda_clip': 10.0, 'window_after_trained_shots': 5}
    with pytest.warns(DeprecationWarning, match=f"'{old}' is deprecated; use '{new}'") as record:
        setattr(params, old, value)
    assert len(deprecations(record)) == 1
    assert getattr(params, new) == value if old == 'flow_parameters' else \
        getattr(params, new)() == HOOK_VALUES[old]
    assert old not in params.__dict__
    # once per process: the second assignment does not warn again
    recwarn.clear()
    setattr(params, old, value)
    assert deprecations(recwarn) == []


@pytest.mark.parametrize('old', sorted(_legacy.PARAMS_FIELDS))
def test_old_field_in_a_params_file(tmp_path, old, recwarn):
    new = _legacy.PARAMS_FIELDS[old]
    if old == 'flow_parameters':
        extra = OLD.replace("chain_type = 'flow'", "chain_type = 'cgaimmd'")
    else:
        extra = NEW + f"def {old}(*args):\n    return {HOOK_VALUES[old]!r}\n"
        if old == 'flow_prepare_function':
            extra = extra.replace("'placement': 'none', ", '')
    params = aimmd.Params.load(write_params(tmp_path, extra), save=False)
    assert deprecations(recwarn) == [
        f"the params field '{old}' is deprecated; use '{new}' (the old names of C-GAIMMD are "
        f"read for one release)"]
    if old == 'flow_parameters':
        assert params.cgaimmd_parameters == {'placement': 'none', 'lambda_clip': 10.0,
                                             'window_after_trained_shots': 5}
    else:
        assert getattr(params, new)() == HOOK_VALUES[old]
        # written under the new name only, as an import of the old function
        text = str(params.__str__(params.parent))
        assert f'import {old} as {new}' in text
    assert all(not name.startswith('flow') for name in params.__dict__)


@pytest.mark.parametrize('old', ['flow_parameters', 'flow_lambda_sampler'])
def test_old_field_as_keyword(tmp_path, old, recwarn):
    fname = write_params(tmp_path, NEW)
    if old == 'flow_parameters':
        value = {'n_targets': 4, 'lambda_clip': 10.0, 'placement': 'none'}
    else:
        def value(n, context):
            return np.zeros(n)
    params = aimmd.Params.load(fname, save=False, **{old: value})
    assert len(deprecations(recwarn)) == 1
    assert getattr(params, _legacy.PARAMS_FIELDS[old]) == value
    with pytest.raises(TypeError, match="both 'flow_lambda_sampler' and its new name"):
        aimmd.Params.load(fname, save=False, flow_lambda_sampler=None,
                          cgaimmd_lambda_sampler=None)


@pytest.mark.parametrize('value', ['flow', 'FLOW', 'fl', 'Flow shooting'])
def test_chain_type_flow(tmp_path, value, recwarn):
    params = aimmd.Params.load(write_params(tmp_path, NEW.replace("chain_type = 'cgaimmd'",
                                                                  "chain_type = 'rfps'")),
                               save=False)
    with pytest.warns(DeprecationWarning, match="chain_type .* is deprecated; use chain_type "
                                                "'cgaimmd'") as record:
        params.chain_type = value
    assert params.chain_type == 'cgaimmd'
    assert len(deprecations(record)) == 1


def test_an_old_params_file_loads_and_saves_new_names(tmp_path, recwarn, monkeypatch):
    params = aimmd.Params.load(write_params(tmp_path, OLD + textwrap.dedent('''
        def flow_validity_function(positions, context):
            return None
        flow_lambda_sampler = None
        ''')), save=False)
    found = deprecations(recwarn)
    assert len(found) == 4                     # chain_type, flow_parameters and the two hooks
    assert sum("chain_type 'flow'" in message for message in found) == 1
    assert params.chain_type == 'cgaimmd'
    monkeypatch.chdir(params.parent)                  # as the Launcher saves, from the run's folder
    saved = params.save(os.path.join(params.parent, 'saved.py'))
    text = open(saved).read()
    assert "chain_type = 'cgaimmd'" in text
    assert 'import flow_validity_function as cgaimmd_validity_function' in text
    for line in text.splitlines():
        assert not line.startswith(('flow_', "chain_type = 'flow'")), line
    _legacy._reset()
    recwarn.clear()
    reloaded = aimmd.Params.load(os.path.join(params.parent, 'saved.py'), save=False)
    assert deprecations(recwarn) == []
    assert reloaded.cgaimmd_parameters == params.cgaimmd_parameters


# ---------------------------------------------------------------------------
# the ODE settings of the old flow_parameters
# ---------------------------------------------------------------------------

def test_ode_settings_map_to_the_generator(tmp_path, recwarn):
    params = aimmd.Params.load(write_params(tmp_path, NEW), save=False)
    with pytest.warns(DeprecationWarning, match=r"cgaimmd_generator = \{'method': 'ode', "
                                                r"'sampler': 'rk4', 'n_steps': 8, 'chunk': 32\}"
                      ) as record:
        params.flow_parameters = {'ode_sampler': 'rk4', 'ode_steps': 8, 'generate_chunk': 32,
                                  'n_targets': 4, 'lambda_clip': 10.0, 'placement': 'none'}
    assert len(deprecations(record)) == 1
    assert params.cgaimmd_generator == {'method': 'ode', 'sampler': 'rk4', 'n_steps': 8,
                                        'chunk': 32}
    assert params.cgaimmd_parameters == {'n_targets': 4, 'lambda_clip': 10.0,
                                         'placement': 'none'}
    from aimmd.cgaimmd.generation import make_generator
    from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
    assert make_generator(params.cgaimmd_generator, params.network) == ODEGenerator(
        sampler='rk4', n_steps=8, chunk=32)


def test_generate_chunk_alone_is_a_setting_of_the_default_method(tmp_path, recwarn):
    """The archived BioEmu2 params set only generate_chunk (their network had its own generate)."""
    params = aimmd.Params.load(write_params(tmp_path, OLD.replace(
        "flow_parameters = {", "flow_parameters = {'generate_chunk': 64, ")), save=False)
    assert params.cgaimmd_generator == {'chunk': 64}
    assert 'generate_chunk' not in params.cgaimmd_parameters
    from aimmd.cgaimmd.generation import make_generator
    assert make_generator(params.cgaimmd_generator, params.network).settings()['chunk'] == 64
    assert len(deprecations(recwarn)) == 2                          # chain_type, flow_parameters


def test_old_and_new_in_one_file_is_an_error(tmp_path):
    with pytest.raises(TypeError, match="sets both 'flow_parameters' and its new name "
                                        "'cgaimmd_generator'"), \
            pytest.warns(DeprecationWarning):
        aimmd.Params.load(write_params(tmp_path, NEW + textwrap.dedent('''
            flow_parameters = {'ode_steps': 4}
            cgaimmd_generator = 'ode'
            ''')), save=False)


# ---------------------------------------------------------------------------
# compatibility modules
# ---------------------------------------------------------------------------

def test_old_module_names_resolve_with_one_warning(recwarn):
    from aimmd.cgaimmd.types import CandidateContext, ValidityResult
    from aimmd.cgaimmd.values import ProtocolValuesFunction
    with pytest.warns(DeprecationWarning, match='aimmd.flow.types.FlowContext is deprecated; '
                                                'use aimmd.cgaimmd.types.CandidateContext'
                      ) as record:
        from aimmd.flow.types import FlowContext
    assert FlowContext is CandidateContext
    assert len(deprecations(record)) == 1
    from aimmd.flow.types import ValidityResult as OldValidityResult
    from aimmd.flow.values import FlowValuesFunction
    from aimmd.flow import types as old_types
    assert OldValidityResult is ValidityResult and FlowValuesFunction is ProtocolValuesFunction
    assert old_types.FlowContext is CandidateContext                  # warned before: no repeat
    assert len(deprecations(recwarn)) == 2                         # the two other old names
    import aimmd.flow.protocol
    from aimmd.cgaimmd.protocol import is_cgaimmd_network
    assert aimmd.flow.protocol.is_flow_network is is_cgaimmd_network
    from aimmd.flow.sampler import ODEGenerator, to_flow
    from aimmd.cgaimmd.flow_matching import sampler
    assert ODEGenerator is sampler.ODEGenerator and to_flow is sampler.to_flow


def test_every_old_name_resolves_or_says_why():
    """A name of the tables resolves to its new object, or names the module that is missing."""
    for old_module, names in _legacy.NAMES.items():
        module = importlib.import_module(old_module)
        for name, (new_module, new_name) in names.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    value = getattr(module, name)
            except ImportError as exc:
                assert new_module in str(exc) and 'not available' in str(exc)
                assert importlib.util.find_spec(new_module.rsplit('.', 1)[0]) is None or \
                    importlib.util.find_spec(new_module) is None
            else:
                assert value is getattr(importlib.import_module(new_module), new_name)


def test_names_whose_new_module_is_missing():
    """Modules of later streams: the error names the new module (the shim is robust to that)."""
    if importlib.util.find_spec('aimmd.cgaimmd.dataset') is None:
        with pytest.raises(ImportError, match='aimmd.network.flow_fit._resolve_exclusions is '
                                              'now aimmd.cgaimmd.dataset.resolve_exclusions'):
            from aimmd.network.flow_fit import _resolve_exclusions             # noqa: F401
    if importlib.util.find_spec('aimmd.cgaimmd.preparation') is None:
        with pytest.raises(ImportError, match='aimmd.cgaimmd.preparation.descriptors.'
                                              'GeneratorAtoms'):
            from aimmd.flow.descriptors import FlowAtoms                       # noqa: F401
        import aimmd.flow.validate as validate
        with pytest.raises(ImportError, match='preparation.minimize.polish'):
            validate._polish
    # unknown private and dunder names are plain AttributeErrors (introspection keeps working)
    import aimmd.network.flow_fit as flow_fit
    assert not hasattr(flow_fit, '_no_such_private_name')
    assert not hasattr(flow_fit, '__wrapped__')


def test_dropped_names_say_why():
    import aimmd.flow.harmonic as harmonic
    with pytest.raises(ImportError, match='harmonic prior is not part of C-GAIMMD'):
        harmonic.HarmonicPrior
    with pytest.raises(ImportError, match='mixture prior is not part of C-GAIMMD'):
        from aimmd.flow.prior import MixturePrior                               # noqa: F401
    with pytest.raises(ImportError, match='generation goes through aimmd.cgaimmd.generation'):
        from aimmd.flow.sampler import sample                                   # noqa: F401
    with pytest.raises(ImportError, match='cfaimmd'):
        from aimmd.network.joint_painn import convert_cfaimmd_state_dict        # noqa: F401


def test_every_old_submodule_is_importable():
    for old in _legacy.MODULES:
        if old.startswith('aimmd.flow.'):
            module = importlib.import_module(old)
            assert module.__name__ == old
            assert getattr(importlib.import_module('aimmd.flow'), old.rsplit('.', 1)[1]) is module


# ---------------------------------------------------------------------------
# values read from files and hook objects
# ---------------------------------------------------------------------------

def test_translate(recwarn):
    with pytest.warns(DeprecationWarning, match="restrain 'flow_heavy' is deprecated; use "
                                                "restrain 'generator_heavy'") as record:
        assert _legacy.translate(_legacy.RESTRAINT_MODES, 'flow_heavy',
                                 'restrain') == 'generator_heavy'
    assert len(deprecations(record)) == 1
    recwarn.clear()
    assert _legacy.translate(_legacy.RESTRAINT_MODES, 'flow_heavy', 'restrain') == \
        'generator_heavy'
    assert deprecations(recwarn) == []
    assert _legacy.translate(_legacy.RESTRAINT_MODES, 'none', 'restrain') == 'none'
    assert _legacy.translate(_legacy.RESTRAINT_MODES, ['unhashable'], 'restrain') == [
        'unhashable']
    assert _legacy.translate(_legacy.GENERATOR_LABELS, 'network.generate',
                             'generator method') == 'bioemu2_edm'
    assert _legacy.translate(_legacy.GENERATOR_LABELS, 'aimmd sampler',
                             'generator method') == 'ode'
    assert _legacy.SIDECAR_SUFFIXES == {'.flow.json': '.cgaimmd.json'}
    assert _legacy.SIDECAR_SCHEMAS == {'aimmd.flow.shot/1': 'aimmd.cgaimmd.shot/1'}
    assert _legacy.IN_FLIGHT_MARKERS['.flow_shooting_point.json'] == \
        '.cgaimmd_shooting_point.json'
    assert _legacy.CHAIN_TYPES == {'flow': 'cgaimmd'}
