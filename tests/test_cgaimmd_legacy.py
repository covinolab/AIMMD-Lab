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
                                        r"'cgaimmd_generator'; move its keys 'ode_steps' into "
                                        r"'cgaimmd_generator' \(as 'n_steps'\), then delete "
                                        "'flow_parameters'"), \
            pytest.warns(DeprecationWarning):
        aimmd.Params.load(write_params(tmp_path, NEW + textwrap.dedent('''
            flow_parameters = {'ode_steps': 4}
            cgaimmd_generator = 'ode'
            ''')), save=False)


def test_a_half_migrated_file_is_told_where_each_key_goes(tmp_path):
    """Deleting flow_parameters alone would drop its loop settings: the message says where they go."""
    old = "{'window_after_trained_shots': 5, 'ode_steps': 4, 'lambda_tolerance': 0.5}"
    advice = (r"move its keys 'ode_steps' into 'cgaimmd_generator' \(as 'n_steps'\) and its "
              r"keys 'window_after_trained_shots', 'lambda_tolerance' into "
              r"'cgaimmd_parameters', then delete 'flow_parameters'")
    fname = write_params(tmp_path, NEW + f"flow_parameters = {old}\ncgaimmd_generator = 'ode'\n")
    with pytest.raises(TypeError, match="params.py sets both 'flow_parameters' and its new "
                                        "name 'cgaimmd_generator'; " + advice), \
            pytest.warns(DeprecationWarning):
        aimmd.Params.load(fname, save=False)
    fname = write_params(tmp_path / 'keywords', NEW)
    with pytest.raises(TypeError, match="both 'flow_parameters' and its new name "
                                        "'cgaimmd_generator' were given; " + advice):
        aimmd.Params.load(fname, save=False, cgaimmd_generator='ode',
                          flow_parameters=eval(old))


def test_an_old_field_is_assigned_whole_or_not_at_all(tmp_path, recwarn):
    params = aimmd.Params.load(write_params(tmp_path, NEW), save=False)
    before = dict(params.cgaimmd_parameters)
    with pytest.raises(TypeError, match="can't update 'flow_parameters'.*bogus_key"):
        params.flow_parameters = {'ode_steps': 4, 'bogus_key': 1}
    assert params.cgaimmd_generator is None                 # not {'method': 'ode', 'n_steps': 4}
    assert params.cgaimmd_parameters == before
    assert '_read_old_names' not in params.__dict__


# ---------------------------------------------------------------------------
# networks of the old code: the generation method the old worker chose
# ---------------------------------------------------------------------------

ARCHIVED_NETWORK = '''
import torch

class ArchivedNetwork(torch.nn.Module):
    """The members of the archived BioEmu2 network (bioemu2_committor.aimmd_flow):
    a committor, a stand-in velocity field and prior, its own generate, and no
    default_generator."""
    n_atoms = 1
    lambda_clip = 10.0
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(torch.as_tensor(x, dtype=torch.float32).reshape(len(x), -1)[:, :3])
    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        return torch.randn((n, 1, 3), generator=generator)
    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        return torch.zeros_like(x_t)
    def generate(self, lam, n=None, *, sampler='midpoint', n_steps=16, chunk=128,
                 generator=None):
        return torch.zeros((len(lam), 1, 3))

class Network(ArchivedNetwork):
    pass

network = Network()
'''

WITHOUT_GENERATE = ARCHIVED_NETWORK.replace('''class Network(ArchivedNetwork):
    pass''', '''class Network(ArchivedNetwork):
    generate = None                                     # an old PaiNN-like network''')

ARCHIVED_BIOEMU2_PARAMS = OLD.replace("flow_parameters = {", "flow_parameters = {'generate_chunk': 64, ")
"""What the five archived BioEmu2 params set: chain_type 'flow' and generate_chunk only."""


def test_the_archived_bioemu2_params_load(tmp_path, recwarn, capsys, monkeypatch):
    """Their network has its own generate and no default_generator: the old worker used generate."""
    params = aimmd.Params.load(write_params(tmp_path, ARCHIVED_BIOEMU2_PARAMS + ARCHIVED_NETWORK),
                               save=False)
    assert params.chain_type == 'cgaimmd'
    assert params.cgaimmd_generator == {'method': 'bioemu2_edm', 'chunk': 64}
    found = deprecations(recwarn)
    assert len(found) == 3                       # chain_type, flow_parameters, the method
    assert ("letting the old worker pick the generation method of Network (it used its own "
            "generate()) is deprecated; use cgaimmd_generator = {'method': 'bioemu2_edm', "
            "'chunk': 64}") in found[-1]
    # the method cannot be used here (no adaptor, or not a BioEmu2Network): a warning, as
    # the params of an old run load for analysis; a worker's start-up check refuses them
    out = capsys.readouterr().out
    assert 'These params were read through old names' in out
    if importlib.util.find_spec('aimmd.cgaimmd.bioemu2') is None:
        assert "generation method 'bioemu2_edm'" in out and 'aimmd.cgaimmd.bioemu2' in out
    # saved under the new names, with the method the old worker used
    monkeypatch.chdir(params.parent)
    text = open(params.save(os.path.join(params.parent, 'saved.py'))).read()
    assert "cgaimmd_generator = {'method': 'bioemu2_edm', 'chunk': 64}" in text
    assert "chain_type = 'cgaimmd'" in text
    # the same params under the new names are refused: a network needs a default_generator()
    # or the field, and the message recommends no method
    with pytest.raises(TypeError, match=r'Network has no default_generator\(\), so '
                                        r'params.cgaimmd_generator must name') as info:
        aimmd.Params.load(write_params(tmp_path / 'new', NEW + ARCHIVED_NETWORK), save=False)
    assert "'ode'" not in str(info.value).split('known methods')[0]


def test_an_old_network_without_generate_gets_the_ode(tmp_path, recwarn, capsys):
    params = aimmd.Params.load(write_params(tmp_path, ARCHIVED_BIOEMU2_PARAMS + WITHOUT_GENERATE),
                               save=False)
    assert params.cgaimmd_generator == {'method': 'ode', 'chunk': 64}
    assert any("(it used AIMMD's ODE sampler) is deprecated; use cgaimmd_generator = "
               "{'method': 'ode', 'chunk': 64}" in message for message in deprecations(recwarn))
    assert 'These params were read through old names' not in capsys.readouterr().out
    # chain_type 'flow' and no generator settings at all
    params = aimmd.Params.load(write_params(tmp_path / 'bare', OLD + WITHOUT_GENERATE), save=False)
    assert params.cgaimmd_generator == {'method': 'ode'}


def test_an_old_network_with_generate_never_gets_the_ode(tmp_path, recwarn):
    """The old worker called a network's own generate even when ODE settings were given."""
    params = aimmd.Params.load(write_params(tmp_path, OLD + ARCHIVED_NETWORK), save=False)
    assert params.cgaimmd_generator == {'method': 'bioemu2_edm'}
    params = aimmd.Params.load(write_params(tmp_path / 'ode_keys', OLD.replace(
        "flow_parameters = {", "flow_parameters = {'ode_sampler': 'rk4', 'ode_steps': 8, "
                               "'generate_chunk': 32, ") + ARCHIVED_NETWORK), save=False)
    assert params.cgaimmd_generator == {'method': 'bioemu2_edm', 'chunk': 32}
    assert any("(it used its own generate(), which ignored 'sampler', 'n_steps')" in message
               for message in deprecations(recwarn))
    # assigned later, the old field is completed the same way
    params.flow_parameters = {'generate_chunk': 16, 'placement': 'none', 'lambda_clip': 10.0,
                              'window_after_trained_shots': 5}
    assert params.cgaimmd_generator == {'method': 'bioemu2_edm', 'chunk': 16}


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
                assert f'would be {new_module}.{new_name}' in str(exc)
                assert 'not available' in str(exc)
                assert importlib.util.find_spec(new_module.rsplit('.', 1)[0]) is None or \
                    importlib.util.find_spec(new_module) is None
            else:
                assert value is getattr(importlib.import_module(new_module), new_name)


def test_names_whose_new_module_is_missing():
    """Modules of later streams: the error names the new module (the shim is robust to that)."""
    if importlib.util.find_spec('aimmd.cgaimmd.dataset') is None:
        with pytest.raises(ImportError, match='aimmd.network.flow_fit._resolve_exclusions would '
                                              'be aimmd.cgaimmd.dataset.resolve_exclusions'):
            from aimmd.network.flow_fit import _resolve_exclusions             # noqa: F401
    if importlib.util.find_spec('aimmd.cgaimmd.preparation') is None:
        with pytest.raises(ImportError, match='aimmd.cgaimmd.preparation.descriptors.'
                                              'GeneratorAtoms'):
            from aimmd.flow.descriptors import FlowAtoms                       # noqa: F401
        import aimmd.flow.validate as validate
        with pytest.raises(ImportError, match='preparation.minimize.polish'):
            validate._polish
    # names the tables do not list are plain AttributeErrors (introspection keeps working)
    import aimmd.network.flow_fit as flow_fit
    assert not hasattr(flow_fit, '_no_such_private_name')
    assert not hasattr(flow_fit, '__wrapped__')


def test_introspection_of_every_compatibility_module(recwarn):
    """Code that walks sys.modules and asks every aimmd module for a name must not break.

    (tests/test_mda_reader_cache.py does ``hasattr(module, 'MDA_CACHE')`` for
    every ``aimmd.*`` module.)
    """
    for old in _legacy.MODULES:
        module = importlib.import_module(old)
        for name in ('NO_SUCH_PUBLIC_NAME', '_no_such_private_name', '__wrapped__'):
            assert not hasattr(module, name), (old, name)
    assert deprecations(recwarn) == []


def test_dropped_names_say_why():
    import aimmd.flow.harmonic as harmonic
    with pytest.raises(ImportError, match='harmonic prior is not part of C-GAIMMD'):
        harmonic.HarmonicPrior
    with pytest.raises(AttributeError, match='harmonic prior is not part of C-GAIMMD'):
        harmonic.anything_else
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
