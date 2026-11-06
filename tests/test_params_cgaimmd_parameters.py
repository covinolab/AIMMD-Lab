"""The ``cgaimmd_parameters`` field, its defaults and ``Params.cgaimmd_settings``.

``cgaimmd_parameters`` holds every scalar setting of the C-GAIMMD shooting
loop and of the C-GAIMMD devices as one validated dict. Only the keys the user
gave are stored (so ``Params.save`` writes no defaults and a later change of a
default reaches restarted runs); ``Params.cgaimmd_settings`` fills in the
defaults. How candidates are generated is not among them: that is
``cgaimmd_generator`` (``tests/test_cgaimmd_generation.py``).
"""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.params.utils import CGAIMMD_PARAMETER_DEFAULTS, canonical_cgaimmd_parameters


# the documented defaults, key by key
SPEC_DEFAULTS = {
    'lambda_law': 'uniform',
    'lambda_window': 'bins',
    'lambda_clip': None,
    'acceptance': 'transition_region',
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

PARAMS_SOURCE = '''
import numpy as np
import torch
from math import inf, nan

engine = 'toy'
cgaimmd_parameters = {cgaimmd_parameters}

def states_function(trajectory):
    return np.array(['R'] * len(trajectory), dtype='<U1')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''


def _write_params(tmp_path, cgaimmd_parameters, name='params.py'):
    fname = Path(tmp_path) / name
    fname.write_text(PARAMS_SOURCE.format(cgaimmd_parameters=cgaimmd_parameters))
    return str(fname)


def test_cgaimmd_parameters_defaults(tmp_path):
    """The defaults are the spec's table; unset means all defaults."""
    assert dict(CGAIMMD_PARAMETER_DEFAULTS) == SPEC_DEFAULTS
    assert list(CGAIMMD_PARAMETER_DEFAULTS) == list(SPEC_DEFAULTS)
    # a plain dict: it passes its own validation unchanged, deep-copies, pickles and dumps
    assert type(CGAIMMD_PARAMETER_DEFAULTS) is dict
    assert canonical_cgaimmd_parameters(CGAIMMD_PARAMETER_DEFAULTS) == SPEC_DEFAULTS
    assert copy.deepcopy(CGAIMMD_PARAMETER_DEFAULTS) == SPEC_DEFAULTS
    assert pickle.loads(pickle.dumps(CGAIMMD_PARAMETER_DEFAULTS)) == SPEC_DEFAULTS
    assert json.loads(json.dumps(CGAIMMD_PARAMETER_DEFAULTS, default=list)) == {
        k: list(v) if isinstance(v, tuple) else v for k, v in SPEC_DEFAULTS.items()}

    field = aimmd.Params.__dataclass_fields__['cgaimmd_parameters']
    assert field.default is None
    assert field.metadata['description'].startswith(
        'Settings of the C-GAIMMD shooting loop (chain_type=\'cgaimmd\') and of the')

    params = aimmd.Params.load(_write_params(tmp_path, None), save=False)
    assert params.cgaimmd_parameters is None
    settings = params.cgaimmd_settings
    assert settings == SPEC_DEFAULTS
    # a fresh dict every time: mutating it changes neither the params nor the defaults
    settings['n_targets'] = 3
    assert params.cgaimmd_settings['n_targets'] == 64
    assert CGAIMMD_PARAMETER_DEFAULTS == SPEC_DEFAULTS

    # the placeholder has it too (C-GAIMMD code reads it on test namespaces)
    assert aimmd.Params.placeholder.cgaimmd_settings == SPEC_DEFAULTS


@pytest.mark.parametrize('bad', [
    {'bogus': 1},
    {'lambda_law': 'gaussian'},
    {'lambda_law': 1},
    {'lambda_window': (1.0, -1.0)},
    {'lambda_window': (0.0, 0.0)},
    {'lambda_window': (-1.0, float('inf'))},
    {'lambda_window': (-1.0, 0.0, 1.0)},
    {'lambda_window': 'bin'},
    {'lambda_window': 3.0},
    {'lambda_clip': 0.0},
    {'lambda_clip': -1.0},
    {'lambda_clip': float('nan')},
    {'lambda_clip': True},
    {'acceptance': 'accept'},
    {'lambda_tolerance': 0},
    {'lambda_tolerance': '1'},
    {'window_after_trained_shots': -1},
    {'window_after_trained_shots': 5.0},
    {'window_after_trained_shots': True},
    {'raw_lambda_tolerance': 0.0},
    {'n_targets': 0},
    {'candidates_per_target': 0},
    {'generate_chunk': 64},           # generation settings belong to cgaimmd_generator
    {'ode_sampler': 'rk4'},
    {'ode_steps': 8},
    {'placement': 'center'},
    {'final_check_round': -1},
    {'final_check_round': 2.5},
    {'final_check_round': 'always'},
    {'generation_device': 'gpu'},
    {'values_device': 'cuda:'},
    {'training_device': 'cuda:-1'},
    {'training_device': 0},
    {'max_refills': 0},
    {'wait_seconds': -1.0},
    {'generation_oom_halvings': -1},
    {'generation_seed': -3},
    {'generation_seed': 1.5},
    {'keep_scratch': 1},
])
def test_cgaimmd_parameters_rejects(bad, tmp_path):
    """Unknown keys, wrong types and out-of-range values raise TypeError naming the key."""
    (key,) = bad
    with pytest.raises(TypeError, match=repr(key)):
        canonical_cgaimmd_parameters(bad)

    params = aimmd.Params.load(_write_params(tmp_path, None), save=False)
    with pytest.raises(TypeError, match=repr(key)):
        params.cgaimmd_parameters = bad
    assert params.cgaimmd_parameters is None          # the old value is kept

    with pytest.raises(TypeError):
        aimmd.Params.load(_write_params(tmp_path, repr(bad), 'bad.py'), save=False)


def test_acceptance_rules():
    """'transition_region' (the default: no committor filter), 'target' and 'window' are accepted."""
    assert CGAIMMD_PARAMETER_DEFAULTS['acceptance'] == 'transition_region'
    for rule in ('transition_region', 'target', 'window'):
        assert canonical_cgaimmd_parameters({'acceptance': rule}) == {'acceptance': rule}
    with pytest.raises(TypeError, match="'acceptance'"):
        canonical_cgaimmd_parameters({'acceptance': 'transition region'})


def test_raw_lambda_prefilter_needs_a_committor_rule(tmp_path):
    """raw_lambda_tolerance filters by the committor: an error unless acceptance is 'target'/'window'."""
    for value in ({'raw_lambda_tolerance': 0.5},
                  {'acceptance': 'transition_region', 'raw_lambda_tolerance': 0.5},
                  {'raw_lambda_tolerance': 0.5, 'acceptance': 'transition_region'}):
        with pytest.raises(TypeError, match="'raw_lambda_tolerance'.*'transition_region'"):
            canonical_cgaimmd_parameters(value)
    for rule in ('target', 'window'):
        value = {'acceptance': rule, 'raw_lambda_tolerance': 0.5}
        assert canonical_cgaimmd_parameters(value) == value
    # None is the default and means no prefilter: allowed with every rule
    assert canonical_cgaimmd_parameters({'raw_lambda_tolerance': None}) == {
        'raw_lambda_tolerance': None}
    with pytest.raises(TypeError, match="'raw_lambda_tolerance'"):
        aimmd.Params.load(_write_params(tmp_path, "{'raw_lambda_tolerance': 0.5}"), save=False)


def test_generation_keys_point_to_the_generator_field():
    for key in ('generate_chunk', 'ode_sampler', 'ode_steps'):
        with pytest.raises(TypeError, match=f"unknown cgaimmd_parameters key '{key}'.*"
                                            f"set by cgaimmd_generator"):
            canonical_cgaimmd_parameters({key: 8})
    with pytest.raises(TypeError) as info:
        canonical_cgaimmd_parameters({'bogus': 1})
    assert 'cgaimmd_generator' not in str(info.value)


def test_cgaimmd_parameters_must_be_a_dict():
    for bad in ([('n_targets', 8)], 'n_targets=8', 8):
        with pytest.raises(TypeError, match='cgaimmd_parameters'):
            canonical_cgaimmd_parameters(bad)
    assert canonical_cgaimmd_parameters(None) is None
    assert canonical_cgaimmd_parameters({}) == {}


def test_lists_become_tuples():
    """Normalization: lists -> tuples, numbers -> plain Python int/float."""
    value = canonical_cgaimmd_parameters({
        'lambda_window': [-3, np.float32(3.5)],
        'lambda_tolerance': 2,
        'wait_seconds': np.int64(5),
        'n_targets': np.int64(8),
        'lambda_clip': np.float64(10.0),
        'final_check_round': np.int32(3),
        'generation_seed': 7,
    })
    assert value['lambda_window'] == (-3.0, 3.5)
    assert isinstance(value['lambda_window'], tuple)
    assert all(type(v) is float for v in value['lambda_window'])
    assert type(value['lambda_tolerance']) is float and value['lambda_tolerance'] == 2.0
    assert type(value['wait_seconds']) is float and value['wait_seconds'] == 5.0
    assert type(value['n_targets']) is int and value['n_targets'] == 8
    assert type(value['lambda_clip']) is float
    assert type(value['final_check_round']) is int
    assert value['generation_seed'] == 7
    # 'bins' and the other string choices pass unchanged
    assert canonical_cgaimmd_parameters({'lambda_window': 'bins'}) == {'lambda_window': 'bins'}
    for spec in ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:12'):
        assert canonical_cgaimmd_parameters({'values_device': spec}) == {'values_device': spec}
    for choice in (None, 'auto', 0, 2):
        assert canonical_cgaimmd_parameters({'final_check_round': choice}) == {
            'final_check_round': choice}


def test_stores_user_keys_only(tmp_path):
    """Only the given keys are stored and saved; cgaimmd_settings fills in the rest."""
    params = aimmd.Params.load(_write_params(
        tmp_path, "{'n_targets': 8, 'lambda_window': [-2, 2]}"), save=False)
    assert params.cgaimmd_parameters == {'n_targets': 8, 'lambda_window': (-2.0, 2.0)}
    settings = params.cgaimmd_settings
    assert settings['n_targets'] == 8
    assert settings['lambda_window'] == (-2.0, 2.0)
    assert {k: v for k, v in settings.items()
            if k not in ('n_targets', 'lambda_window')} == {
        k: v for k, v in SPEC_DEFAULTS.items() if k not in ('n_targets', 'lambda_window')}

    saved = Path(params.save(tmp_path / 'saved.py'))
    lines = [line for line in saved.read_text().splitlines()
             if line.startswith('cgaimmd_parameters')]
    assert lines == ["cgaimmd_parameters = {'n_targets': 8, 'lambda_window': (-2.0, 2.0)}"]

    reloaded = aimmd.Params.load(str(saved), save=False)
    assert reloaded.cgaimmd_parameters == params.cgaimmd_parameters
    assert reloaded.cgaimmd_settings == params.cgaimmd_settings

    # assignment (any chain type) validates and stores the given keys only
    params.cgaimmd_parameters = {'keep_scratch': True}
    assert params.cgaimmd_parameters == {'keep_scratch': True}
    assert params.cgaimmd_settings['keep_scratch'] is True
    assert params.cgaimmd_settings['n_targets'] == 64
    params.cgaimmd_parameters = None
    assert params.cgaimmd_parameters is None
    saved = Path(params.save(tmp_path / 'saved_none.py'))
    assert 'cgaimmd_parameters = None' in saved.read_text().splitlines()
