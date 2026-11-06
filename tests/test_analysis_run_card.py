"""Every setting of a run in one record (``aimmd.analysis.run_card``)."""

import dataclasses
import json
import math
import os

import numpy as np
import pytest
import torch

from aimmd.analysis.run_card import RUN_CARD_SCHEMA, describe_value, run_card, run_card_rows


BIAS = '''
record_bias = True
bias_source = 'reader'
bias_reactive_threshold = 0.25


def bias_function(trajectory):
    return np.zeros(len(trajectory))
'''


def load(folder, name='params.py'):
    import aimmd
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        return aimmd.Params.load(name, save=False)
    finally:
        os.chdir(cwd)


@pytest.fixture
def toy_campaign(tmp_path):
    """The toy C-GAIMMD campaign folder with a network and its fit metadata in ``run/``, as the
    trainer writes them after a trained round."""
    from tests._cgaimmd_helpers import write_fake_network_file
    from tests.test_toy_1d_cgaimmd import make_campaign
    folder = make_campaign(tmp_path / 'campaign')
    params = load(folder)
    sha = write_fake_network_file(folder / 'run', params.network, round=3, n_trained_shots=12,
                                  metadata={'fit': 'flow_matching_fit', 'seed': 7,
                                            'keywords': {'steps_per_round': 5}})
    return {'folder': folder, 'run': str(folder / 'run'), 'params': params, 'sha': sha}


def test_every_params_field_is_in_the_card(toy_campaign):
    import aimmd
    card = run_card(toy_campaign['params'], toy_campaign['run'])
    names = {field.name for field in dataclasses.fields(aimmd.Params)}
    assert set(card['params']) == names
    assert card['schema'] == RUN_CARD_SCHEMA
    assert card['chain_type'] == 'cgaimmd' and card['states'] == 'ARB'
    assert card['params_file'].endswith('params.py') and len(card['params_sha256']) == 64
    assert {'chain_type', 'cgaimmd_generator', 'cgaimmd_parameters', 'initial_network',
            'masses_from_topology'} <= set(card['non_default'])
    assert 'nbins' not in card['non_default'] and 'reweight_parameters' not in card['non_default']
    assert card['params']['nbins'] == 10
    assert card['params']['reweight_parameters'] == {'free_threshold': 20}
    assert card['params']['cutoff_max'] == 20.0
    assert card['params']['states_function'].endswith('states_function')
    json.dumps(card)                                         # JSON-serializable throughout


def test_the_card_holds_the_cgaimmd_generator_and_settings(toy_campaign):
    from aimmd.params.utils import CGAIMMD_PARAMETER_DEFAULTS
    card = run_card(toy_campaign['params'], toy_campaign['run'])
    section = card['cgaimmd']
    assert set(section['settings']) == set(CGAIMMD_PARAMETER_DEFAULTS)
    assert section['settings']['lambda_tolerance'] == 3.0
    assert section['settings']['window_after_trained_shots'] == 5
    generator = section['generator']
    assert generator['method'] == 'ode' and generator['error'] is None
    assert generator['settings'] == {**generator['settings'], 'sampler': 'midpoint',
                                     'n_steps': 8, 'chunk': 16}
    assert generator['settings']['guidance_scale'] == 1.0
    assert generator['spec']['method'] == 'ode'
    assert section['hooks']['cgaimmd_validity_function'].endswith('cgaimmd_validity_function')
    assert section['hooks']['cgaimmd_prepare_function'] is None


def test_the_card_holds_the_network_and_the_fit_of_the_last_round(toy_campaign):
    from tests._analysis_ring_run import snapshot
    before = snapshot(toy_campaign['folder'])
    card = run_card(toy_campaign['params'], toy_campaign['run'])
    assert snapshot(toy_campaign['folder']) == before          # reads only
    network = card['network']
    assert network['class'].endswith('Network')
    assert network['n_parameters'] == sum(p.numel() for p in
                                          toy_campaign['params'].network.parameters())
    assert 'Linear' in network['architecture'] or 'Sequential' in network['architecture']
    assert card['network_files']['network_sha256'] == toy_campaign['sha']
    last = card['fit']['last_round']
    assert (last['round'], last['n_trained_shots'], last['seed']) == (3, 12, 7)
    assert last['keywords'] == {'steps_per_round': 5}
    assert card['fit']['function'] is not None
    assert card['initial_network'].endswith('initial_network.h5')
    assert card['engine']['engine'] == 'toy' and 'toy_mdrun' in card['engine']
    assert card['aimmd']['version'] and card['software']['python']
    # without a run folder there is no last round
    alone = run_card(toy_campaign['params'])
    assert alone['fit']['last_round'] is None and alone['network_files'] is None


def test_the_card_holds_the_bias(toy_campaign):
    assert run_card(toy_campaign['params'])['bias'] is None
    with open(toy_campaign['folder'] / 'params.py', 'a') as handle:
        handle.write(BIAS)
    card = run_card(load(toy_campaign['folder']))
    assert card['bias']['record_bias'] is True
    assert card['bias']['bias_source'] == 'reader'
    assert card['bias']['bias_reactive_threshold'] == 0.25
    assert card['bias']['bias_function'].endswith('bias_function')
    assert card['params']['record_bias'] is True


def test_a_plain_run_has_no_cgaimmd_section(tmp_path):
    from tests._analysis_ring_run import RING_PARAMS, write_frames
    (tmp_path / 'params.py').write_text(RING_PARAMS)
    write_frames(tmp_path / 'initial.xtc', np.linspace(1, 6, 11), np.arange(11.0))
    card = run_card(load(tmp_path))
    assert card['cgaimmd'] is None and card['chain_type'] == 'rfps'
    assert card['params']['free_restart_source'] == 'equilibrium'
    assert 'free_restart_source' in card['non_default']


def test_rows_for_a_table(toy_campaign):
    card = run_card(toy_campaign['params'], toy_campaign['run'])
    rows = run_card_rows(card)
    assert all(len(row) == 3 and all(isinstance(x, str) for x in row) for row in rows)
    assert ('chain_type', '', 'cgaimmd') in rows
    assert ('cgaimmd', 'generator.settings.n_steps', '8') in rows
    assert ('fit', 'last_round.seed', '7') in rows
    params_rows = run_card_rows(card, sections=['params'])
    assert {row[0] for row in params_rows} == {'params'}
    assert ('params', 'cutoff_min', '0.5') in params_rows
    latex = run_card_rows(card, sections=['params', 'cgaimmd'], latex=True)
    assert ('params', 'cutoff\\_min', '0.5') in latex
    assert not any('_' in text.replace('\\_', '') for row in latex for text in row)
    assert len(run_card_rows(card, skip_none=True)) < len(rows)


def test_describe_value():
    assert describe_value(None) is None and describe_value(True) is True
    assert describe_value(3) == 3 and describe_value('a') == 'a'
    assert describe_value(math.inf) == 'inf' and describe_value(math.nan) == 'nan'
    assert describe_value(np.float32(0.5)) == 0.5 and describe_value(np.int64(2)) == 2
    assert describe_value(np.arange(3)) == [0, 1, 2]
    assert describe_value(np.zeros((100, 3))) == {'array': [100, 3], 'dtype': 'float64'}
    assert describe_value((1, [2.0, np.inf])) == [1, [2.0, 'inf']]
    assert describe_value({'a': (1,), 2: None}) == {'a': [1], '2': None}
    long = describe_value(list(range(100)))
    assert long == {'length': 100, 'first': [0, 1, 2], 'last': [97, 98, 99]}
    module = describe_value(torch.nn.Linear(2, 3))
    assert module == {'class': 'torch.nn.modules.linear.Linear', 'n_parameters': 9}
    assert describe_value(np.mean) == 'numpy.mean'
    assert describe_value(dict) == 'builtins.dict'

    class Thing:
        def __repr__(self):
            return 'thing'
    assert describe_value(Thing()) == {'class': f'{__name__}.test_describe_value.<locals>.Thing',
                                       'repr': 'thing'}
    json.dumps(describe_value({'x': [np.float64(1.0), torch.nn.ReLU(), Thing()]}))
