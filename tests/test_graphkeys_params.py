"""The ``descriptor_cache`` switch of Params (graph keys instead of descriptors).

'npy' (the default) must return exactly the historical compute tuples;
'graphkeys' swaps in the key function and the keyed values function, but
only together with a ``descriptors_function``. The field is validated and
round-trips through the saved params file. Runs without torch_geometric.
"""

from pathlib import Path as PosixPath
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.network import graph_keys as gk


PARAMS_SOURCE = '''
import numpy as np
import torch

engine = 'toy'
DESCRIPTOR_CACHE_LINE


def states_function(trajectory):
    return np.full(len(trajectory), 'R', dtype='<U1')


def descriptors_function(trajectory):
    return np.array([ts.positions.ravel().copy() for ts in trajectory])


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''


def _params_file(folder, line="descriptor_cache = 'graphkeys'"):
    path = PosixPath(folder) / 'params.py'
    path.write_text(PARAMS_SOURCE.replace('DESCRIPTOR_CACHE_LINE', line))
    return path


def _placeholder(**fields):
    params = aimmd.Params.placeholder
    params.__dict__.update(fields)
    return params


def descriptors_function(trajectory):
    return np.zeros((len(trajectory), 3), dtype=np.float32)


def descriptor_transform(x):
    return x


# ------------------------------------------------------------ properties --
def test_npy_mode_keeps_the_historical_tuples():
    params = _placeholder(descriptors_function=descriptors_function)
    assert params.descriptor_cache == 'npy'
    assert not params.graphkeys_mode
    assert params.graphkeys_function is None
    assert params.descriptors_source == 'descriptors'
    assert params.compute_descriptors_args == (descriptors_function,
                                               'descriptors')
    assert params.compute_values_args == (params.values_function, 'values',
                                          'descriptors')
    assert params.pipeline == (params.compute_descriptors_args,
                               params.compute_states_args,
                               params.compute_values_args)


def test_npy_mode_without_descriptors_uses_coordinates():
    params = _placeholder()
    assert params.descriptors_source == 'coordinates'
    assert params.compute_descriptors_args is None
    assert params.compute_values_args == (params.values_function, 'values',
                                          'coordinates')
    assert len(params.pipeline) == 2


def test_graphkeys_mode_swaps_in_the_key_functions():
    params = _placeholder(descriptor_cache='graphkeys',
                          descriptors_function=descriptors_function,
                          descriptor_transform=descriptor_transform)
    assert params.graphkeys_mode
    assert params.descriptors_source == 'graphkeys'
    keys_function = params.graphkeys_function
    assert isinstance(keys_function, gk.GraphKeysFunction)
    assert keys_function.descriptors_function is descriptors_function
    assert keys_function.descriptor_transform is descriptor_transform
    assert params.graphkeys_function is keys_function, 'built once'

    assert params.compute_descriptors_args == (keys_function, 'graphkeys')
    function, target, source = params.compute_values_args
    assert isinstance(function, gk.KeyedFunction)
    assert function.function is params.values_function
    assert function.keys_function is keys_function
    assert (target, source) == ('values', 'graphkeys')
    assert len(params.pipeline) == 3
    assert params.pipeline[0] == (keys_function, 'graphkeys')
    assert params.pipeline[2][2] == 'graphkeys'


def test_key_function_follows_the_functions_it_wraps():
    params = _placeholder(descriptor_cache='graphkeys',
                          descriptors_function=descriptors_function,
                          descriptor_transform=descriptor_transform)
    first = params.graphkeys_function

    def other_transform(x):
        return x

    params.__dict__['descriptor_transform'] = other_transform
    second = params.graphkeys_function
    assert second is not first
    assert second.descriptor_transform is other_transform


def test_graphkeys_without_descriptors_function_routes_coordinates():
    params = _placeholder(descriptor_cache='graphkeys')
    assert not params.graphkeys_mode
    assert params.graphkeys_function is None
    assert params.descriptors_source == 'coordinates'
    assert params.compute_descriptors_args is None
    assert params.compute_values_args[2] == 'coordinates'


def test_uses_graph_keys_reads_any_params_like_object():
    assert not gk.uses_graph_keys(SimpleNamespace())
    assert not gk.uses_graph_keys(
        SimpleNamespace(descriptors_function=descriptors_function))
    assert not gk.uses_graph_keys(SimpleNamespace(descriptor_cache='graphkeys'))
    assert gk.uses_graph_keys(SimpleNamespace(
        descriptor_cache='graphkeys', descriptors_function=descriptors_function))


# ------------------------------------------------- validation, round trip --
@pytest.mark.parametrize('value', ['keys', 'NPZ', '', 1, None])
def test_invalid_value_is_a_value_error_at_load(tmp_path, value):
    params_file = _params_file(tmp_path, f'descriptor_cache = {value!r}')
    with pytest.raises(ValueError, match='descriptor_cache'):
        aimmd.Params.load(str(params_file), save=False)


def test_invalid_value_is_a_value_error_on_assignment(tmp_path):
    params = aimmd.Params.load(str(_params_file(tmp_path)), save=False)
    with pytest.raises(ValueError, match="'npy' or 'graphkeys'"):
        params.descriptor_cache = 'descriptors'
    assert params.descriptor_cache == 'graphkeys'


def test_value_is_normalised(tmp_path):
    params_file = _params_file(tmp_path, "descriptor_cache = ' GraphKeys '")
    params = aimmd.Params.load(str(params_file), save=False)
    assert params.descriptor_cache == 'graphkeys'


@pytest.mark.parametrize('line, expected', [
    ("descriptor_cache = 'graphkeys'", 'graphkeys'),
    ('', 'npy'),
])
def test_field_round_trips_into_params1(tmp_path, line, expected):
    params_file = _params_file(tmp_path, line)
    params = aimmd.Params.load(str(params_file))      # saves params1.py
    assert params.descriptor_cache == expected
    saved = tmp_path / 'params1.py'
    assert saved.exists()
    assert f'descriptor_cache = {expected!r}' in saved.read_text()
    reloaded = aimmd.Params.load(str(saved), save=False)
    assert reloaded.descriptor_cache == expected
    assert reloaded.graphkeys_mode == (expected == 'graphkeys')
    assert reloaded.descriptors_source == (
        'graphkeys' if expected == 'graphkeys' else 'descriptors')
