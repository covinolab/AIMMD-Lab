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


# ------------------------------------------------ Params.load: initial paths --
SEED_SOURCE = '''
import numpy as np
import torch
from tests._helpers_graphkeys import SqliteToyCache
from tests._helpers_graphkeys import descriptors_function as _coordinates

engine = 'toy'
DESCRIPTOR_CACHE_LINE
initial_paths = ['initial.xtc']
CACHE = SqliteToyCache('graphs.sqlite')


def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x < 2, 'A', np.where(x > 8, 'B', 'R')).astype('<U1')


def descriptors_function(trajectory):
    rows = _coordinates(trajectory)
    SIDE_EFFECT
    return rows


def descriptor_transform(x):
    return CACHE.transform(x)


def values_function(x):
    VALUES


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''

VALUES = {
    'toy': 'return CACHE.values(x)',
    'raises': 'return CACHE.values(np.asarray(x).reshape(len(x), -1, 3))',
    'numbers': 'return np.asarray(x, dtype=float)[:, 0]',
    'shape': 'return np.zeros((len(x), 2))',
}


def _sweep(fname, n_frames=20, offset=0.0):
    """Atom 0 sweeps x from 0 to 10 (A -> R -> B); atom 1 is fixed."""
    import MDAnalysis as mda
    universe = mda.Universe.empty(2, trajectory=True)
    with mda.Writer(str(fname), 2) as writer:
        for x in np.linspace(0.0, 10.0, n_frames):
            universe.atoms.positions = np.array(
                [[x, 5.0, 5.0], [5.0 + offset, 5.0, 5.0]], dtype=np.float32)
            writer.write(universe.atoms)


def _seed_params(folder, cache='graphkeys', values='toy', side_effect=False):
    folder = PosixPath(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'initial.xtc').exists():
        _sweep(folder / 'initial.xtc')
    source = (SEED_SOURCE
              .replace('DESCRIPTOR_CACHE_LINE', f'descriptor_cache = {cache!r}')
              .replace('SIDE_EFFECT', 'CACHE.transform(rows)' if side_effect
                       else 'pass')
              .replace('VALUES', VALUES[values]))
    (folder / 'params.py').write_text(source)
    return str(folder / 'params.py')


def _seed_rows(params):
    path = params.initial_paths[0]
    return np.asarray(path.coordinates, dtype=np.float32)


def _foreign_cache(folder, n=5):
    """A graph cache that holds graphs, none of them of the seed frames."""
    from tests._helpers_graphkeys import SqliteToyCache
    cache = SqliteToyCache(PosixPath(folder) / 'graphs.sqlite')
    cache.add(np.random.default_rng(1).uniform(0, 1, (n, 6)))
    return cache


def test_load_keys_the_initial_paths_and_never_touches_descriptors(
        tmp_path, monkeypatch):
    from tests._helpers_graphkeys import SqliteToyCache, forbid_descriptor_files
    params_file = _seed_params(tmp_path)
    monkeypatch.chdir(tmp_path)          # initial paths are relative to it
    with forbid_descriptor_files(monkeypatch) as opened:
        params = aimmd.Params.load(params_file, save=False)
    assert opened == []
    path = params.initial_paths[0]
    assert 'descriptors' not in path.__dict__
    keys = path.__dict__['graphkeys']
    assert keys.dtype == np.uint8 and keys.shape == (len(path), 32)
    assert np.array_equal(keys, graph_keys_of(_seed_rows(params)))
    # the graphs of every seed frame were built on the way
    assert SqliteToyCache(tmp_path / 'graphs.sqlite').count() == len(path)
    assert not list(tmp_path.glob('*.descriptors.npy'))
    assert not list(tmp_path.glob('*.graphkeys.npy')), 'kept in memory'


def graph_keys_of(rows):
    from aimmd.core.graphkey import graph_keys
    return graph_keys(rows)


def test_selftest_stops_a_load_whose_keys_the_cache_does_not_know(tmp_path):
    _foreign_cache(tmp_path)
    params_file = _seed_params(tmp_path)
    with pytest.raises(RuntimeError, match='none of the .* frames') as info:
        aimmd.Params.load(params_file, save=False)
    assert 'AIMMD_GRAPHKEYS_SKIP_SELFTEST' in str(info.value)
    assert 'graphs.sqlite' in str(info.value)


def test_selftest_can_be_skipped(tmp_path, monkeypatch, capsys):
    _foreign_cache(tmp_path)
    monkeypatch.setenv('AIMMD_GRAPHKEYS_SKIP_SELFTEST', '1')
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    assert 'graphkeys' in params.initial_paths[0].__dict__
    assert 'check skipped' in capsys.readouterr().out


def test_selftest_passes_on_a_cache_that_knows_the_seed(tmp_path):
    # an earlier load (empty cache: nothing to check) fills the cache, ...
    aimmd.Params.load(_seed_params(tmp_path), save=False)
    # ... and later loads find the seed's graphs in it
    _foreign_cache(tmp_path)
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    assert 'graphkeys' in params.initial_paths[0].__dict__


def test_switching_an_npy_run_with_a_graph_cache_passes(tmp_path):
    """The npy-mode loads built the seed graphs as a side effect."""
    _foreign_cache(tmp_path)
    aimmd.Params.load(_seed_params(tmp_path, cache='npy', side_effect=True),
                      save=False)
    params = aimmd.Params.load(_seed_params(tmp_path, side_effect=True),
                               save=False)
    assert params.graphkeys_mode


def test_selftest_ignores_graphs_the_keying_itself_stores(tmp_path):
    """A descriptors_function that builds graphs cannot fool the check."""
    _foreign_cache(tmp_path)
    with pytest.raises(RuntimeError, match='none of the'):
        aimmd.Params.load(_seed_params(tmp_path, side_effect=True),
                          save=False)


def test_selftest_does_not_compare_with_the_default_pickle_protocol(
        tmp_path, monkeypatch):
    """Python 3.14 pickles with protocol 5 by default; keys must not care."""
    import pickle
    aimmd.Params.load(_seed_params(tmp_path), save=False)
    monkeypatch.setattr(pickle, 'DEFAULT_PROTOCOL', 5)
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    assert params.graphkeys_mode


@pytest.mark.parametrize('values, message', [
    ('raises', 'failed on a graph-key row'),
    ('numbers', 'does not look graphs up by key'),
    ('shape', r'expected \(1,\)'),
])
def test_values_function_must_evaluate_graph_keys(tmp_path, values, message):
    with pytest.raises(RuntimeError, match=message) as info:
        aimmd.Params.load(_seed_params(tmp_path, values=values), save=False)
    assert 'process_descriptors_pyg' in str(info.value)


def test_skipping_the_selftest_keeps_the_shape_check(tmp_path, monkeypatch):
    monkeypatch.setenv('AIMMD_GRAPHKEYS_SKIP_SELFTEST', 'yes')
    aimmd.Params.load(_seed_params(tmp_path, values='numbers'), save=False)
    with pytest.raises(RuntimeError, match=r'expected \(1,\)'):
        aimmd.Params.load(_seed_params(tmp_path, values='shape'), save=False)


def test_values_check_survives_a_graph_store_that_did_not_land(tmp_path):
    """The check repairs a seed graph that is missing, like a value pass."""
    from tests._helpers_graphkeys import SqliteToyCache
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    first = params.initial_paths[0].__dict__['graphkeys'][0].tobytes().hex()
    cache = SqliteToyCache(tmp_path / 'graphs.sqlite')
    cache.conn.execute('DELETE FROM graphs_cache WHERE key = ?', (first,))
    cache.conn.commit()
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    assert params.graphkeys_mode
    assert cache.conn.execute('SELECT COUNT(*) FROM graphs_cache WHERE key = ?',
                              (first,)).fetchone()[0] == 1


def test_update_switches_the_initial_paths_between_modes(tmp_path):
    params = aimmd.Params.load(_seed_params(tmp_path, cache='npy'), save=False)
    path = params.initial_paths[0]
    assert 'descriptors' in path.__dict__ and 'graphkeys' not in path.__dict__
    rows = path.__dict__['descriptors']

    params.update(descriptor_cache='graphkeys', save=False)
    path = params.initial_paths[0]
    assert 'descriptors' not in path.__dict__
    assert np.array_equal(path.__dict__['graphkeys'], graph_keys_of(rows))

    params.update(descriptor_cache='npy', save=False)
    path = params.initial_paths[0]
    assert 'graphkeys' not in path.__dict__
    assert np.array_equal(path.__dict__['descriptors'], rows)


def test_launcher_exports_the_initial_graph_keys(tmp_path, monkeypatch):
    from tests._helpers_graphkeys import forbid_descriptor_files
    params = aimmd.Params.load(_seed_params(tmp_path), save=False)
    monkeypatch.chdir(tmp_path)          # initial paths are relative to it
    folder = tmp_path / 'run1' / 'initialARB'
    folder.mkdir(parents=True)
    stale = folder / 'initial.xtc.descriptors.npy'
    np.save(stale, np.zeros((3, 6)))
    launcher = aimmd.Launcher(params, str(tmp_path / 'run1'))
    launcher._update(n=0)
    with forbid_descriptor_files(monkeypatch) as opened:
        launcher._build()
    assert opened == []
    assert not stale.exists(), 'the stale descriptors were wiped'
    exported = folder / 'initial.xtc.graphkeys.npy'
    assert exported.exists()
    assert np.array_equal(np.load(exported),
                          params.initial_paths[0].__dict__['graphkeys'])
    assert not list(folder.glob('*.descriptors.npy'))
    # a worker reading the exported seed gets the same keys
    seed = aimmd.PathEnsemble(str(folder / '*'))[0]
    assert np.array_equal(seed.graphkeys, np.load(exported))


# --------------------------------------------- Params.load: multi-system --
MULTI_SOURCE = '''
import numpy as np
import torch
from tests._helpers_graphkeys import SqliteToyCache
from tests._helpers_graphkeys import descriptors_function as _coordinates

engine = 'toy'
multi_system = True
multi_system_share_network = True
system_ids = ['s1', 's2']
topology = ['s1.xtc', 's2.xtc']
initial_paths = [['s1.xtc'], ['s2.xtc']]
descriptor_cache = 'graphkeys'
CACHES = {'s1': SqliteToyCache('s1.sqlite'), 's2': SqliteToyCache('s2.sqlite')}
SEEN = []


def states_function(trajectory, system_id=None):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x < 2, 'A', np.where(x > 8, 'B', 'R')).astype('<U1')


def descriptors_function(trajectory, system_id=None):
    return _coordinates(trajectory)


def descriptor_transform(x, system_id=None):
    return CACHES[system_id].transform(x)


def values_function(x, system_id=None):
    SEEN.append(system_id)
    return CACHES[system_id].values(x)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''


def _multi_params(folder):
    folder = PosixPath(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _sweep(folder / 's1.xtc', offset=0.0)
    _sweep(folder / 's2.xtc', offset=1.0)
    (folder / 'params.py').write_text(MULTI_SOURCE)
    return str(folder / 'params.py')


def test_multi_system_load_keys_each_system_in_its_own_cache(tmp_path,
                                                           monkeypatch):
    from tests._helpers_graphkeys import SqliteToyCache
    params = aimmd.Params.load(_multi_params(tmp_path), save=False)
    monkeypatch.chdir(tmp_path)          # initial paths are relative to it
    for sid, group in zip(params.system_ids, params.initial_paths):
        path = group[0]
        assert 'descriptors' not in path.__dict__
        rows = np.asarray(path.coordinates, dtype=np.float32)
        assert np.array_equal(path.__dict__['graphkeys'], graph_keys_of(rows))
        assert SqliteToyCache(tmp_path / f'{sid}.sqlite').count() == len(path)
    # each system's values check ran with its system_id
    module_seen = params.values_function.__globals__['SEEN']
    assert {'s1', 's2'} <= set(module_seen)


def test_multi_system_selftest_names_the_system(tmp_path):
    folder = tmp_path / 'multi'
    params_file = _multi_params(folder)
    from tests._helpers_graphkeys import SqliteToyCache
    SqliteToyCache(folder / 's2.sqlite').add(
        np.random.default_rng(2).uniform(0, 1, (4, 6)))
    with pytest.raises(RuntimeError, match="system 's2'"):
        aimmd.Params.load(params_file, save=False)
