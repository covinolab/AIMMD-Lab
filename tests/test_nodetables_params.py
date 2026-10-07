"""A node-table featurizer in a params file.

`aimmd.Params` stores functions, not objects: it unwraps a bound method to
its bare function, which would drop the featurizer (``FEATURIZER.
descriptors_function`` then fails with "missing 1 required positional
argument"). Params files therefore use module-level wrapper functions, and
Params refuses a featurizer's bound method with a message saying so. With the
wrappers and the pinned, checked series name, Params.load featurizes the seed
into the named series and writes a params1.py that workers load the same way.
Needs only numpy and MDAnalysis (the values here are not computed from
graphs).
"""
import functools
import os
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

import aimmd
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      NodeTableFeaturizer)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames, toy_universe,
                                   write_toy_gro, write_toy_xtc)

# the ligand moves along y: A (y < 7) -> R -> B (y > 13), split across x
FRAMES = toy_frames(12, seed=4, ligand_y=np.linspace(5.0, 15.0, 12))

PARAMS_SOURCE = '''
import numpy as np
import torch
import MDAnalysis as mda
from aimmd.network.nodetables import NodeTableFeaturizer

engine = 'toy'
initial_paths = ['initial.xtc']
topology = 'toy.gro'

FEATURIZER = NodeTableFeaturizer(
    mda.Universe('toy.gro', to_guess=['types', 'bonds']),
    {system!r}, {environment!r}, None, cutoff={cutoff!r})
descriptors_series = FEATURIZER.check_series({series!r})


def states_function(trajectory):
    y = np.array([ts.positions[:6, 1].mean() for ts in trajectory])
    out = np.full(len(y), 'R', dtype='<U1')
    out[y < 7.0] = 'A'
    out[y > 13.0] = 'B'
    return out


def values_function(rows, verbose=False):
    return (np.asarray(rows)[:, 0] - 40.0) / 10.0


def toy_mdrun(ts):
    ts.positions[:] = ts.positions + 0.1


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(torch.as_tensor(x, dtype=torch.float32)[:, :1])

network = Network()
'''

WRAPPERS = '''

def descriptors_function(trajectory):
    return FEATURIZER.descriptors_function(trajectory)
'''


def _toy_featurizer(folder):
    universe = mda.Universe(str(Path(folder) / 'toy.gro'),
                            to_guess=['types', 'bonds'])
    return NodeTableFeaturizer(universe, SYSTEM_SELECTION,
                               ENVIRONMENT_SELECTION, None, cutoff=CUTOFF)


def _params_file(folder, functions=WRAPPERS, series=None):
    """The toy params file, its topology and its initial path in `folder`."""
    os.makedirs(folder, exist_ok=True)
    write_toy_gro(Path(folder) / 'toy.gro', FRAMES[0])
    write_toy_xtc(Path(folder) / 'initial.xtc', FRAMES)
    if series is None:
        series = _toy_featurizer(folder).series
    source = PARAMS_SOURCE.format(
        system=SYSTEM_SELECTION, environment=ENVIRONMENT_SELECTION,
        cutoff=CUTOFF, series=series) + functions
    params_file = Path(folder) / 'params.py'
    params_file.write_text(source)
    return params_file


def test_module_level_wrappers_load_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params_file = _params_file(tmp_path)
    featurizer = _toy_featurizer(tmp_path)
    series = featurizer.series

    params = aimmd.Params.load(params_file)         # saves params1.py

    assert params.descriptors_series == series
    assert params.compute_descriptors_args[1] == series
    path, = params.initial_paths
    rows = getattr(path, series)
    assert len(path) < len(FRAMES)                  # the A -> B block
    assert rows.shape == (len(path), featurizer.width) and rows[:, 0].all()
    np.testing.assert_array_equal(
        rows, path.compute(featurizer.descriptors_function))

    # the params file of the workers loads the featurizer the same way
    saved = Path(params.path)
    assert saved.name == 'params1.py'
    assert f'descriptors_series = {series!r}' in saved.read_text()
    reloaded = aimmd.Params.load(saved, save=False)
    assert reloaded.descriptors_series == series
    reloaded_path, = reloaded.initial_paths
    np.testing.assert_array_equal(getattr(reloaded_path, series), rows)
    np.testing.assert_array_equal(
        reloaded_path.compute(reloaded.descriptors_function), rows)


@pytest.mark.parametrize('name, method', [
    ('descriptors_function', 'descriptors_function'),
    ('descriptor_transform', 'graphs'),
    ('values_function', 'batch_dict')])
def test_a_bound_method_in_a_params_file_is_refused_clearly(
        tmp_path, monkeypatch, name, method):
    monkeypatch.chdir(tmp_path)
    functions = WRAPPERS if name != 'descriptors_function' else ''
    functions += f'\n{name} = FEATURIZER.{method}\n'
    params_file = _params_file(tmp_path, functions)

    with pytest.raises(TypeError) as info:
        aimmd.Params.load(params_file, save=False)

    message = str(info.value)
    assert f'{name!r}' in message and f'NodeTableFeaturizer.{method}' in message
    assert 'module-level' in message
    assert 'missing 1 required positional argument' not in message


def _multi_system_featurizers():
    featurizer = NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                                     ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF)
    return MultiSystemNodeTableFeaturizer({'0': featurizer})


@pytest.mark.parametrize('make, method', [
    (lambda: NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                                 ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF),
     'descriptors_function'),
    (_multi_system_featurizers, 'descriptors_function'),
    (_multi_system_featurizers, 'graphs')])
def test_assigning_a_bound_method_is_refused(make, method):
    params = aimmd.Params.placeholder
    bound = getattr(make(), method)
    field = ('descriptors_function' if method == 'descriptors_function'
             else 'descriptor_transform')
    with pytest.raises(TypeError, match='module-level'):
        setattr(params, field, bound)
    assert getattr(params, field) is None


@pytest.mark.parametrize('wrap', [
    lambda f: functools.partial(f.descriptors_function),
    lambda f: functools.partial(functools.partial(f.descriptors_function)),
    lambda f: functools.partial(type(f).descriptors_function, f)])
def test_a_partial_over_a_featurizer_is_refused(wrap):
    """A functools.partial is no bound method, but Params cannot store it
    either: the generated params1.py would read 'from functools import
    descriptors_function', and every worker's load would fail at job
    start."""
    featurizer = NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                                     ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF)
    params = aimmd.Params.placeholder
    with pytest.raises(TypeError, match='module-level') as info:
        params.descriptors_function = wrap(featurizer)
    message = str(info.value)
    assert 'functools.partial' in message
    assert 'NodeTableFeaturizer.descriptors_function' in message
    # the wrapper to write instead, with the partial's remaining parameters
    assert 'def descriptors_function(trajectory):' in message
    assert 'INSTANCE.descriptors_function(trajectory)' in message
    assert params.descriptors_function is None


def test_a_partial_over_a_featurizer_in_a_params_file_is_refused(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params_file = _params_file(
        tmp_path, 'import functools\n' + WRAPPERS.replace(
            'def descriptors_function(trajectory):\n'
            '    return FEATURIZER.descriptors_function(trajectory)\n',
            'descriptors_function = functools.partial(\n'
            '    FEATURIZER.descriptors_function)\n'))
    assert 'functools.partial(' in Path(params_file).read_text()

    with pytest.raises(TypeError, match='functools.partial'):
        aimmd.Params.load(params_file, save=False)


def test_other_bound_methods_are_still_unwrapped():
    class Functions:
        def descriptors_function(trajectory):        # no self, used unbound
            return np.zeros((len(trajectory), 1))

    params = aimmd.Params.placeholder
    params.descriptors_function = Functions().descriptors_function
    assert params.descriptors_function is Functions.descriptors_function


def test_a_mismatched_pinned_series_fails_the_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params_file = _params_file(tmp_path, series='descriptors-gn0000000000')
    with pytest.raises(ValueError, match="'descriptors-gn0000000000'"):
        aimmd.Params.load(params_file, save=False)
