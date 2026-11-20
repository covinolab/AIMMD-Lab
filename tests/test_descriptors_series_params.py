"""The `descriptors_series` field names the cache series of the descriptors.

`descriptors_function` output is cached per trajectory in
``{trajectory}.{descriptors_series}.npy``. The default, 'descriptors', is the
historical series; any other name starts with 'descriptors-' and gives a
separate series (e.g. one per featurizer). The field drives the compute
argument tuples, the seed series that `Params.load` attaches to the initial
paths, the params file written for the workers and the initialARB export of
the launcher.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.launcher import Launcher
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import simple_descriptors_function, write_trajectory

SERIES = "descriptors-toy"

# one atom, A -> R -> R -> R -> B for the states_function of build_params_file
X = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def _positions(x):
    x = np.asarray(x, dtype=np.float32)
    zeros = np.zeros_like(x)
    return np.stack([x, zeros, zeros], 1)[:, None, :]


PARAMS_SOURCE = '''
import numpy as np
import torch

engine = 'toy'
initial_paths = ['initial.xtc']
topology = 'initial.xtc'


def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    out = np.full(len(x), 'R', dtype='<U1')
    out[x <= -0.5] = 'A'
    out[x >= 0.5] = 'B'
    return out


def descriptors_function(trajectory):
    return np.array([ts.positions[:, 0].copy() for ts in trajectory],
                    dtype=float)


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


def _params_file(folder, series=SERIES):
    """A toy params file next to its initial path; chdir into `folder`
    before using the loaded params (the initial path name is relative)."""
    os.makedirs(folder, exist_ok=True)
    write_trajectory(folder, stem="initial", positions=_positions(X))
    source = PARAMS_SOURCE
    if series is not None:
        source += f"descriptors_series = {series!r}\n"
    params_file = Path(folder) / "params.py"
    params_file.write_text(source)
    return params_file


def _descriptor_attributes(path):
    return sorted(name for name in path.__dict__
                  if name == "descriptors" or name.startswith("descriptors-"))


def test_default_series_is_the_historical_one():
    params = aimmd.Params.placeholder
    assert params.descriptors_series == "descriptors"
    assert params.compute_descriptors_args is None
    assert params.compute_values_args[2] == "coordinates"

    params.__dict__["descriptors_function"] = simple_descriptors_function
    assert params.compute_descriptors_args == (simple_descriptors_function,
                                               "descriptors")
    assert params.compute_values_args[1:] == ("values", "descriptors")
    assert params.pipeline[0] == params.compute_descriptors_args


def test_compute_arguments_use_the_named_series():
    params = aimmd.Params.placeholder
    params.__dict__.update(descriptors_function=simple_descriptors_function,
                           descriptors_series=SERIES)
    assert params.compute_descriptors_args == (simple_descriptors_function,
                                               SERIES)
    assert params.compute_values_args[1:] == ("values", SERIES)
    descriptors, states, values = params.pipeline
    assert descriptors == (simple_descriptors_function, SERIES)
    assert states[1] == "states"
    assert values[1:] == ("values", SERIES)

    # without a descriptors_function the series plays no role
    params.__dict__["descriptors_function"] = None
    assert params.compute_descriptors_args is None
    assert params.compute_values_args[2] == "coordinates"


@pytest.mark.parametrize("series", [
    "descriptors-gne38bf950a1", "descriptors-toy_v2", "descriptors-a.b"])
def test_valid_series_names(series):
    params = aimmd.Params.placeholder
    params.descriptors_series = series
    assert params.descriptors_series == series


@pytest.mark.parametrize("series", [
    "", "states", "values", "descriptor", "descriptorsX", "descriptors_gn",
    "descriptors-", "descriptors-a/b", "descriptors-a b", "descriptors-*",
    "Descriptors-a", None, 3])
def test_invalid_series_names_raise_value_error(series):
    params = aimmd.Params.placeholder
    with pytest.raises(ValueError, match="descriptors_series"):
        params.descriptors_series = series
    assert params.descriptors_series == "descriptors"


def test_invalid_series_name_in_params_file(tmp_path):
    params_file = _params_file(tmp_path, series="graphs")
    with pytest.raises(ValueError, match="descriptors_series"):
        aimmd.Params.load(params_file, save=False)


def test_params_file_round_trip(tmp_path, monkeypatch):
    """The field reaches the params file the workers load (params1.py)."""
    monkeypatch.chdir(tmp_path)
    params_file = _params_file(tmp_path)

    params = aimmd.Params.load(params_file)        # saves params1.py

    assert params.descriptors_series == SERIES
    saved = Path(params.path)
    assert saved.name == "params1.py"
    assert f"descriptors_series = {SERIES!r}" in saved.read_text()
    reloaded = aimmd.Params.load(saved, save=False)
    assert reloaded.descriptors_series == SERIES
    assert reloaded.compute_descriptors_args[1] == SERIES

    # the default is written out too, and reads back as the default
    default = aimmd.Params.load(_params_file(tmp_path / "default", None))
    assert "descriptors_series = 'descriptors'" in Path(default.path).read_text()
    assert aimmd.Params.load(default.path, save=False).descriptors_series \
        == "descriptors"


def test_seed_series_is_stored_under_the_named_series(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = aimmd.Params.load(_params_file(tmp_path), save=False)

    path, = params.initial_paths
    assert _descriptor_attributes(path) == [SERIES]
    np.testing.assert_array_equal(getattr(path, SERIES), path.positions[:, :, 0])


def test_changing_the_series_refeaturizes_the_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = aimmd.Params.load(_params_file(tmp_path), save=False)

    params.descriptors_series = "descriptors-other"
    path, = params.initial_paths
    assert _descriptor_attributes(path) == ["descriptors-other"]
    np.testing.assert_array_equal(getattr(path, "descriptors-other"),
                                  path.positions[:, :, 0])

    params.descriptors_series = "descriptors"      # back to the default
    path, = params.initial_paths
    assert _descriptor_attributes(path) == ["descriptors"]


MULTI_SYSTEM_SOURCE = '''
import numpy as np
import torch

engine = 'toy'
multi_system = True
system_ids = ['s1', 's2']
topology = ['s1.xtc', 's2.xtc']
initial_paths = [['s1.xtc'], ['s2.xtc']]
descriptors_series = 'descriptors-ms'


def states_function(trajectory, system_id=None):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    out = np.full(len(x), 'R', dtype='<U1')
    out[x <= -0.5] = 'A'
    out[x >= 0.5] = 'B'
    return out


def descriptors_function(trajectory, system_id=None):
    offset = 10.0 if system_id == 's2' else 0.0
    return np.array([ts.positions[:1, 0] + offset for ts in trajectory])


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(torch.as_tensor(x, dtype=torch.float32)[:, :1])

network = Network()
'''


def test_multi_system_seed_series(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_trajectory(tmp_path, stem="s1", positions=_positions(X))
    write_trajectory(tmp_path, stem="s2", positions=np.concatenate(
        [_positions(X), _positions(X)], axis=1))
    (tmp_path / "params.py").write_text(MULTI_SYSTEM_SOURCE)

    params = aimmd.Params.load(str(tmp_path / "params.py"), save=False)

    assert params.compute_descriptors_args[1] == "descriptors-ms"
    for group, offset in zip(params.initial_paths, (0.0, 10.0)):
        path, = group
        assert _descriptor_attributes(path) == ["descriptors-ms"]
        np.testing.assert_allclose(getattr(path, "descriptors-ms")[:, 0],
                                   path.positions[:, 0, 0] + offset)


def test_launcher_exports_the_named_series(tmp_path, monkeypatch):
    """initialARB/<name>.xtc.<series>.npy holds the seed rows."""
    monkeypatch.chdir(tmp_path)
    params = aimmd.Params.load(_params_file(tmp_path), save=False)
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_cpus", lambda: 4)
    monkeypatch.setattr("aimmd.launcher._helpers.get_num_gpus", lambda: 0)

    launcher = Launcher([params], [str(tmp_path / "run")])
    launcher._update(n=1, n1=0, n2=0, nrounds=0, cpus_per_task=1,
                     gpus_per_task=0, ntasks_per_node=1)
    launcher._build()

    folder = tmp_path / "run" / "initialARB"
    exported = folder / "initial.xtc"
    assert exported.exists()
    series = np.load(get_cache_fname(exported, SERIES))
    np.testing.assert_array_equal(
        series, getattr(params.initial_paths[0], SERIES))
    assert not os.path.exists(get_cache_fname(exported, "descriptors"))
    # (hidden files are MDAnalysis offsets)
    assert sorted(name for name in os.listdir(folder)
                  if not name.startswith(".")) == sorted([
        "initial.xtc", f"initial.xtc.{SERIES}.npy", "initial.xtc.states.npy"])
