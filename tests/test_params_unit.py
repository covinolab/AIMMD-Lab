from pathlib import Path
import shutil

import numpy as np
import pytest

import aimmd
from tests._helpers_unit import build_params_file, build_path, simple_descriptors_function


def test_placeholder_params_properties():
    """`Params.placeholder` should expose a coherent minimal configuration."""

    params = aimmd.Params.placeholder
    assert params.sorted_states == "ARB"
    assert params.compute_states_args[1] == "states"
    assert params.compute_values_args[1] == "values"
    assert len(params.pipeline) == 2


def test_params_load_save_update_and_paths(tmp_path):
    """Load, update, save, and re-read a tiny toy-engine parameter set."""

    initial = build_path(
        tmp_path,
        stem="initial",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    params_file = build_params_file(tmp_path, initial.fname)

    params = aimmd.Params.load(params_file, save=False)
    assert params.engine == "toy"
    assert len(params.initial_paths) == 1
    assert params.path.is_file()

    # Saving should emit a canonical params file that can serve as the run's
    # source of truth.
    # Save explicitly inside `tmp_path` so the test cannot leak a top-level
    # `params.py` into the repository when AIMMD resolves relative paths.
    saved = Path(params.save(tmp_path / "saved_params.py"))
    assert saved.exists()

    # `Params.update()` saves by default, so keep this update in-memory only to
    # avoid writing a stray top-level `params.py` during the test run.
    params.update(nbins=5, descriptors_function=simple_descriptors_function, save=False)
    assert params.nbins == 5
    assert params.compute_descriptors_args[1] == "descriptors"

    # `Params.pathensemble` expects an AIMMD-style run folder, so we create the
    # minimal chain layout it knows how to scan.
    run_dir = tmp_path / "run"
    chain_dir = run_dir / "chainR0"
    chain_dir.mkdir(parents=True)
    shutil.copy2(initial.fname, chain_dir / "path000001.xtc")
    ensemble = params.pathensemble(run_dir)
    assert len(ensemble) >= 1


def test_multi_system_params_fields_and_roundtrip(tmp_path):
    """Multi-system params: list topology, per-system universes, system_id-aware
    functions, shared-network path resolution, and a save/reload round-trip."""
    source = '''
import numpy as np
import torch

engine = 'toy'
multi_system = True
multi_system_share_network = True
system_ids = ['G2', 'G4']
topology = ['G2.gro', 'G4.gro']
atom_types = ['H', 'C', 'N', 'O', 'NA', 'BR']

def states_function(trajectory, system_id=None):
    cut = 1.0 if system_id == 'G2' else 1.5
    return np.array(['R'] * len(trajectory), dtype='<U1')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''
    params_file = tmp_path / "params.py"
    params_file.write_text(source)

    params = aimmd.Params.load(str(params_file), save=False)
    assert params.multi_system and params.multi_system_share_network
    assert params.system_ids == ['G2', 'G4']
    assert params.topology == ['G2.gro', 'G4.gro']
    assert params.atom_types == ['H', 'C', 'N', 'O', 'NA', 'BR']
    # per-system universe cache exists (dummy topologies -> None, but keyed)
    assert set(params.__dict__['_universes']) == {'G2', 'G4'}
    # shared network resolves to the run root from a per-system subfolder
    assert params._network_fname('run1/G2') == 'run1/networkARB.h5'
    assert params._network_fname('run1') == 'run1/networkARB.h5'

    # save and reload round-trips the list/grouped fields (topology entries are
    # rewritten to paths relative to the saved file, exactly as single-system)
    saved = Path(params.save(tmp_path / "saved.py"))
    reloaded = aimmd.Params.load(str(saved), save=False)
    assert reloaded.system_ids == ['G2', 'G4']
    assert isinstance(reloaded.topology, list) and len(reloaded.topology) == 2
    assert reloaded.multi_system is True
    assert reloaded.multi_system_share_network is True
    assert reloaded.atom_types == ['H', 'C', 'N', 'O', 'NA', 'BR']


def test_multi_system_bias_reactive_threshold_per_system(tmp_path):
    """A per-system `bias_reactive_threshold` list is validated, resolved via
    `bias_reactive_threshold_of`, and survives a save/reload round-trip."""
    source = '''
import numpy as np
import torch

engine = 'toy'
multi_system = True
multi_system_share_network = True
system_ids = ['G2', 'G4']
topology = ['G2.gro', 'G4.gro']
record_bias = True
bias_source = 'file'
bias_reactive_threshold = [0.5, 0.3]

def states_function(trajectory, system_id=None):
    return np.array(['R'] * len(trajectory), dtype='<U1')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''
    params_file = tmp_path / "params.py"
    params_file.write_text(source)

    params = aimmd.Params.load(str(params_file), save=False)
    assert params.bias_reactive_threshold == [0.5, 0.3]
    assert params.bias_reactive_threshold_of('G2') == 0.5
    assert params.bias_reactive_threshold_of('G4') == 0.3

    saved = Path(params.save(tmp_path / "saved.py"))
    reloaded = aimmd.Params.load(str(saved), save=False)
    assert reloaded.bias_reactive_threshold == [0.5, 0.3]
    assert reloaded.bias_reactive_threshold_of('G4') == 0.3

    # a wrong-length list is rejected
    bad = source.replace('[0.5, 0.3]', '[0.5, 0.3, 0.1]')
    (tmp_path / "bad.py").write_text(bad)
    with pytest.raises(TypeError):
        aimmd.Params.load(str(tmp_path / "bad.py"), save=False)


def test_subsample_caps_validation_and_roundtrip(tmp_path):
    """subsample_caps validates keys/values, resolves per system, and round-trips
    through save/reload."""
    source = '''
import numpy as np
import torch

engine = 'toy'
multi_system = True
multi_system_share_network = True
system_ids = ['G2', 'G4']
topology = ['G2.gro', 'G4.gro']
subsample_caps = [{'shot': 100, 'free': 500, 'in_state': 5000}, None]

def states_function(trajectory, system_id=None):
    return np.array(['R'] * len(trajectory), dtype='<U1')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x[:, :1])
network = Network()
'''
    params_file = tmp_path / "params.py"
    params_file.write_text(source)

    params = aimmd.Params.load(str(params_file), save=False)
    assert params.subsample_caps_of('G2') == {'shot': 100, 'free': 500,
                                              'in_state': 5000}
    assert params.subsample_caps_of('G4') is None     # per-system None = uncapped

    saved = Path(params.save(tmp_path / "saved.py"))
    reloaded = aimmd.Params.load(str(saved), save=False)
    assert reloaded.subsample_caps_of('G2')['shot'] == 100

    # bad key and non-positive value are rejected
    for bad_caps in ("{'bogus': 1}", "{'shot': 0}", "{'shot': -5}"):
        bad = source.replace(
            "[{'shot': 100, 'free': 500, 'in_state': 5000}, None]", bad_caps)
        (tmp_path / "bad.py").write_text(bad)
        with pytest.raises(TypeError):
            aimmd.Params.load(str(tmp_path / "bad.py"), save=False)


def test_params_validation_rejects_bad_network(tmp_path):
    """Network assignment should fail if the runtime interface is incomplete."""

    initial = build_path(
        tmp_path,
        stem="initial2",
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
    )
    params_file = build_params_file(tmp_path, initial.fname)
    params = aimmd.Params.load(params_file, save=False)

    class BadNetwork:
        pass

    try:
        params.update(network=BadNetwork())
    except TypeError:
        pass
    else:
        raise AssertionError("Expected network validation failure")
