"""The BioEmu2 network (aimmd.cgaimmd.bioemu2.network) on the tiny model.

It follows the committor-only C-GAIMMD network protocol, is built from a system
folder without loading BioEmu2, saves only its head and adapters, never copies
BioEmu2, and reproduces the values of the old networks; the networks of the
archived campaigns load through the key map (tested on synthetic checkpoints
with the old parameter names, assembled from the goldens).
"""

import copy
import pickle

import numpy as np
import pytest
import torch

from ._bioemu2_fixtures import (golden, golden_state, tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_peptide_system, tiny_protein_ligand_system)

pytestmark = pytest.mark.bioemu2

SCALER_KEYS = ('mean', 'std', 'log_sigma_mean', 'log_sigma_std')


def _scaler(golden, which):
    return {key: golden[f'{which}.network.scaler.{key}'] for key in SCALER_KEYS}


def _system(request, which):
    return request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')


def test_the_committor_protocol_and_nothing_of_generation(tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    from aimmd.cgaimmd.protocol import check_network_protocol, is_cgaimmd_network, n_atoms_of
    network = BioEmu2Network(tiny_peptide_system.folder)            # a folder: loaded without loading BioEmu2
    assert network._engine is None
    check_network_protocol(network)
    assert is_cgaimmd_network(network)
    assert n_atoms_of(network) == tiny_peptide_system.n_atoms == 166
    assert network.lambda_clip == 10.0 and network.default_generator() == {'method': 'bioemu2_edm'}
    for member in ('velocity', 'sample_prior', 'generate'):
        assert not hasattr(network, member)


@pytest.mark.parametrize('which, readout, pair, n_features', [('peptide', 'flat', False, 10 * 32),
                                                              ('ligand', 'pooled', True, 3 * 32)])
def test_widths_and_defaults_come_from_the_system_folder(which, readout, pair, n_features, request):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    system = _system(request, which)
    network = BioEmu2Network(system)
    assert network.readout == readout and (network.pair_adapter is not None) == pair
    assert network.n_features == n_features and network.widths == (32, 16, 8)
    assert network.single_adapter.settings()['dim'] == 16
    other = BioEmu2Network(system, readout='pooled' if readout == 'flat' else 'flat', pair_adapter=not pair)
    assert other.readout != readout and (other.pair_adapter is not None) != pair
    settings = network.settings()
    assert settings['readout'] == readout and settings['widths'] == [32, 16, 8]
    assert settings['checkpoint_path'] == system.data['trunk']['checkpoint_path']
    with pytest.raises(ValueError, match='readout must be one of'):
        BioEmu2Network(system, readout='mean')


def test_the_state_dict_holds_only_the_head_and_the_adapters(tiny_protein_ligand_system):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_protein_ligand_system, hidden=(16,))
    network.engine()                                                 # BioEmu2 loaded: still not in the state
    prefixes = {key.split('.')[0] for key in network.state_dict()}
    assert prefixes == {'head', 'head_ema', 'single_adapter', 'pair_adapter'}
    assert sum(p.numel() for p in network.parameters()) < 20000
    other = BioEmu2Network(tiny_protein_ligand_system.folder, hidden=(16,))
    other.load_state_dict(network.state_dict())
    network.engine().close()


@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_values_of_the_old_networks(which, request, golden):
    """The old networks' values with a seeded head: flat (chignolin code) and pooled (holo code) features."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    from aimmd.cgaimmd.values import protocol_values_function
    network = BioEmu2Network(_system(request, which), hidden=(16,), scaler=_scaler(golden, which), value_batch=3)
    network.head_ema.load_state_dict(golden_state(golden, f'{which}.network.head_ema'))
    x = golden[f'{which}.x_md']
    with torch.no_grad():
        values = network(torch.as_tensor(x).reshape(len(x), -1))
    assert values.shape == (4, 1) and not values.requires_grad
    np.testing.assert_allclose(values.numpy(), golden[f'{which}.network.values'], rtol=1e-5, atol=1e-5)
    # through AIMMD's values function of C-GAIMMD networks (inference mode, flat descriptors)
    values_function = protocol_values_function(network)
    np.testing.assert_allclose(values_function(x.reshape(len(x), -1)), values.numpy().reshape(-1), rtol=1e-6,
                               atol=1e-6)
    network.engine().close()


def test_copies_and_pickles_never_carry_bioemu2(tiny_peptide_system, golden):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_peptide_system, hidden=(16,), scaler=_scaler(golden, 'peptide'))
    x = torch.as_tensor(golden['peptide.x_md'][:2]).reshape(2, -1)
    with torch.no_grad():
        values = network(x)
    model = network.engine().model
    for clone in (copy.deepcopy(network), pickle.loads(pickle.dumps(network))):
        assert clone._engine is None and clone._model is None
        with torch.no_grad():
            assert torch.equal(clone(x), values)
        assert clone.engine().model is model                         # one BioEmu2 per process and device
        clone.engine().close()
    network.engine().close()


def test_a_given_model_is_used_and_must_be_on_the_device(tiny_peptide_system, tiny_bioemu2_model):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_peptide_system, model=tiny_bioemu2_model)
    assert network.engine().model is tiny_bioemu2_model
    network.engine().close()
    if torch.cuda.is_available():                                    # pragma: no cover - CPU in CI
        network.to('cuda')
        with pytest.raises(ValueError, match='move the network or give a model on its device'):
            network.engine()


def test_an_engine_built_in_inference_mode_still_trains(tiny_peptide_system):
    """AIMMD's values pass runs in inference mode; a fit in the same process must still get gradients."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_peptide_system, hidden=(16,))
    x = torch.as_tensor(tiny_peptide_system.template[None].repeat(2, 0), dtype=torch.float32).reshape(2, -1)
    with torch.inference_mode():
        network(x)                                                   # the engine is built here
    denoiser = network.engine().denoiser
    coords = torch.zeros(2, denoiser.n_pad, 3)
    coords[:, :tiny_peptide_system.n_bioemu2_atoms] = network.to_bioemu2(x)
    loss = denoiser.denoise(coords + 1.0, torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0])).square().mean()
    loss.backward()
    assert network.single_adapter.out.weight.grad is not None
    network.engine().close()


def test_reset_gives_identity_adapters_and_keeps_the_scaler(tiny_protein_ligand_system, golden):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_protein_ligand_system, hidden=(16,), scaler=_scaler(golden, 'ligand'))
    with torch.no_grad():
        for p in network.parameters():
            p.add_(0.5)
    network.reset_parameters(seed=3)
    for adapter in network.adapters():
        assert torch.equal(adapter.out.weight, torch.zeros_like(adapter.out.weight))
    assert torch.equal(network.head.mean, torch.as_tensor(golden['ligand.network.scaler.mean']))
    for (key, a), b in zip(network.head.state_dict().items(), network.head_ema.state_dict().values()):
        assert torch.equal(a, b), key
    first = copy.deepcopy(network.state_dict())
    network.reset_parameters(seed=3)
    for key, value in network.state_dict().items():
        assert torch.equal(value, first[key]), key


# --------------------------------------------------------------------------- the archived networks

def _old_checkpoint(golden, which):
    """A state dict with the parameter names of the old BioEmu2FlowNetwork / HoloFlowNetwork."""
    head = golden_state(golden, f'{which}.network.head_ema')
    state = {**{f'head.{k}': v for k, v in head.items()}, **{f'head_ema.{k}': v for k, v in head.items()},
             **{f'adapter.{k}': v for k, v in golden_state(golden, f'{which}.denoiser.state.single_adapter').items()}}
    if which == 'ligand':
        state.update({f'pair_adapter.{k}': v
                      for k, v in golden_state(golden, 'ligand.denoiser.state.pair_adapter').items()})
    return state


@pytest.mark.parametrize('which, readout', [('peptide', 'flat'), ('ligand', 'pooled')])
def test_archived_networks_load_through_the_key_map(which, readout, request, golden, tmp_path):
    from aimmd.cgaimmd.bioemu2.network import archived_settings, load_archived_network
    system = _system(request, which)
    fname = tmp_path / 'networkARB.h5'
    torch.save(_old_checkpoint(golden, which), fname)                 # as the old campaigns saved them
    settings = archived_settings(torch.load(fname, weights_only=True))
    assert settings['hidden'] == [16] and settings['pair_adapter'] == (which == 'ligand')
    network = load_archived_network(fname, system.folder, value_batch=2)
    assert network.readout == readout and network.settings()['adapter_hidden'] == 128
    x = golden[f'{which}.x_md']
    with torch.no_grad():
        values = network(torch.as_tensor(x).reshape(len(x), -1))
    np.testing.assert_allclose(values.numpy(), golden[f'{which}.network.values'], rtol=1e-5, atol=1e-5)
    for key, value in golden_state(golden, f'{which}.denoiser.state.single_adapter').items():
        assert torch.equal(network.single_adapter.state_dict()[key], value)
    network.engine().close()


def test_what_the_archived_loader_refuses(tiny_peptide_system, tiny_protein_ligand_system, golden):
    from aimmd.cgaimmd.bioemu2.network import archived_state_dict, load_archived_network
    with pytest.raises(KeyError, match="'velocity.weight' is not a parameter of an archived BioEmu2 network"):
        archived_state_dict({'velocity.weight': torch.zeros(1)})
    assert set(archived_state_dict({'adapter.out.bias': torch.zeros(1)})) == {'single_adapter.out.bias'}
    with pytest.raises(ValueError, match='the archived head reads 96 features'):
        load_archived_network(_old_checkpoint(golden, 'ligand'), tiny_peptide_system)
    with pytest.raises(ValueError, match='the archived head reads 320 features'):
        load_archived_network(_old_checkpoint(golden, 'peptide'), tiny_protein_ligand_system)


# --------------------------------------------------------------------------- in a params file

PARAMS_FILE = '''\
import numpy as np
from aimmd.cgaimmd.bioemu2 import BioEmu2Network, sequential_fit

engine = 'toy'
states = 'ARB'
chain_type = 'cgaimmd'
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
round_shooting_index = True
gen_temperature = 300.0
cgaimmd_parameters = {{'placement': 'none', 'window_after_trained_shots': 5}}
{generator}

def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01

def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))

def descriptors_function(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)

class Network(BioEmu2Network):
    def __init__(self):
        super().__init__({system!r}, hidden=(16,))

network = Network()

def fit(params, pathensemble, verbose=False, worker=None):
    return sequential_fit(params, pathensemble, verbose=verbose, worker=worker, min_shots_for_generator=100)
'''


def _params_folder(tmp_path, system, generator=''):
    from tests._helpers_unit import write_trajectory
    for kind, x0 in (('A', -1.0), ('B', 1.0)):
        positions = np.repeat(system.template[None], 3, axis=0).astype(np.float32)
        positions[:, 0, 0] = x0 + 0.05 * np.arange(3)              # atom 0's x decides the state
        write_trajectory(tmp_path, stem=f'seeds_{kind}', positions=positions)
    fname = tmp_path / 'params.py'
    fname.write_text(PARAMS_FILE.format(system=str(system.folder), generator=generator))
    return fname


def test_a_params_file_with_the_network_and_its_default_generator(tiny_peptide_system, tmp_path):
    """Params build the network's generator at load (rule 2): 'bioemu2_edm', or the one named."""
    import aimmd
    from aimmd.cgaimmd.bioemu2 import BioEmu2Generator
    from aimmd.cgaimmd.generation import make_generator
    from aimmd.cgaimmd.values import ProtocolValuesFunction
    params = aimmd.Params.load(str(_params_folder(tmp_path, tiny_peptide_system)), save=False)
    assert params.cgaimmd_generator is None and type(params.network).__name__ == 'Network'
    assert isinstance(params.values_function, ProtocolValuesFunction)
    generator = make_generator(params.cgaimmd_generator, params.network)
    assert isinstance(generator, BioEmu2Generator) and generator.settings()['n_steps'] == 200
    named = tmp_path / 'named'
    named.mkdir()
    params = aimmd.Params.load(str(_params_folder(named, tiny_peptide_system, generator=(
        "cgaimmd_generator = {'method': 'bioemu2_edm', 'n_steps': 50, 'cfg_scale': 2.0}"))), save=False)
    assert make_generator(params.cgaimmd_generator, params.network).settings() == {
        'n_steps': 50, 'cfg_scale': 2.0, 'cfg_schedule': None, 'chunk': 16}
