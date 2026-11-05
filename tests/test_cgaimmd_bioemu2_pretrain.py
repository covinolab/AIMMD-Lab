"""The initial network of a BioEmu2 run (aimmd.cgaimmd.bioemu2.pretrain) on the tiny model.

Identity adapters, a scaler fitted on the frames, a head that separates the two
states, a saved network file with its record, and the same result for the same
seed.
"""

import json

import numpy as np
import pytest
import torch

from ._bioemu2_fixtures import tiny_bioemu2_checkpoint, tiny_peptide_system  # noqa: F401

pytestmark = pytest.mark.bioemu2


def _frames(system, n=12, seed=0):
    """Frames of two 'states': the template and a version stretched along x, with noise."""
    rng = np.random.default_rng(seed)
    template = system.template
    stretched = template * np.array([1.4, 1.0, 1.0])
    frames = np.stack([template if k % 2 == 0 else stretched for k in range(n)])
    frames = frames + 0.1 * rng.normal(size=frames.shape)
    states = np.array([1 if k % 2 == 0 else 2 for k in range(n)])
    return frames.astype(np.float32), states


def test_pretraining_separates_the_states_and_saves(tiny_peptide_system, tmp_path):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, pretrain_initial_network
    network = BioEmu2Network(tiny_peptide_system, hidden=(16,))
    with torch.no_grad():
        for adapter in network.adapters():
            adapter.out.weight.add_(1.0)
    frames, states = _frames(tiny_peptide_system)
    fname = tmp_path / 'pretrained' / 'networkARB.h5'
    record = pretrain_initial_network(network, frames, states, steps=60, batch=8, lr=3e-3, seed=2,
                                      network_file=fname)
    for adapter in network.adapters():
        assert torch.equal(adapter.out.weight, torch.zeros_like(adapter.out.weight))
    assert not torch.equal(network.head.mean, torch.zeros_like(network.head.mean))
    assert record['n_first_state'] == record['n_last_state'] == 6
    assert record['ema_logit_median_last_state'] > record['ema_logit_median_first_state']
    with torch.no_grad():
        values = network(torch.as_tensor(frames).reshape(len(frames), -1))[:, 0]
    assert values[states == 2].mean() > values[states == 1].mean()
    saved = torch.load(fname, weights_only=True)
    assert set(saved) == set(network.state_dict())
    meta = json.loads(fname.with_suffix('.pretrain.json').read_text())
    assert meta['sha256'] == record['sha256'] and meta['steps'] == 60 and meta['seed'] == 2
    again = BioEmu2Network(tiny_peptide_system, hidden=(16,))
    pretrain_initial_network(again, frames, states, steps=60, batch=8, lr=3e-3, seed=2)
    for key, value in again.state_dict().items():
        assert torch.equal(value, saved[key]), key
    network.engine().close()


def test_pretraining_needs_both_states(tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, pretrain_initial_network
    network = BioEmu2Network(tiny_peptide_system, hidden=(16,))
    frames, _ = _frames(tiny_peptide_system, n=4)
    with pytest.raises(ValueError, match='with frames of both'):
        pretrain_initial_network(network, frames, [1, 1, 1, 1], steps=1)
    with pytest.raises(ValueError, match='with frames of both'):
        pretrain_initial_network(network, frames, [0, 1, 2, 1], steps=1)


def test_the_pretraining_learning_rate_warms_up_and_decays(tiny_peptide_system, monkeypatch):
    """Step k runs at ``lr min(1, (k + 1) / 100) (1 + cos(pi k / steps)) / 2``: 100 warm-up steps, cosine decay."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, pretrain_initial_network
    rates, real_step = [], torch.optim.AdamW.step

    def step(self, *args, **kwargs):
        rates.append(self.param_groups[0]['lr'])
        return real_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    network = BioEmu2Network(tiny_peptide_system, hidden=(16,))
    frames, states = _frames(tiny_peptide_system)
    steps, lr = 150, 2e-3
    pretrain_initial_network(network, frames, states, steps=steps, batch=4, lr=lr, seed=1)
    expected = [lr * min(1.0, (k + 1) / 100) * 0.5 * (1 + np.cos(np.pi * k / steps)) for k in range(steps)]
    assert rates == pytest.approx(expected, rel=1e-12, abs=0)
    network.engine().close()
