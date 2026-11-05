"""The trunk adapters, the conditioned denoiser and the guidance schedule (aimmd.cgaimmd.bioemu2.adapters).

On the tiny model: fresh adapters leave BioEmu2 exactly as it is; the goldens of
the old ``finetune.TrunkAdapter`` / ``ConditionedDenoiser`` (single, learned null,
single + pair) and of its guidance hook are reproduced; the guidance schedule
parses and picks its windows like the generator-adherence campaign's. The
adapter and schedule tests without a model run without bioemu2.
"""

import math

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.bioemu2.adapters import (GuidanceSchedule, TrunkAdapter, lambda_features,
                                            parse_guidance_schedule)

from ._bioemu2_fixtures import (golden, golden_state, tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_peptide_system, tiny_protein_ligand_system)
from ._cgaimmd_bioemu2_golden import (ADAPTER_KEEP, ADAPTER_LAMBDAS, CFG_TARGET, CFG_WEIGHT, DENOISER_LAMBDAS,
                                      DENOISER_SIGMAS, DENOISER_UNCOND)

# --------------------------------------------------------------------------- without a model


def test_a_fresh_adapter_is_the_identity():
    adapter = TrunkAdapter(16, hidden=8, e_dim=4)
    x = torch.randn(3, 5, 16)
    lam, keep = torch.tensor([-12.0, 0.0, 3.0]), torch.tensor([True, False, True])
    assert torch.equal(adapter(x, lam, keep), x)
    null = TrunkAdapter(16, hidden=8, e_dim=4, learned_null=True)
    assert torch.equal(null.null, torch.zeros(4)) and torch.equal(null(x, lam, keep), x)
    assert adapter.settings() == dict(dim=16, hidden=8, n_freq=8, e_dim=4, learned_null=False)


def test_reset_is_the_old_fresh_adapter_and_leaves_the_global_stream():
    adapter = TrunkAdapter(6, hidden=4, learned_null=True)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(1.0)
    state = torch.random.get_rng_state()
    adapter.reset_parameters(seed=11)
    assert torch.equal(torch.random.get_rng_state(), state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        old = TrunkAdapter(6, hidden=4, learned_null=True).state_dict()
    for key, value in old.items():
        assert torch.equal(adapter.state_dict()[key], value)
    assert torch.equal(adapter.out.weight, torch.zeros_like(adapter.out.weight))


def test_unlabelled_rows_without_and_with_a_learned_null():
    torch.manual_seed(0)
    plain, null = TrunkAdapter(6, hidden=4), TrunkAdapter(6, hidden=4, learned_null=True)
    for adapter in (plain, null):
        with torch.no_grad():
            for p in adapter.parameters():
                p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(2, 3, 6)
    lam, keep = torch.tensor([1.0, 1.0]), torch.tensor([True, False])
    out = plain(x, lam, keep)
    assert torch.equal(out[1], x[1]) and not torch.allclose(out[0], x[0])       # no label: unchanged
    out = null(x, lam, keep)
    assert not torch.allclose(out[1], x[1])                                     # no label: the learned null
    # the null row does not depend on the label it was handed
    assert torch.equal(null(x, torch.tensor([1.0, -7.0]), keep)[1], out[1])


def test_lambda_features_clip_at_lambda_clip():
    freq = torch.pi * torch.logspace(-1, 0.5, 8)
    assert torch.equal(lambda_features(torch.tensor([25.0]), freq), lambda_features(torch.tensor([10.0]), freq))
    assert lambda_features(torch.tensor([2.0]), freq).shape == (1, 17)


def test_the_guidance_schedule():
    windows = parse_guidance_schedule('8-inf:3:learned, 2-8:2.5:bioemu2')
    assert windows == ((8.0, math.inf, 3.0, 'learned'), (2.0, 8.0, 2.5, 'bioemu2'))
    assert parse_guidance_schedule(None) == () and parse_guidance_schedule('') == ()
    assert parse_guidance_schedule([(8, float('inf'), 3, 'learned')]) == ((8.0, math.inf, 3.0, 'learned'),)
    schedule = GuidanceSchedule(1.5, '8-inf:3:learned,2-8:2.5:bioemu2')
    assert schedule.at(100.0) == (3.0, 'learned') and schedule.at(8.0) == (3.0, 'learned')   # first window wins
    assert schedule.at(4.0) == (2.5, 'bioemu2') and schedule.at(1.0) == (1.5, 'learned')
    assert schedule.spec() == '8-inf:3:learned,2-8:2.5:bioemu2'
    assert GuidanceSchedule(1.0, schedule.spec()).windows == schedule.windows
    assert GuidanceSchedule().is_off and GuidanceSchedule(1.0, '0-1:1:bioemu2').is_off
    assert not GuidanceSchedule(2.0).is_off and not schedule.is_off
    for spec, message in (('8-inf:3', "is not 'lo-hi:weight:null'"), ('8-inf:3:cfg', "unknown null 'cfg'"),
                          ('9-8:3:learned', 'needs 0 <= lo <= hi'), ('x-8:3:learned', "noise level 'x'"),
                          ('1-8:nan:learned', 'must be finite')):
        with pytest.raises(ValueError, match=message):
            parse_guidance_schedule(spec)
    with pytest.raises(ValueError, match='finite number'):
        GuidanceSchedule(float('inf'))


# --------------------------------------------------------------------------- on the tiny model


def _network(system, golden, prefix, *, learned_null=False):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(system, hidden=(16,), learned_null=learned_null)
    network.single_adapter.load_state_dict(golden_state(golden, f'{prefix}.single_adapter'))
    if network.pair_adapter is not None:
        network.pair_adapter.load_state_dict(golden_state(golden, f'{prefix}.pair_adapter'))
    return network


@pytest.mark.bioemu2
@pytest.mark.parametrize('which, prefix, learned_null', [('peptide', 'denoiser', False),
                                                         ('peptide', 'denoiser_null', True),
                                                         ('ligand', 'denoiser', False)])
def test_goldens_of_the_old_conditioned_denoiser(which, prefix, learned_null, request, golden):
    """Labelled rows, unlabelled rows, no label, plain BioEmu2 and the guidance hook of the old code."""
    from aimmd.cgaimmd.bioemu2.sampling import GuidedModel
    system = request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')
    network = _network(system, golden, f'{which}.{prefix}.state', learned_null=learned_null)
    denoiser = network.engine().denoiser
    x, sigma = torch.as_tensor(golden[f'{which}.{prefix}.x']), torch.tensor(DENOISER_SIGMAS)
    with torch.no_grad():
        got = dict(labelled=denoiser.denoise(x, sigma, torch.tensor(DENOISER_LAMBDAS), torch.tensor(DENOISER_UNCOND)),
                   no_label=denoiser.denoise(x, sigma, None), plain=denoiser.plain_denoise(x, sigma),
                   cfg=GuidedModel(denoiser, torch.full((3,), CFG_TARGET), GuidanceSchedule(CFG_WEIGHT)).denoise(
                       None, None, x, sigma).denoised_coords)
    # the peptide's trunk comes from the build step, the old code embedded at engine start (1.4e-6 apart)
    tolerance = 2e-5 if which == 'peptide' else 1e-6
    for name, value in got.items():
        np.testing.assert_allclose(value.numpy(), golden[f'{which}.{prefix}.{name}'], rtol=0, atol=tolerance,
                                   err_msg=name)
    adapter = f'{which}.adapter_null' if learned_null else f'{which}.adapter'
    with torch.no_grad():
        out = network.single_adapter(torch.as_tensor(golden[f'{adapter}.single.x']), torch.tensor(ADAPTER_LAMBDAS),
                                     torch.tensor(ADAPTER_KEEP))
    np.testing.assert_allclose(out.numpy(), golden[f'{adapter}.single.out'], rtol=0, atol=1e-6)
    network.engine().close()


@pytest.mark.bioemu2
@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_zero_adapters_are_plain_bioemu2(which, request):
    """Fresh adapters: the conditioned denoiser gives exactly what the frozen model gives on the plain trunk."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, _api
    system = request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')
    network = BioEmu2Network(system, hidden=(16,), learned_null=True)
    engine = network.engine()
    denoiser, model = engine.denoiser, engine.model
    k = 3
    x = torch.zeros(k, denoiser.n_pad, 3)
    x[:, :system.n_bioemu2_atoms] = torch.as_tensor(system.reference, dtype=torch.float32)
    sigma = torch.tensor([0.5, 4.0, 16.0])
    x = x + sigma[:, None, None] * torch.randn(x.shape, generator=torch.Generator().manual_seed(3))
    t = denoiser.trunk

    def rep(value):
        return value.expand(k, *value.shape[1:])
    trunk_k = _api.TrunkEmbeddings(s_inputs=rep(t.s_inputs), single=rep(t.single), pair=rep(t.pair),
                                   relative_position_encoding=rep(t.relative_position_encoding),
                                   distogram=rep(t.distogram))
    with torch.no_grad():
        context = model.compute_static_context(_api.expand_batch(denoiser.batch, k), trunk_k, 1)
        plain = model.denoise(context, trunk_k, x, sigma).denoised_coords
        for lam, uncond in ((torch.tensor([-12.0, 0.0, 4.0]), None), (torch.tensor([1.0, 2.0, 3.0]),
                                                                      torch.tensor([True, False, True])),
                            (None, None)):
            assert torch.equal(denoiser.denoise(x, sigma, lam, uncond), plain)
        assert torch.equal(denoiser.plain_denoise(x, sigma), plain)
    engine.close()


@pytest.mark.bioemu2
def test_the_pair_adapter_is_trained_through_the_static_context(tiny_protein_ligand_system):
    """The pair embedding reaches the denoiser only through the static context, which keeps gradients."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(tiny_protein_ligand_system, hidden=(16,))
    assert network.pair_adapter is not None
    denoiser = network.engine().denoiser
    x = torch.zeros(2, denoiser.n_pad, 3)
    x[:, :tiny_protein_ligand_system.n_bioemu2_atoms] = torch.as_tensor(tiny_protein_ligand_system.reference,
                                                                       dtype=torch.float32)
    x = x + 2.0 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    out = denoiser.denoise(x, torch.tensor([2.0, 2.0]), torch.tensor([-1.0, 1.0]))
    out.square().sum().backward()
    for adapter in network.adapters():                  # the zero output layer gets a gradient
        assert adapter.out.weight.grad is not None and adapter.out.weight.grad.abs().sum() > 0
    network.engine().close()
