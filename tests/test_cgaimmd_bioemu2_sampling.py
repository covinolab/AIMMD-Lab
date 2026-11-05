"""The generation method 'bioemu2_edm' (aimmd.cgaimmd.bioemu2.sampling) on the tiny model.

- the generator contract through the registry (``make_generator``,
  ``check_generator``, the network's default, the start-up numerics check);
- the old sampler's samples (``guided_sampler.sample`` driven as the old networks
  drove it, guidance weight 1 and 2, and the noise-level guidance schedule with
  a learned null of the generator-adherence campaign) are reproduced by
  BioEmu2's public ``EDMSampler`` behind :class:`GuidedModel`: the random
  streams are aligned, and what is left is float32 rounding of a reordered
  Euler update (below 1e-4 Å on coordinates of 25 Å after 5 steps);
- generation is reproducible from the caller's ``torch.Generator`` and leaves
  the global random stream alone; samples are completed to the generator atoms.
"""

import json

import numpy as np
import pytest
import torch

from ._bioemu2_fixtures import (golden, golden_state, tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_peptide_system, tiny_protein_ligand_system)
from ._cgaimmd_bioemu2_golden import GEN_BATCH, GUIDANCE_SCHEDULE, SAMPLER_LAMBDAS, SAMPLER_SEED, SAMPLER_STEPS

pytestmark = pytest.mark.bioemu2

SAMPLE_ATOL = 2e-4
"""Ångström: the old sampler's samples against the public sampler's (both float32; measured 4.5e-5)."""


def _system(request, which):
    return request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')


def _network(system, golden=None, prefix=None, **kwargs):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(system, hidden=(16,), **kwargs)
    if golden is not None:
        network.single_adapter.load_state_dict(golden_state(golden, f'{prefix}.single_adapter'))
        if network.pair_adapter is not None:
            network.pair_adapter.load_state_dict(golden_state(golden, f'{prefix}.pair_adapter'))
    return network


def test_the_generator_contract_through_the_registry(tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from aimmd.cgaimmd.generation import (available_generators, canonical_generator_spec, check_generator,
                                          is_generator, make_generator)
    network = _network(tiny_peptide_system)
    assert 'bioemu2_edm' in available_generators()
    default = make_generator(None, network)                          # the network's default_generator()
    assert isinstance(default, BioEmu2Generator) and is_generator(default)
    assert default.method == 'bioemu2_edm'
    assert default.settings() == {'n_steps': 200, 'cfg_scale': 1.0, 'cfg_schedule': None, 'chunk': 16}
    assert check_generator(default, network) == []
    spec = {'method': 'bioemu2_edm', 'n_steps': 30, 'cfg_scale': 2.0, 'cfg_schedule': '8-inf:3:learned',
            'chunk': 4}
    built = make_generator(spec, network)
    assert built.settings() == {key: value for key, value in spec.items() if key != 'method'}
    assert make_generator({'method': 'bioemu2_edm', **built.settings()}) == built
    json.dumps(built.settings())
    assert make_generator({'chunk': 64}, network).settings()['chunk'] == 64      # the old generate_chunk
    assert canonical_generator_spec(spec) == spec
    problems = check_generator(default, TinyFlowNetwork())
    assert problems == ["generator 'bioemu2_edm': 'bioemu2_edm' samples with the trunk adapters of a "
                        "BioEmu2Network (aimmd.cgaimmd.bioemu2.BioEmu2Network); got TinyFlowNetwork"]
    for settings, message in (({'n_steps': 1}, 'n_steps must be >= 2'), ({'chunk': 0}, 'chunk must be an int'),
                              ({'cfg_schedule': '1-2:3'}, "is not 'lo-hi:weight:null'"),
                              ({'cfg_scale': 'two'}, 'cfg_scale must be a number'),
                              ({'sampler': 'euler'}, 'sampler')):
        with pytest.raises((TypeError, ValueError), match=message):
            make_generator({'method': 'bioemu2_edm', **settings})


@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_goldens_of_the_old_sampler(which, request, golden):
    """guided_sampler.sample as the old networks drove it: chunks of GEN_BATCH, seed + chunk start."""
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    network = _network(_system(request, which), golden, f'{which}.denoiser.state')
    for weight in (1.0, 2.0):
        generator = BioEmu2Generator(n_steps=SAMPLER_STEPS, cfg_scale=weight, chunk=GEN_BATCH)
        samples = generator.sample_bioemu2(network, torch.tensor(SAMPLER_LAMBDAS), seed=SAMPLER_SEED)
        np.testing.assert_allclose(samples.numpy(), golden[f'{which}.sample.cfg{weight:g}'], rtol=0,
                                   atol=SAMPLE_ATOL)
    network.engine().close()


def test_golden_of_the_guidance_schedule_with_a_learned_null(tiny_peptide_system, golden):
    """adherence_lib.guidance_hook: windows against the learned null and against plain BioEmu2."""
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    network = _network(tiny_peptide_system, golden, 'peptide.sample_schedule', learned_null=True)
    generator = BioEmu2Generator(n_steps=SAMPLER_STEPS, cfg_schedule=GUIDANCE_SCHEDULE, chunk=GEN_BATCH)
    samples = generator.sample_bioemu2(network, torch.tensor(SAMPLER_LAMBDAS), seed=SAMPLER_SEED)
    np.testing.assert_allclose(samples.numpy(), golden['peptide.sample_schedule.coords'], rtol=0, atol=SAMPLE_ATOL)
    # the schedule matters: without it the samples are others
    plain = BioEmu2Generator(n_steps=SAMPLER_STEPS, chunk=GEN_BATCH).sample_bioemu2(
        network, torch.tensor(SAMPLER_LAMBDAS), seed=SAMPLER_SEED)
    assert np.abs(plain.numpy() - golden['peptide.sample_schedule.coords']).max() > 1e-2
    network.engine().close()


@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_generate_is_reproducible_completed_and_leaves_the_global_stream(which, request, golden):
    from aimmd.cgaimmd.generation import make_generator
    system = _system(request, which)
    network = _network(system, golden, f'{which}.denoiser.state')
    generator = make_generator({'method': 'bioemu2_edm', 'n_steps': 3, 'chunk': 2}, network)
    lam = torch.tensor([-3.0, 0.0, 2.0])
    state = torch.random.get_rng_state()
    first = generator.generate(network, lam, generator=torch.Generator().manual_seed(5))
    assert torch.equal(torch.random.get_rng_state(), state)
    again = generator.generate(network, lam, generator=torch.Generator().manual_seed(5))
    other = generator.generate(network, lam, generator=torch.Generator().manual_seed(6))
    assert first.shape == (3, system.n_atoms, 3) and first.dtype == torch.float32 and not first.requires_grad
    assert torch.isfinite(first).all() and torch.equal(first, again) and not torch.equal(first, other)
    # the BioEmu2 atoms are the sampler's, the others completed from the template
    seed = int(torch.randint(2 ** 31 - 1, (1,), generator=torch.Generator().manual_seed(5)))
    samples = generator.sample_bioemu2(network, lam, seed=seed).numpy()
    np.testing.assert_allclose(first.numpy()[:, system.bioemu2_to_md], samples, rtol=0, atol=1e-5)
    np.testing.assert_allclose(first.numpy(), system.complete(samples), rtol=0, atol=1e-4)
    network.engine().close()


def test_a_non_finite_sample_comes_back_as_nan(tiny_peptide_system, monkeypatch):
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    network = _network(tiny_peptide_system)
    generator = BioEmu2Generator(n_steps=2)
    samples = torch.as_tensor(np.stack([tiny_peptide_system.reference] * 2), dtype=torch.float32)
    samples[1, 3, 0] = float('nan')
    monkeypatch.setattr(BioEmu2Generator, 'sample_bioemu2', lambda self, net, lam, seed: samples)
    out = generator.generate(network, torch.tensor([0.0, 1.0]))
    assert torch.isfinite(out[0]).all() and torch.isnan(out[1]).all()


def test_the_start_up_check_with_one_generation_step(tiny_peptide_system):
    """What a shooting worker runs before it starts: protocol, generator check, one tiny generation."""
    from aimmd.cgaimmd.protocol import check_network_numerics, check_network_protocol
    network = _network(tiny_peptide_system)
    check_network_protocol(network)
    coordinates = torch.as_tensor(tiny_peptide_system.template[None].repeat(2, 0), dtype=torch.float32)
    report = check_network_numerics(network, coordinates, coordinates.reshape(2, -1),
                                    generation={'method': 'bioemu2_edm', 'n_steps': 2, 'chunk': 2})
    assert report['generation'] == {'method': 'bioemu2_edm', 'settings': {'n_steps': 2, 'cfg_scale': 1.0,
                                                                          'cfg_schedule': None, 'chunk': 2}}
    assert list(report['sample_shape']) == [2, 166, 3]
    network.engine().close()
