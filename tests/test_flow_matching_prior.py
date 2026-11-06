"""The prior of flow-matching networks: ``aimmd.cgaimmd.flow_matching.prior.GaussianPrior``.

The center-of-mass-free isotropic Gaussian is the only prior of C-GAIMMD's
PaiNN network (the flow branch's mixture and harmonic priors are not ported).
"""

import pytest
import torch

from aimmd.cgaimmd.flow_matching import flowmath
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.flow_matching.prior import PRIORS, GaussianPrior


TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=False,
            edge_backend='torch')
TOKENS = [6, 1, 1, 8, 6, 1, 7]
M = len(TOKENS)


def test_samples_are_center_of_mass_free_standard_normal():
    """Shape, dtype, zero mean over the atoms, per-coordinate variance (M - 1) / M."""
    prior = GaussianPrior(6)
    z = prior.sample(4000, generator=torch.Generator().manual_seed(0))
    assert z.shape == (4000, 6, 3) and z.dtype == torch.float32
    assert float(z.mean(dim=1).abs().max()) < 1e-5
    assert abs(float(z.var()) - 5.0 / 6.0) < 0.02
    # it is the projected standard normal of flowmath, draw for draw
    assert torch.equal(z, flowmath.gaussian_prior((4000, 6, 3),
                                                  generator=torch.Generator().manual_seed(0)))
    assert prior.sample(0).shape == (0, 6, 3)


def test_draws_only_from_the_given_stream():
    """Reproducible from a generator; the global random stream is left alone."""
    prior = GaussianPrior(5)
    state = torch.get_rng_state()
    a = prior.sample(7, generator=torch.Generator().manual_seed(3))
    b = prior.sample(7, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), state)
    assert not torch.equal(a, prior.sample(7, generator=torch.Generator().manual_seed(4)))


def test_mirror_images_are_as_likely_as_the_originals():
    """The prior is parity symmetric: the handedness of four atoms is + and - equally often.

    This is why an E(3) velocity field generates both enantiomers and the SE(3)
    network (parity breaking) or reflected validity checks are needed.
    """
    z = GaussianPrior(4).sample(20000, generator=torch.Generator().manual_seed(1))
    handedness = torch.sign(torch.det(z[:, 1:] - z[:, :1]))
    assert abs(float((handedness > 0).float().mean()) - 0.5) < 0.015


def test_settings_and_validation():
    prior = GaussianPrior(3)
    assert PRIORS == ('gaussian',) and prior.kind == 'gaussian'
    assert prior.settings() == {'kind': 'gaussian', 'n_atoms': 3}
    assert prior.n_atoms == 3 and repr(prior) == 'GaussianPrior(n_atoms=3)'
    assert prior == GaussianPrior(3) and prior != GaussianPrior(4)
    assert len({prior, GaussianPrior(3)}) == 1
    for bad in (0, -1, 2.0, True, None):
        with pytest.raises(ValueError, match='n_atoms'):
            GaussianPrior(bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match='prior samples'):
            prior.sample(bad)


def test_the_network_samples_its_gaussian_prior():
    """JointPaiNN.sample_prior is the Gaussian prior's draw; lam and uncond_mask do not change it."""
    torch.manual_seed(0)
    net = JointPaiNN(M, TOKENS, **TINY)
    assert net.prior == GaussianPrior(M)
    assert not any(key.startswith('prior') for key in net.state_dict())     # nothing to save
    lam = torch.tensor([-5.0, 0.0, 3.0])
    z = net.sample_prior(lam, 3, generator=torch.Generator().manual_seed(2))
    assert z.shape == (3, M, 3) and z.device == net.device
    assert torch.equal(z, net.prior.sample(3, generator=torch.Generator().manual_seed(2)))
    for other_lam, mask in ((None, None), (-lam, None), (lam, torch.tensor([True, False, True]))):
        again = net.sample_prior(other_lam, 3, uncond_mask=mask,
                                 generator=torch.Generator().manual_seed(2))
        assert torch.equal(again, z)
    # in Angstrom the prior is coord_scale times wider than in flow units
    x = net.from_flow(z)
    assert torch.allclose(x, flowmath.center(z) * TINY['coord_scale'])
