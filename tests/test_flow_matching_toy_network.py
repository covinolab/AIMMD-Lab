"""``aimmd.cgaimmd.flow_matching.toy.TinyFlowNetwork``, the minimal flow-matching network."""

import numpy as np
import pytest
import torch

from aimmd.cgaimmd import generation, protocol
from aimmd.cgaimmd.flow_matching import sampler
from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork


def test_minimal_protocol_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA touched at construction')

    for name in ('is_available', 'device_count', 'init', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    net = TinyFlowNetwork()
    monkeypatch.undo()

    for member in protocol.REQUIRED_MEMBERS + sampler.REQUIRED_MEMBERS:
        assert callable(getattr(net, member))
    assert net.n_atoms == 1 and type(net.n_atoms) is int
    # optional members: only default_generator (and lambda_clip when given),
    # so the end-to-end tests exercise every other fallback
    for member in protocol.OPTIONAL_MEMBERS + sampler.OPTIONAL_MEMBERS:
        if member == 'default_generator':
            assert net.default_generator() == {'method': 'ode'}
        else:
            assert not hasattr(net, member), member
    assert protocol.is_cgaimmd_network(net)
    assert generation.make_generator(None, net) == ODEGenerator()
    assert TinyFlowNetwork(lambda_clip=5).lambda_clip == 5.0
    assert type(TinyFlowNetwork(lambda_clip=5).lambda_clip) is float
    protocol.check_network_protocol(net)

    # a local subclass with the default constructor, as a params file defines it
    class Network(TinyFlowNetwork):
        pass

    local = Network()
    protocol.check_network_protocol(local)
    assert protocol.n_atoms_of(local) == 1

    wide = TinyFlowNetwork(n_atoms=3, hidden=8, prior_center=0.0, prior_width=1.0)
    x = torch.randn(5, 3, 3)
    assert wide(x.reshape(5, 9)).shape == (5, 1)
    assert wide.velocity(x, torch.rand(5), torch.randn(5)).shape == (5, 3, 3)
    assert wide.sample_prior(None, 4).shape == (4, 3, 3)
    with pytest.raises(ValueError):
        TinyFlowNetwork(n_atoms=0)
    with pytest.raises(ValueError):
        TinyFlowNetwork(lambda_clip=0.0)


def test_zero_initialized_outputs():
    net = TinyFlowNetwork()
    x = torch.randn(64, 3, requires_grad=True)
    lam = net(x)
    assert lam.shape == (64, 1)
    assert torch.all(lam == 0)
    lam.sum().backward()
    assert torch.all(x.grad == 0)                        # dλ/dx vanishes at init, too

    z = torch.randn(8, 1, 3)
    t = torch.rand(8)
    for lam_arg in (None, torch.randn(8)):
        assert torch.all(net.velocity(z, t, lam_arg) == 0)

    # an untrained network returns its prior samples
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    samples = ODEGenerator(sampler='midpoint', n_steps=4, chunk=4000).generate(
        net, torch.zeros(4000), generator=g1)
    prior = net.sample_prior(torch.zeros(4000), 4000, generator=g2)
    assert torch.equal(samples, prior)
    assert abs(float(samples.mean()) - 1.5) < 0.02
    assert abs(float(samples.std()) - 0.3) < 0.02

    # the committor looks at x[:, 0] only (the first coordinate of the first atom)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p))
    x = torch.randn(10, 3)
    y = x.clone()
    y[:, 1:] = torch.randn(10, 2)
    torch.testing.assert_close(net(x), net(y))
    assert not torch.allclose(net(x), net(x + torch.tensor([0.5, 0.0, 0.0])))

    # the null token is not lambda = 0; uncond_mask rows are null rows
    z = torch.randn(6, 1, 3)
    t = torch.rand(6)
    v_null = net.velocity(z, t, None)
    v_zero = net.velocity(z, t, torch.zeros(6))
    assert not torch.allclose(v_null, v_zero)
    mask = torch.tensor([True, False, True, False, False, True])
    v_mask = net.velocity(z, t, torch.zeros(6), uncond_mask=mask)
    torch.testing.assert_close(v_mask[mask], v_null[mask])
    torch.testing.assert_close(v_mask[~mask], v_zero[~mask])


def test_trainable():
    """Both heads learn (zero-initialized output layers still get gradients)."""
    torch.manual_seed(0)
    net = TinyFlowNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(32, 3)
    target = (x[:, :1] > 0).float()
    z = torch.randn(32, 1, 3)
    t = torch.rand(32)
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), target)
        loss = loss + ((net.velocity(z, t, torch.zeros(32)) - 1.0) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.all(net(x) == 0)
    assert not torch.all(net.velocity(z, t, torch.zeros(32)) == 0)
    # the network works in float32 on the CPU
    assert all(p.dtype == torch.float32 for p in net.parameters())
    assert np.isfinite(net(x).detach().numpy()).all()
