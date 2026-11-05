"""The committor head of the BioEmu2 network (aimmd.cgaimmd.bioemu2.head): torch only, no bioemu2 needed.

The scaler stays through a reset, a seeded reset is the old ``fresh_head``,
the scaler equals numpy's moments, and the two losses have their closed forms.
"""

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.bioemu2.head import (LAMBDA_CLIP, SIGMA_REF, STD_FLOOR, CommittorHead, FeatureMLP, fit_scaler,
                                        in_state_nll, shooting_point_nll)


def test_constants_are_the_production_ones():
    assert (SIGMA_REF, LAMBDA_CLIP, STD_FLOOR) == (0.05, 10.0, 1e-3)


def test_head_shapes_scaler_and_floor():
    head = CommittorHead(6, hidden=(8, 4), mean=np.arange(6.0), std=[2.0, 0.0, 1.0, 1.0, 1.0, 1e-6],
                         log_sigma_mean=-1.0, log_sigma_std=0.5)
    assert head.n_features == 6 and head.hidden == [8, 4]
    assert torch.equal(head.std, torch.tensor([2.0, STD_FLOOR, 1.0, 1.0, 1.0, STD_FLOOR]))
    out = head(torch.randn(5, 6), torch.full((5,), SIGMA_REF))
    assert out.shape == (5,) and out.dtype == torch.float32
    # half-precision features are accepted (the fit stores them so)
    x = torch.randn(3, 6)
    sigma = torch.tensor([0.05, 1.0, 10.0])
    assert torch.allclose(head(x.half(), sigma), head(x.half().float(), sigma))
    # the input is [z-scored features, z-scored log sigma]
    z = torch.cat([(x - head.mean) / head.std, ((torch.log(sigma) + 1.0) / 0.5)[:, None]], 1)
    assert torch.allclose(head(x, sigma), head.mlp(z).reshape(-1))
    scaler = head.scaler()
    other = CommittorHead(6, hidden=(8, 4))
    other.set_scaler(scaler)
    for name in ('mean', 'std', 'ls_mean', 'ls_std'):
        assert torch.equal(getattr(other, name), getattr(head, name))


def test_reset_keeps_the_scaler_and_a_seed_reproduces_the_old_fresh_head():
    head = CommittorHead(4, hidden=(5,), mean=np.ones(4), std=np.full(4, 3.0))
    with torch.no_grad():
        for p in head.mlp.parameters():
            p.add_(1.0)
    state = torch.random.get_rng_state()
    head.reset_parameters(seed=7)
    assert torch.equal(torch.random.get_rng_state(), state)        # the global stream is left as it was
    with torch.random.fork_rng(devices=[]):                        # the old fresh_head
        torch.manual_seed(7)
        old = FeatureMLP(5, [5]).state_dict()
    for key, value in old.items():
        assert torch.equal(head.mlp.state_dict()[key], value)
    assert torch.equal(head.mean, torch.ones(4)) and torch.equal(head.std, torch.full((4,), 3.0))
    head.reset_parameters()                                        # unseeded: nn.Linear's own initialisation
    assert not torch.equal(head.mlp.out.weight, old['out.weight'])


def test_fit_scaler_is_the_column_moments():
    rng = np.random.default_rng(0)
    features = rng.normal(3.0, 2.0, size=(5000, 3)).astype(np.float16)
    sigmas = np.exp(rng.uniform(-3, 3, 5000))
    scaler = fit_scaler(features, sigmas)
    f = features.astype(np.float64)
    np.testing.assert_allclose(scaler['mean'], f.mean(0), rtol=1e-6)
    np.testing.assert_allclose(scaler['std'], f.std(0), rtol=1e-5)
    assert scaler['mean'].dtype == np.float32
    assert scaler['log_sigma_mean'] == pytest.approx(np.log(sigmas).mean())
    assert scaler['log_sigma_std'] == pytest.approx(np.log(sigmas).std())
    with pytest.raises(ValueError, match='at least one feature row'):
        fit_scaler(np.zeros((0, 3)), [])


def test_the_losses():
    lam = torch.tensor([0.0, 2.0, -1.0])
    n_first, n_last = torch.tensor([1.0, 0.0, 2.0]), torch.tensor([1.0, 2.0, 0.0])
    p = torch.sigmoid(lam)
    expected = -(n_last * torch.log(p) + n_first * torch.log(1 - p)) / (n_first + n_last)
    assert torch.allclose(shooting_point_nll(lam, n_first, n_last), expected.mean())
    high = torch.sigmoid(torch.tensor(LAMBDA_CLIP))
    state = torch.tensor([1, 2, 2])
    target = torch.where(state == 2, high, 1 - high)
    expected = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    assert torch.allclose(in_state_nll(lam, state), expected.mean())
