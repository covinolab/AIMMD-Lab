"""The fixed-step ODE of flow-matching networks and its generator ``'ode'``.

``aimmd.cgaimmd.flow_matching.sampler``: the integrator, :func:`generate` and
:class:`ODEGenerator`, the generator that C-GAIMMD picks by the name ``'ode'``
(``aimmd.cgaimmd.generation``).
"""

import json

import numpy as np
import pytest
import torch

from aimmd.cgaimmd import generation, protocol
from aimmd.cgaimmd.flow_matching import sampler
from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork


@pytest.mark.parametrize('method', sampler.SAMPLERS)
@pytest.mark.parametrize('n_steps', [1, 3, 16])
def test_integrate_linear_field_exact(method, n_steps):
    z = torch.randn(4, 2, 3)
    c = torch.randn(1, 2, 3)
    seen = []

    def constant(x, t):
        seen.append(t)
        return c.expand_as(x)

    out = sampler.integrate(constant, z, sampler=method, n_steps=n_steps)
    torch.testing.assert_close(out, z + c, atol=1e-6, rtol=1e-6)
    assert all(t.shape == (4,) and t.dtype == z.dtype for t in seen)
    assert len(seen) == {'euler': 1, 'midpoint': 2, 'rk4': 4}[method] * n_steps
    assert float(seen[0][0]) == 0.0
    # backwards (t0 > t1) undoes it
    back = sampler.integrate(constant, out, t0=1.0, t1=0.0, sampler=method, n_steps=n_steps)
    torch.testing.assert_close(back, z, atol=1e-6, rtol=1e-6)

    # time-linear field dx/dt = 2t: x(1) = x(0) + 1; midpoint and rk4 are exact
    def linear_in_t(x, t):
        return (2 * t)[:, None, None].expand_as(x)

    out = sampler.integrate(linear_in_t, z, sampler=method, n_steps=n_steps)
    expected = 1.0 if method != 'euler' else 1.0 - 1.0 / n_steps
    torch.testing.assert_close(out, z + expected, atol=1e-5, rtol=1e-5)

    # dx/dt = 3t^2: rk4 (Simpson) is exact
    if method == 'rk4':
        out = sampler.integrate(lambda x, t: (3 * t ** 2)[:, None, None].expand_as(x), z,
                                sampler=method, n_steps=n_steps)
        torch.testing.assert_close(out, z + 1.0, atol=1e-5, rtol=1e-5)


# the exact one-step amplification of each explicit method on dx/dt = a x (h = a dt)
AMPLIFICATION = {'euler': lambda h: 1 + h,
                 'midpoint': lambda h: 1 + h + h ** 2 / 2,
                 'rk4': lambda h: 1 + h + h ** 2 / 2 + h ** 3 / 6 + h ** 4 / 24}


@pytest.mark.parametrize('method', sampler.SAMPLERS)
@pytest.mark.parametrize('n_steps', [1, 2, 5])
@pytest.mark.parametrize('a', [0.7, -1.3])
def test_integrate_state_dependent_field(method, n_steps, a):
    """dx/dt = a x tests the x-updates between the stages (the fields above ignore x)."""
    z = torch.randn(3, 2, 3, dtype=torch.float64)
    out = sampler.integrate(lambda x, t: a * x, z, sampler=method, n_steps=n_steps)
    factor = AMPLIFICATION[method](a / n_steps) ** n_steps
    torch.testing.assert_close(out, z * factor, rtol=1e-12, atol=1e-12)
    # backwards from t = 1 to t = 0 applies the amplification of h = -a dt
    back = sampler.integrate(lambda x, t: a * x, z, t0=1.0, t1=0.0, sampler=method,
                             n_steps=n_steps)
    torch.testing.assert_close(back, z * AMPLIFICATION[method](-a / n_steps) ** n_steps,
                               rtol=1e-12, atol=1e-12)


def test_integrate_convergence_orders():
    """Doubling n_steps divides the error at t = 1 by 2, 4 and 16 (orders 1, 2, 4)."""
    import math
    z = torch.ones(1, 1, 1, dtype=torch.float64)
    exact = math.exp(1.0)
    for method, order in (('euler', 1), ('midpoint', 2), ('rk4', 4)):
        errors = [abs(float(sampler.integrate(lambda x, t: x, z, sampler=method,
                                              n_steps=n)[0, 0, 0]) - exact)
                  for n in (8, 16, 32)]
        for coarse, fine in zip(errors, errors[1:]):
            assert coarse / fine == pytest.approx(2 ** order, rel=0.15), (method, errors)


def test_integrate_rejects_bad_arguments():
    z = torch.zeros(2, 1, 3)
    with pytest.raises(ValueError, match='sampler'):
        sampler.integrate(lambda x, t: x, z, sampler='dopri5')
    for n_steps in (0, -1, 2.5):
        with pytest.raises(ValueError, match='n_steps'):
            sampler.integrate(lambda x, t: x, z, n_steps=n_steps)


class Recorder:
    """Proxy that records every attribute a caller touches on the network."""

    def __init__(self, network):
        object.__setattr__(self, '_network', network)
        object.__setattr__(self, 'touched', [])

    def __getattr__(self, name):
        self.touched.append(name)
        return getattr(self._network, name)


class RequiredOnly:
    """The protocol's callable members plus velocity and sample_prior (no n_atoms)."""

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.velocity_calls = 0

    def forward(self, inp):
        return inp[:, :1]

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        self.velocity_calls += 1
        return torch.ones_like(x_t) * (0.0 if lam is None else lam[:, None, None])

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        return torch.zeros(n, 2, 3)

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return None

    def parameters(self):
        return iter([self.weight])

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def test_generator_uses_only_flow_matching_members():
    proxy = Recorder(RequiredOnly())
    lam = torch.tensor([1.0, -2.0, 0.5])
    out = ODEGenerator(n_steps=4, chunk=2).generate(proxy, lam)
    # velocity lam per sample over t in [0, 1] from a zero prior
    torch.testing.assert_close(out, lam[:, None, None].expand(3, 2, 3))
    touched = set(proxy.touched)
    assert {'velocity', 'sample_prior'} <= touched
    allowed = set(protocol.REQUIRED_MEMBERS) | set(sampler.REQUIRED_MEMBERS)
    probed_optional = {'n_atoms', 'from_flow'}                # looked up, absent: fallbacks
    assert touched <= allowed | probed_optional, touched
    assert proxy._network.velocity_calls == 2 * 4 * 2        # 2 chunks, 4 midpoint steps


def test_a_network_generate_member_is_not_used():
    """The 'ode' method integrates the velocity field; it never calls another method's members."""

    class Generating(RequiredOnly):
        def generate(self, *args, **kwargs):
            raise AssertionError("the 'ode' generator must not call network.generate")

    out = ODEGenerator(n_steps=2).generate(Generating(), torch.zeros(3))
    assert out.shape == (3, 2, 3)


def test_chunking_reproducible():
    torch.manual_seed(1)
    net = TinyFlowNetwork(n_atoms=2, hidden=8)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    lam = torch.linspace(-3, 3, 7)
    a = sampler.generate(net, lam, chunk=3, generator=torch.Generator().manual_seed(5))
    b = sampler.generate(net, lam, chunk=3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert a.shape == (7, 2, 3) and a.dtype == torch.float32 and a.device.type == 'cpu'
    assert not a.requires_grad

    # with a deterministic prior, chunking does not change the result
    class Deterministic(TinyFlowNetwork):
        def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
            base = torch.zeros(n, self.n_atoms, 3)
            return base + (0.0 if lam is None else lam[:, None, None] / 10)

    det = Deterministic(n_atoms=2, hidden=8)
    det.load_state_dict(net.state_dict())
    one = sampler.generate(det, lam, chunk=100)
    many = sampler.generate(det, lam, chunk=2)
    torch.testing.assert_close(one, many)

    # null-token generation needs n; lam and n must agree
    assert sampler.generate(net, None, n=5, chunk=2).shape == (5, 2, 3)
    with pytest.raises(ValueError, match='n'):
        sampler.generate(net, None)
    with pytest.raises(ValueError, match='len'):
        sampler.generate(net, lam, n=3)
    assert sampler.generate(net, lam, n=7).shape == (7, 2, 3)
    assert sampler.generate(net, torch.zeros(0)).shape == (0, 2, 3)
    with pytest.raises(ValueError, match='chunk'):
        sampler.generate(net, lam, chunk=0)
    # lam may be a list or numpy array
    assert sampler.generate(net, [0.0, 1.0]).shape == (2, 2, 3)


def test_no_projection():
    """No center-of-mass projection: a translating field translates the output."""

    class Translating(RequiredOnly):
        def velocity(self, x_t, t, lam, *, uncond_mask=None):
            v = torch.zeros_like(x_t)
            v[..., 0] = 1.0                                   # every atom moves +1 in x
            return v

    out = sampler.generate(Translating(), torch.zeros(3), n_steps=3)
    expected = torch.zeros(3, 2, 3)
    expected[..., 0] = 1.0
    torch.testing.assert_close(out, expected)

    class Scaled(Translating):
        def from_flow(self, z):
            return z * 2.0

    torch.testing.assert_close(sampler.generate(Scaled(), torch.zeros(3), n_steps=3),
                               expected * 2.0)


# ---------------------------------------------------------------------------
# the generator 'ode'
# ---------------------------------------------------------------------------

class CommittorOnly(torch.nn.Module):
    """A C-GAIMMD network with a committor and nothing to generate with."""

    n_atoms = 2

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, inp):
        return self.linear(torch.flatten(torch.as_tensor(inp, dtype=torch.float32), 1))


def test_ode_generator_settings():
    generator = ODEGenerator()
    assert generator.method == 'ode'
    assert generator.settings() == {'sampler': 'midpoint', 'n_steps': 16, 'chunk': 128,
                                    'guidance_scale': 1.0}
    other = ODEGenerator(sampler='rk4', n_steps=np.int64(4), chunk=3, guidance_scale=np.int64(3))
    settings = other.settings()
    assert settings == {'sampler': 'rk4', 'n_steps': 4, 'chunk': 3, 'guidance_scale': 3.0}
    assert type(settings['n_steps']) is int and type(settings['guidance_scale']) is float
    assert json.loads(json.dumps(settings)) == settings
    # make_generator({'method': 'ode', **settings()}) builds an equal generator
    assert generation.make_generator({'method': 'ode', **settings}) == other
    assert generation.make_generator('ode') == generator
    assert generator != other
    assert 'rk4' in repr(other)
    for bad in ({'sampler': 'dopri5'}, {'n_steps': 0}, {'n_steps': 2.5}, {'chunk': 0},
                {'chunk': True}, {'guidance_scale': -1.0}, {'guidance_scale': float('nan')},
                {'guidance_scale': True}, {'guidance_scale': '3'}):
        with pytest.raises(ValueError):
            ODEGenerator(**bad)
    with pytest.raises(TypeError, match="'ode'.*bogus"):
        generation.make_generator({'method': 'ode', 'bogus': 1})
    assert generation.check_generator(generator, TinyFlowNetwork()) == []


def test_ode_generator_registered_on_import():
    assert generation._FACTORIES.get('ode') is ODEGenerator
    generation.unregister_generator('ode')
    assert 'ode' in generation.available_generators()          # still known: found on use
    assert isinstance(generation.make_generator('ode'), ODEGenerator)
    assert generation._FACTORIES.get('ode') is ODEGenerator


def test_ode_check_names_what_the_network_lacks():
    generator = ODEGenerator()
    assert generator.check(TinyFlowNetwork()) == []
    assert generator.check(RequiredOnly()) == []
    # a network with only a committor: both flow-matching members, by their signature
    protocol.check_network_protocol(CommittorOnly())
    assert generator.check(CommittorOnly()) == [
        'needs velocity(x_t, t, lam), which the network lacks',
        'needs sample_prior(lam, n, *, generator=None), which the network lacks']
    # one member missing: only that one is named
    only_velocity = type('OnlyVelocity', (CommittorOnly,), {
        'velocity': lambda self, x_t, t, lam, *, uncond_mask=None: x_t})()
    assert generator.check(only_velocity) == [
        'needs sample_prior(lam, n, *, generator=None), which the network lacks']
    # members with the wrong signature, or set but not callable
    bad = type('Bad', (RequiredOnly,), {
        'velocity': lambda self, x_t, t: x_t,
        'sample_prior': lambda self, lam, n: None,
        'from_flow': 3.0})()
    problems = generator.check(bad)
    assert len(problems) == 3
    assert problems[0].startswith('velocity must accept velocity(x_t, t, lam)')
    assert problems[1].startswith('sample_prior must accept sample_prior(lam, n, *, '
                                  'generator=None)')
    assert problems[2] == 'from_flow is set but not callable (expected from_flow(z))'
    # **kwargs satisfies any keyword
    loose = type('Loose', (RequiredOnly,), {
        'velocity': lambda self, x_t, t, lam, **kwargs: x_t,
        'sample_prior': lambda self, *args, **kwargs: None})()
    assert generator.check(loose) == []
    # check_generator prefixes the method
    assert generation.check_generator(generator, CommittorOnly()) == [
        "generator 'ode': " + message for message in generator.check(CommittorOnly())]


def test_generate_names_a_member_of_the_wrong_shape():
    lam = torch.zeros(2)
    cases = {
        'sample_prior': type('P', (RequiredOnly,), {
            'sample_prior': lambda self, lam, n, *, generator=None: torch.zeros(n, 3, 3),
            'n_atoms': 2}),
        'velocity': type('V', (RequiredOnly,), {
            'velocity': lambda self, x_t, t, lam, *, uncond_mask=None: x_t[:, :1]}),
        'from_flow': type('F', (RequiredOnly,), {'from_flow': lambda self, z: z[:, :1]}),
    }
    for member, network in cases.items():
        with pytest.raises(ValueError, match=rf'{member}\(.*returned shape'):
            ODEGenerator(n_steps=1).generate(network(), lam)


def test_ode_generator_through_the_protocol_numerics():
    net = TinyFlowNetwork(n_atoms=2, hidden=8)
    coords = np.random.default_rng(0).normal(size=(3, 2, 3)).astype(np.float32)
    report = protocol.check_network_numerics(net, coords, coords.reshape(3, 6),
                                             generation=ODEGenerator(n_steps=2))
    assert report['generation'] == {'method': 'ode', 'settings': {
        'sampler': 'midpoint', 'n_steps': 2, 'chunk': 128, 'guidance_scale': 1.0}}
    assert report['sample_shape'] == [2, 2, 3]


# ---------------------------------------------------------------------------
# classifier-free guidance
# ---------------------------------------------------------------------------

class LinearField(RequiredOnly):
    """A velocity field that a hand computation can follow: ``v = a x + b lam`` (``b lam`` -> ``c`` on null rows).

    The conditional and the unconditional field differ, the field depends on
    ``x``, and the null token is not ``lam = 0``.
    """

    a, b, c = -0.5, 0.4, 0.3
    n_atoms = 2

    def __init__(self):
        super().__init__()
        self.calls = []

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        self.calls.append(None if uncond_mask is None else uncond_mask.clone())
        drive = (torch.full((len(x_t),), self.c) if lam is None
                 else self.b * torch.as_tensor(lam, dtype=x_t.dtype))
        if lam is not None and uncond_mask is not None:
            drive = torch.where(uncond_mask, torch.full_like(drive, self.c), drive)
        return self.a * x_t + drive[:, None, None]

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        return torch.randn(n, 2, 3, generator=generator)


def test_guidance_scale_one_is_no_guidance_bit_for_bit():
    """guidance_scale=1 evaluates the conditional velocity once per stage, as before: identical bits."""
    torch.manual_seed(3)
    net = TinyFlowNetwork(n_atoms=2, hidden=8)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(0.5 * torch.randn_like(p))
    lam = torch.linspace(-4, 4, 5)
    for method in sampler.SAMPLERS:
        plain = sampler.generate(net, lam, sampler=method, n_steps=3,
                                 generator=torch.Generator().manual_seed(9))
        one = ODEGenerator(sampler=method, n_steps=3, guidance_scale=1.0).generate(
            net, lam, generator=torch.Generator().manual_seed(9))
        assert torch.equal(plain, one)
    # no unconditional evaluation at all
    field = LinearField()
    ODEGenerator(n_steps=2, guidance_scale=1).generate(field, lam,
                                                       generator=torch.Generator().manual_seed(0))
    assert field.calls and all(mask is None for mask in field.calls)


def _hand_guided_euler(x0, lam, w, a, b, c, n_steps):
    """Euler steps of v_u + w (v_c - v_u) for v_c = a x + b lam, v_u = a x + c, by hand (float64)."""
    x = x0.to(torch.float64)
    lam = lam.to(torch.float64)[:, None, None]
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        v_c = a * x + b * lam
        v_u = a * x + c
        x = x + dt * (w * v_c + (1.0 - w) * v_u)
    return x


@pytest.mark.parametrize('w', [0.0, 2.0, 3.0])
def test_guidance_matches_a_hand_computation(w):
    """On a linear toy field the guided ODE is w v_cond + (1 - w) v_uncond, integrated by hand."""
    field = LinearField()
    lam = torch.tensor([-3.0, 0.0, 1.5, 4.0])
    out = ODEGenerator(sampler='euler', n_steps=5, guidance_scale=w).generate(
        field, lam, generator=torch.Generator().manual_seed(4))
    x0 = torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(4))   # the same prior draw
    expected = _hand_guided_euler(x0, lam, w, field.a, field.b, field.c, 5)
    torch.testing.assert_close(out.to(torch.float64), expected, rtol=1e-6, atol=1e-6)
    # two velocity evaluations per stage, the second on null rows only
    masks = [m for m in field.calls if m is not None]
    assert len(field.calls) == 2 * 5 and len(masks) == 5
    assert all(bool(m.all()) and len(m) == 4 for m in masks)
    # closed form of one Euler step, w v_c + (1 - w) v_u = a x + w b lam + (1 - w) c
    one = ODEGenerator(sampler='euler', n_steps=1, guidance_scale=w).generate(
        LinearField(), lam, generator=torch.Generator().manual_seed(4))
    closed = x0 + (field.a * x0 + (w * field.b * lam + (1 - w) * field.c)[:, None, None])
    torch.testing.assert_close(one, closed, rtol=1e-6, atol=1e-6)


def test_guidance_with_the_null_token_is_plain():
    """lam=None generates with the null token: nothing to guide, one evaluation per stage."""
    field = LinearField()
    guided = sampler.generate(field, None, n=3, n_steps=2, guidance_scale=3.0,
                              generator=torch.Generator().manual_seed(1))
    plain = sampler.generate(field, None, n=3, n_steps=2,
                             generator=torch.Generator().manual_seed(1))
    assert torch.equal(guided, plain)
    assert len(field.calls) == 2 * 2 * 2 and all(mask is None for mask in field.calls)


def test_guidance_draws_the_same_prior():
    """The prior does not depend on the guidance weight: the same generator, the same prior draw."""
    seen = []

    class Recording(LinearField):
        def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
            z = super().sample_prior(lam, n, generator=generator)
            seen.append(z.clone())
            return z

    for w in (1.0, 3.0):
        ODEGenerator(n_steps=2, guidance_scale=w).generate(
            Recording(), torch.zeros(3), generator=torch.Generator().manual_seed(2))
    assert torch.equal(seen[0], seen[1])


def test_guidance_check_requires_uncond_mask_only_when_on():
    """check() asks for velocity's keyword uncond_mask only when guidance is on."""

    class NoMask(RequiredOnly):
        def velocity(self, x_t, t, lam):
            return torch.zeros_like(x_t)

        def sample_prior(self, lam, n, *, generator=None):
            return torch.zeros(n, 2, 3)

    assert ODEGenerator().check(NoMask()) == []
    assert ODEGenerator(guidance_scale=1.0).check(NoMask()) == []
    out = ODEGenerator(n_steps=2).generate(NoMask(), torch.zeros(2))
    assert out.shape == (2, 2, 3)
    problems = ODEGenerator(guidance_scale=3.0).check(NoMask())
    assert len(problems) == 1
    assert problems[0].startswith('velocity must accept velocity(x_t, t, lam, *, '
                                  'uncond_mask=None)')
    assert generation.check_generator(ODEGenerator(guidance_scale=2.0), TinyFlowNetwork()) == []
    assert sampler.flow_matching_problems(NoMask()) == []
    assert sampler.flow_matching_problems(NoMask(), uncond_mask=True) == problems


def test_guidance_through_the_params_spec():
    """{'method': 'ode', 'guidance_scale': w} builds a guided generator; its settings record w."""
    built = generation.make_generator({'method': 'ode', 'guidance_scale': 3.0, 'n_steps': 2},
                                      TinyFlowNetwork())
    assert isinstance(built, ODEGenerator) and built.guidance_scale == 3.0
    assert built.settings()['guidance_scale'] == 3.0
    assert 'guidance_scale=3.0' in repr(built)
    assert generation.make_generator({'method': 'ode', **built.settings()}) == built
