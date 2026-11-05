"""The duck-typed C-GAIMMD network protocol, ``aimmd.cgaimmd.protocol``.

The protocol covers the committor only; what a generation method needs from
the network is checked by the generator (``tests/test_flow_matching_sampler.py``
for ``'ode'``, ``tests/test_cgaimmd_generation.py`` for the interface).
"""

import math

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd import protocol
from aimmd.cgaimmd.flow_matching import sampler
from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork

ODE = ODEGenerator(n_steps=2)


class CommittorOnly:
    """A C-GAIMMD network without torch.nn.Module: the protocol members and nothing else.

    Committor: lambda = x0 - 1 (first coordinate); two atoms.
    """

    n_atoms = 2

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.training = True

    def forward(self, inp):
        return (inp[:, :1] - 1.0) * self.weight

    def state_dict(self):
        return {'weight': self.weight.detach().clone()}

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.weight.copy_(sd['weight'])

    def parameters(self):
        return iter([self.weight])

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


class DuckNetwork:
    """A flow-matching network without torch.nn.Module and without n_atoms.

    Committor: lambda = x0 - 1 (first coordinate); velocity: constant (1, 0, 0)
    per atom; prior: N(0, 1) of shape (n, M, 3), which gives n_atoms.
    """

    def __init__(self, n_atoms=2):
        self._m = n_atoms
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.training = True

    def forward(self, inp):
        return (inp[:, :1] - 1.0) * self.weight

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        out = torch.zeros_like(x_t)
        out[..., 0] = 1.0
        return out

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        return torch.randn(n, self._m, 3, generator=generator)

    def state_dict(self):
        return {'weight': self.weight.detach().clone()}

    def load_state_dict(self, sd):
        with torch.no_grad():
            self.weight.copy_(sd['weight'])

    def parameters(self):
        return iter([self.weight])

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


def _without(name):
    """A DuckNetwork subclass whose `name` member is missing (or not callable)."""
    return type(f'DuckWithout_{name}', (DuckNetwork,), {name: None})()


def _coords(net, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, protocol.n_atoms_of(net), 3)).astype(np.float32)


def _dense_input(coords):
    return torch.flatten(torch.as_tensor(coords, dtype=torch.float32), start_dim=1)


def test_tiny_network_passes():
    for net in (TinyFlowNetwork(), TinyFlowNetwork(n_atoms=3, lambda_clip=5.0), DuckNetwork()):
        protocol.check_network_protocol(net)
        coords = _coords(net)
        report = protocol.check_network_numerics(net, coords, _dense_input(coords),
                                                 generation=ODE)
        assert report['n_atoms'] == protocol.n_atoms_of(net)
        assert report['n_frames'] == 3
        assert report['device'] == 'cpu'
        assert report['generation']['method'] == 'ode'
    assert protocol.is_cgaimmd_network(TinyFlowNetwork())
    assert not protocol.is_cgaimmd_network(DuckNetwork())            # no default_generator
    doc = protocol.check_network_protocol.__doc__
    for member in protocol.REQUIRED_MEMBERS + protocol.OPTIONAL_MEMBERS + ('n_atoms',):
        assert member in doc


def test_a_network_with_only_a_committor_passes():
    """No generation member is part of the protocol (a network never fakes another method's)."""
    net = CommittorOnly()
    for member in ('velocity', 'sample_prior', 'generate', 'default_generator'):
        assert not hasattr(net, member)
    protocol.check_network_protocol(net)
    assert protocol.n_atoms_of(net) == 2
    coords = _coords(net)
    report = protocol.check_network_numerics(net, coords, _dense_input(coords))
    assert report['generation'] is None and report['sample_shape'] is None
    assert report['forward_range'] is not None
    # not a C-GAIMMD network by itself: it does not name a generation method
    assert not protocol.is_cgaimmd_network(net)
    named = type('Named', (CommittorOnly,), {'default_generator': lambda self: 'ode'})()
    assert protocol.is_cgaimmd_network(named)
    # the same as a torch module
    module = type('Module', (torch.nn.Module,), {
        'n_atoms': 2, 'forward': lambda self, inp: inp[:, :1]})()
    protocol.check_network_protocol(module)


def test_n_atoms_is_required_unless_a_prior_gives_it():
    no_atoms = type('NoAtoms', (CommittorOnly,), {'n_atoms': None})()
    with pytest.raises(TypeError, match='required attribute n_atoms is missing'):
        protocol.check_network_protocol(no_atoms)
    with pytest.raises(TypeError, match='has no n_atoms'):
        protocol.n_atoms_of(no_atoms)
    protocol.check_network_protocol(DuckNetwork())                   # the prior gives it
    assert protocol.n_atoms_of(DuckNetwork(n_atoms=3)) == 3


@pytest.mark.parametrize('member', protocol.REQUIRED_MEMBERS)
def test_missing_required_member_reported(member):
    assert member in ('forward', 'state_dict', 'load_state_dict', 'parameters', 'to', 'eval',
                      'train')
    net = _without(member)
    with pytest.raises(TypeError, match=rf'\b{member}\b'):
        protocol.check_network_protocol(net)
    assert not protocol.is_cgaimmd_network(
        type('Named', (type(net),), {'default_generator': lambda self: 'ode'})())


def test_every_problem_listed_at_once():
    net = type('Broken', (CommittorOnly,), {'forward': None, 'train': None,
                                            'n_atoms': 0})()
    with pytest.raises(TypeError) as info:
        protocol.check_network_protocol(net)
    message = str(info.value)
    for name in ('forward', 'train', 'n_atoms'):
        assert name in message


def test_optional_member_fallbacks():
    duck = DuckNetwork(n_atoms=4)
    # n_atoms_of: the attribute, else the prior's shape without touching the global RNG
    state = torch.get_rng_state()
    assert protocol.n_atoms_of(duck) == 4
    assert torch.equal(torch.get_rng_state(), state)
    assert protocol.n_atoms_of(TinyFlowNetwork(n_atoms=5)) == 5

    x = torch.randn(2, 4, 3)
    # to_flow / from_flow / pair (flow-matching members): identity without the members
    assert sampler.to_flow(duck, x) is x
    assert sampler.from_flow(duck, x) is x
    x0 = torch.randn(2, 4, 3)
    assert sampler.pair(duck, x, x0) is x

    class WithOptional(DuckNetwork):
        n_atoms = 4
        lambda_clip = 7.5

        def to_flow(self, x):
            return x / 2

        def from_flow(self, z):
            return z * 2

        def flow_pair(self, x1, x0):
            return x1.flip(1)

        def committor(self, x):
            return x[:, 0, 0] * 0 + 42.0

    net = WithOptional()
    torch.testing.assert_close(sampler.to_flow(net, x), x / 2)
    torch.testing.assert_close(sampler.from_flow(net, x), x * 2)
    torch.testing.assert_close(sampler.pair(net, x, x0), x.flip(1))

    # lambda_clip_of: the network's value, else the setting (required then)
    assert protocol.lambda_clip_of(net) == 7.5
    assert protocol.lambda_clip_of(net, {'lambda_clip': None}) == 7.5
    assert protocol.lambda_clip_of(duck, {'lambda_clip': 3}) == 3.0
    assert protocol.lambda_clip_of(TinyFlowNetwork(lambda_clip=2.0), None) == 2.0
    assert isinstance(protocol.lambda_clip_of(duck, {'lambda_clip': 3}), float)

    # coordinate_committor: committor when present ...
    out = protocol.coordinate_committor(net, x.numpy())
    assert out.shape == (2,) and out.dtype == torch.float32
    assert torch.all(out == 42.0)
    # ... else forward of the flat values-function input
    coords = _coords(duck, n=5)
    out = protocol.coordinate_committor(duck, coords)
    expected = duck.forward(_dense_input(coords))[:, 0]
    torch.testing.assert_close(out, expected)
    assert out.shape == (5,) and out.dtype == torch.float32
    # a dense descriptor_transform is applied to the flat rows first
    out = protocol.coordinate_committor(
        duck, coords, descriptor_transform=lambda rows: np.asarray(rows) + 1.0)
    torch.testing.assert_close(out, expected + 1.0)
    # batch_size chunks give the same values
    torch.testing.assert_close(protocol.coordinate_committor(duck, coords, batch_size=2), expected)
    # autograd flows through the dense fallback unless the caller disables it
    xt = torch.as_tensor(coords).requires_grad_(True)
    protocol.coordinate_committor(duck, xt).sum().backward()
    assert xt.grad is not None and xt.grad[:, 0, 0].eq(1).all()
    with torch.no_grad():
        assert not protocol.coordinate_committor(duck, xt).requires_grad
    # an empty batch (a refill without survivors) is a (0,) tensor, the network is not called
    def never(*args, **kwargs):
        raise AssertionError('the network was called on an empty batch')

    for network, kwargs in ((TinyFlowNetwork(), {}),
                            (_with(forward=never), {}),
                            (_with(forward=never), {'descriptor_transform': never}),
                            (type('C', (DuckNetwork,), {'committor': never})(), {}),
                            (_with(forward=never), {'batch_size': 2})):
        m = protocol.n_atoms_of(network)
        for empty in (np.zeros((0, m, 3)), torch.zeros(0, m, 3)):
            out = protocol.coordinate_committor(network, empty, **kwargs)
            assert out.shape == (0,) and out.dtype == torch.float32
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='batch_size'):
            protocol.coordinate_committor(duck, coords, batch_size=bad)
        with pytest.raises(ValueError, match='batch_size'):
            protocol.coordinate_committor(duck, coords[:0], batch_size=bad)
    # network_device: first parameter, CPU without parameters
    assert protocol.network_device(duck) == torch.device('cpu')
    assert protocol.network_device(type('P', (DuckNetwork,), {
        'parameters': lambda self: iter(())})()) == torch.device('cpu')


@pytest.mark.graph
def test_coordinate_committor_graph_fallback():
    from torch_geometric.data import Data

    class GraphNet(DuckNetwork):
        def forward(self, inp):
            assert isinstance(inp, dict)
            pos = inp['positions']
            out = torch.zeros(len(inp['ptr']) - 1)
            out = out.index_add(0, inp['batch'], pos[:, 0])
            return out[:, None]

    net = GraphNet(n_atoms=3)
    coords = _coords(net, n=4)

    def transform(rows):
        rows = np.asarray(rows).reshape(len(rows), -1, 3)
        return [Data(positions=torch.as_tensor(r), num_nodes=len(r)) for r in rows]

    out = protocol.coordinate_committor(net, coords, descriptor_transform=transform)
    torch.testing.assert_close(out, torch.as_tensor(coords[:, :, 0].sum(axis=1)))


def test_lambda_clip_conflict_raises():
    net = TinyFlowNetwork(lambda_clip=10.0)
    with pytest.raises(TypeError, match='lambda_clip'):
        protocol.lambda_clip_of(net, {'lambda_clip': 5.0})
    # float32 buffers: agreement within 1e-6 relative is not a conflict
    assert protocol.lambda_clip_of(net, {'lambda_clip': 10.0 * (1 + 1e-7)}) == 10.0
    with pytest.raises(TypeError, match='lambda_clip'):
        protocol.lambda_clip_of(DuckNetwork(), {'lambda_clip': None})
    with pytest.raises(TypeError, match='lambda_clip'):
        protocol.lambda_clip_of(DuckNetwork())
    for bad in (0.0, -1.0, math.nan, math.inf, 'ten'):
        broken = type('B', (DuckNetwork,), {'lambda_clip': bad})()
        with pytest.raises(TypeError, match='lambda_clip'):
            protocol.lambda_clip_of(broken)
        with pytest.raises(TypeError, match='lambda_clip'):
            protocol.check_network_protocol(broken)
    tensor_clip = type('T', (DuckNetwork,), {'lambda_clip': torch.tensor(4.0)})()
    assert protocol.lambda_clip_of(tensor_clip) == 4.0


def _with(**members):
    return type('Variant', (DuckNetwork,), members)()


def test_bad_optional_member_signature():
    bad = {
        'committor': [lambda self: None, lambda self, x, y: None],
        'reset_parameters': [lambda self, x: None],
        'default_generator': [lambda self, network: None],
    }
    for member, variants in bad.items():
        for fn in variants:
            with pytest.raises(TypeError, match=member):
                protocol.check_network_protocol(_with(**{member: fn}))
        with pytest.raises(TypeError, match=f"optional member '{member}' is set but not callable"):
            protocol.check_network_protocol(_with(**{member: 3}))
    good = {
        'committor': [lambda self, x: None, lambda self, *args: None],
        'reset_parameters': [lambda self: None],
        'default_generator': [lambda self: 'ode', lambda self, **kwargs: None],
    }
    for member, variants in good.items():
        for fn in variants:
            protocol.check_network_protocol(_with(**{member: fn}))
    # members of a generation method are not the protocol's business
    for member, fn in (('velocity', lambda self, x, t: x), ('sample_prior', lambda self: None),
                       ('generate', lambda self: None), ('to_flow', lambda self: None)):
        protocol.check_network_protocol(_with(**{member: fn}))
    for n_atoms in (0, -2, '3', True, 2.0):
        with pytest.raises(TypeError, match='n_atoms'):
            protocol.check_network_protocol(_with(n_atoms=n_atoms))
    protocol.check_network_protocol(_with(n_atoms=np.int64(2)))


def test_is_cgaimmd_network():
    named = {'default_generator': lambda self: {'method': 'ode'}}
    assert protocol.is_cgaimmd_network(TinyFlowNetwork())
    assert protocol.is_cgaimmd_network(_with(**named))
    assert not protocol.is_cgaimmd_network(DuckNetwork())
    assert not protocol.is_cgaimmd_network(aimmd.network.utils.placeholder)
    assert not protocol.is_cgaimmd_network(torch.nn.Linear(3, 1))
    assert not protocol.is_cgaimmd_network(None)
    assert not protocol.is_cgaimmd_network(object())
    assert not protocol.is_cgaimmd_network(_with(committor=lambda self: None, **named))

    # a torch Module that does not override forward (nn.Module's own raises when called)
    class NoForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def velocity(self, x_t, t, lam, *, uncond_mask=None):
            return x_t

        def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
            return torch.zeros(n, 1, 3)

        def default_generator(self):
            return 'ode'

    assert callable(NoForward().forward)                  # it exists, so check 1 alone passed
    assert not protocol.is_cgaimmd_network(NoForward())
    with pytest.raises(TypeError, match='forward is not implemented'):
        protocol.check_network_protocol(NoForward())
    with_forward = type('WithForward', (NoForward,), {'forward': lambda self, inp: inp[:, :1]})
    assert protocol.is_cgaimmd_network(with_forward())
    assert protocol.is_cgaimmd_network(type('Sub', (TinyFlowNetwork,), {})())


def test_numerics_shapes():
    net = DuckNetwork(n_atoms=2)
    coords = _coords(net)
    report = protocol.check_network_numerics(net, coords, _dense_input(coords), n_generate=3,
                                             generation=ODE)
    assert report['sample_shape'] == [3, 2, 3]
    assert net.training is True                           # the training flag is restored

    with pytest.raises(ValueError, match='coordinates'):
        protocol.check_network_numerics(net, coords[:, :1], _dense_input(coords))

    cases = {
        'forward': _with(forward=lambda self, inp: inp[:, 0]),
        'velocity': _with(velocity=lambda self, x, t, lam, *, uncond_mask=None: x[:, :1]),
        # a non-finite prior shows in the generated candidates
        'non-finite': _with(sample_prior=lambda self, lam, n, *, uncond_mask=None,
                            generator=None: torch.full((n, 2, 3), float('nan'))),
        'committor': _with(committor=lambda self, x: torch.zeros(len(x), 1)),
        'device': _with(velocity=lambda self, x, t, lam, *, uncond_mask=None:
                        torch.zeros(x.shape, device='meta')),
    }
    for what, broken in cases.items():
        with pytest.raises(ValueError, match=what):
            protocol.check_network_numerics(broken, coords, _dense_input(coords),
                                            generation=ODE)

    # exceptions inside the network are reported as ValueError naming the member
    def raising(self, x, t, lam, *, uncond_mask=None):
        raise RuntimeError('shape mismatch in layer 3')

    with pytest.raises(ValueError, match="'ode' generator: generate.*RuntimeError: shape "
                                         "mismatch in layer 3$"):
        protocol.check_network_numerics(_with(velocity=raising), coords, _dense_input(coords),
                                        generation=ODE)
    # a generator the network cannot serve: the generate call reports it
    with pytest.raises(ValueError, match="'ode' generator: generate.*sample_prior"):
        protocol.check_network_numerics(CommittorOnly(), _coords(CommittorOnly()),
                                        generation=ODE)
    # a spec is built for the network; one that cannot be built is reported
    report = protocol.check_network_numerics(net, coords, generation={'method': 'ode',
                                                                      'n_steps': 1})
    assert report['generation']['settings']['n_steps'] == 1
    with pytest.raises(ValueError, match='could not be built.*no_such_method'):
        protocol.check_network_numerics(net, coords, generation='no_such_method')

    # an exception while finding n_atoms (the prior draw of n_atoms_of) is a ValueError too
    def unattached(self, lam, n, *, uncond_mask=None, generator=None):
        raise RuntimeError('prior bank not attached')

    with pytest.raises(ValueError, match=r'n_atoms_of.*RuntimeError: prior bank not attached'):
        protocol.check_network_numerics(_with(sample_prior=unattached), coords,
                                        _dense_input(coords), generation=ODE)
    with pytest.raises(ValueError, match='n_atoms'):
        protocol.check_network_numerics(_with(n_atoms=0), coords, _dense_input(coords))

    # the global torch RNG is not consumed
    state = torch.get_rng_state()
    protocol.check_network_numerics(net, coords, _dense_input(coords), generation=ODE)
    assert torch.equal(torch.get_rng_state(), state)


class FixedGenerator:
    """A generator that returns what it was given (for the numerics check)."""

    method = 'fixed_output'

    def __init__(self, output):
        self.output = output

    def check(self, network):
        return []

    def generate(self, network, lam, *, generator=None):
        return self.output(len(lam))

    def settings(self):
        return {}


def test_numerics_check_the_generator_output():
    net = CommittorOnly()
    coords = _coords(net)
    good = FixedGenerator(lambda n: torch.zeros(n, 2, 3))
    report = protocol.check_network_numerics(net, coords, generation=good, n_generate=4)
    assert report['generation'] == {'method': 'fixed_output', 'settings': {}}
    assert report['sample_shape'] == [4, 2, 3]
    cases = {
        r'returned shape \(2, 1, 3\), expected \(2, 2, 3\)': lambda n: torch.zeros(n, 1, 3),
        'non-finite': lambda n: torch.full((n, 2, 3), float('nan')),
        'device meta': lambda n: torch.zeros((n, 2, 3), device='meta'),
        'returned list': lambda n: [0.0] * n,
    }
    for message, output in cases.items():
        with pytest.raises(ValueError, match=f"'fixed_output' generator: generate.*{message}"):
            protocol.check_network_numerics(net, coords, generation=FixedGenerator(output))


class ModeRecorder(DuckNetwork):
    """A DuckNetwork that records (training, grad enabled) in every network call."""

    def __init__(self, n_atoms=2, has_n_atoms=True):
        super().__init__(n_atoms)
        if has_n_atoms:
            self.n_atoms = n_atoms
        self.seen = []

    def _record(self, member):
        self.seen.append((member, self.training, torch.is_grad_enabled()))

    def forward(self, inp):
        self._record('forward')
        return super().forward(inp)

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        self._record('velocity')
        return super().velocity(x_t, t, lam, uncond_mask=uncond_mask)

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        self._record('sample_prior')
        return super().sample_prior(lam, n, uncond_mask=uncond_mask, generator=generator)


def test_numerics_eval_mode_no_grad():
    """check_network_numerics runs the network in eval mode under no_grad, then restores it."""
    net = ModeRecorder()
    coords = _coords(net)
    assert net.training and torch.is_grad_enabled()
    protocol.check_network_numerics(net, coords, _dense_input(coords), generation=ODE)
    assert {member for member, _, _ in net.seen} == {'forward', 'velocity', 'sample_prior'}
    assert all(not training and not grad for _, training, grad in net.seen), net.seen
    assert net.training is True and torch.is_grad_enabled()
    # the n_atoms_of fallback draws under no_grad as well
    net = ModeRecorder(has_n_atoms=False)
    protocol.check_network_numerics(net, _coords(net), _dense_input(_coords(net)),
                                    generation=ODE)
    assert all(not grad for _, _, grad in net.seen), net.seen


def test_numerics_forward_mismatch_raises():
    coords = _coords(DuckNetwork(n_atoms=2), n=4)
    inp = _dense_input(coords)

    def committor(offset):
        return lambda self, x: torch.as_tensor(x)[:, 0, 0] - 1.0 + offset

    report = protocol.check_network_numerics(_with(committor=committor(1e-6)), coords, inp)
    assert report['committor_max_abs_diff'] < 1e-4
    with pytest.raises(ValueError, match='committor'):
        protocol.check_network_numerics(_with(committor=committor(0.1)), coords, inp)
    # a looser tolerance accepts it
    protocol.check_network_numerics(_with(committor=committor(0.1)), coords, inp, atol=0.2)
    # without forward_input the comparison is skipped, the committor's shape still checked
    report = protocol.check_network_numerics(_with(committor=committor(0.1)), coords, None)
    assert report['committor_max_abs_diff'] is None
    with pytest.raises(ValueError, match='committor'):
        protocol.check_network_numerics(
            _with(committor=lambda self, x: torch.zeros(len(x), 2)), coords, None)


def test_state_dict_weights_only(tmp_path):
    for kwargs in ({}, {'n_atoms': 3, 'lambda_clip': 4.0, 'hidden': 8}):
        net = TinyFlowNetwork(**kwargs)
        with torch.no_grad():
            for p in net.parameters():
                p.copy_(torch.randn_like(p))
        sd = net.state_dict()
        assert all(isinstance(v, torch.Tensor) for v in sd.values())
        fname = tmp_path / 'net.h5'
        torch.save(sd, fname)
        loaded = torch.load(fname, weights_only=True)
        other = TinyFlowNetwork(**kwargs)
        other.load_state_dict(loaded)                     # strict
        coords = _coords(net)
        torch.testing.assert_close(other(_dense_input(coords)), net(_dense_input(coords)))
        x = torch.as_tensor(coords)
        t = torch.full((3,), 0.3)
        lam = torch.tensor([-1.0, 0.0, 2.0])
        torch.testing.assert_close(other.velocity(x, t, lam), net.velocity(x, t, lam))


def test_coordinate_committor_batch_dict_transform():
    """A transform that returns a batch dict (node-table graph mode) is used as it is."""
    seen = []

    class DictNet(CommittorOnly):
        def forward(self, inp):
            seen.append(inp)
            assert isinstance(inp, dict)
            return inp['positions'].reshape(-1, 2, 3)[:, :, 0].sum(dim=1, keepdim=True)

    def transform(rows):
        rows = np.asarray(rows, dtype=np.float32)
        return {'positions': torch.as_tensor(rows), 'n_graphs': len(rows)}

    net = DictNet()
    coords = _coords(net, n=4)
    out = protocol.coordinate_committor(net, coords, descriptor_transform=transform, batch_size=3)
    torch.testing.assert_close(out, torch.as_tensor(coords[:, :, 0].sum(axis=1)))
    assert [inp['n_graphs'] for inp in seen] == [3, 1]
    # the numerics check takes such a dict as forward input
    report = protocol.check_network_numerics(
        net, coords, transform(coords.reshape(4, 6)))
    assert report['forward_range'] is not None
