"""The generation interface of C-GAIMMD, ``aimmd.cgaimmd.generation``.

The generation method (BioEmu2's diffusion sampler, the flow-matching ODE,
later others) is picked through one interface: a generator has a ``method``
name, ``check(network)``, ``generate(network, lam, *, generator=None)`` and
``settings()``; ``make_generator`` builds one from a method name, a settings
dict, a generator object or the network's ``default_generator()``.
"""

import json

import pytest
import torch

from aimmd.cgaimmd import generation
from aimmd.cgaimmd.flow_matching.sampler import ODEGenerator
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.generation import (available_generators, canonical_generator_spec,
                                      check_generator, is_generator, make_generator,
                                      register_generator, unregister_generator)


class ShiftedPrior:
    """A generation method defined in this test module: the prior, shifted by the target."""

    method = 'test_shifted_prior'

    def __init__(self, scale=1.0, chunk=4):
        if not scale > 0:
            raise ValueError(f'scale must be > 0, got {scale!r}')
        self.scale = float(scale)
        self.chunk = int(chunk)

    def check(self, network):
        if callable(getattr(network, 'sample_prior', None)):
            return []
        return ['needs sample_prior(lam, n, *, generator=None), which the network lacks']

    def generate(self, network, lam, *, generator=None):
        parts = []
        for start in range(0, len(lam), self.chunk):
            part = lam[start:start + self.chunk]
            prior = network.sample_prior(part, len(part), generator=generator)
            parts.append(prior + self.scale * part[:, None, None])
        return torch.cat(parts)

    def settings(self):
        return {'scale': self.scale, 'chunk': self.chunk}


class CommittorOnly(torch.nn.Module):
    """A C-GAIMMD network with a committor and nothing else."""

    n_atoms = 1

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, inp):
        return self.linear(inp)


def _with_default(default):
    """A TinyFlowNetwork whose default_generator() returns `default`."""
    return type('WithDefault', (TinyFlowNetwork,), {'default_generator': lambda self: default})()


@pytest.fixture
def shifted():
    register_generator('test_shifted_prior', ShiftedPrior)
    yield ShiftedPrior
    unregister_generator('test_shifted_prior')


def test_a_generator_registered_here_is_built_from_name_dict_and_object(shifted):
    network = TinyFlowNetwork()
    by_name = make_generator('test_shifted_prior')
    by_dict = make_generator({'method': 'test_shifted_prior', 'scale': 2.0, 'chunk': 3})
    instance = ShiftedPrior(scale=0.5)
    assert make_generator(instance) is instance
    assert type(by_name) is ShiftedPrior and by_name.settings() == {'scale': 1.0, 'chunk': 4}
    assert by_dict.settings() == {'scale': 2.0, 'chunk': 3}
    for built in (by_name, by_dict, instance):
        assert built.method == 'test_shifted_prior'
        settings = built.settings()
        assert json.loads(json.dumps(settings)) == settings          # JSON-able
        rebuilt = make_generator({'method': built.method, **settings})
        assert rebuilt.settings() == settings
        assert check_generator(built, network) == []
        out = built.generate(network, torch.linspace(-1, 1, 5),
                             generator=torch.Generator().manual_seed(0))
        assert out.shape == (5, 1, 3)
    # what it needs from a network is its own business
    assert check_generator(by_name, CommittorOnly()) == [
        "generator 'test_shifted_prior': needs sample_prior(lam, n, *, generator=None), "
        "which the network lacks"]
    # settings it does not accept: the method is named
    with pytest.raises(TypeError, match="'test_shifted_prior'.*bogus"):
        make_generator({'method': 'test_shifted_prior', 'bogus': 1})
    with pytest.raises(ValueError, match="'test_shifted_prior'.*scale must be > 0"):
        make_generator({'method': 'test_shifted_prior', 'scale': -1})
    assert 'test_shifted_prior' in available_generators()


def test_registry_rules(shifted):
    assert register_generator('test_shifted_prior', ShiftedPrior) is ShiftedPrior   # no-op
    with pytest.raises(ValueError, match='already registered'):
        register_generator('test_shifted_prior', lambda **settings: ShiftedPrior(**settings))
    other = register_generator('test_shifted_prior', ODEGenerator, replace=True)
    assert other is ODEGenerator
    assert isinstance(make_generator('test_shifted_prior'), ODEGenerator)
    for name in ('', '1st', 'has space', None, 3, 'a/b'):
        with pytest.raises(ValueError, match='generation method name'):
            register_generator(name, ShiftedPrior)
    with pytest.raises(TypeError, match='callable'):
        register_generator('test_not_callable', 3)
    unregister_generator('test_shifted_prior')
    unregister_generator('test_never_registered')                    # ignored
    with pytest.raises(ValueError, match="unknown generation method 'test_shifted_prior'"):
        make_generator('test_shifted_prior')
    # a factory that does not return a generator
    register_generator('test_returns_none', lambda: None)
    try:
        with pytest.raises(TypeError, match='lacks the generator members'):
            make_generator('test_returns_none')
    finally:
        unregister_generator('test_returns_none')


def test_none_takes_the_network_default(shifted):
    assert make_generator(None, TinyFlowNetwork()) == ODEGenerator()
    assert isinstance(make_generator(None, _with_default('test_shifted_prior')), ShiftedPrior)
    built = make_generator(None, _with_default({'method': 'test_shifted_prior', 'scale': 3.0}))
    assert built.settings() == {'scale': 3.0, 'chunk': 4}
    instance = ShiftedPrior(chunk=2)
    assert make_generator(None, _with_default(instance)) is instance
    with pytest.raises(TypeError, match="without 'method'"):
        make_generator(None, _with_default({'scale': 3.0}))


def test_settings_without_method_override_the_network_default(shifted):
    """What the old flow_parameters['generate_chunk'] becomes: settings of the default method."""
    assert make_generator({'chunk': 8}, TinyFlowNetwork()) == ODEGenerator(chunk=8)
    network = _with_default({'method': 'test_shifted_prior', 'scale': 3.0})
    assert make_generator({'chunk': 2}, network).settings() == {'scale': 3.0, 'chunk': 2}
    network = _with_default(ShiftedPrior(scale=5.0))
    assert make_generator({'chunk': 7}, network).settings() == {'scale': 5.0, 'chunk': 7}
    with pytest.raises(TypeError, match='cgaimmd_generator'):
        make_generator({'chunk': 2}, CommittorOnly())


def test_none_without_a_default_names_the_params_field():
    with pytest.raises(TypeError, match=r'CommittorOnly has no default_generator\(\), so '
                                        r'params.cgaimmd_generator must name the generation '
                                        r'method'):
        make_generator(None, CommittorOnly())
    with pytest.raises(TypeError, match='cgaimmd_generator'):
        make_generator(None)
    with pytest.raises(TypeError, match='cgaimmd_generator'):
        make_generator(None, _with_default(None))


def test_unknown_and_wrong_specs():
    with pytest.raises(ValueError, match=r"unknown generation method 'nope'; known methods "
                                         r"are \[.*'ode'.*\]"):
        make_generator('nope')
    for bad in (3, ['ode'], object(), {'method': 3}):
        with pytest.raises(TypeError):
            make_generator(bad)


def test_bioemu2_edm_is_known_and_says_what_is_missing():
    """Registered lazily: known by name; its module (bioemu2 extra) is imported on first use."""
    assert 'bioemu2_edm' in available_generators()
    assert generation.is_known_generator('bioemu2_edm')
    assert canonical_generator_spec('bioemu2_edm') == 'bioemu2_edm'        # imports nothing
    assert canonical_generator_spec({'method': 'bioemu2_edm', 'n_steps': 30}) == {
        'method': 'bioemu2_edm', 'n_steps': 30}
    import importlib.util
    if importlib.util.find_spec('aimmd.cgaimmd.bioemu2') is not None:
        pytest.skip('the BioEmu2 adaptor is installed')
    with pytest.raises(ImportError) as info:
        make_generator('bioemu2_edm')
    message = str(info.value)
    assert "generation method 'bioemu2_edm'" in message
    assert 'aimmd.cgaimmd.bioemu2' in message
    assert "needs the 'bioemu2' extra (pip install 'aimmd-lab[bioemu2]')" in message


def test_check_generator_reports_instead_of_raising():
    network = TinyFlowNetwork()

    class Incomplete:
        method = 3
        check = None

    assert check_generator(Incomplete(), network) == [
        'Incomplete is not a generator: method, check, generate, settings missing or of the '
        'wrong type']
    assert not is_generator(Incomplete())

    class Broken(ShiftedPrior):
        method = 'test_broken'

        def settings(self):
            return {'callable': print}

        def check(self, network):
            raise RuntimeError('cannot inspect')

    problems = check_generator(Broken(), network)
    assert problems[0].startswith("generator 'test_broken': settings() is not a "
                                  "JSON-serializable dict")
    assert problems[1] == ("generator 'test_broken': check(network) raised RuntimeError: "
                           "cannot inspect")

    class NotAList(ShiftedPrior):
        def check(self, network):
            return 'fine'

    assert check_generator(NotAList(), network) == [
        "generator 'test_shifted_prior': check(network) returned str, not a list of messages"]


def test_canonical_generator_spec():
    assert canonical_generator_spec(None) is None
    assert canonical_generator_spec('ode') == 'ode'
    instance = ODEGenerator()
    assert canonical_generator_spec(instance) is instance
    assert canonical_generator_spec({'chunk': 4, 'shape': [1, 2]}) == {'chunk': 4,
                                                                        'shape': (1, 2)}
    for bad, message in (('nope', 'unknown generation method'),
                         ({'method': 'nope'}, 'unknown generation method'),
                         ({'method': 'ode', 'n_steps': -1}, 'n_steps'),
                         ({'chunk': {1, 2}}, 'JSON-serializable'),
                         (3.0, 'must be a method name')):
        with pytest.raises(TypeError, match=message):
            canonical_generator_spec(bad)
