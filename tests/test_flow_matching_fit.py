"""The online fit of flow-matching networks, ``aimmd.cgaimmd.flow_matching.fit``.

A toy run folder (``tests._cgaimmd_helpers.build_cgaimmd_run_dir``) and the
minimal ``TinyFlowNetwork`` (dense, every protocol fallback) in the default
lane; a tiny ``JointPaiNN`` where its members matter; graph mode in
``tests/test_flow_matching_fit_graph_mode.py`` (``--rungraph``).

Every round starts from the same weights (``params.initial_network``, else a
seeded ``reset_parameters``) with a new optimizer, EMA and normalizer: the
tests of that are under "every round from the same start". The trainer and a
round are compared with the flow branch's code bit for bit
(``tests/_flow_matching_fit_golden.py``).
"""

import copy
import functools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.flow_matching import fit as ff
from aimmd.cgaimmd.flow_matching import flowmath
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.dataset import extract_dataset
from aimmd.params.utils import create_default_values_function
from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble import PathEnsemble
from aimmd.pathensemble.utils import assemble_pathensemble
from aimmd.worker.utils import get_initial_frames_for_training
from tests import _flow_matching_fit_golden as golden
from tests._cgaimmd_helpers import (build_cgaimmd_run_dir, cgaimmd_params_namespace,
                                    write_fake_network_file)
from tests._helpers_unit import state_labels_from_positions, write_trajectory


FAST = dict(steps_per_round=3, warmup_steps=0, n_sp=4, n_anchor=4, n_fm=4, idle_sleep_s=0.0,
            round_oom_wait_s=0.0, normalizer_warmup=0)


class FakeWorker:
    """The members of WorkerTrain the fit reads: _directory, task, must_stop."""

    def __init__(self, directory, task='train', stop=None):
        self._directory = str(directory)
        self.directory = str(directory)
        self.task = task
        self.stop = stop

    @property
    def must_stop(self):
        return bool(self.stop()) if self.stop is not None else False


def _ensemble(params, run):
    chains = params.shot_chains(run, None)
    free = params.free_trajectories(run)
    return assemble_pathensemble(chains, free)


def _values_function(network):
    """AIMMD's default values function, returning float64 as the C-GAIMMD values function does."""
    base = create_default_values_function(network)

    def values_function(descriptors, *args, **kwargs):
        return np.asarray(base(descriptors, *args, **kwargs), dtype=np.float64)
    return values_function


def _state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _randomized(net, seed=0, scale=0.5):
    return golden.randomize(net, seed=seed, scale=scale)


def _setup(tmp_path, network=None, *, n_chains=3, n_shots=4, n_free=1, n_atoms=1, seed=0,
           initial=True, **overrides):
    """A run folder, its params (``initial_network`` = the network's weights now), a trainer stand-in."""
    tmp_path = Path(tmp_path)
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=n_chains, n_shots=n_shots, n_free=n_free,
                               n_atoms=n_atoms)
    if network is None:
        torch.manual_seed(seed)
        network = _randomized(TinyFlowNetwork(n_atoms=n_atoms, lambda_clip=10.0), seed=seed)
    if initial:
        fname = tmp_path / 'initial_network.h5'
        torch.save(network.state_dict(), fname)
        overrides.setdefault('initial_network', str(fname))
    overrides.setdefault('values_function', _values_function(network))
    params = cgaimmd_params_namespace(network=network, **overrides)
    worker = FakeWorker(rd.run)
    return SimpleNamespace(rd=rd, run=rd.run, params=params, worker=worker,
                           pe=_ensemble(params, rd.run), network=network,
                           start=_state(network))


def _fit(s, **kwargs):
    return ff.flow_matching_fit(s.params, s.pe, worker=s.worker, **{**FAST, **kwargs})


def _state_file(s):
    return json.loads(Path(f'{s.run}/flow_fit{s.params.sorted_states}/state.json').read_text())


def _rounds(s):
    return [json.loads(line) for line in
            Path(f'{s.run}/flow_fit{s.params.sorted_states}/rounds.jsonl').read_text().splitlines()]


@pytest.fixture
def trainers(monkeypatch):
    """Every FlowTrainer the fit creates, with the network's weights at its creation (``start_state``)."""
    made = []

    class Recording(ff.FlowTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_state = _state(self.network)
            self.created_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            made.append(self)

    monkeypatch.setattr(ff, 'FlowTrainer', Recording)
    return made


# --------------------------------------------------------------------------------------------
# signature, keywords, refusals
# --------------------------------------------------------------------------------------------

def test_wrapper_accepted_by_params():
    """The params-file wrapper has the four parameters Params requires."""
    def fit(params, pathensemble, verbose=False, worker=None):
        return ff.flow_matching_fit(params, pathensemble, verbose=verbose, worker=worker,
                                    graphs=False, steps_per_round=10)
    params = aimmd.Params.placeholder.copy()
    params._setattr('fit', fit, process_and_check=False)
    assert params.fit.__name__ == 'fit'
    import inspect
    assert tuple(inspect.signature(ff.flow_matching_fit).parameters)[:4] == (
        'params', 'pathensemble', 'verbose', 'worker')
    assert set(ff.FLOW_FIT_DEFAULTS) >= {'graphs', 'steps_per_round', 'lr', 'ema_decay',
                                         'grad_penalty', 'anchor_weight', 'holdout_seed',
                                         'relabel_chunk', 'state_dir', 'cache_dataset',
                                         'relabel_every', 'basin_draw', 'split_backward',
                                         'holdout_every', 'early_stopping_patience'}
    assert ff.FLOW_FIT_DEFAULTS['anchor_weight'] == 0.5
    assert ff.FLOW_FIT_DEFAULTS['holdout_seed'] == 12345
    assert ff.FLOW_FIT_DEFAULTS['trainer_seed'] is None
    for off in ('split_backward', 'graphs'):
        assert ff.FLOW_FIT_DEFAULTS[off] is False
    assert ff.FLOW_FIT_DEFAULTS['early_stopping_patience'] is None
    assert ff.FLOW_FIT_DEFAULTS['basin_draw'] == 'uniform'


def test_old_name_resolves_with_a_warning():
    """aimmd.network.flow_fit.flow_fit is this fit (one release), with one DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match='flow_matching_fit'):
        from aimmd.network.flow_fit import flow_fit
    assert flow_fit is ff.flow_matching_fit
    with pytest.warns(DeprecationWarning):
        from aimmd.network.flow_fit import FlowTrainer, LossNormalizer, release_trainer
    assert (FlowTrainer, LossNormalizer, release_trainer) == (
        ff.FlowTrainer, ff.LossNormalizer, ff.release_trainer)


def test_unknown_kwarg(tmp_path):
    """An unknown keyword is a TypeError; a bad value of a known one a ValueError naming it."""
    s = _setup(tmp_path)
    with pytest.raises(TypeError, match='learning_rate'):
        ff.flow_matching_fit(s.params, s.pe, worker=s.worker, learning_rate=1e-3)
    for bad, match in (({'lr_schedule': 'linear'}, 'lr_schedule'),
                       ({'t_schedule': 'beta'}, 't_schedule'),
                       ({'steps_per_round': 0}, 'steps_per_round'),
                       ({'ema_decay': 1.5}, 'ema_decay'), ({'graphs': 1}, 'graphs'),
                       ({'split_backward': 1}, 'split_backward'),
                       ({'trainer_seed': -1}, 'trainer_seed'),
                       ({'holdout_every': -1}, 'holdout_every'),
                       ({'early_stopping_patience': 0}, 'early_stopping_patience'),
                       ({'early_stopping_patience': 3}, 'holdout_every >= 1')):
        with pytest.raises(ValueError, match=match):
            ff.flow_matching_fit(s.params, s.pe, worker=s.worker, **bad)
    assert not os.path.exists(f'{s.run}/flow_fitARB')            # refused before any set-up


def test_multi_system_raises(tmp_path):
    s = _setup(tmp_path)
    with pytest.raises(NotImplementedError):
        ff.flow_matching_fit(s.params, [s.pe, s.pe], worker=s.worker, **FAST)
    s.params.__dict__['multi_system'] = True
    with pytest.raises(NotImplementedError):
        ff.flow_matching_fit(s.params, s.pe, worker=s.worker, **FAST)


def test_rescale_committor_raises(tmp_path):
    s = _setup(tmp_path, rescale_committor=True)
    with pytest.raises(TypeError, match='rescale_committor'):
        _fit(s)


def test_first_call_checks_protocol(tmp_path):
    s = _setup(tmp_path, network=torch.nn.Linear(3, 1), initial=False)
    with pytest.raises(TypeError, match='protocol'):
        _fit(s)


def test_the_network_needs_the_flow_matching_members_with_uncond_mask(tmp_path):
    """The trainer drops labels with uncond_mask in velocity and sample_prior: both must take it."""

    class NoMask(TinyFlowNetwork):
        def velocity(self, x_t, t, lam):
            return super().velocity(x_t, t, lam)

    class PriorWithoutMask(TinyFlowNetwork):
        def sample_prior(self, lam, n, *, generator=None):
            return super().sample_prior(lam, n, generator=generator)

    for cls, member in ((NoMask, 'velocity'), (PriorWithoutMask, 'sample_prior')):
        s = _setup(tmp_path / member, network=cls(n_atoms=1, lambda_clip=10.0))
        with pytest.raises(TypeError, match=rf'not a flow-matching network: {member} must '
                                            rf'accept .*uncond_mask'):
            _fit(s)


def test_the_network_needs_a_start(tmp_path):
    """Without initial_network a network must reset itself; the error says what to set."""
    s = _setup(tmp_path, initial=False)
    assert not callable(getattr(s.network, 'reset_parameters', None))
    with pytest.raises(TypeError, match='set initial_network'):
        _fit(s)


# --------------------------------------------------------------------------------------------
# small pieces
# --------------------------------------------------------------------------------------------

def test_committor_nll_and_normalizer():
    logits = torch.tensor([0.0, 2.0, -1.0])
    y = torch.tensor([0.5, 1.0, 0.0])
    n = torch.tensor([2.0, 1.0, 1.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    assert torch.allclose(ff.committor_nll(logits, y, n), (bce * n).sum() / n.sum())
    norm = ff.LossNormalizer(decay=0.5, warmup=1, max_ratio=10.0)
    first = norm('q', torch.tensor(4.0))
    assert float(first) == 4.0                           # warm-up: identity
    norm.advance()
    assert float(norm('q', torch.tensor(2.0))) == pytest.approx(2.0 / 3.0)   # (4 + 2) / 2
    norm.advance()
    assert float(norm('q', torch.tensor(1e9))) == pytest.approx(1e9 / (0.5 * 3 + 0.5 * 30))
    state = norm.state_dict()
    assert state == {'step': 2, 'terms': {'q': {'value': 16.5, 'count': 3}}}
    other = ff.LossNormalizer(decay=0.5, warmup=1)
    other.load_state_dict(json.loads(json.dumps(state)))
    assert other.state_dict() == state
    off = ff.LossNormalizer(enabled=False, warmup=0)
    off.advance()
    assert float(off('q', torch.tensor(3.0))) == 3.0


def test_stale_descriptor_cache_raises(tmp_path):
    s = _setup(tmp_path)
    fname = s.rd.shots[0].fnames[0]
    np.save(get_cache_fname(fname, 'descriptors'), np.zeros((7, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='migrate_descriptors'):
        _fit(s)


def test_the_fit_reads_the_params_series(tmp_path, trainers):
    """The dataset comes from params.descriptors_series (here a series of its own)."""
    s = _setup(tmp_path)
    for path in list(s.rd.shots) + list(s.rd.free):
        fname = path.fnames[0]
        rows = np.load(get_cache_fname(fname, 'descriptors'))
        np.save(get_cache_fname(fname, 'descriptors-generator_atoms'), rows + 0.25)
    s.params.__dict__['descriptors_series'] = 'descriptors-generator_atoms'
    _, info = _fit(s)
    data = trainers[-1].dataset
    assert data.notes['series'] == 'descriptors-generator_atoms'
    assert info['network_metadata']['series'] == 'descriptors-generator_atoms'
    plain = extract_dataset(s.pe, states='ARB', n_atoms=1, free_stride=5)
    np.testing.assert_array_equal(data.sp_rows, plain.sp_rows + np.float32(0.25))


# --------------------------------------------------------------------------------------------
# the flow branch's trainer and round, bit for bit
# --------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _golden():
    return torch.load(golden.GOLDEN, map_location='cpu', weights_only=True)


def _same_outcome(new, old, exact=True):
    if exact:
        assert new['totals'] == old['totals']
        assert new['terms'] == old['terms']
    else:
        np.testing.assert_allclose(new['totals'], old['totals'], rtol=1e-5, atol=1e-8)
        assert [t.keys() for t in new['terms']] == [t.keys() for t in old['terms']]
        for a, b in zip(new['terms'], old['terms']):
            for key in a:
                assert a[key] == pytest.approx(b[key], rel=1e-5, abs=1e-8), key
    for key in ('live', 'ema'):
        assert new[key].keys() == old[key].keys()
        for name in old[key]:
            if exact:
                assert torch.equal(new[key][name], old[key][name]), (key, name)
            else:
                torch.testing.assert_close(new[key][name], old[key][name], rtol=1e-5,
                                           atol=1e-6)
    assert new['optimizer'].keys() == old['optimizer'].keys()
    for k in old['optimizer']:
        for name, value in old['optimizer'][k].items():
            mine = new['optimizer'][k][name]
            if exact:
                assert torch.equal(mine, value), (k, name)
            else:
                torch.testing.assert_close(mine.to(torch.float64), value.to(torch.float64),
                                           rtol=1e-4, atol=1e-9)
    if exact:
        assert new['normalizer'] == old['normalizer']
    assert new['skipped_steps'] == old['skipped_steps']
    assert new['rng'] == old['rng'] and torch.equal(new['generator'], old['generator'])


@pytest.mark.parametrize('name', sorted(golden.STEP_CASES))
def test_trainer_steps_match_the_flow_branch(name):
    """FlowTrainer steps equal the flow branch's (bb98943), bit for bit: losses, weights, optimizer, streams."""
    _same_outcome(golden.run_step_case(golden.new_api(), name), _golden()['steps'][name])


def _same_round(new, old):
    assert new['losses'] == old['losses']
    assert _equal(new['network'], old['network'])
    for key in ('held_out_chains', 'n_trained_shots', 'loss_terms', 'holdout_nll_ema',
                'holdout_nll_live', 'relabels_in_round', 'sp_fname'):
        assert new[key] == old[key], key
    for key in ('fm_pool', 'fm_lambda', 'fr_w'):
        assert torch.equal(new[key], old[key]), key


def test_rounds_match_the_flow_branch_started_fresh(tmp_path, monkeypatch, trainers):
    """Round 1 is the flow branch's first call; round 2 its fresh call with seed + 1000, bit for bit.

    The flow branch continued its second call from the first (warm start), so
    its fresh call from the same start is what a round of this fit must equal.
    """
    g = _golden()['rounds']
    monkeypatch.setattr(ff, 'path_weights', golden.ramp_weights)
    api = golden.new_api()
    for kind, cases in (('tiny', ('tiny_seed5', 'tiny_seed1005')), ('painn', ('painn_seed5',))):
        run, manifest = golden.write_fixture(tmp_path / kind)
        network = golden.build_network(api, kind, golden.N_ATOMS)
        network.load_state_dict(g['start'][kind])
        fname = tmp_path / f'initial_{kind}.h5'
        torch.save(g['start'][kind], fname)
        params = golden.round_params(network, initial_network=str(fname))
        worker = golden.FakeWorker(run)
        pe = golden.fixture_ensemble(run, manifest)
        settings = golden.round_settings(cases[0])            # the same trainer_seed every round
        for r, name in enumerate(cases, start=1):
            losses, info = ff.flow_matching_fit(params, pe, worker=worker, **settings)
            assert info['round'] == r and info['network_metadata']['round_seed'] == 5 + 1000 * (r - 1)
            trainer = trainers[-1]
            new = golden.round_outcome(losses, info, network, trainer.dataset, trainer.fm_pool,
                                       trainer.fm_lambda, run)
            _same_round(new, g[name])


@pytest.mark.parametrize('name', sorted(golden.STEP_CASES))
def test_split_backward_trains_like_one_backward(name):
    """Two backward passes per step give the same training to float32 rounding (the run-local patch's test)."""
    case = golden.STEP_CASES[name]
    patched = dict(golden.STEP_CASES)
    patched[name] = {**case, 'settings': {**case['settings'], 'split_backward': True}}
    old = _golden()['steps'][name]
    try:
        golden.STEP_CASES[name] = patched[name]
        new = golden.run_step_case(golden.new_api(), name)
    finally:
        golden.STEP_CASES[name] = case
    _same_outcome(new, old, exact=False)


# --------------------------------------------------------------------------------------------
# split_backward
# --------------------------------------------------------------------------------------------

def _hand(network, m=2, held=(), **settings):
    return golden.hand_trainer(golden.new_api(), network, m, held=held, **settings)


def test_split_backward_off_is_bit_identical_to_absent():
    """split_backward=False and a settings dict without the key train the same bits."""
    results = []
    for settings in ('absent', 'off'):
        torch.manual_seed(0)
        net = _randomized(TinyFlowNetwork(n_atoms=2, lambda_clip=10.0))
        trainer = _hand(net, p_uncond=0.5, grad_penalty=0.2)
        if settings == 'absent':
            del trainer.settings['split_backward']
        else:
            assert trainer.settings['split_backward'] is False
        steps = [trainer.step() for _ in range(5)]
        results.append(golden.trainer_outcome(trainer, steps))
    _same_outcome(results[0], results[1])


def test_split_backward_frees_the_committor_part_first(monkeypatch):
    """With split_backward the committor part's gradients exist before the flow-matching loss is formed."""
    seen = []
    original = ff.FlowTrainer._fm_loss

    def spy(self, *args, **kwargs):
        seen.append(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in self.live.parameters()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ff.FlowTrainer, '_fm_loss', spy)
    for split in (True, False):
        torch.manual_seed(0)
        trainer = _hand(_randomized(TinyFlowNetwork(n_atoms=2, lambda_clip=10.0)),
                        split_backward=split)
        trainer.step()
    assert seen == [True, False]


def test_split_backward_without_flow_matching_or_committor_terms():
    """Either part alone: the split step trains like the single backward pass."""
    for extra in (dict(n_fm=0), dict(n_sp=0, n_anchor=0), dict(fm_weight=0.0, grad_penalty=2.0)):
        outcomes = []
        for split in (False, True):
            torch.manual_seed(0)
            trainer = _hand(_randomized(TinyFlowNetwork(n_atoms=2, lambda_clip=10.0)),
                            split_backward=split, **extra)
            steps = [trainer.step() for _ in range(3)]
            outcomes.append(golden.trainer_outcome(trainer, steps))
        _same_outcome(outcomes[1], outcomes[0], exact=False)


def test_split_backward_in_a_fit(tmp_path):
    s = _setup(tmp_path)
    losses, info = _fit(s, split_backward=True)
    assert info['status'] == 'trained' and len(losses) == FAST['steps_per_round']
    assert info['network_metadata']['settings']['split_backward'] is True


# --------------------------------------------------------------------------------------------
# every round from the same start (no warm start)
# --------------------------------------------------------------------------------------------

def test_every_round_starts_from_the_initial_network(tmp_path, trainers):
    """Rounds 1-3 start from initial_network's weights, with a fresh optimizer, EMA and normalizer."""
    s = _setup(tmp_path)
    file_state = torch.load(s.params.initial_network, weights_only=True)
    for r in (1, 2, 3):
        losses, info = _fit(s, ema_decay=0.9, lr=1e-2)
        assert info['round'] == r and info['status'] == 'trained'
        trainer = trainers[-1]
        assert _equal(trainer.start_state, file_state)            # the start, every round
        assert trainer.created_optimizer_state['state'] == {}     # a fresh optimizer
        assert trainer.normalizer.step == FAST['steps_per_round']  # counted from 0 this round
        assert trainer.step_count == FAST['steps_per_round']
        assert not _equal(_state(s.network), file_state)          # and the round trained
        assert info['network_metadata']['start'] == {
            'kind': 'initial_network', 'file': os.path.abspath(s.params.initial_network),
            'sha256': aimmd.cgaimmd.io.file_sha256(s.params.initial_network)}
    assert len({id(t) for t in trainers}) == 3


def test_round_two_does_not_depend_on_round_one_weights(tmp_path, trainers):
    """Whatever params.network holds when a round starts (round 1's result or anything else),
    the round's result is the same: the start, the labels and the streams come from the
    start weights, the data and the seed only."""
    results = []
    for case in ('as_trained', 'replaced', 'new_process'):
        s = _setup(tmp_path / case)
        _fit(s, ema_decay=0.9, lr=1e-2, relabel_every=1, trainer_seed=3)   # (folders differ)
        if case == 'replaced':
            with torch.no_grad():                               # another round 1 result
                for p in s.network.parameters():
                    p.add_(torch.randn_like(p))
        elif case == 'new_process':
            ff._RUNS.clear()
            s.network.load_state_dict(_randomized(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0),
                                                  seed=7).state_dict())
        losses, info = _fit(s, ema_decay=0.9, lr=1e-2, relabel_every=1, trainer_seed=3)
        assert info['round'] == 2
        trainer = trainers[-1]
        results.append((losses, _state(s.network), trainer.fm_lambda.copy(),
                        info['record']['loss_terms'], trainer.rng.bit_generator.state))
    for other in results[1:]:
        assert other[0] == results[0][0]
        assert _equal(other[1], results[0][1])
        assert np.array_equal(other[2], results[0][2])
        assert other[3] == results[0][3] and other[4] == results[0][4]


def test_rounds_differ_only_through_the_data_and_the_round_seed(tmp_path, trainers):
    """With the same data, round 2 equals round 1 run with the round seed of round 2."""
    s = _setup(tmp_path / 'two_rounds')
    _fit(s, trainer_seed=11)
    second, _ = _fit(s, trainer_seed=11)
    t = _setup(tmp_path / 'one_round')
    first, info = _fit(t, trainer_seed=1011)
    assert info['round'] == 1 and second == first
    assert _equal(_state(s.network), _state(t.network))


def test_flow_matching_labels_never_come_from_the_values_caches(tmp_path, monkeypatch, trainers):
    """The labels are the round's own EMA's committor: values caches with other values change nothing."""
    monkeypatch.setattr(ff, 'path_weights', golden.ramp_weights)    # the caches feed nothing else
    results = []
    for case in ('as_written', 'scrambled'):
        s = _setup(tmp_path / case)
        if case == 'scrambled':
            for path in list(s.rd.shots) + list(s.rd.free):
                cache = get_cache_fname(path.fnames[0], 'values')
                np.save(cache, np.load(cache) * -3.0 + 7.0)
        write_fake_network_file(s.run, s.network)
        _fit(s, trainer_seed=1)
        losses, _ = _fit(s, fill_missing_values=False, trainer_seed=1)
        results.append((losses, trainers[-1].fm_lambda.copy(), _state(s.network)))
    assert results[0][0] == results[1][0] and np.array_equal(results[0][1], results[1][1])
    assert _equal(results[0][2], results[1][2])


def test_reset_start_without_initial_network(tmp_path, trainers):
    """A network with reset_parameters starts every round from the same seeded reset; the caller's streams are untouched."""
    torch.manual_seed(0)
    net = golden.build_network(golden.new_api(), 'painn', 3)
    s = _setup(tmp_path, network=net, n_atoms=3, initial=False, n_chains=2, n_shots=2)
    assert getattr(s.params, 'initial_network', None) is None
    starts = []
    for r in (1, 2):
        before = torch.random.get_rng_state()
        _, info = _fit(s, steps_per_round=1, trainer_seed=4)
        assert torch.equal(before, torch.random.get_rng_state())   # global stream unchanged
        starts.append(trainers[-1].start_state)
        assert info['network_metadata']['start'] == {'kind': 'reset_parameters', 'seed': 4}
    assert _equal(starts[0], starts[1])
    assert not _equal(starts[0], s.start)                      # a reset, not the given weights
    expected = copy.deepcopy(net)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(4)
        expected.reset_parameters()
    assert _equal(starts[0], _state(expected))


def test_the_round_counter_survives_a_new_process(tmp_path, trainers):
    """The round counter and the held-out chains come from state.json (and network<s>.json)."""
    s = _setup(tmp_path, n_chains=4, n_shots=2)
    _fit(s)
    _fit(s)
    assert _state_file(s)['round'] == 2
    ff._RUNS.clear()                                              # a new trainer process
    _, info = _fit(s)
    assert info['round'] == 3
    assert info['network_metadata']['round_seed'] == info['network_metadata']['seed'] + 2000
    # the trainer's network json (round 9: a fit state lost) counts too
    write_fake_network_file(s.run, s.network, round=9)
    _, info = _fit(s)
    assert info['round'] == 10 and _state_file(s)['round'] == 10
    # releasing the process memory loses nothing on disk
    assert ff.release_trainer(f'{s.run}/flow_fitARB') is True
    assert ff.release_trainer(f'{s.run}/flow_fitARB') is False
    _, info = _fit(s)
    assert info['round'] == 11


def test_the_old_warm_start_checkpoint_gives_only_its_bookkeeping(tmp_path, trainers, capsys):
    """A flow branch state.pt (schema aimmd.flow_fit/1): round counter and held-out chains continue, weights not."""
    s = _setup(tmp_path, n_chains=4, n_shots=2)
    state_dir = Path(f'{s.run}/flow_fitARB')
    state_dir.mkdir()
    other = _state(_randomized(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), seed=9))
    torch.save({'schema': 'aimmd.flow_fit/1', 'round': 4, 'held_out_chains': [3],
                'noop_done': True, 'n_trained_shots': 6, 'live': other, 'ema': other,
                'step': 400}, state_dir / 'state.pt')
    _, info = _fit(s)
    assert 'warm-start checkpoint of the flow branch' in capsys.readouterr().out
    assert info['round'] == 5 and info['record']['held_out_chains'] == [3]
    assert _equal(trainers[-1].start_state, s.start)
    assert _state_file(s)['round'] == 5                          # state.json from now on


def test_metadata_complete(tmp_path):
    """network_metadata names every setting, the seed and where it came from, the start and the network."""
    s = _setup(tmp_path, n_chains=4, n_shots=3)
    kwargs = dict(basin_draw='density', holdout_every=2, relabel_every=2, split_backward=True,
                  exclude_frames={s.rd.shots[0].fnames[0]: {1, 2}}, t_schedule='power:2')
    losses, info = _fit(s, **kwargs)
    metadata = info['network_metadata']
    assert json.loads(json.dumps(metadata)) == metadata             # plain JSON
    assert set(metadata['settings']) == set(ff.FLOW_FIT_DEFAULTS)
    assert metadata['settings']['basin_draw'] == 'density'
    assert metadata['settings']['exclude_frames'] == {s.rd.shots[0].fnames[0]: [1, 2]}
    assert metadata['settings']['steps_per_round'] == FAST['steps_per_round']
    assert metadata['keywords']['t_schedule'] == 'power:2'
    assert set(metadata['keywords']) == set(FAST) | set(kwargs)
    assert metadata['round'] == 1 and metadata['steps'] == 3 and metadata['status'] == 'trained'
    assert metadata['fit'] == 'aimmd.cgaimmd.flow_matching.fit'
    seed = aimmd.cgaimmd.fitting.fit_seed(s.params, s.run)
    assert (metadata['seed'], metadata['seed_source']) == seed and seed[1] == 'run folder'
    assert metadata['round_seed'] == metadata['seed']
    assert metadata['start']['kind'] == 'initial_network'
    assert metadata['series'] == 'descriptors' and metadata['ema_decay'] == 0.999
    assert metadata['held_out_chains'] == info['record']['held_out_chains']
    assert metadata['n_trained_shots'] == info['record']['n_trained_shots'] == 9
    assert metadata['network'] == {'class': 'aimmd.cgaimmd.flow_matching.toy.TinyFlowNetwork',
                                   'n_parameters': sum(p.numel() for p in s.network.parameters()),
                                   'n_atoms': 1, 'lambda_clip': 10.0}
    assert metadata['early_stopping']['every'] == 2
    record = _rounds(s)[0]
    for key in ('round', 'status', 'steps', 'skipped_steps', 'fm_dropped', 'loss_terms',
                'holdout_nll_ema', 'holdout_nll_live', 'n_trained_shots', 'held_out_chains',
                'relabels_in_round', 'start', 'seed', 'round_seed', 'data', 'seconds',
                'device', 'time_end', 'early_stopping'):
        assert key in record, key
    assert set(record['seconds']) >= {'data', 'start', 'relabel', 'train', 'total'}
    state = _state_file(s)
    assert state['schema'] == 'aimmd.flow_fit/2' and state['round'] == 1
    assert state['seed'] == seed[0] and state['keywords']['basin_draw'] == 'density'


def test_seeds_of_replicates_differ_and_a_seed_reproduces(tmp_path):
    """The seed comes from trainer_seed, else generation_seed, else the run folder: replicates differ."""
    a = _setup(tmp_path / 'replicate1')
    b = _setup(tmp_path / 'replicate2')
    _, info_a = _fit(a)
    _, info_b = _fit(b)
    assert info_a['network_metadata']['seed'] != info_b['network_metadata']['seed']
    c = _setup(tmp_path / 'replicate3', cgaimmd_parameters={'generation_seed': 77})
    _, info_c = _fit(c)
    assert (info_c['network_metadata']['seed'], info_c['network_metadata']['seed_source']) == \
        (77, 'generation_seed')
    first, _ = _fit(_setup(tmp_path / 'again1'), trainer_seed=3)
    second, _ = _fit(_setup(tmp_path / 'again2'), trainer_seed=3)
    assert first == second


# --------------------------------------------------------------------------------------------
# one round
# --------------------------------------------------------------------------------------------

class _BufferedNet(TinyFlowNetwork):
    """TinyFlowNetwork with a buffer that training-mode forward calls change (as running statistics do)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('calls', torch.zeros(()))

    def forward(self, inp):
        if self.training:
            self.calls += 1.0
        return super().forward(inp)


def test_ema_is_network(tmp_path, trainers):
    """params.network holds the EMA of the live weights; decay None: the live weights.

    Decay 0.9 tells ``d ema + (1 - d) live`` from the swapped roles; the
    buffers are copied from the live network, not averaged.
    """
    torch.manual_seed(0)
    s = _setup(tmp_path, network=_randomized(_BufferedNet(n_atoms=1, lambda_clip=10.0)))
    losses, info = _fit(s, steps_per_round=1, ema_decay=0.9)
    assert len(losses) == 1 and info['status'] == 'trained'
    live = _state(trainers[-1].live)
    after = _state(s.network)
    for name, _ in s.network.named_parameters():
        assert torch.allclose(after[name], 0.9 * s.start[name] + 0.1 * live[name], atol=1e-7)
    assert not all(torch.equal(s.start[k], live[k]) for k in s.start)
    assert float(live['calls']) > 0 and torch.equal(after['calls'], live['calls'])
    t = _setup(tmp_path / 'follow')
    _fit(t, ema_decay=None)
    assert _equal(_state(t.network), _state(trainers[-1].live))


def test_info_counts_training_shots(tmp_path, trainers):
    s = _setup(tmp_path, n_chains=4, n_shots=3)
    losses, info = _fit(s)
    held = set(trainers[-1].held_out_chains)
    assert len(held) == 1                                # max(1, round(0.1 * 4))
    n_train = int(sum(g not in held for g in trainers[-1].dataset.sp_group))
    assert info['record']['n_trained_shots'] == n_train == 9
    assert info['status'] == 'trained' and len(losses) == 3
    assert info['record']['holdout_nll_ema'] is not None


def test_committor_terms_use_forward(tmp_path, monkeypatch):
    """The committor terms go through forward (AIMMD's input path); committor is never called."""
    class Net(TinyFlowNetwork):
        def committor(self, x):
            x = torch.as_tensor(x, dtype=torch.float32)
            return self.forward(x.reshape(len(x), -1))[:, 0]
    torch.manual_seed(0)
    s = _setup(tmp_path, network=Net(n_atoms=1, lambda_clip=10.0))
    _fit(s)                                              # first call: start-up checks
    monkeypatch.setattr(Net, 'committor', lambda self, x: pytest.fail('committor called'))
    losses, info = _fit(s, grad_penalty=0.1)
    assert info['status'] == 'trained' and len(losses) == 3


def test_gradient_penalty_dense(tmp_path, trainers):
    """L_gp = mean((d sum lambda / d x)^2) on the SP batch, differentiated into the weights."""
    s = _setup(tmp_path)
    trainer_fit = _fit(s, steps_per_round=1, grad_penalty=0.1)
    trainer = trainers[-1]
    terms = trainer.last_terms
    assert 'L_gradpen' in terms and terms['L_gradpen'] >= 0.0
    assert trainer_fit[1]['record']['loss_terms']['L_gradpen'] == pytest.approx(terms['L_gradpen'])
    rows = trainer.dataset.sp_rows[:3]
    lam, pos = trainer.committor_fn(trainer.live, rows, True)
    (grad,) = torch.autograd.grad(lam.sum(), pos)
    assert grad.shape == (3, 3)
    t = _setup(tmp_path / 'none')
    _fit(t, steps_per_round=1, grad_penalty=0.0)
    assert 'L_gradpen' not in trainers[-1].last_terms


def test_instate_conditioning(tmp_path, monkeypatch, trainers):
    """Basin FM samples are conditioned at -+L; without R frames every FM sample is a basin one."""
    calls = []
    real = TinyFlowNetwork.velocity

    def spy(self, x_t, t, lam, *, uncond_mask=None):
        calls.append(None if lam is None else lam.detach().clone())
        return real(self, x_t, t, lam, uncond_mask=uncond_mask)
    monkeypatch.setattr(TinyFlowNetwork, 'velocity', spy)
    s = _setup(tmp_path)
    _fit(s, steps_per_round=1, n_fm=10, instate_fm_share=0.2, fm_max_sample_loss=None)
    lam = calls[-1]
    assert lam.tolist()[:2] == [-10.0, 10.0]             # round(0.2 x 10) = 2 basin frames
    assert bool((lam[2:].abs() < 10.0).all())            # R frames: their clipped lambda-hat
    trainer = trainers[-1]
    assert set(trainer.dataset.fr_state[trainer.fm_pool].tolist()) == {0}
    calls.clear()
    trainer.set_fm_targets(np.zeros(0, dtype=np.int64), np.zeros(0))
    trainer.step()
    assert sorted(calls[-1].tolist()) == [-10.0] * 5 + [10.0] * 5


def test_holdout_by_chain_stable(tmp_path, trainers):
    """Held-out chains: drawn once with the fixed seed, persisted, out of the NLL and the FM pool."""
    s = _setup(tmp_path, n_chains=5, n_shots=2)
    _fit(s)
    trainer = trainers[-1]
    chains = sorted(set(trainer.dataset.sp_group.tolist()))
    expected = sorted(np.random.default_rng(12345).choice(chains, 1, replace=False).tolist())
    assert list(trainer.held_out_chains) == expected
    held = expected[0]
    assert all(trainer.dataset.sp_group[i] != held for i in trainer.sp_train)
    assert all(trainer.dataset.sp_group[i] == held for i in trainer.sp_hold)
    assert not bool((trainer.dataset.fr_group[trainer.fm_pool] == held).any())
    anchors = np.concatenate([trainer.anchor_first, trainer.anchor_last])
    assert bool((trainer.dataset.fr_group[anchors] == held).any())      # anchors keep them
    assert _state_file(s)['held_out_chains'] == expected
    # a new process keeps the chains; a new chain is a training chain
    ff._RUNS.clear()
    rd = build_cgaimmd_run_dir(tmp_path / 'more', n_chains=7, n_shots=2)
    shutil.copytree(f'{rd.run}/chainR6', f'{s.run}/chainR6')
    s.pe = _ensemble(s.params, s.run)
    _fit(s, hold_out_anchor_frames=True)
    trainer = trainers[-1]
    assert list(trainer.held_out_chains) == expected
    anchors = np.concatenate([trainer.anchor_first, trainer.anchor_last])
    assert not bool((trainer.dataset.fr_group[anchors] == held).any())
    u = _setup(tmp_path / 'named', n_chains=3)
    _fit(u, holdout_chains=[0, 2])
    assert list(trainers[-1].held_out_chains) == [0, 2]


def test_internal_shooting_chains_are_not_held_out(tmp_path):
    """Groups are reactive-chain numbers: chain<A>0 frames are group -1, never held out with chainR0."""
    s = _setup(tmp_path, n_chains=2, n_shots=2)
    shutil.copytree(f'{s.run}/chainR0', f'{s.run}/chainA0')
    s.pe = _ensemble(s.params, s.run)
    data = extract_dataset(s.pe, states='ARB', n_atoms=1)
    internal = np.array(['chainA0' in f for f in data.fr_fname])
    assert internal.any() and (data.fr_group[internal] == -1).all()
    trainer = ff.FlowTrainer(s.network, ff._settings_from({}), lambda_clip=10.0, device='cpu',
                             committor_fn=ff._MDFrameInput(s.params, False))
    trainer.set_data(data, (0,), hold_out_anchor_frames=True)
    pool = set(trainer.fm_candidates.tolist())
    anchors = set(np.concatenate([trainer.anchor_first, trainer.anchor_last]).tolist())
    chain_r0 = np.array(['chainR0' in f for f in data.fr_fname])
    assert not (set(np.flatnonzero(chain_r0).tolist()) & (pool | anchors))
    assert set(np.flatnonzero(internal & (data.fr_state != 0)).tolist()) <= anchors
    assert set(np.flatnonzero(internal & (data.fr_state == 0)).tolist()) <= pool


def test_margins_never_anchors(tmp_path, monkeypatch, trainers):
    """The trainer's margins (initial-path frames) have reweighting weight 0: never anchors."""
    s = _setup(tmp_path)
    positions = np.zeros((4, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = [-1.2, -1.1, 1.1, 1.2]
    (Path(s.run) / 'initialARB').mkdir()
    fname = write_trajectory(Path(s.run) / 'initialARB', stem='initial', ext='.xtc',
                             positions=positions, times=np.arange(4.0))
    stored = aimmd.Path(fname).positions
    save_npy(get_cache_fname(fname, 'states'), state_labels_from_positions(stored, states='ARB'))
    save_npy(get_cache_fname(fname, 'descriptors'), stored.reshape(4, -1).astype(np.float32))
    margins = get_initial_frames_for_training(PathEnsemble(fname), 'ARB')

    def broken(*args, **kwargs):
        raise RuntimeError('no reweighting')
    for uniform in (False, True):
        ff._RUNS.clear()
        if uniform:                                      # the uniform fallback keeps them out too
            monkeypatch.setattr(ff, 'path_weights', broken)
        _fit(SimpleNamespace(**{**s.__dict__, 'pe': s.pe + margins}))
        trainer = trainers[-1]
        anchors = np.concatenate([trainer.anchor_first, trainer.anchor_last])
        assert len(anchors) and fname not in set(trainer.dataset.fr_fname[anchors].tolist())


def test_reweight_failure_uniform(tmp_path, monkeypatch, capsys, trainers):
    s = _setup(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError('nothing to reweight')
    monkeypatch.setattr(type(s.pe), 'reweight', broken)
    losses, info = _fit(s)
    assert info['status'] == 'trained'
    out = capsys.readouterr().out
    assert 'flow-matching fit: reweighting failed' in out and 'weighted uniformly' in out
    assert set(np.unique(trainers[-1].dataset.fr_w).tolist()) <= {1.0, 5.0}     # 5: free stride


def test_fill_missing_values(tmp_path):
    """Missing R-frame values (0.0) are computed with params.values_function before reweighting."""
    s = _setup(tmp_path)
    fname = s.rd.shots[0].fnames[0]
    values = np.load(get_cache_fname(fname, 'values'))
    values[2] = 0.0
    np.save(get_cache_fname(fname, 'values'), values)
    _fit(s, fill_missing_values=False)
    assert np.load(get_cache_fname(fname, 'values'))[2] == 0.0
    _fit(s)
    assert np.load(get_cache_fname(fname, 'values'))[2] != 0.0


@pytest.mark.parametrize('stored, returned', [(np.float64, np.float32), (np.float32, np.float64)])
def test_fill_missing_values_casts_to_the_cache_dtype(tmp_path, stored, returned):
    """Missing values are written in the dtype of the values caches, whatever the values function returns."""
    s = _setup(tmp_path)
    base = create_default_values_function(s.network)

    def values_function(descriptors, *args, **kwargs):
        return (np.asarray(base(descriptors, *args, **kwargs)) * 0.0 + 0.25).astype(returned)
    s.params.__dict__['values_function'] = values_function
    for path in list(s.rd.shots) + list(s.rd.free):
        fname = get_cache_fname(path.fnames[0], 'values')
        np.save(fname, np.load(fname).astype(stored))
    fname = get_cache_fname(s.rd.shots[0].fnames[0], 'values')
    values = np.load(fname)
    values[2] = 0.0
    np.save(fname, values)
    _, info = _fit(s)
    after = np.load(fname)
    assert info['status'] == 'trained' and after.dtype == stored
    assert after[2] == pytest.approx(0.25)


def test_values_caches_of_two_dtypes_raise(tmp_path):
    """A missing value in a cache of another dtype than the run's first cache: RuntimeError naming both."""
    s = _setup(tmp_path)
    files = [path.fnames[0] for path in list(s.rd.shots) + list(s.rd.free)]
    for k, fname in enumerate(files):
        cache = get_cache_fname(fname, 'values')
        np.save(cache, np.load(cache).astype(np.float32 if k == 0 else np.float64))
    first = ff._values_cache_dtype(s.pe)
    odd = files[0] if first == np.float64 else files[1]       # a cache of the other dtype
    cache = get_cache_fname(odd, 'values')
    values = np.load(cache)
    values[2] = 0.0
    np.save(cache, values)
    with pytest.raises(RuntimeError, match='more than one dtype') as info:
        _fit(s)
    assert "'<f4'" in str(info.value) and "'<f8'" in str(info.value)
    _, info = _fit(s, fill_missing_values=False)                 # the documented way out
    assert info['status'] == 'trained'


def test_exclude_frames(tmp_path, monkeypatch, trainers):
    """exclude_frames removes frames and SPs; a relative JSON path is found next to the params file.

    The JSON path is looked up in ``params.parent`` (the working directory
    for params without a file, like the test's placeholder params), the file
    names inside it against the run directory.
    """
    s = _setup(tmp_path)
    whole = s.rd.shots[0].fnames[0]
    partial = s.rd.shots[1].fnames[0]
    spec = {'exclude': {os.path.relpath(whole, s.run): 'all',
                        os.path.relpath(partial, s.run): [0, 3]}}
    params_dir = tmp_path / 'params_dir'
    params_dir.mkdir()
    (params_dir / 'exclude.json').write_text(json.dumps(spec))
    monkeypatch.chdir(params_dir)                        # placeholder params: parent is '.'
    assert os.path.samefile(ff._params_directory(s.params), params_dir)
    _fit(s, exclude_frames='exclude.json')
    data = trainers[-1].dataset
    assert whole not in set(data.fr_fname.tolist()) and whole not in data.sp_fname
    locs = data.fr_loc[data.fr_fname == partial]
    assert 0 not in locs and 3 not in locs and 1 in locs
    assert partial not in data.sp_fname                   # its shooting frame (3) is excluded
    assert data.notes['excluded_shooting_points'] == 2
    monkeypatch.chdir(tmp_path)                          # not relative to the run directory
    with pytest.raises(FileNotFoundError, match='exclude.json'):
        _fit(s, exclude_frames='exclude.json')
    monkeypatch.setattr(type(s.params), 'parent', property(lambda self: params_dir))
    _, info = _fit(s, exclude_frames='exclude.json')      # a params file's own directory
    assert info['record']['data']['excluded_shooting_points'] == 2


def test_relabelling_uses_the_ema_clipped(tmp_path, trainers):
    """The labels are the EMA network's committor of the pool frames, clipped to -+L (not the live one)."""
    torch.manual_seed(0)
    s = _setup(tmp_path, network=_steep(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0)))
    seen = {}
    original = ff._relabel

    def spy(context, data):
        trainer = context['trainer']
        if trainer.step_count == 4:
            seen['ema'] = copy.deepcopy(trainer.network).eval()
            seen['live'] = copy.deepcopy(trainer.live).eval()
        return original(context, data)
    ff._relabel = spy
    try:
        _fit(s, ema_decay=0.9, lr=1e-2, steps_per_round=5, relabel_every=4)
    finally:
        ff._relabel = original
    trainer = trainers[-1]
    pool, rows = trainer.fm_pool, trainer.dataset.fr_rows[trainer.fm_pool]
    with torch.no_grad():
        lam_ema = seen['ema'](torch.as_tensor(rows))[:, 0].double().numpy()
        lam_live = seen['live'](torch.as_tensor(rows))[:, 0].double().numpy()
    assert len(pool) and (np.abs(lam_ema) > 10.0).any() and (np.abs(lam_ema) < 10.0).any()
    assert np.abs(lam_ema - lam_live).max() > 1e-3                   # live and EMA differ
    assert np.allclose(trainer.fm_lambda, np.clip(lam_ema, -10.0, 10.0), atol=1e-5)
    assert trainer.fm_lambda.max() == 10.0 or trainer.fm_lambda.min() == -10.0


def _steep(net, slope=4.0, scale=2.0, offset=-10.0):
    """Committor weights with logits beyond -+10 on the toy R frames: 16 scale silu(slope x) + offset."""
    with torch.no_grad():
        first, last = net.committor_net[0], net.committor_net[2]
        first.weight.fill_(slope)
        first.bias.zero_()
        last.weight.fill_(scale)
        last.bias.fill_(offset)
    return net


def test_holdout_nll_of_ema_and_live(tmp_path, trainers):
    """The round record scores the held-out shooting points with the EMA and with the live weights."""
    s = _setup(tmp_path, n_chains=4, n_shots=3)
    _, info = _fit(s, ema_decay=0.9, lr=1e-2, steps_per_round=5)
    trainer = trainers[-1]
    record = info['record']
    assert len(trainer.sp_hold)
    assert record['holdout_nll_ema'] == trainer.holdout_nll(trainer.network)
    assert record['holdout_nll_live'] == trainer.holdout_nll(trainer.live)
    assert record['holdout_nll_ema'] != record['holdout_nll_live']


def test_joint_painn_round(tmp_path):
    """The PaiNN network trains in dense mode (OT pairing, to_flow, the Gaussian prior)."""
    torch.manual_seed(0)
    net = golden.build_network(golden.new_api(), 'painn', 3)
    s = _setup(tmp_path, network=net, n_atoms=3)
    losses, info = _fit(s, grad_penalty=0.1)
    assert info['status'] == 'trained' and all(np.isfinite(losses))
    assert set(_state(s.network)) == set(s.start)


def test_posttrain_under_rfps_and_tps(tmp_path):
    """Post-training: the fit validates the protocol itself and runs under any chain type."""
    for chain_type in ('rfps', 'tps'):
        s = _setup(tmp_path / chain_type, chain_type=chain_type,
                   cgaimmd_parameters={'lambda_clip': 10.0, 'training_device': 'cpu'})
        losses, info = _fit(s)
        assert info['status'] == 'trained' and len(losses) == 3


# --------------------------------------------------------------------------------------------
# the trainer on a hand-built dataset
# --------------------------------------------------------------------------------------------

def test_step_terms_replay():
    """Every loss term of one step equals its formula, replayed from the same draws."""
    torch.manual_seed(0)
    api = golden.new_api()
    net = golden.build_network(api, 'paired', 2)
    weights = dict(committor_weight=1.3, anchor_weight=0.7, grad_penalty=0.2, fm_weight=0.9)
    trainer = _hand(net, p_uncond=0.5, instate_fm_share=1 / 3, **weights)
    trainer.step()                                       # the normalizer has magnitudes now
    rng_state = copy.deepcopy(trainer.rng.bit_generator.state)
    torch_state = trainer.generator.get_state()
    live = copy.deepcopy(trainer.live)
    normalizer = copy.deepcopy(trainer.normalizer)
    total, terms = trainer.step()

    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    g = torch.Generator()
    g.set_state(torch_state)
    data, L, m = trainer.dataset, 10.0, 2
    i = rng.choice(trainer.sp_train, 6)
    x = torch.as_tensor(data.sp_rows[i]).requires_grad_(True)
    lam = live(x)[:, 0]
    first, last = torch.as_tensor(data.sp_r_first[i]), torch.as_tensor(data.sp_r_last[i])
    l_sp = ff.committor_nll(lam, last / (first + last), first + last)
    (grad,) = torch.autograd.grad(lam.sum(), x, create_graph=True)
    l_gp = grad.pow(2).mean()
    ia, ib = rng.choice(trainer.anchor_first, 2), rng.choice(trainer.anchor_last, 2)
    lam_a = live(torch.as_tensor(np.concatenate([data.fr_rows[ia], data.fr_rows[ib]])))[:, 0]
    y = torch.sigmoid(torch.tensor([-L, -L, L, L]))
    l_anchor = ff.committor_nll(lam_a, y, torch.ones(4))
    saved, trainer.rng = trainer.rng, rng
    j, lam_hat = trainer._draw_fm(6)
    trainer.rng = saved
    lam_t = torch.as_tensor(lam_hat)
    u = torch.rand(6, generator=g) < 0.5
    x0 = live.sample_prior(lam_t, 6, uncond_mask=u, generator=g)
    x1 = torch.as_tensor(data.fr_x[j]).flip(1)
    t = flowmath.sample_time(6, 'uniform', generator=g)
    x_t = (1 - t).reshape(-1, 1, 1) * x0 + t.reshape(-1, 1, 1) * x1
    v = live.velocity(x_t, t, lam_t, uncond_mask=u)
    l_fm = ((v - (x1 - x0)) ** 2).sum(dim=(1, 2)).div(m).mean()

    assert terms['L_sp'] == pytest.approx(float(l_sp), rel=1e-5)
    assert terms['L_gradpen'] == pytest.approx(float(l_gp), rel=1e-5)
    assert terms['L_anchor'] == pytest.approx(float(l_anchor), rel=1e-5)
    assert terms['L_FM'] == pytest.approx(float(l_fm), rel=1e-5)
    lq = 1.3 * l_sp + 0.2 * l_gp + 0.7 * l_anchor
    expected = normalizer('q', lq.detach()) + 0.9 * normalizer('fm', l_fm.detach())
    assert total == pytest.approx(float(expected), rel=1e-5)
    assert float(normalizer('q', lq.detach())) != pytest.approx(float(lq))   # N_q is not 1
    assert 'skipped' not in terms and trainer.skipped_steps == 0


def test_committor_direction():
    """Trained on (2, 0) at -0.3, (0, 2) at +0.3 and the anchors, lambda increases toward the last state."""
    torch.manual_seed(0)
    net = TinyFlowNetwork(n_atoms=1, lambda_clip=10.0)
    trainer = _hand(net, m=1, lr=1e-2, ema_decay=None, n_sp=8, n_anchor=8, n_fm=8)
    for _ in range(200):
        trainer.step()
    with torch.no_grad():
        lam = trainer.network(torch.as_tensor(golden.rows([-1.0, -0.3, 0.3, 1.0], 1)))[:, 0]
    lam = lam.tolist()
    assert lam[0] < lam[1] < 0.0 < lam[2] < lam[3], lam


def test_lr_factor_warmup_and_cosine():
    """Linear warm-up to the base rate, then constant or a cosine to 0 at total_steps (default: the round)."""
    net = TinyFlowNetwork(n_atoms=1, lambda_clip=10.0)
    cosine = _hand(net, m=1, warmup_steps=10, lr_schedule='cosine', total_steps=110)
    assert [cosine._lr_factor(k) for k in (0, 4, 9)] == pytest.approx([0.1, 0.5, 1.0])
    assert cosine._lr_factor(10) == pytest.approx(1.0)
    assert cosine._lr_factor(60) == pytest.approx(0.5)
    assert cosine._lr_factor(110) == pytest.approx(0.0, abs=1e-12)
    assert cosine._lr_factor(500) == pytest.approx(0.0, abs=1e-12)
    round_long = _hand(net, m=1, warmup_steps=10, lr_schedule='cosine', steps_per_round=110)
    assert [round_long._lr_factor(k) for k in (10, 60, 110)] == pytest.approx([1.0, 0.5, 0.0],
                                                                                abs=1e-12)
    shorter = _hand(net, m=1, warmup_steps=10, lr_schedule='cosine', steps_per_round=200,
                    total_steps=110)                     # anneals early, then lr 0 to the end
    assert [shorter._lr_factor(k) for k in (60, 110, 199)] == pytest.approx([0.5, 0.0, 0.0],
                                                                             abs=1e-12)
    constant = _hand(net, m=1, warmup_steps=4)
    assert [constant._lr_factor(k) for k in (0, 3, 4, 1000)] == pytest.approx([0.25, 1, 1, 1])
    constant.step()
    assert constant.optimizer.param_groups[0]['lr'] == pytest.approx(0.25 * 3e-4)


def test_cosine_longer_than_a_round_is_refused(tmp_path):
    """total_steps > steps_per_round with the cosine is refused: every round restarts the schedule.

    The flow branch's templates set ``total_steps`` to "rounds x steps per
    round" for warm-started rounds; under the restart such a cosine would end
    every round at a learning-rate factor of 0.88 (16,000 against 4,000 steps).
    """
    old_template = dict(steps_per_round=4000, total_steps=16000, warmup_steps=500,
                        lr_schedule='cosine')
    with pytest.raises(ValueError, match='total_steps=16000 is larger than steps_per_round=4000'):
        ff._settings_from(old_template)
    for fine in (dict(old_template, total_steps=4000), dict(old_template, total_steps=None),
                 dict(old_template, total_steps=2000),
                 dict(old_template, lr_schedule='constant')):    # (constant: total_steps unused)
        ff._settings_from(fine)
    s = _setup(tmp_path)
    with pytest.raises(ValueError, match='total_steps=4 is larger than steps_per_round=3'):
        _fit(s, lr_schedule='cosine', total_steps=4)
    assert not os.path.exists(f'{s.run}/flow_fitARB')            # refused before any set-up


def test_fm_runaway_guard():
    """Samples above fm_max_sample_loss are dropped and the loss is that of the kept ones."""
    class Runaway(TinyFlowNetwork):
        def velocity(self, x_t, t, lam, *, uncond_mask=None):
            v = super().velocity(x_t, t, lam, uncond_mask=uncond_mask)
            return v + 1e6 * (t > 0.99).to(v.dtype).reshape(-1, 1, 1)
    torch.manual_seed(0)
    net = _randomized(Runaway(n_atoms=2, lambda_clip=10.0))
    trainer = _hand(net, fm_max_sample_loss=1e3)
    g = torch.Generator().manual_seed(1)
    x0, x1 = torch.randn(4, 2, 3, generator=g), torch.randn(4, 2, 3, generator=g)
    t = torch.tensor([0.2, 0.999, 0.5, 0.7])
    lam, u = torch.zeros(4), torch.zeros(4, dtype=torch.bool)
    x_t, target = (1 - t).reshape(-1, 1, 1) * x0 + t.reshape(-1, 1, 1) * x1, x1 - x0
    loss, dropped = trainer._fm_loss(x_t, t, lam, u, target)
    keep = torch.tensor([True, False, True, True])
    v = TinyFlowNetwork.velocity(trainer.live, x_t[keep], t[keep], lam[keep],
                                 uncond_mask=u[keep])
    assert dropped == 1
    assert float(loss) == pytest.approx(float(((v - target[keep]) ** 2).sum(dim=(1, 2))
                                              .div(2).mean()), rel=1e-6)
    loss, dropped = trainer._fm_loss(x_t, torch.full((4,), 0.995), lam, u, target)
    assert loss is None and dropped == 4
    nan = x_t.clone()
    nan[0] = float('nan')
    loss, dropped = trainer._fm_loss(nan, t.clamp_max(0.9), lam, u, target)
    assert dropped == 1 and torch.isfinite(loss)            # NaN samples are dropped too
    trainer.settings['fm_max_sample_loss'] = None
    loss, dropped = trainer._fm_loss(x_t, t, lam, u, target)
    assert dropped == 0 and float(loss) > 1e10             # the guard off: the runaway counts


@pytest.mark.parametrize('split', [False, True])
def test_nonfinite_gradient_step_skipped(split):
    """A step with a non-finite gradient norm changes neither the weights nor the EMA; it is counted."""
    class Broken(TinyFlowNetwork):
        def forward(self, inp):
            return super().forward(inp) * float('nan')
    torch.manual_seed(0)
    net = Broken(n_atoms=2, lambda_clip=10.0)
    trainer = _hand(net, ema_decay=0.5, split_backward=split)
    live, ema = _state(trainer.live), _state(trainer.network)
    total, terms = trainer.step()
    assert terms['skipped'] == 1.0 and trainer.skipped_steps == 1 and trainer.step_count == 1
    assert _equal(live, _state(trainer.live)) and _equal(ema, _state(trainer.network))
    assert not trainer.optimizer.state                     # no optimizer step was taken


def test_fm_draws_strata_uniform_and_weights():
    """R frames of the FM batch: a stratum uniformly among the non-empty ones, a frame by weight."""
    torch.manual_seed(0)
    net = TinyFlowNetwork(n_atoms=1, lambda_clip=10.0)
    trainer = _hand(net, m=1, n_lambda_bins=4)
    data = trainer.dataset
    pool = np.array([6, 7, 8, 9])                          # weights 1, 3, 1, 2
    trainer.set_fm_targets(pool, np.array([-2.0, -1.0, 7.0, 7.5]))   # strata [-5, 0) and [5, 10]
    trainer.settings['instate_fm_share'] = 0.0
    j, lam = trainer._draw_fm(20000)
    assert set(j.tolist()) <= set(pool.tolist())
    low = np.isin(j, [6, 7])
    assert low.mean() == pytest.approx(0.5, abs=0.02)                  # strata uniform
    assert (j[low] == 7).mean() == pytest.approx(3 / 4, abs=0.02)      # weights 1 : 3
    assert (j[~low] == 9).mean() == pytest.approx(2 / 3, abs=0.02)     # weights 1 : 2
    assert set(np.round(lam[j == 8], 3).tolist()) == {7.0}             # its own lambda-hat
    trainer.settings['instate_fm_share'] = 0.5
    j, lam = trainer._draw_fm(10)
    assert lam[:5].tolist() == [-10.0, -10.0, 10.0, 10.0, 10.0]      # half per state at -+L
    assert set(data.fr_state[j[:5]].tolist()) == {1, 2}


def _frequencies(indices, members):
    indices = np.asarray(indices)
    return np.array([(indices == k).mean() for k in members])


@pytest.mark.parametrize('draw, first, last', [
    ('uniform', [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]),
    ('density', [1 / 8, 6 / 8, 1 / 8], [2 / 4, 1 / 4, 1 / 4])])
def test_basin_draws(draw, first, last):
    """Basin frames uniform by default; with basin_draw='density' proportional to fr_w.

    The hand dataset's in-state weights are 1 : 6 : 1 (first state) and 2 : 1 : 1 (last state).
    """
    torch.manual_seed(0)
    trainer = _hand(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), m=1, basin_draw=draw)
    np.testing.assert_allclose(_frequencies(trainer._draw_basin(1, 40000), [0, 1, 2]), first,
                               atol=0.01)
    np.testing.assert_allclose(_frequencies(trainer._draw_basin(2, 40000), [3, 4, 5]), last,
                               atol=0.01)
    trainer.settings['instate_fm_share'] = 1.0
    j, lam = trainer._draw_fm(40000)
    np.testing.assert_allclose(_frequencies(j[lam < 0], [0, 1, 2]), first, atol=0.015)
    np.testing.assert_allclose(_frequencies(j[lam > 0], [3, 4, 5]), last, atol=0.015)
    trainer.settings['n_anchor'] = 4000
    seen = []
    inner = trainer.committor_fn

    def spy(net, rows, requires_grad):
        if not requires_grad and len(rows) == 4000:
            seen.append(np.round(np.asarray(rows)[:, 0], 3))
        return inner(net, rows, requires_grad)
    trainer.committor_fn = spy
    trainer.step()
    xs = trainer.dataset.fr_x[:, 0, 0].round(3)
    anchors = seen[0]
    np.testing.assert_allclose([(anchors[:2000] == xs[k]).mean() for k in (0, 1, 2)], first,
                               atol=0.04)
    np.testing.assert_allclose([(anchors[2000:] == xs[k]).mean() for k in (3, 4, 5)], last,
                               atol=0.04)


def test_basin_draw_uniform_is_the_old_stream():
    """The default draws the numbers the fit drew before the keyword existed (bitwise)."""
    torch.manual_seed(0)
    trainer = _hand(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), m=1)
    assert trainer.settings['basin_draw'] == 'uniform' == ff.FLOW_FIT_DEFAULTS['basin_draw']
    state = trainer.rng.bit_generator.state
    old = np.random.default_rng()
    old.bit_generator.state = state
    expected = old.choice(trainer.anchor_first, 7)
    np.testing.assert_array_equal(trainer._draw_basin(1, 7), expected)


def test_basin_draw_validated(tmp_path):
    s = _setup(tmp_path)
    with pytest.raises(ValueError, match='basin_draw'):
        ff.flow_matching_fit(s.params, s.pe, worker=s.worker, basin_draw='equilibrium')


def test_basin_draw_density_in_a_fit(tmp_path, trainers):
    """The keyword reaches the trainer through the fit and is recorded."""
    s = _setup(tmp_path)
    _, info = _fit(s, basin_draw='density')
    trainer = trainers[-1]
    assert trainer.settings['basin_draw'] == 'density'
    w = trainer.dataset.fr_w[trainer.anchor_first]
    np.testing.assert_allclose(trainer._basin_p[1], w / w.sum())
    assert info['network_metadata']['settings']['basin_draw'] == 'density'
    assert _state_file(s)['keywords']['basin_draw'] == 'density'


def test_sp_draws_skip_held_out_chains():
    """Shooting points of held-out chains never enter the NLL; they are scored by holdout_nll."""
    seen = []
    torch.manual_seed(0)
    trainer = _hand(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), m=1, held=(1,),
                    grad_penalty=0.1)
    inner = trainer.committor_fn

    def spy(net, rows, requires_grad):
        if requires_grad:                                   # the SP term (grad_penalty > 0)
            seen.append(np.asarray(rows)[:, 0].copy())
        return inner(net, rows, requires_grad)
    trainer.committor_fn = spy
    for _ in range(30):
        trainer.step()
    drawn = np.concatenate(seen)
    assert len(seen) == 30 and not np.isin(np.round(drawn, 3), [-0.35, 0.35]).any()
    assert set(trainer.sp_hold.tolist()) == {4, 5}
    assert trainer.holdout_nll(trainer.network) is not None


def test_p_uncond_masks(monkeypatch):
    """p_uncond is the probability of the null token in flow matching: 1 -> every row, 0 -> none."""
    masks = []
    real = TinyFlowNetwork.velocity

    def spy(self, x_t, t, lam, *, uncond_mask=None):
        masks.append(uncond_mask.clone())
        return real(self, x_t, t, lam, uncond_mask=uncond_mask)
    monkeypatch.setattr(TinyFlowNetwork, 'velocity', spy)
    for p, expected in ((1.0, True), (0.0, False)):
        masks.clear()
        trainer = _hand(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), m=1, p_uncond=p)
        trainer.step()
        assert masks and all(bool((mask == expected).all()) for mask in masks)
    trainer = _hand(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0), m=1, p_uncond=0.5,
                    n_fm=4000)
    masks.clear()
    trainer.step()
    assert float(masks[0].float().mean()) == pytest.approx(0.5, abs=0.04)


# --------------------------------------------------------------------------------------------
# held-out evaluation and early stopping
# --------------------------------------------------------------------------------------------

def test_holdout_every_logs_and_does_not_change_the_training(tmp_path):
    """holdout_every alone records the held-out curve; the training is bit for bit that without it."""
    results = []
    for every in (0, 2):
        s = _setup(tmp_path / f'every{every}', n_chains=4, n_shots=3)
        losses, info = _fit(s, steps_per_round=5, holdout_every=every, lr=1e-2, trainer_seed=1)
        results.append((losses, _state(s.network), info))
    assert results[0][0] == results[1][0] and _equal(results[0][1], results[1][1])
    curve = results[1][2]['record']['early_stopping']['curve']
    assert [step for step, _ in curve] == [2, 4, 5]                 # every 2 steps and the end
    assert curve[-1][1] == results[1][2]['record']['holdout_nll_ema']
    assert results[1][2]['record']['early_stopping']['restored'] is False
    assert results[0][2]['record']['early_stopping'] is None


def test_early_stopping_restores_the_best_ema(tmp_path, monkeypatch):
    """With early_stopping_patience the round stops after that many evaluations without a lower held-out NLL
    and the network ends as the EMA of the best evaluation."""
    scripted = iter([3.0, 2.0, 2.5, 2.6, 1.0])
    snapshots = {}
    real = ff.FlowTrainer.holdout_nll

    def holdout_nll(self, net):
        if net is self.network and self.step_count and self.step_count not in snapshots:
            snapshots[self.step_count] = _state(net)
            return next(scripted)
        return real(self, net)
    monkeypatch.setattr(ff.FlowTrainer, 'holdout_nll', holdout_nll)
    s = _setup(tmp_path, n_chains=4, n_shots=3)
    losses, info = _fit(s, steps_per_round=20, holdout_every=2, early_stopping_patience=2,
                        lr=1e-2, ema_decay=0.5)
    stopping = info['record']['early_stopping']
    assert len(losses) == 8 and info['status'] == 'trained'      # evaluations at 2, 4, 6, 8
    assert 'early stopping' in info['reason']
    assert stopping['curve'] == [[2, 3.0], [4, 2.0], [6, 2.5], [8, 2.6]]
    assert stopping['best_step'] == 4 and stopping['best_holdout_nll'] == 2.0
    assert stopping['restored'] is True
    assert _equal(_state(s.network), snapshots[4])
    assert info['network_metadata']['early_stopping']['best_step'] == 4


def test_early_stopping_without_held_out_shooting_points(tmp_path):
    """Without held-out chains nothing is evaluated: the round trains its full budget and says so."""
    s = _setup(tmp_path, n_chains=1, n_shots=4)
    losses, info = _fit(s, steps_per_round=4, holdout_every=1, early_stopping_patience=1)
    assert len(losses) == 4
    assert info['record']['early_stopping']['note'] == ('no held-out shooting points: nothing '
                                                        'evaluated')


# --------------------------------------------------------------------------------------------
# no data, stopping, out of memory, budgets
# --------------------------------------------------------------------------------------------

def test_first_round_noop(tmp_path):
    """A fresh run without basin frames: one no-op success so that the trainer writes the first bins."""
    s = _setup(tmp_path, nbins=4)
    before = _state(s.params.network)
    losses, info = ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker, **FAST)
    assert losses == [0.0] and info['status'] == 'no-op' and info['round'] == 1
    assert _equal(before, _state(s.params.network))
    assert _state_file(s)['noop_done'] is True
    assert info['network_metadata']['round'] == 1 and info['network_metadata']['steps'] == 0
    assert len(_rounds(s)) == 1
    losses, info = ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker, **FAST)
    assert losses == [] and info['status'] == 'skipped' and info['round'] == 1
    assert info['network_metadata'] is None
    assert len(_rounds(s)) == 1
    losses, info = _fit(s)
    assert info['round'] == 2 and info['status'] == 'trained'


def test_noop_memory_per_state_dir(tmp_path):
    s = _setup(tmp_path, nbins=4)
    ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker, **FAST)
    ff._RUNS.clear()
    _, info = ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker, **FAST)
    assert info['status'] == 'skipped'                   # noop_done read from state.json
    _, info = ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker,
                                   state_dir=f'{s.run}/other_state', **FAST)
    assert info['status'] == 'no-op'                     # another state_dir, another memory
    t = _setup(tmp_path / 'with_bins', nbins=4)
    np.save(f'{t.run}/binsARB.npy', np.linspace(-5, 5, 11))
    _, info = ff.flow_matching_fit(t.params, PathEnsemble(), worker=t.worker, **FAST)
    assert info['status'] == 'skipped'                   # bins exist: nobody waits
    u = _setup(tmp_path / 'one_bin', nbins=1)
    _, info = ff.flow_matching_fit(u.params, PathEnsemble(), worker=u.worker, **FAST)
    assert info['status'] == 'skipped'


def test_later_no_data_skips(tmp_path, monkeypatch):
    s = _setup(tmp_path)
    np.save(f'{s.run}/binsARB.npy', np.linspace(-5, 5, 11))
    sleeps = []
    monkeypatch.setattr(ff.time, 'sleep', lambda seconds: sleeps.append(seconds))
    losses, info = ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker,
                                        **{**FAST, 'idle_sleep_s': 3.0})
    assert losses == [] and info['status'] == 'skipped' and info['reason']
    assert sum(sleeps) == pytest.approx(3.0) and max(sleeps) <= 1.0
    s.worker.stop = lambda: True
    sleeps.clear()
    ff.flow_matching_fit(s.params, PathEnsemble(), worker=s.worker,
                         **{**FAST, 'idle_sleep_s': 30.0})
    assert sum(sleeps) <= 1.0                            # must_stop polled during the sleep


def test_kinetics_convergence_uses_the_fraction_state_dir(tmp_path):
    """A kinetics-convergence fit keeps its state in the fraction's folder, never the run's."""
    s = _setup(tmp_path)
    write_fake_network_file(s.run, s.network, round=5)
    s.worker.task = 'kinetics_convergence'
    s.worker.kinetics_convergence_state_dir = str(tmp_path / 'kcv050')
    losses, info = _fit(s, state_dir=str(tmp_path / 'ignored'))
    assert info['status'] == 'trained' and len(losses) == FAST['steps_per_round']
    assert info['round'] == 1                              # the run's round 5 does not count
    assert json.loads((tmp_path / 'kcv050' / 'state.json').read_text())['round'] == 1
    assert not os.path.exists(tmp_path / 'ignored')
    assert not os.path.exists(f'{s.run}/flow_fitARB')
    _, info = _fit(s)
    assert info['round'] == 2
    # without a fraction folder nothing is kept between calls
    s.worker.kinetics_convergence_state_dir = None
    for _ in range(2):
        _, info = _fit(s)
        assert info['round'] == 1
    assert not os.path.exists(f'{s.run}/flow_fitARB')


def test_kinetics_convergence_task_has_no_noop(tmp_path, monkeypatch):
    """Without data a kinetics-convergence round is skipped at once: no no-op, no idle sleep."""
    s = _setup(tmp_path, nbins=4)
    s.worker.task = 'kinetics_convergence'
    s.worker.kinetics_convergence_state_dir = str(tmp_path / 'kcv100')

    def no_sleep(seconds, stopping):
        raise AssertionError('a kinetics-convergence round must not sleep')
    monkeypatch.setattr(ff, '_sleep', no_sleep)
    losses, info = _fit(s, min_basin_frames=10 ** 6, idle_sleep_s=60.0)
    assert losses == [] and info['status'] == 'skipped'
    assert 'min_basin_frames' in info['reason']


def test_must_stop_mid_round(tmp_path, trainers):
    s = _setup(tmp_path)
    s.worker.stop = lambda: bool(trainers) and trainers[-1].step_count >= 2
    losses, info = _fit(s, steps_per_round=10)
    assert len(losses) == 2 and info['status'] == 'stopped' and info['round'] == 1
    assert info['network_metadata']['steps'] == 2 and info['network_metadata']['status'] == 'stopped'
    assert _state_file(s)['round'] == 1


def test_max_round_seconds(tmp_path, monkeypatch):
    """With every step taking 1 s, a budget of 6.5 s from the start of the call runs exactly 7 steps."""
    clock = [0.0]
    monkeypatch.setattr(ff, '_clock', lambda: clock[0])
    real = ff.FlowTrainer.step

    def step(self):
        clock[0] += 1.0
        return real(self)
    monkeypatch.setattr(ff.FlowTrainer, 'step', step)
    s = _setup(tmp_path)
    losses, info = _fit(s, steps_per_round=100, max_round_seconds=6.5)
    assert len(losses) == 7 and info['status'] == 'trained'       # checked before steps 1-8
    assert info['reason'] == 'max_round_seconds=6.5 reached'
    assert info['record']['steps'] == 7 and info['round'] == 1


def test_max_round_seconds_data_step_over_budget(tmp_path, monkeypatch):
    """When the data step alone takes longer than the budget, the round still runs one step."""
    clock = [0.0]
    monkeypatch.setattr(ff, '_clock', lambda: clock[0])
    real = ff._assemble

    def slow(context):
        clock[0] += 100.0
        return real(context)
    monkeypatch.setattr(ff, '_assemble', slow)
    s = _setup(tmp_path)
    losses, info = _fit(s, steps_per_round=10, max_round_seconds=5.0)
    assert len(losses) == 1 and info['status'] == 'trained' and info['round'] == 1
    assert 'max_round_seconds' in info['reason']


def test_oom_restarts_the_round_from_its_start(tmp_path, monkeypatch, trainers):
    """A CUDA OOM restarts the round from its start weights and seed: the result equals a round without it."""
    clean = _setup(tmp_path / 'clean')
    expected, _ = _fit(clean, steps_per_round=5, trainer_seed=2)
    real = ff.FlowTrainer.step
    raised = []

    def step(self):
        if self.step_count == 2 and not raised:
            raised.append(True)
            raise torch.OutOfMemoryError('CUDA out of memory (fake)')
        return real(self)
    monkeypatch.setattr(ff.FlowTrainer, 'step', step)
    s = _setup(tmp_path / 'oom')
    losses, info = _fit(s, steps_per_round=5, trainer_seed=2)
    assert raised and losses == expected and _equal(_state(s.network), _state(clean.network))
    assert info['record']['oom_retries'] == 1 and info['record']['steps'] == 5
    assert info['round'] == 1 and _equal(trainers[-2].start_state, trainers[-1].start_state)
    raised.clear()
    monkeypatch.setattr(ff.FlowTrainer, 'step',
                        lambda self: (_ for _ in ()).throw(torch.OutOfMemoryError('always')))
    with pytest.raises(torch.OutOfMemoryError):
        _fit(s, round_oom_retries=2)


def test_oom_in_the_record_repeats_the_record_only(tmp_path, monkeypatch, trainers):
    """An OOM while the round record is built repeats that part only; the round counts once."""
    real_record = ff._record
    raised = []

    def record(context, status, *args, **kwargs):
        if status == 'trained' and not raised:
            raised.append('record')
            raise torch.OutOfMemoryError('CUDA out of memory (fake, held-out NLL)')
        return real_record(context, status, *args, **kwargs)
    monkeypatch.setattr(ff, '_record', record)
    s = _setup(tmp_path)
    losses, info = _fit(s)
    assert raised == ['record'] and len(losses) == 3 and len(trainers) == 1
    assert info['round'] == 1 and _state_file(s)['round'] == 1 and len(_rounds(s)) == 1
    # a no-op round repeated after an OOM stays the no-op (no sleep, no second count)
    t = _setup(tmp_path / 'noop', nbins=4)
    calls = []

    def noop_record(context, status, *args, **kwargs):
        calls.append(status)
        if status == 'no-op' and calls.count('no-op') == 1:
            raise torch.OutOfMemoryError('CUDA out of memory (fake, no-op record)')
        return real_record(context, status, *args, **kwargs)
    monkeypatch.setattr(ff, '_record', noop_record)
    monkeypatch.setattr(ff, '_sleep', lambda seconds, stopping:
                        pytest.fail('the repeated no-op slept') if seconds else False)
    losses, info = ff.flow_matching_fit(t.params, PathEnsemble(), worker=t.worker,
                                        **{**FAST, 'idle_sleep_s': 5.0})
    assert losses == [0.0] and info['status'] == 'no-op' and info['round'] == 1
    assert calls == ['no-op', 'no-op']


def test_fit_without_state_dir_keeps_its_rounds_in_memory(tmp_path):
    """No worker and no state_dir: nothing is written; repeated calls count rounds in memory."""
    s = _setup(tmp_path)
    before = sorted(Path(s.run).rglob('*'))
    _, first = ff.flow_matching_fit(s.params, s.pe, fill_missing_values=False, **FAST)
    _, second = ff.flow_matching_fit(s.params, s.pe, fill_missing_values=False, **FAST)
    assert first['round'] == 1 and second['round'] == 2
    assert first['network_metadata']['seed_source'] == 'default'
    assert sorted(Path(s.run).rglob('*')) == before
    assert ('memory', id(s.params.network)) in ff._RUNS


def test_log_every_writes_progress_records(tmp_path):
    s = _setup(tmp_path)
    _fit(s, steps_per_round=4, log_every=2)
    lines = _rounds(s)
    assert [r.get('progress', False) for r in lines] == [True, True, False]
    assert [r['round'] for r in lines] == [1, 1, 1]


# --------------------------------------------------------------------------------------------
# relabelling within a round (relabel_every)
# --------------------------------------------------------------------------------------------

def test_relabel_every_default_is_one_relabel_per_round(tmp_path, capsys):
    """relabel_every=0 (default) relabels once, at the start of the round, and warns once per process."""
    s = _setup(tmp_path)
    calls = []
    original = ff._relabel

    def spy(context, data):
        calls.append(context['trainer'].step_count)
        return original(context, data)
    ff._relabel = spy
    try:
        _, info = _fit(s, steps_per_round=6)
        out = capsys.readouterr().out
        _fit(s, steps_per_round=6)
    finally:
        ff._relabel = original
    assert calls == [0, 0]
    assert info['record']['relabels_in_round'] == 0
    assert 'relabel_in_round' not in info['record']['seconds']
    assert 'labels of the reactive frames are the committor of the start network' in out
    assert 'start network' not in capsys.readouterr().out          # once per process


def test_relabel_every_relabels_within_the_round_from_the_current_ema(tmp_path, trainers):
    """relabel_every=k relabels after steps k, 2k, ... (not after the last), recomputed from that step's EMA."""
    torch.manual_seed(0)
    s = _setup(tmp_path, network=_steep(TinyFlowNetwork(n_atoms=1, lambda_clip=10.0)))
    calls = []
    original = ff._relabel
    seen = {}

    def spy(context, data):
        trainer = context['trainer']
        calls.append(trainer.step_count)
        out = original(context, data)
        seen[trainer.step_count] = (copy.deepcopy(trainer.network).eval(),
                                    trainer.fm_pool.copy(), trainer.fm_lambda.copy())
        return out
    ff._relabel = spy
    try:
        _, info = _fit(s, steps_per_round=7, relabel_every=3, ema_decay=0.9, lr=1e-2)
    finally:
        ff._relabel = original
    assert calls == [0, 3, 6]
    record = info['record']
    assert record['relabels_in_round'] == 2 and record['steps'] == 7
    assert record['seconds']['relabel_in_round'] >= 0.0
    trainer = trainers[-1]
    for step in (3, 6):
        ema, pool, lam = seen[step]
        rows = trainer.dataset.fr_rows[pool]
        with torch.no_grad():
            expected = ema(torch.as_tensor(rows))[:, 0].double().numpy()
        assert np.allclose(lam, np.clip(expected, -10.0, 10.0), atol=1e-5)
    ema0, _, lam0 = seen[0]
    ema6, pool6, lam6 = seen[6]
    rows = trainer.dataset.fr_rows[pool6]
    with torch.no_grad():
        start = np.clip(ema0(torch.as_tensor(rows))[:, 0].double().numpy(), -10.0, 10.0)
    assert np.abs(start - lam6).max() > 1e-4                 # the EMA moved: new targets
    assert _rounds(s)[-1]['relabels_in_round'] == 2


def test_relabel_every_is_validated(tmp_path):
    s = _setup(tmp_path)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match='relabel_every'):
            _fit(s, relabel_every=bad)
