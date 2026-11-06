"""End-state pretraining, ``aimmd.cgaimmd.flow_matching.pretrain``.

The network learns the two end states before any shot exists; its file is the
``initial_network`` every round of the online fit starts from.
"""

import json

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.flow_matching import fit as ff
from aimmd.cgaimmd.flow_matching import pretrain
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from tests._cgaimmd_helpers import flat_coordinates_descriptors, x_states_function
from tests._helpers_unit import write_trajectory


TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=False,
            edge_backend='torch')


def _frames(n_per_source=(20, 20), m=3, seed=0):
    """Two basins of an m-atom chain (x of atom 0 decides the state), one source per block."""
    rng = np.random.default_rng(seed)
    coords, states, source = [], [], []
    base = np.stack([np.zeros(m), 1.5 * np.arange(m), np.zeros(m)], axis=1)
    for k, n in enumerate(n_per_source):
        state = 'A' if k % 2 == 0 else 'B'
        x = base + rng.normal(0.0, 0.1, size=(n, m, 3))
        x[:, :, 0] += -1.5 if state == 'A' else 1.5
        coords.append(x)
        states.append(np.full(n, state))
        source.append(np.full(n, k))
    return {'coordinates': np.concatenate(coords).astype(np.float32),
            'state': np.concatenate(states).astype('<U1'),
            'source': np.concatenate(source),
            'trajectories': [f'traj{k}.xtc' for k in range(len(n_per_source))]}


def test_estimate_coord_scale():
    """sqrt(mean |x - centroid|^2 / 3): the per-coordinate RMS of centered frames."""
    x = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                  [[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]]) + 7.0
    expected = np.sqrt((1 + 1 + 4 + 4) / 4 / 3)
    assert pretrain.estimate_coord_scale(x) == pytest.approx(expected)
    rng = np.random.default_rng(0)
    blob = rng.normal(0.0, 2.0, size=(500, 30, 3))
    assert pretrain.estimate_coord_scale(blob) == pytest.approx(2.0, rel=0.03)


def test_collect_end_state_frames(tmp_path):
    """Descriptors and states of every stride-th frame; other states dropped; random cap in time order."""
    import MDAnalysis as mda
    structure = tmp_path / 'one.pdb'
    universe = mda.Universe.empty(1, trajectory=True)
    universe.add_TopologyAttr('names', ['X'])
    universe.add_TopologyAttr('types', ['X'])
    universe.atoms.positions = np.zeros((1, 3))
    universe.dimensions = np.array([50, 50, 50, 90, 90, 90], dtype=np.float32)
    universe.atoms.write(str(structure))
    files = []
    for k, xs in enumerate(([-1.0, -1.1, 0.0, -1.2, -1.3, -1.4], [1.0, 1.1, 1.2, 0.1, 1.3, 1.4])):
        positions = np.zeros((len(xs), 1, 3), dtype=np.float32)
        positions[:, 0, 0] = xs
        files.append(write_trajectory(tmp_path, stem=f'basin{k}', positions=positions,
                                      times=np.arange(len(xs), dtype=float)))
    frames = pretrain.collect_end_state_frames(
        files, structure=str(structure), descriptors_function=flat_coordinates_descriptors,
        states_function=x_states_function, states='ARB')
    assert frames['coordinates'].shape == (10, 1, 3) and frames['coordinates'].dtype == np.float32
    assert frames['state'].tolist() == ['A'] * 5 + ['B'] * 5            # R frames dropped
    assert frames['source'].tolist() == [0] * 5 + [1] * 5
    assert frames['trajectories'] == [str(f) for f in files]
    strided = pretrain.collect_end_state_frames(
        files, structure=str(structure), descriptors_function=flat_coordinates_descriptors,
        states_function=x_states_function, states='ARB', stride=2)
    assert np.allclose(strided['coordinates'][:, 0, 0], [-1.0, -1.3, 1.0, 1.2, 1.3], atol=1e-2)
    capped = pretrain.collect_end_state_frames(
        files, structure=str(structure), descriptors_function=flat_coordinates_descriptors,
        states_function=x_states_function, states='ARB', max_frames=4, seed=3)
    assert len(capped['state']) == 4
    order = [(s, x) for s, x in zip(capped['source'], capped['coordinates'][:, 0, 0])]
    assert order == sorted(order, key=lambda item: (item[0], abs(item[1])))        # time order
    # blocks of the full system: any block size gives the same frames (strides included)
    for stride, chunk in ((1, 1), (1, 4), (2, 1), (2, 2)):
        blocked = pretrain.collect_end_state_frames(
            files, structure=str(structure), descriptors_function=flat_coordinates_descriptors,
            states_function=x_states_function, states='ARB', stride=stride, chunk_frames=chunk)
        whole = strided if stride == 2 else frames
        assert np.array_equal(blocked['coordinates'], whole['coordinates'])
        assert blocked['state'].tolist() == whole['state'].tolist()
        assert blocked['source'].tolist() == whole['source'].tolist()
    sizes = []

    def recording(trajectory):
        sizes.append(trajectory.n_frames)
        return flat_coordinates_descriptors(trajectory)
    pretrain.collect_end_state_frames(
        files, structure=str(structure), descriptors_function=recording,
        states_function=x_states_function, states='ARB', chunk_frames=4)
    assert sizes == [4, 2, 4, 2]                          # never more than a block in memory
    with pytest.raises(ValueError, match='chunk_frames'):
        pretrain.collect_end_state_frames(
            files, structure=str(structure), descriptors_function=flat_coordinates_descriptors,
            states_function=x_states_function, states='ARB', chunk_frames=0)


def test_holdout_split_rules():
    """With >= 5 sources whole trajectories are held out, else the last fraction of each."""
    few = pretrain._holdout_split(np.repeat([0, 1], 10), 0.2, np.random.default_rng(0))
    assert few.tolist() == ([False] * 8 + [True] * 2) * 2
    source = np.repeat(np.arange(6), 5)
    many = pretrain._holdout_split(source, 0.2, np.random.default_rng(0))
    held = sorted(set(source[many].tolist()))
    assert len(held) == 1                                             # round(0.2 x 6) = 1
    assert all(many[source == s].all() for s in held)
    assert not many[~np.isin(source, held)].any()


def test_steps_required(tmp_path):
    torch.manual_seed(0)
    net = JointPaiNN(3, [6, 6, 6], **TINY)
    with pytest.raises(TypeError, match='steps'):
        pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB')


def test_pretraining_validates_its_keywords(tmp_path):
    """Bad values raise with pretraining's own names (not the online fit's)."""
    net = TinyFlowNetwork(n_atoms=3, lambda_clip=10.0)
    for bad, match in (({'steps': 0}, 'steps must be an int >= 1'),
                       ({'steps': 2.5}, 'steps must be an int >= 1'),
                       ({'batch_size': 1}, 'batch_size must be an int >= 2'),
                       ({'batch_size': 0}, 'batch_size must be an int >= 2'),
                       ({'seed': -1}, 'seed'), ({'holdout_fraction': 1.0}, 'holdout_fraction'),
                       ({'checkpoint_steps': (0,)}, 'checkpoint_steps'),
                       ({'lr': -1.0}, 'pretrain_end_states: lr must be'),
                       ({'lr_schedule': 'linear'}, 'pretrain_end_states: lr_schedule')):
        kwargs = {'steps': 2, 'device': 'cpu', **bad}
        with pytest.raises(ValueError, match=match) as info:
            pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', **kwargs)
        assert 'flow_matching_fit' not in str(info.value)
    with pytest.raises(TypeError, match="pretrain_end_states\\(\\) got an unexpected keyword "
                                        "argument 'stepz'"):
        pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=2, stepz=3)


def test_the_old_prior_keywords(tmp_path):
    """prior='gaussian' (and None) is what pretraining always does: one warning; other priors raise with the reason."""
    from aimmd.cgaimmd import _legacy
    net = TinyFlowNetwork(n_atoms=3, lambda_clip=10.0)
    with pytest.warns(DeprecationWarning, match="pretrain_end_states\\(prior='gaussian'\\)"):
        report = pretrain.pretrain_end_states(net, _frames(), tmp_path / 'g', states='ARB',
                                              steps=1, batch_size=4, device='cpu',
                                              prior='gaussian', prior_sigma=None)
    assert report['steps_done'] == 1
    for bad in ({'prior': 'mixture', 'prior_sigma': 0.5}, {'prior_sigma': 0.5},
                {'prior_bank': 'train'}, {'prior': 'harmonic'}):
        with pytest.raises(TypeError, match='is not supported: .*not part of C-GAIMMD'):
            pretrain.pretrain_end_states(net, _frames(), tmp_path / 'bad', states='ARB',
                                         steps=1, device='cpu', **bad)
    assert not (tmp_path / 'bad').exists()                       # refused before anything
    assert ('pretrain_end_states', 'prior') in _legacy.DROPPED_KEYWORDS


def test_the_flow_branch_gaussian_pretraining_call(tmp_path):
    """The production call of the flow branch (Gaussian arm) runs: its mixture-prior keywords are that branch's defaults.

    ``analysis/campaigns/2026-10-02_universal-prior/chignolin/pretrain_end_states.py``
    passed ``prior='gaussian'`` with ``prior_bank='holdout', prior_bank_max=5000,
    prior_pi_null=0.5, prior_pi_temperature=1.0``, which did nothing with the
    Gaussian prior. Other values of those keywords still raise.
    """
    old_defaults = dict(prior='gaussian', prior_bank='holdout', prior_bank_max=5000,
                        prior_pi_null=0.5, prior_pi_temperature=1.0)
    net = TinyFlowNetwork(n_atoms=3, lambda_clip=10.0)
    with pytest.warns(DeprecationWarning) as caught:
        report = pretrain.pretrain_end_states(net, _frames(), tmp_path / 'g', states='ARB',
                                              steps=1, batch_size=4, device='cpu',
                                              **old_defaults)
    assert report['steps_done'] == 1
    warned = ' '.join(str(w.message) for w in caught)
    for name in old_defaults:
        assert f'pretrain_end_states({name}=' in warned
    pretrain.pretrain_end_states(net, _frames(), tmp_path / 'i', states='ARB', steps=1,
                                 batch_size=4, device='cpu', prior_pi_temperature=1,
                                 prior_bank_max=5000.0)                # equal numbers
    for bad in ({'prior_bank_max': 2000}, {'prior_pi_null': 0.3}, {'prior_pi_temperature': 2.0},
                {'prior_bank_max': '5000'}, {'prior_pi_temperature': True},
                {'prior_bank': 'train'}):
        with pytest.raises(TypeError, match='is not supported: .*not part of C-GAIMMD'):
            pretrain.pretrain_end_states(net, _frames(), tmp_path / 'bad', states='ARB',
                                         steps=1, device='cpu', **{**old_defaults, **bad})
    assert not (tmp_path / 'bad').exists()


def test_cosine_spans_the_pretraining_steps(tmp_path, monkeypatch):
    """Pretraining is one round of `steps` steps: more steps than the fit's steps_per_round are fine."""
    trainers = []

    class Recorded(ff.FlowTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)
    monkeypatch.setattr(ff, 'FlowTrainer', Recorded)
    monkeypatch.setitem(ff.FLOW_FIT_DEFAULTS, 'steps_per_round', 2)    # below `steps`
    net = TinyFlowNetwork(n_atoms=3, lambda_clip=10.0)
    report = pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=4,
                                          batch_size=4, warmup_steps=1, device='cpu')
    assert report['steps_done'] == 4
    (trainer,) = trainers
    assert trainer.settings['total_steps'] == trainer.settings['steps_per_round'] == 4
    assert trainer._lr_factor(3) > 0.0 and trainer._lr_factor(4) == pytest.approx(0.0, abs=1e-12)


def test_divergence_guard(tmp_path, monkeypatch, capsys):
    """Consecutive non-finite gradients: a warning every 100, RuntimeError at the limit, no network file."""
    class Broken(TinyFlowNetwork):
        def forward(self, inp):
            return super().forward(inp) * float('nan')
    assert pretrain.MAX_CONSECUTIVE_SKIPPED_STEPS == 1000             # the prototype's
    monkeypatch.setattr(pretrain, 'MAX_CONSECUTIVE_SKIPPED_STEPS', 200)
    net = Broken(n_atoms=3, lambda_clip=10.0)
    with pytest.raises(RuntimeError, match='200 consecutive steps'):
        pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=500,
                                     batch_size=4, device='cpu')
    assert capsys.readouterr().out.count('consecutive steps skipped') == 2   # at 100 and 200
    assert not (tmp_path / 'networkARB.h5').exists()
    report = pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=4,
                                          batch_size=4, device='cpu')
    assert report['skipped_steps'] == 4                       # below the limit: written


def test_tiny_pretrain_writes_loadable_network(tmp_path, monkeypatch):
    """network<s>.h5 (tensors only, the EMA), step checkpoints and pretrain.json."""
    trainers = []

    class Recorded(ff.FlowTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)
    monkeypatch.setattr(ff, 'FlowTrainer', Recorded)
    torch.manual_seed(0)
    net = JointPaiNN(3, [6, 6, 6], **TINY)
    frames = _frames((12, 12, 12, 12, 12, 12))
    out = tmp_path / 'pretrained'
    before = {k: v.clone() for k, v in net.state_dict().items()}
    report = pretrain.pretrain_end_states(net, frames, out, states='BRA', steps=4, batch_size=6,
                                          ema_decay=0.9, checkpoint_steps=(2,), warmup_steps=1,
                                          device='cpu', seed=1)
    assert (out / 'networkARB.h5').exists() and (out / 'checkpoint_step000002.pt').exists()
    sd = torch.load(out / 'networkARB.h5', weights_only=True)
    fresh = JointPaiNN(3, [6, 6, 6], **TINY)
    fresh.load_state_dict(sd)                              # strict: the network's own keys only
    assert all(torch.equal(sd[k], v) for k, v in net.state_dict().items())
    assert not all(torch.equal(before[k], sd[k]) for k in before)
    on_disk = json.loads((out / 'pretrain.json').read_text())
    assert on_disk == json.loads(json.dumps(report))
    assert report['steps_done'] == 4 and report['states'] == 'BRA'
    assert report['trajectories'] == frames['trajectories']
    assert report['split']['kind'].startswith('by trajectory')
    assert report['holdout_anchor_nll'] is not None and np.isfinite(report['holdout_anchor_nll'])
    assert 'unbiased' in report['note']
    assert report['prior'] == {'kind': 'gaussian', 'n_atoms': 3}
    assert report['loss_history'][0]['steps'] == [0, 4]
    assert report['settings']['steps'] == 4 and report['settings']['ema_decay'] == 0.9
    assert report['saved_weights'] == 'ema'
    # the file is the EMA (the trainer's network), not the live weights
    (trainer,) = trainers
    assert trainer.network is net and trainer.seed == 1
    live = trainer.live.state_dict()
    assert all(torch.equal(sd[k], v.cpu()) for k, v in trainer.network.state_dict().items())
    assert not all(torch.equal(sd[k], live[k].cpu()) for k in sd)
    # the held-out anchor NLL: committor_nll of the held-out frames against sigmoid(-+L)
    held = pretrain._holdout_split(frames['source'], 0.1, np.random.default_rng(1))
    x = torch.as_tensor(frames['coordinates'][held])
    net.eval()
    with torch.no_grad():
        lam = net.committor(x).double()
    y = torch.sigmoid(torch.as_tensor(np.where(frames['state'][held] == 'B', 10.0, -10.0)))
    expected = float(ff.committor_nll(lam, y, torch.ones_like(y)))
    assert report['holdout_anchor_nll'] == pytest.approx(expected, rel=1e-5)


def test_checkpoints_hold_the_saved_weights(tmp_path):
    """checkpoint_step<k>.pt is the EMA at step k (what network<s>.h5 would hold after k steps)."""
    common = dict(states='ARB', batch_size=6, ema_decay=0.5, lr_schedule='constant', lr=1e-2,
                  device='cpu', seed=2)
    frames = _frames((12, 12, 12, 12, 12, 12))
    torch.manual_seed(0)
    short = JointPaiNN(3, [6, 6, 6], **TINY)
    pretrain.pretrain_end_states(short, frames, tmp_path / 'two', steps=2, **common)
    torch.manual_seed(0)
    net = JointPaiNN(3, [6, 6, 6], **TINY)
    report = pretrain.pretrain_end_states(net, frames, tmp_path / 'four', steps=4,
                                          checkpoint_steps=(2, 4), **common)
    assert report['checkpoints'] == ['checkpoint_step000002.pt', 'checkpoint_step000004.pt']
    at_two = torch.load(tmp_path / 'four' / 'checkpoint_step000002.pt', weights_only=True)
    after_two = torch.load(tmp_path / 'two' / 'networkARB.h5', weights_only=True)
    assert at_two.keys() == after_two.keys()
    assert all(torch.equal(at_two[k], after_two[k]) for k in at_two)
    at_four = torch.load(tmp_path / 'four' / 'checkpoint_step000004.pt', weights_only=True)
    final = torch.load(tmp_path / 'four' / 'networkARB.h5', weights_only=True)
    assert all(torch.equal(at_four[k], final[k]) for k in final)
    assert not all(torch.equal(at_two[k], at_four[k]) for k in at_two)


def test_the_minimal_network_pretrains(tmp_path, monkeypatch):
    """Only protocol and flow-matching members are used: the minimal network pretrains."""
    draws = []
    real = TinyFlowNetwork.sample_prior

    def spy(self, lam, n, *, uncond_mask=None, generator=None):
        draws.append(n)
        return real(self, lam, n, uncond_mask=uncond_mask, generator=generator)
    monkeypatch.setattr(TinyFlowNetwork, 'sample_prior', spy)
    torch.manual_seed(0)
    net = TinyFlowNetwork(n_atoms=3)                     # the minimal network: no lambda_clip
    with pytest.raises(TypeError, match='lambda_clip'):
        pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=2,
                                     device='cpu')
    stops = iter([False, False, True])
    report = pretrain.pretrain_end_states(net, _frames(), tmp_path, states='ARB', steps=10,
                                          batch_size=4, device='cpu',
                                          must_stop=lambda: next(stops), lambda_clip=8.0)
    assert report['steps_done'] == 2 and report['stopped'] is True
    assert report['lambda_clip'] == 8.0 and report['settings']['lambda_clip'] == 8.0
    assert draws and report['prior'] == {'kind': "the network's own sample_prior",
                                         'network': 'TinyFlowNetwork'}
    assert torch.load(tmp_path / 'networkARB.h5', weights_only=True).keys() == \
        net.state_dict().keys()
    assert report['saved_weights'] == 'live'
    with pytest.raises(TypeError, match='lambda_clip'):           # a network's own L wins, or agrees
        pretrain.pretrain_end_states(TinyFlowNetwork(n_atoms=3, lambda_clip=10.0), _frames(),
                                     tmp_path, states='ARB', steps=1, device='cpu',
                                     lambda_clip=8.0)


def test_a_network_without_the_flow_matching_members_is_refused(tmp_path):
    class NoMask(TinyFlowNetwork):
        def velocity(self, x_t, t, lam):
            return super().velocity(x_t, t, lam)
    with pytest.raises(TypeError, match='not a flow-matching network: velocity must accept'):
        pretrain.pretrain_end_states(NoMask(n_atoms=3, lambda_clip=10.0), _frames(), tmp_path,
                                     states='ARB', steps=1, device='cpu')


def test_the_pretrained_network_is_the_start_of_every_fit_round(tmp_path, monkeypatch):
    """network<s>.h5 of pretraining, set as initial_network, is what each online round starts from."""
    from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace
    from aimmd.pathensemble.utils import assemble_pathensemble
    torch.manual_seed(0)
    net = JointPaiNN(3, [6, 6, 6], **TINY)
    pretrain.pretrain_end_states(net, _frames(), tmp_path / 'pretrained', states='ARB', steps=3,
                                 batch_size=4, device='cpu')
    initial = tmp_path / 'pretrained' / 'networkARB.h5'
    starts = []

    class Recorded(ff.FlowTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            starts.append({k: v.clone() for k, v in self.network.state_dict().items()})
    monkeypatch.setattr(ff, 'FlowTrainer', Recorded)
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=2, n_atoms=3)
    params = cgaimmd_params_namespace(network=net, initial_network=str(initial))
    pe = assemble_pathensemble(params.shot_chains(rd.run, None), params.free_trajectories(rd.run))

    class Worker:
        _directory = rd.run
        task = 'train'
        must_stop = False
    for _ in range(2):
        losses, info = ff.flow_matching_fit(params, pe, worker=Worker(), steps_per_round=2,
                                            warmup_steps=0, n_sp=2, n_anchor=2, n_fm=2,
                                            idle_sleep_s=0.0, fill_missing_values=False)
        assert info['status'] == 'trained'
    saved = torch.load(initial, weights_only=True)
    assert len(starts) == 2
    for start in starts:
        assert start.keys() == saved.keys() and all(torch.equal(start[k], saved[k]) for k in saved)
