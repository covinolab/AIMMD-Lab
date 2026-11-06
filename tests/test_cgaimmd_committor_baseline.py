"""The distance-matrix committor baseline (``aimmd.cgaimmd.analysis.committor_baseline``).

The pair distances and their scaler, the network (shape, invariance, the
selection, the scaler in the saved weights, reset), and AIMMD's own fit on a
toy run whose committor is a known function of one distance: two atoms,
shooting points spread over that distance, each half committed with that
committor. Nothing is written into the run folder, hidden files included.
"""

import os

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.analysis import committor_baseline as baseline
from aimmd.cgaimmd.analysis import validate
from aimmd.path.utils import get_cache_fname
from tests._cgaimmd_helpers import flat_coordinates_descriptors
from tests._helpers_unit import write_trajectory


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.diag(r))[None]


def test_pair_distances_and_the_scaler():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 5, 3))
    d = baseline.pair_distances(x, [4, 0, 2])
    assert d.shape == (50, 3)
    assert d[7, 0] == pytest.approx(np.linalg.norm(x[7, 4] - x[7, 0]))
    assert d[7, 2] == pytest.approx(np.linalg.norm(x[7, 0] - x[7, 2]))
    np.testing.assert_allclose(baseline.pair_distances(x.reshape(50, -1), [4, 0, 2]), d)
    mean, std = baseline.distance_scaler(x, [4, 0, 2])
    np.testing.assert_allclose(mean, d.mean(0))
    np.testing.assert_allclose(std, d.std(0))
    w = np.where(np.arange(50) < 10, 3.0, 1.0)
    wmean, _ = baseline.distance_scaler(x, [4, 0, 2], weights=w)
    np.testing.assert_allclose(wmean, (w[:, None] * d).sum(0) / w.sum())
    still = np.repeat(x[:1], 4, axis=0)
    assert np.all(baseline.distance_scaler(still, [0, 1])[1] == baseline.MIN_STD)
    with pytest.raises(ValueError, match='two atoms'):
        baseline.pair_distances(x, [1])
    with pytest.raises(ValueError, match='twice'):
        baseline.pair_distances(x, [1, 1])
    with pytest.raises(ValueError, match='weights'):
        baseline.distance_scaler(x, [0, 1], weights=np.zeros(50))


def test_distance_mlp():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 5, 3)).astype(np.float32)
    selection = [3, 0, 4]
    mean, std = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 2.0])
    torch.manual_seed(0)
    net = baseline.DistanceMLP(selection, n_atoms=5, hidden=(8, 4), mean=mean, std=std)
    out = net(torch.as_tensor(x.reshape(6, -1)))
    assert out.shape == (6, 1)
    assert torch.allclose(net(torch.as_tensor(x)), out)
    # the scaler is applied
    want = (baseline.pair_distances(x, selection) - mean) / std
    np.testing.assert_allclose(net.features(x).detach().numpy(), want, rtol=1e-5, atol=1e-5)
    # moved and rotated molecules give the same committor
    moved = x @ _rotation(2).T.astype(np.float32) + 4.0
    assert torch.allclose(net(torch.as_tensor(moved)), out, atol=1e-5)
    # a permuted selection permutes the features
    swapped = baseline.DistanceMLP([0, 3, 4], n_atoms=5, hidden=(8, 4))
    np.testing.assert_allclose(swapped.features(x).detach().numpy()[:, [0, 2, 1]],
                               baseline.pair_distances(x, selection), rtol=1e-5)
    # the scaler and the selection are saved with the weights; reset keeps them
    state = net.state_dict()
    assert {'atom_indices', 'pair_first', 'pair_second', 'mean', 'std'} <= set(state)
    other = baseline.DistanceMLP(selection, n_atoms=5, hidden=(8, 4))
    other.load_state_dict(state)
    assert torch.allclose(other(torch.as_tensor(x)), out)
    before = net.output.weight.detach().clone()
    net.reset_parameters()
    assert not torch.equal(net.output.weight, before)
    assert torch.equal(net.mean, torch.as_tensor(mean, dtype=torch.float32))
    assert net.settings() == {'n_atoms': 5, 'n_selected_atoms': 3, 'n_pairs': 3,
                              'hidden': [8, 4], 'activation': 'SiLU'}
    # AIMMD's default values function takes it like any committor network
    from aimmd.params.utils import create_default_values_function
    values = create_default_values_function(net)(x.reshape(6, -1))
    assert values.shape == (6,)
    with pytest.raises(ValueError, match='lie in'):
        baseline.DistanceMLP([0, 7], n_atoms=5)
    with pytest.raises(ValueError, match='one per pair'):
        baseline.DistanceMLP([0, 1, 2], n_atoms=5, mean=np.zeros(2))


DISTANCE_MIDPOINT = 2.0
STEEPNESS = 4.0


def _committor(d):
    """The toy run's committor: a known function of the distance of the two atoms."""
    return _sigmoid(STEEPNESS * (np.asarray(d) - DISTANCE_MIDPOINT))


def _frames(d):
    """Two atoms at distance `d` along a tilted axis, molecule shifted with its distance."""
    d = np.asarray(d, dtype=float)
    axis = np.array([0.6, 0.0, 0.8])
    x = np.zeros((len(d), 2, 3), dtype=np.float32)
    x[:, 0] = np.array([5.0, 5.0, 5.0])
    x[:, 1] = x[:, 0] + d[:, None] * axis
    return x


def _write(fname, d, labels, times, bias=None):
    written = write_trajectory(os.path.dirname(fname), stem=os.path.basename(fname)[:-4],
                               ext='.xtc', positions=_frames(d), times=times)
    stored = aimmd.Path(written).positions
    save_npy(get_cache_fname(written, 'states'), np.asarray(labels, dtype='<U1'))
    save_npy(get_cache_fname(written, 'descriptors'),
             np.asarray(stored, dtype=np.float32).reshape(len(stored), -1))
    if bias is not None:
        save_npy(get_cache_fname(written, 'bias'), np.asarray(bias, dtype=float))
    return written


def _toy_run(tmp_path, n_shots=160, seed=0, bias_in_a=None):
    """Free MD in A (d ~ 1) and B (d ~ 3); shots from d in [1.2, 2.8], each half committed with q(d)."""
    rng = np.random.default_rng(seed)
    run = tmp_path / 'run'
    for state, center in (('A', 1.0), ('B', 3.0)):
        folder = run / f'free{state}'
        folder.mkdir(parents=True)
        d = center + 0.1 * rng.standard_normal(30)
        bias = None if bias_in_a is None else np.full(30, bias_in_a if state == 'A' else 0.0)
        _write(str(folder / 'traj000001.part0001.xtc'), d, [state] * 30, np.arange(30.0), bias)
    folder = run / 'chainR0'
    folder.mkdir(parents=True)
    si = 3
    for n in range(1, n_shots + 1):
        d0 = rng.uniform(1.2, 2.8)
        ends = ['B' if u < _committor(d0) else 'A' for u in rng.random(2)]
        d = np.full(7, d0)
        d[0] = 1.0 if ends[0] == 'A' else 3.0
        d[-1] = 1.0 if ends[1] == 'A' else 3.0
        labels = [ends[0]] + ['R'] * 5 + [ends[1]]
        _write(str(folder / f'path{n:06d}.xtc'), d, labels, np.abs(np.arange(7.0) - si))
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(states='ARB', trajectory_extension='.xtc',
                           descriptors_function=flat_coordinates_descriptors,
                           round_shooting_index=True, record_bias=bias_in_a is not None)
    return run, params


def _data_files(root):
    """{file: (size, mtime, inode)} of every file under `root`, hidden files (locks, frame
    indices) included."""
    out = {}
    for directory, _, files in os.walk(root):
        for name in files:
            stat = os.stat(os.path.join(directory, name))
            out[os.path.relpath(os.path.join(directory, name), root)] = (
                stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return out


def test_scaler_from_the_free_simulations(tmp_path):
    run, params = _toy_run(tmp_path, n_shots=2)
    scaler = baseline.run_distance_scaler(params, run, [0, 1])
    assert scaler['n_frames'] == 60 and scaler['bias_applied'] is False
    rows = np.concatenate([np.load(get_cache_fname(f, 'descriptors')) for f in scaler['files']])
    d = baseline.pair_distances(rows, [0, 1])[:, 0]
    assert scaler['mean'][0] == pytest.approx(d.mean(), rel=1e-6)
    few = baseline.run_distance_scaler(params, run, [0, 1], max_frames=10)
    assert few['n_frames'] == 10
    with pytest.raises(FileNotFoundError, match='no .descriptors-other'):
        baseline.run_distance_scaler(params, run, [0, 1], series='descriptors-other')


def test_a_run_with_a_bias_is_refused_unless_allowed(tmp_path):
    """Frames of biased simulations become neither an input scaler nor training data by default;
    allowed explicitly, the scaler weights every frame by exp(bias)."""
    run, params = _toy_run(tmp_path, n_shots=4, bias_in_a=1.0)
    with pytest.raises(ValueError, match='records a bias.*allow_biased_frames'):
        baseline.run_distance_scaler(params, run, [0, 1])
    with pytest.raises(ValueError, match='records a bias.*allow_biased_frames'):
        baseline.train_on_run(params, run, baseline.DistanceMLP([0, 1], n_atoms=2), epochs=1)
    plain = baseline.run_distance_scaler(params, run, [0, 1], bias=False,
                                         allow_biased_frames=True)
    assert plain['n_frames'] == 60 and plain['bias_applied'] is False
    rows = np.concatenate([np.load(get_cache_fname(f, 'descriptors')) for f in plain['files']])
    d = baseline.pair_distances(rows, [0, 1])[:, 0]
    assert plain['mean'][0] == pytest.approx(d.mean(), rel=1e-6)
    weighted = baseline.run_distance_scaler(params, run, [0, 1], allow_biased_frames=True)
    assert weighted['bias_applied'] is True                             # 'auto': the run's bias
    w = np.where(np.arange(60) < 30, np.e, 1.0)                         # free A first
    assert weighted['mean'][0] == pytest.approx((w * d).sum() / w.sum(), rel=1e-6)
    assert weighted['frames_without_bias'] == 0
    torch.manual_seed(0)
    result = baseline.train_on_run(params, run, baseline.DistanceMLP([0, 1], n_atoms=2,
                                                                     hidden=(4,)),
                                   epochs=2, allow_biased_frames=True)
    assert result['n_shots'] == 4
    # a free simulation without its bias cache: weighted 1, with a warning
    os.remove(get_cache_fname(str(run / 'freeA' / 'traj000001.part0001.xtc'), 'bias'))
    with pytest.warns(UserWarning, match='30 free-simulation frames have no row'):
        partial = baseline.run_distance_scaler(params, run, [0, 1], allow_biased_frames=True)
    assert partial['frames_without_bias'] == 30
    assert partial['mean'][0] == pytest.approx(d.mean(), rel=1e-6)


def test_train_on_the_shots_of_a_run(tmp_path):
    """AIMMD's fit on the run's own shots learns the committor of the toy (a function of the distance)."""
    run, params = _toy_run(tmp_path)
    from aimmd import _config
    for cache in (_config.MDA_CACHE, _config.NPY_CACHE):
        cache.clear()                                           # every file is opened afresh
    before = _data_files(run)
    scaler = baseline.run_distance_scaler(params, run, [0, 1])
    torch.manual_seed(0)
    mlp = baseline.DistanceMLP([0, 1], n_atoms=2, hidden=(16, 16), mean=scaler['mean'],
                               std=scaler['std'])
    result = baseline.train_on_run(params, run, mlp, epochs=300, batch_size=256, lr=3e-3)
    assert _data_files(run) == before                           # nothing written into the run
    assert result['n_shots'] == 160 and result['n_free_trajectories'] == 2
    assert len(result['losses']) > 0 and result['fit_settings']['epochs'] == 300
    grid = np.linspace(1.3, 2.7, 15)
    with torch.no_grad():
        lam = mlp(torch.as_tensor(_frames(grid))).numpy().reshape(-1)
    assert np.all(np.diff(lam) > 0)                             # increases with the distance
    assert np.max(np.abs(_sigmoid(lam) - _committor(grid))) < 0.25
    # scored on whole configurations through the run's descriptors, or on the coordinates
    reference = validate.CommittorReference('toy', _frames(grid), 100 * (1 - _committor(grid)),
                                            100 * _committor(grid))
    report, lam_params = validate.score_committor(
        validate.params_committor(params, network=mlp, descriptor_transform=None), reference)
    _, lam_direct = validate.score_committor(validate.network_committor(mlp), reference)
    np.testing.assert_allclose(lam_params, lam_direct, atol=1e-5)
    assert report['spearman'] == pytest.approx(1.0)
    # held-out chains: only the chains asked for
    torch.manual_seed(0)
    other = baseline.DistanceMLP([0, 1], n_atoms=2, hidden=(4,))
    held = baseline.train_on_run(params, run, other, chains=[0], epochs=5)
    assert held['chains'] == [0] and held['n_shots'] == 160
    assert baseline.train_on_run(params, run, other, chains=[1], epochs=5)['n_shots'] == 0
