"""Committor parity and the validation protocol (``aimmd.cgaimmd.analysis.validate``).

Ported from the flow branch's ``tests/test_flow_validate.py`` (M1 validity,
M2 committor accuracy, M3 generator against committor, the guard, the JSON
report, the model card, the leaderboard), with samples drawn through the
generation interface, plus the new parts: Spearman and the binomial-noise
ceiling of R², the reference loaders (an npz in the NANMA format, a trajectory
with shot counts in the chignolin and trypsin formats, the free simulations of
a run), the committor of any AIMMD network through its params, and the parity
figure. CPU only; the GROMACS polish runs with ``--runslow``.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.analysis import validate
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.protocol import coordinate_committor
from aimmd.cgaimmd.types import PreparedFrame, ValidityResult
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace

DATA = Path(__file__).parent / 'data'
NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'
NANMA_REFERENCE = DATA / 'nanma_committor_reference_subset.npz'
TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=False)
TOKENS = [6, 1, 1, 8, 6, 1, 7]
M = len(TOKENS)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _molecule(n=4, seed=0, noise=0.15):
    """(n, M, 3) float64 Å frames of a 7-atom blob."""
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.0, 0.0, 1.4],
                     [-1.5, -0.2, -0.3], [-1.9, -1.1, 0.2], [1.2, 1.3, 1.1]])
    return base[None] + noise * rng.standard_normal((n, M, 3))


def _rotation(seed=0):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))[None]
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _randomize(network, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return network


def _joint(randomize=False, seed=0):
    torch.manual_seed(seed)
    net = JointPaiNN(M, TOKENS, edge_backend='torch', **TINY)
    return (_randomize(net, seed) if randomize else net).eval()


def _counts(lam, n=40, seed=0, a=0.0, b=1.0):
    rng = np.random.default_rng(seed)
    q = _sigmoid(a + b * np.asarray(lam, float))
    n_last = rng.binomial(n, q).astype(float)
    return n - n_last, n_last


class FrameBankGenerator:
    """A generator stand-in: frames of a bank plus Gaussian noise (a near-perfect generator of MD)."""

    method = 'test_frame_bank'

    def __init__(self, frames, sigma=0.01):
        self.frames = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
        self.sigma = float(sigma)

    def check(self, network):
        return []

    def generate(self, network, lam, *, generator=None):
        pick = torch.randint(0, len(self.frames), (len(lam),), generator=generator)
        noise = torch.randn((len(lam),) + tuple(self.frames.shape[1:]), generator=generator)
        return self.frames[pick] + self.sigma * noise

    def settings(self):
        return {'n_frames': int(len(self.frames)), 'sigma': self.sigma}


# ---------------------------------------------------------------------------
# M2: committor metrics
# ---------------------------------------------------------------------------

def _hand_metrics(lam, nA, nB, clip=1e-6):
    """An independent loop implementation of the statistics of committor_metrics."""
    p = np.clip(_sigmoid(lam), clip, 1 - clip)
    n = nA + nB
    q = nB / n
    ll = sum(b * math.log(pi) + a * math.log(1 - pi) for a, b, pi in zip(nA, nB, p)) / n.sum()
    qc = np.clip(q, clip, 1 - clip)
    sat = sum(b * math.log(qi) + a * math.log(1 - qi) for a, b, qi in zip(nA, nB, qc)) / n.sum()
    qbar = nB.sum() / n.sum()
    const = (nB.sum() * math.log(qbar) + nA.sum() * math.log(1 - qbar)) / n.sum()
    praw = _sigmoid(lam)
    r2 = 1 - ((q - praw) ** 2).sum() / ((q - q.mean()) ** 2).sum()
    qw = (n * q).sum() / n.sum()
    r2w = 1 - (n * (q - praw) ** 2).sum() / (n * (q - qw) ** 2).sum()
    tr = (q > 0.1) & (q < 0.9)
    rmse_tr = math.sqrt((n[tr] * (praw[tr] - q[tr]) ** 2).sum() / n[tr].sum())
    bias_tr = (n[tr] * (praw[tr] - q[tr])).sum() / n[tr].sum()
    sat_tr = int((tr & ((praw < 0.05) | (praw > 0.95))).sum())
    noise = [qi * (1 - qi) / (ni - 1) for qi, ni in zip(q, n)]
    ceiling = 1 - sum(noise) / ((q - q.mean()) ** 2).sum()
    return dict(loglik_per_endpoint=ll, loglik_saturated=sat, loglik_constant=const, r2=r2,
                r2_weighted=r2w, rmse_transition=rmse_tr, bias_transition=bias_tr,
                n_saturated_transition=sat_tr, n_transition=int(tr.sum()),
                mse_binomial_noise=float(np.mean(noise)), r2_noise_ceiling=ceiling)


def test_committor_metrics_hand_values():
    scipy_stats = pytest.importorskip('scipy.stats')
    lam = np.array([-6.0, -2.0, -0.5, 0.0, 0.4, 1.5, 3.0, 9.0, -40.0])
    nA = np.array([40, 30, 25, 20, 16, 8, 3, 0, 1], dtype=float)
    nB = np.array([0, 10, 15, 20, 24, 32, 37, 40, 39], dtype=float)
    got = validate.committor_metrics(lam, nA, nB)
    want = _hand_metrics(lam, nA, nB)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-10, abs=1e-12), key
    assert got['spearman'] == pytest.approx(scipy_stats.spearmanr(_sigmoid(lam), nB / (nA + nB))[0])
    assert got['n_entries'] == 9 and got['n_endpoints'] == int((nA + nB).sum())
    assert np.isfinite(got['loglik_per_endpoint'])           # the certain-and-wrong entry is clipped
    tighter = validate.committor_metrics(lam, nA, nB, prob_clip=1e-9)
    assert tighter['loglik_per_endpoint'] < got['loglik_per_endpoint']
    assert tighter['settings']['prob_clip'] == 1e-9


def test_noise_ceiling_is_what_the_exact_committor_scores():
    """On binomial counts the exact committor's R² is below 1, by about the noise ceiling."""
    rng = np.random.default_rng(11)
    lam = rng.uniform(-3, 3, 400)
    r2, ceiling = [], []
    for seed in range(40):
        nA, nB = _counts(lam, n=20, seed=seed)
        m = validate.committor_metrics(lam, nA, nB)
        r2.append(m['r2'])
        ceiling.append(m['r2_noise_ceiling'])
    assert np.mean(r2) == pytest.approx(np.mean(ceiling), abs=0.01)
    assert np.mean(ceiling) < 0.95
    # one endpoint per entry: no noise estimate
    single = validate.committor_metrics([0.0, 1.0], [1.0, 0.0], [0.0, 1.0])
    assert single['r2_noise_ceiling'] is None and single['mse_binomial_noise'] is None
    assert single['r2_noise_ceiling_entries'] is None and single['n_noise_ceiling_entries'] == 0
    # mixed: the ceiling and the R² held against it cover the same entries (n >= 2)
    lam = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    nA, nB = np.array([1.0, 15.0, 10.0, 6.0, 1.0]), np.array([0.0, 5.0, 10.0, 14.0, 19.0])
    mixed = validate.committor_metrics(lam, nA, nB)
    assert mixed['n_noise_ceiling_entries'] == 4 and mixed['n_entries'] == 5
    want = validate.committor_metrics(lam[1:], nA[1:], nB[1:])
    assert mixed['r2_noise_ceiling_entries'] == pytest.approx(want['r2'])
    assert mixed['r2_noise_ceiling'] == pytest.approx(want['r2_noise_ceiling'])
    assert mixed['r2'] != pytest.approx(want['r2'])
    # the two halves of one two-way shot are not independent: no ceiling
    halves = validate.committor_metrics(lam, np.full(5, 1.0), np.full(5, 1.0),
                                        independent_endpoints=False)
    assert halves['r2_noise_ceiling'] is None and halves['mse_binomial_noise'] is None
    reference = validate.CommittorReference('held-out', _molecule(5), [2, 1, 1, 0, 0],
                                            [0, 1, 1, 2, 2], kind='heldout_chains')
    report, _ = validate.score_committor(lambda positions: lam, reference)
    assert report['r2_noise_ceiling'] is None and report['r2'] is not None


def test_calibration_fit_recovers_known_line():
    lam = np.linspace(-5, 5, 41)
    q = _sigmoid(0.5 + 0.7 * lam)
    n = 1000.0
    a, b = validate.calibration_fit(lam, n * (1 - q), n * q)
    assert a == pytest.approx(0.5, abs=1e-6) and b == pytest.approx(0.7, abs=1e-6)
    a, b = validate.calibration_fit(lam, n * (1 - _sigmoid(lam)), n * _sigmoid(lam))
    assert a == pytest.approx(0.0, abs=1e-6) and b == pytest.approx(1.0, abs=1e-6)
    nA, nB = _counts(lam, n=20, seed=3, a=-0.3, b=1.6)
    a, b = validate.calibration_fit(lam, nA, nB)

    def nll(a_, b_):
        s = a_ + b_ * lam
        return float(np.sum(nB * np.logaddexp(0, -s) + nA * np.logaddexp(0, s)))

    best = nll(a, b)
    for da, db in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert nll(a + da, b + db) >= best - 1e-9
    # separated counts: the ridge keeps the slope finite and moderate
    separated = (lam > 0).astype(float)
    _, steep = validate.calibration_fit(lam, 1 - separated, separated, ridge=1e-2)
    assert np.isfinite(steep) and 3 < steep < 100


def test_calibration_and_metrics_refuse_what_they_cannot_judge():
    """Non-finite logits are an error (a NaN once read as the calibration (0, 1)); with every
    endpoint in one state the calibration line is not identifiable."""
    for bad in ([-2.0, 0.0, 2.0, np.nan], [-np.inf, 0.0, 1.0, 2.0]):
        with pytest.raises(ValueError, match='not finite'):
            validate.calibration_fit(bad, [3, 2, 1, 0], [0, 1, 2, 3])
        with pytest.raises(ValueError, match='not finite'):
            validate.committor_metrics(bad, [3, 2, 1, 0], [0, 1, 2, 3])
    # an entry without endpoints may carry any logit: it is dropped first
    ok = validate.committor_metrics([-2.0, 0.0, 2.0, np.nan], [3, 2, 1, 0], [0, 1, 2, 0])
    assert ok['n_entries'] == 3
    assert validate.calibration_fit([-2.0, 0.0, 2.0], [5, 5, 5], [0, 0, 0]) == (None, None)
    one_state = validate.committor_metrics([-2.0, 0.0, 2.0], [5, 5, 5], [0, 0, 0])
    assert one_state['calibration_slope'] is None and one_state['calibration_intercept'] is None


def test_metrics_of_a_constant_model_and_degenerate_sets():
    nA = np.array([10.0, 0.0, 5.0])
    nB = np.array([0.0, 10.0, 5.0])
    got = validate.committor_metrics(np.zeros(3), nA, nB)
    assert got['loglik_per_endpoint'] == pytest.approx(math.log(0.5))
    assert got['calibration_slope'] is None and got['spearman'] is None
    end_only = validate.committor_metrics(np.array([-3.0, 3.0]), np.array([10.0, 0.0]),
                                          np.array([0.0, 10.0]))
    assert end_only['n_transition'] == 0 and end_only['rmse_transition'] is None
    with pytest.raises(ValueError, match='no endpoints'):
        validate.committor_metrics(np.zeros(2), np.zeros(2), np.zeros(2))


def test_score_committor_scores_included_entries_only():
    x = _molecule(6)
    lam_true = np.linspace(-3, 3, 6)
    nA, nB = _counts(lam_true, seed=1)
    include = np.array([True, True, False, True, True, True])
    reference = validate.CommittorReference('set', x, nA, nB, include=include,
                                            notes={'source': 'synthetic'})

    def committor(positions):
        assert positions.shape == (6, M, 3)
        return lam_true.copy()

    report, lam = validate.score_committor(committor, reference)
    assert np.array_equal(lam, lam_true)
    want = validate.committor_metrics(lam_true[include], nA[include], nB[include])
    assert report['loglik_per_endpoint'] == want['loglik_per_endpoint']
    assert report['n_scored'] == 5 and report['n_excluded'] == 1
    assert report['kind'] == 'brute_force' and report['notes'] == {'source': 'synthetic'}
    np.testing.assert_allclose(reference.committor, nB / (nA + nB))
    with pytest.raises(ValueError, match='kind'):
        validate.CommittorReference('bad', x, nA, nB, kind='umbrella')
    with pytest.raises(ValueError, match='shape'):
        validate.CommittorReference('bad', x, nA[:3], nB[:3])
    # a committor that takes the boxes gets them
    seen = {}

    def boxed(positions, dimensions=None):
        seen['dimensions'] = dimensions
        return lam_true.copy()

    with_box = validate.CommittorReference('box', x, nA, nB, dimensions=[30, 30, 30, 90, 90, 90])
    validate.score_committor(boxed, with_box)
    assert seen['dimensions'].shape == (6, 6)
    with pytest.raises(ValueError, match='non-finite'):
        validate.score_committor(lambda positions: np.full(6, np.nan), reference)


def test_heldout_chain_reference(tmp_path):
    """Shooting points and two-way outcomes of the held-out chains of a run folder."""
    from aimmd import _config
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=3, n_shots=4, n_atoms=2)
    params = cgaimmd_params_namespace(network=TinyFlowNetwork(n_atoms=2))
    resident = set(_config.NPY_CACHE._cache)
    reference = validate.heldout_chain_reference(params, rd.run, [1])
    assert set(_config.NPY_CACHE._cache) <= resident            # what the call read is evicted
    assert reference.kind == 'heldout_chains'
    assert len(reference.n_first) == 4
    assert np.all(reference.n_first + reference.n_last == 2)
    assert set(reference.groups.tolist()) == {1}
    rows = [np.asarray(p.positions, dtype=float)[rd.shooting_index] for p in rd.chains[1]]
    got = {tuple(np.round(r.reshape(-1), 4)) for r in reference.positions.reshape(4, -1)}
    assert got == {tuple(np.round(r.reshape(-1), 4)) for r in rows}
    assert reference.notes['chains'] == [1] and reference.notes['n_shooting_points_read'] == 4
    assert reference.notes['descriptors_series'] == 'descriptors'
    excluded = validate.heldout_chain_reference(
        params, rd.run, [1, 2], n_atoms=2,
        exclude_frames={str(rd.chains[1][0].fnames[0]): 'all'})
    assert len(excluded.n_first) == 7
    assert excluded.notes['excluded_shooting_points'] == 1
    # the shots of a run that records a bias are a reference only when the caller allows it
    params.__dict__['record_bias'] = True
    with pytest.raises(ValueError, match='records a bias.*allow_biased_frames'):
        validate.heldout_chain_reference(params, rd.run, [1])
    allowed = validate.heldout_chain_reference(params, rd.run, [1], allow_biased_frames=True)
    np.testing.assert_array_equal(allowed.n_last, reference.n_last)


# ---------------------------------------------------------------------------
# reference loaders and the committor of any AIMMD network
# ---------------------------------------------------------------------------

def _nanma_params(network, descriptors_function, values_function=None):
    import aimmd
    from aimmd.params.utils import create_default_values_function
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(states='ARB', network=network, descriptors_function=descriptors_function,
                           descriptor_transform=None,
                           values_function=values_function or create_default_values_function(network))
    return params


class DihedralNetwork(torch.nn.Module):
    """An ordinary AIMMD network on (phi, psi) descriptors: logit = w . (cos, sin) features + b."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.5, 2.0, -1.0, 0.3]]))
            self.linear.bias.fill_(0.2)

    def forward(self, x):
        rad = torch.deg2rad(torch.as_tensor(x, dtype=torch.float32))
        return self.linear(torch.cat([torch.cos(rad), torch.sin(rad)], dim=1))


def test_npz_reference_in_the_nanma_format_scored_with_any_network():
    """The NANMA subset: coordinates in nm, counts nA/nB; an ordinary descriptor network and the flat-coordinate route agree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('nanma_states', NANMA / 'states.py')
    states = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(states)
    reference = validate.CommittorReference.from_npz(NANMA_REFERENCE)
    with np.load(NANMA_REFERENCE) as z:
        xyz, nA, nB, q, phi, psi = (z[k] for k in ('xyz_nm', 'nA', 'nB', 'q', 'phi', 'psi'))
    assert reference.name == 'nanma_committor_reference_subset' and len(reference) == 16
    np.testing.assert_allclose(reference.positions, 10.0 * xyz.astype(float))
    np.testing.assert_array_equal(reference.n_first, nA)
    np.testing.assert_allclose(reference.committor, q)
    assert reference.notes['units'] == 'nm' and len(reference.notes['sha256']) == 64
    assert reference.notes['meta']['source_meta']['shots'].startswith('one-way')
    assert reference.notes['shots_without_end_state'] == 0
    # an ordinary AIMMD network on (phi, psi) descriptors, through its params
    network = DihedralNetwork()
    params = _nanma_params(network, states.dihedrals)
    report, lam = validate.score_committor(validate.params_committor(params), reference)
    with torch.no_grad():
        want = network(torch.tensor(np.stack([phi, psi], 1))).numpy().reshape(-1)
    np.testing.assert_allclose(lam, want, atol=2e-3)          # dihedrals from the coordinates
    assert report['n_endpoints'] == int((nA + nB).sum())
    assert report == validate.score_committor(lambda x: lam, reference)[0] | {
        'seconds': report['seconds']}
    # the network object given directly; the same numbers
    alone = validate.params_committor(_nanma_params(DihedralNetwork(), states.dihedrals),
                                      network=network, batch_size=5)
    np.testing.assert_allclose(alone(reference.positions), lam, atol=1e-6)
    # a C-GAIMMD network on flat coordinates: through its params or directly
    tiny = TinyFlowNetwork(n_atoms=22)
    with torch.no_grad():
        tiny.committor_net[-1].weight.fill_(0.3)
    flat = _nanma_params(tiny, lambda trajectory: np.array([ts.positions.reshape(-1)
                                                            for ts in trajectory]))
    via_params = validate.params_committor(flat)(reference.positions)
    direct = validate.network_committor(tiny)(reference.positions)
    np.testing.assert_allclose(via_params, direct, atol=1e-5)
    # atoms and units
    heavy = validate.CommittorReference.from_npz(NANMA_REFERENCE, atoms=[4, 6, 8, 14],
                                                 name='heavy')
    assert heavy.positions.shape == (16, 4, 3) and heavy.name == 'heavy'
    angstrom = validate.CommittorReference.from_arrays('a', xyz, nA, nB, units='nm')
    np.testing.assert_allclose(angstrom.positions, reference.positions)
    with pytest.raises(ValueError, match='units'):
        validate.CommittorReference.from_arrays('a', xyz, nA, nB, units='pm')


def _write_frames(fname, positions, box=(40.0, 40.0, 40.0, 90.0, 90.0, 90.0)):
    import MDAnalysis as mda
    universe = mda.Universe.empty(positions.shape[1], trajectory=True)
    with mda.Writer(str(fname), n_atoms=positions.shape[1]) as writer:
        for t, x in enumerate(positions):
            universe.atoms.positions = x
            universe.dimensions = list(box)
            universe.trajectory.ts.time = float(t)
            writer.write(universe.atoms)


def test_trajectory_reference_with_shot_counts(tmp_path):
    """The chignolin format (n_A, n_B) and the trypsin format (fraction and shots per frame)."""
    rng = np.random.default_rng(5)
    frames = np.round(rng.uniform(0, 20, size=(6, 5, 3)), 2).astype(np.float32)
    folder = tmp_path / 'set'
    folder.mkdir()
    _write_frames(folder / 'frames.xtc', frames)
    _write_frames(folder / 'more.xtc', frames[:2] + 1.0)
    np.savez(folder / 'outcomes.npz', n_A=np.array([40, 39, 20, 2, 0, 10]),
             n_B=np.array([0, 1, 20, 38, 42, 30]))
    np.savez(folder / 'ground_truth.npz', sampled=np.array([0.0, 0.5, 0.25, 1.0, 0.75, 0.5]),
             nshots=np.array([48.0, 46.0, 40.0, 32.0, 44.0, 48.0]))
    before = sorted(os.listdir(folder))
    n_first, n_last = validate.shot_counts(folder / 'outcomes.npz')
    assert n_first.tolist() == [40, 39, 20, 2, 0, 10]
    reference = validate.CommittorReference.from_trajectory(folder / 'frames.xtc', n_first, n_last)
    assert sorted(os.listdir(folder)) == before                # no index file next to the frames
    assert reference.name == 'frames' and reference.positions.shape == (6, 5, 3)
    np.testing.assert_allclose(reference.positions, frames, atol=0.006)
    assert reference.dimensions.shape == (6, 6) and reference.dimensions[0, 0] == 40.0
    assert reference.notes['trajectory'] == [str(folder / 'frames.xtc')]
    a, b = validate.shot_counts(folder / 'ground_truth.npz', fraction_last='sampled',
                                n_total='nshots')
    assert b.tolist() == [0.0, 23.0, 10.0, 32.0, 33.0, 24.0]
    assert (a + b).tolist() == [48.0, 46.0, 40.0, 32.0, 44.0, 48.0]
    with pytest.raises(ValueError, match='go together'):
        validate.shot_counts(folder / 'ground_truth.npz', fraction_last='sampled')
    subset = validate.CommittorReference.from_trajectory(
        [folder / 'frames.xtc', folder / 'more.xtc'], np.ones(3), np.ones(3), frames=[0, 6, 7],
        atoms=[0, 2], name='picked')
    assert subset.positions.shape == (3, 2, 3)
    np.testing.assert_allclose(subset.positions[1], frames[0][[0, 2]] + 1.0, atol=0.006)
    assert subset.notes['frames'] == [0, 6, 7] and subset.notes['atoms'] == [0, 2]


def test_reference_from_the_free_simulations(tmp_path):
    from aimmd.analysis.free_md import FreeMDReference
    rows = np.arange(4 * 6, dtype=float).reshape(4, 6)
    free = FreeMDReference(name='free', files=['a.xtc'], file=np.zeros(4, int),
                           frame=np.arange(4), trajectory=np.zeros(4, int),
                           n_first=np.array([1, 1, 0, 0]), n_last=np.array([0, 0, 1, 1]),
                           groups=np.array([0, 0, 1, 1]), values=np.zeros(4),
                           descriptors=rows, notes={'free_restart_source': 'equilibrium'})
    reference = validate.CommittorReference.from_free_md(free, n_atoms=2)
    assert reference.kind == 'free_md_forward' and reference.positions.shape == (4, 2, 3)
    assert reference.groups.tolist() == [0, 0, 1, 1]
    assert reference.notes['free_restart_source'] == 'equilibrium'
    report, _ = validate.score_committor(lambda x: np.array([-2.0, -1.0, 1.0, 2.0]), reference)
    assert report['kind'] == 'free_md_forward' and report['n_endpoints'] == 4
    free.descriptors = None
    with pytest.raises(ValueError, match='descriptor rows'):
        validate.CommittorReference.from_free_md(free)
    with pytest.raises(ValueError, match='atoms'):
        validate.CommittorReference.from_free_md(free, positions=np.zeros((4, 3, 3)), n_atoms=2)


def test_parity_figure(tmp_path):
    reference = validate.CommittorReference.from_npz(NANMA_REFERENCE)
    lam_true = np.log(np.clip(reference.committor, 0.02, 0.98) / (1 - np.clip(reference.committor,
                                                                            0.02, 0.98)))
    rng = np.random.default_rng(0)
    groups = validate.CommittorReference('held-out chains', reference.positions, np.ones(16),
                                         np.ones(16), kind='heldout_chains')
    png, pdf = validate.parity_figure(
        [('distance-matrix network', reference, lam_true + rng.normal(0, 0.8, 16)),
         ('C-GAIMMD model', reference, 1.5 * lam_true),
         ('C-GAIMMD model, held-out chains', groups, lam_true)], tmp_path / 'figures' / 'parity',
        title='NANMA committor parity')
    assert Path(png).stat().st_size > 20_000 and Path(pdf).exists()
    with pytest.raises(ValueError, match='at least one'):
        validate.parity_figure([], tmp_path / 'empty')


# ---------------------------------------------------------------------------
# sampling through the generation interface
# ---------------------------------------------------------------------------

def test_draw_samples_through_the_generation_interface():
    torch.manual_seed(0)
    net = _randomize(TinyFlowNetwork(n_atoms=3, lambda_clip=10.0))
    one = validate.draw_samples(net, [-2.0, 0.0, 2.0], 5, chunk=2, seed=1)
    assert one.positions.shape == (3, 5, 3, 3)
    assert one.settings['generator']['method'] == 'ode' and one.settings['chunk'] == 2
    again = validate.draw_samples(net, [-2.0, 0.0, 2.0], 5, chunk=2, seed=1)
    assert np.array_equal(one.positions, again.positions)
    other = validate.draw_samples(net, [-2.0, 0.0, 2.0], 5, chunk=2, seed=2)
    assert not np.allclose(one.positions, other.positions)
    # a level keeps its samples whatever other levels are drawn with it
    alone = validate.draw_samples(net, [2.0], 5, chunk=2, seed=1)
    assert np.array_equal(alone.positions[0], one.positions[2])
    steps = validate.draw_samples(net, [0.0], 2, generator={'method': 'ode', 'n_steps': 2})
    assert steps.settings['generator']['settings']['n_steps'] == 2
    bank = validate.draw_samples(net, [0.0, 1.0], 4, generator=FrameBankGenerator(np.zeros((3, 3, 3))))
    assert bank.settings['generator']['method'] == 'test_frame_bank'
    assert np.abs(bank.positions).max() < 0.1

    class Refuses(FrameBankGenerator):
        def check(self, network):
            return ['needs a velocity field']

    with pytest.raises(ValueError, match='needs a velocity field'):
        validate.draw_samples(net, [0.0], 2, generator=Refuses(np.zeros((1, 3, 3))))


# ---------------------------------------------------------------------------
# M1: validity
# ---------------------------------------------------------------------------

def _water_box(n_waters=2, seed=0):
    rng = np.random.default_rng(seed)
    solute = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]])
    blocks = [solute] + [water + np.array([5.0 * (k + 1), 0.0, 0.0]) for k in range(n_waters)]
    x = np.concatenate(blocks) + 0.01 * rng.standard_normal((2 + 3 * n_waters, 3))
    return x, np.array([6, 8] + [8, 1, 1] * n_waters)


def test_waters_intact_is_label_free():
    x, elements = _water_box(2)
    assert validate.waters_intact(x[None], elements, n_solute=2, n_waters=2).tolist() == [True]
    swapped = x.copy()
    swapped[[3, 6]] = swapped[[6, 3]]
    assert validate.waters_intact(swapped[None], elements, n_solute=2, n_waters=2)[0]
    broken = x.copy()
    broken[4] += np.array([0.0, 0.0, 3.0])
    merged = x.copy()
    merged[5:8] = x[2:5] + np.array([0.0, 1.6, 0.0])
    out = validate.waters_intact(np.stack([broken, merged]), elements, n_solute=2, n_waters=2)
    assert out.tolist() == [False, False]
    assert validate.waters_intact(x[None], elements, n_solute=8, n_waters=0).tolist() == [True]


class _FakeValidity:
    """ok unless atom 0 sits at x > 0.5 ('clash') or the sample is non-finite."""

    def __call__(self, positions, context=None):
        x = np.asarray(positions, dtype=float)
        ok = np.isfinite(x).all(axis=(1, 2)) & (x[:, 0, 0] <= 0.5)
        reason = np.array(['' if o else 'clash' for o in ok], dtype=object)
        return ValidityResult(positions=x[:, ::-1].copy(), ok=ok, reason=reason,
                              info=[{} for _ in ok])


def test_validity_statistics_per_level():
    x = np.zeros((2, 4, 3, 3))
    x[0, :, 0, 0] = [0.0, 1.0, 0.0, 0.0]
    x[1, :, 0, 0] = [1.0, 1.0, 0.0, 2.0]
    check = validate.check_validity(x, _FakeValidity())
    assert check['ok'].tolist() == [[True, False, True, True], [False, False, True, False]]
    assert np.array_equal(check['positions'], x[:, :, ::-1])
    levels = validate.validity_summary([-1.0, 1.0], check)
    assert levels[0]['valid'] == 0.75 and levels[1]['valid'] == 0.25
    assert levels[1]['reasons'] == {'clash': 0.75}
    unchecked = validate.check_validity(x, None)
    assert unchecked['ok'].all() and unchecked['checked'] is False
    assert not unchecked['reflected'].any() and levels[0]['reflected'] == 0.0


class _FakeReflectingValidity(_FakeValidity):
    def __call__(self, positions, context=None):
        r = super().__call__(positions, context)
        x = np.asarray(positions, dtype=float)
        return ValidityResult(positions=r.positions, ok=r.ok, reason=r.reason,
                              info=[{'reflected': bool(v < 0)} for v in x[:, 0, 0]])


def test_reflected_fraction_per_level():
    x = np.zeros((1, 4, 3, 3))
    x[0, :, 0, 0] = [-1.0, -0.5, 0.0, 1.0]
    check = validate.check_validity(x, _FakeReflectingValidity())
    assert check['reflected'].tolist() == [[True, True, False, False]]
    assert validate.validity_summary([0.0], check)[0]['reflected'] == 0.5
    single = validate.check_validity(x[0], _FakeReflectingValidity())
    assert single['reflected'].tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# M3 and the guard
# ---------------------------------------------------------------------------

def test_generator_consistency_statistics():
    scipy_stats = pytest.importorskip('scipy.stats')
    requested = np.array([-2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    measured = np.array([-2.5, -1.0, 0.2, 0.4, 1.0, 5.0, 2.9])
    got = validate.generator_consistency(requested, measured)
    assert got['spearman'] == pytest.approx(scipy_stats.spearmanr(requested, measured)[0])
    assert got['within_tolerance'] == pytest.approx(6 / 7)
    assert got['median_abs_error'] == pytest.approx(np.median(np.abs(measured - requested)))
    per = {row['lambda']: row for row in got['per_level']}
    assert per[2.0]['n'] == 3 and per[2.0]['median'] == pytest.approx(2.9)
    assert got['transition_region']['n'] == 7
    assert validate.generator_consistency(requested, np.zeros(7))['spearman'] is None
    assert validate.generator_consistency(np.zeros(0), np.zeros(0))['n'] == 0


def test_symmetric_rmsd_minimizes_over_symmetry_and_rotation():
    x = _molecule(1, seed=4)[0]
    y = x.copy()
    y[[1, 2]] = y[[2, 1]]
    y = y @ _rotation(3).T + np.array([4.0, -1.0, 2.0])
    assert validate.symmetric_rmsd(x[None], y[None])[0] > 0.1
    spec = SimpleNamespace(kwargs=lambda: {'groups': (), 'factors': (((1, 2), ((0, 1), (1, 0))),),
                                           'molecule_blocks': ()})
    assert validate.symmetric_rmsd(x[None], y[None], symmetry=spec)[0] < 1e-5
    assert validate.symmetric_rmsd(x[None], y[None], atoms=[0, 3, 4, 6])[0] < 1e-5
    mirror = x * np.array([-1.0, 1.0, 1.0])
    assert validate.symmetric_rmsd(x[None], mirror[None])[0] > 0.1
    groups = SimpleNamespace(kwargs=lambda: {'groups': ((1, 2),), 'factors': (),
                                             'molecule_blocks': ()})
    assert validate.symmetric_rmsd(x[None], y[None], symmetry=groups)[0] < 1e-5


def test_nearest_frames_finds_rotated_copies():
    frames = _molecule(40, seed=5, noise=0.4)
    rng = np.random.default_rng(0)
    pick = np.array([7, 21, 33])
    x = np.stack([frames[i] @ _rotation(i).T + 3.0 for i in pick])
    x = x + 1e-3 * rng.standard_normal(x.shape)
    distance, index = validate.nearest_frames(x, frames, k=3, chunk=7)
    assert index.tolist() == pick.tolist() and np.all(distance < 5e-3)
    swapped = x.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    spec = SimpleNamespace(kwargs=lambda: {'groups': ((1, 2),), 'factors': (),
                                           'molecule_blocks': ()})
    d_sym, i_sym = validate.nearest_frames(swapped, frames, symmetry=spec, k=3)
    assert i_sym.tolist() == pick.tolist() and np.all(d_sym < 5e-3)


def test_guard_statistics_against_the_heldout_baseline():
    rng = np.random.default_rng(1)
    baseline = rng.uniform(1.0, 2.0, 200)
    copies = np.array([0.01, 0.02, 1.5, 1.8])
    got = validate.guard_summary(copies, baseline)
    assert got['baseline_p5'] == pytest.approx(float(np.percentile(baseline, 5)))
    assert got['fraction_below_baseline_p5'] == pytest.approx(0.5)
    assert got['median'] == pytest.approx(np.median(copies))


# ---------------------------------------------------------------------------
# one call per checkpoint
# ---------------------------------------------------------------------------

def _setup_validation():
    net = _joint(randomize=True)
    bank = _molecule(10, seed=1, noise=0.1)
    x_ref = _molecule(8, seed=7)
    lam_ref = coordinate_committor(net, x_ref).detach().numpy().astype(float)
    nA, nB = _counts(lam_ref, n=30, seed=0)
    reference = validate.CommittorReference('synthetic brute force', x_ref, nA, nB,
                                            notes={'what': 'counts drawn from the model'})
    return SimpleNamespace(net=net, generator=FrameBankGenerator(bank[:5], sigma=0.01),
                           reference=reference, training=bank[:5],
                           heldout=_molecule(6, seed=9, noise=0.1))


def test_validate_writes_report_card_and_samples(tmp_path):
    s = _setup_validation()
    out = tmp_path / 'tiny_model'
    report = validate.validate(
        s.net, out, label='tiny joint model', system='seven-atom blob', levels=[-2.0, 0.0, 2.0],
        n_per_level=8, generator=s.generator, references=[s.reference],
        independent_committor=lambda x: np.zeros(len(x)), independent_name='zero model',
        training_frames=s.training, heldout_frames=s.heldout, guard_per_level=4,
        adherence_boot=20, metadata={'checkpoint': {'path': 'none'}})
    files = report['files']
    for key in ('json', 'card_png', 'card_pdf', 'samples'):
        assert Path(files[key]).exists(), key
    assert sorted(os.listdir(tmp_path)) == ['tiny_model.json', 'tiny_model_card.pdf',
                                            'tiny_model_card.png', 'tiny_model_samples.npz']
    on_disk = validate.load_report(files['json'])
    assert on_disk['schema'] == validate.SCHEMA == 'aimmd.cgaimmd.validate/1'
    assert on_disk['label'] == 'tiny joint model' and on_disk['system'] == 'seven-atom blob'
    assert on_disk['settings']['generator'] == {'method': 'test_frame_bank',
                                                'settings': {'n_frames': 5, 'sigma': 0.01}}
    assert [row['lambda'] for row in on_disk['m1']['levels']] == [-2.0, 0.0, 2.0]
    assert on_disk['m1']['levels'][0]['n'] == 8 and on_disk['m1']['checked'] is False
    m2 = on_disk['m2']['synthetic brute force']
    lam = np.load(files['samples'])['m2_logits_0']
    want = validate.committor_metrics(lam, s.reference.n_first, s.reference.n_last)
    assert m2['loglik_per_endpoint'] == pytest.approx(want['loglik_per_endpoint'])
    assert m2['spearman'] == pytest.approx(want['spearman'])
    m3 = on_disk['m3']
    assert m3['own']['n'] == 24 and m3['independent']['n'] == 24
    assert m3['independent_name'] == 'zero model'
    assert set(m3['own_adherence']) >= {'resolution_all', 'resolution_band', 'median_slopes',
                                        'intervals'}
    assert m3['independent_adherence']['slope_all'] == pytest.approx(0.0, abs=1e-12)
    guard = on_disk['guard']
    assert guard['training']['n_samples'] == 12
    # the stand-in generator copies training frames (+ 0.01 A noise): the guard sees it
    assert guard['training']['median'] < 0.05
    assert guard['training']['fraction_below_baseline_p5'] == 1.0
    assert on_disk['metadata'] == {'checkpoint': {'path': 'none'}}
    assert on_disk['timing']['total_seconds'] > 0 and on_disk['settings']['seed'] == 0
    assert 'aimmd_version' in on_disk['environment']


def test_validate_without_optional_parts(tmp_path):
    torch.manual_seed(0)
    net = _randomize(TinyFlowNetwork(n_atoms=3, lambda_clip=10.0))
    report = validate.validate(net, tmp_path / 'toy', levels=[0.0], n_per_level=4,
                               generator={'method': 'ode', 'n_steps': 2}, card=False,
                               save_samples=False)
    assert report['m2'] == {} and report['guard'] is None
    assert report['m3']['independent'] is None and report['m3']['independent_adherence'] is None
    assert 'card_png' not in report['files'] and 'samples' not in report['files']
    assert json.loads(Path(report['files']['json']).read_text())['m1']['levels'][0]['valid'] == 1.0


def test_model_card_takes_a_bare_checkpoint_path(tmp_path):
    torch.manual_seed(0)
    net = _randomize(TinyFlowNetwork(n_atoms=3, lambda_clip=10.0))
    meta = {'checkpoint': '/some/run/networkARB.h5', 'checkpoint_sha256': 'ab' * 32}
    report = validate.validate(net, tmp_path / 'toy', levels=[0.0, 1.0], n_per_level=4,
                               generator={'method': 'ode', 'n_steps': 2}, save_samples=False,
                               metadata=meta, adherence_boot=0)
    assert Path(report['files']['card_png']).exists()
    assert report['metadata'] == meta


def test_redraw_card_from_the_saved_arrays(tmp_path):
    s = _setup_validation()
    report = validate.validate(s.net, tmp_path / 'm', levels=[-1.0, 1.0], n_per_level=4,
                               generator=s.generator, references=[s.reference],
                               training_frames=s.training, heldout_frames=s.heldout,
                               guard_per_level=2, card=False, adherence_boot=0)
    png, pdf = validate.redraw_card(report['files']['json'])
    assert png == str(tmp_path / 'm_card.png') and Path(png).stat().st_size > 10_000
    assert Path(pdf).exists()


def test_old_reports_are_read_with_the_new_names(tmp_path, recwarn):
    from aimmd.cgaimmd import _legacy
    _legacy._reset()
    old = {'schema': 'aimmd.flow.validate/1', 'label': 'archived',
           'settings': {'sampler': 'midpoint', 'n_steps': 16, 'chunk': 128,
                        'generate': 'aimmd.flow.sampler.generate'}}
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(old))
    report = validate.load_report(path)
    assert report['schema'] == 'aimmd.cgaimmd.validate/1'
    assert report['settings']['generator'] == {
        'method': 'ode', 'settings': {'sampler': 'midpoint', 'n_steps': 16, 'chunk': 128}}
    messages = [str(w.message) for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert any("'aimmd.flow.validate/1'" in m for m in messages)
    path.write_text(json.dumps({'schema': 'something else'}))
    with pytest.raises(ValueError, match='not an aimmd.cgaimmd.validate report'):
        validate.load_report(path)


class _ShiftMinimizer:
    """A minimize-function stand-in: moves every atom by 0.1 Å, fails for x[0,0] > 0.4."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, context):
        self.calls += 1
        assert context.network is not None and Path(context.scratch).is_dir()
        x = np.asarray(frame.positions, dtype=float)
        if x[0, 0] - frame.dimensions[0] / 2 > 0.4:
            return PreparedFrame(positions=x, dimensions=frame.dimensions, ok=False,
                                 reason='minimize_failed: test')
        return PreparedFrame(positions=x + 0.1, dimensions=frame.dimensions,
                             record={'heavy_rms_displacement_A': 0.17})


def test_validate_polish(tmp_path):
    torch.manual_seed(0)
    net = _randomize(TinyFlowNetwork(n_atoms=3, prior_center=0.0, prior_width=0.3,
                                     lambda_clip=10.0), scale=0.0)
    minimizer = _ShiftMinimizer()
    report = validate.validate(net, tmp_path / 'polished', levels=[0.0, 1.0], n_per_level=6,
                               generator={'method': 'ode', 'n_steps': 2},
                               validity=_FakeValidity(), minimizer=minimizer,
                               polish_per_level=4, polish_box=[30.0, 30.0, 30.0, 90, 90, 90],
                               polish_params=SimpleNamespace(), card=False, adherence_boot=0)
    assert minimizer.calls == 8
    for row in report['m1']['levels']:
        polish = row['polish']
        assert polish['n'] == 4 and polish['minimize_failed'] + polish['minimized'] == 4
        assert 0.0 <= polish['valid_after'] <= 1.0
    n_after = sum(round(row['polish']['valid_after'] * 4) for row in report['m1']['levels'])
    assert report['m3']['own_after_polish']['n'] == n_after
    assert report['m3']['independent_after_polish'] is None
    assert sorted(os.listdir(tmp_path)) == ['polished.json', 'polished_samples.npz']
    with pytest.raises(ValueError, match='polish_params'):
        validate.validate(net, tmp_path / 'x', levels=[0.0], n_per_level=2,
                          minimizer=minimizer, polish_per_level=2, card=False)


def test_validate_nanma_end_to_end(tmp_path):
    """Real validity, symmetry and guard on NANMA (no GROMACS), with frames of free MD as samples."""
    pytest.importorskip('networkx')
    mda = pytest.importorskip('MDAnalysis')
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    from aimmd.cgaimmd.preparation.validity import TopologyValidity

    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    universe = mda.Universe(str(NANMA / 'conf.gro'), [str(NANMA / 'free_md_A.xtc'),
                                                      str(NANMA / 'free_md_B.xtc')])
    frames = np.array([descriptors.select(ts.positions, ts.dimensions)
                       for ts in universe.trajectory])
    net = JointPaiNN.from_topology(str(NANMA / 'topol.top'), descriptors, edge_backend='torch',
                                   **TINY)
    net.eval()
    validity = TopologyValidity(str(NANMA / 'topol.top'), descriptors,
                                node_labels=net.atom_tokens_numpy(),
                                clash_reference=[str(NANMA / 'free_md_A.xtc'),
                                                 str(NANMA / 'free_md_B.xtc')])
    lam_ref = coordinate_committor(net, frames).detach().numpy().astype(float)
    reference = validate.CommittorReference('free MD stand-in', frames, *_counts(lam_ref))
    report = validate.validate(
        net, tmp_path / 'nanma', levels=[-4.0, 0.0, 4.0], n_per_level=6,
        generator=FrameBankGenerator(frames[:5], sigma=0.01), validity=validity,
        references=[reference], training_frames=frames[:5], heldout_frames=frames[5:10],
        guard_per_level=6, adherence_boot=0)
    assert all(row['valid'] == 1.0 for row in report['m1']['levels'])
    assert report['m1']['checked'] is True
    assert report['guard']['symmetry'] == 'network'
    n_heavy = int((net.atomic_numbers != 1).sum())
    assert report['guard']['atoms'] == f'heavy atoms of the generator atoms ({n_heavy})'
    assert report['guard']['training']['median'] < 0.05
    assert Path(report['files']['card_png']).stat().st_size > 10_000


def test_leaderboard_table_and_figure(tmp_path):
    s = _setup_validation()
    paths = []
    for k, label in enumerate(('model one', 'model two')):
        report = validate.validate(
            s.net, tmp_path / f'm{k}', label=label, system='blob', levels=[-1.0, 1.0],
            n_per_level=4, generator=s.generator, references=[s.reference],
            training_frames=s.training, heldout_frames=s.heldout, guard_per_level=2, seed=k,
            card=False, adherence_boot=0)
        paths.append(report['files']['json'])
    board = validate.leaderboard(paths, tmp_path / 'board', title='test leaderboard')
    assert [row['label'] for row in board['rows']] == ['model one', 'model two']
    for key in ('csv', 'markdown', 'json', 'png', 'pdf'):
        assert Path(board['files'][key]).exists(), key
    csv = Path(board['files']['csv']).read_text().splitlines()
    assert csv[0].startswith('label,system,generator,') and len(csv) == 3
    assert 'synthetic brute force' in Path(board['files']['markdown']).read_text()
    row = board['rows'][0]
    assert row['generator'] == 'test_frame_bank'
    assert row['m2:synthetic brute force:loglik_per_endpoint'] == pytest.approx(
        validate.load_report(paths[0])['m2']['synthetic brute force']['loglik_per_endpoint'])


@pytest.mark.slow
@pytest.mark.gromacs
def test_gromacs_polish_nanma(tmp_path, gmx_executable):
    """Restrained EM of generated NANMA samples with GromacsMinimizer, through validate's polish."""
    mda = pytest.importorskip('MDAnalysis')
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    from aimmd.cgaimmd.preparation.minimize import GromacsMinimizer
    from aimmd.cgaimmd.preparation.validity import TopologyValidity

    descriptors = GeneratorDescriptors(str(NANMA / 'conf.gro'), GeneratorAtoms(22))
    universe = mda.Universe(str(NANMA / 'conf.gro'), str(NANMA / 'free_md_A.xtc'))
    frames = np.array([descriptors.select(ts.positions, ts.dimensions)
                       for ts in universe.trajectory])
    net = JointPaiNN.from_topology(str(NANMA / 'topol.top'), descriptors, edge_backend='torch',
                                   **TINY)
    net.eval()
    minimizer = GromacsMinimizer(em_mdp=str(NANMA / 'em.mdp'), restrain='generator_heavy',
                                 grompp=f'{gmx_executable} grompp -maxwarn 5',
                                 mdrun=f'{gmx_executable} mdrun -ntmpi 1 -ntomp 1 -nb cpu -pin off',
                                 topology=str(NANMA / 'topol.top'))
    polish_params = SimpleNamespace(universe=mda.Universe(str(NANMA / 'conf.gro')),
                                    parent=str(NANMA))
    validity = TopologyValidity(str(NANMA / 'topol.top'), descriptors,
                                node_labels=net.atom_tokens_numpy())
    report = validate.validate(net, tmp_path / 'polish', levels=[0.0], n_per_level=3,
                               generator=FrameBankGenerator(frames, sigma=0.05),
                               validity=validity, minimizer=minimizer, polish_per_level=3,
                               polish_params=polish_params, polish_workers=2, card=False,
                               adherence_boot=0)
    polish = report['m1']['levels'][0]['polish']
    assert polish['minimized'] == 3 and polish['valid_after'] == 1.0
    assert 0.0 < polish['median_heavy_rms_displacement_A'] < 0.2
