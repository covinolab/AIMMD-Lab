"""Target committor laws of C-GAIMMD shooting and the lambda window."""

import json
from math import inf

import numpy as np
import pytest

from aimmd.analysis.utils import bin_centers
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd import lambda_law
from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble import PathEnsemble
from aimmd.worker import utils as worker_utils
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, make_candidate_context
from tests._helpers_unit import write_trajectory

L = 10.0


def _context(tmp_path, *, bins=None, densities=None, params_overrides=None, **settings):
    settings.setdefault('lambda_clip', L)
    context = make_candidate_context(tmp_path, **settings)
    context.params.__dict__.update(params_overrides or {})
    context.bins = None if bins is None else np.asarray(bins, dtype=float)
    context.densities = None if densities is None else np.asarray(densities, dtype=float)
    return context


def _trainer_cutoffs(reweight_parameters, bins):
    """The trainer's former inline rule (aimmd.worker._train before the reweighting helpers)."""
    rw_p = dict(reweight_parameters)
    sp_cutoff_min, sp_cutoff_max = bins[0], bins[-1]
    if sp_cutoff_min == -inf and bins[1] < +inf:
        sp_cutoff_min = bins[1]
    if sp_cutoff_max == +inf and bins[-2] > -inf:
        sp_cutoff_max = bins[-2]
    rw_p.setdefault('sp_cutoff_min', sp_cutoff_min)
    rw_p.setdefault('sp_cutoff_max', sp_cutoff_max)
    return float(rw_p['sp_cutoff_min']), float(rw_p['sp_cutoff_max'])


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------

def test_uniform(tmp_path):
    """'uniform': lo + (hi - lo) U(0, 1) from the context's stream; reproducible."""
    context = _context(tmp_path, lambda_window=(-2.0, 3.0))
    targets, record = lambda_law.draw_targets(4000, context)
    assert targets.shape == (4000,) and targets.dtype == np.float64
    assert targets.min() >= -2.0 and targets.max() <= 3.0
    assert abs(targets.mean() - 0.5) < 0.1
    assert record == {'name': 'uniform', 'window': [-2.0, 3.0], 'window_source': 'setting'}
    expected = -2.0 + 5.0 * np.random.default_rng(0).random(4000)
    np.testing.assert_array_equal(targets, expected)
    empty, _ = lambda_law.draw_targets(0, _context(tmp_path, lambda_window=(-2.0, 3.0)))
    assert empty.shape == (0,)


def test_window_clipped_to_lambda_clip(tmp_path):
    """A (lo, hi) window is clipped to [-L, L]; so are the draws."""
    window, record = lambda_law.resolve_window({'lambda_window': (-6.0, 6.0)}, 4.0, None)
    assert window == (-4.0, 4.0)
    assert record == {'setting': [-6.0, 6.0], 'window': [-4.0, 4.0], 'source': 'setting',
                      'lambda_clip': 4.0}
    assert lambda_law.resolve_window({'lambda_window': (-1.0, 2.0)}, 4.0, None)[0] == (-1.0, 2.0)
    context = _context(tmp_path, lambda_clip=3.0, lambda_window=(-6.0, 6.0))
    targets, record = lambda_law.draw_targets(500, context)
    assert targets.min() >= -3.0 and targets.max() <= 3.0
    assert record['window'] == [-3.0, 3.0]
    with pytest.raises(ValueError, match='empty'):
        lambda_law.resolve_window({'lambda_window': (5.0, 6.0)}, 4.0, None)


@pytest.mark.parametrize('seed', range(6))
def test_window_bins_equals_trainer_cutoffs(seed):
    """lambda_window='bins' is the trainer's sp cutoffs (reweight_parameters_with_bins), clipped."""
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.uniform(-8.0, 8.0, size=rng.integers(3, 9)))
    lower = -inf if seed % 2 else edges[0] - 1.0
    upper = +inf if seed % 3 else edges[-1] + 1.0
    bins = np.concatenate([[lower], edges, [upper]])
    for reweight_parameters in ({}, {'free_threshold': 20}, {'sp_cutoff_min': -1.5},
                                {'sp_cutoff_min': -2.5, 'sp_cutoff_max': 1.0}):
        lo, hi = _trainer_cutoffs(reweight_parameters, bins)
        window, record = lambda_law.resolve_window(
            {'lambda_window': 'bins'}, 100.0, bins, reweight_parameters=reweight_parameters)
        assert window == (lo, hi)
        assert record['source'] == 'bins' and record['setting'] == 'bins'
        clipped, _ = lambda_law.resolve_window(
            {'lambda_window': 'bins'}, 2.0, bins, reweight_parameters=reweight_parameters)
        assert clipped == (max(lo, -2.0), min(hi, 2.0))
        assert reweight_parameters.get('sp_cutoff_max', 'unset') in ('unset', 1.0)   # not modified


def test_window_default_is_the_params_default():
    """Without a lambda_window setting the window is CGAIMMD_PARAMETER_DEFAULTS' (no own default)."""
    from aimmd.params.utils import CGAIMMD_PARAMETER_DEFAULTS
    default = CGAIMMD_PARAMETER_DEFAULTS['lambda_window']
    bins = np.array([-inf, -1.0, 0.0, 2.0, inf])
    assert (lambda_law.resolve_window({}, 4.0, bins)
            == lambda_law.resolve_window({'lambda_window': default}, 4.0, bins))


def test_window_bins_before_bins_exist():
    """Without bins the 'bins' window is [-L, L], recorded as 'bins (no bins yet)'."""
    for bins in (None, [], np.zeros(0)):
        window, record = lambda_law.resolve_window({'lambda_window': 'bins'}, 5.0, bins)
        assert window == (-5.0, 5.0)
        assert record == {'setting': 'bins', 'window': [-5.0, 5.0],
                          'source': 'bins (no bins yet)', 'lambda_clip': 5.0}


def test_finite_window_bins():
    """Edges clipped to the window; bins without a finite positive width inside it dropped."""
    bins = np.array([-inf, -3.0, -1.0, 0.0, 1.0, 4.0, inf])
    low, high, index = lambda_law.finite_window_bins(bins, (-2.0, 2.0))
    np.testing.assert_array_equal(index, [1, 2, 3, 4])
    np.testing.assert_array_equal(low, [-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(high, [-1.0, 0.0, 1.0, 2.0])
    low, high, index = lambda_law.finite_window_bins(bins, (-10.0, 10.0))
    np.testing.assert_array_equal(index, range(6))
    assert low[0] == -10.0 and high[-1] == 10.0
    low, high, index = lambda_law.finite_window_bins(np.array([0.0, 1.0]), (2.0, 3.0))
    assert len(index) == 0 and len(low) == 0 and len(high) == 0


# ---------------------------------------------------------------------------
# the 'bins' law
# ---------------------------------------------------------------------------

BINS = np.array([-inf, -2.0, -1.0, 0.0, 0.05, 1.0, 2.0, inf])
DENSITIES = np.array([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05])


def _values_function(descriptors):
    """A values function of the shooting-point x (not the cached values, x itself)."""
    return 10.0 * np.asarray(descriptors, dtype=float)[:, 0]


def test_bins_law_uses_aimmd_density_helpers(tmp_path, monkeypatch):
    """Lorentzian, density_adjustment and shared adjustment through aimmd.worker.utils's helpers."""
    run = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3)
    chains = [PathEnsemble(chain) for chain in run.chains]
    # an in-flight shot of another chain folder (RFPS counts frame 0 of back{ext}, through
    # its descriptors cache)
    (tmp_path / 'run' / 'chainR5').mkdir()
    back = write_trajectory(tmp_path / 'run' / 'chainR5', stem='back',
                            positions=np.array([[[-0.25, 0.0, 0.0]]], dtype=np.float32))
    save_npy(get_cache_fname(back, 'descriptors'), np.array([[-0.25, 0.0, 0.0]], np.float32))
    calls = []
    real_lorentzian = worker_utils.apply_lorentzian
    real_adjustment = worker_utils.apply_density_adjustment

    def spy_lorentzian(densities, bins, lorentzian):
        calls.append(('lorentzian', np.array(densities), lorentzian))
        return real_lorentzian(densities, bins, lorentzian)

    def spy_adjustment(densities, populations, shared, merged, adjustment):
        calls.append(('adjustment', np.array(populations), np.array(shared), np.array(merged),
                      adjustment))
        return real_adjustment(densities, populations, shared, merged, adjustment)

    monkeypatch.setattr(worker_utils, 'apply_lorentzian', spy_lorentzian)
    monkeypatch.setattr(worker_utils, 'apply_density_adjustment', spy_adjustment)
    context = _context(tmp_path, bins=BINS, densities=DENSITIES, lambda_law='bins',
                       lambda_window=(-3.0, 3.0),
                       params_overrides={'lorentzian': 1.5, 'density_adjustment': inf,
                                         'shared_density_adjustment': True, 'nbins': 7,
                                         'values_function': _values_function})
    context.folder = str(tmp_path / 'run' / 'chainR0')
    targets, record = lambda_law.draw_targets(20000, context, chain=chains[0],
                                              shooting_chains=chains)

    # independent computation of RFPS's adjustments
    own = np.asarray(chains[0].shooting('values'), dtype=float)          # cached values
    populations = np.histogram(own, BINS)[0]
    shared_x = np.array([path[path.shooting_index].positions[0, 0] for chain in run.chains
                         for path in chain] + [-0.25])
    shared = np.histogram(10.0 * shared_x, BINS)[0]
    expected = DENSITIES / DENSITIES.sum()
    expected = expected * (bin_centers(BINS) ** 2 + 1.5 ** 2)
    expected /= expected.sum()
    expected = expected * (populations + 1) * (shared + 1)
    expected /= expected.sum()

    kinds = [call[0] for call in calls]
    assert kinds == ['lorentzian', 'adjustment']
    np.testing.assert_array_equal(calls[1][1], populations)
    np.testing.assert_array_equal(calls[1][2], shared)
    np.testing.assert_array_equal(calls[1][3], np.ones(len(DENSITIES)))
    assert calls[1][4] == inf
    assert record['name'] == 'bins' and record['window'] == [-3.0, 3.0]
    assert record['window_source'] == 'setting'
    assert record['lorentzian'] == 1.5 and record['density_adjustment'] == 'inf'
    np.testing.assert_allclose(record['adjusted_densities'], expected, rtol=1e-12)
    np.testing.assert_array_equal(record['densities'], DENSITIES)
    np.testing.assert_array_equal(np.asarray(record['bins'], dtype=float), BINS)
    assert record['bins'][0] == '-inf' and record['bins'][-1] == 'inf'   # strict JSON
    json.dumps(record, allow_nan=False)

    # draws: bin k (clipped to the window) with weight 1 / density_k, uniform inside it
    low, high, index = lambda_law.finite_window_bins(BINS, (-3.0, 3.0))
    weights = 1.0 / expected[index]
    weights /= weights.sum()
    counts = np.array([np.sum((targets >= a) & (targets < b)) for a, b in zip(low, high)])
    np.testing.assert_allclose(counts / len(targets), weights, atol=0.015)
    assert targets.min() >= -3.0 and targets.max() <= 3.0
    inside = targets[(targets >= 1.0) & (targets < 2.0)]               # uniform inside a bin
    assert abs(inside.mean() - 1.5) < 0.05

    # a finite density_adjustment histograms only the chain's last values
    calls.clear()
    context = _context(tmp_path, bins=BINS, densities=DENSITIES, lambda_law='bins',
                       lambda_window=(-3.0, 3.0),
                       params_overrides={'density_adjustment': 2, 'nbins': 7})
    lambda_law.draw_targets(10, context, chain=chains[0], shooting_chains=chains)
    assert [call[0] for call in calls] == ['adjustment']                 # lorentzian = inf
    np.testing.assert_array_equal(calls[0][1], np.histogram(own[-2:], BINS)[0])
    np.testing.assert_array_equal(calls[0][2], np.zeros(len(DENSITIES)))

    # nbins <= 1 turns the density adjustment (and the shared one) off, as in RFPS
    calls.clear()
    context = _context(tmp_path, bins=BINS, densities=DENSITIES, lambda_law='bins',
                       params_overrides={'nbins': 1, 'shared_density_adjustment': True})
    _, record = lambda_law.draw_targets(10, context, chain=chains[0], shooting_chains=chains)
    assert calls == [] and record['density_adjustment'] == 0


def test_shared_density_counts_what_rfps_selection_counts(tmp_path, monkeypatch):
    """The shared density step is RFPS selection's own helper pair: a registered shot without
    descriptor rows is featurized and counted (the old hand copy skipped it), and the backward
    half another worker is running is only read, never written (without the descriptor rows
    its owner writes, it is not counted, as in RFPS selection)."""
    run = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=2)
    chains = [PathEnsemble(chain) for chain in run.chains]
    lost = run.chains[1][0]                                       # its descriptor rows are lost
    import os
    os.remove(get_cache_fname(lost.fname, 'descriptors'))
    (tmp_path / 'run' / 'chainR5').mkdir()
    back = write_trajectory(tmp_path / 'run' / 'chainR5', stem='back',
                            positions=np.array([[[-0.25, 0.0, 0.0]]], dtype=np.float32))
    calls = []
    real_collect = worker_utils.collect_shared_shooting_points
    real_populations = worker_utils.shared_shooting_populations

    def collect(*args, **kwargs):
        calls.append('collect')
        return real_collect(*args, **kwargs)

    def populations(*args, **kwargs):
        calls.append('populations')
        return real_populations(*args, **kwargs)

    monkeypatch.setattr(worker_utils, 'collect_shared_shooting_points', collect)
    monkeypatch.setattr(worker_utils, 'shared_shooting_populations', populations)
    seen = []
    real_adjustment = worker_utils.apply_density_adjustment

    def spy(densities, own, shared, merged, adjustment):
        seen.append(np.array(shared))
        return real_adjustment(densities, own, shared, merged, adjustment)

    monkeypatch.setattr(worker_utils, 'apply_density_adjustment', spy)
    context = _context(tmp_path, bins=BINS, densities=DENSITIES, lambda_law='bins',
                       lambda_window=(-3.0, 3.0),
                       params_overrides={'density_adjustment': inf, 'nbins': 7,
                                         'shared_density_adjustment': True,
                                         'values_function': _values_function})
    context.folder = str(tmp_path / 'run' / 'chainR0')
    lambda_law.draw_targets(10, context, chain=chains[0], shooting_chains=chains)
    assert calls == ['collect', 'populations']
    shared_x = np.array([path[path.shooting_index].positions[0, 0] for chain in run.chains
                         for path in chain])
    np.testing.assert_array_equal(seen[0], np.histogram(10.0 * shared_x, BINS)[0])
    assert seen[0].sum() == 4                                     # the 4 registered shots
    assert os.path.exists(get_cache_fname(lost.fname, 'descriptors'))      # featurized
    assert not os.path.exists(get_cache_fname(back, 'descriptors'))        # only read


def test_bins_law_flag_gives_plain_inverse_density(tmp_path, monkeypatch):
    """BINS_LAW_AIMMD_ADJUSTMENTS = False: plain 1/density."""
    monkeypatch.setattr(lambda_law, 'BINS_LAW_AIMMD_ADJUSTMENTS', False)
    run = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=3)
    context = _context(tmp_path, bins=BINS, densities=DENSITIES, lambda_law='bins',
                       params_overrides={'lorentzian': 1.5, 'density_adjustment': inf})
    _, record = lambda_law.draw_targets(10, context, chain=PathEnsemble(run.chains[0]))
    np.testing.assert_allclose(record['adjusted_densities'], DENSITIES / DENSITIES.sum())
    assert record['lorentzian'] == 'inf' and record['density_adjustment'] == 0
    json.dumps(record, allow_nan=False)


def test_bins_empty_density_floor(tmp_path):
    """An empty bin (<= 1e-12) gets the smallest non-empty density in the window, not 1/1e-15."""
    bins = np.array([-inf, -1.0, 0.0, 1.0, 2.0, inf])
    densities = np.array([0.1, 0.4, 1e-15, 0.2, 0.3])
    densities = densities / densities.sum()
    context = _context(tmp_path, bins=bins, densities=densities, lambda_law='bins',
                       lambda_window=(-2.0, 3.0), params_overrides={'density_adjustment': 0})
    targets, record = lambda_law.draw_targets(40000, context)
    floored = np.where(densities > 1e-12, densities, 0.1 / densities.sum())
    floored /= floored.sum()
    np.testing.assert_allclose(record['adjusted_densities'], floored, rtol=1e-12)
    weights = 1.0 / floored
    weights /= weights.sum()
    counts = np.histogram(targets, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])[0] / len(targets)
    np.testing.assert_allclose(counts, weights, atol=0.01)
    assert abs(counts[2] - counts[0]) < 0.02                  # as likely as the sparsest bin

    # the floor is the smallest non-empty density among the bins in the window, not overall
    densities = np.array([0.01, 0.4, 1e-15, 0.2, 0.3])        # 0.01 lies outside the window
    context = _context(tmp_path, bins=bins, densities=densities, lambda_law='bins',
                       lambda_window=(-0.5, 1.5), params_overrides={'density_adjustment': 0})
    _, record = lambda_law.draw_targets(10, context)
    floored = np.where(densities > 1e-12, densities, 0.2)      # min over bins 1, 2, 3
    np.testing.assert_allclose(record['adjusted_densities'], floored / floored.sum(), rtol=1e-12)


def test_bins_law_weights_the_bins_inside_the_window(tmp_path):
    """A window that drops bins: bin index[j] is drawn with weight 1 / density[index[j]], not j's."""
    bins = np.array([-inf, -3.0, -1.0, 1.0, 3.0, inf])
    densities = np.array([0.05, 0.10, 0.50, 0.25, 0.10])       # unequal, sums to 1
    # lambda_window='bins' is (-3, 3), the trainer's cutoffs: the two outer bins are dropped;
    # a (lo, hi) window above -1 drops the two lowest bins
    for window, edges in (('bins', [-3.0, -1.0, 1.0, 3.0]), ((-0.5, 5.0), [-0.5, 1.0, 3.0, 5.0])):
        context = _context(tmp_path, bins=bins, densities=densities, lambda_law='bins',
                           lambda_window=window, params_overrides={'density_adjustment': 0})
        targets, record = lambda_law.draw_targets(40000, context)
        low, high, index = lambda_law.finite_window_bins(bins, tuple(record['window']))
        assert record['window'] == [edges[0], edges[-1]] and len(index) == 3
        np.testing.assert_allclose(record['adjusted_densities'], densities, rtol=1e-12)
        weights = 1.0 / densities[index]
        weights /= weights.sum()
        counts = np.histogram(targets, edges)[0] / len(targets)
        np.testing.assert_allclose(counts, weights, atol=0.01)
        assert abs(weights - 1.0 / 3.0).max() > 0.1               # the bins are not interchangeable


def test_bins_uniform_inside_a_bin(tmp_path):
    """Inside the drawn bin lambda* is uniform (not the bin center)."""
    context = _context(tmp_path, bins=[-inf, 0.0, 1.0, inf], densities=[0.3, 0.4, 0.3],
                       lambda_law='bins', lambda_window=(0.0, 1.0),
                       params_overrides={'density_adjustment': 0})
    targets, _ = lambda_law.draw_targets(4000, context)       # the window holds one bin
    assert targets.min() >= 0.0 and targets.max() <= 1.0
    assert abs(targets.mean() - 0.5) < 0.02 and abs(targets.std() - 12 ** -0.5) < 0.02


def test_bins_density_adjustment_takes_the_last_values(tmp_path, monkeypatch):
    """A finite density_adjustment histograms the chain's last shooting values, not its first."""
    run = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=3)   # shooting values 0.01, 0.02, 0.03
    chain = PathEnsemble(run.chains[0])
    np.testing.assert_allclose(chain.shooting('values'), [0.01, 0.02, 0.03], atol=1e-6)
    seen = []
    real = worker_utils.apply_density_adjustment

    def spy(densities, populations, shared, merged, adjustment):
        seen.append(np.array(populations))
        return real(densities, populations, shared, merged, adjustment)

    monkeypatch.setattr(worker_utils, 'apply_density_adjustment', spy)
    context = _context(tmp_path, bins=[-inf, 0.0, 0.015, 0.025, inf], densities=[0.25] * 4,
                       lambda_law='bins', lambda_window=(-1.0, 1.0),
                       params_overrides={'density_adjustment': 2, 'nbins': 4})
    lambda_law.draw_targets(5, context, chain=chain)
    np.testing.assert_array_equal(seen[0], [0, 0, 1, 1])        # 0.02 and 0.03

    # only accepted paths count (chain[chain.accepted], as select_shooting_point): the last
    # path excluded, the last two accepted values are 0.01 and 0.02
    chain.accepted = [True, True, False]
    lambda_law.draw_targets(5, context, chain=chain)
    np.testing.assert_array_equal(seen[1], [0, 1, 1, 0])

    # a negative density_adjustment (Params only rounds it) is off, as max(value, 0) in RFPS
    context.params.__dict__['density_adjustment'] = -1
    _, record = lambda_law.draw_targets(5, context, chain=chain)
    assert len(seen) == 2 and record['density_adjustment'] == 0


def test_bins_fallback_before_bins_exist(tmp_path):
    """No bins yet: uniform on the window, record 'bins (uniform: no bins yet)', logged once."""
    context = _context(tmp_path, lambda_law='bins', lambda_window=(-1.0, 2.0))
    targets, record = lambda_law.draw_targets(1000, context)
    assert record == {'name': 'bins (uniform: no bins yet)', 'window': [-1.0, 2.0],
                      'window_source': 'setting'}
    np.testing.assert_array_equal(targets, -1.0 + 3.0 * np.random.default_rng(0).random(1000))
    assert len(context.log.messages) == 1 and 'no bins yet' in context.log.messages[0]
    lambda_law.draw_targets(10, context)
    assert len(context.log.messages) == 1
    # lambda_window='bins' as well: the window is [-L, L]
    context = _context(tmp_path, lambda_law='bins', lambda_window='bins', lambda_clip=4.0)
    targets, record = lambda_law.draw_targets(100, context)
    assert record['window'] == [-4.0, 4.0] and record['window_source'] == 'bins (no bins yet)'
    # a window that overlaps no bin falls back the same way
    context = _context(tmp_path, bins=[0.0, 1.0], densities=[1.0], lambda_law='bins',
                       lambda_window=(2.0, 3.0))
    targets, record = lambda_law.draw_targets(100, context)
    assert record['name'] == 'bins (uniform: no bins in the window)'
    assert targets.min() >= 2.0 and targets.max() <= 3.0


def test_callable_validated(tmp_path):
    """cgaimmd_lambda_sampler replaces the law; its output is checked; record names the function."""
    seen = []

    def my_sampler(n, context):
        seen.append(context.rng)
        return np.full(n, 0.5)

    context = _context(tmp_path, lambda_window=(-1.0, 1.0))
    context.params.__dict__['cgaimmd_lambda_sampler'] = my_sampler
    targets, record = lambda_law.draw_targets(5, context)
    np.testing.assert_array_equal(targets, np.full(5, 0.5))
    assert targets.dtype == np.float64
    assert seen == [context.rng]
    assert record == {'name': 'callable',
                      'function': f'{__name__}.test_callable_validated.<locals>.my_sampler',
                      'window': [-1.0, 1.0], 'window_source': 'setting'}
    for bad, match in ((lambda n, c: np.zeros(n + 1), 'shape'),
                       (lambda n, c: np.full(n, np.nan), 'finite'),
                       (lambda n, c: np.full(n, 1.5), 'window'),
                       (lambda n, c: np.zeros((n, 1)), 'shape')):
        context.params.__dict__['cgaimmd_lambda_sampler'] = bad
        with pytest.raises(ValueError, match=match):
            lambda_law.draw_targets(3, context)


def test_unknown_law_raises(tmp_path):
    """A law name outside LAWS is an error (canonical_cgaimmd_parameters also rejects it)."""
    assert lambda_law.LAWS == ('uniform', 'bins')
    context = _context(tmp_path)
    context.settings['lambda_law'] = 'greenwood'
    with pytest.raises(ValueError, match='lambda_law'):
        lambda_law.draw_targets(3, context)
