"""The trainer's reweighting repeated offline (``aimmd.analysis.reweighting``).

Two synthetic runs of the ring of ``tests/_analysis_ring_run.py`` (free
simulations from both wells and two-way shots, written as AIMMD's workers
write them, 12,000 free-simulation frames and 16 shots each), each with one
round of the real trainer in this process: one unbiased, one with a static
bias inside state A. The trainer's numbers are recorded through wrappers
that change nothing (the weights of its last reweighting; the result of its
bias-reweighted rate estimate), so the offline results can be compared
with what the trainer computed, not only with the four digits it printed.

The free-energy checks compare with the ring's exact answer. At this amount
of data the largest deviation over ten bins is 0.1-0.36 kT (six seeds); the
tolerance of 0.4 kT is set from that spread, and the slow lane repeats the
check with three times the data at 0.25 kT.
"""

import json
import os
import shutil

import numpy as np
import pytest

from aimmd.analysis.reweighting import (FreeEnergy, check_output_outside, frame_table,
                                        free_energy, free_energy_convergence,
                                        free_energy_difference, populations, reweight_run,
                                        trainer_rate_estimates)
from tests._analysis_ring_run import (build_ring_run, expected_free_energy, metropolis,
                                      ring_states, snapshot, train_once, write_frames)

FES_TOLERANCE = 0.4
"""kT; the largest deviation over ten bins from the exact free energy (see the module docstring)."""


def _load_params(folder):
    import aimmd
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        return aimmd.Params.load('params.py', save=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='module')
def ring(tmp_path_factory):
    """The unbiased ring run after one trainer round: folder, params, the trainer's numbers."""
    folder = build_ring_run(tmp_path_factory.mktemp('ring') / 'campaign', seed=0)
    trained = train_once(folder)
    return {'folder': folder, 'run': str(folder / 'run'), 'trained': trained,
            'params': _load_params(folder)}


@pytest.fixture(scope='module')
def biased_ring(tmp_path_factory):
    """The ring run with a static bias inside A, after one trainer round."""
    folder = build_ring_run(tmp_path_factory.mktemp('biased_ring') / 'campaign', seed=0,
                            biased=True)
    trained = train_once(folder)
    return {'folder': folder, 'run': str(folder / 'run'), 'trained': trained,
            'params': _load_params(folder)}


@pytest.fixture(scope='module')
def ring_reweighting(ring):
    return reweight_run(ring['params'], ring['run'], fractions=(0.5, 1.0))


@pytest.fixture(scope='module')
def biased_reweightings(biased_ring):
    """The biased run reweighted with the bias applied (the default) and without it."""
    return (reweight_run(biased_ring['params'], biased_ring['run']),
            reweight_run(biased_ring['params'], biased_ring['run'], bias=False))


def x_of(reader):
    """The ring coordinate of every frame of a reader."""
    return np.array([frame.positions[0, 0] for frame in reader])


# ════════════════════════════════════════════════════════════════════════════
# the trainer's numbers
# ════════════════════════════════════════════════════════════════════════════

def test_cached_values_reproduce_the_trainers_rates(ring, ring_reweighting):
    """values='cached' gives the rates the trainer computed after its last round (to 1e-6 relative,
    in fact to the last bit) and prints them as the trainer printed them."""
    result, trained = ring_reweighting, ring['trained']
    assert np.isfinite(trained['k12']) and np.isfinite(trained['k21'])
    assert result.k12 == pytest.approx(trained['k12'], rel=1e-6)
    assert result.k21 == pytest.approx(trained['k21'], rel=1e-6)
    assert result.k12 == trained['k12'] and result.k21 == trained['k21']
    assert f'k12 estimate: {result.k12:.3e} [1/dt]' in trained['out']
    assert f'k21 estimate: {result.k21:.3e} [1/dt]' in trained['out']
    assert result.values == 'cached' and result.network is None
    assert result.bias_applied is False and result.k12_bias_reweighted is None
    assert result.left_out == {'free_frames': {}, 'shots': []}
    assert result.n_shots == 16
    assert result.free_frames + result.shot_frames == int(np.sum(result.n_frames))
    assert result.rate('A', 'B') == result.k12 and result.rate('B', 'A') == result.k21
    with pytest.raises(ValueError, match='end states'):
        result.rate('A', 'R')


def test_bins_and_densities_equal_the_trainers_files(ring, ring_reweighting):
    """The adaptive bins and the excursion density on them equal ``bins{s}.npy`` and
    ``densities{s}.npy``, which the trainer wrote after the same reweighting."""
    bins = np.load(os.path.join(ring['run'], 'binsARB.npy'))
    densities = np.load(os.path.join(ring['run'], 'densitiesARB.npy'))
    np.testing.assert_array_equal(ring_reweighting.bins, bins)
    np.testing.assert_allclose(ring_reweighting.densities, densities, rtol=1e-12, atol=0)
    assert ring_reweighting.reweight_parameters['sp_cutoff_min'] == bins[0]
    assert ring_reweighting.reweight_parameters['sp_cutoff_max'] == bins[-1]


def test_the_trainers_log_parser_reads_the_estimate_with_its_data(ring, ring_reweighting,
                                                                  tmp_path):
    """trainer_rate_estimates reads the printed estimate, the round, the shots and the frames."""
    log = tmp_path / 'trainARB.log'
    log.write_text(ring['trained']['out'])
    estimates = trainer_rate_estimates(log)
    assert len(estimates['k12']) == 1
    assert estimates['k12'][0] == float(f'{ring_reweighting.k12:.3e}')
    assert estimates['k21'][0] == float(f'{ring_reweighting.k21:.3e}')
    assert estimates['round'][0] == 1
    assert estimates['n_shots'][0] == 16
    assert estimates['n_frames'][0] == int(np.sum(ring_reweighting.n_frames))
    assert np.isfinite(estimates['time'][0])
    assert np.isnan(estimates['k12_bias_reweighted'][0])


def test_recomputed_values_equal_the_cached_ones(ring, ring_reweighting):
    """values='recompute' scores every reactive frame again in memory: with the run's own network
    it gives the cached rates (to float32 rounding), and it writes nothing."""
    before = snapshot(ring['run'])
    params = _load_params(ring['folder'])
    result = reweight_run(params, ring['run'], values='recompute')
    assert snapshot(ring['run']) == before
    assert result.values == 'recompute'
    assert result.network == os.path.join(ring['run'], 'networkARB.h5')
    assert float(params.network.scale.detach()) == 4.0                 # the run's network was loaded
    assert result.k12 == pytest.approx(ring_reweighting.k12, rel=1e-5)
    assert result.k21 == pytest.approx(ring_reweighting.k21, rel=1e-5)
    table = frame_table(result)
    cached = frame_table(ring_reweighting)
    np.testing.assert_allclose(table.value, cached.value, rtol=0, atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# subsets
# ════════════════════════════════════════════════════════════════════════════

def test_the_whole_data_as_a_subset_reproduces_all_data(ring_reweighting):
    """fraction 1.0 is all the data; fraction 0.5 holds half the shots and of every free folder."""
    half, whole = ring_reweighting.subsets
    assert whole.fraction == 1.0 and whole.fraction_by == 'shots'
    assert whole.k12 == ring_reweighting.k12 and whole.k21 == ring_reweighting.k21
    np.testing.assert_array_equal(whole.weights_forward, ring_reweighting.weights_forward)
    assert half.n_shots == 8
    assert half.free_frames == pytest.approx(ring_reweighting.free_frames / 2, abs=4)
    assert np.isfinite(half.k12) and np.isfinite(half.k21)
    rows = ring_reweighting.convergence()
    assert [row['fraction'] for row in rows] == [0.5, 1.0, 1.0]
    assert [row['n_shots'] for row in rows] == [8, 16, 16]
    assert rows[0]['registered_until'] <= rows[-1]['registered_until']
    summary = ring_reweighting.summary()
    json.dumps(summary)
    assert [s['fraction'] for s in summary['subsets']] == [0.5, 1.0]


def test_subsets_by_registration_time(ring):
    """fraction_by='time' keeps the shots registered in the first part of the registration span."""
    files = sorted(os.path.join(ring['run'], 'chainR0', name)
                   for name in os.listdir(os.path.join(ring['run'], 'chainR0'))
                   if name.endswith('.xtc'))
    times = sorted(os.path.getmtime(f) for f in files)
    result = reweight_run(ring['params'], ring['run'], fractions=(0.5,), fraction_by='time')
    limit = times[0] + 0.5 * (times[-1] - times[0])
    (subset,) = result.subsets
    assert subset.fraction_by == 'time'
    assert subset.n_shots == sum(t <= limit for t in times)


def test_arguments_are_checked(ring):
    with pytest.raises(ValueError, match='values'):
        reweight_run(ring['params'], ring['run'], values='new')
    with pytest.raises(ValueError, match='bias'):
        reweight_run(ring['params'], ring['run'], bias='yes')
    with pytest.raises(ValueError, match='records no bias'):
        reweight_run(ring['params'], ring['run'], bias=True)
    with pytest.raises(ValueError, match='fraction_by'):
        reweight_run(ring['params'], ring['run'], fraction_by='frames')
    with pytest.raises(ValueError, match='fractions'):
        reweight_run(ring['params'], ring['run'], fractions=(0.0,))
    with pytest.raises(FileNotFoundError):
        reweight_run(ring['params'], ring['run'] + '_missing')


# ════════════════════════════════════════════════════════════════════════════
# frames, populations, free energies
# ════════════════════════════════════════════════════════════════════════════

def test_the_frame_table_covers_the_counted_frames_once(ring, ring_reweighting):
    """Every counted frame carries its path's weight; the frames of a free trajectory appear
    once; states and values are those of the caches."""
    table = frame_table(ring_reweighting, cv='descriptors')
    weights = ring_reweighting.path_weights
    counted = np.asarray(ring_reweighting.n_frames)
    nonzero = weights != 0
    assert len(table) == int(counted[nonzero].sum())
    assert np.sum(table.path_weight) == pytest.approx(np.sum(weights * counted), rel=1e-12)
    free = ~table.is_shot
    keys = set(zip(table.file[free], table.frame[free]))
    assert len(keys) == int(free.sum())                       # no free frame twice
    x = table.column('descriptors')
    np.testing.assert_array_equal(table.state, ring_states(x))
    reactive = table.state == 'R'
    np.testing.assert_allclose(table.value[reactive],
                               4.0 * (2 * np.clip(np.where(x[reactive] < 6, (x[reactive] - 2) / 3,
                                                           (10 - x[reactive]) / 3), 0, 1) - 1),
                               atol=1e-5)
    assert np.all(table.value[~reactive] == 0)
    assert set(table.last_visited) <= {'A', 'B', ''}
    assert np.all(table.last_visited[table.state == 'A'] == 'A')
    assert table.bias_applied is False and np.all(table.bias == 0)
    np.testing.assert_array_equal(table.weight, table.path_weight)
    # a callable on the reader gives the same coordinate
    from_reader = frame_table(ring_reweighting, cv={'x': x_of})
    np.testing.assert_allclose(from_reader.column('x'), x, atol=1e-6)


def test_populations_sum_to_one(ring_reweighting):
    result = ring_reweighting.populations()
    assert sum(result['state'].values()) == pytest.approx(1.0)
    assert sum(result['last_visited'].values()) == pytest.approx(1.0)
    assert set(result['state']) == {'A', 'R', 'B'}
    assert result['total_weight'] > 0
    # each well holds 0.31 of exp(-U) (A: x < 2, B: 5 <= x < 7); 12,000 frames allow +-0.08
    exact = np.exp(-expected_free_energy(np.arange(11.0)))
    exact /= exact.sum()
    assert exact[:2].sum() == pytest.approx(0.312, abs=1e-3)
    for state, share in (('A', exact[:2].sum()), ('B', exact[5:7].sum())):
        assert result['state'][state] == pytest.approx(share, abs=0.08)


def test_populations_of_given_frames():
    out = populations(['A', 'R', 'B', 'R'], ['A', 'A', 'B', 'B'], [1, 1, 2, 0], states='ARB')
    assert out['state'] == {'A': 0.25, 'R': 0.25, 'B': 0.5}
    assert out['last_visited'] == {'A': 0.5, 'B': 0.5, 'none': 0.0}
    assert populations(['A'], ['A'], [0.0])['state']['A'] == 0.0


def test_the_free_energy_of_the_ring_is_its_potential(ring_reweighting):
    """The reweighted free energy along x equals U(x) (the exact answer) within the noise."""
    table = frame_table(ring_reweighting, cv={'x': x_of})
    surface = free_energy(table, bins=10, range=[(0, 10)])
    assert surface.names == ('x',)
    exact = expected_free_energy(surface.edges[0])
    deviation = surface.free_energy - exact
    deviation -= np.sum(surface.probability * deviation)
    assert np.abs(deviation).max() < FES_TOLERANCE, deviation
    # without the weights (a plain histogram of the frames) the shots' oversampling of R shows
    plain = free_energy(table.cv[:, 0], bins=10, range=[(0, 10)])
    assert isinstance(plain, FreeEnergy) and np.isfinite(plain.free_energy).all()


def test_free_energy_convergence_uses_one_grid(ring_reweighting):
    out = free_energy_convergence(ring_reweighting, 'descriptors', cv_source='descriptors',
                                  bins=10, range=[(0, 10)])
    full = out['full']
    assert [entry['fraction'] for entry in out['subsets']] == [0.5, 1.0]
    for entry in out['subsets']:
        for mine, reference in zip(entry['free_energy'].edges, full.edges):
            np.testing.assert_array_equal(mine, reference)
    assert out['subsets'][1]['difference']['rms'] == pytest.approx(0.0, abs=1e-12)
    assert out['subsets'][1]['difference']['coverage'] == pytest.approx(1.0)
    assert out['subsets'][0]['difference']['rms'] > 0


# ════════════════════════════════════════════════════════════════════════════
# the static bias
# ════════════════════════════════════════════════════════════════════════════

def test_bias_reweighted_rates_equal_the_trainers(biased_ring, biased_reweightings):
    """With record_bias, the rates go through AIMMD's bias-reweighted estimator: the result is the
    trainer's, and its checks are in the report."""
    trained = biased_ring['trained']
    assert trained['bias'] is not None
    k12_rw, k21_rw, gamma1, gamma2 = trained['bias']
    result, plain = biased_reweightings
    assert result.bias_applied is True
    assert result.k12 == trained['k12'] and result.k21 == trained['k21']
    assert result.k12_bias_reweighted == pytest.approx(k12_rw, rel=1e-6)
    assert result.k21_bias_reweighted == pytest.approx(k21_rw, rel=1e-6)
    np.testing.assert_allclose(result.gamma_forward, gamma1, rtol=1e-12)
    np.testing.assert_allclose(result.gamma_backward, gamma2, rtol=1e-12)
    assert 'Bias check passed' in result.report
    assert f'k12 bias-reweighted: {result.k12_bias_reweighted:.3e}' in trained['out']
    # the bias only acts in A: it slows the departure from A, not the one from B
    assert result.k12_bias_reweighted < result.k12
    assert result.k21_bias_reweighted == pytest.approx(result.k21, rel=1e-12)
    assert result.rate('A', 'B') == result.k12_bias_reweighted
    assert result.rate('A', 'B', bias_reweighted=False) == result.k12
    assert plain.bias_applied is False and plain.k12_bias_reweighted is None
    assert plain.k12 == result.k12
    estimates = trainer_rate_estimates_from_text(trained['out'])
    assert estimates['k12_bias_reweighted'][0] == float(f'{k12_rw:.3e}')


def trainer_rate_estimates_from_text(text):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as handle:
        handle.write(text)
    try:
        return trainer_rate_estimates(handle.name)
    finally:
        os.remove(handle.name)


def test_bias_weights_give_the_unbiased_free_energy(biased_reweightings):
    """Frames weighted with exp(bias) give the free energy of the unbiased ring, U(x); without the
    factor they give that of the biased dynamics, U(x) + V(x). Both are exact answers."""
    corrected = frame_table(biased_reweightings[0], cv='descriptors')
    uncorrected = frame_table(biased_reweightings[1], cv='descriptors')
    in_a = corrected.state == 'A'
    assert corrected.bias_applied and np.all(corrected.bias[in_a] > 0)
    assert np.all(corrected.bias[corrected.state == 'R'] == 0)
    np.testing.assert_allclose(corrected.weight, corrected.path_weight * np.exp(corrected.bias))
    surfaces = {}
    for name, table, with_bias in (('corrected', corrected, False),
                                   ('uncorrected', uncorrected, True)):
        surface = free_energy(table, bins=10, range=[(0, 10)])
        for expectation in (False, True):
            exact = expected_free_energy(surface.edges[0], with_bias=expectation)
            deviation = surface.free_energy - exact
            deviation -= np.sum(surface.probability * deviation)
            surfaces[name, expectation] = np.abs(deviation).max()
    assert surfaces['corrected', False] < FES_TOLERANCE
    assert surfaces['uncorrected', True] < FES_TOLERANCE
    # each matches its own answer better than the other one
    assert surfaces['corrected', False] < surfaces['corrected', True]
    assert surfaces['uncorrected', True] < surfaces['uncorrected', False]
    populations_ = corrected.populations()['state']
    assert populations_['A'] == pytest.approx(populations_['B'], abs=0.1)


def test_exp_bias_weights_recover_unbiased_populations_of_a_markov_chain():
    """A Metropolis chain on the ring under U + V, its frames weighted with exp(V): the weighted
    histogram is exp(-U), the unweighted one exp(-U - V)."""
    rng = np.random.default_rng(3)
    x = metropolis(rng, 1.0, 200000, biased=True)
    from tests._analysis_ring_run import bias
    edges = np.linspace(0, 10, 11)
    for weights, with_bias in ((np.exp(bias(x)), False), (np.ones(len(x)), True)):
        surface = free_energy(x, weights, bins=[edges])
        exact = expected_free_energy(edges, with_bias=with_bias)
        deviation = surface.free_energy - exact
        deviation -= np.sum(surface.probability * deviation)
        assert np.abs(deviation).max() < 0.1, deviation


# ════════════════════════════════════════════════════════════════════════════
# reading only, and data the trainer has not seen yet
# ════════════════════════════════════════════════════════════════════════════

def test_nothing_is_written_into_the_run_folder(ring, ring_reweighting, tmp_path):
    from aimmd.analysis.free_md import free_md_rates, free_md_reference
    before = snapshot(ring['run'])
    result = reweight_run(ring['params'], ring['run'], fractions=(0.5,))
    table = frame_table(result, cv={'x': x_of})
    free_energy_convergence(result, 'descriptors', cv_source='descriptors', bins=5)
    free_md_rates(ring['params'], ring['run'])
    free_md_reference(ring['params'], ring['run'], descriptors=True)
    assert snapshot(ring['run']) == before
    with pytest.raises(ValueError, match='inside the run folder'):
        result.save(os.path.join(ring['run'], 'analysis', 'rates'))
    with pytest.raises(ValueError, match='inside the run folder'):
        table.save(os.path.join(ring['run'], 'table.npz'))
    json_file, npz_file = result.save(tmp_path / 'out' / 'rates')
    assert json.loads(open(json_file).read())['k12'] == result.k12
    assert np.load(npz_file)['weights_forward'].shape == result.weights_forward.shape
    saved = np.load(table.save(tmp_path / 'out' / 'table.npz'))
    np.testing.assert_array_equal(saved['weight'], table.weight)
    assert snapshot(ring['run']) == before
    check_output_outside(tmp_path / 'elsewhere', ring['run'])
    with pytest.raises(ValueError):
        check_output_outside(ring['run'], ring['run'])


def test_frames_without_values_are_left_out(ring, ring_reweighting, tmp_path):
    """Frames written after the trainer's last value pass read as logit 0; with until_values (the
    default) free trajectories are cut at their first reactive frame without a value and such
    shots are left out, which gives the trainer's numbers back. A part file with no reactive
    frame has no values file and is never cut."""
    folder = tmp_path / 'campaign'
    shutil.copytree(ring['folder'], folder, symlinks=True)
    run = folder / 'run'
    rng = np.random.default_rng(7)
    # free A grows by a part file in A only (no values file), then by one with reactive frames
    # (a restarted mdrun repeats the last frame and time of the previous part; AIMMD joins the
    # parts by their times)
    last_part = sorted((run / 'freeA').glob('traj000001.part*.xtc'))[-1]
    import aimmd
    last = aimmd.Path(str(last_part))
    x_last, t_last = float(last.positions[-1, 0, 0]), float(last.times[-1])
    in_a = np.clip(np.full(30, 1.0) + 0.01 * rng.standard_normal(30), 0.5, 1.5)
    in_a[0] = x_last
    n = int(last_part.stem.split('part')[1]) + 1
    write_frames(run / 'freeA' / f'traj000001.part{n:04d}.xtc', in_a, t_last + np.arange(30.0))
    reactive = np.concatenate([[in_a[-1]], np.linspace(2.2, 4.0, 20)])
    write_frames(run / 'freeA' / f'traj000001.part{n + 1:04d}.xtc', reactive,
                 t_last + 29 + np.arange(21.0))
    # a new shot without values
    xs = np.concatenate([np.linspace(1.5, 3.0, 8), np.linspace(3.0, 5.5, 8)[1:]])
    write_frames(run / 'chainR0' / 'path000017.xtc', xs, np.abs(np.arange(15.0) - 7))
    params = _load_params(folder)
    result = reweight_run(params, str(run))
    (cut,) = result.left_out['free_frames'].values()
    assert cut == 20                    # the reactive part (its first frame repeats the A part's last)
    assert [os.path.basename(f) for f in result.left_out['shots']] == ['path000017.xtc']
    assert result.n_shots == 16
    # the 30 frames in A were kept: one counted frame more per kept frame than before
    assert result.free_frames == ring_reweighting.free_frames + 29
    without = reweight_run(params, str(run), until_values=False)
    assert without.left_out == {'free_frames': {}, 'shots': []}
    assert without.n_shots == 17
    assert (without.k12, without.k21) != (result.k12, result.k21)


# ════════════════════════════════════════════════════════════════════════════
# free energies from arrays, the log parser
# ════════════════════════════════════════════════════════════════════════════

def test_free_energy_of_weighted_points_is_exact():
    """Points on a grid weighted with exp(-U) give F = U - min U exactly."""
    x = np.repeat(np.arange(5) + 0.5, 3)
    energy = np.array([0.0, 1.0, 2.5, 0.5, 3.0])
    weights = np.exp(-energy[np.repeat(np.arange(5), 3)])
    surface = free_energy(x, weights, bins=5, range=[(0, 5)])
    np.testing.assert_allclose(surface.free_energy, energy - energy.min(), atol=1e-12)
    np.testing.assert_array_equal(surface.n_frames, 3)
    np.testing.assert_allclose(surface.centers[0], np.arange(5) + 0.5)
    # two dimensions, empty bins, and too few frames
    xy = np.array([[0.5, 0.5], [0.5, 0.5], [1.5, 0.5]])
    surface = free_energy(xy, [1.0, 1.0, 1.0], bins=2, range=[(0, 2), (0, 2)], min_frames=2)
    assert surface.free_energy[0, 0] == 0.0
    assert np.isinf(surface.free_energy[1, 0]) and np.isinf(surface.free_energy[0, 1])
    with pytest.raises(ValueError, match='negative'):
        free_energy([0.5], [-1.0], bins=1)
    with pytest.raises(ValueError, match='weights for'):
        free_energy([0.5, 1.0], [1.0], bins=1)


def test_free_energy_difference():
    a = free_energy([0.5, 1.5, 1.5], bins=2, range=[(0, 2)])
    b = free_energy([0.5, 0.5, 1.5], bins=2, range=[(0, 2)])
    assert free_energy_difference(a, a) == {'rms': 0.0, 'shift': 0.0, 'coverage': 1.0}
    out = free_energy_difference(a, b)
    assert out['coverage'] == 1.0 and out['rms'] > 0
    c = free_energy([0.5], bins=2, range=[(0, 2)])
    assert free_energy_difference(c, b)['coverage'] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match='grids'):
        free_energy_difference(a, free_energy([0.5], bins=3, range=[(0, 2)]))


LOG = """
Loading current path ensemble (Wed Oct 28 13:20:44 2026)
run:   0%|          | 1017/500000 [11:19:55<4947:15:15, 35.69s/ paths]\r run:   0%|  | 1018/500000 [11:20:00<4947:15:15, 35.69s/ paths]
Training the network (round 334, (Wed Oct 28 13:20:56 2026))
Reweighting the full path ensemble (Wed Oct 28 13:49:42 2026)
    k12 estimate: 1.812e-05 [1/dt]
    k21 estimate: 7.393e-04 [1/dt]
    796505 frames (excluded margins)
Bias check passed: mean |bias| in 'R' = 0.000 kT
    k12 bias-reweighted: 1.000e-05 [1/dt]
    k21 bias-reweighted: 7.000e-04 [1/dt]
run: 1100 paths [00:00, ? paths/s]
Reweighting the full path ensemble (Wed Nov  4 01:02:03 2026)
    k12 estimate: nan [1/dt]
    k21 estimate: 2.000e-04 [1/dt]
    10 frames (excluded margins)
Reweighting the full path ensemble (Wed Nov  4 02:00:00 2026)
run:   5%|5         | 26251/500000 [44:40<13:06:40, 10.04 frames/s]\x1b[A    k12 estimate: 1.555e-05 [1/dt]
    k21 estimate: 6.560e-04 [1/dt]
    26251 frames (excluded margins)
"""
"""A trainer log in the format of the archived campaigns; the last block has a progress bar of the
value pass in front of its first estimate, as in the chignolin campaign's log of 28 Oct."""

MULTI_LOG = """
Training the network (round 2, (Wed Oct 28 13:20:56 2026))
... [system 's1'] loaded 12 shot path(s) in 0.1s + 2 free trajectory(ies) in 0.0s
... [system 's2'] loaded 7 shot path(s) in 0.1s + 2 free trajectory(ies) in 0.0s
[system 's1'] adaptation bins (Wed Oct 28 13:21:00 2026)
    [system 's1'] k12 estimate: 1.000e-03 [1/dt]
    [system 's1'] k21 estimate: 2.000e-03 [1/dt]
    [system 's1'] k12 bias-reweighted: 5.000e-04 [1/dt]
    [system 's1'] k21 bias-reweighted: 2.000e-03 [1/dt]
[system 's2'] adaptation bins (Wed Oct 28 13:22:00 2026)
    [system 's2'] k12 estimate: 3.000e-03 [1/dt]
    [system 's2'] k21 estimate: 4.000e-03 [1/dt]
"""


def test_the_log_parser_on_single_and_multi_system_logs(tmp_path):
    log = tmp_path / 'trainARB.log'
    log.write_text(LOG)
    out = trainer_rate_estimates(log)
    np.testing.assert_array_equal(out['round'], [334, 334, 334])
    np.testing.assert_array_equal(out['n_shots'], [1018, 1100, 1100])
    np.testing.assert_array_equal(out['n_frames'], [796505, 10, 26251])
    np.testing.assert_allclose(out['k12'], [1.812e-05, np.nan, 1.555e-05])
    np.testing.assert_allclose(out['k21'], [7.393e-04, 2.0e-04, 6.56e-4])
    np.testing.assert_allclose(out['k12_bias_reweighted'], [1.0e-05, np.nan, np.nan])
    np.testing.assert_allclose(out['k21_bias_reweighted'], [7.0e-04, np.nan, np.nan])
    import datetime
    gap = (datetime.datetime(2026, 11, 4, 1, 2, 3)
           - datetime.datetime(2026, 10, 28, 13, 49, 42)).total_seconds()
    assert out['time'][1] - out['time'][0] == pytest.approx(gap, abs=3600)   # a DST change
    assert list(out['system']) == ['', '', '']
    log.write_text(MULTI_LOG)
    out = trainer_rate_estimates(log)
    assert list(out['system']) == ['s1', 's2']
    np.testing.assert_array_equal(out['n_shots'], [12, 7])
    np.testing.assert_allclose(out['k12'], [1e-3, 3e-3])
    np.testing.assert_allclose(out['k12_bias_reweighted'], [5e-4, np.nan])
    assert out['time'][1] - out['time'][0] == 60.0
    np.testing.assert_array_equal(out['round'], [2, 2])


# ════════════════════════════════════════════════════════════════════════════
# more than one system, cross-checks, imports
# ════════════════════════════════════════════════════════════════════════════

def test_a_multi_system_run_is_reweighted_per_system(ring, ring_reweighting, tmp_path):
    """A multi-system run (here twice the ring, as systems s1 and s2): one reweighting per system
    folder, each the single-system result of that folder."""
    from tests._analysis_ring_run import RING_PARAMS
    folder = tmp_path / 'multi'
    folder.mkdir()
    shutil.copy(ring['folder'] / 'initial.xtc', folder / 'initial.xtc')
    shutil.copy(ring['folder'] / 'initial.xtc.states.npy', folder / 'initial.xtc.states.npy')
    source = RING_PARAMS.replace(
        "initial_paths = 'initial.xtc'",
        "multi_system = True\nmulti_system_share_network = True\nsystem_ids = ['s1', 's2']\n"
        "topology = ['initial.xtc', 'initial.xtc']\n"
        "initial_paths = [['initial.xtc'], ['initial.xtc']]")
    (folder / 'params.py').write_text(source)
    for system in ('s1', 's2'):
        shutil.copytree(ring['run'], folder / 'run' / system, symlinks=True)
    params = _load_params(folder)
    assert params.multi_system
    results = reweight_run(params, str(folder / 'run'))
    assert set(results) == {'s1', 's2'}
    for system, result in results.items():
        assert result.system_id == system
        assert result.run_dir == str(folder / 'run' / system)
        assert result.k12 == ring_reweighting.k12 and result.k21 == ring_reweighting.k21
    one = reweight_run(params, str(folder / 'run'), system_id='s2')
    assert one.k12 == ring_reweighting.k12
    from aimmd.analysis.free_md import free_md_rates
    with pytest.raises(ValueError, match='system_id'):
        free_md_rates(params, str(folder / 'run'))
    assert free_md_rates(params, str(folder / 'run'), system_id='s1')['n_trajectories'] == 2


def test_the_reweighted_rates_agree_with_the_free_simulations(ring, ring_reweighting):
    """The ring's free simulations cross back and forth: the reweighted rates lie inside the 95 %
    Poisson limits of the rates counted in them."""
    from aimmd.analysis.free_md import free_md_rates
    counted = free_md_rates(ring['params'], ring['run'])
    assert counted['AB']['transitions'] > 5 and counted['BA']['transitions'] > 5
    assert counted['AB']['lower'] < ring_reweighting.k12 < counted['AB']['upper']
    assert counted['BA']['lower'] < ring_reweighting.k21 < counted['BA']['upper']


def test_frame_table_rows_can_be_selected(ring_reweighting):
    table = frame_table(ring_reweighting, cv='descriptors')
    shots = table.select(table.is_shot)
    assert len(shots) == int(table.is_shot.sum()) and np.all(shots.is_shot)
    assert shots.cv.shape == (len(shots), 1) and shots.cv_names == ('descriptors',)
    assert shots.files is table.files
    assert set(np.unique(shots.path)) <= set(np.flatnonzero(ring_reweighting.is_shot))
    with pytest.raises(ValueError, match='no collective variables'):
        frame_table(ring_reweighting).column('x')


def test_the_modules_import_under_the_docs_stubs():
    """The analysis modules import where the docs are built (torch, MDAnalysis and the other
    heavy packages replaced by the stubs of docs/source/conf.py)."""
    import subprocess
    import sys
    import textwrap
    import aimmd
    root = os.path.dirname(os.path.dirname(os.path.abspath(aimmd.__file__)))
    code = textwrap.dedent(f'''
        import importlib, runpy, sys
        BLOCKED = ('torch', 'MDAnalysis', 'mdtraj', 'matplotlib', 'tqdm', 'filelock', 'psutil',
                   'networkx', 'torch_geometric', 'torch_cluster', 'lz4')

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in BLOCKED:
                    raise ModuleNotFoundError(name)
                return None

        sys.meta_path.insert(0, Block())
        runpy.run_path({os.path.join(root, 'docs', 'source', 'conf.py')!r})
        for name in ('aimmd.analysis.reweighting', 'aimmd.analysis.free_md',
                     'aimmd.analysis.run_card'):
            importlib.import_module(name)
        print('imported')
    ''')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([root] + ([os.environ['PYTHONPATH']]
                                                    if os.environ.get('PYTHONPATH') else [])))
    proc = subprocess.run([sys.executable, '-c', code], cwd=root, env=env, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == 'imported'
