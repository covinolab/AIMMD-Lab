"""Generator adherence: the dial numbers and the samples next to MD (``aimmd.cgaimmd.analysis.adherence``).

Synthetic dials with a known answer: an affine dial with Gaussian noise, a flat
dial, a step at +-2 with a flat core, two flat wings with a gap; feature sets
drawn from known distributions for variety, near-copies, departure and the
structural match; the data-ceiling draw. numpy and scipy only.
"""

import math

import numpy as np
import pytest

from aimmd.cgaimmd.analysis import adherence

LEVELS = np.arange(-5.0, 6.0)


def _requests(n_per_level=64, levels=LEVELS):
    return np.repeat(np.asarray(levels, dtype=float), n_per_level)


# ---------------------------------------------------------------------------
# the dial
# ---------------------------------------------------------------------------

def test_affine_dial_recovers_noise_over_slope():
    """lambda = a + b lambda* + N(0, s): resolution = s / b within the bootstrap interval."""
    rng = np.random.default_rng(0)
    requested = _requests(96)
    a, b, s = 0.4, 0.7, 1.1
    judged = a + b * requested + rng.normal(0.0, s, len(requested))
    m = adherence.adherence_metrics(requested, judged, n_boot=300, seed=1)
    fit = m['all_requests']
    assert fit['slope'] == pytest.approx(b, abs=0.03)
    assert fit['offset_at_zero'] == pytest.approx(a, abs=0.1)
    assert fit['residual_robust_sd'] == pytest.approx(s, rel=0.08)
    lo, hi = m['intervals']['resolution_all']
    assert lo < s / b < hi
    assert fit['resolution'] == pytest.approx(s / b, rel=0.1)
    # the band and the wings see the same line
    assert m['band']['resolution'] == pytest.approx(s / b, rel=0.2)
    assert m['wings']['resolution'] == pytest.approx(s / b, rel=0.2)
    for region in ('core', 'band', 'low_wing', 'high_wing'):
        slope = m['median_slopes'][region]
        assert slope['interval'][0] < b < slope['interval'][1], region
    assert m['median_slopes']['core']['requests'] == [-1.0, 0.0, 1.0]
    assert m['n'] == len(requested) and m['n_missing'] == 0
    # neighbouring requests one logit apart with noise s / b in request units
    assert 0.5 < fit['neighbour_auc_mean'] < 1.0
    assert fit['spearman'] > 0.8


def test_flat_dial_has_infinite_resolution():
    rng = np.random.default_rng(2)
    requested = _requests(32)
    judged = rng.normal(0.0, 1.0, len(requested))           # ignores the request
    judged -= np.polyfit(requested, judged, 1)[0] * requested  # exactly flat on average
    m = adherence.adherence_metrics(requested, judged, n_boot=50)
    assert m['all_requests']['resolution'] == math.inf
    assert m['intervals']['resolution_all'][1] == math.inf
    assert abs(m['all_requests']['neighbour_auc_mean'] - 0.5) < 0.1


def test_step_at_the_band_edges_with_a_flat_core():
    """Medians jump at +-2 but are flat inside: band slope > 0, core slope about 0 (the lesson of 4 Nov)."""
    rng = np.random.default_rng(3)
    requested = _requests(64)
    judged = np.where(requested >= 2, 4.0, np.where(requested <= -2, -4.0, 0.0))
    judged = judged + rng.normal(0.0, 0.3, len(requested))
    slopes = adherence.median_slopes(requested, judged, n_boot=200)
    assert slopes['band']['slope'] > 1.0
    assert slopes['band']['interval'][0] > 0.5
    assert abs(slopes['core']['slope']) < 0.1
    assert slopes['core']['interval'][0] < 0.0 < slopes['core']['interval'][1]


def test_gap_between_the_wings_is_not_rewarded():
    """Flat wings at -5 and +5: one line through both wings would be steep; the wing fit is flat."""
    rng = np.random.default_rng(4)
    requested = _requests(48)
    judged = np.where(requested < 0, -5.0, 5.0) + rng.normal(0.0, 0.5, len(requested))
    outside = np.abs(requested) > 2
    naive = adherence.affine_fit(requested[outside], judged[outside])
    assert naive['slope'] > 1.0                               # what the gap would pretend
    assert naive['resolution'] < 2.0
    wings = adherence.wing_resolution(requested, judged)
    assert abs(wings['slope']) < 0.15                         # 0 within noise (SE about 0.04)
    assert wings['resolution'] == math.inf or wings['resolution'] > 3.0
    low, high = wings['intercepts']                             # each wing's line at lambda* = 0
    assert low - 4.0 * wings['slope'] == pytest.approx(-5.0, abs=0.1)
    assert high + 4.0 * wings['slope'] == pytest.approx(5.0, abs=0.1)
    # a real slope in the wings is still found
    sloped = 0.8 * requested + rng.normal(0.0, 0.5, len(requested))
    assert adherence.wing_resolution(requested, sloped)['slope'] == pytest.approx(0.8, abs=0.05)
    assert adherence.wing_resolution(np.zeros(3), np.zeros(3)) is None


def test_one_request_per_wing_leaves_the_wing_slope_undetermined():
    """With one request in each wing the two intercepts fit both wings: the gap between them is
    not a slope. A wing with two requests measures it again, even when the other wing is empty."""
    rng = np.random.default_rng(5)
    requested = np.repeat([-3.0, 3.0], 50)
    judged = np.where(requested < 0, -5.0, 5.0) + rng.normal(0.0, 0.1, len(requested))
    wings = adherence.wing_resolution(requested, judged)
    assert math.isnan(wings['slope']) and math.isnan(wings['resolution'])
    assert wings['intercepts'] == [pytest.approx(-5.0, abs=0.05), pytest.approx(5.0, abs=0.05)]
    m = adherence.adherence_metrics(requested, judged, n_boot=20)
    assert math.isnan(m['wings']['resolution'])
    assert all(math.isnan(v) for v in m['intervals']['resolution_wings'])
    one_wing = np.repeat([-4.0, -3.0], 50)
    sloped = adherence.wing_resolution(one_wing, 0.5 * one_wing + rng.normal(0, 0.1, 100))
    assert sloped['slope'] == pytest.approx(0.5, abs=0.1)
    assert math.isnan(sloped['intercepts'][1])


def test_neighbour_auc_and_missing_values():
    assert adherence.neighbour_auc([0, 1], [2, 3]) == 1.0
    assert adherence.neighbour_auc([0, 1], [0, 1]) == 0.5
    assert adherence.neighbour_auc([1.0], [1.0]) == 0.5
    assert math.isnan(adherence.neighbour_auc([], [1.0]))
    requested = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])
    judged = np.array([-1.2, np.nan, 0.8, 1.1, np.inf])
    m = adherence.adherence_metrics(requested, judged, n_boot=0)
    assert m['n'] == 3 and m['n_missing'] == 2
    assert m['intervals']['resolution_all'] == [math.nan, math.nan] or all(
        math.isnan(v) for v in m['intervals']['resolution_all'])
    per = {p['request']: p for p in m['per_request']}
    assert per[1.0]['n'] == 2 and per[1.0]['median'] == pytest.approx(0.95)
    with pytest.raises(ValueError, match='requests'):
        adherence.adherence_metrics([0.0, 1.0], [0.0])
    # called directly, the dial functions refuse judged values that are not finite (instead of a
    # NaN slope with an infinite resolution); the neighbour AUC leaves them out
    for function in (adherence.affine_fit, adherence.wing_resolution, adherence.median_slopes):
        with pytest.raises(ValueError, match='not finite'):
            function([-3.0, 0.0, 3.0], [-3.0, np.nan, 3.0])
    assert adherence.neighbour_auc([0.0, np.nan], [2.0, np.inf, 3.0]) == 1.0


# ---------------------------------------------------------------------------
# the samples next to MD
# ---------------------------------------------------------------------------

def test_structural_match_is_zero_for_equal_distributions():
    rng = np.random.default_rng(5)
    labels = rng.uniform(-3, 3, 4000)
    reference_cv = rng.normal(0.0, 1.0, 4000)
    requested = np.repeat([-2.0, 0.0, 2.0], 200)
    out = adherence.structural_match(requested, np.tile(np.sort(reference_cv[:200]), 3),
                                     labels, reference_cv)
    assert out['median_all'] < 0.15                           # sampling noise only
    same = adherence.structural_match(np.zeros(50), reference_cv[:50], np.zeros(50),
                                      reference_cv[:50])
    assert same['per_request'][0.0] == 0.0
    shifted = adherence.structural_match(np.zeros(50), reference_cv[:50] + 0.7, np.zeros(50),
                                         reference_cv[:50])
    assert shifted['per_request'][0.0] == pytest.approx(0.7)
    few = adherence.structural_match(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))
    assert math.isnan(few['per_request'][0.0]) and few['n_reference'][0.0] == 5


def test_pair_distance_features_and_distances():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(5, 4, 3))
    features = adherence.pair_distance_features([0, 2, 3])
    f = features(x)
    assert f.shape == (5, 3)
    assert f[1, 0] == pytest.approx(np.linalg.norm(x[1, 0] - x[1, 2]), rel=1e-5)
    # alignment-free: a rotated, shifted copy has the same features
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.allclose(features(x @ q.T + 3.0), f, atol=1e-4)
    d = adherence.pairwise_feature_distances(f)
    assert d.shape == (5, 5) and np.allclose(np.diag(d), 0)
    assert d[0, 1] == pytest.approx(adherence.feature_distances(f[0], f[1]), rel=1e-6)
    near, index = adherence.nearest_distances(f[[3, 1]] + 1e-6, f, chunk=2)
    assert index.tolist() == [3, 1] and np.all(near < 1e-4)
    with pytest.raises(ValueError, match='two atoms'):
        adherence.pair_distance_features([1])


def _cloud(rng, n, width, center=0.0, dim=6):
    return center + width * rng.normal(size=(n, dim))


def test_variety_is_one_for_samples_drawn_from_the_reference():
    rng = np.random.default_rng(7)
    levels = [-1.0, 0.0, 1.0]
    labels = rng.choice(levels, 3000) + rng.uniform(-0.4, 0.4, 3000)
    features = _cloud(rng, 3000, 1.0)
    reference = adherence.label_binned_reference(labels, features, levels, n_max=200, seed=1)
    for level in levels:
        assert reference[level]['n_in_window'] > 500
        assert len(reference[level]['indices']) == 200
    requested = np.repeat(levels, 120)
    same = adherence.sample_quality(requested, _cloud(rng, len(requested), 1.0), reference,
                                    n_boot=100)
    for level in levels:
        entry = same['per_request'][level]
        assert entry['variety_ratio'] == pytest.approx(1.0, abs=0.12)
        assert entry['variety_ratio_interval'][1] > 0.9
    assert same['variety_interval_below'] == []
    collapsed = adherence.sample_quality(requested, _cloud(rng, len(requested), 0.3), reference,
                                         n_boot=100)
    assert collapsed['variety_ratio_lowest'] < 0.4
    assert sorted(collapsed['variety_interval_below']) == levels
    assert collapsed['settings']['variety_threshold'] == 0.8


def test_near_copies_and_departure():
    rng = np.random.default_rng(8)
    levels = [0.0, 1.0]
    training = _cloud(rng, 400, 1.0)
    # an MD trajectory: consecutive frames 0.05 apart -> the near-copy line
    walk = np.cumsum(0.05 * rng.normal(size=(201, 6)), axis=0)
    line = adherence.consecutive_frame_threshold(walk[:-1], walk[1:])
    assert line == pytest.approx(0.05, rel=0.3)               # RMS of 0.05-wide steps
    labels = rng.choice(levels, 2000) + rng.uniform(-0.4, 0.4, 2000)
    md = _cloud(rng, 2000, 1.0)
    reference = adherence.label_binned_reference(labels, md, levels, training_features=training,
                                                 near_copy_threshold=line, seed=2)
    requested = np.repeat(levels, 100)
    copies = training[rng.choice(len(training), len(requested))] + 1e-4
    out = adherence.sample_quality(requested, copies, reference, training_features=training,
                                   near_copy_threshold=line, baseline_near_copy_share=0.02,
                                   n_boot=100)
    assert out['near_copy_share'] == pytest.approx(1.0)
    assert out['near_copy_share_interval'][0] > 0.95
    assert out['near_copy_line'] == pytest.approx(0.07)
    assert out['near_copy_excess'] == pytest.approx(0.98)
    assert out['departure_ratio'] < 0.05                        # on top of training frames
    fresh = adherence.sample_quality(requested, _cloud(rng, len(requested), 1.0), reference,
                                     training_features=training, near_copy_threshold=line,
                                     n_boot=50)
    assert fresh['near_copy_share'] < 0.05
    assert fresh['departure_ratio'] == pytest.approx(1.0, abs=0.15)
    far = adherence.sample_quality(requested, _cloud(rng, len(requested), 1.0, center=4.0),
                                   reference, training_features=training, n_boot=0)
    assert far['departure_ratio'] > 1.25 and far['near_copy_share'] is None
    assert far['settings']['departure_threshold'] == 1.25


def test_data_ceiling_respects_the_window():
    rng = np.random.default_rng(9)
    labels = rng.uniform(-6, 6, 5000)
    labels[::7] = np.nan
    valid = np.ones(len(labels), dtype=bool)
    valid[:1000] = False
    out = adherence.data_ceiling(labels, LEVELS, n_per_level=40, valid=valid, seed=3)
    assert len(out['indices']) == len(out['requested']) == len(out['labels'])
    assert np.all(np.abs(out['labels'] - out['requested']) <= 0.25)
    assert np.all(out['indices'] >= 1000)                       # only allowed frames
    assert np.all(np.isfinite(out['labels']))
    for level in LEVELS:
        assert (out['requested'] == level).sum() == min(40, out['n_in_window'][level])
    again = adherence.data_ceiling(labels, LEVELS, n_per_level=40, valid=valid, seed=3)
    assert np.array_equal(again['indices'], out['indices'])
    # the ceiling of a perfect judge: training frames at their labels follow the dial
    m = adherence.adherence_metrics(out['requested'], out['labels'], n_boot=0)
    assert m['all_requests']['slope'] == pytest.approx(1.0, abs=0.02)
    assert m['all_requests']['resolution'] < 0.3
