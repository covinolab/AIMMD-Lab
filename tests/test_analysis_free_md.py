"""Rates and a committor reference from the free simulations alone (``aimmd.analysis.free_md``).

The run-folder tests use a Markov chain on the sites 0, ..., 8 of a line (a
random walk with a site-dependent drift; site 0 is state A, site 8 state B,
the others the reactive region R), written as the free simulations of an
AIMMD run: the particle's ``x`` coordinate is the site. Its committor is known
exactly (the solution of a linear system), and so are its transition counts
(they are counted on the generated sequence).
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from aimmd.analysis.free_md import (forward_outcomes, free_md_calibration, free_md_rates,
                                    free_md_reference, poisson_rate_interval,
                                    transition_counts)

N_SITES = 9


def jump_probabilities():
    """Per site: probability to step up and to step down (the rest: stay)."""
    up = np.array([0.3] + [0.25 + 0.03 * i for i in range(1, N_SITES - 1)] + [0.0])
    down = np.array([0.0] + [0.3] * (N_SITES - 2) + [0.3])
    return up, down


def exact_committor():
    """Probability to reach site N - 1 before site 0, from every site."""
    up, down = jump_probabilities()
    n = N_SITES
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0
    rhs[-1] = 1.0
    for i in range(1, n - 1):
        matrix[i, i] = up[i] + down[i]
        matrix[i, i + 1] = -up[i]
        matrix[i, i - 1] = -down[i]
    return np.linalg.solve(matrix, rhs)


def markov_chain(rng, start, n_steps):
    up, down = jump_probabilities()
    sites = np.empty(n_steps, dtype=int)
    site = start
    draws = rng.random(n_steps)
    for t in range(n_steps):
        sites[t] = site
        if draws[t] < up[site]:
            site += 1
        elif draws[t] < up[site] + down[site]:
            site -= 1
    return sites


def labels_of(sites):
    out = np.full(len(sites), 'R', dtype='<U1')
    out[np.asarray(sites) == 0] = 'A'
    out[np.asarray(sites) == N_SITES - 1] = 'B'
    return out


def cluster_estimate(outcome, groups):
    """Mean outcome and its standard error with the frames of one group counted as one draw."""
    mean = outcome.mean()
    totals = {}
    for o, g in zip(outcome, groups):
        n, s = totals.get(g, (0, 0.0))
        totals[g] = (n + 1, s + (o - mean))
    variance = sum(s ** 2 for _, s in totals.values()) / len(outcome) ** 2
    return mean, np.sqrt(variance)


# ════════════════════════════════════════════════════════════════════════════
# pure functions
# ════════════════════════════════════════════════════════════════════════════

def test_poisson_limits_are_the_exact_garwood_limits():
    rate, lower, upper = poisson_rate_interval(0, 10.0)
    assert (rate, lower) == (0.0, 0.0)
    assert upper == pytest.approx(3.688879 / 10.0, rel=1e-6)          # -ln(0.025) / T
    rate, lower, upper = poisson_rate_interval(1, 1.0)
    assert (lower, upper) == pytest.approx((0.025318, 5.571643), rel=1e-5)
    rate, lower, upper = poisson_rate_interval(10, 2.0)
    assert rate == 5.0
    assert (lower, upper) == pytest.approx((4.795389 / 2, 18.390356 / 2), rel=1e-6)
    rate, lower, upper = poisson_rate_interval(10, 2.0, confidence=0.68)
    assert lower == pytest.approx(chi2.ppf(0.16, 20) / 4)
    assert upper == pytest.approx(chi2.ppf(0.84, 22) / 4)
    assert all(np.isnan(poisson_rate_interval(3, 0.0)))
    with pytest.raises(ValueError):
        poisson_rate_interval(-1, 1.0)
    with pytest.raises(ValueError):
        poisson_rate_interval(1, 1.0, confidence=1.0)


def test_poisson_limits_cover_the_true_rate():
    """Exact intervals are conservative: they cover the true rate at least 95 % of the time."""
    rng = np.random.default_rng(0)
    for mean in (0.5, 3.0, 20.0):
        counts = rng.poisson(mean, size=4000)
        limits = np.array([poisson_rate_interval(n, 1.0)[1:] for n in counts])
        covered = (limits[:, 0] <= mean) & (mean <= limits[:, 1])
        assert covered.mean() >= 0.95


def test_transition_counts_by_hand():
    labels = list('RAARRBRBBARRA')
    out = transition_counts(labels, states='ARB')
    assert out['transitions'] == {('A', 'B'): 1, ('B', 'A'): 1}
    assert out['frames'] == {'A': 8, 'B': 4}            # the first R frame is on no side
    assert out['time'] == {'A': 8.0, 'B': 4.0}
    doubled = transition_counts(labels, frame_time=np.full(len(labels), 2.0))
    assert doubled['time'] == {'A': 16.0, 'B': 8.0}
    other = transition_counts(list('XYZY'), states='XYZ')     # any letters
    assert other['transitions'] == {('X', 'Z'): 1, ('Z', 'X'): 0}
    assert other['frames'] == {'X': 2, 'Z': 2}
    with pytest.raises(ValueError):
        transition_counts(labels, frame_time=[1.0])


def test_forward_outcomes_by_hand():
    outcome, excursion = forward_outcomes(list('RRARRBRRR'))
    np.testing.assert_array_equal(outcome, [0, 0, 0, 1, 1, 1, -1, -1, -1])
    np.testing.assert_array_equal(excursion, [0, 0, -1, 1, 1, -1, 2, 2, 2])
    outcome, excursion = forward_outcomes([])
    assert len(outcome) == 0 and len(excursion) == 0


def test_forward_outcomes_recover_the_committor_of_a_markov_chain():
    """The share of a site's frames whose next end state is B is the committor of that site."""
    rng = np.random.default_rng(1)
    sites = markov_chain(rng, 0, 400000)
    outcome, excursion = forward_outcomes(labels_of(sites))
    exact = exact_committor()
    for site in range(1, N_SITES - 1):
        mask = (sites == site) & (outcome >= 0)
        mean, error = cluster_estimate(outcome[mask], excursion[mask])
        assert abs(mean - exact[site]) < 4 * error + 1e-3, (site, mean, exact[site], error)
        assert error < 0.03


# ════════════════════════════════════════════════════════════════════════════
# a run folder
# ════════════════════════════════════════════════════════════════════════════

PARAMS = '''
import numpy as np

engine = 'toy'
initial_paths = 'initial.xtc'
toy_slowdown = 0.0
free_restart_source = '{restart}'


def toy_mdrun(ts):
    pass


def states_function(trajectory):
    x = np.array([frame.positions[0, 0] for frame in trajectory])
    out = np.full(len(x), 'R', dtype='<U1')
    out[x < 0.5] = 'A'
    out[x > {last} - 0.5] = 'B'
    return out
'''

BIAS = '''
record_bias = True
bias_source = 'reader'


def bias_function(trajectory):
    x = np.array([frame.positions[0, 0] for frame in trajectory])
    return np.where(x < 0.5, 0.5, 0.0)
'''


def write_trajectory_with_caches(fname, sites, times, *, values=None, bias=None):
    import MDAnalysis as mda
    from aimmd.cache.npy import save_npy
    from aimmd.path.utils import get_cache_fname
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe.empty(1, trajectory=True)
    with mda.Writer(str(fname), n_atoms=1) as writer:
        for site, time in zip(sites, times):
            universe.trajectory.ts.positions = np.array([[site, 0.0, 0.0]], dtype=np.float32)
            universe.trajectory.ts.time = float(time)
            universe.trajectory.ts.dimensions = np.array([20, 20, 20, 90, 90, 90],
                                                         dtype=np.float32)
            writer.write(universe.atoms)
    save_npy(get_cache_fname(str(fname), 'states'), labels_of(sites))
    save_npy(get_cache_fname(str(fname), 'descriptors'),
             np.asarray(sites, dtype=np.float32)[:, None])
    if values is not None:
        save_npy(get_cache_fname(str(fname), 'values'), np.asarray(values, dtype=np.float32))
    if bias is not None:
        save_npy(get_cache_fname(str(fname), 'bias'), np.asarray(bias, dtype=float))
    return str(fname)


def exact_logit(sites):
    q = np.clip(exact_committor()[np.asarray(sites)], 1e-6, 1 - 1e-6)
    return np.where(labels_of(sites) == 'R', np.log(q / (1 - q)), 0.0)


@pytest.fixture(scope='module')
def markov_run(tmp_path_factory):
    """Free simulations of the Markov chain: free A (two part files; values for the first only)
    and free B (one file), with states, descriptors and bias caches."""
    folder = tmp_path_factory.mktemp('markov') / 'campaign'
    folder.mkdir()
    rng = np.random.default_rng(2)
    for restart, name in (('equilibrium', 'params.py'), ('crossing', 'params_crossing.py')):
        (folder / name).write_text(PARAMS.format(restart=restart, last=N_SITES - 1))
    (folder / 'params_biased.py').write_text(
        PARAMS.format(restart='equilibrium', last=N_SITES - 1) + BIAS)
    write_trajectory_with_caches(folder / 'initial.xtc', np.arange(N_SITES),
                                 np.arange(N_SITES, dtype=float))
    run = folder / 'run'
    sequences = {}
    a = markov_chain(rng, 0, 30001)
    half = 15000
    write_trajectory_with_caches(run / 'freeA' / 'traj000001.part0001.xtc', a[:half + 1],
                                 np.arange(half + 1, dtype=float), values=exact_logit(a[:half + 1]),
                                 bias=np.where(a[:half + 1] == 0, 0.5, 0.0))
    write_trajectory_with_caches(run / 'freeA' / 'traj000001.part0002.xtc', a[half:],
                                 np.arange(half, len(a), dtype=float),
                                 bias=np.where(a[half:] == 0, 0.5, 0.0))
    sequences['A'] = a
    b = markov_chain(rng, N_SITES - 1, 20000)
    write_trajectory_with_caches(run / 'freeB' / 'traj000001.part0001.xtc', b,
                                 np.arange(len(b), dtype=float),
                                 bias=np.where(b == 0, 0.5, 0.0))
    sequences['B'] = b
    return {'folder': folder, 'run': str(run), 'sequences': sequences}


def load(folder, name='params.py'):
    import aimmd
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        return aimmd.Params.load(name, save=False)
    finally:
        os.chdir(cwd)


def test_free_md_rates_count_the_transitions(markov_run):
    params = load(markov_run['folder'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')                   # 'equilibrium' restarts: no warning
        out = free_md_rates(params, markov_run['run'])
    expected = {'A': 0.0, 'B': 0.0}
    transitions = {('A', 'B'): 0, ('B', 'A'): 0}
    for sites in markov_run['sequences'].values():
        counts = transition_counts(labels_of(sites))
        for s in 'AB':
            expected[s] += counts['time'][s]
        for key in transitions:
            transitions[key] += counts['transitions'][key]
    assert out['n_trajectories'] == 2 and out['bias_applied'] is False
    assert out['frames_without_state'] == 0
    for s, o in (('A', 'B'), ('B', 'A')):
        entry = out[s + o]
        assert entry['transitions'] == transitions[(s, o)] > 10
        assert entry['time'] == expected[s] == entry['frames']
        assert (entry['rate'], entry['lower'], entry['upper']) == pytest.approx(
            poisson_rate_interval(transitions[(s, o)], expected[s]))
        assert entry['lower'] < entry['rate'] < entry['upper']


def test_free_md_rates_use_boosted_time_under_a_bias(markov_run):
    """With record_bias, every frame counts exp(bias): here exp(0.5) for the frames in A."""
    params = load(markov_run['folder'], 'params_biased.py')
    out = free_md_rates(params, markov_run['run'])
    plain = free_md_rates(params, markov_run['run'], bias=False)
    assert out['bias_applied'] is True and plain['bias_applied'] is False
    boosted = 0.0
    for sites in markov_run['sequences'].values():
        labels = labels_of(sites)
        counts = transition_counts(labels, frame_time=np.exp(np.where(sites == 0, 0.5, 0.0)))
        boosted += counts['time']['A']
    assert out['AB']['time'] == pytest.approx(boosted, rel=1e-12)
    assert out['AB']['time'] > plain['AB']['time']
    assert out['AB']['transitions'] == plain['AB']['transitions']
    assert out['AB']['rate'] < plain['AB']['rate']
    assert out['BA'] == pytest.approx(plain['BA'])        # the B side has no frame in A
    with pytest.raises(ValueError, match='records no bias'):
        free_md_rates(load(markov_run['folder']), markov_run['run'], bias=True)


def test_crossing_restarts_are_flagged(markov_run):
    params = load(markov_run['folder'], 'params_crossing.py')
    with pytest.warns(UserWarning, match='crossing'):
        out = free_md_rates(params, markov_run['run'])
    assert out['free_restart_source'] == 'crossing'
    with pytest.warns(UserWarning, match='crossing'):
        free_md_reference(params, markov_run['run'])


def test_free_md_reference_recovers_the_committor_of_a_markov_chain(markov_run):
    """The run's free simulations, read from the run folder, give per site the exact committor."""
    params = load(markov_run['folder'])
    reference = free_md_reference(params, markov_run['run'], descriptors=True)
    assert reference.kind == 'free_md_forward' and reference.states == 'ARB'
    sites = np.rint(reference.descriptors[:, 0]).astype(int)
    assert set(sites) <= set(range(1, N_SITES - 1))      # reactive frames only
    np.testing.assert_array_equal(reference.n_first + reference.n_last, 1)
    exact = exact_committor()
    for site in range(1, N_SITES - 1):
        mask = sites == site
        mean, error = cluster_estimate(reference.n_last[mask].astype(float),
                                       reference.groups[mask])
        assert abs(mean - exact[site]) < 4 * error + 1e-3, (site, mean, exact[site], error)
    # the frames point back into the files; the groups are excursions of one trajectory each
    for f in range(len(reference.files)):
        mask = reference.file == f
        assert reference.frame[mask].max() < 15001 + 20000
    for group in np.unique(reference.groups)[:50]:
        mask = reference.groups == group
        assert len(set(reference.trajectory[mask])) == 1
        assert len(set(reference.n_last[mask])) == 1     # one outcome per excursion
    # the cached values: present for the first part file, missing (nan) for the second
    first_part = np.array([reference.files[f].endswith('part0001.xtc') and 'freeA' in
                           reference.files[f] for f in reference.file])
    assert np.all(np.isfinite(reference.values[first_part]))
    np.testing.assert_allclose(reference.values[first_part], exact_logit(sites[first_part]),
                               atol=1e-5)
    assert np.all(np.isnan(reference.values[~first_part]))
    censored = reference.notes['censored_frames']
    assert censored >= 0 and reference.notes['free_restart_source'] == 'equilibrium'
    strided = free_md_reference(params, markov_run['run'], stride=5)
    assert len(strided) < len(reference) / 4
    with pytest.raises(ValueError, match='stride'):
        free_md_reference(params, markov_run['run'], stride=0)


def test_free_md_calibration_compares_predicted_and_observed(markov_run):
    """With the exact committor as the model, predicted and observed outcomes agree per bin."""
    params = load(markov_run['folder'])
    reference = free_md_reference(params, markov_run['run'], descriptors=True)
    sites = np.rint(reference.descriptors[:, 0]).astype(int)
    rows = free_md_calibration(reference, exact_logit(sites))
    assert sum(row['frames'] for row in rows) == len(reference)
    for row in rows:
        if row['frames'] > 200:
            # judged on the scale of the excursions: frames of one excursion share the outcome
            spread = 4 * np.sqrt(row['excursions']) * row['frames'] / row['excursions']
            assert abs(row['predicted'] - row['observed']) < spread, row
    # the run's cached values (part of the frames) are the default
    default = free_md_calibration(reference)
    assert sum(row['frames'] for row in default) == int(np.isfinite(reference.values).sum())
    with pytest.raises(ValueError, match='logits for'):
        free_md_calibration(reference, [0.0])
