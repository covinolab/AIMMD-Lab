"""The pure refactors of ``aimmd.worker.utils`` leave AIMMD's behaviour unchanged.

Three helpers are factored out of existing code so that code which must apply
exactly AIMMD's rules (a generator's own fit and selection law, for example)
can call them instead of copying them:

- ``reweight_parameters_with_bins`` -- the trainer's ``sp_cutoff`` injection
  (``_train.py``, single- and multi-system trainer);
- ``apply_lorentzian`` and ``apply_density_adjustment`` -- the density steps of
  ``select_shooting_point``.

Each is compared, bit for bit, with the code it replaced: the old inline
snippets are reproduced below, and ``select_shooting_point`` as it was right
before the refactor is kept verbatim (docstring aside) as
``_select_shooting_point_before_refactor``. Both selectors run on the same
synthetic pools with the same seeds; the shooting points, every
``np.random.choice`` call (its probabilities byte by byte), the log and the
densities TPS saves must be identical."""

import itertools
import os
import re
from glob import glob
from math import inf

import numpy as np
import pytest
import torch

import aimmd
from aimmd._config import NPY_CACHE, print
from aimmd.analysis.utils import bin_centers, merge_empty_bins
from aimmd.cache.npy import save_npy
from aimmd.core.utils import now, process_state
from aimmd.path import Path
from aimmd.pathensemble import PathEnsemble
from aimmd.worker import utils as worker_utils
from aimmd.worker.utils import (apply_density_adjustment, apply_lorentzian, ensure_descriptors,
                                is_initial_path, reweight_parameters_with_bins,
                                select_shooting_point)
from tests._helpers_unit import TinyNetwork, build_path, simple_descriptors_function


# ---------------------------------------------------------------------------
# the replaced inline code, verbatim
# ---------------------------------------------------------------------------

def _old_sp_cutoffs_single_system(reweight_parameters, bins):
    """_train.py, single-system trainer, under "find sp_cutoff_min and sp_cutoff_max"."""
    rw_p = reweight_parameters.copy()
    if len(bins):
        sp_cutoff_min = bins[+0]
        sp_cutoff_max = bins[-1]
        if sp_cutoff_min == -inf and bins[+1] < +inf:
            sp_cutoff_min = bins[+1]
        if sp_cutoff_max == +inf and bins[-2] > -inf:
            sp_cutoff_max = bins[-2]
        if 'sp_cutoff_min' not in rw_p:
            rw_p['sp_cutoff_min'] = sp_cutoff_min
        if 'sp_cutoff_max' not in rw_p:
            rw_p['sp_cutoff_max'] = sp_cutoff_max
    return rw_p


def _old_sp_cutoffs_multi_system(reweight_parameters, bins):
    """_train.py, multi-system trainer (per-system bins / densities / rates)."""
    rw_p = reweight_parameters.copy()
    if len(bins):
        sp_cutoff_min, sp_cutoff_max = bins[0], bins[-1]
        if sp_cutoff_min == -inf and bins[1] < +inf:
            sp_cutoff_min = bins[1]
        if sp_cutoff_max == +inf and bins[-2] > -inf:
            sp_cutoff_max = bins[-2]
        rw_p.setdefault('sp_cutoff_min', sp_cutoff_min)
        rw_p.setdefault('sp_cutoff_max', sp_cutoff_max)
    return rw_p


def _old_lorentzian(densities, bins, lorentzian):
    """select_shooting_point: the Lorentzian step (after densities /= densities.sum())."""
    if lorentzian < inf:
        centers = bin_centers(bins)
        densities *= centers ** 2 + lorentzian ** 2
        densities /= densities.sum()
    return densities


def _old_density_adjustment(densities, populations_for_adjustment,
                            shared_populations_for_adjustment, merged_bin_counts,
                            density_adjustment):
    """select_shooting_point: the population adjustment and the final normalization."""
    if density_adjustment:
        densities *= populations_for_adjustment + merged_bin_counts
        densities *= shared_populations_for_adjustment + merged_bin_counts
    densities /= densities.sum()
    return densities




# ---------------------------------------------------------------------------
# reweight_parameters_with_bins
# ---------------------------------------------------------------------------

def _bin_cases():
    rng = np.random.default_rng(7)
    cases = [np.array([]), np.array([-inf, inf]), np.array([-inf, 0.0, inf]),
             np.array([-1.0, 1.0]), np.array([-inf, -1.0, 2.0]), np.array([-3.0, 0.5, inf])]
    for n in (2, 3, 5, 11):
        inner = np.sort(rng.normal(scale=4.0, size=n))
        for low, high in itertools.product((True, False), repeat=2):
            edges = inner
            if low:
                edges = np.concatenate([[-inf], edges])
            if high:
                edges = np.concatenate([edges, [inf]])
            cases.append(edges)
    return cases


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        assert type(a[key]) is type(b[key]), key
        assert a[key] == b[key] or (a[key] != a[key] and b[key] != b[key]), key


@pytest.mark.parametrize('user', [{}, {'free_threshold': 20},
                                  {'sp_cutoff_min': -0.3},
                                  {'sp_cutoff_max': 4.0, 'free_threshold': 5},
                                  {'sp_cutoff_min': -2.0, 'sp_cutoff_max': 2.0}])
def test_reweight_parameters_with_bins_equals_trainer(user):
    for bins in _bin_cases():
        for as_list in (False, True):
            edges = list(bins) if as_list else bins
            before = dict(user)
            new = reweight_parameters_with_bins(user, edges)
            assert user == before                                # input untouched
            assert new is not user
            _same(new, _old_sp_cutoffs_single_system(user, edges))
            _same(new, _old_sp_cutoffs_multi_system(user, edges))
    # no bins (yet): an unchanged copy (the old code had no None case)
    new = reweight_parameters_with_bins(user, None)
    assert new == user and new is not user


def test_trainer_uses_reweight_parameters_with_bins(monkeypatch, tmp_path):
    """The single-system trainer passes exactly the helper's dict to reweight."""
    from tests.test_worker_runtime_unit import TinyTrainWorker

    initial = build_path(tmp_path, stem='train_initial',
                         positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                                            dtype=np.float32))
    reweight_kwargs = []

    class FakeEnsemble:
        weights = np.array([1.0])
        fnames = np.array([initial.fname])
        n_frames = np.array([len(initial)])

        def __len__(self):
            return 1

        def __add__(self, other):
            return self

        def compute(self, *args, **kwargs):
            return 1

        def reweight(self, *args, **kwargs):
            reweight_kwargs.append(kwargs)
            return (np.array([1.0]), None, None, None, np.array([-1.0, 1.0]),
                    np.array([0.2, 0.8]))

        def project(self, bins, source='values'):
            return np.array([2.0, 1.0, 1.0], dtype=float)

        def types(self, pattern=None):
            return np.array([True])

    ensemble = FakeEnsemble()
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        sorted_states='ARB', chain_type='rfps', nbins=3, cutoff_min=0.5, cutoff_max=5.0,
        fit=lambda params, pathensemble, **kwargs: (
            [1.0], [1.0], np.array([0.0]), np.array([1.0]), np.array([[1.0, 0.0]])),
        marginal_bins='all', network_batch_size=4, rescale_committor=False,
        reweight_parameters={'free_threshold': 7, 'sp_cutoff_max': 9.0},
        trajectory_extension='.xtc',
        compute_values_args=(lambda x: np.array([0.0]), 'values', 'positions'),
        network_save_interval=1,
        update_network=lambda directory, timeout=0, raise_if_failure=False: None,
        shot_chains=lambda directory, target_state=None, old=None: [ensemble],
        free_trajectories=lambda directory, old=None: [], network=TinyNetwork())
    bins = np.array([-np.inf, -1.25, 0.5, np.inf])
    helper_calls = []
    real_helper = worker_utils.reweight_parameters_with_bins

    def spy(reweight_parameters, bins_):
        helper_calls.append((dict(reweight_parameters), np.array(bins_)))
        return real_helper(reweight_parameters, bins_)

    monkeypatch.setattr('aimmd.worker._train.reweight_parameters_with_bins', spy)
    monkeypatch.setattr('aimmd.worker._train.assemble_pathensemble', lambda *a, **k: ensemble)
    monkeypatch.setattr('aimmd.worker._train.compute_bins', lambda *a, **k: bins)
    monkeypatch.setattr('aimmd.worker._train.save_npy', lambda fname, arr: None)
    monkeypatch.setattr('aimmd.worker._train.replace_in_cache', lambda *a, **k: None)
    monkeypatch.setattr('aimmd.worker._train.torch.save', lambda state, fname: None)
    monkeypatch.setattr('aimmd.worker._train.shutil.copyfile', lambda src, dst: None)
    TinyTrainWorker(params, aimmd.PathEnsemble(initial), tmp_path)._train(
        nrounds=1, keep_running=False)
    assert len(helper_calls) == 1
    assert helper_calls[0][0] == {'free_threshold': 7, 'sp_cutoff_max': 9.0}
    np.testing.assert_array_equal(helper_calls[0][1], bins)
    expected = _old_sp_cutoffs_single_system(params.reweight_parameters, bins)
    assert reweight_kwargs and all(
        {k: v for k, v in kwargs.items() if k != 'source'} == expected
        for kwargs in reweight_kwargs)
    assert params.reweight_parameters == {'free_threshold': 7, 'sp_cutoff_max': 9.0}


# ---------------------------------------------------------------------------
# apply_lorentzian / apply_density_adjustment
# ---------------------------------------------------------------------------

def _bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_density_helpers_equal_inline_code(dtype):
    rng = np.random.default_rng(11)
    for n_bins in (1, 2, 5, 9):
        inner = np.sort(rng.normal(scale=2.0, size=n_bins - 1))
        for bins in (np.concatenate([[-inf], inner, [inf]]),
                     np.sort(rng.normal(scale=3.0, size=n_bins + 1))):
            densities = rng.random(n_bins).astype(dtype) + dtype(1e-3)
            densities /= densities.sum()
            for lorentzian in (inf, 0.3, 1.0, 5.0):
                before = densities.copy()
                new = apply_lorentzian(densities, bins, lorentzian)
                assert _bitwise_equal(densities, before)          # input untouched
                assert new is not densities
                assert _bitwise_equal(new, _old_lorentzian(densities.copy(), bins, lorentzian))
            for adjustment in (0, 3, inf):
                populations = rng.integers(0, 6, n_bins)
                shared = rng.integers(0, 4, n_bins)
                merged = rng.integers(1, 3, n_bins).astype(float)
                before = densities.copy()
                new = apply_density_adjustment(densities, populations, shared, merged,
                                               adjustment)
                assert _bitwise_equal(densities, before)
                assert new is not densities
                if not adjustment:                                # identity (a copy)
                    assert _bitwise_equal(new, densities)
                    new /= new.sum()                              # the caller normalizes
                assert _bitwise_equal(new, _old_density_adjustment(
                    densities.copy(), populations, shared, merged, adjustment))


# ---------------------------------------------------------------------------
# select_shooting_point before and after the refactor
# ---------------------------------------------------------------------------

BINS = np.array([-inf, -1.5, -0.5, 0.0, 0.5, 1.5, inf])
DENSITIES = np.array([0.3, 0.1, 0.25, 0.2, 1e-13, 0.15])

# (x of the frames, values of the frames): excursions A R ... R B / B R ... R A
POOL = [([-1.0, -0.3, -0.1, 0.1, 0.3, 1.0], [-6.0, -1.0, -0.2, 0.3, 0.8, 6.0]),
        ([1.0, 0.2, 0.1, -0.2, -1.0], [6.0, -0.8, -0.3, 0.1, -6.0]),
        ([-1.0, 0.1, 0.2, 0.3, 0.4, 1.0], [-6.0, 0.2, 0.6, 1.0, 1.2, 6.0])]
CHAIN = [([-1.0, -0.2, 0.2, 1.0], [-6.0, -1.0, 0.2, 6.0], 1),
         ([1.0, 0.3, -0.3, -1.0], [6.0, 0.7, 0.3, -6.0], 2),
         ([-1.0, 0.1, 0.4, 1.0], [-6.0, -0.2, 0.9, 6.0], 1),
         ([-1.0, 0.0, 0.3, 1.0], [-6.0, 0.4, 1.4, 6.0], 2)]


def _positions(x):
    return np.array([[[v, 0.0, 0.0]] for v in x], dtype=np.float32)


def _setup(tmp_path, *, densities_dtype=np.float64):
    run = tmp_path / 'run'
    folder = run / 'chainR0'
    other = run / 'chainR1'
    folder.mkdir(parents=True)
    other.mkdir(parents=True)
    network = TinyNetwork()
    torch.save(network.state_dict(), run / 'networkARB.h5')
    save_npy(str(run / 'binsARB.npy'), BINS)
    save_npy(str(run / 'densitiesARB.npy'), DENSITIES.astype(densities_dtype))
    pool = [build_path(folder, stem=f'pool{i}', positions=_positions(x), values=np.array(v),
                       shooting_index=1)
            for i, (x, v) in enumerate(POOL)]
    chain = [build_path(folder, stem=f'path{i + 1:06d}', positions=_positions(x),
                        values=np.array(v), shooting_index=si)
             for i, (x, v, si) in enumerate(CHAIN)]
    other_chain = [build_path(other, stem='path000001', positions=_positions(CHAIN[1][0]),
                              values=np.array(CHAIN[1][1]), shooting_index=1)]
    # an in-flight segment of the other chain (shared density adjustment)
    build_path(other, stem='back', positions=_positions([0.1, 0.2]),
               values=np.array([0.1, 0.2]), shooting_index=0)
    return {'folder': str(folder), 'pool': [p.fname for p in pool],
            'chain': [(p.fname, p.shooting_index) for p in chain],
            'other': [(p.fname, p.shooting_index) for p in other_chain], 'network': network}


def _params(setup, **fields):
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states='ARB', selection_pool_size=3, free_overriding_states='',
        free_overriding_attempts=0, free_overriding_recovery_rate=0.0, nbins=6,
        trajectory_extension='.xtc', network=setup['network'],
        descriptors_function=simple_descriptors_function,
        values_function=lambda d: np.asarray(d, dtype=float)[:, 0],
        uniform_selection_on_initial_paths=False, lorentzian=inf, density_adjustment=inf,
        shared_density_adjustment=False, chain_type='rfps')
    params.__dict__.update(fields)
    return params


def _run(selector, setup, params, seed, monkeypatch, capsys):
    """One selection with fresh pool/chain objects; returns everything observable."""
    pool = PathEnsemble([Path(fname) for fname in setup['pool']])
    chain = PathEnsemble([Path(fname, shooting_index=si) for fname, si in setup['chain']])
    other = PathEnsemble([Path(fname, shooting_index=si) for fname, si in setup['other']])
    calls = []
    real_choice = np.random.choice

    def recording_choice(a, *args, **kwargs):
        result = real_choice(a, *args, **kwargs)
        p = kwargs.get('p')
        calls.append((a if np.isscalar(a) else ('array', len(a), np.asarray(a).tobytes()),
                      None if p is None else (np.asarray(p).dtype.str,
                                              np.asarray(p).tobytes()),
                      np.asarray(result).tobytes()))
        return result

    monkeypatch.setattr(np.random, 'choice', recording_choice)
    capsys.readouterr()
    np.random.seed(seed)
    try:
        point = selector(pool, params, setup['folder'], chain, free_trajectories=[],
                         shooting_chains=[chain, other], target_state='R')
    finally:
        monkeypatch.setattr(np.random, 'choice', real_choice)
    log = [line for line in capsys.readouterr().out.splitlines()
           if 'Shooting initialization completed' not in line]
    saved = None
    if params.chain_type == 'tps':
        fname = f"{setup['folder']}/densitiesARB.npy"
        with open(fname, 'rb') as handle:
            saved = handle.read()
        os.remove(fname)
    return {'positions': np.asarray(point.positions).tobytes(), 'frame': point.frame,
            'calls': calls, 'log': log, 'pool_left': [p.fname for p in pool],
            'saved_densities': saved}


SCENARIOS = [dict(lorentzian=lorentzian, density_adjustment=adjustment,
                  shared_density_adjustment=shared)
             for lorentzian in (inf, 1.2) for adjustment in (inf, 0, 2)
             for shared in (False, True)]
SCENARIOS += [dict(lorentzian=0.7, density_adjustment=inf, shared_density_adjustment=True,
                   chain_type='tps', densities_dtype=np.float32),
              dict(lorentzian=inf, density_adjustment=3, shared_density_adjustment=False,
                   chain_type='tps', densities_dtype=np.float32),
              dict(lorentzian=2.0, density_adjustment=0, shared_density_adjustment=True,
                   chain_type='tps', densities_dtype=np.float64)]


@pytest.mark.parametrize('scenario', SCENARIOS,
                         ids=[re.sub(r'[^a-z0-9=.,_-]', '', str(sorted(
                             (k, getattr(v, '__name__', v)) for k, v in s.items())))
                              for s in SCENARIOS])
def test_density_helpers_leave_selection_unchanged(scenario, tmp_path, monkeypatch, capsys):
    scenario = dict(scenario)
    setup = _setup(tmp_path, densities_dtype=scenario.pop('densities_dtype', np.float64))
    params = _params(setup, **scenario)
    for seed in (0, 1, 2, 3):
        old = _run(_select_shooting_point_before_refactor, setup, params, seed, monkeypatch, capsys)
        new = _run(select_shooting_point, setup, params, seed, monkeypatch, capsys)
        assert new == old
        assert any(p is not None for _, p, _ in old['calls'])     # a bin was drawn
        if params.chain_type == 'tps':
            assert old['saved_densities'] is not None


# ---------------------------------------------------------------------------
# the frozen reference: select_shooting_point before the refactor
# ---------------------------------------------------------------------------

def _select_shooting_point_before_refactor(pool, params, folder,
                                           chain=None,
                                           free_trajectories=[],
                                           shooting_chains=[],
                                           target_state=1):
    """select_shooting_point right before the refactor (frozen reference; nodetable base)."""

    # params
    lorentzian = params.lorentzian
    states = params.states
    t = process_state(target_state, states)
    states = params.sorted_states
    pool_size = params.selection_pool_size

    # easy situation: internal shooting
    if t != states[1]:
        pool_index = np.random.choice(len(pool))
        path = pool[pool_index]
        fname = path.fname
        if path.middle('states') == t:
            indices = path.internal('indices')
        else:
            indices = np.flatnonzero(path.states == t)
        if not len(indices):
            raise RuntimeError(f'no frames available for {fname} in {t}')
        index = np.random.choice(indices)
        k, loc = path._get_local_loc(index)
        print(f'=== selecting path {fname!r}')
        print(f'=== selecting frame {loc}')

        # remove from pool and return
        if len(pool) >= pool_size:
            print(f'xxx removed {fname} from pool')
            pool.pop(pool_index)

        return path[index]

    # next params
    free_overriding_states = params.free_overriding_states
    if free_overriding_states == 'all':
        free_overriding_states = '.'
    overriding_types = [f'{s}{t}' for s in free_overriding_states]
    overriding_attempts = params.free_overriding_attempts
    overriding_rate = params.free_overriding_recovery_rate
    compute_values_args = params.compute_values_args
    ext = params.trajectory_extension
    nbins = params.nbins
    density_adjustment = max(params.density_adjustment, 0)
    shared_density_adjustment = params.shared_density_adjustment
    if nbins <= 1:  # no density adjustment in this case
        density_adjustment = 0
        shared_density_adjustment = False
    overriding_bins = np.zeros(nbins, dtype=bool)
    overriding_bins[params.free_overriding_bins] = True
    
    # before the chain has produced its first path with non-zero weight (for
    # TPS: its first accepted transition), the pool can only hold seeds
    # (TPS only: RFPS path weights depend on the selection probabilities)
    uniform_on_seed = (getattr(params, 'uniform_selection_on_initial_paths',
                               False)
                       and str(params.chain_type).lower().startswith('tp')
                       and (chain is None or chain.path is None))

    # process chain
    if chain is not None:
        chain = chain[chain.accepted]
    else:
        chain = PathEnsemble()
    
    # shared density adjustment: obtain shared_shooting_points
    # (registered chain paths, then frame 0 of the backward halves in flight)
    shared_shooting_points = registered_shooting_points = Path()
    if shared_density_adjustment:
        shooting_chains = PathEnsemble(shooting_chains)
        registered_shooting_points = shooting_chains.shooting('self').join()
        shared_shooting_points = registered_shooting_points
        for fname in glob(f'{folder}/../chain{t}*/back{ext}'):
            shared_shooting_points += Path(fname, stop=1)

    # overriding configurations
    candidate_paths = PathEnsemble()
    for trajectory in free_trajectories:
        candidate_paths += trajectory.split().extract(*overriding_types)
    overriding = overriding_unique = candidate_paths.sample(
        overriding_attempts)
    if overriding_attempts:
        print(f'*** shortlisted {len(overriding)} overriding frames '
              f'from {len(candidate_paths)} candidates')
        overriding_unique = PathEnsemble(overriding).merge()

    # clear cache to avoid picking the wrong values
    NPY_CACHE.clear()

    # network parameters, bins, densities
    if nbins > 1:
        params.update_network(f'{folder}/..')
        bins, densities = params.load_bins_and_densities(f'{folder}/..')

        # compute simulated values only where there are none (yet)
        # (featurize first the frames whose descriptors are missing)
        ensure_descriptors(pool, params, missing_values_only=True)
        ensure_descriptors(overriding_unique, params,
                           missing_values_only=True)
        n1 = pool.compute(*compute_values_args,
                          raise_if_error=True)
        n2 = overriding_unique.compute(*compute_values_args,
                                       raise_if_error=True)
        
        # also recompute initial paths' values if they still feature in pool
        initial_paths_in_pool = PathEnsemble(
            [path for path in pool if is_initial_path(path)])
        ensure_descriptors(initial_paths_in_pool, params)
        n1 += initial_paths_in_pool.compute(
            *compute_values_args, overwrite=True)

        print(f'*** updated {n1 + n2} frame values ({n1} from pool)')
        # need to compute overriding values already here to be sure that
        # both pool and overriding values are evaluated on the same NN model
        # this is because I am not re-evaluating pre-existing values

    else:
        bins = np.array([-inf, +inf])
        densities = np.array([1.])

    # immediately get all values & population histograms
    # (such that there is a lower risk of desync)
    pool_values = pool.values
    pool_shooting_values = pool.shooting('values')
    overriding_values = overriding.values
    chain_shooting_values = chain.shooting('values')
    populations = np.histogram(chain_shooting_values, bins)[0]
    # for the purpose of density adjustment
    if 0 < density_adjustment < inf:
        populations_for_adjustment = np.histogram(
            chain_shooting_values[-density_adjustment:], bins)[0]
    elif density_adjustment:
        populations_for_adjustment = populations
    else:
        populations_for_adjustment = np.zeros_like(populations)
    if shared_density_adjustment:
        try:  # catch instabilities in try/except loop
            # featurize registered paths only: the halves in flight belong
            # to other workers, which write their series while ingesting
            # them (and replace them with every shot), so they are only read
            ensure_descriptors(registered_shooting_points, params)
            shared_populations_for_adjustment = np.histogram(
                shared_shooting_points.compute(
                    compute_values_args[0], '',
                    *compute_values_args[2:]),
                bins)[0]
        except Exception as exception:
            shared_populations_for_adjustment = np.zeros_like(populations)
            print("!!! Warning: in computing shared_populations_for_adjustment:"
                  f" {exception}")
    else:
        shared_populations_for_adjustment = np.zeros_like(populations)

    # report selection pool
    report, histograms = pool.report(bins=bins, values=pool_values)
    print(f'\nSelection pool\n{report}')
    if nbins > 1:
        print(f'*** current pool shooting interfaces: {pool_shooting_values}')
        print(f'*** populations  {populations}')
        print(f'*** for adjust.  {populations_for_adjustment}')
        print(f'*** shared adj.  {shared_populations_for_adjustment}')
    
    # normalize histograms, average in "combined" histogram
    norms = np.maximum(histograms.sum(axis=1), 1.0)
    histograms /= norms[:, None]
    combined_histograms = histograms.mean(axis=0)
    
    # density adjustment (lorentzian)
    densities /= densities.sum()
    print(f'*** densities    {densities}')
    if lorentzian < inf:
        centers = bin_centers(bins)
        densities *= centers ** 2 + lorentzian ** 2
        densities /= densities.sum()
        print(f'    (after applying the Loretzian) {densities}')
    
    # merge empty bins, update histograms and densities
    keepers = combined_histograms > 0
    processed_overriding_bins = overriding_bins
    merged_bin_counts = np.ones(len(densities))
    if not keepers.all():
        (bins, merged_bin_counts,
         processed_overriding_bins, *merged_histograms
        ) = merge_empty_bins(
            bins, keepers, overriding_bins,
            *histograms, combined_histograms,
            densities, populations,
            populations_for_adjustment, shared_populations_for_adjustment
        )
        processed_overriding_bins = processed_overriding_bins.astype(bool)
        histograms = merged_histograms[:len(histograms)]
        combined_histograms, densities, populations = merged_histograms[-5:-2]
        populations_for_adjustment = merged_histograms[-2]
        shared_populations_for_adjustment = merged_histograms[-1]
        if len(bins) - 1 < nbins:
            print(f'*** merged {nbins - len(bins) + 1} internal empty bins:')
            print(f'    bins         {bins}')
            print(f'    merged count {merged_bin_counts}')
            print(f'    populations  {populations}')
            print(f'    for adjust.  {populations_for_adjustment}')
            print(f'    shared adj.  {shared_populations_for_adjustment}')
            print(f'    densities    {densities}')          
        
        # to preserve target distribution: divide densities by merged bin counts
        densities /= merged_bin_counts
    
    # density adjustment (populations)
    if density_adjustment:
        densities *= populations_for_adjustment + merged_bin_counts
        densities *= shared_populations_for_adjustment + merged_bin_counts
    densities /= densities.sum()
    print(f'    (adjusted)   {densities}')

    # report also overriding
    if overriding_bins.any() and overriding_attempts:
        print(f'*** overriding   {processed_overriding_bins}')
    
    # choose path
    pool_index = np.random.choice(len(pool))
    path = pool[pool_index]
    values = pool_values[pool_index]
    if not path.is_excursion(states):  # check for consistency
        raise RuntimeError(
            f'{path.fname!r} should be an excursion, found {path.type} instead')
    indices = path.internal('indices')
    locs = path.internal('locs')
    fname = path.fname
    print(f'=== selecting path {fname!r}')
    
    # seed path, no transition in the chain yet: the network has not seen a
    # single transition, so its values on the seed carry no committor
    # information and must not decide the shooting point. Draw uniformly among
    # the internal frames (plain TPS selection). With chain_type='tps' and
    # selection_pool_size=1 this is exact: the first transition is accepted
    # with probability 1 whatever the selection probability was.
    histogram = histograms[pool_index]
    mask = histogram > 0
    if uniform_on_seed and is_initial_path(path):
        i = np.random.choice(len(indices))
        k = None
        overriding = None
        print(f'=== uniform selection among the {len(indices)} internal '
              f'frames of the initial path (no transition in the chain yet; '
              f'uniform_selection_on_initial_paths=True)')

    # assign selection probabilities (weights)
    elif mask.any():
        bin_weights = np.zeros(len(histogram))
        bin_weights[mask] = combined_histograms[mask] / densities[mask]
        bin_weights /= bin_weights.sum()
        print(f'*** sel weights  {bin_weights}')
        
        # select bin
        k = np.random.choice(len(bin_weights), p=bin_weights)
        print(f'=== selecting bin {k}: {bins[k:k+2]}')

        # select shooting point among candidates in bin
        candidates = np.flatnonzero(np.digitize(values, bins) - 1 == k)
        i = np.random.choice(candidates)

    else:  # special situation: FFS-like
        i = np.argmin(np.abs(values))
        print(f'!!! outside bins range')
        k = None

    # get shooting point and report info
    shooting_point = path[indices[i]]
    loc = locs[i]
    if len(bins) - 1 > 1:
        value = values[i]
        print(f'=== selecting frame {loc} (value: {value:.3f})')
    else:
        print(f'=== selecting frame {loc}')
    
    if overriding_types and k is not None:
        if not processed_overriding_bins[k]:
            print('*** skipped overriding because the SP bin is '
                  'not in overriding_bins')
            overriding = None
        elif np.digitize(pool_shooting_values[pool_index], bins) - 1 == k:
            if np.random.random() > overriding_rate:
                print(f'*** skipped overriding because the old SP is in the '
                      f'same bin (rec. rate = {overriding_rate})')
                overriding = None
            else:
                # on a very rare occasion: still override
                print(f'*** rescued overriding with '
                      f'rec. rate = {overriding_rate}')
    
    if overriding:
        candidates = np.flatnonzero(np.digitize(
            overriding_values, bins) - 1 == k)
        if len(candidates):
            i = np.random.choice(candidates)
            path = overriding
            loc = overriding.locs[i]
            name = overriding.filenames[i]
            shooting_point = path[i]  # ALL
            if nbins > 1:
                value = overriding_values[i]
                print(f'=== overriding with {name}, {loc} '
                      f'(value: {value:.3f})')
            else:
                print(f'=== overriding with {name}, {loc}')
        else:
            print(f'*** no overriding candidates in bin {k} '
                  f'({len(overriding)} frames in total)')

    # remove from pool
    if len(pool) >= pool_size:
        print(f'xxx removed {fname} from pool')
        pool.pop(pool_index)

    # save params for TPS
    if params.chain_type == 'tps':
        torch.save(params.network.state_dict(),
                   f'{folder}/network{states}.h5')
        save_npy(f'{folder}/bins{states}.npy', bins)
        save_npy(f'{folder}/densities{states}.npy', densities)
    
    print(f'Shooting initialization completed {now()}\n')
    return shooting_point
