"""The offline reweighting on longer runs (slow lane).

- The ring of ``tests/_analysis_ring_run.py`` with 40,000 free-simulation
  frames and 30 shots: the reweighted free energy matches the exact one to
  0.25 kT (largest deviation over ten bins; 0.08-0.18 kT over three seeds),
  without and with a static bias in A.
- The toy 1-D C-GAIMMD campaign of ``tests/test_toy_1d_cgaimmd_campaign.py``
  (free simulations, shots from the 'ode' generator, a trainer round with the
  flow-matching fit), with and without a recorded bias: the offline rates are
  the trainer's, and the run card holds what the fit wrote.
"""

import dataclasses
import os

import numpy as np
import pytest
import torch

import aimmd
from aimmd.analysis.free_md import free_md_rates, free_md_reference
from aimmd.analysis.reweighting import frame_table, free_energy, reweight_run
from aimmd.analysis.run_card import run_card
from tests._analysis_ring_run import build_ring_run, expected_free_energy, snapshot, train_once
from tests.test_toy_1d_cgaimmd import restore_signals, short_sys_path  # noqa: F401

pytestmark = pytest.mark.slow

TOLERANCE = 0.25
"""kT; the largest deviation over ten bins from the exact free energy at 40,000 frames."""


def _load(folder):
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        return aimmd.Params.load('params.py', save=False)
    finally:
        os.chdir(cwd)


def _deviation(table, *, with_bias):
    surface = free_energy(table, bins=10, range=[(0, 10)])
    exact = expected_free_energy(surface.edges[0], with_bias=with_bias)
    deviation = surface.free_energy - exact
    deviation -= np.sum(surface.probability * deviation)
    return np.abs(deviation).max()


@pytest.mark.parametrize('biased', [False, True])
def test_the_free_energy_of_a_longer_ring_run(tmp_path, biased):
    folder = build_ring_run(tmp_path / 'campaign', seed=11, n_free=20000, part=5000,
                            n_shots=30, biased=biased)
    trained = train_once(folder)
    params = _load(folder)
    result = reweight_run(params, str(folder / 'run'), fractions=(0.25, 0.5))
    assert result.k12 == trained['k12'] and result.k21 == trained['k21']
    table = frame_table(result, cv='descriptors')
    assert _deviation(table, with_bias=False) < TOLERANCE
    if biased:
        k12_rw, k21_rw, _, _ = trained['bias']
        assert result.k12_bias_reweighted == pytest.approx(k12_rw, rel=1e-12)
        assert result.k21_bias_reweighted == pytest.approx(k21_rw, rel=1e-12)
        plain = frame_table(reweight_run(params, str(folder / 'run'), bias=False),
                            cv='descriptors')
        assert _deviation(plain, with_bias=True) < TOLERANCE
    # the rates of the two channels' ring: both directions within a factor 1.5 of each other
    # (the wells are mirror images); the bias-reweighted ones for the biased run
    k12 = result.k12_bias_reweighted if biased else result.k12
    k21 = result.k21_bias_reweighted if biased else result.k21
    assert 1 / 1.5 < k12 / k21 < 1.5
    for subset in result.subsets:
        assert np.isfinite(subset.k12) and np.isfinite(subset.k21)


def _spy(monkeypatch):
    from aimmd.pathensemble import PathEnsemble
    from aimmd.pathensemble import bias_utils
    calls, bias_calls = [], []
    reweight = PathEnsemble.reweight
    bias_rates = bias_utils.bias_reweighted_rates

    def reweight_spy(self, *args, **kwargs):
        result = reweight(self, *args, **kwargs)
        calls.append((np.asarray(self.n_frames).copy(), np.asarray(result[0]).copy()))
        return result

    def bias_spy(*args, **kwargs):
        result = bias_rates(*args, **kwargs)
        bias_calls.append(result)
        return result

    monkeypatch.setattr(PathEnsemble, 'reweight', reweight_spy)
    monkeypatch.setattr(bias_utils, 'bias_reweighted_rates', bias_spy)
    return calls, bias_calls


@pytest.mark.parametrize('with_bias', [False, True])
def test_a_cgaimmd_toy_campaign(tmp_path, monkeypatch, restore_signals,  # noqa: F811
                                short_sys_path, with_bias):  # noqa: F811
    """Free simulations, bootstrap shots and one trained round of the flow-matching fit: the
    offline reweighting gives the trainer's rates and densities; the run card has every params
    field and the metadata the fit stored; nothing is written into the run folder."""
    from tests._cgaimmd_toy_campaign import make_trained_campaign
    from tests.test_toy_1d_cgaimmd_campaign import BIAS, SHORT_WINDOW, run_campaign
    folder = make_trained_campaign(tmp_path / 'campaign', 'flow_matching',
                                   SHORT_WINDOW + (BIAS if with_bias else ''))
    monkeypatch.chdir(folder)
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    calls, bias_calls = _spy(monkeypatch)
    run = run_campaign(folder, params, n_target=0)       # the trainer's round is the last step
    monkeypatch.undo()
    monkeypatch.chdir(folder)
    (n1, w1), (n2, w2) = calls[-2:]
    trainer = [1 / np.sum(w1 * n1) if np.sum(w1 * n1) else np.nan,
               1 / np.sum(w2 * n2) if np.sum(w2 * n2) else np.nan]
    before = snapshot(run)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    result = reweight_run(params, run)
    assert np.all(np.isfinite(trainer)), trainer
    np.testing.assert_allclose([result.k12, result.k21], trainer, rtol=1e-6)
    densities = os.path.join(run, 'densitiesARB.npy')
    if os.path.exists(densities):
        np.testing.assert_allclose(result.densities, np.load(densities), rtol=1e-12)
    if with_bias:
        k12_rw, k21_rw, _, _ = bias_calls[-1]
        np.testing.assert_allclose([result.k12_bias_reweighted, result.k21_bias_reweighted],
                                   [k12_rw, k21_rw], rtol=1e-6)
    card = run_card(params, run)
    assert set(card['params']) == {field.name for field in dataclasses.fields(aimmd.Params)}
    assert card['cgaimmd']['generator']['method'] == 'ode'
    from aimmd.cgaimmd.flow_matching.fit import FIT_NAME
    assert card['fit']['last_round']['fit'] == FIT_NAME
    assert card['fit']['last_round']['keywords'] == {'steps_per_round': 5, 'relabel_every': 2}
    assert (card['bias'] is not None) == with_bias
    with pytest.warns(UserWarning, match='crossing'):     # the toy keeps the default restarts
        rates = free_md_rates(params, run)
    assert rates['n_trajectories'] >= 2
    if with_bias:                                         # biased frames: only when allowed
        with pytest.raises(ValueError, match='records a bias'):
            free_md_reference(params, run)
    with pytest.warns(UserWarning, match='crossing'):
        free_md_reference(params, run, allow_biased_frames=with_bias)
    assert snapshot(run) == before
