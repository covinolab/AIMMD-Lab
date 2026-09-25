"""Shooting-point selection on initial (seed) paths.

Regression tests for the kcmpd09 "trial2" failure: a TPS chain whose only
source of shooting points was the seed shot 324 times from the same seed frame
(0 of 592 halves reached B) and never from the seed frames near the transition
state, because
  1. ``is_initial_path`` never recognised a seed, so the seed's values were
     computed once, by the first network (trained on 2 frames), and never
     re-evaluated;
  2. the value-guided rule picks among the frames that fall inside the bins,
     whose upper edge stays at +0.5 until B-side data exist, so the seed frames
     near B (values > 0.5) were never selectable.
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker.utils import is_initial_path, select_shooting_point
from tests._helpers_unit import TinyNetwork, build_path

# trial2-like seed block: A, 16 R frames, B. Values: frames 1-11 far on the A
# side, 12 and 13 as logged (-14.267, -2.289), 14-16 on the B side (the frames
# near the transition state), 17 = B.
SEED_VALUES = np.array([-41.935] + [-30.0] * 11
                       + [-14.267, -2.289, 5.29, 7.70, 14.25, 18.37])
# the last trial2 bins: upper edge at cutoff_min = +0.5
BINS = np.linspace(-12.849, 0.5, 11)


def _seed(folder):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(SEED_VALUES)
    x = np.linspace(-1.0, 1.0, n)
    x[1:-1] = np.linspace(-0.45, 0.45, n - 2)          # all internal frames in R
    positions = np.stack([np.stack([x, np.zeros(n), np.zeros(n)], axis=1),
                          np.stack([x + 1, np.zeros(n), np.zeros(n)], axis=1)],
                         axis=1).astype(np.float32)
    return build_path(folder, stem="seed", positions=positions,
                      values=SEED_VALUES, shooting_index=13)


def _params(monkeypatch, chain_type="tps", **extra):
    params = aimmd.Params.placeholder
    params.__dict__.update(states="ARB", selection_pool_size=1,
                           chain_type=chain_type, nbins=len(BINS) - 1,
                           network=TinyNetwork(), **extra)
    params.__dict__["update_network"] = lambda *args, **kwargs: None
    params.__dict__["load_bins_and_densities"] = (
        lambda *args, **kwargs: (BINS.copy(), np.ones(len(BINS) - 1)))
    # values come from the cache written by build_path; no network evaluation
    monkeypatch.setattr(PathEnsemble, "compute", lambda self, *a, **k: 0)
    return params


def _picks(tmp_path, params, capsys, chain=None, n=200):
    seed = _seed(tmp_path / "run" / "initialARB")
    folder = tmp_path / "run" / "chainR0"
    folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    picks = []
    for _ in range(n):
        select_shooting_point(PathEnsemble(seed), params, str(folder),
                              chain=chain, target_state="R")
        out = capsys.readouterr().out
        picks.append(int(re.findall(r"=== selecting frame (\d+)", out)[-1]))
    return np.array(picks)


def test_is_initial_path_recognises_the_seed_folder(tmp_path):
    for fname in ("run/initialARB/seed.xtc", "/abs/run/initialARB/seed.xtc",
                  "initialRAB/x.xtc"):
        assert is_initial_path(SimpleNamespace(fname=fname))
    for fname in ("run/chainR0/path000001.xtc", "/abs/run/freeA/traj.xtc",
                  "initial_selected.xtc"):
        assert not is_initial_path(SimpleNamespace(fname=fname))
    assert is_initial_path(_seed(tmp_path / "run" / "initialARB"))


def test_seed_values_are_reevaluated_at_every_selection(tmp_path, monkeypatch,
                                                        capsys):
    params = _params(monkeypatch, uniform_selection_on_initial_paths=False)
    calls = []

    def recording_compute(self, *args, overwrite=False, **kwargs):
        calls.append((len(self), overwrite))
        return 0
    monkeypatch.setattr(PathEnsemble, "compute", recording_compute)
    _picks(tmp_path, params, capsys, n=3)
    # one overwrite=True call with the seed per selection
    assert sum(1 for n, ow in calls if ow and n == 1) == 3


def test_value_guided_rule_reproduces_the_trial2_lock(tmp_path, monkeypatch,
                                                      capsys):
    params = _params(monkeypatch, uniform_selection_on_initial_paths=False)
    picks = _picks(tmp_path, params, capsys)
    assert set(picks) == {13}


def test_uniform_selection_on_seed_until_first_transition(tmp_path,
                                                          monkeypatch, capsys):
    params = _params(monkeypatch)          # the default must switch it on
    assert params.uniform_selection_on_initial_paths is True
    picks = _picks(tmp_path, params, capsys)
    assert set(picks) <= set(range(1, 17))          # internal frames only
    assert {14, 15, 16} <= set(picks)               # near-TS frames reachable
    counts = np.bincount(picks, minlength=17)[1:]
    assert counts.min() > 0 and counts.max() < 30   # ~12.5 each of 200


def test_uniform_selection_stops_once_the_chain_has_a_transition(
        tmp_path, monkeypatch, capsys):
    params = _params(monkeypatch)
    x = np.array([-1.0, 0.0, 1.0])
    positions = np.stack([np.stack([x, 0 * x, 0 * x], 1),
                          np.stack([x + 1, 0 * x, 0 * x], 1)], 1)
    accepted = build_path(tmp_path, stem="accepted", positions=positions,
                          values=np.array([-1.0, 0.0, 1.0]))
    chain = PathEnsemble(accepted)
    assert chain.path is not None
    picks = _picks(tmp_path, params, capsys, chain=chain, n=20)
    assert set(picks) == {13}


def test_uniform_selection_is_tps_only(tmp_path, monkeypatch, capsys):
    params = _params(monkeypatch, chain_type="rfps")
    picks = _picks(tmp_path, params, capsys, n=20)
    assert set(picks) == {13}
