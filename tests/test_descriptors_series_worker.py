"""Shooting workers keep descriptors in `params.descriptors_series`.

register_path copies the rows of the backward and forward halves into the
series of the new path, TPS acceptance and the selection-time value passes
read it, and "ensure source" fills it. With a named series, a stale
`*.descriptors.npy` next to the same trajectories (from before a change of
series) is never opened. Params-like objects without the field (e.g.
SimpleNamespace) keep the historical 'descriptors'.
"""
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import load_npy, save_npy
from aimmd.pathensemble import PathEnsemble
from aimmd.path.utils import get_cache_fname
from aimmd.worker import utils as worker_utils
from aimmd.worker.utils import (accept_or_reject_last_path, register_path,
                                select_shooting_point)
from tests._helpers_unit import (build_path, forbid_opening,
                                 simple_descriptors_function)
from tests.test_ensure_source import (BINS, CURRENT_X, DENSITIES, LEADING_X,
                                      _Counting, _params, _positions,
                                      _shooting_point_bias, _values, _x)

SERIES = "descriptors-gn0123456789"
STALE = 100.0          # offset of the stale legacy rows: wrong values if read


def _with_series(path, series=SERIES, stale=True, rows=True):
    """Give `path` its rows in `series` and a stale legacy series."""
    legacy = get_cache_fname(path.fname, "descriptors")
    featurized = np.asarray(simple_descriptors_function(path.reader),
                            dtype=float)
    if rows:
        save_npy(get_cache_fname(path.fname, series), featurized)
    if stale:
        save_npy(legacy, featurized + STALE)
    elif os.path.exists(legacy):
        os.remove(legacy)
    return path


def _path(folder, stem, x, shooting_index, drop=(), **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    path = build_path(folder, stem=stem, positions=_positions(x),
                      shooting_index=shooting_index)
    for attribute in drop:
        os.remove(get_cache_fname(path.fname, attribute))
    return _with_series(path, **kwargs)


# ---------------------------------------------------------------------------
# register_path
# ---------------------------------------------------------------------------
BACK_X = np.array([0.0, -0.3, -0.6, -0.9])
FORW_X = np.array([0.0, 0.3, 0.6, 0.9])


def _halves(tmp_path):
    back = _path(tmp_path, "back", BACK_X, 1)
    forw = _path(tmp_path, "forw", FORW_X, 1)
    return back, forw


def _register(path, **kwargs):
    NPY_CACHE.clear()
    chain = PathEnsemble()
    register_path(path, chain, eneconv=None, **kwargs)
    NPY_CACHE.clear()
    return chain[-1]


def test_register_path_copies_the_named_series(tmp_path, monkeypatch):
    back, forw = _halves(tmp_path)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")

    registered = _register(back[::-1] + forw[1:], descriptors_series=SERIES)

    expected = np.asarray(simple_descriptors_function(registered.reader))
    np.testing.assert_array_equal(
        load_npy(get_cache_fname(registered.fname, SERIES)), expected)
    assert not os.path.exists(get_cache_fname(registered.fname, "descriptors"))
    assert opened == []


def test_register_path_backward_only_with_named_series(tmp_path, monkeypatch):
    back, forw = _halves(tmp_path)
    path = back[len(back) - 1::-1] + forw[1:1]     # as the shoot loop builds it
    opened = forbid_opening(monkeypatch, ".descriptors.npy")

    registered = _register(path, descriptors_series=SERIES)

    np.testing.assert_array_equal(
        load_npy(get_cache_fname(registered.fname, SERIES)),
        np.asarray(simple_descriptors_function(registered.reader)))
    assert not os.path.exists(get_cache_fname(registered.fname, "descriptors"))
    assert opened == []


def test_register_path_half_without_named_series(tmp_path, monkeypatch):
    """A half begun before the change of series: zero rows, legacy unread."""
    back = _path(tmp_path, "back", BACK_X, 1, rows=False)
    forw = _path(tmp_path, "forw", FORW_X, 1)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")

    registered = _register(back[::-1] + forw[1:], descriptors_series=SERIES)

    series = load_npy(get_cache_fname(registered.fname, SERIES))
    expected = np.asarray(simple_descriptors_function(registered.reader))
    assert not series[:len(back)].any()
    np.testing.assert_array_equal(series[len(back):], expected[len(back):])
    # the ledger refills the zero rows of the named series
    NPY_CACHE.clear()
    assert registered.compute(simple_descriptors_function, SERIES) == len(back)
    NPY_CACHE.clear()
    np.testing.assert_array_equal(
        load_npy(get_cache_fname(registered.fname, SERIES)), expected)
    assert opened == []


def test_register_path_defaults_to_descriptors(tmp_path):
    back, forw = _halves(tmp_path)
    registered = _register(back[::-1] + forw[1:])
    legacy = load_npy(get_cache_fname(registered.fname, "descriptors"))
    np.testing.assert_array_equal(
        legacy, np.asarray(simple_descriptors_function(registered.reader))
        + STALE)
    assert not os.path.exists(get_cache_fname(registered.fname, SERIES))


def test_shoot_loop_registers_into_the_params_series(tmp_path, monkeypatch):
    from tests.test_sweep_validation import _TinySweepWorker, _sweep_params

    (tmp_path / "initialARB").mkdir()
    seed = build_path(tmp_path / "initialARB", stem="seed",
                      positions=_positions(LEADING_X), shooting_index=3)
    params = _sweep_params()
    params.__dict__.update(descriptors_function=_Counting(),
                           values_function=_values, descriptors_series=SERIES)
    chain = PathEnsemble()
    params.__dict__["shot_chains"] = lambda directory, t, k=None: chain
    worker = _TinySweepWorker(params, PathEnsemble(seed), tmp_path)
    monkeypatch.setattr(worker, "_simulate",
                        lambda *a, **k: (2, 3, "A", 1), raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    monkeypatch.setattr(
        "aimmd.worker._shoot.Path",
        lambda *a, **k: seed.copy() if not a else aimmd.Path(*a, **k))
    calls = []

    def fake_register(path, chain_, eneconv, **kwargs):
        calls.append(kwargs)
        chain_._paths.append(path)
        path.weight = 1.0
        worker.must_stop = True

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)
    monkeypatch.setattr("aimmd.worker._shoot.compute_shooting_point_value",
                        lambda *a, **k: None)
    NPY_CACHE.clear()

    worker._shoot(target_state="R", k=0)

    assert [kwargs["descriptors_series"] for kwargs in calls] == [SERIES]
    NPY_CACHE.clear()


# ---------------------------------------------------------------------------
# value passes
# ---------------------------------------------------------------------------
def _acceptance(out):
    return float(re.findall(r"acceptance probability: ([\d.]+)", out)[-1])


def test_tps_acceptance_reads_the_named_series(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "run" / "chainR0"
    leading = _path(folder, "path000001", LEADING_X, 3)
    current = _path(folder, "path000002", CURRENT_X, 2)
    params = _params(descriptors_series=SERIES)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    NPY_CACHE.clear()

    accept_or_reject_last_path(PathEnsemble(leading, current), params)

    expected = _shooting_point_bias(current) / _shooting_point_bias(leading)
    assert _acceptance(capsys.readouterr().out) == pytest.approx(
        expected, abs=1e-3)
    assert params.descriptors_function.n_frames == 0     # rows were there
    assert opened == []
    NPY_CACHE.clear()


def test_tps_acceptance_fills_a_missing_named_series(tmp_path, capsys,
                                                     monkeypatch):
    """R7 for a path registered before the change of series."""
    folder = tmp_path / "run" / "chainR0"
    leading = _path(folder, "path000001", LEADING_X, 3, rows=False)
    current = _path(folder, "path000002", CURRENT_X, 2)
    params = _params(descriptors_series=SERIES)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    NPY_CACHE.clear()

    accept_or_reject_last_path(PathEnsemble(leading, current), params)

    expected = _shooting_point_bias(current) / _shooting_point_bias(leading)
    assert _acceptance(capsys.readouterr().out) == pytest.approx(
        expected, abs=1e-3)
    assert params.descriptors_function.n_frames == len(leading) - 2
    series = load_npy(get_cache_fname(leading.fname, SERIES))
    np.testing.assert_array_equal(
        series[1:len(leading) - 1],
        np.asarray(simple_descriptors_function(leading.reader))[1:-1])
    assert opened == []
    NPY_CACHE.clear()


def test_pool_selection_reads_the_named_series(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3, drop=("values",))
    params = _params(descriptors_series=SERIES)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    NPY_CACHE.clear()
    np.random.seed(0)

    select_shooting_point(PathEnsemble(path), params, str(folder),
                          target_state="R")

    np.testing.assert_allclose(
        load_npy(get_cache_fname(path.fname, "values")), 10.0 * _x(path))
    assert params.descriptors_function.n_frames == 0
    assert opened == []
    NPY_CACHE.clear()


def test_pool_selection_fills_a_missing_named_series(tmp_path, capsys,
                                                     monkeypatch):
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3, drop=("values",),
                 rows=False)
    params = _params(descriptors_series=SERIES)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    NPY_CACHE.clear()
    np.random.seed(0)

    select_shooting_point(PathEnsemble(path), params, str(folder),
                          target_state="R")

    np.testing.assert_allclose(
        load_npy(get_cache_fname(path.fname, "values")), 10.0 * _x(path))
    assert params.descriptors_function.n_frames == len(path)
    assert os.path.exists(get_cache_fname(path.fname, SERIES))
    assert opened == []
    NPY_CACHE.clear()


def test_zero_weight_shooting_point_value_reads_the_named_series(
        tmp_path, monkeypatch):
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 4, drop=("values",))
    params = _params(descriptors_series=SERIES)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    NPY_CACHE.clear()

    value = worker_utils.compute_shooting_point_value(path, params)

    np.testing.assert_allclose(value, [10.0 * _x(path)[4]])
    assert params.descriptors_function.n_frames == 0
    assert opened == []
    NPY_CACHE.clear()


def test_tps_acceptance_with_params_without_the_field(tmp_path, capsys):
    """A SimpleNamespace without descriptors_series reads 'descriptors'."""
    folder = tmp_path / "run" / "chainR0"
    leading = _path(folder, "path000001", LEADING_X, 3, stale=False,
                    rows=False)
    current = _path(folder, "path000002", CURRENT_X, 2, stale=False,
                    rows=False)
    for path in (leading, current):        # the legacy series, correct rows
        save_npy(get_cache_fname(path.fname, "descriptors"),
                 np.asarray(simple_descriptors_function(path.reader)))
    params = SimpleNamespace(
        states="ARB", network_batch_size=4096,
        descriptors_function=simple_descriptors_function,
        values_function=_values,
        update_network=lambda *args, **kwargs: None,
        load_bins_and_densities=lambda *args, **kwargs: (
            BINS.copy(), DENSITIES.copy()))
    NPY_CACHE.clear()

    accept_or_reject_last_path(PathEnsemble(leading, current), params)

    expected = _shooting_point_bias(current) / _shooting_point_bias(leading)
    assert _acceptance(capsys.readouterr().out) == pytest.approx(
        expected, abs=1e-3)
    NPY_CACHE.clear()
