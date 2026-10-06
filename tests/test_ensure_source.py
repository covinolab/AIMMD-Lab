"""Worker value passes featurize paths whose descriptor series is missing.

A shooting worker evaluates the network on cached descriptors in three places:
shooting-point selection (pool and overriding frames), TPS acceptance and the
value of a zero-weight path's shooting point. `Path.compute` with a cached
series as source skips a file whose series does not exist or is too short,
silently when ``raise_if_error=False``. A path can lack its series after a
lost file, after a crash between the two halves of a registration, or after a
switch to another descriptor series (the path was registered before it).
TPS acceptance then got no values for that path and raised IndexError (review
R7), selection raised TypeError, and the zero-weight shooting-point value was
never cached.

Each of these sites now runs the descriptor computation on the same frames
first ("ensure source"); for complete series it only checks the ledger.
"""
import os
import re

import numpy as np
import pytest

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import load_npy
from aimmd.pathensemble import PathEnsemble
from aimmd.path.utils import get_cache_fname
from aimmd.worker import utils as worker_utils
from aimmd.worker.utils import accept_or_reject_last_path, select_shooting_point
from tests._helpers_unit import build_path, simple_descriptors_function

BINS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
DENSITIES = np.array([0.1, 0.2, 0.3, 0.4])

# A R R R R R R B and A R R R R B; no frame sits on a bin edge or at x = 0
LEADING_X = np.array([-0.9, -0.45, -0.3, -0.2, 0.1, 0.3, 0.45, 0.8])
CURRENT_X = np.array([-0.8, -0.4, -0.1, 0.2, 0.35, 0.7])


def _values(descriptors):
    """Committor logit of the toy model: 10 x of the first atom."""
    return 10.0 * np.asarray(descriptors, dtype=float)[:, 0]


class _Counting:
    """Descriptor function that counts the frames it featurizes."""

    def __init__(self):
        self.n_frames = 0

    def __call__(self, trajectory):
        result = simple_descriptors_function(trajectory)
        self.n_frames += len(result)
        return result


def _params(**extra):
    params = aimmd.Params.placeholder
    params.__dict__.update(
        states="ARB", chain_type="tps", selection_pool_size=1,
        nbins=len(BINS) - 1, descriptors_function=_Counting(),
        values_function=_values, uniform_selection_on_initial_paths=False,
        free_overriding_states="", **extra)
    params.__dict__["update_network"] = lambda *args, **kwargs: None
    params.__dict__["load_bins_and_densities"] = (
        lambda *args, **kwargs: (BINS.copy(), DENSITIES.copy()))
    return params


def _positions(x):
    x = np.asarray(x, dtype=float)
    zeros = np.zeros_like(x)
    return np.stack([np.stack([x, zeros, zeros], 1),
                     np.stack([x + 2.0, zeros, zeros], 1)], 1).astype(np.float32)


def _path(folder, stem, x, shooting_index, drop=("descriptors",)):
    folder.mkdir(parents=True, exist_ok=True)
    path = build_path(folder, stem=stem, positions=_positions(x),
                      shooting_index=shooting_index)
    for attribute in drop:
        os.remove(get_cache_fname(path.fname, attribute))
    return path


def _x(path):
    """x of the first atom as stored in the xtc (precision included)."""
    return path.positions[:, 0, 0].astype(float)


def _shooting_point_bias(path):
    """Selection bias of a path's shooting point, as the acceptance step
    computes it, from values evaluated directly on the trajectory."""
    values = 10.0 * _x(path)[1:-1]
    bin_weights = np.array(list(1 / DENSITIES) + [0.0])
    biases = bin_weights[np.digitize(values, BINS) - 1]
    biases /= biases.sum() or 1.0
    return biases[path.shooting_index - 1] or 1.0


@pytest.mark.parametrize("missing", ["leading", "current", "both"])
def test_tps_acceptance_on_paths_without_series(tmp_path, capsys, missing):
    """R7: the leading path was registered before its series existed."""
    folder = tmp_path / "run" / "chainR0"
    drop_leading = ("descriptors",) if missing in ("leading", "both") else ()
    drop_current = ("descriptors",) if missing in ("current", "both") else ()
    leading = _path(folder, "path000001", LEADING_X, 3, drop_leading)
    current = _path(folder, "path000002", CURRENT_X, 2, drop_current)
    chain = PathEnsemble(leading, current)
    params = _params()
    NPY_CACHE.clear()

    accept_or_reject_last_path(chain, params)

    out = capsys.readouterr().out
    acceptance = float(re.findall(r"acceptance probability: ([\d.]+)", out)[-1])
    expected = _shooting_point_bias(current) / _shooting_point_bias(leading)
    assert acceptance == pytest.approx(expected, abs=1e-3)
    assert expected != pytest.approx(1.0)
    # the internal frames of the paths now have their series
    for path in (leading, current):
        series = load_npy(get_cache_fname(path.fname, "descriptors"))
        np.testing.assert_array_equal(
            series[1:len(path) - 1],
            np.asarray(simple_descriptors_function(path.reader))[1:-1])
    NPY_CACHE.clear()


def test_pool_selection_on_path_without_series(tmp_path, capsys):
    """Neither descriptors nor values: both are computed before selection."""
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3,
                 drop=("descriptors", "values"))
    params = _params()
    NPY_CACHE.clear()
    np.random.seed(0)

    select_shooting_point(PathEnsemble(path), params, str(folder),
                          target_state="R")

    assert "=== selecting frame" in capsys.readouterr().out
    np.testing.assert_allclose(
        load_npy(get_cache_fname(path.fname, "values")), 10.0 * _x(path))
    assert params.descriptors_function.n_frames == len(path)
    NPY_CACHE.clear()


def test_pool_selection_leaves_series_alone_when_values_exist(tmp_path, capsys):
    """Ensuring the source costs nothing when no value has to be computed."""
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3)    # values cached
    params = _params()
    NPY_CACHE.clear()
    np.random.seed(0)

    select_shooting_point(PathEnsemble(path), params, str(folder),
                          target_state="R")

    assert params.descriptors_function.n_frames == 0
    assert not os.path.exists(get_cache_fname(path.fname, "descriptors"))
    NPY_CACHE.clear()


def test_pool_selection_fills_only_the_frames_lacking_values(tmp_path, capsys):
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3)
    values_fname = get_cache_fname(path.fname, "values")
    values = load_npy(values_fname).copy()
    values[[2, 5]] = 0.0                               # not computed yet
    np.save(values_fname, values)
    params = _params()
    NPY_CACHE.clear()
    np.random.seed(0)

    select_shooting_point(PathEnsemble(path), params, str(folder),
                          target_state="R")

    assert params.descriptors_function.n_frames == 2
    np.testing.assert_allclose(load_npy(values_fname)[[2, 5]],
                               10.0 * _x(path)[[2, 5]])
    NPY_CACHE.clear()


def test_zero_weight_shooting_point_value_without_series(tmp_path):
    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 4,
                 drop=("descriptors", "values"))
    path.weight = 0.0
    params = _params()
    NPY_CACHE.clear()

    value = worker_utils.compute_shooting_point_value(path, params)

    np.testing.assert_allclose(value, [10.0 * _x(path)[4]])
    cached = load_npy(get_cache_fname(path.fname, "values"))
    assert cached[4] == pytest.approx(10.0 * _x(path)[4])
    assert params.descriptors_function.n_frames == 1
    NPY_CACHE.clear()


def test_shoot_loop_caches_zero_weight_shooting_point_value(tmp_path, monkeypatch):
    """The shooting worker uses the same helper for a zero-weight path."""
    from tests.test_sweep_validation import _TinySweepWorker, _sweep_params

    (tmp_path / "initialARB").mkdir()
    seed = build_path(tmp_path / "initialARB", stem="seed",
                      positions=_positions(LEADING_X), shooting_index=3)
    params = _sweep_params()                      # rfps, nbins = 1
    params.__dict__.update(descriptors_function=_Counting(),
                           values_function=_values)
    chain = PathEnsemble()
    params.__dict__["shot_chains"] = lambda directory, t, k=None: chain
    worker = _TinySweepWorker(params, PathEnsemble(seed), tmp_path)
    folder = tmp_path / "chainR0"

    # one completed backward and forward half, engine fully mocked
    monkeypatch.setattr(worker, "_simulate",
                        lambda *a, **k: (2, 3, "A", 1), raising=False)
    monkeypatch.setattr("aimmd.worker._shoot.remove", lambda *a, **k: None)
    monkeypatch.setattr(
        "aimmd.worker._shoot.Path",
        lambda *a, **k: seed.copy() if not a else aimmd.Path(*a, **k))
    registered = []

    def fake_register(path, chain_, eneconv, **kwargs):
        # an R-R-R path: not complete, so the worker gives it zero weight
        new = _path(folder, "path000001", np.linspace(-0.3, 0.3, len(path)),
                    0, drop=("descriptors", "values"))
        path._fnames, path._first, path._last = new._fnames, [0], [len(path) - 1]
        chain_.append(path)
        registered.append(path)
        worker.must_stop = True

    monkeypatch.setattr("aimmd.worker._shoot.register_path", fake_register)
    NPY_CACHE.clear()

    worker._shoot(target_state="R", k=0)

    path, = registered
    assert path.weight == 0.0
    si = path.shooting_index
    cached = load_npy(get_cache_fname(path.fname, "values"))
    assert cached is not None and cached[si] == pytest.approx(10.0 * _x(path)[si])
    NPY_CACHE.clear()


def test_ledger_reads_nothing_for_frames_the_conditions_exclude(tmp_path, monkeypatch):
    """`Path.compute` applies the conditions first and checks the target
    ledger only on the frames they leave."""
    import aimmd.path._compute as compute_module

    folder = tmp_path / "run" / "chainR0"
    path = _path(folder, "path000001", LEADING_X, 3, drop=())
    calls = []
    real = compute_module._ledger_rows

    def spy(fname, locs):
        calls.append(list(locs))
        return real(fname, locs)

    monkeypatch.setattr(compute_module, "_ledger_rows", spy)
    NPY_CACHE.clear()
    values = load_npy(get_cache_fname(path.fname, "values"))

    # every frame has a value: no ledger check at all
    assert path.compute(simple_descriptors_function, "descriptors",
                        conditions={"values": worker_utils._lacks_value}) == 0
    assert calls == []

    # two frames lack a value: the ledger is asked about those two only
    values = values.copy()
    values[[2, 5]] = 0.0
    np.save(get_cache_fname(path.fname, "values"), values)
    NPY_CACHE.clear()
    assert path.compute(simple_descriptors_function, "descriptors",
                        conditions={"values": worker_utils._lacks_value}) == 0
    assert calls == [[2, 5]]
    NPY_CACHE.clear()
