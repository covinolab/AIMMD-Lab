"""TPS acceptance after a value-guided shooting-point selection.

With ``selection_pool_size=1``, `select_shooting_point` draws a bin k with
probability proportional to n_k / d_k (n_k internal frames of the leading path
in bin k, d_k the selection density of the bin) and then one frame of that bin
uniformly. Frame x of path X is therefore selected with probability

    p_sel(x | X) = w(x) / sum_{y internal to X} w(y),    w = 1 / d(bin).

For two-way shooting with momenta drawn symmetrically, the Metropolis
acceptance of the new path X_new, whose shooting point is x, is

    min(1, p_sel(x | X_new) / p_sel(x | X_old)) h_AB(X_new)
        = min(1, sum_{X_old} w / sum_{X_new} w) h_AB(X_new),

because x is the same frame in both paths. The frame the leading path was
itself shot from (``leading.shooting_index``) does not enter. Up to ce2f005 the
acceptance step evaluated the leading path's selection probability at that
frame instead of at x, which multiplies the ratio by
w(x) / w(own shooting point of X_old).

The acceptance tests in test_ensure_source.py and
test_descriptors_series_worker.py (feat/nodetable-descriptor-series) use two
paths whose shooting points share a bin, where both formulas give the same
number; the paths here separate them. Only `tests._helpers_unit` is imported,
so the file runs on main as well.
"""
import re

import numpy as np
import pytest

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import load_npy, save_npy
from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble import PathEnsemble
from aimmd.worker.utils import accept_or_reject_last_path, select_shooting_point
from tests._helpers_unit import build_path, simple_descriptors_function

# value = 10 x (x of the first atom); states: x <= -0.5 A, x >= 0.5 B, else R
BINS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
DENSITIES = np.array([0.1, 0.2, 0.3, 0.4])          # w = 10, 5, 10/3, 2.5

# Leading path X_old: A R R R R R R B. Internal values -4.5 -3 -2 1 3 4.5,
# bins 1 1 1 2 2 2, w = 5 5 5 10/3 10/3 10/3, sum 25.
# The new shooting point x is its frame 6 (x = 0.45, value 4.5, bin 2).
LEADING_X = np.array([-0.9, -0.45, -0.3, -0.2, 0.1, 0.3, 0.45, 0.8])
X_IN_LEADING = 6
# New path X_new, shot from x (its frame 7): A R R R R R R R B. Internal
# values -4.5 -4 -3.5 -3 -1 2 4.5, w = 5 (five times) and 10/3 (twice),
# sum 31.67. Correct acceptance 25 / 31.67 = 0.789. The old formula, with the
# leading path's own shooting point at frame 3 (bin 1, w = 5), gave
# 0.789 * (10/3) / 5 = 0.526.
NEW_X = np.array([-0.8, -0.45, -0.4, -0.35, -0.3, -0.1, 0.2, 0.45, 0.7])
X_IN_NEW = 7


def _values(descriptors):
    """Committor logit of the toy model: 10 x of the first atom."""
    return 10.0 * np.asarray(descriptors, dtype=float)[:, 0]


def _params():
    params = aimmd.Params.placeholder
    params.__dict__.update(
        states="ARB", chain_type="tps", selection_pool_size=1,
        nbins=len(BINS) - 1, descriptors_function=simple_descriptors_function,
        values_function=_values, uniform_selection_on_initial_paths=False,
        free_overriding_states="")
    params.__dict__["update_network"] = lambda *args, **kwargs: None
    params.__dict__["load_bins_and_densities"] = (
        lambda *args, **kwargs: (BINS.copy(), DENSITIES.copy()))
    return params


def _x(path):
    """x of the first atom as stored in the xtc (precision included)."""
    return path.positions[:, 0, 0].astype(float)


def _path(folder, stem, x, shooting_index):
    folder.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    zeros = np.zeros_like(x)
    positions = np.stack([np.stack([x, zeros, zeros], 1),
                          np.stack([x + 2.0, zeros, zeros], 1)], 1)
    path = build_path(folder, stem=stem, positions=positions.astype(np.float32),
                      shooting_index=shooting_index)
    # cached values as the network gives them
    save_npy(get_cache_fname(path.fname, "values"), 10.0 * _x(path))
    return path


def _weights(path):
    """w of the internal frames of `path`, from the values the acceptance step
    computes on the trajectory."""
    bin_weights = np.array(list(1 / DENSITIES) + [0.0])
    return bin_weights[np.digitize(10.0 * _x(path)[1:-1], BINS) - 1]


def _paths(tmp_path, own_shooting_index=3):
    """The leading path, shot from its frame `own_shooting_index`, and the new
    path, shot from its frame X_IN_LEADING."""
    folder = tmp_path / "run" / "chainR0"
    leading = _path(folder, "path000001", LEADING_X, own_shooting_index)
    new = _path(folder, "path000002", NEW_X, X_IN_NEW)
    assert _x(leading)[X_IN_LEADING] == _x(new)[X_IN_NEW]   # the same frame
    return folder, leading, new


def _accept(leading, new, params, capsys):
    """Run the acceptance step on the chain (leading, new); return the
    printed (unclipped) acceptance probability."""
    capsys.readouterr()
    NPY_CACHE.clear()
    accept_or_reject_last_path(PathEnsemble(leading, new), params)
    NPY_CACHE.clear()
    out = capsys.readouterr().out
    return float(re.findall(r"acceptance probability: ([\d.]+)", out)[-1])


def _selection_probability(path, index, params, folder, monkeypatch):
    """Probability with which `select_shooting_point` (pool = `path` alone)
    picks frame `index` of `path`: the probability of the bin it draws,
    divided by the number of internal frames of `path` in that bin."""
    drawn = []
    choice = np.random.choice

    def spy(a, *args, **kwargs):
        if kwargs.get("p") is not None:
            drawn.append(np.array(kwargs["p"], dtype=float))
        return choice(a, *args, **kwargs)

    NPY_CACHE.clear()
    with monkeypatch.context() as patch:
        patch.setattr(np.random, "choice", spy)
        select_shooting_point(PathEnsemble(path), params, str(folder),
                              target_state="R")
    NPY_CACHE.clear()
    p_bins, = drawn
    # the bins the selection used, as it saved them for the acceptance step
    bins = load_npy(f"{folder}/bins{params.sorted_states}.npy")
    k = np.digitize(10.0 * _x(path)[1:-1], bins) - 1
    return p_bins[k[index - 1]] / np.count_nonzero(k == k[index - 1])


def test_acceptance_is_the_ratio_of_the_selection_normalizations(tmp_path,
                                                                 capsys):
    _, leading, new = _paths(tmp_path)

    acceptance = _accept(leading, new, _params(), capsys)

    expected = _weights(leading).sum() / _weights(new).sum()
    assert acceptance == pytest.approx(expected, abs=1e-3)
    # x and the leading path's own shooting point lie in different bins, so
    # the old formula gives another number
    w = _weights(leading)
    old = expected * w[X_IN_LEADING - 1] / w[leading.shooting_index - 1]
    assert old != pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize("own_shooting_index", range(1, len(LEADING_X) - 1))
def test_acceptance_ignores_the_leading_paths_own_shooting_point(
        tmp_path, capsys, own_shooting_index):
    _, leading, new = _paths(tmp_path, own_shooting_index)

    acceptance = _accept(leading, new, _params(), capsys)

    expected = _weights(leading).sum() / _weights(new).sum()
    assert acceptance == pytest.approx(expected, abs=1e-3)


def test_detailed_balance_with_the_selection_probabilities(tmp_path, capsys,
                                                           monkeypatch):
    """p_sel(x | X_old) acc(X_old -> X_new) = p_sel(x | X_new) acc(X_new ->
    X_old), with p_sel read off `select_shooting_point` itself."""
    folder, leading, new = _paths(tmp_path)
    params = _params()

    p_forward = _selection_probability(leading, X_IN_LEADING, params, folder,
                                       monkeypatch)
    p_backward = _selection_probability(new, X_IN_NEW, params, folder,
                                        monkeypatch)
    # the selection rule is w(x) / sum w, the premise of the acceptance step
    for path, index, p in ((leading, X_IN_LEADING, p_forward),
                           (new, X_IN_NEW, p_backward)):
        w = _weights(path)
        assert p == pytest.approx(w[index - 1] / w.sum())

    # forward: X_old leads (it was shot from its frame 3), X_new is shot from x
    forward = min(1.0, _accept(leading, new, params, capsys))
    # backward: X_new leads (it was shot from x), X_old is shot from x
    backward = min(1.0, _accept(
        aimmd.Path(new.fname, shooting_index=X_IN_NEW, weight=1.0),
        aimmd.Path(leading.fname, shooting_index=X_IN_LEADING, weight=1.0),
        params, capsys))

    # the printed acceptance has three decimals
    assert p_forward * forward == pytest.approx(p_backward * backward,
                                                rel=5e-3)
