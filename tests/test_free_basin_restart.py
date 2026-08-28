"""
Unit tests for the equilibrium (in-basin) free-simulation restart.

Why this exists
---------------
`aimmd.worker._free._free` restarts every new free trajectory from the last
frame the previous one spent in the target state — the frame it escaped from,
which sits on the state boundary. Each first passage therefore starts from the
boundary-entry distribution instead of the equilibrium distribution inside the
state. The two agree only when in-state relaxation is fast compared with the
escape time; when they disagree, the first-passage times stop being exponential
and `k = N / Σ(w·L·γ)` is biased toward faster rates.

`get_basin_frames_for_free_restart` draws the restart configuration from the
frames the accumulated free trajectories of that state actually spent inside it.

Tests
-----
test_draws_only_in_state_frames        — never seeds from an R/B frame
test_never_draws_frame_zero            — a history frame always exists
test_history_frame_is_the_predecessor  — pair is in forward time order
test_occupancy_is_uniform_over_frames  — long dwells dominate, as occupancy does
test_unbiased_weighting_prefers_the_fill — exp(bias) draw favours the core
test_unbiased_falls_back_without_bias  — no silent dropping of uncached dwells
test_returns_none_for_the_reactive_state — refuses where it is meaningless
test_returns_none_without_in_state_frames — caller keeps its own fallback
test_min_frames_gates_the_draw         — historical behaviour until pool is big
test_seed_bias_is_the_history_frame_bias — .part0000 gets its real bias
test_excluded_frames_are_never_drawn   — indicted frames masked by _get('states')
"""

import numpy as np
import pytest

from aimmd.worker.utils import get_basin_frames_for_free_restart


class FakeFreePath:
    """Minimal stand-in for aimmd.path.Path over a per-frame state/bias list."""

    def __init__(self, states, bias=None, exclude_from=-1):
        self.states_full = np.asarray(list(states), dtype='<U1')
        self._bias = None if bias is None else np.asarray(bias, dtype=float)
        self._exclude_from = exclude_from

    def __len__(self):
        return len(self.states_full)

    def _get(self, attribute, raise_if_missing=False):
        if attribute in ('states', 'true_states'):
            states = self.states_full.copy()
            if attribute == 'states' and self._exclude_from >= 0:
                states[self._exclude_from:] = ''
            return states
        if attribute == 'bias':
            if self._bias is None:
                if raise_if_missing:
                    raise TypeError('no bias cache')
                return np.zeros(len(self), dtype=float)
            return self._bias.copy()
        raise AttributeError(attribute)

    def __getitem__(self, key):
        assert isinstance(key, slice)
        segment = FakeFreePath(self.states_full[key],
                              None if self._bias is None else self._bias[key])
        segment.origin = (key.start, key.stop)
        segment.source = self
        return segment

    @property
    def states(self):
        return self._get('states')

    @property
    def locs(self):
        return np.arange(len(self))

    @property
    def filenames(self):
        return np.array(['fake.xtc'] * len(self))


def _draw_many(pool, n=400, **kwargs):
    np.random.seed(1234)
    return [get_basin_frames_for_free_restart(pool, 'A', 'R', **kwargs)
            for _ in range(n)]


def test_draws_only_in_state_frames():
    path = FakeFreePath('AAARRRAAARRRB')
    for frames, _ in _draw_many([path], n=200):
        assert frames is not None
        assert frames.states[1] == 'A'


def test_never_draws_frame_zero():
    """A restart needs a history frame, so index 0 must never be selected."""
    path = FakeFreePath('AAAB')
    for frames, _ in _draw_many([path], n=100):
        assert frames.origin[0] >= 0
        assert frames.origin[1] - frames.origin[0] == 2


def test_history_frame_is_the_predecessor():
    """The pair is (j-1, j), forward in time: `.part0000` is the real past.

    The historical boundary restart hands over (exit+1, exit) reversed, i.e. it
    writes a *future* frame as the seed's history.
    """
    path = FakeFreePath('RAAAAB')
    frames, _ = _draw_many([path], n=1)[0]
    start, stop = frames.origin
    assert stop == start + 2


def test_occupancy_is_uniform_over_frames():
    """Frame-uniform == occupancy weighted: a long dwell dominates a short one.

    This is the property that repairs the pathology: one trajectory that spent
    1000 frames in the basin core outvotes 10 that only grazed the boundary.
    """
    long_dwell = FakeFreePath('R' + 'A' * 1000 + 'RB')
    short = [FakeFreePath('RAAB') for _ in range(10)]
    picks = [frames.source is long_dwell
             for frames, _ in _draw_many([long_dwell] + short, n=400)]
    # 1000 in-state frames against 10 x 2 -> 1000/1020 = 98 %
    assert np.mean(picks) > 0.95


def test_unbiased_weighting_prefers_the_fill():
    """exp(bias) weighting draws from the *unbiased* in-state distribution."""
    states = 'R' + 'A' * 10 + 'B'
    bias = [0.0] + [0.0] * 5 + [6.0] * 5 + [0.0]     # deep fill in the core
    path = FakeFreePath(states, bias)
    deep = [frames.origin[1] - 1 >= 6
            for frames, _ in _draw_many([path], n=400, weighting='unbiased')]
    assert np.mean(deep) > 0.95, 'exp(6) should dominate exp(0)'
    flat = [frames.origin[1] - 1 >= 6
            for frames, _ in _draw_many([path], n=400, weighting='occupancy')]
    assert 0.3 < np.mean(flat) < 0.7, 'occupancy weighting must stay uniform'


def test_unbiased_falls_back_without_bias(capsys):
    """An uncached trajectory must not be silently excluded from the draw.

    The uncached one is typically the long in-basin dwell — the single most
    important candidate — so dropping it would be worse than ignoring the bias.
    """
    cached = FakeFreePath('RAAAB', [0.0, 6.0, 6.0, 6.0, 0.0])
    uncached = FakeFreePath('R' + 'A' * 50 + 'B')
    frames, _ = _draw_many([cached, uncached], n=1, weighting='unbiased')[0]
    assert frames is not None
    assert 'occupancy' in capsys.readouterr().out


def test_returns_none_for_the_reactive_state():
    path = FakeFreePath('RRRAAA')
    assert get_basin_frames_for_free_restart(
        [path], 'R', 'R') == (None, None)


def test_returns_none_without_in_state_frames():
    path = FakeFreePath('RRRB')
    assert get_basin_frames_for_free_restart(
        [path], 'A', 'R') == (None, None)
    assert get_basin_frames_for_free_restart(
        [], 'A', 'R') == (None, None)


def test_min_frames_gates_the_draw():
    path = FakeFreePath('RAAB')       # two in-state frames
    assert get_basin_frames_for_free_restart(
        [path], 'A', 'R', min_frames=100) == (None, None)
    frames, _ = get_basin_frames_for_free_restart(
        [path], 'A', 'R', min_frames=2)
    assert frames is not None


def test_seed_bias_is_the_history_frame_bias():
    """`.part0000` must carry its real bias, not 0.

    For a trajectory only a few frames long, calling the history frame's
    exp(bias) = 1 instead of exp(6) = 403 changes gamma by a large factor in the
    direction of a faster rate.
    """
    path = FakeFreePath('RAAAB', [0.0, 6.0, 6.0, 6.0, 0.0])
    frames, seed_bias = _draw_many([path], n=1)[0]
    j = frames.origin[1] - 1
    assert seed_bias is not None
    np.testing.assert_allclose(seed_bias, [path._bias[j - 1]])


def test_excluded_frames_are_never_drawn():
    """Frames past `_exclude_from` are masked by `_get('states')`."""
    path = FakeFreePath('RAAAAAAAAB', exclude_from=4)
    for frames, _ in _draw_many([path], n=100):
        assert frames.origin[1] <= 4
