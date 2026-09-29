"""The shooting index of a two-way path loaded from disk.

`Path.find_shooting_index` reconstructs the shooting frame (time 0) of a path
whose first frames run backwards in time as ``-t0 / dt`` from the first two
time stamps. That ratio is only nearly integral when the frame spacing is not
exact in binary: with 0.05 ps frames ``t0 = 0.2`` and ``dt = -0.05000000000000002``
give ``3.9999999999999982``, and truncating it with ``int`` puts the shooting
index one frame before the t = 0 frame.

The shoot worker's in-memory path has the right index (it is sliced from the
segments); every process that loads the path from disk (`params.shot_paths`:
the trainer, a restarted shoot worker) gets the truncated one. Truncation stays
the default; ``Params.round_shooting_index=True`` (or
``shooting_index='find_nearest'``) takes the nearest frame.
"""
import numpy as np
import pytest

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import build_params_file, write_trajectory


# a two-way path with 0.05 ps frames: 4 backward frames, the shooting point,
# 2 forward frames (times as the shoot worker writes them)
TIMES = np.array([0.2, 0.15, 0.1, 0.05, 0.0, 0.05, 0.1])
X = np.array([-1.0, -0.4, -0.3, -0.1, 0.0, 0.2, 1.0])
T0_INDEX = 4


def _write_path(folder, stem='path000001'):
    positions = np.zeros((len(X), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = X
    fname = write_trajectory(folder, stem=stem, positions=positions,
                             times=TIMES)
    save_npy(get_cache_fname(fname, 'states'),
             np.array(['A', 'R', 'R', 'R', 'R', 'R', 'B'], dtype='<U1'))
    return fname


def test_find_truncates_by_default(tmp_path):
    """The old behaviour, kept: one frame before t = 0."""
    path = aimmd.Path(_write_path(tmp_path), shooting_index='find')
    assert path.shooting_index == T0_INDEX - 1
    assert path.find_shooting_index() == T0_INDEX - 1


def test_find_nearest_takes_the_t0_frame(tmp_path):
    path = aimmd.Path(_write_path(tmp_path), shooting_index='find_nearest')
    assert path.shooting_index == T0_INDEX
    assert path.find_shooting_index(nearest=True) == T0_INDEX
    assert np.isclose(path.times[path.shooting_index], 0.0)


def test_find_nearest_matches_the_assembled_shot(tmp_path):
    """The shoot worker's own index: `back[n - 1::-1] + forw[1:]`."""
    positions = np.zeros((5, 1, 3), dtype=np.float32)
    write_trajectory(tmp_path, stem='back', positions=positions,
                     times=np.arange(5) * 0.05)
    write_trajectory(tmp_path, stem='forw', positions=positions[:3],
                     times=np.arange(3) * 0.05)
    back, forw = aimmd.Path(), aimmd.Path()
    back.extend(str(tmp_path / 'back.xtc'))
    forw.extend(str(tmp_path / 'forw.xtc'))
    shot = back[4::-1] + forw[1:3]
    shot.write(str(tmp_path / 'path000001.xtc'))
    fname = str(tmp_path / 'path000001.xtc')

    assert shot.shooting_index == T0_INDEX
    assert aimmd.Path(fname, shooting_index='find').shooting_index == 3
    assert (aimmd.Path(fname, shooting_index='find_nearest').shooting_index
            == shot.shooting_index)


@pytest.mark.parametrize('times, expected', [
    ([0.0, 0.05, 0.1], 0),              # starts at the shooting point
    ([3.0, 2.0, 1.0, 0.0, 1.0], 3),     # exact spacing: both agree
    ([0.3, 0.2, 0.1], 2),               # clamped to the last frame
])
def test_find_nearest_agrees_where_truncation_is_exact(tmp_path, times,
                                                      expected):
    positions = np.zeros((len(times), 1, 3), dtype=np.float32)
    fname = write_trajectory(tmp_path, positions=positions,
                             times=np.array(times))
    assert aimmd.Path(fname, shooting_index='find').shooting_index == expected
    assert (aimmd.Path(fname, shooting_index='find_nearest').shooting_index
            == expected)


@pytest.mark.parametrize('round_index, expected', [
    (None, T0_INDEX - 1), (False, T0_INDEX - 1), (True, T0_INDEX)])
def test_shot_paths_follow_round_shooting_index(tmp_path, round_index,
                                                expected):
    chain = tmp_path / 'run' / 'chainR0'
    chain.mkdir(parents=True)
    _write_path(chain)
    initial = write_trajectory(tmp_path, stem='initial')
    params = aimmd.Params.load(build_params_file(tmp_path, initial),
                               save=False)
    if round_index is None:
        assert params.round_shooting_index is False
    else:
        params.update(round_shooting_index=round_index, save=False)

    shot = params.shot_chains(str(tmp_path / 'run'), 'R', 0)

    assert len(shot) == 1
    assert shot[0].shooting_index == expected
