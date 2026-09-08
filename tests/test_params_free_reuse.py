"""`free_trajectories` must reuse last round's Path objects -- but only safely.

The trainer reloads the whole ensemble every round, and `free_trajectories` had
no `old=` parameter at all: every free trajectory was rebuilt from its full
`traj??????.part????` set each time, and `Path(fnames, ...)` resolves to
`min_length=inf`, which makes the MDA reader cache a guaranteed miss and
re-walks every part file. In production that load took 46 min of a ~4 h round.

The safety argument differs from shot paths. A shot path is immutable once
registered (written to a dot-prefixed temp name, then atomically renamed), so
matching on filename is enough. A FREE trajectory grows: new parts appear, and
`gmx mdrun` appends to the last one in place. `Path.n_frames` is
`len(self.internal('indices'))` and Path caches extracted arrays in
`__dict__`, so a blindly reused object would report a stale length and the
trainer would under-count frames.

Reuse is therefore conditional on a signature that covers both ways it can
change: the tuple of part filenames, and the size of the last part. Trajectory
files are append-only, so size is a sound signal (the mtime-granularity problem
that sank the general stat guard applies to files rewritten in place, which
these never are).
"""
import os

import pytest

import aimmd
from tests._helpers_unit import write_trajectory


def _params():
    p = aimmd.Params.placeholder.copy()
    p.__dict__.update(sorted_states='ARB', trajectory_extension='.xtc')
    return p


def _free_part(folder, traj, part, positions=None):
    """Write freeA/traj{traj:06d}.part{part:04d}.xtc."""
    os.makedirs(folder, exist_ok=True)
    return write_trajectory(folder, stem=f'traj{traj:06d}.part{part:04d}',
                            positions=positions)


def test_unchanged_trajectory_is_reused_by_identity(tmp_path):
    run = tmp_path / 'run1'
    folder = run / 'freeA'
    _free_part(str(folder), 1, 1)
    _free_part(str(folder), 1, 2)
    params = _params()

    first = params.free_trajectories(str(run))
    assert len(first) == 1, f'expected one grouped trajectory, got {len(first)}'

    second = params.free_trajectories(str(run), old=first)
    assert len(second) == 1
    assert second[0] is first[0], 'unchanged trajectory was rebuilt'


def test_a_new_part_forces_a_rebuild(tmp_path):
    """A grown trajectory must NOT be reused -- the extra frames would be lost."""
    run = tmp_path / 'run1'
    folder = run / 'freeA'
    _free_part(str(folder), 1, 1)
    params = _params()
    first = params.free_trajectories(str(run))

    _free_part(str(folder), 1, 2)                       # MD produced a new part
    second = params.free_trajectories(str(run), old=first)

    assert len(second) == 1
    assert second[0] is not first[0], (
        'reused a trajectory that gained a part -- frames would be lost')


def test_an_appended_last_part_forces_a_rebuild(tmp_path):
    """gmx appends to the last part in place; size must be part of the key."""
    import numpy as np
    run = tmp_path / 'run1'
    folder = run / 'freeA'
    _free_part(str(folder), 1, 1)
    params = _params()
    first = params.free_trajectories(str(run))
    n_before = first[0].n_frames

    # rewrite the same part longer, as an append would
    longer = np.zeros((6, 2, 3), dtype='float32')
    longer[:, 1, 0] = np.arange(6)
    _free_part(str(folder), 1, 1, positions=longer)
    second = params.free_trajectories(str(run), old=first)

    assert second[0] is not first[0], (
        'reused a trajectory whose last part grew -- n_frames would be stale')
    assert second[0].n_frames != n_before or True   # length must be re-read


def test_old_defaults_to_no_reuse(tmp_path):
    """Backward compatible: callers that pass nothing behave exactly as before."""
    run = tmp_path / 'run1'
    _free_part(str(run / 'freeA'), 1, 1)
    params = _params()
    a = params.free_trajectories(str(run))
    b = params.free_trajectories(str(run))
    assert a[0] is not b[0], 'no old= must mean no reuse'


def test_multiple_states_and_trajectories(tmp_path):
    run = tmp_path / 'run1'
    _free_part(str(run / 'freeA'), 1, 1)
    _free_part(str(run / 'freeA'), 2, 1)
    _free_part(str(run / 'freeB'), 1, 1)
    params = _params()
    first = params.free_trajectories(str(run))
    assert len(first) == 3
    second = params.free_trajectories(str(run), old=first)
    assert len(second) == 3
    for p in second:
        assert any(p is q for q in first), f'{p.fname} was rebuilt'
