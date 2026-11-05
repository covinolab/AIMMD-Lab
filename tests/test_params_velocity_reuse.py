"""`gen_temperature < 0` reuses the shooting point's velocities.

The field description promises: "If < 0: reuse velocities from the parent
trajectory (requires `trajectory_extension == '.trr'` so velocities are
available)". `Params.initialize_simulation` flips the sign before every
branch after the first, so a single ``deffnm`` (a free simulation) continues
with the frame's velocities, and a two-way shot (``back`` first, then
``forw``) starts ``back`` with them and ``forw`` with them reversed.

A frame without velocities (e.g. from an ``.xtc``) falls back to zero
velocities, with a warning.
"""
import numpy as np
from MDAnalysis.coordinates.core import reader as Reader

import aimmd
from tests._helpers_unit import build_params_file, write_trajectory


POSITIONS = np.array([[[-1.0, 0, 0], [0, 1, 0]],
                      [[0.0, 0, 0], [0, 2, 0]],
                      [[1.0, 0, 0], [0, 3, 0]]], dtype=np.float32)
VELOCITIES = np.array([[[1.0, 2, 3], [4, 5, 6]],
                       [[7.0, 8, 9], [-1, -2, -3]],
                       [[0.5, 0, 0], [0, 0, 0.5]]], dtype=np.float32)


def _params(tmp_path, ext, velocities):
    fname = write_trajectory(tmp_path, stem='parent', ext=ext,
                             positions=POSITIONS, velocities=velocities)
    params = aimmd.Params.load(build_params_file(tmp_path, fname), save=False)
    params.update(gen_temperature=-1.0, save=False)
    return params, aimmd.Path(fname)


def _first_frame_velocities(fname):
    return np.array(Reader(fname)[0].velocities)


def test_negative_gen_temperature_reuses_the_frame_velocities(tmp_path):
    params, parent = _params(tmp_path, '.trr', VELOCITIES)
    back, forw = str(tmp_path / 'back'), str(tmp_path / 'forw')

    params.initialize_simulation(parent[1:2], back, forw)

    np.testing.assert_allclose(_first_frame_velocities(f'{back}.trr'),
                               VELOCITIES[1])
    np.testing.assert_allclose(_first_frame_velocities(f'{forw}.trr'),
                               -VELOCITIES[1])
    # the parent's reader must not have been flipped in place
    np.testing.assert_allclose(parent[1].velocities, VELOCITIES[1])


def test_negative_gen_temperature_single_branch_continues(tmp_path):
    """One `deffnm` from a Timestep, as the free worker and check_engine use."""
    params, parent = _params(tmp_path, '.trr', VELOCITIES)
    seed = str(tmp_path / 'seed')

    params.initialize_simulation(parent[2], seed)

    np.testing.assert_allclose(_first_frame_velocities(f'{seed}.trr'),
                               VELOCITIES[2])


def test_negative_gen_temperature_without_velocities_warns(tmp_path, capsys):
    params, parent = _params(tmp_path, '.xtc', None)
    back, forw = str(tmp_path / 'back'), str(tmp_path / 'forw')

    params.initialize_simulation(parent[1:2], back, forw)

    assert 'gen_temperature' in capsys.readouterr().out
    assert Reader(f'{forw}.xtc')[0].positions.shape == (2, 3)
