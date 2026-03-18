from pathlib import Path

import numpy as np
import torch
from MDAnalysis import Universe, Writer

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname


DEFAULT_DIMENSIONS = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float32)


def write_trajectory(
    tmp_path,
    stem="traj",
    ext=".xtc",
    positions=None,
    velocities=None,
    times=None,
    dimensions=None,
):
    if positions is None:
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    positions = np.asarray(positions, dtype=np.float32)
    n_frames, n_atoms, _ = positions.shape

    if times is None:
        times = np.arange(n_frames, dtype=float)
    if dimensions is None:
        dimensions = np.repeat(DEFAULT_DIMENSIONS[None, :], n_frames, axis=0)
    dimensions = np.asarray(dimensions, dtype=np.float32)

    fname = Path(tmp_path) / f"{stem}{ext}"
    universe = Universe.empty(n_atoms, trajectory=True)
    with Writer(str(fname), n_atoms=n_atoms) as writer:
        for i in range(n_frames):
            ts = universe.trajectory.ts
            ts.positions = positions[i]
            ts.time = float(times[i])
            ts.dimensions = dimensions[i]
            if velocities is not None:
                ts.velocities = np.asarray(velocities[i], dtype=np.float32)
            writer.write(universe.atoms)
    return str(fname)


def state_labels_from_positions(positions, states="ARB"):
    positions = np.asarray(positions, dtype=float)
    x = positions[:, 0, 0]
    out = np.full(len(x), states[1], dtype="<U1")
    out[x <= -0.5] = states[0]
    out[x >= 0.5] = states[2]
    return out


def simple_states_function(traj, states="ARB"):
    if hasattr(traj, "positions") and np.asarray(traj.positions).ndim == 2:
        positions = np.asarray(traj.positions, dtype=float)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in traj], dtype=float)
    return state_labels_from_positions(positions, states=states)


def simple_descriptors_function(traj):
    if hasattr(traj, "positions") and np.asarray(traj.positions).ndim == 2:
        positions = np.asarray(traj.positions, dtype=float)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in traj], dtype=float)
    return positions[:, :, 0]


class TinyNetwork(aimmd.network.Rescalable):
    def __init__(self):
        super().__init__(max_knots=8)
        self.linear = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight[:] = torch.tensor([[1.0, -0.5]])

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.linear(x)


def build_path(
    tmp_path,
    stem="traj",
    ext=".xtc",
    positions=None,
    states=None,
    values=None,
    descriptors=None,
    shooting_index=1,
    weight=1.0,
    exclude_from=-1,
):
    fname = write_trajectory(tmp_path, stem=stem, ext=ext, positions=positions)
    path = aimmd.Path(fname, shooting_index=shooting_index, weight=weight, exclude_from=exclude_from)
    positions = path.positions
    if states is None:
        states = state_labels_from_positions(positions)
    if values is None:
        values = positions[:, 0, 0].astype(float)
    if descriptors is None:
        descriptors = positions[:, :, 0].astype(float)
    save_npy(get_cache_fname(fname, "states"), np.asarray(states, dtype="<U1"))
    save_npy(get_cache_fname(fname, "values"), np.asarray(values, dtype=float))
    save_npy(get_cache_fname(fname, "descriptors"), np.asarray(descriptors, dtype=float))
    return aimmd.Path(fname, shooting_index=shooting_index, weight=weight, exclude_from=exclude_from)


def build_params_file(tmp_path, initial_fname):
    path = Path(tmp_path) / "params.py"
    source = f"""
import numpy as np
from pathlib import Path
import aimmd

def states_function(traj):
    if hasattr(traj, 'positions') and np.asarray(traj.positions).ndim == 2:
        positions = np.asarray(traj.positions, dtype=float)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in traj], dtype=float)
    x = positions[:, 0, 0]
    out = np.full(len(x), 'R', dtype='<U1')
    out[x <= -0.5] = 'A'
    out[x >= 0.5] = 'B'
    return out

def descriptors_function(traj):
    if hasattr(traj, 'positions') and np.asarray(traj.positions).ndim == 2:
        positions = np.asarray(traj.positions, dtype=float)[None]
    else:
        positions = np.array([ts.positions.copy() for ts in traj], dtype=float)
    return positions[:, :, 0]

def toy_mdrun(ts):
    ts.positions[:] = ts.positions + 0.1
    ts.time = ts.time + 1.0

initial_paths = ['{Path(initial_fname).name}']
engine = 'toy'
toy_slowdown = 0.0
network = aimmd.network.utils.placeholder
trajectory_extension = '{Path(initial_fname).suffix}'
topology = '{Path(initial_fname).name}'
"""
    path.write_text(source)
    return path
