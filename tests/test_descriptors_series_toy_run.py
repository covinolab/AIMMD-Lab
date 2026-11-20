"""A toy AIMMD run with a named descriptors series, end to end.

Train, free simulation, shooting and training again with
``descriptors_series = 'descriptors-toy'``; then a resume next to stale
``*.descriptors.npy`` files (as left by a run that changed series), which must
never be opened; then a rollback to the historical 'descriptors' series, which
refills that series and does not crash.

Every trajectory of the run gets complete rows in the series of its phase,
equal to the descriptors computed from its own frames.
"""
import os
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import pytest
from MDAnalysis import Universe

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import forbid_opening

SERIES = "descriptors-toy"
TOY_1D = Path(__file__).parent / "toy_1d"
WALLTIME = 120

PARAMS_SOURCE = '''
import numpy as np
import torch
from aimmd.network import fit as _fit
from aimmd.network.rescalable import Rescalable

engine = 'toy'
toy_slowdown = 0.0
initial_paths = 'initial.xtc'
free_overriding_states = 'all'
descriptors_series = 'descriptors-toy'


def toy_mdrun(ts):
    for _ in range(100):
        ts.positions = (ts.positions + .02 * np.random.normal()) % 10


def states_function(trajectory):
    x = np.array([frame.positions[0, 0] for frame in trajectory])
    result = np.full(len(x), 'S', dtype='<U1')
    result[(x < 1) | (x > 9)] = 'A'
    result[(x >= 1) & (x < 2)] = 'R'
    result[(x >= 2) & (x < 3)] = 'B'
    return result


def descriptors_function(trajectory):
    return np.array([frame.positions[0, :1].copy() for frame in trajectory],
                    dtype=np.float32)


class Network(Rescalable):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(1, 16)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(16, 1)
        self.reset_parameters()
    def forward(self, x):
        return self.output(self.activation(self.input(x[:, :1])))
    def reset_parameters(self):
        self.input.reset_parameters()
        self.output.reset_parameters()

network = Network()


def fit(params, pathensemble, verbose=False, worker=None):
    return _fit(params, pathensemble, nbins=0, cutoff_min=0.5, cutoff_max=20.,
                state_bins='all', augment='no', lr=1e-3,
                loss_bayesian_factor=0, loss_smoothening_weight=0,
                loss_regularization_weight=0, epochs=30, batch_size=4096,
                stop=50., train_validation_early_stopping=False,
                in_memory=False, graphs=False, verbose=False, worker=worker)
'''


def _trajectories(run):
    """Every trajectory of the run with a states series."""
    return sorted(name[:-len(".states.npy")] for name in
                  glob(f"{run}/**/*.states.npy", recursive=True))


def _featurized(trajectory):
    universe = Universe(trajectory)
    return np.array([ts.positions[0, :1].copy() for ts in universe.trajectory],
                    dtype=np.float32)


def _check_series(run, series):
    """Each trajectory has complete rows in `series`, equal to its frames.

    (Up to one float32 ulp: an xtc of at most 9 atoms stores uncompressed
    nm floats, so the rewrite of a path at registration can move the last
    bit of the Angstrom values its rows were computed from.)
    """
    trajectories = _trajectories(run)
    assert trajectories
    for trajectory in trajectories:
        rows = np.load(get_cache_fname(trajectory, series))
        expected = _featurized(trajectory)
        assert rows.shape == expected.shape, trajectory
        assert rows.any(axis=1).all(), trajectory
        np.testing.assert_allclose(rows, expected, rtol=1e-6, atol=0,
                                   err_msg=trajectory)
    return trajectories


def _plant_stale_descriptors(trajectories):
    """Legacy rows next to every trajectory (written outside the guard's
    name, then renamed into place)."""
    planted = []
    for trajectory in trajectories:
        fname = get_cache_fname(trajectory, "descriptors")
        temp = f"{trajectory}.planted.npy"
        save_npy(temp, _featurized(trajectory))
        os.replace(temp, fname)
        planted.append(fname)
    return planted


def _legacy_files(folder):
    return sorted(glob(f"{folder}/**/*.descriptors.npy", recursive=True))


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    shutil.copy(TOY_1D / "initial.xtc", tmp_path / "initial.xtc")
    (tmp_path / "params.py").write_text(PARAMS_SOURCE)
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    yield tmp_path
    NPY_CACHE.clear()


def test_toy_run_with_named_series_resume_and_rollback(toy_run):
    run = str(toy_run / "run1")

    # --- a run with the named series ----------------------------------------
    with pytest.MonkeyPatch.context() as guard:
        opened = forbid_opening(guard, ".descriptors.npy")

        params = aimmd.Params.load("params.py")
        assert params.descriptors_series == SERIES
        aimmd.Worker(params, "run1", walltime=WALLTIME).train(nrounds=1)
        aimmd.Worker(params, "run1", nframes=60,
                     walltime=WALLTIME).free(0, 0, 1)
        aimmd.Worker(params, "run1", nsteps=3, walltime=WALLTIME).shoot(1, 0)
        aimmd.Worker(params, "run1", walltime=WALLTIME).train(nrounds=1)

        assert opened == []
    assert _legacy_files(toy_run) == []
    assert os.path.exists(f"{run}/networkARB.h5")
    trajectories = _check_series(run, SERIES)
    assert any("/chainR0/path" in name for name in trajectories)
    assert any("/freeA/traj" in name for name in trajectories)
    assert any("/initialARB/" in name for name in trajectories)
    n_paths = len(glob(f"{run}/chainR0/path*.xtc"))
    assert n_paths >= 3

    # --- resume next to stale legacy files -----------------------------------
    planted = _plant_stale_descriptors(trajectories + [str(toy_run / "initial.xtc")])
    with pytest.MonkeyPatch.context() as guard:
        opened = forbid_opening(guard, ".descriptors.npy")

        params = aimmd.Params.load("params.py")
        aimmd.Worker(params, "run1", nsteps=6, walltime=WALLTIME).shoot(1, 0)
        aimmd.Worker(params, "run1", nframes=120,
                     walltime=WALLTIME).free(0, 0, 1)
        aimmd.Worker(params, "run1", walltime=WALLTIME).train(nrounds=1)

        assert opened == []
    assert _legacy_files(toy_run) == sorted(planted)
    assert len(glob(f"{run}/chainR0/path*.xtc")) > n_paths
    trajectories = _check_series(run, SERIES)

    # --- rollback to the historical series -----------------------------------
    params = aimmd.Params.load("params.py", descriptors_series="descriptors")
    assert params.descriptors_series == "descriptors"
    assert params.compute_values_args[2] == "descriptors"
    n_paths = len(glob(f"{run}/chainR0/path*.xtc"))
    aimmd.Worker(params, "run1", nsteps=n_paths + 2,
                 walltime=WALLTIME).shoot(1, 0)
    aimmd.Worker(params, "run1", walltime=WALLTIME).train(nrounds=1)

    # every trajectory of the ensemble has its legacy rows again
    _check_series(run, "descriptors")
    values = glob(f"{run}/**/*.values.npy", recursive=True)
    assert values
