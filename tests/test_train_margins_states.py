"""The trainer's margin frames use the run's state labels, not 'ARB'.

`get_initial_frames_for_training` extracts the initial paths' frames in the
first and last state (the "margins" every fit sees on top of the path
ensemble). Its `states` argument defaults to 'ARB', so a run whose states are
labelled differently must pass its own labels or the extraction finds no
frame in 'A' and raises before the first round.
"""
import numpy as np
import pytest

import aimmd
import aimmd.worker._train as train_mod
from aimmd.worker._train import WorkerTrain
from tests._helpers_unit import TinyNetwork, build_path


STATES = 'CRD'


class TinyTrainWorker(WorkerTrain):
    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self._directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0


class FakeEnsemble:
    """Path-ensemble stand-in with the API `_train` needs; records what fit sees."""

    def __init__(self, fname, n_frames):
        self.weights = np.array([1.0])
        self.fnames = np.array([fname])
        self.n_frames = np.array([n_frames])
        self.added = []

    def __len__(self):
        return 1

    def __add__(self, other):
        self.added.append(other)
        return self

    def compute(self, *args, **kwargs):
        return 0

    def reweight(self, *args, **kwargs):
        return (np.array([1.0]), None, None, None,
                np.array([-1.0, 1.0]), np.array([0.2, 0.8]))

    def project(self, bins, source='values'):
        return np.array([2.0, 1.0])

    def types(self, pattern=None):
        return np.array([True])


def _params_and_initial(tmp_path, **fields):
    initial = build_path(
        tmp_path, stem='initial',
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                           dtype=np.float32),
        states=np.array(list(STATES)))
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(states=STATES, network=TinyNetwork(), **fields)
    assert params.sorted_states == STATES
    return params, aimmd.PathEnsemble(initial)


def test_train_builds_margins_from_the_run_states(monkeypatch, tmp_path):
    ensemble = None
    params, initial = _params_and_initial(
        tmp_path,
        chain_type='rfps',
        fit=lambda params, pathensemble, **kwargs: ([1.0],),
        nbins=2,
        trajectory_extension='.xtc',
        network_save_interval=1,
        compute_values_args=(lambda x: np.array([0.0]), 'values', 'positions'),
        update_network=lambda directory, timeout=0, raise_if_failure=False: None,
        shot_chains=lambda directory, target_state=None, old=None: [ensemble],
        free_trajectories=lambda directory, old=None: [],
    )
    ensemble = FakeEnsemble(initial[0].fname, len(initial[0]))
    worker = TinyTrainWorker(params, initial, tmp_path)
    monkeypatch.setattr(train_mod, 'assemble_pathensemble',
                        lambda *args, **kwargs: ensemble)
    monkeypatch.setattr(train_mod, 'compute_bins',
                        lambda *args, **kwargs: np.array([-np.inf, 0., np.inf]))
    monkeypatch.setattr(train_mod, 'save_npy', lambda *args, **kwargs: None)
    monkeypatch.setattr(train_mod, 'replace_in_cache', lambda *args, **kwargs: None)
    monkeypatch.setattr(train_mod.torch, 'save', lambda *args, **kwargs: None)
    monkeypatch.setattr(train_mod.shutil, 'copyfile', lambda *args, **kwargs: None)

    worker._train(nrounds=1, keep_running=False)

    margins = ensemble.added[0]
    assert sorted(np.concatenate([path.states for path in margins])) == ['C', 'D']


class _Reached(Exception):
    pass


def test_kinetics_convergence_builds_margins_from_the_run_states(monkeypatch, tmp_path):
    params, initial = _params_and_initial(
        tmp_path,
        shot_chains=lambda directory, target_state=None, old=None: [],
        free_trajectories=lambda directory, old=None: [],
    )
    worker = TinyTrainWorker(params, initial, tmp_path)
    built = []
    real = train_mod.get_initial_frames_for_training

    def spy(*args, **kwargs):
        built.append(real(*args, **kwargs))
        raise _Reached   # the rest of the analysis is not under test

    monkeypatch.setattr(train_mod, 'get_initial_frames_for_training', spy)
    with pytest.raises(_Reached):
        worker._kinetics_convergence()

    assert sorted(np.concatenate([path.states for path in built[0]])) == ['C', 'D']
