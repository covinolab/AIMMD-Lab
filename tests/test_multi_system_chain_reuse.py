"""The multi-system trainer must offer last round's chains back to shot_chains.

`shot_paths` matches on filename and returns the *existing* Path object when one
is offered via `old=`, so nothing is re-read from disk. The single-system trainer
already did this through `self._shot_chains`; the multi-system trainer used a
loop-local `chains` and passed no `old=`, so every Path was rebuilt from disk on
every reload -- and `must_stop()` runs twice per round.

Rebuilding is expensive by construction: `Path(fname, shooting_index='find')`
resolves to `min_length=inf`, so `MDA_CACHE.get` can never hit
(`len(instance) < inf` is always true) and each construction re-walks the XTC
frame headers and rewrites the offsets sidecar it just deleted. In the LOO
campaign this reached 151 s per `must_stop()` at 26,496 paths, and jobs were
being killed *inside* the reload rather than merely slowed.

Driven in-process against a stub `params`, because `Launcher.run` spawns the
trainer in a subprocess where an in-process monkeypatch cannot reach it.
"""
import numpy as np
import pytest

import aimmd
from aimmd.worker._train import WorkerTrain


class _Net:
    def __call__(self, *a, **k):
        return np.array([[0.0]])

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def train(self, mode=True):
        return self


class _Ensemble:
    """Just enough PathEnsemble surface for one multi-system round."""

    def __init__(self, tag):
        self.tag = tag
        self.n_frames = np.array([1])
        self.fnames = []
        self.weights = np.array([1.0])
        self._paths = []

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(())

    def __add__(self, other):
        return self

    def compute(self, *a, **k):
        return 1

    def subsample(self, *a, **k):
        return self

    def reweight(self, *a, **k):
        return (np.array([1.0]), None, None, None,
                np.array([-1.0, 1.0]), np.array([0.2, 0.8]))

    def project(self, bins, source='values'):
        return np.array([2.0, 1.0], dtype=float)

    def types(self, pattern=None):
        return np.array([True])


class _Worker(WorkerTrain):
    def __init__(self, params, root):
        self.params = params
        self.initial_paths = []
        self._directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """Run one multi-system round, recording every shot_chains call."""
    import os
    for sid in ('s1', 's2'):
        os.makedirs(tmp_path / sid, exist_ok=True)

    calls = []          # (directory, old_object, returned_object)

    def shot_chains(directory, target_state=None, k=None, old=None):
        returned = [_Ensemble(f'{directory}#{len(calls)}')]
        calls.append((directory, old, returned))
        return returned

    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        multi_system=True, multi_system_share_network=True,
        system_ids=['s1', 's2'],
        sorted_states='ARB', chain_type='rfps',
        fit=lambda params, pathensemble, **k: (
            [1.0], [1.0], np.array([0.0]), np.array([1.0]),
            np.array([[1.0, 0.0]])),
        nbins=2, cutoff_min=0.5, cutoff_max=5.0,
        terminal_bin_extension=0.0,
        network_batch_size=4, rescale_committor=False,
        reweight_parameters={}, trajectory_extension='.xtc',
        compute_values_args=(lambda x: np.array([0.0]), 'values', 'positions'),
        compute_descriptors_args=(lambda x: np.array([0.0]), 'descriptors',
                                  'positions'),
        network_save_interval=1, record_bias=False, bias_function=None,
        bias_source='values', subsample_caps=None,
        subsample_caps_of=lambda sid: None,
        bias_reactive_threshold_of=lambda sid: None,
        update_network=lambda directory, timeout=0, raise_if_failure=False: None,
        shot_chains=shot_chains,
        free_trajectories=lambda directory: [],
        network=_Net(),
    )
    for name, value in (
            ('assemble_pathensemble', lambda *a, **k: _Ensemble('assembled')),
            ('compute_bins', lambda *a, **k: np.array([-np.inf, 0.0, np.inf])),
            ('save_npy', lambda fname, arr: None),
            ('replace_in_cache', lambda *a, **k: None)):
        monkeypatch.setattr(f'aimmd.worker._train.{name}', value, raising=False)
    monkeypatch.setattr('aimmd.worker._train.torch.save', lambda s, f: None)
    monkeypatch.setattr('aimmd.worker._train.shutil.copyfile',
                        lambda s, d: None)

    worker = _Worker(params, tmp_path)
    # Deliberately not guarded: if the stub stops completing a round, that
    # should fail loudly rather than skip and silently stop testing the fix.
    worker._train_multi_system(nrounds=1, keep_running=False)
    return calls


def test_reload_offers_the_previous_chains_back(driver):
    per_dir = {}
    for directory, old, returned in driver:
        per_dir.setdefault(directory, []).append((old, returned))

    assert per_dir, 'shot_chains was never called'
    repeated = {d: v for d, v in per_dir.items() if len(v) > 1}
    assert repeated, (
        'no directory was loaded twice; must_stop() should run twice per round, '
        f'saw { {d: len(v) for d, v in per_dir.items()} }')

    for directory, seq in repeated.items():
        for i in range(1, len(seq)):
            old = seq[i][0]
            prev = seq[i - 1][1]
            assert old is prev, (
                f'{directory}: reload {i} was passed {old!r} instead of the '
                f'previous result -- every Path would be rebuilt from disk')


def test_old_is_per_system_not_pooled(driver):
    """A cross-system `old` would make the linear scan O(total^2)."""
    for directory, old, _ in driver:
        if not old:
            continue
        for chain in old:
            tag = getattr(chain, 'tag', '')
            assert tag.startswith(f'{directory}#') or tag == 'assembled', (
                f'{directory} was offered chains belonging to {tag!r}')
