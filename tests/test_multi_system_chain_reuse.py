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


class _Run(list):
    """The recorded shot_chains calls, plus the trainer's captured output.

    A list subclass so existing tests can keep iterating it directly, while the
    logging tests can reach `.out` -- capsys in a test body does not see output
    produced during fixture setup.
    """
    out = ''


@pytest.fixture
def driver(tmp_path, monkeypatch, capsys):
    """Run one multi-system round, recording every shot_chains call."""
    import os
    for sid in ('s1', 's2'):
        os.makedirs(tmp_path / sid, exist_ok=True)

    calls = []          # (directory, old_object, returned_object)

    def shot_chains(directory, target_state=None, k=None, old=None):
        returned = [_Ensemble(f'{directory}#{len(calls)}')]
        calls.append((directory, old, returned))
        return returned

    free_calls = []

    def free_trajectories(directory, target_state=None, old=None):
        returned = []
        free_calls.append((directory, old))
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
        # compute_descriptors_args is a PROPERTY derived from this, so it
        # must be set for the trainer's descriptor sweep to run at all
        descriptors_function=lambda trajectory, system_id=None: np.zeros((1, 1)),
        network_save_interval=1, record_bias=False, bias_function=None,
        bias_source='values', subsample_caps=None,
        subsample_caps_of=lambda sid: None,
        bias_reactive_threshold_of=lambda sid: None,
        update_network=lambda directory, timeout=0, raise_if_failure=False: None,
        shot_chains=shot_chains,
        free_trajectories=free_trajectories,
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

    flush_calls = []
    monkeypatch.setattr('aimmd.worker._train._flush_graph_backlog',
                        lambda: flush_calls.append(1))

    worker = _Worker(params, tmp_path)
    # Deliberately not guarded: if the stub stops completing a round, that
    # should fail loudly rather than skip and silently stop testing the fix.
    worker._train_multi_system(nrounds=1, keep_running=False)
    run = _Run(calls)
    run.free_calls = free_calls
    run.flush_calls = flush_calls
    run.out = capsys.readouterr().out
    print(run.out)              # keep it visible on failure
    return run


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


def test_trainer_reports_each_phase(driver):
    """Every phase of a round must be attributable from the log alone.

    Diagnosing the production stalls repeatedly failed because this loop was
    almost silent: a stall could not be pinned to a system or a step, the
    post-training pass had no instrumentation at all, and the fit's own cache
    behaviour was invisible. All of those cost real GPU-hours to re-derive.
    """
    out = driver.out

    # the load, split so we can see chains-vs-free-trajectories
    assert 'Loading current path ensembles' in out
    assert 'shot path(s) in' in out and 'free trajectory(ies) in' in out
    assert 'Path ensembles loaded in' in out
    # tmpfs headroom, so the ceiling is seen approaching
    assert 'replica(s) staged' in out and 'budget' in out
    # the pre-training value pass, per system and per step
    assert 'Value pass over' in out
    assert 'descriptors:' in out and 'value pass:' in out
    # graph-cache counters attached to the phases
    assert 'graph cache hit=' in out
    # the fit's OWN cache behaviour -- previously invisible, which is why we
    # still cannot explain 619 ms/epoch in production
    assert 'training completed' in out
    completed = [l for l in out.splitlines() if 'training completed' in l]
    assert any('graph cache hit=' in l for l in completed), (
        f'no cache counters on the fit line: {completed}')
    # the post-training pass, which had no instrumentation at all before
    assert 'Post-training value pass over' in out
    assert 'Post-training value pass complete in' in out


def test_trainer_reports_are_per_system(driver):
    """A stall must be attributable to one system, not just to 'the value pass'."""
    out = driver.out
    for sid in ('s1', 's2'):
        assert f"[system '{sid}']" in out, f'no per-system line for {sid}'


def test_free_trajectories_are_offered_back(driver):
    """The trainer must thread `old=` into free_trajectories too.

    This was the one remaining un-reused part of the ensemble load, and the
    load was 46 min of a ~4 h production round. Reuse inside
    free_trajectories is conditional (a free trajectory grows), so offering a
    stale list is safe -- but the trainer has to offer it at all.
    """
    per_dir = {}
    for directory, old in driver.free_calls:
        per_dir.setdefault(directory, []).append(old)

    assert per_dir, 'free_trajectories was never called'
    repeated = {d: v for d, v in per_dir.items() if len(v) > 1}
    assert repeated, f'no directory reloaded; saw { {d: len(v) for d, v in per_dir.items()} }'
    for directory, olds in repeated.items():
        for i, old in enumerate(olds[1:], start=1):
            assert old is not None, (
                f'{directory}: reload {i} passed no old= to free_trajectories')


def test_trainer_writes_its_computed_graphs_back_each_round(driver):
    """Both value passes must hand the round's new graphs to the shared cache.

    The trainer computes graphs the MD writers have not reached yet. Keeping
    them only in its memo and its tmpfs replica means they are recomputed every
    round forever, because both are rebuilt from the real database at the top of
    the next round -- and each recompute is a read that stops that database's
    WAL from resetting. Two sites must fire per round: after the value pass and
    after the post-training value pass.
    """
    assert len(driver.flush_calls) >= 2, (
        f'expected the backlog to be flushed after both value passes, saw '
        f'{len(driver.flush_calls)}')


# ------------------------------------ the backlog flush helper in isolation --
def test_flush_helper_is_a_no_op_without_the_graph_stack(monkeypatch):
    """A run with no graph cache must not pay to import the GNN stack.

    `_flush_graph_backlog` runs up to four times per round. Importing
    `graph_utils` there drags in torch_geometric/mlcolvar/mdtraj even for a toy
    1-D run that never stores a graph -- which pushed `test_toy_1d` past its
    training-time guard. If `graph_utils` was never imported, nothing was ever
    stored, so there is nothing to flush.
    """
    import sys
    from aimmd.worker import _train

    monkeypatch.delitem(sys.modules, 'aimmd.network.graph_utils', raising=False)
    called = []
    monkeypatch.setattr('builtins.__import__',
                        lambda *a, **k: called.append(a[:1]) or (_ for _ in ()).throw(
                            AssertionError('must not import anything')))
    _train._flush_graph_backlog()          # must simply return
    assert called == []


def test_flush_helper_reports_what_was_written(monkeypatch, capsys):
    """When the cache is in play, the round's write-back is attributable."""
    import sys, types
    from aimmd.worker import _train

    stub = types.ModuleType('aimmd.network.graph_utils')
    stub.flush_pending_writes = lambda: {'/nowhere/graphs_cache_G4.sqlite': 25086}
    monkeypatch.setitem(sys.modules, 'aimmd.network.graph_utils', stub)

    _train._flush_graph_backlog()

    out = capsys.readouterr().out
    assert 'graphs_cache_G4.sqlite' in out
    assert '25,086' in out


def test_flush_helper_never_raises(monkeypatch):
    """A cache write must never propagate into the training loop."""
    import sys, types
    from aimmd.worker import _train

    stub = types.ModuleType('aimmd.network.graph_utils')

    def boom():
        raise RuntimeError('cache exploded')

    stub.flush_pending_writes = boom
    monkeypatch.setitem(sys.modules, 'aimmd.network.graph_utils', stub)
    _train._flush_graph_backlog()          # must not raise
