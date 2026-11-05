"""A shoot worker managing several chains loads each chain from its own folder.

With ``nchains_per_worker > 1`` a shoot worker started with ``k`` manages the
folders ``chain{t}{k}``, ``chain{t}{k + 1}``, ... On a restart into an existing
run directory every one of them must be loaded from its own folder: the next
path of a chain is named ``path{len(chain) + 1:06g}`` inside the folder being
shot, so a chain loaded from the wrong folder either leaves a gap in the
numbering or overwrites an existing path file.
"""
import numpy as np

import aimmd
from aimmd.pathensemble import PathEnsemble
from aimmd.worker._shoot import WorkerShoot
from tests._helpers_unit import build_path


TRANSITION = np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32)


class TinyShootWorker(WorkerShoot):
    """Concrete host for `_shoot` that stops once all chains are loaded."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def _existing_run(tmp_path, lengths, k=0):
    """A run directory whose chains chainR{k}, chainR{k+1}, ... hold `lengths` paths."""
    root = tmp_path / 'run'
    for offset, n in enumerate(lengths):
        folder = root / f'chainR{k + offset}'
        folder.mkdir(parents=True)
        for i in range(n):
            build_path(folder, stem=f'path{i + 1:06g}', positions=TRANSITION)
    return root


def _load_chains(monkeypatch, tmp_path, lengths, k=0):
    """Run `_shoot` up to the end of its chain-loading loop; return the loaded chains."""
    root = _existing_run(tmp_path, lengths, k=k)
    initial = build_path(tmp_path, stem='initial', positions=TRANSITION)
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        chain_type='rfps',
        at_least_one_transition_in_pool=False,
        nbins=1,                       # no network/bins gate
        selection_pool_size=1,
        free_overriding_states='',
        engine='toy',
        trajectory_extension='.xtc',
    )
    worker = TinyShootWorker(params, PathEnsemble(initial), root)

    loaded = []

    def record_and_stop(pool, size, chain, *args, **kwargs):
        loaded.append(chain)
        if len(loaded) == len(lengths):
            worker.must_stop = True    # leave `_shoot` before the main loop

    monkeypatch.setattr('aimmd.worker._shoot.update_selection_pool',
                        record_and_stop)
    worker._shoot(target_state='R', k=k, sweep=False,
                  nchains_per_worker=len(lengths))
    return root, worker, loaded


def test_each_managed_chain_is_loaded_from_its_own_folder(monkeypatch, tmp_path):
    root, worker, loaded = _load_chains(monkeypatch, tmp_path, lengths=[1, 3])

    assert [len(chain) for chain in loaded] == [1, 3]
    for chain_id, chain in enumerate(loaded):
        folders = {fname.rsplit('/', 2)[-2] for fname in chain.fnames}
        assert folders == {f'chainR{chain_id}'}
    assert worker.total_steps == 4


def test_chains_of_a_worker_with_nonzero_k(monkeypatch, tmp_path):
    """The second worker (k=2) manages chainR2 and chainR3, not chainR2 twice."""
    root, worker, loaded = _load_chains(monkeypatch, tmp_path, lengths=[2, 1], k=2)

    assert [len(chain) for chain in loaded] == [2, 1]
    assert all('/chainR3/' in fname for fname in loaded[1].fnames)
