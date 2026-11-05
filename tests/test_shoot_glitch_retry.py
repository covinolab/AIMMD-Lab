"""A state-definition-glitch retry shoots from the NEW shooting point.

With ``retry_with_state_definition_glitches=True`` the shoot worker handles a
shooting point whose first frame reads back in a state (``... should be in
R``) by deleting ``back*``/``forw*`` and selecting a new shooting point. The
new ``back``/``forw`` files have the same names as the deleted ones, so
everything this process cached about the deleted files -- the open MDAnalysis
readers in ``MDA_CACHE`` and the state/descriptor arrays in ``NPY_CACHE`` --
must go with them; otherwise the retry is judged on the deleted trajectory's
frames and glitches again, forever.

The test runs the real `_shoot` loop on the toy engine: the first selected
frame is in A (a glitch by construction), every later one is in R. The
segment files are written either by the toy engine's own initialisation,
which happens to reopen the segment readers through `MDA_CACHE.load`, or
behind the cache's back, the way GROMACS' mdrun writes them.

- At the default ``trajectory_update_batch_size`` (1000) the retry is judged
  on the deleted segment's ``states`` array that `NPY_CACHE` still holds.
- At batch size 1 the cached one-frame reader of the deleted ``back.xtc`` is
  long enough to be served again, so `MDA_CACHE` must be emptied too. (With
  GROMACS the same happens at any batch size whenever the new file cannot be
  opened yet: `MDA_CACHE.get` then returns the reader it holds.)
"""
import glob
import os

import numpy as np
import pytest
from MDAnalysis import Universe, Writer

import aimmd
from aimmd import MDA_CACHE, NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.pathensemble import PathEnsemble
from aimmd.worker._shoot import WorkerShoot
from aimmd.worker._simulate import WorkerSimulate
from aimmd.worker.utils import evict_segment_caches
from tests._helpers_unit import build_params_file, build_path, write_trajectory


class TinyShootWorker(WorkerShoot, WorkerSimulate):
    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.must_stop = False
        self.termination_timeout = 1.0
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def write_like_mdrun(frame, *deffnms):
    """Write each segment's first frame without touching AIMMD's caches."""
    positions = frame.positions[-1]
    universe = Universe.empty(len(positions), trajectory=True)
    universe.trajectory.ts.positions = positions
    universe.trajectory.ts.dimensions = frame.dimensions[-1]
    for deffnm in deffnms:
        with Writer(f'{deffnm}.xtc', len(positions)) as writer:
            writer.write(universe)


@pytest.mark.parametrize('initialisation, batch_size', [
    ('toy', 1000), ('like_mdrun', 1000), ('like_mdrun', 1)])
def test_glitch_retry_does_not_reuse_the_deleted_trajectory(
        monkeypatch, tmp_path, initialisation, batch_size):
    initial = build_path(
        tmp_path, stem='initial',
        positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                           dtype=np.float32))
    params = aimmd.Params.load(build_params_file(tmp_path, initial.fname),
                               save=False)
    params.update(nbins=1, selection_pool_size=1,
                  trajectory_update_batch_size=batch_size,
                  retry_with_state_definition_glitches=True, save=False)
    if initialisation == 'like_mdrun':
        params.__dict__['initialize_simulation'] = write_like_mdrun
    worker = TinyShootWorker(params, PathEnsemble(initial), tmp_path / 'run')

    selections = []

    def select(pool, params, folder, chain, **kwargs):
        selections.append(len(selections))
        if len(selections) >= 5:     # a loop: give up instead of hanging
            worker.must_stop = True
        # frame 0 is in A (the glitch), frame 1 in R
        return initial[0:1] if len(selections) == 1 else initial[1:2]

    real_register = aimmd.worker._shoot.register_path

    def register_and_stop(path, chain, *args, **kwargs):
        real_register(path, chain, *args, **kwargs)
        worker.must_stop = True

    monkeypatch.setattr('aimmd.worker._shoot.select_shooting_point', select)
    monkeypatch.setattr('aimmd.worker._shoot.register_path', register_and_stop)

    worker._shoot(target_state='R', k=0, sweep=False)

    assert len(selections) == 2, (
        f'{len(selections)} selections: the retry kept re-reading the '
        f'glitched shooting point')
    assert worker.total_steps == 1
    registered = aimmd.Path(str(tmp_path / 'run' / 'chainR0' / 'path000001.xtc'),
                            shooting_index='find')
    assert registered.states[registered.shooting_index] == 'R'


def test_evict_segment_caches_forgets_only_the_folders_segments(tmp_path):
    """The folder's back/forw entries go, in both caches; nothing else does.

    Entries are found from the caches' keys, so a segment array that exists
    only in memory (as the segments' ``states`` arrays do) is evicted too.
    The eviction goes through :func:`aimmd.cache.forget`, which also removes
    the deleted trajectories' MDAnalysis offsets files.
    """
    chain0, chain1 = tmp_path / 'chainR0', tmp_path / 'chainR1'
    chain0.mkdir()
    chain1.mkdir()
    keep, drop = [], []
    for folder, stem in ((chain0, 'back'), (chain0, 'forw'),
                         (chain0, 'path000001'), (chain1, 'back')):
        xtc = write_trajectory(folder, stem=stem)
        save_npy(f'{xtc}.states.npy', np.array(list('RRR')))
        MDA_CACHE.get(xtc)
        NPY_CACHE.get(f'{xtc}.states.npy')
        entries = [xtc, f'{xtc}.states.npy']
        (drop if folder == chain0 and stem != 'path000001' else keep).extend(
            entries)
    in_memory = str(chain0 / 'back.xtc.descriptors.npy')   # not on disk
    NPY_CACHE.put(in_memory, np.zeros((3, 2), dtype=np.float32))
    drop.append(in_memory)
    assert glob.glob(str(chain0 / '.back.xtc_offsets.*'))

    # relative folder: the keys are matched by absolute path
    evicted = evict_segment_caches(os.path.relpath(chain0))

    assert evicted == len(drop)
    resident = set(MDA_CACHE._cache) | set(NPY_CACHE._cache)
    assert not resident & set(drop)
    assert set(keep) <= resident
    assert not glob.glob(str(chain0 / '.back.xtc_offsets.*'))
    assert glob.glob(str(chain1 / '.back.xtc_offsets.*'))
    assert evict_segment_caches(str(chain0)) == 0
