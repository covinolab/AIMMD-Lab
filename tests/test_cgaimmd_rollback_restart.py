"""A C-GAIMMD chain restarted mid-shot after a checkpoint rollback, with its in-flight record.

A C-GAIMMD shot writes its in-flight record (``.cgaimmd_shooting_point.json``
and ``.npy``) before its segments exist, so that a restart can attach the
record to the path it registers. When the job dies and the next one resumes
``mdrun`` from a checkpoint older than the last frame AIMMD read,
``aimmd.worker.rollback`` first cuts the half and every per-frame cache
(``<trajectory>.<series>.npy``) back to the checkpoint. The record is not a
per-frame cache of a trajectory, so it must survive the rollback, and the
registered shot must carry it.

No MD runs: ``gmx mdrun`` is the ``FakeGromacs`` of
``tests/test_restart_rollback.py`` (it does to the files what GROMACS does,
here appending), and ``gmx dump -cp`` is replaced by a checkpoint object.
Everything else is AIMMD's own code: the C-GAIMMD branch of
``WorkerShoot._shoot`` (resume with the record, registration, sidecar),
``WorkerSimulate._simulate`` with the rollback, ``register_path`` and the
params' pipeline. The toy has 12 atoms; its CV is x of atom 0 (A below 20,
B above 40); the shooting point has x = 30 (R).
"""

import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd import MDA_CACHE, NPY_CACHE
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.sidecar import (in_flight_fname, in_flight_positions_fname,
                                   read_shot_sidecar, write_in_flight)
from aimmd.cgaimmd import source as cgaimmd_source
from aimmd.path.utils import get_cache_fname
from aimmd.worker import _shoot, rollback
from aimmd.worker._shoot import WorkerShoot
from aimmd.worker._simulate import WorkerSimulate
from tests._cgaimmd_helpers import ScriptedShootingPointSource, cgaimmd_params_namespace
from tests.test_restart_rollback import (DT, N_ATOMS, FakeGromacs, Session, checkpoint_after,
                                         on_disk, write_frames, x_run1, x_run2)


def label(x):
    return 'A' if x < 20 else ('B' if x > 40 else 'R')


def states_function(trajectory):
    return np.array([label(float(ts.positions[0, 0])) for ts in trajectory], dtype='<U1')


def descriptors_function(trajectory):
    """The flat coordinates (copied: a reader reuses its buffer for the next frame)."""
    return np.array([ts.positions.ravel().copy() for ts in trajectory], dtype=np.float32)


class ShootWorker(WorkerShoot, WorkerSimulate):
    """The shoot worker's C-GAIMMD branch with the real simulation loop (as Worker composes them)."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = self._directory = str(root)
        self.must_stop = False
        self.termination_timeout = 1.0
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def new_process():
    MDA_CACHE.clear()
    NPY_CACHE.clear()


@pytest.fixture
def shot(tmp_path, monkeypatch):
    """A C-GAIMMD run whose first shot was in flight when its job died (see the module notes)."""
    run = tmp_path / 'run'
    folder = run / 'chainR0'
    folder.mkdir(parents=True)
    sources = tmp_path / 'sources'
    sources.mkdir()
    initial = write_frames(tmp_path / 'initial.xtc', [10.0, 12.0, 11.0], 0)
    params = cgaimmd_params_namespace(
        engine='gromacs', gmx_mdrun='gmx mdrun', gmx_grompp='gmx grompp -maxwarn 1',
        gmx_eneconv=None, max_length=np.inf, trajectory_update_batch_size=1000, nbins=1,
        network=TinyFlowNetwork(n_atoms=N_ATOMS), states_function=states_function,
        descriptors_function=descriptors_function, record_bias=False,
        free_trajectories=lambda directory, old=None: [],
        retry_with_state_definition_glitches=False)
    # the shot as _shoot left it: its record, its two .tpr files, back.xtc with 100 frames of
    # realisation 1 (all in R) and their caches; forw not started
    run1 = write_frames(sources / 'run1.xtc', [x_run1(i) for i in range(100)], 0)
    shooting_point = on_disk(run1)
    positions = np.asarray(aimmd.Path(run1).positions[0], dtype=np.float64)
    record = {'schema': 'aimmd.cgaimmd.shot/1', 'source': 'cgaimmd', 'path_name': 'path000001',
              'chain': 'chainR0', 'lambda_target': 0.5, 'note': 'before the job died'}
    write_in_flight(str(folder), record, positions)
    for half in ('back', 'forw'):
        (folder / f'{half}.tpr').write_text('tpr')
        (folder / f'{half}.log').write_text('x' * 4000)
    gromacs = FakeGromacs(Session(run1))
    monkeypatch.setattr('aimmd.params._methods.execute_command', gromacs)
    new_process()
    result = ShootWorker(params, aimmd.PathEnsemble(aimmd.Path(initial)), run)._simulate(
        str(folder / 'back'), aimmd.Path(), 'R', 'shoot')
    assert result[0] is None                                # the job died before B
    assert len(np.load(f'{folder}/back.xtc.states.npy')) == 100
    new_process()
    return SimpleNamespace(run=run, folder=folder, params=params, initial=initial,
                           sources=sources, positions=positions, record=record,
                           shooting_point=shooting_point)


def test_restart_after_a_rollback_keeps_the_in_flight_record(shot, monkeypatch, capsys):
    folder, sources = shot.folder, shot.sources
    back_xtc = f'{folder}/back.xtc'
    # the next job resumes back from a checkpoint written after frame 29: realisation 2 climbs
    # into B from there; forw (no checkpoint) runs into A
    checkpoint = checkpoint_after(back_xtc, 30)
    open(checkpoint.fname, 'w').close()

    def read_checkpoint(fname, gmx):
        assert fname.endswith('back.cpt'), fname
        return checkpoint

    monkeypatch.setattr(rollback, 'read_checkpoint', read_checkpoint)
    run2 = write_frames(sources / 'run2.xtc', [x_run2(i, 30) for i in range(30, 130)], 30, seed=30)
    forw = write_frames(sources / 'forw.xtc',
                        [30.0 - 1.2 * i for i in range(40)], 0, seed=1)
    gromacs = FakeGromacs(Session(run2, cut=checkpoint.output(back_xtc).offset), Session(forw))
    monkeypatch.setattr('aimmd.params._methods.execute_command', gromacs)
    markers = []
    real_rollback = rollback.rollback_to_checkpoint

    def spy_rollback(deffnm, *args, **kwargs):
        out = real_rollback(deffnm, *args, **kwargs)
        markers.append((os.path.isfile(in_flight_fname(str(folder))),
                        os.path.isfile(in_flight_positions_fname(str(folder)))))
        return out

    monkeypatch.setattr('aimmd.worker._simulate.rollback_to_checkpoint', spy_rollback)
    sources_made = []

    def make_cgaimmd_source(params, worker, directory):
        source = ScriptedShootingPointSource(params, worker, directory, points=())
        sources_made.append(source)
        return source

    monkeypatch.setattr(cgaimmd_source, 'make_cgaimmd_source', make_cgaimmd_source)
    worker = ShootWorker(shot.params, aimmd.PathEnsemble(aimmd.Path(shot.initial)), shot.run)
    real_register = _shoot.register_path

    def register_path(*args, **kwargs):
        real_register(*args, **kwargs)
        worker.must_stop = True                             # one shot

    monkeypatch.setattr(_shoot, 'register_path', register_path)
    capsys.readouterr()
    worker._shoot(target_state='R', k=0, sweep=False)
    out = capsys.readouterr().out

    # the rollback ran on back (and only there), and the record survived it
    entry = [line for line in out.splitlines() if line.startswith('!!! CHECKPOINT ROLLBACK')]
    assert len(entry) == 1 and 'back.xtc cut from 100 to 30 frames' in entry[0], entry
    assert markers == [(True, True)]
    # the shot resumed with its record: no new candidate, the record in the sidecar
    assert sources_made[0].called('next_shooting_point') == []
    fname = f'{folder}/path000001.xtc'
    sidecar = read_shot_sidecar(fname)
    assert sidecar['source'] == 'cgaimmd' and sidecar['path_name'] == 'path000001'
    assert sidecar['note'] == 'before the job died' and sidecar['lambda_target'] == 0.5
    assert sidecar['lambda_registered'] is not None
    assert not os.path.exists(in_flight_fname(str(folder)))
    assert not os.path.exists(in_flight_positions_fname(str(folder)))
    # the registered path: its shooting frame is the record's configuration, it runs from B
    # (its back half, reversed, stopped at its first B frame of realisation 2) to A, and every
    # cached row is the row of the frame now at its index (no row of the lost realisation)
    path = aimmd.Path(fname, shooting_index=sidecar['registration']['shooting_index'])
    frame = np.asarray(path.positions[path.shooting_index], dtype=float)
    assert np.abs(frame - shot.positions).max() <= 0.01
    disk = on_disk(fname)
    states = np.load(get_cache_fname(fname, 'states'))
    np.testing.assert_array_equal(states, disk['states'])
    assert states[0] == 'B' and states[-1] == 'A' and (states[1:-1] == 'R').all()
    rows = np.load(get_cache_fname(fname, 'descriptors'))
    np.testing.assert_allclose(rows, disk['descriptors'] - 100.0, atol=1e-3)
    assert sidecar['registration']['n_frames'] == len(disk['times'])
    assert not glob.glob(f'{folder}/*.flow.json')
