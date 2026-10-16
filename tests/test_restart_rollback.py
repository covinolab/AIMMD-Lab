"""
Rolling trajectories and their per-frame caches back to the restart checkpoint.

When GROMACS resumes from a checkpoint older than the last frame AIMMD has read
(a job killed before it wrote its last checkpoint, or a checkpoint replaced by
an older copy), it simulates the frames after the checkpoint again under the
same frame indices. AIMMD takes cached row i for frame i, so before the
rollback it kept the rows of the lost frames: a half stopped at its old cached
length + 1 instead of at its first frame in a state, and free runs were spliced
from two realisations.

No MD runs here. ``gmx mdrun`` is replaced, inside
``aimmd.params._methods.execute_command``, by :class:`FakeGromacs`, which does
to the files what GROMACS does: appending, it refuses a file shorter than the
checkpoint's length, truncates the file to it and appends whole frames in
bursts; with ``-noappend`` it writes part ``s + 1``. ``gmx dump -cp`` is
replaced by a checkpoint object (``aimmd.worker.rollback.read_checkpoint`` is
monkeypatched). Everything else is AIMMD's own code:
``WorkerSimulate._simulate`` and its stop condition, ``Path.extend`` with the
worker's pipeline, the compute ledger, the real ``Params.run_simulation``, and
the rollback.

The toy: 12 atoms (compressed XTC frames of varying size, as in production),
the CV is x of atom 0, A below 20, B above 40, one frame every 10 ps.
"""

import glob
import hashlib
import os
from dataclasses import dataclass
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.XTC import XTCReader
from MDAnalysis.lib.formats.libmdaxdr import XTCFile

from aimmd import MDA_CACHE, NPY_CACHE
from aimmd.cache.npy import save_npy, update_npy
from aimmd.engines.gromacs import Checkpoint, CheckpointError, CheckpointOutput
from aimmd.params._methods import ParamsMethods
from aimmd.path import Path
from aimmd.worker import rollback
from aimmd.worker._simulate import WorkerSimulate

N_ATOMS = 12
DT = 10.0
BOX = np.array([60, 60, 60, 90, 90, 90], dtype=np.float32)
SERIES = 'descriptors-toy'          # a 2-D descriptor series (node tables)


# ---------------------------------------------------------------------------
# the toy system and what the params file gives the worker
# ---------------------------------------------------------------------------

def label(x):
    return 'A' if x < 20 else ('B' if x > 40 else 'R')


def states_function(trajectory):
    return np.array([label(float(ts.positions[0, 0])) for ts in trajectory],
                    dtype='<U1')


def descriptors_function(trajectory):
    """Every coordinate of the frame, shifted so that no row is all zero."""
    return np.array([ts.positions.ravel() + 100.0 for ts in trajectory],
                    dtype=np.float32)


def cv_function(trajectory):
    return np.array([float(ts.positions[0, 0]) for ts in trajectory])


PIPELINE = [(descriptors_function, SERIES), (cv_function, 'cv'),
            (states_function, 'states'), (None, 'values')]


def x_run1(i):
    """Realisation 1 of a shooting half: oscillates in R (25 to 35)."""
    return 30.0 + 5.0 * np.sin(i / 7.0)


def x_run2(i, restart):
    """Realisation 2, simulated again from frame `restart` on: climbs into B."""
    if i < restart:
        return x_run1(i)
    return min(x_run1(restart) + 1.2 * (i - restart), 45.0 + np.sin(i))


def x_free(i):
    """A free run inside A."""
    return 15.0 + 2.0 * np.sin(i / 5.0)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def write_frames(fname, xs, first, seed=0):
    """Frames `first`, `first + 1`, ... (t = index * DT) with x of atom 0 = xs."""
    universe = mda.Universe.empty(N_ATOMS, trajectory=True)
    rng = np.random.default_rng(seed)
    with mda.Writer(str(fname), N_ATOMS) as writer:
        for k, x in enumerate(xs):
            positions = (30.0 + rng.normal(size=(N_ATOMS, 3))).astype(np.float32)
            positions[0] = (x, 30.0, 30.0)
            universe.trajectory.ts.positions = positions
            universe.trajectory.ts.dimensions = BOX
            universe.trajectory.ts.time = (first + k) * DT
            writer.write(universe.atoms)
    return str(fname)


def frame_bytes(fname):
    """Byte offset of every frame of `fname`, then its size."""
    with XTCFile(str(fname)) as file:
        return [int(offset) for offset in file.offsets] + [os.path.getsize(fname)]


def on_disk(fname):
    """Times, labels, descriptor rows and CV of the frames now in `fname`."""
    reader = XTCReader(str(fname), refresh_offsets=True)
    try:
        positions = np.array([ts.positions.copy() for ts in reader])
        times = np.array([ts.time for ts in reader])
    finally:
        reader.close()
    cv = positions[:, 0, 0].astype(float)
    return dict(times=times, states=np.array([label(x) for x in cv]),
                descriptors=positions.reshape(len(positions), -1) + 100.0, cv=cv)


def assert_caches_match(xtc, nrows=None):
    """Every cached row of `xtc` is the row of the frame now at its index."""
    disk = on_disk(xtc)
    n = len(disk['times']) if nrows is None else nrows
    states = np.load(f'{xtc}.states.npy')
    descriptors = np.load(f'{xtc}.{SERIES}.npy')
    cv = np.load(f'{xtc}.cv.npy')
    assert len(states) == len(descriptors) == len(cv) == n
    np.testing.assert_array_equal(states, disk['states'][:n])
    np.testing.assert_allclose(descriptors, disk['descriptors'][:n], atol=1e-4)
    np.testing.assert_allclose(cv, disk['cv'][:n], atol=1e-4)


def rows(fname):
    """Number of rows of a .npy file."""
    return len(np.load(fname))


def state(xtc):
    """Size of a trajectory, rows of its caches, and the md5 of all of them."""
    caches = sorted(glob.glob(glob.escape(xtc) + '.*.npy'))
    md5 = {os.path.basename(f): hashlib.md5(open(f, 'rb').read()).hexdigest()
           for f in [xtc] + caches}
    rows = {f[len(xtc) + 1:-len('.npy')]: len(np.load(f)) for f in caches}
    return dict(size=os.path.getsize(xtc), rows=rows, md5=md5)


# ---------------------------------------------------------------------------
# GROMACS and PLUMED, emulated on the files
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """What one `gmx mdrun` call writes."""
    source: str             # an xtc holding the frames this session writes
    cut: int = 0            # appending: the checkpoint's length of the file
    part: int = None        # -noappend: the part number this session writes
    burst: int = 10         # frames written between two polls of the worker


class FakeGromacs:
    """Stands in for execute_command in aimmd.params._methods."""

    def __init__(self, *sessions):
        self.sessions = list(sessions)
        self.commands = []
        self.seen = []      # the trajectory's files when mdrun started

    def __call__(self, command, stop_condition=lambda: False, **kwargs):
        self.commands.append(command)
        session = self.sessions.pop(0)
        folder = command.split('&&')[0].strip()[len('cd '):].strip()
        base = command.split('-deffnm ')[1].split()[0]
        if session.part is None:
            assert '-noappend' not in command
            target = os.path.join(folder, f'{base}.xtc')
            if os.path.exists(target):
                self.seen.append(state(target))
                assert os.path.getsize(target) >= session.cut, \
                    'GROMACS would refuse to append'
                os.truncate(target, session.cut)
            else:
                self.seen.append({})
                open(target, 'wb').close()
        else:
            assert '-noappend' in command
            target = os.path.join(folder, f'{base}.part{session.part:04d}.xtc')
            self.seen.append(sorted(os.listdir(folder)))
            open(target, 'wb').close()      # -nobackup overwrites a part
        offsets = frame_bytes(session.source)
        for a in range(0, len(offsets) - 1, session.burst):
            b = min(a + session.burst, len(offsets) - 1)
            with open(session.source, 'rb') as src, open(target, 'ab') as dst:
                src.seek(offsets[a])
                dst.write(src.read(offsets[b] - offsets[a]))
            if stop_condition():
                break
        return 0


class SimWorker(WorkerSimulate):
    def __init__(self, params):
        self.params = params
        self.must_stop = False
        self.termination_timeout = 1.0


def make_params(record_bias=False):
    params = SimpleNamespace(
        engine='gromacs', trajectory_extension='.xtc', max_length=np.inf,
        trajectory_update_batch_size=1000, pipeline=PIPELINE, states='ARB',
        gmx_mdrun='gmx mdrun', gmx_grompp='gmx grompp -maxwarn 1',
        record_bias=record_bias, bias_source='file')
    params.run_simulation = (
        lambda *args, **kwargs: ParamsMethods.run_simulation(params, *args, **kwargs))
    return params


def new_process():
    """A new job: a new process, whose in-memory caches start empty."""
    MDA_CACHE.clear()
    NPY_CACHE.clear()


def checkpoint_after(xtc, nframes, part=1, time=None):
    """The checkpoint GROMACS writes when `xtc` holds its first `nframes` frames."""
    offset = frame_bytes(xtc)[nframes]
    size = min(offset, 1 << 20)
    with open(xtc, 'rb') as fh:
        fh.seek(offset - size)
        digest = hashlib.md5(fh.read(size)).hexdigest()
    deffnm = xtc[:-len('.xtc')]
    name = os.path.basename(deffnm)
    return Checkpoint(f'{deffnm}.cpt', part, nframes * 5000,
                      nframes * DT if time is None else time,
                      (CheckpointOutput(f'{name}.log', 1234, -1, ''),
                       CheckpointOutput(f'{name}.xtc', offset, size, digest)))


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run sessions of the toy half chainR0/back through the worker."""
    folder = tmp_path / 'run' / 'chainR0'
    folder.mkdir(parents=True)
    sources = tmp_path / 'sources'
    sources.mkdir()
    deffnm = str(folder / 'back')
    open(f'{deffnm}.tpr', 'w').close()
    with open(f'{deffnm}.log', 'w') as fh:   # the log, which mdrun appends to
        fh.write('x' * 4000)
    new_process()

    def job(*sessions, checkpoint=None, params=None, path=None):
        gromacs = FakeGromacs(*sessions)
        monkeypatch.setattr('aimmd.params._methods.execute_command', gromacs)
        if checkpoint is not None:
            open(checkpoint.fname, 'w').close()
            monkeypatch.setattr(rollback, 'read_checkpoint',
                                lambda fname, gmx: checkpoint)
        path = Path() if path is None else path
        result = SimWorker(params or make_params())._simulate(
            deffnm, path, 'R', 'shoot')
        return result, gromacs, path

    def job1(n=100, params=None):
        """mdrun writes realisation 1 (n frames, all in R); the job then ends."""
        source = write_frames(sources / 'run1.xtc', [x_run1(i) for i in range(n)], 0)
        job(Session(source), params=params)
        new_process()

    def realisation2(restart, n=100):
        return write_frames(sources / f'run2_{restart}.xtc',
                            [x_run2(i, restart) for i in range(restart, restart + n)],
                            restart, seed=restart)

    return SimpleNamespace(deffnm=deffnm, xtc=f'{deffnm}.xtc', folder=folder,
                           job=job, job1=job1, realisation2=realisation2)


# ---------------------------------------------------------------------------
# (a) appending restart from an old checkpoint
# ---------------------------------------------------------------------------

def test_old_checkpoint_rolls_the_half_back_and_it_stops_at_its_first_b_frame(
        run, capsys):
    run.job1()
    xtc = run.xtc
    assert rows(f'{xtc}.states.npy') == 100
    # more caches: values and bias are cut too; sweep_frame is not per frame
    save_npy(f'{xtc}.values.npy', np.linspace(1, 2, 100))
    save_npy(f'{xtc}.bias.npy', np.linspace(3, 4, 100))
    save_npy(f'{xtc}.sweep_frame.npy', np.array([7]))
    capsys.readouterr()

    # job 2 resumes from a checkpoint written after frame 29 (as on Oct 9):
    # mdrun simulates frames 30, 31, ... again; this time B comes at frame 43
    checkpoint = checkpoint_after(xtc, 30)
    result, gromacs, path = run.job(
        Session(run.realisation2(30), cut=checkpoint.output(xtc).offset),
        checkpoint=checkpoint)
    out = capsys.readouterr().out

    entry = [line for line in out.splitlines()
             if line.startswith('!!! CHECKPOINT ROLLBACK')]
    assert entry == [
        f'!!! CHECKPOINT ROLLBACK {xtc}: mdrun resumes from back.cpt (part 1, '
        f'step 150000, t = 300 ps); dropped the 70 frames written after it '
        f'(t = 300 to 990 ps; 690.0 ps of simulated time past the checkpoint), '
        f'back.xtc cut from 100 to 30 frames; caches cut to 30 rows: bias (100), '
        f'cv (100), {SERIES} (100), states (100), values (100)']
    # the log entry comes before mdrun starts, which finds the file and the
    # caches cut to the checkpoint
    assert out.index('!!! CHECKPOINT ROLLBACK') < out.index('+++ starting')
    assert gromacs.seen[0]['size'] == checkpoint.output(xtc).offset
    assert gromacs.seen[0]['rows'] == {'bias': 30, 'cv': 30, SERIES: 30,
                                       'states': 30, 'sweep_frame': 1,
                                       'values': 30}
    # the half stops at its true first B frame (44 frames), not at the old
    # cached length + 1 (101)
    assert result[0] + result[3] == 44 and result[2] == 'B'
    assert f'xxx {run.deffnm} completed after 44 frames in B' in out
    # every cached row belongs to the frame now at its index
    assert_caches_match(xtc)
    assert rows(f'{xtc}.values.npy') == rows(f'{xtc}.bias.npy') == 30
    assert rows(f'{xtc}.sweep_frame.npy') == 1
    assert len(path) == len(on_disk(xtc)['times']) == 50


def test_without_the_rollback_the_half_stops_late(run, capsys, monkeypatch):
    """Control: the scenario above reproduces the bug when nothing rolls back."""
    monkeypatch.setattr('aimmd.worker._simulate.rollback_to_checkpoint',
                        lambda *args, **kwargs: None)
    run.job1()
    checkpoint = checkpoint_after(run.xtc, 30)
    result, _, _ = run.job(
        Session(run.realisation2(30), cut=checkpoint.output(run.xtc).offset),
        checkpoint=checkpoint)
    assert result[0] + result[3] == 101
    states = np.load(f'{run.xtc}.states.npy')
    assert (states[43:100] == 'R').all()                     # stale rows
    assert (on_disk(run.xtc)['states'][43:100] == 'B').all()


def test_the_in_memory_path_of_the_same_process_is_cut(run, capsys):
    """A Path that already read the lost frames (same process) loses them."""
    run.job1()
    path = Path()
    path.extend(run.xtc, pipeline=PIPELINE[:-1])
    assert len(path) == 100
    checkpoint = checkpoint_after(run.xtc, 30)
    result, _, path = run.job(
        Session(run.realisation2(30), cut=checkpoint.output(run.xtc).offset),
        checkpoint=checkpoint, path=path)
    assert result[0] + result[3] == 44
    assert path._fnames == [run.xtc] and len(path) == 50
    np.testing.assert_allclose(path.times, on_disk(run.xtc)['times'])
    assert_caches_match(run.xtc)


# ---------------------------------------------------------------------------
# (b) clean restart: the checkpoint is at the end of the trajectory
# ---------------------------------------------------------------------------

def test_clean_restart_changes_nothing(run, capsys):
    run.job1()
    before = state(run.xtc)
    capsys.readouterr()
    checkpoint = checkpoint_after(run.xtc, 100)
    result, gromacs, _ = run.job(
        Session(run.realisation2(100), cut=checkpoint.output(run.xtc).offset),
        checkpoint=checkpoint)
    out = capsys.readouterr().out
    assert gromacs.seen == [before]          # mdrun found the files untouched
    assert 'CHECKPOINT ROLLBACK' not in out
    assert (f'... {run.xtc} ends at back.cpt (part 1, step 500000, t = 1000 ps) '
            f'(100 frames): nothing to roll back') in out
    assert result[0] + result[3] == 106 and result[2] == 'B'
    assert_caches_match(run.xtc)


def test_caches_longer_than_the_trajectory_are_cut(run, capsys):
    """Rows beyond the last frame belong to frames that no longer exist."""
    run.job1()
    xtc = run.xtc
    # leftovers of a longer realisation: 20 more rows, labelled B
    update_npy(f'{xtc}.states.npy', np.array(['B'] * 20), np.arange(100, 120))
    update_npy(f'{xtc}.cv.npy', np.full(20, 45.0), np.arange(100, 120))
    update_npy(f'{xtc}.{SERIES}.npy', np.ones((20, 3 * N_ATOMS), np.float32),
               np.arange(100, 120))
    capsys.readouterr()
    checkpoint = checkpoint_after(xtc, 100)
    result, _, _ = run.job(
        Session(run.realisation2(100), cut=checkpoint.output(xtc).offset),
        checkpoint=checkpoint)
    out = capsys.readouterr().out
    assert (f'!!! CHECKPOINT ROLLBACK {xtc}: mdrun resumes from back.cpt (part 1, '
            f'step 500000, t = 1000 ps); back.xtc ends at the checkpoint (100 '
            f'frames); caches cut to 100 rows: cv (120), {SERIES} (120), '
            f'states (120)') in out
    # with the stale B rows the half would have stopped after 101 frames
    assert result[0] + result[3] == 106
    assert_caches_match(xtc)


# ---------------------------------------------------------------------------
# when nothing is rolled back
# ---------------------------------------------------------------------------

def _fake_gmx(tmp_path, body):
    script = tmp_path / 'fake_gmx'
    script.write_text('#!/bin/sh\n' + body)
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize('how', ['reader raises', 'gmx dump fails'])
def test_unreadable_checkpoint_warns_and_rolls_nothing_back(
        run, capsys, monkeypatch, tmp_path, how):
    run.job1()
    before = state(run.xtc)
    capsys.readouterr()
    offset = checkpoint_after(run.xtc, 30).output(run.xtc).offset
    open(f'{run.deffnm}.cpt', 'w').close()
    params = make_params()
    if how == 'reader raises':
        def broken(fname, gmx):
            raise CheckpointError("'gmx dump -cp back.cpt' failed with exit "
                                  "code 1: Checkpoint file corrupted/truncated")
        monkeypatch.setattr(rollback, 'read_checkpoint', broken)
    else:   # the real reader, running a gmx that fails like GROMACS does
        gmx = _fake_gmx(tmp_path, 'printf "Fatal error:\\nCheckpoint file '
                                  'corrupted/truncated\\n" >&2\nexit 1\n')
        params.gmx_grompp = f'{gmx} grompp -maxwarn 1'
    result, gromacs, _ = run.job(Session(run.realisation2(30), cut=offset),
                                 params=params)
    out = capsys.readouterr().out
    assert (f'!!! CHECKPOINT {run.deffnm}.cpt could not be read, so nothing was '
            f'rolled back to it') in out
    assert 'Checkpoint file corrupted/truncated' in out
    assert gromacs.seen == [before] and len(gromacs.commands) == 1
    # the old behaviour, which is all that is left without the checkpoint
    assert result[0] + result[3] == 101


def _call(run, checkpoint):
    path = Path()
    path.extend(run.xtc)
    return rollback.rollback_to_checkpoint(
        run.deffnm, path, make_params(), noappend=False), path


def test_checkpoint_longer_than_the_trajectory_is_left_to_gromacs(
        run, capsys, monkeypatch):
    run.job1()
    before = state(run.xtc)
    good = checkpoint_after(run.xtc, 100)
    record = good.output(run.xtc)
    checkpoint = Checkpoint(good.fname, 1, 1, 1.0, (CheckpointOutput(
        'back.xtc', record.offset + 100, record.checksum_size, record.checksum),))
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda f, g: checkpoint)
    result, path = _call(run, checkpoint)
    assert 'GROMACS will refuse to append; nothing rolled back' in result.message
    assert f'it accounts for {record.offset + 100} bytes of {run.xtc}' in \
        result.message
    assert not result.rolled_back and len(path) == 100
    assert state(run.xtc) == before


def test_checkpoint_of_another_trajectory_is_left_to_gromacs(run, monkeypatch):
    run.job1()
    before = state(run.xtc)
    good = checkpoint_after(run.xtc, 30)
    record = good.output(run.xtc)
    checkpoint = Checkpoint(good.fname, 1, 1, 1.0, (CheckpointOutput(
        'back.xtc', record.offset, record.checksum_size, '0' * 32),))
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda f, g: checkpoint)
    result, path = _call(run, checkpoint)
    assert f'{run.xtc} does not match it' in result.message
    assert 'GROMACS will refuse to append; nothing rolled back' in result.message
    assert len(path) == 100 and state(run.xtc) == before


@pytest.mark.parametrize('broken', ['missing', 'short'])
def test_another_output_file_gromacs_refuses_stops_the_rollback(
        run, monkeypatch, broken):
    """GROMACS checks every output file it recorded, not only the trajectory."""
    run.job1()
    before = state(run.xtc)
    good = checkpoint_after(run.xtc, 30)
    edr = f'{run.deffnm}.edr'
    with open(edr, 'w') as fh:
        fh.write('e' * (10 if broken == 'short' else 100))
    if broken == 'missing':
        os.remove(edr)
    checkpoint = Checkpoint(good.fname, 1, 1, 1.0, good.outputs + (
        CheckpointOutput('back.edr', 50),))
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda f, g: checkpoint)
    result, path = _call(run, checkpoint)
    assert 'GROMACS will refuse to append; nothing rolled back' in result.message
    assert (f'{edr} is missing' if broken == 'missing'
            else f'it accounts for 50 bytes of {edr}, which has 10') in \
        result.message
    assert len(path) == 100 and state(run.xtc) == before


def test_checkpoint_without_the_trajectory_is_reported(run, monkeypatch):
    run.job1()
    before = state(run.xtc)
    checkpoint = Checkpoint(f'{run.deffnm}.cpt', 1, 1, 1.0,
                            (CheckpointOutput('back.log', 10),))
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda f, g: checkpoint)
    result, _ = _call(run, checkpoint)
    assert result.message.endswith('lists no output file back.xtc: nothing '
                                   'rolled back')
    assert state(run.xtc) == before


def test_no_checkpoint_no_rollback(run, capsys, monkeypatch):
    """A first session (no .cpt) never reads a checkpoint."""
    monkeypatch.setattr(rollback, 'read_checkpoint',
                        lambda *args: pytest.fail('read a checkpoint'))
    run.job1()
    out = capsys.readouterr().out
    assert 'CHECKPOINT' not in out


# ---------------------------------------------------------------------------
# (c) free run (-noappend): later parts are moved aside
# ---------------------------------------------------------------------------

@pytest.fixture
def free(tmp_path, monkeypatch):
    """A free run whose parts 2 and 3 came after the checkpoint of part 1."""
    folder = tmp_path / 'run' / 'freeA'
    folder.mkdir(parents=True)
    sources = tmp_path / 'sources'
    sources.mkdir()
    deffnm = str(folder / 'traj000001')
    new_process()

    # the parts on disk: seed, part 1 (t = 0..490), and the parts 2 and 3 of a
    # continuation that the checkpoint does not know (t = 500..1390)
    write_frames(f'{deffnm}.part0000.xtc', [x_free(-1)], -1, seed=10)
    write_frames(f'{deffnm}.part0001.xtc', [x_free(i) for i in range(50)], 0, seed=11)
    write_frames(f'{deffnm}.part0002.xtc', [x_free(i) for i in range(50, 100)], 50,
                 seed=12)
    write_frames(f'{deffnm}.part0003.xtc', [x_free(i) for i in range(100, 140)],
                 100, seed=13)
    parts = sorted(glob.glob(f'{deffnm}.part????.xtc'))
    Path(parts, remove_overlapping_frames=True, pipeline=PIPELINE[:-1])
    for part in parts[1:]:
        save_npy(f'{part}.bias.npy', on_disk(part)['times'] / 1000)
        for ext in ('_COLVAR', '.log', '.edr'):
            open(part.replace('.xtc', ext), 'w').close()
    open(f'{deffnm}.tpr', 'w').close()
    new_process()

    # mdrun resumes from a checkpoint of part 1 at t = 400 ps and writes part 2
    checkpoint = Checkpoint(f'{deffnm}.cpt', 1, 200000, 400.0, (
        CheckpointOutput('traj000001.part0001.log', 1234),
        CheckpointOutput('traj000001.part0001.xtc', 5678)))
    open(checkpoint.fname, 'w').close()
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda fname, gmx: checkpoint)
    resumed = write_frames(sources / 'resumed.xtc',
                           [x_free(i) + 0.5 for i in range(40, 70)], 40, seed=14)
    gromacs = FakeGromacs(Session(resumed, part=2))
    monkeypatch.setattr('aimmd.params._methods.execute_command', gromacs)
    return SimpleNamespace(deffnm=deffnm, folder=folder, gromacs=gromacs,
                           resumed=resumed)


@pytest.mark.parametrize('same_process', [False, True])
def test_free_run_moves_later_parts_aside_and_does_not_splice(free, capsys,
                                                              same_process):
    path = Path()
    if same_process:   # the worker's Path already holds the later parts
        path.extend(f'{free.deffnm}.part????.xtc', pipeline=PIPELINE[:-1])
        assert len(path) == 141
    result = SimWorker(make_params())._simulate(free.deffnm, path, 'A', 'free')
    out = capsys.readouterr().out

    moved = glob.glob(str(free.folder / 'rolled_back_*'))
    assert len(moved) == 1
    moved = moved[0]
    entry = [line for line in out.splitlines()
             if line.startswith('!!! CHECKPOINT ROLLBACK')]
    assert entry == [
        f'!!! CHECKPOINT ROLLBACK {free.deffnm}: mdrun resumes from traj000001.cpt '
        f'(part 1, step 200000, t = 400 ps) with part 0002; moved the 16 files '
        f'of parts 0002-0003 (90 frames; t = 500 to 1390 ps; 990.0 ps of '
        f'simulated time past the checkpoint) to {moved}']
    # nothing deleted: every file of parts 2 and 3 is in the folder
    names = sorted(os.listdir(moved))
    for part in ('part0002', 'part0003'):
        for suffix in ('.xtc', '.xtc.states.npy', f'.xtc.{SERIES}.npy',
                       '.xtc.cv.npy', '.xtc.bias.npy', '_COLVAR', '.log', '.edr'):
            assert f'traj000001.{part}{suffix}' in names
    # mdrun started with parts 0 and 1 only; it wrote a new part 2
    assert 'traj000001.part0002.xtc' not in free.gromacs.seen[0]
    assert not os.path.exists(f'{free.deffnm}.part0003.xtc')
    resumed_times = on_disk(free.resumed)['times']
    np.testing.assert_allclose(on_disk(f'{free.deffnm}.part0002.xtc')['times'],
                               resumed_times)

    # the trajectory: seed, part 1 up to the restart, then the new part 2 --
    # no frame of the abandoned continuation, time strictly increasing
    expected = np.concatenate([[-10.0], np.arange(0, 400, DT), resumed_times])
    np.testing.assert_allclose(path.times, expected)
    assert result[1] == len(expected)
    # what the trainer rebuilds from the files is the same trajectory
    rebuilt = Path(sorted(glob.glob(f'{free.deffnm}.part????.xtc')),
                   remove_overlapping_frames=True)
    np.testing.assert_allclose(rebuilt.times, expected)
    assert_caches_match(f'{free.deffnm}.part0002.xtc')


def test_free_run_without_later_parts_is_quiet(free, capsys, monkeypatch):
    checkpoint = Checkpoint(f'{free.deffnm}.cpt', 3, 1, 1390.0)
    monkeypatch.setattr(rollback, 'read_checkpoint', lambda fname, gmx: checkpoint)
    before = sorted(os.listdir(free.folder))
    result = rollback.rollback_to_checkpoint(free.deffnm, Path(), make_params(),
                                             noappend=True)
    assert not result.rolled_back and result.kept == {}
    assert result.message == (f'... {free.deffnm} resumes from traj000001.cpt '
                              f'(part 3, step 1, t = 1390 ps) with part 0004: '
                              f'nothing to roll back')
    assert sorted(os.listdir(free.folder)) == before


def test_checkpoint_without_its_trajectory_keeps_the_relic_cleanup(
        run, capsys, monkeypatch):
    """A .cpt with no trajectory beside it is removed, and mdrun starts anew."""
    monkeypatch.setattr(rollback, 'read_checkpoint',
                        lambda *args: pytest.fail('read a checkpoint'))
    open(f'{run.deffnm}.cpt', 'w').close()
    open(f'{run.deffnm}.log', 'w').close()
    source = write_frames(os.path.join(os.path.dirname(run.folder), 'src.xtc'),
                          [x_run2(i, 0) for i in range(20)], 0)
    result, gromacs, _ = run.job(Session(source))
    assert not os.path.exists(f'{run.deffnm}.cpt')
    assert os.path.exists(f'{run.deffnm}.tpr')
    assert gromacs.seen == [{}]           # mdrun started without a trajectory
    assert 'CHECKPOINT' not in capsys.readouterr().out
    assert result[0] + result[3] == 10    # x_run2(i, 0) enters B at frame 9


def test_trr_trajectories_are_rolled_back_too(tmp_path, monkeypatch):
    deffnm = str(tmp_path / 'back')
    universe = mda.Universe.empty(N_ATOMS, trajectory=True)
    with mda.Writer(f'{deffnm}.trr', N_ATOMS) as writer:
        for k in range(10):
            universe.trajectory.ts.positions = np.full((N_ATOMS, 3), k, np.float32)
            universe.trajectory.ts.velocities = np.ones((N_ATOMS, 3), np.float32)
            universe.trajectory.ts.dimensions = BOX
            universe.trajectory.ts.time = k * DT
            writer.write(universe.atoms)
    save_npy(f'{deffnm}.trr.states.npy', np.array(list('R' * 10)))
    from MDAnalysis.lib.formats.libmdaxdr import TRRFile
    with TRRFile(f'{deffnm}.trr') as file:
        offset = int(file.offsets[4])
    checkpoint = Checkpoint(f'{deffnm}.cpt', 1, 4, 40.0,
                            (CheckpointOutput('back.trr', offset),))
    result = rollback.rollback_appended(f'{deffnm}.trr', checkpoint)
    assert result.rolled_back and result.kept == {f'{deffnm}.trr': 4}
    assert 'back.trr cut from 10 to 4 frames' in result.message
    assert os.path.getsize(f'{deffnm}.trr') == offset
    assert rows(f'{deffnm}.trr.states.npy') == 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_frame_caches_lists_per_frame_series_only(tmp_path):
    xtc = str(tmp_path / 'path000001.xtc')
    for name in ('path000001.xtc.states.npy', f'path000001.xtc.{SERIES}.npy',
                 'path000001.xtc.sweep_frame.npy', 'temp.path000001.xtc.states.npy',
                 'path000001.xtc2.states.npy', 'path0000011.xtc.states.npy',
                 'path000001.xtc_offsets.npz'):
        (tmp_path / name).write_bytes(b'')
    assert rollback.frame_caches(xtc) == [
        (SERIES, f'{xtc}.{SERIES}.npy'), ('states', f'{xtc}.states.npy')]


def test_truncate_path_and_path_truncate(tmp_path):
    a = write_frames(tmp_path / 'a.xtc', [30.0] * 5, 0)
    b = write_frames(tmp_path / 'b.xtc', [30.0] * 5, 5)
    path = Path([a, b])
    path.states = np.array(list('RRRRRRRRRR'))
    assert rollback.truncate_path(path, {b: 2}) == 3
    assert (path._fnames, path._first, path._last) == ([a, b], [0, 0], [4, 1])
    assert len(path.__dict__['states']) == 7
    assert rollback.truncate_path(path, {b: 5, 'other.xtc': 0}) == 0
    assert rollback.truncate_path(path, {a: 0}) == 7
    assert len(path) == 0 and path._fnames == []
    # backward segments: the first frame read is the last one of the file
    back = Path(a)[::-1]
    assert rollback.truncate_path(back, {a: 3}) == 5
    assert Path(a).truncate(9) == 0
