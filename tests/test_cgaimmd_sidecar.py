"""In-flight records, per-shot sidecars, their reconciliation, provenance and the shot table."""

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.core.utils import remove
from aimmd.cgaimmd import _legacy
from aimmd.cgaimmd import io as cgaimmd_io
from aimmd.cgaimmd import sidecar
from aimmd.cgaimmd.io import file_sha256
from aimmd.pathensemble import PathEnsemble
from tests._helpers_unit import build_path, simple_descriptors_function


# a shot A -> R -> R -> R -> B of two atoms; the state is set by the x of atom 0
# (x <= -0.5 is A, x >= 0.5 is B, else R); atom 1 sits 1 A away in y. Values are
# multiples of 0.1 A, which xtc (0.001 nm) stores exactly.
_X = np.array([-1.0, -0.3, 0.1, 0.3, 1.0])
SHOT = np.zeros((5, 2, 3), dtype=np.float32)
SHOT[:, 0, 0] = _X
SHOT[:, 1, 0] = _X
SHOT[:, 1, 1] = 1.0
SHOOTING_INDEX = 2
LAMBDA_SP = 2.0 * 0.1                              # values_function at the shooting frame

# the keys of every sidecar, in their order
SIDECAR_KEYS = ('schema', 'source', 'note', 'stale_record', 'path', 'chain', 'created',
                'registered', 'recovered_after_restart', 'lambda_target', 'lambda_raw',
                'lambda_final', 'lambda_registered', 'state_final', 'lambda_law', 'acceptance',
                'network', 'generator', 'pool', 'stage1_rejected', 'candidates', 'validity',
                'prepare', 'minimize', 'cost_seconds', 'seed', 'device', 'aimmd', 'software',
                'bias', 'registration')
AIMMD_KEYS = {'version', 'git_commit', 'git_dirty', 'git_reason', 'git_source', 'source_sha256'}
LEGACY = Path(__file__).resolve().parent / 'data' / 'legacy_cgaimmd'


def values_function(descriptors):
    return 2.0 * np.asarray(descriptors, dtype=float)[:, 0]


def _params(engine='toy'):
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(states='ARB', engine=engine, trajectory_extension='.xtc',
                           descriptors_function=simple_descriptors_function,
                           values_function=values_function)
    return params


def _folder(tmp_path, name='chainR0'):
    folder = tmp_path / 'run' / name
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _shot(folder, stem='path000001', positions=SHOT, values=True):
    path = build_path(folder, stem=stem, positions=positions, shooting_index=SHOOTING_INDEX)
    if not values:
        os.remove(f'{path.fname}.values.npy')
    return path


def _replaced_names(monkeypatch):
    """Record the target basename of every os.replace (the moment a file appears)."""
    names = []
    real_replace = os.replace

    def spy(src, dst):
        names.append(os.path.basename(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(cgaimmd_io.os, 'replace', spy)
    return names


def _record(path_name='path000001', **extra):
    record = {'schema': sidecar.SIDECAR_SCHEMA, 'source': 'cgaimmd', 'path_name': path_name,
              'lambda_target': 0.4, 'lambda_raw': 0.35, 'lambda_final': 0.3,
              'state_final': 'R', 'created': '2026-09-29T10:00:00+00:00'}
    record.update(extra)
    return record


def test_in_flight_roundtrip(tmp_path):
    folder = _folder(tmp_path)
    assert sidecar.in_flight_fname(folder) == f'{folder}/.cgaimmd_shooting_point.json'
    assert sidecar.in_flight_positions_fname(folder) == f'{folder}/.cgaimmd_shooting_point.npy'
    assert sidecar.read_in_flight(folder) is None

    positions = SHOT[SHOOTING_INDEX].astype(np.float64) + 1e-9
    record = _record()
    del record['created']
    sidecar.write_in_flight(folder, record, positions)
    assert 'positions_sha256' not in record                     # the caller's dict is not changed
    stored = sidecar.read_in_flight(folder)
    assert stored['path_name'] == 'path000001'
    assert stored['positions_sha256'] == file_sha256(sidecar.in_flight_positions_fname(folder))
    assert stored['created'].endswith('+00:00')
    for key, value in record.items():
        assert stored[key] == value
    loaded = np.load(sidecar.in_flight_positions_fname(folder))
    assert loaded.dtype == np.float32 and loaded.shape == (2, 3)
    np.testing.assert_array_equal(loaded, positions.astype(np.float32))
    assert [p for p in os.listdir(folder) if '.tmp.' in p] == []

    sidecar.clear_in_flight(folder)
    assert sidecar.read_in_flight(folder) is None
    assert not os.path.exists(sidecar.in_flight_positions_fname(folder))
    sidecar.clear_in_flight(folder)                              # clearing twice is fine

    # a missing or empty 'created' is set to now
    for created in (None, ''):
        sidecar.write_in_flight(folder, _record(created=created), positions)
        assert sidecar.read_in_flight(folder)['created'].endswith('+00:00')
    sidecar.clear_in_flight(folder)

    with pytest.raises(ValueError, match='path_name'):
        sidecar.write_in_flight(folder, {'source': 'cgaimmd'}, positions)
    with pytest.raises(ValueError, match='positions'):
        sidecar.write_in_flight(folder, _record(), positions[:, :2])
    assert sidecar.read_in_flight(folder) is None

    # an unreadable marker is reported, not raised
    Path(sidecar.in_flight_fname(folder)).write_text('{"path_name": ')
    broken = sidecar.read_in_flight(folder)
    assert broken['source'] == 'unknown' and 'unreadable' in broken['note']
    assert broken['path_name'] is None


def test_in_flight_write_order(tmp_path, monkeypatch):
    """The coordinates (.npy) appear before the record (.json) that names their sha256."""
    folder = _folder(tmp_path)
    names = _replaced_names(monkeypatch)
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    assert names == ['.cgaimmd_shooting_point.npy', '.cgaimmd_shooting_point.json']


def test_sidecar_keys(tmp_path):
    folder = _folder(tmp_path)
    params = _params()
    path = _shot(folder)
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    fname = sidecar.write_shot_sidecar(path, sidecar.read_in_flight(folder), params=params,
                                       states='ARB', target_state='R')
    assert fname == f'{path.fname}.cgaimmd.json' == sidecar.sidecar_fname(path.fname)
    data = sidecar.read_shot_sidecar(path.fname)
    assert data == json.loads(Path(fname).read_text())
    assert sidecar.SIDECAR_KEYS == SIDECAR_KEYS
    # every key; the only extras are the record's own keys
    assert set(data) - set(SIDECAR_KEYS) <= set(sidecar.read_in_flight(folder))
    assert data['schema'] == sidecar.SIDECAR_SCHEMA == 'aimmd.cgaimmd.shot/1'
    assert data['source'] == 'cgaimmd'
    assert data['stale_record'] is None
    assert data['path'] == 'path000001.xtc'
    assert data['chain'] == 'chainR0'
    assert data['created'] == '2026-09-29T10:00:00+00:00'
    assert data['registered'].endswith('+00:00')
    assert data['recovered_after_restart'] is False
    assert data['lambda_target'] == 0.4 and data['lambda_final'] == 0.3
    # the registered shooting frame's value, through params.values_function
    assert data['lambda_registered'] == pytest.approx(LAMBDA_SP, abs=1e-6)
    assert data['registration'] == {'n_frames': 5, 'path_type': str(path.type),
                                    'shooting_index': SHOOTING_INDEX,
                                    'complete': bool(path.is_complete('R', 'ARB')),
                                    'lambda_registered_missing': None}
    assert data['aimmd']['version'] == aimmd.__version__
    assert set(data['aimmd']) == AIMMD_KEYS
    assert data['aimmd']['git_commit'] == sidecar.aimmd_git_commit()
    assert data['aimmd']['git_dirty'] == sidecar.aimmd_git_dirty()
    assert data['aimmd']['source_sha256'] == sidecar.source_sha256()
    # a commit, or an explicit reason why there is none
    assert data['aimmd']['git_commit'] is not None or data['aimmd']['git_reason']
    assert data['software'] == sidecar.software_versions()
    assert data['bias'] is None                                   # the toy run has no bias
    for key in ('lambda_law', 'acceptance', 'network', 'generator', 'pool', 'validity',
                'prepare', 'minimize', 'cost_seconds', 'seed', 'device', 'candidates',
                'stage1_rejected', 'note'):
        assert data[key] is None                                 # not in this record: null
    assert sidecar.read_shot_sidecar(str(tmp_path / 'nothing.xtc')) is None

    commit = sidecar.aimmd_git_commit()
    assert commit is None or (len(commit) in (40, 64) and set(commit) <= set('0123456789abcdef'))

    # no record at all: source 'unknown', still registered and valued
    other = _shot(folder, stem='path000002')
    data = json.loads(Path(sidecar.write_shot_sidecar(
        other, None, params=params, states='ARB', target_state='R')).read_text())
    assert data['source'] == 'unknown' and data['stale_record'] is None
    assert data['lambda_registered'] == pytest.approx(LAMBDA_SP, abs=1e-6)


def test_unknown_origin_record(tmp_path):
    folder = _folder(tmp_path)
    record = sidecar.unknown_origin_record(folder, 'path000003')
    assert record['schema'] == sidecar.SIDECAR_SCHEMA
    assert record['source'] == 'resumed_without_cgaimmd_record'
    assert record['path_name'] == 'path000003'
    assert record['chain'] == 'chainR0'
    json.dumps(record)
    path = _shot(folder, stem='path000003')
    data = json.loads(Path(sidecar.write_shot_sidecar(
        path, record, params=_params(), states='ARB', target_state='R')).read_text())
    assert data['source'] == 'resumed_without_cgaimmd_record'


def test_reconcile_recovered(tmp_path):
    folder = _folder(tmp_path)
    params = _params()
    path = _shot(folder)
    chain = PathEnsemble(path)
    assert sidecar.reconcile_in_flight(folder, chain, params) == 'none'
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    # crash between register_path and the sidecar write
    assert sidecar.reconcile_in_flight(folder, chain, params) == 'recovered'
    data = sidecar.read_shot_sidecar(path.fname)
    assert data['recovered_after_restart'] is True
    assert data['source'] == 'cgaimmd'
    assert data['registration']['shooting_index'] == SHOOTING_INDEX
    assert sidecar.read_in_flight(folder) is None

    # crash between the sidecar write and clear_in_flight: the marker is dropped,
    # even though segments are still initialized, and the sidecar is not touched
    before = Path(sidecar.sidecar_fname(path.fname)).read_bytes()
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    build_path(folder, stem='back', positions=SHOT)
    build_path(folder, stem='forw', positions=SHOT)
    assert params.check_if_initialized(f'{folder}/back', f'{folder}/forw')
    assert sidecar.reconcile_in_flight(folder, chain, params) == 'cleared'
    assert sidecar.read_in_flight(folder) is None
    assert Path(sidecar.sidecar_fname(path.fname)).read_bytes() == before


def test_reconcile_kept_gromacs(tmp_path):
    folder = _folder(tmp_path)
    params = _params(engine='gromacs')
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    for name in ('back.tpr', 'forw.tpr'):
        Path(folder, name).write_bytes(b'tpr')
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == 'kept'
    assert sidecar.read_in_flight(folder)['path_name'] == 'path000001'


def test_reconcile_kept_toy_engine(tmp_path):
    folder = _folder(tmp_path)
    params = _params(engine='toy')
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    build_path(folder, stem='back', positions=SHOT[:3])
    build_path(folder, stem='forw.part0000', positions=SHOT[2:])
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == 'kept'
    assert sidecar.read_in_flight(folder) is not None


def test_reconcile_cleared(tmp_path, capsys):
    folder = _folder(tmp_path)
    params = _params(engine='gromacs')
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == 'cleared'
    assert sidecar.read_in_flight(folder) is None
    assert not os.path.exists(sidecar.in_flight_positions_fname(folder))
    assert 'path000001' in capsys.readouterr().out
    # one half initialized (a crash between the two grompp calls)
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    Path(folder, 'back.tpr').write_bytes(b'tpr')
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == 'cleared'
    assert sidecar.read_in_flight(folder) is None
    # an empty file is not initialized
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    Path(folder, 'forw.tpr').write_bytes(b'')
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == 'cleared'


def test_reconcile_unreadable_marker(tmp_path):
    """A corrupt marker names no path: cleared unless both segments are initialized."""
    folder = _folder(tmp_path)
    params = _params(engine='gromacs')
    for initialized, expected in ((False, 'cleared'), (True, 'kept')):
        sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
        Path(sidecar.in_flight_fname(folder)).write_text('{"path_name": "path0')
        assert sidecar.read_in_flight(folder)['path_name'] is None
        for name in ('back.tpr', 'forw.tpr'):
            if initialized:
                Path(folder, name).write_bytes(b'tpr')
        assert sidecar.reconcile_in_flight(folder, PathEnsemble(), params) == expected
        present = [os.path.exists(f) for f in (sidecar.in_flight_fname(folder),
                                                sidecar.in_flight_positions_fname(folder))]
        assert present == [initialized, initialized]


def test_names_survive_back_forw_removal(tmp_path):
    folder = _folder(tmp_path)
    path = _shot(folder)
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    sidecar.write_shot_sidecar(path, sidecar.read_in_flight(folder), params=_params(),
                               states='ARB', target_state='R')
    for name in ('back.xtc', 'forw.xtc', 'back.tpr', 'forw.part0000.xtc'):
        Path(folder, name).write_bytes(b'x')
    remove(f'{folder}/*back*', f'{folder}/*forw*')
    left = sorted(os.listdir(folder))
    assert '.cgaimmd_shooting_point.json' in left and '.cgaimmd_shooting_point.npy' in left
    assert 'path000001.xtc.cgaimmd.json' in left
    assert not [name for name in left if 'back' in name or 'forw' in name]
    # path loaders do not pick them up
    assert glob.glob(f'{folder}/path??????.xtc') == [path.fname]
    for name in (sidecar.in_flight_fname(folder), sidecar.in_flight_positions_fname(folder),
                 sidecar.sidecar_fname(path.fname)):
        base = os.path.basename(name)
        assert 'back' not in base and 'forw' not in base


def test_sidecar_renumbered_path_verified_by_coordinates(tmp_path):
    folder = _folder(tmp_path)
    params = _params()
    _shot(folder, stem='path000001')
    # the in-flight shot was expected as path000001 but registered as path000002;
    # its stored frame differs by xtc precision and a translation
    renumbered = _shot(folder, stem='path000002')
    positions = SHOT[SHOOTING_INDEX] + np.array([[0.004, 0.0, 0.0], [-0.004, 0.0, 0.003]],
                                                dtype=np.float32)
    sidecar.write_in_flight(folder, _record('path000001'), positions + 3.0)
    assert sidecar.shooting_point_matches(renumbered, folder) is True
    data = json.loads(Path(sidecar.write_shot_sidecar(
        renumbered, sidecar.read_in_flight(folder), params=params, states='ARB',
        target_state='R')).read_text())
    assert data['source'] == 'cgaimmd'
    assert data['lambda_target'] == 0.4
    assert 'registered under a different index than expected' in data['note']
    assert 'verified by coordinates' in data['note']
    assert data['path'] == 'path000002.xtc'


def test_sidecar_renumbered_path_mismatch_quarantined(tmp_path):
    folder = _folder(tmp_path)
    params = _params()
    renumbered = _shot(folder, stem='path000002')
    record = _record('path000001')
    other = SHOT[SHOOTING_INDEX] + np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
                                            dtype=np.float32)
    shifted = np.concatenate([other, other + 1.0])                # a different shape
    for positions, expected in ((other, False), (shifted, False)):
        sidecar.write_in_flight(folder, record, positions)
        assert sidecar.shooting_point_matches(renumbered, folder) is expected
    sidecar.write_in_flight(folder, record, other)
    stored = sidecar.read_in_flight(folder)
    data = json.loads(Path(sidecar.write_shot_sidecar(
        renumbered, stored, params=params, states='ARB', target_state='R')).read_text())
    assert data['source'] == 'unknown'
    assert data['stale_record'] == stored
    assert data['lambda_target'] is None
    assert data['lambda_registered'] == pytest.approx(LAMBDA_SP, abs=1e-6)
    # without the .npy the match cannot be decided: quarantined as well
    os.remove(sidecar.in_flight_positions_fname(folder))
    assert sidecar.shooting_point_matches(renumbered, folder) is None
    data = json.loads(Path(sidecar.write_shot_sidecar(
        renumbered, stored, params=params, states='ARB', target_state='R')).read_text())
    assert data['source'] == 'unknown'


def test_lambda_registered_does_not_write_values_cache(tmp_path):
    folder = _folder(tmp_path)
    params = _params()
    fresh = _shot(folder, stem='path000001', values=False)
    sidecar.write_shot_sidecar(fresh, _record(), params=params, states='ARB', target_state='R')
    assert not os.path.exists(f'{fresh.fname}.values.npy')
    assert sidecar.read_shot_sidecar(fresh.fname)['lambda_registered'] == pytest.approx(
        LAMBDA_SP, abs=1e-6)

    cached = _shot(folder, stem='path000002')
    values = f'{cached.fname}.values.npy'
    before = (Path(values).read_bytes(), os.stat(values).st_mtime_ns)
    sidecar.write_shot_sidecar(cached, _record('path000002'), params=params, states='ARB',
                               target_state='R')
    assert (Path(values).read_bytes(), os.stat(values).st_mtime_ns) == before


# ---------------------------------------------------------------------------
# lambda_registered: featurized first, never silently None
# ---------------------------------------------------------------------------

def test_lambda_registered_featurizes_a_missing_descriptor_row(tmp_path):
    """A shooting frame without its descriptor row is featurized (the series cache is written,
    the values cache is not); the value is the values function of that row."""
    folder = _folder(tmp_path)
    path = _shot(folder, values=False)
    os.remove(f'{path.fname}.descriptors.npy')
    sidecar.write_shot_sidecar(path, _record(), params=_params(), states='ARB',
                               target_state='R')
    data = sidecar.read_shot_sidecar(path.fname)
    assert data['lambda_registered'] == pytest.approx(LAMBDA_SP, abs=1e-6)
    assert data['registration']['lambda_registered_missing'] is None
    rows = np.load(f'{path.fname}.descriptors.npy')
    np.testing.assert_allclose(rows[SHOOTING_INDEX], SHOT[SHOOTING_INDEX, :, 0], atol=1e-6)
    assert not os.path.exists(f'{path.fname}.values.npy')


def test_lambda_registered_under_a_named_series(tmp_path):
    """The run's descriptors_series is read and, where missing, written (not 'descriptors')."""
    folder = _folder(tmp_path)
    path = _shot(folder, values=False)
    os.remove(f'{path.fname}.descriptors.npy')
    params = _params()
    params.__dict__['descriptors_series'] = 'descriptors-cgtest'
    sidecar.write_shot_sidecar(path, _record(), params=params, states='ARB', target_state='R')
    assert sidecar.read_shot_sidecar(path.fname)['lambda_registered'] == pytest.approx(
        LAMBDA_SP, abs=1e-6)
    assert os.path.exists(f'{path.fname}.descriptors-cgtest.npy')
    assert not os.path.exists(f'{path.fname}.descriptors.npy')


def test_lambda_registered_missing_is_explained_and_logged(tmp_path, capsys):
    """When no value can be had, the sidecar says why and the worker log has a WARNING."""
    def broken(trajectory):
        raise RuntimeError('descriptor bug (a test)')

    folder = _folder(tmp_path)
    path = _shot(folder, values=False)
    os.remove(f'{path.fname}.descriptors.npy')
    params = _params()
    params.__dict__['descriptors_function'] = broken
    sidecar.write_shot_sidecar(path, _record(), params=params, states='ARB', target_state='R')
    data = sidecar.read_shot_sidecar(path.fname)
    assert data['lambda_registered'] is None
    reason = data['registration']['lambda_registered_missing']
    assert 'RuntimeError' in reason and 'descriptor bug' in reason
    assert f'WARNING: no lambda_registered for {path.fname}' in capsys.readouterr().out

    def nan_values(descriptors):
        return np.full(len(descriptors), np.nan)

    other = _shot(folder, stem='path000002')
    params = _params()
    params.__dict__['values_function'] = nan_values
    sidecar.write_shot_sidecar(other, _record('path000002'), params=params, states='ARB',
                               target_state='R')
    data = sidecar.read_shot_sidecar(other.fname)
    assert data['lambda_registered'] is None
    assert 'not finite' in data['registration']['lambda_registered_missing']
    from types import SimpleNamespace
    assert sidecar.lambda_registered(SimpleNamespace(shooting_index=None), params) == (
        None, 'the registered path has no shooting index')


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _fake_git(answers, calls):
    """A subprocess.run stand-in: `answers[command]` is (returncode, stdout, stderr) or an exception."""
    def run(args, **kwargs):
        calls.append(args[1])
        answer = answers[args[1]]
        if isinstance(answer, Exception):
            raise answer
        code, out, err = answer
        return subprocess.CompletedProcess(args, code, stdout=out, stderr=err)
    return run


@pytest.fixture
def fresh_git_provenance():
    sidecar.git_provenance.cache_clear()
    yield
    sidecar.git_provenance.cache_clear()


def _checkout(root, head='ref: refs/heads/main', refs=None, packed=None):
    """A folder that looks like a clone of AIMMD-Lab: root/aimmd/__init__.py and root/.git."""
    package = root / 'aimmd'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    git_dir = root / '.git'
    git_dir.mkdir()
    (git_dir / 'HEAD').write_text(head + '\n')
    for ref, commit in (refs or {}).items():
        (git_dir / ref).parent.mkdir(parents=True, exist_ok=True)
        (git_dir / ref).write_text(commit + '\n')
    if packed:
        (git_dir / 'packed-refs').write_text(
            '# pack-refs with: peeled fully-peeled sorted\n'
            + ''.join(f'{commit} {ref}\n' for ref, commit in packed.items()))
    return package


def test_git_provenance_looked_up_once(monkeypatch, fresh_git_provenance):
    """git runs once per process (ls-files, rev-parse, status); the result is cached."""
    calls = []
    monkeypatch.setattr(sidecar.subprocess, 'run', _fake_git(
        {'ls-files': (0, '__init__.py\n', ''), 'rev-parse': (0, 'ab' * 20 + '\n', ''),
         'status': (0, '', '')}, calls))
    for _ in range(3):
        assert sidecar.aimmd_git_commit() == 'ab' * 20
        assert sidecar.aimmd_git_dirty() is False
    assert calls == ['ls-files', 'rev-parse', 'status']
    assert sidecar.git_provenance() == {'git_commit': 'ab' * 20, 'git_dirty': False,
                                        'git_reason': None, 'git_source': 'git'}


def test_git_provenance_untracked_package_names_the_reason(monkeypatch, fresh_git_provenance,
                                                           tmp_path):
    """git answers 'not tracked': no commit, and the reason says so (no fallback to .git files)."""
    package = _checkout(tmp_path / 'clone', refs={'refs/heads/main': 'cd' * 20})
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))
    calls = []
    monkeypatch.setattr(sidecar.subprocess, 'run', _fake_git(
        {'ls-files': (1, '', "error: pathspec '__init__.py' did not match any file(s) known "
                             "to git\n")}, calls))
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] is None and provenance['git_dirty'] is None
    assert 'not tracked' in provenance['git_reason'] and calls == ['ls-files']


@pytest.mark.parametrize('failure', [
    FileNotFoundError('git'),                                              # no git on PATH
    (128, '', "fatal: detected dubious ownership in repository at '/e/project1/x'\n"),
    subprocess.TimeoutExpired('git', 10)])
def test_git_provenance_reads_the_git_files_when_git_cannot_answer(
        monkeypatch, fresh_git_provenance, tmp_path, failure):
    """No git executable, a repository git refuses, a time-out: the commit is read from the
    clone's .git files and the reason names what git could not do (the null commits of the
    JUPITER sidecars had no reason)."""
    package = _checkout(tmp_path / 'clone', refs={'refs/heads/main': 'cd' * 20})
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))
    monkeypatch.setattr(sidecar.subprocess, 'run', _fake_git({'ls-files': failure}, []))
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] == 'cd' * 20 and provenance['git_source'] == 'git files'
    assert provenance['git_dirty'] is None
    reason = provenance['git_reason']
    assert 'dirty flag is unknown' in reason
    assert {FileNotFoundError: 'not found on PATH', tuple: 'dubious ownership',
            subprocess.TimeoutExpired: 'longer than 10 s'}[type(failure)] in reason


def test_git_files_packed_detached_and_worktree(monkeypatch, fresh_git_provenance, tmp_path):
    """The .git reader follows packed-refs, a detached HEAD and a worktree's .git file."""
    monkeypatch.setattr(sidecar.subprocess, 'run',
                        _fake_git({'ls-files': FileNotFoundError('git')}, []))
    cases = {
        'packed': (_checkout(tmp_path / 'packed', packed={'refs/heads/main': 'aa' * 20}),
                   'aa' * 20),
        'detached': (_checkout(tmp_path / 'detached', head='bb' * 20), 'bb' * 20),
    }
    # a worktree: root/.git is a file naming its git dir, whose commondir holds the refs
    main = tmp_path / 'main'
    _checkout(main, refs={'refs/heads/feature': 'ee' * 20})
    worktree_git = main / '.git' / 'worktrees' / 'wt'
    worktree_git.mkdir(parents=True)
    (worktree_git / 'HEAD').write_text('ref: refs/heads/feature\n')
    (worktree_git / 'commondir').write_text('../..\n')
    worktree = tmp_path / 'wt'
    (worktree / 'aimmd').mkdir(parents=True)
    (worktree / 'aimmd' / '__init__.py').write_text('')
    (worktree / '.git').write_text(f'gitdir: {worktree_git}\n')
    cases['worktree'] = (worktree / 'aimmd', 'ee' * 20)
    for name, (package, commit) in cases.items():
        sidecar.git_provenance.cache_clear()
        monkeypatch.setattr(sidecar, '_package_dir', lambda package=package: str(package))
        assert sidecar.aimmd_git_commit() == commit, name


def test_git_provenance_without_any_git(monkeypatch, fresh_git_provenance, tmp_path):
    """No git and no .git next to the package: no commit, and both reasons are given."""
    package = tmp_path / 'site-packages' / 'aimmd'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))
    monkeypatch.setattr(sidecar.subprocess, 'run',
                        _fake_git({'ls-files': FileNotFoundError('git')}, []))
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] is None and provenance['git_source'] is None
    assert 'not found on PATH' in provenance['git_reason']
    assert 'has no .git' in provenance['git_reason']


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_git_provenance_of_untracked_install(tmp_path, monkeypatch, fresh_git_provenance):
    """An install inside another project's work tree records no commit (not the project's)."""
    def git(*args):
        return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                              cwd=tmp_path, check=True, capture_output=True, text=True).stdout

    git('init', '-q')
    (tmp_path / 'README').write_text('project\n')
    git('add', 'README')
    git('commit', '-q', '-m', 'project')
    package = tmp_path / 'env' / 'site-packages' / 'aimmd'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))
    assert sidecar.aimmd_git_commit() is None                  # untracked: not the project's HEAD
    assert sidecar.aimmd_git_dirty() is None
    assert 'not tracked' in sidecar.git_provenance()['git_reason']
    git('add', '.')
    git('commit', '-q', '-m', 'track the package')
    sidecar.git_provenance.cache_clear()
    assert sidecar.aimmd_git_commit() == git('rev-parse', 'HEAD').strip()
    assert sidecar.aimmd_git_dirty() is False
    (package / '__init__.py').write_text('changed = True\n')
    (tmp_path / 'README').write_text('elsewhere\n')              # outside the package: ignored
    sidecar.git_provenance.cache_clear()
    assert sidecar.aimmd_git_dirty() is True
    git('checkout', '--', 'env')
    (package / 'new_module.py').write_text('')                     # untracked in the package
    sidecar.git_provenance.cache_clear()
    assert sidecar.aimmd_git_dirty() is True


def test_source_sha256_identifies_the_code(monkeypatch, tmp_path):
    """A digest of the package's .py files: changes with any of them, ignores caches."""
    package = tmp_path / 'aimmd'
    (package / 'sub').mkdir(parents=True)
    (package / '__init__.py').write_text('a = 1\n')
    (package / 'sub' / 'module.py').write_text('b = 2\n')
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))

    def digest():
        sidecar.source_sha256.cache_clear()
        return sidecar.source_sha256()

    try:
        first = digest()
        assert len(first) == 64 and digest() == first
        (package / '__pycache__').mkdir()
        (package / '__pycache__' / 'x.cpython-313.pyc').write_bytes(b'cache')
        (package / 'data.txt').write_text('not code')
        assert digest() == first
        (package / 'sub' / 'module.py').write_text('b = 3\n')
        assert digest() != first
    finally:
        sidecar.source_sha256.cache_clear()


def test_software_versions_read_metadata_without_importing():
    """Python and the distribution versions; nothing is imported (bioemu2 stays unloaded)."""
    import importlib.metadata
    import sys
    versions = sidecar.software_versions()
    assert set(versions) == {'python', *sidecar.SOFTWARE_DISTRIBUTIONS}
    assert versions['torch'] == importlib.metadata.version('torch')
    assert versions['numpy'] == np.__version__
    for name in ('bioemu2', 'cuequivariance', 'cuequivariance_torch'):
        assert versions[name] is None or isinstance(versions[name], str)
    if 'bioemu2' not in sys.modules:
        sidecar.software_versions()
        assert 'bioemu2' not in sys.modules


def test_bias_settings(tmp_path):
    """None without a bias; the PLUMED file of gmx_mdrun (absolute, hashed) and the bias fields."""
    params = _params()
    assert sidecar.bias_settings(params) is None
    params = _params(engine='gromacs')
    params.__dict__.update(gmx_mdrun='gmx mdrun -ntomp 2', path=Path(tmp_path))  # parent: tmp_path
    assert sidecar.bias_settings(params) is None
    plumed = tmp_path / 'opes_static.dat'
    plumed.write_text('OPES_METAD ... (a test)\n')

    def bias_function(fname):
        return np.zeros(1)

    params.__dict__.update(gmx_mdrun='gmx mdrun -ntomp 2 -plumed opes_static.dat',
                           record_bias=True, bias_source='file', bias_function=bias_function,
                           bias_reactive_threshold=0.5)
    record = sidecar.bias_settings(params)
    assert record == {'record_bias': True, 'bias_source': 'file',
                      'bias_function': f'{__name__}.test_bias_settings.<locals>.bias_function',
                      'bias_reactive_threshold': 0.5, 'plumed_file': str(plumed),
                      'plumed_sha256': file_sha256(plumed), 'preparation_unbiased': True}
    json.dumps(record)
    plumed.unlink()                                                   # recorded, but unreadable
    assert sidecar.bias_settings(params)['plumed_sha256'] is None
    # a sidecar written for such a run records it
    folder = _folder(tmp_path)
    path = _shot(folder)
    params.__dict__['descriptors_function'] = simple_descriptors_function
    data = json.loads(Path(sidecar.write_shot_sidecar(
        path, _record(), params=params, states='ARB', target_state='R')).read_text())
    assert data['bias']['plumed_file'] == str(plumed) and data['bias']['record_bias'] is True


# ---------------------------------------------------------------------------
# old names: archived sidecars and in-flight markers
# ---------------------------------------------------------------------------

def test_archived_flow_sidecar_is_read_with_the_new_names(tmp_path):
    """A sidecar of the October 2026 chignolin campaign (*.xtc.flow.json, paths stripped) is
    found and read: schema, source and generator label in the new names, one warning each."""
    _legacy._reset()
    folder = _folder(tmp_path)
    fname = f'{folder}/path000001.xtc'
    shutil.copy(LEGACY / 'archived_chignolin_path000001.xtc.flow.json', f'{fname}.flow.json')
    original = json.loads((LEGACY / 'archived_chignolin_path000001.xtc.flow.json').read_text())
    assert original['schema'] == 'aimmd.flow.shot/1' and original['source'] == 'flow'
    assert original['generator']['method'] == 'network.generate'
    with pytest.warns(DeprecationWarning) as caught:
        data = sidecar.read_shot_sidecar(fname)
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 4, messages                      # suffix, schema, source, generator
    assert any("'.flow.json'" in m and "'.cgaimmd.json'" in m for m in messages)
    assert data['schema'] == 'aimmd.cgaimmd.shot/1' and data['source'] == 'cgaimmd'
    assert data['generator']['method'] == 'bioemu2_edm'
    assert data['lambda_registered'] == original['lambda_registered']
    assert data['acceptance'] == original['acceptance']
    assert os.listdir(folder) == ['path000001.xtc.flow.json']   # nothing written
    # a new sidecar wins over an old one
    Path(sidecar.sidecar_fname(fname)).write_text(json.dumps({'schema': 'new'}))
    assert sidecar.read_shot_sidecar(fname) == {'schema': 'new'}


def test_archived_in_flight_marker_is_found_translated_and_cleared(tmp_path):
    """The marker a stopped run of the old code left (.flow_shooting_point.json) is read with
    the new names, used by reconcile and the coordinate match, and cleared with the new one."""
    _legacy._reset()
    folder = _folder(tmp_path, 'chainR1')
    shutil.copy(LEGACY / 'archived_chignolin_flow_shooting_point.json',
                f'{folder}/.flow_shooting_point.json')
    np.save(f'{folder}/.flow_shooting_point.npy', SHOT[SHOOTING_INDEX])
    with pytest.warns(DeprecationWarning, match='flow_shooting_point'):
        record = sidecar.read_in_flight(folder)
    assert record['path_name'] == 'path000204'
    assert record['schema'] == sidecar.SIDECAR_SCHEMA and record['source'] == 'cgaimmd'
    assert record['generator']['method'] == 'bioemu2_edm'
    # the registered path matches the old marker's coordinates
    path = _shot(folder, stem='path000204')
    assert sidecar.shooting_point_matches(path, folder) is True
    # crash between register_path and the sidecar write: recovered into a new-name sidecar
    params = _params()
    assert sidecar.reconcile_in_flight(folder, PathEnsemble(path), params) == 'recovered'
    data = sidecar.read_shot_sidecar(path.fname)
    assert os.path.exists(sidecar.sidecar_fname(path.fname))
    assert data['schema'] == sidecar.SIDECAR_SCHEMA and data['source'] == 'cgaimmd'
    assert data['recovered_after_restart'] is True and data['path_name'] == 'path000204'
    assert not [name for name in os.listdir(folder) if 'shooting_point' in name]


def test_write_in_flight_replaces_an_old_marker(tmp_path):
    folder = _folder(tmp_path)
    Path(folder, '.flow_shooting_point.json').write_text('{"path_name": "path000001"}')
    Path(folder, '.flow_shooting_point.npy').write_bytes(b'old')
    sidecar.write_in_flight(folder, _record(), SHOT[SHOOTING_INDEX])
    assert sorted(os.listdir(folder)) == ['.cgaimmd_shooting_point.json',
                                          '.cgaimmd_shooting_point.npy']
    assert sidecar.read_in_flight(folder)['source'] == 'cgaimmd'


# ---------------------------------------------------------------------------
# the per-shot table
# ---------------------------------------------------------------------------

def _listing(root):
    """{relative file: (size, mtime_ns)} of every file under `root` (hidden ones too)."""
    out = {}
    for directory, _, files in os.walk(root):
        for name in files:
            fname = os.path.join(directory, name)
            stat = os.stat(fname)
            out[os.path.relpath(fname, root)] = (stat.st_size, stat.st_mtime_ns)
    return out


def test_shot_table_and_summary(tmp_path):
    """One row per shot (new sidecars, old sidecars, shots without one); nothing is written."""
    _legacy._reset()
    run = tmp_path / 'run'
    params = _params()
    folder0 = _folder(tmp_path, 'chainR0')
    folder1 = _folder(tmp_path, 'chainR1')
    shot1 = _shot(folder0, stem='path000001')                   # A ... B: reactive
    sidecar.write_shot_sidecar(shot1, _record(
        acceptance={'rule': 'target', 'enforced': True}, network={'round': 3, 'n_trained_shots': 7,
                                                                  'sha256': 'ab' * 32},
        generator={'method': 'ode', 'settings': {}, 'batch_size': 8},
        candidates={'tried': 3, 'rejected': {'state_B': 2}}, stage1_rejected={'clash': 1},
        cost_seconds={'generate': 1.0, 'total': 2.5}), params=params, states='ARB',
        target_state='R')
    back = SHOT[::-1].copy()
    back[-1, 0, 0] = -1.0                                         # B ... A then back to A
    shot2 = _shot(folder0, stem='path000002', positions=np.concatenate([SHOT[:3], SHOT[:2][::-1]]))
    shutil.copy(LEGACY / 'archived_chignolin_path000001.xtc.flow.json',
                f'{shot2.fname}.flow.json')                       # an old sidecar
    _shot(folder1, stem='path000001')                              # an RFPS shot: no sidecar
    (run / 'chainA0').mkdir()
    _shot(str(run / 'chainA0'), stem='path000001')                  # internal shooting: not listed
    before = _listing(run)

    rows = sidecar.shot_table(run, params=params)
    assert _listing(run) == before                              # read only: no lock, cache, offsets
    assert [set(row) for row in rows] == [set(sidecar.SHOT_TABLE_COLUMNS)] * 3
    by_name = {(row['chain'], row['path']): row for row in rows}
    assert set(by_name) == {('chainR0', 'path000001.xtc'), ('chainR0', 'path000002.xtc'),
                            ('chainR1', 'path000001.xtc')}
    assert [row['shot_index'] for row in rows] == [0, 1, 2]
    assert [row['registered'] for row in rows] == sorted(row['registered'] for row in rows)
    new = by_name[('chainR0', 'path000001.xtc')]
    assert (new['first_state'], new['last_state'], new['reactive'], new['n_frames']) == (
        'A', 'B', True, 5)
    assert new['source'] == 'cgaimmd' and new['lambda_target'] == 0.4
    assert new['lambda_registered'] == pytest.approx(LAMBDA_SP, abs=1e-6)
    assert (new['acceptance_rule'], new['acceptance_enforced'], new['network_round'],
            new['network_trained_shots'], new['generator_method']) == ('target', True, 3, 7, 'ode')
    assert new['candidates_rejected'] == {'state_B': 2} and new['stage1_rejected'] == {'clash': 1}
    assert new['seconds_generate'] == 1.0 and new['seconds_total'] == 2.5
    assert new['seconds_prepare'] is None and new['shooting_index'] == SHOOTING_INDEX
    old = by_name[('chainR0', 'path000002.xtc')]
    assert old['source'] == 'cgaimmd' and old['generator_method'] == 'bioemu2_edm'
    assert (old['first_state'], old['last_state'], old['reactive']) == ('A', 'A', False)
    assert old['registered'] == '2026-10-28T01:04:48+00:00'
    assert old['network_round'] == 62 and old['candidates_tried'] == 2
    rfps = by_name[('chainR1', 'path000001.xtc')]
    assert rfps['source'] is None and rfps['candidates_rejected'] is None
    assert rfps['reactive'] is True and rfps['registered'].endswith('+00:00')

    summary = sidecar.summarise_shots(rows, frame_interval_ps=5.0, window=2)
    assert summary['n_shots'] == 3 and summary['n_reactive'] == 2
    assert summary['reactive_share'] == pytest.approx(2 / 3)
    assert summary['per_chain'] == {'chainR0': {'shots': 2, 'reactive': 1},
                                    'chainR1': {'shots': 1, 'reactive': 1}}
    assert summary['path_types'] == {'AA': 1, 'AB': 2}
    assert summary['sources'] == {'cgaimmd': 2, 'none': 1}
    assert summary['md_frames_in_shots'] == 15
    assert summary['md_ns_in_shots'] == pytest.approx(15 * 5.0 / 1000.0)
    assert summary['cumulative_reactive'] == [int(rows[0]['reactive']),
                                              sum(int(r['reactive']) for r in rows[:2]), 2]
    assert [window[:2] for window in summary['reactive_share_in_windows']] == [(0, 2), (2, 3)]
    assert summary['candidates_tried'] == 5
    assert summary['candidates_rejected'] == {'state_A': 1, 'state_B': 2}
    json.dumps(summary)
    empty = sidecar.summarise_shots([])
    assert empty['n_shots'] == 0 and empty['reactive_share'] is None
    assert empty['md_ns_in_shots'] is None
    with pytest.raises(ValueError, match='window'):
        sidecar.summarise_shots(rows, window=0)
    assert sidecar.shot_table(tmp_path / 'nothing') == []
