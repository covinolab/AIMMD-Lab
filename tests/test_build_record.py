"""The build record: the git commit a built AIMMD-Lab comes from.

``setup.py`` writes ``aimmd/_build_record.json`` into every built package
(wheels, installs from a folder, source distributions), and the C-GAIMMD
sidecars read it when git and pip's install record name no commit
(:func:`aimmd.cgaimmd.sidecar.git_provenance`). Before, a wheel or a folder
install recorded ``git_commit: null``.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from aimmd.cgaimmd import sidecar


ROOT = Path(__file__).resolve().parents[1]
NOT_TRACKED = (1, '', "error: pathspec '__init__.py' did not match any file(s) known to git\n")


@pytest.fixture
def fresh_git_provenance():
    sidecar.git_provenance.cache_clear()
    yield
    sidecar.git_provenance.cache_clear()


def _setup_module():
    """``setup.py`` as a module (its ``setup()`` call only runs as ``__main__``)."""
    spec = importlib.util.spec_from_file_location('aimmd_setup_py', ROOT / 'setup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_git(answer):
    def run(args, **kwargs):
        if isinstance(answer, BaseException):
            raise answer
        code, out, err = answer
        return subprocess.CompletedProcess(args, code, out, err)
    return run


class _Distribution:
    """An ``importlib.metadata`` distribution stand-in with pip's install record `record`."""

    def __init__(self, root, record):
        self.root, self.record, self.version = Path(root), record, '0.1.0'

    def locate_file(self, name):
        return self.root / name

    def read_text(self, name):
        return None if self.record is None else json.dumps(self.record)


def _installed(tmp_path, monkeypatch, install_record, build_record):
    package = tmp_path / 'site-packages' / 'aimmd'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    if build_record is not None:
        (package / '_build_record.json').write_text(json.dumps(build_record))
    monkeypatch.setattr(sidecar, '_package_dir', lambda: str(package))
    monkeypatch.setattr(sidecar.importlib.metadata, 'distributions',
                        lambda name=None: [_Distribution(package.parent, install_record)])
    monkeypatch.setattr(sidecar.subprocess, 'run', _fake_git(NOT_TRACKED))
    return package


WHEEL_FILE = {'url': 'file:///wheels/aimmd_lab-0.1.0-py3-none-any.whl', 'archive_info': {}}
RECORD = {'schema': 'aimmd.build_record/1', 'git_commit': 'ab' * 20, 'git_dirty': False,
          'git_reason': None, 'built_at': '2026-11-06T10:00:00+00:00',
          'built_from': 'git checkout'}


def test_sidecar_reads_the_build_record(tmp_path, monkeypatch, fresh_git_provenance):
    """An install from a wheel file: no git, no commit in pip's record -> the build record's."""
    _installed(tmp_path, monkeypatch, WHEEL_FILE, RECORD)
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] == 'ab' * 20
    assert provenance['git_source'] == 'build record'
    assert provenance['git_dirty'] is False
    assert '_build_record.json' in provenance['git_reason']
    assert 'not tracked' in provenance['git_reason']
    assert sidecar.aimmd_provenance()['git_commit'] == 'ab' * 20


def test_pips_install_record_comes_before_the_build_record(tmp_path, monkeypatch,
                                                           fresh_git_provenance):
    """An install from git names its commit in direct_url.json, which is read first."""
    from_git = {'url': 'git+file:///clone', 'vcs_info': {'vcs': 'git', 'commit_id': 'cd' * 20}}
    _installed(tmp_path, monkeypatch, from_git, RECORD)
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] == 'cd' * 20 and provenance['git_source'] == 'direct_url'


@pytest.mark.parametrize('record, reason', [
    (None, 'has no build record'),
    ({**RECORD, 'git_commit': None, 'git_reason': 'built from a folder that is not a git '
                                                  'checkout of AIMMD-Lab'},
     'names no commit (built from a folder that is not a git checkout'),
])
def test_no_commit_without_a_usable_build_record(tmp_path, monkeypatch, fresh_git_provenance,
                                                 record, reason):
    _installed(tmp_path, monkeypatch, WHEEL_FILE, record)
    provenance = sidecar.git_provenance()
    assert provenance['git_commit'] is None and provenance['git_source'] is None
    assert reason in provenance['git_reason'] and 'archive' in provenance['git_reason']


def _git(folder, *args):
    return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                          cwd=folder, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_build_record_of_a_checkout(tmp_path):
    """setup.py's record: the commit of the checkout and the dirty flag of aimmd/ (bytecode aside)."""
    setup_py = _setup_module()
    checkout = tmp_path / 'checkout'
    (checkout / 'aimmd').mkdir(parents=True)
    (checkout / 'aimmd' / '__init__.py').write_text('')
    _git(checkout, 'init', '-q')
    _git(checkout, 'add', '.')
    _git(checkout, 'commit', '-q', '-m', 'aimmd')
    record = setup_py.build_record(str(checkout))
    assert record['git_commit'] == _git(checkout, 'rev-parse', 'HEAD')
    assert record['git_dirty'] is False and record['built_from'] == 'git checkout'
    (checkout / 'aimmd' / '__pycache__').mkdir()
    (checkout / 'aimmd' / '__pycache__' / 'x.cpython-313.pyc').write_bytes(b'')
    assert setup_py.build_record(str(checkout))['git_dirty'] is False
    (checkout / 'aimmd' / 'new.py').write_text('')
    assert setup_py.build_record(str(checkout))['git_dirty'] is True
    # a folder that is not a checkout: no commit, and why
    plain = tmp_path / 'plain'
    (plain / 'aimmd').mkdir(parents=True)
    record = setup_py.build_record(str(plain))
    assert record['git_commit'] is None and record['built_from'] == 'folder'
    assert 'not a git checkout' in record['git_reason']


def _source_tree(target):
    """The files a build of AIMMD-Lab needs, copied from this checkout into a new git repository."""
    target.mkdir()
    for name in ('setup.py', 'pyproject.toml', 'README.md', 'LICENSE'):
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / 'aimmd', target / 'aimmd',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '_build_record.json'))
    _git(target, 'init', '-q')
    _git(target, 'add', '.')
    _git(target, 'commit', '-q', '-m', 'source')
    return _git(target, 'rev-parse', 'HEAD')


def _pip(*args, cwd):
    subprocess.run([sys.executable, '-m', 'pip', *args, '--no-deps', '--no-index',
                    '--no-cache-dir', '--disable-pip-version-check', '-q'],
                   cwd=cwd, check=True, capture_output=True, text=True)


def _wheel_record(dist):
    (wheel,) = sorted(Path(dist).glob('aimmd_lab-*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        return wheel, json.loads(archive.read('aimmd/_build_record.json'))


@pytest.mark.slow                                            # three pip builds: about 30 s
@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_wheels_and_source_distributions_carry_the_commit(tmp_path):
    """A wheel built from a checkout, and one built from its sdist, record the checkout's commit;
    installed from the wheel file, the sidecars read it from there."""
    source = tmp_path / 'source'
    commit = _source_tree(source)

    # wheel from the checkout
    _pip('wheel', '--no-build-isolation', '-w', str(tmp_path / 'dist'), str(source), cwd=tmp_path)
    wheel, record = _wheel_record(tmp_path / 'dist')
    assert record['git_commit'] == commit and record['git_dirty'] is False
    assert record['built_from'] == 'git checkout' and record['schema'] == 'aimmd.build_record/1'
    assert not (source / 'aimmd' / '_build_record.json').exists()   # never in the source tree

    # sdist from the checkout, then a wheel from the unpacked sdist (as python -m build does)
    subprocess.run([sys.executable, '-c', 'import sys; from setuptools import build_meta; '
                                          'print(build_meta.build_sdist(sys.argv[1]))',
                    str(tmp_path / 'sdist')], cwd=source, check=True, capture_output=True)
    (sdist,) = sorted((tmp_path / 'sdist').glob('aimmd_lab-*.tar.gz'))
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / 'unpacked', filter='data')
    (unpacked,) = sorted((tmp_path / 'unpacked').iterdir())
    assert json.loads((unpacked / 'aimmd' / '_build_record.json').read_text())[
        'git_commit'] == commit
    _pip('wheel', '--no-build-isolation', '-w', str(tmp_path / 'dist_from_sdist'),
         str(unpacked), cwd=tmp_path)
    _, record = _wheel_record(tmp_path / 'dist_from_sdist')
    assert record['git_commit'] == commit and record['built_from'] == 'source distribution'

    # installed from the wheel file: no git, pip records no commit, the build record does
    _pip('install', '--target', str(tmp_path / 'target'), str(wheel), cwd=tmp_path)
    out = subprocess.run(
        [sys.executable, '-c', 'import json; from aimmd.cgaimmd import sidecar; '
                               'print(json.dumps(sidecar.git_provenance()))'],
        cwd=tmp_path, check=True, capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(tmp_path / 'target')}).stdout
    provenance = json.loads(out.strip().splitlines()[-1])
    assert provenance['git_commit'] == commit and provenance['git_source'] == 'build record'
