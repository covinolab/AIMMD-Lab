"""The one write protocol of C-GAIMMD code, ``aimmd.cgaimmd.io``."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from aimmd.cgaimmd import io as cgaimmd_io


def _leftovers(directory):
    return sorted(p.name for p in Path(directory).iterdir() if '.tmp.' in p.name)


def test_atomic_write_bytes_replace_and_prev(tmp_path, monkeypatch):
    fname = str(tmp_path / 'state.pt')
    cgaimmd_io.atomic_write_bytes(fname, b'one')
    assert Path(fname).read_bytes() == b'one'
    assert not os.path.exists(fname + '.prev')

    cgaimmd_io.atomic_write_bytes(fname, b'two', keep_previous=True)
    assert Path(fname).read_bytes() == b'two'
    assert Path(fname + '.prev').read_bytes() == b'one'

    cgaimmd_io.atomic_write_bytes(fname, b'three')              # no keep_previous: .prev untouched
    assert Path(fname).read_bytes() == b'three'
    assert Path(fname + '.prev').read_bytes() == b'one'
    assert _leftovers(tmp_path) == []

    # the temporary file is f'{fname}.tmp.{pid}', fsync'ed before the rename
    seen = {}
    real_replace, real_fsync = os.replace, os.fsync

    def spy_replace(src, dst):
        seen.setdefault('replace', []).append((os.path.basename(src), os.path.basename(dst)))
        return real_replace(src, dst)

    def spy_fsync(fd):
        seen['fsync'] = seen.get('fsync', 0) + 1
        return real_fsync(fd)

    monkeypatch.setattr(cgaimmd_io.os, 'replace', spy_replace)
    monkeypatch.setattr(cgaimmd_io.os, 'fsync', spy_fsync)
    cgaimmd_io.atomic_write_bytes(fname, b'four', keep_previous=True)
    assert seen['replace'] == [('state.pt', 'state.pt.prev'),
                               (f'state.pt.tmp.{os.getpid()}', 'state.pt')]
    assert seen['fsync'] >= 1

    # a failing write leaves the old file and no temporary behind
    def broken_replace(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(cgaimmd_io.os, 'replace', broken_replace)
    with pytest.raises(OSError, match='disk full'):
        cgaimmd_io.atomic_write_bytes(fname, b'five')
    monkeypatch.undo()
    assert Path(fname).read_bytes() == b'four'
    assert _leftovers(tmp_path) == []


def test_atomic_write_json(tmp_path):
    fname = str(tmp_path / 'record.json')
    obj = {'b': 1, 'a': [1.5, None], 'n': np.int64(3), 'x': np.float32(0.5),
           'arr': np.arange(3), 'p': tmp_path / 'x', 'flag': np.bool_(True)}
    cgaimmd_io.atomic_write_json(fname, obj)
    text = Path(fname).read_text()
    expected = {'b': 1, 'a': [1.5, None], 'n': 3, 'x': 0.5, 'arr': [0, 1, 2],
                'p': str(tmp_path / 'x'), 'flag': True}
    assert json.loads(text) == expected
    assert text == json.dumps(expected, indent=1, sort_keys=True)
    cgaimmd_io.atomic_write_json(fname, {'v': 2}, keep_previous=True)
    assert json.loads(Path(fname + '.prev').read_text()) == expected
    with pytest.raises(TypeError):
        cgaimmd_io.atomic_write_json(fname, {'bad': object()})
    assert json.loads(Path(fname).read_text()) == {'v': 2}


def test_atomic_torch_save_returns_sha(tmp_path):
    fname = str(tmp_path / 'network.h5')
    state = {'w': torch.arange(6, dtype=torch.float32).reshape(2, 3),
             'nested': {'b': torch.tensor([1, 2], dtype=torch.int64)}}
    sha = cgaimmd_io.atomic_torch_save(state, fname, keep_previous=True)
    data = Path(fname).read_bytes()
    assert sha == hashlib.sha256(data).hexdigest()
    assert sha == cgaimmd_io.file_sha256(fname)
    loaded = torch.load(fname, weights_only=True)
    assert torch.equal(loaded['w'], state['w'])
    assert torch.equal(loaded['nested']['b'], state['nested']['b'])
    sha2 = cgaimmd_io.atomic_torch_save({'w': torch.zeros(1)}, fname, keep_previous=True)
    assert sha2 != sha
    assert cgaimmd_io.file_sha256(fname + '.prev') == sha


def test_file_sha256_retries(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(cgaimmd_io.time, 'sleep', lambda seconds: sleeps.append(seconds))

    # missing file: `retries` attempts (retries - 1 waits), then None
    assert cgaimmd_io.file_sha256(str(tmp_path / 'missing'), retries=3, wait=0.5) is None
    assert sleeps == [0.5, 0.5]
    sleeps.clear()
    assert cgaimmd_io.file_sha256(str(tmp_path / 'missing'), retries=1) is None     # a poller
    assert sleeps == []
    assert cgaimmd_io.file_sha256(str(tmp_path / 'missing')) is None                # the defaults
    assert sleeps == [0.2] * 4
    for bad in (0, -1, 2.5, True, None):
        with pytest.raises(ValueError, match='retries'):
            cgaimmd_io.file_sha256(str(tmp_path / 'missing'), retries=bad)

    fname = tmp_path / 'data.bin'
    payload = os.urandom(3 * 2 ** 20 + 7)                    # more than one read chunk
    fname.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    sleeps.clear()
    assert cgaimmd_io.file_sha256(str(fname)) == expected
    assert sleeps == []

    # a file being replaced: transient OSErrors are retried
    real_open = open
    failures = {'left': 2}

    def flaky_open(name, *args, **kwargs):
        if str(name) == str(fname) and failures['left']:
            failures['left'] -= 1
            raise OSError(116, 'Stale file handle')
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(cgaimmd_io, 'open', flaky_open, raising=False)
    assert cgaimmd_io.file_sha256(str(fname), retries=5, wait=0.2) == expected
    assert sleeps == [0.2, 0.2]

    # a persistent error is raised after the last try
    def broken_open(name, *args, **kwargs):
        raise PermissionError(13, 'Permission denied')

    sleeps.clear()
    monkeypatch.setattr(cgaimmd_io, 'open', broken_open, raising=False)
    with pytest.raises(PermissionError):
        cgaimmd_io.file_sha256(str(fname), retries=4, wait=0.1)
    assert sleeps == [0.1, 0.1, 0.1]
