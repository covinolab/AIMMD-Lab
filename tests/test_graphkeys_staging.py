"""The staging deadline of the /dev/shm graph-cache replica.

Graph-key runs keep the shared graph cache, so its replica keeps being
staged at every trainer start, under a wall-clock deadline. A fixed 300 s
deadline gives up on a cache of more than about 55 GB at the 183 MB/s
measured on JUPITER, about 20 job-days of a production campaign; every
lookup then goes to the shared filesystem. By default the deadline now
grows with the cache, ``max(300 s, size / 50 MB/s)``, and an explicit
``AIMMD_STAGE_DEADLINE`` still wins.

shm_cache imports only the standard library, so these run in the default
suite.
"""

import os
import subprocess
import sys

import pytest

from aimmd.network import shm_cache
from tests.test_shm_cache import _isolate, _make_cache          # noqa: F401


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def scaled(monkeypatch):
    """No explicit deadline: the size-scaled default applies."""
    monkeypatch.setattr(shm_cache, '_STAGE_DEADLINE_SECONDS', None)


def test_deadline_is_300_s_up_to_15_gb(scaled):
    assert shm_cache.stage_deadline(0) == 300.0
    assert shm_cache.stage_deadline(6.87e9) == 300.0     # trial3 today
    assert shm_cache.stage_deadline(15e9) == 300.0


def test_deadline_grows_with_the_cache_beyond(scaled):
    assert shm_cache.stage_deadline(55e9) == pytest.approx(1100.0)
    assert shm_cache.stage_deadline(100e9) == pytest.approx(2000.0)
    assert (shm_cache.stage_deadline(60e9)
            > shm_cache.stage_deadline(50e9) > 300.0)


def test_an_explicit_deadline_wins(monkeypatch):
    monkeypatch.setattr(shm_cache, '_STAGE_DEADLINE_SECONDS', 120.0)
    assert shm_cache.stage_deadline(0) == 120.0
    assert shm_cache.stage_deadline(1e12) == 120.0


def _imported(env):
    """``_STAGE_DEADLINE_SECONDS`` and the deadline of 1 TB, in a fresh
    interpreter with ``env``."""
    code = ('from aimmd.network import shm_cache as s; '
            'print(s._STAGE_DEADLINE_SECONDS, s.stage_deadline(1e12))')
    environ = {k: v for k, v in os.environ.items()
               if k != 'AIMMD_STAGE_DEADLINE'}
    environ.update(env, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES='')
    out = subprocess.run([sys.executable, '-W', 'ignore', '-c', code],
                         env=environ, capture_output=True, text=True,
                         check=True, timeout=300)
    return out.stdout.split()[-2:]


def test_the_environment_variable_sets_the_deadline():
    assert _imported({}) == ['None', '20000.0']
    assert _imported({'AIMMD_STAGE_DEADLINE': ''}) == ['None', '20000.0']
    assert _imported({'AIMMD_STAGE_DEADLINE': '42'}) == ['42.0', '42.0']


def test_staging_uses_the_deadline_of_the_cache_size(tmp_path, monkeypatch,
                                                     scaled):
    conn = _make_cache(tmp_path / 'c.sqlite',
                       {chr(97 + i): i for i in range(6)})
    nbytes = os.path.getsize(tmp_path / 'c.sqlite')
    if os.path.exists(f'{tmp_path}/c.sqlite-wal'):
        nbytes += os.path.getsize(f'{tmp_path}/c.sqlite-wal')
    asked, given = [], []
    real_deadline = shm_cache.stage_deadline
    real_copy = shm_cache._snapshot_copy

    def deadline(size):
        asked.append(size)
        return real_deadline(size)

    def copy(conn, db_path, partial, deadline_s):
        given.append(deadline_s)
        return real_copy(conn, db_path, partial, deadline_s)

    monkeypatch.setattr(shm_cache, 'stage_deadline', deadline)
    monkeypatch.setattr(shm_cache, '_snapshot_copy', copy)
    assert shm_cache.stage_cache(conn) is not None
    assert asked == [nbytes]
    assert given == [300.0]


def test_a_blown_deadline_names_it(tmp_path, monkeypatch, capsys, scaled):
    conn = _make_cache(tmp_path / 'c.sqlite', {'a': 1})
    monkeypatch.setattr(shm_cache, 'stage_deadline', lambda size: 0.0)
    assert shm_cache.stage_cache(conn) is None
    assert 'within 0s' in capsys.readouterr().out
