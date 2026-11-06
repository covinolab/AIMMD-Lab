"""What bounds a free worker whose trajectory never completes: the walltime, not nframes.

A free trajectory that stays in its state is one long engine run. The worker
counts its frames (``nframes``) when a trajectory is complete or the engine
returns, so that it stops with complete trajectories, which the trainer can
use; the walltime is checked by the engine's stop condition while the engine
runs. Bound such runs with ``walltime`` (as the GROMACS vacuum test and the
tutorials do): in vacuum NANMA a free run with ``nframes=20`` wrote 176,106
frames in 75 s.
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

import aimmd


TOY = Path(__file__).resolve().parent / 'toy_1d'

STILL = '''


def toy_mdrun(ts):
    """Never move: the free trajectory stays in its state and never completes."""
    pass
'''

WALLTIME = 6.0


@pytest.fixture
def still_toy(tmp_path, monkeypatch):
    """tests/toy_1d with an engine that never moves, in a folder of its own."""
    for name in ('params.py', 'initial.xtc'):
        shutil.copy(TOY / name, tmp_path / name)
    with open(tmp_path / 'params.py', 'a') as handle:
        handle.write(STILL)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))      # Params.load adds the folder
    return aimmd.Params.load('params.py', save=False)


def test_the_walltime_bounds_one_long_free_trajectory(still_toy, capsys):
    """nframes=20 is not reached inside the one engine run; the walltime stops it."""
    t0 = time.time()
    aimmd.Worker(still_toy, 'run', nframes=20, walltime=WALLTIME).free(0, 0, 1)
    elapsed = time.time() - t0
    out = capsys.readouterr().out
    assert 'stopping: walltime reached' in out, out[-2000:]
    assert 'frame limit reached' not in out
    assert WALLTIME <= elapsed < WALLTIME + 30
    assert 'run/freeA/traj000001 hit' in out                 # the frames read meanwhile
