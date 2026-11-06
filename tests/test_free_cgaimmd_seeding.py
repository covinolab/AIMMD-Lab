"""Free workers seeded from per-basin initial files under ``chain_type='cgaimmd'``.

A C-GAIMMD run needs no initial transition: its initial paths are typically
end-state frames, one file per basin. A free worker seeds from every initial
path, and a path with no frame in its target state (a B-only file for a free-A
worker) used to kill it. With ``skip_unusable=True``, which ``_free`` passes
under ``'cgaimmd'``, such paths are skipped and ``RuntimeError`` is raised only
when none remains; the default is the old behaviour.
"""

import numpy as np
import pytest

import aimmd
from aimmd.worker._free import WorkerFree
from aimmd.worker.utils import get_initial_frames_for_free_simulations
from tests._cgaimmd_helpers import cgaimmd_params_namespace
from tests._helpers_unit import build_path


def one_atom(x):
    positions = np.zeros((len(x), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = x
    return positions


def basin_files(tmp_path):
    a_only = build_path(tmp_path, stem='free_md_A', positions=one_atom([-1.0, -1.2, -0.9, -1.1]))
    b_only = build_path(tmp_path, stem='free_md_B', positions=one_atom([1.0, 1.2, 0.9, 1.1]))
    return a_only, b_only


# ---------------------------------------------------------------------------
# get_initial_frames_for_free_simulations
# ---------------------------------------------------------------------------

def test_skip_unusable_default_unchanged(tmp_path):
    a_only, b_only = basin_files(tmp_path)
    with pytest.raises(RuntimeError, match='must allow to extract'):
        get_initial_frames_for_free_simulations(aimmd.PathEnsemble([a_only, b_only]), 'A', 'R')
    couples = get_initial_frames_for_free_simulations(aimmd.PathEnsemble([a_only]), 'A', 'R')
    assert len(couples) == 1


@pytest.mark.parametrize('target, expected', [('A', 'free_md_A'), ('B', 'free_md_B')])
def test_skip_unusable_at_the_boundary(tmp_path, target, expected, capsys):
    a_only, b_only = basin_files(tmp_path)
    couples = get_initial_frames_for_free_simulations(
        aimmd.PathEnsemble([a_only, b_only]), target, 'R', skip_unusable=True)
    assert len(couples) == 1
    assert expected in couples[0].fname
    assert (couples[0].states == target).all()
    other = 'free_md_B' if target == 'A' else 'free_md_A'
    assert f'{other}' in capsys.readouterr().out                    # the skipped path is named


def test_skip_unusable_none_left_raises(tmp_path):
    a_only, _ = basin_files(tmp_path)
    with pytest.raises(RuntimeError, match="no initial path .* 'B'"):
        get_initial_frames_for_free_simulations(aimmd.PathEnsemble([a_only]), 'B', 'R',
                                                skip_unusable=True)


def test_skip_unusable_away_from_the_boundary(tmp_path):
    """Other positions need a transition block: paths without one are skipped."""
    transition = build_path(tmp_path, stem='transition',
                            positions=one_atom([-1.4, -1.2, -1.0, -0.2, 0.2, 1.0, 1.2]))
    _, b_only = basin_files(tmp_path)
    untrimmed = {'transition.xtc': transition, 'free_md_B.xtc': b_only}
    couples = get_initial_frames_for_free_simulations(
        aimmd.PathEnsemble([b_only, transition]), 'A', 'R', position=0.0,
        untrimmed_paths=untrimmed, states='ARB', skip_unusable=True)
    assert len(couples) == 1 and 'transition' in couples[0].fname
    np.testing.assert_allclose(couples[0].positions[-1, 0, 0], -1.4)
    with pytest.raises(ValueError, match='holds no'):
        get_initial_frames_for_free_simulations(
            aimmd.PathEnsemble([b_only, transition]), 'A', 'R', position=0.0,
            untrimmed_paths=untrimmed, states='ARB')


# ---------------------------------------------------------------------------
# the free worker
# ---------------------------------------------------------------------------

class TinyFreeWorker(WorkerFree):
    """Concrete host for `_free` (as in tests/test_worker_runtime_unit.py)."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root)
        self._directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def start_free_worker(tmp_path, monkeypatch, target, chain_type):
    """Run `_free` until its first trajectory is initialized; return what it was seeded with."""
    a_only, b_only = basin_files(tmp_path)
    init_calls = []

    def initialize_simulation(frames, deffnm):
        init_calls.append((frames, deffnm))
        worker.must_stop = True                           # stop once the first run is seeded

    params = cgaimmd_params_namespace(
        chain_type=chain_type, nbins=1, trajectory_update_batch_size=2,
        pipeline=['states', 'descriptors', 'values'], extra_free_frames=0,
        restart_free_simulations_with_transitions='',
        check_if_initialized=lambda deffnm: False,
        shot_chains=lambda directory, r, old=None: [],
        initialize_simulation=initialize_simulation)
    root = tmp_path / f'run_{target}'
    root.mkdir()
    worker = TinyFreeWorker(params, aimmd.PathEnsemble([a_only, b_only]), root)

    def fake_simulate(*args, **kwargs):
        return (None, 0, '', 0)

    monkeypatch.setattr(worker, '_simulate', fake_simulate, raising=False)
    monkeypatch.setattr('aimmd.worker._free.remove', lambda *args, **kwargs: None)
    worker._free(target_state=target, k=0, total=1, wait=False)
    return init_calls


@pytest.mark.parametrize('target', ['A', 'B'])
def test_free_seeding_per_basin_files(tmp_path, monkeypatch, target):
    init_calls = start_free_worker(tmp_path, monkeypatch, target, 'cgaimmd')
    assert len(init_calls) == 1
    frames, deffnm = init_calls[0]
    assert (frames.states == target).all()
    assert f'free_md_{target}' in frames.fname


def test_free_seeding_per_basin_files_rfps_unchanged(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='must allow to extract'):
        start_free_worker(tmp_path, monkeypatch, 'A', 'rfps')
