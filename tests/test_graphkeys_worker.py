"""Worker side of graph keys: registration, value passes, trainer verify.

- ``register_path`` copies the per-frame network series of the back and
  forw halves into the registered path: graph keys in graph-key runs
  (never opening a ``*.descriptors.npy``), descriptors otherwise. A half
  that lacks the series gives zero key rows (repaired on first use), and
  no longer crashes for descriptors.

Runs without torch_geometric (dict-backed, key-aware toy cache).
"""

import os

import numpy as np
import pytest

from aimmd import Path
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.core.graphkey import graph_keys, pad_keys
from aimmd.network import graph_keys as gk
from aimmd.network import graph_lookup
from aimmd.pathensemble import PathEnsemble
from aimmd.worker.utils import register_path
from tests._helpers_graphkeys import (ToyCache, descriptors_function,
                                      forbid_descriptor_files)
from tests._helpers_unit import write_trajectory


@pytest.fixture(autouse=True)
def _clean_state():
    NPY_CACHE.clear()
    gk.reset_repair_stats()
    yield
    assert graph_lookup._OVERLAYS == [], 'an overlay leaked out of its block'
    NPY_CACHE.clear()


def _half(folder, stem, n_frames, seed, keys=True, descriptors=False,
          key_rows=None):
    """A shooting half: trajectory, states and (optionally) its series."""
    # whole Angstroms: Path.write (registration) re-encodes them exactly
    rng = np.random.default_rng(seed)
    positions = rng.integers(1, 10, (n_frames, 2, 3)).astype(np.float32)
    fname = write_trajectory(folder, stem=stem, positions=positions)
    rows = np.asarray(Path(fname).coordinates, dtype=np.float32)
    save_npy(f'{fname}.states.npy', np.full(n_frames, 'R', dtype='<U1'))
    if keys:
        stored = graph_keys(rows)
        save_npy(f'{fname}.graphkeys.npy',
                 stored if key_rows is None else stored[:key_rows])
    if descriptors:
        save_npy(f'{fname}.descriptors.npy', rows)
    return fname, rows


def _shot(folder, n_back=5, n_forw=4, **kwargs):
    """back/forw halves and the path _shoot assembles from them."""
    back_kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith('back_')}
    forw_kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith('forw_')}
    back, back_rows = _half(folder, 'back', n_back, 1, **back_kwargs)
    forw, forw_rows = _half(folder, 'forw', n_forw, 2, **forw_kwargs)
    path = Path(back)[n_back - 1::-1] + Path(forw)[1:]
    rows = np.concatenate([back_rows[::-1], forw_rows[1:]])
    return path, rows


def test_register_path_copies_key_rows_and_never_opens_descriptors(
        tmp_path, monkeypatch):
    path, rows = _shot(tmp_path, back_descriptors=True, forw_descriptors=True)
    chain = PathEnsemble()
    with forbid_descriptor_files(monkeypatch) as opened:
        register_path(path, chain, frame_series=('graphkeys',))
    assert opened == []
    fname = chain[0].fname
    assert fname.endswith('path000001.xtc')
    assert not os.path.exists(f'{fname}.descriptors.npy')
    stored = np.load(f'{fname}.graphkeys.npy')
    assert np.array_equal(stored, graph_keys(rows))
    # the keys still match the frames of the re-written trajectory
    written = np.asarray(Path(fname).coordinates, dtype=np.float32)
    assert np.array_equal(graph_keys(written), stored)


def test_register_path_of_a_single_half(tmp_path):
    back, rows = _half(tmp_path, 'back', 5, 1)
    path = Path(back)[::-1]
    chain = PathEnsemble()
    register_path(path, chain, frame_series=('graphkeys',))
    stored = np.load(f'{chain[0].fname}.graphkeys.npy')
    assert np.array_equal(stored, graph_keys(rows[::-1]))


@pytest.mark.parametrize('back, forw', [
    ({'back_keys': False}, {}),
    ({}, {'forw_key_rows': 2}),
    ({'back_key_rows': 0}, {'forw_keys': False}),
])
def test_register_path_zero_fills_missing_key_halves(tmp_path, back, forw):
    """A half keyed only partly (e.g. in flight at the switch) gives zeros."""
    path, rows = _shot(tmp_path, **back, **forw)
    chain = PathEnsemble()
    register_path(path, chain, frame_series=('graphkeys',))
    fname = chain[0].fname
    stored = np.load(f'{fname}.graphkeys.npy')
    expected = graph_keys(rows)
    assert stored.shape == expected.shape
    keyed = stored.any(axis=1)
    assert not keyed.all()
    assert np.array_equal(stored[keyed], expected[keyed])

    # the first value pass keys the zero rows and finds the graphs
    toy = ToyCache()
    keys_function = gk.GraphKeysFunction(descriptors_function, toy.transform)
    keyed_values = gk.KeyedFunction(toy.values, keys_function)
    values = chain[0].compute(keyed_values, '', 'graphkeys')
    written = np.asarray(Path(fname).coordinates, dtype=np.float32)
    assert np.array_equal(values, toy.values(written))
    NPY_CACHE.clear()
    assert np.array_equal(np.load(f'{fname}.graphkeys.npy'), expected)


@pytest.mark.parametrize('back, forw', [
    ({}, {'forw_descriptors': True}),            # was: None[back_indices]
    ({'back_descriptors': True}, {}),
])
def test_register_path_with_half_missing_descriptors_does_not_crash(
        tmp_path, capsys, back, forw):
    path, rows = _shot(tmp_path, back_keys=False, forw_keys=False,
                       **back, **forw)
    chain = PathEnsemble()
    register_path(path, chain)                   # default: ('descriptors',)
    fname = chain[0].fname
    assert not os.path.exists(f'{fname}.descriptors.npy')
    assert 'descriptors' in capsys.readouterr().out
    # nothing half-written: the next ledger pass computes the whole path
    assert chain[0].compute(descriptors_function, 'descriptors') == len(rows)
    NPY_CACHE.clear()
    assert np.array_equal(np.load(f'{fname}.descriptors.npy'), rows)


def test_register_path_with_short_descriptors_does_not_crash(tmp_path):
    path, rows = _shot(tmp_path, back_keys=False, forw_keys=False,
                       back_descriptors=True, forw_descriptors=True)
    forw = path._fnames[1]
    save_npy(f'{forw}.descriptors.npy',
             np.load(f'{forw}.descriptors.npy')[:2])
    chain = PathEnsemble()
    register_path(path, chain)
    assert not os.path.exists(f'{chain[0].fname}.descriptors.npy')


def test_register_path_npy_mode_is_unchanged(tmp_path, capsys):
    path, rows = _shot(tmp_path, back_keys=False, forw_keys=False,
                       back_descriptors=True, forw_descriptors=True)
    chain = PathEnsemble()
    register_path(path, chain)
    fname = chain[0].fname
    assert np.array_equal(np.load(f'{fname}.descriptors.npy'), rows)
    assert not os.path.exists(f'{fname}.graphkeys.npy')
    assert 'descriptors' not in capsys.readouterr().out.replace(
        'path000001', '')


def test_register_path_without_any_series_is_quiet(tmp_path, capsys):
    """Runs without descriptors_function have no series to copy."""
    path, _ = _shot(tmp_path, back_keys=False, forw_keys=False)
    chain = PathEnsemble()
    register_path(path, chain)
    assert not os.path.exists(f'{chain[0].fname}.descriptors.npy')
    assert 'descriptors' not in capsys.readouterr().out


@pytest.mark.parametrize('cache, series', [('npy', ('descriptors',)),
                                           ('graphkeys', ('graphkeys',))])
def test_shoot_registers_the_series_of_the_run(monkeypatch, tmp_path, cache,
                                               series):
    """The shooting task hands register_path the run's network series."""
    import aimmd
    from aimmd.worker._shoot import WorkerShoot
    from tests._helpers_unit import build_path

    class Shooter(WorkerShoot):
        def __init__(self, params, initial_paths, root):
            self.params = params
            self.initial_paths = initial_paths
            self.directory = self._directory = str(root)
            self.must_stop = False
            self.total_steps = self.total_frames = 0
            self._location = self.log_file = self.original_stdout = ''

    initial = build_path(tmp_path, stem='initial', positions=np.array(
        [[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32))
    chain = PathEnsemble()
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        descriptor_cache=cache, descriptors_function=descriptors_function,
        states='ARB', chain_type='rfps', nbins=1, max_length=10,
        selection_pool_size=2, free_overriding_states='', engine='toy',
        check_if_initialized=lambda *deffnms: False,
        shot_chains=lambda directory, t, k=None: chain,
        shot_paths=lambda directory, prefix, t, k=None: chain,
        free_trajectories=lambda directory, old=None: [],
        initialize_simulation=lambda shooting_point, *deffnms: None)
    worker = Shooter(params, PathEnsemble(initial), tmp_path)
    monkeypatch.setattr('aimmd.worker._shoot.update_selection_pool',
                        lambda *args, **kwargs: PathEnsemble())
    monkeypatch.setattr('aimmd.worker._shoot.select_shooting_point',
                        lambda *args, **kwargs: initial[1:2])
    monkeypatch.setattr('aimmd.worker._shoot.remove', lambda *args: None)
    back = build_path(tmp_path, stem='back_seg')
    forw = build_path(tmp_path, stem='forw_seg')
    results = iter([(0, len(back), 'A', len(back)),
                    (0, len(forw), 'B', len(forw))])
    monkeypatch.setattr(worker, '_simulate',
                        lambda *args, **kwargs: next(results), raising=False)
    received = []

    def register(path, chain_, eneconv, **kwargs):
        received.append(kwargs.get('frame_series'))
        chain_.append(path)
        worker.must_stop = True

    monkeypatch.setattr('aimmd.worker._shoot.register_path', register)
    paths = iter([back, forw, aimmd.Path(), aimmd.Path()])
    monkeypatch.setattr('aimmd.worker._shoot.Path', lambda *args, **kwargs:
                        next(paths) if not args else aimmd.Path(*args, **kwargs))
    worker._shoot(target_state='R', k=0, sweep=False)
    assert received == [series]
