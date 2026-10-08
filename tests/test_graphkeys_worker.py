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


# --------------------------------------------------------- ensure source --
def _transition(folder, stem, n_frames=7, shift=0.0, series=()):
    """An A -> B transition (only its end frames in A and B), with series."""
    x = np.linspace(-1.0, 1.0, n_frames)
    x[1:-1] = np.linspace(-0.45, 0.45, n_frames - 2)
    positions = np.zeros((n_frames, 2, 3), dtype=np.float32)
    positions[:, 0, 0] = x
    positions[:, 1] = [3.0 + shift, 4.0, 5.0]
    fname = write_trajectory(folder, stem=stem, positions=positions)
    states = np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))
    save_npy(f'{fname}.states.npy', states.astype('<U1'))
    rows = np.asarray(Path(fname).coordinates, dtype=np.float32)
    if 'descriptors' in series:
        save_npy(f'{fname}.descriptors.npy', rows)
    if 'graphkeys' in series:
        save_npy(f'{fname}.graphkeys.npy', graph_keys(rows))
    return fname, rows


def _value_params(cache, toy, bins):
    import aimmd
    params = aimmd.Params.placeholder.copy()
    params.__dict__.update(
        states='ARB', chain_type='tps', selection_pool_size=1,
        nbins=len(bins) - 1, descriptor_cache=cache,
        descriptors_function=descriptors_function,
        descriptor_transform=toy.transform, values_function=toy.values,
        _default_values_function=False, network_batch_size=4096)
    params.__dict__['update_network'] = lambda *args, **kwargs: None
    params.__dict__['load_bins_and_densities'] = (
        lambda *args, **kwargs: (bins.copy(), np.arange(1.0, len(bins))))
    return params


# the series each mode reads, and the series a path of the other mode has
_OTHER = {'npy': ('graphkeys',), 'graphkeys': ('descriptors',)}
_OWN = {'npy': ('descriptors',), 'graphkeys': ('graphkeys',)}


def _tps_chain(folder, cache, leading_series):
    folder.mkdir(parents=True, exist_ok=True)
    leading, _ = _transition(folder, 'path000001', 7, 0.0, leading_series)
    current, _ = _transition(folder, 'path000002', 9, 1.0, _OWN[cache])
    chain = PathEnsemble([Path(leading, shooting_index=3),
                          Path(current, shooting_index=4)])
    return chain


@pytest.mark.parametrize('cache', ['npy', 'graphkeys'])
def test_tps_acceptance_on_a_leading_path_of_the_other_mode(
        tmp_path, monkeypatch, capsys, cache):
    """The R7 crash: no values for the leading path, then an IndexError."""
    toy = ToyCache()
    bins = np.array([-np.inf, 10.0, 12.0, 14.0, np.inf])
    params = _value_params(cache, toy, bins)
    monkeypatch.setattr(np.random, 'random', lambda: 0.0)   # always accept

    # reference: the leading path has its series
    chain = _tps_chain(tmp_path / 'ref', cache, _OWN[cache])
    from aimmd.worker.utils import accept_or_reject_last_path
    accept_or_reject_last_path(chain, params)
    reference = [line for line in capsys.readouterr().out.splitlines()
                 if 'acceptance probability' in line]

    # the leading path was registered in the other mode
    NPY_CACHE.clear()
    chain = _tps_chain(tmp_path / 'switched', cache, _OTHER[cache])
    accept_or_reject_last_path(chain, params)
    lines = [line for line in capsys.readouterr().out.splitlines()
             if 'acceptance probability' in line]
    assert lines == reference and len(lines) == 1
    assert chain[-1].weight == 1.0
    # the leading path's internal frames have their series now
    leading = chain[0].fname
    rows = np.asarray(Path(leading).coordinates, dtype=np.float32)
    internal = slice(1, len(rows) - 1)
    NPY_CACHE.clear()
    if cache == 'npy':
        stored = np.load(f'{leading}.descriptors.npy')
        assert np.array_equal(stored[internal], rows[internal])
    else:
        stored = np.load(f'{leading}.graphkeys.npy')
        assert np.array_equal(stored[internal], graph_keys(rows)[internal])


@pytest.mark.parametrize('cache', ['npy', 'graphkeys'])
def test_pool_selection_on_a_path_of_the_other_mode(tmp_path, cache):
    from aimmd.worker.utils import select_shooting_point
    toy = ToyCache()
    bins = np.array([-np.inf, 10.0, 12.0, 14.0, np.inf])
    params = _value_params(cache, toy, bins)
    params.__dict__['chain_type'] = 'rfps'
    folder = tmp_path / 'chainR0'
    folder.mkdir()
    fname, rows = _transition(folder, 'path000001', 7, 0.0, _OTHER[cache])
    pool = PathEnsemble(Path(fname, shooting_index=3))
    np.random.seed(0)
    point = select_shooting_point(pool, params, str(folder), target_state='R')
    assert point.n_atoms == 2
    NPY_CACHE.clear()
    values = np.load(f'{fname}.values.npy')
    assert np.array_equal(values, toy.values(rows))


@pytest.mark.parametrize('cache', ['npy', 'graphkeys'])
def test_shared_density_never_writes_another_workers_halves(tmp_path,
                                                            cache):
    """The shared density reads the in-flight back halves of the other
    workers (frame 0), but must never write their series files.

    Those files belong to the worker that runs the shot: it removes them
    when the shot ends and recreates them for the next one. A row written
    by another worker from the old trajectory would survive into the next
    shot's file and be taken as computed, a wrong row for its first frame.
    Without the series, the frame is left out of the shared density, as
    in npy runs before graph keys.
    """
    from aimmd.worker.utils import select_shooting_point
    toy = ToyCache()
    bins = np.array([-np.inf, 10.0, 12.0, 14.0, np.inf])
    params = _value_params(cache, toy, bins)
    params.__dict__.update(chain_type='rfps', shared_density_adjustment=True)
    own = tmp_path / 'chainR0'
    own.mkdir()
    fname, _ = _transition(own, 'path000001', 7, 0.0, _OWN[cache])
    pool = PathEnsemble(Path(fname, shooting_index=3))
    other = tmp_path / 'chainR1'
    other.mkdir()
    back, _ = _transition(other, 'back', 5, 2.0)       # no series yet
    np.random.seed(0)
    point = select_shooting_point(pool, params, str(own),
                                  shooting_chains=[pool],
                                  target_state='R')
    assert point.n_atoms == 2
    written = sorted(p.name for p in other.glob('back.xtc.*.npy'))
    assert written == ['back.xtc.states.npy'], written


def test_ensure_source_is_a_no_op_without_descriptors(tmp_path):
    import aimmd
    from aimmd.worker.utils import ensure_source
    fname, _ = _transition(tmp_path, 'path000001')
    params = aimmd.Params.placeholder
    assert ensure_source(PathEnsemble(Path(fname)), params) == 0
    assert not os.path.exists(f'{fname}.descriptors.npy')


# --------------------------------------------------------- trainer verify --
def _keyed_path(folder, toy, n_frames=6, stem='path000001', seed=7):
    rng = np.random.default_rng(seed)
    positions = rng.integers(1, 10, (n_frames, 2, 3)).astype(np.float32)
    fname = write_trajectory(folder, stem=stem, positions=positions)
    rows = np.asarray(Path(fname).coordinates, dtype=np.float32)
    toy.cache(rows)
    save_npy(f'{fname}.graphkeys.npy', graph_keys(rows))
    return fname, rows


def _keys_params(toy, cache='graphkeys'):
    from types import SimpleNamespace
    keys_function = gk.GraphKeysFunction(descriptors_function, toy.transform)
    return SimpleNamespace(descriptor_cache=cache,
                           descriptors_function=descriptors_function,
                           graphkeys_function=keys_function)


def test_trainer_verify_repairs_injected_missing_graphs(tmp_path, capsys,
                                                        monkeypatch):
    from aimmd.worker._train import _verify_graph_keys
    monkeypatch.delenv('AIMMD_GRAPHKEYS_VERIFY', raising=False)
    toy = ToyCache()
    fname, rows = _keyed_path(tmp_path, toy)
    other, _ = _keyed_path(tmp_path, toy, 4, 'path000002', seed=8)
    hexes = gk.keys_to_hex(graph_keys(rows))
    for h in hexes[1:3]:
        del toy.store[h]                            # lost graphs
    stored = np.load(f'{other}.graphkeys.npy')
    stored[0] = 0                                   # a frame never keyed
    save_npy(f'{other}.graphkeys.npy', stored)
    NPY_CACHE.clear()

    ensemble = PathEnsemble([Path(fname), Path(other)])
    _verify_graph_keys(_keys_params(toy), ensemble)
    out = capsys.readouterr().out
    assert '10 frame(s) checked' in out
    assert '1 without a key, 2 without a graph; 3 repaired (0 stale)' in out
    assert all(h in toy.store for h in hexes)
    NPY_CACHE.clear()
    assert np.array_equal(np.load(f'{other}.graphkeys.npy'), graph_keys(
        np.asarray(Path(other).coordinates, dtype=np.float32)))


def test_trainer_verify_is_silent_in_npy_runs_and_when_off(tmp_path, capsys,
                                                           monkeypatch):
    from aimmd.worker._train import _verify_graph_keys
    toy = ToyCache()
    fname, rows = _keyed_path(tmp_path, toy)
    del toy.store[gk.keys_to_hex(graph_keys(rows[:1]))[0]]
    _verify_graph_keys(_keys_params(toy, cache='npy'), Path(fname))
    monkeypatch.setenv('AIMMD_GRAPHKEYS_VERIFY', '0')
    _verify_graph_keys(_keys_params(toy), Path(fname))
    assert capsys.readouterr().out == ''
    assert len(toy.store) == len(rows) - 1


@pytest.mark.graph
def test_trainer_verify_in_reader_role_defers_the_write(tmp_path, monkeypatch,
                                                        capsys):
    """Graphs the trainer repairs wait in the pending backlog for the flush."""
    from types import SimpleNamespace
    from aimmd.network import shm_cache
    from aimmd.worker._train import _verify_graph_keys, _flush_graph_backlog
    from tests.test_graphkeys_repair import (_GraphParams, _graph_trajectory,
                                             _import_graph_utils)
    monkeypatch.delenv('AIMMD_GRAPHKEYS_VERIFY', raising=False)
    monkeypatch.setenv('AIMMD_SHM_DIR', 'off')
    gu = _import_graph_utils()
    conn = gu.init_db(str(tmp_path / 'g.sqlite'))
    functions = _GraphParams(gu, {None: conn})
    keys_function = gk.GraphKeysFunction(functions.descriptors_function,
                                         functions.descriptor_transform)
    fname = _graph_trajectory(tmp_path, 5)
    keys = keys_function(Path(fname).reader)           # writer role: stored
    save_npy(f'{fname}.graphkeys.npy', keys)
    lost = gk.keys_to_hex(keys[2:4])
    conn.executemany('DELETE FROM graphs_cache WHERE key = ?',
                     [(h,) for h in lost])
    conn.commit()
    conn._aimmd_memo.clear()

    shm_cache.set_reader_role()
    params = SimpleNamespace(descriptor_cache='graphkeys',
                             descriptors_function=functions.descriptors_function,
                             graphkeys_function=keys_function)
    _verify_graph_keys(params, Path(fname))
    assert '2 without a graph; 2 repaired' in capsys.readouterr().out
    count = 'SELECT COUNT(*) FROM graphs_cache'
    assert conn.execute(count).fetchone()[0] == 3, 'no write in reader role'
    assert shm_cache.pending_count(conn) == 2
    _flush_graph_backlog()
    assert conn.execute(count).fetchone()[0] == 5
    assert gu.graphs_present(keys, conn).all()
    conn.close()


def test_ensure_source_reads_no_complete_descriptor_file(tmp_path,
                                                         monkeypatch):
    """npy runs: complete descriptor files (many GB in production) stay
    unread; missing and short ones are computed."""
    from aimmd.worker.utils import ensure_source
    toy = ToyCache()
    params = _value_params('npy', toy, np.array([-np.inf, np.inf]))
    complete, rows = _transition(tmp_path, 'complete', 7, 0.0, ('descriptors',))
    missing, _ = _transition(tmp_path, 'missing', 6, 1.0)
    short, short_rows = _transition(tmp_path, 'short', 5, 2.0, ('descriptors',))
    save_npy(f'{short}.descriptors.npy', short_rows[:2])
    read = []
    for name in ('get', 'load', 'pop'):
        original = getattr(NPY_CACHE, name)

        def spy(fname, *args, _original=original, **kwargs):
            read.append(str(fname))
            return _original(fname, *args, **kwargs)

        monkeypatch.setattr(NPY_CACHE, name, spy)
    ensemble = PathEnsemble([Path(complete), Path(missing), Path(short)])
    assert ensure_source(ensemble, params) == 6 + 3
    assert not any(name.startswith(complete) for name in read), read
    assert ensure_source(Path(complete)[1:-1], params) == 0
    assert not any(name.startswith(complete) for name in read), read
    monkeypatch.undo()
    NPY_CACHE.clear()
    for fname in (missing, short):
        assert np.array_equal(np.load(f'{fname}.descriptors.npy'),
                              np.asarray(Path(fname).coordinates,
                                         dtype=np.float32))
