"""Graph keys as a per-frame series of the path layer.

``<traj>.graphkeys.npy`` is read like the other cached series, but rows that
were never written -- a missing or short file -- come back as zero rows
("not computed") in frame order, also for reversed slices. ``Path.compute``
keeps the ledger on it, and a value pass whose source is ``'graphkeys'``
repairs missing graphs from the trajectory frames of each chunk
(``graph_keys.call_with_repair``). Runs without torch_geometric.
"""

import os

import numpy as np
import pytest

from aimmd import Path
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.core.graphkey import graph_keys, keys_to_hex, pad_keys
from aimmd.network import graph_keys as gk
from aimmd.network import graph_lookup
from aimmd.path import utils as path_utils
from tests._helpers_graphkeys import ToyCache, descriptors_function
from tests._helpers_unit import write_trajectory


def _trajectory(folder, n_frames=6, stem='traj', seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(1.0, 9.0, (n_frames, 2, 3)).astype(np.float32)
    return write_trajectory(folder, stem=stem, positions=positions)


def _rows(fname):
    return np.asarray(Path(fname).coordinates, dtype=np.float32)


def _key_file(fname):
    return f'{fname}.graphkeys.npy'


@pytest.fixture(autouse=True)
def _clean_state():
    NPY_CACHE.clear()
    gk.reset_repair_stats()
    yield
    assert graph_lookup._OVERLAYS == [], 'an overlay leaked out of its block'
    NPY_CACHE.clear()


# ------------------------------------------------------------- _extract --
def test_missing_key_file_reads_as_zero_rows(tmp_path):
    fname = _trajectory(tmp_path)
    path = Path(fname)
    for view, n in ((path, 6), (path[::-1], 6), (path[2:5], 3),
                    (path[4:0:-1], 4)):
        keys = view.graphkeys
        assert keys.dtype == np.uint8 and keys.shape == (n, 32)
        assert not keys.any()
    assert not os.path.exists(_key_file(fname)), 'reading must not create it'


def test_short_key_file_is_zero_padded_in_frame_order(tmp_path):
    fname = _trajectory(tmp_path)
    keys = graph_keys(_rows(fname))
    save_npy(_key_file(fname), keys[:3])
    path = Path(fname)
    assert np.array_equal(path.graphkeys[:3], keys[:3])
    assert not path.graphkeys[3:].any()
    # reversed views: frames 5, 4, 3 have no row yet, frames 2, 1, 0 do
    assert np.array_equal(path[::-1].graphkeys[3:], keys[2::-1])
    assert not path[::-1].graphkeys[:3].any()
    assert np.array_equal(path[4:0:-1].graphkeys[2:], keys[2:0:-1])
    assert not path[4:0:-1].graphkeys[:2].any()


def test_key_rows_across_files_and_directions(tmp_path):
    a = _trajectory(tmp_path, 5, 'a', seed=1)
    b = _trajectory(tmp_path, 4, 'b', seed=2)
    keys_a, keys_b = graph_keys(_rows(a)), graph_keys(_rows(b))
    save_npy(_key_file(a), keys_a)
    save_npy(_key_file(b), keys_b[:2])                   # short
    path = Path(a)[::-1] + Path(b)[1:]
    expected = np.concatenate([keys_a[::-1], keys_b[1:2],
                               np.zeros((2, 32), dtype=np.uint8)])
    assert np.array_equal(path.graphkeys, expected)


def test_malformed_key_file_is_an_error(tmp_path):
    fname = _trajectory(tmp_path)
    save_npy(_key_file(fname), np.ones((6, 3)))
    with pytest.raises(AttributeError, match='not a graph-key file'):
        Path(fname).graphkeys


def test_descriptors_of_a_keys_only_trajectory_are_an_error(tmp_path):
    """No silent zeros where a run caches graph keys instead of descriptors."""
    fname = _trajectory(tmp_path)
    path = Path(fname)
    assert not path.descriptors.any()        # no series at all: zeros, as before
    save_npy(_key_file(fname), graph_keys(_rows(fname)))
    NPY_CACHE.clear()
    with pytest.raises(AttributeError, match="descriptor_cache='graphkeys'"):
        Path(fname).descriptors
    with pytest.raises(RuntimeError, match='path.coordinates'):
        Path(fname)._extract(0, 'descriptors')


def test_descriptors_file_wins_over_key_file(tmp_path):
    fname = _trajectory(tmp_path)
    rows = _rows(fname)
    save_npy(f'{fname}.descriptors.npy', rows)
    save_npy(_key_file(fname), graph_keys(rows))
    assert np.array_equal(Path(fname).descriptors, rows)


# --------------------------------------------------------------- ledger --
def test_key_ledger_fills_missing_short_and_zero_rows(tmp_path):
    fname = _trajectory(tmp_path)
    keys = graph_keys(_rows(fname))
    toy = ToyCache()
    decoded = []

    def counting_descriptors_function(trajectory):
        rows = descriptors_function(trajectory)
        decoded.append(len(rows))
        return rows

    kf = gk.GraphKeysFunction(counting_descriptors_function, toy.transform)
    path = Path(fname)
    assert path.compute(kf, 'graphkeys') == 6
    assert np.array_equal(np.load(_key_file(fname)), keys)
    assert len(toy.built) == 6
    assert path.compute(kf, 'graphkeys') == 0          # nothing left to do

    # a short file with one zero row: 1 + 2 frames to decode, no graph to
    # build (they are cached)
    stored = keys[:4].copy()
    stored[1] = 0
    save_npy(_key_file(fname), stored)
    NPY_CACHE.clear()
    decoded.clear()
    assert path.compute(kf, 'graphkeys') == 3
    assert sum(decoded) == 3
    assert np.array_equal(np.load(_key_file(fname)), keys)
    assert len(toy.built) == 6


def test_key_ledger_on_reversed_slices(tmp_path):
    fname = _trajectory(tmp_path)
    keys = graph_keys(_rows(fname))
    kf = gk.GraphKeysFunction(descriptors_function, ToyCache().transform)
    path = Path(fname)
    assert path[4:1:-1].compute(kf, 'graphkeys') == 3
    stored = np.load(_key_file(fname))
    assert stored.shape == (5, 32)
    assert np.array_equal(stored[2:5], keys[2:5])
    assert not stored[:2].any()
    NPY_CACHE.clear()
    assert path[::-1].compute(kf, 'graphkeys', batch_size=2) == 3
    assert np.array_equal(np.load(_key_file(fname)), keys)


def test_key_ledger_with_conditions(tmp_path):
    fname = _trajectory(tmp_path)
    keys = graph_keys(_rows(fname))
    save_npy(f'{fname}.states.npy', np.array(list('ARRBRR'), dtype='<U1'))
    kf = gk.GraphKeysFunction(descriptors_function, ToyCache().transform)
    n = Path(fname).compute(kf, 'graphkeys',
                            conditions={'states': lambda s: s == 'R'})
    assert n == 4
    stored = pad_keys(np.load(_key_file(fname)), 6)
    in_r = np.array(list('ARRBRR')) == 'R'
    assert np.array_equal(stored[in_r], keys[in_r])
    assert not stored[~in_r].any()


# ---------------------------------------------------------- value passes --
def _keyed_values(toy, transform=None):
    kf = gk.GraphKeysFunction(descriptors_function, transform or toy.transform)
    return gk.KeyedFunction(toy.values, kf), kf


def test_value_pass_on_keys_fills_and_equals_coordinate_values(tmp_path):
    fname = _trajectory(tmp_path)
    rows = _rows(fname)
    toy = ToyCache()
    keyed, _ = _keyed_values(toy)
    path = Path(fname)
    values = path.compute(keyed, 'values', 'graphkeys', return_result=True)
    assert np.array_equal(values, toy.values(rows))
    assert np.array_equal(np.load(f'{fname}.values.npy'), values)
    # the zero rows were keyed on the way
    assert np.array_equal(np.load(_key_file(fname)), graph_keys(rows))
    assert gk.repair_stats()['filled'] == 6


def test_value_pass_repairs_a_lost_graph(tmp_path, capsys):
    fname = _trajectory(tmp_path)
    rows = _rows(fname)
    toy = ToyCache()
    keyed, kf = _keyed_values(toy)
    path = Path(fname)
    path.compute(kf, 'graphkeys')
    lost = keys_to_hex(graph_keys(rows[3:4]))[0]
    del toy.store[lost]
    values = path.compute(keyed, '', 'graphkeys')
    assert np.array_equal(values, toy.values(rows))
    assert lost in toy.store
    assert gk.repair_stats()['retries'] == 1
    assert 'repaired 1 frame(s)' in capsys.readouterr().out


def test_value_pass_on_reversed_multi_file_path_in_small_batches(tmp_path):
    a = _trajectory(tmp_path, 5, 'a', seed=1)
    b = _trajectory(tmp_path, 4, 'b', seed=2)
    toy = ToyCache()
    keyed, kf = _keyed_values(toy)
    Path(a).compute(kf, 'graphkeys')               # a keyed, b not at all
    toy.store.pop(keys_to_hex(graph_keys(_rows(a)[1:2]))[0])
    path = Path(a)[::-1] + Path(b)[1:]
    expected = toy.values(np.concatenate([_rows(a)[::-1], _rows(b)[1:]]))
    values = path.compute(keyed, '', 'graphkeys', batch_size=3)
    assert np.array_equal(values, expected)
    assert np.array_equal(pad_keys(np.load(_key_file(b)), 4)[1:],
                          graph_keys(_rows(b))[1:])


def test_value_pass_forwards_system_id(tmp_path):
    fname = _trajectory(tmp_path)
    toy = ToyCache()
    seen = []

    def transform(x, system_id=None):
        seen.append(('transform', system_id))
        return toy.transform(x)

    def values(x, system_id=None):
        seen.append(('values', system_id))
        return toy.values(x)

    kf = gk.GraphKeysFunction(descriptors_function, transform)
    keyed = gk.KeyedFunction(values, kf)
    result = Path(fname).compute(keyed, '', 'graphkeys', system_id='s1')
    assert np.array_equal(result, toy.values(_rows(fname)))
    assert ('values', 's1') in seen and ('transform', 's1') in seen
    assert all(sid == 's1' for _, sid in seen)


def test_repair_is_only_for_key_sources_and_keyed_functions(tmp_path,
                                                            monkeypatch):
    """compute_batch picks the repair by the source name, not by dtype."""
    fname = _trajectory(tmp_path)
    rows = _rows(fname)
    save_npy(f'{fname}.descriptors.npy', rows)
    toy = ToyCache()
    keyed, _ = _keyed_values(toy)

    calls = []
    real = gk.call_with_repair

    def spy(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(gk, 'call_with_repair', spy)

    # a keyed function on another source: called directly
    values = Path(fname).compute(keyed, '', 'descriptors')
    assert np.array_equal(values, toy.values(rows)) and calls == []

    # a plain function on graph keys: called directly, on the stored rows
    received = []
    Path(fname).compute(lambda x: received.append(x.copy()) or np.zeros(len(x)),
                        '', 'graphkeys')
    assert calls == [] and received[0].shape == (6, 32)
    assert not received[0].any()

    # the keyed function on graph keys: through the repair
    Path(fname).compute(keyed, '', 'graphkeys')
    assert calls == [keyed]


def test_compute_batch_without_refs_calls_the_function(tmp_path):
    """Direct callers that pass no frame refs get today's plain call."""
    keyed = gk.KeyedFunction(lambda x: np.arange(len(x)), None)
    result = path_utils.compute_batch(
        keyed, [np.zeros((3, 32), dtype=np.uint8)], [], False,
        return_result=True)
    assert np.array_equal(result, [0, 1, 2])
