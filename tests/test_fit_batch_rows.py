"""`_load_batch_descriptors` never evicts from NPY_CACHE.

A fit batch draws a few rows from many per-trajectory series files. Loading
every file of a batch whole into NPY_CACHE is right while all of them fit in
its budget; beyond that each load evicts other files, and every batch reloads
whole files to use a few rows of each. So per file of a batch:

1. a resident copy with enough rows is used, as before;
2. a file that fits in the room left in the budget is loaded whole and kept,
   as before (a run whose files fit behaves as it always did);
3. otherwise only the drawn rows are read (`read_npy_rows`) and nothing is
   kept; NPY_CACHE.get remains the fallback for files read_npy_rows cannot
   read.

The rows, their order and their duplicates are those of a whole-file load in
every case, and a small fit gives the same losses with any budget.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd.cache.npy as npy_module
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import TinyNetwork
from tests.test_descriptors_series_fit import (
    _install_regularization_extractors)
from tests.test_fit_graphs_in_memory import BACK, FORW, ROWS, VALUES
from tests.test_network_fit_unit import DummyPathEnsemble

fit_module = importlib.import_module("aimmd.network.fit")


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    NPY_CACHE.clear()
    # what an earlier test left in total_size must not shrink the room here
    monkeypatch.setattr(NPY_CACHE, "total_size", 0)
    yield
    NPY_CACHE.clear()


def _write(tmp_path, lengths, width=3, stem="traj"):
    """One float32 file per length, with distinct values in every row."""
    paths, arrays = [], []
    for k, length in enumerate(lengths):
        array = (np.arange(length * width, dtype=np.float32)
                 .reshape(length, width) + 1000 * k)
        path = str(tmp_path / f"{stem}{k}.descriptors.npy")
        save_npy(path, array)
        paths.append(path)
        arrays.append(array)
    return paths, arrays


def _charge(path):
    """What NPY_CACHE charges for keeping the whole file."""
    return NPY_CACHE._size(np.load(path))


def _expected(paths, arrays, npy_paths, locs):
    by_path = dict(zip(paths, arrays))
    return np.array([by_path[path][loc] for path, loc in zip(npy_paths, locs)])


@pytest.fixture
def reads(monkeypatch):
    """Record every read_npy_rows call of fit as ``(fname, indices)``."""
    calls = []
    real = npy_module.read_npy_rows

    def spy(fname, indices, *args, **kwargs):
        calls.append((fname, [int(i) for i in indices]))
        return real(fname, indices, *args, **kwargs)

    monkeypatch.setattr(fit_module, "read_npy_rows", spy)
    return calls


class _Loads(list):
    """Whole-file loads by NPY_CACHE; `np` lists every np.load."""

    def __init__(self):
        super().__init__()
        self.np = []

    def clear(self):
        super().clear()
        self.np.clear()


@pytest.fixture
def loads(monkeypatch):
    """Record every whole-file load, by NPY_CACHE or by np.load."""
    calls = _Loads()
    real_load_npy, real_np_load = npy_module.load_npy, np.load

    def spy_load_npy(fname, *args, **kwargs):
        calls.append(fname)
        return real_load_npy(fname, *args, **kwargs)

    def spy_np_load(fname, *args, **kwargs):
        calls.np.append(str(fname))
        return real_np_load(fname, *args, **kwargs)

    monkeypatch.setattr(npy_module, "load_npy", spy_load_npy)
    monkeypatch.setattr(np, "load", spy_np_load)
    return calls


def _state():
    return list(NPY_CACHE._cache), NPY_CACHE.total_size


def test_files_that_fit_are_kept_as_before(tmp_path, monkeypatch, reads,
                                           loads):
    paths, arrays = _write(tmp_path, [7, 4, 9])
    charges = [_charge(path) for path in paths]
    loads.clear()
    # room for exactly the three files
    monkeypatch.setattr(NPY_CACHE, "max_size", sum(charges) + 1)
    p0, p1, p2 = paths
    npy_paths = np.array([p1, p0, p1, p2, p0, p1])
    locs = np.array([3, 6, 3, 8, 0, 1])

    out = fit_module._load_batch_descriptors(npy_paths, locs)

    np.testing.assert_array_equal(
        out, _expected(paths, arrays, npy_paths, locs))
    assert out.dtype == np.float32
    assert reads == []
    assert loads == [p1, p0, p2]           # once each, in order of first use
    assert _state() == ([p1, p0, p2], sum(charges))

    # all resident now: the next batch reads nothing
    loads.clear()
    out = fit_module._load_batch_descriptors(npy_paths[::-1], locs[::-1])
    np.testing.assert_array_equal(
        out, _expected(paths, arrays, npy_paths[::-1], locs[::-1]))
    assert loads == [] and reads == []


def test_files_beyond_the_budget_are_read_by_rows(tmp_path, monkeypatch,
                                                  reads, loads):
    paths, arrays = _write(tmp_path, [7, 4, 9, 5])
    p0, p1, p2, p3 = paths
    NPY_CACHE.get(p0)
    # p0 stays resident; no room for any other file
    monkeypatch.setattr(NPY_CACHE, "max_size", NPY_CACHE.total_size + 10)
    before = _state()
    loads.clear()
    npy_paths = np.array([p1, p0, p2, p1, p3, p0, p1, p2])
    locs = np.array([3, 6, 8, 3, 0, 6, 1, 2])

    out = fit_module._load_batch_descriptors(npy_paths, locs)

    np.testing.assert_array_equal(
        out, _expected(paths, arrays, npy_paths, locs))
    assert out.dtype == np.float32
    assert loads == [] and loads.np == []  # no whole file is read
    # only the drawn rows of the files that are not resident, in batch
    # order with their duplicates
    assert reads == [(p1, [3, 3, 1]), (p2, [8, 2]), (p3, [0])]
    assert _state() == before              # nothing admitted or evicted


def test_a_file_larger_than_the_budget_is_read_by_rows(tmp_path, monkeypatch,
                                                       reads, loads):
    paths, arrays = _write(tmp_path, [3, 40])
    small, big = paths
    charge_small = _charge(small)
    monkeypatch.setattr(NPY_CACHE, "max_size",
                        charge_small + _charge(big) // 2)
    loads.clear()
    npy_paths = np.array([big, small, big, big])
    locs = np.array([39, 2, 0, 39])

    out = fit_module._load_batch_descriptors(npy_paths, locs)

    np.testing.assert_array_equal(
        out, _expected(paths, arrays, npy_paths, locs))
    assert reads == [(big, [39, 0, 39])]
    assert loads == [small]                # the small file still fits
    assert _state() == ([small], charge_small)

    # nothing fits in an empty cache with a budget below the file
    NPY_CACHE.clear()
    monkeypatch.setattr(NPY_CACHE, "max_size", _charge(big) // 2)
    reads.clear()
    loads.clear()
    out = fit_module._load_batch_descriptors(np.array([big, big]),
                                             np.array([5, 4]))
    np.testing.assert_array_equal(out, arrays[1][[5, 4]])
    assert reads == [(big, [5, 4])] and loads == [] and loads.np == []
    assert _state() == ([], 0)


def test_unreadable_rows_fall_back_to_the_cache(tmp_path, monkeypatch, loads):
    paths, arrays = _write(tmp_path, [7, 4])
    monkeypatch.setattr(NPY_CACHE, "max_size", 10)
    calls = []

    def refuse(fname, indices, *args, **kwargs):
        calls.append(fname)
        return None

    monkeypatch.setattr(fit_module, "read_npy_rows", refuse)
    loads.clear()
    npy_paths = np.array([paths[1], paths[0], paths[1]])
    locs = np.array([3, 6, 0])

    out = fit_module._load_batch_descriptors(npy_paths, locs)

    np.testing.assert_array_equal(
        out, _expected(paths, arrays, npy_paths, locs))
    assert calls == [paths[1], paths[0]]
    assert loads == [paths[1], paths[0]]   # NPY_CACHE.get, as before


def test_rows_past_the_end_still_fail(tmp_path, monkeypatch):
    paths, _ = _write(tmp_path, [4])
    monkeypatch.setattr(NPY_CACHE, "max_size", 10)
    with pytest.raises(IndexError):
        fit_module._load_batch_descriptors(np.array(paths * 2),
                                           np.array([1, 4]))


def test_a_missing_file_still_fails(tmp_path, monkeypatch):
    missing = str(tmp_path / "missing.descriptors.npy")
    for budget in (10, type(NPY_CACHE).max_size):
        monkeypatch.setattr(NPY_CACHE, "max_size", budget)
        with pytest.raises(RuntimeError, match="Could not load descriptor"):
            fit_module._load_batch_descriptors(np.array([missing]),
                                               np.array([0]))


@pytest.mark.parametrize("room", [True, False])
def test_a_resident_copy_too_short_is_not_used(tmp_path, monkeypatch, reads,
                                               loads, room):
    paths, arrays = _write(tmp_path, [20])
    path = paths[0]
    stale = arrays[0][:3] + 0.5
    NPY_CACHE.put(path, stale)
    if not room:
        monkeypatch.setattr(NPY_CACHE, "max_size", NPY_CACHE.total_size + 10)
    loads.clear()
    npy_paths = np.array([path, path])
    locs = np.array([1, 15])

    out = fit_module._load_batch_descriptors(npy_paths, locs)

    np.testing.assert_array_equal(out, arrays[0][[1, 15]])
    if room:                               # reloaded and kept
        assert reads == [] and loads == [path]
        assert len(NPY_CACHE.peek(path)) == 20
        assert NPY_CACHE.total_size == _charge(path)
    else:                                  # rows read, the copy left alone
        assert reads == [(path, [1, 15])] and loads == []
        assert NPY_CACHE.peek(path) is stale


def test_a_resident_copy_long_enough_is_used(tmp_path, monkeypatch, reads,
                                             loads):
    paths, arrays = _write(tmp_path, [7])
    path = paths[0]
    resident = arrays[0] + 0.5             # tells the copy from the file
    NPY_CACHE.put(path, resident)
    monkeypatch.setattr(NPY_CACHE, "max_size", 10)
    loads.clear()

    out = fit_module._load_batch_descriptors(np.array([path, path]),
                                             np.array([6, 2]))

    np.testing.assert_array_equal(out, resident[[6, 2]])
    assert reads == [] and loads == []


def test_routed_batches_take_the_same_path(tmp_path, monkeypatch, reads,
                                           loads):
    paths_a, arrays_a = _write(tmp_path, [6, 5], stem="a")
    paths_b, arrays_b = _write(tmp_path, [8], width=4, stem="b")
    monkeypatch.setattr(NPY_CACHE, "max_size", 10)
    loads.clear()
    npy_paths = np.array([paths_b[0], paths_a[1], paths_a[0], paths_b[0],
                          paths_a[1]])
    locs = np.array([7, 4, 0, 7, 2])
    system_id = np.array([1, 0, 0, 1, 0])

    def transform(block, system_id=None):
        return [(system_id, tuple(row)) for row in np.asarray(block)]

    out = fit_module._load_batch_descriptors_routed(
        npy_paths, locs, system_id, ["sa", "sb"], transform, True,
        graphs=True)

    rows = dict(zip(paths_a + paths_b, arrays_a + arrays_b))
    assert out == [(("sa", "sb")[s], tuple(rows[p][loc]))
                   for p, loc, s in zip(npy_paths, locs, system_id)]
    assert sorted(reads) == sorted([(paths_a[1], [4, 2]), (paths_a[0], [0]),
                                    (paths_b[0], [7, 7])])
    assert loads == [] and loads.np == [] and _state() == ([], 0)


def _install_file_extractor(monkeypatch, fnames):
    """Synthetic extraction: frame `category` of the file `category % 3`."""

    def fake_extract(pathensemble, indices, *sources):
        category = int(indices[0])
        assert sources[-2:] == ("filenames", "locs")
        refs = (np.array([fnames[category % len(fnames)]]),
                np.array([category]))
        back = np.array([BACK[category]])
        forw = np.array([FORW[category]])
        if sources[0] == "values":
            return (np.arange(1), back, forw,
                    np.array([VALUES[category]]), *refs, 1)
        return (np.arange(1), back, forw, *refs, 1)

    monkeypatch.setattr(fit_module, "extract_indices_and_series", fake_extract)
    monkeypatch.setattr(
        fit_module, "compute_bins",
        lambda *args, **kwargs: np.array([-np.inf, -0.5, 0.5, np.inf]))
    monkeypatch.setattr(
        fit_module, "merge_marginal_bins",
        lambda bins, values1, values2, min_values=3: (bins, np.ones(3)))


def test_fit_with_a_tiny_budget_gives_the_same_losses(tmp_path, monkeypatch,
                                                      reads):
    fnames = [str(tmp_path / f"traj{k}.xtc") for k in range(3)]
    for k, fname in enumerate(fnames):
        save_npy(get_cache_fname(fname, "descriptors"), ROWS + 0.01 * k)
    _install_file_extractor(monkeypatch, fnames)
    _install_regularization_extractors(monkeypatch, fnames[0], [])
    budget = NPY_CACHE.max_size

    def run(max_size):
        NPY_CACHE.clear()
        monkeypatch.setattr(NPY_CACHE, "max_size", max_size)
        np.random.seed(0)
        torch.manual_seed(0)
        params = SimpleNamespace(network=TinyNetwork(), sorted_states="ARB",
                                 descriptors_function=True,
                                 descriptor_transform=lambda rows: rows)
        losses, *_ = fit_module.fit(
            params, DummyPathEnsemble(), nbins=1, in_memory=False,
            graphs=False, epochs=3, batch_size=4, lsr_weight=0.1,
            mar_weight=0.1, stop=100.0)
        weights = params.network.linear.weight.detach().clone()
        return losses, weights, _state()

    losses, weights, state = run(budget)
    assert reads == [] and len(state[0]) == 3
    tiny_losses, tiny_weights, tiny_state = run(10)
    assert reads and tiny_state == ([], 0)

    assert losses and tiny_losses == losses
    assert torch.equal(tiny_weights, weights)
