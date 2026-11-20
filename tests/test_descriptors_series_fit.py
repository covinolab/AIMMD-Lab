"""`fit` trains on the rows of `params.descriptors_series`.

With ``in_memory=True`` fit extracts the series by name; with
``in_memory=False`` it collects per-frame file references and loads the rows
of ``{trajectory}.{descriptors_series}.npy`` per batch, for the committor
loss and for the LSR pairs and MAR sequences. A stale ``*.descriptors.npy``
next to the same trajectory is never opened. Params-like objects without the
field keep 'descriptors'.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import TinyNetwork, forbid_opening
from tests.test_fit_graphs_in_memory import ROWS, _install_extractor
from tests.test_network_fit_unit import DummyPathEnsemble

fit_module = importlib.import_module("aimmd.network.fit")

SERIES = "descriptors-gn0123456789"


def _params(seen, **extra):
    def descriptor_transform(rows):
        seen.append(np.array(rows))
        return rows

    return SimpleNamespace(network=TinyNetwork(), sorted_states="ARB",
                           descriptors_function=True,
                           descriptor_transform=descriptor_transform, **extra)


def _install_regularization_extractors(monkeypatch, fname, names):
    """LSR pairs and MAR sequences as file references into `fname`."""

    def lsr_pairs(pathensemble, key, lagtime, name):
        names.append(name)
        if name == "filenames":
            return np.array([fname] * 3), np.array([fname] * 3), 1, 3
        if name == "locs":
            return np.array([2, 4, 6]), np.array([3, 5, 7]), 1, 3
        return ROWS[[2, 4, 6]], ROWS[[3, 5, 7]], 1, 3

    def mar_sequences(paths, key, lagtime, name):
        names.append(name)
        if name == "filenames":
            return [np.array([fname] * 3)], 1, 1
        if name == "locs":
            return [np.array([4, 6, 8])], 1, 1
        return [ROWS[[4, 6, 8]]], 1, 1

    monkeypatch.setattr(fit_module, "extract_lsr_pairs", lsr_pairs)
    monkeypatch.setattr(fit_module, "extract_mar_sequences", mar_sequences)


def test_per_batch_loading_reads_the_named_series(tmp_path, monkeypatch):
    fname = str(tmp_path / "traj.xtc")
    save_npy(get_cache_fname(fname, SERIES), ROWS)
    save_npy(get_cache_fname(fname, "descriptors"), ROWS + 100.0)   # stale
    NPY_CACHE.clear()
    requested, names, seen, loaded = [], [], [], []
    _install_extractor(monkeypatch, fname, requested)
    _install_regularization_extractors(monkeypatch, fname, names)
    load = fit_module._load_batch_descriptors

    def spy(npy_paths, locs):
        loaded.extend(np.asarray(npy_paths).tolist())
        return load(npy_paths, locs)

    monkeypatch.setattr(fit_module, "_load_batch_descriptors", spy)
    opened = forbid_opening(monkeypatch, ".descriptors.npy")
    np.random.seed(0)

    losses, *_ = fit_module.fit(
        _params(seen, descriptors_series=SERIES), DummyPathEnsemble(),
        nbins=1, in_memory=False, graphs=False, epochs=2, batch_size=4,
        lsr_weight=0.1, mar_weight=0.1, stop=100.0)

    assert losses
    assert set(names) == {"filenames", "locs"}
    assert loaded and set(loaded) == {get_cache_fname(fname, SERIES)}
    # every row the network saw comes from the named series
    rows = np.concatenate(seen)
    assert (rows[:, None, :] == ROWS[None]).all(axis=2).any(axis=1).all()
    assert opened == []
    NPY_CACHE.clear()


@pytest.mark.parametrize("extra, series", [
    ({"descriptors_series": SERIES}, SERIES),
    ({}, "descriptors")])
def test_in_memory_extracts_the_named_series(tmp_path, monkeypatch, extra,
                                             series):
    requested, names = [], []
    _install_extractor(monkeypatch, str(tmp_path / "traj.xtc"), requested)
    _install_regularization_extractors(monkeypatch, str(tmp_path / "traj.xtc"),
                                       names)
    np.random.seed(0)

    losses, *_ = fit_module.fit(
        _params([], **extra), DummyPathEnsemble(), nbins=1, in_memory=True,
        graphs=False, epochs=2, batch_size=4, lsr_weight=0.1, mar_weight=0.1,
        stop=100.0)

    assert losses
    # (values, series) for the shot categories, (series,) for the others
    assert requested and all(sources[-1] == series for sources in requested)
    assert names and set(names) == {series}


def test_params_without_the_field_load_descriptors_per_batch(tmp_path,
                                                             monkeypatch):
    fname = str(tmp_path / "traj.xtc")
    save_npy(get_cache_fname(fname, "descriptors"), ROWS)
    NPY_CACHE.clear()
    requested, names, loaded = [], [], []
    _install_extractor(monkeypatch, fname, requested)
    _install_regularization_extractors(monkeypatch, fname, names)
    load = fit_module._load_batch_descriptors

    def spy(npy_paths, locs):
        loaded.extend(np.asarray(npy_paths).tolist())
        return load(npy_paths, locs)

    monkeypatch.setattr(fit_module, "_load_batch_descriptors", spy)
    np.random.seed(0)

    losses, *_ = fit_module.fit(
        _params([]), DummyPathEnsemble(), nbins=1, in_memory=False,
        graphs=False, epochs=2, batch_size=4, lsr_weight=0.1, mar_weight=0.1,
        stop=100.0)

    assert losses
    assert set(loaded) == {get_cache_fname(fname, "descriptors")}
    NPY_CACHE.clear()
