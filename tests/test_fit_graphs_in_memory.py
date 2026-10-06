"""`fit(in_memory=True, graphs=True)` warns and trains with in_memory=False.

With graph descriptors, ``in_memory=True`` applies ``descriptor_transform`` to
the whole training set up front and keeps every graph object in RAM (about
90 kB per frame for the kcmpd09 PaiNN graphs, i.e. tens of GB per round). Per
batch loading (``in_memory=False``) is the mode graph networks are meant to
train in, so the combination now warns and falls back to it.

These tests replace torch_geometric's ``Batch`` with a stub, so they run
without the optional graph dependencies.
"""
import importlib
import sys
import types
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import TinyNetwork
from tests.test_network_fit_unit import DummyPathEnsemble

fit_module = importlib.import_module("aimmd.network.fit")

# one frame per path category of DummyPathEnsemble (same numbers as the
# synthetic data of test_network_fit_unit)
ROWS = np.array([[-2.0, -1.0], [2.0, 1.5], [-1.0, -0.5], [1.0, 0.5],
                 [-0.2, 0.3], [0.4, -0.1], [-0.8, -0.2], [0.8, 0.2],
                 [-0.4, 0.6], [0.6, -0.4]])
VALUES = [None, None, -1.5, 1.5, -0.2, 0.2, -0.8, 0.8, -0.4, 0.4]
BACK = [True, True, True, False, True, True, True, True, True, True]
FORW = [True, True, False, True, True, True, True, True, True, True]


class _Batch:
    """Stand-in for torch_geometric.data.Batch on 1D "graphs" (rows)."""

    def __init__(self, x):
        self.x = x

    @classmethod
    def from_data_list(cls, data_list):
        return cls(torch.as_tensor(np.stack(data_list), dtype=torch.float32))

    def to(self, device):
        return self

    def to_dict(self):
        return self.x


@pytest.fixture
def stub_torch_geometric(monkeypatch):
    package = types.ModuleType("torch_geometric")
    data = types.ModuleType("torch_geometric.data")
    data.Batch = _Batch
    package.data = data
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data)


def _install_extractor(monkeypatch, fname, requested):
    """Synthetic extraction answering either raw rows or file references."""

    def fake_extract(pathensemble, indices, *sources):
        requested.append(sources)
        category = int(indices[0])
        if sources[-2:] == ("filenames", "locs"):
            refs = (np.array([fname]), np.array([category]))
        else:
            refs = (ROWS[category:category + 1],)
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


def _params(transform_calls):
    def descriptor_transform(rows):
        transform_calls.append(len(rows))
        return [row for row in np.asarray(rows)]       # one "graph" per row

    return SimpleNamespace(network=TinyNetwork(), sorted_states="ARB",
                           descriptors_function=True,
                           descriptor_transform=descriptor_transform)


def test_in_memory_with_graphs_warns_and_loads_per_batch(
        tmp_path, monkeypatch, stub_torch_geometric):
    fname = str(tmp_path / "traj.xtc")
    save_npy(get_cache_fname(fname, "descriptors"), ROWS)
    NPY_CACHE.clear()
    requested, transform_calls = [], []
    _install_extractor(monkeypatch, fname, requested)
    np.random.seed(0)

    with pytest.warns(UserWarning, match="in_memory"):
        losses, *_ = fit_module.fit(
            _params(transform_calls), DummyPathEnsemble(), nbins=1,
            in_memory=True, graphs=True, epochs=3, batch_size=4,
            stop=100.0)

    assert losses
    # per-frame file references were collected, never the raw rows
    assert all(sources[-2:] == ("filenames", "locs") for sources in requested)
    # the transform only ever saw one batch, never the whole training set
    assert transform_calls and max(transform_calls) <= 4
    NPY_CACHE.clear()


@pytest.mark.parametrize("in_memory, graphs", [(False, True), (True, False)])
def test_other_combinations_do_not_warn(tmp_path, monkeypatch,
                                        stub_torch_geometric, in_memory, graphs):
    requested = []
    _install_extractor(monkeypatch, str(tmp_path / "traj.xtc"), requested)
    worker = SimpleNamespace(termination_signal=True)  # stop after extraction
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_module.fit(_params([]), DummyPathEnsemble(),
                                in_memory=in_memory, graphs=graphs,
                                worker=worker)
    assert result == ([], [], [], [], [])
    expected = ("filenames", "locs") if not in_memory else ("descriptors",)
    assert requested == [expected]
