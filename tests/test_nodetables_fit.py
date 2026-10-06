"""`fit(graphs=True)` with a descriptor_transform that returns a batch dict.

`NodeTableFeaturizer.batch_dict` builds the network input of a batch directly
(one radius_graph call), which is cheaper than building a list of graphs and
batching it with ``Batch.from_data_list`` (11.8 against 16.5 ms per 32-frame
batch for kcmpd09). fit uses such a dict as it is, only moved to the
network's device, for the committor batches, the validation set and the LSR
and MAR terms; a list of graphs is batched as before. With a fixed seed both
give the same losses and weights. A multi-system fit routes frames per system
and needs graphs; a dict there is refused with a clear error.

The first tests replace torch_geometric with a stub (no graph dependencies);
the comparison with real graphs needs ``--rungraph``.
"""
import copy
import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests.test_descriptors_series_fit import (
    _install_regularization_extractors)
from tests.test_fit_graphs_in_memory import ROWS, _install_extractor
from tests.test_network_fit_unit import DummyPathEnsemble

fit_module = importlib.import_module("aimmd.network.fit")

SERIES = "descriptors-gn0123456789"
FIT_KWARGS = dict(nbins=1, in_memory=False, graphs=True, epochs=4,
                  batch_size=4, lsr_weight=0.1, mar_weight=0.1, stop=100.0)


class _RefusingBatch:
    """Stand-in for torch_geometric's Batch that must not be used."""

    @classmethod
    def from_data_list(cls, data_list):
        raise AssertionError("a batch dict must not be batched again")


@pytest.fixture
def refusing_torch_geometric(monkeypatch):
    package = types.ModuleType("torch_geometric")
    data = types.ModuleType("torch_geometric.data")
    data.Batch = _RefusingBatch
    package.data = data
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data)


class _DictNetwork(aimmd.network.Rescalable):
    """Linear network on the 'x' entry of a batch dict."""

    def __init__(self, seen):
        super().__init__(max_knots=8)
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.seen = seen

    def forward(self, batch):
        self.seen.append(batch)
        return self.linear(batch["x"])


def _series_file(tmp_path, rows):
    fname = str(tmp_path / "traj.xtc")
    save_npy(get_cache_fname(fname, SERIES), rows)
    NPY_CACHE.clear()
    return fname


def test_a_batch_dict_is_used_as_it_is(tmp_path, monkeypatch,
                                       refusing_torch_geometric):
    fname = _series_file(tmp_path, ROWS.astype(np.float32))
    _install_extractor(monkeypatch, fname, [])
    _install_regularization_extractors(monkeypatch, fname, [])
    transformed, seen = [], []

    def descriptor_transform(rows):
        batch = {"x": torch.as_tensor(np.asarray(rows), dtype=torch.float32),
                 "n": len(rows)}
        transformed.append(batch)
        return batch

    params = SimpleNamespace(network=_DictNetwork(seen), sorted_states="ARB",
                             descriptors_function=True,
                             descriptor_transform=descriptor_transform,
                             descriptors_series=SERIES)
    np.random.seed(0)

    losses, *_ = fit_module.fit(params, DummyPathEnsemble(), **FIT_KWARGS)

    assert losses
    # committor batches, LSR pairs and MAR sequences
    assert len(transformed) > FIT_KWARGS["epochs"] * 2
    assert all(isinstance(batch, dict) for batch in seen)
    # the same tensors (on the network's device), other entries untouched
    tensors = {id(batch["x"]) for batch in transformed}
    assert all(id(batch["x"]) in tensors for batch in seen)
    assert all(batch["n"] == len(batch["x"]) for batch in seen)
    NPY_CACHE.clear()


def test_graph_batch_moves_a_dict_to_the_device():
    batch = {"positions": torch.zeros(3, 3), "ptr": torch.tensor([0, 3]),
             "label": "kept"}
    moved = fit_module._graph_batch(batch, torch.device("meta"))
    assert moved["positions"].device.type == "meta"
    assert moved["ptr"].device.type == "meta"
    assert moved["label"] == "kept"
    assert batch["positions"].device.type == "cpu"     # input untouched


def test_a_multi_system_fit_refuses_a_batch_dict(tmp_path):
    fname = _series_file(tmp_path, ROWS.astype(np.float32))
    npy = np.array([get_cache_fname(fname, SERIES)] * 4)

    with pytest.raises(TypeError, match="list of graphs"):
        fit_module._load_batch_descriptors_routed(
            npy, np.arange(4), np.array([0, 1, 0, 1]), ["s1", "s2"],
            lambda rows: {"x": torch.as_tensor(rows)}, False, graphs=True)
    NPY_CACHE.clear()


# ----------------------------------------------------------------------------
# with node tables and real graphs

class _TinyGNN(aimmd.network.Rescalable):
    """One message-passing step over the edges, summed per graph. Records
    its parameters at every call (fit may restore the initial ones at the
    end)."""

    def __init__(self, n_types):
        super().__init__(max_knots=8)
        self.embed = torch.nn.Linear(n_types, 4)
        self.out = torch.nn.Linear(4, 1)
        self.trace = []

    def forward(self, batch):
        self.trace.append(torch.cat([parameter.detach().flatten().clone()
                                     for parameter in self.parameters()]))
        h = self.embed(batch["node_attrs"])
        source, target = batch["edge_index"]
        distance = (batch["positions"][source]
                    - batch["positions"][target]).norm(dim=1, keepdim=True)
        h = h + torch.zeros_like(h).index_add(
            0, target, h[source] * torch.exp(-distance))
        pooled = torch.zeros(len(batch["ptr"]) - 1, h.shape[1]).index_add(
            0, batch["batch"], torch.tanh(h))
        return self.out(pooled)


@pytest.mark.graph
def test_dict_and_graph_batches_train_identically(tmp_path, monkeypatch):
    pytest.importorskip("torch_geometric")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from aimmd.network.nodetables import NodeTableFeaturizer
    from tests._nodetables_toy import (ATOM_TYPES, CUTOFF,
                                       ENVIRONMENT_SELECTION, SYSTEM_SELECTION,
                                       toy_frames, toy_universe)

    featurizer = NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                                     ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF)
    fname = _series_file(tmp_path,
                         featurizer.rows_from_coordinates(toy_frames(10)))
    _install_extractor(monkeypatch, fname, [])
    _install_regularization_extractors(monkeypatch, fname, [])
    torch.manual_seed(0)
    initial = _TinyGNN(len(ATOM_TYPES))
    transforms = {"graphs": featurizer.graphs,
                  "batch_dict": featurizer.batch_dict}

    results = {}
    for name, transform in transforms.items():
        calls = []

        def descriptor_transform(rows, transform=transform, calls=calls):
            calls.append(len(rows))
            return transform(rows)

        params = SimpleNamespace(network=copy.deepcopy(initial),
                                 sorted_states="ARB",
                                 descriptors_function=True,
                                 descriptor_transform=descriptor_transform,
                                 descriptors_series=SERIES)
        np.random.seed(0)
        losses, *_ = fit_module.fit(params, DummyPathEnsemble(), **FIT_KWARGS)
        results[name] = (losses, params.network.state_dict(), calls,
                         params.network.trace)
        NPY_CACHE.clear()

    losses, weights, calls, trace = results["graphs"]
    assert losses and len(calls) > FIT_KWARGS["epochs"] * 2
    assert results["batch_dict"][0] == losses
    assert results["batch_dict"][2] == calls
    # the same parameters at every network call, and they did train
    assert len(results["batch_dict"][3]) == len(trace) > len(losses)
    assert all(torch.equal(a, b)
               for a, b in zip(results["batch_dict"][3], trace))
    assert any(not torch.equal(trace[0], parameters) for parameters in trace)
    for key, value in weights.items():                 # bitwise, NaN == NaN
        torch.testing.assert_close(results["batch_dict"][1][key], value,
                                   rtol=0, atol=0, equal_nan=True)
