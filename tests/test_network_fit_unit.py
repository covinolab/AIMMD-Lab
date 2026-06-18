"""Focused unit tests for the large `aimmd.network.fit` training routine.

These tests deliberately replace the expensive path-ensemble data gathering
steps with tiny synthetic arrays. That lets us exercise the training control
flow, bin-based weighting logic, and stop/validation branches in isolation.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

fit_module = importlib.import_module("aimmd.network.fit")
from tests._helpers_unit import TinyNetwork


class DummyPathEnsemble:
    """Minimal pathensemble stub exposing only the type strings used by `fit`.

    The production implementation infers training categories from 4-character
    path codes. We provide one path per category so the synthetic extractor can
    hand back deterministic values/descriptors for every branch in `fit`.
    """

    def __init__(self):
        self._types = np.array(
            [
                "RAAA",  # in A
                "RBBB",  # in B
                "ARAA",  # free A -> A
                "BRBB",  # free B -> B
                "ARBA",  # free A -> B
                "BRAB",  # free B -> A
                "ARAR",  # shot A -> A
                "BRBR",  # shot B -> B
                "ARBR",  # shot A -> B
                "BRAR",  # shot B -> A
            ],
            dtype="<U4",
        )

    def types(self):
        return self._types

    def __add__(self, other):
        """Concatenate path codes (used by fit's pooled bin computation for
        multi-system; compute_bins is monkeypatched, so only the codes matter)."""
        merged = DummyPathEnsemble.__new__(DummyPathEnsemble)
        merged._types = np.concatenate([self._types, other._types])
        return merged


def _params():
    """Build the smallest Params-like object accepted by `fit`.

    Only the attributes actually read inside the function are provided here.
    """

    return SimpleNamespace(
        network=TinyNetwork(),
        sorted_states="ARB",
        descriptors_function=True,
        descriptor_transform=lambda x: x,
    )


def _install_synthetic_extractors(monkeypatch):
    """Replace data extraction/binning with deterministic synthetic outputs.

    Each path category contributes a tiny descriptor matrix with two features so
    `TinyNetwork` can train on it. The returned values are chosen to populate
    multiple bins and to keep both reaction outcomes represented.
    """

    category_data = {
        0: {"desc": np.array([[[-2.0, -1.0]]], dtype=np.float32)},
        1: {"desc": np.array([[[2.0, 1.5]]], dtype=np.float32)},
        2: {
            "values": np.array([-1.5]),
            "desc": np.array([[[-1.0, -0.5]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([False]),
        },
        3: {
            "values": np.array([1.5]),
            "desc": np.array([[[1.0, 0.5]]], dtype=np.float32),
            "back": np.array([False]),
            "forw": np.array([True]),
        },
        4: {
            "values": np.array([-0.2]),
            "desc": np.array([[[-0.2, 0.3]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
        5: {
            "values": np.array([0.2]),
            "desc": np.array([[[0.4, -0.1]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
        6: {
            "values": np.array([-0.8]),
            "desc": np.array([[[-0.8, -0.2]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
        7: {
            "values": np.array([0.8]),
            "desc": np.array([[[0.8, 0.2]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
        8: {
            "values": np.array([-0.4]),
            "desc": np.array([[[-0.4, 0.6]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
        9: {
            "values": np.array([0.4]),
            "desc": np.array([[[0.6, -0.4]]], dtype=np.float32),
            "back": np.array([True]),
            "forw": np.array([True]),
        },
    }

    def fake_extract(pathensemble, indices, *sources):
        """Return the synthetic payload for the unique category in `indices`."""

        idx = int(indices[0])
        payload = category_data[idx]
        if len(sources) == 1:
            desc = payload["desc"]
            n = len(desc)
            return np.arange(n), np.ones(n, dtype=bool), np.ones(n, dtype=bool), desc, 1
        values = payload["values"]
        desc = payload["desc"]
        back = payload["back"]
        forw = payload["forw"]
        return np.arange(len(values)), back, forw, values, desc, 1

    monkeypatch.setattr(fit_module, "extract_indices_and_series", fake_extract)
    monkeypatch.setattr(
        fit_module,
        "compute_bins",
        lambda *args, **kwargs: np.array([-np.inf, -0.5, 0.5, np.inf], dtype=float),
    )
    monkeypatch.setattr(
        fit_module,
        "merge_marginal_bins",
        lambda bins, values1, values2, min_values=3: (
            bins,
            np.array([1.0, 1.0, 1.0], dtype=float),
        ),
    )


def test_fit_rejects_invalid_options():
    """Input validation should fail before any training-side effects happen."""

    params = _params()
    pathensemble = DummyPathEnsemble()

    with pytest.raises(TypeError):
        fit_module.fit(params, pathensemble, batching_strategy="bad")
    with pytest.raises(TypeError):
        fit_module.fit(params, pathensemble, augment="bad")
    with pytest.raises(TypeError):
        fit_module.fit(params, pathensemble, sparse_update_max_frames=3)


def test_fit_trains_on_synthetic_data_with_validation(monkeypatch):
    """A tiny synthetic dataset should be enough to traverse the main fit loop.

    This test enables several optional penalties/branches at once so we cover:
    - selection-probability shaping from bins,
    - validation-split setup,
    - L1 regularization,
    - in-memory descriptor handling,
    - and the final return contract.
    """

    _install_synthetic_extractors(monkeypatch)
    np.random.seed(0)

    params = _params()
    pathensemble = DummyPathEnsemble()
    losses, scales, values, selection_probabilities, results = fit_module.fit(
        params,
        pathensemble,
        augment="yes",
        nbins=2,
        state_bins="all",
        transition_path_upweighting=1.5,
        train_validation_early_stopping=True,
        early_stopping_min_samples=1,
        early_stopping_patience=2,
        # The current implementation keeps the full probability vector while
        # shrinking `training_set_size`, so a non-zero split can make the
        # sampling dimensions inconsistent. Using a zero-sized validation set
        # still exercises the branch without triggering that known issue.
        early_stopping_split=0.0,
        loss_regularization_weight=0.01,
        epochs=2,
        batch_size=4,
        stop=100.0,
        verbose=False,
    )

    # The fit routine should complete at least one optimizer step and preserve
    # its basic output invariants even on this heavily simplified dataset.
    assert losses
    assert scales
    assert values.ndim == 1
    assert results.shape[1] == 2
    assert np.isclose(selection_probabilities.sum(), 1.0)
    assert np.isfinite(results).all()


def test_fit_supports_smoothness_penalty_without_validation(monkeypatch):
    """The smoothness penalty should work during ordinary training batches.

    The validation path wraps the loss evaluation in `torch.no_grad()`, so the
    production code currently cannot combine validation loss computation with the
    gradient-based smoothness penalty. This test isolates the smoothness branch
    on its own, where descriptor gradients are expected to be available.
    """

    _install_synthetic_extractors(monkeypatch)
    np.random.seed(2)

    losses, scales, *_ = fit_module.fit(
        _params(),
        DummyPathEnsemble(),
        augment="yes",
        nbins=1,
        loss_smoothening_weight=0.01,
        epochs=1,
        batch_size=4,
        stop=100.0,
        verbose=False,
    )

    assert losses
    assert scales


def test_fit_stops_cleanly_when_worker_requests_termination(monkeypatch):
    """A worker termination signal should short-circuit the expensive logic.

    `fit` checks the worker after each extraction step so AIMMD can abort
    cooperatively instead of finishing a full training round.
    """

    _install_synthetic_extractors(monkeypatch)
    params = _params()
    worker = SimpleNamespace(termination_signal=True)

    result = fit_module.fit(params, DummyPathEnsemble(), worker=worker, epochs=1)
    assert result == ([], [], [], [], [])


def test_fit_restores_safe_weights_when_output_scale_blows_up(monkeypatch):
    """Very small `stop` thresholds should trigger the overflow safety branch.

    This covers the code path where AIMMD decides the network logits became too
    large and restores one of the previously saved state dictionaries.
    """

    _install_synthetic_extractors(monkeypatch)
    np.random.seed(1)

    params = _params()
    losses, scales, *_ = fit_module.fit(
        params,
        DummyPathEnsemble(),
        augment="yes",
        nbins=1,
        epochs=2,
        batch_size=4,
        stop=0.01,
        verbose=False,
    )

    assert len(losses) == 1
    assert scales[0] >= 0.01


def test_fit_single_pe_equals_one_element_list(monkeypatch):
    """A 1-element LIST of PathEnsembles must reduce to the single-system path.

    ``fit(params, [pe])`` and ``fit(params, pe)`` execute identical code, so with
    the same RNG seed and a fresh network they must produce identical output —
    the balancing is a strict no-op for a single system.
    """
    _install_synthetic_extractors(monkeypatch)
    pe = DummyPathEnsemble()

    def run(arg):
        import torch
        np.random.seed(7)
        torch.manual_seed(7)
        params = _params()
        return fit_module.fit(params, arg, augment="no", nbins=2,
                              state_bins="all", epochs=3, batch_size=4,
                              stop=100.0, in_memory=True, verbose=False)

    losses_single, _, _, sp_single, _ = run(pe)
    losses_list, _, _, sp_list, _ = run([pe])
    assert losses_single == losses_list
    assert np.allclose(sp_single, sp_list)


def test_fit_multi_system_pools_and_balances(monkeypatch):
    """``fit`` accepts a LIST of PathEnsembles, pools them, and balances systems.

    With two (identical) synthetic systems the pooled training set is twice as
    large, selection probabilities still normalize to 1, and each in-state anchor
    block carries equal mass per system (1/N per bin, including state bins).
    """
    _install_synthetic_extractors(monkeypatch)
    np.random.seed(0)
    params = _params()
    pe = DummyPathEnsemble()

    losses, scales, values, selection_probabilities, results = fit_module.fit(
        params, [pe, pe], augment="no", nbins=2, state_bins="all",
        epochs=3, batch_size=4, stop=100.0, in_memory=True, verbose=False)

    assert losses                                    # trained at least once
    assert results.shape[1] == 2
    assert np.isclose(selection_probabilities.sum(), 1.0)
    # selection_probabilities and results stay aligned after pooling/keepers
    assert values.ndim == 1
    assert len(selection_probabilities) == len(results)
    assert np.isfinite(results).all()


def test_assemble_inmemory_multi_dense_and_graphs():
    """The in-memory multi-system descriptor assembler filters each system's
    block by keepers, transforms it with that system's id, and concatenates.

    Dense -> a stacked 2D array (different-atom-count blocks are transformed to a
    common width per system); graphs -> a flat list of graph objects.
    """
    blocks = [(0, np.array([[1.0, 2.0], [3.0, 4.0]])),  # system 0, 2 frames
              (1, np.array([[5.0, 6.0]]))]               # system 1, 1 frame
    keepers = np.array([True, False, True])              # drop the 2nd frame
    labels = ['s1', 's2']

    # dense: scale system 's2' by 10 to prove the per-system id reaches transform
    def dense_transform(block, system_id=None):
        return np.asarray(block) * (10.0 if system_id == 's2' else 1.0)

    dense = fit_module._assemble_inmemory_multi(
        blocks, keepers, dense_transform, True, labels, graphs=False)
    np.testing.assert_allclose(dense, [[1.0, 2.0], [50.0, 60.0]])

    # graphs: a list, one (tagged) entry per kept frame, in global order
    def graph_transform(block, system_id=None):
        return [(system_id, float(row[0])) for row in np.asarray(block)]

    graphs = fit_module._assemble_inmemory_multi(
        blocks, keepers, graph_transform, True, labels, graphs=True)
    assert graphs == [('s1', 1.0), ('s2', 5.0)]


def test_load_batch_descriptors_routed_groups_by_system(monkeypatch):
    """A mixed-system batch is grouped by system, transformed per system, and
    reassembled in the original batch order."""
    # stub the raw loader: 'path' encodes the value, loc unused
    monkeypatch.setattr(fit_module, "_load_batch_descriptors",
                        lambda paths, locs: np.array([[float(p)] for p in paths]))

    paths = np.array([10.0, 20.0, 30.0])
    locs = np.array([0, 0, 0])
    system_id = np.array([1, 0, 1])           # interleaved systems
    labels = ['s1', 's2']

    def transform(block, system_id=None):
        bump = 100.0 if system_id == 's2' else 0.0
        return np.asarray(block) + bump

    out = fit_module._load_batch_descriptors_routed(
        paths, locs, system_id, labels, transform, True, graphs=False)
    # frames 0 and 2 are system 1 ('s2', +100); frame 1 is system 0 ('s1')
    np.testing.assert_allclose(out, [[110.0], [20.0], [130.0]])


def test_assign_balanced_uniform_splits_mass_per_system():
    """In-state anchor mass is split equally across systems present in a block."""
    sp = np.zeros(5)
    system_id = np.array([0, 0, 1, 1, 1])     # 2 frames sys0, 3 frames sys1
    fit_module._assign_balanced_uniform(sp, 0, 5, system_id, total_mass=1.0)
    assert np.isclose(sp.sum(), 1.0)
    assert np.isclose(sp[:2].sum(), 0.5)      # system 0 carries half
    assert np.isclose(sp[2:].sum(), 0.5)      # system 1 carries half
    assert np.isclose(sp[0], sp[1])           # uniform within a system
    assert np.isclose(sp[2], sp[3]) and np.isclose(sp[3], sp[4])


def test_default_wrapper_forwards_to_fit(monkeypatch):
    """`default` is a thin wrapper and should forward its arguments unchanged."""

    called = {}

    def fake_fit(**kwargs):
        called.update(kwargs)
        return ("sentinel",)

    monkeypatch.setattr(fit_module, "fit", fake_fit)
    params = _params()
    pathensemble = DummyPathEnsemble()

    result = fit_module.default(params, pathensemble, verbose=False, worker="w")
    assert result == ("sentinel",)
    assert called["params"] is params
    assert called["pathensemble"] is pathensemble
    assert called["verbose"] is False
    assert called["worker"] == "w"
