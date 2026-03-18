import numpy as np
import torch

from aimmd.network.rescalable import Rescalable
from aimmd.network.rescale_utils import find_knots_and_values, rescale
from aimmd.network.utils import PlaceholderNetwork, extract_indices_and_series


def test_placeholder_network_and_series_extraction():
    """Pin down the placeholder network and the path-series flattener."""

    network = PlaceholderNetwork()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = network(x)
    # The placeholder network is intentionally trivial: it returns the first
    # feature column unchanged so downstream code can still run without a model.
    np.testing.assert_allclose(out.detach().numpy().ravel(), np.array([1.0, 3.0]))

    class StubPath:
        def __init__(self):
            self.type = "ARBR"
            self.shooting_index = 1
            self.indices = np.array([0, 1, 2])

        def internal(self, name):
            assert name == "indices"
            return np.array([0, 1, 2])

        def get(self, name, start, stop, raise_if_missing=True):
            assert name == "values"
            return np.array([10.0, 20.0, 30.0])

    indices, back, forw, series, n_selected = extract_indices_and_series([StubPath()], None, "values")
    # The helper concatenates internal indices and also marks which frames belong
    # to the backward and forward pieces relative to the shooting point.
    np.testing.assert_array_equal(indices, np.array([0, 1, 2]))
    np.testing.assert_array_equal(back, np.array([True, True, False]))
    np.testing.assert_array_equal(forw, np.array([False, True, True]))
    np.testing.assert_allclose(series, np.array([10.0, 20.0, 30.0]))
    assert n_selected == 1


def test_rescale_utils_and_rescalable_mixin():
    """Check both direct rescaling and the `Rescalable` module wrapper."""

    q = np.array([-1.0, 0.0, 1.0])
    # With two knots the map is simply linear between the endpoints.
    np.testing.assert_allclose(rescale(q.copy(), [-1.0, 1.0], [-2.0, 2.0]), np.array([-2.0, 0.0, 2.0]))

    knots, values = find_knots_and_values(
        np.array([-2.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 2.0]),
        np.array([4.0, 2.0, 1.0]),
        np.array([4.0, 2.0, 1.0]),
    )
    assert len(knots) == len(values)

    class DemoNetwork(Rescalable):
        def __init__(self):
            super().__init__(max_knots=4)

        def forward(self, x):
            return torch.as_tensor(x, dtype=torch.float32).clone()

    network = DemoNetwork()
    network.set_knots_and_values([-1.0, 1.0], [-2.0, 2.0])
    out = network(torch.tensor([-1.0, 0.0, 1.0]))
    # The mixin applies the post-forward remapping transparently on call.
    np.testing.assert_allclose(out.detach().numpy(), np.array([-2.0, 0.0, 2.0]))
