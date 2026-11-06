"""The protocol values function: ``aimmd.cgaimmd.values``.

The default values function of C-GAIMMD networks (under every chain type) and
of ``chain_type='cgaimmd'``.
Unlike AIMMD's ``create_default_values_function``, which fixes the network's
device when it is created (at params load, on the CPU), it resolves the device
at every call: inside a worker task the network is moved to
``device_for('values', params)``, outside one it stays where it is and no CUDA
function is called; inputs always go to the network's current device.
"""

import numpy as np
import pytest
import torch

from aimmd import _config
from aimmd.cgaimmd import values as cgaimmd_values
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.values import ProtocolValuesFunction, protocol_values_function
from tests._cgaimmd_helpers import cgaimmd_params_namespace


class LinearNetwork(TinyFlowNetwork):
    """TinyFlowNetwork whose committor is 2 x0 - 1 (a non-trivial, exact forward)."""

    def __init__(self, n_atoms=1):
        super().__init__(n_atoms=n_atoms, hidden=4)
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, inp):
        x = torch.flatten(torch.as_tensor(inp, dtype=torch.float32), start_dim=1)
        return (self.scale * x[:, :1] - 1.0).reshape(-1, 1)


class RecordingNetwork(LinearNetwork):
    """Records the device of every forward input and every ``to`` call."""

    def __init__(self, n_atoms=1):
        super().__init__(n_atoms=n_atoms)
        self.input_devices = []
        self.moves = []

    def forward(self, inp):
        self.input_devices.append(inp.device)
        return torch.zeros(inp.shape[0], 1)

    def to(self, *args, **kwargs):
        self.moves.append(args[0] if args else kwargs.get('device'))
        return self


def test_dense():
    network = LinearNetwork(n_atoms=2)
    network.train()
    function = ProtocolValuesFunction(network, batch_size=3)
    rows = np.random.default_rng(0).normal(size=(7, 6)).astype(np.float32)
    values = function(rows)
    assert values.shape == (7,) and values.dtype == np.float64
    np.testing.assert_allclose(values, 2.0 * rows[:, 0] - 1.0, rtol=1e-6, atol=1e-6)
    assert not network.training                                     # eval mode, as AIMMD's default
    # the batch size only chunks the evaluation
    np.testing.assert_array_equal(values, ProtocolValuesFunction(network, batch_size=100)(rows))
    # float64 descriptors and a system_id keyword are accepted
    np.testing.assert_allclose(function(rows.astype(np.float64), system_id=None), values)
    assert protocol_values_function(network, batch_size=3)(rows).tolist() == values.tolist()


def test_no_grad_and_inputs_not_modified():
    network = LinearNetwork()
    rows = np.array([[0.25, 1.0, 2.0]], dtype=np.float32)
    copy = rows.copy()
    values = ProtocolValuesFunction(network)(rows)
    np.testing.assert_array_equal(rows, copy)
    assert isinstance(values, np.ndarray)
    assert network.scale.grad is None


def test_read_only_rows_are_copied_not_shared():
    """Read-only rows (a cached path's coordinates) become a tensor of their own: torch cannot
    share read-only memory (it warns once per process when asked to); writable float32 rows
    are still used without a copy."""
    from aimmd.cgaimmd.protocol import _values_input
    rows = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    rows.flags.writeable = False
    x = _values_input(rows, 'cpu', None)
    assert x.data_ptr() != rows.ctypes.data
    np.testing.assert_array_equal(x.numpy(), rows)
    writable = rows.copy()
    assert _values_input(writable, 'cpu', None).data_ptr() == writable.ctypes.data
    values = ProtocolValuesFunction(LinearNetwork())(rows)
    np.testing.assert_allclose(values, 2.0 * rows[:, 0] - 1.0, rtol=1e-6, atol=1e-6)


def test_descriptor_transform_dense_output():
    """A transform returning an array is applied before the network, as in AIMMD."""
    network = LinearNetwork()
    calls = []

    def descriptor_transform(rows):
        calls.append(np.asarray(rows).shape)
        return np.asarray(rows) * 2.0

    rows = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    values = ProtocolValuesFunction(network, descriptor_transform)(rows)
    np.testing.assert_allclose(values, 2.0 * (2.0 * rows[:, 0]) - 1.0)
    assert calls == [(2, 3)]


def test_descriptor_transform_with_system_id():
    network = LinearNetwork()
    seen = []

    def descriptor_transform(rows, system_id=None):
        seen.append(system_id)
        return rows

    function = ProtocolValuesFunction(network, descriptor_transform)
    function(np.zeros((1, 3), dtype=np.float32), system_id='lig1')
    function(np.zeros((1, 3), dtype=np.float32))
    assert seen == ['lig1', None]


@pytest.mark.graph
def test_graph_lists():
    """A transform returning PyG ``Data`` is batched with ``Batch.from_data_list(...).to_dict()``."""
    torch_geometric = pytest.importorskip('torch_geometric')
    from torch_geometric.data import Data

    class GraphNetwork(TinyFlowNetwork):
        def __init__(self):
            super().__init__(n_atoms=2)
            self.inputs = []

        def forward(self, inp):
            self.inputs.append(inp)
            positions = inp['positions']
            batch = inp['batch']
            out = torch.zeros(int(batch.max()) + 1, 1)
            return out.index_add(0, batch, positions[:, :1])

    def descriptor_transform(rows):
        rows = np.asarray(rows, dtype=np.float32).reshape(len(rows), 2, 3)
        return [Data(positions=torch.as_tensor(r), edge_index=torch.zeros(2, 0, dtype=torch.long),
                     num_nodes=2) for r in rows]

    network = GraphNetwork()
    rows = np.arange(18, dtype=np.float32).reshape(3, 6)
    values = ProtocolValuesFunction(network, descriptor_transform, batch_size=2)(rows)
    np.testing.assert_allclose(values, rows[:, 0] + rows[:, 3])
    assert isinstance(network.inputs[0], dict) and 'ptr' in network.inputs[0]
    assert [len(inp['ptr']) - 1 for inp in network.inputs] == [2, 1]
    assert torch_geometric is not None


def test_device_by_task(monkeypatch):
    """Inside a task the network is moved to ``device_for('values', params)``, outside it is not."""
    network = RecordingNetwork()
    params = cgaimmd_params_namespace(network=network, cgaimmd_parameters={'values_device': 'cpu'})
    function = ProtocolValuesFunction(network, params=params)
    rows = np.zeros((2, 3), dtype=np.float32)
    function(rows)
    assert network.moves == []                                       # outside a task: no move
    monkeypatch.setattr(_config, 'WORKER_TASK', 'shoot')
    function(rows)
    assert network.moves == [torch.device('cpu')]


def test_device_resolved_per_call(monkeypatch):
    """Train then shoot in one process: the training, then the values device, at every call."""
    resolved = []

    def fake_resolve(spec):
        resolved.append(spec)
        return torch.device('cpu')

    monkeypatch.setattr('aimmd.cgaimmd.devices.resolve_device', fake_resolve)
    network = LinearNetwork()
    params = cgaimmd_params_namespace(network=network, cgaimmd_parameters={
        'training_device': 'cuda:1', 'values_device': 'cuda:2'})
    function = ProtocolValuesFunction(network, params=params)
    rows = np.zeros((1, 3), dtype=np.float32)
    for task in ('train', 'train', 'shoot', 'kinetics_convergence', 'free'):
        monkeypatch.setattr(_config, 'WORKER_TASK', task)
        function(rows)
    assert resolved == ['cuda:1', 'cuda:1', 'cuda:2', 'cuda:1', 'cuda:2']


def test_no_cuda_query_outside_task(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('torch.cuda was queried outside a worker task')

    for name in ('is_available', 'device_count', 'current_device', 'init', 'is_initialized',
                 'set_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    network = LinearNetwork()
    params = cgaimmd_params_namespace(network=network)
    values = ProtocolValuesFunction(network, params=params)(np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(values, [1.0, 1.0])


def test_follows_network_device(monkeypatch):
    """Inputs go to the network's device at call time (AIMMD's default fixes it at build time)."""
    network = RecordingNetwork()
    function = ProtocolValuesFunction(network)
    rows = np.zeros((2, 3), dtype=np.float32)
    function(rows)
    assert network.input_devices == [torch.device('cpu')]
    other = torch.device('meta')
    monkeypatch.setattr(cgaimmd_values, 'network_device', lambda net: other)
    function(rows)
    assert network.input_devices[-1] == other


def test_width_mismatch_hint():
    network = LinearNetwork(n_atoms=2)
    function = ProtocolValuesFunction(network)
    with pytest.raises(ValueError, match='migrate_descriptors') as info:
        function(np.zeros((3, 5), dtype=np.float32))
    assert '5' in str(info.value) and '6' in str(info.value)


def test_width_uses_n_atoms_fallback():
    """A flow-matching network without ``n_atoms`` is measured through its prior (the fallback)."""

    class NoAtomsNetwork(LinearNetwork):
        def __init__(self):
            super().__init__(n_atoms=1)
            del self.n_atoms

        def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
            return torch.randn((int(n), 1, 3), generator=generator)

    network = NoAtomsNetwork()
    assert not hasattr(network, 'n_atoms')
    function = ProtocolValuesFunction(network)
    assert function(np.zeros((1, 3), dtype=np.float32)).shape == (1,)
    with pytest.raises(ValueError, match='migrate_descriptors'):
        function(np.zeros((1, 2), dtype=np.float32))


def test_empty_input():
    network = RecordingNetwork()
    function = ProtocolValuesFunction(network)
    for empty in (np.zeros((0, 3), dtype=np.float32), np.zeros(0), []):
        values = function(empty)
        assert values.shape == (0,) and values.dtype == np.float64
    assert network.input_devices == []                               # the network is not called


def test_wrong_output_size_raises():
    class TwoOutputs(LinearNetwork):
        def forward(self, inp):
            return torch.zeros(inp.shape[0], 2)

    with pytest.raises(ValueError, match=r'\(B, 1\)'):
        ProtocolValuesFunction(TwoOutputs())(np.zeros((2, 3), dtype=np.float32))


def test_descriptor_transform_batch_dict():
    """A transform returning a batch dict (node-table graph mode) feeds the network as it is."""

    class DictNetwork(LinearNetwork):
        def __init__(self):
            super().__init__(n_atoms=2)
            self.inputs = []

        def forward(self, inp):
            self.inputs.append(inp)
            return inp['x'][:, :1] + inp['x'][:, 3:4]

    def descriptor_transform(rows):
        return {'x': torch.as_tensor(np.asarray(rows, dtype=np.float32)), 'n_graphs': len(rows)}

    network = DictNetwork()
    rows = np.arange(18, dtype=np.float32).reshape(3, 6)
    values = ProtocolValuesFunction(network, descriptor_transform, batch_size=2)(rows)
    np.testing.assert_allclose(values, rows[:, 0] + rows[:, 3])
    assert [inp['n_graphs'] for inp in network.inputs] == [2, 1]
    # a batch dict given directly is evaluated as one batch (its 'ptr' gives B when present)
    batch = descriptor_transform(rows)
    np.testing.assert_allclose(ProtocolValuesFunction(network)(batch), rows[:, 0] + rows[:, 3])
    with pytest.raises(ValueError, match=r'\(B, 1\)'):
        ProtocolValuesFunction(network)({**batch, 'ptr': torch.arange(3)})
