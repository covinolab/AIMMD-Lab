"""The PaiNN flow-matching network ``aimmd.cgaimmd.flow_matching.painn.JointPaiNN``.

Tiny architecture (8 channels x 2 layers), float32, torch edge backend in the
default lane; the graph-pipeline forward needs ``--rungraph``. Ported from the
flow branch's ``tests/test_joint_painn.py`` (``bb98943``), without its
prototype converters and its mixture prior, plus:

- the port reproduces the flow branch: its recorded checkpoints and outputs
  (``tests/data/joint_painn_reference.pt``, and the golden of every switch,
  ``tests/data/flow_matching_painn_flowdev_bb98943.pt``);
- the network is a C-GAIMMD network whose generator is ``'ode'``
  (``make_generator``, ``check_network_numerics``);
- nothing of the dropped priors and converters can be reached.
"""

import ast
import importlib
import inspect
import io
import pkgutil
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd.cgaimmd.flow_matching as flow_matching
from aimmd.cgaimmd import _legacy, protocol
from aimmd.cgaimmd.flow_matching import flowmath, sampler
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.generation import check_generator, make_generator


DATA = Path(__file__).parent / 'data'

TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=False)
TOKENS = [6, 1, 1, 8, 6, 1, 7]
M = len(TOKENS)


def _molecule(n=4, seed=0):
    """(n, M, 3) float32 Å frames of a 7-atom blob with bonds of ~1.5 A (so the graph has edges)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.0, 0.0, 1.4],
                         [-1.5, -0.2, -0.3], [-1.9, -1.1, 0.2], [1.2, 1.3, 1.1]])
    return base.unsqueeze(0) + 0.15 * torch.randn(n, M, 3, generator=g)


def _randomize(network, seed=0, scale=0.3):
    """Give every parameter random values (zero-initialized outputs would hide every symmetry)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return network


def _net(seed=0, randomize=True, **overrides):
    torch.manual_seed(seed)
    kwargs = {**TINY, 'edge_backend': 'torch', **overrides}
    net = JointPaiNN(M, TOKENS, **kwargs)
    return _randomize(net, seed) if randomize else net


def _rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def _mirror():
    return torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


def _deprecations(record):
    return [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]


# ---------------------------------------------------------------------------
# the port reproduces the flow branch
# ---------------------------------------------------------------------------

def _reference_model(name):
    """The two models of ``tests/data/record_joint_painn_reference.py``, built with this package."""
    tiny = {k: v for k, v in TINY.items() if k != 'parity_breaking'}
    tiny['edge_backend'] = 'torch'
    if name == 'element_gaussian':
        return JointPaiNN(M, TOKENS, parity_breaking=False, **tiny)
    assert name == 'residue_bondedges_parity_mixture'
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6)]
    distance = torch.full((M, M), 4, dtype=torch.long)
    distance.fill_diagonal_(0)
    for i, j in bonds:
        distance[i, j] = distance[j, i] = 1
    for _ in range(4):
        for k in range(M):
            distance = torch.minimum(distance, distance[:, k:k + 1] + distance[k:k + 1, :])
    return JointPaiNN(M, [1001, 1002, 1002, 1003, 1004, 1005, 1006], atomic_numbers=TOKENS,
                      parity_breaking=True, bond_distances=distance.clamp_max(4).numpy(), **tiny)


def test_flow_branch_checkpoints_load_unchanged():
    """The flow branch's recorded checkpoints load strictly and give its outputs bit for bit.

    ``joint_painn_reference.pt`` was recorded on the flow branch (``722b6df``,
    torch 2.7.1). The second model carried a mixture prior: its ``prior.*``
    buffers are refused with the reason, the rest loads strictly, and every
    output that does not depend on the prior is the recorded one. Its prior
    draw and the pairing against a prior draw are not comparable (the network
    now always uses the Gaussian prior). Exact equality is required when
    torch and the CPU capability are the recording's, else 1e-6.
    """
    record = torch.load(DATA / 'joint_painn_reference.pt', weights_only=True)
    meta = record['meta']
    exact = (meta['torch'] == str(torch.__version__)
             and meta['cpu_capability'] == torch.backends.cpu.get_cpu_capability())
    for name, item in record['models'].items():
        net = _reference_model(name)
        state_dict = item['state_dict']
        prior_keys = sorted(k for k in state_dict if k.startswith('prior.'))
        if prior_keys:
            with pytest.raises(ValueError, match='mixture or harmonic'):
                net.load_state_dict(state_dict)
            state_dict = {k: v for k, v in state_dict.items() if k not in prior_keys}
        assert set(net.state_dict()) == set(state_dict), name           # no new keys
        net.load_state_dict(state_dict)                                    # strict
        net.eval()
        out, x = item['outputs'], item['outputs']['x']
        z, t, lam, mask = out['z'], out['t'], out['lam'], out['mask']
        with torch.no_grad():
            mine = {'z': net.to_flow(x), 'committor': net.committor(x),
                    'forward_dense': net(x.reshape(len(x), -1)),
                    'velocity': net.velocity(z, t, lam),
                    'velocity_null': net.velocity(z, t, None),
                    'velocity_masked': net.velocity(z, t, lam, uncond_mask=mask)}
            if not prior_keys:
                mine['prior_seed0'] = net.sample_prior(
                    lam, len(x), generator=torch.Generator().manual_seed(0))
                mine['flow_pair'] = net.flow_pair(z, net.sample_prior(
                    None, len(x), generator=torch.Generator().manual_seed(1)))
        assert len(mine) == (6 if prior_keys else 8)
        for key, value in mine.items():
            if exact:
                assert torch.equal(value, out[key]), (name, key)
            else:
                assert torch.allclose(value, out[key], atol=1e-6), (name, key)


@pytest.mark.parametrize('name', ['chiro_painn', 'from_topology_element',
                                  'from_topology_residue_atomname'])
def test_port_reproduces_the_flow_branch_golden(name, tmp_path):
    """Every switch on: built here, the flow branch's state dict loads and every output is its own.

    The golden (``tests/_flow_matching_painn_golden.py``) was recorded with the
    flow branch's code at ``bb98943``: element and residue/atom-name tokens
    from the topology, bond edges, topology edges, random-walk encodings,
    parity breaking, the OT pairing, ``forward`` on a graph batch dict and a
    seeded generation by the ODE (here through ``make_generator``). The
    topology-derived buffers must be identical before any weight is loaded.
    """
    golden = pytest.importorskip('tests._flow_matching_painn_golden')
    record = torch.load(golden.GOLDEN, weights_only=True)
    item = record['models'][name]
    exact = (record['meta']['torch'] == str(torch.__version__)
             and record['meta']['cpu_capability'] == torch.backends.cpu.get_cpu_capability())
    api = golden.new_api()
    net = golden.build(name, api, tmp_path)
    own = net.state_dict()
    assert set(own) == set(item['state_dict'])
    for key, value in item['state_dict'].items():
        if not key.endswith(('weight', 'bias', 'null_token')):          # buffers: from the topology
            assert torch.equal(own[key], value), key
    net.load_state_dict(item['state_dict'])
    out = golden.outputs(net, api)
    assert set(out) == set(item['outputs'])
    for key, value in item['outputs'].items():
        if exact or value.dtype == torch.bool:
            assert torch.equal(out[key], value), key
        else:
            assert torch.allclose(out[key], value, rtol=1e-5, atol=1e-5), key


# ---------------------------------------------------------------------------
# symmetries and inputs
# ---------------------------------------------------------------------------

def test_rotation_equivariance():
    """Committor invariant (1e-5), velocity equivariant (1e-4) under proper rotations."""
    net = _net().eval()
    x = _molecule()
    rot = _rotation(1)
    with torch.no_grad():
        lam_x, lam_r = net.committor(x), net.committor(x @ rot.T)
        assert torch.allclose(lam_x, lam_r, atol=1e-5)
        assert float(lam_x.abs().max()) > 1e-3                  # not trivially zero
        z = net.to_flow(x)
        t = torch.tensor([0.1, 0.4, 0.7, 1.0])
        lam = torch.tensor([-2.0, 0.0, 1.0, 12.0])
        v = net.velocity(z, t, lam)
        v_rot = net.velocity(z @ rot.T, t, lam)
        assert torch.allclose(v @ rot.T, v_rot, atol=1e-4)
        assert float(v.abs().max()) > 1e-3
        v_null = net.velocity(z @ rot.T, t, None)
        assert torch.allclose(net.velocity(z, t, None) @ rot.T, v_null, atol=1e-4)


def test_translation_invariance():
    """Committor and velocity ignore a common translation; velocity is center-of-mass free."""
    net = _net().eval()
    x = _molecule()
    shift = torch.tensor([3.7, -2.0, 11.0])
    with torch.no_grad():
        assert torch.allclose(net.committor(x), net.committor(x + shift), atol=1e-5)
        z = x / 1.5
        t = torch.full((4,), 0.5)
        lam = torch.zeros(4)
        v = net.velocity(z, t, lam)
        assert torch.allclose(v, net.velocity(z + shift, t, lam), atol=1e-4)
        assert float(v.mean(dim=1).abs().max()) < 1e-5            # COM removed per sample
        # flow units: centered, divided by coord_scale; from_flow inverts up to the translation
        assert torch.allclose(net.to_flow(x), flowmath.center(x) / 1.5)
        assert torch.allclose(net.from_flow(net.to_flow(x + shift)), flowmath.center(x), atol=1e-5)


def test_parity_symmetric_by_default():
    """Without parity breaking a mirror image has the same committor and a mirrored velocity."""
    net = _net().eval()
    x = _molecule()
    p = _mirror()
    with torch.no_grad():
        assert torch.allclose(net.committor(x), net.committor(x @ p.T), atol=1e-5)
        z, t, lam = net.to_flow(x), torch.full((4,), 0.3), torch.full((4,), 1.0)
        assert torch.allclose(net.velocity(z, t, lam) @ p.T, net.velocity(z @ p.T, t, lam),
                              atol=1e-4)
    assert net.parity_breaking is False
    assert not any('cross' in k for k in net.state_dict())


def test_parity_breaking_detects_mirror():
    """ITO's cross-product term tells a structure from its mirror image; rotations are still symmetries."""
    net = _net(parity_breaking=True).eval()
    assert net.parity_breaking is True
    assert any('filter_cross' in k for k in net.state_dict())
    x = _molecule()
    with torch.no_grad():
        lam, lam_mirror = net.committor(x), net.committor(x @ _mirror().T)
        assert float((lam - lam_mirror).abs().max()) > 1e-3
        assert torch.allclose(lam, net.committor(x @ _rotation(2).T), atol=1e-5)
    # freshly built, the cross term is exactly zero (filter_cross zero-initialized): E(3) function
    fresh = _net(parity_breaking=True, randomize=False)
    for msg in fresh.trunk.messages:
        assert not bool(msg.filter_cross.weight.any()) and not bool(msg.filter_cross.bias.any())
    with pytest.raises(ValueError, match='n_layers'):
        JointPaiNN(M, TOKENS, **{**TINY, 'parity_breaking': True, 'n_layers': 1})


def test_bond_edges_onehot():
    """bond_distances -> trunk.bond_class = clip(d, 1, 4) - 1; the filters see 4 more features."""
    distance = np.full((M, M), 4, dtype=np.int16)
    np.fill_diagonal(distance, 0)
    for i, j, d in [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (4, 5, 1), (1, 2, 2), (3, 6, 3),
                    (0, 6, 2)]:
        distance[i, j] = distance[j, i] = d
    net = _net(bond_distances=distance)
    expected = torch.as_tensor((np.clip(distance, 1, 4) - 1).astype(np.uint8))
    assert net.trunk.bond_class.dtype == torch.uint8
    assert torch.equal(net.trunk.bond_class, expected)
    assert net.trunk.messages[0].filter.in_features == TINY['n_basis'] + 4
    plain = _net()
    assert plain.trunk.messages[0].filter.in_features == TINY['n_basis']
    assert 'trunk.bond_class' not in plain.state_dict()
    negative = distance.copy()
    negative[3, 6] = negative[6, 3] = -1                    # no negative "disconnected" marker
    with pytest.raises(ValueError, match='bond_distances'):
        _net(bond_distances=negative)
    x = _molecule()
    other = distance.copy()
    other[1, 2] = other[2, 1] = 4
    changed = _net(bond_distances=other)
    with torch.no_grad():
        assert float((net.committor(x) - changed.committor(x)).abs().max()) > 1e-4
        assert torch.allclose(net.committor(x), net.committor(x @ _rotation(3).T), atol=1e-5)


def test_forward_dense_equals_committor():
    """forward: (B, 3 M) flat descriptors (AIMMD's input) -> (B, 1), equal to committor."""
    net = _net().eval()
    x = _molecule(5)
    with torch.no_grad():
        out = net(x.reshape(5, -1))
        assert out.shape == (5, 1) and out.dtype == torch.float32
        assert torch.allclose(out[:, 0], net.committor(x), atol=1e-6)
        assert torch.allclose(net(x)[:, 0], out[:, 0], atol=1e-6)
    with pytest.raises(ValueError):
        net(torch.zeros(2, 3 * M + 3))
    # the graph dict of build_graph_batch (edges built with the same call) gives the same logits
    from aimmd.cgaimmd.flow_matching.graph_batch import build_graph_batch
    with torch.no_grad():
        d = build_graph_batch(x + 50.0, TINY['cutoff'], backend='torch')
        assert torch.allclose(net(d)[:, 0], out[:, 0], atol=1e-5)
        bad = dict(d, ptr=torch.tensor([0, 3, 5 * M]))
        with pytest.raises(ValueError, match='generator atoms'):
            net(bad)
    # chunked evaluation in eval mode equals one call
    chunked = _net(committor_chunk=2).eval()
    with torch.no_grad():
        assert torch.allclose(chunked.committor(x), net.committor(x), atol=1e-6)


def test_gradients_through_forward():
    """forward is differentiable in the positions, twice (the gradient penalty's create_graph)."""
    net = _net()
    rows = _molecule(3).reshape(3, -1).requires_grad_(True)
    lam = net(rows)[:, 0]
    (grad,) = torch.autograd.grad(lam.sum(), rows, create_graph=True)
    penalty = grad.pow(2).mean()
    penalty.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in net.parameters())


def _structure(tmp_path, positions):
    """A PDB of the 7 generator atoms (the structure FlowGraphTransform builds its universe from)."""
    import MDAnalysis as mda
    universe = mda.Universe.empty(M, n_residues=1, atom_resindex=[0] * M, trajectory=True)
    universe.add_TopologyAttr('names', ['C1', 'H1', 'H2', 'O1', 'C2', 'H3', 'N1'])
    universe.add_TopologyAttr('types', ['C', 'H', 'H', 'O', 'C', 'H', 'N'])
    universe.add_TopologyAttr('elements', ['C', 'H', 'H', 'O', 'C', 'H', 'N'])
    universe.add_TopologyAttr('resnames', ['MOL'])
    universe.add_TopologyAttr('resids', [1])
    universe.atoms.positions = np.asarray(positions)
    universe.dimensions = np.array([30, 30, 30, 90, 90, 90], dtype=np.float32)
    structure = str(tmp_path / 'generator_atoms.pdb')
    universe.atoms.write(structure)
    return structure


@pytest.mark.graph
def test_forward_graph_equals_committor(tmp_path):
    """forward on the cached graphs of MD frames equals committor on their coordinates."""
    pytest.importorskip('aimmd.network.graph_utils')
    from torch_geometric.data import Batch
    from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
    x = _molecule(4) + 20.0
    structure = _structure(tmp_path, x[0].numpy())
    transform = FlowGraphTransform(structure, TINY['cutoff'],
                                   graph_cache=tmp_path / 'graphs.sqlite', write_graph_cache=True)
    try:
        data = transform(x.reshape(4, -1).numpy())
    finally:
        transform.close()
    net = _net(edge_backend='pyg').eval()
    with torch.no_grad():
        d = Batch.from_data_list(data).to_dict()
        assert torch.allclose(net(d)[:, 0], net.committor(x), atol=1e-4)
        torch_net = _net(edge_backend='torch').eval()
        assert torch.allclose(torch_net.committor(x), net.committor(x), atol=1e-5)


@pytest.mark.graph
def test_forward_graph_precision_steep_committor(tmp_path):
    """A steep committor near 1/2: graphs of the cache's default box match committor within the start-up tolerance.

    The graph pipeline moves every frame to the center of its box in float32,
    so the rounding error of the graph positions grows with the box. The
    start-up checks accept ``|forward(graph) - committor| <= 1e-4 + 1e-4
    |lambda|`` (``protocol.check_network_numerics``); with the default box of
    100 A a committor with gradients of more than 10 per A stays well inside
    it, while a 1000 A box is several times worse. The default (cache-free)
    route keeps the coordinates as they are and has no such error.
    """
    pytest.importorskip('aimmd.network.graph_utils')
    from torch_geometric.data import Batch
    from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
    x = _molecule(64) + torch.tensor([23.7, 41.2, 17.9])
    rows = x.reshape(len(x), -1).numpy()
    structure = _structure(tmp_path, x[0].numpy())
    net = _net(edge_backend='torch').eval()
    head = net.committor_head.heads[0][-1]
    with torch.no_grad():
        head.weight.mul_(10.0)
        head.bias.sub_(net.committor(x).mean() / M)          # lambda ~ 0: the tightest tolerance
        lam = net.committor(x).double()
    x_grad = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(net.committor(x_grad).sum(), x_grad)
    assert float(grad.reshape(len(x), -1).norm(dim=1).median()) > 10.0     # steep
    assert float(lam.abs().median()) < 3.0                      # near 1/2: tight tolerance

    def worst(box_length):
        transform = FlowGraphTransform(
            structure, TINY['cutoff'], box_length=box_length, write_graph_cache=True,
            graph_cache=tmp_path / f'graphs_box{box_length:g}.sqlite')
        try:
            data = transform(rows)
        finally:
            transform.close()
        with torch.no_grad():
            out = net(Batch.from_data_list(data).to_dict())[:, 0].double()
        return float(((out - lam).abs() / (1e-4 + 1e-4 * lam.abs())).max())

    default = worst(FlowGraphTransform.__init__.__kwdefaults__['box_length'])
    assert default < 0.5                                   # measured: 0.24 (100 A)
    assert worst(1000.0) > 4 * default                     # measured: 3.0, it would fail
    with torch.no_grad():
        plain = net(FlowGraphTransform(structure, TINY['cutoff'], edge_backend='torch')(rows))
    assert float(((plain[:, 0].double() - lam).abs() / (1e-4 + 1e-4 * lam.abs())).max()) < 0.05
    transform = FlowGraphTransform(structure, TINY['cutoff'], graph_cache=tmp_path /
                                   'graphs_box100.sqlite')
    data = transform(rows[:4])
    transform.close()
    report = protocol.check_network_numerics(net, x[:4].numpy(),
                                             Batch.from_data_list(data).to_dict())
    assert report['n_atoms'] == M


def test_architecture_kwargs_required():
    """Every architecture hyperparameter is required (no silent defaults from another system)."""
    for name in TINY:
        kwargs = {k: v for k, v in TINY.items() if k != name}
        with pytest.raises(TypeError, match=name):
            JointPaiNN(M, TOKENS, **kwargs)
    with pytest.raises(TypeError):
        JointPaiNN(M, TOKENS, TINY['coord_scale'])                     # keyword-only
    with pytest.raises(TypeError, match='unexpected keyword'):
        JointPaiNN(M, TOKENS, no_such_switch=True, **TINY)
    with pytest.raises(ValueError):
        JointPaiNN(M, TOKENS[:-1], **TINY)                            # token count
    with pytest.raises(ValueError, match='atomic_numbers'):
        JointPaiNN(M, [1001 + i for i in range(M)], **TINY)          # tokens are not elements
    net = JointPaiNN(M, [1001, 1002, 1002, 1003, 1001, 1002, 1004], atomic_numbers=TOKENS,
                     **TINY)
    assert net.atomic_numbers.tolist() == TOKENS
    assert net.trunk.z_table.tolist() == [1001, 1002, 1003, 1004]
    assert net.atom_tokens_numpy().tolist() == [1001, 1002, 1002, 1003, 1001, 1002, 1004]


def _write_ethanol(tmp_path):
    """A 9-atom ethanol topology (self-contained, no force-field include) and its structure."""
    top = tmp_path / 'topol.top'
    top.write_text("""
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
CT 6 12.011 0.0 A 0.35 0.27
HC 1 1.008 0.0 A 0.25 0.12
OH 8 15.999 0.0 A 0.31 0.71
HO 1 1.008 0.0 A 0.0 0.0

[ moleculetype ]
ETH 3

[ atoms ]
1 CT 1 ETH C1 1 -0.18 12.011
2 HC 1 ETH H11 1 0.06 1.008
3 HC 1 ETH H12 1 0.06 1.008
4 HC 1 ETH H13 1 0.06 1.008
5 CT 1 ETH C2 1 0.14 12.011
6 HC 1 ETH H21 1 0.03 1.008
7 HC 1 ETH H22 1 0.03 1.008
8 OH 1 ETH O 1 -0.68 15.999
9 HO 1 ETH HO 1 0.42 1.008

[ bonds ]
1 2
1 3
1 4
1 5
5 6
5 7
5 8
8 9

[ system ]
ethanol

[ molecules ]
ETH 1
""")
    positions = np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                          [-0.63, 0.63, -0.63], [1.5, 0.0, 0.0], [1.9, 1.0, 0.0],
                          [1.9, -0.5, 0.9], [2.0, -0.7, -1.1], [2.9, -0.7, -1.1]]) + 10.0
    lines = ['ethanol', '9']
    names = ['C1', 'H11', 'H12', 'H13', 'C2', 'H21', 'H22', 'O', 'HO']
    for i, (name, pos) in enumerate(zip(names, positions / 10.0)):
        lines.append(f'{1:5d}{"ETH":<5s}{name:>5s}{i + 1:5d}{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}')
    lines.append('   3.00000   3.00000   3.00000')
    gro = tmp_path / 'conf.gro'
    gro.write_text('\n'.join(lines) + '\n')
    return str(top), str(gro)


def _ethanol_descriptors(gro):
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    return GeneratorDescriptors(gro, GeneratorAtoms(n_solute=9))


def test_from_topology_tokens_and_bonds(tmp_path):
    """from_topology builds tokens, elements, bond classes and the OT spec from the topology tooling."""
    pytest.importorskip('networkx')
    from aimmd.cgaimmd.preparation import topology
    top, gro = _write_ethanol(tmp_path)
    descriptors = _ethanol_descriptors(gro)
    net = JointPaiNN.from_topology(top, descriptors, **{**TINY, 'edge_backend': 'torch'})
    mt = topology.model_topology(top, n_solute=9)
    assert net.n_atoms == 9
    assert net.atomic_numbers.tolist() == list(mt.elements)
    assert all(int(t) > 1000 for t in net.atom_tokens_numpy())      # residue_atomname tokens
    distance = topology.topological_distances(9, mt.bonds, max_distance=4)
    assert torch.equal(net.trunk.bond_class,
                       torch.as_tensor((np.clip(distance, 1, 4) - 1).astype(np.uint8)))
    assert net.ot_symmetry is not None
    element = JointPaiNN.from_topology(top, descriptors, embedding='element', bond_edges=False,
                                       symmetry='none', **{**TINY, 'edge_backend': 'torch'})
    assert element.atom_tokens_numpy().tolist() == list(mt.elements)
    assert 'trunk.bond_class' not in element.state_dict()


def test_from_topology_missing_architecture_kwarg(tmp_path):
    pytest.importorskip('networkx')
    top, gro = _write_ethanol(tmp_path)
    kwargs = {k: v for k, v in TINY.items() if k != 'velocity_radial'}
    with pytest.raises(TypeError, match='velocity_radial'):
        JointPaiNN.from_topology(top, _ethanol_descriptors(gro), **kwargs)


def test_load_state_dict_checks(recwarn):
    """Atom count and scales are checked; a flow-branch prior in a checkpoint is refused with the reason."""
    net = _net()
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    sd = torch.load(buffer, weights_only=True)
    fresh = _net(seed=5)
    fresh.load_state_dict(sd)                                          # strict
    x = _molecule()
    with torch.no_grad():
        assert torch.equal(fresh.committor(x), net.committor(x))
    # another atom count
    bigger = JointPaiNN(M + 1, TOKENS + [1], **TINY)
    with pytest.raises(ValueError, match='atoms'):
        bigger.load_state_dict(sd)
    # coord_scale / lambda_clip differ: warned, the checkpoint (a buffer) wins
    other = JointPaiNN(M, TOKENS, **{**TINY, 'coord_scale': 2.0})
    with pytest.warns(UserWarning, match='coord_scale'):
        other.load_state_dict(sd)
    assert other.coord_scale == pytest.approx(1.5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _net(seed=3).load_state_dict(sd)
    # a checkpoint of the flow branch with another prior: refused strictly, the rest loads leniently
    with_prior = dict(sd, **{'prior.bank_a': torch.zeros(2, M, 3),
                             'prior.bank_b': torch.zeros(3, M, 3), 'prior.sigma': torch.tensor(0.4)})
    with pytest.raises(ValueError, match=r'prior\.bank_a.*not part of C-GAIMMD.*strict=False'):
        fresh.load_state_dict(with_prior)
    third = _net(seed=7)
    with pytest.warns(UserWarning, match=r'not loaded: prior\.bank_a, prior\.bank_b, prior\.sigma'):
        result = third.load_state_dict(with_prior, strict=False)
    assert result.unexpected_keys == [] and result.missing_keys == []
    with torch.no_grad():
        assert torch.equal(third.committor(x), net.committor(x))
    assert _legacy.dropped_state_keys(sd) == {}


def test_state_dict_layout_and_reset():
    """Buffers of the flow branch's layout, float scalars read from them, reset_parameters re-initializes."""
    net = _net(randomize=False)
    sd = net.state_dict()
    for key in ('atomic_numbers', 'atom_tokens', 'coord_scale', 'lambda_clip', 'trunk.z_table',
                'committor_head.heads.0.0.weight', 'velocity_head.gate.0.weight'):
        assert key in sd
    assert sd['coord_scale'].dtype == torch.float32 and sd['coord_scale'].dim() == 0
    assert sd['atom_tokens'].dtype == torch.long
    assert isinstance(net.lambda_clip, float) and net.lambda_clip == 10.0
    assert isinstance(net.n_atoms, int) and net.n_atoms == M
    assert net.cutoff == TINY['cutoff']
    buffer = io.BytesIO()
    torch.save(sd, buffer)
    buffer.seek(0)
    torch.load(buffer, weights_only=True)
    # untrained: committor ~ 0 (last layer N(0, 1e-6)), velocity exactly 0 (zero-initialized)
    x = _molecule()
    with torch.no_grad():
        assert float(net.committor(x).abs().max()) < 0.1
        assert not bool(net.velocity(net.to_flow(x), torch.full((4,), 0.5), None).any())
    _randomize(net)
    with torch.no_grad():
        assert bool(net.velocity(net.to_flow(x), torch.full((4,), 0.5), None).any())
    net.reset_parameters()
    with torch.no_grad():
        assert not bool(net.velocity(net.to_flow(x), torch.full((4,), 0.5), None).any())
    assert net.coord_scale == 1.5 and net.atom_tokens.tolist() == TOKENS
    assert not hasattr(net, 'generate')                   # generation goes through a generator


def test_flow_pair_uses_ot_symmetry():
    """flow_pair: proper Kabsch (always) plus the permutations of ot_symmetry."""
    net = _net()
    x0 = flowmath.center(_molecule(3, seed=4))
    swapped = x0[:, [0, 2, 1, 3, 4, 5, 6]]
    rot = _rotation(5)
    x1 = swapped @ rot.T
    plain = net.flow_pair(x1, x0)                          # rotation only: H1/H2 stay swapped
    assert float((plain - x0).abs().max()) > 1e-2
    spec = SimpleNamespace(kwargs=lambda: {'groups': (), 'factors': (((1, 2), ((0, 1), (1, 0))),),
                                           'molecule_blocks': ()})
    paired = _net(ot_symmetry=spec).flow_pair(x1, x0)
    assert torch.allclose(paired, x0, atol=1e-4)


def test_no_cuda_at_construction(monkeypatch):
    """Construction (and CPU use) never touches CUDA."""
    def boom(*args, **kwargs):
        raise AssertionError('CUDA touched')
    for name in ('is_available', 'current_device', 'device_count', 'init', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, boom)
    net = _net(randomize=False)
    with torch.no_grad():
        net.committor(_molecule())
        net.velocity(net.sample_prior(None, 2), torch.full((2,), 0.5), None)
    assert not torch.cuda.is_initialized()


# ---------------------------------------------------------------------------
# a C-GAIMMD network whose generator is the flow-matching ODE
# ---------------------------------------------------------------------------

def test_protocol_and_the_ode_generator():
    """The committor protocol passes; the default generator is 'ode'; generation and start-up checks run."""
    net = _net()
    protocol.check_network_protocol(net)
    assert protocol.is_cgaimmd_network(net)
    assert protocol.n_atoms_of(net) == M
    assert protocol.lambda_clip_of(net, {'lambda_clip': None}) == 10.0
    assert net.default_generator() == {'method': 'ode'}
    assert sampler.flow_matching_problems(net) == []
    generator = make_generator(None, net)                 # params.cgaimmd_generator unset
    assert isinstance(generator, sampler.ODEGenerator)
    assert generator == make_generator('ode', net) == sampler.ODEGenerator()
    assert check_generator(generator, net) == []
    x = _molecule(4)
    report = protocol.check_network_numerics(net, x.numpy(), x.reshape(4, -1).numpy(),
                                             generation='ode', n_generate=3)
    assert report['n_atoms'] == M and report['committor_max_abs_diff'] < 1e-4
    assert report['generation'] == {'method': 'ode', 'settings': generator.settings()}
    assert tuple(report['sample_shape']) == (3, M, 3)
    # a generation through the interface is the ODE from the network's own prior, reproducibly
    lam = torch.linspace(-2.0, 2.0, 5)
    ode = make_generator({'method': 'ode', 'sampler': 'rk4', 'n_steps': 3, 'chunk': 2}, net)
    out = ode.generate(net, lam, generator=torch.Generator().manual_seed(3))
    assert out.shape == (5, M, 3) and torch.isfinite(out).all()
    again = sampler.generate(net, lam, sampler='rk4', n_steps=3, chunk=2,
                             generator=torch.Generator().manual_seed(3))
    assert torch.equal(out, again)
    # coordinate_committor uses the committor fast path
    assert torch.allclose(protocol.coordinate_committor(net, x), net.committor(x), atol=1e-6)


def test_graph_forward_input_passes_the_start_up_check(tmp_path):
    """check_network_numerics with the default graph route's batch dict as the forward input."""
    from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
    x = _molecule(4) + 7.0
    transform = FlowGraphTransform(_structure(tmp_path, x[0].numpy()), TINY['cutoff'],
                                   edge_backend='torch')
    net = _net()
    report = protocol.check_network_numerics(net, x.numpy(), transform(x.reshape(4, -1).numpy()),
                                             generation={'method': 'ode', 'n_steps': 2})
    assert report['committor_max_abs_diff'] < 1e-4


# ---------------------------------------------------------------------------
# what was dropped stays dropped
# ---------------------------------------------------------------------------

DROPPED_WORDS = re.compile(r'harmonic|mixture|cfaimmd|mace', re.IGNORECASE)
"""The harmonic and mixture priors and the prototype (MACE) converters are not part of C-GAIMMD."""

NEW_MODULES = ('flowmath', 'prior', 'graph_batch', 'graphs', 'painn', 'fit', 'pretrain')
"""The modules of this port."""


def _public_names(obj):
    return [name for name in dir(obj) if not name.startswith('__')]


def test_dropped_priors_and_converters_are_not_reachable():
    """No module, function, class, attribute or parameter of the port is named after them."""
    found = []
    for info in pkgutil.iter_modules(flow_matching.__path__):
        module = importlib.import_module(f'aimmd.cgaimmd.flow_matching.{info.name}')
        for name in _public_names(module):
            value = getattr(module, name)
            if DROPPED_WORDS.search(name):
                found.append(f'{module.__name__}.{name}')
            if inspect.isclass(value) and value.__module__ == module.__name__:
                found += [f'{module.__name__}.{name}.{member}' for member in _public_names(value)
                          if DROPPED_WORDS.search(member)]
            if callable(value):
                try:
                    parameters = inspect.signature(value).parameters
                except (TypeError, ValueError):
                    continue
                found += [f'{module.__name__}.{name}({p})' for p in parameters
                          if DROPPED_WORDS.search(p)]
    net = _net(randomize=False)
    found += [f'JointPaiNN().{name}' for name in _public_names(net) if DROPPED_WORDS.search(name)]
    found += [f'JointPaiNN.from_topology({p})'
              for p in inspect.signature(JointPaiNN.from_topology).parameters
              if DROPPED_WORDS.search(p)]
    assert found == []
    assert not hasattr(net, 'attach_prior')
    for name in ('MixturePrior', 'HarmonicPrior', 'convert_cfaimmd_state_dict'):
        for module in NEW_MODULES:
            assert not hasattr(importlib.import_module(f'aimmd.cgaimmd.flow_matching.{module}'),
                               name)


def test_the_port_does_not_mention_them():
    """Not even the source of the ported modules (docstrings and messages included) names them.

    The reasons for the old names live in ``aimmd.cgaimmd._legacy`` alone.
    """
    root = Path(flow_matching.__file__).parent
    for module in NEW_MODULES:
        source = (root / f'{module}.py').read_text()
        assert DROPPED_WORDS.findall(source) == [], module
        ast.parse(source)


def test_old_names_say_why(recwarn):
    """Old imports and keywords of the dropped parts raise with the reason; None keywords warn once."""
    with pytest.raises(ImportError, match='cfaimmd'):
        from aimmd.network.joint_painn import convert_cfaimmd_state_dict      # noqa: F401
    with pytest.raises(ImportError, match='mixture prior is not part of C-GAIMMD'):
        from aimmd.flow.prior import MixturePrior                             # noqa: F401
    import aimmd.flow.harmonic as harmonic
    with pytest.raises(ImportError, match='harmonic prior is not part of C-GAIMMD'):
        harmonic.HarmonicPrior
    # the old module name of the network resolves to this class, with one warning
    from aimmd.network.joint_painn import JointPaiNN as old
    assert old is JointPaiNN
    assert len([m for m in _deprecations(recwarn) if 'aimmd.network.joint_painn.JointPaiNN' in m
                and 'aimmd.cgaimmd.flow_matching.painn.JointPaiNN' in m]) == 1
    # the old prior keywords: None (the Gaussian) is accepted with one warning, anything else raises
    recwarn.clear()
    for _ in range(2):
        net = JointPaiNN(M, TOKENS, prior=None, harmonic_prior=None, **TINY)
    assert net.prior.kind == 'gaussian'
    messages = _deprecations(recwarn)
    assert len(messages) == 2
    assert any("JointPaiNN(prior=None) is deprecated" in m for m in messages)
    assert any("JointPaiNN(harmonic_prior=None) is deprecated" in m
               and 'harmonic prior is not part of C-GAIMMD' in m for m in messages)
    with pytest.raises(TypeError, match=r'JointPaiNN\(prior=\.\.\.\) is not supported'):
        JointPaiNN(M, TOKENS, prior=object(), **TINY)
    with pytest.raises(TypeError, match=r'harmonic_prior=\.\.\.\) is not supported: the harmonic'):
        JointPaiNN(M, TOKENS, harmonic_prior={'kappa': 1.0}, **TINY)


def test_old_template_keywords_of_from_topology(tmp_path, recwarn):
    """The flow branch's templates pass harmonic_prior=None for the Gaussian: still accepted, with a warning.

    The templates call it on the subclass a params file defines.
    """
    pytest.importorskip('networkx')

    class Network(JointPaiNN):
        pass

    top, gro = _write_ethanol(tmp_path)
    descriptors = _ethanol_descriptors(gro)
    net = Network.from_topology(top, descriptors, embedding='element', harmonic_prior=None,
                                **{**TINY, 'edge_backend': 'torch'})
    assert net.n_atoms == 9 and type(net) is Network
    assert any('harmonic_prior=None' in m for m in _deprecations(recwarn))
    with pytest.raises(TypeError, match='harmonic prior is not part of C-GAIMMD'):
        Network.from_topology(top, descriptors, harmonic_prior={'kappa': 0.3},
                              **{**TINY, 'edge_backend': 'torch'})
    with pytest.raises(TypeError, match="unexpected keyword argument 'no_such_switch'"):
        Network(M, TOKENS, no_such_switch=1, **TINY)


PARAMS = '''\
"""A C-GAIMMD params file with the PaiNN network in graph mode (the default route, no cache)."""
import numpy as np
from aimmd.cgaimmd.flow_matching.graphs import FlowGraphTransform
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN

engine = 'toy'
states = 'ARB'
chain_type = 'cgaimmd'
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
round_shooting_index = True
gen_temperature = 300.0
cgaimmd_parameters = {'placement': 'none', 'window_after_trained_shots': 5}


def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01


def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))


def descriptors_function(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)


_graphs = FlowGraphTransform('generator_atoms.pdb', 4.0, edge_backend='torch')


def descriptor_transform(descriptors):
    return _graphs(descriptors)


class Network(JointPaiNN):
    """Local subclass: Params.save writes 'from <this module> import network'."""


network = Network(3, [6, 6, 6], bond_distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                  coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6,
                  cutoff=4.0, cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,),
                  velocity_hidden=8, velocity_radial=6, velocity_min_distance=0.7,
                  parity_breaking=True, edge_backend='torch')
'''


def test_a_params_file_with_the_network_in_graph_mode(tmp_path, monkeypatch):
    """chain_type='cgaimmd' with the PaiNN network and graph mode's default route: it loads, its
    generator is 'ode', its values come through the batch dicts, and no graph cache is written."""
    import MDAnalysis as mda
    import aimmd
    from aimmd.cgaimmd.values import ProtocolValuesFunction
    from tests._helpers_unit import write_trajectory
    for kind, x0 in (('A', -1.5), ('B', 1.5)):
        positions = np.zeros((4, 3, 3), dtype=np.float32)
        positions[:, :, 0] = x0 + 0.05 * np.arange(4)[:, None]
        positions[:, :, 1] = 1.5 * np.arange(3)[None, :]
        write_trajectory(tmp_path, stem=f'seeds_{kind}', positions=positions)
    universe = mda.Universe.empty(3, n_residues=1, atom_resindex=[0] * 3, trajectory=True)
    universe.add_TopologyAttr('names', ['C1', 'C2', 'C3'])
    universe.add_TopologyAttr('types', ['C'] * 3)
    universe.add_TopologyAttr('resnames', ['MOL'])
    universe.add_TopologyAttr('resids', [1])
    universe.atoms.positions = np.array([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 3.0, 0.0]])
    universe.dimensions = np.array([30, 30, 30, 90, 90, 90], dtype=np.float32)
    universe.atoms.write(str(tmp_path / 'generator_atoms.pdb'))
    (tmp_path / 'params.py').write_text(PARAMS)
    monkeypatch.chdir(tmp_path)
    params = aimmd.Params.load(str(tmp_path / 'params.py'), save=False)
    network = _randomize(params.network, 4, scale=0.2).eval()
    assert isinstance(params.values_function, ProtocolValuesFunction)
    assert params.values_function.descriptor_transform is params.descriptor_transform
    assert isinstance(make_generator(params.cgaimmd_generator, network), sampler.ODEGenerator)
    g = np.random.default_rng(0)
    rows = (np.tile([0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0], (5, 1))
            + 0.2 * g.normal(size=(5, 9))).astype(np.float32)
    assert isinstance(params.descriptor_transform(rows), dict)
    values = params.values_function(rows)
    with torch.no_grad():
        dense = network.committor(torch.as_tensor(rows)).double().numpy()
    assert float(np.abs(dense).max()) > 1e-3
    assert np.allclose(values, dense, atol=1e-5)
    assert not list(tmp_path.rglob('*.sqlite*'))
