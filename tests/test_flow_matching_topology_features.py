"""Topology features of the PaiNN network, and the force-field geometry they are built from.

- **Topology edges** (``topology_geometry``): every pair within three bonds is
  an edge at every distance, carrying the force-field reference distances.
- **Random-walk positional encodings** (``positional_encoding``).
- **Reference geometry** (``aimmd.cgaimmd.preparation.topology.reference_geometry``):
  the bond lengths and 1-3 / 1-4 distance ranges of the force field, never of
  a frame, which the topology edges carry.

Both features are off by default, and then the network is the one without
them (its checkpoints and outputs: ``tests/test_flow_matching_painn.py``).
Tiny architecture, float32, torch edge backend; the graph-pipeline forward
needs ``--rungraph``. Ported from the flow branch's
``tests/test_flow_topology_features.py`` and the reference-geometry tests of
its ``tests/test_flow_harmonic.py`` (``bb98943``); the harmonic prior itself
is not part of C-GAIMMD.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aimmd.cgaimmd.flow_matching import flowmath
from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
from aimmd.cgaimmd.preparation import topology as T

from ._cgaimmd_topology_fixtures import (ETHANOL_POSITIONS, ETHANOL_TOP, MIXED_TOP,
                                         write as _write, write_ethanol_gro as _write_ethanol_gro)

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'

TINY = dict(coord_scale=1.7, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, parity_breaking=False,
            edge_backend='torch')
SOLUTE_TOP = ETHANOL_TOP.replace('NA  1\nSOL 3', 'SOL 3')
ELEMENTS = [6, 1, 1, 1, 6, 1, 1, 8, 1]
M = len(ELEMENTS)


def _randomize(network, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return network


def _geometry(tmp_path):
    top = T.read_topology(_write(tmp_path, SOLUTE_TOP))
    mt = T.model_topology(top, n_solute=M)
    return T.reference_geometry(top, mt), mt


def _net(tmp_path, *, topology_edges=True, pe=False, seed=0, randomize=True, **overrides):
    geometry, mt = _geometry(tmp_path)
    kwargs = dict(TINY, **overrides)
    if topology_edges:
        kwargs['topology_geometry'] = geometry
    if pe:
        kwargs['positional_encoding'] = T.random_walk_encoding(M, mt.bonds, steps=16)
    torch.manual_seed(seed)
    net = JointPaiNN(M, ELEMENTS, **kwargs)
    return _randomize(net, seed) if randomize else net


def _frames(n=4, seed=0, stretch=1.0):
    g = torch.Generator().manual_seed(seed)
    base = torch.as_tensor(ETHANOL_POSITIONS, dtype=torch.float32) * stretch
    return base.unsqueeze(0) + 0.1 * torch.randn(n, M, 3, generator=g)


def _rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


def test_defaults_add_no_parameters_or_buffers(tmp_path):
    """No topology_geometry, no positional_encoding: the state dict and filter widths of before."""
    plain = _net(tmp_path, topology_edges=False)
    assert not any('topology' in k or 'positional' in k or 'pe_to_embed' in k
                   for k in plain.state_dict())
    assert plain.trunk.messages[0].filter.in_features == TINY['n_basis']
    assert plain.velocity_head.pair[0].in_features == 2 * plain.trunk.n_scalar + TINY['velocity_radial']
    assert plain.trunk.topology_edges is False and plain.trunk.positional_encoding_steps == 0


# ---------------------------------------------------------------------------
# topology edges
# ---------------------------------------------------------------------------

def test_merge_topology_edges():
    """The radius graph merged with the topology template: every topology edge once, radius-only edges kept."""
    from aimmd.cgaimmd.flow_matching.graphs import merge_topology_edges, topology_edge_template
    pairs = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
    separation = np.array([1, 1, 2, 1])
    template = topology_edge_template(5, pairs, separation, np.array([1.0, 1.1, 1.8, 1.2]),
                                      np.array([1.0, 1.1, 1.8, 1.2]))
    assert template['index'].shape == (2, 8)
    assert template['separation'].tolist() == [1, 1, 2, 1] * 2
    assert template['pair_mask'].shape == (5, 5) and bool(template['pair_mask'][1, 0])
    assert not bool(template['pair_mask'][0, 3])
    # two samples of 5 atoms; the radius graph has the topology pair (0, 1) and the far pair (0, 4)
    edge_index = torch.tensor([[1, 0, 4, 0, 6, 9],
                               [0, 1, 0, 4, 5, 5]])                # global [source, target]
    batch = torch.arange(2).repeat_interleave(5)
    ptr = torch.tensor([0, 5, 10])
    merged, ids = merge_topology_edges(edge_index, batch, ptr, template['index'],
                                       template['pair_mask'])
    edges = list(map(tuple, merged.t().tolist()))
    assert len(edges) == len(set(edges))                              # no duplicates
    topo = {(int(a) + o, int(b) + o) for o in (0, 5) for a, b in template['index'].t().tolist()}
    radius_only = {(4, 0), (0, 4), (9, 5)}
    assert set(edges) == topo | radius_only
    for (a, b), k in zip(edges, ids.tolist()):
        if k < 0:
            assert (a, b) in radius_only
        else:
            o = 0 if a < 5 else 5
            assert (a - o, b - o) == tuple(template['index'][:, k].tolist())


def test_topology_edges_present_at_any_distance(tmp_path):
    """Every pair within three bonds is an edge at every distance; the indicator says radius-only, topology-only or both."""
    net = _net(tmp_path)
    geometry, _ = _geometry(tmp_path)
    x = torch.cat([_frames(1), _frames(1, stretch=4.0)])          # compact, then stretched apart
    pos = flowmath.center(x).reshape(-1, 3)
    batch, ptr = net._layout(2, pos.device)
    radius = net._edges(pos, batch)
    merged, ids = net._topology_edges(radius, batch, ptr)
    edges = set(map(tuple, merged.t().tolist()))
    for sample in range(2):
        for i, j in geometry.pairs:
            a, b = int(i) + sample * M, int(j) + sample * M
            assert (a, b) in edges and (b, a) in edges
    features, envelope = net.trunk.edge_features(pos, batch, ptr, merged, ids)
    n_basis = TINY['n_basis']
    one_hot = features[:, n_basis:n_basis + 4]
    indicator = features[:, n_basis + 4:n_basis + 7]
    reference = features[:, n_basis + 7:n_basis + 9]
    r = (pos[merged[0]] - pos[merged[1]]).norm(dim=-1)
    topo = ids >= 0
    assert torch.equal(one_hot.argmax(dim=1)[~topo], torch.full((int((~topo).sum()),), 3))
    sep = torch.as_tensor(net.trunk.topology_separation[ids[topo]].long())
    assert torch.equal(one_hot.argmax(dim=1)[topo], sep - 1)
    assert torch.allclose(indicator[~topo], torch.tensor([1.0, 0.0, 0.0]).expand(int((~topo).sum()), 3))
    far = topo & (r >= TINY['cutoff'])
    near = topo & (r < 0.5 * TINY['cutoff'])
    assert int(far.sum()) > 0 and int(near.sum()) > 0
    assert torch.allclose(indicator[far], torch.tensor([0.0, 1.0, 0.0]).expand(int(far.sum()), 3))
    assert bool((indicator[near, 2] > 0.5).all())                    # mostly "both" well inside
    assert torch.allclose(indicator[topo].sum(dim=1), torch.ones(int(topo.sum())))
    # the force-field reference distances of every topology edge
    lookup = {(int(i), int(j)): (lo, hi) for (i, j), lo, hi in
              zip(geometry.pairs, geometry.distance_min, geometry.distance_max)}
    local = merged - ptr[batch[merged[0]]].unsqueeze(0)
    for k in torch.nonzero(topo).reshape(-1)[:50].tolist():
        i, j = sorted(local[:, k].tolist())
        assert reference[k].tolist() == pytest.approx(lookup[(i, j)], abs=1e-5)
    assert torch.equal(reference[~topo], torch.zeros(int((~topo).sum()), 2))
    # the long-range basis reaches beyond the cutoff; the messages of topology edges are not cut off
    long_range = features[:, n_basis + 9:]
    assert long_range.shape[1] == net.trunk.topology_n_basis + 1
    assert float(long_range[far].abs().sum(dim=1).min()) > 0.0
    assert torch.equal(envelope[topo], torch.ones(int(topo.sum())))
    assert bool((envelope[~topo] <= 1.0).all()) and bool((envelope[~topo] >= 0.0).all())


def test_long_range_basis_is_smooth_and_bounded(tmp_path):
    """Gaussians on [0, r_topo] under a cosine envelope at r_topo, plus r / (r + r_topo)."""
    net = _net(tmp_path, topology_cutoff=10.0, topology_n_basis=8)
    r = torch.linspace(0.0, 30.0, 3001)
    basis = net.trunk.topology_rbf(r)
    assert basis.shape == (3001, 9)
    assert torch.equal(basis[r >= 10.0, :-1], torch.zeros(int((r >= 10.0).sum()), 8))
    assert torch.allclose(basis[:, -1], r / (r + 10.0))
    jumps = (basis[1:] - basis[:-1]).abs().max()
    assert float(jumps) < 0.05                                      # no discontinuity
    assert bool((basis[1:, -1] > basis[:-1, -1]).all())             # the magnitude beyond r_topo


def test_messages_reach_bonded_partners_beyond_the_cutoff(tmp_path):
    """With topology edges a bonded atom farther than the cutoff still informs its partner's velocity."""
    x = _frames(1)
    x[0, 8] = x[0, 7] + torch.tensor([0.0, -9.0, 0.0])          # HO 9 Å from O, far from all
    moved = x.clone()
    moved[0, 8] += torch.tensor([1.5, 0.0, 0.0])
    t, lam = torch.full((1,), 0.2), torch.full((1,), 1.0)
    for topology, should_change in ((True, True), (False, False)):
        net = _net(tmp_path, topology_edges=topology, bond_distances=None).eval()
        with torch.no_grad():
            pos = flowmath.center(x)[0]
            assert float((pos[8] - pos[:8]).norm(dim=-1).min()) > TINY['cutoff']
            v_a = net.velocity(net.to_flow(x), t, lam)
            v_b = net.velocity(net.to_flow(moved), t, lam)
        # the center of mass moves with HO, so compare the oxygen relative to C2
        change = float(((v_a[0, 7] - v_a[0, 4]) - (v_b[0, 7] - v_b[0, 4])).abs().max())
        assert (change > 1e-4) == should_change, (topology, change)


def test_velocity_is_smooth_where_a_bonded_pair_crosses_the_cutoff(tmp_path):
    """A topology edge does not switch at the cutoff: the velocity is continuous in the bond length."""
    pair = T.ReferenceGeometry(n_atoms=2, bonds=np.array([[0, 1]]),
                               bond_lengths=np.array([1.2]), pairs=np.array([[0, 1]]),
                               separation=np.array([1]), distance_min=np.array([1.2]),
                               distance_max=np.array([1.2]))
    torch.manual_seed(0)
    with_topology = _randomize(JointPaiNN(2, [6, 8], topology_geometry=pair, **TINY), 1).eval()
    without = _randomize(JointPaiNN(2, [6, 8], **TINY), 1).eval()
    rc, eps = TINY['cutoff'], 1e-4

    def velocity(net, r):
        x = torch.tensor([[[0.0, 0.0, 0.0], [r, 0.3, 0.0]]])
        x[0, 1] *= r / float(x[0, 1].norm())
        with torch.no_grad():
            return net.velocity(net.to_flow(x), torch.full((1,), 0.3), torch.zeros(1))

    jump = float((velocity(with_topology, rc + eps) - velocity(with_topology, rc - eps)).abs().max())
    slope = float((velocity(with_topology, rc - 0.05) - velocity(with_topology, rc - 0.05 - 2 * eps))
                  .abs().max())
    assert jump < 10 * slope + 1e-5
    # the radius graph alone loses the edge there: the test would see that jump
    radius_jump = float((velocity(without, rc + eps) - velocity(without, rc - eps)).abs().max())
    assert radius_jump > 100 * (jump + 1e-7)


def test_symmetries_with_topology_features(tmp_path):
    """Rotations, translations and the topology's automorphisms are symmetries; the velocity is COM free."""
    for options in ({}, {'pe': True}, {'pe': True, 'parity_breaking': True}):
        # weights of scale 0.2: at 0.3 the random two-layer trunk amplifies the extra messages
        # of the topology edges to |v| ~ 1e10, where float32 rounding hides the symmetry
        net = _randomize(_net(tmp_path, randomize=False, **options), 0, scale=0.2).eval()
        x = _frames(3, stretch=2.0)
        rot = _rotation(1)
        t, lam = torch.tensor([0.1, 0.5, 0.9]), torch.tensor([-2.0, 0.0, 3.0])
        with torch.no_grad():
            lam_x = net.committor(x)
            assert torch.allclose(lam_x, net.committor(x @ rot.T), atol=1e-5)
            assert torch.allclose(lam_x, net.committor(x + 3.0), atol=1e-5)
            z = net.to_flow(x)
            v = net.velocity(z, t, lam)
            assert torch.allclose(v @ rot.T, net.velocity(z @ rot.T, t, lam), atol=1e-4)
            assert float(v.mean(dim=1).abs().max()) < 1e-5
            # the methyl hydrogens (1, 2, 3) are one orbit: tokens, encodings and topology agree
            perm = torch.tensor([0, 2, 3, 1, 4, 5, 6, 7, 8])
            assert torch.allclose(lam_x, net.committor(x[:, perm]), atol=1e-5)
            assert torch.allclose(v[:, perm], net.velocity(z[:, perm], t, lam), atol=1e-4)
            assert float(v.abs().max()) > 1e-3


def test_gradients_with_topology_features(tmp_path):
    """forward stays twice differentiable (the gradient penalty) with both features on."""
    net = _net(tmp_path, pe=True)
    rows = _frames(2, stretch=2.0).reshape(2, -1).requires_grad_(True)
    (grad,) = torch.autograd.grad(net(rows)[:, 0].sum(), rows, create_graph=True)
    grad.pow(2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_simple_fit_recovers_bonded_distances_from_prior_samples(tmp_path):
    """Sanity, not a benchmark: from x_t at t = 0.05 a linear fit of the edge features gives the bond lengths.

    With topology edges every bond is an edge of every sample, whatever the
    prior sample, and its features carry the force-field length, so a
    least-squares fit predicts every bond's ``b0`` exactly, and the data's
    bond lengths (which differ from ``b0`` by up to 0.045 Å in this frame) as
    well as ``b0`` does. With the radius graph alone a Gaussian x0 hides many
    bonds beyond the cutoff, and the bonds that are edges carry no length
    (the fit is off by about the spread of the ``b0``).
    """
    geometry, _ = _geometry(tmp_path)
    x1 = torch.as_tensor(ETHANOL_POSITIONS, dtype=torch.float32)
    bonds = [tuple(map(int, b)) for b in geometry.bonds]
    data = {b: float((x1[b[0]] - x1[b[1]]).norm()) for b in bonds}
    b0 = {b: float(length) for b, length in zip(bonds, geometry.bond_lengths)}
    data_vs_b0 = math.sqrt(np.mean([(data[b] - b0[b]) ** 2 for b in bonds]))
    spread = float(np.std(geometry.bond_lengths))
    t = 0.05
    for topology in (True, False):
        net = _net(tmp_path, topology_edges=topology,
                   bond_distances=T.topological_distances(M, geometry.bonds)).eval()
        z0 = net.sample_prior(None, 64, generator=torch.Generator().manual_seed(0))
        x0 = net.from_flow(z0)
        xt = (1.0 - t) * x0 + t * x1
        pos = xt.reshape(-1, 3)
        batch, ptr = net._layout(len(xt), pos.device)
        edge_index, ids = net._topology_edges(net._edges(pos, batch), batch, ptr)
        features, _ = net.trunk.edge_features(pos, batch, ptr, edge_index, ids)
        local = (edge_index - ptr[batch[edge_index[0]]].unsqueeze(0)).t().tolist()
        rows, keys = [], []
        for k, (i, j) in enumerate(local):
            if (i, j) in data:
                rows.append(features[k])
                keys.append((i, j))
        coverage = len(keys) / (len(xt) * len(bonds))
        a = torch.cat([torch.stack(rows).double(), torch.ones(len(rows), 1).double()], dim=1)

        def fit(target):
            y = torch.tensor([target[key] for key in keys], dtype=torch.float64).unsqueeze(1)
            coef = torch.linalg.lstsq(a, y, driver="gelsd").solution   # rank deficient
            return float((a @ coef - y).pow(2).mean().sqrt())

        if not topology:
            assert coverage < 0.9, coverage
            assert fit(b0) > 0.3 * spread, (fit(b0), spread)
            continue
        assert coverage == 1.0
        assert fit(b0) < 1e-3, fit(b0)
        assert fit(data) <= data_vs_b0 + 1e-3, (fit(data), data_vs_b0)


# ---------------------------------------------------------------------------
# random-walk positional encodings
# ---------------------------------------------------------------------------

def test_positional_encoding_enters_the_embedding(tmp_path):
    """The encodings enter s0 through a zero-initialized map: switching them on keeps an untrained model's function."""
    _, mt = _geometry(tmp_path)
    pe = T.random_walk_encoding(M, mt.bonds, steps=16)
    torch.manual_seed(0)
    plain = JointPaiNN(M, ELEMENTS, **TINY)
    torch.manual_seed(0)
    encoded = JointPaiNN(M, ELEMENTS, positional_encoding=pe, **TINY)
    assert encoded.trunk.positional_encoding_steps == 16
    assert torch.allclose(encoded.trunk.positional_encoding, torch.as_tensor(pe, dtype=torch.float32))
    assert not bool(encoded.trunk.pe_to_embed.weight.any())
    shared = {k: v for k, v in plain.state_dict().items()}
    missing = encoded.load_state_dict(shared, strict=False).missing_keys
    assert sorted(missing) == ['trunk.pe_to_embed.weight', 'trunk.positional_encoding']
    x = _frames(2)
    with torch.no_grad():
        assert torch.allclose(plain.committor(x), encoded.committor(x), atol=1e-6)
        encoded.trunk.pe_to_embed.weight.normal_(0.0, 0.5)
        assert float((plain.committor(x) - encoded.committor(x)).abs().max()) > 1e-5
    with pytest.raises(ValueError, match='positional_encoding'):
        JointPaiNN(M, ELEMENTS, positional_encoding=pe[:-1], **TINY)


# ---------------------------------------------------------------------------
# from the params file
# ---------------------------------------------------------------------------

def _descriptors(structure):
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    return GeneratorDescriptors(structure, GeneratorAtoms(n_solute=M))


def test_from_topology_switches(tmp_path):
    """from_topology(topology_edges=True, random_walk_pe=True): the topology's geometry and encodings, never a frame's."""
    pytest.importorskip('networkx')
    top = _write(tmp_path, SOLUTE_TOP)
    gro = _write_ethanol_gro(tmp_path)
    stretched = _write_ethanol_gro(tmp_path, 1.4 * ETHANOL_POSITIONS, 'stretched.gro')
    nets = []
    for structure in (gro, stretched):
        nets.append(JointPaiNN.from_topology(top, _descriptors(structure), topology_edges=True,
                                             random_walk_pe=True, random_walk_steps=8,
                                             topology_cutoff=12.0, **TINY))
    a, b = nets
    for key in ('trunk.topology_index', 'trunk.topology_separation', 'trunk.topology_reference',
                'trunk.topology_pairs', 'trunk.positional_encoding'):
        assert key in a.state_dict()
        assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
    geometry, mt = _geometry(tmp_path)
    assert a.trunk.topology_index.shape == (2, 2 * len(geometry.pairs))
    assert a.trunk.positional_encoding.shape == (M, 8)
    assert torch.allclose(a.trunk.positional_encoding,
                          torch.as_tensor(T.random_walk_encoding(M, mt.bonds, steps=8),
                                          dtype=torch.float32))
    assert a.trunk.topology_cutoff == pytest.approx(12.0)
    default = JointPaiNN.from_topology(top, _descriptors(gro), **TINY)
    assert not any('topology' in k or 'positional' in k for k in default.state_dict())
    # a checkpoint of a model with features loads into a model built the same way
    fresh = JointPaiNN.from_topology(top, _descriptors(gro), topology_edges=True,
                                     random_walk_pe=True, random_walk_steps=8,
                                     topology_cutoff=12.0, **TINY)
    _randomize(a, 3)
    fresh.load_state_dict(a.state_dict())
    x = _frames(2)
    with torch.no_grad():
        assert torch.equal(fresh.committor(x), a.committor(x))


# ---------------------------------------------------------------------------
# the force-field reference geometry of the topology edges
# ---------------------------------------------------------------------------

def _ethanol(tmp_path, n_waters=2):
    """(Topology, ModelTopology) of ethanol + Na+ (the solute, 10 atoms) and `n_waters` waters."""
    top = T.read_topology(_write(tmp_path, ETHANOL_TOP))
    return top, T.model_topology(top, n_solute=10, n_waters=n_waters)


def _law_of_cosines(a, b, theta_deg):
    return math.sqrt(a * a + b * b - 2.0 * a * b * math.cos(math.radians(theta_deg)))


def _dihedral_distance(b1, b2, b3, theta1, theta2, phi):
    """|x_i - x_l| of the chain i-j-k-l built explicitly with dihedral `phi` (all in Å, degrees)."""
    t1, t2, p = (math.radians(v) for v in (theta1, theta2, phi))
    j = np.zeros(3)
    k = np.array([b2, 0.0, 0.0])
    i = j + b1 * np.array([math.cos(t1), math.sin(t1), 0.0])
    l = k + b3 * np.array([-math.cos(t2), math.sin(t2) * math.cos(p), math.sin(t2) * math.sin(p)])
    return float(np.linalg.norm(i - l))


def test_reference_geometry_from_type_lookups(tmp_path):
    """b0 from [ bondtypes ], theta0 from [ angletypes ]; 1-2, 1-3 and 1-4 distances in Å."""
    top, mt = _ethanol(tmp_path, n_waters=0)
    geometry = T.reference_geometry(top, mt)
    assert geometry.n_atoms == 10
    assert np.array_equal(geometry.bonds, mt.bonds)
    lengths = {tuple(b): float(d) for b, d in zip(geometry.bonds, geometry.bond_lengths)}
    assert lengths[(0, 1)] == pytest.approx(1.090)            # CT-HC, nm -> Å
    assert lengths[(0, 4)] == pytest.approx(1.529)            # CT-CT
    assert lengths[(4, 7)] == pytest.approx(1.410)            # OH CT given as "OH CT" (reversed)
    assert lengths[(7, 8)] == pytest.approx(0.945)
    # every pair within three bonds, with its topological separation; the ion has none
    distance = T.topological_distances(10, mt.bonds, max_distance=4)
    expected = {(i, j): int(distance[i, j]) for i in range(10) for j in range(i + 1, 10)
                if 1 <= distance[i, j] <= 3}
    got = {tuple(p): int(s) for p, s in zip(geometry.pairs, geometry.separation)}
    assert got == expected
    assert not any(9 in p for p in got)
    pair = {tuple(p): (lo, hi) for p, lo, hi in zip(geometry.pairs, geometry.distance_min,
                                                     geometry.distance_max)}
    # 1-2: the bond length; 1-3: the law of cosines with theta0
    assert pair[(0, 1)] == pytest.approx((1.090, 1.090))
    d13 = _law_of_cosines(1.529, 1.410, 109.5)                # C1 ... O
    assert pair[(0, 7)] == pytest.approx((d13, d13))
    hh = _law_of_cosines(1.090, 1.090, 107.8)                 # H11 ... H12
    assert pair[(1, 2)] == pytest.approx((hh, hh))
    # 1-4: cis and trans distances of the dihedral, from the 1-2 / 1-3 geometry
    cis = _dihedral_distance(1.090, 1.529, 1.410, 110.7, 109.5, 0.0)       # H11-C1-C2-O
    trans = _dihedral_distance(1.090, 1.529, 1.410, 110.7, 109.5, 180.0)
    assert pair[(1, 7)] == pytest.approx((cis, trans), abs=1e-6)
    cis = _dihedral_distance(1.529, 1.410, 0.945, 109.5, 108.5, 0.0)       # C1-C2-O-HO
    trans = _dihedral_distance(1.529, 1.410, 0.945, 109.5, 108.5, 180.0)
    assert pair[(0, 8)] == pytest.approx((cis, trans), abs=1e-6)
    assert np.all(geometry.distance_min <= geometry.distance_max + 1e-12)


def test_reference_geometry_settles_constraints_macros_and_bond_types(tmp_path):
    """SETTLE doh/dhh, [ constraints ] via [ constrainttypes ], explicit and macro parameters, OPLS bond types."""
    top, mt = _ethanol(tmp_path, n_waters=2)
    geometry = T.reference_geometry(top, mt)
    pair = {tuple(p): (lo, hi) for p, lo, hi in zip(geometry.pairs, geometry.distance_min,
                                                     geometry.distance_max)}
    for oxygen in (10, 13):                                   # the two flow waters
        assert pair[(oxygen, oxygen + 1)] == pytest.approx((0.9572, 0.9572))
        assert pair[(oxygen, oxygen + 2)] == pytest.approx((0.9572, 0.9572))
        assert pair[(oxygen + 1, oxygen + 2)] == pytest.approx((1.5139, 1.5139))   # dhh, no angle
    molecule = np.zeros(mt.n_atoms, dtype=int)
    for k, atoms in enumerate(mt.molecules):
        molecule[atoms] = k
    assert np.all(molecule[geometry.pairs[:, 0]] == molecule[geometry.pairs[:, 1]])

    top = T.read_topology(_write(tmp_path, MIXED_TOP, 'mixed.top'))
    mt = T.model_topology(top, n_solute=6)
    geometry = T.reference_geometry(top, mt)
    lengths = {tuple(b): float(d) for b, d in zip(geometry.bonds, geometry.bond_lengths)}
    assert lengths == pytest.approx({(0, 1): 1.100,           # bond_type column CX HX
                                     (0, 2): 1.080,           # explicit on the line
                                     (0, 3): 1.000,           # #define gb_test, funct 2
                                     (0, 5): 1.100,
                                     (3, 4): 1.010})          # [ constraints ] via constrainttypes
    pair = {tuple(p): (lo, hi) for p, lo, hi in zip(geometry.pairs, geometry.distance_min,
                                                     geometry.distance_max)}
    assert pair[(2, 3)][0] == pytest.approx(_law_of_cosines(1.080, 1.000, 111.0))   # explicit
    assert pair[(0, 4)][0] == pytest.approx(_law_of_cosines(1.000, 1.010, 112.0))   # UB funct 5
    assert pair[(1, 4)] == pytest.approx(
        (_dihedral_distance(1.100, 1.000, 1.010, 110.0, 112.0, 0.0),
         _dihedral_distance(1.100, 1.000, 1.010, 110.0, 112.0, 180.0)), abs=1e-6)


def test_reference_geometry_missing_parameters_raise(tmp_path):
    """A bond without a length or a 1-3 pair without an angle is an error naming the atoms."""
    no_type = MIXED_TOP.replace('CX  HX  1  0.1100  300000.0\n', '')
    top = T.read_topology(_write(tmp_path, no_type, 'a.top'))
    with pytest.raises(ValueError, match=r'MOL1\.H1'):
        T.reference_geometry(top, T.model_topology(top, n_solute=6))
    connection = MIXED_TOP.replace('1 6 1\n', '1 6 5\n')        # funct 5: a connection, no length
    top = T.read_topology(_write(tmp_path, connection, 'b.top'))
    with pytest.raises(ValueError, match='no length'):
        T.reference_geometry(top, T.model_topology(top, n_solute=6))
    no_angle = MIXED_TOP.replace('2 1 6 1\n', '').replace('HX  CX  HX  1  109.0  300.0\n', '')
    top = T.read_topology(_write(tmp_path, no_angle, 'c.top'))
    with pytest.raises(ValueError, match='angle'):
        T.reference_geometry(top, T.model_topology(top, n_solute=6))


def test_reference_geometry_nanma_matches_its_md_frame():
    """NANMA (CHARMM27 from GROMACS' share directory): the force-field geometry is that of an MD frame.

    The frame is used only to check the parameters were read right (units,
    type lookups); the geometry itself never sees coordinates.
    """
    top = T.read_topology(NANMA / 'topol.top')
    if top.unresolved_includes:
        pytest.skip(f'force field not found: {top.unresolved_includes}')
    mt = T.model_topology(top, n_solute=22)
    geometry = T.reference_geometry(top, mt)
    import MDAnalysis as mda
    x = mda.Universe(str(NANMA / 'conf.gro')).atoms.positions.astype(float)
    d = np.linalg.norm(x[geometry.pairs[:, 0]] - x[geometry.pairs[:, 1]], axis=1)
    sep = geometry.separation
    assert (sep == 1).sum() == len(mt.bonds) == 21
    assert np.abs(d - geometry.distance_min)[sep == 1].max() < 0.1
    assert np.abs(d - geometry.distance_min)[sep == 2].max() < 0.2
    inside = (d >= geometry.distance_min - 0.2) & (d <= geometry.distance_max + 0.2)
    assert inside[sep == 3].all()


def test_random_walk_encoding():
    """k-step return probabilities of the bond graph's random walk: known values, orbit invariance."""
    # H2 + an isolated atom: the walk on one bond returns at every even step
    pe = T.random_walk_encoding(3, [(0, 1)], steps=4)
    assert pe.shape == (3, 4)
    assert np.allclose(pe[0], [0.0, 1.0, 0.0, 1.0]) and np.allclose(pe[1], pe[0])
    assert np.allclose(pe[2], 0.0)                              # no bond, no walk
    # a three-atom chain: the center returns with 1 at even steps, an end with 1/2
    pe = T.random_walk_encoding(3, [(0, 1), (1, 2)], steps=4)
    assert np.allclose(pe[1], [0.0, 1.0, 0.0, 1.0])
    assert np.allclose(pe[0], [0.0, 0.5, 0.0, 0.5]) and np.allclose(pe[2], pe[0])
    # a methyl-like star: the three hydrogens (one orbit) get the same encoding
    pe = T.random_walk_encoding(5, [(0, 1), (0, 2), (0, 3), (0, 4)], steps=16)
    assert pe.shape == (5, 16) and np.allclose(pe[1:], pe[1])
    assert np.allclose(pe[0, 1], 1.0) and np.allclose(pe[1, 1], 0.25)
    assert np.all(pe[:, 0] == 0.0)                               # no self loops
    with pytest.raises(ValueError):
        T.random_walk_encoding(3, [(0, 1)], steps=0)
