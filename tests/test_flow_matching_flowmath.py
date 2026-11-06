"""Flow mathematics of ``aimmd.cgaimmd.flow_matching.flowmath``.

Centering and the center-of-mass-free prior, proper Kabsch alignment,
permutation pairing (Hungarian groups, exact factor enumeration, whole
molecules), the probability path, flow-time schedules and Haar rotations.
Ported from the flow branch's ``tests/test_flow_math.py`` (``bb98943``).
"""

import itertools
import math

import pytest
import torch

from aimmd.cgaimmd.flow_matching import flowmath


def _random_rotation(generator, dtype=torch.float64):
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_center_and_com_free():
    """center removes the (weighted) centroid; the Gaussian prior lives in the COM-free subspace."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 7, 3, generator=g, dtype=torch.float64) + 3.0
    centered = flowmath.center(x)
    assert torch.allclose(centered.mean(dim=-2), torch.zeros(5, 3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(centered, x - x.mean(dim=-2, keepdim=True))
    w = torch.arange(1.0, 8.0, dtype=torch.float64)
    weighted = flowmath.center(x, weights=w)
    com = (weighted * w.reshape(1, -1, 1)).sum(dim=-2) / w.sum()
    assert torch.allclose(com, torch.zeros(5, 3, dtype=torch.float64), atol=1e-12)
    v = torch.randn(4, 6, 3, generator=g)
    assert torch.allclose(flowmath.project_com_free(v).sum(dim=-2), torch.zeros(4, 3), atol=1e-5)

    z = flowmath.gaussian_prior((2000, 6, 3), generator=torch.Generator().manual_seed(1))
    assert z.shape == (2000, 6, 3) and z.dtype == torch.float32
    assert float(z.mean(dim=-2).abs().max()) < 1e-5
    # the projection removes 3 of 3 M degrees of freedom: per-coordinate variance (M - 1) / M
    assert abs(float(z.var()) - 5.0 / 6.0) < 0.03
    again = flowmath.gaussian_prior((2000, 6, 3), generator=torch.Generator().manual_seed(1))
    assert torch.equal(z, again)                       # drawn only from the generator


def test_kabsch_proper_rotation():
    """Kabsch recovers a rotation exactly and never aligns by a reflection."""
    g = torch.Generator().manual_seed(2)
    target = flowmath.center(torch.randn(4, 9, 3, generator=g, dtype=torch.float64))
    rotations = torch.stack([_random_rotation(g) for _ in range(4)])
    mobile = target @ rotations.transpose(-1, -2)       # rotated copies
    aligned = flowmath.kabsch_align(mobile, target)
    assert torch.allclose(aligned, target, atol=1e-10)

    # a mirror image cannot be aligned onto the original by a proper rotation
    mirror = target * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    aligned = flowmath.kabsch_align(mirror, target)
    assert float((aligned - target).pow(2).sum(dim=(-1, -2)).min()) > 1e-3
    # the map applied is a proper rotation: distances kept, handedness kept
    gram = aligned @ aligned.transpose(-1, -2)
    assert torch.allclose(gram, mirror @ mirror.transpose(-1, -2), atol=1e-10)
    triple = torch.det(aligned[:, 1:4] - aligned[:, :1])
    assert torch.allclose(torch.sign(triple), torch.sign(torch.det(mirror[:, 1:4] - mirror[:, :1])))


def test_factor_align_equals_hungarian():
    """Exact enumeration of symmetric-group factors gives the Hungarian assignment of each group."""
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn(16, 10, 3, generator=g)
    groups = [[1, 2, 3], [5, 6], [7, 8, 9]]
    factors = [(group, list(itertools.permutations(range(len(group))))) for group in groups]
    # scramble x0 within the groups and add noise
    x1 = x0.clone()
    for n in range(len(x1)):
        for group in groups:
            perm = torch.randperm(len(group), generator=g)
            x1[n, group] = x0[n, [group[int(i)] for i in perm]]
    x1 = x1 + 0.05 * torch.randn(x1.shape, generator=g)
    hungarian = flowmath.permutation_align(x1, x0, groups)
    enumerated = flowmath.factor_align(x1, x0, flowmath.PermutationFactors(factors))
    plain_tuples = flowmath.factor_align(x1, x0, factors)
    assert torch.equal(hungarian, enumerated)
    assert torch.equal(enumerated, plain_tuples)
    assert float((enumerated - x0).abs().max()) < 0.5   # unscrambled
    pf = flowmath.PermutationFactors(factors)
    assert pf.order == 6 * 2 * 6 and pf.n_candidates == 6 + 2 + 6
    with pytest.raises(ValueError):
        flowmath.PermutationFactors([((0, 1), [(1, 0)])])          # no identity
    with pytest.raises(ValueError):
        flowmath.PermutationFactors([((0, 1), [(0, 1)]), ((1, 2), [(0, 1)])])   # overlapping


def test_molecule_blocks_pairing():
    """Whole molecules are permuted as blocks (by their anchor atom), hydrogens follow their oxygen."""
    g = torch.Generator().manual_seed(4)
    n_waters = 4
    solute = torch.randn(3, 3, generator=g)
    waters = torch.stack([torch.randn(3, generator=g) * 4 + torch.tensor([0.0, 0.1, 0.2]) * k
                          for k in range(n_waters)])
    offsets = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0]])
    x0 = torch.cat([solute] + [w + offsets for w in waters]).unsqueeze(0)       # (1, 15, 3)
    blocks = [[3 + 3 * k, 4 + 3 * k, 5 + 3 * k] for k in range(n_waters)]
    order = [2, 0, 3, 1]
    x1 = torch.cat([solute] + [waters[k] + offsets for k in order]).unsqueeze(0)
    paired = flowmath.ot_pair(x1, x0, molecule_blocks=blocks, rotate=False)
    assert torch.allclose(paired, flowmath.center(x0), atol=1e-5)
    # a rotated copy in the right order: Kabsch undoes the rotation, the blocks stay put
    rot = _random_rotation(g, dtype=torch.float32)
    paired = flowmath.ot_pair(flowmath.center(x0) @ rot.T, x0, molecule_blocks=blocks)
    assert torch.allclose(paired, flowmath.center(x0), atol=1e-4)
    # alternating never increases the pairing cost over the plain Kabsch alignment
    scrambled = flowmath.center(x1) @ rot.T
    plain = flowmath.kabsch_align(flowmath.center(scrambled), flowmath.center(x0))
    paired = flowmath.ot_pair(scrambled, x0, molecule_blocks=blocks, n_rounds=3)
    cost = lambda y: float((y - flowmath.center(x0)).pow(2).sum())   # noqa: E731
    assert cost(paired) <= cost(plain) + 1e-4


def test_ot_pair_alternates_rotation_and_permutation():
    """Every round rotates, then permutes: a second Kabsch after the permutation lowers the cost."""
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(1, 8, 3, generator=g, dtype=torch.float64)
    x1 = torch.randn(1, 8, 3, generator=g, dtype=torch.float64)
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    a, b = flowmath.center(x1), flowmath.center(x0)
    first = flowmath.permutation_align(flowmath.kabsch_align(a, b), b, groups)
    second = flowmath.permutation_align(flowmath.kabsch_align(first, b), b, groups)
    one = flowmath.ot_pair(x1, x0, groups=groups, n_rounds=1)
    two = flowmath.ot_pair(x1, x0, groups=groups, n_rounds=2)
    assert torch.allclose(one, first) and torch.allclose(two, second)
    cost = lambda y: float((y - b).pow(2).sum())           # noqa: E731
    assert cost(two) < 0.9 * cost(one)                     # measured: 10.8 against 15.0
    assert cost(flowmath.permutation_align(first, b, groups)) == pytest.approx(cost(one))


def test_molecule_blocks_assigned_by_the_first_atom():
    """Whole molecules are assigned by their first atom (the oxygen of a water), not another one.

    Water 1's oxygen sits at water A's and its first hydrogen near water B's,
    and the other way round for water 2: by the oxygens nothing moves, by the
    first hydrogens the two waters would be swapped.
    """
    x0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]], dtype=torch.float64)
    x1 = torch.tensor([[0.5, 0.0, 0.0], [10.5, 0.0, 0.0], [0.5, 1.0, 0.0],
                       [9.5, 0.0, 0.0], [0.5, 0.2, 0.0], [9.5, 1.0, 0.0]], dtype=torch.float64)
    blocks = [[0, 1, 2], [3, 4, 5]]
    paired = flowmath.ot_pair(x1[None], x0[None], molecule_blocks=blocks, rotate=False)
    assert torch.allclose(paired, flowmath.center(x1[None]))                  # kept by the O
    by_hydrogen = flowmath._molecule_permutation_align(
        flowmath.center(x1[None]), flowmath.center(x0[None]), blocks, anchor=1)
    assert torch.allclose(by_hydrogen[0, :3], flowmath.center(x1[None])[0, 3:])  # swapped


def test_ot_pair_rotation_only_and_centering():
    """Without permutations ot_pair is one proper Kabsch alignment of the centered samples."""
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(3, 6, 3, generator=g, dtype=torch.float64) + 2.0
    x1 = torch.randn(3, 6, 3, generator=g, dtype=torch.float64) - 1.0
    paired = flowmath.ot_pair(x1, x0)
    expected = flowmath.kabsch_align(flowmath.center(x1), flowmath.center(x0))
    assert torch.allclose(paired, expected)
    assert torch.allclose(paired.mean(dim=-2), torch.zeros(3, 3, dtype=torch.float64), atol=1e-12)
    assert torch.equal(flowmath.ot_pair(x1, x0, rotate=False), flowmath.center(x1))


def test_interpolate():
    g = torch.Generator().manual_seed(6)
    x0, x1 = torch.randn(2, 4, 5, 3, generator=g)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    xt, target = flowmath.interpolate(x0, x1, t)
    assert torch.allclose(xt[0], x0[0]) and torch.allclose(xt[3], x1[3])
    assert torch.allclose(xt[1], 0.75 * x0[1] + 0.25 * x1[1])
    assert torch.equal(target, x1 - x0)


def test_sample_time_schedules():
    """uniform, power:k (Beta(k, 1)) and logitnormal:m,s draw from the generator only."""
    assert flowmath.parse_time_schedule('uniform') == ('uniform', ())
    assert flowmath.parse_time_schedule('power:2') == ('power', (2.0,))
    assert flowmath.parse_time_schedule('logitnormal:0.5,1') == ('logitnormal', (0.5, 1.0))
    for bad in ('uniform:1', 'power:0', 'power:x', 'logitnormal:1', 'logitnormal:0,-1', 'beta'):
        with pytest.raises(ValueError):
            flowmath.parse_time_schedule(bad)
    n = 20000
    t = flowmath.sample_time(n, 'uniform', generator=torch.Generator().manual_seed(7))
    assert t.shape == (n,) and float(t.min()) >= 0 and float(t.max()) <= 1
    assert abs(float(t.mean()) - 0.5) < 0.01
    t = flowmath.sample_time(n, 'power:2', generator=torch.Generator().manual_seed(7))
    assert abs(float(t.mean()) - 2.0 / 3.0) < 0.01                 # Beta(2, 1)
    assert abs(float((t > 0.8).float().mean()) - 0.36) < 0.015
    t = flowmath.sample_time(n, 'logitnormal:1,0.5', generator=torch.Generator().manual_seed(7))
    assert abs(float(torch.logit(t).mean()) - 1.0) < 0.02
    assert abs(float(torch.logit(t).std()) - 0.5) < 0.02
    state = torch.get_rng_state()
    a = flowmath.sample_time(10, 'power:3', generator=torch.Generator().manual_seed(8))
    b = flowmath.sample_time(10, 'power:3', generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and torch.equal(torch.get_rng_state(), state)


def test_random_rotations_haar_det_one():
    """Quaternion rotations are orthogonal with det +1 and uniform (Haar) on SO(3)."""
    rot = flowmath.random_rotations(20000, dtype=torch.float64,
                                    generator=torch.Generator().manual_seed(9))
    eye = torch.eye(3, dtype=torch.float64).expand(len(rot), 3, 3)
    assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-12)
    assert torch.allclose(torch.det(rot), torch.ones(len(rot), dtype=torch.float64), atol=1e-12)
    # Haar: a fixed vector is rotated uniformly onto the sphere; the trace has density
    # (1 - cos w) / pi of the rotation angle w, i.e. E[trace] = 0 and E[trace^2] = 1
    v = rot @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert float(v.mean(dim=0).abs().max()) < 0.02
    assert abs(float((v[:, 2] ** 2).mean()) - 1.0 / 3.0) < 0.01
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert abs(float(trace.mean())) < 0.03 and abs(float((trace ** 2).mean()) - 1.0) < 0.05
    # drawn only from the generator (torch's CPU randn orders its draws differently for
    # small and large n, so a prefix of a longer draw is not compared)
    state = torch.get_rng_state()
    once = flowmath.random_rotations(5, dtype=torch.float64,
                                     generator=torch.Generator().manual_seed(9))
    again = flowmath.random_rotations(5, dtype=torch.float64,
                                      generator=torch.Generator().manual_seed(9))
    assert torch.equal(once, again) and torch.equal(torch.get_rng_state(), state)


def test_no_heavy_imports_at_module_level():
    """scipy.optimize is imported only when the Hungarian pairing runs."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(flowmath))
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in top for alias in node.names]
    modules = [getattr(node, 'module', None) or '' for node in top]
    assert not any('scipy' in name for name in names + modules)


def test_uniform_time_is_torch_rand():
    """The 'uniform' schedule is ``torch.rand`` of the generator (the prototype's draw order)."""
    assert math.isclose(flowmath.sample_time(1, generator=torch.Generator().manual_seed(0)).item(),
                        torch.rand(1, generator=torch.Generator().manual_seed(0)).item())
