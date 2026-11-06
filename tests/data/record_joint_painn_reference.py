"""Record the outputs of ``JointPaiNN`` before the universal-prior features (run by hand).

Kept from the flow branch (``dev/committor-flow``, ``bb98943``), unchanged
but for this paragraph, as the record of how ``joint_painn_reference.pt`` was
made. It imports the flow branch's names (``aimmd.flow.prior.MixturePrior``,
``aimmd.network.joint_painn``) and runs only on that branch: the mixture prior
is not part of C-GAIMMD. On this branch
``tests/test_flow_matching_painn.py::test_flow_branch_checkpoints_load_unchanged``
builds the same two models with ``aimmd.cgaimmd.flow_matching.painn.JointPaiNN``,
loads the stored state dicts strictly (without the mixture prior's
``prior.*`` buffers) and checks that every output that does not depend on the
prior is the same.

On the flow branch,
``tests/test_flow_topology_features.py::test_features_off_reproduce_recorded_outputs``
loads ``joint_painn_reference.pt`` and checks that a model built by the
current code with the topology features off (the default) gives exactly
these outputs: the features must not change the network's function unless
they are switched on.

The file was generated on ``dev/committor-flow`` at ``722b6df`` (before the
features existed) with::

    PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES="" python tests/data/record_joint_painn_reference.py

Two tiny models (8 channels x 2 layers, torch edge backend, seeded random
weights so that no zero-initialized term hides): element tokens with the
Gaussian prior, and residue-style tokens with bond edges, parity breaking and
a mixture prior.
"""

import platform
import sys
from pathlib import Path

import torch

from aimmd.flow.prior import MixturePrior
from aimmd.network.joint_painn import JointPaiNN


TINY = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, edge_backend='torch')
TOKENS = [6, 1, 1, 8, 6, 1, 7]
RESIDUE_TOKENS = [1001, 1002, 1002, 1003, 1004, 1005, 1006]
BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6)]
M = len(TOKENS)


def topological_distance(n, bonds, cap=4):
    d = torch.full((n, n), cap, dtype=torch.long)
    d.fill_diagonal_(0)
    for i, j in bonds:
        d[i, j] = d[j, i] = 1
    for _ in range(cap):
        for k in range(n):
            d = torch.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
    return d.clamp_max(cap)


def molecule(n, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.0, 0.0, 1.4],
                         [-1.5, -0.2, -0.3], [-1.9, -1.1, 0.2], [1.2, 1.3, 1.1]])
    return base.unsqueeze(0) + 0.15 * torch.randn(n, M, 3, generator=g)


def randomize(network, seed, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return network


def build(name):
    if name == 'element_gaussian':
        net = JointPaiNN(M, TOKENS, parity_breaking=False, **TINY)
    elif name == 'residue_bondedges_parity_mixture':
        prior = MixturePrior(molecule(3, 11), molecule(2, 12), 0.4)
        net = JointPaiNN(M, RESIDUE_TOKENS, atomic_numbers=TOKENS, parity_breaking=True,
                         bond_distances=topological_distance(M, BONDS).numpy(), prior=prior,
                         **TINY)
    else:
        raise ValueError(name)
    return net


def outputs(net):
    net.eval()
    x = molecule(5, 3)
    z = net.to_flow(x)
    t = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    lam = torch.tensor([-12.0, -1.0, 0.0, 2.0, 7.0])
    mask = torch.tensor([False, True, False, True, False])
    with torch.no_grad():
        out = {'x': x, 'z': z, 't': t, 'lam': lam, 'mask': mask,
               'committor': net.committor(x),
               'forward_dense': net(x.reshape(5, -1)),
               'velocity': net.velocity(z, t, lam),
               'velocity_null': net.velocity(z, t, None),
               'velocity_masked': net.velocity(z, t, lam, uncond_mask=mask),
               'prior_seed0': net.sample_prior(lam, 5, generator=torch.Generator().manual_seed(0)),
               'flow_pair': net.flow_pair(z, net.sample_prior(
                   None, 5, generator=torch.Generator().manual_seed(1)))}
    return out


def main():
    record = {'meta': {'torch': str(torch.__version__), 'python': sys.version.split()[0],
                       'machine': platform.machine(),
                       'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                       'commit': '722b6df'},
              'models': {}}
    for seed, name in enumerate(('element_gaussian', 'residue_bondedges_parity_mixture')):
        torch.manual_seed(seed)
        net = randomize(build(name), seed)
        record['models'][name] = {'state_dict': net.state_dict(), 'outputs': outputs(net)}
    target = Path(__file__).with_name('joint_painn_reference.pt')
    torch.save(record, target)
    print(f'wrote {target}')


if __name__ == '__main__':
    main()
