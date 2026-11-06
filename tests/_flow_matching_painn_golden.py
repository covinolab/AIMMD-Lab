"""The port-equivalence golden of the PaiNN flow-matching network.

The network of the flow branch, ``aimmd.network.joint_painn.JointPaiNN`` at
flowdev ``bb98943``, was built once for each model of :data:`MODELS`, and its
state dict and outputs are stored in
``tests/data/flow_matching_painn_flowdev_bb98943.pt``.
``tests/test_flow_matching_painn.py`` builds the same models with
:class:`aimmd.cgaimmd.flow_matching.painn.JointPaiNN`, loads the stored state
dicts strictly and checks that every output is the same.

The models are tiny (8 channels x 2 layers, torch edge backend) and their
weights are seeded random numbers of scale 0.2, so that no zero-initialized
term hides. They cover every switch the network has, on the 9-atom ethanol of
``tests/_cgaimmd_topology_fixtures.py``:

- ``'chiro_painn'``: element tokens, bond edges, topology edges (range 10 Å,
  8 Gaussians), random-walk encodings (8 steps) and parity breaking (SE(3)),
  with a pairing over the methyl hydrogens;
- ``'from_topology_element'`` and ``'from_topology_residue_atomname'``:
  ``JointPaiNN.from_topology`` on the ethanol topology with topology edges
  and encodings, ``tie='orbits'`` and ``symmetry='topology'``: tokens,
  elements, bond classes, the topology buffers and the pairing all come from
  the topology tooling.

The outputs are the committor, ``forward`` on dense rows and on the batch dict
of ``build_graph_batch``, the three velocities (labelled, null token, masked
rows), a seeded prior sample, the OT pairing and a seeded generation by the
flow-matching ODE (midpoint, 4 steps).

Recording it again is only needed when the models change. From the root of
this checkout, with ``<export>`` an empty folder::

    git archive bb98943 aimmd | tar -x -C <export>
    PYTHONPATH=<export> python tests/_flow_matching_painn_golden.py \\
        tests/data/flow_matching_painn_flowdev_bb98943.pt

``aimmd`` is then imported from ``<export>`` (the script refuses to record
from this checkout). This module imports only the standard library and torch
at module level; the old and the new package are reached through
the namespaces of :func:`old_api` and :func:`new_api`.
"""

import importlib.util
import itertools
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'data' / 'flow_matching_painn_flowdev_bb98943.pt'
"""The recorded state dicts and outputs of the old network."""

TINY = dict(coord_scale=1.7, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6, cutoff=4.0,
            cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,), velocity_hidden=8,
            velocity_radial=6, velocity_min_distance=0.7, edge_backend='torch')
"""The tiny architecture of every model."""

MODELS = ('chiro_painn', 'from_topology_element', 'from_topology_residue_atomname')
"""The recorded models, in recording order (model ``k`` is built and randomized with seed ``k``)."""

N_SOLUTE = 9
"""The ethanol atoms (no ion, no water)."""


def _fixtures():
    """``tests/_cgaimmd_topology_fixtures.py``, loaded from its file (no ``tests`` package needed)."""
    spec = importlib.util.spec_from_file_location(
        '_cgaimmd_topology_fixtures', Path(__file__).with_name('_cgaimmd_topology_fixtures.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_api():
    """The flow branch's names (``aimmd`` from a ``git archive bb98943 aimmd`` export)."""
    from aimmd.flow import descriptors, sampler, topology
    from aimmd.network.graph_batch import build_graph_batch
    from aimmd.network.joint_painn import JointPaiNN

    def generate(net, lam, generator):
        return sampler.generate(net, lam, sampler='midpoint', n_steps=4, generator=generator)

    return SimpleNamespace(
        JointPaiNN=JointPaiNN, topology=topology, build_graph_batch=build_graph_batch,
        generate=generate,
        descriptors=lambda gro: descriptors.FlowDescriptors(
            gro, descriptors.FlowAtoms(n_solute=N_SOLUTE)))


def new_api():
    """This checkout's names; generation through the generator ``'ode'`` of the generation interface."""
    from aimmd.cgaimmd.generation import make_generator
    from aimmd.cgaimmd.flow_matching.graph_batch import build_graph_batch
    from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
    from aimmd.cgaimmd.preparation import descriptors, topology

    def generate(net, lam, generator):
        ode = make_generator({'method': 'ode', 'sampler': 'midpoint', 'n_steps': 4}, net)
        return ode.generate(net, lam, generator=generator)

    return SimpleNamespace(
        JointPaiNN=JointPaiNN, topology=topology, build_graph_batch=build_graph_batch,
        generate=generate,
        descriptors=lambda gro: descriptors.GeneratorDescriptors(
            gro, descriptors.GeneratorAtoms(n_solute=N_SOLUTE)))


def randomize(network, seed, scale=0.2):
    """Seeded random values for every parameter."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in network.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return network


def frames(fixtures, n=4, seed=0):
    """``(n, 9, 3)`` ethanol frames: the fixture geometry plus noise, half of them stretched 2x."""
    g = torch.Generator().manual_seed(seed)
    base = torch.as_tensor(fixtures.ETHANOL_POSITIONS, dtype=torch.float32)
    stretch = torch.tensor([1.0, 2.0] * (n // 2) + [1.0] * (n % 2)).reshape(n, 1, 1)
    return base.unsqueeze(0) * stretch + 0.1 * torch.randn(n, N_SOLUTE, 3, generator=g)


def build(name, api, folder):
    """The model `name` of :data:`MODELS`, built with `api` (files written into `folder`)."""
    fixtures = _fixtures()
    folder = Path(folder)
    top = fixtures.write(folder, fixtures.ETHANOL_TOP.replace('NA  1\nSOL 3', 'SOL 3'))
    if name == 'chiro_painn':
        T = api.topology
        topology = T.read_topology(top)
        mt = T.model_topology(topology, n_solute=N_SOLUTE)
        methyl = (((1, 2, 3), tuple(itertools.permutations(range(3)))),)
        pairing = SimpleNamespace(kwargs=lambda: {'groups': (), 'factors': methyl,
                                                  'molecule_blocks': ()})
        return api.JointPaiNN(
            N_SOLUTE, list(mt.elements), bond_distances=T.topological_distances(
                N_SOLUTE, mt.bonds, max_distance=4),
            topology_geometry=T.reference_geometry(topology, mt),
            positional_encoding=T.random_walk_encoding(N_SOLUTE, mt.bonds, steps=8),
            topology_cutoff=10.0, topology_n_basis=8, parity_breaking=True,
            ot_symmetry=pairing, **TINY)
    embedding = name[len('from_topology_'):]
    gro = fixtures.write_ethanol_gro(folder)
    return api.JointPaiNN.from_topology(
        str(top), api.descriptors(gro), embedding=embedding, tie='orbits', bond_edges=True,
        symmetry='topology', topology_edges=True, random_walk_pe=True, random_walk_steps=8,
        topology_cutoff=10.0, parity_breaking=True, **TINY)


def outputs(net, api):
    """Every output the golden compares (eval mode, no gradients)."""
    fixtures = _fixtures()
    net.eval()
    x = frames(fixtures)
    n = len(x)
    t = torch.tensor([0.0, 0.3, 0.7, 1.0])
    lam = torch.tensor([-12.0, -1.0, 0.5, 6.0])
    mask = torch.tensor([False, True, False, True])
    with torch.no_grad():
        z = net.to_flow(x)
        graph = api.build_graph_batch(x + 20.0, net.cutoff, backend='torch')
        x0 = net.sample_prior(None, n, generator=torch.Generator().manual_seed(1))
        return {
            'x': x, 'z': z, 't': t, 'lam': lam, 'mask': mask,
            'committor': net.committor(x),
            'forward_dense': net(x.reshape(n, -1)),
            'forward_graph': net(graph),
            'velocity': net.velocity(z, t, lam),
            'velocity_null': net.velocity(z, t, None),
            'velocity_masked': net.velocity(z, t, lam, uncond_mask=mask),
            'prior_seed0': net.sample_prior(lam, n, generator=torch.Generator().manual_seed(0)),
            'flow_pair': net.flow_pair(z, x0),
            'generated': api.generate(net, lam[:3], torch.Generator().manual_seed(7)),
        }


def record(target):
    """Build every model with the old code and save the golden to `target`."""
    import aimmd
    if str(ROOT) in str(Path(aimmd.__file__).resolve()):
        raise SystemExit(f'aimmd is imported from this checkout ({aimmd.__file__}); put the '
                         f'flowdev bb98943 export first on PYTHONPATH')
    api = old_api()
    golden = {'meta': {'torch': str(torch.__version__), 'python': sys.version.split()[0],
                       'machine': platform.machine(),
                       'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                       'commit': 'bb98943', 'aimmd': str(aimmd.__file__)},
              'models': {}}
    with tempfile.TemporaryDirectory() as folder:
        for seed, name in enumerate(MODELS):
            torch.manual_seed(seed)
            net = randomize(build(name, api, folder), seed)
            golden['models'][name] = {'state_dict': net.state_dict(),
                                      'outputs': outputs(net, api)}
    torch.save(golden, target)
    print(f'wrote {target}')


if __name__ == '__main__':
    record(sys.argv[1] if len(sys.argv) > 1 else GOLDEN)
