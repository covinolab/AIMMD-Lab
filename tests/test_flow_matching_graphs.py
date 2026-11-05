"""Bond-edge features of ``aimmd.cgaimmd.flow_matching.graphs`` (plain torch).

Ported from the flow branch's ``tests/test_flow_graphs.py`` (``bb98943``).
"""

import torch

from aimmd.cgaimmd.flow_matching.graphs import bond_edge_features


def _edge_set(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


def test_bond_edge_features():
    """One-hot of the topological distance class, looked up by molecule-local index."""
    # a 4-atom chain 0-1-2-3: distances 1, 2, 3 bonds; class = clip(d, 1, 4) - 1
    distance = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    bond_class = (distance.clamp(1, 4) - 1).to(torch.uint8)
    # two molecules of 4 atoms, edges given in global indices
    edge_index = torch.tensor([[0, 3, 5, 4, 7],
                               [1, 0, 7, 7, 4]])
    batch = torch.arange(2).repeat_interleave(4)
    ptr = torch.tensor([0, 4, 8])
    features = bond_edge_features(edge_index, batch, ptr, bond_class)
    assert features.shape == (5, 4) and features.dtype == torch.float32
    assert features.sum(dim=1).tolist() == [1.0] * 5
    assert features.argmax(dim=1).tolist() == [0, 2, 1, 2, 2]
    # disconnected pairs carry the largest class (> 3 bonds)
    far = torch.full((4, 4), 3, dtype=torch.uint8)
    assert bond_edge_features(edge_index, batch, ptr, far).argmax(dim=1).tolist() == [3] * 5


# ---------------------------------------------------------------------------
# the default route: one batch dict, no cache, no file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the SQLite graph cache: an explicit switch
# ---------------------------------------------------------------------------
