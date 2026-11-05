"""
aimmd.cgaimmd.flow_matching.graphs
==================================

The bond-edge and topology-edge features of the PaiNN network.

:func:`bond_edge_features` is the bond-edge wrapper: a 4-way one-hot of the
topological distance of every edge (1, 2, 3 or more than 3 bonds), which the
PaiNN network appends to the radial features of every graph.

:func:`topology_edge_template` and :func:`merge_topology_edges` give the
PaiNN network its optional *topology edges*: every pair of atoms within three
bonds is an edge at every distance, merged with the radius graph of every
input, so that the bonded structure is visible to the network even when a
prior sample puts bonded atoms far apart.
"""

import numpy as np
import torch


N_BOND_CLASSES = 4
"""The number of topological distance classes of a bond edge (1, 2, 3 and more than 3 bonds, disconnected included)."""


TOPOLOGY_INDICATORS = ('radius_only', 'topology_only', 'both')
"""The channels of the topology-edge indicator, in order (a radius-graph edge more than three bonds apart, a topology edge beyond the cutoff, one within it)."""


def bond_edge_features(edge_index, batch, ptr, bond_class):
    """One-hot topological distance class of every edge.

    Parameters
    ----------
    edge_index : torch.Tensor
        ``(2, E)`` long ``[source, target]`` in batch (global) atom indices.
    batch : torch.Tensor
        ``(N,)`` long sample index of every atom.
    ptr : torch.Tensor
        ``(B + 1,)`` long offset of every sample's first atom.
    bond_class : torch.Tensor
        ``(M, M)`` uint8 class matrix, ``clip(topological distance, 1, 4) - 1``.

    Returns
    -------
    torch.Tensor
        ``(E, 4)`` float32 one-hot of the classes (1, 2, 3, > 3 bonds).

    Notes
    -----
    Each edge is looked up by its molecule-local indices
    ``edge_index - ptr[batch[edge_index]]``, so every sample must hold the
    ``M`` atoms of `bond_class` in flow order. Disconnected pairs carry the
    last class.
    """
    bond_class = bond_class.to(edge_index.device)
    start = ptr.to(edge_index.device)[batch.to(edge_index.device)[edge_index[0]]]
    local = edge_index - start.unsqueeze(0)
    classes = bond_class[local[0], local[1]].long()
    return torch.nn.functional.one_hot(classes, N_BOND_CLASSES).to(torch.float32)


def topology_edge_template(n_atoms, pairs, separation, distance_min, distance_max):
    """The fixed topology edges of one sample, both directions.

    Parameters
    ----------
    n_atoms : int
        Number ``M`` of flow atoms.
    pairs : array_like
        ``(P, 2)`` atom pairs within three bonds (``aimmd.cgaimmd.preparation.topology.ReferenceGeometry.pairs``).
    separation : array_like
        ``(P,)`` their topological distances, 1, 2 or 3.
    distance_min, distance_max : array_like
        ``(P,)`` their force-field reference distances, Å.

    Returns
    -------
    dict
        ``index`` ``(2, 2 P)`` long ``[source, target]`` (the pairs, then the reversed pairs), ``separation`` ``(2 P,)`` uint8, ``reference`` ``(2 P, 2)`` float32 ``(distance_min, distance_max)``, ``pair_mask`` ``(M, M)`` bool.

    Raises
    ------
    ValueError
        For pairs out of range or on one atom, a separation outside 1-3, or arrays of different lengths.
    """
    n_atoms = int(n_atoms)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    separation = np.asarray(separation, dtype=np.int64).reshape(-1)
    reference = np.stack([np.asarray(distance_min, dtype=np.float64).reshape(-1),
                          np.asarray(distance_max, dtype=np.float64).reshape(-1)], axis=1) \
        if len(pairs) else np.zeros((0, 2))
    if not len(pairs) == len(separation) == len(reference):
        raise ValueError('pairs, separation, distance_min and distance_max must have the same '
                         'length')
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= n_atoms
                       or bool((pairs[:, 0] == pairs[:, 1]).any())):
        raise ValueError(f'topology pairs must be distinct atoms in [0, {n_atoms})')
    if len(separation) and (separation.min() < 1 or separation.max() > 3):
        raise ValueError('topology pairs must be 1, 2 or 3 bonds apart')
    index = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
    mask = np.zeros((n_atoms, n_atoms), dtype=bool)
    mask[index[0], index[1]] = True
    return {'index': torch.as_tensor(index, dtype=torch.long),
            'separation': torch.as_tensor(np.concatenate([separation, separation]),
                                          dtype=torch.uint8),
            'reference': torch.as_tensor(np.concatenate([reference, reference]),
                                         dtype=torch.float32),
            'pair_mask': torch.as_tensor(mask)}


def merge_topology_edges(edge_index, batch, ptr, template_index, pair_mask):
    """The radius graph merged with the topology edges of every sample, each edge once.

    Parameters
    ----------
    edge_index : torch.Tensor
        ``(2, E)`` long radius-graph edges ``[source, target]`` in batch (global) atom indices.
    batch : torch.Tensor
        ``(N,)`` long sample index of every atom.
    ptr : torch.Tensor
        ``(B + 1,)`` long offset of every sample's first atom (every sample holds the ``M`` flow atoms in flow order).
    template_index : torch.Tensor
        ``(2, T)`` long topology edges of one sample (molecule-local), from :func:`topology_edge_template`.
    pair_mask : torch.Tensor
        ``(M, M)`` bool, True for the topology pairs.

    Returns
    -------
    tuple of torch.Tensor
        ``(edge_index (2, E'), topology_ids (E',))``: the radius-graph edges that are not topology pairs, then the ``T`` topology edges of every sample in sample order; ``topology_ids`` is the template column of a topology edge and -1 for a radius-only edge.

    Notes
    -----
    A radius-graph edge between a topology pair is dropped and comes back
    as the topology edge, so the result has no duplicates when the radius
    graph has none. Pure torch, on the device of `edge_index`.
    """
    device = edge_index.device
    batch, ptr = batch.to(device), ptr.to(device)
    template_index, pair_mask = template_index.to(device), pair_mask.to(device)
    start = ptr[batch[edge_index[0]]]
    local = edge_index - start.unsqueeze(0)
    radius_only = edge_index[:, ~pair_mask[local[0], local[1]]]
    n_samples, n_template = int(ptr.numel()) - 1, int(template_index.shape[1])
    topology = (template_index.unsqueeze(1) + ptr[:-1].reshape(1, -1, 1)).reshape(2, -1)
    ids = torch.cat([torch.full((radius_only.shape[1],), -1, dtype=torch.long, device=device),
                     torch.arange(n_template, device=device).repeat(n_samples)])
    return torch.cat([radius_only, topology], dim=1), ids
