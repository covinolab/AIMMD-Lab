"""
aimmd.cgaimmd.flow_matching.graph_batch
=======================================

Radius graphs built on the fly, for configurations that are not on disk.

Generator states -- the interpolants ``x_t`` of flow matching and the ODE, and
freshly generated candidates -- change at every step and are never cached;
their graphs are built here, on the device of the positions, without per-frame
Python work. MD frames reach a graph network through
``params.descriptor_transform`` (:class:`aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`),
whose default route builds their graphs here as well, so generated and MD
configurations get the same graphs. The ``'pyg'`` backend is the
``radius_graph`` call that ``aimmd.network.graph_utils.get_graphs_pyg`` makes,
so these graphs also agree with those of AIMMD's graph pipeline and its cache;
the ``'torch'`` backend needs nothing beyond torch (a per-sample distance
matrix).

``torch_geometric`` and ``torch_cluster`` are imported only inside the
functions that need them.
"""

import importlib.util
import warnings
from numbers import Integral

import torch

from .. import _once


BACKENDS = ('auto', 'pyg', 'torch')
"""The edge backends, ``'pyg'`` (``torch_geometric.nn.radius_graph``), ``'torch'`` (pure torch) and ``'auto'``."""


def _resolve_backend(backend):
    """``'auto'`` -> ``'pyg'`` if ``torch_cluster`` can be imported, else ``'torch'``."""
    if backend not in BACKENDS:
        raise ValueError(f'unknown edge backend {backend!r}; expected one of {BACKENDS}')
    if backend == 'auto':
        return 'pyg' if importlib.util.find_spec('torch_cluster') is not None else 'torch'
    return backend


def _check_cap(max_num_neighbors):
    if isinstance(max_num_neighbors, bool) or not isinstance(max_num_neighbors, Integral) \
            or max_num_neighbors < 1:
        raise ValueError(f'max_num_neighbors must be an int >= 1, got {max_num_neighbors!r}')
    return int(max_num_neighbors)


def _pyg_edges(pos, batch, cutoff, max_num_neighbors):
    from torch_geometric.nn import radius_graph
    return radius_graph(x=pos, r=float(cutoff), batch=batch, loop=False,
                        max_num_neighbors=max_num_neighbors)


def _cap_rows(mask, d, max_num_neighbors):
    """Keep, in every row of `mask` (a target atom), the `max_num_neighbors` nearest sources."""
    big = torch.where(mask, d, torch.full_like(d, float('inf')))
    keep = big.argsort(dim=-1)[..., :max_num_neighbors]
    capped = torch.zeros_like(mask)
    capped.scatter_(-1, keep, True)
    return mask & capped


def _torch_edges(pos, batch, cutoff, max_num_neighbors):
    n = pos.shape[0]
    device = pos.device
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    counts = torch.bincount(batch)
    contiguous = bool((batch[1:] >= batch[:-1]).all()) if n > 1 else True
    if contiguous and bool((counts == counts[0]).all()):
        # every sample has k atoms, in order: one (B, k, k) distance tensor instead of (Bk, Bk)
        b, k = int(counts.numel()), int(counts[0])
        grid = pos.reshape(b, k, 3)
        d = torch.cdist(grid, grid, compute_mode='donot_use_mm_for_euclid_dist')
        eye = torch.eye(k, dtype=torch.bool, device=device).unsqueeze(0)
        mask = (d < cutoff) & ~eye
        if max_num_neighbors < k - 1:
            mask = _cap_rows(mask, d, max_num_neighbors)
        sample, target, source = mask.nonzero(as_tuple=True)
        offset = sample * k
        return torch.stack([source + offset, target + offset], dim=0)
    # ragged: samples of different sizes or in any order
    d = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0),
                    compute_mode='donot_use_mm_for_euclid_dist').squeeze(0)
    same = batch.unsqueeze(0) == batch.unsqueeze(1)
    mask = (d < cutoff) & same
    mask.fill_diagonal_(False)
    if max_num_neighbors < n - 1:
        mask = _cap_rows(mask, d, max_num_neighbors)
    target, source = mask.nonzero(as_tuple=True)
    return torch.stack([source, target], dim=0)


def radius_edges(pos, batch, cutoff, *, max_num_neighbors=128, backend='auto'):
    """All pairs of atoms of one sample closer than `cutoff`, as an edge index.

    Parameters
    ----------
    pos : torch.Tensor
        ``(N, 3)`` positions (Å), all samples concatenated.
    batch : torch.Tensor
        ``(N,)`` long sample index of every atom.
    cutoff : float
        Edge cutoff, Å.
    max_num_neighbors : int, optional
        At most this many sources per target atom (the cap ``graph_utils`` applies).
    backend : {'auto', 'pyg', 'torch'}, optional
        ``'pyg'``: ``torch_geometric.nn.radius_graph``; ``'torch'``: per-sample distance matrices; ``'auto'``: ``'pyg'`` if ``torch_cluster`` is importable.

    Returns
    -------
    torch.Tensor
        ``(2, E)`` long ``[source, target]`` on the device of `pos`.

    Raises
    ------
    ValueError
        For an unknown backend or a bad `max_num_neighbors`.

    Notes
    -----
    No self loops, no periodic images, pairs within one sample only. The
    ``'pyg'`` backend is ``radius_graph(x=pos, r=cutoff, batch=batch,
    loop=False, max_num_neighbors=max_num_neighbors)``, the call
    ``graph_utils.get_graphs_pyg`` makes per frame; the ``'torch'`` backend is
    the prototype's ``build_edges`` (a ``(B, k, k)`` distance tensor when every
    sample has ``k`` atoms in order, else one masked matrix). Both give the
    same edge set as long as no atom reaches the cap; at the cap PyG keeps
    arbitrary neighbours and the torch backend the nearest. A
    ``RuntimeWarning`` is issued once per process
    (:func:`aimmd.cgaimmd._once.once`) when an atom reaches the cap, since its
    neighbour list may then be truncated.

    The edges are built from detached positions; gradients reach `pos` only
    through the caller's use of ``pos[edge_index]``.
    """
    backend = _resolve_backend(backend)
    max_num_neighbors = _check_cap(max_num_neighbors)
    with torch.no_grad():
        pos = pos.detach()
        batch = batch.to(device=pos.device, dtype=torch.long)
        if backend == 'pyg':
            edges = _pyg_edges(pos, batch, cutoff, max_num_neighbors)
        else:
            edges = _torch_edges(pos, batch, float(cutoff), max_num_neighbors)
        if edges.shape[1]:
            degree = torch.bincount(edges[1], minlength=pos.shape[0])
            if int(degree.max()) >= max_num_neighbors \
                    and _once.once(('radius_edges max_num_neighbors', max_num_neighbors)):
                warnings.warn(f'radius_edges: an atom has reached max_num_neighbors='
                              f'{max_num_neighbors} neighbours within {float(cutoff)} A; its '
                              f'neighbour list may be truncated', RuntimeWarning, stacklevel=2)
    return edges


def build_graph_batch(positions, cutoff, *, node_attrs=None, max_num_neighbors=128,
                      backend='auto'):
    """Graph batch of ``B`` configurations, as the dict of a PyG ``Batch``.

    Parameters
    ----------
    positions : torch.Tensor
        ``(B, M, 3)`` positions, Å.
    cutoff : float
        Edge cutoff, Å.
    node_attrs : torch.Tensor, optional
        ``(M, T)`` per-atom attributes (repeated for every sample) or ``(B * M, T)``; None: ``(B * M, 0)``.
    max_num_neighbors, backend : optional
        See :func:`radius_edges`.

    Returns
    -------
    dict
        ``positions`` ``(B M, 3)``, ``edge_index`` ``(2, E)``, ``batch`` ``(B M,)``, ``ptr`` ``(B + 1,)``, ``node_attrs``, ``shifts`` ``(E, 3)`` zeros.

    Notes
    -----
    The keys and layout are those of ``Batch.from_data_list(data).to_dict()``
    for ``Data`` built like ``graph_utils.get_graphs_pyg`` does, all on the
    device of `positions`, without per-frame Python work. The positions keep
    their autograd history.
    """
    b, m = int(positions.shape[0]), int(positions.shape[1])
    device = positions.device
    flat = positions.reshape(b * m, 3)
    batch = torch.arange(b, device=device).repeat_interleave(m)
    edges = radius_edges(flat, batch, cutoff, max_num_neighbors=max_num_neighbors,
                         backend=backend)
    if node_attrs is None:
        attrs = torch.zeros((b * m, 0), dtype=positions.dtype, device=device)
    else:
        attrs = torch.as_tensor(node_attrs, device=device)
        if attrs.dim() == 3:
            attrs = attrs.reshape(b * m, -1)
        elif attrs.shape[0] == m and b != 1:
            attrs = attrs.repeat(b, 1)
        if attrs.shape[0] != b * m:
            raise ValueError(f'node_attrs must be (M, T) or (B M, T), got {tuple(attrs.shape)} '
                             f'for B={b}, M={m}')
    return {'positions': flat, 'edge_index': edges, 'batch': batch,
            'ptr': torch.arange(0, (b + 1) * m, m, device=device)[:b + 1],
            'node_attrs': attrs,
            'shifts': torch.zeros((edges.shape[1], 3), dtype=positions.dtype, device=device)}
