"""
aimmd.cgaimmd.flow_matching.painn
=================================

:class:`JointPaiNN`, the PaiNN flow-matching network of C-GAIMMD: one
equivariant PaiNN trunk with two heads.

- The **committor head** is an invariant scalar, the logit committor toward
  ``sorted_states[2]``, a per-atom MLP of the trunk's scalars evaluated at
  ``(x, t = 1, null token)`` and sum-pooled. It is AIMMD's ``forward``, and
  all that the C-GAIMMD network protocol asks of the network
  (:mod:`aimmd.cgaimmd.protocol`).
- The **velocity head** is an equivariant vector field
  ``v(x_t, t | lambda)`` in flow units: gated trunk vectors plus a learned
  pairwise term along every edge, center-of-mass free. Integrated from the
  Gaussian prior (:mod:`aimmd.cgaimmd.flow_matching.prior`) by the generator
  ``'ode'`` (:class:`aimmd.cgaimmd.flow_matching.sampler.ODEGenerator`, the
  network's :meth:`~JointPaiNN.default_generator`), it generates
  configurations at a requested committor level.

The trunk is a pure-torch PaiNN (Schütt, Unke, Gastegger, 2021) with FiLM
conditioning on flow time and committor level, optional bond-edge features
and an optional parity-breaking cross-product term (ITO), which makes the
network SE(3)- instead of E(3)-equivariant, so that it tells a structure from
its mirror image. Module and buffer names are those of the flow branch
(``aimmd.network.joint_painn`` of ``dev/committor-flow``), so its checkpoints
load unchanged.

Two optional topology features, both off by default, let the network see the
bonded structure at every flow time, in particular early in the flow, where a
prior sample puts bonded atoms farther apart than the cutoff:

- **topology edges** (``topology_geometry``): every pair within three bonds
  is an edge at any distance, merged with the radius graph, carrying the
  topological class, an indicator (radius-only, topology-only, both), the
  force-field reference distances and a radial basis of the current distance
  that is not cut off at the radial cutoff;
- **random-walk positional encodings** (``positional_encoding``): k-step
  return probabilities of the bond graph, added to the atoms' embedding.

The model needs only torch; graphs of coordinates are built on the fly by
:func:`aimmd.cgaimmd.flow_matching.graph_batch.radius_edges` (PyG when
``torch_cluster`` is installed, else pure torch), and graph-mode inputs
(:class:`aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`) are used as
they come.

Like every C-GAIMMD network, the class used in a params file must be defined
there (``Params.save`` writes ``from <module of the class> import
network``):

>>> from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
>>> class Network(JointPaiNN):
...     pass

The model is built on the CPU and never touches CUDA at construction;
C-GAIMMD code moves it at use.
"""

import math
import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn

from .. import _legacy
from . import flowmath
from .graph_batch import BACKENDS, radius_edges
from .graphs import (N_BOND_CLASSES, TOPOLOGY_INDICATORS, bond_edge_features,
                     merge_topology_edges, topology_edge_template)
from .prior import GaussianPrior


ARCHITECTURE_KEYS = ('coord_scale', 'lambda_clip', 'n_channels', 'n_layers', 'n_basis', 'cutoff',
                     'cond_dim', 'lambda_embedding_scale', 'committor_hidden', 'velocity_hidden',
                     'velocity_radial', 'velocity_min_distance', 'parity_breaking')
"""The required architecture keywords of :class:`JointPaiNN` (and of ``from_topology``)."""


# --------------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------------

def safe_norm(v, dim=-1, keepdim=False, eps=1e-8):
    """``sqrt(|v|^2 + eps^2)``: the Euclidean norm, smooth and twice differentiable at ``v = 0``.

    Parameters
    ----------
    v : torch.Tensor
        Vectors along `dim`.
    dim : int, optional
        Axis of the components.
    keepdim : bool, optional
        Keep the reduced axis.
    eps : float, optional
        Smoothing length.

    Returns
    -------
    torch.Tensor
        The smoothed norms.

    Notes
    -----
    The gradient penalty of the flow-matching fit differentiates the trunk twice; the
    second derivative of ``v.norm().clamp_min(eps)`` is NaN at ``v = 0``,
    which a vector channel is whenever an atom has no neighbour.
    """
    return (v.pow(2).sum(dim=dim, keepdim=keepdim) + eps * eps).sqrt()


class SinusoidalEmbedding(nn.Module):
    """Transformer-style ``[sin, cos](scale x omega_k)`` features, ``omega_k = 10^(-4 k / (dim / 2))``.

    Parameters
    ----------
    dim : int
        Even number of features.
    scale : float, optional
        Multiplies the input before the frequencies.
    max_period : float, optional
        Longest period of the frequency ladder.
    """

    def __init__(self, dim, scale=1.0, max_period=1e4):
        super().__init__()
        if dim % 2:
            raise ValueError(f'SinusoidalEmbedding dim must be even, got {dim}')
        self.dim, self.scale, self.max_period = int(dim), float(scale), float(max_period)

    def forward(self, x):
        """``(B,)`` inputs to ``(B, dim)`` features."""
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period)
                          * torch.arange(half, device=x.device, dtype=dtype) / half)
        angle = (x.reshape(-1, 1).to(dtype) * self.scale) * freqs.reshape(1, -1)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class ConditionEmbedding(nn.Module):
    """``c = MLP([emb(t), emb(c_lambda)])`` with a learned null token for the unconditional branch.

    Parameters
    ----------
    dim : int
        Width ``d`` of the condition vector (even).
    lambda_scale : float
        Scale of the committor-level embedding (the squashed level is in ``[-1, 1]``).

    Notes
    -----
    The null token (initialized N(0, 0.02^2)) replaces the level embedding for
    ``c_lambda = None`` and for the rows of ``uncond_mask``; it is a different
    input from level 0 (the transition state). The MLP is ``2 d -> 2 d -> d``
    with SiLU.
    """

    def __init__(self, dim, lambda_scale):
        super().__init__()
        self.dim = int(dim)
        self.t_embed = SinusoidalEmbedding(dim, scale=1.0)
        self.lambda_embed = SinusoidalEmbedding(dim, scale=lambda_scale)
        self.null_token = nn.Parameter(torch.zeros(dim))
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
        self.reset_special()

    def reset_special(self):
        """Draw the null token from N(0, 0.02^2), as at construction."""
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, t, c, uncond_mask=None):
        """``(B, dim)`` condition vectors of flow times `t` and squashed levels `c` (None: null token)."""
        te = self.t_embed(t)
        if c is None:
            le = self.null_token.unsqueeze(0).expand(te.shape[0], -1)
        else:
            le = self.lambda_embed(c)
            if uncond_mask is not None:
                mask = torch.as_tensor(uncond_mask, device=te.device).to(torch.bool)
                le = torch.where(mask.reshape(-1, 1), self.null_token.unsqueeze(0), le)
        return self.mlp(torch.cat([te, le], dim=-1))


class FiLM(nn.Module):
    """``s <- s (1 + gamma(c)) + beta(c)`` on the invariant channels, zero-initialized (identity)."""

    def __init__(self, cond_dim, n_scalar):
        super().__init__()
        self.n_scalar = int(n_scalar)
        self.proj = nn.Linear(cond_dim, 2 * n_scalar)
        self.reset_special()

    def reset_special(self):
        """Zero the projection, so that the layer starts as the identity."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features, cond_per_atom):
        """Modulate the first ``n_scalar`` columns of `features` with the per-atom condition."""
        gamma, beta = self.proj(cond_per_atom).chunk(2, dim=-1)
        head = features[:, :self.n_scalar] * (1.0 + gamma) + beta
        if features.shape[1] == self.n_scalar:
            return head
        return torch.cat([head, features[:, self.n_scalar:]], dim=-1)


class CosineCutoff(nn.Module):
    """``0.5 (cos(pi r / r_max) + 1)`` inside the cutoff, 0 beyond (smooth at ``r_max``)."""

    def __init__(self, r_max):
        super().__init__()
        self.register_buffer('r_max', torch.tensor(float(r_max)))

    def forward(self, r):
        """Cutoff weights of distances `r`."""
        return torch.where(r < self.r_max, 0.5 * (torch.cos(math.pi * r / self.r_max) + 1.0),
                           torch.zeros_like(r))


class BesselBasis(nn.Module):
    """PaiNN's radial basis ``sin(n pi r / r_max) / r``, ``n = 1 .. n_basis``."""

    def __init__(self, n_basis, r_max):
        super().__init__()
        self.register_buffer('r_max', torch.tensor(float(r_max)))
        self.register_buffer('freqs', torch.arange(1, n_basis + 1, dtype=torch.float32) * math.pi)
        self.n_basis = int(n_basis)

    def forward(self, r):
        """``(E, n_basis)`` basis values of distances `r` (Å)."""
        r = r.clamp_min(1e-6).reshape(-1, 1)
        return torch.sin(self.freqs.reshape(1, -1) * r / self.r_max) / r


def _cosine_cutoff(r, r_max):
    """``0.5 (cos(pi r / r_max) + 1)`` for ``r < r_max``, 0 beyond (the :class:`CosineCutoff` function)."""
    return torch.where(r < r_max, 0.5 * (torch.cos(math.pi * r / r_max) + 1.0),
                       torch.zeros_like(r))


class TopologyBasis(nn.Module):
    """Radial basis of the topology edges: Gaussians on ``[0, r_max]`` under a cosine envelope, and ``r / (r + r_max)``.

    Parameters
    ----------
    n_basis : int
        Number of Gaussians (centers ``linspace(0, r_max, n_basis)``, width their spacing).
    r_max : float
        Range of the Gaussians and of their envelope, Å (the topology cutoff, larger than the radial cutoff).

    Notes
    -----
    The Gaussians vanish smoothly at `r_max`; the last feature, ``r / (r +
    r_max)``, keeps growing beyond it, so the distance of a bonded pair is
    never lost however far apart a prior sample puts it.
    """

    def __init__(self, n_basis, r_max):
        super().__init__()
        self.n_basis = int(n_basis)
        self.register_buffer('r_max', torch.tensor(float(r_max)))
        self.register_buffer('centres', torch.linspace(0.0, float(r_max), self.n_basis))
        self.register_buffer('width', torch.tensor(float(r_max) / max(self.n_basis - 1, 1)))

    def forward(self, r):
        """``(E, n_basis + 1)`` features of distances `r` (Å)."""
        r = r.reshape(-1, 1)
        gauss = torch.exp(-((r - self.centres.reshape(1, -1)) ** 2) / (2.0 * self.width ** 2))
        envelope = _cosine_cutoff(r, self.r_max)
        return torch.cat([gauss * envelope, r / (r + self.r_max)], dim=-1)


class _Message(nn.Module):
    """PaiNN's continuous-filter message; with parity breaking also ITO's cross-product term."""

    def __init__(self, n_channels, n_edge, r_max):
        super().__init__()
        self.n = int(n_channels)
        self.n_edge = int(n_edge)
        self.phi = nn.Sequential(nn.Linear(n_channels, n_channels), nn.SiLU(),
                                 nn.Linear(n_channels, 3 * n_channels))
        self.filter = nn.Linear(n_edge, 3 * n_channels)
        self.cutoff = CosineCutoff(r_max)
        self.parity_breaking = False

    def add_parity_breaking(self):
        """Create the cross-product term (separate modules, ``filter_cross`` zero-initialized)."""
        if self.parity_breaking:
            return
        n = self.n
        self.phi_cross = nn.Sequential(nn.Linear(n, n), nn.SiLU(), nn.Linear(n, n))
        self.filter_cross = nn.Linear(self.n_edge, n)
        self.parity_breaking = True
        self.reset_special()

    def reset_special(self):
        """Zero ``filter_cross`` (if present), so that the cross-product term starts at zero."""
        if self.parity_breaking:
            nn.init.zeros_(self.filter_cross.weight)
            nn.init.zeros_(self.filter_cross.bias)

    def forward(self, s, v, edge_index, r_hat, r_len, rbf, envelope=None):
        """Scalar and vector messages summed onto the receiving atoms (`envelope` replaces the cosine cutoff)."""
        sender, receiver = edge_index
        cut = (self.cutoff(r_len) if envelope is None else envelope).reshape(-1, 1)
        x = self.phi(s)[sender] * (self.filter(rbf) * cut)
        ds, dvv, dvs = torch.split(x, self.n, dim=-1)
        dv = v[sender] * dvv.unsqueeze(-1) + r_hat.unsqueeze(1) * dvs.unsqueeze(-1)
        if self.parity_breaking:
            # edge vector x the receiving atom's own vectors: rotation equivariant, parity even
            gate = self.phi_cross(s)[sender] * self.filter_cross(rbf) * cut
            v_dst = v[receiver]
            cross = torch.linalg.cross(r_hat.unsqueeze(1).expand_as(v_dst), v_dst, dim=-1)
            dv = dv + gate.unsqueeze(-1) * cross
        s_out = torch.zeros_like(s).index_add_(0, receiver, ds)
        v_out = torch.zeros_like(v).index_add_(0, receiver, dv)
        return s_out, v_out


class _Update(nn.Module):
    """PaiNN's node update: scalars learn from vectors through norms and inner products."""

    def __init__(self, n_channels):
        super().__init__()
        self.n = int(n_channels)
        self.U = nn.Linear(n_channels, n_channels, bias=False)
        self.V = nn.Linear(n_channels, n_channels, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_channels, n_channels), nn.SiLU(),
                                 nn.Linear(n_channels, 3 * n_channels))

    def forward(self, s, v):
        """Scalar and vector updates of every atom."""
        uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)
        vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)
        a = self.mlp(torch.cat([s, safe_norm(vv, dim=-1)], dim=-1))
        avv, asv, ass = torch.split(a, self.n, dim=-1)
        return ass + asv * (uv * vv).sum(-1), avv.unsqueeze(-1) * uv


class PaiNNTrunk(nn.Module):
    """PaiNN trunk with FiLM conditioning: positions and tokens to invariant scalars and equivariant vectors.

    Parameters
    ----------
    tokens : array-like
        ``(M,)`` embedding tokens of the atoms; the embedding table is over their sorted unique values.
    n_channels, n_layers, n_basis : int
        Channels, message/update blocks and Bessel functions.
    cutoff : float
        Radial cutoff, Å.
    cond_dim : int
        Width of the ``(t, lambda)`` condition vector.
    lambda_embedding_scale : float
        Scale of the committor-level embedding.
    bond_class : torch.Tensor, optional
        ``(M, M)`` uint8 topological distance classes; appends the 4-class bond one-hot to every edge.
    parity_breaking : bool, optional
        Add ITO's cross-product message term (SE(3) instead of E(3)); needs ``n_layers >= 2``.
    topology : dict, optional
        Topology edges, the dict of :func:`aimmd.cgaimmd.flow_matching.graphs.topology_edge_template`; None: the radius graph alone.
    positional_encoding : torch.Tensor, optional
        ``(M, K)`` node encodings (random-walk return probabilities) added to the embedding; None: none.
    topology_cutoff : float, optional
        Range of the topology edges' radial basis, Å (with `topology`).
    topology_n_basis : int, optional
        Gaussians of that basis (with `topology`).

    Notes
    -----
    Scalars out: the concatenation of every layer's scalars
    (``n_scalar = n_channels * n_layers``); vectors: the last layer's
    (``n_channels``). ``s0 = Emb(token) + W_c c`` (``W_c`` zero-initialized),
    ``v0 = 0``; with positional encodings ``+ W_pe pe`` (``W_pe``
    zero-initialized, no bias).

    With topology edges the edge features are, in this order: the Bessel
    basis (multiplied by the cosine cutoff on topology edges), the 4-class
    one-hot of the topological distance (radius-only edges are more than
    three bonds apart), the indicator of :data:`aimmd.cgaimmd.flow_matching.graphs.TOPOLOGY_INDICATORS`
    (on a topology edge ``1 - f_cut(r)`` and ``f_cut(r)``, the cosine cutoff:
    a smooth version of "beyond / within the cutoff"), the reference
    distances ``(d_min, d_max)`` in Å (zero on radius-only edges) and
    :class:`TopologyBasis` of the current distance (zero on radius-only
    edges). Radius-only edges keep the cosine cutoff as the envelope of their
    messages; topology edges are a fixed set and have none, so a bonded pair
    sends its messages at any distance and the network stays smooth when it
    crosses the cutoff. The class one-hot comes from the edge set, which
    equals the `bond_class` lookup.
    """

    def __init__(self, tokens, *, n_channels, n_layers, n_basis, cutoff, cond_dim,
                 lambda_embedding_scale, bond_class=None, parity_breaking=False, topology=None,
                 positional_encoding=None, topology_cutoff=15.0, topology_n_basis=16):
        super().__init__()
        values = sorted({int(z) for z in np.asarray(tokens).reshape(-1)})
        self.register_buffer('z_table', torch.tensor(values, dtype=torch.long))
        self.n_channels, self.n_layers, self.r_max = int(n_channels), int(n_layers), float(cutoff)
        if parity_breaking and n_layers < 2:
            raise ValueError('parity_breaking needs n_layers >= 2: the vector features start at '
                             'zero, so the cross-product term vanishes in the first block')
        self.parity_breaking = bool(parity_breaking)
        n_edge = int(n_basis)
        self.bond_edges = bond_class is not None
        if self.bond_edges:
            self.register_buffer('bond_class', torch.as_tensor(bond_class).to(torch.uint8))
            n_edge += N_BOND_CLASSES
        self.topology_edges = topology is not None
        self.topology_cutoff = float(topology_cutoff)
        self.topology_n_basis = int(topology_n_basis)
        if self.topology_edges:
            self.register_buffer('topology_index', torch.as_tensor(topology['index'])
                                 .to(torch.long))
            self.register_buffer('topology_separation', torch.as_tensor(topology['separation'])
                                 .to(torch.uint8))
            self.register_buffer('topology_reference', torch.as_tensor(topology['reference'])
                                 .to(torch.float32))
            self.register_buffer('topology_pairs', torch.as_tensor(topology['pair_mask'])
                                 .to(torch.bool))
            if not self.bond_edges:
                n_edge += N_BOND_CLASSES
            n_edge += len(TOPOLOGY_INDICATORS) + 2 + self.topology_n_basis + 1
        self.n_edge = n_edge
        self.embedding = nn.Embedding(len(values), n_channels)
        self.rbf = BesselBasis(n_basis, cutoff)
        self.messages = nn.ModuleList(_Message(n_channels, n_edge, cutoff) for _ in range(n_layers))
        self.updates = nn.ModuleList(_Update(n_channels) for _ in range(n_layers))
        self.cond = ConditionEmbedding(cond_dim, lambda_embedding_scale)
        self.cond_to_embed = nn.Linear(cond_dim, n_channels)
        self.film = nn.ModuleList(FiLM(cond_dim, n_channels) for _ in range(n_layers))
        if self.parity_breaking:
            for message in self.messages:
                message.add_parity_breaking()
        if self.topology_edges:
            self.topology_rbf = TopologyBasis(self.topology_n_basis, self.topology_cutoff)
        self.positional_encoding_steps = 0
        if positional_encoding is not None:
            encoding = torch.as_tensor(positional_encoding).to(torch.float32)
            self.positional_encoding_steps = int(encoding.shape[1])
            self.register_buffer('positional_encoding', encoding)
            self.pe_to_embed = nn.Linear(self.positional_encoding_steps, n_channels, bias=False)
        self.n_scalar = self.n_channels * self.n_layers
        self.n_vector = self.n_channels
        self.reset_special()

    def reset_special(self):
        """Zero the condition-to-embedding map ``W_c`` (and ``W_pe``), as at construction."""
        nn.init.zeros_(self.cond_to_embed.weight)
        nn.init.zeros_(self.cond_to_embed.bias)
        if self.positional_encoding_steps:
            nn.init.zeros_(self.pe_to_embed.weight)

    def token_index(self, tokens):
        """Rows of the embedding table of `tokens` (ValueError for a token the trunk was not built with)."""
        index = torch.searchsorted(self.z_table, tokens).clamp_max(len(self.z_table) - 1)
        if not bool((self.z_table[index] == tokens).all()):
            missing = sorted(set(tokens[self.z_table[index] != tokens].tolist()))
            raise ValueError(f'PaiNNTrunk was built without the tokens {missing}')
        return index

    def _edge_features(self, r_len, edge_index, batch, ptr, topology_ids):
        rbf = self.rbf(r_len)
        if not self.topology_edges:
            if self.bond_edges:
                rbf = torch.cat([rbf, bond_edge_features(edge_index, batch, ptr,
                                                         self.bond_class).to(rbf.dtype)], dim=-1)
            return rbf, None
        if topology_ids is None:
            raise ValueError('a trunk with topology edges needs the topology_ids of '
                             'aimmd.cgaimmd.flow_matching.graphs.merge_topology_edges')
        is_topology = topology_ids >= 0
        ids = topology_ids.clamp_min(0)
        flag = is_topology.to(rbf.dtype)
        cut = _cosine_cutoff(r_len, self.r_max)
        bessel = torch.where(is_topology.unsqueeze(-1), rbf * cut.unsqueeze(-1), rbf)
        classes = torch.where(is_topology, self.topology_separation[ids].long() - 1,
                              torch.full_like(ids, N_BOND_CLASSES - 1))
        one_hot = torch.nn.functional.one_hot(classes, N_BOND_CLASSES).to(rbf.dtype)
        indicator = torch.stack([1.0 - flag, flag * (1.0 - cut), flag * cut], dim=-1)
        reference = self.topology_reference[ids].to(rbf.dtype) * flag.unsqueeze(-1)
        long_range = self.topology_rbf(r_len) * flag.unsqueeze(-1)
        features = torch.cat([bessel, one_hot, indicator, reference, long_range], dim=-1)
        return features, torch.where(is_topology, torch.ones_like(cut), cut)

    def edge_features(self, pos, batch, ptr, edge_index, topology_ids=None):
        """Edge features and message envelopes of the given edges.

        Parameters
        ----------
        pos : torch.Tensor
            ``(N, 3)`` positions, Å.
        batch : torch.Tensor
            ``(N,)`` long sample index of every atom.
        ptr : torch.Tensor
            ``(B + 1,)`` long offsets of the samples.
        edge_index : torch.Tensor
            ``(2, E)`` long edges (with topology edges: merged by :func:`aimmd.cgaimmd.flow_matching.graphs.merge_topology_edges`).
        topology_ids : torch.Tensor, optional
            ``(E,)`` long template column of every topology edge, -1 elsewhere (required with topology edges).

        Returns
        -------
        tuple
            ``(features (E, n_edge), envelope (E,) or None)``; None means the cosine cutoff of the radius graph.
        """
        sender, receiver = edge_index
        r_len = (pos[sender] - pos[receiver]).norm(dim=-1)
        return self._edge_features(r_len, edge_index, batch, ptr, topology_ids)

    def forward(self, pos, tokens, batch, ptr, t, c=None, uncond_mask=None, *, edge_index,
                topology_ids=None, edge_features=None):
        """``(scalars (N, C L), vectors (N, C, 3))`` of positions (Å) and tokens on the given edges.

        `topology_ids` marks the topology edges of a merged edge set;
        `edge_features` is a precomputed ``(features, envelope)`` of
        :meth:`edge_features` for the same edges.
        """
        sender, receiver = edge_index
        rij = pos[sender] - pos[receiver]
        r_len = rij.norm(dim=-1)
        r_hat = rij / r_len.clamp_min(1e-8).unsqueeze(-1)
        if edge_features is None:
            rbf, envelope = self._edge_features(r_len, edge_index, batch, ptr, topology_ids)
        else:
            rbf, envelope = edge_features
        cond = self.cond(t, c, uncond_mask)[batch]
        s = self.embedding(self.token_index(tokens)) + self.cond_to_embed(cond)
        if self.positional_encoding_steps:
            local = torch.arange(pos.shape[0], device=pos.device) - ptr.to(pos.device)[batch]
            s = s + self.pe_to_embed(self.positional_encoding[local].to(s.dtype))
        v = torch.zeros(pos.shape[0], self.n_channels, 3, device=pos.device, dtype=pos.dtype)
        scalars = []
        for message, update, film in zip(self.messages, self.updates, self.film):
            ds, dv = message(s, v, edge_index, r_hat, r_len, rbf, envelope)
            s, v = s + ds, v + dv
            ds, dv = update(s, v)
            s, v = s + ds, v + dv
            s = film(s, cond)
            scalars.append(s)
        return torch.cat(scalars, dim=-1), v


class CommittorHead(nn.Module):
    """Per-atom MLP ``n_scalar -> hidden -> 1`` (SiLU) of the invariant scalars, sum-pooled per sample.

    Parameters
    ----------
    n_scalar : int
        Input width.
    hidden : tuple of int
        Hidden widths.

    Notes
    -----
    The last layer starts with weights N(0, 1e-6) and bias 0, so an
    untrained model has ``lambda ~ 0`` (committor 1/2) everywhere. Kept as
    ``heads.0`` of a ``ModuleList``, the key names of the flow branch's
    checkpoints (an ensemble of heads is not supported).
    """

    def __init__(self, n_scalar, hidden):
        super().__init__()
        layers, width = [], int(n_scalar)
        for h in hidden:
            layers += [nn.Linear(width, int(h)), nn.SiLU()]
            width = int(h)
        layers.append(nn.Linear(width, 1))
        self.heads = nn.ModuleList([nn.Sequential(*layers)])
        self.reset_special()

    def reset_special(self):
        """Last layer: bias 0 and weights of standard deviation 1e-3, so that the logits start near 0."""
        for head in self.heads:
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, std=1e-3)

    def forward(self, scalars, batch, n_samples):
        """``(n_samples,)`` logits: the per-atom outputs summed per sample."""
        per_atom = self.heads[0](scalars).reshape(-1)
        pooled = torch.zeros(n_samples, device=scalars.device, dtype=scalars.dtype)
        return pooled.index_add(0, batch, per_atom)


class VelocityHead(nn.Module):
    """``v_i = sum_c g_c(s_i) v_ic + sum_j phi(s_i, s_j, rho(r_ij)) r_hat_ij``, in flow units, COM free.

    Parameters
    ----------
    n_scalar, n_vector : int
        Trunk feature widths.
    n_radial : int
        Gaussians of the pair term's radial basis on ``[0, cutoff]``.
    cutoff : float
        Radial cutoff, Å.
    hidden : int
        Hidden width of the gate and pair MLPs.
    min_distance : float
        ``d_min`` of ``r~ = sqrt(r^2 + d_min^2)``, Å.
    n_edge_attr : int, optional
        Width of extra edge attributes appended to the pair MLP's input (the topology features); 0: none.

    Notes
    -----
    The Gaussian width is ``3 cutoff / (n_radial - 1)``; the last layers of
    the gate and the pair MLP are zero-initialized, so an untrained head
    returns 0. The result is divided by ``coord_scale`` (the trunk works in
    Å) and its per-sample mean is removed. With topology edges the pair term
    runs over them too and also sees their attributes (class, indicator,
    reference distances, long-range basis), so it can pull a bonded pair
    together from any distance.
    """

    def __init__(self, n_scalar, n_vector, n_radial, cutoff, hidden, min_distance,
                 n_edge_attr=0):
        super().__init__()
        self.r_max = float(cutoff)
        self.min_distance = float(min_distance)
        self.gate = nn.Sequential(nn.Linear(n_scalar, hidden), nn.SiLU(),
                                  nn.Linear(hidden, n_vector))
        self.register_buffer('centres', torch.linspace(0.0, cutoff, n_radial))
        self.register_buffer('width', torch.tensor(3.0 * cutoff / max(n_radial - 1, 1)))
        self.n_edge_attr = int(n_edge_attr)
        self.pair = nn.Sequential(nn.Linear(2 * n_scalar + n_radial + self.n_edge_attr, hidden),
                                  nn.SiLU(), nn.Linear(hidden, 1))
        self.reset_special()

    def reset_special(self):
        """Zero the last layers of the gate and the pair MLP, so that the head starts at 0."""
        for layer in (self.pair[-1], self.gate[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, scalars, vectors, pos, edge_index, n_samples, coord_scale, edge_attr=None):
        """``(n_samples, M, 3)`` velocities in flow units, center-of-mass free."""
        v = (self.gate(scalars).unsqueeze(-1) * vectors).sum(dim=1)
        sender, receiver = edge_index
        rij = pos[sender] - pos[receiver]
        r_len = (rij.pow(2).sum(-1, keepdim=True) + self.min_distance ** 2).sqrt()
        r_hat = rij / r_len
        rbf = torch.exp(-((r_len - self.centres.reshape(1, -1)) ** 2) / (2 * self.width ** 2))
        inputs = [scalars[sender], scalars[receiver], rbf]
        if edge_attr is not None:
            inputs.append(edge_attr.to(rbf.dtype))
        w = self.pair(torch.cat(inputs, dim=-1))
        v = v.index_add(0, receiver, w * r_hat)
        v = (v / coord_scale).reshape(n_samples, -1, 3)
        return v - v.mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------------------------
# the joint model
# --------------------------------------------------------------------------------------------

def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) \
            or value <= 0:
        raise ValueError(f'{name} must be a finite number > 0, got {value!r}')
    return float(value)


def _count(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an int >= {minimum}, got {value!r}')
    return int(value)


def _bond_class(bond_distances, n_atoms):
    d = torch.as_tensor(np.asarray(bond_distances)).to(torch.long)
    if d.shape != (n_atoms, n_atoms):
        raise ValueError(f'bond_distances must be ({n_atoms}, {n_atoms}), got {tuple(d.shape)}')
    if bool((d < 0).any()):
        raise ValueError('bond_distances must be >= 0; give disconnected pairs a distance of 4 or '
                         'more (aimmd.cgaimmd.preparation.topology.topological_distances(..., '
                         'max_distance=4) does), not a negative marker')
    return (d.clamp(1, N_BOND_CLASSES) - 1).to(torch.uint8)


class JointPaiNN(nn.Module):
    """The PaiNN flow-matching network: PaiNN trunk, committor head and velocity head.

    Parameters
    ----------
    n_atoms : int
        Number ``M`` of generator atoms.
    atom_tokens : array-like
        ``(M,)`` embedding tokens (``aimmd.cgaimmd.flow_matching.embedding.atom_tokens``; atomic numbers for the ``'element'`` scheme).
    coord_scale : float
        Å per flow unit (``aimmd.cgaimmd.flow_matching.pretrain.estimate_coord_scale``).
    lambda_clip : float
        The conditioning range ``L`` of the committor level.
    n_channels : int
        PaiNN channels.
    n_layers : int
        PaiNN message/update blocks.
    n_basis : int
        Bessel radial functions.
    cutoff : float
        Radial cutoff of the graphs, Å.
    cond_dim : int
        Width of the ``(t, lambda)`` condition vector (even).
    lambda_embedding_scale : float
        Scale of the embedding of the squashed level ``tanh(lambda / L)``.
    committor_hidden : tuple of int
        Hidden widths of the committor head.
    velocity_hidden : int
        Hidden width of the velocity head's gate and pair MLPs.
    velocity_radial : int
        Gaussians of the velocity head's pair term.
    velocity_min_distance : float
        Distance floor of the pair term, Å.
    parity_breaking : bool
        Add the cross-product term that makes the trunk SE(3)- instead of E(3)-equivariant (needs ``n_layers >= 2``).
    atomic_numbers : array-like, optional
        ``(M,)`` true elements; default: `atom_tokens`, which must then all be <= 118.
    bond_distances : array-like, optional
        ``(M, M)`` non-negative topological distances (bonds; 4 or more for disconnected pairs); enables the bond-edge features.
    ot_symmetry : object, optional
        ``aimmd.cgaimmd.preparation.symmetry.OTSpec`` (anything with ``kwargs()``); None: rotations only.
    ot_rounds : int, optional
        Alternations of rotation and permutations in :meth:`flow_pair`.
    max_num_neighbors : int, optional
        Neighbour cap of the graphs built on the fly (as ``graph_utils``).
    edge_backend : {'auto', 'pyg', 'torch'}, optional
        Backend of :func:`aimmd.cgaimmd.flow_matching.graph_batch.radius_edges`.
    committor_chunk : int, optional
        Frames per trunk call of :meth:`committor` in eval mode; None: ``max(1, 8192 // M)``.
    topology_geometry : aimmd.cgaimmd.preparation.topology.ReferenceGeometry, optional
        Force-field geometry of the generator atoms; enables the topology edges (every pair within three bonds, at any distance). None: the radius graph alone.
    positional_encoding : array-like, optional
        ``(M, K)`` node encodings (:func:`aimmd.cgaimmd.preparation.topology.random_walk_encoding`) added to the embedding. None: none.
    topology_cutoff : float, optional
        Range ``r_topo`` of the topology edges' radial basis, Å (default 15; used with `topology_geometry`).
    topology_n_basis : int, optional
        Gaussians of that basis (default 16; used with `topology_geometry`).

    Raises
    ------
    TypeError
        If an architecture keyword is missing (they have no defaults), or for an unknown keyword.
    ValueError
        For inconsistent sizes or out-of-range values.

    Notes
    -----
    Every architecture hyperparameter is required, so a params file that
    leaves one out fails at construction instead of silently getting another
    system's architecture. The NANMA values of the flow branch's campaigns
    (an example, not defaults) are ``n_channels=128, n_layers=4, n_basis=20,
    cutoff=5.0, cond_dim=64, lambda_embedding_scale=4.0,
    committor_hidden=(128,), velocity_hidden=128, velocity_radial=32,
    velocity_min_distance=0.7, lambda_clip=10.0`` with
    ``coord_scale=1.6495``. The keywords with defaults are implementation
    switches that do not change the network's function.

    The network is a C-GAIMMD network (:mod:`aimmd.cgaimmd.protocol`: the
    committor ``forward``, ``committor``, ``n_atoms``, ``lambda_clip``,
    ``reset_parameters``) whose :meth:`default_generator` is the
    flow-matching ODE ``'ode'``. ``velocity``, ``sample_prior``, ``to_flow``,
    ``from_flow`` and ``flow_pair`` are the members that generator and the
    flow-matching fit use (:mod:`aimmd.cgaimmd.flow_matching.sampler`); the
    network has no ``generate`` of its own.

    Flow units are ``to_flow(x) = (x - mean_atoms(x)) / coord_scale``. The
    prior is the center-of-mass-free Gaussian
    (:class:`~aimmd.cgaimmd.flow_matching.prior.GaussianPrior`, the attribute
    ``prior``), and the velocity removes the center of mass per sample.
    ``coord_scale`` and ``lambda_clip`` are float32 buffers (the checkpoint
    wins over the constructor), read as floats by the attributes of the same
    names.

    Graph-mode inputs (the batch dict of
    :class:`~aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`, or of
    AIMMD's graph pipeline) must hold exactly the ``M`` generator atoms in
    network order per graph; their ``node_attrs`` are ignored in favour of the
    model's own tokens.

    The topology features (``topology_geometry``, ``positional_encoding``)
    are off by default, and then the model, its state dict and its outputs
    are those of the model without them. With topology edges the radius graph
    of every input (on the fly or from a graph dict) is merged with the fixed
    topology pairs (:func:`aimmd.cgaimmd.flow_matching.graphs.merge_topology_edges`) in the
    committor and the velocity alike; the velocity head's pair term then also
    sees the topology edges' attributes. The reference distances come from
    the force field, never from a frame. Both features are buffers of the
    state dict (``trunk.topology_*``, ``trunk.positional_encoding``), so a
    checkpoint loads only into a model built with the same switches.
    """

    def __init__(self, n_atoms, atom_tokens, *, coord_scale, lambda_clip, n_channels, n_layers,
                 n_basis, cutoff, cond_dim, lambda_embedding_scale, committor_hidden,
                 velocity_hidden, velocity_radial, velocity_min_distance, parity_breaking,
                 atomic_numbers=None, bond_distances=None, ot_symmetry=None, ot_rounds=2,
                 max_num_neighbors=128, edge_backend='auto', committor_chunk=None,
                 topology_geometry=None, positional_encoding=None, topology_cutoff=15.0,
                 topology_n_basis=16, **old_keywords):
        super().__init__()
        _legacy.dropped_keywords('JointPaiNN', old_keywords)       # also for subclasses
        n_atoms = _count('n_atoms', n_atoms)
        tokens = torch.as_tensor(np.asarray(atom_tokens).astype(np.int64)).reshape(-1)
        if len(tokens) != n_atoms:
            raise ValueError(f'atom_tokens has {len(tokens)} entries for n_atoms={n_atoms}')
        if atomic_numbers is None:
            if bool((tokens < 1).any()) or bool((tokens > 118).any()):
                raise ValueError('atom_tokens are not atomic numbers (some are outside 1..118): '
                                 'pass atomic_numbers, the true elements')
            elements = tokens.clone()
        else:
            elements = torch.as_tensor(np.asarray(atomic_numbers).astype(np.int64)).reshape(-1)
            if len(elements) != n_atoms:
                raise ValueError(f'atomic_numbers has {len(elements)} entries for '
                                 f'n_atoms={n_atoms}')
        coord_scale = _positive('coord_scale', coord_scale)
        lambda_clip = _positive('lambda_clip', lambda_clip)
        cutoff = _positive('cutoff', cutoff)
        lambda_embedding_scale = _positive('lambda_embedding_scale', lambda_embedding_scale)
        velocity_min_distance = _positive('velocity_min_distance', velocity_min_distance)
        n_channels = _count('n_channels', n_channels)
        n_layers = _count('n_layers', n_layers)
        n_basis = _count('n_basis', n_basis)
        cond_dim = _count('cond_dim', cond_dim, 2)
        if cond_dim % 2:
            raise ValueError(f'cond_dim must be even, got {cond_dim}')
        velocity_hidden = _count('velocity_hidden', velocity_hidden)
        velocity_radial = _count('velocity_radial', velocity_radial, 2)
        committor_hidden = tuple(_count('committor_hidden entries', h) for h in committor_hidden)
        if not isinstance(parity_breaking, (bool, np.bool_)):
            raise ValueError(f'parity_breaking must be a bool, got {parity_breaking!r}')
        if edge_backend not in BACKENDS:
            raise ValueError(f'edge_backend must be one of {BACKENDS}, got {edge_backend!r}')
        self.ot_rounds = _count('ot_rounds', ot_rounds)
        self.max_num_neighbors = _count('max_num_neighbors', max_num_neighbors)
        self.edge_backend = edge_backend
        self.committor_chunk = None if committor_chunk is None \
            else _count('committor_chunk', committor_chunk)
        self._cutoff = cutoff
        topology = None
        if topology_geometry is not None:
            if int(topology_geometry.n_atoms) != n_atoms:
                raise ValueError(f'topology_geometry has {topology_geometry.n_atoms} atoms for '
                                 f'n_atoms={n_atoms}')
            topology_cutoff = _positive('topology_cutoff', topology_cutoff)
            if topology_cutoff <= cutoff:
                raise ValueError(f'topology_cutoff ({topology_cutoff}) must exceed the radial '
                                 f'cutoff ({cutoff})')
            topology_n_basis = _count('topology_n_basis', topology_n_basis, 2)
            topology = topology_edge_template(
                n_atoms, topology_geometry.pairs, topology_geometry.separation,
                topology_geometry.distance_min, topology_geometry.distance_max)
        if positional_encoding is not None:
            positional_encoding = np.asarray(positional_encoding, dtype=np.float64)
            if positional_encoding.ndim != 2 or positional_encoding.shape[0] != n_atoms \
                    or positional_encoding.shape[1] < 1:
                raise ValueError(f'positional_encoding must be (n_atoms={n_atoms}, K >= 1), got '
                                 f'shape {positional_encoding.shape}')
            if not np.all(np.isfinite(positional_encoding)):
                raise ValueError('positional_encoding must be finite')

        self.register_buffer('atomic_numbers', elements)
        self.register_buffer('atom_tokens', tokens)
        # written into _buffers: the float attributes of the same names are properties
        self._buffers['coord_scale'] = torch.tensor(coord_scale, dtype=torch.float32)
        self._buffers['lambda_clip'] = torch.tensor(lambda_clip, dtype=torch.float32)
        self.trunk = PaiNNTrunk(
            tokens, n_channels=n_channels, n_layers=n_layers, n_basis=n_basis, cutoff=cutoff,
            cond_dim=cond_dim, lambda_embedding_scale=lambda_embedding_scale,
            bond_class=None if bond_distances is None else _bond_class(bond_distances, n_atoms),
            parity_breaking=bool(parity_breaking), topology=topology,
            positional_encoding=positional_encoding, topology_cutoff=topology_cutoff,
            topology_n_basis=topology_n_basis)
        self.committor_head = CommittorHead(self.trunk.n_scalar, committor_hidden)
        # with topology edges the pair term also sees the edge attributes after the Bessel basis
        self.velocity_head = VelocityHead(self.trunk.n_scalar, self.trunk.n_vector,
                                          velocity_radial, cutoff, velocity_hidden,
                                          velocity_min_distance,
                                          n_edge_attr=self._n_edge_attr(n_basis))
        self.prior = GaussianPrior(n_atoms)
        """The prior of the flow (:class:`~aimmd.cgaimmd.flow_matching.prior.GaussianPrior`)."""
        self.ot_symmetry = ot_symmetry
        self._ot_kwargs = self._ot_arguments(ot_symmetry)

    # -- attributes --------------------------------------------------------------------------

    @property
    def n_atoms(self):
        """Number ``M`` of generator atoms."""
        return int(self.atom_tokens.numel())

    @property
    def coord_scale(self):
        """Å per flow unit (the float of the ``coord_scale`` buffer)."""
        return float(self._buffers['coord_scale'])

    @property
    def lambda_clip(self):
        """The conditioning range ``L`` (the float of the ``lambda_clip`` buffer)."""
        return float(self._buffers['lambda_clip'])

    @property
    def cutoff(self):
        """Radial cutoff of the graphs, Å."""
        return self._cutoff

    @property
    def parity_breaking(self):
        """True for the SE(3) trunk (cross-product term), False for the E(3) one."""
        return bool(self.trunk.parity_breaking)

    @property
    def device(self):
        """Device of the model's buffers."""
        return self.atom_tokens.device

    @property
    def topology_edges(self):
        """True if the graphs carry the topology edges (``topology_geometry``)."""
        return bool(self.trunk.topology_edges)

    def _n_edge_attr(self, n_basis):
        """Width of the edge attributes the velocity head sees: the trunk's edge features after the Bessel basis."""
        return self.trunk.n_edge - int(n_basis) if self.trunk.topology_edges else 0

    def atom_tokens_numpy(self):
        """The ``(M,)`` embedding tokens as a numpy array (node labels of validity checks)."""
        return self.atom_tokens.detach().cpu().numpy().copy()

    @staticmethod
    def _ot_arguments(ot_symmetry):
        if ot_symmetry is None:
            return {}
        kwargs = dict(ot_symmetry.kwargs())
        factors = kwargs.get('factors') or ()
        if not isinstance(factors, flowmath.PermutationFactors):
            factors = flowmath.PermutationFactors(factors)
        return {'groups': tuple(tuple(int(i) for i in g) for g in kwargs.get('groups') or ()),
                'factors': factors,
                'molecule_blocks': tuple(tuple(int(i) for i in b)
                                         for b in kwargs.get('molecule_blocks') or ())}

    # -- flow units, prior and pairing --------------------------------------------------------

    def _as_frames(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        m = self.n_atoms
        if x.dim() == 1:
            x = x.reshape(1, m, 3)
        elif x.dim() == 2:
            if x.shape[1] != 3 * m:
                raise ValueError(f'flat input must have width {3 * m} (3 x {m} generator atoms), '
                                 f'got shape {tuple(x.shape)}')
            x = x.reshape(-1, m, 3)
        if x.dim() != 3 or tuple(x.shape[1:]) != (m, 3):
            raise ValueError(f'expected (B, {m}, 3) coordinates, got shape {tuple(x.shape)}')
        return x

    def to_flow(self, x):
        """Å ``(B, M, 3)`` -> flow units: ``(x - mean_atoms(x)) / coord_scale``."""
        x = self._as_frames(x)
        return flowmath.center(x) / self._buffers['coord_scale']

    def from_flow(self, z):
        """Flow units ``(B, M, 3)`` -> Å: ``(z - mean_atoms(z)) * coord_scale``."""
        z = self._as_frames(z)
        return flowmath.center(z) * self._buffers['coord_scale']

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        """Prior samples ``(n, M, 3)`` in flow units, center-of-mass free.

        Parameters
        ----------
        lam : torch.Tensor or None
            ``(n,)`` committor logits; the Gaussian prior ignores them.
        n : int
            Number of samples.
        uncond_mask : torch.Tensor, optional
            ``(n,)`` bool rows of the unconditional branch; the Gaussian prior ignores them
            (accepted because the flow-matching fit passes them).
        generator : torch.Generator, optional
            On the model's device.

        Returns
        -------
        torch.Tensor
            ``(n, M, 3)`` float32 on the model's device: :meth:`GaussianPrior.sample
            <aimmd.cgaimmd.flow_matching.prior.GaussianPrior.sample>`.
        """
        return self.prior.sample(int(n), device=self.device, generator=generator)

    def flow_pair(self, x1, x0):
        """OT pairing: `x1` acted on by the proper rotation and permutation closest to `x0`.

        Parameters
        ----------
        x1 : torch.Tensor
            ``(B, M, 3)`` data samples, flow units.
        x0 : torch.Tensor
            ``(B, M, 3)`` prior samples, flow units.

        Returns
        -------
        torch.Tensor
            The paired, centered `x1`.

        Notes
        -----
        ``flowmath.ot_pair(x1, x0, **ot_symmetry.kwargs(), n_rounds=ot_rounds)``:
        alternating proper Kabsch rotation (never a reflection), exact factor
        enumeration, Hungarian groups and whole-molecule blocks.
        """
        return flowmath.ot_pair(x1, x0, n_rounds=self.ot_rounds, **self._ot_kwargs)

    # -- graphs --------------------------------------------------------------------------------

    def _layout(self, n_samples, device):
        m = self.n_atoms
        batch = torch.arange(n_samples, device=device).repeat_interleave(m)
        ptr = torch.arange(0, (n_samples + 1) * m, m, device=device)[:n_samples + 1]
        return batch, ptr

    def _edges(self, pos, batch):
        return radius_edges(pos, batch, self._cutoff, max_num_neighbors=self.max_num_neighbors,
                            backend=self.edge_backend)

    def _topology_edges(self, edge_index, batch, ptr):
        """``(edge_index, topology_ids)``: the radius graph merged with the topology edges, or ``(edge_index, None)`` without them."""
        if not self.trunk.topology_edges:
            return edge_index, None
        return merge_topology_edges(edge_index, batch, ptr, self.trunk.topology_index,
                                    self.trunk.topology_pairs)

    # -- the committor branch ------------------------------------------------------------------

    def _committor_of(self, pos, n_samples, batch, ptr, edge_index):
        edge_index, topology_ids = self._topology_edges(edge_index, batch, ptr)
        tokens = self.atom_tokens.repeat(n_samples)
        t = torch.ones(n_samples, device=pos.device, dtype=pos.dtype)
        scalars, _ = self.trunk(pos, tokens, batch, ptr, t, None, None, edge_index=edge_index,
                                topology_ids=topology_ids)
        return self.committor_head(scalars, batch, n_samples)

    def _committor(self, x):
        b = x.shape[0]
        pos = flowmath.center(x).reshape(b * self.n_atoms, 3)
        batch, ptr = self._layout(b, x.device)
        return self._committor_of(pos, b, batch, ptr, self._edges(pos, batch))

    def committor(self, x):
        """Logit committor of coordinates that are not on disk.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, M, 3)`` (or flat ``(B, 3 M)``) coordinates in Å.

        Returns
        -------
        torch.Tensor
            ``(B,)`` float32 logits toward ``sorted_states[2]``, equal to ``forward`` on the same frames.

        Notes
        -----
        Evaluated at ``(x, t = 1, null token)`` with graphs built on the fly;
        in eval mode in chunks of ``committor_chunk`` frames. Differentiable
        in `x`.
        """
        x = self._as_frames(x)
        chunk = self.committor_chunk or max(1, 8192 // self.n_atoms)
        if not self.training and x.shape[0] > chunk:
            return torch.cat([self._committor(x[i:i + chunk]) for i in range(0, len(x), chunk)])
        return self._committor(x)

    def forward(self, inp):
        """AIMMD's contract: the values-function input to ``(B, 1)`` logit committors.

        Parameters
        ----------
        inp : torch.Tensor or dict
            Dense ``(B, 3 M)`` Å descriptors, or a graph batch dict (the layout of ``Batch.from_data_list(...).to_dict()``).

        Returns
        -------
        torch.Tensor
            ``(B, 1)`` float32 logits toward ``sorted_states[2]``.

        Raises
        ------
        ValueError
            For a width other than ``3 M``, or graphs that do not hold exactly the ``M`` generator atoms.

        Notes
        -----
        A graph dict brings its own edges (``edge_index``: from
        :class:`~aimmd.cgaimmd.flow_matching.graphs.FlowGraphTransform`, or from
        AIMMD's graph cache); its positions may be translated as a whole (the
        graph pipeline centers them in its box), which the model ignores.
        """
        if isinstance(inp, Mapping):
            return self._forward_graphs(inp).unsqueeze(-1)
        return self.committor(inp).unsqueeze(-1)

    def _forward_graphs(self, graphs):
        m = self.n_atoms
        pos = torch.as_tensor(graphs['positions'], device=self.device).to(torch.float32)
        ptr = graphs.get('ptr')
        if ptr is None:
            ptr = torch.tensor([0, pos.shape[0]], device=pos.device)
        ptr = torch.as_tensor(ptr, device=pos.device)
        counts = ptr[1:] - ptr[:-1]
        if pos.shape[0] % m or not bool((counts == m).all()):
            raise ValueError(f'every graph must hold the {m} generator atoms in network order; got '
                             f'graphs of {sorted(set(counts.tolist()))} atoms')
        b = int(counts.numel())
        pos = flowmath.center(pos.reshape(b, m, 3)).reshape(b * m, 3)
        batch, ptr = self._layout(b, pos.device)
        edge_index = torch.as_tensor(graphs['edge_index'], device=pos.device).to(torch.long)
        return self._committor_of(pos, b, batch, ptr, edge_index)

    # -- the generative branch -------------------------------------------------------------------

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        """Velocity ``v(x_t, t | lam)`` in flow units.

        Parameters
        ----------
        x_t : torch.Tensor
            ``(B, M, 3)`` states, flow units.
        t : torch.Tensor
            ``(B,)`` flow times in ``[0, 1]``.
        lam : torch.Tensor or None
            ``(B,)`` committor logits (squashed to ``tanh(lam / L)``); None: the null token.
        uncond_mask : torch.Tensor, optional
            ``(B,)`` bool rows that use the null token.

        Returns
        -------
        torch.Tensor
            ``(B, M, 3)``, center-of-mass free per sample.
        """
        x_t = self._as_frames(x_t)
        b = x_t.shape[0]
        pos = self.from_flow(x_t).reshape(b * self.n_atoms, 3)
        batch, ptr = self._layout(b, pos.device)
        edge_index = self._edges(pos, batch)
        t = torch.as_tensor(t, dtype=pos.dtype, device=pos.device).reshape(-1).expand(b)
        c = None
        if lam is not None:
            lam = torch.as_tensor(lam, dtype=pos.dtype, device=pos.device).reshape(-1)
            c = torch.tanh(lam / self._buffers['lambda_clip'])
        if uncond_mask is not None:
            uncond_mask = torch.as_tensor(uncond_mask, device=pos.device).to(torch.bool)
        tokens = self.atom_tokens.repeat(b)
        if not self.trunk.topology_edges:
            scalars, vectors = self.trunk(pos, tokens, batch, ptr, t, c, uncond_mask,
                                          edge_index=edge_index)
            return self.velocity_head(scalars, vectors, pos, edge_index, b,
                                      self._buffers['coord_scale'])
        edge_index, topology_ids = self._topology_edges(edge_index, batch, ptr)
        features = self.trunk.edge_features(pos, batch, ptr, edge_index, topology_ids)
        scalars, vectors = self.trunk(pos, tokens, batch, ptr, t, c, uncond_mask,
                                      edge_index=edge_index, edge_features=features)
        return self.velocity_head(scalars, vectors, pos, edge_index, b,
                                  self._buffers['coord_scale'],
                                  edge_attr=features[0][:, self.trunk.rbf.n_basis:])

    # -- state dict -------------------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """``torch.nn.Module.load_state_dict`` with checks of the atom count and of the old priors.

        Parameters
        ----------
        state_dict : mapping
            A state dict of this class (also one of the flow branch, whose keys are the same).
        strict : bool, optional
            As for torch.
        assign : bool, optional
            As for torch.

        Returns
        -------
        NamedTuple
            torch's ``missing_keys`` / ``unexpected_keys``.

        Raises
        ------
        ValueError
            If the checkpoint's atom count differs from the model's, or (with
            `strict`) if it holds a prior of the flow branch (``prior.*``
            entries), which this network does not have.
        RuntimeError
            For torch's key mismatches.

        Notes
        -----
        A checkpoint of the flow branch with another prior than the Gaussian
        (``prior.*`` buffers) is refused with the reason; ``strict=False``
        loads the rest of it (trunk and heads) with a warning. Its committor
        head is usable, but its velocity was trained from another prior, so
        its generator needs training again. If ``coord_scale`` or
        ``lambda_clip`` differ from the model's by more than 1e-6 relative a
        ``UserWarning`` is issued; the checkpoint's values (buffers) are
        loaded.
        """
        for key in ('atom_tokens', 'atomic_numbers'):
            if key in state_dict and int(state_dict[key].numel()) != self.n_atoms:
                raise ValueError(f'the checkpoint is for {int(state_dict[key].numel())} atoms, '
                                 f'the model has {self.n_atoms} atoms')
        dropped = _legacy.dropped_state_keys(state_dict)
        if dropped:
            keys = sorted(key for keys in dropped.values() for key in keys)
            reasons = '; '.join(dropped)
            if strict:
                raise ValueError(f'the checkpoint holds {", ".join(keys)}: {reasons}; pass '
                                 f'strict=False to load the rest of it')
            warnings.warn(f'not loaded: {", ".join(keys)} ({reasons})', UserWarning,
                          stacklevel=2)
            state_dict = {key: value for key, value in state_dict.items() if key not in keys}
        for key in ('coord_scale', 'lambda_clip'):
            if key in state_dict:
                new, old = float(state_dict[key]), float(self._buffers[key])
                if not math.isclose(new, old, rel_tol=1e-6, abs_tol=0.0):
                    warnings.warn(f'the checkpoint has {key}={new!r}, the model was built with '
                                  f'{old!r}; the checkpoint value is used', UserWarning,
                                  stacklevel=2)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def reset_parameters(self):
        """Re-initialize every submodule as at construction (buffers unchanged).

        Linear and embedding layers get torch's default initialization (from
        the global random stream), then every module's ``reset_special`` sets
        the zero and near-zero starts of construction (FiLM, the condition
        map, the last layers of both heads, the cross-product filters, the
        null token).
        """
        for module in self.modules():
            if module is not self and isinstance(module, (nn.Linear, nn.Embedding)):
                module.reset_parameters()
        for module in self.modules():
            special = getattr(module, 'reset_special', None)
            if module is not self and callable(special):
                special()

    def default_generator(self):
        """The generation method of this network: the flow-matching ODE, ``{'method': 'ode'}``.

        Used when ``params.cgaimmd_generator`` is not set
        (:func:`aimmd.cgaimmd.generation.make_generator`); the ODE integrates
        :meth:`velocity` from :meth:`sample_prior`
        (:class:`aimmd.cgaimmd.flow_matching.sampler.ODEGenerator`).
        """
        return {'method': 'ode'}

    # -- construction from a topology ---------------------------------------------------------

    @classmethod
    def from_topology(cls, topology, descriptors, *, embedding='residue_atomname', tie='orbits',
                      chain_token=True, bond_edges=True, symmetry='topology',
                      permutation_groups=(), symmetry_max_factor_order=720,
                      symmetry_twins_only_min_atoms=None, topology_edges=False,
                      random_walk_pe=False, random_walk_steps=16, **architecture):
        """Build the model from a GROMACS topology and the descriptors of the generator atoms.

        Parameters
        ----------
        topology : str
            GROMACS ``.top`` of the system.
        descriptors : aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors
            The generator atoms (solute, cluster waters) and their structure file.
        embedding : {'residue_atomname', 'element'}, optional
            Token scheme of ``aimmd.cgaimmd.flow_matching.embedding.atom_tokens``.
        tie : {'orbits', 'none'}, optional
            Merge the tokens of atoms in one automorphism orbit.
        chain_token : bool, optional
            Include the chain (molecule) id in the residue_atomname key.
        bond_edges : bool, optional
            Append the 4-class bond one-hot to every edge.
        symmetry : {'topology', 'topology+approximate', 'groups', 'none'}, optional
            Mode of ``aimmd.cgaimmd.preparation.symmetry.ot_spec`` for the OT pairing.
        permutation_groups : sequence, optional
            Atom groups for ``symmetry='groups'``.
        symmetry_max_factor_order : int, optional
            Largest factor order enumerated exactly.
        symmetry_twins_only_min_atoms : int, optional
            Molecule size above which only twin swaps are searched.
        topology_edges : bool, optional
            Give the trunk the topology edges of the force-field geometry (``topology_geometry``).
        random_walk_pe : bool, optional
            Add the random-walk positional encodings of the bond graph to the embedding.
        random_walk_steps : int, optional
            ``K``, the walk lengths of the encodings (default 16).
        **architecture
            Every required keyword of the constructor (``coord_scale``, ``lambda_clip``, ``n_channels``, ...), plus its optional switches (``topology_cutoff``, ``topology_n_basis``, ...).

        Returns
        -------
        JointPaiNN
            An instance of `cls`.

        Raises
        ------
        TypeError
            Naming the missing architecture keywords.
        ValueError
            If the topology's generator atoms do not match the descriptors.

        Notes
        -----
        ``model_topology`` gives the records and bonds of the generator atoms
        (the first ``S`` topology atoms plus ``K`` waters of the cluster);
        ``symmetry.analyse`` its automorphism orbits and factors;
        ``embedding.atom_tokens`` the tokens; ``topological_distances`` the
        bond classes; ``symmetry.ot_spec`` the pairing (raising
        ``OTTokenMismatch`` if a permutation exchanges atoms with different
        tokens). Needs ``networkx`` (the ``cgaimmd`` extra).

        The topology edges take their lengths from
        ``aimmd.cgaimmd.preparation.topology.reference_geometry`` (the force
        field's ``b0``, constraint lengths and ``theta0``, never a frame's).
        The encodings come from the bond graph of ``model_topology``.

        The SE(3) network with element tokens and topology edges
        (``embedding='element', topology_edges=True, parity_breaking=True``)
        generated the most valid chignolin structures of the flow branch's
        campaigns; the default embedding is the flow branch's.
        """
        missing = [key for key in ARCHITECTURE_KEYS if key not in architecture]
        if missing:
            raise TypeError(f'{cls.__name__}.from_topology() missing required architecture '
                            f'keyword argument(s): {", ".join(repr(k) for k in missing)}')
        for key, value in (('topology_edges', topology_edges), ('random_walk_pe', random_walk_pe)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{key} must be a bool, got {value!r}')
        from ..preparation import symmetry as generator_symmetry
        from ..preparation import topology as generator_topology
        from . import embedding as generator_embedding
        generator_atoms = descriptors.generator_atoms
        cluster = generator_atoms.cluster
        mt = generator_topology.model_topology(
            topology, n_solute=generator_atoms.n_solute, n_waters=generator_atoms.n_waters,
            water_resname=cluster.water_resname if cluster is not None else 'SOL',
            atoms_per_water=cluster.atoms_per_water if cluster is not None else 3)
        n_atoms = len(mt.records)
        if n_atoms != descriptors.n_atoms:
            raise ValueError(f'the topology gives {n_atoms} generator atoms, the descriptors '
                             f'{descriptors.n_atoms}')
        import MDAnalysis as mda
        names = list(mda.Universe(descriptors.structure).atoms.names[:generator_atoms.n_solute])
        top_names = [record.name for record in mt.records[:generator_atoms.n_solute]]
        if names != top_names:
            bad = next(i for i, (a, b) in enumerate(zip(names, top_names)) if a != b) \
                if len(names) == len(top_names) else min(len(names), len(top_names))
            raise ValueError(f'atom names of {descriptors.structure} and {topology} differ '
                             f'(first difference at solute atom {bad})')
        report = generator_symmetry.analyse(mt, max_factor_order=symmetry_max_factor_order,
                                            twins_only_min_atoms=symmetry_twins_only_min_atoms)
        tokens = generator_embedding.atom_tokens(mt.records, embedding, tie,
                                                 orbits=report.orbits, chain_token=chain_token)
        bonds = generator_topology.topological_distances(n_atoms, mt.bonds, max_distance=4) \
            if bond_edges else None
        ot = generator_symmetry.ot_spec(report, mode=symmetry, tokens=tokens,
                                        groups=permutation_groups)
        if topology_edges:
            architecture['topology_geometry'] = generator_topology.reference_geometry(topology, mt)
        if random_walk_pe:
            architecture['positional_encoding'] = generator_topology.random_walk_encoding(
                n_atoms, mt.bonds, steps=random_walk_steps)
        return cls(n_atoms, tokens, atomic_numbers=mt.elements, bond_distances=bonds,
                   ot_symmetry=ot, **architecture)
