"""
aimmd.cgaimmd.flow_matching.flowmath
====================================

The mathematics of equivariant conditional flow matching: centering and the
center-of-mass-free Gaussian prior, the optimal-transport pairing of a data
sample with a prior sample (proper Kabsch rotation, permutations of
interchangeable atoms, whole molecules), the straight-line probability path
and the flow-time schedules, and Haar-random rotations.

Nothing here knows about molecules, AIMMD or the committor: functions take
tensors of shape ``(..., M, 3)`` (``M`` atoms) and return tensors. Every random
draw takes a ``generator``, so that a trainer that owns its own random stream
(the flow-matching fit, end-state pretraining) draws nothing from the global one.

Ported from the flow-matching prototype; the ODE integrators are
:mod:`aimmd.cgaimmd.flow_matching.sampler`. ``scipy.optimize`` is imported only
inside :func:`permutation_align` and the whole-molecule pairing.
"""

import math

import torch


# --------------------------------------------------------------------------------------------
# center-of-mass-free geometry
# --------------------------------------------------------------------------------------------

def center(x, weights=None):
    """Remove the (optionally weighted) centroid along the atom axis.

    Parameters
    ----------
    x : torch.Tensor
        ``(..., M, 3)`` coordinates.
    weights : torch.Tensor, optional
        ``(M,)`` unnormalized weights (e.g. masses); None gives the geometric centroid.

    Returns
    -------
    torch.Tensor
        ``x`` minus its centroid, same shape and dtype.

    Notes
    -----
    Flow code uses the geometric centroid: the prior is isotropic in
    unweighted coordinates, so the subspace it lives in is the unweighted one.
    """
    if weights is None:
        return x - x.mean(dim=-2, keepdim=True)
    w = weights.to(x.dtype).reshape(*([1] * (x.dim() - 2)), -1, 1)
    return x - (x * w).sum(dim=-2, keepdim=True) / w.sum(dim=-2, keepdim=True)


def project_com_free(v):
    """Project displacements ``(..., M, 3)`` onto the zero-net-translation subspace."""
    return v - v.mean(dim=-2, keepdim=True)


def gaussian_prior(shape, *, device=None, dtype=None, generator=None):
    """Standard normal samples projected onto the center-of-mass-free subspace.

    Parameters
    ----------
    shape : tuple of int
        ``(n, M, 3)`` (any leading dimensions).
    device, dtype : optional
        Of the result; dtype None is float32.
    generator : torch.Generator, optional
        The random stream (on `device`); None uses the global one.

    Returns
    -------
    torch.Tensor
        The projected samples, flow units.

    Notes
    -----
    This is not a standard normal in ``R^{3M}``: it has 3 fewer degrees of
    freedom, so every coordinate has variance ``(M - 1) / M``. The pairing and
    the flow-matching loss use only differences inside the subspace.
    """
    dtype = torch.float32 if dtype is None else dtype
    z = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    return project_com_free(z)


def random_rotations(n, *, device=None, dtype=None, generator=None):
    """``(n, 3, 3)`` rotation matrices, uniform (Haar) on SO(3), determinant +1.

    Parameters
    ----------
    n : int
        Number of rotations.
    device, dtype : optional
        Of the result; dtype None is float32.
    generator : torch.Generator, optional
        The random stream.

    Returns
    -------
    torch.Tensor
        Proper rotations; never a reflection.

    Notes
    -----
    A normalized 4-D standard normal is a uniform unit quaternion, and the
    quaternion-to-matrix map pushes the uniform measure on the 3-sphere to
    the Haar measure on SO(3).
    """
    dtype = torch.float32 if dtype is None else dtype
    q = torch.randn(n, 4, device=device, dtype=dtype, generator=generator)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


# --------------------------------------------------------------------------------------------
# optimal-transport pairing
# --------------------------------------------------------------------------------------------

def kabsch_align(mobile, target, weights=None):
    """Rotate `mobile` onto `target` by the proper rotation of least squared deviation.

    Parameters
    ----------
    mobile, target : torch.Tensor
        ``(..., M, 3)``, both already centered; batched over leading dimensions.
    weights : torch.Tensor, optional
        ``(M,)`` weights of the squared deviation.

    Returns
    -------
    torch.Tensor
        The rotated `mobile`.

    Notes
    -----
    The reflection guard (flipping the least significant singular direction
    when the optimum would be improper) is essential: an improper
    "alignment" would map one enantiomer onto the other.
    """
    if weights is None:
        cov = mobile.transpose(-1, -2) @ target
    else:
        w = weights.to(mobile.dtype).reshape(*([1] * (mobile.dim() - 2)), -1, 1)
        cov = (mobile * w).transpose(-1, -2) @ target
    u, _, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)))
    flip = torch.ones_like(u)
    flip[..., :, -1] = d.unsqueeze(-1)
    rot = (vh.transpose(-1, -2) @ (u * flip).transpose(-1, -2)).transpose(-1, -2)
    return mobile @ rot


def _assignment(cost):
    """Row permutation of a ``(k, k)`` numpy cost matrix: slot ``j`` takes old slot ``inv[j]``."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    inv = torch.empty(len(rows), dtype=torch.long)
    inv[torch.as_tensor(cols)] = torch.as_tensor(rows)
    return inv


def permutation_align(mobile, target, groups):
    """Permute interchangeable atoms within each group to minimize the squared deviation.

    Parameters
    ----------
    mobile, target : torch.Tensor
        ``(..., M, 3)`` coordinates.
    groups : sequence of sequence of int
        Atom index lists; each group is permuted independently (Hungarian algorithm).

    Returns
    -------
    torch.Tensor
        `mobile` with every group relabelled; the other atoms unchanged.
    """
    if not groups:
        return mobile
    out = mobile.clone()
    flat = mobile.reshape(-1, mobile.shape[-2], 3)
    flat_target = target.reshape(-1, target.shape[-2], 3)
    flat_out = out.reshape(-1, out.shape[-2], 3)
    for group in groups:
        idx = torch.as_tensor(list(group), device=mobile.device, dtype=torch.long)
        a = flat[:, idx, :]
        b = flat_target[:, idx, :]
        cost = torch.cdist(a, b).pow(2).detach().cpu().numpy()
        perm = torch.stack([_assignment(cost[n]) for n in range(cost.shape[0])]).to(mobile.device)
        flat_out[:, idx, :] = torch.gather(a, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
    return flat_out.reshape_as(mobile)


def _molecule_permutation_align(mobile, target, blocks, anchor=0):
    """Permute whole molecules (blocks moved together), assigned by their anchor atoms."""
    if len(blocks) < 2:
        return mobile
    block_idx = torch.as_tensor([list(b) for b in blocks], device=mobile.device, dtype=torch.long)
    flat = mobile.reshape(-1, mobile.shape[-2], 3)
    flat_target = target.reshape(-1, target.shape[-2], 3)
    out = flat.clone()
    a = flat[:, block_idx[:, anchor], :]
    b = flat_target[:, block_idx[:, anchor], :]
    cost = torch.cdist(a, b).pow(2).detach().cpu().numpy()
    for n in range(cost.shape[0]):
        src = block_idx[_assignment(cost[n]).to(mobile.device)]
        out[n, block_idx.reshape(-1), :] = flat[n, src.reshape(-1), :]
    return out.reshape_as(mobile)


class PermutationFactors:
    """Finite permutation groups with disjoint supports: the direct factors of a symmetry group.

    Parameters
    ----------
    factors : sequence of tuple
        ``(support, elements)`` pairs: ``support`` lists the ``k`` atoms a factor moves; ``elements`` its group members as length-``k`` position tuples.

    Raises
    ------
    ValueError
        For an empty or repeating support, overlapping supports, an element that is no permutation, or no identity.

    Notes
    -----
    Element ``e`` sends the coordinates of atom ``support[e[j]]`` to slot
    ``support[j]``. The elements must form a group (closed under inversion,
    so the direction convention does not matter); only the presence of the
    identity and that every element is a permutation are checked. The
    identity is put first, so that exact cost ties keep the input labelling.
    Index tensors are built once and cached per device, so
    :func:`factor_align` does no host-to-device copy inside a training step.
    ``aimmd.cgaimmd.preparation.symmetry.OTSpec.factors`` holds the plain tuples.
    """

    def __init__(self, factors=()):
        items = []
        used = set()
        for support, elements in factors:
            s = tuple(int(i) for i in support)
            k = len(s)
            if k == 0 or len(set(s)) != k:
                raise ValueError(f'factor support must be non-empty and without repeats, got {s}')
            if used & set(s):
                raise ValueError(f'factor supports must be disjoint; {sorted(used & set(s))} '
                                 f'appear twice')
            used |= set(s)
            elems = [tuple(int(v) for v in e) for e in elements]
            ident = tuple(range(k))
            for e in elems:
                if len(e) != k or sorted(e) != list(ident):
                    raise ValueError(f'element {e} is not a permutation of the {k} positions of '
                                     f'support {s}')
            if ident not in elems:
                raise ValueError(f'the elements of support {s} do not contain the identity')
            elems = [ident] + [e for e in dict.fromkeys(elems) if e != ident]
            items.append((s, elems))
        self.factors = items
        buckets = {}
        for s, elems in items:
            buckets.setdefault((len(s), len(elems)), []).append((s, elems))
        self._host = []
        for _, bucket in sorted(buckets.items()):
            slots = torch.tensor([s for s, _ in bucket], dtype=torch.long)            # (g, k)
            positions = torch.tensor([e for _, e in bucket], dtype=torch.long)        # (g, p, k)
            sources = torch.gather(slots.unsqueeze(1).expand(-1, positions.shape[1], -1), 2,
                                   positions)                                         # (g, p, k)
            self._host.append((slots, sources))
        self._device = {}

    def __len__(self):
        return len(self.factors)

    @property
    def order(self):
        """Order of the product group."""
        return int(math.prod(len(e) for _, e in self.factors))

    @property
    def n_candidates(self):
        """Candidate evaluations per :func:`factor_align` call (the sum of the factor orders)."""
        return int(sum(len(e) for _, e in self.factors))

    def on(self, device):
        """The index tensors on `device` (cached)."""
        key = torch.device(device)
        key = (key.type, key.index)
        if key not in self._device:
            self._device[key] = [(s.to(device), g.to(device)) for s, g in self._host]
        return self._device[key]

    def __repr__(self):
        return (f'PermutationFactors({len(self)} factors, order {self.order}, '
                f'{self.n_candidates} candidates)')


def factor_align(mobile, target, factors):
    """Apply, factor by factor, the group element that minimizes the squared deviation.

    Parameters
    ----------
    mobile, target : torch.Tensor
        ``(..., M, 3)`` coordinates.
    factors : PermutationFactors or sequence or None
        The factors (plain ``(support, elements)`` tuples are converted); None or empty: identity.

    Returns
    -------
    torch.Tensor
        `mobile` relabelled.

    Notes
    -----
    Exact: for a direct product of groups with disjoint supports the cost is
    a sum of per-factor costs, so the best element of the product is the
    product of the per-factor best elements, each found by enumeration.
    Batched over samples and over factors of equal (support size, order),
    with device-side gathers only. For a symmetric group on ``k`` atoms this
    is the Hungarian assignment of :func:`permutation_align` (they can differ
    only on exact ties).
    """
    if factors is None:
        return mobile
    if not isinstance(factors, PermutationFactors):
        factors = PermutationFactors(factors)
    if len(factors) == 0:
        return mobile
    m = mobile.shape[-2]
    flat = mobile.reshape(-1, m, 3)
    flat_target = target.reshape(-1, m, 3)
    out = flat.clone()
    b = flat.shape[0]
    for slots, sources in factors.on(mobile.device):
        g, p, k = sources.shape
        cand = flat[:, sources, :]                                          # (B, g, p, k, 3)
        tgt = flat_target[:, slots, :]                                      # (B, g, k, 3)
        cost = (cand - tgt.unsqueeze(2)).pow(2).sum(dim=(-1, -2))           # (B, g, p)
        best = cost.argmin(dim=-1)                                          # (B, g)
        sel = torch.gather(cand, 2, best[:, :, None, None, None].expand(b, g, 1, k, 3)).squeeze(2)
        out[:, slots.reshape(-1), :] = sel.reshape(b, g * k, 3)
    return out.reshape_as(mobile)


def ot_pair(x1, x0, *, groups=(), factors=(), molecule_blocks=(), rotate=True, n_rounds=2):
    """Act on the data sample `x1` with the symmetry element that brings it closest to `x0`.

    Parameters
    ----------
    x1 : torch.Tensor
        ``(..., M, 3)`` data samples (flow units).
    x0 : torch.Tensor
        ``(..., M, 3)`` prior samples; not changed.
    groups : sequence of sequence of int, optional
        Hungarian groups (orbits too large to enumerate).
    factors : PermutationFactors or sequence, optional
        Disjoint-support factors enumerated exactly (:func:`factor_align`).
    molecule_blocks : sequence of sequence of int, optional
        Identical whole molecules (e.g. waters), permuted as blocks by their first atom.
    rotate : bool, optional
        Include the proper Kabsch rotation.
    n_rounds : int, optional
        Rotation and permutations do not commute, so they are alternated this many times.

    Returns
    -------
    torch.Tensor
        The transformed, centered `x1`.

    Notes
    -----
    Both samples are centered first. Each round applies, in this order, the
    rotation, the molecule blocks, the factors and the Hungarian groups; with
    nothing to permute one round is done (a rotation alone is idempotent).
    Alternating never increases the cost. Two rounds were enough for NANMA
    in the prototype (the second changes the cost by under 1 %).
    """
    x1 = center(x1)
    x0 = center(x0)
    if factors is not None and not isinstance(factors, PermutationFactors):
        factors = PermutationFactors(factors)
    has_factors = factors is not None and len(factors) > 0
    permutes = has_factors or bool(groups) or bool(molecule_blocks)
    for _ in range(max(1, int(n_rounds))):
        if rotate:
            x1 = kabsch_align(x1, x0)
        if molecule_blocks:
            x1 = _molecule_permutation_align(x1, x0, molecule_blocks)
        if has_factors:
            x1 = factor_align(x1, x0, factors)
        if groups:
            x1 = permutation_align(x1, x0, groups)
        if not permutes:
            break
    return x1


# --------------------------------------------------------------------------------------------
# probability path and flow time
# --------------------------------------------------------------------------------------------

def interpolate(x0, x1, t):
    """The conditional optimal-transport path and its velocity.

    Parameters
    ----------
    x0, x1 : torch.Tensor
        ``(B, M, 3)`` prior and data samples.
    t : torch.Tensor
        ``(B,)`` flow times.

    Returns
    -------
    tuple of torch.Tensor
        ``x_t = (1 - t) x0 + t x1`` and the target velocity ``x1 - x0``.
    """
    tt = t.reshape(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - tt) * x0 + tt * x1, x1 - x0


def parse_time_schedule(schedule):
    """Parse a flow-time schedule without drawing random numbers.

    Parameters
    ----------
    schedule : str
        ``'uniform'``, ``'power:k'`` (k > 0) or ``'logitnormal:m,s'`` (s > 0).

    Returns
    -------
    tuple
        ``(kind, parameters)``.

    Raises
    ------
    ValueError
        For anything else.
    """
    kind, _, arg = str(schedule).partition(':')
    try:
        if kind == 'uniform' and not arg:
            return 'uniform', ()
        if kind == 'power':
            k = float(arg)
            if k > 0 and math.isfinite(k):
                return 'power', (k,)
        if kind == 'logitnormal':
            m, s = (float(v) for v in arg.split(','))
            if s > 0 and math.isfinite(m) and math.isfinite(s):
                return 'logitnormal', (m, s)
    except ValueError:
        pass
    raise ValueError(f'invalid time schedule {schedule!r} (uniform | power:k with k > 0 | '
                     f'logitnormal:m,s with s > 0)')


def sample_time(n, schedule='uniform', *, device=None, generator=None):
    """Draw the ``(n,)`` flow times of a flow-matching batch.

    Parameters
    ----------
    n : int
        Number of samples.
    schedule : str, optional
        See :func:`parse_time_schedule`.
    device : optional
        Of the result.
    generator : torch.Generator, optional
        The random stream.

    Returns
    -------
    torch.Tensor
        ``(n,)`` float32 times in ``[0, 1]``.

    Notes
    -----
    - ``'uniform'``: ``torch.rand(n)``.
    - ``'power:k'``: ``u ** (1 / k)``, i.e. Beta(k, 1) with density
      ``k t^(k-1)``; k > 1 weights training toward the data end.
    - ``'logitnormal:m,s'``: ``sigmoid(m + s z)``, z standard normal.

    Changing the time distribution changes only how the loss is weighted over
    ``t``, not the minimizer (the marginal velocity field).
    """
    kind, parameters = parse_time_schedule(schedule)
    if kind == 'uniform':
        return torch.rand(n, device=device, generator=generator)
    if kind == 'power':
        return torch.rand(n, device=device, generator=generator).pow(1.0 / parameters[0])
    m, s = parameters
    return torch.sigmoid(m + s * torch.randn(n, device=device, generator=generator))
