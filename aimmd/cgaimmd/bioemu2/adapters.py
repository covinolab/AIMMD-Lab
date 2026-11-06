"""
aimmd.cgaimmd.bioemu2.adapters
==============================

How a frozen BioEmu2 is conditioned on a target committor: small adapters on
its trunk embeddings, and classifier-free guidance.

**Trunk adapters.** BioEmu2's structure module reads the sequence trunk's
single embedding ``s`` (per token) and pair embedding ``z`` (per token pair).
A :class:`TrunkAdapter` adds a committor-dependent correction,

``s_i' = s_i + W2 SiLU(W1 [LayerNorm(s_i), e(lambda)])``

with ``e(lambda)`` an embedding of the target logit committor (clipped at
:data:`~.head.LAMBDA_CLIP`). ``W2`` starts at zero, so a fresh adapter leaves
BioEmu2 exactly as it is. The pair adapter does the same for ``z``. Every
weight of BioEmu2 stays frozen: the generator is BioEmu2 seen through
committor-conditioned trunk embeddings, the analogue of conditioning a frozen
image diffusion model through its text embeddings.

**One copy of the system per sample.** BioEmu2 samples one structure per
system copy; with an adapter every sample has its own conditioned trunk, so k
samples are k copies of the system, and BioEmu2's static context is rebuilt
from the k conditioned trunks (:class:`ConditionedDenoiser`).

**No label.** A row without a label keeps the trunk as it is (plain BioEmu2),
unless the adapters have a *learned null*: then such a row gets a learned
null embedding instead of ``e(lambda)`` and is adapted too. Training with a
label dropout ``p_uncond`` (:func:`aimmd.cgaimmd.bioemu2.fit.sequential_round`)
teaches the learned null; without a learned null the dropped rows give no
gradient and only waste their share of the batch.

**Classifier-free guidance** (:class:`GuidanceSchedule`). At every noise level
``sigma`` the sampler mixes the conditioned and a null denoiser output,
``D = D_null + w (D_lambda - D_null)``. ``w = 1`` (the default, as in both
production campaigns) is the conditioned model alone. The weight can depend on
the noise level (windows ``'lo-hi:weight:null'``), and the null is the
adapters' own null (``'learned'``; plain BioEmu2 without a learned null) or
plain BioEmu2 (``'bioemu2'``, adapters off).

The parameter names of :class:`TrunkAdapter` are those of earlier BioEmu2
C-GAIMMD runs, so their adapters load unchanged.
"""

# Provenance: Ported from ``bioemu2_committor/finetune.py`` (``TrunkAdapter``,
# ``lambda_features``, the trunk kinds of ``ConditionedDenoiser``; LoRA and the
# token-conditioning embedding were dropped) and the guidance schedule of the
# generator-adherence campaign (``adherence_lib.parse_guidance_schedule``,
# ``null_denoise``, ``guidance_hook``).

import math
from numbers import Real

import numpy as np
import torch
from torch import nn

from .head import LAMBDA_CLIP

__all__ = ['LAMBDA_CLIP', 'GUIDANCE_NULLS', 'lambda_features', 'TrunkAdapter', 'ConditionedDenoiser',
           'GuidanceSchedule', 'parse_guidance_schedule']

GUIDANCE_NULLS = ('learned', 'bioemu2')
"""The nulls of classifier-free guidance: the adapters' own null, or plain BioEmu2 (adapters off)."""


def lambda_features(lam, freq):
    """``(k, 1 + 2 F)``: ``[lambda / 4, sin(lambda f / 4), cos(lambda f / 4)]`` of logits clipped at +-LAMBDA_CLIP."""
    lam = lam.clamp(-LAMBDA_CLIP, LAMBDA_CLIP)[:, None]
    return torch.cat([lam / 4, torch.sin(lam * freq / 4), torch.cos(lam * freq / 4)], dim=1)


class TrunkAdapter(nn.Module):
    """``x' = x + W2 SiLU(W1 [LayerNorm(x), e(lambda)])`` per token (single) or token pair (pair).

    Parameters
    ----------
    dim : int
        Width of the trunk embedding it adapts (read from the trunk).
    hidden : int, optional
        Hidden width (production: 128 single, 64 pair).
    n_freq : int, optional
        Frequencies of the committor features.
    e_dim : int, optional
        Width of the committor embedding ``e(lambda)``.
    learned_null : bool, optional
        Rows without a label get a learned null embedding and are adapted
        too; otherwise they stay unchanged.

    Notes
    -----
    ``W2`` (``out``) is zero after construction and after
    :meth:`reset_parameters`: the adapter is then the identity.
    """

    def __init__(self, dim, hidden=128, n_freq=8, e_dim=32, learned_null=False):
        super().__init__()
        self.register_buffer('freq', torch.pi * torch.logspace(-1, 0.5, int(n_freq)))
        self.lam_embed = nn.Sequential(nn.Linear(1 + 2 * int(n_freq), int(e_dim)), nn.SiLU())
        self.norm = nn.LayerNorm(int(dim))
        self.inp = nn.Linear(int(dim) + int(e_dim), int(hidden))
        self.out = nn.Linear(int(hidden), int(dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.learned_null = bool(learned_null)
        if self.learned_null:
            self.null = nn.Parameter(torch.zeros(int(e_dim)))

    def settings(self):
        """The constructor arguments (``dim``, ``hidden``, ``n_freq``, ``e_dim``, ``learned_null``)."""
        return dict(dim=self.norm.normalized_shape[0], hidden=self.inp.out_features, n_freq=len(self.freq),
                    e_dim=self.lam_embed[0].out_features, learned_null=self.learned_null)

    def forward(self, x, lam, keep):
        """``x (k, ..., dim)``, ``lam (k,)`` logits, ``keep (k,)`` bool (rows that carry a label) -> ``x'``."""
        e = self.lam_embed(lambda_features(lam.to(x.dtype), self.freq))          # (k, e_dim)
        if self.learned_null:
            e = torch.where(keep.view(-1, 1), e, self.null.to(e.dtype).expand_as(e))
        e = e.view(e.shape[0], *([1] * (x.ndim - 2)), e.shape[1]).expand(*x.shape[:-1], e.shape[1])
        d = self.out(torch.nn.functional.silu(self.inp(torch.cat([self.norm(x), e], dim=-1))))
        if self.learned_null:
            return x + d
        return x + d * keep.view(-1, *([1] * (x.ndim - 1))).to(d.dtype)

    def reset_parameters(self, seed=None):
        """Fresh weights with a zero output layer (the identity), the learned null at zero.

        With `seed`, the weights are those of a new adapter built after
        ``torch.manual_seed(seed)`` on a forked CPU random stream (the global
        stream is left as it was): the old ``fresh_adapter``. Without, they
        are drawn from the global stream, as ``nn.Linear`` draws them.
        """
        if seed is None:
            state = TrunkAdapter(**self.settings()).state_dict()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                state = TrunkAdapter(**self.settings()).state_dict()
        self.load_state_dict(state)


class ConditionedDenoiser:
    """Frozen BioEmu2 whose trunk embeddings are conditioned on target committors by trunk adapters.

    Parameters
    ----------
    model : InferenceModel
        A frozen BioEmu2 model.
    trunk : bioemu2.TrunkEmbeddings
        The trunk embeddings of one system (float32).
    batch : BatchAllAtomStructure
        BioEmu2's sampling batch of that system (one copy).
    single_adapter : TrunkAdapter
        The adapter of the single embedding (the network's module: training
        it changes what this object samples).
    pair_adapter : TrunkAdapter, optional
        The adapter of the pair embedding.

    Notes
    -----
    Not a module: it holds references, nothing of it is saved. The static
    context is computed with gradients (``_api.static_context_with_grad``),
    so a pair adapter, which reaches the denoiser only through it, is trained.
    """

    def __init__(self, model, trunk, batch, single_adapter, pair_adapter=None):
        from . import _api
        self._api = _api
        self.model = model
        self.trunk = trunk
        self.batch = batch
        self.single_adapter = single_adapter
        self.pair_adapter = pair_adapter

    @property
    def device(self):
        return torch.device(self.model.device)

    @property
    def n_pad(self):
        """Atoms per sample of BioEmu2's coordinates (the real atoms first, then padding)."""
        return int(self.batch.coords.shape[1])

    def labels(self, lam, k, uncond=None):
        """``(lam (k,) float32, keep (k,) bool)``: `lam` None means no label for any row."""
        device = self.device
        if lam is None:
            return torch.zeros(k, device=device), torch.zeros(k, dtype=torch.bool, device=device)
        lam = torch.as_tensor(lam, dtype=torch.float32, device=device).reshape(k)
        keep = (torch.ones(k, dtype=torch.bool, device=device) if uncond is None
                else ~torch.as_tensor(uncond, dtype=torch.bool, device=device).reshape(k))
        return lam, keep

    def conditioned(self, lam, keep, *, adapters=True):
        """``(trunk_k, context_k)``: k copies of the system with conditioned trunks and their static context.

        ``adapters=False`` gives k copies of the plain trunk (the null
        ``'bioemu2'`` of guidance).
        """
        t, k = self.trunk, len(lam)

        def rep(value):
            return value.expand(k, *value.shape[1:])
        if adapters:
            single = self.single_adapter(rep(t.single).float(), lam, keep)
            pair = (self.pair_adapter(rep(t.pair).float(), lam, keep) if self.pair_adapter is not None
                    else rep(t.pair))
        else:
            single = rep(t.single).float()
            pair = rep(t.pair).float() if self.pair_adapter is not None else rep(t.pair)
        trunk_k = self._api.TrunkEmbeddings(s_inputs=rep(t.s_inputs), single=single, pair=pair,
                                            relative_position_encoding=rep(t.relative_position_encoding),
                                            distogram=rep(t.distogram))
        context_k = self._api.static_context_with_grad(self.model, self._api.expand_batch(self.batch, k), trunk_k)
        return trunk_k, context_k

    def denoise_with(self, trunk_k, context_k, x, sigma):
        """BioEmu2's denoised coordinates of ``x (k, n_pad, 3)`` at `sigma` with conditioned trunks."""
        sigma = self._sigmas(sigma, len(x), x.device)
        return self.model.denoise(context=context_k, trunk_embeddings=trunk_k, coords=x, sigmas=sigma).denoised_coords

    @staticmethod
    def _sigmas(sigma, k, device):
        return sigma if torch.is_tensor(sigma) else torch.full((k,), float(sigma), device=device)

    def denoise(self, x, sigma, lam=None, uncond=None):
        """One pass of the conditioned model on ``x (k, n_pad, 3)`` at `sigma` (float or ``(k,)``).

        Rows in `uncond`, and every row for ``lam=None``, get no label (the
        adapters' null). Differentiable with respect to the adapters.
        """
        lam, keep = self.labels(lam, len(x), uncond)
        trunk_k, context_k = self.conditioned(lam, keep)
        return self.denoise_with(trunk_k, context_k, x, sigma)

    def plain_denoise(self, x, sigma):
        """One pass of plain BioEmu2 (adapters off) on ``x (k, n_pad, 3)``, through the same k-copy route."""
        lam, keep = self.labels(None, len(x))
        trunk_k, context_k = self.conditioned(lam, keep, adapters=False)
        return self.denoise_with(trunk_k, context_k, x, sigma)


# ---------------------------------------------------------------------------
# classifier-free guidance
# ---------------------------------------------------------------------------

def _decimal(value):
    """`value` as the shortest decimal that reads back as the same float, never with an exponent.

    The schedule text separates ``lo`` and ``hi`` by ``'-'``, so ``'1e-05'``
    would not read back; ``inf`` stays ``'inf'``.
    """
    return np.format_float_positional(float(value), trim='-')


def _number(text, what, spec):
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f'guidance schedule {spec!r}: {what} {text!r} is not a number') from None
    return value


def parse_guidance_schedule(spec):
    """``'lo-hi:weight:null,...'`` -> ``((lo, hi, weight, null), ...)``.

    Each window applies the guidance weight `weight` against `null`
    (:data:`GUIDANCE_NULLS`) at noise levels ``lo <= sigma <= hi``
    (Ångström; `hi` may be ``inf``). The first window that contains a noise
    level decides; outside every window :class:`GuidanceSchedule` uses its
    ``scale`` against the learned null.

    Examples
    --------
    >>> parse_guidance_schedule('8-inf:3:learned,2-8:3:bioemu2')
    ((8.0, inf, 3.0, 'learned'), (2.0, 8.0, 3.0, 'bioemu2'))
    """
    if spec is None:
        return ()
    if not isinstance(spec, str):                         # ((lo, hi, weight, null), ...)
        try:
            spec = ','.join(f'{_decimal(lo)}-{_decimal(hi)}:{_decimal(w)}:{null}' for lo, hi, w, null in spec)
        except (TypeError, ValueError):
            raise ValueError(f'a guidance schedule is a string or a sequence of (lo, hi, weight, null), '
                             f'got {spec!r}') from None
    out = []
    for part in [p.strip() for p in spec.split(',') if p.strip()]:
        fields = part.split(':')
        if len(fields) != 3 or fields[0].count('-') != 1:
            raise ValueError(f"guidance schedule {spec!r}: window {part!r} is not 'lo-hi:weight:null'")
        lo_text, hi_text = fields[0].split('-')
        lo, hi = _number(lo_text, 'noise level', spec), _number(hi_text, 'noise level', spec)
        weight = _number(fields[1], 'weight', spec)
        null = fields[2].strip()
        if null not in GUIDANCE_NULLS:
            raise ValueError(f'guidance schedule {spec!r}: unknown null {null!r}, expected one of {GUIDANCE_NULLS}')
        if not (0 <= lo <= hi) or math.isnan(hi):
            raise ValueError(f'guidance schedule {spec!r}: window {part!r} needs 0 <= lo <= hi')
        if not math.isfinite(weight):
            raise ValueError(f'guidance schedule {spec!r}: the weight of {part!r} must be finite')
        out.append((lo, hi, weight, null))
    return tuple(out)


class GuidanceSchedule:
    """Classifier-free guidance weight and null at each noise level of the sampler.

    Parameters
    ----------
    scale : float, optional
        The weight outside every window, against the learned null (1: no
        guidance, the production setting).
    windows : str or sequence, optional
        Noise-level windows with their own weight and null
        (:func:`parse_guidance_schedule`).

    Notes
    -----
    ``D = D_null + w (D_lambda - D_null)`` at noise level ``sigma``, with
    ``(w, null) = at(sigma)``. With ``w = 1`` the null pass is skipped.
    """

    def __init__(self, scale=1.0, windows=None):
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
            raise ValueError(f'the guidance scale must be a finite number, got {scale!r}')
        self.scale = float(scale)
        self.windows = parse_guidance_schedule(windows)

    def at(self, sigma):
        """``(weight, null)`` at noise level `sigma` (Ångström)."""
        sigma = float(sigma)
        for lo, hi, weight, null in self.windows:
            if lo <= sigma <= hi:
                return weight, null
        return self.scale, 'learned'

    @property
    def is_off(self):
        """True when every noise level has weight 1 (the conditioned model alone)."""
        return self.scale == 1.0 and all(w == 1.0 for _, _, w, _ in self.windows)

    def spec(self):
        """The windows as a canonical string that reads back exactly (None without windows)."""
        if not self.windows:
            return None
        return ','.join(f'{_decimal(lo)}-{_decimal(hi)}:{_decimal(w)}:{null}' for lo, hi, w, null in self.windows)

    def __repr__(self):
        return f'GuidanceSchedule(scale={self.scale:g}, windows={self.spec()!r})'
