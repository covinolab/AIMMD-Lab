"""
aimmd.cgaimmd.bioemu2.sampling
==============================

The generation method ``'bioemu2_edm'``: BioEmu2's own EDM sampler, run on a
:class:`~aimmd.cgaimmd.bioemu2.network.BioEmu2Network`'s committor-conditioned
trunk embeddings, with classifier-free guidance.

:class:`BioEmu2Generator` implements :class:`aimmd.cgaimmd.generation.Generator`
and is registered under ``'bioemu2_edm'`` when this module is imported (the
generator registry imports it on first use). It is the default generator of a
``BioEmu2Network``. Its settings:

``n_steps`` (200)
    Steps of BioEmu2's EDM sampler (Karras schedule, churn, a random rigid
    motion per step, as BioEmu2 samples; both production campaigns used 200).
``cfg_scale`` (1.0)
    The classifier-free guidance weight against the adapters' null
    (:class:`.adapters.GuidanceSchedule`); 1 is no guidance (production).
``cfg_schedule`` (None)
    Noise-level windows with their own weight and null,
    ``'lo-hi:weight:null,...'`` (:func:`.adapters.parse_guidance_schedule`);
    outside them ``cfg_scale`` applies.
``chunk`` (16)
    Samples per sampler call (each sample is one copy of the system).

How it samples: :class:`GuidedModel` wraps the frozen model, the network's
adapters, the requested logits and the guidance schedule as a BioEmu2
``InferenceModel``, and BioEmu2's public ``EDMSampler`` samples with it, one
system copy per requested logit. Every chunk starts from
``torch.manual_seed(seed + chunk start)`` on a forked random stream (the global stream is left as it
was), with ``seed`` drawn from the ``torch.Generator`` the caller passes, so a
generation is reproducible and its chunks never share their noise.

The samples (the BioEmu2 atoms, centred) are completed to the system's
generator atoms (:meth:`.system.BioEmu2System.complete`: hydrogens, ions and
other atoms BioEmu2 does not model, from the template). A sample with
non-finite coordinates comes back as NaN, which every validity check rejects.

No copy of bioemu2 code is kept: the samples of the copy of bioemu2's
``edm_sampler`` with a denoise hook that earlier runs used are reproduced by
this module (``tests/test_cgaimmd_bioemu2_sampling.py``).
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch

from . import _api
from ..generation import _register_builtin
from .adapters import GuidanceSchedule
from .head import LAMBDA_CLIP

__all__ = ['BioEmu2Generator', 'GuidedModel', 'GuidedDenoiseOutput']


@dataclass
class GuidedDenoiseOutput:
    """What :meth:`GuidedModel.denoise` returns: the guided denoised coordinates (``score`` is not computed)."""

    denoised_coords: torch.Tensor
    score: torch.Tensor = None


class GuidedModel:
    """BioEmu2's ``InferenceModel`` protocol for one chunk of conditioned samples.

    Parameters
    ----------
    denoiser : ConditionedDenoiser
        The frozen model with the network's adapters (:mod:`.adapters`).
    lam : torch.Tensor
        ``(k,)`` requested logits, one per system copy (clipped at
        ``LAMBDA_CLIP`` by the adapters).
    guidance : GuidanceSchedule
        The classifier-free guidance at each noise level.

    Notes
    -----
    ``compute_static_context`` is the frozen model's (BioEmu2's sampler reads
    its flags); :meth:`denoise` ignores the context and trunk it is handed and
    uses the k conditioned trunks of the requested labels, and for guidance
    the null trunks, each computed once per chunk.
    """

    supports_propagation = False

    def __init__(self, denoiser, lam, guidance):
        self.denoiser = denoiser
        self.guidance = guidance
        self.lam = torch.as_tensor(lam, dtype=torch.float32, device=denoiser.device).reshape(-1)
        self._contexts = {}

    @property
    def device(self):
        return self.denoiser.device

    @property
    def edm_schedule(self):
        return self.denoiser.model.edm_schedule

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        return self.denoiser.model.compute_static_context(batch, trunk_embeddings, multiplicity)

    def _trunks(self, which):
        found = self._contexts.get(which)
        if found is None:
            k = len(self.lam)
            with torch.no_grad():
                if which == 'conditioned':
                    found = self.denoiser.conditioned(*self.denoiser.labels(self.lam, k))
                else:
                    found = self.denoiser.conditioned(*self.denoiser.labels(None, k), adapters=which == 'learned')
            self._contexts[which] = found
        return found

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        """The guided denoised coordinates of ``coords (k, n_pad, 3)`` at ``sigmas (k,)``."""
        if len(coords) != len(self.lam):
            raise ValueError(f'{len(coords)} coordinate rows for {len(self.lam)} requested logits')
        weight, null = self.guidance.at(float(sigmas.reshape(-1)[0]))
        with torch.no_grad():
            conditioned = self.denoiser.denoise_with(*self._trunks('conditioned'), coords, sigmas)
            if weight == 1.0:
                return GuidedDenoiseOutput(conditioned)
            unconditioned = self.denoiser.denoise_with(*self._trunks(null), coords, sigmas)
            return GuidedDenoiseOutput(unconditioned + weight * (conditioned - unconditioned))

    def embed(self, *args, **kwargs):
        return self.denoiser.model.embed(*args, **kwargs)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be an int >= 1, got {value!r}')
    return int(value)


class BioEmu2Generator:
    """The generator ``'bioemu2_edm'``: BioEmu2's EDM sampler with the network's adapters (module docstring).

    Parameters
    ----------
    n_steps : int, optional
        Sampler steps (>= 2).
    cfg_scale : float, optional
        Guidance weight outside the windows of `cfg_schedule` (1: none).
    cfg_schedule : str, optional
        Guidance windows ``'lo-hi:weight:null,...'``.
    chunk : int, optional
        Samples per sampler call.
    """

    method = 'bioemu2_edm'
    """The registry name of this method."""

    def __init__(self, n_steps=200, cfg_scale=1.0, cfg_schedule=None, chunk=16):
        self.n_steps = _positive_int(n_steps, 'n_steps')
        if self.n_steps < 2:
            raise ValueError(f'n_steps must be >= 2 (BioEmu2 sampler), got {n_steps!r}')
        if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, Real):
            raise ValueError(f'cfg_scale must be a number, got {cfg_scale!r}')
        self.guidance = GuidanceSchedule(float(cfg_scale), cfg_schedule)
        self.chunk = _positive_int(chunk, 'chunk')

    @property
    def cfg_scale(self):
        return self.guidance.scale

    @property
    def cfg_schedule(self):
        return self.guidance.spec()

    def __repr__(self):
        return (f'{type(self).__name__}(n_steps={self.n_steps}, cfg_scale={self.cfg_scale:g}, '
                f'cfg_schedule={self.cfg_schedule!r}, chunk={self.chunk})')

    def __eq__(self, other):
        return type(other) is type(self) and other.settings() == self.settings()

    def settings(self):
        """``{'n_steps', 'cfg_scale', 'cfg_schedule', 'chunk'}``."""
        return {'n_steps': self.n_steps, 'cfg_scale': self.cfg_scale, 'cfg_schedule': self.cfg_schedule,
                'chunk': self.chunk}

    def check(self, network):
        """What `network` lacks for this method: it must be a :class:`.network.BioEmu2Network`."""
        from .network import BioEmu2Network
        if not isinstance(network, BioEmu2Network):
            return [f"'bioemu2_edm' samples with the trunk adapters of a BioEmu2Network "
                    f'(aimmd.cgaimmd.bioemu2.BioEmu2Network); got {type(network).__name__}']
        return []

    # ------------------------------------------------------------------ sampling
    def sample_bioemu2(self, network, lam, *, seed):
        """``(B, M, 3)`` float32 BioEmu2 atoms (Ångström, centred, on the network's device) at logits `lam`.

        Chunk ``s`` (its first index) starts from ``torch.manual_seed(seed + s)``
        on a forked random stream.
        """
        engine = network.engine()
        denoiser = engine.denoiser
        lam = torch.as_tensor(lam, dtype=torch.float32, device=network.device).reshape(-1).clamp(
            -LAMBDA_CLIP, LAMBDA_CLIP)
        n_real = network.system.n_bioemu2_atoms
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        out = []
        for start in range(0, len(lam), self.chunk):
            lam_k = lam[start:start + self.chunk]
            model = GuidedModel(denoiser, lam_k, self.guidance)
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(int(seed) + start)
                coords = _api.edm_sample(model, denoiser.batch, denoiser.trunk, len(lam_k), self.n_steps)
            x = coords.reshape(len(lam_k), -1, 3)[:, :n_real].float()
            out.append(x - x.mean(dim=1, keepdim=True))
        if not out:
            return torch.zeros(0, n_real, 3, device=network.device)
        return torch.cat(out).to(network.device)

    def generate(self, network, lam, *, generator=None):
        """``(B,)`` target logits -> ``(B, N, 3)`` float32 generator atoms (Ångström) on the network's device."""
        if generator is not None:
            seed = int(torch.randint(2 ** 31 - 1, (1,), generator=generator, device=generator.device))
        else:
            seed = int(torch.randint(2 ** 31 - 1, (1,)))
        x_bioemu2 = self.sample_bioemu2(network, lam, seed=seed).cpu().numpy().astype(np.float64)
        out = np.full((len(x_bioemu2), network.n_atoms, 3), np.nan)
        good = np.isfinite(x_bioemu2).reshape(len(x_bioemu2), -1).all(axis=1)
        if good.any():
            out[good] = network.system.complete(x_bioemu2[good])
        return torch.as_tensor(out, dtype=torch.float32, device=network.device)


_register_builtin('bioemu2_edm', BioEmu2Generator)
