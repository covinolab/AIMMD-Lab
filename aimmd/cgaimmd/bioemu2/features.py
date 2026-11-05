"""
aimmd.cgaimmd.bioemu2.features
==============================

The features of frozen BioEmu2 that a C-GAIMMD committor head reads.

Convention (the one of both archived production campaigns, now for any
system):

1. **Input**: the BioEmu2 atoms of a structure (:meth:`BioEmu2Features.features`)
   or its generator atoms (:meth:`BioEmu2Features.forward`, mapped by the
   :class:`~aimmd.cgaimmd.bioemu2.system.BioEmu2System`), in Ångström, whole
   molecules, any rigid pose; BioEmu2's padding rows are allowed and cut.
2. **Pose**: centred on all BioEmu2 atoms and rotated by the rigid fit of
   the system's core atoms onto the reference pose (Kabsch, uniform
   weights; differentiable). BioEmu2 is not invariant to rotations (absolute
   positions enter linearly), so the canonical pose makes the features
   invariant to rigid motions of the input. ``align=False`` only centres.
3. **One denoiser pass** of the frozen model (float32, eval mode) at the
   noise level ``sigma_ref`` (Ångström) on the clean coordinates, or at one
   noise level per structure: the EDM preconditioning is applied by the
   model; nothing is noised.
4. **Token features**: the input ``a`` of ``structure_module.readout``,
   ``(B, n_tokens, width)`` (width 768 for bioemu2-preview, read from the
   model), captured by a forward pre-hook (:class:`._api.ReadoutCapture`).
5. **Readout**: ``'flat'`` concatenates every token's features
   (``n_tokens * width``; the chignolin campaign); ``'pooled'`` concatenates
   their means over the system's token pools, ligand, pocket and protein in
   this order (``3 * width``; protein only for a system without ligand; the
   trypsin campaign).

The trunk embeddings and the static context do not depend on the
coordinates: they are loaded (or given) and computed once.
"""

import weakref

import numpy as np
import torch

from . import _api
from .head import SIGMA_REF

__all__ = ['BioEmu2Features', 'SIGMA_REF', 'READOUTS', 'POOLS', 'canonical_pose']

READOUTS = ('flat', 'pooled')
POOLS = ('ligand', 'pocket', 'protein')
"""Order of the token pools in the pooled readout."""


def canonical_pose(x, reference_core, core):
    """Centre ``x (B, M, 3)`` on all its atoms and rotate it by the fit of ``x[:, core]`` onto `reference_core`.

    Parameters
    ----------
    x : torch.Tensor
        ``(B, M, 3)``.
    reference_core : torch.Tensor
        ``(C, 3)`` reference positions of the core atoms.
    core : torch.Tensor
        ``(C,)`` long, the core atoms of `x`.

    Returns
    -------
    torch.Tensor
        ``(B, M, 3)`` float32; the proper rotation is computed in float64 and
        is differentiable.
    """
    x_core = x.index_select(1, core)
    xc = x_core - x_core.mean(dim=1, keepdim=True)
    rc = reference_core - reference_core.mean(dim=0, keepdim=True)
    h = xc.double().transpose(1, 2) @ rc.double()
    u, _, vh = torch.linalg.svd(h)
    d = torch.sign(torch.linalg.det(u @ vh))
    diag = torch.ones(len(x), 3, dtype=h.dtype, device=h.device)
    diag[:, 2] = d
    rotation_t = u @ torch.diag_embed(diag) @ vh
    return ((x - x.mean(dim=1, keepdim=True)).double() @ rotation_t).to(torch.float32)


class BioEmu2Features(torch.nn.Module):
    """Features of frozen BioEmu2 for the committor head of a system (module docstring).

    Parameters
    ----------
    system : BioEmu2System
        The system (checked against the model's bioemu2 when this object is
        built).
    model : InferenceModel
        A frozen BioEmu2 model (``_api.load_model``). It is not registered as
        a submodule: it is never saved with the features and ``.to()`` does not
        move it.
    trunk : bioemu2.TrunkEmbeddings, optional
        Default: the system folder's ``trunk_embeddings.pt``.
    sigma_ref : float, optional
        Noise level (Ångström, > 0) of a clean structure's features.
    readout : {'flat', 'pooled'}, optional
    align : bool, optional
        Rotate onto the reference pose (default) or only centre.
    allow_version_mismatch : bool, optional
        See :meth:`BioEmu2System.check`.

    Notes
    -----
    The readout hook stays on the model until :meth:`close`. Every method is
    differentiable with respect to the coordinates; wrap calls in
    ``torch.no_grad()`` when no gradient is needed.
    """

    def __init__(self, system, model, *, trunk=None, sigma_ref=SIGMA_REF, readout='flat', align=True,
                 allow_version_mismatch=False):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}, got {readout!r}')
        if not sigma_ref > 0:
            raise ValueError(f'sigma_ref must be > 0 (the model takes log(sigma)), got {sigma_ref!r}')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('the BioEmu2 model must be frozen and in eval mode (as _api.load_model returns it)')
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'system', system)
        self.device = torch.device(model.device)
        self.sigma_ref = float(sigma_ref)
        self.readout = readout
        self.align = bool(align)
        self.topology, self.batch = _api.sampling_inputs(system.bioemu2_system(), self.device)
        system.check(self.topology, allow_version_mismatch=allow_version_mismatch)
        self.trunk = _api.load_trunk(system.trunk_path, self.device) if trunk is None else trunk
        self.n_atoms = system.n_bioemu2_atoms
        self.n_pad = int(self.batch.coords.shape[1])
        self.n_tokens = system.n_tokens
        if int(self.trunk.single.shape[1]) != self.n_tokens:
            raise ValueError(f'the trunk embeddings have {self.trunk.single.shape[1]} tokens, the system '
                             f'{self.n_tokens}: they belong to another system')
        self.token_width = _api.token_feature_width(model)
        self.context = model.compute_static_context(self.batch, self.trunk, 1)
        self.register_buffer('bioemu2_index', torch.as_tensor(system.bioemu2_to_md, dtype=torch.long,
                                                              device=self.device))
        self.register_buffer('core', torch.as_tensor(system.core, dtype=torch.long, device=self.device))
        reference = np.asarray(system.reference, dtype=np.float64)[system.core]
        self.register_buffer('reference_core', torch.as_tensor(reference - reference.mean(0), dtype=torch.float32,
                                                               device=self.device))
        pools = system.token_pools
        self.pools = tuple(name for name in POOLS if name in pools)
        for name in self.pools:
            self.register_buffer(f'pool_{name}', torch.as_tensor(pools[name], dtype=torch.long,
                                                                 device=self.device))
        self._readout = _api.ReadoutCapture(model)
        # the model is shared by every network of the process: the hook (and the tensors it last captured) goes
        # with this object, also when it is dropped without close() (a discarded network copy)
        self._remove_readout = weakref.finalize(self, self._readout.remove)

    @property
    def n_features(self):
        """Width of :meth:`features`: ``n_tokens * width`` (flat) or ``len(pools) * width`` (pooled)."""
        return (self.n_tokens if self.readout == 'flat' else len(self.pools)) * self.token_width

    def to_bioemu2(self, x):
        """Generator atoms ``(B, N, 3)`` or flat ``(B, 3N)`` -> BioEmu2 atoms ``(B, M, 3)`` float32."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, self.system.n_atoms, 3)
        return x.index_select(1, self.bioemu2_index)

    def canonical(self, x_bioemu2):
        """BioEmu2 atoms ``(B, >= M, 3)`` -> the ``M`` real atoms in the canonical pose (float32)."""
        x = torch.as_tensor(x_bioemu2, device=self.device)[:, :self.n_atoms].to(torch.float32)
        if not self.align:
            return x - x.mean(dim=1, keepdim=True)
        return canonical_pose(x, self.reference_core, self.core)

    def _sigmas(self, n, sigmas):
        if sigmas is None:
            return torch.full((n,), self.sigma_ref, dtype=torch.float32, device=self.device)
        return torch.as_tensor(sigmas, dtype=torch.float32, device=self.device).reshape(n)

    def denoise(self, x_canonical, sigmas=None):
        """One frozen denoiser pass on ``(B, M, 3)`` canonical atoms at `sigmas` (default ``sigma_ref``).

        Returns BioEmu2's ``DenoiseOutput``; the readout inputs of the pass are
        in ``self._readout``.
        """
        n = len(x_canonical)
        padded = torch.cat([x_canonical, torch.zeros(n, self.n_pad - self.n_atoms, 3, dtype=torch.float32,
                                                     device=self.device)], dim=1)
        self._readout.clear()
        return self.model.denoise(self.context, self.trunk, padded, self._sigmas(n, sigmas))

    def token_features(self, x_bioemu2, sigmas=None):
        """``(B, n_tokens, width)``: the readout's token features of BioEmu2 atoms at `sigmas`."""
        self.denoise(self.canonical(x_bioemu2), sigmas)
        return self._readout.a[:, :self.n_tokens]

    def pooled(self, tokens):
        """``(B, n_tokens, width)`` -> ``(B, len(pools) * width)``: the means over the token pools."""
        return torch.cat([tokens.index_select(1, getattr(self, f'pool_{name}')).float().mean(1)
                          for name in self.pools], dim=1)

    def features(self, x_bioemu2, sigmas=None):
        """``(B, n_features)`` features of BioEmu2 atoms ``(B, >= M, 3)`` at `sigmas` (default ``sigma_ref``)."""
        tokens = self.token_features(x_bioemu2, sigmas)
        if self.readout == 'flat':
            return tokens.reshape(len(tokens), -1)
        return self.pooled(tokens)

    def forward(self, x, sigmas=None):
        """``(B, n_features)`` features of generator atoms (``(B, N, 3)`` or flat) at `sigmas`."""
        return self.features(self.to_bioemu2(x), sigmas)

    def identity(self):
        """What features made by this object depend on (for metadata and sidecars)."""
        return dict(bioemu2_version=_api.bioemu2_version(), bioemu2_file=_api.bioemu2.__file__,
                    precision='32-true', matmul_precision=None, sigma_ref_angstrom=self.sigma_ref,
                    readout=self.readout, pools=list(self.pools), align=self.align,
                    system=str(self.system.folder) if self.system.folder is not None else None,
                    trunk=self.system.data.get('trunk'), n_tokens=self.n_tokens, token_width=self.token_width,
                    n_features=self.n_features, n_core_atoms=int(len(self.core)),
                    layer='structure_module.readout input a (token features)',
                    n_parameters=int(sum(p.numel() for p in self.model.parameters())),
                    parameter_checksum=_api.parameter_checksum(self.model))

    def close(self):
        """Remove the readout hook from the model (done anyway when this object is garbage-collected)."""
        self._remove_readout()
