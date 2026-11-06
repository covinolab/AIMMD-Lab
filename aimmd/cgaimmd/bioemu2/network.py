"""
aimmd.cgaimmd.bioemu2.network
=============================

:class:`BioEmu2Network`: the C-GAIMMD network of a BioEmu2 run. It is a
committor model (AIMMD's network contract and the C-GAIMMD network protocol
of :mod:`aimmd.cgaimmd.protocol`) that also carries the trunk adapters the
generator ``'bioemu2_edm'`` (:mod:`.sampling`) conditions BioEmu2 with.

What AIMMD sees:

- ``forward(descriptors)``: the logit committor toward ``sorted_states[2]``,
  ``(B, 1)``, of the EMA head on the frozen BioEmu2 features of the structures
  (descriptors = the flat Ångström coordinates of the system's generator
  atoms, ``(B, 3 N)``);
- ``n_atoms`` (the generator atoms ``N``), ``lambda_clip``, ``reset_parameters``,
  ``default_generator()`` -> ``{'method': 'bioemu2_edm'}`` and ``provenance()``
  (the checkpoint, its Hugging Face snapshot and the parameter checksum of the
  frozen model, which every sidecar records);
- ``state_dict()``: the head, its EMA copy and the adapters. Frozen BioEmu2 is
  never saved: every process loads it once (offline on compute nodes), from
  the checkpoint the system folder records unless another is named.

It has no ``velocity``, ``sample_prior`` or ``generate``: BioEmu2 is a
diffusion model, and generation is the generator's business
(:mod:`aimmd.cgaimmd.generation`).

The network is built from a system folder (``python -m aimmd.cgaimmd.bioemu2
build-system``) without loading BioEmu2: the widths of the head and the
adapters are read from the folder. BioEmu2 is loaded on first use, on the
network's device, and shared by every network of the process that uses the
same checkpoint on the same device. Without bioemu2 the network can still be
built, loaded and saved (to analyse a run elsewhere); its first real use raises
the ImportError that says how to install bioemu2.

:func:`load_archived_network` reads the networks of the archived production
campaigns (``BioEmu2FlowNetwork`` and ``HoloFlowNetwork`` of the old project
code) into a :class:`BioEmu2Network` through :data:`ARCHIVED_KEY_MAP`, and
reads their architecture from the weights. A :class:`BioEmu2Network` of the
same architecture also loads their state dicts directly (AIMMD's
``Params.update_network`` on an archived run folder): the old name of the
single adapter is read for one release, with one ``DeprecationWarning``.

Ported from ``bioemu2_committor/aimmd_flow.py`` (``BioEmu2FlowNetwork``) and
``aimmd_flow_holo.py`` (``HoloFlowNetwork``), merged; the probability-flow
``velocity`` / ``sample_prior`` and ``generate`` were dropped (DESIGN 8.3).
"""

import weakref
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .adapters import ConditionedDenoiser, TrunkAdapter
from .head import LAMBDA_CLIP, SIGMA_REF, CommittorHead
from .system import BioEmu2System

__all__ = ['BioEmu2Network', 'READOUTS', 'ARCHIVED_KEY_MAP', 'archived_state_dict', 'archived_settings',
           'load_archived_network']

READOUTS = ('flat', 'pooled')
"""The feature readouts (:mod:`.features`)."""
_POOLS = ('ligand', 'pocket', 'protein')

_MODELS = weakref.WeakValueDictionary()
"""Loaded BioEmu2 models by (checkpoint, checkpoint file, device): one per process while a network uses it."""


def _shared_model(checkpoint, checkpoint_path, device):
    from . import _api
    key = (None if checkpoint_path is not None else str(checkpoint),
           None if checkpoint_path is None else str(Path(checkpoint_path).resolve()), str(device))
    model = _MODELS.get(key)
    if model is None:
        with torch.inference_mode(False):
            model = _api.load_model(checkpoint, checkpoint_path=checkpoint_path, device=device)
        _MODELS[key] = model
    return model


class _Engine:
    """Frozen BioEmu2 on one device: the features of the head and the conditioned denoiser."""

    def __init__(self, network, device):
        from .features import BioEmu2Features
        self.device = device
        if network._model is not None:
            model = network._model
            if torch.device(model.device) != device:
                raise ValueError(f'the BioEmu2 model given to the network is on {model.device}, the network on '
                                 f'{device}; move the network or give a model on its device')
        else:
            model = _shared_model(network.checkpoint, network.checkpoint_path, device)
        # never inside inference mode: the trunk and the contexts are used for training too
        with torch.inference_mode(False), torch.no_grad():
            self.features = BioEmu2Features(network.system, model, sigma_ref=network.sigma_ref,
                                            readout=network.readout,
                                            allow_version_mismatch=network.allow_version_mismatch)
        self.model = model
        widths = (self.features.token_width, int(self.features.trunk.single.shape[-1]),
                  int(self.features.trunk.pair.shape[-1]))
        if widths != network.widths:
            self.features.close()
            raise ValueError(f'the BioEmu2 model has token, single and pair widths {widths}, the network was built '
                             f'for {network.widths} (the system folder {network.system.folder}): the folder was '
                             f'built with another checkpoint')
        self.denoiser = ConditionedDenoiser(model, self.features.trunk, self.features.batch, network.single_adapter,
                                            network.pair_adapter)

    def close(self):
        self.features.close()


class BioEmu2Network(nn.Module):
    """The committor network of a BioEmu2 C-GAIMMD run, with the adapters of its generator (module docstring).

    Parameters
    ----------
    system : BioEmu2System or str or Path
        The system, or its folder (loaded without a check; the check against
        the installed bioemu2 runs when BioEmu2 is first used).
    readout : {'flat', 'pooled'}, optional
        The feature readout (:mod:`.features`). Default: ``'flat'`` for a
        system without ligand (the chignolin campaign), ``'pooled'`` with one
        (the trypsin campaign).
    pair_adapter : bool, optional
        Adapt the pair embedding as well as the single one. Default: only for
        a system with a ligand (as in production).
    hidden : sequence of int, optional
        Hidden widths of the committor head.
    scaler : dict or str or Path, optional
        The head's feature scaler (:func:`.head.fit_scaler`; or an ``.npz``
        holding it). Usually it comes with the initial network's weights.
    adapter_hidden, pair_hidden, e_dim, n_freq : int, optional
        Widths of the adapters (:class:`.adapters.TrunkAdapter`).
    learned_null : bool, optional
        Adapters with a learned null embedding (needed for guidance against a
        learned null and for a label dropout that trains it).
    sigma_ref : float, optional
        Noise level (Ångström) of the features of a clean structure.
    value_batch : int, optional
        Structures per BioEmu2 pass in :meth:`forward`.
    checkpoint, checkpoint_path : optional
        The BioEmu2 checkpoint (a name of the Hugging Face repository, or a
        local file); default: the one the system folder was built with.
    model : InferenceModel, optional
        A loaded frozen BioEmu2 model to use instead (on the network's device;
        not saved, not copied).
    allow_version_mismatch : bool, optional
        See :meth:`.system.BioEmu2System.check`.
    """

    lambda_clip = LAMBDA_CLIP
    """The conditioning range of the generator (logits beyond it are clipped)."""

    def __init__(self, system, *, readout=None, pair_adapter=None, hidden=(512,), scaler=None, adapter_hidden=128,
                 pair_hidden=64, e_dim=32, n_freq=8, learned_null=False, sigma_ref=SIGMA_REF, value_batch=64,
                 checkpoint=None, checkpoint_path=None, model=None, allow_version_mismatch=False):
        super().__init__()
        if not isinstance(system, BioEmu2System):
            system = BioEmu2System.load(system, check=False)
        trunk = system.data.get('trunk')
        if not trunk or 'token_feature_width' not in trunk:
            raise ValueError(f'the system {system.folder} has no trunk embeddings record: build its folder with '
                             f'python -m aimmd.cgaimmd.bioemu2 build-system')
        has_ligand = bool(system.ligands)
        readout = ('pooled' if has_ligand else 'flat') if readout is None else readout
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}, got {readout!r}')
        pair_adapter = has_ligand if pair_adapter is None else bool(pair_adapter)
        if not sigma_ref > 0:
            raise ValueError(f'sigma_ref must be > 0, got {sigma_ref!r}')
        self.system = system
        self.n_atoms = system.n_atoms
        self.readout = readout
        self.sigma_ref = float(sigma_ref)
        self.value_batch = int(value_batch)
        self.allow_version_mismatch = bool(allow_version_mismatch)
        self.checkpoint = checkpoint if checkpoint is not None or checkpoint_path is not None else (
            trunk.get('checkpoint') or 'bioemu2-preview')
        self.checkpoint_path = (checkpoint_path if checkpoint is not None or checkpoint_path is not None
                                else trunk.get('checkpoint_path'))
        self.widths = (int(trunk['token_feature_width']), int(trunk['single_width']), int(trunk['pair_width']))
        pools = [name for name in _POOLS if name in system.token_pools]
        n_features = (system.n_tokens if readout == 'flat' else len(pools)) * self.widths[0]
        if isinstance(scaler, (str, Path)):
            scaler = dict(np.load(scaler))
        scaler = dict(scaler or {})
        head = dict(n_features=n_features, hidden=tuple(int(h) for h in hidden), mean=scaler.get('mean'),
                    std=scaler.get('std'), log_sigma_mean=float(scaler.get('log_sigma_mean', 0.0)),
                    log_sigma_std=float(scaler.get('log_sigma_std', 1.0)))
        self.head = CommittorHead(**head)
        self.head_ema = CommittorHead(**head).requires_grad_(False)
        self.head_ema.load_state_dict(self.head.state_dict())
        self.single_adapter = TrunkAdapter(self.widths[1], hidden=adapter_hidden, n_freq=n_freq, e_dim=e_dim,
                                           learned_null=learned_null)
        self.pair_adapter = (TrunkAdapter(self.widths[2], hidden=pair_hidden, n_freq=n_freq, e_dim=e_dim,
                                          learned_null=learned_null) if pair_adapter else None)
        self.learned_null = bool(learned_null)
        object.__setattr__(self, '_model', model)              # plain attributes: never saved, never moved
        object.__setattr__(self, '_engine', None)

    # ------------------------------------------------------------------ description
    @property
    def device(self):
        return self.head.mean.device

    @property
    def n_features(self):
        """Width of the head's features."""
        return self.head.n_features

    def adapters(self):
        """The trunk adapters (single, then pair if present)."""
        return [self.single_adapter] + ([self.pair_adapter] if self.pair_adapter is not None else [])

    def settings(self):
        """The JSON-able settings of this network (constructor arguments and widths)."""
        single = self.single_adapter.settings()
        return dict(system=None if self.system.folder is None else str(self.system.folder), readout=self.readout,
                    pair_adapter=self.pair_adapter is not None, hidden=list(self.head.hidden),
                    adapter_hidden=single['hidden'],
                    pair_hidden=None if self.pair_adapter is None else self.pair_adapter.settings()['hidden'],
                    e_dim=single['e_dim'], n_freq=single['n_freq'], learned_null=self.learned_null,
                    sigma_ref=self.sigma_ref, n_features=self.n_features, widths=list(self.widths),
                    checkpoint=self.checkpoint,
                    checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path))

    def default_generator(self):
        """The generation method of this network: ``{'method': 'bioemu2_edm'}`` (:mod:`.sampling`)."""
        return {'method': 'bioemu2_edm'}

    def provenance(self):
        """What this network's values and samples depend on besides its weights (for the sidecars).

        Returns
        -------
        dict
            ``bioemu2_version``; ``checkpoint`` (the checkpoint name, None for a checkpoint
            file) and ``checkpoint_path``; ``hf_snapshot``, the Hugging Face snapshot the
            checkpoint comes from (None when it cannot be told, e.g. a checkpoint file outside
            the cache); ``parameter_checksum`` of the loaded BioEmu2 model (sha1 over its
            parameters and buffers, computed once per model); ``n_parameters``; the
            ``system`` folder and its ``trunk`` record; ``readout`` and ``sigma_ref``.

        Notes
        -----
        BioEmu2 is loaded if it is not yet (the shooting worker calls this once at start-up,
        after its numerical checks have used the model).
        """
        from . import _api
        model = self.engine().model
        return {'bioemu2_version': _api.bioemu2_version(),
                'checkpoint': None if self.checkpoint_path is not None else self.checkpoint,
                'checkpoint_path': None if self.checkpoint_path is None else str(self.checkpoint_path),
                'hf_snapshot': _api.checkpoint_snapshot(self.checkpoint, self.checkpoint_path),
                'parameter_checksum': _api.model_parameter_checksum(model),
                'n_parameters': int(sum(p.numel() for p in model.parameters())),
                'system': None if self.system.folder is None else str(self.system.folder),
                'trunk': self.system.data.get('trunk'), 'readout': self.readout,
                'sigma_ref': float(self.sigma_ref)}

    def __getstate__(self):
        # copies and pickles never carry BioEmu2 (it is loaded again on first use)
        state = super().__getstate__() if hasattr(super(), '__getstate__') else self.__dict__.copy()
        state = dict(state)
        state['_model'] = None
        state['_engine'] = None
        return state

    def __repr__(self):
        return (f'{type(self).__name__}(system={str(self.system.folder)!r}, readout={self.readout!r}, '
                f'pair_adapter={self.pair_adapter is not None}, learned_null={self.learned_null}, '
                f'n_features={self.n_features})')

    # ------------------------------------------------------------------ BioEmu2
    def engine(self):
        """Frozen BioEmu2 on the network's device: ``.model``, ``.features``, ``.denoiser``.

        Built on first use (BioEmu2 is loaded then, once per process and
        device) and again after the network moved to another device.
        """
        engine = self._engine
        device = self.device
        if engine is not None and engine.device == device:
            return engine
        if engine is not None:
            engine.close()
            object.__setattr__(self, '_engine', None)
        # bioemu2 draws random numbers while it prepares a system: keep them off the global streams
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        if device.type == 'cuda' and not devices:
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            engine = _Engine(self, device)
        object.__setattr__(self, '_engine', engine)
        return engine

    def to_bioemu2(self, x):
        """Generator atoms ``(B, N, 3)`` or flat ``(B, 3 N)`` -> BioEmu2 atoms ``(B, M, 3)`` float32 on the device."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, self.n_atoms, 3)
        index = torch.as_tensor(self.system.bioemu2_to_md, dtype=torch.long, device=self.device)
        return x.index_select(1, index)

    def features(self, x, sigmas=None):
        """``(B, n_features)`` frozen BioEmu2 features of generator atoms `x` at `sigmas` (default ``sigma_ref``)."""
        return self.engine().features(torch.as_tensor(x, dtype=torch.float32, device=self.device), sigmas)

    # ------------------------------------------------------------------ the committor
    def forward(self, descriptors):
        """AIMMD's contract: the logit committor toward ``sorted_states[2]``, ``(B, 1)``, of the EMA head."""
        x = torch.as_tensor(descriptors, dtype=torch.float32, device=self.device).reshape(-1, self.n_atoms, 3)
        out = []
        for i in range(0, len(x), self.value_batch):
            features = self.features(x[i:i + self.value_batch])
            sigma = torch.full((len(features),), self.sigma_ref, device=self.device)
            out.append(self.head_ema(features, sigma))
        return torch.cat(out).reshape(-1, 1) if out else torch.zeros(0, 1, device=self.device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # the archived networks called the single adapter 'adapter' (ARCHIVED_KEY_MAP): read for one release, so
        # that AIMMD's own loading (Params.update_network) reads their run folders; never written
        old, new = f'{prefix}adapter.', f'{prefix}single_adapter.'
        renamed = [key for key in state_dict if key.startswith(old)]
        if renamed and not any(key.startswith(new) for key in state_dict):
            from .._legacy import warn_renamed
            warn_renamed("the parameter prefix 'adapter.' of an archived BioEmu2 network", "'single_adapter.'",
                         key=('BioEmu2 network parameter', 'adapter'), stacklevel=5)
            for key in renamed:
                state_dict[new + key[len(old):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def reset_parameters(self, seed=None):
        """A fresh head (its scaler kept), the EMA head equal to it, and identity adapters.

        With `seed`: the head from ``seed + 1``, the adapters from ``seed + 2``,
        ``seed + 3`` (as one round of :func:`.fit.sequential_round` with that
        round seed); the global random stream is left as it was.
        """
        self.head.reset_parameters(None if seed is None else int(seed) + 1)
        self.head_ema.load_state_dict(self.head.state_dict())
        for j, adapter in enumerate(self.adapters()):
            adapter.reset_parameters(None if seed is None else int(seed) + 2 + j)


# ---------------------------------------------------------------------------
# the networks of the archived campaigns
# ---------------------------------------------------------------------------

ARCHIVED_KEY_MAP = {'head': 'head', 'head_ema': 'head_ema', 'adapter': 'single_adapter',
                    'pair_adapter': 'pair_adapter'}
"""First component of a parameter name in the archived networks -> in :class:`BioEmu2Network`.

The archived campaigns saved ``BioEmu2FlowNetwork`` (chignolin, 27 Oct 2026:
``head``, ``head_ema``, ``adapter``) and ``HoloFlowNetwork`` (trypsin, 28-30
Oct 2026: also ``pair_adapter``). The head (``CommittorHead``, the old
``JointHead``) and the adapters keep their old parameter names.
"""


def archived_state_dict(state):
    """The state dict of an archived network with the parameter names of :class:`BioEmu2Network`.

    Raises
    ------
    KeyError
        For a parameter outside :data:`ARCHIVED_KEY_MAP` (another kind of network).
    """
    out = {}
    for key, value in state.items():
        first, _, rest = key.partition('.')
        if first not in ARCHIVED_KEY_MAP or not rest:
            raise KeyError(f'{key!r} is not a parameter of an archived BioEmu2 network (known: '
                           f'{sorted(ARCHIVED_KEY_MAP)}); is this an archived BioEmu2FlowNetwork / HoloFlowNetwork?')
        out[f'{ARCHIVED_KEY_MAP[first]}.{rest}'] = value
    return out


def _linear_widths(state, prefix):
    layers = sorted({int(key[len(prefix):].split('.')[0]) for key in state
                     if key.startswith(prefix) and key.endswith('.weight')})
    return [int(state[f'{prefix}{i}.weight'].shape[0]) for i in layers]


def archived_settings(state):
    """The architecture of an archived network, read from its weights.

    Returns
    -------
    dict
        ``hidden``, ``n_features``, ``adapter_hidden``, ``e_dim``, ``n_freq``,
        ``learned_null``, ``pair_adapter``, ``pair_hidden``, ``single_width``,
        ``pair_width`` (the last two must match the system's trunk).
    """
    state = archived_state_dict(state)
    single = 'single_adapter.'
    out = dict(hidden=_linear_widths(state, 'head.mlp.hidden.'), n_features=int(state['head.mean'].shape[0]),
               adapter_hidden=int(state[f'{single}inp.weight'].shape[0]),
               e_dim=int(state[f'{single}lam_embed.0.weight'].shape[0]),
               n_freq=int(state[f'{single}freq'].shape[0]), learned_null=f'{single}null' in state,
               single_width=int(state[f'{single}norm.weight'].shape[0]),
               pair_adapter='pair_adapter.inp.weight' in state, pair_hidden=None, pair_width=None)
    if out['pair_adapter']:
        out['pair_hidden'] = int(state['pair_adapter.inp.weight'].shape[0])
        out['pair_width'] = int(state['pair_adapter.norm.weight'].shape[0])
    return out


def load_archived_network(file, system, *, readout=None, **kwargs):
    """A :class:`BioEmu2Network` with the weights of an archived network file.

    Parameters
    ----------
    file : str or Path or dict
        The archived network (``torch.save(network.state_dict())``, e.g.
        ``networkARB.h5`` or ``pretrained/networkARB.h5`` of the archived
        campaigns), or its state dict.
    system : BioEmu2System or str or Path
        The system the network was trained on (its folder).
    readout : {'flat', 'pooled'}, optional
        Default: the readout whose width matches the archived head.
    **kwargs
        Further :class:`BioEmu2Network` arguments (``model``, ``checkpoint``,
        ``value_batch``, ...).

    Raises
    ------
    ValueError
        If the archived head or adapters do not fit the system (feature or
        trunk widths).
    """
    state = file if isinstance(file, dict) else torch.load(file, map_location='cpu', weights_only=True)
    settings = archived_settings(state)
    if not isinstance(system, BioEmu2System):
        system = BioEmu2System.load(system, check=False)
    if readout is None:
        trunk = system.data.get('trunk') or {}
        width = int(trunk.get('token_feature_width', 0))
        pools = [name for name in _POOLS if name in system.token_pools]
        by_width = {system.n_tokens * width: 'flat', len(pools) * width: 'pooled'}
        readout = by_width.get(settings['n_features'])
        if readout is None:
            raise ValueError(f"the archived head reads {settings['n_features']} features; the system "
                             f'{system.folder} gives {system.n_tokens * width} (flat) or {len(pools) * width} '
                             f'(pooled)')
    network = BioEmu2Network(system, readout=readout, pair_adapter=settings['pair_adapter'],
                             hidden=settings['hidden'], adapter_hidden=settings['adapter_hidden'],
                             pair_hidden=settings['pair_hidden'] or 64, e_dim=settings['e_dim'],
                             n_freq=settings['n_freq'], learned_null=settings['learned_null'], **kwargs)
    if network.n_features != settings['n_features']:
        raise ValueError(f"the archived head reads {settings['n_features']} features, the network's "
                         f'{network.readout} readout of the system gives {network.n_features}')
    have = (settings['single_width'], settings['pair_width'] if settings['pair_adapter'] else network.widths[2])
    if have != network.widths[1:]:
        raise ValueError(f'the archived adapters have trunk widths {have}, the system folder records '
                         f'{network.widths[1:]}')
    network.load_state_dict(archived_state_dict(state))
    return network
