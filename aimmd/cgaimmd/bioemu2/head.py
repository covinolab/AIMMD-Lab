"""
aimmd.cgaimmd.bioemu2.head
==========================

The committor head of a BioEmu2 C-GAIMMD network: a small MLP on the frozen
BioEmu2 features of a structure (:mod:`.features`) and the noise level they
were taken at.

``logit q = MLP([z-scored features, z-scored log sigma])``

- At ``sigma_ref`` on a clean structure it is the committor toward
  ``sorted_states[2]``. The fit also trains it on structures noised at
  ``sigma`` (the noise-conditioned committor), which regularises it and keeps
  the noise input meaningful.
- The z-score scaler (:func:`fit_scaler`) is fitted once, on the frames of the
  initial network (:func:`aimmd.cgaimmd.bioemu2.pretrain.pretrain_initial_network`),
  and stays frozen: every round's fresh head keeps it (it is part of the
  head's buffers, so it is saved and loaded with the network).
- The losses of the fit: :func:`shooting_point_nll` (two-way outcomes of
  shooting points) and :func:`in_state_nll` (in-state frames toward
  ``sigmoid(+-LAMBDA_CLIP)``).

Only torch and numpy: a network loads and its head runs without bioemu2.

Ported from the BioEmu2 project code of the archived production campaigns
(``bioemu2_committor/joint.py``: ``JointHead`` is :class:`CommittorHead`,
``fit_scaler``; ``bioemu2_committor/head.py``: ``FeatureMLP``, ``STD_FLOOR``;
``aimmd_flow_fit.py``: ``_sp_nll``, ``_anchor``, ``fresh_head``). The buffer
and layer names are the old ones, so the archived networks load with a plain
key map (:func:`aimmd.cgaimmd.bioemu2.network.archived_state_dict`).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['SIGMA_REF', 'SIGMA_RANGE', 'LAMBDA_CLIP', 'STD_FLOOR', 'FeatureMLP', 'CommittorHead', 'fit_scaler',
           'shooting_point_nll', 'in_state_nll']

SIGMA_REF = 0.05
"""Noise level (Ångström) of the features of a clean structure (both production campaigns)."""
SIGMA_RANGE = (0.05, 20.0)
"""Noise levels (Ångström) of the noised copies the head is trained on (drawn log-uniformly)."""
LAMBDA_CLIP = 10.0
"""Largest committor logit the generator is conditioned on, and the in-state target ``sigmoid(+-LAMBDA_CLIP)``."""
STD_FLOOR = 1e-3
"""Smallest standard deviation of the scaler (constant features are not blown up)."""


class FeatureMLP(nn.Module):
    """``(B, in_dim) -> (B, 1)`` logits: ``Linear``/``SiLU`` layers of the widths `hidden`, then ``Linear(., 1)``.

    Parameters
    ----------
    in_dim : int
        Input width.
    hidden : sequence of int, optional
        Widths of the hidden layers.
    """

    def __init__(self, in_dim, hidden=(64, 64, 32)):
        super().__init__()
        self.in_dim = int(in_dim)
        layers, previous = [], self.in_dim
        for width in hidden:
            layers += [nn.Linear(previous, int(width)), nn.SiLU()]
            previous = int(width)
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(previous, 1)

    def forward(self, z):
        return self.out(self.hidden(z.reshape(len(z), -1)))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class CommittorHead(nn.Module):
    """``(features (B, n_features), sigma (B,)) -> (B,)`` committor logits, with a frozen z-score scaler.

    Parameters
    ----------
    n_features : int
        Width of the BioEmu2 features (:attr:`.features.BioEmu2Features.n_features`).
    hidden : sequence of int, optional
        Hidden widths of the MLP (production: one layer of 512).
    mean, std : array-like, optional
        The feature scaler (default 0 and 1); `std` is floored at :data:`STD_FLOOR`.
    log_sigma_mean, log_sigma_std : float, optional
        The scaler of ``log sigma``.

    Notes
    -----
    The scaler lives in buffers (``mean``, ``std``, ``ls_mean``, ``ls_std``):
    it is saved and loaded with the network, and :meth:`reset_parameters`
    keeps it.
    """

    def __init__(self, n_features, hidden=(512,), mean=None, std=None, log_sigma_mean=0.0, log_sigma_std=1.0):
        super().__init__()
        n_features = int(n_features)
        self.mlp = FeatureMLP(n_features + 1, list(hidden))
        self.hidden = [int(h) for h in hidden]
        self.n_features = n_features
        self.register_buffer('mean', torch.zeros(n_features) if mean is None
                             else torch.as_tensor(np.asarray(mean), dtype=torch.float32).reshape(n_features))
        self.register_buffer('std', torch.ones(n_features) if std is None
                             else torch.as_tensor(np.asarray(std), dtype=torch.float32).reshape(
                                 n_features).clamp_min(STD_FLOOR))
        self.register_buffer('ls_mean', torch.tensor(float(log_sigma_mean)))
        self.register_buffer('ls_std', torch.tensor(max(float(log_sigma_std), STD_FLOOR)))

    def forward(self, features, sigma):
        z = (features.float() - self.mean) / self.std
        log_sigma = ((torch.log(sigma.float()) - self.ls_mean) / self.ls_std)[:, None]
        return self.mlp(torch.cat([z, log_sigma], dim=1)).reshape(-1)

    def set_scaler(self, scaler):
        """Set the scaler from a :func:`fit_scaler` dict (``mean``, ``std``, ``log_sigma_mean``, ``log_sigma_std``)."""
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(np.asarray(scaler['mean']), dtype=torch.float32))
            self.std.copy_(torch.as_tensor(np.asarray(scaler['std']), dtype=torch.float32).clamp_min(STD_FLOOR))
            self.ls_mean.fill_(float(scaler.get('log_sigma_mean', 0.0)))
            self.ls_std.fill_(max(float(scaler.get('log_sigma_std', 1.0)), STD_FLOOR))

    def scaler(self):
        """The scaler as a :func:`fit_scaler` dict (numpy arrays and floats)."""
        return dict(mean=self.mean.detach().cpu().numpy(), std=self.std.detach().cpu().numpy(),
                    log_sigma_mean=float(self.ls_mean), log_sigma_std=float(self.ls_std))

    def reset_parameters(self, seed=None):
        """A fresh MLP (``nn.Linear`` initialisation), the scaler kept.

        With `seed`, the weights are those of a ``FeatureMLP`` built after
        ``torch.manual_seed(seed)`` on a forked CPU random stream (the global
        stream is left as it was): the old ``fresh_head``.
        """
        if seed is None:
            self.mlp.reset_parameters()
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            state = FeatureMLP(self.n_features + 1, self.hidden).state_dict()
        self.mlp.load_state_dict(state)


def fit_scaler(features, sigmas):
    """The z-score scaler of features ``(n, D)`` and of ``log sigma`` (float64 sums, no full-size copy).

    Returns
    -------
    dict
        ``mean``, ``std`` (``(D,)`` float32; the floor is applied by the head),
        ``log_sigma_mean``, ``log_sigma_std`` (float).
    """
    n = len(features)
    if not n:
        raise ValueError('the scaler needs at least one feature row')
    s1 = np.zeros(features.shape[1])
    s2 = np.zeros(features.shape[1])
    for i in range(0, n, 4096):
        chunk = np.asarray(features[i:i + 4096], np.float64)
        s1 += chunk.sum(0)
        s2 += (chunk * chunk).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    log_sigma = np.log(np.asarray(sigmas, np.float64))
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), log_sigma_mean=float(log_sigma.mean()),
                log_sigma_std=float(log_sigma.std()))


def shooting_point_nll(logits, n_first, n_last):
    """Binomial negative log-likelihood of two-way shooting outcomes, per shot, averaged.

    `n_first` / `n_last` count the trajectories that ended in
    ``sorted_states[0]`` / ``sorted_states[2]``; the logit is toward the last.
    """
    return ((n_last * F.softplus(-logits) + n_first * F.softplus(logits)) / (n_first + n_last)).mean()


def in_state_nll(logits, state):
    """Cross-entropy of in-state frames (state codes 1, 2) toward ``sigmoid(-+LAMBDA_CLIP)``, averaged."""
    high = torch.sigmoid(torch.tensor(LAMBDA_CLIP, device=logits.device))
    target = torch.where(state == 2, high, 1 - high)
    return (target * F.softplus(-logits) + (1 - target) * F.softplus(logits)).mean()
