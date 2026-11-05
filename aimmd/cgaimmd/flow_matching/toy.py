"""
aimmd.cgaimmd.flow_matching.toy
===============================

:class:`TinyFlowNetwork`: the documented minimal flow-matching C-GAIMMD network.

It is the network of the toy 1-D end-to-end C-GAIMMD test, of the tutorial and
of the protocol documentation: a few-parameter MLP that implements exactly the
required members of the C-GAIMMD network protocol
(:mod:`aimmd.cgaimmd.protocol`) plus ``n_atoms`` and ``default_generator`` (and
``lambda_clip`` only when given), and the two members of a flow-matching
network that the ``'ode'`` generator integrates (``velocity`` and
``sample_prior``, :mod:`aimmd.cgaimmd.flow_matching.sampler`), so that
everything it takes part in exercises every other fallback. It ships in the
package so that spawned workers, which import the params file copy of a run,
and notebooks can import it.

Like every C-GAIMMD network it is used through a class defined in the params
file (``Params.save`` writes ``from <module of the class> import network``):

>>> from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
>>> class Network(TinyFlowNetwork):
...     pass
>>> network = Network()
"""

import torch


_DEFAULT_LAMBDA_SCALE = 10.0


class TinyFlowNetwork(torch.nn.Module):
    """A minimal flow-matching network: MLP committor on ``x[:, 0]``, MLP velocity, Gaussian prior.

    Parameters
    ----------
    n_atoms : int, optional
        Number ``M`` of generated atoms (the toy 1-D engine has one).
    hidden : int, optional
        Width of the hidden layers.
    prior_center : float, optional
        Center of the Gaussian prior, in Å, for every coordinate.
    prior_width : float, optional
        Standard deviation of the prior, in Å.
    lambda_clip : float, optional
        The conditioning range ``L``; when given it is the protocol attribute ``lambda_clip``.

    Notes
    -----
    - ``forward(inp)``: ``inp`` is the dense ``(B, 3 M)`` Å input of AIMMD's
      values function; the committor is ``MLP(inp[:, :1])`` (the first
      coordinate of the first atom). Its output layer is zero-initialized, so
      an untrained network returns ``lambda = 0`` everywhere.
    - ``velocity(x_t, t, lam, *, uncond_mask=None)``: ``MLP([x_t (flat), t,
      tanh(lam / L), null])`` on all coordinates, with ``null = 1`` (and the
      lambda feature 0) for ``lam=None`` and for the rows of ``uncond_mask``;
      ``L`` is ``lambda_clip``, or 10.0 when it is not given. The output layer
      is zero-initialized, so an untrained network returns its prior samples.
    - ``sample_prior(lam, n, *, uncond_mask=None, generator=None)``:
      ``prior_center + prior_width * N(0, 1)`` of shape ``(n, M, 3)``; it
      ignores ``lam``.
    - ``default_generator()``: ``{'method': 'ode'}``, the flow-matching ODE
      with its default settings.
    - Without ``lambda_clip`` the network has no such attribute and C-GAIMMD
      code takes ``cgaimmd_parameters['lambda_clip']``.
    - Flow units are Å (no ``to_flow``/``from_flow``) and absolute: nothing is
      centered, so the network suits a one-atom toy system (placement
      ``'none'``).
    - Construction does not touch CUDA; the network is float32 on the CPU
      until C-GAIMMD code moves it.
    """

    def __init__(self, n_atoms=1, hidden=16, prior_center=1.5, prior_width=0.3,
                 lambda_clip=None):
        super().__init__()
        n_atoms = int(n_atoms)
        if n_atoms < 1:
            raise ValueError(f'n_atoms must be >= 1, got {n_atoms}')
        hidden = int(hidden)
        self.n_atoms = n_atoms
        self.prior_center = float(prior_center)
        self.prior_width = float(prior_width)
        if lambda_clip is not None:
            lambda_clip = float(lambda_clip)
            if not lambda_clip > 0:
                raise ValueError(f'lambda_clip must be > 0, got {lambda_clip}')
            self.lambda_clip = lambda_clip
        self._lambda_scale = lambda_clip if lambda_clip is not None else _DEFAULT_LAMBDA_SCALE
        dim = 3 * n_atoms
        self.committor_net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))
        self.velocity_net = torch.nn.Sequential(
            torch.nn.Linear(dim + 3, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, dim))
        for head in (self.committor_net, self.velocity_net):
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)

    def forward(self, inp):
        """Logit committor ``(B, 1)`` of the dense ``(B, 3 M)`` input (Å)."""
        x = torch.as_tensor(inp, dtype=torch.float32)
        x = torch.flatten(x, start_dim=1)
        return self.committor_net(x[:, :1])

    def velocity(self, x_t, t, lam, *, uncond_mask=None):
        """Velocity ``(B, M, 3)`` at ``x_t`` ``(B, M, 3)``, time ``t`` ``(B,)``, level ``lam``."""
        batch = x_t.shape[0]
        flat = x_t.reshape(batch, -1)
        t = torch.as_tensor(t, dtype=flat.dtype, device=flat.device).reshape(batch, 1)
        null = torch.ones(batch, 1, dtype=flat.dtype, device=flat.device)
        level = torch.zeros(batch, 1, dtype=flat.dtype, device=flat.device)
        if lam is not None:
            lam = torch.as_tensor(lam, dtype=flat.dtype, device=flat.device).reshape(batch, 1)
            level = torch.tanh(lam / self._lambda_scale)
            null = torch.zeros_like(null)
            if uncond_mask is not None:
                mask = torch.as_tensor(uncond_mask, dtype=torch.bool,
                                       device=flat.device).reshape(batch, 1)
                level = torch.where(mask, torch.zeros_like(level), level)
                null = mask.to(flat.dtype)
        features = torch.cat([flat, t, level, null], dim=1)
        return self.velocity_net(features).reshape(x_t.shape)

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        """``(n, M, 3)`` Gaussian prior samples around ``prior_center`` (Å); ignores ``lam``."""
        device = next(self.parameters()).device
        noise = torch.randn((int(n), self.n_atoms, 3), generator=generator, device=device,
                            dtype=torch.float32)
        return self.prior_center + self.prior_width * noise

    def default_generator(self):
        """The generation method of this network: the flow-matching ODE, ``{'method': 'ode'}``."""
        return {'method': 'ode'}
