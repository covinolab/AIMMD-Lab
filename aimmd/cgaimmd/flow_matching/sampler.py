"""
aimmd.cgaimmd.flow_matching.sampler
===================================

The fixed-step ODE of flow-matching networks and its generator ``'ode'``.

A flow-matching network's generator is the velocity field ``v(x_t, t, lam)``
of an ODE from the prior (``t = 0``) to configurations (``t = 1``) at the
committor level ``lam``. This module holds the members such a network
provides (the contract of the ``'ode'`` method; the C-GAIMMD network protocol
itself covers the committor only) and the code that integrates them:

- :func:`integrate`: a generic fixed-step integrator;
- :func:`generate`: prior samples integrated along ``velocity``;
- :class:`ODEGenerator`: the generator ``'ode'``
  (:mod:`aimmd.cgaimmd.generation`), registered when this module is imported.

The members of a flow-matching network, beyond the protocol:

- ``velocity(x_t, t, lam, *, uncond_mask=None) -> Tensor (B, M, 3)``: flow
  units; ``t`` ``(B,)`` in ``[0, 1]``; ``lam`` ``(B,)`` logits or None (the
  learned null token, never the same as ``lam = 0``); ``uncond_mask`` ``(B,)``
  bool rows that use the null token. The keyword ``uncond_mask`` is needed
  only for classifier-free guidance (``guidance_scale != 1``) and by the
  flow-matching fit (:mod:`~aimmd.cgaimmd.flow_matching.fit`), which drops
  labels with it; plain generation calls ``velocity(x_t, t, lam)``.
- ``sample_prior(lam, n, *, generator=None) -> Tensor (n, M, 3)``: flow units.
- optional ``to_flow(x)``, ``from_flow(z)``: Å to flow units and back
  [:func:`to_flow`, :func:`from_flow`: identity]; optional
  ``flow_pair(x1, x0)``: the optimal-transport pairing of the fit
  [:func:`pair`: identity].

The sampler never centers or projects: a network whose flow lives in a
subspace (the PaiNN reference model: center-of-mass free) returns velocities
in that subspace itself. That is what lets a one-atom toy network, whose
coordinates are absolute, use the same code.
"""

import math
from numbers import Integral, Real

import torch

from .. import protocol
from ..generation import _register_builtin


SAMPLERS = ('euler', 'midpoint', 'rk4')
"""Fixed-step integrators, with 1, 2 and 4 velocity evaluations per step."""

REQUIRED_MEMBERS = ('velocity', 'sample_prior')
"""Members a network must have for the ``'ode'`` method."""

OPTIONAL_MEMBERS = ('to_flow', 'from_flow', 'flow_pair')
"""Members a flow-matching network may have; each has an identity fallback."""

_SIGNATURES = {
    'velocity': (('x_t', 't', 'lam'), {}, 'velocity(x_t, t, lam)'),
    'sample_prior': (('lam', 'n'), {'generator': None},
                     'sample_prior(lam, n, *, generator=None)'),
    'to_flow': (('x',), {}, 'to_flow(x)'),
    'from_flow': (('z',), {}, 'from_flow(z)'),
    'flow_pair': (('x1', 'x0'), {}, 'flow_pair(x1, x0)'),
}


def to_flow(network, x):
    """Å -> flow units: ``network.to_flow(x)``, else `x` (identity)."""
    method = getattr(network, 'to_flow', None)
    return method(x) if callable(method) else x


def from_flow(network, z):
    """Flow units -> Å: ``network.from_flow(z)``, else `z` (identity)."""
    method = getattr(network, 'from_flow', None)
    return method(z) if callable(method) else z


def pair(network, x1, x0):
    """OT pairing: ``network.flow_pair(x1, x0)``, else `x1` (identity).

    Returns `x1` (flow units) acted on by the symmetry element that brings it
    closest to the prior sample `x0`.
    """
    method = getattr(network, 'flow_pair', None)
    return method(x1, x0) if callable(method) else x1


_VELOCITY_WITH_MASK = (('x_t', 't', 'lam'), {'uncond_mask': None},
                       'velocity(x_t, t, lam, *, uncond_mask=None)')
"""The signature of ``velocity`` that classifier-free guidance and the fit need."""


def flow_matching_problems(network, *, uncond_mask=False):
    """What `network` lacks to be integrated by this module, one message per problem.

    Parameters
    ----------
    network : object
        The candidate flow-matching network.
    uncond_mask : bool, optional
        Also require the keyword ``uncond_mask`` of ``velocity`` (the null
        token on chosen rows): classifier-free guidance and the
        flow-matching fit use it.

    Returns
    -------
    list of str
        Empty when ``velocity`` and ``sample_prior`` exist, are callable and
        bind their arguments (``velocity``: ``(x_t, t, lam)``, and with
        `uncond_mask` the keyword ``uncond_mask``; ``sample_prior``:
        ``(lam, n)`` and the keyword ``generator``) and every present optional
        member binds its arguments (``to_flow``, ``from_flow``: one;
        ``flow_pair``: two). ``**kwargs`` satisfies any keyword.
    """
    problems = []
    for name in REQUIRED_MEMBERS + OPTIONAL_MEMBERS:
        args, kwargs, expected = _SIGNATURES[name]
        if name == 'velocity' and uncond_mask:
            args, kwargs, expected = _VELOCITY_WITH_MASK
        member = getattr(network, name, None)
        if member is None:
            if name in REQUIRED_MEMBERS:
                problems.append(f'needs {expected}, which the network lacks')
            continue
        if not callable(member):
            problems.append(f'{name} is set but not callable (expected {expected})')
            continue
        error = protocol._bind(member, args, kwargs)
        if error is not None:
            problems.append(f'{name} must accept {expected} ({error})')
    return problems


def integrate(velocity_fn, z, *, t0=0.0, t1=1.0, sampler='midpoint', n_steps=16):
    """Integrate ``dx/dt = velocity_fn(x, t)`` from `t0` to `t1` with fixed steps.

    Parameters
    ----------
    velocity_fn : callable
        ``velocity_fn(x, t) -> dx/dt``, with ``t`` a ``(B,)`` tensor (dtype and device of `z`).
    z : torch.Tensor
        ``(B, ...)`` start values at `t0`.
    t0, t1 : float, optional
        Start and end time; ``t0 > t1`` integrates backwards.
    sampler : {'euler', 'midpoint', 'rk4'}, optional
        Explicit Euler, explicit midpoint (the default) or classical Runge-Kutta.
    n_steps : int, optional
        Number of equal steps (>= 1).

    Returns
    -------
    torch.Tensor
        The values at `t1`; no projection of any kind is applied.

    Raises
    ------
    ValueError
        For an unknown `sampler` or ``n_steps < 1``.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f'unknown ODE sampler {sampler!r}; expected one of {SAMPLERS}')
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
        raise ValueError(f'n_steps must be an int >= 1, got {n_steps!r}')
    n_steps = int(n_steps)
    dt = (t1 - t0) / n_steps
    x = z

    def at(t):
        return torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)

    for i in range(n_steps):
        t = t0 + i * dt
        if sampler == 'euler':
            x = x + dt * velocity_fn(x, at(t))
        elif sampler == 'midpoint':
            k1 = velocity_fn(x, at(t))
            k2 = velocity_fn(x + 0.5 * dt * k1, at(t + 0.5 * dt))
            x = x + dt * k2
        else:
            k1 = velocity_fn(x, at(t))
            k2 = velocity_fn(x + 0.5 * dt * k1, at(t + 0.5 * dt))
            k3 = velocity_fn(x + 0.5 * dt * k2, at(t + 0.5 * dt))
            k4 = velocity_fn(x + dt * k3, at(t + dt))
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _checked(what, value, shape):
    """`value` if it is a tensor of `shape`, else a ValueError naming the member."""
    if not torch.is_tensor(value):
        raise ValueError(f'{what} returned {type(value).__name__}, expected a tensor of shape '
                         f'{tuple(shape)}')
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{what} returned shape {tuple(value.shape)}, expected {tuple(shape)}')
    return value


def _check_chunk(chunk):
    if isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1:
        raise ValueError(f'chunk must be an int >= 1, got {chunk!r}')
    return int(chunk)


def _check_guidance_scale(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) \
            or value < 0:
        raise ValueError(f'guidance_scale must be a finite number >= 0, got {value!r}')
    return float(value)


def generate(network, lam, *, n=None, sampler='midpoint', n_steps=16, chunk=128,
             generator=None, guidance_scale=1.0):
    """Generate configurations by integrating the network's velocity field.

    Parameters
    ----------
    network : object
        A flow-matching network (only ``velocity``, ``sample_prior``,
        ``parameters``, ``n_atoms`` and the optional ``from_flow`` are used).
    lam : array-like or None
        ``(n,)`` target committor logits, or None for the null token (then `n` is required).
    n : int, optional
        Number of samples; must equal ``len(lam)`` when both are given.
    sampler, n_steps : optional
        See :func:`integrate`.
    chunk : int, optional
        Samples per network call (>= 1).
    generator : torch.Generator, optional
        On the network's device; passed to ``sample_prior``.
    guidance_scale : float, optional
        Classifier-free guidance weight ``w >= 0``: the ODE integrates ``v_u +
        w (v_c - v_u)``, with ``v_c = velocity(x_t, t, lam)`` and ``v_u =
        velocity(x_t, t, lam, uncond_mask=<all rows>)`` (the null token).
        1 (the default) is plain conditional generation and evaluates
        ``velocity`` once per stage, exactly as without guidance; any other
        value evaluates it twice. With ``lam=None`` there is nothing to guide
        and `guidance_scale` is ignored.

    Returns
    -------
    torch.Tensor
        ``(n, M, 3)`` float32 in Å on the network's device, without autograd history.

    Raises
    ------
    ValueError
        For inconsistent `lam` and `n`, a bad `chunk`, `sampler`, `n_steps`
        or `guidance_scale`, or a ``sample_prior``, ``velocity`` or
        ``from_flow`` output of the wrong shape (the message names the member).

    Notes
    -----
    For each chunk of at most `chunk` samples: ``z =
    network.sample_prior(lam_chunk, n_chunk, generator=generator)``, ``x =
    integrate(lambda x, t: network.velocity(x, t, lam_chunk), z, ...)`` (or
    the guided velocity), then :func:`from_flow`. The prior does not depend
    on `guidance_scale`, and guidance draws no random numbers, so the same
    `generator` gives the same prior samples at every guidance weight. The
    network's train/eval mode is left as it is.
    """
    device = protocol.network_device(network)
    if lam is None:
        if n is None:
            raise ValueError('lam=None (the null token) needs n, the number of samples')
        lam_all, total = None, int(n)
    else:
        lam_all = torch.as_tensor(lam, dtype=torch.float32, device=device).reshape(-1)
        total = len(lam_all)
        if n is not None and int(n) != total:
            raise ValueError(f'n={n} differs from len(lam)={total}')
    chunk = _check_chunk(chunk)
    guidance_scale = _check_guidance_scale(guidance_scale)
    n_atoms = protocol.n_atoms_of(network)
    out = []
    with torch.no_grad():
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            lam_chunk = None if lam_all is None else lam_all[start:stop]
            shape = (stop - start, n_atoms, 3)
            z = _checked('sample_prior(lam, n)',
                         network.sample_prior(lam_chunk, stop - start, generator=generator),
                         shape)

            def velocity(x, t):
                return _checked('velocity(x_t, t, lam)', network.velocity(x, t, lam_chunk),
                                shape)

            if guidance_scale != 1.0 and lam_chunk is not None:
                velocity = _guided(network, lam_chunk, shape, guidance_scale)
            x = integrate(velocity, z, sampler=sampler, n_steps=n_steps)
            out.append(_checked('from_flow(z)', from_flow(network, x), shape).to(torch.float32))
    if not out:
        return torch.zeros((0, n_atoms, 3), dtype=torch.float32, device=device)
    return torch.cat(out)


def _guided(network, lam, shape, weight):
    """The classifier-free-guided velocity ``v_u + weight (v_c - v_u)`` at the levels `lam`."""
    null = torch.ones(len(lam), dtype=torch.bool, device=lam.device)

    def velocity(x, t):
        v_c = _checked('velocity(x_t, t, lam)', network.velocity(x, t, lam), shape)
        v_u = _checked('velocity(x_t, t, lam, uncond_mask=...)',
                       network.velocity(x, t, lam, uncond_mask=null), shape)
        return v_u + weight * (v_c - v_u)
    return velocity


class ODEGenerator:
    """The generator ``'ode'``: the flow-matching ODE from the network's prior.

    Parameters
    ----------
    sampler : {'euler', 'midpoint', 'rk4'}, optional
        The fixed-step integrator (:func:`integrate`).
    n_steps : int, optional
        Integration steps from the prior (``t = 0``) to configurations (``t = 1``).
    chunk : int, optional
        Candidates per network call.
    guidance_scale : float, optional
        Classifier-free guidance weight ``w >= 0`` (:func:`generate`): 1, the
        default, is plain conditional generation; ``w > 1`` pushes the samples
        further toward their committor level. Offline, on chignolin, ``w = 3``
        raised the committor adherence of PaiNN samples at about the same
        validity; the default stays 1 until the online loop has been tested
        with guidance.

    Notes
    -----
    It implements :class:`aimmd.cgaimmd.generation.Generator`:
    :meth:`check` names what the network lacks of a flow-matching network
    (:func:`flow_matching_problems`; the keyword ``uncond_mask`` of
    ``velocity`` only when guidance is on), :meth:`generate` is
    :func:`generate` with these settings, and :meth:`settings` returns them.
    """

    method = 'ode'
    """The registry name of this method."""

    def __init__(self, sampler='midpoint', n_steps=16, chunk=128, guidance_scale=1.0):
        if sampler not in SAMPLERS:
            raise ValueError(f'unknown ODE sampler {sampler!r}; expected one of {SAMPLERS}')
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
            raise ValueError(f'n_steps must be an int >= 1, got {n_steps!r}')
        self.sampler = str(sampler)
        """The fixed-step integrator."""
        self.n_steps = int(n_steps)
        """Integration steps."""
        self.chunk = _check_chunk(chunk)
        """Candidates per network call."""
        self.guidance_scale = _check_guidance_scale(guidance_scale)
        """Classifier-free guidance weight (1: off)."""

    def __repr__(self):
        return (f'{type(self).__name__}(sampler={self.sampler!r}, n_steps={self.n_steps}, '
                f'chunk={self.chunk}, guidance_scale={self.guidance_scale!r})')

    def __eq__(self, other):
        return type(other) is type(self) and other.settings() == self.settings()

    def check(self, network):
        """What `network` lacks for this method (see :func:`flow_matching_problems`).

        ``velocity`` must accept the keyword ``uncond_mask`` only when
        guidance is on (``guidance_scale != 1``).
        """
        return flow_matching_problems(network, uncond_mask=self.guidance_scale != 1.0)

    def generate(self, network, lam, *, generator=None):
        """``(B,)`` target logits -> ``(B, M, 3)`` Å candidates on the network's device."""
        return generate(network, lam, sampler=self.sampler, n_steps=self.n_steps,
                        chunk=self.chunk, generator=generator,
                        guidance_scale=self.guidance_scale)

    def settings(self):
        """``{'sampler', 'n_steps', 'chunk', 'guidance_scale'}``."""
        return {'sampler': self.sampler, 'n_steps': self.n_steps, 'chunk': self.chunk,
                'guidance_scale': self.guidance_scale}


# a user's replacement of 'ode' registered before this import is kept
_register_builtin('ode', ODEGenerator)
