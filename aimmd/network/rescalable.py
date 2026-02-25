"""
aimmd.network.rescalable
=======================

Rescaling mixin for network modules.

This module defines :class:`Rescalable`, a mixin that augments a `torch.nn.Module`
by applying an optional *output rescaling* step after the normal forward call.

The rescaling is represented by two 1D buffers:

- ``rescale_knots``: knot locations in the original output coordinate,
- ``rescale_values``: corresponding target values in the rescaled coordinate.

The actual transformation is performed by :func:`aimmd.network.rescale_utils.rescale`
and is applied **in-place** to the network output under ``torch.no_grad()``.

Typical usage
-------------
Subclass `Rescalable` together with a standard `nn.Module` implementation:

- define `forward(...)` as usual,
- call :meth:`set_knots_and_values` to enable rescaling.

Rescaling is disabled by default: buffers are filled with NaNs and only entries
up to the first NaN are considered active.

"""

# external
import torch
from abc import ABC
from math import nan
from torch import nn
from collections.abc import Iterable

# internal imports
from .rescale_utils import rescale


# rescalable network module
class Rescalable(ABC, nn.Module):
    """
    Mixin that applies output rescaling at the end of `__call__`.

    This class is intended to be used with multiple inheritance alongside an
    `nn.Module` that implements `forward(...)`. By overriding :meth:`__call__`,
    it ensures:

    1) the usual PyTorch module call path runs first (hooks, autocast, etc.),
    2) the resulting output `q` is optionally rescaled based on stored knots
       and values.

    Rescaling parameters are stored as **buffers** so they are:
    - moved with the module across devices (`.to(...)`),
    - included in module state (unless explicitly filtered),
    - not treated as trainable parameters.

    Notes
    -----
    - The rescaling is applied under ``torch.no_grad()`` and mutates the output
      tensor in-place (as implemented by :func:`rescale`).
    - Rescaling is considered active when `rescale_knots` contains at least one
      non-NaN entry.

    """

    def __init__(self, max_knots=100):
        """
        Parameters
        ----------
        max_knots : int, optional
            Maximum number of knots/values stored in the internal buffers.
            Two buffers of shape ``(max_knots,)`` are registered:
            ``rescale_knots`` and ``rescale_values``. Entries are initialized to
            NaN and only non-NaN entries are used.

        """
        super().__init__()
        # fixed-size buffers; NaN means "inactive slot"
        self.register_buffer("rescale_knots", torch.full((max_knots,), nan))
        self.register_buffer("rescale_values", torch.full((max_knots,), nan))

    def __call__(self, *args, **kwargs):
        """Invoke the module and apply optional output rescaling.

        Parameters
        ----------
        *args, **kwargs
            Forward arguments passed to the next class in the MRO (typically
            `nn.Module.__call__`, which dispatches to `forward` and handles hooks).

        Returns
        -------
        q : torch.Tensor
            The module output, potentially rescaled. If rescaling is active,
            the returned tensor is the same object as produced by the forward
            call but modified in-place by :func:`rescale`.

        """
        # let nn.Module handle forward hooks, autocast, etc.
        q = super().__call__(*args, **kwargs)

        # identify active knot/value entries (non-NaN)
        keepers = ~torch.isnan(self.rescale_knots)
        if keepers.any():
            # slice active portion
            knots = self.rescale_knots[keepers]
            values = self.rescale_values[keepers]
            # apply transformation without tracking gradients
            with torch.no_grad():
                q = rescale(q, knots, values)
        return q

    def set_knots_and_values(self, knots, values):
        """Set the active rescaling definition.

        Parameters
        ----------
        knots : Iterable[float]
            Knot positions in the original coordinate. Must be iterable.
        values : Iterable[float]
            Target values at each knot. Must be iterable and have the same
            length as `knots`.

        Raises
        ------
        TypeError
            If `knots` or `values` are not iterable, or if they do not have the
            same length.

        Notes
        -----
        - Values are copied into the buffers starting at index 0.
        - Only the first ``len(knots)`` entries are overwritten; remaining buffer
          entries are left unchanged. If you need to clear previous knots, call
          :meth:`reset_parameters` first (or set fewer knots after a reset).

        """
        # basic validation (iterability and matching lengths)
        if not isinstance(knots, Iterable):
            raise TypeError(f'knots must be iterable, got {knots!r}')
        if not isinstance(values, Iterable):
            raise TypeError(f'values must be iterable, got {values!r}')
        if len(knots) != len(values):
            raise TypeError(f'knots and values must have the same length, '
                            f'got {len(knots)} != {len(values)}')

        # store as tensors (default dtype/device follow torch.tensor behavior)
        self.rescale_knots[:len(knots)] = torch.tensor(knots)
        self.rescale_values[:len(knots)] = torch.tensor(values)

    def reset_parameters(self):
        """Reset module parameters and clear rescaling buffers.

        This cooperates with multiple inheritance by calling `reset_parameters`
        on the next class in the MRO if it exists, then clears the rescaling
        buffers by filling them with NaNs.

        Notes
        -----
        - Clearing the buffers disables rescaling until new knots/values are set.

        """
        # cooperate with multiple inheritance
        if hasattr(super(), "reset_parameters"):
            super().reset_parameters()

        # clear rescaling buffers
        with torch.no_grad():
            self.rescale_knots.fill_(nan)
            self.rescale_values.fill_(nan)
