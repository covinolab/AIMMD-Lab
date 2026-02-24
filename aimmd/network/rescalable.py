"""
...
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
    Mixin that applies output rescaling at the end of __call__.
    Can be mixed into any nn.Module.
    """
    
    def __init__(self, max_knots=100):
        super().__init__()
        self.register_buffer("rescale_knots", torch.full((max_knots,), nan))
        self.register_buffer("rescale_values", torch.full((max_knots,), nan))
    
    def __call__(self, *args, **kwargs):
        # let nn.Module handle forward hooks, autocast, etc.
        q = super().__call__(*args, **kwargs)

        keepers = ~torch.isnan(self.rescale_knots)
        if keepers.any():
            knots = self.rescale_knots[keepers]
            values = self.rescale_values[keepers]
            with torch.no_grad():
                q = rescale(q, knots, values)
        return q

    def set_knots_and_values(self, knots, values):
        if not isinstance(knots, Iterable):
            raise TypeError(f'knots must be iterable, got {knots!r}')
        if not isinstance(values, Iterable):
            raise TypeError(f'values must be iterable, got {values!r}')
        if len(knots) != len(values):
            raise TypeError(f'knots and values must have the same length, '
                            f'got {len(knots)} != {len(values)}')
        self.rescale_knots[:len(knots)] = torch.tensor(knots)
        self.rescale_values[:len(knots)] = torch.tensor(values)
    
    def reset_parameters(self):
        # cooperate with multiple inheritance
        if hasattr(super(), "reset_parameters"):
            super().reset_parameters()

        with torch.no_grad():
            self.rescale_knots.fill_(nan)
            self.rescale_values.fill_(nan)
