"""
aimmd.network
=============

Neural-network utilities for AIMMD.

This subpackage contains lightweight components used to train and apply neural
networks that predict the logit committor (or committor-like coordinates) from
AIMMD path-sampling data.

Public API
----------
fit
    Train ``aimmd.Params.network`` from a :class:`~aimmd.pathensemble.PathEnsemble`.
placeholder
    Default stateless network object used when no user model is configured.
Rescalable
    Mixin that applies an optional output rescaling step after a module call.
rescale
    Piecewise-linear rescaling function used by :class:`Rescalable` and related
    training/analysis utilities.

Notes
-----
- Only a small subset of objects is re-exported here to keep the public surface
  stable.
- Training modifies the network in-place; see :func:`aimmd.network.fit.fit`.

"""

from .fit import fit
from .utils import placeholder
from .rescalable import Rescalable
from .rescale_utils import rescale
