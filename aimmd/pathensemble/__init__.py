"""
aimmd.pathensemble
=================

Path-ensemble container and reweighting utilities.

This subpackage defines :class:`~aimmd.pathensemble.PathEnsemble`, AIMMD's main
high-level container for *path sampling* data. A ``PathEnsemble`` holds a list
of :class:`~aimmd.path.Path` objects (or path-like inputs that can be converted
to ``Path``), and provides:

- **collection**: load, store, slice, merge, and sample path data,
- **projection**: build weighted histograms of user-defined observables along
  paths (e.g., a committor proxy, CVs, descriptors),
- **reweighting**: compute statistically meaningful weights for paths sampled
  with shooting / path sampling, and derive factors/crossing probabilities.

Design
------
``PathEnsemble`` is implemented as a composition of small mixins to keep concerns
separate (I/O, properties, magic methods, projection, reweighting, reporting).

The public API is the :class:`PathEnsemble` class exported by this module.
"""

# public mixins (assembled below)
from ._io import PathEnsembleIO
from ._magic import PathEnsembleMagic
from ._report import PathEnsembleReport
from ._helpers import PathEnsembleHelpers
from ._methods import PathEnsembleMethods
from ._project import PathEnsembleProject
from ._reweight import PathEnsembleReweight
from ._positions import PathEnsemblePositions
from ._properties import PathEnsembleProperties


# -----------------------------------------------------------------------------
# Public class
# -----------------------------------------------------------------------------
class PathEnsemble(
    PathEnsembleHelpers,
    PathEnsembleMagic,
    PathEnsembleProperties,
    PathEnsembleMethods,
    PathEnsemblePositions,
    PathEnsembleProject,
    PathEnsembleReweight,
    PathEnsembleIO,
    PathEnsembleReport):
    """
    Container for a collection of sampled paths.

    The main scope of this class is to **collect and reweight paths sampled with
    path sampling** (e.g., shooting-based approaches). It provides a list-like
    interface over underlying :class:`~aimmd.path.Path` objects, and adds
    ensemble-level operations such as:

    - selection by type/pattern, splitting and merging,
    - computing ensemble-level properties (lengths, accepted flags, weights),
    - projecting observables into weighted histograms,
    - reweighting excursions and internal segments.

    Notes
    -----
    The actual initialization logic lives in :meth:`PathEnsembleHelpers._init`
    and is aliased here to keep the top-level class minimal.
    """

    # NOTE: the "real" constructor is implemented in the helpers mixin.
    __init__ = PathEnsembleHelpers._init


__all__ = ['PathEnsemble']
