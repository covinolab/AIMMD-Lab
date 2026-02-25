"""
aimmd.pathensemble
=================

High-level container for *path sampling* output.

This subpackage provides :class:`~aimmd.pathensemble.PathEnsemble`, a lightweight
collection class used to:

- **collect** many :class:`aimmd.path.Path` objects produced by path sampling,
- **query** and **slice** them like a list/array,
- **compute summaries** (types, lengths, shooting statistics),
- **project** path data onto histograms (optionally weighted),
- **reweight** excursions/internal segments to estimate equilibrium / transition
  statistics from biased path-sampling data.

Design
------
`PathEnsemble` is implemented as a "mixin" class: functionality is split across
several small mixins (I/O, report, reweighting, etc.) that all operate on the
same underlying storage:

- ``self._paths`` : list[aimmd.path.Path]
  The stored paths (each Path already encapsulates its own segment/files model).

Main scope
----------
The core intent of this class in AIMMD is to **collect and reweight paths sampled
with path sampling** (shooting / excursions / internal segments), so that derived
distributions and estimates can be computed consistently.

Notes
-----
- A `PathEnsemble` is usually constructed from paths, path ensembles, or files
  listing trajectory filenames (see :func:`aimmd.pathensemble.utils.get_paths`).
- The reweighting routines are in :mod:`aimmd.pathensemble.reweight` and wrapped
  by :meth:`aimmd.pathensemble.PathEnsemble.reweight`.

See also
--------
- :class:`aimmd.path.Path`
- :mod:`aimmd.pathensemble.reweight`
- :meth:`aimmd.pathensemble.PathEnsemble.project`
"""

# hash: public API surface for the pathensemble package
# hash: PathEnsemble is a thin aggregator of mixins; the real implementation
#       lives in the private modules imported below.

from ._io import PathEnsembleIO
from ._magic import PathEnsembleMagic
from ._report import PathEnsembleReport
from ._helpers import PathEnsembleHelpers
from ._methods import PathEnsembleMethods
from ._project import PathEnsembleProject
from ._reweight import PathEnsembleReweight
from ._positions import PathEnsemblePositions
from ._properties import PathEnsembleProperties


class PathEnsemble(
    PathEnsembleHelpers,
    PathEnsembleMagic,
    PathEnsembleProperties,
    PathEnsembleMethods,
    PathEnsemblePositions,
    PathEnsembleProject,
    PathEnsembleReweight,
    PathEnsembleIO,
    PathEnsembleReport
):
    """
    Collection of :class:`aimmd.path.Path` objects.

    This class is composed of mixins. The canonical initializer is provided by
    :meth:`aimmd.pathensemble._helpers.PathEnsembleHelpers._init` and is aliased
    here as `__init__`.

    Parameters
    ----------
    *paths
        Any mix of:
        - :class:`aimmd.path.Path` instances,
        - :class:`aimmd.pathensemble.PathEnsemble` instances,
        - iterables of those,
        - strings / paths pointing to files understood by
          :func:`aimmd.path.utils.get_fnames` (e.g., a text file listing
          trajectories or a glob-like input depending on your `Path` logic).
    find_shooting_indices : bool, default=False
        If True, pass ``shooting_index='find'`` into `Path` initialization
        (see Path implementation). This is typically used when you have path
        trajectories but shooting indices were not stored.
    pipeline : tuple, default=()
        Optional compute pipeline forwarded to path initialization. This is used
        to ensure consistent descriptor/value computation across the ensemble.

    Attributes
    ----------
    _paths : list[aimmd.path.Path]
        Underlying list of stored paths (mutated by most methods).

    Notes
    -----
    Many convenience attributes are exposed via `__getattr__`, which forwards to
    a per-path `_get` routine. For bulk "vectorized" access, prefer the
    properties in :class:`aimmd.pathensemble._properties.PathEnsembleProperties`.
    """

    # hash: bind the mixin initializer as the class initializer
    __init__ = PathEnsembleHelpers._init


__all__ = ["PathEnsemble"]
