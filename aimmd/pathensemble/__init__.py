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

# public API surface for the pathensemble package
# PathEnsemble is a thin aggregator of mixins; the real implementation
# lives in the private modules imported below.

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
    Collection of :class:`~aimmd.path.Path` objects with vectorized operations.

    A :class:`PathEnsemble` is a container for paths used throughout AIMMD to
    represent:

    - shooting chains (ordered sequences of sampled paths),
    - selection pools (bounded candidate sets for shooting-point selection),
    - free-trajectory ensembles (collections of long unbiased trajectories),
    - merged or extracted ensembles used for analysis and training.

    In addition to list-like behavior, PathEnsemble provides convenience methods
    that operate *across* all member paths, often in a vectorized / batched way:

    - compute per-frame quantities (states/descriptors/values) for the ensemble,
    - project values into bins and estimate densities,
    - reweight paths according to sampling schemes,
    - extract subsets by path type patterns or by end-state transitions,
    - merge paths into a single frame collection (for sweep shooting and
      reporting).

    Representation
    -------------
    An ensemble stores paths (typically in a list-like attribute such as
    ``_paths``). Many performance-critical operations may operate directly on
    this internal structure.

    Each path carries segment metadata and refers to trajectory files on disk.
    Ensemble methods therefore often interact with the global caches:

    - ``aimmd._config.MDA_CACHE`` for trajectory readers,
    - ``aimmd._config.NPY_CACHE`` for cached `.npy` arrays.

    Typical usage in AIMMD
    ----------------------
    Shooting
        The worker maintains a shooting chain as a PathEnsemble. New paths are
        appended via registration utilities. A separate PathEnsemble is used as
        the selection pool.

    Training
        A PathEnsemble assembled from current chains and free trajectories is
        used to compute values and to fit the committor model. Projection and
        reweighting are performed on the ensemble to produce adaptive bins and
        densities.

    Validation / sweep
        A merged ensemble (frame collection) is used to deterministically select
        frames and repeatedly shoot from them for brute-force committor
        estimation.

    Notes
    -----
    - Many ensemble operations assume that per-path cached arrays (states,
      descriptors, values) exist or can be computed on demand, and that their
      frame ordering matches the path’s frame ordering.
    - Some methods intentionally mutate paths in place (e.g., attaching weights,
      updating cached attributes). This is acceptable in worker contexts where
      the ensemble is treated as a live view of the filesystem-backed run state.
    """
    
    # bind the mixin initializer as the class initializer
    __init__ = PathEnsembleHelpers._init


__all__ = ["PathEnsemble"]
