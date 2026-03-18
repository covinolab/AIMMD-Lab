Path Ensemble Package
=====================

Primary Class
-------------

:class:`aimmd.PathEnsemble` stores collections of :class:`aimmd.Path` objects
and provides ensemble-level operations.

Internal Modules
----------------

:mod:`aimmd.pathensemble._helpers`
   Construction and normalization.

:mod:`aimmd.pathensemble._properties`
   Aggregate ensemble properties.

:mod:`aimmd.pathensemble._methods`
   Extraction, joining, sampling, and type-based queries.

:mod:`aimmd.pathensemble._positions`
   Position-based reductions across all paths.

:mod:`aimmd.pathensemble._project`
   Projection of path data into histograms and bins.

:mod:`aimmd.pathensemble._reweight`
   High-level reweighting interface.

:mod:`aimmd.pathensemble._report`
   Reporting helpers for chain and sweep statistics.

:mod:`aimmd.pathensemble._io`
   Load/save helpers for path ensembles.

:mod:`aimmd.pathensemble._magic`
   Basic collection behavior.

Statistical Support
-------------------

:mod:`aimmd.pathensemble.reweight`
   Array-level numerical routines used by the high-level reweighting logic.

:mod:`aimmd.pathensemble.utils`
   Assembly and projection helpers used across the package.

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/pathensemble`.
