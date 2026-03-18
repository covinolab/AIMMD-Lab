Parameters Package
==================

Primary Class
-------------

:class:`aimmd.Params` is the central configuration object for AIMMD runs.

Internal Organization
---------------------

The implementation is split into mixins:

:mod:`aimmd.params._fields`
   Declares the full parameter schema and default values.

:mod:`aimmd.params._helpers`
   Initialization, validation, and normalization helpers.

:mod:`aimmd.params._properties`
   Derived properties and cached accessors.

:mod:`aimmd.params._methods`
   Engine-facing operations such as simulation initialization and network
   updates from disk.

:mod:`aimmd.params._paths`
   Run-loading helpers that reconstruct chains, free trajectories, and path
   ensembles from a working directory.

:mod:`aimmd.params._io`
   Save/load support for Python-based parameter files.

:mod:`aimmd.params._magic`
   ``repr``/comparison/attribute-management behavior.

Supporting Utility Module
-------------------------

:mod:`aimmd.params.utils`
   Source-tracking helpers and small utilities used by the persistence layer.

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/params`.
