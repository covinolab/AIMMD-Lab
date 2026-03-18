Public Surface
==============

Top-Level Entry Points
----------------------

The package-level imports in :mod:`aimmd` expose the main user-facing objects:

- :class:`aimmd.Params`
- :class:`aimmd.Path`
- :class:`aimmd.PathEnsemble`
- :class:`aimmd.Worker`
- :class:`aimmd.Launcher`
- :mod:`aimmd.utils`

The package import also triggers :mod:`aimmd._init`, which populates the shared
runtime state stored in :mod:`aimmd._config`.

Internal Bootstrap Modules
--------------------------

``aimmd.__main__``
   Minimal multiprocessing-safe entry point for ``python -m aimmd``.

``aimmd._init``
   Import-time initialization routine that resolves executables, creates
   caches, patches a few runtime behaviors, and populates configuration.

``aimmd._config``
   Module-level singleton storage for runtime-resolved paths, caches, and
   defaults.

Main Object Relationships
-------------------------

``Params`` -> ``Launcher`` and ``Worker``
   The parameter object is consumed by both orchestration and worker execution.

``Path`` -> ``PathEnsemble``
   Individual paths are grouped into ensembles for reporting, projection, and
   reweighting.

``Worker`` -> ``Path`` / ``PathEnsemble``
   Workers write trajectories and update the ensemble stored on disk.

``Launcher`` -> ``Worker``
   The launcher builds and supervises one or more worker processes.

See Also
--------

The full autodoc pages for this layer live in :doc:`lowlevel/public`.
