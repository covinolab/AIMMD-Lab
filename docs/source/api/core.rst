Core Infrastructure
===================

Core Utilities
--------------

The foundational utilities are concentrated in :mod:`aimmd.core`:

:mod:`aimmd.core.utils`
   General-purpose helpers for array handling, frame indexing, state parsing,
   path naming, and trajectory-related convenience routines.

:mod:`aimmd.core.base`
   Base abstractions used by array-like containers in the codebase.

:mod:`aimmd.core.decorators`
   Small decorators such as the custom ``classproperty`` and mixed
   class-or-instance method support.

Caches
------

:mod:`aimmd.cache.base`
   Abstract cache interface.

:mod:`aimmd.cache.npy`
   Safe concurrent access to cached ``.npy`` arrays.

:mod:`aimmd.cache.mda`
   Robust opening and caching of MDAnalysis trajectory readers.

Execution Helpers
-----------------

:mod:`aimmd.execute.utils`
   Shell-command execution with cooperative stopping.

:mod:`aimmd.execute.base`
   Base executor abstraction.

:mod:`aimmd.execute.threads` and :mod:`aimmd.execute.processes`
   Lightweight thread-based and process-based executors used by the runtime.

Resource Helpers
----------------

:mod:`aimmd.resources.cpu` and :mod:`aimmd.resources.gpu`
   Inspect the CPU and GPU resources visible to the current process.

:mod:`aimmd.resources.binding`
   Bind a worker process to a local CPU/GPU subset.

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/core`.
