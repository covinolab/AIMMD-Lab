Execution and Orchestration
===========================

Workers
-------

:class:`aimmd.Worker` is the atomic runtime unit. Its internal modules split
the worker implementation by concern:

:mod:`aimmd.worker._helpers`
   Initialization, signal handling, resource binding, and progress helpers.

:mod:`aimmd.worker._properties`
   Accessors for stop conditions, logging, and initial paths.

:mod:`aimmd.worker._run`
   Task dispatch.

:mod:`aimmd.worker._simulate`
   Engine-facing simulation update loop.

:mod:`aimmd.worker._shoot`
   Committor-guided two-way shooting.

:mod:`aimmd.worker._free`
   Unbiased free simulations around states.

:mod:`aimmd.worker._train`
   Network training, value recomputation, binning, and density updates.

:mod:`aimmd.worker._magic` and :mod:`aimmd.worker.utils`
   Support code and reusable helper routines for worker tasks.

Launchers
---------

:class:`aimmd.Launcher` coordinates one or more workers and one or more runs.
Its implementation lives in:

:mod:`aimmd.launcher._helpers`
   Setup, normalization, and resource allocation.

:mod:`aimmd.launcher._properties`
   Read-only accessors.

:mod:`aimmd.launcher._methods`
   High-level user actions such as adding runs and writing SLURM job scripts.

:mod:`aimmd.launcher._build`
   Directory construction and worker-argument assembly.

:mod:`aimmd.launcher._run`
   Local execution and fail-fast supervision.

:mod:`aimmd.launcher._magic`
   Minimal container-like behavior.

Engines
-------

:mod:`aimmd.engines.toy`
   Pure-Python toy engine used in tests and development.

:mod:`aimmd.engines`
   Bundled GROMACS minimization template exposed via :mod:`aimmd.engines`.

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/execution`.
