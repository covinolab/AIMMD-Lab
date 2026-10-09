Utilities
=========

Lower-level helper subpackages used throughout AIMMD. Most users never call
these directly, but they are part of the public surface.

Resources
---------

CPU/GPU introspection and process resource binding.

.. automodule:: aimmd.resources.cpu
   :members:

.. automodule:: aimmd.resources.gpu
   :members:

.. automodule:: aimmd.resources.binding
   :members:

Execution
---------

Streaming subprocess execution and simple thread/process executors.

.. automodule:: aimmd.execute.utils
   :members:

.. automodule:: aimmd.execute.processes
   :members:

.. automodule:: aimmd.execute.threads
   :members:

Caching
-------

Robust, lock-protected caches for trajectory readers and NumPy arrays.

.. automodule:: aimmd.cache.npy
   :members:

.. automodule:: aimmd.cache.mda
   :members:

Core utilities
--------------

.. automodule:: aimmd.core.utils
   :members:

Descriptor series of a run
--------------------------

The check, before a job, that every trajectory of a run has the params'
descriptor series, and the refill of a series in one process of the run.

.. automodule:: aimmd.core.series
   :members: find_trajectories, series_coverage, SeriesCoverage,
             register_series, series_policy, SeriesPolicy,
             check_series_coverage, ensure_series_coverage,
             missing_series_message, run_folder, SeriesCoverageError

Engines
-------

The pure-Python toy engine used by the tests and tutorials.

.. automodule:: aimmd.engines.toy
   :members:
