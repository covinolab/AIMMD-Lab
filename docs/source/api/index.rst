API Reference
=============

AIMMD has a deliberately small public surface: five high-level classes plus a
handful of helper subpackages. Each class is assembled internally from several
mixin modules (see :doc:`../developer_guide`), but is documented here as the
single composed class you actually use, including its inherited members.

Core classes
------------

.. autosummary::
   :nosignatures:

   aimmd.Params
   aimmd.Path
   aimmd.PathEnsemble
   aimmd.Worker
   aimmd.Launcher

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - Role
   * - :class:`aimmd.Params`
     - Central run configuration: states, engine, network, data functions, sampling controls.
   * - :class:`aimmd.Path`
     - A single trajectory-backed transition path (segment-aware, lazily cached).
   * - :class:`aimmd.PathEnsemble`
     - A collection of paths with projection, reporting, and reweighting.
   * - :class:`aimmd.Worker`
     - Atomic execution unit running one ``shoot`` / ``free`` / ``train`` task.
   * - :class:`aimmd.Launcher`
     - Orchestrates runs: builds directories, allocates resources, launches workers or emits SLURM scripts.

.. toctree::
   :hidden:
   :caption: Core classes

   params
   path
   pathensemble
   worker
   launcher

.. toctree::
   :hidden:
   :caption: Subpackages

   network
   analysis
   utilities
   config
