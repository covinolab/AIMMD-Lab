Overview
========

What AIMMD Does
---------------

AIMMD stands for *AI for Molecular Mechanism Discovery*. Its core purpose is to
sample rare transitions more efficiently than straightforward equilibrium
molecular dynamics by combining:

- path sampling based on two-way shooting,
- a learned committor-like model that guides where to shoot,
- free simulations near metastable states,
- and reweighting machinery that maps the biased sampled paths back to
  equilibrium observables.

At a high level, AIMMD creates a feedback loop:

1. Start from one or more initial transition paths.
2. Shoot new trajectories from selected frames in or near the reactive region.
3. Train a neural network to predict a committor-like coordinate from the data
   collected so far.
4. Use the updated model to rebalance where future shooting points are chosen.
5. Reweight the resulting path ensemble to estimate free energies, rates, and
   mechanisms.

Core Objects
------------

The repository is organized around a small set of high-level objects:

:class:`aimmd.Params`
   Central configuration object. It stores state definitions, engine settings,
   descriptor and value functions, the neural network, training hooks, sampling
   controls, and path-loading helpers.

:class:`aimmd.Path`
   Lightweight representation of a trajectory-backed path. A path stores frame
   indexing metadata and filenames rather than duplicating heavy trajectory
   data in memory.

:class:`aimmd.PathEnsemble`
   Container for many paths. It supports slicing, extraction by path type,
   projection, reporting, and reweighting.

:class:`aimmd.Worker`
   Atomic execution unit. A worker runs exactly one task at a time:
   ``shoot``, ``free``, or ``train``.

:class:`aimmd.Launcher`
   Orchestrator for one or more runs. It builds directory layouts, allocates
   worker resources, launches local workers, and can emit SLURM job scripts.

Main Workflow in the Code
-------------------------

The implementation mirrors the sampling logic described in the papers and the
README:

- :meth:`aimmd.Worker.shoot` workers generate new excursions and transition paths by two-way
  shooting.
- :meth:`aimmd.Worker.free` workers provide unbiased trajectory segments around states.
- :meth:`aimmd.Worker.train` workers fit or refresh the committor model, recompute values,
  build adaptive bins, and update densities used by the shooting logic.

The path objects written to disk are then assembled into path ensembles for
analysis and reweighting.

Repository Map
--------------

The most important packages are:

:mod:`aimmd.params`
   Parameter schema, validation, persistence, and run-loading helpers.

:mod:`aimmd.path` and :mod:`aimmd.pathensemble`
   Filesystem-backed trajectory containers and ensemble-level operations.

:mod:`aimmd.worker` and :mod:`aimmd.launcher`
   Runtime execution and orchestration.

:mod:`aimmd.network`
   Committor-model training, rescaling, and optional graph-related utilities.

:mod:`aimmd.analysis`
   Analysis helpers for binning, confidence intervals, committor grids, and
   path-lineage utilities.

:mod:`aimmd.resources` and :mod:`aimmd.execute`
   CPU/GPU introspection, resource binding, shell-command execution, and simple
   thread/process executors.

:mod:`aimmd.cache`
   Robust caches for trajectory readers and NumPy arrays.

:mod:`aimmd.engines`
   Engine-facing code, including the pure-Python toy engine used by tests and
   development workflows.
