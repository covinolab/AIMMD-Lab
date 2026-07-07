Concepts
========

What AIMMD Does
---------------

AIMMD stands for *AI for Molecular Mechanism Discovery*. Its core purpose is to
sample rare transitions far more efficiently than straightforward equilibrium
molecular dynamics, by combining:

- path sampling based on two-way shooting,
- a learned committor-like model that guides *where* to shoot,
- free simulations near metastable states, and
- reweighting machinery that maps the sampled paths back to equilibrium
  observables.

At a high level, AIMMD runs a feedback loop:

1. Start from one or more initial transition paths.
2. Shoot new trajectories from selected frames in or near the reactive region.
3. Train a neural network to predict a committor-like coordinate from the data
   collected so far.
4. Use the updated model to rebalance where future shooting points are chosen.
5. Reweight the resulting path ensemble to estimate free energies, rates, and
   mechanisms.

Because the committor model improves as data accumulates and immediately feeds
back into shooting-point selection, the reactive region is explored
increasingly uniformly — the basis of the rejection-free path sampling scheme
(see :doc:`scientific_background`).

Core Objects
------------

The package is organized around a small set of high-level objects (documented in
full in the :doc:`api/index`):

:class:`aimmd.Params`
   Central configuration object. It stores state definitions, engine settings,
   descriptor and value functions, the neural network, training hooks, sampling
   controls, and path-loading helpers. A run is defined by a ``params.py`` file
   loaded into a ``Params`` instance.

:class:`aimmd.Path`
   Lightweight representation of a trajectory-backed path. A path stores frame
   indexing metadata and filenames rather than duplicating heavy trajectory data
   in memory; states, descriptors, and committor values are cached lazily beside
   the trajectory.

:class:`aimmd.PathEnsemble`
   Container for many paths. It supports slicing, filtering by path type,
   projection, reporting, and reweighting into equilibrium observables.

:class:`aimmd.Worker`
   Atomic execution unit. A worker runs exactly one task at a time: ``shoot``,
   ``free``, or ``train``.

:class:`aimmd.Launcher`
   Orchestrator for one or more runs. It builds directory layouts, allocates
   worker resources, launches local workers, and can emit SLURM job scripts.

The typical path from configuration to results is covered in :doc:`quickstart`
and, in more depth, in :doc:`workflow`.
