Package and configuration
=========================

Importing :mod:`aimmd` runs :func:`aimmd._init.initialize` once, which resolves
the Python interpreter and the GROMACS executable, computes package paths,
instantiates the shared reader/array caches, and applies a few runtime tweaks.
If neither ``gmx`` nor ``gmx_mpi`` is on ``PATH``, import raises
``EnvironmentError``.

Top-level names
---------------

``aimmd.__version__``
   The package version string.

``aimmd.utils``
   Alias of :mod:`aimmd.core.utils` (general indexing/concatenation helpers).

The public classes :class:`~aimmd.Params`, :class:`~aimmd.Path`,
:class:`~aimmd.PathEnsemble`, :class:`~aimmd.Worker`, and :class:`~aimmd.Launcher`
are documented on their own pages.

Configuration constants
------------------------

These are populated at import time and re-exported at the package top level:

``GROMACS``
   Absolute path to the resolved ``gmx`` / ``gmx_mpi`` executable.

``PARENT``
   Absolute path to the installed ``aimmd`` package directory.

``WORKER``
   Path to ``aimmd/worker/run.py``, the standalone worker launcher invoked by
   ``srun`` in generated SLURM scripts.

``EM_MDP``
   Path to the bundled energy-minimization ``.mdp`` file used by the GROMACS engine.

``NPY_CACHE`` / ``MDA_CACHE``
   Process-wide singleton caches for NumPy arrays and MDAnalysis trajectory
   readers, respectively.
