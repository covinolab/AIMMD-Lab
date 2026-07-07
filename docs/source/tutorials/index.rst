Tutorials
=========

These Jupyter notebooks run end to end on the built-in toy engine — no GROMACS
and no GPU required. They are rendered here from their **saved outputs**; to run
them yourself, launch them from ``examples/notebooks/`` in a clone of the
repository.

.. toctree::
   :maxdepth: 1

   1_toy_1d
   2_multi_system

- **1D toy system** (:doc:`1_toy_1d`) — a complete single-system AIMMD run on a
  1-D double-well toy model, from configuration through launching, training, and
  analysis of the resulting path ensemble.
- **Multi-system run** (:doc:`2_multi_system`) — two toy systems with different
  atom counts in one run, trained with a single shared committor network
  (contrasted with per-system networks), including per-system committor and rate
  analysis.
