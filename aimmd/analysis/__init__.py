"""
aimmd.analysis
==============

Analysis utilities used across AIMMD.

This subpackage contains small numerical helpers used by training and
post-processing, in particular routines for:

- defining bin edges for projections / committor-like analysis,
- handling bins that include marginal ``±inf`` boundaries,
- computing simple confidence intervals for binomial outcomes,
- solving a 2D committor field by relaxation on a grid.

The implementations live in :mod:`aimmd.analysis.utils`.

Three modules read a run folder after (or during) a run, without writing into
it; import them by name:

- :mod:`aimmd.analysis.reweighting`: the trainer's reweighting repeated
  offline: rates, per-frame equilibrium weights, populations, free energies
  along any collective variable, and how they change with the amount of data;
  with a static bias, the bias-reweighted rates and unbiased weights;
- :mod:`aimmd.analysis.free_md`: rates counted in the free simulations (with
  exact Poisson limits), and a committor reference from where their reactive
  frames went next;
- :mod:`aimmd.analysis.run_card`: every setting of a run in one record, for
  parameter tables.
"""

from .utils import *

