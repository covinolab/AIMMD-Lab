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
"""

from .utils import *

