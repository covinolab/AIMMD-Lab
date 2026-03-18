Analysis Package
================

The :mod:`aimmd.analysis` package is intentionally small and focused. Most of
the numerical analysis support lives in :mod:`aimmd.analysis.utils`.

What It Covers
--------------

``compute_bins``
   Build adaptive value-space bins used by training and sampling.

``merge_empty_bins`` and ``merge_marginal_bins``
   Post-process bin layouts when data are sparse.

``binomial_mean_and_confidence_interval``
   Small statistical helper for outcome probabilities.

``solve_committor_by_relaxation``
   Grid-based committor solver for analysis and validation workflows.

``find_path_lineages`` and ``plot_path_lineages``
   Path-chain lineage reconstruction and visualization helpers.

See Also
--------

The full autodoc pages for this layer live in :doc:`lowlevel/analysis`.
