Analysis
========

The :mod:`aimmd.analysis` subpackage provides the numerical post-processing used
for adaptive binning and for turning sampled paths into rates and free-energy
profiles: bin construction and merging, binomial confidence intervals, committor
fields by relaxation, rate-estimate extraction from logs, and path-lineage
utilities.

Three further modules read a run folder after (or during) a run and never write
into it: the trainer's reweighting repeated offline (rates, per-frame
equilibrium weights, populations and free energies along any collective
variable), the rates and a committor reference from the free simulations alone,
and a record of every setting of a run. They apply to every chain type,
C-GAIMMD included; import them by name.

.. automodule:: aimmd.analysis.utils
   :members:
   :show-inheritance:

Reweighting, populations, free energies and rates of a run
----------------------------------------------------------

.. automodule:: aimmd.analysis.reweighting

Rates and a committor reference from the free simulations
---------------------------------------------------------

.. automodule:: aimmd.analysis.free_md

Every setting of a run
----------------------

.. automodule:: aimmd.analysis.run_card
