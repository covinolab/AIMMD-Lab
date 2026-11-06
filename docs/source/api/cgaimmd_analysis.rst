C-GAIMMD: judging a model and its generator
===========================================

The tools that judge a C-GAIMMD model after (or during) a run: the accuracy of
its committor against brute-force shooting, how closely its generator follows
the committor it is asked for, whether generated configurations have that
committor when shot from, and a plain committor model trained on the same shots
to compare with. Rates and free energies of a run are generic AIMMD analysis
(:doc:`analysis`). None of these tools writes into a run folder: results are
returned, and files are written only where the caller names them.

.. automodule:: aimmd.cgaimmd.analysis
   :no-members:

Committor parity and the validation protocol
--------------------------------------------

.. automodule:: aimmd.cgaimmd.analysis.validate

Generator adherence
-------------------

.. automodule:: aimmd.cgaimmd.analysis.adherence

Shooting validation of generated configurations
-----------------------------------------------

.. automodule:: aimmd.cgaimmd.analysis.shooting_validation

Distance-matrix committor baseline
----------------------------------

.. automodule:: aimmd.cgaimmd.analysis.committor_baseline
