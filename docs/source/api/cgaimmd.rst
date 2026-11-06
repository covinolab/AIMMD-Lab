C-GAIMMD
========

The API of :mod:`aimmd.cgaimmd`, the tooling of committor-conditioned Generative
AIMMD (``chain_type='cgaimmd'``). The method and how to run it are described
in :doc:`../cgaimmd` and :doc:`../cgaimmd_production`; how the pieces fit
together is in :doc:`../cgaimmd_internals`. The settings live in three
:class:`aimmd.Params` fields: ``cgaimmd_parameters`` (read through
``Params.cgaimmd_settings``), ``cgaimmd_generator`` and the
``cgaimmd_*_function`` hooks (:doc:`../parameters`).

.. automodule:: aimmd.cgaimmd
   :no-members:

The reference is split by what the code does:

- :doc:`cgaimmd_core`: the network protocol, the generator interface and its
  registry, the shared types, devices, file writes, the training data of a
  round and what the online fits share;
- :doc:`cgaimmd_shooting`: the shooting-point source of a shooting worker,
  the target-committor laws, the candidate pool, the per-shot records and
  the shot table, restarts and post-training, the preflight and the PLUMED
  bias of the shots;
- :doc:`cgaimmd_preparation`: the GROMACS tools that turn a generated
  structure into a shooting point (descriptors, validity checks, the polish,
  re-solvation, minimization, topology and symmetry);
- :doc:`cgaimmd_flow_matching`: the PaiNN flow-matching network, its ODE
  generator ``'ode'``, its fit and its end-state pretraining;
- :doc:`cgaimmd_bioemu2`: the BioEmu2 adaptor (the ``bioemu2`` extra), its
  generator ``'bioemu2_edm'``, its network, fit and system folder.
- :doc:`cgaimmd_analysis`: the tools that judge a model and its generator
  after a run (committor parity against brute-force sets, generator
  adherence, shooting validation, a distance-matrix committor baseline).

.. toctree::
   :hidden:

   cgaimmd_core
   cgaimmd_shooting
   cgaimmd_preparation
   cgaimmd_flow_matching
   cgaimmd_bioemu2
   cgaimmd_analysis
