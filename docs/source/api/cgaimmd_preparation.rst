C-GAIMMD: candidate preparation
===============================

The GROMACS tools that turn a generated configuration of the generator atoms
into a shooting point of the run's full system. A params file wraps them in
its hooks: ``descriptors_function``
(:class:`~aimmd.cgaimmd.preparation.descriptors.GeneratorDescriptors`),
``cgaimmd_validity_function``
(:class:`~aimmd.cgaimmd.preparation.validity.TopologyValidity` or
:class:`~aimmd.cgaimmd.preparation.validity.PolishedValidity`),
``cgaimmd_prepare_function``
(:class:`~aimmd.cgaimmd.preparation.solvate.GromacsSolvator`) and
``cgaimmd_minimize_function``
(:class:`~aimmd.cgaimmd.preparation.minimize.GromacsMinimizer`).

.. automodule:: aimmd.cgaimmd.preparation
   :no-members:

Generator atoms and descriptors
-------------------------------

.. automodule:: aimmd.cgaimmd.preparation.descriptors

Validity checks
---------------

.. automodule:: aimmd.cgaimmd.preparation.validity

Minimization and the polish
---------------------------

.. automodule:: aimmd.cgaimmd.preparation.minimize

Re-solvation
------------

.. automodule:: aimmd.cgaimmd.preparation.solvate

GROMACS topology
----------------

.. automodule:: aimmd.cgaimmd.preparation.topology

Permutation symmetries
----------------------

.. automodule:: aimmd.cgaimmd.preparation.symmetry

GROMACS plumbing
----------------

.. automodule:: aimmd.cgaimmd.preparation.gromacs
