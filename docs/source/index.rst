AIMMD Documentation
===================

.. rst-class:: hero

AIMMD implements AI-enhanced path sampling for rare-event molecular dynamics.
The repository combines simulation orchestration, path data structures,
committor-model training, and reweighting utilities so that one workflow can
produce transition paths, update the learned reaction coordinate, and recover
mechanistic and thermodynamic observables.

This documentation is organized around how the codebase may be used in scientific practice:

- what the method is trying to compute,
- how the repository is structured,
- how runs are configured and launched,
- how outputs are analyzed and reweighted,
- and where the main APIs live.

.. rst-class:: quicklinks

Start here:

- :doc:`overview`
- :doc:`installation`
- :doc:`workflow`
- :doc:`api/index`

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   installation
   scientific_background
   workflow
   parameters
   examples
   developer_guide
   api/index
