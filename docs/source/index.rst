AIMMD
=====

**AIMMD** (*AI for Molecular Mechanism Discovery*) is a Python package for
**AI-enhanced path sampling** of rare molecular transitions. Short unbiased
simulations are shot from points chosen by a learned committor model, producing
a diverse ensemble of reactive trajectories — often many more transition events
than plain equilibrium sampling at comparable cost. That ensemble is reweighted
to recover free-energy profiles and transition rates, and the learned committor
doubles as an interpretable reaction coordinate. Runs scale from a laptop
(multiprocessing) to HPC clusters (SLURM ``srun``), driving GROMACS or a
built-in toy engine.

Highlights
----------

- **Committor-guided two-way shooting** in the reactive region.
- **Rejection-free path sampling** (``rfps``) *and* classic TPS acceptance (``tps``).
- **On-the-fly training and reweighting**: the committor model, adaptive bins,
  and densities update continuously as data arrives.
- **Free energies and rates** from a single self-consistent path ensemble.
- **Multi-system / multi-ligand runs** with one shared committor network.
- **Biased dynamics** (OPES/PLUMED) with Tiwary-Parrinello rate reweighting.
- **Graph-neural-network** committor models (optional).
- **Local or HPC** execution from the same parameter file.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Install AIMMD and run your first toy example end to end.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Executable notebooks: a 1-D toy system and a multi-ligand run.

   .. grid-item-card:: Concepts
      :link: concepts
      :link-type: doc

      The method and the objects that implement it.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      The five public classes and helper subpackages.

To cite AIMMD, see :doc:`scientific_background`. AIMMD is released under the MIT
License.

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: User Guide

   concepts
   workflow
   parameters
   advanced

.. toctree::
   :hidden:
   :caption: Background

   scientific_background

.. toctree::
   :hidden:
   :caption: Reference

   api/index

.. toctree::
   :hidden:
   :caption: Developer

   developer_guide
