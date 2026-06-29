Examples and Tests
==================

Tests as Executable Documentation
---------------------------------

The repository includes both unit tests and integrations.
The integration tests are a good way to understand how the repository is intended
to be used because they exercise realistic workflows instead of only unit-level
helpers.

``tests/test_params.py``
   Focuses on parameter loading, persistence, source tracking, updates, and
   relative-path behavior.

``tests/test_toy_1d.py``
   Runs an end-to-end AIMMD workflow on the toy engine. This is the most useful
   lightweight example when you want to understand the control flow without a
   full GROMACS setup.

``tests/test_multi_system.py``
   Runs end-to-end MULTI-SYSTEM (multi-ligand) workflows on the toy engine: two
   systems with different atom counts in one run, trained with a shared network
   and with separate networks, plus per-system kinetics convergence. A compact
   reference for the multi-system control flow without GROMACS.

``tests/test_retinal.py``
   Runs a more realistic integration workflow using the retinal test system and
   GROMACS-based dynamics.

The retinal folder also includes a complete example ``params.py`` file together
with topology, coordinates, and force-field assets.

Notebook Example
----------------

The repository includes ``examples/notebooks/1_toy_1d.ipynb``. It complements
the toy-engine test by showing how to run AIMMD, in addition to how to analyse the outputs.

``examples/notebooks/2_multi_system.ipynb`` extends this to a **multi-system
(multi-ligand)** run: it configures one params file for two toy systems with
different atom counts, trains a single shared committor network on both
(balanced), contrasts the separate-network mode, and analyses the per-system
committor and rates. Like the single-system notebook it runs end-to-end on the
toy engine (no GROMACS/GPU).