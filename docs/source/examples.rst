Examples and Tests
==================

Tests as Executable Documentation
---------------------------------

The tests are a good way to understand how the repository is intended
to be used because they exercise realistic workflows instead of only unit-level
helpers.

``tests/test_params.py``
   Focuses on parameter loading, persistence, source tracking, updates, and
   relative-path behavior.

``tests/test_toy_1d.py``
   Runs an end-to-end AIMMD workflow on the toy engine. This is the most useful
   lightweight example when you want to understand the control flow without a
   full GROMACS setup.

``tests/test_retinal.py``
   Runs a more realistic integration workflow using the retinal test system and
   GROMACS-based dynamics.

The retinal folder also includes a complete example ``params.py`` file together
with topology, coordinates, and force-field assets.

Notebook Example
----------------

The repository includes ``examples/notebooks/1_toy_1d.ipynb``. It complements
the toy-engine test by showing how to run AIMMD, in addition to how to analyse the outputs.