Developer Guide
===============

Repository Structure
--------------------

The repository is split into runtime packages, examples, tests, docs, and
developer utilities:

``aimmd/``
   Main Python package.

``docs/``
   Sphinx documentation sources and build outputs.

``tests/``
   Integration-style tests that exercise the core workflows.

``examples/``
   Notebook-based examples and supporting example data.

``devtools/``
   Environment and maintenance helpers.

How the Package is Layered
--------------------------

The package itself has a fairly consistent layering:

Low-level infrastructure
   ``aimmd.core``, ``aimmd.cache``, ``aimmd.execute``, and
   ``aimmd.resources`` provide generic utilities, caches, executors, and
   resource binding helpers.

Data containers
   ``aimmd.path`` and ``aimmd.pathensemble`` define the main filesystem-backed
   trajectory data model.

Configuration and orchestration
   ``aimmd.params``, ``aimmd.worker``, and ``aimmd.launcher`` convert a
   scientific configuration into concrete simulation and training work.

Learning and analysis
   ``aimmd.network`` and ``aimmd.analysis`` implement committor fitting,
   rescaling, binning, and post-processing.

Why So Many Private Modules?
----------------------------

Several major classes are assembled from mixins:

- ``Params``,
- ``Path``,
- ``PathEnsemble``,
- ``Worker``,
- ``Launcher``.

This is not accidental. The codebase uses private modules such as
``_helpers``, ``_methods``, ``_properties``, and ``_io`` to separate concerns
inside otherwise very large classes. The public classes then combine those
mixins into the user-facing API.

That means the private modules are part of the implementation architecture even
when the public entry point is just one class.

Import-Time Initialization
--------------------------

Importing ``aimmd`` triggers ``aimmd._init.initialize()``. This sets global
configuration, creates caches, resolves the GROMACS executable, patches a few
runtime behaviors, and stores shared objects in ``aimmd._config``.

That import-time behavior is important to know when:

- writing tests,
- using AIMMD on clusters,
- or generating documentation through autodoc.

Suggested Reading Order for New Contributors
--------------------------------------------

If you are onboarding to the codebase, a practical reading order is:

1. ``README.md`` for the scientific and operational overview.
2. ``tests/test_toy_1d.py`` and ``tests/retinal/params.py`` for concrete usage.
3. ``aimmd/params/__init__.py`` and ``aimmd/params/_fields.py`` for the main
   configuration surface.
4. ``aimmd/worker/_shoot.py``, ``aimmd/worker/_free.py``, and
   ``aimmd/worker/_train.py`` for the runtime loop.
5. ``aimmd/pathensemble/reweight.py`` and ``aimmd/analysis/utils.py`` for the
   statistical post-processing layer.

Tests as Executable Documentation
---------------------------------

The test suite doubles as worked examples of the intended API usage:

``tests/test_params.py``
   Parameter loading, persistence, source tracking, updates, and relative-path
   behavior.

``tests/test_toy_1d.py``
   A full end-to-end AIMMD workflow on the toy engine — the most useful
   lightweight example for following the control flow without a GROMACS setup.

``tests/test_multi_system.py``
   End-to-end multi-system (multi-ligand) workflows on the toy engine: two
   systems with different atom counts, trained with shared and with separate
   networks, plus per-system kinetics convergence.

``tests/test_retinal.py``
   A more realistic integration workflow using the retinal test system and
   GROMACS-based dynamics. The ``tests/retinal/`` folder includes a complete
   ``params.py`` with topology, coordinates, and force-field assets.

Building the Documentation
--------------------------

The Sphinx sources live under ``docs/source``. Install the documentation
dependencies and build:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html

The docs are configured so that ``import aimmd`` succeeds without the heavy
runtime dependencies or GROMACS: ``docs/source/conf.py`` inserts the repository
root on ``sys.path``, installs lightweight stubs for ``torch``, ``MDAnalysis``,
``mdtraj``, ``matplotlib`` and the graph stack, and patches executable resolution
so the GROMACS check passes. Only ``numpy`` and ``scipy`` are real dependencies
of the build — this is what lets Read the Docs build the API reference without
installing the package. Example notebooks are rendered from their saved outputs
(``nb_execution_mode = "off"``); they are not executed at build time.

The parameter reference on :doc:`parameters` is generated at build time by the
small ``aimmd-params`` Sphinx extension in ``docs/source/_ext/`` directly from
the :class:`aimmd.Params` dataclass fields.

Contributing
------------

Contributions are welcome. A typical workflow:

1. Install in editable mode with the test and docs extras::

      pip install -e ".[tests,docs]"

2. Make your change on a feature branch.
3. Run the tests (``pytest tests/``) and, if you touched documented code, build
   the docs to check the API reference still renders.
4. Follow the existing style: NumPy-style docstrings (rendered by
   ``sphinx.ext.napoleon``), and keep the mixin layering described above when
   extending the major classes.
5. Open a pull request against ``main`` at https://github.com/covinolab/AIMMD-Lab.

See ``CONTRIBUTING.md`` in the repository root for the short version.
