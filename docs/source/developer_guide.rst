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
