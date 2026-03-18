Installation
============

Recommended Environment
-----------------------

It is recommended to install AIMMD in a clean conda environment. Python 3.13 is tested and supported, but other versions may work as well. Create a new environment with:

.. code-block:: bash

   conda create -n aimmd python=3.13


Prerequisites
-------------

AIMMD depends on a working GROMACS installation. The package expects either
``gmx`` or ``gmx_mpi`` to be available on ``PATH`` because import-time
initialization resolves the executable and uses it to configure engine-facing
defaults.

For lightweight testing, the GROMACS can be installed from
``conda-forge``:

.. code-block:: bash

   conda install conda-forge::gromacs

For production work, we recommend building GROMACS from source for
better performance and cluster-specific tuning.

Installing AIMMD
----------------

The package is not published on PyPI yet. Install it from a local clone in
editable mode:

.. code-block:: bash

   pip install -e .

Optional Graph-Network Dependencies
-----------------------------------

Graph-neural-network workflows require additional packages that are not
installed by default. The following stack is confirmed to work
for Linux with an NVIDIA GPU compatible with CUDA 11.8 in a Python 3.13
environment:

.. code-block:: bash

   pip install torch==2.7.1 -f https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp313-cp313-manylinux_2_28_x86_64.whl
   pip install torch-geometric==2.7.0
   pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp313-cp313-linux_x86_64.whl
   pip install mlcolvar

These packages matter only if you are using the optional graph utilities in
``aimmd.network.graph_utils`` or graph-based descriptor pipelines.

Verifying the Installation
--------------------------

The installation can be verified by running the test suite.

.. code-block:: bash

   pip install pytest
   pytest tests/


Building the Documentation
--------------------------

The Sphinx sources live under ``docs/source`` and the docs can be built from
``docs`` with:

.. code-block:: bash

   make html

The generated HTML will be written to ``docs/build/html``.
