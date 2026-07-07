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

Install the latest release from PyPI:

.. code-block:: bash

   pip install aimmd-lab

The distribution is named ``aimmd-lab``, but the import name is ``aimmd``:

.. code-block:: python

   import aimmd

To work on AIMMD itself, install from a local clone in editable mode with the
optional test and documentation dependencies:

.. code-block:: bash

   git clone https://github.com/covinolab/AIMMD.git
   cd AIMMD
   pip install -e ".[tests,docs]"

Optional Graph-Network Dependencies
-----------------------------------

Graph-neural-network workflows need extra packages that are not installed by
default (``torch-cluster`` in particular can be awkward to build). The ``graphs``
extra pulls in ``torch-geometric`` and ``torch-cluster``:

.. code-block:: bash

   pip install "aimmd-lab[graphs]"
   pip install mlcolvar

If the default wheels do not match your CUDA / Python build, install them
explicitly. The following is **one confirmed-working example** for Linux with an
NVIDIA GPU on CUDA 11.8 and Python 3.13 — adjust the versions and index URLs for
your own setup:

.. code-block:: bash

   pip install torch==2.7.1 -f https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp313-cp313-manylinux_2_28_x86_64.whl
   pip install torch-geometric==2.7.0
   pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0%2Bcu118/torch_cluster-1.6.3%2Bpt27cu118-cp313-cp313-linux_x86_64.whl
   pip install mlcolvar

These packages matter only if you use the optional graph utilities in
``aimmd.network.graph_utils`` or graph-based descriptor pipelines.

Verifying the Installation
--------------------------

The installation can be verified by running the test suite.

.. code-block:: bash

   pip install pytest
   pytest tests/


Building the Documentation
--------------------------

The Sphinx sources live under ``docs/source``. Install the documentation
dependencies and build from the ``docs`` directory:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make html

The generated HTML is written to ``docs/build/html``. See the
:doc:`developer_guide` for how the build imports the package without the heavy
runtime dependencies.
