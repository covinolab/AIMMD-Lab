Network Package
===============

Primary Role
------------

The :mod:`aimmd.network` package contains the model-fitting and output-rescaling
logic used by AIMMD committor learning.

Key Modules
-----------

:mod:`aimmd.network.fit`
   Main training routine used by default in AIMMD runs.

:mod:`aimmd.network.rescalable`
   Defines the ``Rescalable`` mixin used by AIMMD-compatible networks.

:mod:`aimmd.network.rescale_utils`
   Numerical helpers for fitting and applying piecewise-linear output
   rescaling.

:mod:`aimmd.network.utils`
   Placeholder network and small data-extraction helpers.

Optional Graph Workflow
-----------------------

The repository also contains :mod:`aimmd.network.graph_utils` for graph-based
descriptor pipelines. That module is optional and depends on the additional
packages described in the installation guide:

- ``torch-geometric``
- ``torch-cluster``
- ``mlcolvar``

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/network`.
