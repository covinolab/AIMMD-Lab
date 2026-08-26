Network
=======

The :mod:`aimmd.network` subpackage trains and applies the committor model and
provides output rescaling. The committor network itself is a user-supplied
:class:`torch.nn.Module` passed via :attr:`Params.network`; AIMMD supplies the
training loop, a placeholder network, and rescaling utilities.

Training
--------

.. automodule:: aimmd.network.fit
   :members:
   :show-inheritance:

Output rescaling
----------------

.. automodule:: aimmd.network.rescalable
   :members:
   :show-inheritance:

.. automodule:: aimmd.network.rescale_utils
   :members:
   :show-inheritance:

Utilities
---------

.. automodule:: aimmd.network.utils
   :members:
   :show-inheritance:

Graph-neural-network support
----------------------------

The module ``aimmd.network.graph_utils`` provides optional support for
graph-neural-network committor models (atom-coordinate descriptors, PyG graph
construction and SQLite graph caching, and a shared ``atom_types`` one-hot
encoding for multi-system runs). It requires the optional ``graphs`` extras
(``torch-geometric``, ``torch-cluster``) plus ``mlcolvar`` and ``lz4``; install
them with ``pip install "aimmd-lab[graphs]"`` and see :doc:`../advanced`.

Graph-cache acceleration
------------------------

Graph-cache lookups dominate the trainer's wall clock once the cache grows to
tens of GB on a shared filesystem, because the file never stays resident. The
module below transparently serves those lookups from a node-local ``/dev/shm``
replica and an in-process memo, falling back to the database on any miss or
failure. It is enabled automatically for the trainer and needs no configuration;
the environment variables in its docstring exist only to constrain or disable it.
Unlike ``graph_utils`` it has no optional dependencies, so it imports anywhere.

.. automodule:: aimmd.network.shm_cache
   :members:
   :show-inheritance:
