C-GAIMMD: BioEmu2
=================

C-GAIMMD with BioEmu2, Microsoft's protein ensemble emulator, as the
generator. It needs the ``bioemu2`` extra (``pip install
'aimmd-lab[bioemu2]'``, or ``'aimmd-lab[bioemu2-cuda]'`` for the
cuEquivariance kernels of the production model; Python 3.12 or later). A
params file names a system folder (built once with ``python -m
aimmd.cgaimmd.bioemu2 build-system``), builds a
:class:`~aimmd.cgaimmd.bioemu2.network.BioEmu2Network` from it, wraps
:func:`~aimmd.cgaimmd.bioemu2.fit.sequential_fit` in its ``fit`` and names
the network written by
:func:`~aimmd.cgaimmd.bioemu2.pretrain.pretrain_initial_network` as
``initial_network``.

.. automodule:: aimmd.cgaimmd.bioemu2
   :no-members:

Network
-------

.. automodule:: aimmd.cgaimmd.bioemu2.network

Generator ``'bioemu2_edm'``
---------------------------

.. automodule:: aimmd.cgaimmd.bioemu2.sampling

Online fit
----------

.. automodule:: aimmd.cgaimmd.bioemu2.fit

Initial network
---------------

.. automodule:: aimmd.cgaimmd.bioemu2.pretrain

System
------

.. automodule:: aimmd.cgaimmd.bioemu2.system

Building a system folder
------------------------

.. automodule:: aimmd.cgaimmd.bioemu2.build

Features
--------

.. automodule:: aimmd.cgaimmd.bioemu2.features

Committor head
--------------

.. automodule:: aimmd.cgaimmd.bioemu2.head

Trunk adapters and guidance
---------------------------

.. automodule:: aimmd.cgaimmd.bioemu2.adapters
