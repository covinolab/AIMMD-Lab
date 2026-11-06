C-GAIMMD: networks, generators and shared pieces
================================================

What every C-GAIMMD run uses, whatever its generator: the protocol a network
follows (the committor only), the interface through which candidates are
generated and the registry of generation methods, the shared types that
cross the params-file hooks, the device roles, the atomic file writes, the
training data of a round and what the two online fits share.

Network protocol
----------------

.. automodule:: aimmd.cgaimmd.protocol

Generators
----------

.. automodule:: aimmd.cgaimmd.generation

Shared types
------------

.. automodule:: aimmd.cgaimmd.types

Protocol values function
------------------------

.. automodule:: aimmd.cgaimmd.values

Training data of a round
------------------------

.. automodule:: aimmd.cgaimmd.dataset

What the online fits share
--------------------------

.. automodule:: aimmd.cgaimmd.fitting

Devices
-------

.. automodule:: aimmd.cgaimmd.devices

Atomic writes and hashing
-------------------------

.. automodule:: aimmd.cgaimmd.io
