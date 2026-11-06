C-GAIMMD: flow matching (PaiNN)
===============================

The flow-matching generator of C-GAIMMD: the PaiNN network that carries a
committor head and a velocity field, the ODE generator ``'ode'`` that
integrates that field, the online fit that trains both every round from the
same start, and the end-state pretraining that is that start. A params file
builds the network (usually with
:meth:`~aimmd.cgaimmd.flow_matching.painn.JointPaiNN.from_topology`), wraps
:func:`~aimmd.cgaimmd.flow_matching.fit.flow_matching_fit` in its ``fit``
and names the pretrained network as ``initial_network``.

.. automodule:: aimmd.cgaimmd.flow_matching
   :no-members:

ODE generator
-------------

.. automodule:: aimmd.cgaimmd.flow_matching.sampler

PaiNN network
-------------

.. automodule:: aimmd.cgaimmd.flow_matching.painn

Online fit
----------

.. automodule:: aimmd.cgaimmd.flow_matching.fit

End-state pretraining
---------------------

.. automodule:: aimmd.cgaimmd.flow_matching.pretrain

Prior
-----

.. automodule:: aimmd.cgaimmd.flow_matching.prior

Atom tokens
-----------

.. automodule:: aimmd.cgaimmd.flow_matching.embedding

Flow-matching mathematics
-------------------------

.. automodule:: aimmd.cgaimmd.flow_matching.flowmath

Graph mode
----------

.. automodule:: aimmd.cgaimmd.flow_matching.graphs

Radius graphs
-------------

.. automodule:: aimmd.cgaimmd.flow_matching.graph_batch

Minimal network
---------------

.. automodule:: aimmd.cgaimmd.flow_matching.toy
