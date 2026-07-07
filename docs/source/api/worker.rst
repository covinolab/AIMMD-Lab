Worker
======

:class:`aimmd.Worker` is the atomic execution unit. A worker runs exactly one
task at a time — ``shoot`` (committor-guided two-way shooting), ``free``
(unbiased simulation from a state), ``train`` (fit the committor model and
refresh the adaptive sampling state), or ``kinetics_convergence`` — handling
signal-driven cooperative termination, resource binding, and log redirection.

.. autoclass:: aimmd.Worker
   :members:
   :inherited-members:
   :show-inheritance:
