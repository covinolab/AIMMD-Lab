C-GAIMMD: the shooting loop
===========================

What a C-GAIMMD shooting worker does to obtain a shooting point, and what it
leaves behind: the shooting-point source with its two stages and the final
acceptance check, the target-committor laws, the candidate pool and its
seeds, the in-flight records and per-shot sidecars with the shot table, the
tools for restarts and post-training, the preflight and the PLUMED bias of
the shots.

Shooting-point source
---------------------

.. automodule:: aimmd.cgaimmd.source

Target-committor laws
---------------------

.. automodule:: aimmd.cgaimmd.lambda_law

Candidate pool, network watcher and seeds
-----------------------------------------

.. automodule:: aimmd.cgaimmd.pool

In-flight records, sidecars and the shot table
----------------------------------------------

.. automodule:: aimmd.cgaimmd.sidecar

Restarts and post-training
--------------------------

.. automodule:: aimmd.cgaimmd.restart

Preflight
---------

.. automodule:: aimmd.cgaimmd.preflight

PLUMED bias of the shots
------------------------

.. automodule:: aimmd.cgaimmd.plumed
