Launcher
========

:class:`aimmd.Launcher` orchestrates one or more runs. It normalizes inputs,
computes worker counts and CPU/GPU allocations, builds the on-disk directory
layout, and then either launches workers locally (via spawned processes) or
emits a SLURM job script that starts them with ``srun``.

.. autoclass:: aimmd.Launcher
   :members:
   :inherited-members:
   :show-inheritance:
