Path
====

:class:`aimmd.Path` is a lightweight, segment-aware reference to one or more
on-disk trajectory files. Heavy data (states, descriptors, committor values) is
loaded lazily and cached beside the trajectory as ``.npy`` files.

.. autoclass:: aimmd.Path
   :members:
   :inherited-members:
   :show-inheritance:
