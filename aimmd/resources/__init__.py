"""
aimmd.resources
===============

Runtime resource introspection and binding helpers.

This subpackage provides minimal utilities for:

- querying the resources visible to the current process (CPUs/GPUs),
- binding a worker process to a subset of those resources.

This is used to make multi-worker execution deterministic and avoid accidental
oversubscription on workstations and HPC nodes.

Public API
----------
get_num_cpus / get_available_cpus
    CPU visibility as seen by the current process (honors CPU affinity / cgroups
    when available).
get_num_gpus / get_available_gpus
    GPU visibility as seen by the current process (honors CUDA_VISIBLE_DEVICES).
bind_resources
    Bind the current process to a local slice of CPUs/GPUs based on a `localid`.

Notes
-----
This module only re-exports the public functions. Implementations live in:
- :mod:`aimmd.resources.cpu`
- :mod:`aimmd.resources.gpu`
- :mod:`aimmd.resources.binding`
"""

from .cpu import get_num_cpus, get_available_cpus
from .gpu import get_num_gpus, get_available_gpus
from .binding import bind_resources
