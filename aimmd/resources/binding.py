"""
aimmd.resources.binding
======================

Worker resource binding.

This module defines :func:`bind_resources`, a utility used by AIMMD workers to
bind a process to a subset of CPUs and GPUs based on a per-node local index.

The goal is to make local multi-worker execution predictable:

- on workstations: map worker 0..K-1 to disjoint slices of CPU cores and GPUs
- on HPC with Slurm: respect allocations that are already restricted by cpusets
  and CUDA_VISIBLE_DEVICES, while still allowing explicit binding if requested.
  However, the default mode in this case is 'skip'.

Binding model
-------------
`bind_resources` operates on the *resources visible to the current process*:

- CPUs are obtained from :func:`aimmd.resources.cpu.get_available_cpus`
- GPUs are obtained from :func:`aimmd.resources.gpu.get_available_gpus`

The selection rule is:

- if resources_per_task == 'all':
    use all visible resources
- if resources_per_task == 'skip':
    do not bind, only report visible resources
- if resources_per_task is an integer:
    assign a contiguous slice starting at ``localid * resources_per_task``;
    if this exceeds the visible set, wrap around modulo the number available
    (and print a note about possible oversubscription)

Actual binding side effects
---------------------------
CPU binding (when enabled) sets:
- ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``
- ``torch.set_num_threads(...)``
- attempts to set process CPU affinity via ``psutil.Process().cpu_affinity(...)``

GPU binding (when enabled) sets:
- ``CUDA_VISIBLE_DEVICES``
- ``GPU_DEVICE_ORDINAL`` (used by some runtimes, including ROCm paths in torch)

No other resource managers are touched.

Notes
-----
This function is intentionally pragmatic:
- it tries to bind resources when asked,
- but if affinity operations fail it continues and only emits a warning.
"""

# external
import os
import torch
import psutil

# aimmd imports
from .cpu import get_available_cpus
from .gpu import get_available_gpus
from .._config import print


# function
def bind_resources(localid, cpus_per_task='skip', gpus_per_task='skip'):
    """
    Bind the current process to CPU/GPU resources for the given worker local id.

    Parameters
    ----------
    localid : int
        Index of the worker on the current node. This is used to choose which
        subset of resources the worker should take when `cpus_per_task` and/or
        `gpus_per_task` are integers.

    cpus_per_task : str or int, default 'skip'
        CPU allocation policy for this worker.

        - 'all' : bind to all visible CPUs
        - 'skip': do not bind; only report visible CPUs
        - int   : number of CPUs to allocate to this worker

    gpus_per_task : str or int, default 'skip'
        GPU allocation policy for this worker.

        - 'all' : bind to all visible GPUs
        - 'skip': do not bind; only report visible GPUs
        - 0     : bind to no GPU (unset visibility)
        - int   : number of GPUs to allocate to this worker

    Notes
    -----
    The selected resource lists are computed from the visible resources of the
    current process. This means scheduler restrictions (cpusets, CUDA_VISIBLE_DEVICES)
    are honored automatically.
    """

    # process cpus_per_task and gpus_per_task
    try:
        cpus_per_task = int(cpus_per_task)
    except:
        cpus_per_task = str(cpus_per_task).lower()
    try:
        gpus_per_task = int(gpus_per_task)
    except:
        gpus_per_task = str(gpus_per_task).lower()

    # check correctness
    if cpus_per_task not in ['all', 'skip'] and (
        type(cpus_per_task) is int and cpus_per_task <= 0):
        raise TypeError(f'cpus_per_task must be either a positive '
                        f'integer or "all", "skip"')
    if gpus_per_task not in ['all', 'skip'] and (
        type(gpus_per_task) is int and gpus_per_task < 0):
        raise TypeError(f'gpus_per_task must be either 0, a positive '
                        f'integer, or "all", "skip"')

    print(f'\nWorker\'s resources info')
    print(f'-----------------------')
    print(f'LocalID {localid}')

    # find available cpus
    cpus_available = get_available_cpus()

    # find available gpus, using torch to avoid
    # extra dependency, on cuda or ROCm
    gpus_available = get_available_gpus()

    def _determine_resources(resources_per_task,
                             resources_available,
                             resources_name):
        """Standardized operations for CPUs/GPUs.
        Returns: resources list (to bind)."""

        num_resources_available = len(resources_available)

        # check if requested resources are available
        if resources_per_task != 'all' and (
            resources_per_task > num_resources_available):
            raise RuntimeError(
                f"{num_resources_available} {resources_name}s available "
                f"but {resources_per_task} requested for task.")

        # determine the actual resources allocated for the task
        if resources_per_task == 'all' or (
            resources_per_task == num_resources_available):
            # this happens when running srun on HPC clusters
            # or when requiring "all" resources to be used
            start = 0
            stop = num_resources_available
        else:
            # this happens when running on a node/workstation
            # with a few resources per task
            start = localid * resources_per_task
            stop = start + resources_per_task

        # notify the user if this worker is oversubscribing resources
        if stop > num_resources_available:
            print(f"[Note] Worker may be oversubscribing {resources_name}s"
                  f" (available: {num_resources_available}, "
                  f" required: {resources_per_task} per task)")

        return [resources_available[i % num_resources_available]
                for i in range(start, stop)]

    # CPU binding
    if cpus_per_task != 'skip':
        cpus = _determine_resources(cpus_per_task, cpus_available, 'CPU')
        cpus_per_task = len(cpus)

        # actual binding
        os.environ["OMP_NUM_THREADS"] = str(cpus_per_task)
        os.environ["MKL_NUM_THREADS"] = str(cpus_per_task)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
        torch.set_num_threads(cpus_per_task)
        try:
            psutil.Process().cpu_affinity(cpus)
            cpus = ",".join([str(id) for id in cpus])
        except Exception as exception:
            print(f"[Warning] Could not set CPU affinity "
                  f"with {cpus}: {exception}")
    else:
        cpus = ",".join([str(id) for id in cpus_available])

    # GPU binding
    if gpus_per_task != 'skip':
        if gpus_per_task:
            gpus = _determine_resources(gpus_per_task, gpus_available, 'GPU')
            gpus = ",".join([str(id) for id in gpus])
        else:  # even if there are GPUs, you asked for none
            gpus = ""

        # actual binding
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # for NVIDIA GPUs, and also ROCm picked up by torch
        os.environ["GPU_DEVICE_ORDINAL"] = gpus
        # for ROCm GPUs, Gromacs will use OpenCL
    else:
        gpus = ",".join([str(id) for id in gpus_available])

    if not gpus:
        gpus = "none"

    # report resource allocation
    print(f'CPU ids: {cpus}')
    print(f'GPU ids: {gpus}')
    print(f'-----------------------')
