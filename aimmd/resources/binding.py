"""
...
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
    Parameters
    ----------
    cpus_per_task: str or int, defaut 'skip'
        Number of CPUs to allocate per task
        if 'all': each worker takes them all (explicitly bind resources)
        if 'skip': just report available resources, do not explicitly bind
    gpus_per_task: str or int, default 'skip'
        Number of GPUs to allocate per task
        if 'all': each worker takes them all (explicitly bind resources)
        if 'skip': just report available resources, do not explicitly bind
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
