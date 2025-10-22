import os
import torch
import numpy as np
import psutil
import warnings

# call just once, for compatibility issues
# when spawning multiple processes
torch.set_num_interop_threads(1)

def get_available_cpus():
    try:
        return sorted(list(set(os.sched_getaffinity(0))))
    except:
        return list(range(psutil.cpu_count(logical=False)))


def get_num_cpus():
    return len(get_available_cpus())


def get_num_gpus():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore *all* warnings in this block
        try:
            return torch.cuda.device_count()
        except Exception as exception:
            print(f'[Exception]: {exception}')
            return 0


def get_available_gpus():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])


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
    if (type(cpus_per_task) is str and
        cpus_per_task not in ['all', 'skip']
       ) or cpus_per_task <= 0:
        raise TypeError(f'cpus_per_task must be either a positive '
                        f'integer or "all", "skip"')
    if (type(gpus_per_task) is str and
        gpus_per_task not in ['all', 'skip']
       ) or gpus_per_task < 0:
        raise TypeError(f'gpus_per_task must be either 0, a positive '
                        f'integer, or "all", "skip"')
    
    print(f'Worker\'s resources info')
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
                f"but {resources_per_task} requested per task.")
        
        # determine the actual resources allocated for the task
        if resources_per_task == 'all' or (
            resources_per_task == num_resources_available):
            # this happens when running srun on HPC clusters
            # or when requiring "all" resources to be used        
            start = None
            stop = None
        else:
            # this happens when running on a node/workstation
            # with a few resources per task
            start = localid * resources_per_task
            stop = start + resources_per_task
        
        # notify the user if this worker is oversubscribing resources
        if stop and stop > num_resources_available:
            print(f"[Note] Worker may be oversubscribing {resources_name}\n"
                  f"  available {resources_name}s: {num_resources_available}"
                  f"\n  {resources_name}s per task: {resources_per_task}")
        
        return resources_available[start:stop]
    
    # CPU binding
    if cpus_per_task != 'skip':
        cpus = _determine_resources(cpus_per_task, cpus_available, 'CPU')
        cpus_per_tasks = len(cpus)
        
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
            cpus = "all"
    else:
        cpus = ",".join([str(id) for id in cpus_available])
    
    # GPU binding
    if gpus_per_task != 'skip' and gpus_per_task:
        gpus = _determine_resources(gpus_per_task, gpus_available, 'GPU')
        gpus_per_tasks = len(gpus)
        
        # GPU binding
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
    print(f'-----------------------\n')
