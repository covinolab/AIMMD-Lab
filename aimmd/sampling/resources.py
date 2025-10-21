import os
import torch
import numpy as np
import psutil

# call just once, for compatibility issues
# when spawning multiple processes
torch.set_num_interop_threads(1)

def get_available_cpus():
    try:
        return sorted(list(set(os.sched_getaffinity(0))))
    except:
        return list(range(psutil.cpu_count(logical=False)))

def get_num_gpus():
    return torch.cuda.device_count()

def get_available_gpus():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])

def bind_resources(localid, cpus_per_task=None, gpus_per_task=0):
    """
    cpus_per_task: None or 0 means take all available
    gpus_per_task: None means take all available
    """
    
    print(f'Worker\'s resources info')
    print(f'-----------------------')
    print(f'LocalID {localid}')
    
    # find available cpus
    available_cpus = get_available_cpus()
    
    # find available gpus, using torch to avoid
    # extra dependency, on cuda or ROCm
    num_gpus_avail = get_num_gpus()
    
    # determine the actual cpus allocated for the task
    if not cpus_per_task or len(available_cpus) <= cpus_per_task:
        # this happens when running srun on HPC clusters
        # or when requiring "all" cpus to be used        
        start = None
        stop = None
    else:
        # this happens when running on a node/workstation
        # with a few cpus per task
        start = localid * cpus_per_task
        stop = start + cpus_per_task
    cpus = available_cpus[start:stop]
    
    # CPU binding
    if cpus_per_task is not None and cpus_per_task > 0:
        cpus_per_task = len(cpus)
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
    
    # GPU binding
    if gpus_per_task is not None:
        
        # check if requested GPU resources are available
        if gpus_per_task == 0 and num_gpus_avail == 0:
            raise RuntimeError(
                f"No GPUs available but {gpus_per_task} requested")
        if gpus_per_task > num_gpus_avail:
            raise RuntimeError(f"Only {num_gpus_avail} GPUs available but "
                               f"{gpus_per_task} requested per task.")
        
        # determine the actual gpus allocated for the task
        if gpus_per_task > 0:
            start = localid * gpus_per_task
            stop = start + gpus_per_task
            gpus = np.arange(start, stop) % num_gpus_avail
            gpus = ",".join([str(id) for id in gpus])
            
            # notify the user if this worker is oversubscribing a GPU
            if stop > num_gpus_avail:
                print(f"[Note] Worker may be oversubscribing GPUs\n"
                      f"  available GPUs: {num_gpus_avail}\n"
                      f"  GPUs per task: {gpus_per_task}")
            
            # GPU binding
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            # for NVIDIA GPUs, and also ROCm picked up by torch
            os.environ["GPU_DEVICE_ORDINAL"] = gpus
            # for ROCm GPUs, Gromacs will use OpenCL
        else:
            gpus = "none"
    else:
        gpus = ",".join([str(id) for id in range(num_gpus_avail)])
    
    # report resource allocation
    print(f'CPU ids: {cpus}')
    print(f'GPU ids: {gpus}')
    print(f'-----------------------\n')
