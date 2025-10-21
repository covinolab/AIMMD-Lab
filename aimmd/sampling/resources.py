import os
import torch
import numpy as np
import psutil

def get_available_cpus():
    return sorted(list(set(os.sched_getaffinity(0))))

def get_num_gpus():
    return torch.cuda.device_count()

def get_available_gpus():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])

def bind_resources(localid, cpus_per_task=1, gpus_per_task=0):
    print(f'Worker\'s resources info')
    print(f'-----------------------')
    print(f'LocalID {localid}')
    
    # find available cpus
    available_cpus = get_available_cpus()
    
    # find available gpus, using  to avoid
    # extra dependency, on cuda or ROCm
    num_gpus_avail = get_num_gpus()
    
    # CPU binding
    os.environ["OMP_NUM_THREADS"] = str(cpus_per_task)
    os.environ["MKL_NUM_THREADS"] = str(cpus_per_task)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    if len(available_cpus) == cpus_per_task:
        start = None
        stop = None
    else:
        start = localid * cpus_per_task
        stop = start + cpus_per_task
    cpus = available_cpus[start:stop]
    try:
        psutil.Process().cpu_affinity(cpus)
    except Exception as exception:
        print(f"[Warning] Could not set CPU affinity "
              f"with {cpus}: {exception}")
        cpus = []
    cpus = ",".join([str(id) for id in cpus])
    
    # check if requested GPU resources are available
    if gpus_per_task > 0 and num_gpus_avail == 0:
        raise RuntimeError(f"No GPUs available but {gpus_per_task} requested")
    if gpus_per_task > num_gpus_avail:
        raise RuntimeError(f"Only {num_gpus_avail} GPUs available but "
                           f"{gpus_per_task} requested per task.")

    # GPU binding
    if gpus_per_task > 0:
        start = localid * gpus_per_task
        stop = start + gpus_per_task
        gpus = np.arange(start, stop) % num_gpus_avail
        gpus = ",".join([str(id) for id in gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # for NVIDIA GPUs, and also ROCm picked up by torch
        os.environ["GPU_DEVICE_ORDINAL"] = gpus
        # for ROCm GPUs, Gromacs will use OpenCL
        
        # notify the user if this worker is oversubscribing a GPU
        if stop > num_gpus_avail:
            print(f"[Note] Worker may be oversubscribing GPUs\n"
                  f"  available GPUs: {num_gpus_avail}\n"
                  f"  GPUs per task: {gpus_per_task}")
    
    # report resource allocation
    if cpus:
        print(f"CPU ids: {cpus}")
    else:
        print(f"CPU ids: all")
    if gpus_per_task > 0:
        print(f"GPU ids: {gpus}")
    else:
        print(f"No GPUs allocated")
    print(f'-----------------------\n')
