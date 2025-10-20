import os
import torch

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
