import os
import subprocess

def get_available_cpus():
    return sorted(list(set(os.sched_getaffinity(0))))

def get_num_gpus():
    try:
        return len(subprocess.check_output(
            ["nvidia-smi", "--list-gpus"]
        ).decode().strip().split("\n"))
    except:
        return 0

def get_available_gpus():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])
