"""
...
"""

# external
import os
import psutil

# cpu infos
def get_available_cpus():
    try:
        return sorted(list(set(os.sched_getaffinity(0))))
    except:
        return list(range(psutil.cpu_count(logical=False)))

def get_num_cpus():
    return len(get_available_cpus())
