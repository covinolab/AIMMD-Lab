"""
aimmd.resources.cpu
==================

CPU visibility utilities.

This module provides small helpers to query which CPUs are available to the
current process.

The returned CPU set is meant to reflect process-level restrictions imposed by:
- Linux CPU affinity (e.g., Slurm cgroups, taskset, sched_setaffinity),
- container/cgroup cpuset limitations.

If affinity information is not available, the implementation falls back to the
number of physical cores reported by psutil.

Notes
-----
- On Linux, :func:`os.sched_getaffinity` is the preferred source because it
  reflects the scheduler-visible CPU mask for the current PID.
- The fallback uses ``psutil.cpu_count(logical=False)`` (physical cores).
"""

# external
import os
import psutil


# cpu infos
def get_available_cpus():
    """
    Return the CPU ids currently available to the process.

    Returns
    -------
    list[int]
        Sorted list of CPU ids. On Linux, this is derived from
        ``os.sched_getaffinity(0)`` and therefore respects cpuset/affinity
        restrictions. If that fails, returns ``range(psutil.cpu_count(logical=False))``.

    Notes
    -----
    This function is intentionally conservative: it reports the CPUs the process
    can actually run on, not necessarily all CPUs installed on the machine.
    """
    try:
        return sorted(list(set(os.sched_getaffinity(0))))
    except:
        return list(range(psutil.cpu_count(logical=False)))


def get_num_cpus():
    """
    Return the number of CPUs currently available to the process.

    Returns
    -------
    int
        ``len(get_available_cpus())``.
    """
    return len(get_available_cpus())
