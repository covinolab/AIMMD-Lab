"""
Just for running on cluster.
"""

# external
import sys

# aimmd imports
from aimmd.worker import Worker

if __name__ == '__main__':
    Worker(*sys.argv[1:11]).run(*sys.argv[11:])
    # 1: params, 2: directory, 3: localid,
    # 4: cpus_per_task, 5: gpus_per_task,
    # 6: log_file, 7: walltime,
    # 8: nsteps, 9: nframes,
    # 10: termination timeout,
    # 11+: worker run arguments
