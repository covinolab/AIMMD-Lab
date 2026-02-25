"""
aimmd.worker.run
================

Command-line entry point for launching a single AIMMD worker.

This script is intended to be executed as an isolated process, typically via
a batch scheduler such as SLURM using ``srun``. It constructs a
:class:`~aimmd.worker.Worker` from command-line arguments and dispatches a task
via :meth:`~aimmd.worker.Worker.run`.

The design is intentionally minimal: argument parsing is positional and kept
outside of the AIMMD library API. This keeps the worker startup cost low and
avoids importing heavier scheduler/CLI dependencies.

Usage
-----
The script expects arguments in two blocks:

1) Worker constructor arguments (10 values):

   1. params
   2. directory
   3. localid
   4. cpus_per_task
   5. gpus_per_task
   6. log_file
   7. walltime
   8. nsteps
   9. nframes
   10. termination_timeout

2) Worker task dispatch arguments (11+):

   - task name and task-specific positional/keyword arguments consumed by
     :meth:`aimmd.worker.Worker.run`.

Notes
-----
- All values arrive as strings from the shell; the Worker initialization logic
  is responsible for parsing/casting where necessary.
- The script slices ``sys.argv`` directly; missing or extra arguments will
  result in Python exceptions (intentionally, to fail fast on misconfigured
  launch commands).
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
