"""
aimmd.execute
============

Execution helpers for AIMMD.

This package provides lightweight utilities to run:
- external shell commands with streaming output and stop conditions
  (`execute_command`),
- collections of Python callables in parallel using either threads
  (`ThreadExecutor`) or processes (`ProcessExecutor`).

Public API
----------
execute_command
    Run a shell command with real-time output forwarding and cooperative stopping.
ThreadExecutor
    Run Python callables concurrently using `threading.Thread`.
ProcessExecutor
    Run Python callables concurrently using `multiprocessing.Process` (spawn).
"""

from .utils import execute_command
from .threads import ThreadExecutor
from .processes import ProcessExecutor
