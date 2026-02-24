"""
...
"""

# external
import os
import signal

# aimmd imports
from .base import TaskExecutor, ctx

# process executor
class ProcessExecutor(TaskExecutor):
    
    __task_name__ = 'Process'
    
    def _initialize(self, target):
        return ctx.Process(target=target)
    
    def _terminate(self, localid):
        """Graceful termination of the process."""
        task = self._tasks[localid]
        if task and task.is_alive():
            task.terminate()
    
    def _kill(self, localid):
        """Forcefully kill the process (alias to terminate in Python)."""
        task = self._tasks[localid]
        if task and task.is_alive():
            task.terminate()
    
    def _close(self, localid):
        """Join the process to clean up resources."""
        task = self._tasks[localid]
        if task:
            task.join(timeout=0.1)  # small timeout to avoid blocking forever
    
    def _closed(self, localid):
        if self._tasks[localid] is None:
            return False
        return self._tasks[localid]._closed
