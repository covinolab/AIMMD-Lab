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
        try:
            os.kill(self._tasks[localid].pid, signal.SIGINT)
        except:
            pass
    
    def _kill(self, localid):
        self._tasks[localid].kill()
        self._tasks[localid].join()

    def _close(self, localid):
        if self._tasks[localid] is not None:
            self._tasks[localid].close()
    
    def _closed(self, localid):
        if self._tasks[localid] is None:
            return False
        return self._tasks[localid]._closed
