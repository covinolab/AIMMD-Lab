"""
...
"""

# external
import threading

# aimmd imports
from .base import TaskExecutor

# process executor
class ThreadExecutor(TaskExecutor):
    
    __task_name__ = 'Thread'
    
    def _initialize(self, target):
        return threading.Thread(target=target, daemon=True)
