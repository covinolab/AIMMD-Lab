"""
...
"""

# external
import os
import time
import traceback
from abc import ABC

# aimmd imports
from .._config import MDA_CACHE, NPY_CACHE, print
from ..core.utils import now

# worker run method
class WorkerRun(ABC):

    def run(self, task, *args, **kwargs):
        
        # initialize
        self.task = task
        
        try:
            # always go params' directory
            cwd = os.getcwd()
            self._directory = os.path.relpath(
                self.directory, self.params.parent)
            # directory relative to params' folder
            os.chdir(self.params.parent)

            # report
            if self.log_file == self.original_stdout:
                print(f"Press Control+C to interrupt.")
            else:
                print(f"Starting: worker{self.localid}, {task} {now()}")
            
            # bind resources
            self._bind_resources()
            
            # clear caches
            MDA_CACHE.clear()
            NPY_CACHE.clear()
            
            # update stop condition and remove from kwargs
            self._update_stop_condition(**kwargs)

            # reset time, total steps, total frames
            self._t0 = time.time()
            self._total_steps = 0
            self._total_frames = 0
            
            # execute task
            if task == 'shoot':
                return self._shoot(*args, **kwargs)
            if task == 'free':
                return self._free(*args, **kwargs)
            if task == 'train':
                return self._train(*args, **kwargs)
            raise TypeError(f'{task} not implemented in Worker.run')

        except Exception as exception:
            if self.log_file != self.original_stdout:
                traceback.print_exc(file=self.original_stdout)
            raise exception
        
        finally:
            os.chdir(cwd)  # back to main folder
            self._directory = self.directory
            self._terminate_operations()
            self._reset_stop_condition()
