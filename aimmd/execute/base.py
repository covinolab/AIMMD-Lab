"""
...
"""

# external
import time
import numpy as np
import functools
import traceback
import multiprocessing
from abc import ABC, abstractmethod

# aimmd imports
from .utils import target_wrapper
from ..core.utils import now

ctx = multiprocessing.get_context('spawn')

class TaskExecutor(ABC):
    """For both list of processes or threads."""
    
    def __init__(self):
        self._tasks = []
        self._names = []
        self._idents = []
        self._targets = []
        self._args = []
        self._kwargs = []
    
    def __len__(self):
        return len(self._tasks)
    
    def __iter__(self):
        return iter(self._tasks)
    
    def __getitem__(self, key):
        if key is None:
            return list(self._tasks)
        if hasattr(key, '__len__'):
            return [self[key] for key in np.arange(len(self))[key].flatten()]
        return self._tasks[key]

    def __repr__(self):
        text = [f'{self.__class__.__name__} with {len(self)} '
                f'task{"s" if len(self) != 1 else ""}']
        if len(self):
            text.append(':')
        for i in range(len(self)):
            text.append(f'\n... {self._names[i]}')
        return ''.join(text)
    
    @abstractmethod
    def _initialize(self, target, *args, **kwargs):
        pass

    def _alive(self, localid):
        if self._tasks[localid] is None:
            return False
        if self._closed(localid):
            return False
        return self._tasks[localid].is_alive()
    
    def _terminate(self, localid):
        pass

    def _kill(self, localid):
        pass

    def _close(self, localid):
        pass

    def _closed(self, localid):
        return False

    def _reset(self, localid, timeout=20.):
        """Replace task with pristine task. Allows to restart."""
        
        # close old process
        self.stop(localid, timeout=timeout)
        self._close(localid)
        self._tasks[localid] = None
        self._idents[localid] = ''
    
    def _build(self, localid):
        target = self._targets[localid]
        args = self._args[localid]
        kwargs = self._kwargs[localid]
        name = self._names[localid]
        target = functools.partial(
            target_wrapper, target, name, *args, **kwargs)
        self._tasks[localid] = self._initialize(target)        
        return self._tasks[localid]
    
    @property
    def tasks(self):
        return list(self._tasks)

    @property
    def idents(self):
        return list(self._idents)
    
    @property
    def targets(self):
        return list(self._targets)

    @property
    def args(self):
        return list(self._args)

    @property
    def kwargs(self):
        return list(self._kwargs)
    
    @property
    def alive(self):
        return np.array([self._alive(localid)
                         for localid in range(len(self))], dtype=bool)
    
    def add(self, target, *args, name='', **kwargs):
        if not name:
            name = [f'{target.__name__}(']
            for arg in args:
                name.append(f'{arg}, ')
            for key, arg in kwargs.items():
                name.append(f'{key}={arg}, ')
            name[-1] = name[-1].rstrip(', ')
            name = ''.join(name + [')'])
        
        self._tasks.append(None)
        self._names.append(name)
        self._idents.append('')
        self._targets.append(target)
        self._args.append(args)
        self._kwargs.append(kwargs)
    
    def sequential(self, key=None):
        localids = np.arange(len(self))[key].flatten()
        for localid in localids:
            target_wrapper(target, name, *args, **kwargs)
    
    def run(self, key=None, parallel=True, timeout=20.):
        localids = np.arange(len(self))[key].flatten()
        
        if not len(localids):
            return
        
        if not parallel or len(localids) == 1:
            return self.sequential(localids)
        
        for i, localid in enumerate(localids):

            # reset and start task
            self._reset(localid)
            task = self._build(localid)
            task.start()
    
    def stop(self, key=None, timeout=20.):

        # which localids to stop?
        localids = list(np.arange(len(self))[key].flatten())

        # graceful termination
        t0 = time.time()
        terminating = set()  # call termination only once
        while time.time() - t0 < timeout and len(localids):
            for localid in localids[:]:
                if self._alive(localid):
                    if localid not in terminating:
                        self._terminate(localid)
                        terminating.add(localid)
                else:
                    localids.remove(localid)
        
        # force termination within timeout
        for localid in localids:
            if self._alive(localid):
                self._kill(localid)
                time.sleep(timeout)
        
        # final swipe
        for localid in localids:
            if self._alive(localid):
                print(f'Warning: some processes are still alive {now()}')
                break
    
    def clear(self, timeout=20.):
        """Close all tasks and remove them"""
        self.stop(self.alive, timeout=timeout)
        for localid in range(len(self)):
            self._close(localid)
        self._tasks = []
        self._names = []
        self._idents = []
        self._targets = []
        self._args = []
        self._kwargs = []
