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
    
    __task_name__ = 'Task'
    
    def __init__(self):
        self._tasks = []
        self._names = []
        self._idents = []
        self._targets = []
        self._args = []
        self._kwargs = []
        self._receivers = []
    
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
            text.append(':\n... ')
        for i in range(len(self)):
            text.append(f':\n... {self._name(localid)}')
        return ''.join(text)

    @abstractmethod
    def _initialize(self, target, *args, **kwargs):
        pass

    def _name(self, localid):
        return (f'{self.__task_name__}{self._idents[localid]}: '
                f'{self._names[localid]}')

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

    def _reset(self, localid):
        """Replace task with pristine task. Allows to restart."""

        # cannot reset if still alive
        if self._alive(localid):
            raise RuntimeError(f'[{self._name(localid)}] '
                               f'still alive, cannot reset')

        # close old process
        self._close(localid)
        self._tasks[localid] = None
        self._idents[localid] = ''
        self._receivers[localid] = None
    
    def _build(self, localid):
        target = self._targets[localid]
        args = self._args[localid]
        kwargs = self._kwargs[localid]
        receiver1, receiver2 = multiprocessing.Pipe()
        target = functools.partial(target_wrapper, self._name(localid),
            receiver1, receiver2, target, *args, **kwargs)

        # receiver is a function that always returns
        # latest message; you may want to change its name
        closed = False
        last_message = ''
        def receiver():
            nonlocal last_message, closed
            
            if closed:
                return last_message
            
            try:
                while receiver1.poll():
                    last_message = receiver1.recv()
            except EOFError:
                closed = True

            if last_message and last_message != 'running':
                receiver1.close()
            
            return last_message
        
        self._tasks[localid] = self._initialize(target)
        self._receivers[localid] = receiver
        
        return self._tasks[localid], self._receivers[localid]
    
    @property
    def tasks(self):
        return list(self._tasks)

    @property
    def names(self):
        return list(self._names)

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
        self._receivers.append(None)
    
    def sequential(self, key=None):
        localids = np.arange(len(self))[key].flatten()
        for localid in localids:
            try:
                self._targets[localid](
                    *self._args[localid], **self._kwargs[localid])
                print(f'[{self._name(localid)}] exited correctly {now()}')
            except Exception as exception:
                raise RuntimeError(
                    f'[{self._name(localid)}] exited with error: '
                    f'{exception} {now()}\n{traceback.format_exc()}')
        
    def run(self, key=None, parallel=True, timeout=20.):
        localids = np.arange(len(self))[key].flatten()
        
        if not len(localids):
            return
        
        if not parallel or len(localids) == 1:
            return self.sequential(localids)
        
        for i, localid in enumerate(localids):
            self._reset(localid)
            
            task, receiver = self._build(localid)
            task.start()
                        
            # wait for ready
            while not receiver():
                continue
            
            # register identification number and update name in process
            self._idents[localid] = f' {task.ident}'
            print(f'[{self._name(localid)}] started {now()}')
            if args := self._args[localid]:
                print(f'...   args: {self._args[localid]}')
            if kwargs := self._kwargs[localid]:
                print(f'... kwargs: {kwargs}')
            
            # do not proceed if some process already stopped
            for i, receiver in enumerate(self._receivers[:i + 1]):
                name = self._name(i)
                message = receiver()
                if message == 'error':
                    raise RuntimeError(
                        f'[{name}] exited with error {now()}')
                if message == 'complete':
                    print(f'[{name}] exited correctly {now()}')
    
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
                        print(f'[{self._name(localid)}] '
                              f'sent termination signal {now()}')
                        terminating.add(localid)
                else:
                    self._close(localid)
                    localids.remove(localid)
        
        # force termination
        for localid in localids:
            if self._alive(localid):
                self._kill(localid)
                print(f'[{self._name(localid)}] sent kill signal {now()}')
        
        # final swipe
        for localid in localids:
            if not self._alive(localid):
                self._close(localid)
            else:
                print(f'[{self._name(localid)}] still alive {now()}')
