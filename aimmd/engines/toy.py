"""
Integrator for toy systems compatible with worker.sh.
Any dimensions work.
"""

# external
import os
import sys
import time
import signal
from math import inf
from MDAnalysis import Universe, Writer

# aimmd imports
from .._config import MDA_CACHE
from ..execute import ThreadExecutor
from ..core.utils import remove

# class
class ToyEngine:
    
    def __init__(self, mdrun=None, slowdown=.01, extension='.xtc'):
        """mdrun: how to evolve timestep"""
        self.mdrun = mdrun or (lambda x: None)
        self.slowdown = slowdown
        self.extension = extension
        self.must_stop = False
    
    def __call__(self, deffnm,
                 backup=True,
                 noappend=False,
                 stop_condition=lambda : False,
                 walltime=inf,
                 termination_timeout=20., # here just for compatibility
                 raise_if_failure=True,
                 log_file='stdout'):
        t0 = time.time()
        if log_file == 'stdout':
            log_file = sys.stdout
        
        # noappend
        if noappend:
            i = 0
            old = None
            backup = ''

            # sweep forward
            while True:
                fname = f'{deffnm}.part{i:04g}{self.extension}'
                if not (os.path.exists(fname) and os.path.getsize(fname)):
                    break
                old = fname
                i += 1

            # no data
            if not old:
                error_msg = f'{fname!r} not existing or corrupted'
                if not raise_if_failure:
                    if log_file:
                        print(f'Warning: {error_msg}', file=log_file)
                    return 1
                raise RuntimeError(error_msg)
            
            # sweep and check backward
            for i in range(i - 1, -1, -1):
                old = f'{deffnm}.part{i:04g}.xtc'
                reader = MDA_CACHE.load(old)
                if reader:
                    break
                error_msg = f'{old!r} not existing or corrupted'
                if log_file:
                    print(f'Warning: {error_msg}', file=log_file)
                fname = old
            if not reader:
                if not raise_if_failure:
                    if log_file:
                        print(f'Warning: {error_msg}', file=log_file)
                    return 1
                raise RuntimeError(error_msg)
            if log_file:
                print(f'Creating new file {fname!r}', file=log_file)

        # append
        else:
            fname = f'{deffnm}{self.extension}'
            backup = backup and f'{deffnm}_backup{self.extension}'
            reader = MDA_CACHE.load(fname)
            if not reader:
                if not backup or not os.path.exists(backup):
                    error_msg = f'{fname!r} not existing or corrupted'
                    if not raise_if_failure:
                        if log_file:
                            print(f'Warning: {error_msg}', file=log_file)
                        return 1
                    raise RuntimeError(error_msg)
                if log_file:
                    print(f'Warning: loading {backup!r} due to {exception}',
                          file=log_file)
                reader = MDA_CACHE.open(backup)
                if not reader:
                    error_msg = f'{backup!r} not existing or corrupted'
                    if not raise_if_failure:
                        if log_file:
                            print(f'Warning: {error_msg}', file=log_file)
                        return 1
                    raise RuntimeError(error_msg)
            if log_file:
                print(f'Appending to {fname}', file=log_file)

        # finally run simulation
        n_atoms = reader.trajectory.n_atoms
        universe = Universe.empty(n_atoms, trajectory=True)
        ts = universe.trajectory.ts

        # go to other fname, otherwise can't write to file
        if not noappend:
            temp = backup or f'{deffnm}_temp{self.extension}'
            os.rename(fname, temp)
            reader = MDA_CACHE.open(temp)
            if reader is None:
                error_msg = f'{temp!r} not existing or corrupted'
                if not raise_if_failure:
                    if log_file:
                        print(f'Warning: {error_msg}', file=log_file)
                    return 1
                raise RuntimeError(error_msg)
            restore_backup = True

        def main():
        
            # write file
            try:
                with Writer(fname, n_atoms) as writer:
        
                    # retrieve information from frames
                    if noappend:
                        frame = reader[-1]
                        ts.positions = frame.positions
                        ts.time = frame.time
                    else:  # rewrite old frames
                        for frame in reader:
                            ts.positions = frame.positions
                            ts.time = frame.time
                            writer.write(universe)
                        remove(temp, verbose=False)
                    
                    # actual simulation
                    while time.time() - t0 < walltime and not self.must_stop:
                        time.sleep(self.slowdown)
                        self.mdrun(ts)
                        ts.time += 1.0
                        writer.write(universe)
                        if log_file:
                            print(f'Reached time {ts.time}\r',
                                  end='', file=log_file)
    
            # restore backup/create new backup
            finally:
                if backup:
                    if restore_backup:
                        os.rename(backup, fname)
                    else:
                        os.copy(fname, backup)
                self.must_stop = True
        
        def update_stop_condition():
            while not self.must_stop:
                self.must_stop = stop_condition()
                time.sleep(.01)  # avoid freezing
        
        threads = ThreadExecutor()
        threads.add(main, name='toy engine run')
        threads.add(update_stop_condition, name='toy engine check')
        threads.run()
        while threads.alive.any():
            continue
