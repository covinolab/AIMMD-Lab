"""
aimmd.engines.toy
=================

Toy simulation engine.

This module implements :class:`ToyEngine`, a minimal integrator that advances
an MDAnalysis timestep object in pure Python and writes an output trajectory.

Intended use
------------
- Method development.
- Code development and debugging.
- Unit tests / integration tests where GROMACS is not required.
- Demonstration runs with a controllable slowdown.

Core idea
---------
The engine:
1) loads an existing trajectory file (via `MDA_CACHE`),
2) creates a new writable trajectory file (either appending or creating a new part),
3) replays old frames (if appending),
4) generates new frames by repeatedly calling a user-supplied `mdrun(ts)` callback,
5) stops when walltime is reached or `stop_condition()` becomes true.

Concurrency model
-----------------
The engine runs two threads:
- one thread performs trajectory writing and timestep updates,
- one thread polls `stop_condition()` at short intervals.

This mirrors how "real" engines behave in AIMMD: a long-running simulation is
interruptible by an external condition.

Notes
-----
- This is a *toy* engine. It does not perform physical integration by itself.
  The `mdrun` callback is responsible for modifying `ts.positions` (and any
  other state) in a meaningful way.
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


class ToyEngine:
    """
    Minimal engine that advances an MDAnalysis timestep via a callback.

    Parameters
    ----------
    mdrun : callable, optional
        Function that advances the MDAnalysis timestep. It is called as
        `mdrun(ts)` where `ts` is an MDAnalysis Timestep. If None, a no-op
        function is used (`ts` remains unchanged).
    slowdown : float, default 0.01
        Seconds to sleep between generated frames. This is only for throttling
        the toy engine to avoid tight loops. Moreover, it avoid that the rest
        of the AIMMD code does not keep up with the simualtion speed.
    extension : str, default '.xtc'
        Trajectory extension used for input/output files.
    
    Notes
    -----
    `ToyEngine` is callable. Calling an instance runs the simulation loop and
    returns a status code (0/1) or raises, depending on `raise_if_failure`.
    """

    def __init__(self, mdrun=None, slowdown=.01, extension='.xtc'):
        """
        Construct a ToyEngine instance.

        Parameters
        ----------
        mdrun : callable, optional
            Callback used to evolve the timestep.
        slowdown : float, default 0.01
            Sleep time between frames (seconds).
        extension : str, default '.xtc'
            Output trajectory extension.
        """
        # `mdrun` defines how positions/time evolve each step.
        self.mdrun = mdrun or (lambda x: None)
        self.slowdown = slowdown
        self.extension = extension
        self.must_stop = False

    def __call__(
        self,
        deffnm,
        backup=True,
        noappend=False,
        stop_condition=lambda: False,
        walltime=inf,
        termination_timeout=20.,  # kept for compatibility
        raise_if_failure=True,
        log_file='stdout'
    ):
        """
        Run the toy simulation and write trajectory output.

        Parameters
        ----------
        deffnm : str
            Base name for trajectory files (GROMACS-style "deffnm").
            Output is written to `f"{deffnm}{extension}"` unless `noappend`
            is True, in which case `.partXXXX` files are used.
        backup : bool, default True
            Whether to use/maintain a backup file when appending.
            (Behavior preserved exactly as implemented.)
        noappend : bool, default False
            If True, continue from the latest `.partXXXX` file and create a new
            `.partXXXX` file. If False, append by rewriting old frames and then
            adding new ones.
        stop_condition : callable, default `lambda: False`
            Polled periodically from a separate thread. When it returns True,
            the engine stops.
        walltime : float, default inf
            Maximum runtime in seconds.
        termination_timeout : float, default 20.0
            Present only for interface compatibility. Not used by the toy engine.
        raise_if_failure : bool, default True
            If True, raise exceptions on missing/corrupted input. If False,
            return status code 1 on failures.
        log_file : str or file-like, default 'stdout'
            Where to write log output. If `'stdout'`, uses `sys.stdout`.

        Returns
        -------
        int
            Status code:
            - 0 for normal completion (implicit; the code does not explicitly return 0),
            - 1 for handled failures when `raise_if_failure` is False.

        Raises
        ------
        RuntimeError
            If required input trajectory files are missing/corrupted and
            `raise_if_failure` is True.

        Side Effects
        ------------
        - Reads and writes trajectory files on disk.
        - May rename/move existing trajectories when appending.
        - Uses threads to run simulation and stop-condition polling.

        Notes
        -----
        The behavior of backup/restore is preserved as in the code:
        - in append mode, the existing trajectory is temporarily renamed,
          replayed into a new file, and then removed.
        - a backup may be created/restored depending on flags.
        """
        t0 = time.time()
        if log_file == 'stdout':
            log_file = sys.stdout

        # noappend mode: find latest part file, then start a new part.
        if noappend:
            i = 0
            old = None
            backup = ''

            # Sweep forward until a missing/corrupted part is found.
            while True:
                fname = f'{deffnm}.part{i:04g}{self.extension}'
                if not (os.path.exists(fname) and os.path.getsize(fname)):
                    break
                old = fname
                i += 1

            # No existing part file: cannot continue.
            if not old:
                error_msg = f'{fname!r} not existing or corrupted'
                if not raise_if_failure:
                    if log_file:
                        print(f'Warning: {error_msg}', file=log_file)
                    return 1
                raise RuntimeError(error_msg)

            # Sweep backward to find the most recent readable trajectory.
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

        # append mode: load the existing trajectory and append by rewriting it.
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
                print(f'Appending to {fname!r}', file=log_file)

        # Create an empty universe used for writing frames.
        n_atoms = reader.trajectory.n_atoms
        universe = Universe.empty(n_atoms, trajectory=True)
        ts = universe.trajectory.ts

        # In append mode, rename the original file so we can write a new one.
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
            """
            Main writer loop.

            Notes
            -----
            This function writes the output trajectory file:
            - optionally replays old frames (append mode),
            - then generates new frames until walltime or stop condition.
            """
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
            """
            Poll `stop_condition()` until the engine stops.

            Notes
            -----
            This loop sleeps briefly between polls to reduce CPU usage.
            """
            while not self.must_stop:
                self.must_stop = stop_condition()
                time.sleep(.01)  # avoid freezing

        # Run simulation while checking wheter you must stop at the same time
        threads = ThreadExecutor()
        threads.add(main, name='toy engine run')
        threads.add(update_stop_condition, name='toy engine check')
        threads.run()
        while threads.alive.any():
            continue
