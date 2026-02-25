"""
...
"""

# external imports
import os
import numpy as np
from abc import ABC
from math import inf
from glob import glob
from MDAnalysis import Universe, Writer

# aimmd imports
from .utils import get_fnames, get_last_time_and_dt, get_cache_fname
from .._config import MDA_CACHE, DEFAULT_DIMENSIONS

# Path's input/output methods
class PathIO(ABC):

    def extend(self, fnames,
               nframes=inf, skip=0,
               remove_overlapping_frames=True,
               pipeline=()):
        """If fnames is None it just extends from latest.
        Returns also how many other frames you could add (of reader).
        """
        
        if skip is None:
            skip = 0
        else:
            skip = max(skip, 0)
        frames_to_add = nframes or inf 
        if not frames_to_add:
            return 0, 0
        
        # get fnames
        new_fnames = get_fnames(fnames)
        if not new_fnames:
            return 0, 0

        # initialize
        frames_left = 0
        added_frames = 0
        length = 0  # of current reader
        start = len(self)  # where you start computing from
        skip = max(skip - start, 0)
        t1 = None
                
        # update current last file in path
        n_files = len(self._fnames)
        if n_files and not skip and (fname := self._fnames[-1]) in new_fnames:
            end = self._last[-1] + 1
            if self._first[-1] >= end:
                raise RuntimeError("can't extend path running backwards")
            new_fnames = new_fnames[new_fnames.index(fname) + 1:]
            
            # min length that you want to add
            min_length = end + frames_to_add
            
            # how many new frames? (get them through reader)
            reader = MDA_CACHE.get(fname, min_length)
            if reader is None:  # special case: shrink the path instead
                self._fnames.pop(-1)
                self._last.pop(-1)
                added_frames = self._first.pop(-1) - end
                return added_frames, -1
            length = len(reader)
            
            # calculate progress
            frames_left = length - end
            delta = min(frames_left, frames_to_add)
            frames_left -= delta
            frames_to_add -= delta
            added_frames += delta
            
            # update
            self._last[-1] += delta

            # get time info
            try:  # try/except to prevent i/o errors with concurrent access
                t1, dt = get_last_time_and_dt(reader, self._last[-1])
            except:
                new_frames = []

        length = 0
        
        # update new files
        for fname in new_fnames:
            if frames_to_add <= 0:
                frames_left = -1  # the next path (you never know)
                break
            
            # get reader or times
            min_length = skip + frames_to_add
            reader = MDA_CACHE.get(fname, min_length)  
            if reader is None:  # special case: skip
                continue
            length = len(reader)
            
            # no frames to add
            if length <= skip:
                continue
            
            # remove overlapping frames
            if remove_overlapping_frames and t1 is not None:
                
                # current time
                try:  # try/except to prevent i/o errors with concurrent access
                    t0 = round(reader[skip].time, 3)
                except:
                    break
                
                # rewind previous trajectory up until before t0
                while True:
                    if self._last[-1] < 0:
                        self._fnames.pop(-1)
                        self._first.pop(-1)
                        self._last.pop(-1)
                        break
                    if t1 < t0:
                        break
                    t1 = round(t1 - dt, 3)
                    self._last[-1] -= 1
                    frames_to_add += 1
                    if len(self._fnames) == n_files:
                        start -= 1  # compute one frame more
            
            # no frames to add
            if length <= skip:
                continue
            
            # calculate progress
            frames_left = length - skip
            delta = min(frames_left, frames_to_add)
            frames_left -= delta
            frames_to_add -= delta
            added_frames += delta
            
            # update
            self._fnames.append(fname)
            self._first.append(skip)
            end = skip + delta
            self._last.append(end - 1)
            skip = max(skip - length, 0)
            
            # get time info
            try:  # try/except to prevent i/o errors with concurrent access
                t1, dt = get_last_time_and_dt(reader, self._last[-1])
            except:
                break
        
        # compute from "start": only the new frames
        if pipeline:
            path = self[start:] if start else self
            for args in pipeline:
                path.compute(*args)
        
        # return
        return added_frames, frames_left

    def write(self, filename, key=None, atoms=None,
              t0=None, overwrite=False, return_writer=False):
        """
        ...
        """
        
        # not overwriting
        if not overwrite and os.path.exists(filename):
            raise TypeError(f'{filename!r} exists, cannot write')

        # in memory stuff
        times_in_memory = self.in_memory('times')
        positions_in_memory = self.in_memory('positions')
        velocities_in_memory = self.in_memory('velocities')
        dimensions_in_memory = self.in_memory('dimensions')
        if not (times_in_memory and positions_in_memory and
                velocities_in_memory and dimensions_in_memory):
            reader = self.reader
        else:
            reader = np.arange(len(self))  # mockup
        
        if key is not None:
            key = np.arange(len(self))[key].flatten()
            reader = reader[key]
        else:
            key = range(len(self))
        n_frames = len(reader)

        if not n_frames:
            raise RuntimeError("no frames to write")
        
        # get universe and timestep
        indices = np.arange(self.n_atoms)[atoms].flatten()
        n_atoms = len(indices)
        universe = Universe.empty(n_atoms, trajectory=True)
        ts = universe.trajectory.ts
        
        # get time info
        if n_frames >= 2:
            if times_in_memory:
                t1, t2 = self.__dict__['times'][key[:2]]
            else:
                t1 = reader[0].time
                t2 = reader[1].time
            dt = abs(t2 - t1) or 1.0
        else:
            if times_in_memory:
                t1 = self.__dict__['times'][key[0]]
            else:
                t1 = reader[0].time
            dt = 1.0

        # get direction
        directions = np.ones(n_frames)
        directions[:-1][np.diff(key) < 0] = -1.0
        
        # initialize times list
        times = []
        dt = 1.0
        
        # open writer
        writer = Writer(str(filename), len(indices))
        try:
                        
            # iterate over frames
            for i, frame in enumerate(reader):
                # get info

                # time and dt
                if times_in_memory:
                    t2 = self.__dict__['times'][key[i]]
                else:
                    t2 = frame.time
                if i == 1:
                    dt = abs(t2 - t1)
                t1 = t2
                if t0 is not None:
                    t = t0 + dt * i
                else:
                    t = t1

                # positions
                if positions_in_memory:
                    positions = self.__dict__['positions'][key[i]]
                else:
                    positions = frame.positions[indices].copy()

                # velocities (right direction)
                if velocities_in_memory:
                    velocities = self.__dict__['velocities'][key[i]]
                else:
                    velocities = frame._velocities
                    if velocities.any():
                        velocities = velocities[indices] * direction
                    else:
                        velocities = np.zeros((n_atoms, 3))

                # dimensions
                if dimensions_in_memory:
                    dimensions = self.__dict__['dimensions'][key[i]]
                else:
                    dim = frame.dimensions
                    if dim is None:
                        dimensions = DEFAULT_DIMENSIONS
                    else:
                        dimensions = dim
                
                # update and write frame
                ts.positions = positions
                ts.velocities = velocities * directions[i]
                ts.dimensions = dimensions
                ts.time = t
                writer.write(universe)
                times.append(t)
            
            # return writer or written times
            if return_writer:
                return writer
            writer.close()
            return times
        
        # clear in case of exception
        except Exception as exception:
            writer.close()
            raise exception
