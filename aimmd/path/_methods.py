"""
...
"""

# external
import os
import numpy as np
from abc import ABC
from math import inf
from tqdm import tqdm
from numbers import Integral
from itertools import islice

# aimmd imports
from ._get import PathGet
from .utils import split
from .._config import DEFAULT_DIMENSIONS
from ..core.utils import process_state

# path methods
class PathMethods(ABC):

    def is_complete(self, target_state='R', states='ARB'):
        t = process_state(target_state, states)
        i, r, f = states
        
        # reactive
        if t == r:
            return self.type[:3] in (f'{i}{r}{f}', f'{f}{r}{i}',
                                     f'{i}{r}{i}', f'{f}{r}{f}')
        
        # internal
        path_type = self.type
        if path_type[0] in f'.{t}' or path_type[2] == f'.{t}':
            return False
        if path_type[1] != t:
            return False
        return True
    
    def is_transition(self, states='ARB'):
        path_type = self.type[:3]
        return path_type in (states, states[::-1])

    def is_excursion(self, states='ARB'):
        return (self.initial('states') in (states[0], states[-1]) and
                self.middle('states') == states[1])
    
    def is_internal(self, states='ARB'):
        i, m, f = self.type[:3]
        return self.middle('states') in (states[0], states[2])
    
    def check_stop(self, allowed_states='', max_length=inf,
                   check_first_frame=True):
        """automatically adjusts exclude_from
        offset: current last path from start, so
        that you don't have to check from 0 every time
        
        Returns
        -------
        i: frame where to stop, just before crossing
        None if not existing
        """
        if not len(self):
            return None, 0, '', 0
        
        # get info: states
        exclude_from = self._exclude_from
        try:
            self._exclude_from = -1
            states = self.states
        finally:
            self._exclude_from = exclude_from
        if not states[0]:
            return None, 0, '', 0
        
        # restrict states to where they are useful
        nframes = len(states)
        while nframes > 1 and states[nframes - 1] == '':
            nframes -= 1
        states = states[:nframes]
        
        # get info: lengths
        start, stop = split(states)
        lengths = stop - start
        final_states = states[stop - 1]
        n_split_paths = len(lengths)
        
        # get info: states
        if allowed_states == 'all':
            allowed_states = ''

        # cold start
        if allowed_states:
            if check_first_frame and states[0] not in allowed_states:
                raise RuntimeError(
                    f'{self.fnames[0]}, {self.first[0]} in state {states[0]}, '
                    f'should be in {allowed_states}; consider deleting the '
                    f'trajectory file to allow AIMMD to recreate it')
            elif nframes == 1:  # nothing to do
                return None, 1, states[0], 1
            if stop[0] > 1 and states[1] not in allowed_states:
                return (0, nframes, states[1],
                        2 if states[0] != states[1] else lengths[-1])
        
        # condition 1: excluded
        i1 = n_split_paths
        if self._exclude_from >= 0:
            condition = start > self._exclude_from
            if (where := np.flatnonzero(condition)).size:
                i1 = where[0]

        # condition 2: bad state
        i2 = n_split_paths
        if allowed_states:
            condition = np.logical_and.reduce(
                [final_states != s for s in allowed_states])
            if (where := np.flatnonzero(condition)).size:
                i2 = where[0]
        
        # condition 3: max length
        i3 = n_split_paths
        condition = lengths > max_length
        if (where := np.flatnonzero(condition)).size:
            i3 = where[0]

        # which was first?
        i = min(min(i1, i2), i3)
        
        # return
        if i < n_split_paths:
            last_state = states[max(start[i] + 1, stop[i] - 1)]
            return start[i], nframes, last_state, lengths[i]
        return None, nframes, final_states[-1], lengths[-1]
    
    def partial(self, attribute='self', key=None):
        """path part corresponding to k-th fname"""
        if not isinstance(key, Integral):
            return [self._extract(k, attribute)
                    for k in np.arange(self.n_files)[key].flatten()]
        return self._extract(key, attribute)
    
    def in_memory(self, attribute=None):
        if attribute in (None, 'self', 'reader', 'frames'):
            return ('times' in self.__dict__ and
                    'positions' in self.__dict__ and
                    'velocities' in self.__dict__ and
                    'dimensions' in self.__dict__)
        return attribute in self.__dict__
    
    def split(self, return_start_stop=False, states=None):
        """States may be provided by user"""
        from ..pathensemble import PathEnsemble
        
        try:
            states = self.states
        except:
            states = np.full(len(self), '')
        
        result = PathEnsemble()
        if len(states):
            result._paths = [self[start:stop]
                             for start, stop in zip(*split(states))]
        return result
    
    def to_memory(self):
        positions = []
        velocities = []
        dimensions = []
        times = []
        for frame in self.reader:
            positions.append(frame.positions.copy())
            vel = frame._velocities.copy()
            if vel.size:
                velocities.append(vel)
            else:
                velocities.append(positions[-1] * 0.)
            dim = frame.dimensions
            if dim is None:
                dimensions.append(DEFAULT_DIMENSIONS)
            else:
                dimensions.append(dim.copy())
            times.append(frame.time)
        self.positions = positions
        self.velocities = velocities
        self.dimensions = dimensions
        self.times = times
        return self

    def from_files(self):
        self.positions = None
        self.velocities = None
        self.dimensions = None
        self.times = None
        return self

    def update_exclude_from(self, log_fname):
        if os.path.exists(log_fname):
            self._exclude_from = -1
            with open(log_fname) as file:
                for line in file:
                    fields = line.split()
                    if not fields:
                        continue
                    if fields[0].split('/')[-1] in self.fname:
                        if len(fields) == 1:
                            self._exclude_from = 0
                        else:
                            self._exclude_from = int(fields[1])
                        return

    def copy(self):
        from . import Path
        result = object.__new__(Path)        
        result._fnames = self._fnames[:]
        result._first = self._first[:]
        result._end = self._end[:]
        result._weight = self._weight
        result._exclude_from = self._exclude_from
        result._shooting_index = self._shooting_index
        for attribute, value in islice(
            self.__dict__.items(), 6, None):
            result.__dict__[attribute] = value.copy()
        return result
    
    def find_shooting_index(self):
        return np.argmin(self.times)

    def shooting_result(self, states='ARB'):
        a, r, b = states
        states = self.type
        result = np.zeros(2)
        if states[1] != r:
            return result
        result[0] += states[0] == a
        result[0] += states[2] == a
        result[1] += states[0] == b
        result[1] += states[2] == b
        return result
    
    def sample(self, n_samples, state='internal'):
        from . import Path
        result = Path()
        if not n_samples or not len(self):
            return result
        fnames = []
        first = []
        last = []
        for i in np.random.choice(
            self.internal('indices') if state == 'internal' else
            np.flatnonzero(self.states == state), n_samples):
            k, i = self._get_local_index(i)
            fnames.append(self._fnames[k])
            first.append(i)
            last.append(i)
        result._fnames = fnames
        result._first = first
        result._last = last
        return result

    get = PathGet._get
