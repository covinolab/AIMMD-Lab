import os
import time
import numpy as np
import pickle
import inspect
import warnings
import functools
import MDAnalysis as mda
from itertools import chain
from scipy.special import logit, expit

# quick logging
print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")


def process_kwargs(kwargs, name, default):
    if name in kwargs:
        return kwargs[name]
    return default


def padcumsum(x):
    csum=np.hstack((0,x)) # 3x faster than pad.
    csum.cumsum(out=csum)
    return csum


def has_param(func, param_name):
    signature = inspect.signature(func)
    return param_name in signature.parameters


class MDATrajectory:
    
    def __init__(self, universes, frame_trajectory_indices,
                 frame_trajectory_positions):
        """
        Trajectories is a mdanalysis trajectory iterator.
        """
        self.universes = universes
        self.frame_trajectory_indices = frame_trajectory_indices
        self.frame_trajectory_positions = frame_trajectory_positions
        self.current = 0
    
    def reset(self, mapping=None):
        for universe in self.universes:
            universe.trajectory.rewind()
        self.current = 0
    
    def __repr__(self):
        return f'MDATrajectory with {len(self)} frames'
    
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_trajectory_indices)
    
    def __next__(self):
        if self.current >= len(self.frame_trajectory_indices):
            raise StopIteration
        trajectory_index = self.frame_trajectory_indices[self.current]
        trajectory_position = self.frame_trajectory_positions[self.current]
        self.current += 1
        return (self.universes[trajectory_index].trajectory.
                _read_frame_with_aux(trajectory_position))
    
    def __getitem__(self, key):
        result = []
        key = np.arange(len(self))[key]
        if not hasattr(key, '__len__'):
            current = key
            trajectory_index = int(self.frame_trajectory_indices[current])
            trajectory_position = int(self.frame_trajectory_positions[current])
            return (self.universes[trajectory_index].trajectory.
                    _read_frame_with_aux(trajectory_position))
        return [self.__getitem__(key).copy() for key in key]
    
    def __add__(self, element):
        return MDATrajectory(
            self.universes + element.universes,
            np.append(self.frame_trajectory_indices,
                      element.frame_trajectory_indices +
                      len(self.universes)),
            np.append(self.frame_trajectory_positions,
                      element.frame_trajectory_positions))
    
    @property
    def filenames(self):
        return [universe.trajectory.filename
                for universe in self.universes]
    
    @property
    def filename(self):
        return self.filenames[0] if len(self.filenames) else ''
    
    def close(self):  # TODO check
        for universe in self.universes:
            universe.trajectory.close()
    
    def write(self, filename, frame_indices=None, selection='all',
              invert_velocities=False, joined_shooting_segments=False,
              reset_time=False, overwrite=False):
        if os.path.exists(filename) and not overwrite:
            raise ValueError('Cannot write to an existing file.')
        atom_groups = [universe.select_atoms(selection)
                       for universe in self.universes]
        frame_indices = np.arange(len(self))[frame_indices].ravel()
        if reset_time:
            if len(frame_indices) > 1:
                dt = np.abs(self.__getitem__(frame_indices[-1]).time -
                            self.__getitem__(frame_indices[-2]).time)
                times = -np.arange(len(frame_indices),
                                   dtype=float)[::-1] * dt
            else:
                times = np.array([0.])
        past_shooting_point = False
        with mda.Writer(filename, atom_groups[0].n_atoms) as writer:
            for i, frame_index in enumerate(frame_indices):
                frame_index = int(frame_index)
                frame = self.__getitem__(frame_index)
                if invert_velocities:
                    frame._velocities *= -1
                elif joined_shooting_segments and not past_shooting_point:
                    if frame.time > 0:  # before SP: invert velocities
                        frame._velocities *= -1
                    else:
                        past_shooting_point = False
                if reset_time:
                    frame.time = times[i]
                trajectory_index = self.frame_trajectory_indices[frame_index]
                writer.write(atom_groups[int(trajectory_index)])


class PathEnsemblesIterator:
    def __init__(self, *pathensembles):
        self.pathensembles = pathensembles
        self.length = len(pathensembles)
        self.lengths = np.array([len(pathensemble)
                                 for pathensemble in pathensembles])
        self.current_path = 0
        self.current_pathensemble = 0
        self.end = np.sum(self.lengths)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while self.current_pathensemble < self.length:
            while self.current_path < self.lengths[
                self.current_pathensemble]:
                self.current_path += 1
                return self.pathensembles[self.current_pathensemble][
                    self.current_path - 1]
            else:
                self.current_path = 0
                self.current_pathensemble += 1
                continue
        else:
            raise StopIteration


class AbstractPathEnsemble:
    """
    Functions that PathEnsemble and PathEnsemblesCollection have in common.
    """
    
    """
    Extraction: MDA.
    """
    
    def frames(self, frame_indices=None):
        
        # get frame_trajectory_indices and positions
        frame_trajectory_indices = self.frame_trajectory_indices[
            frame_indices].ravel()
        frame_trajectory_positions = self.frame_trajectory_positions[
            frame_indices].ravel()
        
        if not len(frame_trajectory_indices):
            return MDATrajectory([], [], [])
        
        # get trajectories
        indices = np.unique(frame_trajectory_indices)
        files = np.array(self.trajectory_files)[indices]
        directories = np.array(self.trajectory_directories)[indices]
        topologies = np.array(self.trajectory_topologies)[indices]
        universes = [mda.Universe(f'{directory}/{topology}',
                                  f'{directory}/{file}')
                     for file, directory, topology in zip(
                         files, directories, topologies)]
        
        # process trajectory index to match the subset list
        new_frame_trajectory_indices = np.zeros(
            len(frame_trajectory_indices), dtype=int)
        for i, index in enumerate(indices):
            new_frame_trajectory_indices[frame_trajectory_indices == index] = i
        
        return MDATrajectory(universes, new_frame_trajectory_indices,
                             frame_trajectory_positions)
    
    def path(self, key=None, internal=False, backward=True, forward=True):
        return self.frames(np.concatenate(
            self.frame_indices(key, internal, backward, forward)))
    
    """
    Analysis/reweighting.
    """
    
    def cumulative_time(self, timestep=1., shots=False):
        """
        Return assumed total simulated time (no repeated frames).
        """
        return np.sum(self.internal_lengths) * timestep
    
    def max_values(self, key=None, backward=True, forward=True):
        values = self.values(key, True, backward, forward)
        condition = np.zeros(len(values), dtype=bool)
        if backward:
            condition += self.initial_states[key].ravel() == 'B'
        if forward:
            condition += self.final_states[key].ravel() == 'B'
        return np.array([+np.inf if c else np.max(v, where=(v != 0), initial=-np.inf) # The guard against zero values is to deal with sparse updates
                         for v, c in zip(values, condition)])
    
    def min_values(self, key=None, backward=True, forward=True):
        values = self.values(key, True, backward, forward)
        condition = np.zeros(len(values), dtype=bool)
        if backward:
            condition += self.initial_states[key].ravel() == 'A'
        if forward:
            condition += self.final_states[key].ravel() == 'A'
        return np.array([-np.inf if c else np.min(v, where=(v != 0), initial=np.inf) # The guard against zero values is to deal with sparse updates
                         for v, c in zip(values, condition)])
    
    def density(self, values, shooting_value, neighbors=10):
        """
        Goes to densities.
        """
        
        # process values
        values = values.astype(float)
        values = values[(~np.isinf(values)) * (~np.isnan(values))]
        if (len(values) < 3 or
            np.isinf(shooting_value) or
            np.isnan(shooting_value)):
            return np.inf
        
        # min and max value
        min_value = np.min(values)
        max_value = np.max(values)
        values = np.concatenate(
            [[min_value], np.sort(values[1:-1]), [max_value]])
        differences = np.abs(shooting_value - values)
        begin = 0
        end = len(values)
        while end - begin > neighbors:
            if differences[begin] > differences[end - 1]:
                begin += 1
            else:
                end -= 1
        
        # computation
        upper_boundary = (values[end - 1] +
                            values[min(end, len(values) - 1)]) / 2
        lower_boundary = (values[max(begin - 1, 0)] +
                            values[begin]) / 2
        n_of_neighbors = end - begin
        return np.nan_to_num(
            n_of_neighbors / (upper_boundary - lower_boundary),
            nan=1.0)
    
    def densities(self, key=None, backward=True, forward=True,
                  neighbors=10, norm=10, cutoff=1.):
        """
        n^star at the shooting point
        Works only with shot excursions.
        """
        # process input and intialize result
        path_indices = np.arange(len(self))[key].ravel()
        shooting_values = self.shooting_values[path_indices]
        densities = np.array([
            self.density(values, shooting_value, neighbors)
            for values, shooting_value in zip(
                self.values(path_indices), shooting_values)])
        
        # uniformizing are_shot with 0 < densities < np.inf
        densities[densities <= 0] = np.inf
        mask = np.where(densities < np.inf)[0]
        
        if norm:    
            s = shooting_values[mask]
            d = densities[mask]
            for i, v in zip(mask, shooting_values[mask]):
                keepers = (s > (v - cutoff / 2)) * (s < (v + cutoff / 2))
                if np.sum(keepers) < norm:
                    keepers = np.argsort(np.abs(v - s))[:norm]
                
                densities[i] *= np.mean(1 / d[keepers])
        return densities * np.mean(1 / densities[mask])
    
    def reweight(self, state='A', key=None, **kwargs):
        
        # specific reweighting params
        equilibrium_threshold = process_kwargs(
            kwargs, 'equilibrium_threshold', None)
        theoretical_threshold = process_kwargs(
            kwargs, 'theoretical_threshold', None)
        equilibrium_importance = process_kwargs(
            kwargs, 'equilibrium_importance', 1.)
        crossing_probability_cutoff = process_kwargs(
            kwargs, 'crossing_probability_cutoff', 0.)
        factors_neighbors = process_kwargs(
            kwargs, 'factors_neighbors', 10)
        factors_norm = process_kwargs(
            kwargs, 'factors_norm', 10)
        factors_cutoff = process_kwargs(
            kwargs, 'factors_cutoff', 1.)
        sp_cutoff_min = process_kwargs(
            kwargs, 'sp_cutoff_min', None)
        sp_cutoff_max = process_kwargs(
            kwargs, 'sp_cutoff_max', None)
        corrections = process_kwargs(
            kwargs, 'corrections', None)
        
        # process path indices
        path_indices = np.arange(len(self))[key].ravel()
        if not len(path_indices):
            return (np.array([], dtype=float),  # weights
                    np.array([], dtype=int),  # path_indices (array of indices)
                    np.array([], dtype=int),  # excursions (array of bools)
                    np.array([], dtype=int),  # internal_segments (as above)
                    np.array([], dtype=float),  # crossing probability
                    np.array([], dtype=float),  # m
                    np.array([], dtype=float),  # extremes
                    np.array([], dtype=float),  # shooting values
                    np.array([], dtype=float))  # factors

        # process corrections
        if corrections is None:
            corrections = np.ones(len(path_indices))
        
        # initialize results
        weights = np.zeros(len(path_indices))
        
        # batch info for faster computation
        internal_states = self.internal_states[path_indices]
        initial_states = self.initial_states[path_indices]
        final_states = self.final_states[path_indices]
        shooting_states = self.shooting_states[path_indices]
        shooting_values = self.shooting_values[path_indices]
        are_accepted = self.are_accepted[path_indices]

        # which are excursions?
        excursions = (internal_states == 'R') * are_accepted

        # enforce sp cutoff for shot excursions (not the equilibrium)
        # basically, the factor is computed in a different way
        # and then the path is faked as an equilibrium excursion
        sp_cutoff_densities = np.zeros(len(excursions))
        if sp_cutoff_min is not None and state == 'A':
            sp_cutoff_densities[:] = ((shooting_values < sp_cutoff_min) *
                                      (shooting_states == 'R') * excursions)
            sp_cutoff_densities *= ((initial_states == state) +
                                    (final_states == state))
            k = sp_cutoff_densities > 0
            sp_cutoff_densities[k] = np.array([
                np.sum(value < sp_cutoff_min) for value in
                self.values(path_indices[sp_cutoff_densities > 0],
                          internal=True)]).astype(float)
        if sp_cutoff_max is not None and state == 'B':
            sp_cutoff_densities[:] = ((shooting_values > sp_cutoff_max) *
                                      (shooting_states == 'R') * excursions)
            sp_cutoff_densities *= ((initial_states == state) +
                                    (final_states == state))
            k = sp_cutoff_densities > 0
            sp_cutoff_densities[k] = np.array([
                np.sum(value > sp_cutoff_max) for value in
                self.values(path_indices[sp_cutoff_densities > 0],
                          internal=True)]).astype(float)
        if np.sum(sp_cutoff_densities):
            k = sp_cutoff_densities > 0
            sp_cutoff_densities *= np.mean(1 / sp_cutoff_densities[k])
            sp_cutoff_densities[k] = np.maximum(sp_cutoff_densities[k], 1e-3)
        
        # compute densities for correction factor in m computation
        shot_excursions = excursions * (shooting_states == 'R')
        shot_indices = path_indices[shot_excursions]
        shot_shooting_values = shooting_values[shot_excursions]
        densities = np.ones(len(path_indices))
        densities[shot_excursions] = self.densities(
            shot_indices,
            neighbors=factors_neighbors,
            norm=factors_norm,
            cutoff=factors_cutoff)
        k = sp_cutoff_densities > 0  # change based on sp_cutoff_densities
        densities[k] = sp_cutoff_densities[k]
        corrections[k] /= sp_cutoff_densities[k]
        k = densities > 0
        densities[k] = np.maximum(densities[k], 1e-3)
        k = corrections > 0
        corrections[k] = np.minimum(corrections[k], 1000.)
        
        # classify segments (exclude not accepted paths)
        forward_excursions = excursions * (initial_states == state)
        backward_excursions = excursions * (final_states == state) * (
            shooting_states == 'R') * (sp_cutoff_densities == 0)
        excursions = forward_excursions + backward_excursions
        transitions = excursions * (
            (initial_states != 'R') * (final_states != 'R') *
           ((initial_states != state) + (final_states != state)))
        backward_excursions *= shooting_states == 'R'
        internal_segments = ((internal_states == state) *
                             (initial_states == 'R')) * are_accepted
        
        # compute extremes and shooting_values
        if state == 'A':
            xP_extremes = np.append(
                + self.max_values(
                    path_indices[forward_excursions], backward=False),
                + self.max_values(
                    path_indices[backward_excursions], forward=False))
            xP_shooting_values = np.append(
                + shooting_values[forward_excursions],
                + shooting_values[backward_excursions])
            extremes = + self.max_values(path_indices[excursions])
            shooting_values = + shooting_values[excursions]
        else:
            xP_extremes = np.append(
                - self.min_values(
                    path_indices[forward_excursions], backward=False),
                - self.min_values(
                    path_indices[backward_excursions], forward=False))
            xP_shooting_values = np.append(
                - shooting_values[forward_excursions],
                - shooting_values[backward_excursions])
            extremes = - self.min_values(path_indices[excursions])
            shooting_values = - shooting_values[excursions]
        
        # process extremes and shooting values
        # this is for crossing probability computation
        xP_shooting_states = np.append(
            shooting_states[forward_excursions],
            shooting_states[backward_excursions])
        xP_extremes = np.maximum(-100., xP_extremes)  # limit
        xP_shooting_values = np.maximum(-200., xP_shooting_values)  # limit
        xP_shooting_values[xP_shooting_states == state] = -np.inf  # eq
        xP_factors = np.append(
            corrections[forward_excursions],
            corrections[backward_excursions])
        xP_factors[np.isinf(xP_shooting_values)] *= equilibrium_importance
        k = np.append(  # treat cutoff paths like equilibrium
            sp_cutoff_densities[forward_excursions] > 0,
            np.zeros(np.sum(backward_excursions), dtype=bool))
        xP_shooting_values[k] = - np.inf
        
        # this is for the actual reweighting
        shooting_states = shooting_states[excursions]
        densities = densities[excursions]
        extremes = np.maximum(-100., extremes)  # limit
        shooting_values = np.maximum(-200., shooting_values)  # limit
        # shooting_values[shooting_states != 'R'] = +np.inf  # eq to state
        shooting_values[shooting_states == state] = - np.inf  # eq from state
        shooting_values[sp_cutoff_densities[excursions] > 0] = - np.inf # as ^
        # extremes[shooting_values == np.inf] = +np.inf
        factors = 1 / densities
        shot_transitions = transitions[excursions] * (shooting_states == 'R')
        factors[shot_transitions] /= 2.  # otw counting them double
        
        # sort
        xP_order = np.argsort(xP_extremes)
        xP_extremes = xP_extremes[xP_order]
        xP_shooting_values = xP_shooting_values[xP_order]
        xP_shooting_states = xP_shooting_states[xP_order]
        xP_factors = xP_factors[xP_order]
        order = np.argsort(extremes)
        extremes = extremes[order]
        shooting_values = shooting_values[order]
        shooting_states = shooting_states[order]
        factors = factors[order]
        
        # compute xP: equilibrium part
        if equilibrium_threshold is not None:
            equilibrium_threshold = max(equilibrium_threshold, 1)
            xP_equilibrium_excursions = xP_shooting_states == state
            xP_equilibrium_extremes = xP_extremes[xP_equilibrium_excursions]
            n_eq = len(xP_equilibrium_extremes)
            if n_eq: 
                n_th = np.clip(equilibrium_threshold, 1, n_eq)
                th = xP_equilibrium_extremes[-n_th]
                mask = xP_extremes > th
                xP_extremes = np.append(
                    xP_equilibrium_extremes[:n_eq - n_th + 1],
                    xP_extremes[mask])
                xP_shooting_values = np.append(
                    np.repeat(-np.inf, n_eq - n_th + 1),
                    xP_shooting_values[mask])
                xP_factors = np.append(
                    np.repeat(equilibrium_importance, n_eq - n_th + 1),
                    xP_factors[mask])
            else:
                n_th = 0
        elif len(xP_extremes):
            n_eq = 1
            n_th = 1
        else:
            n_eq = 0
            n_th = 0
        xP = np.ones(len(xP_extremes))
        i = n_eq - n_th
        if n_eq:
            xP[:i + 1] = np.arange(n_eq, n_th - 1, -1.) / n_eq
        
        # compute xP: fitted part
        if theoretical_threshold is None:
            theoretical_threshold = + np.inf
        for i in range(i, len(xP_extremes) - 1):
            if xP_extremes[i] >= theoretical_threshold:
                break
            sv = xP_shooting_values[i:]
            f = xP_factors[i:]
            e = xP_extremes[i:]
            mask = sv < (xP_extremes[i] - crossing_probability_cutoff)
            numerator = f[0]
            denominator = np.sum(f[mask])
            if denominator and mask[0]:
                drop = max(1 -  numerator / denominator, 0.5)
            elif denominator:
                drop = 1.
            else:
                drop = .5
            xP[i + 1] = xP[i] * drop
        
        # compute xP: theoretical part & normalization
        expit_xP_extremes = expit(xP_extremes[i:])
        if len(xP):  # next
            xP[i + 1:] = xP[i] * (
                expit_xP_extremes[0] / expit_xP_extremes[1:])
        
        # compute m: preliminary
        m = np.zeros(len(extremes))
        k = np.isinf(shooting_values)
        m[:np.sum(k)] = np.cumsum(factors[k][::-1])[::-1]
        th = +np.inf
        if np.sum(shooting_values > -np.inf):
            th = np.min(shooting_values[shooting_values > -np.inf])
        
        # compute m: actual computation
        current_extreme = -np.inf
        current_m = 1.
        for i in range(len(m)):
            if extremes[i] > current_extreme:  # need to update
                current_extreme = extremes[i]
                if current_extreme < th:
                    current_m = m[i]
                    continue  # precomputed value
                else:
                    current_m = np.sum(factors[i:] * 
                              (shooting_values[i:] <= current_extreme))
            m[i] = current_m
            if current_extreme == +np.inf:
                m[i:] = current_m
                break
        
        # crossing probability conversion to extremes
        xP = xP[np.minimum(len(xP) - 1,
                    np.searchsorted(xP_extremes, extremes, 'left'))]
        
        # weights (normalized such that transitions = 1)
        n_tr = np.sum(extremes == +np.inf)
        excursions_weights = factors * xP / m
        if len(xP):  # align indices
            excursions_weights /= xP[-1] * expit_xP_extremes[-1]
        weights[np.where(excursions)[0][order]] = excursions_weights
        weights[internal_segments] = 1.
        norm = np.sum(excursions_weights)
        n_internal = np.sum(internal_segments)
        if norm and n_internal:
            weights[internal_segments] = norm / n_internal
        
        # return ordered indices
        internal_segments = path_indices[internal_segments]
        excursions = path_indices[excursions][order]
        
        return (weights, path_indices, internal_segments, excursions,
                xP, m, extremes, shooting_values, factors)

    @property
    def frame_weights(self):
        frame_weights = np.zeros(self.nframes)
        for frame_indices, weight, is_accepted in zip(
                self.frame_indices(internal=True),
                self.weights, self.are_accepted):
            frame_weights[frame_indices] += weight * is_accepted
        return frame_weights


class PathEnsemble(AbstractPathEnsemble):
    """
    Main class.
    """

    __slots__ = ['__trajectory_files',
                 '__frame_trajectory_indices',
                 '__frame_trajectory_positions',
                 '__frame_times',
                 '__frame_simulation_times',
                 '__frame_states',
                 '__frame_descriptors',
                 '__frame_values',
                 '__frame_indices',
                 '__lengths',
                 '__weights',
                 '__shooting_indices',
                 '__selection_biases',
                 '__are_accepted',
                 'directory',
                 'topology']
    _save_descriptors = True

    def __init__(self, directory='.', topology='',
        states_function=None, descriptors_function=None,
        values_function=lambda descriptors: np.repeat(0., len(descriptors))):
        
        """
        Initialize empty `PathEnsemble`. Populate with `append` or `__sum__`.
        """
        
        # trajectory attributes
        self.__trajectory_files = []
        
        # frame attributes
        self.__frame_trajectory_indices = np.empty(0, dtype=int)
        self.__frame_trajectory_positions = np.empty(0, dtype=int)
        self.__frame_times = np.empty(0, dtype=np.float64)
        self.__frame_simulation_times = np.empty(0, dtype=np.float64)
        self.__frame_states = np.empty(0, dtype='<U1')
        self.__frame_descriptors = np.empty((0, 0), dtype=np.float64)
        self.__frame_values = np.empty(0, dtype=np.float64)

        # navigation attributes
        self.__frame_indices = np.empty(0, dtype=int)
        
        # path attributes
        self.__lengths = np.empty(0, dtype=int)
        self.__weights = np.empty(0, dtype=np.float64)
        self.__shooting_indices = np.empty(0, dtype=int)
        self.__selection_biases = np.empty(0, dtype=np.float64)
        self.__are_accepted = np.empty(0, dtype=bool)
        
        # processing
        self.directory = str(directory)
        self.topology = str(topology)
        self.states_function = states_function
        self.descriptors_function = descriptors_function
        self.values_function = values_function
    
    """
    Attributes: files.
    """
    
    @property
    def trajectory_files(self):
        return self.__trajectory_files
    
    @property
    def trajectory_directories(self):
        return np.repeat(self.directory, len(self.__trajectory_files))
    
    @property
    def trajectory_topologies(self):
        return np.repeat(self.topology, len(self.__trajectory_files))
    
    """
    Attributes: frames.
    """
    
    @property
    def frame_trajectory_indices(self):
        return self.__frame_trajectory_indices.copy()
    
    @property
    def frame_trajectory_positions(self):
        return self.__frame_trajectory_positions.copy()
    
    @property
    def frame_times(self):
        return self.__frame_times.copy()
    
    @property
    def frame_simulation_times(self):
        return self.__frame_simulation_times
    
    @property
    def frame_states(self):
        return self.__frame_states
    
    @property
    def frame_descriptors(self):
        return self.__frame_descriptors
    
    @property
    def frame_values(self):
        return self.__frame_values

    @frame_simulation_times.setter
    def frame_simulation_times(self, frame_simulation_times):
        self.__frame_simulation_times[:] = frame_simulation_times

    @frame_states.setter
    def frame_states(self, frame_states):
        self.__frame_states[:] = frame_states

    @frame_descriptors.setter
    def frame_descriptors(self, frame_descriptors):
        self.__frame_descriptors[:] = frame_descriptors

    @frame_values.setter
    def frame_values(self, frame_values):
        self.__frame_values[:] = frame_values
    
    """
    Attributes: paths.
    """
    
    @property
    def lengths(self):
        return self.__lengths.copy()
    
    @property
    def weights(self):
        return self.__weights
    
    @property
    def shooting_indices(self):
        return self.__shooting_indices

    @property
    def selection_biases(self):
        return self.__selection_biases
    
    @property
    def are_accepted(self):
        return self.__are_accepted
    
    @weights.setter
    def weights(self, weights):
        self.__weights[:] = weights
    
    @shooting_indices.setter
    def shooting_indices(self, shooting_indices):
        self.__shooting_indices[:] = shooting_indices

    @selection_biases.setter
    def selection_biases(self, selection_biases):
        self.__selection_biases[:] = selection_biases
    
    @are_accepted.setter
    def are_accepted(self, are_accepted):
        self.__are_accepted[:] = are_accepted
    
    """
    Special properties.
    """
    
    @property
    def nframes(self):
        return self.__frame_trajectory_indices.size
    
    @property
    def boundaries(self):
        return padcumsum(self.__lengths)
    
    """
    Extraction: indexing.
    """
    
    def frame_indices(self, key=None,
                      internal=False, backward=True, forward=True):
        
        # process paths_index
        path_indices = np.arange(len(self))[key].ravel()
        
        # iterate
        boundaries = self.boundaries
        internal_states = self.internal_states[path_indices]
        frame_indices = []
        for path_index, internal_state in zip(path_indices, internal_states):
            if backward and forward:
                frame_indices.append(
                    self.__frame_indices[boundaries[path_index]:
                                         boundaries[path_index + 1]])
            elif backward:
                si = self.__shooting_indices[path_index]
                frame_indices.append(
                    self.__frame_indices[boundaries[path_index]:
                                         boundaries[path_index] + si + 1])
            elif forward:
                si = self.__shooting_indices[path_index]
                frame_indices.append(
                    self.__frame_indices[boundaries[path_index] + si:
                                         boundaries[path_index + 1]])
            else:
                si = self.__shooting_indices[path_index]
                frame_indices.append(
                    self.__frame_indices[[boundaries[path_index] + si]])
            if internal:
                frame_indices[-1] = frame_indices[-1][
                    self.__frame_states[frame_indices[-1]] == internal_state]
        return frame_indices
    
    """
    Extraction: path attributes.
    """
    
    def _extract(self, array, key=None,
                 internal=False, backward=True, forward=True):
        return [array[frame_indices] for frame_indices in
                self.frame_indices(key, internal, backward, forward)]
    
    def trajectory_indices(self, key=None,
                           internal=False, backward=True, forward=True):
        return self._extract(self.__frame_trajectory_indices,
                             key, internal, backward, forward)
    
    def trajectory_filenames(self, key=None,
                          internal=False, backward=True, forward=True):
        trajectory_filenames = np.array([f'{self.directory}/{filename}'
            for filename in self.__trajectory_files])
        return [trajectory_filenames[trajectory_indices]
                for trajectory_indices in self.trajectory_indices(
                    key, internal, backward, forward)]
    
    def trajectory_positions(self, key=None,
                              internal=False, backward=True, forward=True):
        return self._extract(self.__frame_trajectory_positions,
                             key, internal, backward, forward)
    
    def times(self, key=None,
              internal=False, backward=True, forward=True):
        return self._extract(self.__frame_times,
                             key, internal, backward, forward)
    
    def simulation_times(self, key=None,
                         internal=False, backward=True, forward=True):
        return self._extract(self.__frame_simulation_times,
                             key, internal, backward, forward)
    
    def states(self, key=None,
               internal=False, backward=True, forward=True):
        return self._extract(self.__frame_states,
                             key, internal, backward, forward)
    
    def descriptors(self, key=None,
                    internal=False, backward=True, forward=True):
        return self._extract(self.__frame_descriptors,
                             key, internal, backward, forward)
    
    def values(self, key=None,
               internal=False, backward=True, forward=True):
        return self._extract(self.__frame_values,
                             key, internal, backward, forward)
    
    """
    Extraction: path attributes in special positions.
    """
    
    @property
    def initial_frame_indices(self):
        return self.__frame_indices[self.boundaries[:-1]]
    
    @property
    def shooting_frame_indices(self):
        return self.__frame_indices[self.boundaries[:-1] +
                                    self.shooting_indices]
    
    @property
    def final_frame_indices(self):
        return self.__frame_indices[self.boundaries[1:] - 1]
    
    @property
    def initial_frames(self):
        return self.frames(self.initial_frame_indices)
    
    @property
    def shooting_frames(self):
        return self.frames(self.shooting_frame_indices)
    
    @property
    def final_frames(self):
        return self.frames(self.final_frame_indices)
    
    @property
    def initial_trajectory_filenames(self):
        return np.array(self.__trajectory_files)[
            self.initial_trajectory_indices]
    
    @property
    def shooting_trajectory_filenames(self):
        return np.array(self.__trajectory_files)[
            self.shooting_trajectory_indices]
    
    @property
    def final_trajectory_filenames(self):
        return np.array(self.__trajectory_files)[
            self.final_trajectory_indices]
    
    @property
    def initial_trajectory_indices(self):
        return self.__frame_trajectory_indices[self.initial_frame_indices]
    
    @property
    def shooting_trajectory_indices(self):
        return self.__frame_trajectory_indices[self.shooting_frame_indices]
    
    @property
    def final_trajectory_indices(self):
        return self.__frame_trajectory_indices[self.final_frame_indices]
    
    @property
    def initial_trajectory_positions(self):
        return self.__frame_trajectory_positions[self.initial_frame_indices]
    
    @property
    def shooting_trajectory_positions(self):
        return self.__frame_trajectory_positions[self.shooting_frame_indices]
    
    @property
    def final_trajectory_positions(self):
        return self.__frame_trajectory_positions[self.final_frame_indices]
    
    @property
    def initial_times(self):
        return self.__frame_times[self.initial_frame_indices]
    
    @property
    def shooting_times(self):
        return self.__frame_times[self.shooting_frame_indices]
    
    @property
    def final_times(self):
        return self.__frame_times[self.final_frame_indices]
    
    @property
    def initial_simulation_times(self):
        return self.__frame_simulation_times[self.initial_frame_indices]
    
    @property
    def shooting_simulation_times(self):
        return self.__frame_simulation_times[self.shooting_frame_indices]
    
    @property
    def final_simulation_times(self):
        return self.__frame_simulation_times[self.final_frame_indices]
    
    @property
    def completion_times(self):
        result = self.final_simulation_times
        #result[self.final_states == self.internal_states] = np.inf
        return result
    
    @property
    def initial_states(self):
        return self.__frame_states[self.initial_frame_indices]
    
    @property
    def shooting_states(self):
        return self.__frame_states[self.shooting_frame_indices]
    
    @property
    def final_states(self):
        return self.__frame_states[self.final_frame_indices]
    
    @property
    def initial_descriptors(self):
        return self.__frame_descriptors[self.initial_frame_indices]
    
    @property
    def shooting_descriptors(self):
        return self.__frame_descriptors[self.shooting_frame_indices]
    
    @property
    def final_descriptors(self):
        return self.__frame_descriptors[self.final_frame_indices]
    
    @property
    def initial_values(self):
        return self.__frame_values[self.initial_frame_indices]
    
    @property
    def shooting_values(self):
        return self.__frame_values[self.shooting_frame_indices]
    
    @property
    def final_values(self):
        return self.__frame_values[self.final_frame_indices]
    
    """
    Path properties.
    """
    
    @property
    def internal_states(self):
        internal_states = np.repeat('R', len(self))
        mask = self.__lengths <= 2
        internal_states[mask] = self.initial_states[mask]
        mask = self.__lengths > 2
        internal_states[mask] = self.__frame_states[
            self.__frame_indices[self.boundaries[:-1][mask] + 1]]
        return internal_states
    
    @property
    def internal_lengths(self):
        return np.array([np.sum(states == state) for states, state in
                         zip(self.states(), self.internal_states)])
    
    @property
    def are_shot(self):
        return ((self.shooting_frame_indices != self.initial_frame_indices) *
                (self.shooting_frame_indices != self.final_frame_indices))
    
    @property
    def are_equilibrium(self):
        return ((self.shooting_frame_indices == self.initial_frame_indices) +
                (self.shooting_frame_indices == self.final_frame_indices))
    
    @property
    def are_excursions(self):
        return (self.internal_states == 'R') * (
               (self.initial_states != 'R') + (self.final_states != 'R'))
    
    @property
    def are_internal(self):
        return self.internal_states != 'R'
    
    @property
    def shooting_results(self):
        shooting_results = np.zeros((len(self), 2))
        keepers = self.are_excursions
        initial_states = self.initial_states[keepers]
        final_states = self.final_states[keepers]
        shooting_results[keepers, 0] += initial_states == 'A'
        shooting_results[keepers, 0] += final_states == 'A'
        shooting_results[keepers, 1] += initial_states == 'B'
        shooting_results[keepers, 1] += final_states == 'B'
        return shooting_results
    
    @property
    def are_transitions(self):
        shooting_results = self.shooting_results
        return ((np.sum(shooting_results, axis=1) == 2) *
                 shooting_results[:, 0] == 1.)
    
    """
    Update.
    """
    
    def _update(self, **kwargs):
        """
        Directly modify attributes. Attention! Will not check.
        """
        for slot in self.__slots__:
            if slot[:2] == '__':
                name = slot[2:]
                slot = f'_PathEnsemble{slot}'
            else:
                name = slot
            if name in kwargs:
                setattr(self, slot, kwargs[name])

    def reset(self):
        """
        Reset to pristine state.
        Attention! Everything is done in-place. Call with caution.
        """
        self.__trajectory_files = []
        self.__frame_trajectory_indices = np.empty(0, dtype=int)
        self.__frame_trajectory_positions = np.empty(0, dtype=int)
        self.__frame_times = np.empty(0, dtype=np.float64)
        self.__frame_simulation_times = np.empty(0, dtype=np.float64)
        self.__frame_states = np.empty(0, dtype='<U1')
        self.__frame_descriptors = self.__frame_descriptors[:0]  # preserve
        self.__frame_values = np.empty(0, dtype=np.float64)
        self.__frame_indices = np.empty(0, dtype=int)
        self.__lengths = np.empty(0, dtype=int)
        self.__weights = np.empty(0, dtype=np.float64)
        self.__shooting_indices = np.empty(0, dtype=int)
        self.__selection_biases = np.empty(0, dtype=np.float64)
        self.__are_accepted = np.empty(0, dtype=bool)    
    
    def update_states(self, verbose=False):
        """
        Calls again states_function on the frames.
        """
        if has_param(self.states_function, 'verbose'):
            states = self.states_function(self.frames(), verbose=verbose)
        else:
            states = self.states_function(self.frames())
        self.__frame_states[:] = states
    
    def update_descriptors(self, verbose=False):
        """
        Calls again descriptors_function on the frames.
        """
        if self.descriptors_function is not None:
            if has_param(self.descriptors_function, 'verbose'):
                descriptors = self.descriptors_function(
                    self.frames(), verbose=verbose)
            else:
                descriptors = self.descriptors_function(self.frames())
            self.__frame_descriptors = descriptors
        else:
            self.__frame_descriptors = np.zeros((self.nframes, 0))
    
    def update_values(self,
                      only_reactive=False,
                      only_zeros=False,
                      values_function=None, 
                      key=None,
                      check_for_network_change=True,
                      sparse_update_max_frames=-1,
                      verbose=False):
        """
        Calls again descriptors_function over frame_descriptors, and then the network to evaluate values.
        If the network has not changed from the last call, only evaluate 
        values for new additions to the ensemble that have not
        been properly initialised (identical to only_zeros=True).
        If it has changed, proceed to evaluate those frames for which the parameters request it.

        As PathEnsemble doesn't track changes to the network, which is not passed to PathEnsemble at all,
        it will be assumed that if network outputs for the first 3 frames in the path ensemble are unchanged,
        the network hasn't changed (for deterministic value functions).

        Parameters
        ----------
        only_reactive : bool, optional
            If True, only re-evaluate values for frames in reactive states.
            Default is False.
        only_zeros : bool, optional
            If True, only re-evaluate values for frames with value 0.
            Default is False.
        values_function : function, optional
            Function to compute values from descriptors. If None, use the one
            stored in the PathEnsemble. Default is None.
        key : array-like, optional
            Indices of paths to consider. If None, consider all paths. Default is None.
        check_for_network_change : bool, optional
            If True, check if the network has changed since last evaluation.
            Default is True.
        sparse_update_max_frames : int, optional
            If set to other than -1, and if the number of frames to be updated exceeds this
            number, only update a random subset of frames up to this number.
            The other frames will be set to zero values, to avoid carry-over
            between sparse bin updates. Incompatible with
            checking for network change, if set check_for_network_change will set to False.
            Default is -1.
        verbose : bool, optional
            If True, print information about the update process. Default is False.

        Returns
        -------
        None
        """
        if not len(self):
            return
        if values_function is None:
            values_function = self.values_function

        if sparse_update_max_frames != -1:
            check_for_network_change = False

        # check for network change
        if check_for_network_change and len(self.__frame_values)>=3:
            sample_indices = np.array([0, 1, 2])
            if only_zeros:
                sample_indices = np.where(self.__frame_values==0.)[0][:3]
            if len(sample_indices):
                old_values = self.__frame_values[sample_indices].copy()
                if self.__frame_descriptors.shape[1]:
                    new_values = values_function(
                        self.__frame_descriptors[sample_indices])
                else:
                    new_values = values_function(
                        self.frames(sample_indices))
                if np.all(np.abs(old_values - new_values) < 1e-4):
                    only_zeros = True  # network unchanged
                    if verbose:
                        print("Skipping re-evaluation of existing values since network is unchanged.")

        if only_reactive or only_zeros or key is not None or sparse_update_max_frames != -1:
            mask = np.ones(len(self.__frame_values), dtype=bool)
            if key is not None:
                try:
                    keepers = np.concatenate(self.frame_indices(key))
                except:
                    return
                mask = np.zeros(len(self.__frame_values), dtype=bool)
                mask[keepers] = True
            if only_reactive:
                self.__frame_values[self.__frame_states == 'A'] = -np.inf
                self.__frame_values[self.__frame_states == 'B'] = +np.inf
                mask *= self.__frame_states == 'R'
            if only_zeros:
                print(self.__frame_values)
                mask *= self.__frame_values == 0.
            if sparse_update_max_frames != -1:
                # check number of frames to update
                n_frames_to_update = np.sum(mask)
                if n_frames_to_update > sparse_update_max_frames:
                    if verbose:
                        print(f"Sparse update: {n_frames_to_update} frames to update exceeds maximum of {sparse_update_max_frames}. Updating a random subset.")
                    # select random subset
                    indices_to_update = np.where(mask)[0]
                    selected_indices = np.random.choice(
                        indices_to_update,
                        size=sparse_update_max_frames,
                        replace=False)
                    new_mask = np.zeros(len(self.__frame_values), dtype=bool)
                    new_mask[selected_indices] = True
                    mask = new_mask
            if not np.sum(mask):
                return
            if self.__frame_descriptors.shape[1]:
                self.__frame_values[mask] = values_function(
                    self.__frame_descriptors[mask])
            else:
                self.__frame_values[mask] = values_function(
                    self.frames(mask))
        elif self.__frame_descriptors.shape[1]:
            self.__frame_values = values_function(
                self.__frame_descriptors)
        else:
            self.__frame_values = values_function(
                self.__frame_descriptors)

        if sparse_update_max_frames != -1:
            # Set zero values to those frames that have not been sparsely updated
            self.__frame_values[~mask] = 0.

    """
    Paths manipulation.
    """
    
    def prune_trajectory_files(self):
        """
        Remove unused names + merge the same names.
        """
        old_trajectory_files = self.__trajectory_files
        old_frame_trajectory_indices = self.__frame_trajectory_indices
        old_trajectory_indices = np.unique(old_frame_trajectory_indices)
        new_trajectory_files = list(np.unique(np.array(
            old_trajectory_files)[old_trajectory_indices]))
        if len(new_trajectory_files) == len(old_trajectory_files):
            return  # nothing to be done
        new_frame_trajectory_indices = np.zeros(
            len(old_frame_trajectory_indices), dtype=int)
        for i in old_trajectory_indices:
            trajectory_file = old_trajectory_files[i]
            new_trajectory_index = new_trajectory_files.index(trajectory_file)
            mask = old_frame_trajectory_indices == i
            new_frame_trajectory_indices[mask] = new_trajectory_index
        self.__trajectory_files = new_trajectory_files
        self.__frame_trajectory_indices = new_frame_trajectory_indices

    def merge(self, pathensemble):
        if not self.nframes:
            return pathensemble
        if not pathensemble.nframes:
            return self.copy()
        if not len(pathensemble):
            return self.copy()
        assert self.directory == pathensemble.directory
        assert self.topology == pathensemble.topology
        
        result = self.copy()
        
        # trajectory attributes
        n = len(result._PathEnsemble__trajectory_files)
        result._PathEnsemble__trajectory_files = list(
            result._PathEnsemble__trajectory_files) + list(
            pathensemble._PathEnsemble__trajectory_files)
        
        # frame attributes
        result._PathEnsemble__frame_trajectory_indices = np.append(
            result._PathEnsemble__frame_trajectory_indices,
            pathensemble._PathEnsemble__frame_trajectory_indices + n)
        result._PathEnsemble__frame_trajectory_positions = np.append(
            result._PathEnsemble__frame_trajectory_positions,
            pathensemble._PathEnsemble__frame_trajectory_positions)
        result._PathEnsemble__frame_times = np.append(
            result._PathEnsemble__frame_times,
            pathensemble._PathEnsemble__frame_times)
        result._PathEnsemble__frame_simulation_times = np.append(
            result._PathEnsemble__frame_simulation_times,
            pathensemble._PathEnsemble__frame_simulation_times)
        result._PathEnsemble__frame_states = np.append(
            result._PathEnsemble__frame_states,
            pathensemble._PathEnsemble__frame_states)
        result._PathEnsemble__frame_descriptors = np.append(
            result._PathEnsemble__frame_descriptors,
            pathensemble._PathEnsemble__frame_descriptors, axis=0)
        result._PathEnsemble__frame_values = np.append(
            result._PathEnsemble__frame_values,
            pathensemble._PathEnsemble__frame_values)

        # navigation attributes
        result._PathEnsemble__frame_indices = np.append(
            result._PathEnsemble__frame_indices,
            pathensemble._PathEnsemble__frame_indices + self.nframes)
        
        # path attributes
        result._PathEnsemble__lengths = np.append(
            result._PathEnsemble__lengths,
            pathensemble._PathEnsemble__lengths)
        result._PathEnsemble__weights = np.append(
            result._PathEnsemble__weights,
            pathensemble._PathEnsemble__weights)
        result._PathEnsemble__shooting_indices = np.append(
            result._PathEnsemble__shooting_indices,
            pathensemble._PathEnsemble__shooting_indices)
        result._PathEnsemble__selection_biases = np.append(
            result._PathEnsemble__selection_biases,
            pathensemble._PathEnsemble__selection_biases)
        result._PathEnsemble__are_accepted = np.append(
            result._PathEnsemble__are_accepted,
            pathensemble._PathEnsemble__are_accepted)

        # hopefully this is enough
        return result

    
    def append(self, trajectory_file,
               start=0, stop=None, step=1,
               simulation_times=None, weight=1.,
               states=None, descriptors=None,
               verbose=False):
        
        # load trajectory
        try:
            path = f'{self.directory}/{trajectory_file}'
            path = path.split('/')
            trajectory_file = path[-1]
            directory = '/'.join(path[:-1])
            if trajectory_file not in os.listdir(directory):
                raise
            universe =  mda.Universe(
                f'{self.directory}/{self.topology}',
                f'{directory}/{trajectory_file}')
            trajectory = universe.trajectory
        except Exception as exception:
            return 0, np.zeros(0)
        
        # process stop
        if stop is None:
            stop = len(trajectory)
        else:
            stop = min(len(trajectory), stop)
        new_frame_trajectory_positions = np.arange(start, stop, step)
        
        # find new trajectory files and index, remove overlapping frames
        trajectory_file = os.path.relpath(
            f'{directory}/{trajectory_file}',
            self.directory)
        if trajectory_file in self.__trajectory_files:
            new_trajectory_files = []
            trajectory_index = self.__trajectory_files.index(trajectory_file)
            mask = self.__frame_trajectory_indices == trajectory_index
            new_frame_trajectory_positions = np.setdiff1d(
                new_frame_trajectory_positions,
                self.__frame_trajectory_positions[mask])
        else:
            new_trajectory_files = [trajectory_file]
            trajectory_index = len(self.__trajectory_files)
        trajectory = trajectory[new_frame_trajectory_positions]
        
        # run through time to see how much of the trajectory is truly available
        new_frame_times = []
        try:
            nframes = 0
            for frame in trajectory:
                new_frame_times.append(frame.time)
                nframes += 1
        except:
            pass
        new_frame_times = np.array(new_frame_times)
        
        if not nframes:  # nothing added
            return nframes, np.zeros(0)
        trajectory = trajectory[:nframes]
        
        # populate all attributes
        if self.states_function is not None and states is None:
            if has_param(self.states_function, 'verbose'):
                new_frame_states = self.states_function(trajectory,
                    verbose=verbose)
            else:
                new_frame_states = self.states_function(trajectory)
        elif states is not None:
            new_frame_states = np.array(states)
        else:
            new_frame_states = np.repeat('R', len(trajectory))
        if self.descriptors_function is not None and descriptors is None:
            if has_param(self.descriptors_function, 'verbose'):
                new_frame_descriptors = self.descriptors_function(
                    trajectory, verbose=verbose)
            else:
                new_frame_descriptors = self.descriptors_function(trajectory)
        elif descriptors is not None:
            new_frame_descriptors = np.array(descriptors)
        else:
            new_frame_descriptors = np.zeros((nframes, 0))
        new_frame_trajectory_positions = new_frame_trajectory_positions[
            :nframes]  # TODO necessary only if need to recover trajectory
        new_frame_trajectory_indices = np.repeat(trajectory_index, nframes)
        if simulation_times is None:
            new_frame_simulation_times = time.time()
        if not hasattr(simulation_times, '__len__'):
            new_frame_simulation_times = np.repeat(
                new_frame_simulation_times, nframes)
        if new_frame_descriptors.shape[1]:
            new_frame_values = self.values_function(new_frame_descriptors)
        else:
            new_frame_values = self.values_function(
                trajectory[new_frame_trajectory_positions])
        new_frame_indices = np.arange(nframes) + self.nframes
        
        # append to old
        trajectory_files = list(self.__trajectory_files) + list(
            new_trajectory_files)
        if self.nframes:
            frame_trajectory_indices = np.append(
                self.__frame_trajectory_indices, new_frame_trajectory_indices)
            frame_trajectory_positions = np.append(
            self.__frame_trajectory_positions, new_frame_trajectory_positions)
            frame_times = np.append(self.__frame_times, new_frame_times)
            frame_simulation_times = np.append(
                self.__frame_simulation_times, new_frame_simulation_times)
            frame_states = np.append(self.__frame_states, new_frame_states)
            frame_descriptors = np.append(
                self.__frame_descriptors, new_frame_descriptors, axis=0)
            frame_values = np.append(self.__frame_values, new_frame_values)
            frame_indices = np.append(self.__frame_indices, new_frame_indices)
            self.__lengths[-1] += nframes
            lengths = self.__lengths
            weights = self.__weights
            shooting_indices = self.__shooting_indices
            selection_biases = self.__selection_biases
            are_accepted = self.__are_accepted
        else:
            frame_trajectory_indices = new_frame_trajectory_indices
            frame_trajectory_positions = new_frame_trajectory_positions
            frame_times = new_frame_times
            frame_simulation_times = new_frame_simulation_times
            frame_states = new_frame_states
            frame_descriptors = new_frame_descriptors
            frame_values = new_frame_values
            frame_indices = new_frame_indices
            lengths = np.array([nframes])  # a path if none before
            weights = np.array([weight])
            shooting_indices = np.array([np.argmin(new_frame_times)])
            selection_biases = np.ones(1)
            are_accepted = np.ones(1, dtype=bool)
        
        self._update(trajectory_files=trajectory_files ,
                     frame_trajectory_indices=frame_trajectory_indices,
                     frame_trajectory_positions=frame_trajectory_positions,
                     frame_times=frame_times,
                     frame_simulation_times=frame_simulation_times,
                     frame_states=frame_states,
                     frame_descriptors=frame_descriptors,
                     frame_values=frame_values,       
                     frame_indices=frame_indices,
                     lengths=lengths,
                     weights=weights,
                     shooting_indices=shooting_indices,
                     selection_biases=selection_biases,
                     are_accepted=are_accepted)

        universe.trajectory.close()  # Clean up explicitly
        return nframes, new_frame_times
    
    def add_path(self,
                 *trajectory_files,
                 start=0,
                 stop=None,
                 step=1,
                 simulation_times=None,
                 weight=1.,
                 shooting_index=None,
                 selection_bias=1.,
                 is_accepted=True,
                 states=None,  # already cached
                 descriptors=None,
                 verbose=False):
        npaths = len(self)
        
        # add trajectory (trajectories)
        nframes = 0
        times = np.zeros(0)
        for trajectory_file in trajectory_files:
            trajectory_nframes, trajectory_times = self.append(
                trajectory_file, start, stop, step, simulation_times,
                states=states, descriptors=descriptors, verbose=verbose)
            nframes += trajectory_nframes
            times = np.append(times, trajectory_times)
        
        if not nframes:
            os.system('echo "No frames for path, canceling operation!"')
            return 0, np.zeros(0)
        
        if shooting_index is None:
            if nframes:
                shooting_index = np.argmin(times)
            else:
                shooting_index = 0
        
        if npaths:
            self.__lengths[-1] -= nframes
            self.__lengths = np.append(self.__lengths, [nframes])
            self.__weights = np.append(self.__weights, [weight])
            self.__shooting_indices = np.append(self.__shooting_indices,
                                                [shooting_index])
            self.__selection_biases = np.append(self.__selection_biases,
                                                [selection_bias])
            self.__are_accepted = np.append(self.__are_accepted, [is_accepted])
        else:
            self.__weights[-1] = weight
            self.__shooting_indices[-1] = shooting_index
            self.__selection_biases[-1] = selection_bias
            self.__are_accepted[-1] = is_accepted
        return nframes, times
    
    def update_accepted_paths(self, max_excursion_length=np.inf):
        """
        Also removes spurious transitions.
        """
        self.__are_accepted[:] = True
        self.__are_accepted[self.are_excursions *
                           (self.lengths > max_excursion_length)] = False
        initial_states = self.initial_states
        final_states = self.final_states
        internal_states = self.internal_states
        self.__are_accepted[(initial_states == internal_states) *
                            (initial_states == final_states)] = False
        # self.__are_accepted[(initial_states != 'R') *
        #                     (internal_states != 'R') *
        #                     (final_states != 'R')] = False
        return self.__are_accepted[:]
    
    def split(self, max_excursion_length=np.inf):
        if not self.nframes:
            return self
        
        new_indices = []
        new_lengths = []
        new_weights = []
        new_shooting_indices = []
        new_selection_biases = []
        new_are_accepted = []
        
        # original
        for (initial, states, weight, shooting_index,
             selection_bias, are_accepted) in zip(
            self.boundaries, self.states(), self.weights,
            self.shooting_indices, self.selection_biases, self.are_accepted):
            
            # detect boundary crossings
            split_at = np.array([], dtype=int)
            if len(states) > 2:
                split_at = np.where(np.diff(
                    np.vectorize(ord)(states[1:-1])))[0] + 1
            split_at = np.concatenate([[0], split_at, [len(states) - 2]])
            
            # register infos
            for begin, end in zip(split_at, split_at[1:] + 2):
                length = end - begin
                if not length:
                    continue
                new_indices.append(self.__frame_indices[
                    begin + initial:end + initial])
                new_lengths.append(len(new_indices[-1]))
                new_weights.append(weight)
                new_selection_biases.append(selection_bias)
                if end <= shooting_index:
                    new_shooting_indices.append(end - begin - 1)
                elif begin > shooting_index:
                    new_shooting_indices.append(0)
                else:
                    new_shooting_indices.append(shooting_index - begin)
                new_are_accepted.append(are_accepted)
        
        # implement
        self._update(frame_indices=np.concatenate(new_indices),
                     lengths=np.array(new_lengths),
                     weights=np.array(new_weights),
                     shooting_indices=np.array(new_shooting_indices),
                     selection_biases=np.array(new_selection_biases),
                     are_accepted=np.array(new_are_accepted))
        
        self.update_accepted_paths(max_excursion_length)
        
        return self
    
    def unsplit(self):
        if not self.nframes:
            return self
        self.__lengths = np.array([self.nframes])
        self.__frame_indices = np.arange(self.nframes)
        self.__weights = np.array([1.])
        self.__shooting_indices = np.array([np.argmin(self.__frame_times)])
        self.__selection_biases = np.array([1.])
        self.__are_accepted = np.array([True])
        return self

    def invert(self, exclude_shooting_frame=True):
        if exclude_shooting_frame:
            self.__frame_indices = self.__frame_indices[1:]
            self.__lengths[0] -= 1
        self.__frame_indices = self.__frame_indices[::-1]
        self.__lengths = self.__lengths[::-1]


    def remove_overlapping_frames(self):
        """
        Latest produced overlapping frames have priority.
        """
        self.unsplit()
        frame_times = self.frame_times
        _, last_indices = np.unique(frame_times[::-1], return_index=True)        
        keepers = self.nframes - 1 - last_indices
        nframes = len(keepers)
        self._update(
         frame_trajectory_indices=self.__frame_trajectory_indices[keepers],
         frame_trajectory_positions=self.__frame_trajectory_positions[keepers],
         frame_times=self.__frame_times[keepers],
         frame_simulation_times=self.__frame_simulation_times[keepers],
         frame_states=self.__frame_states[keepers],
         frame_descriptors=self.__frame_descriptors[keepers],
         frame_values=self.__frame_values[keepers],
         frame_indices=np.arange(nframes),
         lengths=np.array([nframes]))
        self.prune_trajectory_files()
        return self
    
    """
    Analysis.
    """
    
    def project(self, bins=[-np.inf, +np.inf],
                key=None, f=None, frames=False,
                weights=None, vmin=None, vmax=None,
                backward=True, forward=True):
        """
        Parameters
        ----------
        bins: array-like of floats
              borders of the bins
              the dimension is guessed from here
        f: callable function
           run through all the paths in the path ensemble
           if None: take self.values
        frames: if True: f takes mdanalysis frames as input, otw descr.
        weights: np.array of weights of the path ensemble paths;
                 if None: use standard weights
        vmin, vmax: project only points with RC values between vmin and vmax

        Returns
        -------
        distribution: array-like of size (len(bins) - 1)
                      counts the population in each bin (sum of the weights)
        """
        
        # process bins
        bins = np.array(bins)
        if len(bins.shape) == 1:
            bins = bins.reshape(1, -1)
        
        # process weights
        if weights is None:
            weights = self.__weights[key].ravel()

        # override frames option
        if not self.__frame_descriptors.shape[1]:
            frames = True

        # extract frames and weights
        frame_weights = np.zeros(self.nframes)
        path_indices = np.arange(len(self))[key].ravel()
        for frame_indices, weight, is_accepted in zip(
            self.frame_indices(key, internal=True,
                               backward=backward, forward=forward),
            weights, self.__are_accepted[key].ravel()):
            frame_weights[frame_indices] += weight * is_accepted
        if vmin is not None:
            frame_weights[self.__frame_values < vmin] = 0.
        if vmax is not None:
            frame_weights[self.__frame_values >= vmax] = 0.
        frame_indices = np.where(frame_weights > 0)[0]
        # disregard further frames with zero values
        frame_indices = frame_indices[self.__frame_values[frame_indices] != 0.]
        weights = frame_weights[frame_indices]  # now weights of frames
        
        # get values
        if f is None:
            values = self.__frame_values[frame_indices]
        elif not frames:
            values = f(self.__frame_descriptors[frame_indices])
        else:
            values = f(self.frames(frame_indices))
        
        # process
        if len(bins.shape) > len(values.shape):
            values = np.tile(values, (bins.shape[0], 1)).T
        
        # project
        return np.histogramdd(
            values, bins, density=False, weights=weights)[0].T
    
    """
    Special methods.
    """
    
    def __repr__(self):
        return (f'PathEnsemble with {len(self.__lengths)} '
                f'path{"s" if len(self) != 1 else ""} from '
                f'{len(self.__trajectory_files)} '
                f'trajectory file{"s" * (len(self.__trajectory_files) != 1)}, '
                f'{self.nframes} individual frame{"s" * (self.nframes != 1)}')
    
    def __len__(self):
        return len(self.__lengths)
    
    def __add__(self, entity):
        if type(entity) is PathEnsemblesCollection:
            return PathEnsemblesCollection(self, *entity.pathensembles)
        return PathEnsemblesCollection(self, entity)
    
    def __iter__(self):
        return PathEnsemblesIterator(self)
    
    def __getitem__(self, key):
        
        # initialize result
        result = PathEnsemble(self.directory, self.topology,
            self.states_function, self.descriptors_function,
            self.values_function)
        result._PathEnsemble__trajectory_files = self.__trajectory_files.copy()
        
        # all the remaining frames
        frame_indices = self.frame_indices(key)
        if not len(frame_indices):
            result._PathEnsemble__trajectory_files = []
            return result
        frame_indices = np.concatenate(frame_indices)
        keepers = np.unique(frame_indices)
        
        # convert frame_indices from old to new representation
        mapping = np.zeros(np.max(keepers) + 1, dtype=int)
        for i, j in enumerate(keepers):
            mapping[j] = i
        
        # update
        result._update(
         frame_trajectory_indices=self.__frame_trajectory_indices[keepers],
         frame_trajectory_positions=self.__frame_trajectory_positions[keepers],
         frame_times=self.__frame_times[keepers],
         frame_simulation_times=self.__frame_simulation_times[keepers],
         frame_states=self.__frame_states[keepers],
         frame_descriptors=self.__frame_descriptors[keepers],
         frame_values=self.__frame_values[keepers],
         frame_indices=mapping[frame_indices],
         lengths=self.__lengths[key].ravel(),
         weights=self.__weights[key].ravel(),
         shooting_indices=self.__shooting_indices[key].ravel(),
         selection_biases=self.__selection_biases[key].ravel(),
         are_accepted=self.__are_accepted[key].ravel())
        result.prune_trajectory_files()
        return result
    
    def copy(self):
        return self[:]
    
    def select(self,
               key=None,
               trajectory_filenames=[],
               initial_states=[],
               internal_states=[],
               final_states=[]):
        """
        As getitem, but you can also select by other properties.
        """
        
        # process
        trajectory_filenames = set(trajectory_filenames)
        initial_states = set(initial_states)
        internal_states = set(internal_states)
        final_states = set(final_states)
        
        # select
        path_indices = list(np.arange(len(self))[key].ravel())
        
        # restrict
        for (i,
             path_trajectory_filenames,
             path_initial_state,
             path_internal_state,
             path_final_state) in zip(
            range(len(path_indices) - 1, -1, -1),
            self.trajectory_filenames(path_indices[::-1]),
            self.initial_states[path_indices[::-1]],
            self.internal_states[path_indices[::-1]],
            self.final_states[path_indices[::-1]]):
            if len(set(path_trajectory_filenames).difference(
                trajectory_filenames)):  # there are not requested paths
                path_indices.pop(i)
            elif (len(initial_states) and not
                  initial_states.intersection(path_initial_state)):
                path_indices.pop(i)
            elif (len(internal_states) and not
                  internal_states.intersection(path_internal_state)):
                path_indices.pop(i)
            elif (len(final_states) and not
                  final_states.intersection(path_final_state)):
                path_indices.pop(i)
        
        return self[path_indices]
    
    def crop(self, tmin=0., tmax=+np.inf, max_excursion_length=+np.inf,
             frame_indices=None):
        """
        Attention! Works on equilibrium simulations.
        Also re-splits.
        t0 max simulation time
        """
        frame_indices = np.arange(self.nframes)[frame_indices].ravel()
        if tmin > 0:
            frame_indices = frame_indices[
                self.__frame_simulation_times[frame_indices] >= tmin]
        if tmax < np.inf:
            frame_indices = frame_indices[
                self.__frame_simulation_times[frame_indices] <= tmax]
        result = self[:0]
        result._update(
            trajectory_files=self.trajectory_files.copy(),
            frame_trajectory_indices=self.frame_trajectory_indices[
                frame_indices],
            frame_trajectory_positions=self.frame_trajectory_positions[
                frame_indices],
            frame_times=self.frame_times[frame_indices],
            frame_simulation_times=self.frame_simulation_times[frame_indices],
            frame_states=self.frame_states[frame_indices],
            frame_descriptors=self.frame_descriptors[frame_indices],
            frame_values=self.frame_values[frame_indices])
        result.prune_trajectory_files()
        result.unsplit()
        result.split()
        result.update_accepted_paths(max_excursion_length)
        
        return result
    
    def __getstate__(self):  # for saving/loading
        state = {}
        for slot in self.__slots__:
            if slot[:2] == '__':
                slot = f'_PathEnsemble{slot}'
            if not self.save_descriptors and 'descriptors' in slot:
                continue
            state[slot] = getattr(self, slot)
        return state
    
    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)
    
    def save(self, filename, backup=True, descriptors=True, directory=None):
        """
        Safe to save where the trajectories are (period).
        """
        pathensemble_directory = self.directory
        if directory is not None:
            self.directory = directory
        filename = filename.split('.')
        if len(filename) > 1:
            file = '.'.join(filename[:-1])
            extension = f'.{filename[-1]}'
        else:
            file = filename[0]
            extension = ''
        filename = f'{file}{extension}'
        if backup:
            backup = f'{file}_backup{extension}'
            try:
                os.rename(filename, backup) 
            except:
                pass
        self.save_descriptors = descriptors
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        self.directory = pathensemble_directory
    
    def load(self, filename):
        with open(filename, 'rb') as file:
            result = pickle.load(file)
        if not hasattr(result, 'frame_descriptors'):
            frame_descriptors = np.zeros(
                (len(result.frame_trajectory_indices), 0))
        else:
            frame_descriptors = result.frame_descriptors
        directory = '/'.join(filename.split('/')[:-1])
        if not directory or directory == '.':
            directory = ''
        else:
            directory = f'{directory}/'
        if result.directory != '.':
            directory = f'{directory}/{result.directory}'
        if len(directory) == 0:
            directory = '.'
        if directory[-1] == '/':
            directory = directory[:-1]
        self._update(
            trajectory_files=result.trajectory_files,
            frame_trajectory_indices=result.frame_trajectory_indices,
            frame_trajectory_positions=result.frame_trajectory_positions,
            frame_times=result.frame_times,
            frame_simulation_times=result.frame_simulation_times,
            frame_states=result.frame_states,
            frame_descriptors=frame_descriptors,
            frame_values=result.frame_values,
            frame_indices=result._PathEnsemble__frame_indices,
            lengths=result.lengths,
            weights=result.weights,
            shooting_indices=result.shooting_indices,
            selection_biases=result.selection_biases if hasattr(
                result, 'selection_biases') else np.ones(len(result.weights)),
            are_accepted=result.are_accepted,
            directory=directory,
            topology=result.topology)
        self._check()
        if not hasattr(result, 'frame_descriptors'):
            self.update_descriptors()
        return self
    
    def _check(self):
        assert (len(self.frame_trajectory_indices) ==
                len(self.frame_trajectory_positions) ==
                len(self.frame_times) ==
                len(self.frame_simulation_times) ==
                len(self.frame_states) ==
                len(self.frame_descriptors) ==
                len(self.frame_values))
        assert (len(self.weights) ==
                len(self.shooting_indices) ==
                len(self.selection_biases) ==
                len(self.are_accepted))
        assert (len(self.__frame_indices) == np.sum(self.__lengths))


class PathEnsemblesCollection(AbstractPathEnsemble):
    def __init__(self, *pathensembles):
        self.pathensembles = list(pathensembles)
    
    def _get_attribute(self, attribute, directories=False):
        attributes = [getattr(pathensemble, attribute)
            for pathensemble in self.pathensembles]
        if directories:
            attributes = []
            for pathensemble in self.pathensembles:
                directory = pathensemble.directory
                if directory == '.':
                    directory = ''
                else:
                    directory = f'{directory}/'
                att = getattr(pathensemble, attribute)
                if len(attribute):
                    attributes.append([f'{directory}{attribute}'
                        for attribute in att])
        else:
            attributes = [attribute for attribute in attributes
                               if len(attribute)]
        if len(attributes):
            return np.concatenate(attributes, axis=0)
        return np.zeros(0)
    
    """
    Properties (inherited from PathEnsemble attributes).
    """
    
    @property
    def trajectory_files(self):
        result = []
        for pathensemble in self.pathensembles:
            if pathensemble.directory not in ['.', '']:
                directory = f'{pathensemble.directory}/'
            else:
                directory = ''
            result += [f'{directory}{filename}'
                       for filename in pathensemble.trajectory_files]
        return result
    
    @property
    def trajectory_directories(self):
        return np.repeat('.', len(self.trajectory_files))
    
    @property
    def trajectory_topologies(self):
        if not len(self.pathensembles):
            return np.zeros(0, dtype='<U1')
        return np.concatenate([np.repeat(
            f'{pathensemble.directory}/{pathensemble.topology}', 
            len(pathensemble._PathEnsemble__trajectory_files))
            for pathensemble in self.pathensembles])
    
    @property
    def frame_trajectory_indices(self):
        if not len(self.pathensembles):
            return np.zeros(0, dtype=int)
        n_trajectory_files = [len(pathensemble.trajectory_files)
                              for pathensemble in self.pathensembles]
        return np.concatenate([pathensemble.frame_trajectory_indices + begin
                               for pathensemble, begin in zip(
                self.pathensembles, padcumsum(n_trajectory_files))])
    
    @property
    def frame_trajectory_positions(self):
        return self._get_attribute('frame_trajectory_positions')
    
    @property
    def frame_times(self):
        return self._get_attribute('frame_times')
    
    @property
    def frame_simulation_times(self):
        return self._get_attribute('frame_simulation_times')
    
    @property
    def frame_states(self):
        return self._get_attribute('frame_states')
    
    @property
    def frame_descriptors(self):
        return self._get_attribute('frame_descriptors')
    
    @property
    def frame_values(self):
        return self._get_attribute('frame_values')
    
    @property
    def lengths(self):
        return self._get_attribute('lengths')
    
    @property
    def weights(self):
        return self._get_attribute('weights')
    
    @property
    def shooting_indices(self):
        return self._get_attribute('shooting_indices')

    @property
    def selection_biases(self):
        return self._get_attribute('selection_biases')
    
    @property
    def are_accepted(self):
        return self._get_attribute('are_accepted')

    @property
    def nframes(self):
        return int(np.sum(self.pathensemble_nframes))
    
    @property
    def boundaries(self):
        return padcumsum(self.lengths)  
    
    """
    Original properties (different from PathEnsemble).
    """

    @property
    def npaths(self):
        return np.array([len(pathensemble)
                         for pathensemble in self.pathensembles], dtype=int)
    
    @property
    def pathensemble_nframes(self):
        return np.array([pathensemble.nframes
                         for pathensemble in self.pathensembles], dtype=int)
    
    @property
    def npathensembles(self):
        return len(self.pathensembles)
    
    @property
    def path_indices(self):
        """
        For each path in PathEnsemblesCollection, the indices IN
        its original PathEnsemble.
        """
        if not len(self.pathensembles):
            return np.zeros(0, dtype=int)
        return np.concatenate([np.arange(npaths) for npaths in self.npaths])
    
    @property
    def pathensemble_indices(self):
        """
        For each path in PathEnsemblesCollection, the indices OF
        its original PathEnsemble.
        """
        frame_pathensemble_indices = np.zeros(np.sum(self.npaths), dtype=int)
        for begin in np.cumsum(self.npaths[:-1]):
            frame_pathensemble_indices[begin:] += 1
        return frame_pathensemble_indices
    
    """
    Additional properties.
    """
    
    @property
    def directories(self):
        return [f'{pathensemble.directory}'
                for pathensemble in self.pathensembles]
    
    @property
    def topologies(self):
        return [f'{pathensemble.directory}/{pathensemble.topology}'
                for pathensemble in self.pathensembles]
    
    @property
    def states_functions(self):
        return [pathensemble.states_function
                for pathensemble in self.pathensembles]
    
    @property
    def descriptors_functions(self):
        return [pathensemble.descriptors_function
                for pathensemble in self.pathensembles]
    
    @property
    def values_functions(self):
        return [pathensemble.values_function
                for pathensemble in self.pathensembles]
    
    @property
    def states_function(self):
        if len(self.pathensembles):
            return self.pathensembles[0].states_function
        return None
    
    @property
    def descriptors_function(self):
        if len(self.pathensembles):
            return self.pathensembles[0].descriptors_function
        return None
    
    @property
    def values_function(self):
        if len(self.pathensembles):
            return self.pathensembles[0].values_function
        return None
    
    @states_function.setter
    def states_function(self, states_function):
        for pathensemble in self.pathensembles:
            pathensemble.states_function = states_function
    
    @descriptors_function.setter
    def descriptors_function(self, descriptors_function):
        for pathensemble in self.pathensembles:
            pathensemble.descriptors_function = descriptors_function
    
    @values_function.setter
    def values_function(self, values_function):
        for pathensemble in self.pathensembles:
            pathensemble.values_function = values_function

    """
    Setters (only path properties). Simplest way possible.
    """

    @weights.setter
    def weights(self, weights):
        if not hasattr(weights, '__len__'):
            weights = np.repeat(float(weights), len(self))
        npaths = padcumsum(self.npaths)
        for pathensemble, begin, end in zip(
            self.pathensembles, npaths, npaths[1:]):
            pathensemble.weights[:] = weights[begin:end]
    
    @shooting_indices.setter
    def shooting_indices(self, shooting_indices):
        if not hasattr(shooting_indices, '__len__'):
            shooting_indices = np.repeat(int(shooting_indices), len(self))
        npaths = padcumsum(self.npaths)
        for pathensemble, begin, end in zip(
            self.pathensembles, npaths, npaths[1:]):
            pathensemble.shooting_indices[:] = shooting_indices[begin:end]

    @selection_biases.setter
    def selection_biases(self, selection_biases):
        if not hasattr(weights, '__len__'):
            selection_biases = np.repeat(float(selection_biases), len(self))
        npaths = padcumsum(self.npaths)
        for pathensemble, begin, end in zip(
            self.pathensembles, npaths, npaths[1:]):
            pathensemble.selection_biases[:] = selection_biases[begin:end]
    
    @are_accepted.setter
    def are_accepted(self, are_accepted):
        if not hasattr(are_accepted, '__len__'):
            are_accepted = np.repeat(bool(are_accepted), len(self))
        npaths = padcumsum(self.npaths)
        for pathensemble, begin, end in zip(
            self.pathensembles, npaths, npaths[1:]):
            pathensemble.are_accepted[:] = are_accepted[begin:end]
    
    """
    Extraction.
    """
    
    def _extract(self, method, key=None,
                 internal=False, backward=True, forward=True,
                 offset=False):
        
        # process paths_index
        key = np.arange(len(self))[key].ravel()
        pathensemble_indices = self.pathensemble_indices[key]
        path_indices = self.path_indices[key]
        result = np.zeros(len(key), dtype=object)
        I = np.unique(pathensemble_indices)
        for i, _offset in zip(I,
            padcumsum(self.pathensemble_nframes)[I]):
            pathensemble = self.pathensembles[i]
            mask = np.where(pathensemble_indices == i)[0]
            result[mask] = getattr(pathensemble, method)(
                path_indices[mask], internal, backward, forward)
            if offset:
                result[mask] += _offset
        return list(result)
    
    def frame_indices(self, key=None,
                      internal=False, backward=True, forward=True):
        return self._extract('frame_indices',
                             key, internal, backward, forward, True)
    
    def trajectory_filenames(self, key=None,
                             internal=False, backward=True, forward=True):
        return self._extract('trajectory_filenames',
                             key, internal, backward, forward
                             #, directories=[pathensemble.directory
                             #for pathensemble in self.pathensembles]
                            )
    
    def trajectory_positions(self, key=None,
                             internal=False, backward=True, forward=True):
        return self._extract('trajectory_positions',
                             key, internal, backward, forward)
    
    def times(self, key=None,
              internal=False, backward=True, forward=True):
        return self._extract('times',
                             key, internal, backward, forward)
    
    def simulation_times(self, key=None,
                         internal=False, backward=True, forward=True):
        return self._extract('simulation_times',
                             key, internal, backward, forward)
    
    def states(self, key=None,
               internal=False, backward=True, forward=True):
        return self._extract('states',
                             key, internal, backward, forward)
    
    def descriptors(self, key=None,
                    internal=False, backward=True, forward=True):
        return self._extract('descriptors',
                             key, internal, backward, forward)
    
    def values(self, key=None,
               internal=False, backward=True, forward=True):
        return self._extract('values',
                             key, internal, backward, forward)
    
    @property
    def initial_frame_indices(self):
        return self._get_attribute('initial_frame_indices')
    
    @property
    def shooting_frame_indices(self):
        return self._get_attribute('shooting_frame_indices')
    
    @property
    def final_frame_indices(self):
        return self._get_attribute('final_frame_indices')
    
    def _get_frames(self, attribute):
        if not len(self):
            return MDATrajectory([], [], [])
        frames = getattr(self.pathensembles[0], attribute)
        for pathensemble in self.pathensembles[1:]:
            frames += getattr(pathensemble, attribute)
        return frames
    
    @property
    def initial_frames(self):
        return self._get_frames('initial_frames')
    
    @property
    def shooting_frames(self):
        return self._get_frames('shooting_frames')
    
    @property
    def final_frames(self):
        return self._get_frames('final_frames')
    
    @property
    def initial_trajectory_filenames(self):
        return self._get_attribute('initial_trajectory_filenames', True)
    
    @property
    def shooting_trajectory_filenames(self):
        return self._get_attribute('shooting_trajectory_filenames', True)
    
    @property
    def final_trajectory_filenames(self):
        return self._get_attribute('final_trajectory_filenames', True)
    
    @property
    def initial_trajectory_indices(self):
        return self._get_attribute('initial_trajectory_indices')
    
    @property
    def shooting_trajectory_indices(self):
        return self._get_attribute('shooting_trajectory_indices')
    
    @property
    def final_trajectory_indices(self):
        return self._get_attribute('final_trajectory_indices')
    
    @property
    def initial_trajectory_positions(self):
        return self._get_attribute('initial_trajectory_positions')
    
    @property
    def shooting_trajectory_positions(self):
        return self._get_attribute('shooting_trajectory_positions')
    
    @property
    def final_trajectory_positions(self):
        return self._get_attribute('final_trajectory_positions')
    
    @property
    def initial_times(self):
        return self._get_attribute('initial_times')
    
    @property
    def shooting_times(self):
        return self._get_attribute('shooting_times')
    
    @property
    def final_times(self):
        return self._get_attribute('final_times')
    
    @property
    def initial_simulation_times(self):
        return self._get_attribute('initial_simulation_times')
    
    @property
    def shooting_simulation_times(self):
        return self._get_attribute('shooting_simulation_times')
    
    @property
    def final_simulation_times(self):
        return self._get_attribute('final_simulation_times')
    
    @property
    def completion_times(self):
        return self._get_attribute('completion_times')
    
    @property
    def initial_states(self):
        return self._get_attribute('initial_states')
    
    @property
    def shooting_states(self):
        return self._get_attribute('shooting_states')
    
    @property
    def final_states(self):
        return self._get_attribute('final_states')
    
    @property
    def initial_descriptors(self):
        return self._get_attribute('initial_descriptors')
    
    @property
    def shooting_descriptors(self):
        result = self._get_attribute('shooting_descriptors')
        if len(result.shape) == 1:
            result.reshape((-1, 1))
        return result
    
    @property
    def final_descriptors(self):
        return self._get_attribute('final_descriptors')
    
    @property
    def initial_values(self):
        return self._get_attribute('initial_values')
    
    @property
    def shooting_values(self):
        return self._get_attribute('shooting_values')
    
    @property
    def final_values(self):
        return self._get_attribute('final_values')
    
    """
    Path properties.
    """
    
    @property
    def internal_states(self):
        return self._get_attribute('internal_states')
    
    @property
    def internal_lengths(self):
        return self._get_attribute('internal_lengths')
    
    @property
    def are_shot(self):
        return self._get_attribute('are_shot')
    
    @property
    def are_equilibrium(self):
        return self._get_attribute('are_equilibrium')
    
    @property
    def are_excursions(self):
        return self._get_attribute('are_excursions')
    
    @property
    def are_internal(self):
        return self._get_attribute('are_internal')
    
    @property
    def shooting_results(self):
        return self._get_attribute('shooting_results')
    
    @property
    def are_transitions(self):
        return self._get_attribute('are_transitions')
    
    """
    Update.
    """
    
    def update_states(self, verbose=False):
        for i, pathensemble in enumerate(self.pathensembles):
            if verbose:
                print('Pathensemble', i + 1, 'out of', len(self.pathensembles))
            pathensemble.update_states(verbose=verbose)
    
    def update_descriptors(self, verbose=False):
        for i, pathensemble in enumerate(self.pathensembles):
            if verbose:
                print('Pathensemble', i + 1, 'out of', len(self.pathensembles))
            if pathensemble.descriptors_function is None:
                pathensemble.frame_descriptors[:] = np.zeros(
                    (pathensemble.nframes, 0))
            else:
                pathensemble.update_descriptors(verbose=verbose)

    def update_values(self,
        only_reactive=False, only_zeros=False,
        values_function=None, verbose=False, sparse_update_max_frames=-1):
        for i, pathensemble in enumerate(self.pathensembles):
            if verbose:
                print('Pathensemble', i + 1, 'out of', len(self.pathensembles))
            pathensemble.update_values(
                only_reactive=only_reactive,
                only_zeros=only_zeros,
                values_function=values_function,
                sparse_update_max_frames=int(sparse_update_max_frames / len(self.pathensembles)) if sparse_update_max_frames > 0 else -1,
                verbose=verbose
            )

    """
    Paths manipulation.
    """
    
    def _execute_method(self, method, *args):
        return [getattr(pathensemble, method)(*args)
                for pathensemble in self.pathensembles]
    
    def prune_trajectory_files(self):
        self._execute_method('prune_trajectory_files')
    
    def update_accepted_paths(self, max_excursion_length=np.inf):
        return self._execute_method(
            'update_accepted_paths', max_excursion_length)
    
    def split(self, max_excursion_length=np.inf):
        return self._execute_method('split', max_excursion_length)
    
    def unsplit(self):
        return self._execute_method('unsplit')

    def invert(self, exclude_shooting_frame=True):
        return self._execute_method('invert', exclude_shooting_frame)

    def remove_overlapping_frames(self):
        return self._execute_method('remove_overlapping_frames')
    
    """
    Analysis.
    """
    
    def project(self, bins=[-np.inf, +np.inf],
                key=None, f=None, frames=False,
                weights=None, vmin=None, vmax=None,
                backward=True, forward=True):

        # fixed parameters
        args = [bins, None, f, frames, weights, vmin, vmax, backward, forward]
        
        # select the right key
        key = np.arange(len(self))[key].ravel()
        pathensemble_indices = self.pathensemble_indices[key]
        path_indices = self.path_indices[key]
        
        # a key for each pathensemble in Collection
        results = []
        for i in np.unique(pathensemble_indices):
            args[1] = path_indices[pathensemble_indices == i]
            pathensemble = self.pathensembles[i]
            results.append(getattr(pathensemble, 'project')(*args))    
        return np.sum(results, axis=0)
    
    """
    Special methods.
    """
    
    def __repr__(self):
        return (f'Union of {np.sum(self.pathensemble_nframes > 0)} '
                f'non-empty PathEnsemble instances, {len(self)} '
                f'path{"s" if len(self) != 1 else ""} from '
                f'{len(self.trajectory_files)} '
                f'trajectory file{"s" * (len(self.trajectory_files) != 1)}, '
                f'{self.nframes} individual frame{"s" * (self.nframes != 1)}')
    
    def __len__(self):
        if len(self.pathensembles):
            return np.sum([len(pathensemble)
                           for pathensemble in self.pathensembles])
        return 0
    
    def __add__(self, entity):
        if type(entity) is PathEnsemblesCollection:
            return PathEnsemblesCollection(
                *self.pathensembles, *entity.pathensembles)
        return PathEnsemblesCollection(*self.pathensembles, entity)
    
    def __iter__(self):
        return PathEnsemblesIterator(*self.pathensembles)
    
    def __getitem__(self, key):
        
        # select
        key = np.arange(len(self))[key].ravel()
        pathensemble_indices = self.pathensemble_indices[key]
        path_indices = self.path_indices[key]
        
        # a key for each pathensemble in Collection
        result = PathEnsemblesCollection()
        for i in np.unique(pathensemble_indices):
            pathensemble = self.pathensembles[i]
            result.pathensembles.append(
                pathensemble[path_indices[pathensemble_indices == i]])
        return result
    
    def select(self,
               key=None,
               trajectory_filenames=[],
               initial_states=[],
               internal_states=[],
               final_states=[]):
        """
        As getitem, but you can also select by other properties.
        """
        # select
        key = np.arange(len(self))[key].ravel()
        pathensemble_indices = self.pathensemble_indices[key]
        path_indices = self.path_indices[key]
        
        # a key for each pathensemble in Collection
        result = PathEnsemblesCollection()
        for i in np.unique(pathensemble_indices):
            pathensemble = self.pathensembles[i]
            result.pathensembles.append(
                pathensemble[path_indices[pathensemble_indices == i]],
                trajectory_filenames, initial_stats,
                internal_states, final_states)
        return result
    
    def crop(self, tmin=0., tmax=+np.inf, max_excursion_length=+np.inf):
        return PathEnsemblesCollection(*
            self._execute_method('crop', tmin, tmax, max_excursion_length))
    
    def merge(self):
        """
        Return single pathensemble.
        """
        if not len(self.pathensembles):
            return PathEnsemble()
        nframes = [pathensemble.nframes for pathensemble in self.pathensembles]
        frame_indices = np.concatenate([
            pathensemble._PathEnsemble__frame_indices + begin
            for pathensemble, begin in zip(
            self.pathensembles, padcumsum(nframes))])
        result = self.pathensembles[0][:0]
        result._update(
            trajectory_files=self.trajectory_files,
            frame_trajectory_indices=self.frame_trajectory_indices,
            frame_trajectory_positions=self.frame_trajectory_positions,
            frame_times=self.frame_times,
            frame_simulation_times=self.frame_simulation_times,
            frame_states=self.frame_states,
            frame_descriptors=self.frame_descriptors,
            frame_values=self.frame_values,            
            frame_indices=frame_indices,
            lengths=self.lengths,
            weights=self.weights,
            shooting_indices=self.shooting_indices,
            selection_biases=self.selection_biases,
            are_accepted=self.are_accepted,
            topology=self.topologies[0],
            directory='.')
        result.prune_trajectory_files()
        return result

