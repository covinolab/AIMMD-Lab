"""
...
"""

# exteral
import os
import time
import numpy as np
from abc import ABC
from math import inf
from numbers import Integral

# aimmd imports
from ..path import Path
from .._config import print
from ..core.utils import now, remove, process_state
from ..pathensemble import PathEnsemble
from ..execute.threads import ThreadExecutor
from ..pathensemble.utils import assemble_pathensemble

# worker "free" run method
class WorkerFree(ABC):

    def free(self, target_state=0, k=0, total=1, wait=False):
        """
        Run free simulations.
        total: total number of simulations.
        state: if int, params.states[i], if str: state
        free R -> free excursions (usually from one single state)
        """
        return self.run('free', target_state, k, total, wait)
    
    def _free(self, target_state=0, k=0, total=1, wait=False):
        
        # get/process params
        k = int(k)
        total = int(total)
        directory = self._directory
        params = self.params
        ext = params.trajectory_extension
        len_ext = len(ext)
        batch_size = params.trajectory_update_batch_size
        pipeline = params.pipeline[:-1]  # except for values
        extra_frames = params.extra_free_frames
        # which state are we talking about?
        states = params.states
        t = process_state(target_state, states)
        r = states[1]
        tr = f'{t}{r}'  # allowed states: target & reactive
        folder = f'{directory}/free{t}'
        restart_with_transition = (
            params.restart_free_simulations_with_transitions == 'all' or
            t in params.restart_free_simulations_with_transitions)
        initial_paths = self.initial_paths
        
        # create folder if not existing
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f'+++ created {folder!r}')
        
        # must have network, bins, and descriptors
        if wait and params.nbins > 1:
            print(f'\nWaiting for neural network, bins, densities {now()}')
            while True:
                try:
                    params.update_network(
                        directory, timeout=0, raise_if_failure=True)
                    bins, densities = params.load_bins_and_densities(
                        directory, timeout=0, raise_if_failure=True)
                    break
                except:
                    if self.termination_signal:
                        return
        
        # initialize
        chains = []
        initial_frames = None
        num = k + 1  # first trajectory
        name = f'traj{num:06g}'
        deffnm = f'{folder}/{name}'
        trajectory = Path()
        old_nframes = 0
        
        # main cycle
        print(f"\nCurrent trajectory: {deffnm} {now()}")
        while not self.must_stop:
            
            # update old trajectories while it is possible
            (stop_frame, nframes, last_state, last_length) = \
                self._simulate(deffnm, trajectory, t, 'free', 0, extra_frames)
            simulation_completed = stop_frame is not None
            
            # check mid cycle
            if self.must_stop:
                return
            
            # initialize only when necessary
            if (nframes == old_nframes and
                not simulation_completed and
                not params.check_if_initialized(deffnm)):
                
                # need to find initial_frames
                if restart_with_transition or not initial_frames:

                    # take initial_frames from a list of transitions
                    if restart_with_transition:
                        chains = params.shot_chains(directory, r, old=chains)
                        transitions = assemble_pathensemble(chains).extract(
                            states, states[::-1])
                        if not transitions:
                            transitions = initial_paths
                        i = np.random.choice(len(transitions))
                        path = transitions[i]
                    else:

                        # take initial_frames from initial paths (in order)
                        path = initial_paths[k % len(initial_paths)]
                    
                    if t == r:
                        if np.random.random() > .5:
                            initial_frames = path[:+2]
                        else:
                            initial_frames = path[:-3:-1]
                    elif path.initial('states') == t:
                        initial_frames = path[1::-1]
                    else:
                        initial_frames = path[-2:]
                
                # wipe out garbage
                remove(f'{folder}/*{name}*')
                                
                # report
                fnames = initial_frames.filenames
                locs = initial_frames.locs
                if fnames[0] == fnames[-1]:
                    print(f'\nUsing {fnames[0]} {locs[0]} -> {locs[-1]} '
                          f'for initializing {deffnm}')
                else:
                    print(f'\nUsing {fnames[0]} {locs[0]} -> '
                          f'{fnames[-1]} {locs[-1]} '
                          f'for initializing {deffnm}')
                
                # last frame (initialize simulation)
                # already divide in parts if more than one frame
                params.initialize_simulation(initial_frames, deffnm)

            # update old_nframes
            old_nframes = nframes

            # simulation is completed! go forward
            if simulation_completed:
                self._total_steps += 1
                
                # take initial frames from last valid crossing
                if t == r:
                    stop_frame += last_length - 2
                initial_frames = trajectory[stop_frame:stop_frame + 2]
                if initial_frames.states[1] == t:
                    pass
                elif initial_frames.states[0] == t:
                    initial_frames = initial_frames[::-1]
                else:
                    initial_frames = None  # this should never happen
                    # but it allows to recover from "corrupted" data
                
                # go to next trajectory
                num += total
                name = f'traj{num:06g}'
                deffnm = f'{folder}/{name}'
                trajectory = Path()
                old_nframes = 0
                print(f"\nCurrent trajectory: {deffnm} {now()}")
