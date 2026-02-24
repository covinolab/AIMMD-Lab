"""
...
"""

# external
import os
import time
import numpy as np
from abc import ABC
from math import inf
from numbers import Integral

# aimmd imports
from .utils import register_path
from .utils import select_shooting_point
from .utils import update_selection_pool
from .utils import accept_or_reject_last_path
from ..path import Path
from .._config import print
from ..cache.npy import save_npy
from ..core.utils import now, remove, cycle, process_state
from ..pathensemble import PathEnsemble
from ..execute.threads import ThreadExecutor

# worker "shoot" run method
class WorkerShoot(ABC):
    
    def shoot(self, target_state=1, k=0, sweep=False):
        """
        Run free simulations.
        total: total number of simulations.
        target_state: a (0), r (1), b (2) or any other
        """
        return self.run('shoot', target_state, k, sweep)
    
    def _shoot(self, target_state=1, k=0, sweep=False):
        
        # get/process params
        mode = 'shoot'
        directory = self._directory
        params = self.params
        do_tps = params.chain_type == 'tps'
        # which state are we talking about?
        states = params.states
        if params.at_least_one_transition_in_pool:
            at_least_one = states
        else:
            at_least_one = ''
        t = process_state(target_state, states)
        if not sweep:
            folder = f'{directory}/chain{t}{k}'
        else:
            folder = f'{directory}/sweep{t}{k}'
        initial_paths = self.initial_paths
        initial_paths._paths = cycle(initial_paths._paths, int(k))
        nbins = params.nbins
        max_length = params.max_length
        free_overriding_states = params.free_overriding_states
        
        # sweep
        if sweep:
            sweep_frames = initial_paths.merge()
            sweep_indices = sweep_frames.indices[sweep_frames.states == t]
            sweep_size = len(sweep_indices)
        elif t != states[1]:
            initial_paths = PathEnsemble(initial_paths.sample(1, t))
            pool_size = 1
        elif nbins > 1:
            pool_size = params.selection_pool_size
        else:
            pool_size = 1
        
        # eneconv
        if params.engine == 'gromacs':
            eneconv = params.gmx_eneconv
        else:
            eneconv = None
        
        # create folder if not existing
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f'+++ created {folder!r}')
        
        # load chain and pool
        if not sweep:
            if t == states[1] and not do_tps:
                print(f'\nLoading shooting chain and selection pool {now()}')
                chain = params.shot_chains(directory, t, k)  # weights accounted
                pool = PathEnsemble(f'{folder}/pool.log')
                print(f'... currently {len(chain)} path'
                      f'{"s" if len(chain) != 1 else ""} in shooting chain')
                print(f'... currently {len(pool)} path'
                      f'{"s" if len(pool) != 1 else ""} in selection pool')
            else:
                print(f'\nLoading shooting chain {now()}')
                chain = params.shot_chains(directory, t, k)  # weights accounted
                pool = PathEnsemble()
                print(f'... currently {len(chain)} path'
                      f'{"s" if len(chain) != 1 else ""} in shooting chain')
        else:
            chain = params.shot_paths(directory, 'sweep', t, k)
            print(f'\nReport after {len(chain)} paths')
            chain.report_shooting_results(states, sweep_size)
            print()

        # update total frames and steps
        self._total_frames = sum(chain.n_frames)
        self._total_steps = len(chain)
        
        # must have network, bins, and descriptors
        # only if it makes sense
        if t == states[1] and not sweep and nbins > 1:
            print(f'\nWaiting for neural network, bins, densities {now()}')
            while True:
                try:
                    params.update_network(
                        directory, timeout=0, raise_if_failure=True)
                    bins, densities = params.load_bins_and_densities(
                        directory, timeout=0, raise_if_failure=True)
                    break
                except:
                    if self.must_stop:
                        return
        
        # initialize
        back_simulation_completed = False
        forw_simulation_completed = False
        back = Path()
        forw = Path()
                
        # main cycle
        while not self.must_stop:

            # need to initialize?
            if not params.check_if_initialized(
                f'{folder}/back', f'{folder}/forw'):
                print(f'\nSelecting shooting point for '
                      f'{folder}/path{len(chain) + 1:06g} {now()}')
                
                if not sweep:
                    # update selection pool
                    # (add last chain path to pool if not already there)
                    update_selection_pool(
                        pool, pool_size, chain,
                        initial_paths, at_least_one=at_least_one)
                    
                    # select shooting point
                    shooting_point = select_shooting_point(
                        pool, params, folder, chain,
                        free_trajectories=params.free_trajectories(directory)
                        if free_overriding_states else [],
                        target_state=t)
                
                else:  # sweep
                    index = sweep_indices[len(chain) % len(sweep_indices)]
                    fname_index, loc = sweep_frames._get_local_loc(index)
                    print(f'=== selecting frame '
                          f'{sweep_frames._fnames[fname_index]}, {loc}')
                    shooting_point = sweep_frames[index]
                
                # clean
                remove(f'{folder}/*back*', f'{folder}/*forw*')
                
                # initialize simulation
                params.initialize_simulation(shooting_point,
                    f'{folder}/back', f'{folder}/forw')
                
                if not sweep: # save pool status (removed SP's source)
                    pool.save(f'{folder}/pool.log')
            
            # update existing paths: backward
            if not back_simulation_completed:
                (stop_frame, nframes, last_state, last_length) = \
                    self._simulate(f'{folder}/back', back, t, mode)
                back_simulation_completed = stop_frame is not None
                nframes_back = (stop_frame or 0) + last_length
                if nframes_back >= max_length:
                    forw = Path()  #  no need to simulate at all
                    forw_simulation_completed = True
                    nframes_back = max_length
            
            # check mid cycle
            if self.must_stop:
                return

            if back_simulation_completed and not forw_simulation_completed:
                offset = nframes_back - 1
                (stop_frame, nframes, last_state, last_length) = \
                    self._simulate(f'{folder}/forw', forw, t, mode, offset)
                forw_simulation_completed = stop_frame is not None
                nframes_forw = (stop_frame or 0) + last_length
                nframes_forw = min(nframes_forw, max_length - offset)
            
            # check mid cycle
            if self.must_stop:
                return
            
            if forw_simulation_completed:  # simulation is completed
                # join the two together in a path
                # (will inherit the right shooting index)
                path = back[nframes_back - 1::-1] + forw[1:nframes_forw]
                # save path and add it to chain
                # (zero weight in case of "bad" path)
                register_path(path, chain, eneconv)
                self._total_steps += 1

                # clean and reset
                remove(f'{folder}/*back*', f'{folder}/*forw*')
                back_simulation_completed = False
                forw_simulation_completed = False
                back = Path()
                forw = Path()
                
                if not sweep:
                    
                    # tps (also save weights)
                    if do_tps:
                        accept_or_reject_last_path(chain, params)
                        save_npy(f'{folder}/tps_weights.npy',
                                 chain.weights)
    
                    # do not include in pool in this specific case
                    elif not path.is_complete(t, states):
                        print('xxx path not valid, not including '
                              'it in selection pool')
                        path.weight = 0.
                
                else:  # print sweep summary
                    print(f'\nReport after {len(chain)} paths')
                    chain.report_shooting_results(states, sweep_size)
