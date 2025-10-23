import os
import time
import numpy as np
import functools
import traceback
from tqdm import tqdm
from ..core.utils import (now,
                          remove,
                          stop_simulation,
                          load_initial_paths,
                          load_network_and_projections,
                          initialize_shooting_simulation,
                          update_shooting_chain,
                          update_selection_pool,
                          update_pathensemble,
                          scorporate_pathensembles,
                          update_equilibrium_simulations,
                          update_shooting_simulation,
                          add_path_to_chain,
                          run_acceptance_rejection_on_latest_path)
from ..core.pathensemble import PathEnsemble, PathEnsemblesCollection
                          
inf = float('inf')

# quick logging
print = functools.partial(print, flush=True)

def manage(self, n, nA, nB, eA=0, eB=0,
            log_file=None, nsteps=inf,
            nframes=inf, walltime=inf):
    """
    n: number of replicas dedicated to shooting simulations
       (creates folders if not existing)
    nA: number of replicas dedicated to free simulations around A
    nB: number of replicas dedicated to free simulations around B
    eA: number of replicas dedicated to extending transitions reaching A
    eA: number of replicas dedicated to extending transitions reaching B
    """
    
    # report
    self.log_file = log_file
    print(f"Starting worker: manage ({now()})")
    if not log_file:
        print(f"Press Control+C to interrupt.")
    
    # process arguments
    n = int(n)
    nA = int(nA)
    nB = int(nB)
    eA = int(eA)
    eB = int(eB)
    nsteps = float(nsteps)
    if nsteps < inf:
        nsteps = int(nsteps)
    nframes = float(nframes)
    walltime = float(walltime)
    
    # bind resources
    self.bind_resources()
    
    # initialize output
    pathensemble = None
    
    # get aimmd run parameters (in order of appearance)
    print(f'\nLoading AIMMD run parameters ({now()})')
    directory = self.directory
    topology = self.params.topology
    states_function = self.params.states_function
    descriptors_function = self.params.descriptors_function
    values_function = self.params.values_function
    network = self.params.network
    equilibrium_overriding_states = self.params.equilibrium_overriding_states
    extra_equilibriumA = self.params.extra_equilibriumA
    extra_equilibriumB = self.params.extra_equilibriumB
    extra_equilibriumA_states_map = self.params.extra_equilibriumA_states_map
    extra_equilibriumB_states_map = self.params.extra_equilibriumB_states_map
    selection_pool_size = self.params.selection_pool_size
    at_least_one_transition_in_pool = \
        self.params.at_least_one_transition_in_pool
    do_tps = self.params.do_tps
    if do_tps:
        selection_pool_size = 1
    if self.params.engine == 'gromacs':
        eneconv = self.params.gmx_eneconv
    else:
        eneconv = ''
    trajectory_extension = self.params.trajectory_extension
    save_interval = self.params.save_interval
    
    # create necessary shooting folders if they do not exist yet
    for i in range(n):
        folder = f'{directory}/shots{i}'
        if not os.path.exists(folder):
            os.system(f'mkdir {folder}')
            print(f'+++ created {folder}')
    
    print(f'\nLoading initial path(s) ({now()})')
    initial_paths = load_initial_paths(f'{directory}/initial_paths', topology,
        states_function, descriptors_function, values_function)
    print(f'    {initial_paths}')
    assert initial_paths.nframes
    
    print(f'\nLoading shooting chains ({now()})')
    chains = []
    backwards = []  # shooting simulation segments
    forwards = []
    for chain_id in range(n):
        chain = PathEnsemble()
        update_shooting_chain(chain, chain_id,
            directory, topology, states_function, descriptors_function,
            values_function, load_h5=True)
        chain.save(f'{chain.directory}/chain.h5', directory='.')
        chains.append(chain)
        print(f'    {chain}')
        backward = chain[:0]
        forward = chain[:0]
        # reset values function to speed-up computation
        backward.values_function = lambda descriptors: np.repeat(
          0., len(descriptors))
        forward.values_function = lambda descriptors: np.repeat(
          0., len(descriptors))
        backwards.append(backward)
        forwards.append(forward)
    
    # available transitions
    pathensemble = PathEnsemblesCollection(*chains)
    if len(pathensemble):
        available_transitions = pathensemble[pathensemble.are_transitions]
    else:
        available_transitions = []
    
    print(f'\nLoading selection pools ({now()})')
    pools = []
    for chain_id in range(n):
        print(f'\n    shots{chain_id}')
        chain = chains[chain_id]
        pool = update_selection_pool(
            PathEnsemble(), chain, selection_pool_size,
            None, initial_paths, at_least_one_transition_in_pool, True)
        
        # attempt recovery from unpleasant situation (not supported with TPS)
        if len(chain) and chain.weights[-1] and \
            pool.trajectory_files[-1] != chain.trajectory_files[-1]:
            pool_index = np.load(f'{chain.directory}/pool_index.npy')
            print(f'\n!!! updating pool with missing {chain.directory}/'
                  f'{chain.trajectory_files[-1]}')
            pool = update_selection_pool(
                pool, chain, selection_pool_size, pool_index,
                initial_paths, at_least_one_transition_in_pool)
            remove(f'{chain.directory}/back.xtc')  # call for new simulation
        
        pool.save(f'{chains[chain_id].directory}/pool.h5', directory='.')
        pools.append(pool)
    
    print(f'\nLoading free simulations ({now()})')
    eq_current, eq_completed, ext_current = [], [], []
    update_equilibrium_simulations(
        eq_current, eq_completed, directory, nA, nB, initial_paths, 
        self.params, eA, eB, ext_current, available_transitions,
        save_h5=True, simulate=False, verbose=True)
    
    # update full pathensemble (with equilibrium simulations)
    pathensemble = PathEnsemblesCollection(*chains, *eq_current, *eq_completed)
    shooting_chains, equilibriumA, equilibriumB = \
        scorporate_pathensembles(pathensemble)
    
    # extra equilibriumA and equilibriumB
    if len(extra_equilibriumA):
        print(f'\nLoading extra free simulations around A ({now()})')
        equilibriumA += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[],
            equilibriumA=extra_equilibriumA, equilibriumB=[],
            equilibriumA_states_map=extra_equilibriumA_states_map,
            verbose=True)[0]
    if len(extra_equilibriumB):
        print(f'\nLoading extra free simulations around B ({now()})')
        equilibriumB += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[],
            equilibriumA=[], equilibriumB=extra_equilibriumB,
            equilibriumB_states_map=extra_equilibriumB_states_map,
            verbose=True)[0]  
    
    # will wait until the training part has completed at least one cycle
    print(f'\nWaiting for neural network parameters ({now()})')
    bins, densities = load_network_and_projections(
        network, directory, worker=self)
    
    # update full pathensemble
    pathensemble = shooting_chains + equilibriumA + equilibriumB
    
    # initialize step counter
    step_number = tqdm(total=nsteps, ncols=70, file=self.original_stdout,
        initial=int(sum([len(chain) for chain in chains])))
    
    # stop condition
    t0 = time.time()
    def stop_condition():
        if time.time() - t0 > walltime or step_number.n >= nsteps:
            self.termination_signal = 2  # keyobard interrupt
        return bool(self.termination_signal)
    
    # main cycle
    print(f'\nStarting the main cycle ({now()})')
    while True:
        
        # stop?
        if stop_condition():
            break
        
        if len(pathensemble):
            available_transitions = pathensemble[pathensemble.are_transitions]
        else:
            available_transitions = []
        
        # manage free simulations
        update_equilibrium_simulations(
            eq_current, eq_completed, directory, nA, nB, initial_paths, 
            self.params, eA, eB, ext_current, available_transitions,
            save_h5=True, simulate=True, verbose=False)
        
        # update full pathensemble (with equilibrium simulations)
        pathensemble = PathEnsemblesCollection(
            *chains, *eq_current, *eq_completed)
        shooting_chains, equilibriumA, equilibriumB = \
            scorporate_pathensembles(pathensemble)
        
        if pathensemble.nframes > nframes:
            print(f'\nReached target total number '
                  f'of frames {nframes} ({now()})')
            break
        
        # extra equilibriumA and equilibriumB
        if len(extra_equilibriumA):
            #print(f'\nLoading extra free simulations around A ({now()})')
            equilibriumA += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
                shooting_chains=[],
                equilibriumA=extra_equilibriumA, equilibriumB=[],
                equilibriumA_states_map=extra_equilibriumA_states_map,
                verbose=True)[0]
        if len(extra_equilibriumB):
            #print(f'\nLoading extra free simulations around B ({now()})')
            equilibriumB += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
                shooting_chains=[],
                equilibriumA=[], equilibriumB=extra_equilibriumB,
                equilibriumB_states_map=extra_equilibriumB_states_map,
                verbose=True)[0]
        pathensemble = shooting_chains + equilibriumA + equilibriumB
        
        # manage shooting simulations (permutated k for less bias, two times)
        for k in np.random.permutation(np.arange(n)):
            files = os.listdir(chains[k].directory)
            worker_id = f'{directory}/worker{k + nA + nB + eA + eB}'
            
            # initialize shooting simulation if necessary
            if not ((f'back{trajectory_extension}' in files or
                     f'back.tpr' in files) and
                    (f'forw{trajectory_extension}' in files or
                     f'forw.tpr' in files)):
                stop_simulation(worker_id)  # safety: wait for worker ready
                eq_overriding = PathEnsemblesCollection()
                if 'A' in equilibrium_overriding_states:
                    eq_overriding += equilibriumA
                if 'B' in equilibrium_overriding_states:
                    eq_overriding += equilibriumB
                initialize_shooting_simulation(chains[k], pools[k],
                directory, self.params, shooting_chains, eq_overriding)
                
                # reset segments
                backwards[k] = backwards[k][:0]
                forwards[k] = forwards[k][:0]
            
            # advance simulation
            path, states, descriptors = update_shooting_simulation(
                backwards[k], forwards[k], worker_id, self.params)
            
            # path is not completed
            if not len(path):
                continue
            
            # path is completed: add path to chain
            add_path_to_chain(path, chains[k], states, descriptors,
                              trajectory_extension, eneconv)
            
            # run acceptance/rejection to determine weight
            if do_tps:
                run_acceptance_rejection_on_latest_path(chains[k], network)
            
            # update pool
            pool_index = np.load(f'{chains[k].directory}/pool_index.npy')
            if chains[k].weights[-1]:
                print(f'\nUpdating selection pool shots{k}')
                pools[k] = update_selection_pool(
                    pools[k], chains[k], selection_pool_size, pool_index,
                    initial_paths, at_least_one_transition_in_pool)
            
            # update step number and save
            step_number.update(1)
            print('')
            chains[k].save(f'{chains[k].directory}/chain.h5', directory='.')
            pools[k].save(f'{pools[k].directory}/pool.h5', directory='.')
            if step_number.n % save_interval == 0:
                os.system(f'cp {directory}/network.h5 '
                          f'{directory}/network{step_number.n:06g}.h5')
            
            # stop?
            if stop_condition():
                break
            
            # call for a new simulation
            remove(f'{backwards[k].directory}/back{trajectory_extension}')
            remove(f'{forwards[k].directory}/forw{trajectory_extension}')
            stop_simulation(worker_id)  # safety
            eq_overriding = PathEnsemblesCollection()
            if 'A' in equilibrium_overriding_states:
                eq_overriding += equilibriumA
            if 'B' in equilibrium_overriding_states:
                eq_overriding += equilibriumB
            initialize_shooting_simulation(chains[k], pools[k],
                directory, self.params, shooting_chains, eq_overriding)
            
            # reset segments
            backwards[k] = backwards[k][:0]
            forwards[k] = forwards[k][:0]
            
            # advance simulation
            update_shooting_simulation(backwards[k], forwards[k],
                worker_id, self.params)
    
    # complete
    step_number.close()
    print(f'\nReached {step_number.n} steps ({now()}), '
          f'{pathensemble.nframes} total frames')
    return pathensemble
