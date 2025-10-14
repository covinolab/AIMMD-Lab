from ..core.utils import *

def manage(self, log_file=None, nsteps=int(1e6), nframes=np.inf):
    if log_file:
        log_file=f'{self.directory}/{log_file}'
    
    # get aimmd run parameters
    write(f'\nLoading AIMMD run parameters ({now()})', log_file)
    
    # extract necessary parameters (in order of appearance)
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
    at_least_one_transition_in_pool = self.params.at_least_one_transition_in_pool
    do_tps = self.params.do_tps
    if do_tps:
        selection_pool_size = 1
    eneconv = self.params.gmx_eneconv if self.params.engine == 'gromacs' else ''
    trajectory_extension = self.params.trajectory_extension
    save_interval = self.params.save_interval
    
    write(f'\nLoading initial path(s) ({now()})', log_file)
    initial_paths = load_initial_paths(f'{directory}/initial_paths', topology,
        states_function, descriptors_function, values_function)
    write(f'    {initial_paths}', log_file)
    assert initial_paths.nframes
    
    write(f'\nLoading shooting chains ({now()})', log_file)
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
        write(f'    {chain}', log_file)
        backwards.append(chain[:0])
        forwards.append(chain[:0])
    
    # available transitions
    pathensemble = PathEnsemblesCollection(*chains)
    if len(pathensemble):
        available_transitions = pathensemble[pathensemble.are_transitions]
    else:
        available_transitions = []
    
    write(f'\nLoading selection pools ({now()})', log_file)
    pools = []
    for chain_id in range(n):
        write(f'\n    shots{chain_id}', log_file)
        chain = chains[chain_id]
        pool = update_selection_pool(PathEnsemble(), chain, selection_pool_size,
            None, initial_paths, at_least_one_transition_in_pool, True)
    
        # attempt recovery from unpleasant situation (not supported with TPS)
        if len(chain) and chain.weights[-1] and \
            pool.trajectory_files[-1] != chain.trajectory_files[-1]:
            pool_index = np.load(f'{chain.directory}/pool_index.npy')
            write(f'\n!!! updating pool with missing {chain.directory}/'
                  f'{chain.trajectory_files[-1]}', log_file)
            pool = update_selection_pool(
                pool, chain, selection_pool_size, pool_index,
                initial_paths, at_least_one_transition_in_pool)
            remove(f'{chain.directory}/back.xtc')  # call for new simulation
        
        pool.save(f'{chains[chain_id].directory}/pool.h5', directory='.')
        pools.append(pool)
    
    write(f'\nLoading free simulations ({now()})', log_file)
    eq_current, eq_completed, ext_current = [], [], []
    update_equilibrium_simulations(
        eq_current, eq_completed, directory, nA, nB, initial_paths, 
        aimmd_run_params, eA, eB, ext_current, available_transitions,
        save_h5=True, simulate=False, verbose=True)
    
    # update full pathensemble (with equilibrium simulations)
    pathensemble = PathEnsemblesCollection(*chains, *eq_current, *eq_completed)
    shooting_chains, equilibriumA, equilibriumB = \
        scorporate_pathensembles(pathensemble)
    
    # extra equilibriumA and equilibriumB
    if len(extra_equilibriumA):
        write(f'\nLoading extra free simulations around A ({now()})', log_file)
        equilibriumA += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=extra_equilibriumA, equilibriumB=[],
            equilibriumA_states_map=extra_equilibriumA_states_map,
            verbose=True)[0]
    if len(extra_equilibriumB):
        write(f'\nLoading extra free simulations around B ({now()})', log_file)
        equilibriumB += update_pathensemble(directory, topology,
            states_function, descriptors_function, values_function,
            add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=[], equilibriumB=extra_equilibriumB,
            equilibriumB_states_map=extra_equilibriumB_states_map,
            verbose=True)[0]  
    
    # will wait until the training part completed at least one cycle
    write(f'\nWaiting for neural network parameters ({now()})', log_file)
    bins, densities = load_network_and_projections(network, directory)
    
    # update full pathensemble
    pathensemble = shooting_chains + equilibriumA + equilibriumB
    
    # initialize step counter
    step_number = tqdm(total=nsteps, ncols=70,
        initial=int(sum([len(chain) for chain in chains])))
    
    # main cycle
    write(f'\nStarting the main cycle ({now()})', log_file)
    while step_number.n < nsteps:
        
        # update candidate transitions
        if len(pathensemble):
            available_transitions = pathensemble[pathensemble.are_transitions]
        else:
            available_transitions = []
        
        # manage free simulations
        update_equilibrium_simulations(
            eq_current, eq_completed, directory, nA, nB, initial_paths, 
            aimmd_run_params, eA, eB, ext_current, available_transitions,
            save_h5=True, simulate=True, verbose=False)
        
        # update full pathensemble (with equilibrium simulations)
        pathensemble = PathEnsemblesCollection(*chains, *eq_current, *eq_completed)
        shooting_chains, equilibriumA, equilibriumB = \
            scorporate_pathensembles(pathensemble)
        
        if pathensemble.nframes > nframes:
            write(f'\nReached target total number of frames {nframes} ({now()})', log_file)
            break
        
        # extra equilibriumA and equilibriumB
        if len(extra_equilibriumA):
            #write(f'\nLoading extra free simulations around A ({now()})', log_file)
            equilibriumA += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=extra_equilibriumA, equilibriumB=[],
                equilibriumA_states_map=extra_equilibriumA_states_map,
                verbose=True)[0]
        if len(extra_equilibriumB):
            #write(f'\nLoading extra free simulations around B ({now()})', log_file)
            equilibriumB += update_pathensemble(directory, topology,
                states_function, descriptors_function, values_function,
                add_missing_paths=False, add_missing_frames=False,
            shooting_chains=[], equilibriumA=[], equilibriumB=extra_equilibriumB,
                equilibriumB_states_map=extra_equilibriumB_states_map,
                verbose=True)[0]
        pathensemble = shooting_chains + equilibriumA + equilibriumB
        
        # manage shooting simulations (permutated k for less bias, two times)
        for k in np.random.permutation(np.arange(n)):
            files = os.listdir(chains[k].directory)
            worker_id = f'{directory}/worker{k + nA + nB + eA + eB}.run'
            
            # initialize shooting simulation if necessary
            if not ((f'back{trajectory_extension}' in files or
                     f'back.tpr' in files) and
                    (f'forw{trajectory_extension}' in files or
                     f'forw.tpr' in files)):
                stop_simulation(worker_id)  # safety
                eq_overriding = PathEnsemblesCollection()
                if 'A' in equilibrium_overriding_states:
                    eq_overriding += equilibriumA
                if 'B' in equilibrium_overriding_states:
                    eq_overriding += equilibriumB
                initialize_shooting_simulation(chains[k], pools[k],
                directory, aimmd_run_params, shooting_chains, eq_overriding)
                
                # reset segments
                backwards[k] = backwards[k][:0]
                forwards[k] = forwards[k][:0]
            
            # advance simulation
            path, states, descriptors = update_shooting_simulation(
                backwards[k], forwards[k], worker_id, aimmd_run_params)
            
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
                write(f'\nUpdating selection pool shots{k}', log_file)
                pools[k] = update_selection_pool(
                    pools[k], chains[k], selection_pool_size, pool_index,
                    initial_paths, at_least_one_transition_in_pool)
            
            # update step number and save
            write('', log_file); step_number.update(1); write('\n', log_file)
            chains[k].save(f'{chains[k].directory}/chain.h5', directory='.')
            pools[k].save(f'{pools[k].directory}/pool.h5', directory='.')
            if step_number.n % save_interval == 0:
                os.system(f'cp {directory}/network.h5 '
                          f'{directory}/network{step_number.n:06g}.h5')
            
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
                directory, aimmd_run_params, shooting_chains, eq_overriding)
            
            # reset segments
            backwards[k] = backwards[k][:0]
            forwards[k] = forwards[k][:0]
            
            # advance simulation
            update_shooting_simulation(backwards[k], forwards[k],
                worker_id, aimmd_run_params)
    
    # complete
    step_number.close()
    write(f'\nReached target nsteps {step_number.n} >= {nsteps} ({now()})', log_file)
    
    # two last training rounds (sure to have most updated data)
    write(f'\nLast training rounds', log_file)
    for _ in range(2):
        remove(f'{directory}/network.h5')
        remove(f'{directory}/bins.npy')
        remove(f'{directory}/densities.npy')
        load_network_and_projections(network, directory)
    write(f'*** completed ({now()})', log_file)
    
    # handle dependency
    step_number.close()
