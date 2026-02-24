"""
...
"""

# external
import os
import numpy as np
from abc import ABC
from math import inf
from pathlib import PosixPath
from itertools import islice

# aimmd imports
from ..cache.npy import save_npy
from ..core.utils import remove, unique_path
from ..path.utils import get_cache_fname

# launcher build method
class LauncherBuild(ABC):

    def _build(self):
        """returns arguments and descriptions"""
        offset = 0
        args = []
        descriptions = []
        process_identifiers = []  # keep track of what you will initialize
        # to avoid conflicting processes running together
        termination_timeout = max(self.termination_timeout - 1., 0.0)
        num_processes = sum(self._num_processes)
        if num_processes == 0:
            raise RuntimeError('no processes to run')
        
        # initialize worker id
        i = 0
        
        for run_id in range(len(self)):
            params = self._params[run_id]
            if not params.initial_paths:
                raise TypeError("'initial_paths' missing in aimmd.Params")
            a, r, b = params.states
            sorted_states = params.sorted_states
            params_path = os.path.relpath(params.save())
            directory = self._directories[run_id]
            n = self._n[run_id]
            n1 = self._n1[run_id]
            n2 = self._n2[run_id]
            reactive_region_mode = self._reactive_region_mode[run_id]
            state1_mode = self._state1_mode[run_id]
            state2_mode = self._state2_mode[run_id]
            nsteps = self._nsteps[run_id]
            nframes = self._nframes[run_id]
            nrounds = self._nrounds[run_id]
            walltime = self._walltime
            if nrounds:
                conditions = (inf, inf, inf)
            elif n + n1 + n2:
                conditions = (walltime, nsteps, nframes)
            else:  # not running anything here
                continue

            # main directory
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f'+++ created {directory!r}')
            
            # initial paths
            folder = f'{directory}/initial{sorted_states}'
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f'+++ created {folder!r}')
            remove(f'{folder}/*') 
            for path in params.initial_paths:
                old = path.fname
                fname = unique_path(f'{folder}/{PosixPath(old).name}', '.trr')
                path.write(fname)
                print(f'+++ saved {str(fname)!r} (from: {old!r})')
                for attribute, series in islice(
                    path.__dict__.items(), 6, None):
                    name = get_cache_fname(fname, attribute)
                    save_npy(name, series)
                    print(f'+++ saved {name!r}')
            
            # free simulations
            for t, m in zip([r, a, b], [n, n1, n2]):
                if not m:
                    continue
                if ((t == a and state1_mode == 'shoot') or
                    (t == b and state2_mode == 'shoot')):
                    continue
                if t == r and reactive_region_mode != 'free':
                    continue
                # folders
                folder = f'free{t}'
                dfolder = f'{directory}/{folder}'
                if not os.path.exists(dfolder):
                    os.makedirs(dfolder)
                    print(f'+++ created {dfolder}')
                for k in range(m):
                    process_identifier = f'{dfolder}{k}'
                    if process_identifier in process_identifiers:
                        raise RuntimeError(f"can't initialize process {i} "
                              f"(free {k}) in {dfolder!r} more than once")
                    process_identifiers.append(process_identifier)

                    
                    localid = i % self._ntasks_per_node
                    if num_processes > 1:
                        log_file = f'{folder}/worker{k}.log'
                    else:
                        log_file = 'stdout'
                    args.append(
                        (params_path, directory, localid,
                         self._cpus_per_task, self._gpus_per_task,
                         log_file, *conditions, termination_timeout,
                         'free', t, k, m, nrounds > 0 or n > 0))
                    descriptions.append(f'"{directory}" {folder} (worker{k})')
                    i += 1
            
            # shot simulations
            for t, m in zip([r, a, b], [n, n1, n2]):
                if t == r and reactive_region_mode == 'free':
                    continue
                if ((t == a and state1_mode != 'shoot') or
                    (t == b and state2_mode != 'shoot')):
                    continue
                for k in range(m):
                    if t == r and reactive_region_mode == 'sweep':
                        folder = f'sweep{t}{k}'
                    else:
                        folder = f'chain{t}{k}'
                    dfolder = f'{directory}/{folder}'
                    process_identifier = dfolder
                    if process_identifier in process_identifiers:
                        raise RuntimeError(f"can't initialize process {i} "
                              f"(shoot) in {dfolder!r} more than once")
                    process_identifiers.append(process_identifier)
                    if not os.path.exists(dfolder):
                        os.makedirs(dfolder)
                        print(f'+++ created {dfolder}')
                    sweep = False
                    if t == r:
                        if reactive_region_mode == 'sweep':
                            sweep = True
                        else:
                            os.system(f'touch {dfolder}/pool.log')
                    localid = i % self._ntasks_per_node
                    if num_processes > 1:
                        log_file = f'{folder}/worker.log'
                    else:
                        log_file = 'stdout'
                    noappend = False
                    args.append(
                        (params_path, directory, localid,
                         self._cpus_per_task, self._gpus_per_task,
                         log_file, *conditions, termination_timeout,
                         'shoot', t, k, sweep))
                    descriptions.append(f'"{directory}" {folder}')
                    i += 1
            
            # trainer
            if nrounds:
                localid = i % self._ntasks_per_node
                process_identifier = f'{directory}{sorted_states}'
                if process_identifier in process_identifiers:
                   raise RuntimeError(f"can't initialize process {i} (train "
                          f"{sorted_states}) in {directory!r} more than once")
                process_identifiers.append(process_identifier)
                if num_processes > 1:
                    keep_running = True
                    log_file = f'train{sorted_states}.log'
                else:
                    keep_running = False
                    log_file = 'stdout'
                args.append(
                    (params_path, directory, localid,
                     self._cpus_per_task, self._gpus_per_task,
                     log_file, walltime, nsteps, nframes,
                     termination_timeout, 'train', nrounds, keep_running))
                descriptions.append(f'"{directory}" {sorted_states} trainer')
                i += 1
        
        # combine
        return args, descriptions
