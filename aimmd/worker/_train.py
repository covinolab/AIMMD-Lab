"""
...
"""

# external
import os
import time
import torch
import numpy as np
import shutil
from abc import ABC
from glob import glob
from math import inf, nan
from tqdm import tqdm

# aimmd imports
from .utils import rescale_bins, update_pathensemble
from .._config import NPY_CACHE, print
from ..cache.npy import save_npy
from ..core.utils import now, replace_in_cache
from ..pathensemble import PathEnsemble
from ..network.utils import find_knots_and_values, rescale
from ..execute.threads import ThreadExecutor
from ..analysis.utils import compute_bins
from ..pathensemble.utils import assemble_pathensemble

# train function of worker
class WorkerTrain(ABC):

    def train(self, nrounds=inf, keep_running=False, **kwargs):
        """nrounds: number of training rounds
        keep_running: after nrounds completed
            useful if waiting for new pathensemble data
        kwargs: passed to fit function
        """
        return self.run('train', nrounds, keep_running, **kwargs)
        
    
    def _train(self, nrounds=inf, keep_running=False, **kwargs):
        
        # process arguments
        nrounds = float(nrounds)
                
        # get/process params
        directory = self._directory
        params = self.params
        states = params.sorted_states
        r = states[1]
        network = params.network
        fit = params.fit
        nbins = params.nbins
        do_tps = params.chain_type == 'tps'
        cutoff_min = params.cutoff_min
        cutoff_max = params.cutoff_max
        marginal_bins = params.marginal_bins
        batch_size = params.network_batch_size
        rescale_committor = params.rescale_committor
        reweight_parameters = params.reweight_parameters
        ext = params.trajectory_extension
        compute_values_args = params.compute_values_args
        compute_condition = {'states': lambda state: state == r}
        compute_kwargs = lambda target : {
            'function': compute_values_args[0],
            'target': target,
            'source': compute_values_args[2],
            'conditions': compute_condition,
            'batch_size': batch_size}
        # initalize everywhere; compute just on the reactive region
        save_interval = params.network_save_interval
        initial_paths = self.initial_paths
        margins = PathEnsemble([path[1::-1] for path in initial_paths] +
                               [path[-2::1] for path in initial_paths])
        
        # total steps counter
        counter = tqdm(total=int(self.nsteps) if self.nsteps < inf else None,
                       position=0, file=self.original_stdout)
        
        # check wether you have to stop; updates pathensembles
        def must_stop():
            if self.must_stop:
                return True

            # reset
            self._total_steps = 0
            self._total_frames = 0

            # get chains
            self._shot_chains = params.shot_chains(
                directory, None, getattr(self, '_shot_chains', []))
            for chain in self._shot_chains:
                self._total_frames += sum(chain.n_frames)
                self._total_steps += len(chain)
            
            n = max(self._total_steps - counter.n, 0)
            if n:
                counter.update(n)
            
            # react fast when stop requested
            if self.must_stop:
                return True

            # get free trajectories
            self._free_trajectories = params.free_trajectories(directory)
            for trajectory in self._free_trajectories:
                self._total_frames = trajectory.n_frames
        
        # one cycle
        if must_stop():
            self.termination_signal = 2
            return
        
        # to fit function
        kwargs['worker'] = self
        
        # load the network if it is already possible
        network_fname = f'{directory}/network{states}.h5'
        params.update_network(directory, timeout=0, raise_if_failure=False)
        
        # main cycle
        rounds_done = 0
        while not self.termination_signal:

            # will also update paths loaded in worker
            if must_stop():
                self.termination_signal = 2
                return
            
            # assemble pathensemble
            print(f'\nLoading current path ensemble {now()}')
            pathensemble, added_frames = update_pathensemble(
                self, **compute_kwargs('values'))
            print(f'... {added_frames} new frames')
            
            # check mid-cycle
            if self.termination_signal:
                return
            
            if rounds_done >= nrounds:
                
                # nothing else to do
                if not keep_running:
                    self.termination_signal = 2
                    return

                # not changin source
                source = 'values'

            # train only in this case
            else:
                
                # fit function
                print(f'\nTraining the network '
                      f'(round {rounds_done + 1}, {now()})')
                losses, *_ = fit(params, pathensemble + margins, **kwargs)
                if len(losses):
                    source = 'new'
                    print(f'*** training completed {now()}')
                    
                    # periodically enforce
                    if self.termination_signal:
                        break
                    
                    rounds_done += 1
                
                else:
                    print(f'!!! training failed, reloading {now()}')
                    params.update_network(
                        directory, timeout=0, raise_if_failure=False)
                    source = 'values'
                
                # will also update paths loaded in worker
                if must_stop():
                    self.termination_signal = 2
                    return
                
                # assemble pathensemble (with margins)
                print(f'\nLoading current path ensemble {now()}')
                pathensemble, added_frames2 = update_pathensemble(
                    self, **compute_kwargs('values'))
                print(f'... {added_frames2} new frames')
                added_frames += added_frames2
                
                print(f'\nUpdating the values of '
                      f'all reactive {r} frames {now()}')
                n = pathensemble.compute(
                    **compute_kwargs(source), overwrite=source=='new')
                time.sleep(.1)  # stability
                print(f'... computed {n} values')
                # will fill temp files, replaced later
                
                # check mid-cycle
                if self.termination_signal:
                    return
            
            if source != 'new' and not added_frames:
                # wait for next cycle
                continue
            
            print(f'\nObtaining the adaptation bins {now()}')
            bins = compute_bins(pathensemble, nbins,
                                cutoff_max=cutoff_max,
                                cutoff_min=cutoff_min,
                                find_extremes_with='free',
                                source=source,
                                states=states,
                                marginal_bins=marginal_bins)
            print(f'    bins: {bins}')

            # check mid-cycle
            if self.termination_signal:
                return
            
            print(f'\nReweighting the full path ensemble {now()}')
            if not do_tps:  # if doing tps, already the right weights
                rw_p = reweight_parameters.copy()

                # find sp_cutoff_min and sp_cutoff_max
                sp_cutoff_min = bins[+0]
                sp_cutoff_max = bins[-1]
                if sp_cutoff_min == -inf and bins[+1] < +inf:
                    sp_cutoff_min = bins[+1]
                if sp_cutoff_max == +inf and bins[-2] > -inf:
                    sp_cutoff_max = bins[-2]
                if 'sp_cutoff_min' not in rw_p:
                    rw_p['sp_cutoff_min'] = sp_cutoff_min
                if 'sp_cutoff_max' not in rw_p:
                    rw_p['sp_cutoff_max'] = sp_cutoff_max

                # reweight
                result1 = pathensemble.reweight(
                    states, **rw_p, source=source)
                result2 = pathensemble.reweight(
                    states[::-1], **rw_p, source=source)
                w1, extremes1, xP1 = result1[0], result1[4], result1[5]
                w2, extremes2, xP2 = result2[0], result2[4], result2[5]
                # assign weights only to excursions
                excursions_mask = pathensemble.types(f'.{r}..')
                pathensemble.weights = (w1 + w2) * excursions_mask
                
                # bonus track: estimate rates
                lengths = pathensemble.n_frames
                k12 = np.sum(w1 * lengths)
                k12 = 1 / k12 if k12 else nan
                k21 = np.sum(w2 * lengths)
                k21 = 1 / k21 if k21 else nan
                print(f'    k12 estimate: {k12:.3e} [1/dt]')
                print(f'    k21 estimate: {k21:.3e} [1/dt]')
                
                # only after one training round: rescale committor
                # TODO in the future you may want to adjust it
                # after every reweighting
                if rescale_committor and source == 'new':
                    print(f'\nRescaling committor to match '
                          f'the crossing probability {now()}')
                    knots, values = find_knots_and_values(
                        extremes1, extremes2, xP1, xP2)
                    if not len(knots):
                        print(f'*** rescaling is not possible (yet)')
                    else:
                        print(f'***     knot       value')
                    for knot, value in zip(knots, values):
                        print(f'    {knot:+7.3f} -> {value:+7.3f}')
                        
                    # load interpolation in network
                    network.set_knots_and_values(knots, values)
                    
                    # rescale bins in place
                    rescale_bins(bins, knots, values)
                    print(f'    rescaled bins: {bins}')
                    
                    # rescale all of temp values
                    if len(knots):
                        pathensemble.compute(
                            lambda x: rescale(x, knots, values),
                            'new', 'new', compute_condition,
                            overwrite=True, worker=self)
                        time.sleep(.1)  # stability
            
            # check mid-cycle
            if self.termination_signal:
                break
            
            print(f'\nProjecting the {"T" if do_tps else ""}PE '
                  f'density {now()}')
            densities = pathensemble.project(bins, source=source)
            densities[densities == 0.] = 1e-15
            densities /= densities.sum()
            print(f'    densities: {densities}')
            
            if source == 'new':
                # replace values (as much as possible) all at once
                print(f'\nSubstituting \'...values.npy\' files '
                      f'with \'...new.npy\' {now()}')
                replace_in_cache(NPY_CACHE, '.new.npy', '.values.npy',
                                 set(pathensemble.fnames))
                
                # save network parameters
                print(f'\nSaving network parameters to '
                      f'{network_fname} {now()}')
                torch.save(network.state_dict(), network_fname)
            
            # save bins and densities
            print(f'\nSaving bins and densities {now()}')
            save_npy(f'{directory}/bins{states}.npy', bins)
            save_npy(f'{directory}/densities{states}.npy', densities)
            
            # backup network
            n = (counter.n // save_interval) * save_interval
            backup = f'{network_fname[:-3]}.step{n:06g}.h5'
            if counter.n and not os.path.exists(backup):
                shutil.copyfile(network_fname, backup)
                print(f'*** copied {network_fname!r} to {backup!r}')
