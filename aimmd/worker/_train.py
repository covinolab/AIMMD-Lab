"""
aimmd.worker._train
==================

Training loop for AIMMD workers.

This module defines :class:`WorkerTrain`, a mixin providing the public
:meth:`~WorkerTrain.train` convenience wrapper and the internal
:meth:`~WorkerTrain._train` task implementation used by :meth:`Worker.run`.

At a high level, the training task performs repeated "rounds" in which the
worker:

1) loads / updates the current path ensemble from disk (including any newly
   produced frames),
2) optionally fits the network on the updated data (and margin frames),
3) (re)computes committor-like values on reactive frames,
4) constructs adaptation bins from the current ensemble,
5) reweights the ensemble (if not doing TPS),
6) projects the (T)PE density onto the bins to obtain densities,
7) persists updated artifacts to disk (values files, network state, bins,
   densities, and periodic network backups).

Stop-condition handling is cooperative: the task periodically calls a local
``must_stop()`` helper, which consults :attr:`must_stop` and updates the worker's
internal counters (:attr:`total_steps`, :attr:`total_frames`) with the associated
progress bars. The loop exits by setting :attr:`termination_signal` to a
SIGINT-like marker.

Key inputs come from :class:`~aimmd.params.Params`, notably:

- end states and reactive region label (via :attr:`Params.sorted_states`),
- the network object and fit routine (via :attr:`Params.network` and
  :attr:`Params.fit`),
- reweighting parameters and binning configuration,
- compute pipeline arguments for producing values used for binning/reweighting.

Artifacts written by this task
------------------------------
The training loop may write or update the following files inside the worker
directory:

- ``network{states}.h5``: serialized torch state dict of the network,
- ``bins{states}.npy``: current adaptation bin boundaries,
- ``densities{states}.npy``: projected density on the bins,
- ``...values.npy`` / ``...new.npy`` files within path ensemble folders:
  values are computed and atomically substituted when training produced new
  predictions,
- network backups: ``network{states}.step?????.h5`` (periodic snapshots).

Notes
-----
- The internal helper ``must_stop()`` is expected to be called frequently. It
  updates counters an report progress bars by scanning chains/trajectories on
  disk through :class:`~aimmd.params.Params` helper methods.
- When ``keep_running=True`` and ``nrounds`` rounds are already completed, the
  worker continues to monitor the filesystem and refresh densities/bins whenever
  new frames appear.
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
from ..execute.threads import ThreadExecutor
from ..analysis.utils import compute_bins
from ..pathensemble.utils import assemble_pathensemble
from ..network.rescale_utils import find_knots_and_values, rescale


class WorkerTrain(ABC):

    def train(self, nrounds=1, keep_running=False, **kwargs):
        """
        Public convenience wrapper for the training task.

        This method forwards to :meth:`Worker.run` with ``task='train'`` and
        passes ``nrounds`` and ``keep_running`` as positional arguments.

        Parameters
        ----------
        nrounds : int or float, optional
            Number of training rounds to perform. A "round" corresponds to one
            successful call to the fit routine (i.e., a round only counts if
            training returns at least one loss value). Default is ``1``.
        keep_running : bool, optional
            If ``False``, the worker stops once ``nrounds`` successful rounds
            have completed. If ``True``, the worker keeps monitoring the path
            ensemble and continues to refresh bins/densities when new frames
            appear, even after completing ``nrounds``. Default is ``False``.
        **kwargs
            Additional keyword arguments forwarded to the fit routine via
            :meth:`Worker.run` / :meth:`_train`. Stop-condition keywords
            (typically ``walltime``, ``nsteps``, ``nframes``) may be consumed by
            the run wrapper, depending on the worker implementation.

        Returns
        -------
        object
            Whatever :meth:`Worker.run` returns for the ``'train'`` task.
        """
        return self.run('train', nrounds, keep_running, **kwargs)

    def _train(self, nrounds=inf, keep_running=False, **kwargs):
        """
        Internal implementation of the ``'train'`` worker task.

        Parameters
        ----------
        nrounds : float, optional
            Maximum number of *successful* training rounds to perform. Values
            are converted to ``float`` internally. Default is ``inf``.
        keep_running : bool, optional
            If ``True``, keep looping after ``nrounds`` have completed and wait
            for new data; otherwise stop once the requested rounds are done.
            Default is ``False``.
        **kwargs
            Keyword arguments passed through to the fit routine
            ``params.fit(...)``. This method injects ``worker=self`` into
            ``kwargs`` before calling the fit routine.

        Returns
        -------
        None

        Notes
        -----
        The method exits by setting :attr:`termination_signal` and returning
        ``None``. It is designed to be executed under :meth:`Worker.run`, which
        handles directory management, cache clearing, and cleanup.
        """
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
        compute_kwargs = lambda target: {
            'function': compute_values_args[0],
            'target': target,
            'source': compute_values_args[2],
            'conditions': compute_condition,
            'batch_size': batch_size}
        # initalize everywhere; compute just on the reactive region
        save_interval = params.network_save_interval
        initial_paths = self.initial_paths
        # only transitions
        initial_paths = initial_paths.extract(states, states[::-1])
        margins = PathEnsemble([path[1::-1] for path in initial_paths] +
                               [path[-2::1] for path in initial_paths])

        steps_counter = None
        frames_counter = None

        # routin checking wether you have to stop; it updates pathensembles
        def must_stop(): 
            nonlocal steps_counter, frames_counter
            
            if self.must_stop:
                return True

            # reset
            total_steps = 0
            total_frames = 0

            # get chains
            self._shot_chains = params.shot_chains(
                directory, None,
                old=getattr(self, '_shot_chains', []))
            for chain in self._shot_chains:
                total_frames += sum(chain.n_frames)
                total_steps += len(chain)
            
            # react fast when stop requested
            if self.must_stop:
                return True
            
            # get free trajectories
            self._free_trajectories = params.free_trajectories(directory)
            for trajectory in self._free_trajectories:
                total_frames += trajectory.n_frames
            
            # assign (will update progress bars)
            self.total_steps = total_steps
            self.total_frames = total_frames
        
        # one cycle
        if must_stop():
            self.termination_signal = 2
            return
        
        # to fit function
        kwargs['worker'] = self
        
        # load the network if it is already possible
        print(f'\nLoading pre-existing network parameters {now()}')
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
                    **compute_kwargs(source), overwrite=source == 'new')
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

            # get TPE weights
            if do_tps:
                weights = (pathensemble.weights *
                           pathensemble.are_transitions(states))
            
            # reweight pathensemble
            print(f'\nReweighting the full path ensemble {now()}')
            
            # will update reweighting parameters
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

            # bonus track: estimate rates
            lengths = pathensemble.n_frames
            k12 = np.sum(w1 * lengths)
            k12 = 1 / k12 if k12 else nan
            k21 = np.sum(w2 * lengths)
            k21 = 1 / k21 if k21 else nan
            print(f'    k12 estimate: {k12:.3e} [1/dt]')
            print(f'    k21 estimate: {k21:.3e} [1/dt]')
            print(f'    {pathensemble.n_frames.sum()} frames')
            
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

            # assign weights
            if not do_tps:  # PE weights in reactive region
                excursions_mask = pathensemble.types(f'.{r}..')
                pathensemble.weights = (w1 + w2) * excursions_mask
            else:  # TPE weights
                pathensemble.weights *= pathensemble.are_transitions(states)
            
            print(f'\nProjecting the {"T" * do_tps}PE density {now()}')
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
            n = (self.total_steps // save_interval) * save_interval
            backup = f'{network_fname[:-3]}.step{n:06g}.h5'
            if self.total_steps and not os.path.exists(backup):
                shutil.copyfile(network_fname, backup)
                print(f'*** copied {network_fname!r} to {backup!r}')
