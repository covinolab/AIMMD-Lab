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
import io
import os
import time
import torch
import numpy as np
import shutil
from abc import ABC
from math import inf, nan

# aimmd imports
from .utils import rescale_bins, get_initial_frames_for_training
from .._config import NPY_CACHE, MDA_CACHE, print
from ..cache.npy import save_npy
from ..core.utils import now, replace_in_cache
from ..pathensemble import PathEnsemble
from ..analysis.utils import compute_bins
from ..pathensemble.utils import assemble_pathensemble
from ..network.rescale_utils import find_knots_and_values, rescale
from ..path.utils import get_cache_fname

# WorkerTrain mixin class
class WorkerTrain(ABC):

    def _cache_bias_files(self, pathensemble, bias_function):
        """Cache per-file bias arrays (file-mode bias tracking).

        For each unique trajectory file in ``pathensemble``, ensure
        ``{fname}.bias.npy`` exists and covers at least every frame of the
        underlying trajectory. The cache is rewritten whenever the existing
        ``bias.npy`` is shorter than the current frame count — this matters
        because free trajectories grow between rounds while a stale
        ``bias.npy`` would otherwise satisfy a naive ``is not None`` check
        and silently produce truncated bias slices in
        :func:`compute_bias_corrections`.

        ``bias_function(fname)`` is expected to return a 1-D array of length
        equal to the current trajectory frame count, or ``None`` if no bias
        source is available for that file (in which case the file is left
        alone — a worker-level fallback, e.g. the zero-bias cache written by
        ``_free.py`` for the part0000 seed, may already be in place).

        Parameters
        ----------
        pathensemble : PathEnsemble
            Ensemble whose underlying trajectory files should have up-to-date
            bias caches.
        bias_function : callable
            ``params.bias_function`` (``bias_source='file'``).

        Notes
        -----
        Prints ``Computing bias cache (file mode)`` at invocation and
        ``... wrote bias cache for N trajectory files`` when any rewrite
        happened.
        """
        print(f'\nComputing bias cache (file mode) {now()}')
        seen = set()
        n_cached = 0
        for _path in pathensemble:
            for _fname in _path._fnames:
                if _fname in seen:
                    continue
                seen.add(_fname)
                _cache_fname = get_cache_fname(_fname, 'bias')
                _reader = MDA_CACHE.get(_fname)
                if _reader is None:
                    continue
                _n_traj_frames = len(_reader.trajectory)
                _existing = NPY_CACHE.get(
                    _cache_fname, min_length=_n_traj_frames)
                # Enforce length: `NPY_CACHE.get` returns whatever it finds
                # on disk even if it's shorter than `min_length`, so a short
                # stale file would satisfy `is not None`. Check length too.
                if (_existing is not None
                        and len(_existing) >= _n_traj_frames):
                    continue
                _bias = bias_function(_fname)
                if _bias is None:
                    continue  # no COLVAR for this file; skip
                save_npy(_cache_fname, np.asarray(_bias, dtype=float))
                NPY_CACHE.remove(_cache_fname)
                n_cached += 1
        if n_cached:
            print(f'... wrote bias cache for {n_cached} trajectory files')

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
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'+++ created {directory}')
        params = self.params
        states = params.sorted_states
        r = states[1]
        network = params.network
        fit = params.fit
        nbins = params.nbins
        do_tps = params.chain_type == 'tps'
        cutoff_min = params.cutoff_min
        cutoff_max = params.cutoff_max
        terminal_bin_extension = params.terminal_bin_extension
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
        record_bias = params.record_bias
        bias_function = params.bias_function
        bias_source = params.bias_source
        bias_reactive_threshold = params.bias_reactive_threshold

        # get frames in first and last state for training
        margins = get_initial_frames_for_training(self.initial_paths)
        
        # update kwargs to fit function
        kwargs['worker'] = self
        
        # load the network and bins if it is already possible
        print(f'\nLoading pre-existing network parameters {now()}')
        network_fname = f'{directory}/network{states}.h5'
        params.update_network(directory, timeout=0, raise_if_failure=False)
        try:
            bins = NPY_CACHE.load(f'{directory}/bins{states}.h5')
        except:
            bins = None

        # do you need to stop already?
        if self.must_stop:
            self.termination_signal = 2
            return

        # routine checking wether you have to stop
        # while doing so, it updates the loaded path ensemble
        pathensemble = None
        added_frames = 0  # keep counting until reset
        def must_stop():
            nonlocal pathensemble, added_frames
            
            if self.must_stop:
                self.termination_signal = 2
                return True
            
            # get current number of frames
            old_total_frames = self.total_frames
            
            # reset
            total_steps = 0
            total_frames = 0

            # report
            print(f'\nLoading current path ensemble {now()}')

            # get chains
            self._shot_chains = params.shot_chains(
                directory, None,
                old=getattr(self, '_shot_chains', []))
            for chain in self._shot_chains:
                total_frames += sum(chain.n_frames)
                total_steps += len(chain)
            
            # react fast when stop requested
            if self.must_stop:
                self.termination_signal = 2
                return True
            
            # get free trajectories
            self._free_trajectories = params.free_trajectories(directory)
            for trajectory in self._free_trajectories:
                total_frames += trajectory.n_frames

            # update path ensemble
            pathensemble = assemble_pathensemble(
                self._shot_chains,
                self._free_trajectories)

            # report added frames
            new_frames = total_frames - old_total_frames
            if new_frames > 0:
                print(f'... {new_frames} new frames (excluded margins)')
                added_frames += new_frames
            
            # update progress bars
            self.total_steps = total_steps
            self.total_frames = total_frames
            
            # react fast when stop requested
            if self.must_stop:
                self.termination_signal = 2
                return True

        # main cycle
        rounds_done = 0
        while not self.termination_signal:

            # reset added frames counter
            added_frames = 0

            # update current path ensemble and check stop condition
            if must_stop():
                return

            # Recompute any missing descriptor cache files (e.g. after deletion)
            if params.compute_descriptors_args is not None:
                n = pathensemble.compute(*params.compute_descriptors_args)
                if n:
                    print(f'... (re)computed {n} missing descriptor frames')
                # also recompute initial path descriptor frames, if not present
                n = margins.compute(*params.compute_descriptors_args)
                if n:
                    print(f"... (re)computed {n} missing "
                          "inital path descriptor frames")

            # Compute bias potential per frame (only when record_bias is enabled)
            if record_bias and bias_function is not None:
                if bias_source == 'reader':
                    # reader-based: same convention as states_function / descriptors_function
                    print(f'\nComputing bias cache (reader mode) {now()}')
                    n = pathensemble.compute(
                        function=bias_function,
                        target='bias',
                        source='reader',
                        batch_size=batch_size)
                    if n:
                        print(f'... computed {n} bias frames')
                elif bias_source == 'file':
                    # file-based: bias_function(fname) -> full file array
                    self._cache_bias_files(
                        pathensemble, bias_function)

            print(f'\nComputing the committor values of '
                  f'the new reactive {r} frames {now()}')
            n = pathensemble.compute(**compute_kwargs('values'))
            print(f'... computed {n} values')
            
            # check mid-cycle (do not update path ensemble)
            if self.termination_signal:
                return
            
            if rounds_done >= nrounds:
                
                # nothing else to do
                if not keep_running:
                    self.termination_signal = 2
                    return

                # not changing source
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

                # update current path ensemble and check stop condition
                if must_stop():
                    return

                # (re)compute committor values
                if source == 'new':
                    print(f'\nUpdating the committor values of all '
                          f'the reactive {r} frames {now()}')
                    n = pathensemble.compute(**compute_kwargs('new'), overwrite=True)
                else:
                    print(f'\nComputing the committor values of '
                      f'the new reactive {r} frames {now()}')
                    # will fill temp files ("...new.npy") and replace "values.npy" later
                    # in this way, we minimize the risk of i/o issues
                    n = pathensemble.compute(**compute_kwargs('values'))
                print(f'... computed {n} values')
                
                # check mid-cycle (do not update path ensemble)
                if self.termination_signal:
                    return
            
            if source != 'new' and not added_frames:
                # nothing chanced: can wait for the next cycle
                continue
            
            print(f'\nObtaining the adaptation bins {now()}')
            bins = compute_bins(pathensemble, nbins,
                                cutoff_max=cutoff_max,
                                cutoff_min=cutoff_min,
                                find_extremes_with='free',
                                source=source,
                                states=states,
                                terminal_bin_extension=terminal_bin_extension)
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
            print(f'    {lengths.sum()} frames (excluded margins)')

            # Bias-reweighted rates (only when record_bias is enabled)
            if record_bias:
                # Re-cache bias files right before gamma is computed: between
                # round start and here, free trajectories can have grown by
                # hundreds of frames (see `... N new frames` on ensemble reload
                # after training). A stale bias.npy would silently produce a
                # truncated slice and drag gamma toward 1.0.
                if bias_source == 'file' and bias_function is not None:
                    self._cache_bias_files(pathensemble, bias_function)
                from ..pathensemble.bias_utils import (
                    check_reactive_bias, compute_bias_corrections)
                check_reactive_bias(
                    pathensemble, states, bias_reactive_threshold)
                gamma1 = compute_bias_corrections(pathensemble, w1)
                gamma2 = compute_bias_corrections(pathensemble, w2)
                k12_rw = np.sum(w1 * lengths * gamma1)
                k12_rw = 1 / k12_rw if k12_rw else nan
                k21_rw = np.sum(w2 * lengths * gamma2)
                k21_rw = 1 / k21_rw if k21_rw else nan
                print(f'    k12 bias-reweighted: {k12_rw:.3e} [1/dt]')
                print(f'    k21 bias-reweighted: {k21_rw:.3e} [1/dt]')

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

            # check mid-cycle
            if self.termination_signal:
                return
            
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
            print(f'\nSaving selection bins and densities {now()}')
            save_npy(f'{directory}/bins{states}.npy', bins)
            save_npy(f'{directory}/densities{states}.npy', densities)

            # backup network
            n = (self.total_steps // save_interval) * save_interval
            backup = f'{network_fname[:-3]}.step{n:06g}.h5'
            if self.total_steps and not os.path.exists(backup):
                shutil.copyfile(network_fname, backup)
                print(f'*** copied {network_fname!r} to {backup!r}')

    def kinetics_convergence(self, fractions=None,
                             save_file='kinetics_convergence.npy',
                             network_save_pattern=(
                                 '{directory}/network{states}'
                                 '.kcv{fraction_pct:03d}.h5'),
                             **kwargs):
        """
        Assess how kinetics estimates converge with growing training-set size.

        For each requested fraction the path ensemble is sub-sampled (see
        *Fraction sampling* below), the network is retrained from scratch on
        the sub-sample, and the AIMMD reweighting is used to estimate the rate
        constants ``k12`` and ``k21``.  The results are collected, saved, and
        returned as a structured NumPy array so they can immediately be plotted
        or stored for later analysis.

        This is a convenience wrapper around :meth:`Worker.run` with task
        ``'kinetics_convergence'``; it therefore inherits all of the standard
        run-wrapper behaviour (directory switching, cache clearing, resource
        binding, etc.).

        Parameters
        ----------
        fractions : list of float, optional
            Fractions of the full path ensemble to include in each training
            run.  Values must lie in ``(0, 1]``.  The list is de-duplicated
            and sorted in ascending order before use.  Default is
            ``[0.2, 0.4, 0.6, 0.8, 1.0]`` (five 20 %-increment steps).
        save_file : str, optional
            Path (relative to the worker directory) where the result array is
            written as a ``.npy`` file.  Default is
            ``'kinetics_convergence.npy'``.
        network_save_pattern : str or None, optional
            Format string used to derive the filename for the network
            checkpoint saved after each fraction's training run.  The
            following placeholders are available:

            * ``{directory}`` — the worker directory (same as
              :attr:`Worker.directory`).
            * ``{states}`` — the sorted state-label string, e.g. ``'ARB'``
              (same as :attr:`Params.sorted_states`).
            * ``{fraction}`` — the fraction as a float, e.g. ``0.2``.
            * ``{fraction_pct}`` — the fraction as an integer percentage,
              e.g. ``20``.

            The default pattern
            ``'{directory}/network{states}.kcv{fraction_pct:03d}.h5'``
            produces filenames such as
            ``run_example/networkARB.kcv020.h5``.

            Pass ``None`` to skip saving networks entirely.
        **kwargs
            Additional keyword arguments forwarded to the fit routine (e.g.
            ``epochs``, ``lr``, ``batch_size``).  Stop-condition keys
            (``walltime``, ``nsteps``, ``nframes``) are consumed by the run
            wrapper and do not reach the fit routine.

        Returns
        -------
        numpy.ndarray
            Structured array with dtype
            ``[('fraction', float), ('n_frames', float), ('k12', float),
            ('k21', float), ('k12_rw', float), ('k21_rw', float)]`` and one
            row per requested fraction.  ``n_frames`` is the total frame
            count of the sub-sampled ensemble at that fraction (useful for
            converting fraction-of-training to physical sampling time).
            ``k12``/``k21`` hold the uncorrected AIMMD rate estimates in
            units of ``[1/dt]``; ``k12_rw``/``k21_rw`` hold the
            Tiwary-Parrinello bias-reweighted estimates (filled only when
            ``params.record_bias`` is ``True``, left as ``nan`` otherwise).
            Entries for fractions where training failed are also left as
            ``nan``.

        Fraction sampling
        -----------------
        Paths are sub-sampled *per source* so that every shooting chain and
        every free-simulation trajectory contributes the same fraction to the
        training set.  Concretely:

        * For each shooting chain of length ``N``, the first
          ``max(1, round(N * fraction))`` paths are used.
        * For each free trajectory of length ``N`` (frames), the first
          ``max(1, round(N * fraction))`` frames are used.

        Using the *first* paths/frames preserves the temporal ordering of the
        Markov chains, which is the most natural sub-sample for a convergence
        analysis.

        Notes
        -----
        * The fit routine (``params.fit``) always resets the network parameters
          before training (via ``network.reset_parameters()``), so each fraction
          is trained independently from a random initialisation.
        * The current network state is saved before the analysis and restored
          afterwards, leaving the worker in the same state it was in before
          the call.
        * Temporary ``*.kcv.npy`` cache files are written to disk during value
          computation and removed again before the method returns.
        * When ``params.record_bias`` is ``True``, the per-frame bias cache
          is populated for the sub-sampled ensemble before reweighting, the
          reactive-region bias check (:func:`check_reactive_bias`) is applied,
          and the bias-reweighted rates are computed via
          :func:`compute_bias_corrections` and stored in the ``k12_rw`` /
          ``k21_rw`` fields of the result.

        Examples
        --------
        Basic usage after a completed run::

            worker = aimmd.Worker(params, 'run_example')
            results = worker.kinetics_convergence()
            # saves run_example/networkARB.kcv020.h5, .kcv040.h5, … .kcv100.h5
            print(results['fraction'], results['k12'], results['k21'])

        Custom fractions, fewer training epochs, and a custom network naming::

            results = worker.kinetics_convergence(
                fractions=[0.25, 0.5, 0.75, 1.0],
                network_save_pattern=(
                    'checkpoints/network{states}_f{fraction:.2f}.h5'),
                epochs=100,
            )

        Disable per-fraction network saving::

            results = worker.kinetics_convergence(network_save_pattern=None)

        See Also
        --------
        Worker.train : The standard training task.
        """
        return self.run('kinetics_convergence', fractions, save_file,
                        network_save_pattern, **kwargs)

    def _kinetics_convergence(self, fractions=None,
                               save_file='kinetics_convergence.npy',
                               network_save_pattern=(
                                   '{directory}/network{states}'
                                   '.kcv{fraction_pct:03d}.h5'),
                               **kwargs):
        """
        Internal implementation of the ``'kinetics_convergence'`` task.

        See :meth:`kinetics_convergence` for full user-facing documentation.

        Parameters
        ----------
        fractions : list of float or None
            Requested sub-sampling fractions; defaults to
            ``[0.2, 0.4, 0.6, 0.8, 1.0]``.
        save_file : str
            Output filename for the result array.
        network_save_pattern : str or None
            Format string for per-fraction network checkpoint filenames.
            Available placeholders: ``{directory}``, ``{states}``,
            ``{fraction}``, ``{fraction_pct}``.  ``None`` skips saving.
        **kwargs
            Forwarded to ``params.fit``.  ``worker=self`` is injected
            automatically.

        Returns
        -------
        numpy.ndarray
            Structured array with fields ``fraction``, ``k12``, ``k21``.
        """
        # ── defaults ──────────────────────────────────────────────────────
        if fractions is None:
            fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        fractions = sorted(set(float(f) for f in fractions))

        # ── params ────────────────────────────────────────────────────────
        directory = self._directory
        params = self.params
        states = params.sorted_states
        r = states[1]
        network = params.network
        fit = params.fit
        reweight_parameters = params.reweight_parameters
        nbins = params.nbins
        cutoff_min = params.cutoff_min
        cutoff_max = params.cutoff_max
        terminal_bin_extension = params.terminal_bin_extension
        batch_size = params.network_batch_size
        compute_values_args = params.compute_values_args
        compute_condition = {'states': lambda state: state == r}
        record_bias = params.record_bias
        bias_function = params.bias_function
        bias_source = params.bias_source
        bias_reactive_threshold = params.bias_reactive_threshold

        def _compute_kwargs(target):
            return {
                'function': compute_values_args[0],
                'target': target,
                'source': compute_values_args[2],
                'conditions': compute_condition,
                'batch_size': batch_size,
            }

        kwargs['worker'] = self

        # ── save network state (restored at the end) ──────────────────────
        print(f'\nSaving current network state {now()}')
        device = next(network.parameters()).device
        _buf = io.BytesIO()
        torch.save(network.state_dict(), _buf)
        _saved_state = _buf.getvalue()

        # ── load full path ensemble from disk ─────────────────────────────
        print(f'\nLoading full path ensemble {now()}')
        self._shot_chains = params.shot_chains(
            directory, None,
            old=getattr(self, '_shot_chains', []))
        self._free_trajectories = params.free_trajectories(directory)

        # ── build margins from initial paths (same as _train) ─────────────
        margins = get_initial_frames_for_training(self.initial_paths)

        # ── pre-compute sizes for fraction arithmetic ──────────────────────
        chain_lengths = [len(c) for c in self._shot_chains]
        free_lengths  = [len(t) for t in self._free_trajectories]

        # ── result container ───────────────────────────────────────────────
        # k12_rw/k21_rw are filled only when params.record_bias=True; they stay
        # nan otherwise. Keeping the fields always-present means downstream
        # code can load any kinetics_convergence.npy without branching.
        # n_frames is the total frame count of the sub-sampled ensemble at
        # each fraction; downstream analyses use it to convert fraction-of-
        # training to physical sampling time.
        results = np.full(
            len(fractions), nan,
            dtype=[('fraction', float),
                   ('n_frames', float),
                   ('k12', float), ('k21', float),
                   ('k12_rw', float), ('k21_rw', float)])
        results['fraction'] = fractions

        # ── main loop over fractions ───────────────────────────────────────
        for i, fraction in enumerate(fractions):
            print(f'\n=== Kinetics convergence: fraction {fraction:.2f} ==='
                  f' ({now()})')

            if self.termination_signal:
                break

            # sub-sample chains: first round(N * fraction) paths per chain
            sub_chains = [
                chain[:max(1, round(n * fraction))]
                for chain, n in zip(self._shot_chains, chain_lengths)]

            # sub-sample free trajectories: first round(N * fraction) frames
            sub_free = [
                traj[:max(1, round(n * fraction))]
                for traj, n in zip(self._free_trajectories, free_lengths)]

            pathensemble = assemble_pathensemble(sub_chains, sub_free)

            # Recompute any missing descriptor cache files before computing
            # values — mirrors the equivalent provision in _train so that the
            # convergence loop is robust to missing/deleted descriptor files.
            if params.compute_descriptors_args is not None:
                n = pathensemble.compute(*params.compute_descriptors_args)
                if n:
                    print(f'... (re)computed {n} missing descriptor frames')
                n = margins.compute(*params.compute_descriptors_args)
                if n:
                    print(f'... (re)computed {n} missing initial path '
                          f'descriptor frames')

            # ensure existing value files are present (fill missing only)
            n = pathensemble.compute(**_compute_kwargs('values'))
            if n:
                print(f'... computed {n} missing values')

            if self.termination_signal:
                break

            # train network from scratch on the sub-sampled ensemble
            print(f'Training network on {fraction*100:.0f}% of data '
                  f'{now()}')
            losses, *_ = fit(params, pathensemble + margins, **kwargs)

            if not len(losses):
                print(f'!!! training failed for fraction {fraction:.2f}')
                continue

            print(f'Training completed {now()}')

            if self.termination_signal:
                break

            # compute updated values using 'kcv' target to avoid overwriting
            # production value files on disk
            n = pathensemble.compute(
                **_compute_kwargs('kcv'), overwrite=True)
            print(f'... computed {n} committor values with new network')

            # compute adaptation bins → derive sp_cutoff_min/max
            rw_p = reweight_parameters.copy()
            bins = compute_bins(
                pathensemble, nbins,
                cutoff_max=cutoff_max,
                cutoff_min=cutoff_min,
                find_extremes_with='free',
                source='kcv',
                states=states,
                terminal_bin_extension=terminal_bin_extension)

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

            # populate per-frame bias cache for the sub-sampled ensemble so
            # `compute_bias_corrections` below can compute γ = <exp(bias)> per
            # path. Mirrors the cache step in `_train` (see L341-L356, L482-L488).
            if record_bias and bias_function is not None:
                if bias_source == 'reader':
                    n = pathensemble.compute(
                        function=bias_function,
                        target='bias',
                        source='reader',
                        batch_size=batch_size)
                    if n:
                        print(f'... computed {n} bias frames')
                elif bias_source == 'file':
                    self._cache_bias_files(pathensemble, bias_function)

            # reweight and estimate rates
            result1 = pathensemble.reweight(states, **rw_p, source='kcv')
            result2 = pathensemble.reweight(states[::-1], **rw_p,
                                            source='kcv')
            w1 = result1[0]
            w2 = result2[0]

            lengths = pathensemble.n_frames
            k12 = np.sum(w1 * lengths)
            k12 = 1 / k12 if k12 else nan
            k21 = np.sum(w2 * lengths)
            k21 = 1 / k21 if k21 else nan

            print(f'    k12 estimate: {k12:.3e} [1/dt]')
            print(f'    k21 estimate: {k21:.3e} [1/dt]')
            print(f'    {lengths.sum()} frames in sub-sampled ensemble')

            results['n_frames'][i] = float(lengths.sum())
            results['k12'][i] = k12
            results['k21'][i] = k21

            # Bias-reweighted rates (Tiwary-Parrinello) — only when record_bias
            # is enabled. Same formula as `_train` at L480-L500.
            if record_bias:
                from ..pathensemble.bias_utils import (
                    check_reactive_bias, compute_bias_corrections)
                check_reactive_bias(
                    pathensemble, states, bias_reactive_threshold)
                gamma1 = compute_bias_corrections(pathensemble, w1)
                gamma2 = compute_bias_corrections(pathensemble, w2)
                k12_rw = np.sum(w1 * lengths * gamma1)
                k12_rw = 1 / k12_rw if k12_rw else nan
                k21_rw = np.sum(w2 * lengths * gamma2)
                k21_rw = 1 / k21_rw if k21_rw else nan
                print(f'    k12 bias-reweighted: {k12_rw:.3e} [1/dt]')
                print(f'    k21 bias-reweighted: {k21_rw:.3e} [1/dt]')
                results['k12_rw'][i] = k12_rw
                results['k21_rw'][i] = k21_rw

            # save per-fraction network checkpoint (if requested)
            if network_save_pattern is not None:
                net_fname = network_save_pattern.format(
                    directory=directory,
                    states=states,
                    fraction=fraction,
                    fraction_pct=round(fraction * 100),
                )
                os.makedirs(os.path.dirname(net_fname) or '.', exist_ok=True)
                torch.save(network.state_dict(), net_fname)
                print(f'    saved network to {net_fname!r}')

            # clean up temporary kcv cache files so they do not clutter disk
            for fname in set(pathensemble.fnames):
                kcv_fname = get_cache_fname(fname, 'kcv')
                NPY_CACHE.remove(kcv_fname)
                if os.path.exists(kcv_fname):
                    try:
                        os.remove(kcv_fname)
                    except OSError:
                        pass

        # ── restore original trained network ──────────────────────────────
        print(f'\nRestoring original network state {now()}')
        _buf = io.BytesIO(_saved_state)
        network.load_state_dict(torch.load(_buf, map_location=device,
                                           weights_only=True))

        # ── save and return results ────────────────────────────────────────
        np.save(save_file, results)
        print(f'Saved kinetics convergence results to {save_file!r}')

        return results
