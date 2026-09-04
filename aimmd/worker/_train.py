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
from ..core.utils import now, replace_in_cache, accepts_system_id


def _graph_cache_line():
    """One-line graph-cache counters, for spotting a stalled/ineffective cache.

    ``shm`` is what the replicas actually occupy; ``hit``/``miss`` are replica
    lookups and ``memo`` in-process ones, so a healthy trainer shows hits and
    memo climbing and misses flat. All-misses means the replica is absent or
    stale and every lookup is going to shared storage.
    """
    try:
        from ..network import shm_cache as _sc
        st = _sc.replica_stats()
        return (f"graph cache hit={st.get('hits', 0):,} "
                f"miss={st.get('misses', 0):,} memo={st.get('memo_hits', 0):,} "
                f"shm={st.get('staged_bytes', 0) / 1e9:.1f}GB")
    except Exception:                                          # noqa: BLE001
        return 'graph cache stats unavailable'
from ..pathensemble import PathEnsemble
from ..analysis.utils import compute_bins
from ..pathensemble.utils import assemble_pathensemble
from ..network.rescale_utils import find_knots_and_values, rescale
from ..network import shm_cache
from ..path.utils import get_cache_fname

# WorkerTrain mixin class
class WorkerTrain(ABC):

    def _cache_bias_files(self, pathensemble, bias_function, system_id=None):
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
        system_id : hashable or None
            In a multi-system run, the id of the system whose trajectories are
            being cached. It is forwarded to ``bias_function(fname,
            system_id=...)`` only when the function's signature accepts it (so a
            single file-driven bias function that ignores it keeps working).

        Notes
        -----
        Prints ``Computing bias cache (file mode)`` at invocation and
        ``... wrote bias cache for N trajectory files`` when any rewrite
        happened.
        """
        print(f'\nComputing bias cache (file mode) {now()}')
        from ..pathensemble.bias_utils import derive_bias_from_cumulative_colvar
        pass_sid = system_id is not None and accepts_system_id(bias_function)
        seen = set()
        n_cached = 0
        n_derived = 0
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
                _call = ((lambda f: bias_function(f, system_id=system_id))
                         if pass_sid else bias_function)
                _bias = _call(_fname)
                if _bias is None or len(_bias) < _n_traj_frames:
                    # No per-part _COLVAR slice yet (the free segment's mdrun has
                    # not returned), or it is short. Derive the bias straight from
                    # the trajectory's cumulative COLVAR instead — read-only, and
                    # without touching the slicing machinery.
                    _derived = derive_bias_from_cumulative_colvar(
                        _fname, self.params.trajectory_extension, _call)
                    if _derived is not None and (
                            _bias is None or len(_derived) > len(_bias)):
                        _bias = _derived
                        n_derived += 1
                if _bias is None:
                    continue  # no COLVAR for this file; skip
                save_npy(_cache_fname, np.asarray(_bias, dtype=float))
                NPY_CACHE.remove(_cache_fname)
                n_cached += 1
        if n_cached:
            print(f'... wrote bias cache for {n_cached} trajectory files'
                  + (f' ({n_derived} derived out-of-cache from the cumulative '
                     f'COLVAR)' if n_derived else ''))

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

        Multi-system runs (``params.multi_system``) are dispatched to
        :meth:`_train_multi_system`; single-system behaviour below is unchanged.
        """
        if getattr(self.params, 'multi_system', False):
            return self._train_multi_system(nrounds, keep_running, **kwargs)

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

            # Make the graph cache(s) available in node-local RAM for this cycle.
            # After must_stop(), so a trainer about to exit does not pay for a
            # copy it will never read; inside the loop, because the writers keep
            # appending and a once-per-process replica would go stale.
            # The trainer only READS the shared graph cache: anything it has to
            # compute stays in its memo and its own tmpfs replica, so it never
            # contends with the ~35 MD writers for SQLite's single write lock.
            shm_cache.set_reader_role()
            shm_cache.stage_replicas()

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

            # Value-pass subsample: bounds the (growing) committor value pass and
            # the bins/reweighting that consume it by evaluating them on a random
            # subsample of the ensemble. Identity (the full ensemble, same object)
            # when no caps are configured -> behaviour is unchanged. `fit` always
            # uses the full `pathensemble`. Rebuilt again after training below.
            caps = params.subsample_caps_of(getattr(self, '_system_id', None))
            eval_pe = pathensemble.subsample(caps, states) if caps else pathensemble
            if caps:
                print(f'... value-pass subsample: {len(eval_pe)}/'
                      f'{len(pathensemble)} paths')

            # Compute bias potential per frame (only when record_bias is enabled)
            if record_bias and bias_function is not None:
                if bias_source == 'reader':
                    # reader-based: same convention as states_function / descriptors_function
                    print(f'\nComputing bias cache (reader mode) {now()}')
                    n = eval_pe.compute(
                        function=bias_function,
                        target='bias',
                        source='reader',
                        batch_size=batch_size)
                    if n:
                        print(f'... computed {n} bias frames')
                elif bias_source == 'file':
                    # file-based: bias_function(fname) -> full file array
                    self._cache_bias_files(
                        eval_pe, bias_function)

            print(f'\nComputing the committor values of '
                  f'the new reactive {r} frames {now()}')
            n = eval_pe.compute(**compute_kwargs('values'))
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

                # Persist the freshly trained network immediately, before the
                # (potentially long) value pass + reweighting below. A crash or
                # walltime kill during that downstream work then still leaves the
                # trained model — and its step backup — on disk.
                if source == 'new':
                    print(f'\nSaving network parameters to '
                          f'{network_fname} {now()}')
                    torch.save(network.state_dict(), network_fname)
                    n = (self.total_steps // save_interval) * save_interval
                    backup = f'{network_fname[:-3]}.step{n:06g}.h5'
                    if self.total_steps and not os.path.exists(backup):
                        shutil.copyfile(network_fname, backup)
                        print(f'*** copied {network_fname!r} to {backup!r}')

                # rebuild the eval subsample (frames may have grown during fit)
                eval_pe = (pathensemble.subsample(caps, states) if caps
                           else pathensemble)

                # fit ran for a long time and the writers kept appending, so top
                # the replica up before the re-score reads every frame again
                shm_cache.refresh_replicas()

                # (re)compute committor values
                if source == 'new':
                    print(f'\nUpdating the committor values of all '
                          f'the reactive {r} frames {now()}')
                    n = eval_pe.compute(**compute_kwargs('new'), overwrite=True)
                else:
                    print(f'\nComputing the committor values of '
                      f'the new reactive {r} frames {now()}')
                    # will fill temp files ("...new.npy") and replace "values.npy" later
                    # in this way, we minimize the risk of i/o issues
                    n = eval_pe.compute(**compute_kwargs('values'))
                print(f'... computed {n} values')
                
                # check mid-cycle (do not update path ensemble)
                if self.termination_signal:
                    return
            
            if source != 'new' and not added_frames:
                # nothing chanced: can wait for the next cycle
                continue
            
            print(f'\nObtaining the adaptation bins {now()}')
            bins = compute_bins(eval_pe, nbins,
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
                weights = (eval_pe.weights *
                           eval_pe.are_transitions(states))

            # reweight pathensemble
            print(f'\nReweighting the full path ensemble {now()}')

            # will update reweighting parameters
            rw_p = reweight_parameters.copy()

            # find sp_cutoff_min and sp_cutoff_max (nbins<=0 -> no adaptive bins;
            # compute_bins returns [] and reweight then uses its own defaults)
            if len(bins):
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
            result1 = eval_pe.reweight(
                states, **rw_p, source=source)
            result2 = eval_pe.reweight(
                states[::-1], **rw_p, source=source)
            w1, extremes1, xP1 = result1[0], result1[4], result1[5]
            w2, extremes2, xP2 = result2[0], result2[4], result2[5]

            # bonus track: estimate rates
            lengths = eval_pe.n_frames
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
                    self._cache_bias_files(eval_pe, bias_function)
                from ..pathensemble.bias_utils import bias_reweighted_rates
                k12_rw, k21_rw, gamma1, gamma2 = bias_reweighted_rates(
                    eval_pe, w1, w2, lengths=lengths, states=states,
                    reactive_threshold=bias_reactive_threshold)

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
                    eval_pe.compute(
                        lambda x: rescale(x, knots, values),
                        'new', 'new', compute_condition,
                        overwrite=True, worker=self)
                    time.sleep(.1)  # stability

                # rescaling modified the network in place -> re-save so the file
                # reflects the rescaled model (the immediate post-training save
                # above predates this adjustment).
                print(f'\nRe-saving rescaled network parameters to '
                      f'{network_fname} {now()}')
                torch.save(network.state_dict(), network_fname)

            # check mid-cycle
            if self.termination_signal:
                break

            # adaptive bins / densities drive shooting-point selection; only
            # meaningful when there are adaptive bins (nbins > 0). The projection
            # reads the source='new' committor values, so it must run BEFORE the
            # replace_in_cache below promotes '...new.npy' to '...values.npy'.
            if len(bins):
                # assign weights
                if not do_tps:  # PE weights in reactive region
                    excursions_mask = eval_pe.types(f'.{r}..')
                    eval_pe.weights = (w1 + w2) * excursions_mask
                else:  # TPE weights
                    eval_pe.weights *= eval_pe.are_transitions(states)

                print(f'\nProjecting the {"T" * do_tps}PE density {now()}')
                densities = eval_pe.project(bins, source=source)
                densities[densities == 0.] = 1e-15
                densities /= densities.sum()
                print(f'    densities: {densities}')

            if source == 'new':
                # replace values (as much as possible) all at once. The network
                # itself was already saved right after training (see above).
                print(f'\nSubstituting \'...values.npy\' files '
                      f'with \'...new.npy\' {now()}')
                replace_in_cache(NPY_CACHE, '.new.npy', '.values.npy',
                                 set(eval_pe.fnames))

            # save bins and densities
            if len(bins):
                print(f'\nSaving bins and densities {now()}')
                save_npy(f'{directory}/bins{states}.npy', bins)
                save_npy(f'{directory}/densities{states}.npy', densities)

    def _multi_system_layout(self):
        """Resolve the (subdirectory, system_id) list this trainer serves.

        - shared network ON : one trainer at the run root serves ALL systems;
          returns ``[(<root>/<sid>, sid), ...]`` for every ``system_id``.
        - shared network OFF: one trainer per system, each launched with its own
          subfolder as ``directory``; returns ``[(<subfolder>, <subfolder name>)]``.
        """
        params = self.params
        directory = self._directory
        if params.multi_system_share_network:
            sysids = list(params.system_ids)
            subdirs = [f'{directory}/{sid}' for sid in sysids]
        else:
            sysids = [os.path.basename(os.path.normpath(directory))]
            subdirs = [directory]
        return list(zip(subdirs, sysids))

    def _train_multi_system(self, nrounds=inf, keep_running=False, **kwargs):
        """Training task for multi-system (multi-ligand) runs.

        Mirrors :meth:`_train` but over a list of systems. With a shared network
        the per-system PathEnsembles are pooled and the params ``fit`` function
        receives a LIST (balanced inside :func:`aimmd.network.fit.fit`); the one
        shared network is saved at the run root. Without a shared network this
        trainer serves a single system's subfolder and trains its own network.
        Either way, committor values, adaptation bins/densities and rate
        estimates are computed PER system, in sequence, and written into each
        system's own subfolder.

        When ``params.record_bias`` is set the per-frame bias cache is built per
        system (reader or file mode, with ``system_id`` forwarded), the
        reactive-region bias check uses each system's
        ``bias_reactive_threshold_of(system_id)``, and per-system Tiwary-Parrinello
        bias-reweighted rates are printed alongside the raw ones.

        Limitations (raise clearly): ``chain_type='tps'`` and
        ``rescale_committor`` are not yet supported together with multi_system.
        """
        nrounds = float(nrounds)
        directory = self._directory
        params = self.params
        states = params.sorted_states
        r = states[1]
        network = params.network
        fit = params.fit
        nbins = params.nbins
        cutoff_min = params.cutoff_min
        cutoff_max = params.cutoff_max
        terminal_bin_extension = params.terminal_bin_extension
        batch_size = params.network_batch_size
        reweight_parameters = params.reweight_parameters
        compute_values_args = params.compute_values_args
        save_interval = params.network_save_interval
        share = bool(params.multi_system_share_network)
        record_bias = params.record_bias
        bias_function = params.bias_function
        bias_source = params.bias_source

        if params.chain_type == 'tps':
            raise NotImplementedError(
                "multi_system training currently supports chain_type='rfps' only")
        if params.rescale_committor:
            raise NotImplementedError(
                "rescale_committor is not yet supported with multi_system")

        systems = self._multi_system_layout()
        compute_condition = {'states': lambda state: state == r}

        def values_kwargs(target, system_id):
            return {'function': compute_values_args[0], 'target': target,
                    'source': compute_values_args[2],
                    'conditions': compute_condition,
                    'batch_size': batch_size, 'system_id': system_id}

        def cache_bias(pe, system_id):
            """Per-system bias cache (mirror of the single-system `_train`)."""
            if not (record_bias and bias_function is not None):
                return
            if bias_source == 'reader':
                pe.compute(function=bias_function, target='bias',
                           source='reader', batch_size=batch_size,
                           system_id=system_id)
            elif bias_source == 'file':
                self._cache_bias_files(pe, bias_function, system_id=system_id)

        # per-system margin frames (transitions only), from each subfolder's
        # initial paths
        margins = []
        for subdir, sid in systems:
            ip = PathEnsemble(f'{subdir}/initial{states}/*')
            ip = ip.extract(states, states[::-1])
            margins.append(PathEnsemble([p[1::-1] for p in ip] +
                                        [p[-2::1] for p in ip]))

        kwargs['worker'] = self

        # load shared/own network if available (resolves root vs subfolder)
        print(f'\nLoading pre-existing network parameters {now()}')
        network_fname = params._network_fname(directory)
        params.update_network(directory, timeout=0, raise_if_failure=False)
        if self.must_stop:
            self.termination_signal = 2
            return

        pathensembles = [None] * len(systems)

        # Offer each system's Path objects back to `shot_chains` next reload.
        # `shot_paths` matches on filename and returns the *existing* object, so
        # nothing is re-read from disk; without this every Path is rebuilt, and
        # `Path(fname, shooting_index='find')` resolves to `min_length=inf`,
        # which makes the MDA reader cache a guaranteed miss and re-walks the
        # whole XTC. The single-system trainer has always done this via
        # `self._shot_chains`. Kept per system: a pooled `old` would be correct
        # but would make the linear scan inside `shot_paths` O(total^2).
        shot_chains_by_system = getattr(self, '_shot_chains_by_system', None)
        if (shot_chains_by_system is None
                or len(shot_chains_by_system) != len(systems)):
            shot_chains_by_system = [[] for _ in systems]
        self._shot_chains_by_system = shot_chains_by_system

        def must_stop():
            nonlocal pathensembles
            if self.must_stop:
                self.termination_signal = 2
                return True
            print(f'\nLoading current path ensembles {now()}')
            total_steps = total_frames = 0
            for k, (subdir, sid) in enumerate(systems):
                chains = params.shot_chains(
                    subdir, None, old=shot_chains_by_system[k])
                shot_chains_by_system[k] = chains
                frees = params.free_trajectories(subdir)
                for chain in chains:
                    total_frames += sum(chain.n_frames)
                    total_steps += len(chain)
                for trajectory in frees:
                    total_frames += trajectory.n_frames
                pathensembles[k] = assemble_pathensemble(chains, frees)
                if self.must_stop:
                    self.termination_signal = 2
                    return True
            self.total_steps = total_steps
            self.total_frames = total_frames
            return False

        def make_eval_pes():
            """Per-system path ensembles for the value pass / bins / reweighting.

            With ``params.subsample_caps`` configured these are bounded random
            subsamples (drawn fresh each round); otherwise they are the full
            ensembles (identity -> unchanged behaviour). ``fit`` always uses the
            full ensembles.
            """
            out = []
            for kk, (subdir_, sid_) in enumerate(systems):
                caps = params.subsample_caps_of(sid_)
                if caps:
                    ep = pathensembles[kk].subsample(caps, states)
                    print(f'... [system {sid_!r}] value-pass subsample: '
                          f'{len(ep)}/{len(pathensembles[kk])} paths')
                    out.append(ep)
                else:
                    out.append(pathensembles[kk])
            return out

        rounds_done = 0
        while not self.termination_signal:

            if must_stop():
                return

            # see the equivalent call in _train()
            # The trainer only READS the shared graph cache: anything it has to
            # compute stays in its memo and its own tmpfs replica, so it never
            # contends with the ~35 MD writers for SQLite's single write lock.
            shm_cache.set_reader_role()
            shm_cache.stage_replicas()

            # (re)compute descriptors (full ensemble, for fit) + committor values
            # (on the possibly-subsampled eval ensemble, to bound the value pass)
            #
            # Deliberately chatty: this is the phase that stalled in production
            # (staging is interleaved here -- each system's replica is copied on
            # its first cache lookup -- so a missing "staged ..." line pins the
            # stall to a system, and the per-step timings and cache counters say
            # whether it is descriptor compute, the value pass, or cache I/O).
            eval_pes = make_eval_pes()
            print(f'\nValue pass over {len(systems)} system(s) {now()}')
            for k, (subdir, sid) in enumerate(systems):
                if params.compute_descriptors_args is not None:
                    _t0 = time.time()
                    print(f"... [system {sid!r}] descriptors: computing over "
                          f"{len(pathensembles[k])} paths {now()}")
                    n_desc = pathensembles[k].compute(
                        *params.compute_descriptors_args, system_id=sid)
                    print(f"... [system {sid!r}] descriptors: {n_desc} frame(s) "
                          f"computed in {time.time() - _t0:.1f}s "
                          f"[{_graph_cache_line()}]")
                cache_bias(eval_pes[k], sid)
                _t0 = time.time()
                print(f"... [system {sid!r}] value pass: {len(eval_pes[k])} "
                      f"paths {now()}")
                n_val = eval_pes[k].compute(**values_kwargs('values', sid))
                print(f"... [system {sid!r}] value pass: {n_val} frame(s) "
                      f"in {time.time() - _t0:.1f}s [{_graph_cache_line()}]")
            print(f'Value pass complete {now()}')
            if self.termination_signal:
                return

            if rounds_done >= nrounds:
                if not keep_running:
                    self.termination_signal = 2
                    return
                source = 'values'
            else:
                # one fit call: a LIST of PEs when sharing the network, else a
                # single PE (the existing default fit handles both).
                print(f'\nTraining the network '
                      f'(round {rounds_done + 1}, {now()})')
                if share:
                    fit_input = [pe + margin
                                 for pe, margin in zip(pathensembles, margins)]
                else:
                    fit_input = pathensembles[0] + margins[0]
                losses, *_ = fit(params, fit_input, **kwargs)
                if len(losses):
                    source = 'new'
                    rounds_done += 1
                    print(f'*** training completed {now()}')
                    if self.termination_signal:
                        break
                else:
                    print(f'!!! training failed, reloading {now()}')
                    params.update_network(
                        directory, timeout=0, raise_if_failure=False)
                    source = 'values'
                if must_stop():
                    return
                # Persist the freshly trained (shared) network immediately, before
                # the per-system value pass + reweighting below. A crash or
                # walltime kill during that downstream work then still leaves the
                # trained model — and its step backup — on disk.
                if source == 'new':
                    print(f'\nSaving network parameters to {network_fname} {now()}')
                    torch.save(network.state_dict(), network_fname)
                    n = (self.total_steps // save_interval) * save_interval
                    backup = f'{network_fname[:-3]}.step{n:06g}.h5'
                    if self.total_steps and not os.path.exists(backup):
                        shutil.copyfile(network_fname, backup)
                        print(f'*** copied {network_fname!r} to {backup!r}')
                # rebuild the eval ensembles (frames may have grown during fit)
                # and refresh committor values with the new network
                shm_cache.refresh_replicas()
                eval_pes = make_eval_pes()
                for k, (subdir, sid) in enumerate(systems):
                    pe = eval_pes[k]
                    target = 'new' if source == 'new' else 'values'
                    cache_bias(pe, sid)
                    pe.compute(**values_kwargs(target, sid),
                               overwrite=(source == 'new'))
                if self.termination_signal:
                    return

            # per-system bins / densities / rates (in sequence) + persist.
            # These run on the (possibly subsampled) eval ensemble.
            for k, (subdir, sid) in enumerate(systems):
                pe = eval_pes[k]
                print(f'\n[system {sid!r}] adaptation bins {now()}')
                bins = compute_bins(pe, nbins, cutoff_max=cutoff_max,
                                    cutoff_min=cutoff_min,
                                    find_extremes_with='free', source=source,
                                    states=states,
                                    terminal_bin_extension=terminal_bin_extension)
                rw_p = reweight_parameters.copy()
                # nbins<=0 -> compute_bins returns [] (no adaptive bins); skip the
                # shooting-point cutoff setup (reweight then uses its defaults).
                if len(bins):
                    sp_cutoff_min, sp_cutoff_max = bins[0], bins[-1]
                    if sp_cutoff_min == -inf and bins[1] < +inf:
                        sp_cutoff_min = bins[1]
                    if sp_cutoff_max == +inf and bins[-2] > -inf:
                        sp_cutoff_max = bins[-2]
                    rw_p.setdefault('sp_cutoff_min', sp_cutoff_min)
                    rw_p.setdefault('sp_cutoff_max', sp_cutoff_max)
                w1 = pe.reweight(states, **rw_p, source=source)[0]
                w2 = pe.reweight(states[::-1], **rw_p, source=source)[0]
                frame_lengths = pe.n_frames
                k12 = np.sum(w1 * frame_lengths)
                k12 = 1 / k12 if k12 else nan
                k21 = np.sum(w2 * frame_lengths)
                k21 = 1 / k21 if k21 else nan
                print(f'    [system {sid!r}] k12 estimate: {k12:.3e} [1/dt]')
                print(f'    [system {sid!r}] k21 estimate: {k21:.3e} [1/dt]')

                # Bias-reweighted rates (Tiwary-Parrinello), per system
                if record_bias:
                    if bias_source == 'file' and bias_function is not None:
                        self._cache_bias_files(pe, bias_function, system_id=sid)
                    from ..pathensemble.bias_utils import (
                        bias_reweighted_rates)
                    k12_rw, k21_rw, gamma1, gamma2 = bias_reweighted_rates(
                        pe, w1, w2, lengths=frame_lengths, states=states,
                        reactive_threshold=(
                            params.bias_reactive_threshold_of(sid)),
                        label=f'[system {sid!r}] ')

                # densities/bins drive adaptive shooting-point selection; only
                # meaningful when there are adaptive bins (nbins > 0). The
                # projection reads source='new' values, so it must run BEFORE
                # replace_in_cache promotes '...new.npy' to '...values.npy'.
                if len(bins):
                    excursions_mask = pe.types(f'.{r}..')
                    pe.weights = (w1 + w2) * excursions_mask
                    densities = pe.project(bins, source=source)
                    densities[densities == 0.] = 1e-15
                    densities /= densities.sum()

                if source == 'new':
                    replace_in_cache(NPY_CACHE, '.new.npy', '.values.npy',
                                     set(pe.fnames))
                if len(bins):
                    save_npy(f'{subdir}/bins{states}.npy', bins)
                    save_npy(f'{subdir}/densities{states}.npy', densities)

            # (the shared/own network was already saved right after training,
            # before the per-system value pass / reweighting above)

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
        if getattr(self.params, 'multi_system', False):
            return self._kinetics_convergence_multi_system(
                fractions, save_file, network_save_pattern, **kwargs)

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
                from ..pathensemble.bias_utils import bias_reweighted_rates
                k12_rw, k21_rw, gamma1, gamma2 = bias_reweighted_rates(
                    pathensemble, w1, w2, lengths=lengths, states=states,
                    reactive_threshold=bias_reactive_threshold)
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

    def _kinetics_convergence_multi_system(self, fractions=None,
                                           save_file='kinetics_convergence.npy',
                                           network_save_pattern=(
                                               '{directory}/network{states}'
                                               '.kcv{fraction_pct:03d}.h5'),
                                           **kwargs):
        """Kinetics-convergence for a SHARED multi-system network.

        At each fraction the per-system PathEnsembles are sub-sampled, the one
        shared network is retrained on the LIST of sub-sampled ensembles, and
        rates are estimated **per system, in sequence**. The result is a flat
        structured array with an explicit ``system`` field (one row per
        (fraction, system)), saved at the run root.

        Not-shared multi-system runs should analyze each system's subfolder with
        the ordinary single-system kinetics convergence instead.

        When ``params.record_bias`` is set, the per-system bias cache is built and
        the ``k12_rw`` / ``k21_rw`` columns are filled with the per-system
        Tiwary-Parrinello bias-reweighted estimates (left ``nan`` otherwise).
        """
        params = self.params
        if not params.multi_system_share_network:
            raise NotImplementedError(
                'multi-system kinetics_convergence is implemented for a shared '
                'network; for separate networks run the single-system '
                'kinetics_convergence per system subfolder')
        if params.rescale_committor:
            raise NotImplementedError(
                'rescale_committor not supported with '
                'multi-system kinetics_convergence')
        if fractions is None:
            fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        fractions = sorted(set(float(f) for f in fractions))

        directory = self._directory
        states = params.sorted_states
        r = states[1]
        network = params.network
        fit = params.fit
        reweight_parameters = params.reweight_parameters
        nbins = params.nbins
        cutoff_min, cutoff_max = params.cutoff_min, params.cutoff_max
        terminal_bin_extension = params.terminal_bin_extension
        batch_size = params.network_batch_size
        compute_values_args = params.compute_values_args
        compute_condition = {'states': lambda state: state == r}
        record_bias = params.record_bias
        bias_function = params.bias_function
        bias_source = params.bias_source
        kwargs['worker'] = self
        systems = self._multi_system_layout()

        def values_kwargs(target, system_id):
            return {'function': compute_values_args[0], 'target': target,
                    'source': compute_values_args[2],
                    'conditions': compute_condition,
                    'batch_size': batch_size, 'system_id': system_id}

        def cache_bias(pe, system_id):
            """Per-system bias cache (mirror of the single-system `_train`)."""
            if not (record_bias and bias_function is not None):
                return
            if bias_source == 'reader':
                pe.compute(function=bias_function, target='bias',
                           source='reader', batch_size=batch_size,
                           system_id=system_id)
            elif bias_source == 'file':
                self._cache_bias_files(pe, bias_function, system_id=system_id)

        # save / restore the trained shared network around the experiment
        print(f'\nSaving current network state {now()}')
        device = next(network.parameters()).device
        _buf = io.BytesIO()
        torch.save(network.state_dict(), _buf)
        _saved_state = _buf.getvalue()

        # per-system full chains/free + margins
        sys_chains, sys_free, margins = [], [], []
        _reusable = getattr(self, '_shot_chains_by_system', None) or []
        for _k, (subdir, sid) in enumerate(systems):
            # One-shot: this loop runs once, so reuse saves a single full
            # rescan rather than one per round. Cheap, but not the same win.
            chains = params.shot_chains(
                subdir, None,
                old=_reusable[_k] if _k < len(_reusable) else [])
            frees = params.free_trajectories(subdir)
            sys_chains.append(chains)
            sys_free.append(frees)
            ip = PathEnsemble(f'{subdir}/initial{states}/*')
            ip = ip.extract(states, states[::-1])
            margins.append(PathEnsemble([p[1::-1] for p in ip] +
                                        [p[-2::1] for p in ip]))

        results = np.full(
            len(fractions) * len(systems), nan,
            dtype=[('fraction', float), ('system', 'U32'),
                   ('n_frames', float), ('k12', float), ('k21', float),
                   ('k12_rw', float), ('k21_rw', float)])

        row = 0
        for fraction in fractions:
            print(f'\n=== Kinetics convergence: fraction {fraction:.2f} ==='
                  f' ({now()})')
            if self.termination_signal:
                break
            # sub-sample each system and (re)compute its values
            sub_pes = []
            for k, (subdir, sid) in enumerate(systems):
                sub_chains = [c[:max(1, round(len(c) * fraction))]
                              for c in sys_chains[k]]
                sub_free = [t[:max(1, round(len(t) * fraction))]
                            for t in sys_free[k]]
                pe = assemble_pathensemble(sub_chains, sub_free)
                if params.compute_descriptors_args is not None:
                    pe.compute(*params.compute_descriptors_args, system_id=sid)
                cache_bias(pe, sid)
                pe.compute(**values_kwargs('values', sid))
                sub_pes.append(pe)
            if self.termination_signal:
                break

            print(f'Training shared network on {fraction*100:.0f}% of data '
                  f'{now()}')
            losses, *_ = fit(params, [pe + margin for pe, margin
                                      in zip(sub_pes, margins)], **kwargs)
            if not len(losses):
                print(f'!!! training failed for fraction {fraction:.2f}')
                row += len(systems)
                continue
            if self.termination_signal:
                break

            # per-system rates with the freshly trained shared network
            for k, (subdir, sid) in enumerate(systems):
                pe = sub_pes[k]
                pe.compute(**values_kwargs('kcv', sid), overwrite=True)
                rw_p = reweight_parameters.copy()
                bins = compute_bins(pe, nbins, cutoff_max=cutoff_max,
                                    cutoff_min=cutoff_min,
                                    find_extremes_with='free', source='kcv',
                                    states=states,
                                    terminal_bin_extension=terminal_bin_extension)
                sp_cutoff_min, sp_cutoff_max = bins[0], bins[-1]
                if sp_cutoff_min == -inf and bins[1] < +inf:
                    sp_cutoff_min = bins[1]
                if sp_cutoff_max == +inf and bins[-2] > -inf:
                    sp_cutoff_max = bins[-2]
                rw_p.setdefault('sp_cutoff_min', sp_cutoff_min)
                rw_p.setdefault('sp_cutoff_max', sp_cutoff_max)
                w1 = pe.reweight(states, **rw_p, source='kcv')[0]
                w2 = pe.reweight(states[::-1], **rw_p, source='kcv')[0]
                frame_lengths = pe.n_frames
                k12 = np.sum(w1 * frame_lengths)
                k12 = 1 / k12 if k12 else nan
                k21 = np.sum(w2 * frame_lengths)
                k21 = 1 / k21 if k21 else nan
                results['fraction'][row] = fraction
                results['system'][row] = str(sid)
                results['n_frames'][row] = float(frame_lengths.sum())
                results['k12'][row] = k12
                results['k21'][row] = k21
                print(f'    [system {sid!r}] k12={k12:.3e} k21={k21:.3e} '
                      f'[1/dt], {frame_lengths.sum()} frames')

                # Bias-reweighted (Tiwary-Parrinello) rates, per system
                if record_bias:
                    if bias_source == 'file' and bias_function is not None:
                        self._cache_bias_files(pe, bias_function, system_id=sid)
                    from ..pathensemble.bias_utils import (
                        bias_reweighted_rates)
                    k12_rw, k21_rw, gamma1, gamma2 = bias_reweighted_rates(
                        pe, w1, w2, lengths=frame_lengths, states=states,
                        reactive_threshold=(
                            params.bias_reactive_threshold_of(sid)),
                        label=f'[system {sid!r}] ')
                    results['k12_rw'][row] = k12_rw
                    results['k21_rw'][row] = k21_rw
                for fname in set(pe.fnames):
                    kcv_fname = get_cache_fname(fname, 'kcv')
                    NPY_CACHE.remove(kcv_fname)
                    if os.path.exists(kcv_fname):
                        try:
                            os.remove(kcv_fname)
                        except OSError:
                            pass
                row += 1

            if network_save_pattern is not None:
                net_fname = network_save_pattern.format(
                    directory=directory, states=states, fraction=fraction,
                    fraction_pct=round(fraction * 100))
                os.makedirs(os.path.dirname(net_fname) or '.', exist_ok=True)
                torch.save(network.state_dict(), net_fname)

        print(f'\nRestoring original network state {now()}')
        _buf = io.BytesIO(_saved_state)
        network.load_state_dict(torch.load(_buf, map_location=device,
                                           weights_only=True))
        np.save(save_file, results)
        print(f'Saved per-system kinetics convergence to {save_file!r}')
        return results
