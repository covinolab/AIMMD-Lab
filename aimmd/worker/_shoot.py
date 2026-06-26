"""
aimmd.worker._shoot
==================

Core path-sampling task for AIMMD workers.

This module defines :class:`WorkerShoot`, the worker implementation of **path
sampling**, which is the central simulation engine of AIMMD.

It does (AI-enhanced) path sampling by two-way shooting in the chosen state.

AIMMD's main objective is to **enhance sampling** by producing a *diverse* set
of paths, including more large excursions and transition than unbiased simulations.

The key mechanism is to repeatedly:

1) choose a **shooting point** (a frame) from an existing ensemble, and
2) launch a **backward** and a **forward** simulation from that frame to build a
   new candidate path segment around the reactive region.

In the standard workflow, shooting-point selection is guided by the current
**committor model** (typically a neural network): frames are preferentially
sampled from regions where the committor is most informative for exploring the
reactive tube / separatrix and for producing statistically useful path diversity.
The selection logic is implemented in :func:`aimmd.worker.utils.select_shooting_point`
and operates on a maintained *selection pool* updated continuously from the
current chain and from initial paths.

Two execution modes are supported:

1) **Standard shooting chain** (``sweep=False``)
   This is the production mode used for AIMMD sampling. The worker maintains:

   - a shooting *chain* (the growing sequence of sampled paths),
   - and, when applicable, a *selection pool* of candidate paths used to propose
     shooting points according to the committor-guided strategy.

   Each iteration:

   - updates the selection pool using the existing chain and/or initial paths,
   - selects a shooting point (frame) from the pool (or from overriding free
     trajectories, if configured),
   - initializes a backward and a forward simulation from that point,
   - incrementally runs/extends those simulations until completion,
   - merges the backward and forward pieces into a single path,
   - registers the path into the chain (and optionally applies TPS acceptance),
   - updates tqdm progress bar *always* printed in original `stdout`.

2) **Sweep mode for committor validation** (``sweep=True``)
   Sweep mode serves *model validation* rather than adaptive sampling. It
   deterministically cycles through a predefined set of frames (taken from the
   concatenated ``initial_paths`` ensemble) and repeatedly shoots from them
   to obtain brute-force committor estimates.

   Over many shots from the same starting frame, the empirical committor to end
   state 1 is estimated as:

   ``q ≈ N(reach 1) / (N(reach 0) + N(reach 1))``

   Comparing these brute-force estimates to the network-predicted committor
   provides a direct validation of the committor model.

Folder layout
-------------
Output is written under a per-target, per-worker folder:

- Standard: ``{directory}/chain{t}{k}``
- Sweep:    ``{directory}/sweep{t}{k}``

Within this folder, two engine deffnm prefixes are used:

- ``{folder}/back`` for the backward half
- ``{folder}/forw`` for the forward half

The resulting full path is assembled by reversing the backward piece (so it
runs forward in time away from the shooting point) and concatenating it with
the forward piece (excluding the duplicate shooting-point frame).

State conventions
-----------------
The target state label ``t`` is obtained from ``target_state`` via
:func:`~aimmd.core.utils.process_state`, using ``params.states`` as the mapping.

Some workflows require that at least one transition is present in the selection
pool. This is controlled by :attr:`Params.at_least_one_transition_in_pool`.

Network/bins waiting
--------------------
If the worker is shooting from the reactive state (``t == states[1]``), is not
in sweep mode, and the model uses more than one bin (``nbins > 1``), the task
can block until the network and current bins/densities are available.

TPS vs non-TPS
--------------
If ``params.chain_type == 'tps'``, the chain is treated as a TPS chain and
after each new path is registered, a TPS acceptance step is performed and the
TPS weights are saved to disk.

If not TPS, incomplete paths may be assigned zero weight and excluded from the
selection pool in specific cases.

Expected collaborators
----------------------
This mixin relies on worker components providing:

- :meth:`run` dispatch (from :class:`~aimmd.worker._run.WorkerRun`),
- :meth:`_simulate` (from :class:`~aimmd.worker._simulate.WorkerSimulate`),
- :attr:`initial_paths` and :attr:`must_stop`,
- path and pool utilities from :mod:`aimmd.worker.utils`:
  ``register_path``, ``update_selection_pool``, ``select_shooting_point``,
  ``accept_or_reject_last_path``.

Notes
-----
- The backward simulation is run first. If it hits ``max_length`` (after
  accounting for offsets), the forward simulation may be skipped entirely.
- The method updates :attr:`_total_frames` and :attr:`_total_steps` for
  stop-condition bookkeeping and progress reporting in higher-level loops.
"""

# external
import os
import numpy as np
from abc import ABC

# aimmd imports
from .utils import register_path
from .utils import select_shooting_point
from .utils import update_selection_pool
from .utils import accept_or_reject_last_path
from .utils import get_initial_transitions_for_shooting_chain
from ..path import Path
from .._config import print
from ..cache.npy import save_npy, load_npy
from ..core.utils import now, remove, process_state
from ..pathensemble import PathEnsemble

# worker "shoot" run method
class WorkerShoot(ABC):

    def shoot(self, target_state=1, k=0, sweep=False, nchains_per_worker=1):
        """
        Public convenience wrapper for the shooting task.
        
        Do (AI-enhanced) path sampling by two-way shooting in the chosen state.

        Parameters
        ----------
        target_state : int or str, optional
            Target state used to select shooting points and to name output
            folders.

            - If int, interpreted as an index into ``params.states`` (a, r, b).
            - If str, interpreted as the state label directly.

            The common convention is: ``a (0)``, ``r (1)``, ``b (2)``, but any
            label supported by ``params.states`` is accepted. Default is ``1``.
        k : int, optional
            Worker index used to disambiguate output folders (e.g., ``chainR{k}``)
            and to offset cycling of initial paths. Default is ``0``.
            If `nchains_per_worker > 1`, it is just the *first* shooting chain
            associated with the worker, with the others following sequentially.
        sweep : bool, optional
            If ``False``, run committor-guided shooting with a selection pool
            (enhanced sampling in the reactive region).

            If ``True``, run sweep-mode shooting intended for *committor
            validation*: deterministically cycle through a predefined set of
            frames (from the merged initial ensemble) and repeatedly shoot from
            them to empirically estimate outcome probabilities (e.g., fraction
            reaching state 1 vs state 0). Default is ``False``.
        nchains_per_worker : int, optional, default = 1
            If > 1, the worker will manage more than one chain. A higher value
            of `nchains_per_worker` tends to regularize the training set and
            thus improve performance. If running only one shooting worker and
            `selection_pool_size=1`, `nchains_per_worker=10` is recommended.

        Returns
        -------
        object
            Whatever :meth:`Worker.run` returns for the ``'shoot'`` task.
        """
        return self.run('shoot', target_state, k, sweep, nchains_per_worker)

    def _shoot(self, target_state=1, k=0, sweep=False, nchains_per_worker=1):
        """
        Internal implementation of the ``'shoot'`` worker task.

        Parameters
        ----------
        target_state : int or str, optional
            See :meth:`shoot`.
        k : int, optional
            See :meth:`shoot`.
        sweep : bool, optional
            See :meth:`shoot`.
        nchains_per_worker : int, optional
            See :meth:`shoot`.

        Returns
        -------
        None

        Notes
        -----
        The method exits cooperatively when :attr:`must_stop` becomes ``True``.
        Paths are generated iteratively and appended to the on-disk chain managed
        by :class:`~aimmd.params.Params` accessors.
        """
        # get/process params
        k = int(k)
        nchains_per_worker = int(nchains_per_worker)
        mode = 'shoot'
        directory = self.directory
        params = self.params
        do_tps = params.chain_type == 'tps'
        states = params.states
        if params.at_least_one_transition_in_pool:
            at_least_one = states
        else:
            at_least_one = ''
        t = process_state(target_state, states)
        pool_size = params.selection_pool_size if not sweep else 1
        # retrieve and process paths
        if not sweep:
            initial_paths = get_initial_transitions_for_shooting_chain(
                self.initial_paths, states)
        else:
            initial_paths = self.initial_paths
            for path in initial_paths:
                if not (path.states == t).all():
                    raise RuntimeError(f'all frames in {path.fname} must be '
                                       f'in {t} when `sweep=True`')
        nbins = params.nbins
        max_length = params.max_length
        free_overriding_states = params.free_overriding_states
        if params.engine == 'gromacs':
            eneconv = params.gmx_eneconv
        else:
            eneconv = None
        always_select_inside_the_bins = params.always_select_inside_the_bins

        # given a chain_id < nchains_per_worker, activates folder
        def activate_chain(chain_id):

            # get folders
            folder = (f'sweep{t}{chain_id + k}' if sweep else
                      f'chain{t}{chain_id + k}')
            directory = self.directory or '.'
            _directory = self._directory or '.'

            # exclusively for progress bar
            # location can be different from folder if the params file does not live
            # in the current working directory
            self._location = f'{directory}/{folder}'
            
            # create folder if not existing (along with all intermediate paths)
            self.folder = f'{_directory}/{folder}'
            os.system(f'mkdir -p {self.folder}')

            # update log file
            if not self.log_file == self.original_stdout:
                self.log_file = f'{folder}/worker.log'

            return self.folder
        
        # load initial paths, chains, and pools
        chains = []
        pools = []
        chains_initial_paths = []
        total_frames = 0
        total_steps = 0

        # loop through all chains managed by worker
        for chain_id in range(nchains_per_worker):
            folder = activate_chain(chain_id)

            # chain's initial paths
            chain_initial_paths = PathEnsemble()
            for path_id in range((k + chain_id) * pool_size,
                                 (k + chain_id + 1) * pool_size):
                path_id %= len(initial_paths)
                path = initial_paths[path_id]
                if path not in chain_initial_paths:
                    chain_initial_paths._paths.append(path)

            # load shooting chain and selection pool (Markov chains)
            if not sweep:
                print(f'\nLoading shooting chain and selection pool {now()}')
                chain = params.shot_chains(directory, t, k)  # weights accounted
                pool = PathEnsemble(f'{folder}/pool.log')
                update_selection_pool(
                    pool, pool_size, chain,
                    chain_initial_paths, at_least_one=at_least_one,
                    boundaries=(load_npy(f'{directory}/bins.npy')
                                if always_select_inside_the_bins
                                and t == states[1] else None))
                pool.save(f'{folder}/pool.log')
                print(f'... currently {len(chain)} path'
                    f'{"s" if len(chain) != 1 else ""} in shooting chain')
                print(f'... currently {len(pool)} path'
                    f'{"s" if len(pool) != 1 else ""} in selection pool')
            
            # sweeping simulations
            else:
                chain = params.shot_paths(directory, 'sweep', t, k)
                pool = chain_initial_paths[:1]  # should be already of size 1
                sweep_frames = pool[0]
                sweep_size = len(sweep_frames)
                print(f'\nReport after {len(chain)} paths')
                chain.report_shooting_results(states, sweep_size)
                print()
        
            # update lists, total frames and steps
            chains_initial_paths.append(chain_initial_paths)
            chains.append(chain)
            pools.append(pool)
            total_frames += sum(chain.n_frames)
            total_steps += len(chain)
        self.total_frames = total_frames
        self.total_steps = total_steps

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

            # we always manage the SHORTEST chain
            chain_id = np.argmin([len(chain) for chain in chains])
            folder = activate_chain(chain_id)
            chain = chains[chain_id]
            pool = pools[chain_id]
            chain_initial_paths = chains_initial_paths[chain_id]

            # need to initialize?
            if not params.check_if_initialized(
                f'{folder}/back', f'{folder}/forw'):
                print(f'\nSelecting shooting point for '
                      f'{self._location}/path{len(chain) + 1:06g} {now()}')

                if not sweep:
                    # update selection pool
                    # (add last chain path to pool if not already there)
                    update_selection_pool(
                        pool, pool_size, chain,
                        chain_initial_paths,
                        at_least_one=at_least_one,
                        boundaries=(load_npy(f'{directory}/bins.npy')
                                    if always_select_inside_the_bins
                                    and t == states[1] else None))

                    # select shooting point
                    shooting_point = select_shooting_point(
                        pool, params, folder, chain,
                        free_trajectories=params.free_trajectories(directory)
                        if free_overriding_states else [],
                        shooting_chains=chains,
                        target_state=t)

                else:  # sweep
                    index = len(chain) % len(sweep_frames)
                    fname_index, loc = sweep_frames._get_local_loc(index)
                    print(f'=== selecting frame '
                          f'{sweep_frames._fnames[fname_index]}, {loc}')
                    shooting_point = sweep_frames[index:index + 1]

                # clean
                remove(f'{folder}/*back*', f'{folder}/*forw*')

                # initialize simulation
                params.initialize_simulation(shooting_point,
                    f'{folder}/back', f'{folder}/forw')

                if not sweep:  # save pool status (removed SP's source)
                    pool.save(f'{folder}/pool.log')

            try:
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
            except RuntimeError as exc:
                if (params.retry_with_state_definition_glitches
                        and 'should be in' in str(exc)):
                    print(f'WARNING: state-definition glitch in {folder}: '
                          f'{exc}. Deleting back*/forw* and reselecting a '
                          f'new shooting point '
                          f'(retry_with_state_definition_glitches=True).')
                    remove(f'{folder}/*back*', f'{folder}/*forw*')
                    back_simulation_completed = False
                    forw_simulation_completed = False
                    back = Path()
                    forw = Path()
                    continue
                raise

            # check mid cycle
            if self.must_stop:
                return

            if forw_simulation_completed:  # simulation is completed
                # join the two together in a path
                # (will inherit the right shooting index)
                path = back[nframes_back - 1::-1] + forw[1:nframes_forw]
                # save path and add it to chain
                # (zero weight in case of "bad" path)
                _bias_fn = (params.bias_function
                            if (params.record_bias
                                and params.bias_source == 'file')
                            else None)
                register_path(path, chain, eneconv, bias_function=_bias_fn)
                self.total_steps += 1
                self.total_frames += path.n_frames
                
                # clean and reset
                print()
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

                    if not path.weight:
                        # manually compute shooting point value
                        # otherwise the value will never be updated
                        # if not training, since will never feature
                        # in the selection pool
                        si = path.shooting_index
                        path[si:si + 1].compute(*params.compute_values_args,
                                                return_result=True)
                
                else:  # print sweep summary
                    print(f'\nReport after {len(chain)} paths')
                    chain.report_shooting_results(states, sweep_size)
