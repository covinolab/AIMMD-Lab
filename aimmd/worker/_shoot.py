"""
aimmd.worker._shoot
==================

Core path-sampling task for AIMMD workers.

This module defines :class:`WorkerShoot`, the worker implementation of **path
sampling**, which is the central simulation engine of AIMMD. It performs 
(AI-enhanced) path sampling via two-way shooting in the chosen state.

AIMMD’s main objective is to **enhance sampling** by producing a *diverse* set
of paths, including more large excursions and transitions than unbiased 
simulations.

The key mechanism is to repeatedly:

1. Choose a **shooting point** (a frame) from an existing ensemble.
2. Launch a **backward** and a **forward** simulation from that frame to build 
   a new candidate path segment around the reactive region.

In the standard workflow, shooting-point selection is guided by the current
**committor model** (typically a neural network). Frames are preferentially
sampled from regions where the committor is most informative for exploring the
reactive tube/separatrix. The selection logic is implemented in
:func:`aimmd.worker.utils.select_shooting_point`.

Execution Modes
---------------

1) **Standard Shooting Chain** (``sweep=False``)
   The production mode for AIMMD sampling. The worker maintains a shooting 
   *chain* (a growing sequence of sampled paths). Each iteration:

   - Selects a shooting point from the latest accepted path in the chain (or 
     the initial path if the chain is empty).
   - Initializes and extends backward and forward simulations.
   - Merges segments into a single path and registers it into the chain.
   - Applies TPS acceptance logic if configured.
   - Updates a ``tqdm`` progress bar, which is *always* printed to the 
     original ``stdout`` regardless of file redirection.

2) **Sweep Mode** (``sweep=True``)
   Used for **committor validation** rather than adaptive sampling. It 
   deterministically cycles through a predefined set of frames from the 
   initial file and shoots repeatedly to obtain brute-force committor estimates:
   
   ``q ≈ N(reach 1) / (N(reach 0) + N(reach 1))``

   Comparing these estimates to network predictions provides direct model 
   validation.

Parallelism & Scaling
---------------------
Each worker supports **more than one chain**, allowing for optimization of 
training set uniformity even with a limited number of workers. A value of 
**10 chains per worker** is generally recommended.

Folder Layout
-------------
Output is written to a per-target, per-worker directory:

- **Standard:** ``{directory}/chain{t}{k}``
- **Sweep:** ``{directory}/sweep{t}{k}``

Within these folders, the engine uses ``back`` and ``forw`` prefixes for the 
respective simulation halves. The full path is assembled by reversing the 
backward piece and concatenating it with the forward piece (excluding the 
duplicate shooting-point).

State & Selection Conventions
-----------------------------
- **Target State:** The label ``t`` is processed via 
  :func:`~aimmd.core.utils.process_state`.
- **Bin Constraints:** Some workflows restrict selection to the latest path 
  with values inside allowed bins. This is governed by 
  :attr:`Params.never_select_outside_the_bins`.
- **Network Synchronization:** If shooting from the reactive state with 
  multiple bins (``nbins > 1``), the task may block until the network 
  training completes and the ``state_dict`` is available at 
  ``<directory>/network<states>.h5``.

TPS vs. Non-TPS
---------------
- **TPS:** (``params.chain_type == 'tps'``) An acceptance step is performed 
  after each registration, and TPS weights are saved to disk.
- **Non-TPS:** Incomplete paths are assigned zero weight and excluded from 
  future selection.

Expected Collaborators
----------------------
This mixin relies on components providing:

- :meth:`run` dispatch (from :class:`~aimmd.worker._run.WorkerRun`)
- :meth:`_simulate` (from :class:`~aimmd.worker._simulate.WorkerSimulate`)
- :attr:`must_stop` (stop-condition flag)
- Utilities: ``register_path``, ``select_shooting_point``, and 
  ``accept_or_reject_last_path``.

Notes
-----
- The backward simulation is prioritized. If it reaches ``max_length``, the 
  forward simulation may be skipped.
- The method tracks :attr:`_total_frames` and :attr:`_total_steps` for 
  global progress reporting and stop-condition bookkeeping.
"""

# external
import os
import time
import numpy as np
from abc import ABC
from math import inf
from tqdm import tqdm
from numbers import Integral

# aimmd imports
from .utils import register_path
from .utils import select_shooting_point
from .utils import accept_or_reject_last_path
from ..path import Path
from .._config import print
from ..cache.npy import save_npy
from ..core.utils import now, remove, cycle, process_state
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble
from ..execute.threads import ThreadExecutor


# worker "shoot" run method
class WorkerShoot(ABC):

    def shoot(self, target_state=1, k=0, sweep=False, nchains_per_task=1):
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
            If `nchains_per_task > 1`, it is just the *first* shooting chain
            associated with the worker, with the others following sequentially.
        sweep : bool, optional
            If ``False``, run committor-guided shooting with a Markov chain
            (enhanced sampling in the reactive region).

            If ``True``, run sweep-mode shooting intended for *committor
            validation*: deterministically cycle through a predefined set of
            frames (from the merged initial ensemble) and repeatedly shoot from
            them to empirically estimate outcome probabilities (e.g., fraction
            reaching state 1 vs state 0). Default is ``False``.
        nchains_per_task : int, optional, default = 1
            If > 1, the worker will manage more than one chain. A higher value
            of `nchains_per_task` tends to regularize the training set and thus
            improve performance. If running only one shooting worker,
            `nchains_per_task=10` is recommended.
        
        Returns
        -------
        object
            Whatever :meth:`Worker.run` returns for the ``'shoot'`` task.
        """
        return self.run('shoot', target_state, k, sweep, nchains_per_task)

    def _shoot(self, target_state=1, k=0, sweep=False, nchains_per_task=1):
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
        nchains_per_task : int, optional
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
        mode = 'shoot'
        params = self.params
        do_tps = params.chain_type == 'tps'
        # which state are we talking about?
        states = params.states
        t = process_state(target_state, states)
        nbins = params.nbins
        max_length = params.max_length
        ext = params.trajectory_extension
        free_overriding_states = params.free_overriding_states
        original_stdout = self.log_file == self.original_stdout
        k = int(k)
        nchains_per_task = int(nchains_per_task)

        # exclusively for progress bar
        # location can be different from folder if the params file does not live
        # in the current working directory
        directory = self._directory

        # util for managing multiple chains at the same time
        def set_chain_id(chain_id):
            if not sweep:
                folder = f'chain{t}{k + chain_id}'
            else:
                folder = f'sweep{t}{k + chain_id}'
            self._k = k + chain_id
            self._location = f'{self.directory}/{folder}'
            self._folder = f'{directory}/{folder}'
            if not original_stdout:
                self.log_file = f'{folder}/worker.log'
        
        # eneconv
        if params.engine == 'gromacs':
            eneconv = params.gmx_eneconv
        else:
            eneconv = None  

        # initialize shooting chains managed by worker
        chains = []
        total_frames = 0
        total_steps = 0
        
        # create/overwrite initial frames
        for chain_id in range(nchains_per_task):
            set_chain_id(chain_id)
            while not (initial_path := Path(f'{self._folder}/initial*{ext}')):
                from ..launcher import Launcher
                launcher = Launcher(params, directory)
                launcher._update(n=self._k + 1)
                launcher._build()
            
            if not sweep:
                print(f'\nLoading shooting chain {now()}')
                # paths have zero weight if incomplete
                chain = params.shot_chains(directory, t, self._k)
                print(f'... currently {len(chain)} path'
                      f'{"s" if len(chain) != 1 else ""} in shooting chain')
            else:
                chain = params.shot_paths(directory, 'sweep', t, self._k)
                print(f'\nReport after {len(chain)} paths')
                chain.report_shooting_results(states, sweep_size)
                print()

            # update total frames and steps
            total_frames += sum(chain.n_frames)
            total_steps += len(chain)

            # add chain to the list
            chains.append(chain)
        
        # update total frames and steps
        self.total_frames = total_frames
        self.total_steps = total_steps
        
        # must have network, bins, and descriptors
        # only if it makes sense
        if t == states[1] and not sweep and nbins > 1:
            print(f'\nWaiting for neural network parameters {now()}')
            while True:
                try:
                    params.update_network(directory, timeout=0,
                                          raise_if_failure=True)
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
            chain = chains[chain_id]
            set_chain_id(chain_id)
            
            # need to initialize?
            if not params.check_if_initialized(
                f'{self._folder}/back', f'{self._folder}/forw'):
                print(f'\nSelecting shooting point for '
                      f'{self._folder}/path{len(chain) + 1:06g} {now()}')

                # last path with weight > 0 from the chain
                path = chain.path
                if path is None or sweep:
                    path = Path(f'{self._folder}/initial*{ext}')
                    # force recompute
                    remove(get_cache_fname(path.fname, 'values'))

                # select shooting point
                if not sweep:
                    shooting_point = select_shooting_point(
                        path, params, target_state=t)
                
                else:  # sweep
                    index = len(chain) % len(path)
                    fname_index, loc = path._get_local_loc(index)
                    print(f'=== selecting frame '
                          f'{path._fnames[fname_index]}, {loc}')
                    shooting_point = path[index:index + 1]
                
                # clean
                remove(f'{self._folder}/*back*', f'{self._folder}/*forw*')
                
                # initialize simulation
                params.initialize_simulation(shooting_point,
                    f'{self._folder}/back', f'{self._folder}/forw')
            
            # update existing paths: backward
            if not back_simulation_completed:
                (stop_frame, nframes, last_state, last_length) = \
                    self._simulate(f'{self._folder}/back', back, t, mode)
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
                    self._simulate(f'{self._folder}/forw', forw, t, mode, offset)
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
                self.total_steps += 1
                self.total_frames += path.n_frames
                
                # clean and reset
                print()
                remove(f'{self._folder}/*back*', f'{self._folder}/*forw*')
                back_simulation_completed = False
                forw_simulation_completed = False
                back = Path()
                forw = Path()

                if not sweep:

                    # tps (also save weights)
                    if do_tps:
                        accept_or_reject_last_path(chain, params)
                        save_npy(f'{self._folder}/tps_weights.npy',
                                 chain.weights)

                    # path is not valid
                    elif not path.is_complete(t, states):
                        print('xxx path not valid')
                        path.weight = 0.
                
                else:  # print sweep summary
                    print(f'\nReport after {len(chain)} paths')
                    chain.report_shooting_results(states, sweep_size)
