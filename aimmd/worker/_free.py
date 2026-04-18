"""
aimmd.worker._free
=================

"Free" simulation task for AIMMD workers.

This module defines :class:`WorkerFree`, a mixin implementing the ``free`` task
executed by :meth:`aimmd.worker.Worker.run`. Free simulations are long(er)
unbiased trajectories started from short initial configurations and run until a
stop criterion is met (typically reaching a target state, leaving an allowed
state set, or exceeding a maximum length).

In AIMMD terminology (as used here), a free simulation is organized as a set of
trajectory segments written under a folder:

- ``{directory}/free{t}/traj??????...``

where ``t`` is a chosen "target" state label (string) and ``??????`` is a 6-digits
trajectory index that is stepped by ``total`` to support multiple workers
writing non-overlapping trajectory names.

The core loop proceeds as follows:

1) Maintain a :class:`~aimmd.path.Path` object (``trajectory``) representing the
   current free trajectory being written by the simulation engine.

2) Incrementally extend ``trajectory`` from on-disk trajectory files by calling
   :meth:`WorkerSimulate._simulate` in ``mode='free'``. This method both ingests
   new frames and decides whether a stop event has been reached.

3) If the engine has not yet been initialized for the current ``deffnm``,
   choose suitable *initial frames* (two frames defining a starting direction)
   and call :meth:`~aimmd.params.Params.initialize_simulation`.

4) When a stop event is detected, select initial frames from the last valid
   crossing and advance to the next trajectory name.

State conventions
-----------------
Let ``states = params.states`` and ``r = states[1]`` denote the "reactive"
region/state label used by the broader AIMMD workflow.

- The caller passes ``target_state`` as either:
  - an integer index into ``params.states``, or
  - a string state label.

The helper :func:`~aimmd.core.utils.process_state` normalizes this to the string
label ``t``.

For free simulations, the allowed state set used by :meth:`_simulate` is
constructed as ``f'{t}{r}'`` (i.e., the trajectory is allowed to visit the
target state and the reactive state).

Waiting mode
------------
When ``wait=True`` and the model uses more than one bin (``params.nbins > 1``),
the worker can block until the network and the current ``bins``/``densities``
artifacts are available on disk. This supports workflows where free simulations
depend on a trained network / adaptive sampling state.

Expected collaborators
----------------------
This mixin assumes:

- :meth:`run` dispatch exists (from :class:`~aimmd.worker._run.WorkerRun`),
- :meth:`_simulate` exists (from :class:`~aimmd.worker._simulate.WorkerSimulate`),
- cooperative stop checks via :attr:`must_stop` / :attr:`termination_signal`.

Notes
-----
- The worker writes to a folder per target state (``free{t}``). This prevents
  different target-state free runs from intermixing and simplifies downstream
  analysis.
- Initialization uses two frames (a minimal "direction") so engines that require
  velocities or a previous frame can be seeded consistently.
"""

# exteral
import numpy as np
from abc import ABC

# aimmd imports
from ..path import Path
from .._config import print
from ..cache.npy import save_npy
from ..core.utils import now, remove, process_state
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble
from ..pathensemble.utils import assemble_pathensemble

# WorkerFree mixin class
class WorkerFree(ABC):

    def free(self, target_state=0, k=0, total=1, wait=False):
        """
        Public convenience wrapper for the free-simulation task.

        Parameters
        ----------
        target_state : int or str, optional
            Target state for free simulations.

            - If int, it is interpreted as an index into ``params.states``.
            - If str, it is used directly as the state label.

            Default is ``0``.
        k : int, optional
            Worker slot index within a group of ``total`` free simulations.
            Used to compute the starting trajectory number and to select initial
            paths deterministically when needed. Default is ``0``.
        total : int, optional
            Total number of concurrent free simulations across workers. The
            current worker advances trajectory indices by ``total`` so that each
            worker writes a disjoint subsequence of trajectory names. Default is
            ``1``.
        wait : bool, optional
            If ``True`` and ``params.nbins > 1``, wait until the network and
            current bins/densities can be loaded before starting. Default is
            ``False``.

        Returns
        -------
        object
            Whatever :meth:`Worker.run` returns for the ``'free'`` task.
        """
        return self.run('free', target_state, k, total, wait)

    def _free(self, target_state=0, k=0, total=1, wait=False):
        """
        Internal implementation of the ``'free'`` worker task.

        Parameters
        ----------
        target_state : int or str, optional
            See :meth:`free`.
        k : int, optional
            See :meth:`free`.
        total : int, optional
            See :meth:`free`.
        wait : bool, optional
            See :meth:`free`.

        Returns
        -------
        None

        Notes
        -----
        The method exits cooperatively when :attr:`must_stop` becomes ``True`` or
        when :attr:`termination_signal` is set by the worker.
        """
        # get/process params
        k = int(k)
        total = int(total)
        params = self.params
        ext = params.trajectory_extension
        len_ext = len(ext)
        batch_size = params.trajectory_update_batch_size
        pipeline = params.pipeline[:-1]  # except for values
        extra_frames = params.extra_free_frames
        # which state are we talking about?
        states = params.states
        t = process_state(target_state, states)
        r = states[1]
        tr = f'{t}{r}'  # allowed states: target & reactive
        restart_with_transition = \
            params.restart_free_simulations_with_transitions

        # exclusively for progress bar
        # location can be different from folder if the params file does not live
        # in the current working directory
        self._location = f'{self.directory}/free{t}'
        directory = self._directory
        folder = f'{directory}/free{t}'
        
        # initialize fake launcher, create/overwrite initial frames
        while not (initial_frames_available :=
                   PathEnsemble(f'{folder}/initial*{ext}')):
            from ..launcher import Launcher
            launcher = Launcher(params, directory)
            if t == states[0]:
                launcher._update(n=0, n1=1)
            else:
                launcher._update(n=0, n2=1)
            launcher._build()
        
        # must have network parameters
        if wait and params.nbins > 1:
            print(f'\nWaiting for neural network parameters {now()}')
            while True:
                try:
                    params.update_network(directory, timeout=0,
                                          raise_if_failure=True)
                    break
                except:
                    if self.termination_signal:
                        return

        # initialize
        chains = []
        initial_frames = None
        num = k + 1  # first trajectory
        name = f'traj{num:06g}'
        deffnm = f'{folder}/{name}'
        trajectory = Path()
        old_nframes = 0
        total_frames = [0]

        # main cycle
        print(f"\nCurrent trajectory: {deffnm} {now()}")
        while not self.must_stop:

            # update old trajectories while it is possible
            (stop_frame, nframes, last_state, last_length) = \
                self._simulate(deffnm, trajectory, t, 'free', 0, extra_frames)
            simulation_completed = stop_frame is not None

            # handle total frames progress bar
            if simulation_completed:
                total_frames[-1] = stop_frame + last_length
            else:
                total_frames[-1] = nframes
                self.total_frames = sum(total_frames)
            
            # check mid cycle
            if self.must_stop:
                return
            
            # initialize only when necessary
            if (nframes == old_nframes and
                not simulation_completed and
                not params.check_if_initialized(deffnm)):

                # need to find initial_frames
                if restart_with_transition or not initial_frames:

                    # take initial_frames from a sampled transition
                    if restart_with_transition:
                        chains = params.shot_chains(directory, r, old=chains)
                        transitions = assemble_pathensemble(chains).extract(
                            states, states[::-1])
                        if transitions:
                            path = transitions[np.random.choice(len(transitions))]

                        else:
                            
                            # take initial_frames from those already available
                            initial_frames = initial_frames_available[
                                np.random.choice(len(initial_frames_available))]
                    else:
                        initial_frames = initial_frames_available[
                            k % len(initial_frames_available)]               
                
                # wipe out garbage
                remove(f'{folder}/*{name}*')

                # report
                fnames = initial_frames.filenames
                locs = initial_frames.locs
                if fnames[0] == fnames[-1]:
                    print(f'\nUsing {fnames[0]} {locs[0]} -> {locs[-1]} '
                          f'for initializing {deffnm}')
                else:
                    print(f'\nUsing {fnames[0]} {locs[0]} -> '
                          f'{fnames[-1]} {locs[-1]} '
                          f'for initializing {deffnm}')

                # last frame (initialize simulation)
                # already divide in parts if more than one frame
                params.initialize_simulation(initial_frames, deffnm)

                # PLUMED file-mode bias tracking: the part0000 segment is the
                # python-written seed (initialize_simulation writes
                # initial_frames[:-1] there). GROMACS never runs on it, so
                # PLUMED produces no COLVAR data and run_simulation cannot
                # slice a per-part _COLVAR for it. Without a bias cache here,
                # path._get('bias', raise_if_missing=True) would fail and the
                # whole free trajectory would fall back to gamma=1.0 in the
                # bias correction. Approximate the seed-frame bias as 0
                # (1-frame-out-of-thousands; bypasses bias_function which has
                # no source COLVAR to read).
                if (getattr(params, 'record_bias', False)
                        and getattr(params, 'bias_source', '') == 'file'):
                    seed_n = max(len(initial_frames) - 1, 0)
                    if seed_n > 0:
                        seed_xtc = f'{deffnm}.part0000{ext}'
                        save_npy(get_cache_fname(seed_xtc, 'bias'),
                                 np.zeros(seed_n, dtype=float))

            # update old_nframes
            old_nframes = nframes

            # simulation is completed! go forward
            if simulation_completed:
                self.total_steps += 1
                total_frames.append(0)

                # take initial frames from last valid crossing
                if t == r:
                    stop_frame += last_length - 2
                initial_frames = trajectory[stop_frame:stop_frame + 2]
                if initial_frames.states[1] == t:
                    pass
                elif initial_frames.states[0] == t:
                    initial_frames = initial_frames[::-1]
                else:
                    initial_frames = None  # this should never happen
                    # but it allows to recover from "corrupted" data

                # go to next trajectory
                num += total
                name = f'traj{num:06g}'
                deffnm = f'{folder}/{name}'
                trajectory = Path()
                old_nframes = 0
                print(f"\nCurrent trajectory: {deffnm} {now()}")
