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

where ``t`` is a chosen "target" state label (string) and ``??????`` is a
6-digits trajectory index that is stepped by ``total`` to support multiple
workers writing non-overlapping trajectory names.

The core loop proceeds as follows:

1) Maintain a :class:`~aimmd.path.Path` object (``trajectory``) representing the
   current free trajectory being written by the simulation engine.

2) Incrementally extend ``trajectory`` from on-disk trajectory files by calling
   :meth:`WorkerSimulate._simulate` in ``mode='free'``. This method both ingests
   new frames and decides whether a stop event has been reached.

3) If the engine has not yet been initialized for the current ``deffnm``,
   choose suitable *initial frames* (two frames defining a starting direction)
   and call :meth:`~aimmd.params.Params.initialize_simulation`.

4) When a stop event is detected, select initial frames for the next
   trajectory and advance to the next trajectory name. By default these are
   the last valid crossing, i.e. the frame the trajectory escaped from, which
   lies on the state boundary. ``params.restart_free_simulations_from`` selects
   a different source: ``'transitions'`` (a sampled AIMMD transition path),
   ``'basin'`` (uniform over the frames the accumulated free trajectories spent
   inside the state) or ``'equilibrium'`` (the same pool, reweighted by
   ``exp(bias)`` to the unbiased Boltzmann distribution inside the state), so
   that each first passage starts from inside the state rather than from its
   boundary. See :func:`~aimmd.worker.utils.get_basin_frames_for_free_restart`.

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
- :meth:`_simulate` exists
  (from :class:`~aimmd.worker._simulate.WorkerSimulate`),
- :attr:`initial_paths` exists
  (from :class:`~aimmd.worker._properties.WorkerProperties`),
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
import os
import numpy as np
from abc import ABC

# aimmd imports
from .utils import (get_basin_frames_for_free_restart,
                    get_initial_frames_for_free_simulations)
from ..path import Path
from .._config import print
from ..cache.npy import save_npy
from ..core.utils import now, remove, process_state
from ..path.utils import get_cache_fname
from ..pathensemble import PathEnsemble
from ..pathensemble.utils import assemble_pathensemble

def _basin_weighting_for_mode(mode):
    """
    In-basin candidate weighting for a `params.free_restart_mode` value.

    Returns
    -------
    str or None
        ``'occupancy'`` for ``'basin'`` (uniform over in-state frames, i.e. the
        *biased* equilibrium inside the state), ``'unbiased'`` for
        ``'equilibrium'`` (drawn with probability proportional to ``exp(bias)``,
        i.e. the **unbiased** Boltzmann equilibrium inside the state), and None
        for the sources that do not draw from inside the basin at all
        (``'crossing'``, ``'transitions'``).
    """
    return {'basin': 'occupancy', 'equilibrium': 'unbiased'}.get(mode)


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
        directory = self.directory or '.'
        _directory = self._directory or '.'
        params = self.params
        ext = params.trajectory_extension
        extra_frames = params.extra_free_frames
        states = params.states
        t = process_state(target_state, states)
        r = states[1]
        # retrieve and process paths
        initial_paths = get_initial_frames_for_free_simulations(
                self.initial_paths, t, r)
        # Where this worker's restart configurations come from. Resolved from
        # params.restart_free_simulations_from (which also reads the deprecated
        # restart_free_simulations_with_transitions), and 'crossing' by default,
        # so nothing changes for an existing run. The in-basin sources never
        # apply to the reactive state, where "inside the state" is the barrier
        # region; free_restart_mode already collapses those to 'crossing'.
        restart_mode = params.free_restart_mode(t)
        restart_with_transition = restart_mode == 'transitions'
        basin_weighting = _basin_weighting_for_mode(restart_mode)
        restart_from_basin = basin_weighting is not None
        basin_min_frames = getattr(
            params, 'free_restart_basin_min_frames', 0)

        # get folders
        folder = f'free{t}'

        # exclusively for progress bar
        # location can be different from folder if the params file does not live
        # in the current working directory
        self._location = f'{directory}/{folder}'
        
        # create folder if not existing (along with all intermediate paths)
        folder = self.folder = f'{_directory}/{folder}'
        os.system(f'mkdir -p {folder}')
        
        # must have network, bins, and descriptors
        if wait and params.nbins > 1:
            print(f'\nWaiting for neural network, bins, densities {now()}')
            while True:
                try:
                    params.update_network(
                        directory, timeout=0, raise_if_failure=True)
                    bins, densities = params.load_bins_and_densities(
                        directory, timeout=0, raise_if_failure=True)
                    break
                except:
                    if self.termination_signal:
                        return

        # in-basin restart pool: every free trajectory of this state seen so
        # far. Primed from disk so a requeued worker does not have to rebuild an
        # in-basin sample from scratch; a fresh run starts empty and the very
        # first trajectory is therefore still boundary-seeded (there is nothing
        # else to draw from yet).
        basin_pool = []
        if restart_from_basin:
            try:
                basin_pool = list(params.free_trajectories(_directory, t))
            except Exception as exception:
                print(f'\nWarning: could not prime the in-basin restart pool '
                      f'from {_directory}/free{t}: {exception}')
                basin_pool = []
            print(f'\nFree restarts for {t!r} are drawn from inside the '
                  f'basin (restart_free_simulations_from = {restart_mode!r}, '
                  f'{basin_weighting} weighting, '
                  f'min_frames={basin_min_frames}); pool primed with '
                  f'{len(basin_pool)} trajectories from disk')

        # initialize
        chains = []
        initial_frames = None
        seed_bias = None
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
                    # these sources carry no known history-frame bias
                    seed_bias = None

                    # take initial_frames from a sampled transition
                    if restart_with_transition:
                        chains = params.shot_chains(directory, r, old=chains)
                        transitions = assemble_pathensemble(chains).extract(
                            states, states[::-1])
                        if transitions:
                            initial_frames = transitions[
                                np.random.choice(len(transitions))]
                            if initial_frames.states[0] != t:
                                initial_frames = initial_frames[::-1]
                            initial_frames = initial_frames[:2]
                        else:
                            # take initial frames from initial_paths (random)
                            initial_frames = initial_paths[
                                np.random.choice(len(initial_paths))]
                    else:
                        # take initial_frames from initial paths (in order)
                        initial_frames = initial_paths[k % len(initial_paths)]             
                
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
                # bias correction. A boundary-crossing seed's history frame
                # lies in the (bias-free) reactive region, so 0 is right for it;
                # an in-basin seed's history frame carries the full fill and
                # `seed_bias` holds its recorded value. Approximating THAT as 0
                # would drag gamma down for exactly the shortest trajectories.
                if (getattr(params, 'record_bias', False)
                        and getattr(params, 'bias_source', '') == 'file'):
                    seed_n = max(len(initial_frames) - 1, 0)
                    if seed_n > 0:
                        seed_xtc = f'{deffnm}.part0000{ext}'
                        if (seed_bias is not None
                                and len(seed_bias) == seed_n):
                            seed_values = np.asarray(seed_bias, dtype=float)
                        else:
                            seed_values = np.zeros(seed_n, dtype=float)
                        save_npy(get_cache_fname(seed_xtc, 'bias'),
                                 seed_values)

            # update old_nframes
            old_nframes = nframes

            # simulation is completed! go forward
            if simulation_completed:
                self.total_steps += 1
                total_frames.append(0)

                # take initial frames from inside the basin, when asked to
                initial_frames = None
                seed_bias = None
                if restart_from_basin:
                    basin_pool.append(trajectory)
                    initial_frames, seed_bias = \
                        get_basin_frames_for_free_restart(
                            basin_pool, t, r,
                            weighting=basin_weighting,
                            min_frames=basin_min_frames)
                    if initial_frames is None:
                        print(f'\nWarning: no in-{t} sample to restart from '
                              f'yet; using the boundary crossing instead')
                    else:
                        locs = initial_frames.locs
                        fnames = initial_frames.filenames
                        print(f'\nRestarting from an in-{t} frame drawn from '
                              f'{len(basin_pool)} trajectories '
                              f'({basin_weighting} weighting): '
                              f'{fnames[-1]} {locs[-1]}')

                # take initial frames from last valid crossing
                if initial_frames is None:
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
