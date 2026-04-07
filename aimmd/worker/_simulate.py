"""
aimmd.worker._simulate
=====================

Trajectory-following simulation loop for AIMMD workers.

This module defines :class:`WorkerSimulate`, a mixin providing the internal
:meth:`~WorkerSimulate._simulate` routine used by worker tasks that need to run
(or continue) an engine simulation while *simultaneously* consuming frames from
a trajectory on disk.

The method coordinates three concerns:

1) **Incremental trajectory ingestion**
   The provided :class:`~aimmd.path.Path` (or Path-like object) is extended from
   on-disk trajectory files matching a pattern (either a single file or GROMACS
   ``.part????`` files when ``noappend=True``). Newly added frames are processed
   through the per-frame analysis pipeline (excluding values).

2) **Stopping logic**
   A nested ``stop_condition()`` drives both:
   - *trajectory completion detection* via ``trajectory.check_stop(...)`` (e.g.,
     the path reached a target state or exceeded a maximum length),
   - *global worker stop conditions* via :attr:`must_stop` (walltime/steps/frames
     limits and external termination signals).

   When the simulation runs faster than the worker can ingest frames, the method
   returns early to allow the caller to restart/continue from a partial point.

3) **Engine invocation**
   If simulation can proceed (required input files exist and any stale artifacts
   are cleaned), the method calls :meth:`~aimmd.params.Params.run_simulation`
   and supplies the nested ``stop_condition`` so the engine process can be
   terminated cooperatively.

The return value is the mutable ``check_result`` list, updated in place by
``trajectory.check_stop(...)``.

Expected interface
------------------
This method assumes ``trajectory`` implements at least:

- ``extend(pattern, batch_size, remove_overlapping_frames=True, pipeline=...)``
  -> ``(added_frames, frames_left)``
- ``check_stop(allowed_states=..., max_length=..., check_first_frame=...)``
  -> ``(stop_frame, nframes, last_state, last_length)``
- ``__len__`` and attributes used for reporting (e.g., ``lengths``)

and that the worker instance provides:

- :attr:`params` (a :class:`~aimmd.params.Params` instance),
- :attr:`must_stop` and :attr:`termination_timeout`,

Notes
-----
- The nested ``stop_condition()`` is intentionally stateful (uses ``nonlocal``
  variables) to rate-limit printing and avoid spinning when no new frames are
  available.
- The string prints use the built-in ``print`` function (not AIMMD's wrapped
  print); output routing is therefore controlled by the worker's stdout/stderr
  redirection.
"""

# external
import os
import time
from abc import ABC

# aimmd imports
from ..core.utils import now, remove

# WorkerSimulate mixin class
class WorkerSimulate(ABC):
    def _simulate(self, deffnm, trajectory, t, mode='shoot',
                  offset=0, extra_frames=0):
        """
        Run or continue a simulation while incrementally extending a trajectory.

        Parameters
        ----------
        deffnm : str
            Engine "deffnm" prefix (e.g., GROMACS ``-deffnm``). The method looks
            for engine input/output files derived from this prefix.
        trajectory : aimmd.path.Path or Path-like
            The trajectory/path object to extend as new frames appear on disk.
            The object is mutated in place by calling ``trajectory.extend(...)``.
        t : str
            Target state label(s) used by the stop logic. Interpretation depends
            on ``mode``:

            - ``mode='shoot'``: the simulation is considered complete when the
              trajectory hits the state(s) in ``t``.
            - ``mode='free'``: allowed states are ``f'{t}{states[1]}'``; the
              first-frame check depends on whether ``t`` matches the second
              state label in ``params.states``.
        mode : {'shoot', 'free'}, optional
            Selects file naming conventions and stop-check settings. Default is
            ``'shoot'``.
        offset : int, optional
            Offset applied to the maximum allowed trajectory length. The effective
            maximum is ``params.max_length - offset``. Default is ``0``.
        extra_frames : int, optional
            Additional frames required beyond the computed stop frame before the
            method returns a "complete" stop. This is used to ensure sufficient
            buffer frames are available after a stop event. Default is ``0``.

        Returns
        -------
        list
            A 4-element list ``[stop_frame, nframes, last_state, last_length]``
            as returned by ``trajectory.check_stop(...)`` (updated in place).
            If ``stop_frame`` is not ``None``, the caller should interpret this as
            a completed path (or a restart point for free simulation, depending
            on higher-level logic).

        Notes
        -----
        - This method may return before the simulation is "complete" if the worker
          cannot keep up with trajectory ingestion (``frames_left`` is truthy) or
          if global stop conditions trigger.
        - For GROMACS, simulation cannot start until ``{deffnm}.tpr`` exists.
        - For the toy engine, simulation cannot start until the first trajectory
          file exists (either ``{deffnm}{ext}`` or ``{deffnm}.part0000{ext}``).
        """
        # get args
        params = self.params
        engine = self.params.engine
        ext = params.trajectory_extension
        max_length = params.max_length - offset
        batch_size = params.trajectory_update_batch_size
        check_result = [None, len(trajectory), '', 0]
        pipeline = params.pipeline[:-1]  # except for values
        states = params.states

        if mode == 'shoot':
            noappend = False
            check_stop_args = {'allowed_states': t,
                               'max_length': max_length,
                               'check_first_frame': True}
        elif mode == 'free':
            noappend = True
            check_stop_args = {'allowed_states': f'{t}{states[1]}',
                               'max_length': max_length,
                               'check_first_frame': t != states[1]}

        if not noappend:
            pattern = f'{deffnm}{ext}'
        else:
            pattern = f'{deffnm}.part????{ext}'

        # stop condition
        t0 = time.time()
        old_nframes = 0

        def stop_condition():
            nonlocal t0, old_nframes
            """Only when stopping, returns True, otw False"""
            while True:

                # check stop condition, update general
                check_result[:] = trajectory.check_stop(**check_stop_args)
                stop_frame, nframes, last_state, last_length = check_result

                # reset and stop
                if stop_frame is not None:
                    n = stop_frame + last_length
                    if nframes - n >= extra_frames:  # ok to return
                        print(f'xxx {deffnm} completed after {n} frame'
                              f'{"s" if n != 1 else ""} in {last_state}')
                        return True
                    check_result[0] = None

                # keep on until new frames are added
                added_frames, frames_left = trajectory.extend(
                    pattern, batch_size,
                    remove_overlapping_frames=True,
                    pipeline=pipeline)

                # stop extending because...
                condition1 = time.time() - t0 > 10.0  # too long
                condition2 = nframes >= old_nframes + batch_size  # enough
                condition3 = not added_frames  # no frames to add
                condition4 = self.must_stop
                if condition1 or condition2 or condition3 or condition4:
                    # print the update only in this case
                    if (condition1 or condition2 or condition4
                       ) and nframes > old_nframes:
                        t0 = time.time()
                        old_nframes = nframes
                        report = (f'... {deffnm} hit {nframes} '
                                  f'frame{"s" if nframes != 1 else ""}')
                        if noappend:  # much more...
                            nframes_in_file = trajectory.lengths[-1]
                            nframes_in_file -= len(trajectory) - nframes
                            report += (f' ({nframes_in_file} in file),'
                                       f' last path of length {last_length} '
                                       f'in {last_state}')
                        print(report)
                    break

            # temporarily stop because not keeping up with
            # the simulation speed
            if frames_left:
                return True

            # evaluate stop condition (will update termination signal, too)
            return self.must_stop

        # run just once to update the trajectory
        if stop_condition():
            return check_result

        # no need to simulate
        if (stop_frame := check_result[0]) is not None:
            return check_result

        # can't simulate yet (gromacs)
        if engine == 'gromacs' and not os.path.exists(f'{deffnm}.tpr'):
            return check_result

        # can't simulate yet (toy)
        if (engine == 'toy' and
            ((not noappend and not os.path.exists(f'{deffnm}{ext}')) or
             (noappend and not os.path.exists(f'{deffnm}.part0000{ext}')))):
            return check_result

        # clean relics that don't allow you to simulate
        if (engine == 'gromacs' and not noappend and
            os.path.exists(f'{deffnm}.cpt') and
            not os.path.exists(f'{deffnm}{ext}')):
            remove(f'{deffnm}*', except_for=f'{deffnm}.tpr', verbose=True)

        # simulate
        print(f"+++ starting simulating {deffnm} {now()}")
        params.run_simulation(
            deffnm, noappend=noappend, stop_condition=stop_condition,
            termination_timeout=self.termination_timeout)
        print(f"xxx stopped simulating {deffnm} {now()}")

        # return updated
        return check_result
