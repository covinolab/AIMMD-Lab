"""
...
"""

# external
import os
import time
from abc import ABC

# aimmd imports
from ..path import Path
from ..core.utils import now, remove, process_state

# worker simulate
class WorkerSimulate(ABC):
    def _simulate(self, deffnm, trajectory, t, mode='shoot',
                  offset=0, extra_frames=0):
        """t: target state
        Input
        -----
        trajectory: if provided, appends to that
        Returns
        -------
        Output stop_frame: if not None, it means that free simulation
        has to restart from stop_frame
        nframes = total number of frames in trajectory
        
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
                self._total_frames += added_frames
                
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
