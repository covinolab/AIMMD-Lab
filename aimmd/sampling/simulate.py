import os
import time
import functools
from ..core.utils import (now,
                          get_current_simulation,
                          execute_command)

inf = float('inf')

# quick logging
print = functools.partial(print, flush=True)

def simulate(self, run_file, log_file=None, noappend=False, walltime=inf):
    """
    Continuously run simulations as directed by the run file.
    noappend: bool, add Gromacs' -noappend flag.
    """
    
    if noappend == 'False' or noappend == 'false':
        noappend = False
    else:
        noappend = bool(noappend)
    walltime = float(walltime)
    
    try:
        self.log_file = log_file
        print(f"Starting simulation loop ({now()})...")
        if not log_file:
            print(f"Press Control+C to interrupt.")
        
        # run continuously
        t0 = time.time()
        while True:
            
            # maximum time
            if time.time() - t0 > walltime:
                self.terminate_handler(exit=False)
            
            # received the signal
            if self.interrupt:
                break
            
            # interrupt everything currently running
            self.terminate_handler(report=False, exit=False)
            self.interrupt = False  # ...but continue simulating
            
            # what to simulate
            fname = get_current_simulation(f'{self.directory}/{run_file}')
            if not fname:
                continue  # no job assigned yet
            
            # (re)-start logging
            self.log_file = log_file
            print(f"Starting simulating {fname} ({now()})...")
            
            # determine stop condition
            def stop_condition():
                if get_current_simulation(f'{self.directory}/{run_file}') != fname:
                    print(f"Target simulation file changed ({now()}).")
                    return True
                return self.interrupt
            
            # create command
            cmd = (f'{self.params.mdrun} -deffnm {fname}'
                   f'-cpo {fname}.cpt -cpi {fname}.cpt -cpt .1'
                   f'{"-noappend" if noappend else ""}')
            
            # execute command
            if exit := execute_command(cmd, stop_condition):
                raise RuntimeError(f'{cmd} failed with exit code {exit}')
    
    except SystemExit:
        self.terminate_handler()
    
    except KeyboardInterrupt:
        self.terminate_handler(exit=False)
    
    except Exception as exception:
        print(f'Exception: {exception}')
        self.terminate_handler()
    
    finally:
        self.terminate_handler(exit=False)
