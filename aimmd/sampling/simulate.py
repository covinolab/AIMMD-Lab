import os
import time
import functools
from ..core.utils import (now,
                          remove,
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
    
    # report
    self.log_file = log_file
    print(f"Starting worker: simulate ({now()})")
    if not log_file:
        print(f"Press Control+C to interrupt.")
    
    # cleanup when ending
    self.cleanup = [f'{self.directory}/{run_file}.run',
                    f'{self.directory}/{run_file}.ready']
    
    # control through files
    os.system(f'touch {self.directory}/{run_file}.ready')
    
    # process arguments
    if noappend == 'False' or noappend == 'false':
        noappend = False
    else:
        noappend = bool(noappend)
    walltime = float(walltime)
    
    # bind resources
    self.bind_resources()
    
    # define stop condition
    t0 = time.time()
    fname = ''
    def stop_condition():
        nonlocal fname
        
        # maximum time
        if time.time() - t0 > walltime:
            self.termination_signal = 2  # sigint
            return True
        
        # new simulation
        if get_current_simulation(f'{self.directory}/{run_file}') != fname:
            print(f"Target simulation file changed ({now()}).")
            return True
        
        # do you have to interrupt?
        return bool(self.termination_signal)
    
    # main cycle
    while True:
        
        # received the signal / reached walltime
        if bool(self.termination_signal) or time.time() - t0 > walltime:
            break
        
        # what to simulate
        fname = get_current_simulation(f'{self.directory}/{run_file}')
        if not fname:
            continue  # no job assigned yet
        
        # worker is not ready anymore
        remove(f'{self.directory}/{run_file}.ready')
        
        # create command
        cmd = (f'{self.params.mdrun} -deffnm {fname} '
               f'-cpo {fname}.cpt -cpi {fname}.cpt -cpt .1 '
               f'{"-noappend" if noappend else ""}')
        
        # execute command
        print(f"Starting simulating {fname} ({now()})...")
        if exit := execute_command(cmd, stop_condition,
            termination_timeout=self.termination_timeout):
            os.system(f'touch {self.directory}/{run_file}.ready')
            raise RuntimeError(f'{cmd} failed with exit code {exit}')
        
        # worker is ready again
        os.system(f'touch {self.directory}/{run_file}.ready')
