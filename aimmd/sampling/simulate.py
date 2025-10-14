import os
import pty
import time
import numpy as np
import subprocess
from ..core.utils import get_current_simulation, now

def simulate(self, run_file, log_file=None, noappend=False, walltime=np.inf):
    """
    Continuously run simulations as directed by the run file.
    noappend: bool, add Gromacs' -noappend flag.
    """

    t0 = time.time()
    
    try:
        self.log_file = log_file
        
        # run continuously
        while True:
            
            # received the signal
            if self.interrupt:
                break
            
            print("Starting simulation loop...")
            if not log_file:
                print(f"Press Control+C to interrupt.")
            
            # maximum time
            if time.time() - t0 > walltime:
                break
            
            # interrupt everything currently running
            self.terminate_handler(report=False, exit=False)
            
            # what to simulate
            fname = get_current_simulation(f'{self.directory}/{run_file}')
            if not fname:
                continue  # no job assigned yet
            
            # (re)-start logging
            self.log = open(
                f'{self.directory}/{log_file}', "a+") if log_file else None
            print(f"Starting simulating {fname} ({now()})...")         
            
            # create command
            command = self.params.mdrun.split() + [
                "-deffnm", fname,
                "-cpo", f"{fname}.cpt",
                "-cpi", f"{fname}.cpt",
                "-cpt", ".1"]
            if noappend:
                command.append("-noappend")
                    
            # open pseudo-terminal to capture real-time stdout
            master_fd, slave_fd = pty.openpty()
            
            # run command
            self.process = subprocess.Popen(
                command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
                close_fds=True)
            os.close(slave_fd)
            
            try:
                with os.fdopen(master_fd) as stdout:
                    while True:
                        
                        # received the signal
                        if self.interrupt:
                            break
                        
                        if get_current_simulation(
                            f'{self.directory}/{run_file}') != fname:
                            print("Target simulation file changed ({now()}).")
                            break
                        
                        try:
                            line = stdout.readline()
                            print(line, end="")
                        except OSError:
                            # PTY closed: treat as EOF
                            break
            
            # catch any final PTY read errors cleanly
            except OSError:
                pass
    
    except SystemExit:
        self.terminate_handler()
    
    except KeyboardInterrupt:
        self.terminate_handler(exit=False)
    
    except Exception as exception:
        print(f'Exception: {exception}')
        self.terminate_handler()
    
    finally:
        self.terminate_handler(exit=False)
