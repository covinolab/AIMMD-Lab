import os
import pty
import subprocess
from ..core.utils import get_current_simulation

def simulate(self, run_file, log_file=None, append=False):
    """
    Continuously run simulations as directed by the run file.
    append: bool, if True: do not create new part (Gromacs' -noappend).
    """
    
    self.log = open(log_file, "a+") if log_file else None
    self.report("Starting simulation loop...")
    print(f"Press Control+C to interrupt.")
    
    try:
        
        # run continuously
        while True:
            # what to simulate
            fname = get_current_simulation(run_file)
            if not fname:
                continue  # no job assigned yet
            self.report(f"Starting simulating {fname}...")         
            
            # create command
            command = self.params.mdrun.split() + [
                "-deffnm", fname,
                "-cpo", f"{fname}.cpt",
                "-cpi", f"{fname}.cpt",
                "-cpt", ".1"]
            if not append:
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
                        
                        if get_current_simulation(run_file) != fname:
                            self.report("Simulation changed, terminating.")
                            try:
                                self.process.terminate()
                                self.process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                self.report("Process did not exit in time, killing...")
                                self.process.kill()
                            break
                        
                        try:
                            line = stdout.readline()
                            print(line, end="")
                            if self.log:
                                self.log.write(line)
                                self.log.flush()
                        except OSError:
                            # PTY closed: treat as EOF
                            break
                
                if self.process.poll() is None:
                    self.process.wait()
                self.process = None
            
            # catch any final PTY read errors cleanly
            except OSError:
                pass
    
    except SystemExit:
        self.terminate_handler()
    
    except KeyboardInterrupt:
        self.terminate_handler(exit=False)
    
    except Exception as exception:
        self.report(f'Exception: {exception}')
        self.terminate_handler()
