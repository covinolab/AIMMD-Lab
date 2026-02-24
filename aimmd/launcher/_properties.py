"""
...
"""

# external
import os
import numpy as np
from abc import ABC

# Launcher properties
class LauncherProperties(ABC):
    
    @property
    def directories(self):
        return list(self._directories)

    @property
    def params(self):
        return list(self._params)

    @property
    def paths(self):
        return [params.path for params in self._params]

    @property
    def processes(self):
        return self._processes

    @property
    def job_stop_condition(self):
        """Bash script for stop condition in SLURM job"""
        return f'''stop_condition() {{
        local pids=("$@")
        
        while true; do
            # exit if .terminate file exists
            if [[ -f "{self.directories[0]}/.terminate" ]]; then
                break
            fi
            
            # exit if any PID in the list has terminated
            for pid in "${{pids[@]}}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break 2  # break both loops
                fi
            done
            
            # exit if exceeded (WALLTIME - {int(self.termination_timeout)}) s)
            current_time=$(date +%s)
            elapsed=$((current_time - START_TIME))
            if (( elapsed > WALLTIME - {int(self.termination_timeout)} )); then
                break
            fi
            
            sleep 1
        done
        
        # create terminate (if not existing already)
        touch "{self.directories[0]}/.terminate"
        
        # send termination signal to all PIDs
        for pid in "${{pids[@]}}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
        
        # wait for all to exit cleanly
        wait "${{pids[@]}}" 2>/dev/null
    }}'''
