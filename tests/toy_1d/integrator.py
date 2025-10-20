import os
import sys
import numpy as np
import signal
import shutil
import traceback
import MDAnalysis as mda
from time import sleep

# silence mdanalysis
import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')
logging.getLogger('MDAnalysis').setLevel(logging.ERROR)
logging.getLogger('MDAnalysis.coordinates').setLevel(logging.ERROR)
logging.getLogger('MDAnalysis.topology').setLevel(logging.ERROR)
logging.getLogger('MDAnalysis.analysis').setLevel(logging.ERROR)

"""
Integrator for 1D system compatible with worker.sh.
"""

slowdown = 0.01  # seconds between frames, to avoid imbalances
topology = 'run.gro'  # in main folder

def run(x):
    return np.clip(
        x
        + 0.10 * (x - 0.5) * (0 < x) * (x < 1)  # deterministic drift
        + 0.08 * np.random.normal(),            # stochastic component
        -0.5, 1.5                               # hard boundaries
    )


if __name__ == '__main__':
    
    # cannot run
    if len(sys.argv) < 3:
        print('Usage: python integrator.py -deffnm output_name [-noappend]')
        sys.exit(1)
    
    # simulation file name without extension or part, folder
    name = sys.argv[sys.argv.index('-deffnm') + 1]
    folder = '/'.join(name.split('/')[:-1])
    if not folder:
        folder = '.'
    fname = name.split('/')[-1]
    
    # append to file
    if '-noappend' not in sys.argv:
        file = f'{folder}/{fname}.xtc'
        trajectory = mda.Universe(topology, file, in_memory=True).trajectory
        backup = f'{folder}/{fname}_backup.xtc'
        print(f'Appending to {file}')        
        shutil.copy(file, backup)  # avoid horrible things
        
    else:
        # retrieve last part
        old_file = sorted([file for file in os.listdir(folder) if
            file[:len(fname)] == fname and
            file[-12:-8] == 'part' and
            file[-8:-4].isdigit() and
            file[-4:] == '.xtc'])[-1]
        trajectory = mda.Universe(
            topology, f'{folder}/{old_file}', in_memory=True).trajectory
        file = f'{folder}/{fname}.part{int(old_file[-8:-4]) + 1:04d}.xtc'
        backup = ''
        write(f'Creating new file {file}')
    
    # retrieve last frame's info
    frame = trajectory[-1]
    positions = frame.positions.copy()
    n_atoms = len(positions)
    time = frame.time + 0.
    
    # create an empty *new* Universe with n_atoms (no topology required)
    universe = mda.Universe.empty(n_atoms, trajectory=True)
    
    # create writer
    writer = mda.Writer(file, n_atoms)
    
    # graceful exit operation
    def exit(*args):
        global backup
        try:
            writer.close()
            print('Writer closed successfully.')
        except Exception as error:
            print(f'Error while closing writer: {error}')
        finally:
            if backup:
                print(f'Restoring backup.')
                shutil.copy(backup, file)
            sys.exit(0)
    
    # register signals for clean shutdown
    signal.signal(signal.SIGINT, exit)  # Ctrl+C
    signal.signal(signal.SIGTERM, exit) # kill <pid>
    signal.signal(signal.SIGHUP, exit)  # terminal hangup
    
    # procedure
    def _write(positions, time):
        universe.trajectory.ts.positions = positions
        universe.trajectory.ts.time = time
        writer.write(universe)
    
    # re-write positions before appending
    if backup:
        for frame in trajectory:
            _write(frame.positions, frame.time)
        backup = ''  # not needed anymore    
    
    # simulation loop
    try:
        while True:
            
            # wait
            sleep(slowdown)
            
            # update coordinates
            positions = run(positions)
            time += 1.0
            
            # write frame
            _write(positions, time)
            
            # report
            print('Reached time', time)
    
    except KeyboardInterrupt:
        exit(signal.SIGINT)
    
    except Exception as exception:
        print(exception)
        traceback.print_exc()
        raise exception

