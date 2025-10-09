import os
import sys
import signal
import time
import numpy as np
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
        + 0.10 * (x - 0.5) * (0 < x < 1)  # deterministic drift
        + 0.08 * np.random.normal(),      # stochastic component
        -0.5, 1.5                         # hard boundaries
    )


def graceful_exit(signum=None, frame=None):
    global writer
    print('\nReceived termination signal.')
    try:
        if writer:
            writer.close()
            print('Writer closed successfully.')
    except Exception as error:
        print(f'Error while closing writer: {error}')
    finally:
        sys.exit(0)


if __name__ == '__main__':

    writer = None  # global reference for signal handlers
    
    if len(sys.argv) < 3:
        print('Usage: python integrator.py -deffnm output_name [-noappend]')
        sys.exit(1)
    
    # simulation file name without extension or part
    name = sys.argv[sys.argv.index('-deffnm') + 1]
    
    folder = '/'.join(name.split('/')[:-1])
    if not folder:
        folder = '.'
    fname = name.split('/')[-1]

    # register signals for clean shutdown
    signal.signal(signal.SIGINT, graceful_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_exit)  # kill <pid>
    signal.signal(signal.SIGHUP, graceful_exit)   # terminal hangup

    if '-noappend' not in sys.argv:
        # append to file
        file = f'{folder}/{fname}.xtc'
        print(f'Appending to {file}')
        universe = mda.Universe(topology, file, in_memory=True)
        atoms = universe.atoms
        
        # overwrite
        writer = mda.Writer(f'{folder}/{fname}.xtc', 1)
        for frame in universe.trajectory:
            writer.write(atoms)
        
    else:
        # retrieve last part
        old_file = sorted([file for file in os.listdir(folder) if
            file[:len(fname)] == fname and
            file[-12:-8] == 'part' and
            file[-8:-4].isdigit() and
            file[-4:] == '.xtc'])[-1]
        
        universe = mda.Universe(
            topology, f'{folder}/{old_file}', in_memory=True)
        atoms = universe.atoms

        # create new part
        file = f'{folder}/{fname}.part{int(old_file[-8:-4]) + 1:04d}.xtc'
        print(f'Generating {file}')
        
        writer = mda.Writer(file, 1)
    
    # get last time and position
    frame = universe.trajectory[-1]
    time = frame.time
    positions = frame.positions.copy()

    # simulation loop
    try:
        while True:
            sleep(slowdown)
            
            # update coordinates
            positions[0, 0] = run(positions[0, 0])
            time += 1.0

            # write frame
            frame.positions = positions
            frame.time = time
            writer.write(atoms)
    
    except KeyboardInterrupt:
        graceful_exit(signal.SIGINT)
