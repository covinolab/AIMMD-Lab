"""
...
"""

# external
import os
import time
import numpy as np
import torch
import traceback
from abc import ABC
from glob import glob
from numbers import Integral
from pathlib import PosixPath
from MDAnalysis import Universe, Writer

# aimmd imports
from ..path import Path
from .._config import MDA_CACHE
from ..cache.npy import load_npy
from ..core.utils import randomize_velocities
from ..engines.toy import ToyEngine
from ..pathensemble import PathEnsemble
from ..execute.utils import execute_command

# params' methods
class ParamsMethods(ABC):

    def initialize_simulation(self, frame, *deffnm,
                              timeout=20., verbose=True):
        """Given shooting -> randomize velocities and save.
        If gromacs -> tps, otw trajectory with right extension.
        deffnm: without extension (velocities +/- 1)
        Works even with pathensemble.Path (in that case,
        take last frame; divide in parts if more than one frame)
        """

        previous_frames = []
        if isinstance(frame, Path):
            previous_frames = frame[:-1]
            previous_frames.times = -np.arange(1, len(frame)) * abs(frame.dt)
            frame = frame[-1]
        
        # gen_temperature
        if self.gen_temperature < 0 and hasattr(frame, 'velocities'):
            gen_temperature = 0
        else:
            gen_temperature = self.gen_temperature
        
        # get
        positions = frame.positions
        if gen_temperature < 0:
            velocities = -frame.velocities
        elif (masses := self.masses) is not None:
            velocities = randomize_velocities(masses, gen_temperature)
        else:
            velocities = np.zeros_like(positions)
            print('Warning: could not randomize velocities, '
                  'set them to zero, please update params.topology')
        dimensions = frame.dimensions
        n_atoms = len(positions)

        # timestep you'll write
        universe = Universe.empty(len(positions), trajectory=True)
        ts = universe.trajectory.ts
        ts.positions = positions
        ts.velocities = velocities
        ts.dimensions = dimensions
        ts.time = 0.
        
        for deffnm in deffnm:
            ts.velocities *= -1.0  # invert every time

            # gromacs
            if self.engine == 'gromacs':

                # write temporary frames
                if previous_frames:
                    fname = f'{deffnm}.part0000{self.trajectory_extension}'
                    previous_frames.write(fname)
                
                # use temporary file for grompp
                path = PosixPath(f'{deffnm}.tpr')
                temp = str(path.with_name(f'.{path.stem}.trr'))
                with Writer(temp, n_atoms) as writer:
                    writer.write(universe)
                
                # generate tpr
                execute_command(
                    f'{self.gmx_grompp} -nobackup -f {self.gmx_mdp} '
                    f'-r {self.topology} -c {self.topology} '
                    f'-o {deffnm}.tpr -t {temp}', walltime=timeout,
                    log_file='stdout' if verbose else None,
                    raise_if_failure=True)

            # toy: directly write
            if self.engine == 'toy':
                if previous_frames:
                    fname = f'{deffnm}.part0000{self.trajectory_extension}'
                    writer = previous_frames.write(fname, return_writer=True)
                else:
                    fname = f'{deffnm}{self.trajectory_extension}'
                    writer = Writer(fname, n_atoms)
                try:
                    writer.write(universe)
                finally:
                    writer.close()
                t0 = time.time()
                while time.time() - t0 < 10.:
                    reader = MDA_CACHE.load(fname)
                    if reader:
                        break
                if not reader:
                    raise TypeError(f'{fname!r} not correctly generated')
    
    def run_simulation(self, deffnm, backup=False, cpt=.1,
                       noappend=False, **kwargs):

        # gromacs
        if self.engine == 'gromacs':
            command = f'{self.gmx_mdrun} -deffnm {deffnm}'
            if not backup:
                command += ' -nobackup'
            if cpt:
                command += f' -cpi {deffnm}.cpt -cpt {cpt}'
            if noappend:
                command += ' -noappend'
            return execute_command(command, **kwargs)
        
        # toy: code here -> same as execute command
        if self.engine == 'toy':
            ToyEngine(self.toy_mdrun, self.toy_slowdown)(
                deffnm, backup=backup, noappend=noappend, **kwargs)
    
    def check_engine(self, topology='', deffnm='.check_engine', timeout=10):
        """Will be called by user if necessary.
        filename has no extension"""
        
        # go to the right folder
        cwd = PosixPath('.')
        os.chdir(self.parent)
        if not topology:
            if self._universe:
                ts = self._universe.trajectory[0]
            elif self.initial_paths and (path := self.initial_paths[0]):
                ts = path[0]
            else:
                raise TypeError('please provide topology')
        else:
            reader = MDA_CACHE.open(topology)
            if not reader:
                raise TypeError(f'{topology!r} is not a valid topology')
            ts = reader[0]
        
        try:  # cleanup
            os.system(f'rm -f {deffnm}*')
            
            # initialize 
            self.initialize_simulation(ts, deffnm, timeout=timeout)
            
            # run
            self.run_simulation(deffnm, walltime=timeout)
            
            # check
            assert os.path.getsize(f'{deffnm}{self.trajectory_extension}')
            
            # all fine
            return 0
        
        except Exception as exception:
            traceback.print_exc()
            return 1
        
        finally:  # cleanup and back to the original folder
            os.system(f'rm -f .params_check_engine*')
            os.chdir(cwd)
    
    def update_network(self, directory, timeout=20.,
                       raise_if_failure=True):
        """
        Returns
        -------
        bins, densities: associated to network model.
        """
        
        # find device
        device = next(self.network.parameters()).device

        # find name
        states = self.sorted_states
        network_fname = f'{directory}/network{states}.h5'
        
        # advance only if data are present
        t0 = time.time()
        while True:
            try:
                state_dict = torch.load(network_fname, map_location=device)
                self.network.load_state_dict(state_dict)
                return
            
            # error only after timeout
            except Exception as exception:
                if time.time() - t0 >= timeout:
                    if raise_if_failure:
                        raise exception
                    print(f'Warning: {exception}')
                    return
            time.sleep(.1)

    def load_bins_and_densities(self, directory,
        timeout=20., raise_if_failure=True):
        
        # find name
        states = self.sorted_states
        
        # advance only if data are present
        t0 = time.time()
        while True:
            try:
                bins = load_npy(f'{directory}/bins{states}.npy')
                densities = load_npy(f'{directory}/densities{states}.npy')
                if bins is None:
                    raise RuntimeError('could not load bins')
                if densities is None:
                    raise RuntimeError('could not load bins')
                assert len(bins.shape) == len(densities.shape) == 1
                if len(bins) - 1 != len(densities):
                    raise RuntimeError(
                        f'len(densities) = {len(densities)}, '
                        f'should be len(bins) - 1 = {len(bins) - 1}')
                return bins, densities
            
            # error only after timeout
            except Exception as exception:
                if time.time() - t0 >= timeout:
                    if raise_if_failure:
                        raise exception
                    print(f'Warning: {exception}')
                    return [], []
            time.sleep(.1)
    
    def copy(self):
        from . import Params
        copy = object.__new__(Params)
        copy.__dict__.update(self.__dict__)
        return copy

    def check_if_initialized(self, *deffnms):
        for deffnm in deffnms:
            if self.engine == 'gromacs':
                fname = f'{deffnm}.tpr'
                if not (os.path.exists(fname) and os.path.getsize(fname)):
                    return False
            if self.engine == 'toy':
                fname = f'{deffnm}{self.trajectory_extension}'
                if not (os.path.exists(fname) and os.path.getsize(fname)):
                    fname = f'{deffnm}.part0000{self.trajectory_extension}'
                    if not (os.path.exists(fname) and os.path.getsize(fname)):
                        return False
        return True
