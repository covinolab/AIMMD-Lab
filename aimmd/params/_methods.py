"""
aimmd.params._methods
====================

Engine-dependent operational methods for :class:`aimmd.params.Params`.

This mixin provides high-level methods that *use* the configured engine and
execute real simulations or simulation-adjacent operations.

Provided functionality
----------------------
initialize_simulation(...)
    Create engine-specific input files for a given starting frame:

    - for GROMACS: write a temporary trajectory frame and run `grompp` to build
      a `.tpr`.
    - for toy engine: write a trajectory file directly.

run_simulation(...)
    Run a configured engine simulation segment:

    - for GROMACS: calls `mdrun` via `execute_command`.
    - for toy engine: uses `ToyEngine`.

minimize_energy(...)
    Energy-minimize each frame of a trajectory using GROMACS (writes back a
    minimized trajectory).

check_if_initialized(...)
    Check presence of required output files for initialized simulations.

check_engine(...)
    Lightweight end-to-end check that the engine can initialize and run,
    producing a trajectory file.

update_network(...)
    Wait for a network checkpoint file to appear, then load into the current
    `params.network`.

load_bins_and_densities(...)
    Wait for `.npy` files to appear and validate shapes/consistency.

copy()
    Shallow copy of Params (copies `__dict__`).

Notes
-----
- These methods assume that `Params` fields have already been validated.
- Many methods interact with the filesystem and are sensitive to working
  directory (`params.parent`) and file naming conventions.
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
from .._config import MDA_CACHE, EM_MDP
from ..cache.npy import load_npy
from ..core.utils import randomize_velocities, remove
from ..engines.toy import ToyEngine
from ..pathensemble import PathEnsemble
from ..execute.utils import execute_command


# params' methods
class ParamsMethods(ABC):

    def initialize_simulation(self, frame, *deffnm,
                              timeout=20., verbose=True):
        """
        Initialize engine inputs for a simulation started from a given frame.

        Parameters
        ----------
        frame : MDAnalysis Timestep-like or aimmd.path.Path
            Starting configuration.
            If a `Path` is provided, the last frame is used as the shooting frame,
            while preceding frames are written as `.part0000*` to preserve history
            in toy mode or for bookkeeping in GROMACS initialization.
        *deffnm : str
            One or more output basenames (without extension). For each `deffnm`,
            initialization is performed separately. The method flips velocities
            sign each time (useful for forward/backward branches).
        timeout : float, optional
            Walltime (seconds) used for the `grompp` call via `execute_command`.
        verbose : bool, optional
            If True, print `grompp` output to stdout.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If no `deffnm` is provided, if toy output cannot be read back, etc.

        Notes
        -----
        - Velocity generation:

          * if `params.gen_temperature < 0` and velocities are present: reuse them
            (with a sign flip for the first branch).
          * else randomize from `params.masses` if available.
          * else set zeros (with a warning).
        - For GROMACS:

          * writes a temporary `.trr` file to pass velocities to `grompp` via `-t`.
          * uses `params.topology` for both `-r` and `-c`.
        """
        if not len(deffnm):
            raise TypeError('at least one output name (deffnm) needed')

        previous_frames = []
        if isinstance(frame, Path):
            # If `frame` is a Path, use its last frame as the start and keep
            # the earlier frames (written as `.part0000...`).
            previous_frames = frame[:-1]
            previous_frames.times = -np.arange(1, len(frame)) * abs(frame.dt)
            frame = frame[-1]

        # gen_temperature
        if self.gen_temperature < 0 and hasattr(frame, 'velocities'):
            gen_temperature = 0
        else:
            gen_temperature = self.gen_temperature

        # get positions/velocities/dimensions
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

        # Create an in-memory Universe holding a single frame to write out.
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

                # write temporary frames (history segment)
                if previous_frames:
                    fname = f'{deffnm}.part0000{self.trajectory_extension}'
                    previous_frames.write(fname)

                # use temporary file for grompp (to pass velocities)
                path = PosixPath(f'{deffnm}.tpr')
                temp = str(path.with_name(f'.{path.stem}.trr'))
                with Writer(temp, n_atoms) as writer:
                    writer.write(universe)

                # generate tpr
                try:
                    execute_command(
                        f'{self.gmx_grompp} -nobackup -f {self.gmx_mdp} '
                        f'-r {self.topology} -c {self.topology} '
                        f'-o {deffnm}.tpr -t {temp}', walltime=timeout,
                        log_file='stdout' if verbose else None,
                        raise_if_failure=True)
                finally:
                    remove(temp, verbose=False)

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
                # robustness: wait briefly until reader is available via cache
                t0 = time.time()
                while time.time() - t0 < 10.:
                    reader = MDA_CACHE.load(fname)
                    if reader:
                        break
                if not reader:
                    raise TypeError(f'{fname!r} not correctly generated')

    def run_simulation(self, deffnm, backup=False, cpt=.1,
                       noappend=False, **kwargs):
        """
        Run a simulation segment for the configured engine.

        Parameters
        ----------
        deffnm : str
            Basename used by the engine (GROMACS `-deffnm`).
        backup : bool, optional
            If False, add `-nobackup` for GROMACS output control.
        cpt : float, optional
            If non-zero, enable checkpointing (`-cpi` and `-cpt`) for GROMACS.
        noappend : bool, optional
            If True, add `-noappend` for GROMACS to avoid appending to existing files.
        **kwargs
            Forwarded to `execute_command` (e.g., stop_condition, walltime, log_file).

        Returns
        -------
        int or None
            - For GROMACS: the exit code returned by `execute_command`.
            - For toy engine: returns None (ToyEngine performs its own loop).

        Notes
        -----
        - For toy engine, `ToyEngine(...)` is called with `**kwargs` to allow
          stop-condition / walltime semantics in Python.
        """
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

    def minimize_energy(self, trajectory, out, em_mdp=None):
        """
        Energy-minimize each frame of a trajectory using GROMACS.

        Parameters
        ----------
        trajectory : str or aimmd.path.Path or trajectory-like
            Input trajectory containing frames to minimize. Converted to `Path`.
        out : str or pathlib.Path
            Output filename for the minimized trajectory.
        em_mdp : str, optional
            Override `.mdp` file used for minimization. If None, uses `EM_MDP`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If engine is not GROMACS, or if output would overwrite an input file.

        Notes
        -----
        Implementation strategy:

        - Load positions into memory.
        - For each frame:

          * initialize a temporary run (`initialize_simulation`),
          * run the minimization (`run_simulation`),
          * read minimized coordinates from `<temp>.trr`,
          * clean up temp files.
        - Finally, write the minimized trajectory to `out`.
        """
        # only gromacs
        if self.engine != 'gromacs':
            raise TypeError('energy minimization supported only '
                            'with Gromacs engine')

        # convert to Path if not already
        trajectory = Path(trajectory)

        # is out file ok?
        out = PosixPath(out).resolve()
        for fname in trajectory.fnames:
            if PosixPath(fname).resolve() == out:
                raise TypeError(f"can't overwrite {fname!r} "
                                f"while performing energy minimization")

        # load positions in memory
        trajectory.positions = trajectory.positions

        # just here, use the provided em file (or the default one)
        gmx_mdp = self.gmx_mdp
        self.gmx_mdp = em_mdp or EM_MDP
        try:
            # read positions from temporary file
            for i, ts, fname, loc in zip(range(len(trajectory)),
                                         trajectory,
                                         trajectory.filenames,
                                         trajectory.locs):
                fname = PosixPath(fname)
                temp = str(fname.with_name(f'.{fname.name}_{loc}'))
                self.initialize_simulation(ts, temp)
                self.run_simulation(temp)
                trajectory.positions[i] = Path(f'{temp}.trr').positions[0]
                remove(f'{temp}*', verbose=False)
        finally:
            self.gmx_mdp = gmx_mdp

        # write to "out"
        trajectory.write(out, overwrite=True)

    def check_if_initialized(self, *deffnms):
        """
        Check whether required engine files exist for each deffnm.

        Parameters
        ----------
        *deffnms : str
            One or more simulation basenames.

        Returns
        -------
        bool
            True if all required files exist and are non-empty.

        Notes
        -----
        - For GROMACS: checks `<deffnm>.tpr`.
        - For toy engine: checks `<deffnm><trajectory_extension>` or
          `<deffnm>.part0000<trajectory_extension>`.
        """
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

    def check_engine(self, topology='', deffnm='.check_engine', timeout=10):
        """
        Run a minimal engine self-test in `params.parent`.

        Parameters
        ----------
        topology : str, optional
            Topology/structure file used to obtain a starting frame.
            If empty, attempts to use:
            - `self._universe` first,
            - then the first frame of the first initial path.
        deffnm : str, optional
            Output basename (no extension).
        timeout : float, optional
            Walltime (seconds) used for initialization and execution.

        Returns
        -------
        int
            0 on success, 1 on failure.

        Notes
        -----
        This method:

        - switches to `params.parent`,
        - removes old `deffnm*` files,
        - initializes and runs a short segment,
        - verifies that the expected trajectory file exists and has non-zero size.
        """
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

    def update_network(self, path, timeout=20.,
                       raise_if_failure=True):
        """
        Load a network checkpoint from disk into `self.network`.

        Parameters
        ----------
        path : str
            If path to file: path containing the checkpoint file.
            If path to directory: directory containing the
            `network{params.states}.h5` checkpoint file.
        timeout : float, optional
            Maximum wait time (seconds) for the checkpoint to become readable.
        raise_if_failure : bool, optional
            If True, re-raise the final exception after timeout; otherwise print
            a warning and return.

        Returns
        -------
        None

        Notes
        -----
        The checkpoint name depends on `self.sorted_states` (ensures stable naming
        independent of A/B ordering).
        """
        # find device
        device = next(self.network.parameters()).device
        
        # find name
        if os.path.isfile(path):
            network_fname = path
        elif os.path.isdir(path):
            states = self.sorted_states
            network_fname = f'{path}/network{states}.h5'
        elif raise_if_failure:
            raise FileNotFoundError(f'{path!r} does not exist')
        else:
            print(f'Warning: {path!r} does not exist')
        
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
    
    def copy(self):
        """
        *Shallow*-copy this Params object. Only force reload of inital paths.

        Returns
        -------
        aimmd.params.Params
            A new Params instance with a copied `__dict__` (shallow copy).

        Notes
        -----
        This does not deep-copy mutable objects referenced in `__dict__`.
        """
        from . import Params
        copy = object.__new__(Params)
        copy.__dict__.update(self.__dict__)
        copy.initial_paths = self.initial_paths  # force reload
        return copy
