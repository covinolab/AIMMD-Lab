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
from .utils import (FREE_RESTART_IN_STATE_SOURCES, SEEDING_POSITION_ALIASES,
                    legacy_transitions_replacement, parse_restart_source,
                    parse_seeding_position)
from .._config import MDA_CACHE, EM_MDP
from ..cache.npy import load_npy
from ..core.utils import process_state, randomize_velocities, remove
from ..engines.toy import ToyEngine
from ..pathensemble import PathEnsemble
from ..execute.utils import execute_command



def _colvar_rowcount(fname):
    """Number of data (non-comment) rows in a COLVAR-style file."""
    with open(fname) as fh:
        return sum(1 for line in fh if not line.startswith('#'))


def _part_frame_count(pf):
    """Readable frame count of a trajectory part, or 0 if it cannot be opened."""
    reader = MDA_CACHE.get(pf)
    if reader is None:
        return 0
    try:
        return len(reader)
    except TypeError:
        return 0


def _part_row_ranges(deffnm_dir, deffnm_base, ext, n_rows,
                     allow_partial=False):
    """Map each trajectory part to its row range in the cumulative COLVAR.

    The part-to-row bookkeeping that :func:`_split_cumulative_colvar` performs,
    factored out so the training worker's out-of-cache bias fallback shares it
    rather than carrying a parallel implementation that can drift.

    Parameters
    ----------
    deffnm_dir : str
        Directory holding the cumulative ``COLVAR`` and the part files.
    deffnm_base : str
        Trajectory basename, e.g. ``'traj000002'``.
    ext : str
        Trajectory extension including the dot, e.g. ``'.xtc'``.
    n_rows : int
        Row count of the cumulative COLVAR. Required, so both callers agree on
        one row budget read one way.
    allow_partial : bool, default False
        When the cumulative COLVAR has fewer rows than the parts need, the
        default is to stop at that part — matching the slicer, for which a
        short slice would be taken as final and would shift every later part.
        With True, the **last** part may instead be truncated to the rows that
        exist, so a read-only caller loses a flush lag's worth of frames rather
        than the whole segment. Earlier parts are never truncated.

    Returns
    -------
    dict
        ``{part_number: (start, stop, already_sliced)}``, ``stop`` exclusive.
        ``already_sliced`` marks parts whose ``_COLVAR`` is on disk and whose
        row count was therefore taken as authoritative. ``part0000`` (the
        python-written seed, for which PLUMED produced no rows) is absent, as
        are parts the row budget cannot reach.
    """
    parts = []
    for pf in sorted(glob(os.path.join(
            deffnm_dir, f'{deffnm_base}.part????{ext}'))):
        num = int(os.path.basename(pf)[len(deffnm_base) + len('.part'):][:4])
        if num == 0:
            continue  # seed frame: written by python, no PLUMED rows
        parts.append((num, pf))

    ranges = {}
    cursor = 0
    for i, (num, pf) in enumerate(parts):
        per_part_colvar = pf.replace(ext, '_COLVAR')
        if os.path.exists(per_part_colvar):
            n = _colvar_rowcount(per_part_colvar)
            ranges[num] = (cursor, cursor + n, True)
            cursor += n
            continue

        n = _part_frame_count(pf)
        if n == 0:
            continue  # unopenable or empty: contributes no rows either way

        if cursor + n > n_rows:
            if allow_partial and i == len(parts) - 1 and cursor < n_rows:
                ranges[num] = (cursor, n_rows, False)
            break
        ranges[num] = (cursor, cursor + n, False)
        cursor += n

    return ranges


def _split_cumulative_colvar(deffnm_dir, deffnm_base, ext):
    """Slice a cumulative PLUMED COLVAR file into per-part `_COLVAR` files.

    Background
    ----------
    PLUMED 2.10 (and earlier) writes a single accumulating ``COLVAR`` file
    per chain dir (the file specified by ``PRINT FILE=COLVAR`` in plumed.dat),
    even when GROMACS uses ``-noappend``. With ``RESTART=YES`` set globally
    (e.g. by ``OPES_METAD ... RESTART=YES``) the COLVAR is appended to across
    GROMACS restarts. PLUMED does NOT mirror GROMACS' ``.partNNNN`` naming.

    AIMMD's bias-tracking pipeline expects one ``{deffnm}.partNNNN_COLVAR``
    file per trajectory part (so ``bias_function(traj.partNNNN.xtc)`` can
    locate it). This helper bridges the gap, using :func:`_part_row_ranges`
    to resolve which rows belong to which part.

    Only parts that do not already have a ``_COLVAR`` are written, and never from
    fewer rows than they have frames: a published slice is treated as final by
    later calls, so a partial one would shift every later part's offset.

    The cumulative ``COLVAR`` is left in place: subsequent GROMACS restarts
    keep appending to it, and a later call needs to see all rows.

    Parameters
    ----------
    deffnm_dir : str
        Directory containing the cumulative ``COLVAR`` and the
        ``{deffnm_base}.partNNNN{ext}`` trajectory parts.
    deffnm_base : str
        Trajectory basename (e.g. ``'traj000001'``).
    ext : str
        Trajectory extension including the dot (e.g. ``'.xtc'``).
    """
    cum_colvar = os.path.join(deffnm_dir, 'COLVAR')
    if not os.path.exists(cum_colvar):
        return  # nothing to slice

    # Read header (first comment line) and all data rows
    header = '#! FIELDS time'  # safe fallback if no header is present
    with open(cum_colvar) as fh:
        for line in fh:
            if line.startswith('#'):
                header = line.rstrip('\n')
                break
    rows = np.loadtxt(cum_colvar, comments='#')
    if rows.ndim == 1:
        rows = rows[None, :]
    if len(rows) == 0:
        return

    ranges = _part_row_ranges(deffnm_dir, deffnm_base, ext, len(rows))

    for num, (start, stop, already_sliced) in sorted(ranges.items()):
        if already_sliced:
            continue
        per_part_colvar = os.path.join(
            deffnm_dir, f'{deffnm_base}.part{num:04d}_COLVAR')
        with open(per_part_colvar, 'w') as fh:
            fh.write(header + '\n')
            for row in rows[start:stop]:
                fh.write(' '.join(f'{v:.6f}' for v in row) + '\n')

    # Report a part the row budget could not reach, so a persistent shortfall
    # (e.g. PRINT STRIDE not matching nstxout-compressed) is visible.
    consumed = max((stop for _, stop, _ in ranges.values()), default=0)
    for pf in sorted(glob(os.path.join(
            deffnm_dir, f'{deffnm_base}.part????{ext}'))):
        num = int(os.path.basename(pf)[len(deffnm_base) + len('.part'):][:4])
        if num == 0 or num in ranges:
            continue
        n = _part_frame_count(pf)
        if n:
            print(f'!!! cumulative COLVAR {cum_colvar!r} has only '
                  f'{len(rows) - consumed} unconsumed rows but {pf!r} has '
                  f'{n} frames; skipping (will retry next call)')
            break
        # A zero-frame or unopenable part is also absent from `ranges`; it is not
        # the shortfall, so keep looking rather than silencing the diagnostic.


# params' methods
class ParamsMethods(ABC):

    def _free_restart_spec(self):
        """
        Resolve `free_restart_source` and its deprecated predecessor.

        Returns
        -------
        default : str
            Restart source for states not named explicitly.
        per_state : dict
            ``{state letter: source}`` for the states that were named.

        Raises
        ------
        TypeError
            If both `free_restart_source` and the deprecated
            `restart_free_simulations_with_transitions` ask for something.
        """
        spec = getattr(self, 'free_restart_source', 'crossing')
        legacy = getattr(self, 'restart_free_simulations_with_transitions', '')
        legacy = str(legacy or '').replace(' ', '')

        if legacy:
            replacement = legacy_transitions_replacement(legacy)
            default, per_state = parse_restart_source(spec, states=self.states)
            if default != 'crossing' or per_state:
                raise TypeError(
                    f"'free_restart_source' = {spec!r} and the deprecated "
                    f"'restart_free_simulations_with_transitions' = "
                    f"{legacy!r} both select a free-simulation restart "
                    f"source. Keep only 'free_restart_source'; the deprecated "
                    f"flag is equivalent to {replacement!r}.")
            spec = replacement

        return parse_restart_source(spec, states=self.states)

    def free_restart_source_for(self, state):
        """
        Where free simulations of *state* take their restart configuration.

        Resolves `free_restart_source`, honouring the deprecated
        `restart_free_simulations_with_transitions` when that is the one set.

        Parameters
        ----------
        state : int or str
            State index into `params.states`, or the state label itself.

        Returns
        -------
        str
            One of `aimmd.params.utils.FREE_RESTART_SOURCES`. The in-state
            sources are never returned for the reactive state `states[1]`,
            where they are undefined: a bare in-state default leaves the
            reactive state on ``'crossing'``.

        Raises
        ------
        TypeError
            If both the field and its deprecated predecessor are set.
        """
        state = process_state(state, self.states)
        default, per_state = self._free_restart_spec()
        source = per_state.get(state, default)
        if (source in FREE_RESTART_IN_STATE_SOURCES
                and len(self.states) >= 2 and state == self.states[1]):
            return 'crossing'
        return source

    def free_seeding_position_for(self, state):
        """
        Where the first free simulation of *state* starts inside the state.

        Parameters
        ----------
        state : int or str
            State index into `params.states`, or the state label itself.

        Returns
        -------
        float or str
            A fraction in [0, 1] over the state's run of initial-path frames,
            ordered far-side-first, or ``'random'``. Always the default for the
            reactive state `states[1]`, which has no in-state run.
        """
        state = process_state(state, self.states)
        default, per_state = parse_seeding_position(
            getattr(self, 'free_seeding_position', 'boundary'),
            states=self.states)
        if len(self.states) >= 2 and state == self.states[1]:
            return SEEDING_POSITION_ALIASES['boundary']
        return per_state.get(state, default)

    def untrimmed_initial_paths(self):
        """
        The initial paths as the user gave them, before the transition trim.

        `_process_and_check` replaces every entry of `initial_paths` by its
        transition block, which starts at the last in-state frame before the
        reactive region. Every `free_seeding_position` other than ``'boundary'``
        needs the frames that removes, so the source files are read back here.

        Where the names come from depends on who is asking:

        - in the launching process `initial_paths` is populated, and each
          (trimmed) entry still carries the `fname` of the user's file;
        - in a worker it is empty, because `aimmd/worker/_helpers.py` loads
          params with ``initial_paths=None`` so that a node's slots do not each
          re-read and re-trim the trajectory. The names then come from
          ``_initial_path_files``, recorded by `Params.load` while it was
          chdir'd into the params folder and so resolved by exactly the rule
          the field itself uses - a bare name, a ``'../'`` name, an absolute
          name and a glob pattern all behave the same way.

        Nothing is cached on disk and nothing is written at build time, so a
        hand-edited params.py takes effect on an already-built run folder
        without a rebuild. The cost is one `states_function` pass per worker,
        paid only when a non-default `free_seeding_position` is set.

        Returns
        -------
        dict
            ``{base name of the source file: untrimmed Path}``, with `states`
            computed. Keyed by name rather than by position because a worker
            does not hold `params.initial_paths`: it reads its own copies back
            from ``<run>/initial<states>/``, in whatever order the glob gives
            and with the trajectory extension appended.

        Notes
        -----
        Multi-system runs keep one `PathEnsemble` per system in
        `initial_paths`; those are flattened here, which is safe because the
        source base names are what the lookup matches on.
        """
        groups = getattr(self, 'initial_paths', None) or []
        if isinstance(groups, PathEnsemble):
            sources = [path.fname for path in groups]
        elif groups:
            sources = [path.fname for group in groups for path in group]
        else:
            # A worker loads params with `initial_paths=None`, so the field is
            # empty there and the names come from what the params FILE said,
            # recorded at load time and already resolved to absolute paths.
            sources = list(self.__dict__.get('_initial_path_files') or [])
            if not sources:
                raise TypeError(
                    "cannot reach the untrimmed initial paths: this Params has "
                    "neither 'initial_paths' nor the file names recorded at "
                    "load time. A free_seeding_position other than 'boundary' "
                    "needs them; check that params.py assigns 'initial_paths'.")

        out = {}
        for source in sources:
            base = os.path.basename(str(source))
            if base in out:
                continue
            if not os.path.exists(source):
                raise TypeError(
                    f"the initial path {source!r} named by params.py is not "
                    f"there. A free_seeding_position other than 'boundary' "
                    f"reads the frames the transition trim removed, so the "
                    f"file itself is needed - the copy in the run's "
                    f"initial<states>/ folder is already trimmed.")
            path = Path(source)
            # A cache may sit beside the file, but it is not necessarily
            # usable: an unpopulated one reads back as an array of EMPTY
            # strings rather than raising, so validate instead of merely
            # catching. Recomputing costs one states_function pass over a
            # path of tens of frames.
            usable = False
            try:
                cached = np.asarray(path.states)
                usable = (len(cached) == len(path)
                          and cached.dtype.kind in 'US'
                          and all(len(str(label)) == 1
                                  and str(label) in self.states
                                  for label in cached))
            except Exception:
                usable = False
            if not usable:
                path.states = path.compute(self.states_function)
            out[base] = path
        return out

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
            # Resolve the chain subdirectory and the basename of deffnm.
            # We cd into the chain dir before invoking mdrun so that PLUMED's
            # FILE=COLVAR (and FILE=COLVAR.partNNNN in noappend mode) land in
            # the chain dir rather than the shared cwd.  This eliminates the
            # race condition that occurs when 5 shooting workers all write to
            # the same COLVAR in the process working directory.
            deffnm_abs = os.path.abspath(deffnm)
            deffnm_dir = os.path.dirname(deffnm_abs)
            deffnm_base = os.path.basename(deffnm_abs)
            cmd_parts = [f'cd {deffnm_dir}',
                         f'{self.gmx_mdrun} -deffnm {deffnm_base}']
            if not backup:
                cmd_parts[-1] += ' -nobackup'
            if cpt:
                cmd_parts[-1] += f' -cpi {deffnm_base}.cpt -cpt {cpt}'
            if noappend:
                cmd_parts[-1] += ' -noappend'
            command = ' && '.join(cmd_parts)
            # PLUMED's analog of GROMACS `-nobackup`: PLUMED unconditionally
            # rotates an existing PLUMED.OUT to bck.N.PLUMED.OUT on each new
            # mdrun and hard-aborts at 100 backups. Delete the previous log
            # so PLUMED has nothing to rotate. Nothing in AIMMD reads
            # PLUMED.OUT (bias values are read from COLVAR), so the log can
            # be discarded between segments.
            plumed_out = os.path.join(deffnm_dir, 'PLUMED.OUT')
            if os.path.exists(plumed_out):
                os.remove(plumed_out)
            # Same rationale for PLUMED's per-file backups: PLUMED rotates any
            # existing output file (COLVAR, KERNELS, ...) to bck.N.<file> and
            # hard-aborts at 100 backups. Shooting dirs never hit this (their
            # COLVAR is renamed to {deffnm}_COLVAR after each segment, below),
            # but every free{t} folder shares one COLVAR that PLUMED reopens
            # non-restart at each new trajectory, so bck.N.COLVAR would pile up
            # to 100 and crash the run. The per-part _COLVAR slices are cached
            # before any backup is made and nothing in AIMMD reads bck.* files,
            # so clear them before every mdrun.
            for _bck in glob(os.path.join(deffnm_dir, 'bck.*')):
                os.remove(_bck)
            result = execute_command(command, **kwargs)
            # PLUMED bias: rename COLVAR → {deffnm}_COLVAR so that back and
            # forward segments (and successive free-traj parts) never clobber
            # each other.  Only active when record_bias='file'.
            if (getattr(self, 'record_bias', False)
                    and getattr(self, 'bias_source', '') == 'file'):
                if not noappend:
                    # shoot mode: PLUMED writes COLVAR in the chain dir
                    chain_colvar = os.path.join(deffnm_dir, 'COLVAR')
                    if os.path.exists(chain_colvar):
                        os.rename(chain_colvar, f'{deffnm_abs}_COLVAR')
                else:
                    # free mode: handle two PLUMED naming conventions.
                    # 1) Some PLUMED builds mirror GROMACS noappend numbering
                    #    and produce COLVAR.partNNNN files — rename those.
                    for colvar_f in sorted(glob(
                            os.path.join(deffnm_dir, 'COLVAR.part????'))):
                        part = os.path.basename(colvar_f)[len('COLVAR'):]
                        dest = f'{deffnm_abs}{part}_COLVAR'
                        if not os.path.exists(dest):
                            os.rename(colvar_f, dest)
                    # 2) PLUMED 2.10 (default) writes a single accumulating
                    #    COLVAR file regardless of GROMACS noappend. Slice it
                    #    into per-part {deffnm}.partNNNN_COLVAR files.
                    _split_cumulative_colvar(
                        deffnm_dir, deffnm_base, self.trajectory_extension)
            return result

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

    def _network_fname(self, directory):
        """
        Resolve the network checkpoint filename for a worker directory.

        Single-system (and multi-system without a shared network): the network
        lives in the given directory, ``<directory>/network{states}.h5``.

        Multi-system with a shared network (`multi_system_share_network=True`):
        the ONE shared network lives at the run-folder root, so when
        ``directory`` is a per-system subfolder (its basename is one of
        ``system_ids``) the filename is resolved one level up. The trainer runs
        at the run root, so it already resolves to the root directly.
        """
        states = self.sorted_states
        if (getattr(self, 'multi_system', False)
                and getattr(self, 'multi_system_share_network', False)
                and os.path.basename(os.path.normpath(directory))
                    in (self.system_ids or [])):
            directory = os.path.dirname(os.path.normpath(directory))
        return f'{directory}/network{states}.h5'

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
            network_fname = self._network_fname(path)
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

    def load_bins_and_densities(self, directory,
        timeout=20., raise_if_failure=True):
        """
        Load bin boundaries and densities from `.npy` files.

        Parameters
        ----------
        directory : str
            Directory containing `bins{states}.npy` and `densities{states}.npy`.
        timeout : float, optional
            Maximum wait time (seconds) for files to appear and be readable.
        raise_if_failure : bool, optional
            If True, raise after timeout; otherwise warn and return empty arrays.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            bins : 1D array of bin boundaries (length nbins+1 typically)
            densities : 1D array of per-bin densities (length len(bins)-1)

        Raises
        ------
        Exception
            Any final exception after timeout if `raise_if_failure` is True.

        Notes
        -----
        This function validates:

        - bins and densities are both 1D,
        - `len(densities) == len(bins) - 1`.
        """
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
