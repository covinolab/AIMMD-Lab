"""
aimmd.params._paths
==================

Path/ensemble loading utilities for :class:`aimmd.params.Params`.

This mixin provides methods that (re)loads Path and Pathensemble
objects from an AIMMD run directory layout.

High-level methods
------------------
free_trajectories(directory)
    Collect unsplit free simulation trajectories (potentially composed of parts)
    across states, and apply "indicted" exclusions.

shot_paths(directory, prefix='chain', ...)
    Collect shot paths from shooting-chain folders, optionally updating existing
    cached ensembles.

shot_chains(directory, ...)
    Convenience wrapper around `shot_paths(..., prefix='chain', ...)`.

pathensemble(directory, shot_chains=[])
    Assemble a complete PathEnsemble containing:

    - shot chains (ordered and optionally reusable),
    - free trajectories.

Notes
-----
These methods rely heavily on AIMMD's on-disk naming conventions (folder names,
`traj??????.part????`, `path??????`, and optional `tps_weights.npy` in case of a
TPS run).
"""

# external
import os
import numpy as np
from abc import ABC
from glob import glob
from math import nan
from numbers import Integral
from collections.abc import Iterable

# aimmd imports
from ..path import Path
from ..cache.npy import load_npy
from ..core.utils import process_state
from ..pathensemble import PathEnsemble
from ..pathensemble.utils import assemble_pathensemble


# params' paths loading methods
class ParamsPaths(ABC):

    def free_trajectories(self, directory, target_state=None, old=None):
        """
        Collect (unsplit) free trajectories from an AIMMD run directory.

        Parameters
        ----------
        directory : str
            Base run directory containing `free{target_state}/traj??????.part????`
            trajectory files.
        target_state : str, optional
            If None, load for all states in `self.states`.
            Otherwise interpreted via `process_state(target_state, self.states)`.
        old : iterable of aimmd.path.Path, optional
            Free trajectories from a previous call. Any whose part files are
            byte-for-byte unchanged is returned as the *same object*, so nothing
            is re-read from disk. Without this the trainer rebuilds every free
            trajectory every round, and `Path(fnames, ...)` resolves to
            `min_length=inf`, which makes the MDA reader cache a guaranteed miss
            and re-walks every part file.

            Reuse is conditional, unlike `shot_paths`. A shot path is immutable
            once registered; a free trajectory *grows* -- new parts appear and
            `gmx mdrun` appends to the last one in place. So a trajectory is
            only reused when both its tuple of part filenames and the size of
            its last part are unchanged; otherwise it is rebuilt. Trajectory
            files are append-only, so size is a sound signal here.
        
        Returns
        -------
        list of aimmd.path.Path
            List of reconstructed free trajectories as `Path` objects.
            NOT a `PathEnsemble` object.
        
        Notes
        -----
        - Free trajectories are stored split into parts:

          `traj??????.part????{ext}`.
          This method groups by `traj??????` index and assembles each into a
          `Path(fnames, remove_overlapping_frames=True)`.
        
        - If `indicted.log` is present in `free{state}/`, it is parsed and
          used to exclude frames from trajectories by assigning `_exclude_from`
          or `exclude_from` (depending on code path).
        """

        # initialize
        result = []

        # Index any offered trajectories by their part signature. A miss just
        # means we rebuild, so a stale or partial `old` can only cost time.
        reusable = {}
        for old_traj in (old or []):
            sig = getattr(old_traj, '_aimmd_parts_sig', None)
            if sig is not None:
                reusable.setdefault(sig, old_traj)

        def _parts_sig(part_fnames):
            """Identity of a free trajectory on disk: which parts, and how long
            the last one is (the only one that can still be appended to)."""
            try:
                return (tuple(part_fnames), os.path.getsize(part_fnames[-1]))
            except OSError:
                return None

        def _assemble(part_fnames, name, indicted_map, attr):
            """Reuse the previous object when nothing changed, else rebuild."""
            sig = _parts_sig(part_fnames)
            traj = reusable.get(sig) if sig is not None else None
            if traj is None:
                traj = Path(part_fnames, remove_overlapping_frames=True)
                traj._aimmd_parts_sig = sig
            # re-apply the exclusion every time: indicted.log can change even
            # when the trajectory files have not.
            if name in indicted_map:
                setattr(traj, attr, indicted_map[name])
            return traj

        # get "offset" for determining the path number
        ext = self.trajectory_extension
        offset = len(ext) + 9
        
        # find allowed folders
        if target_state is None:
            states = self.states
        else:
            states = process_state(target_state, self.states)
        
        # iterate on allowed folders
        for t in states:
            # which paths are indicted?
            indicted = {}
            if os.path.exists(f'{directory}/free{t}/indicted.log'):
                with open(f'{directory}/free{t}/indicted.log') as file:
                    for line in file:
                        fields = line.split()
                        if not fields:
                            continue
                        if len(fields) == 1:
                            indicted[fields[0]] = 0
                        else:
                            indicted[fields[0]] = int(fields[1])
            active = ''
            pattern = f'{directory}/free{t}/traj??????.part????{ext}'
            fnames = []
            for fname in sorted(glob(pattern)):
                current = fname[-offset-6:-offset]
                if active != current and fnames:
                    try:
                        result.append(_assemble(
                            fnames, f'traj{active}', indicted, '_exclude_from'))
                    except:
                        continue
                    fnames = [fname]
                else:
                    fnames.append(fname)
                active = current
            # last path
            if fnames:
                try:
                    result.append(_assemble(
                        fnames, f'traj{current}', indicted, 'exclude_from'))
                except:
                    continue

        # all together, as a list
        return result

    def shot_paths(self, directory, prefix='chain',
                   target_state=None, k=None, old=PathEnsemble()):
        """
        Load shot paths (shooting trajectories) from a run directory.

        Parameters
        ----------
        directory : str
            Base run directory containing `{prefix}{state}{k}/path??????{ext}`.
        prefix : str, optional
            Folder prefix (e.g., 'chain', 'sweep', etc.).
        target_state : str, optional
            If None, load for all states in `self.states`.
            Otherwise interpreted via `process_state(target_state, self.states)`.
        k : int, str, iterable, or None, optional
            Shooting chain index/indices to load.

            - int/str: load exactly that chain.
            - iterable: load those chains.
            - None: scan all matching folders and load all chains found.
        old : `PathEnsemble`, optional
            Previously loaded data used for incremental updates.
        
        Returns
        -------
        aimmd.pathensemble.PathEnsemble or list
            For a specific `(target_state, k)` returns a `PathEnsemble`.
            For a broader query returns a list indexed by k and/or state.

        Notes
        -----
        - This function tries to reuse already-loaded paths and only append new
          ones (except the last path which may still be changing on disk).

        - For TPS (`self.chain_type == 'tps'`) it may load `tps_weights.npy` and
          assign weights to newly loaded paths, zeroing weights for non-transitions.
        """
        
        # process "old" (do not copy paths)
        old = assemble_pathensemble(old)
        
        # which state are we talking about?
        states = self.states

        # load all of them
        if target_state is None:
            result = []
            for t in states:
                this = self.shot_paths(directory, prefix, t, k, old)
                if isinstance(this, list):
                    result.extend(this)
                elif this:
                    result.append(this)
            return result

        t = process_state(target_state, states)

        if isinstance(k, (Integral, str)):
            try:
                k = int(k)
            except:
                raise TypeError(f'{k} must be integral, list of integrals, '
                                f'or None, got {k!r}')

            # get info
            folder = f'{directory}/{prefix}{t}{k}'
            ext = self.trajectory_extension

            # initialize shot paths
            shot_paths = PathEnsemble()

            # Index `old` by filename once, instead of rescanning it for every
            # globbed file. The scan was O(M^2) with `Path.fname` as the match
            # key -- itself a property calling another property, 147.6 ns a hit
            # -- which reached 9.55 s at M=11,232 candidates against 3.1 ms for
            # this dict. `setdefault` reproduces the scan's `break` exactly:
            # first match wins.
            #
            # NB this evaluates `.fname` on every candidate, whereas the scan
            # stopped at the first match, so a malformed Path whose `fname`
            # raises now raises here rather than possibly never being touched.
            old_by_fname = {}
            for old_path in old._paths:
                old_by_fname.setdefault(old_path.fname, old_path)

            # iterate through matches
            for fname in sorted(glob(f'{folder}/path??????{ext}')):
                # get it from "old" (will not create copies)
                path = old_by_fname.get(fname)

                # must create new
                if path is None:
                    path = Path(fname, shooting_index='find')
                    if self.chain_type != 'tps':
                        path.weight = path.is_complete(t, states)
                    else:
                        path.weight = nan  # will fill later
                
                # add to shot_paths
                shot_paths.append(path)
            
            # get weights in case of tps
            if 'sweep' not in prefix and self.chain_type == 'tps':
                try:
                    saved_weights = load_npy(f'{folder}/tps_weights.npy')
                    mask = np.flatnonzero(np.isnan(shot_paths.weights))
                    new_weights = saved_weights[mask]
                    # ensure only (new) transitions have nonzero weigths
                    new_weights[~shot_paths[mask].are_transitions(states)] = 0.
                    shot_paths.weights[mask] = new_weights
                except:
                    pass
            
            # return
            return shot_paths

        # for each k
        result = []
        if k is None:
            folders = sorted(glob(f'{directory}/{prefix}{t}*'))
        elif isinstance(k, Iterable):
            folders = [f'{directory}/{prefix}{t}{k}' for k in k]
        else:
            raise TypeError(f'{k} must be integral, list of integrals, '
                            f'or None, got {k!r}')
        for folder in folders:
            if os.path.isdir(folder):
                k = folder.split(f'{prefix}{t}')[-1]
                if k.isdigit():
                    result.append(self.shot_paths(directory, prefix, t, k, old))
        return result

    def shot_chains(self, directory,
                    target_state=None, k=None,
                    old=PathEnsemble()):
        """
        Convenience wrapper for `shot_paths(..., prefix='chain', ...)`.
        """
        return self.shot_paths(directory, 'chain', target_state, k, old)

    def pathensemble(self, directory, shot_chains=[]):
        """
        Assemble a complete PathEnsemble from shot chains + free trajectories.

        Parameters
        ----------
        directory : str
            AIMMD run directory.
        shot_chains : list, optional
            Previously loaded shot chains (used for incremental update).

        Returns
        -------
        aimmd.pathensemble.PathEnsemble
            Full ensemble with the correct state mapping and categories.

        Notes
        -----
        This delegates to `assemble_pathensemble`, which expects:
        - list of shot chains (PathEnsembles),
        - list of free trajectories (Paths).
        """
        return assemble_pathensemble(
            self.shot_chains(directory, None, None, shot_chains),
            self.free_trajectories(directory, None))
