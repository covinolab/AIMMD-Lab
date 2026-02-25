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

    def free_trajectories(self, directory):
        """
        Collect (unsplit) free trajectories from an AIMMD run directory.

        Parameters
        ----------
        directory : str
            Base run directory containing `free{state}/traj??????.part????` files.
        
        Returns
        -------
        list of aimmd.path.Path
            List of reconstructed free trajectories as `Path` objects.
        
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

        # get "offset" for determining the path number
        ext = self.trajectory_extension
        offset = len(ext) + 9

        # iterate on allowed folders
        for t in self.states:
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
                        traj = Path(fnames,
                            remove_overlapping_frames=True)
                        name = f'traj{active}'
                        if name in indicted:
                            traj._exclude_from = indicted[name]
                        result.append(traj)
                    except:
                        continue
                    fnames = [fname]
                else:
                    fnames.append(fname)
                active = current
            # last path
            if fnames:
                try:
                    traj = Path(fnames,
                        remove_overlapping_frames=True)
                    name = f'traj{current}'
                    if name in indicted:
                        traj.exclude_from = indicted[name]
                    result.append(traj)
                except:
                    continue

        # all together, categorized
        pathensemble = PathEnsemble()
        pathensemble._paths = result  # directly assign to be faster
        return result

    def shot_paths(self, directory, prefix='chain',
                   target_state=None, k=None, old=None):
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
        old : optional
            Previously loaded data used for incremental updates.
            May be a `PathEnsemble`, a list of `PathEnsemble`, or compatible
            iterable depending on call mode.

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

        # which state are we talking about?
        states = self.states

        # load all of them
        if target_state is None:
            result = []
            old = old or []
            for t in states:
                this = self.shot_paths(directory, prefix, t, None, old)
                old = old[len(this):]
                result.extend(this)
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

            # initialize with "old"
            old_part = PathEnsemble()
            if isinstance(old, PathEnsemble):
                old_part = old[:-1]
            elif isinstance(old, Iterable) and len(old):
                if isinstance(old[0], PathEnsemble):
                    old_part = old[0][:-1]

            # keep what we already have
            fnames = []
            # except for the last one that may still change
            for path in old_part._paths[:-1]:
                fname = path.fname
                if fname not in fnames:
                    fnames.append(fname)

            # new part
            new_part = PathEnsemble()
            for fname in sorted(glob(f'{folder}/path??????{ext}')):
                if fname not in fnames:
                    path = Path(fname, shooting_index='find')
                    path.weight = path.is_complete(t, states)
                    new_part._paths.append(path)

            # finally: add (do not copy paths)
            shot_paths = old_part + new_part

            # get weights in case of tps
            if 'sweep' not in prefix and self.chain_type == 'tps':
                new_weights = np.zeros(len(new_part))
                try:
                    saved_weights = load_npy(f'{folder}/tps_weights.npy')
                    begin = len(old_part)
                    end = begin + len(new_part)
                    new_weights[:len(saved_weights) - begin
                        ] = saved_weights[begin:end]
                    # ensure only transitions have nonzero weigths
                    new_weights[~new_part.are_transitions(states)] = 0.
                except:
                    pass
                new_part.weights = new_weights

            # return
            return shot_paths

        # for each k
        result = old or []
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
                    k = int(k)
                    if len(result) <= k:
                        result.append(None)
                    result[k] = self.shot_paths(
                        directory, prefix, t, k, result[k])
        return result

    def shot_chains(self, directory, target_state=None, k=None, old=None):
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
            self.free_trajectories(directory))
