"""
aimmd.pathensemble._io
=====================

I/O mixin for :class:`aimmd.pathensemble.PathEnsemble`.

This module implements lightweight persistence of an ensemble as a **text file**
containing one trajectory filename per line. The file stores relative paths with
respect to the output file's parent folder, which makes ensembles relocatable.

Important
---------
This is *not* a full binary serialization of PathEnsemble. Instead, it stores a
flat list of filenames derived from :attr:`PathEnsemble.fnames`.

Loading
-------
:meth:`PathEnsembleIO.load` builds a new :class:`aimmd.pathensemble.PathEnsemble`
from the provided filename and can either:

- return the new instance when called as a class method, or
- mutate the existing instance by appending when called on an instance.

This dual behavior is implemented by :func:`aimmd.core.decorators.class_or_instancemethod`.
"""

# I/O helper layer for path ensembles (text-based persistence)

# external
import os
from abc import ABC
from pathlib import Path as PosixPath

# aimmd imports
from ..core.decorators import class_or_instancemethod


class PathEnsembleIO(ABC):
    """
    Mixin providing file-based save/load for `PathEnsemble`.

    The saved format is a newline-separated list of trajectory file paths.
    """

    def save(self, fname):
        """
        Save the ensemble to a text file listing trajectory filenames.

        Parameters
        ----------
        fname : str or pathlib.Path
            Output file path.

        Notes
        -----
        - Paths are written **relative to the output file's parent directory**.
        - The written list is taken from :attr:`aimmd.pathensemble.PathEnsembleProperties.fnames`,
          i.e., the concatenation of all per-path segment filenames.
        """
        parent = PosixPath(fname).resolve().parent
        with open(fname, "w") as file:
            file.write(
                "\n".join([os.path.relpath(fname, parent) for fname in self.fnames])
            )

    @class_or_instancemethod
    def load(self_or_cls, filename, find_shooting_indices=False, pipeline=()):
        """
        Load a path ensemble from a text file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to a file previously produced by :meth:`save`.
        find_shooting_indices : bool, default=False
            Forwarded to `PathEnsemble` construction. See
            :meth:`aimmd.pathensemble.PathEnsemble.__init__`.
        pipeline : tuple, default=()
            Forwarded to `PathEnsemble` construction. See
            :meth:`aimmd.pathensemble.PathEnsemble.__init__`.

        Returns
        -------
        PathEnsemble
            If called on the class, returns a new instance.

        Notes
        -----
        If called on an instance, this method mutates the receiver by appending
        the loaded ensemble via ``self += instance`` and returns `self`.
        """
        # local import avoids circular imports during package init
        from . import PathEnsemble

        # always create a new instance from the provided file
        instance = PathEnsemble(filename, find_shooting_indices, pipeline)

        # dispatch based on whether invoked on class or instance
        if isinstance(type(self_or_cls), type):
            return instance
        self += instance
        return self
