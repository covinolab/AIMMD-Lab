# pathensemble/_io.py
"""
aimmd.pathensemble._io
=====================

Persistence helpers for :class:`~aimmd.pathensemble.PathEnsemble`.

The I/O model is deliberately lightweight:

- :meth:`PathEnsembleIO.save` writes a plain-text list of relative path filenames
  (one per line), relative to the location of the saved list file.
- :meth:`PathEnsembleIO.load` reads such a list and creates/appends a new
  :class:`~aimmd.pathensemble.PathEnsemble` instance.

This keeps storage robust and transparent: a saved ensemble is simply a list of
path trajectory references, not a binary pickle.
"""

# external
import os
from abc import ABC
from pathlib import PosixPath

# aimmd imports
from ..core.decorators import class_or_instancemethod


# -----------------------------------------------------------------------------
# I/O mixin
# -----------------------------------------------------------------------------
class PathEnsembleIO(ABC):
    def save(self, fname):
        """
        Save the ensemble as a text file listing member path filenames.

        Parameters
        ----------
        fname : str or pathlib.Path
            Destination file. The file content is a newline-separated list of
            filenames for the ensemble, made **relative** to ``fname``'s parent
            directory for portability.

        Notes
        -----
        The filenames written are taken from :attr:`~aimmd.pathensemble.PathEnsemble.fnames`,
        which concatenates the underlying per-path ``_fnames``.
        """
        parent = PosixPath(fname).resolve().parent
        with open(fname, 'w') as file:
            file.write('\n'.join(
                [os.path.relpath(fname, parent)
                 for fname in self.fnames]))

    @class_or_instancemethod
    def load(self_or_cls, filename,
             find_shooting_indices=False, pipeline=()):
        """
        Load an ensemble from a text list file.

        This method supports two call styles:

        - ``PathEnsemble.load(filename, ...)`` returns a new instance.
        - ``ensemble.load(filename, ...)`` appends to the existing instance and
          returns it.

        Parameters
        ----------
        filename : str or pathlib.Path
            Text file written by :meth:`save`, i.e. a newline-separated list of
            path filenames.
        find_shooting_indices : bool, optional
            Forwarded to :meth:`PathEnsembleHelpers._init`.
        pipeline : tuple, optional
            Forwarded to :meth:`PathEnsembleHelpers._init`.

        Returns
        -------
        PathEnsemble
            A new instance if called on the class, otherwise the modified
            instance.
        """

        # do we need to create or select an instance of Params?
        from . import PathEnsemble
        instance = PathEnsemble(filename,
            find_shooting_indices, pipeline)
        if isinstance(type(self_or_cls), type):
            return instance
        self += instance
        return self
