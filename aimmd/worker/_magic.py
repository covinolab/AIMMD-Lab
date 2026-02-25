"""
aimmd.worker._magic
==================

Magic methods for AIMMD worker classes.

This module defines :class:`WorkerMagic`, a small mixin that provides
human-readable representations for worker instances.

The mixin is intentionally minimal and avoids importing other AIMMD modules to
reduce the risk of circular imports during worker startup.

Notes
-----
- The representation relies on the instance exposing a :attr:`directory`
  attribute (typically set by :meth:`~aimmd.worker._helpers.WorkerHelpers._init`).
"""

# external
from abc import ABC


class WorkerMagic(ABC):
    """
    Mixin implementing magic methods for worker instances.

    The primary purpose of this class is to provide a concise ``repr`` that is
    useful in logs and debugging. The mixin does not define any state and
    expects the concrete worker to provide the required attributes.

    Required attributes
    -------------------
    directory : str
        Working directory associated with the worker.
    """

    def __repr__(self):
        """
        Return a concise, human-readable representation of the worker.

        Returns
        -------
        str
            A string of the form ``"Worker of '<directory>'"``.
        """
        return f'Worker of {self.directory!r}'
