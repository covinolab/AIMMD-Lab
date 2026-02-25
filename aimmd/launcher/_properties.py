"""
aimmd.launcher._properties
=========================

Derived properties for AIMMD launcher classes.

This module defines :class:`LauncherProperties`, a small mixin that exposes
read-only views of internal launcher state, such as the managed run directories,
parameter sets, and the process executor.

The properties are intentionally lightweight and return *copies* of internal
lists where appropriate, so external callers do not accidentally mutate the
launcher’s internal bookkeeping.

Expected attributes
-------------------
Concrete launcher classes are expected to define:

- ``self._directories`` : list[str]
- ``self._params`` : list[aimmd.params.Params]
- ``self._processes`` : aimmd.execute.processes.ProcessExecutor
"""

# external
import os
import numpy as np
from abc import ABC


class LauncherProperties(ABC):

    @property
    def directories(self):
        """
        Run directories managed by this launcher.

        Returns
        -------
        list[str]
            Copy of the internal directory list.
        """
        return list(self._directories)

    @property
    def params(self):
        """
        Parameter sets managed by this launcher.

        Returns
        -------
        list[aimmd.params.Params]
            Copy of the internal Params list.
        """
        return list(self._params)

    @property
    def paths(self):
        """
        Parameter file paths for each managed run.

        Returns
        -------
        list[pathlib.Path]
            List of ``params.path`` for each run.
        """
        return [params.path for params in self._params]

    @property
    def processes(self):
        """
        Process executor used to spawn and track worker processes.

        Returns
        -------
        aimmd.execute.processes.ProcessExecutor
            The process executor instance stored in :attr:`_processes`.
        """
        return self._processes
