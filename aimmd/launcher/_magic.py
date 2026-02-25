"""
aimmd.launcher._magic
====================

Magic methods for AIMMD launcher classes.

This module defines :class:`LauncherMagic`, a mixin providing basic dunder
methods for launcher instances:

- ``len(launcher)`` reports the number of managed runs,
- ``launcher_a + launcher_b`` concatenates runs into a new launcher,
- ``repr(launcher)`` provides a concise description suitable for logs.

The mixin is intentionally lightweight and assumes the concrete launcher class
exposes the following attributes/properties:

- :attr:`params` / :attr:`_params` : list of :class:`~aimmd.params.Params`
- :attr:`directories` / :attr:`_directories` : list of str
- :attr:`termination_timeout` : float

Notes
-----
- ``__add__`` returns a :class:`Launcher` instance. This relies on the symbol
  ``Launcher`` being available in the module namespace where this mixin is used.
  This is part of the original design and is preserved here.
"""

# external
from abc import ABC


class LauncherMagic(ABC):
    """
    Mixin implementing magic methods for launcher instances.

    The mixin does not own state. It assumes the concrete launcher stores its
    run list in ``self._params`` and provides public accessors for params and
    directories.
    """

    def __len__(self):
        """
        Return the number of managed runs.

        Returns
        -------
        int
            Number of runs (length of :attr:`_params`).
        """
        return len(self._params)

    def __add__(self, instance):
        """
        Concatenate two launchers.

        Parameters
        ----------
        instance : Launcher
            Another launcher instance. The resulting launcher contains the runs
            from ``self`` followed by the runs from ``instance``.

        Returns
        -------
        Launcher
            New launcher containing concatenated params/directories and using
            the termination timeout of ``self``.

        Notes
        -----
        This method assumes ``instance`` exposes ``params`` and ``directories``
        sequences compatible with concatenation.
        """
        return Launcher(self.params + instance.params,
                        self.directories + instance.directories,
                        self.termination_timeout)

    def __repr__(self):
        """
        Return a concise representation of the launcher.

        Returns
        -------
        str
            A string describing the number of runs and their directories.
        """
        directories = [f'{directory!r}' for directory in self.directories]
        return (f'Launcher of {len(self)} '
                f'run{"s" if len(self) != 1 else ""} '
                f'({", ".join(directories)})')
