'''
aimmd — AI for molecular mechanism discovery
==========================================

Package entry point.

What happens on import
----------------------
Importing :mod:`aimmd` triggers a one-time initialization routine
(:func:`aimmd._init.initialize`). This routine:

- checks/imports required dependencies,
- locates external executables (e.g., GROMACS),
- sets up caches and defaults,
- applies small runtime compatibility tweaks (notably for multiprocessing).

This eager initialization is convenient for interactive use, but note that it
does have side effects on import (environment checks, cache instantiation, etc.).

Public re-exports
-----------------
The package re-exports a selection of commonly used classes and utilities to
provide a compact, user-facing API.

Caveats / maintainability notes
-------------------------------
- This module constructs ``__all__`` dynamically from configuration attributes.
  If new configuration keys are added in :mod:`aimmd._init`, they will
  automatically propagate into the star-import surface.
- This file also defines lightweight summary/representation helpers so that the
  package prints nicely in interactive contexts.
'''

# initialize
from ._init import initialize
initialize()  # initialize once; guarded by aimmd._config._initialized

# basic
from ._config import *  # re-export config values populated by initialize()

# imports
from .core import utils
from .path import Path
from .params import Params
from .worker import Worker
from .launcher import Launcher
from .pathensemble import PathEnsemble

# all/version
__all__ = ['__version__']

# NOTE:
# The next lines assume that a name `_config` is available in this module scope.
# If `_config` is not imported as a module object, this will raise NameError at runtime.
# This comment documents the expectation without changing the logic.
__all__.extend(dir(_config))
__all__.remove('_initialized')
__all__.extend(['utils',
    'Path', 'Params', 'PathEnsemble',
    'Worker', 'Launcher'
])

# Semantic-ish version string for AIMMD.
__version__ = '0.1.0'

# summary/representation
def _summary() -> str:
    """
    Return a short human-readable summary of the package.

    Returns
    -------
    str
        A concise string intended for interactive display.
    """
    return f'AIMMD v{__version__} — AI for molecular mechanism discovery'

def __repr__():
    """
    Provide a friendly representation in interactive contexts.

    Notes
    -----
    This is a module-level helper and is not the same as ``object.__repr__``.
    Some environments may call this when printing the module/package object.
    """
    return _summary()
