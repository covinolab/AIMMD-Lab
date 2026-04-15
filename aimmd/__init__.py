"""
aimmd — AI for molecular mechanism discovery

Public package entry point.

This module:
- triggers one-time initialization (see :func:`aimmd._init.initialize`),
- re-exports selected configuration symbols from :mod:`aimmd._config`,
- exposes the most important high-level classes at package scope.

What happens on import
----------------------
Importing `aimmd` will call `initialize()` immediately. This is intentional,
because many components assume `_config` is populated (e.g., caches, paths,
executables). If you need a "no side effects" import in the future, this file
is where you would introduce an opt-out mechanism.

Public API
----------
The following objects are made available at top-level:

- ``aimmd.utils`` (module): general utilities
- ``aimmd.Path``: path object
- ``aimmd.Params``: configuration container
- ``aimmd.PathEnsemble``: main dataset/ensemble container
- ``aimmd.Worker``: worker logic for running simulations / evaluations
- ``aimmd.Launcher``: orchestration logic

From :mod:`aimmd._config`:
- ``GROMACS``, ``PARENT``, ``WORKER``, ``EM_MDP``
- ``NPY_CACHE``, ``MDA_CACHE``

Versioning
----------
``__version__`` is currently a static string.
"""
 
# initialize once; guarded by aimmd._config._initialized
from ._init import initialize
initialize()

# basic (re-export runtime configuration symbols)
from ._config import *

# imports (high-level public surface)
from .core import utils
from .path import Path
from .params import Params
from .worker import Worker
from .launcher import Launcher
from .pathensemble import PathEnsemble

# version/all
__version__ = '0.1.0'
__all__ = [
    '__version__', 'utils',
    'Path', 'Params', 'PathEnsemble',
    'Worker', 'Launcher', 'GROMACS',
    'PARENT', 'WORKER', 'EM_MDP',
    'NPY_CACHE', 'MDA_CACHE']

# summary/representation
def _summary() -> str:
    # One-line summary used by __repr__.
    return f'AIMMD v{__version__} — AI for molecular mechanism discovery'

def __repr__():
    return _summary()
