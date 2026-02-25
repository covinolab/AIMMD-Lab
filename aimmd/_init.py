"""
aimmd._init
===========

Package initialization logic.

This module provides :func:`initialize`, which performs a **one-time**
initialization of the AIMMD runtime environment and stores the resulting
configuration on :mod:`aimmd._config`.

Why a dedicated initializer?
----------------------------
AIMMD relies on:
- external executables (notably GROMACS),
- scientific Python libraries (NumPy/SciPy/MDAnalysis/etc.),
- multiprocessing-safe behavior (especially when using spawn),
- caching objects used across the package.

By centralizing this work in a single guarded function, we ensure consistent
runtime setup while avoiding repeated side effects on repeated imports.

Idempotency
-----------
The initializer uses the flag ``aimmd._config._initialized`` to ensure that it
only runs once per interpreter session.
"""

from . import _config

def initialize():
    """
    Initialize the AIMMD runtime environment (idempotent).

    This routine:
    - Imports and validates required external dependencies.
    - Locates required executables (e.g., GROMACS) on PATH.
    - Populates global configuration values on :mod:`aimmd._config`.
    - Instantiates shared caches for I/O.
    - Applies small runtime tweaks (warnings suppression, print behavior,
      thread settings for multiprocessing compatibility).

    Notes
    -----
    - The function is safe to call multiple times; it will return immediately if
      initialization has already completed.
    - Fail-fast behavior: if GROMACS cannot be found, an EnvironmentError is raised.
    """
    # Guard against re-initialization (important with repeated imports and
    # multiprocessing spawn where modules may be imported again).
    if _config._initialized:
        return
    
    # external (check dependencies)
    # Importing here ensures that missing dependencies fail early and clearly,
    # and avoids importing heavy modules if initialization is never needed.
    import os
    import sys
    import dill
    import tqdm
    import numpy
    import scipy
    import torch  # make torch work
    import torch._dynamo
    import shutil
    import filelock
    import warnings
    import functools
    import matplotlib
    import MDAnalysis
    import multiprocessing
    from pathlib import PosixPath

    # aimmd imports
    # Shared caches used across the package for efficient trajectory/array reading.
    from .cache import NpyReaderCache, MDAReaderCache
    
    #########################
    # executables and paths #
    #########################
    
    # Absolute path to the currently running Python interpreter.
    _config.PYTHON = sys.executable

    # Locate a GROMACS executable in PATH (support common names).
    _config.GROMACS = shutil.which('gmx') or shutil.which('gmx_mpi')
    if _config.GROMACS is None:
        # Hard requirement: many workflows depend on calling gmx.
        raise EnvironmentError(
            "GROMACS exec not found in PATH. Please install GROMACS and "
            "ensure 'gmx' or 'gmx_mpi' is accessible in your environment.")

    # Resolve the installation directory for AIMMD (used to build paths to
    # internal resources such as worker scripts and engine parameter files).
    _config.PARENT = str(PosixPath(__file__).resolve().parent)

    # Paths to internal helper scripts/files.
    _config.WORKER = f'{_config.PARENT}/worker/run.py'
    _config.EM_MDP = f'{_config.PARENT}/engines/em.mdp'
    
    ##########
    # caches #
    ##########
    
    # Instantiate caches once and share them via the global config.
    _config.NPY_CACHE = NpyReaderCache()
    _config.MDA_CACHE = MDAReaderCache()
    
    #################
    # print options #
    #################
    
    # quick logging
    # Force print() to flush by default to make logs appear promptly,
    # especially useful when running via multiprocessing or in batch systems.
    _config.print = functools.partial(print, flush=True)
    
    # suppress some warnings in MDAnalysis
    # MDAnalysis Reader objects can raise errors in __del__ during interpreter
    # shutdown or when partially constructed; wrap __del__ defensively.
    _old_del = MDAnalysis.coordinates.base.ReaderBase.__del__
    
    def _safe_del(self):
        try:
            _old_del(self)
        except (TypeError, AttributeError):
            # Suppress AttributeError caused by missing internal attributes
            # (e.g., _xdr) during object finalization.
            pass
    
    MDAnalysis.coordinates.base.ReaderBase.__del__ = _safe_del
    
    # Silence noisy user warnings from the XDR reader backend.
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="MDAnalysis.coordinates.XDR")
    
    # more compact array
    # Global NumPy display precision for more readable console output.
    numpy.set_printoptions(precision=3)

    ###################
    # multiprocessing #
    ###################
    
    # call just once, for compatibility issues
    # when spawning multiple processes
    #
    # In some environments, torch interop threading can cause issues when
    # forking/spawning worker processes; restricting interop threads improves
    # stability and avoids oversubscription.
    torch.set_num_interop_threads(1)

    ####################
    # trajectories i/o #
    ####################

    # Default unit cell dimensions used when trajectory files lack explicit box
    # information (lengths + angles).
    _config.DEFAULT_DIMENSIONS = numpy.array([0., 0., 0., 90., 90., 90.])
    
    #########
    # done! #
    #########
    
    # Mark as initialized so subsequent calls are no-ops.
    _config._initialized = True
