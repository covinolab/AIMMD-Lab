"""
aimmd._init
===========

Package initialization routine.

This module defines :func:`initialize`, which is executed at import time by
``aimmd.__init__`` to set up AIMMD runtime configuration.

Responsibilities
----------------
`initialize()` performs four categories of work:

1) Dependency import / availability checks
   Ensures optional and required dependencies import cleanly early, so later
   modules fail fast with clearer error messages.

2) Resolve executables and package-internal paths
   - determines the current Python executable,
   - resolves the GROMACS executable from PATH,
   - computes absolute package path roots (based on ``__file__``),
   - sets paths to worker scripts and engine templates.

3) Create shared caches
   Creates the NPY and MDAnalysis reader caches and stores them in `_config`.
   These caches are referenced in multiple places and should behave as singletons.

4) Configure global behavior
   - sets print flushing (quick logging),
   - suppresses known-noisy MDAnalysis warnings,
   - patches an MDAnalysis destructor to be more defensive during interpreter shutdown,
   - sets NumPy print options,
   - configures PyTorch interop threads for spawn compatibility,
   - defines a default dimensions vector.

Idempotency
-----------
The function uses `_config._initialized` as a guard: if initialization has
already happened, it returns immediately.

Notes
-----
- This routine has intentional side effects (global configuration).
- Heavy imports are done inside the function to avoid import-time cost when
  AIMMD is not actively used.
"""

from . import _config


def _resolve_gromacs():
    """Resolve the GROMACS executable from PATH; warn (do not raise) if absent.

    Sets ``_config.GROMACS`` to the resolved path or ``None`` and returns it.
    Extracted from :func:`initialize` so the resolution/warning behavior can be
    unit-tested without triggering the rest of initialization.
    """
    import shutil
    import warnings
    _config.GROMACS = shutil.which('gmx') or shutil.which('gmx_mpi')
    if _config.GROMACS is None:
        warnings.warn(
            _config._GROMACS_NOT_FOUND_MSG
            + " AIMMD imported successfully; analysis and training will work,"
              " but any GROMACS-engine sampling operation (Launcher.run,"
              " Launcher.create_job, or a shoot/free worker) will raise until"
              " GROMACS is installed and on PATH.",
            RuntimeWarning, stacklevel=2)
    return _config.GROMACS


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
    - If GROMACS cannot be found, a ``RuntimeWarning`` is emitted and
      ``_config.GROMACS`` is left ``None``; GROMACS-engine sampling entry points
      then raise ``EnvironmentError`` via :func:`aimmd._config.require_gromacs`.
    """
    # Idempotency guard: initialization should run once per interpreter.
    if _config._initialized:
        return
    
    # External dependency imports (also acts as a dependency presence check)
    import os
    import sys
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
    from pathlib import Path as PosixPath

    # AIMMD imports that depend on the above dependencies being available
    from .cache import NpyReaderCache, MDAReaderCache
    
    #########################
    # executables and paths #
    #########################
    
    # Path to the current Python interpreter (useful when spawning workers).
    _config.PYTHON = sys.executable

    # Resolve GROMACS executable from PATH. Missing GROMACS now warns (rather
    # than raising) so import / analysis / training work without it; the hard
    # requirement is enforced at the GROMACS-engine sampling entry points via
    # _config.require_gromacs().
    _resolve_gromacs()

    # Package root directory (absolute path).
    _config.PARENT = str(PosixPath(__file__).resolve().parent)

    # Worker launcher script path (used by AIMMD process orchestration).
    _config.WORKER = f'{_config.PARENT}/worker/run.py'

    # Energy minimization mdp template path.
    _config.EM_MDP = f'{_config.PARENT}/engines/em.mdp'
    
    ##########
    # caches #
    ##########
    
    # Shared caches (effectively singletons).
    _config.NPY_CACHE = NpyReaderCache()
    _config.MDA_CACHE = MDAReaderCache()
    
    #################
    # print options #
    #################
    
    # Quick logging: ensure prints flush immediately (useful in HPC logs).
    _config.print = functools.partial(print, flush=True)
    
    # MDAnalysis warning suppression and destructor hardening
    _old_del = MDAnalysis.coordinates.base.ReaderBase.__del__
    
    def _safe_del(self):
        try:
            _old_del(self)
        except (TypeError, AttributeError):
            # Suppress AttributeError caused by missing internal attributes
            # (e.g., _xdr) during interpreter shutdown.
            pass
    
    MDAnalysis.coordinates.base.ReaderBase.__del__ = _safe_del
    
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="MDAnalysis.coordinates.XDR")
    
    # More compact NumPy prints across the package.
    numpy.set_printoptions(precision=3)

    ###################
    # multiprocessing #
    ###################
    
    # Compatibility: call just once, for spawn-related issues when creating
    # multiple processes. This limits PyTorch interop thread usage.
    torch.set_num_interop_threads(1)

    ####################
    # trajectories i/o #
    ####################

    # Default unit cell dimensions for frames missing dimension info.
    _config.DEFAULT_DIMENSIONS = numpy.array([0., 0., 0., 90., 90., 90.])
    
    #########
    # done! #
    #########
    
    _config._initialized = True
