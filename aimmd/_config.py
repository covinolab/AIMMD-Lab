"""
aimmd._config
=============

Internal runtime configuration for AIMMD.

This module stores global configuration values that are populated once at
package initialization time (see :func:`aimmd._init.initialize`).

Purpose
-------
AIMMD needs a small amount of shared, runtime-resolved state:

- paths to key resources inside the package (worker script, engine templates),
- resolved external executables (e.g., GROMACS),
- instantiated caches used across the package,
- small global defaults (e.g., default unit cell dimensions),
- a lightweight "print" wrapper for consistent flushing.

The values are deliberately stored in a module rather than a class:
- they are easy to import from anywhere,
- they behave like a singleton without additional machinery,
- they avoid heavy dependencies.

Initialization protocol
-----------------------
The :func:`aimmd._init.initialize` function is responsible for setting
configuration variables and marking the module as initialized.

Do not set these variables manually unless you know why you are doing it.

Attributes
----------
_initialized : bool
    Sentinel flag to prevent repeated initialization. It is set to ``True``
    at the end of :func:`aimmd._init.initialize`.
"""

# Will become True once initialization is completed.
_initialized = False


# Canonical message shown when no GROMACS executable can be resolved. Kept here
# (rather than in _init) so the import-time warning and the sampling-time guard
# share identical wording, and so launcher/worker can import the guard without a
# circular import (_config imports nothing).
_GROMACS_NOT_FOUND_MSG = (
    "GROMACS exec not found in PATH. Please install GROMACS and "
    "ensure 'gmx' or 'gmx_mpi' is accessible in your environment.")


def require_gromacs():
    """Fail fast if no GROMACS executable was resolved at initialization.

    A no-op when a GROMACS executable is available. Sampling entry points that
    drive the GROMACS engine call this so sampling raises the same
    ``EnvironmentError`` that ``import aimmd`` used to raise, while import,
    analysis, training, and toy-engine sampling stay usable without GROMACS.

    Raises
    ------
    EnvironmentError
        If no ``gmx``/``gmx_mpi`` executable was found on PATH.
    """
    # Read the module attribute at call time so the current value is always
    # seen (and so this is safe even if initialize() never ran).
    if globals().get('GROMACS') is None:
        raise EnvironmentError(_GROMACS_NOT_FOUND_MSG)
