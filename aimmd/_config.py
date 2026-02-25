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
