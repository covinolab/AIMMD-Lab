"""
aimmd._config
=============

Central configuration namespace for AIMMD.

This module is intentionally lightweight and is used as a **shared state**
container for values that are discovered or constructed during package
initialization (see :mod:`aimmd._init`).

Design notes
------------
- AIMMD performs a one-time initialization step (dependency checks, locating
  executables, preparing caches, setting options).
- That initialization populates attributes on this module (e.g. ``PYTHON``,
  ``GROMACS``, cache instances, default box dimensions, etc.).
- The flag ``_initialized`` prevents re-running initialization multiple times,
  which is especially important with multiprocessing and repeated imports.

Public API
----------
Most attributes are added dynamically at runtime by :func:`aimmd._init.initialize`.
Users typically access configuration via ``from aimmd import *`` or via the
package namespace after initialization.

Implementation detail
---------------------
This file exists as a stable import target that can be safely imported by
submodules without triggering expensive initialization on import.
"""

# Will become True once package initialization is completed.
# Used to ensure initialization is idempotent.
_initialized = False
