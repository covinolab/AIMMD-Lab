"""
aimmd.engines
=============

Simulation engine interfaces and templates.

This subpackage contains "engine" implementations that are used by AIMMD to
advance a system in time and write trajectory output. Engines are invoked by
higher-level orchestration code (e.g., workers/executors).

Public API
----------
ToyEngine
    Minimal in-Python engine that writes trajectories through MDAnalysis.
    Intended for tests, demos, and development.

EM_MDP
    Absolute path (string) to the bundled GROMACS energy-minimization `.mdp`
    template shipped with AIMMD.

Why `EM_MDP` is exported here
-----------------------------
`em.mdp` is a data file (not Python code). Exporting its resolved path from
`aimmd.engines` provides a stable import location:

    from aimmd.engines import EM_MDP

Callers can then pass this path to GROMACS or to AIMMD configuration without
hardcoding filesystem layout or relying on `__file__` in user code.
"""

from .toy import ToyEngine
from .._config import EM_MDP
