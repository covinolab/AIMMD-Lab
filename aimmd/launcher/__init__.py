"""
aimmd.launcher
==============

Launcher orchestration for AIMMD runs.

This package defines :class:`~aimmd.launcher.Launcher`, the high-level interface
that coordinates one or more AIMMD runs and spawns multiple worker processes
either locally (via multiprocessing) or indirectly via a scheduler (by writing
a SLURM job script).

Relationship to workers
-----------------------
A :class:`~aimmd.worker.Worker` executes a single AIMMD task (shoot/free/train)
inside one process. The :class:`Launcher` is responsible for:

- managing a list of runs (Params + working directory pairs),
- deciding how many workers to launch for each run and role,
- computing per-worker CPU/GPU allocations from the available resources,
- creating the required directory structure and exporting initial paths,
- starting and supervising processes (fail-fast behavior).

Mixins
------
The launcher is composed from small mixins, each implementing one concern:

- :class:`~aimmd.launcher._helpers.LauncherHelpers`
  Initialization, input normalization, derived configuration, signal handling,
  resource allocation policies.
- :class:`~aimmd.launcher._magic.LauncherMagic`
  Magic methods (``len``, ``add``, ``repr``).
- :class:`~aimmd.launcher._properties.LauncherProperties`
  Read-only accessors for internal state (params, directories, executor).
- :class:`~aimmd.launcher._methods.LauncherMethods`
  User-facing methods to mutate the run list and to write SLURM job scripts.
- :class:`~aimmd.launcher._build.LauncherBuild`
  Filesystem setup and construction of worker argument tuples.
- :class:`~aimmd.launcher._run.LauncherRun`
  Local execution: spawn and supervise worker processes.

Design notes
------------
- ``Launcher.__init__`` is aliased to :meth:`LauncherHelpers._init` to keep a
  single authoritative initialization path.
- Many launcher operations mutate the filesystem (creating directories,
  exporting initial paths). Treat :meth:`LauncherBuild._build` as a setup step.
- Signal handlers (SIGINT/SIGTERM) are installed to terminate all managed
  processes cleanly.

Public API
----------
Only :class:`Launcher` is exported by this package.
"""

from ._run import LauncherRun
from ._build import LauncherBuild
from ._magic import LauncherMagic
from ._helpers import LauncherHelpers
from ._methods import LauncherMethods
from ._properties import LauncherProperties


class Launcher(
    LauncherHelpers,
    LauncherMagic,
    LauncherProperties,
    LauncherMethods,
    LauncherBuild,
    LauncherRun):
    """
    Orchestrate one or more AIMMD runs and spawn/manage worker processes.

    A Launcher groups together *runs*, where each run is defined by:

    - a :class:`~aimmd.params.Params` instance (or a path to a saved Params file),
    - a working directory where trajectories, caches, and logs are created.

    The launcher turns this declarative configuration into an execution plan
    (via :meth:`~aimmd.launcher._build.LauncherBuild._build`) and then executes
    it either:

    - **locally**, by spawning processes and monitoring them
      (:meth:`~aimmd.launcher._run.LauncherRun.run`), or
    - **on a cluster**, by emitting an ``srun``-based SLURM script that launches
      the same workers (:meth:`~aimmd.launcher._methods.LauncherMethods.create_job`).

    The launcher also computes derived values that are needed to run reliably:

    - how many worker processes should be launched for each run and role,
    - per-task CPU/GPU allocations (shared, all, explicit, or skipped binding),
    - stop-condition budgets to pass to workers.

    Notes
    -----
    - This class is built from mixins. The public methods users typically call
      are:

      - :meth:`add` / :meth:`pop` (manage runs),
      - :meth:`create_job` (write a SLURM job script),
      - :meth:`run` (run locally).

    - A launcher installs SIGINT/SIGTERM handlers (via
      :meth:`LauncherHelpers._init`) that clear managed processes. This behavior
      assumes the launcher runs in a main thread in an interactive/script
      context.

    - The launcher uses a fail-fast supervision policy: during local runs, it
      stops all workers when any worker exits; non-zero exit codes are treated
      as failure.
    """

    __init__ = LauncherHelpers._init


__all__ = ['Launcher']
