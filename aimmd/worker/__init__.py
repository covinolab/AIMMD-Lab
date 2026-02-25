"""
aimmd.worker
============

Worker process implementation for AIMMD.

This package defines :class:`~aimmd.worker.Worker`, the concrete worker class
used to run AIMMD tasks as isolated processes (typically under a scheduler such
as SLURM). The worker is built by combining a small set of mixins, each
responsible for a single aspect of worker behavior:

- :class:`~aimmd.worker._helpers.WorkerHelpers`
  Initialization, signal handling, resource binding, stop-condition bookkeeping.
- :class:`~aimmd.worker._properties.WorkerProperties`
  Derived properties for logging, stop checks, and initial-path discovery.
- :class:`~aimmd.worker._run.WorkerRun`
  Task wrapper and dispatch (changes working directory, clears caches, calls the
  selected task).
- :class:`~aimmd.worker._simulate.WorkerSimulate`
  Engine-facing simulation loop that incrementally extends trajectories on disk.
- :class:`~aimmd.worker._shoot.WorkerShoot`
  Core path-sampling task (committor-guided shooting).
- :class:`~aimmd.worker._free.WorkerFree`
  Free simulations.
- :class:`~aimmd.worker._train.WorkerTrain`
  Network training and adaptive-bin/density updates.
- :class:`~aimmd.worker._magic.WorkerMagic`
  Minimal magic methods (e.g., readable ``repr``).

Public API
----------
Only :class:`Worker` is exported by this package.
"""

from ._run import WorkerRun
from ._free import WorkerFree
from ._shoot import WorkerShoot
from ._magic import WorkerMagic
from ._train import WorkerTrain
from ._helpers import WorkerHelpers
from ._simulate import WorkerSimulate
from ._properties import WorkerProperties


class Worker(
    WorkerHelpers,
    WorkerMagic,
    WorkerProperties,
    WorkerRun,
    WorkerSimulate,
    WorkerFree,
    WorkerShoot,
    WorkerTrain):
    """
    Execute AIMMD tasks in an isolated worker process.

    A Worker is the atomic execution unit used by AIMMD on both local machines
    and clusters. It is designed to run as a dedicated process that:

    - installs SIGINT/SIGTERM handlers and reacts cooperatively to termination,
    - optionally binds CPU/GPU resources based on a ``localid`` and policy,
    - redirects stdout/stderr to a per-worker log file when requested,
    - executes exactly one task at a time via :meth:`run`.

    Tasks
    -----
    The canonical task names dispatched by :meth:`run` are:

    - ``'shoot'``: committor-guided path sampling (core enhanced sampling),
    - ``'free'``: free simulations around a chosen state,
    - ``'train'``: update the committor model and adaptive sampling state.

    Parameters
    ----------
    The constructor is aliased to :meth:`WorkerHelpers._init`. See that method
    for the full signature and meaning of worker initialization arguments
    (params/directory/localid, resource policies, stop conditions, logging).

    Notes
    -----
    - Workers intentionally mutate process-global state (signal handlers,
      stdout/stderr redirection). This is safe because workers are meant to run
      as isolated subprocesses.
    - Cache state is cleared per task (via :class:`WorkerRun`) to avoid leaking
      readers/arrays across tasks or directories.
    """

    __init__ = WorkerHelpers._init


__all__ = ['Worker']
