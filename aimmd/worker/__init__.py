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
  **Core path-sampling task**: committor-guided shooting to enhance sampling in
  the reactive region and produce a diverse ensemble of reactive paths.
- :class:`~aimmd.worker._free.WorkerFree`
  Free simulations (typically long unbiased runs) used to support sampling and
  statistics.
- :class:`~aimmd.worker._train.WorkerTrain`
  Network training and adaptive-bin/density updates (committor model updates).
- :class:`~aimmd.worker._magic.WorkerMagic`
  Minimal magic methods (e.g., readable ``repr``).

Design notes
------------
- The worker class is intentionally lean: task logic is split across mixins to
  keep files small and responsibilities local.
- ``Worker.__init__`` is aliased to :meth:`WorkerHelpers._init` to keep a single
  authoritative initialization path.
- Many operations intentionally mutate process-global state (stdout/stderr
  redirection, signal handlers). This is safe because workers are meant to run
  as dedicated subprocesses.

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

    __init__ = WorkerHelpers._init


__all__ = ['Worker']
