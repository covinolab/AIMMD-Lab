"""
aimmd.cgaimmd.types
===================

Shared dataclasses of C-GAIMMD.

These are the objects that cross the interfaces between the shooting-point
source of ``chain_type='cgaimmd'``, the params-file hooks
(``cgaimmd_validity_function``, ``cgaimmd_prepare_function``,
``cgaimmd_minimize_function``, ``cgaimmd_lambda_sampler``) and
``aimmd.worker._shoot``. Every coordinate is in Ångström. Records
(``record``, ``info``, ``metadata``) are JSON-serializable dicts.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class NetworkInfo:
    """Which network weights a shooting worker is using (one loaded ``network{s}.h5``)."""

    file: str
    """The ``network{s}.h5`` the weights were loaded from."""

    sha256: str
    """Hex sha256 of the file bytes that were loaded."""

    round: Optional[int]
    """Training round, from the last ``network{s}.json`` that matched (None before a match)."""

    n_trained_shots: Optional[int]
    """Trained shooting points (training split) of that round, from the same json."""

    metadata: dict
    """That json; ``{}`` if none has matched yet."""

    metadata_matched: bool
    """Whether the json matches *this* sha256."""

    loaded_at: float
    """``time.time()`` of the load."""


@dataclass
class CandidateContext:
    """Everything a C-GAIMMD hook may need about the shot being prepared.

    The shooting-point source builds one per hook call; its ``rng`` and
    ``seed`` are the random stream of that call.
    """

    params: object
    """The run's :class:`aimmd.Params`."""

    network: object
    """``params.network``, holding the weights of :attr:`network_info`."""

    network_info: Optional[NetworkInfo]
    """The loaded network file; None before the first load."""

    directory: str
    """Run directory relative to ``params.parent`` (``worker._directory``, valid inside
    ``Worker.run``, which changes to ``params.parent``)."""

    folder: str
    """Chain folder of the shot."""

    path_name: str
    """Name of the path being prepared, e.g. ``'path000012'``."""

    scratch: str
    """Work directory of this chain, ``f'{folder}/cgaimmd_scratch'``."""

    rng: np.random.Generator
    """Random stream of *this* hook call (the refill's law stream for
    ``cgaimmd_lambda_sampler``, the candidate's preparation stream for preparation and
    minimization)."""

    seed: int
    """The integer seed of that stream (for GROMACS ``gen_seed`` / ``-seed``)."""

    generator: object
    """``torch.Generator`` on the generation device (the random stream of generation)."""

    device: object
    """The generation ``torch.device``."""

    must_stop: Callable[[], bool]
    """``must_stop() -> bool``; hooks that wait or run external programs poll it."""

    log: Callable[[str], None]
    """``log(message)`` prints into the worker log."""

    settings: dict
    """``params.cgaimmd_settings``."""

    bins: Optional[np.ndarray] = None
    """The trainer's current adaptive bins, when loaded."""

    densities: Optional[np.ndarray] = None
    """The trainer's current bin densities, when loaded."""


@dataclass
class ValidityResult:
    """Outcome of ``cgaimmd_validity_function`` on a batch of raw generated samples."""

    positions: np.ndarray
    """``(n, M, 3)`` float64 Å, relabelled to the network's atom order."""

    ok: np.ndarray
    """``(n,)`` bool."""

    reason: np.ndarray
    """Reason codes, an ``(n,)`` object array (``''`` where :attr:`ok`, else a short code such as
    ``'bond_count'``, ``'chirality'`` or ``'clash'``)."""

    info: Optional[list] = None
    """Optional per-sample dicts."""


@dataclass
class PreparedFrame:
    """A candidate turned into a full system (by preparation, then minimization)."""

    positions: np.ndarray
    """``(N, 3)`` float64 Å in ``params.topology`` atom order."""

    dimensions: Optional[np.ndarray]
    """``(6,)`` box (Å, degrees); None for a box-less toy system."""

    ok: bool = True
    """False if the step failed for this candidate (a rejection, not an error)."""

    reason: str = ''
    """Reason code when not :attr:`ok` (``'prepare_failed'``, ``'minimize_failed'``)."""

    record: dict = field(default_factory=dict)
    """What the tool did (settings, messages), for the sidecar."""

    seconds: float = 0.0
    """Wall time spent."""


@dataclass
class ShootingPoint:
    """A generated shooting point, as ``Params.initialize_simulation`` takes it.

    It duck-types the MDAnalysis ``Timestep`` that RFPS selection returns. It
    has no ``velocities``: ``initialize_simulation`` therefore draws fresh
    Maxwell-Boltzmann velocities (``gen_temperature > 0`` is required under
    ``chain_type='cgaimmd'``).
    """

    positions: np.ndarray
    """``(N, 3)`` float32 Å."""

    dimensions: Optional[np.ndarray]
    """``(6,)`` float32 box, or None."""

    record: dict
    """The in-flight record of the shot, without the keys added at registration."""
