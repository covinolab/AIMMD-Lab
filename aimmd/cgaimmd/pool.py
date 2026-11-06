"""
aimmd.cgaimmd.pool
==================

The candidate pool of a C-GAIMMD shooting worker, the watcher of the network
file and the random streams of candidate generation.

A C-GAIMMD shooting worker generates candidates in batches (a *refill*, stage
1: generation, validity, raw committor) and keeps the survivors in a
:class:`CandidatePool`; each shot pops one of them and prepares it (stage 2).
The pool holds the candidates of *one* network: the worker calls
:meth:`NetworkWatcher.refresh` before every pop, and a network file with a new
sha256 resets the pool (a refill outdated before its first pop serves one
pop first, :data:`aimmd.cgaimmd.source.STALE_REFILL_POPS`).
Candidates are i.i.d. given the network, and pops are uniformly random (not
target by target), so consecutive shots rarely share a target level: with
``candidates_per_target=2`` the other candidate of the last target is popped
next with a probability of about one over the pool size.

The watcher also reads ``network{s}.json``, the trainer's metadata of the
network file, for the number of trained shooting points, which decides when
the final check enforces the acceptance rule
(``cgaimmd_parameters['window_after_trained_shots']``).

Seeds are derived, never drawn: :func:`base_seed` (``generation_seed`` or a
hash of the absolute run directory), :func:`refill_seed` (per chain folder,
path and refill) and four independent streams per refill
(:func:`refill_streams`), so that drawing more numbers from one stream never
moves another.

numpy and the standard library at import; torch is imported where networks are
loaded.
"""

import hashlib
import io as _io
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import NetworkInfo


METADATA_WARNING_SECONDS = 60.0
"""Seconds after a load without a matching ``network{s}.json`` before the watcher warns."""

READ_RETRIES = 5
"""Attempts to read the bytes of a network file that changed (an ``OSError`` is retried)."""

READ_WAIT = 0.2
"""Seconds between those attempts."""


def _now():
    """``time.time()`` (a module function, so that tests can move the clock)."""
    return time.time()


def _sleep(seconds):
    """``time.sleep`` (a module function, so that tests can check that nothing waits)."""
    time.sleep(seconds)


def _log(message):
    """Print into the worker log (AIMMD's flushing print, set by ``aimmd._init``)."""
    from .. import _config
    getattr(_config, 'print', print)(message)


def _stat_key(fname):
    """``(st_mtime_ns, st_size, st_ino)`` of `fname`, or None if it does not exist."""
    try:
        stat = os.stat(fname)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def _read_bytes(fname):
    """The bytes of `fname`, retrying ``OSError``; None if it vanished or stays unreadable."""
    for attempt in range(READ_RETRIES):
        try:
            with open(fname, 'rb') as handle:
                return handle.read()
        except FileNotFoundError:
            return None
        except OSError:
            if attempt == READ_RETRIES - 1:
                return None
            _sleep(READ_WAIT)
    return None


class NetworkWatcher:
    """Load ``network{s}.h5`` into ``params.network`` whenever its content changes.

    Parameters
    ----------
    params : aimmd.Params
        The run's params; ``params.network`` receives the weights (``load_state_dict``, strict).
    directory : str
        Run directory (relative to the working directory, i.e. ``params.parent`` inside a worker).
    device : torch.device or str, optional
        Where the network is put after a load; None leaves it on its device.

    Notes
    -----
    :meth:`refresh` checks the file's ``(st_mtime_ns, st_size, st_ino)``;
    only when it changed are the bytes read (``READ_RETRIES`` attempts,
    ``READ_WAIT`` s apart) and hashed. A new sha256 is loaded with
    ``torch.load(BytesIO, map_location='cpu', weights_only=True)``; a file
    that does not load is not read again until its stat changes (one WARNING
    per file version) and the loaded weights stay (they are restored: a failed
    strict ``load_state_dict`` has already copied the tensors whose shapes
    matched). Missing files are found by
    the stat, so polling before the trainer's first network costs no waits.

    ``network{s}.json`` (written by the trainer *before* the ``.h5``)
    is re-read while it does not describe the loaded weights. Its
    ``round`` and ``n_trained_shots`` are taken when its ``sha256`` matches;
    otherwise the last matched values are kept, so a normal hand-off never
    drops the acceptance rule back to the bootstrap rule. If no json has
    matched :data:`METADATA_WARNING_SECONDS` after a load, one WARNING says
    so.
    """

    def __init__(self, params, directory, device=None):
        self.params = params
        self.directory = directory
        self.device = device
        states = params.sorted_states
        self.fname = f'{directory}/network{states}.h5'
        self.json_fname = f'{directory}/network{states}.json'
        self.info: Optional[NetworkInfo] = None
        """The loaded network file and its metadata (None before the first load)."""
        self._seen = None                  # stat key of the .h5 last read
        self._failed = None                # stat key of an .h5 that did not load
        self._json_seen = None             # stat key of the json last parsed
        self._matched = None               # (round, n_trained_shots, metadata) of the last match
        self._warned = None                # sha256 the metadata warning was printed for

    @property
    def loaded(self):
        """Whether a network file has been loaded."""
        return self.info is not None

    def n_trained_shots(self):
        """Trained shots of the last ``network{s}.json`` that matched a loaded network (0 before any)."""
        if self._matched is None:
            return 0
        return int(self._matched[1] or 0)

    def refresh(self):
        """Load the network file if its content changed.

        Returns
        -------
        bool
            True if new weights were loaded; False if the file is missing, unchanged or does not load.
        """
        loaded = self._refresh_network()
        if self.info is not None:
            if not self.info.metadata_matched:
                self._refresh_metadata()
            self._warn_if_unmatched()
        return loaded

    # -- the .h5 ---------------------------------------------------------------

    def _refresh_network(self):
        key = _stat_key(self.fname)
        if key is None or key == self._seen or key == self._failed:
            return False
        data = _read_bytes(self.fname)
        if data is None:                                   # retried next time
            return False
        sha = hashlib.sha256(data).hexdigest()
        if self.info is not None and sha == self.info.sha256:
            self._seen = key                               # touched, same content
            return False
        import torch
        network = self.params.network
        # a failed strict load_state_dict has already copied every tensor whose shape
        # matched: keep a copy to restore the loaded weights
        backup = {name: value.detach().clone() for name, value in network.state_dict().items()}
        try:
            state_dict = torch.load(_io.BytesIO(data), map_location='cpu', weights_only=True)
            network.load_state_dict(state_dict)
        except Exception as exc:                            # noqa: BLE001
            network.load_state_dict(backup)
            self._failed = key
            _log(f'WARNING: {os.path.basename(self.fname)} (sha256 {sha[:8]}) does not load '
                 f'into params.network: {type(exc).__name__}: {exc}')
            return False
        if self.device is not None:
            self.params.network.to(self.device)
        self._seen = key
        self._failed = None
        last = self._matched
        self.info = NetworkInfo(
            file=self.fname, sha256=sha,
            round=None if last is None else last[0],
            n_trained_shots=None if last is None else last[1],
            metadata={} if last is None else dict(last[2]),
            metadata_matched=False, loaded_at=_now())
        self._json_seen = None                             # re-read it for the new weights
        self._refresh_metadata()
        return True

    # -- the .json -------------------------------------------------------------

    def _refresh_metadata(self):
        key = _stat_key(self.json_fname)
        if key is None or key == self._json_seen:
            return
        self._json_seen = key
        try:
            with open(self.json_fname) as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            self._json_seen = None                         # retried next time
            return
        if not isinstance(metadata, dict) or metadata.get('sha256') != self.info.sha256:
            return
        n = metadata.get('n_trained_shots')
        self._matched = (metadata.get('round'), None if n is None else int(n), dict(metadata))
        self.info.round = metadata.get('round')
        self.info.n_trained_shots = self._matched[1]
        self.info.metadata = dict(metadata)
        self.info.metadata_matched = True

    def _warn_if_unmatched(self):
        info = self.info
        if (info.metadata_matched or self._warned == info.sha256
                or _now() - info.loaded_at < METADATA_WARNING_SECONDS):
            return
        self._warned = info.sha256
        _log(f'WARNING: {os.path.basename(self.json_fname)} has not described the loaded '
             f'network (sha256 {info.sha256[:8]}) for {METADATA_WARNING_SECONDS:g} s; trained '
             f'shots kept at {self.n_trained_shots()} (last matched metadata)')


@dataclass
class Candidate:
    """A stage-1 survivor waiting in the pool."""

    positions: np.ndarray
    """``(M, 3)`` float64 Å after the validity function (relabelled to the network's atom order)."""

    lambda_target: float
    """The target committor level it was generated at."""

    lambda_raw: float
    """The network's committor of :attr:`positions` (``aimmd.cgaimmd.protocol.coordinate_committor``)."""

    refill: int
    """The refill counter of the worker that generated it (1 for the worker's first refill)."""

    index: int
    """Its index within the refill's batch of generated samples."""

    refill_seed: int
    """The seed of its refill (:func:`refill_seed`)."""

    prep_seed: int
    """The seed of its own preparation stream (:func:`preparation_stream`, :func:`stream_seed`)."""

    validity_info: Optional[dict]
    """The validity function's per-sample dict, if it gave one."""

    law_record: dict
    """The record of the lambda law that drew its target (``aimmd.cgaimmd.lambda_law.draw_targets``)."""


class CandidatePool:
    """The stage-1 survivors of one network, popped uniformly at random.

    Notes
    -----
    :meth:`extend` adds a refill's candidates together with the sha256 of the
    network that generated them and the refill's pop stream; pops use the
    pop stream of the refill that last extended the pool. Extending a pool
    that holds another network's candidates is an error: the worker resets
    the pool first (:meth:`reset`).
    """

    def __init__(self):
        self._candidates = []
        self._pop_rng = None
        self.network_sha256: Optional[str] = None
        """sha256 of the network whose candidates the pool holds (None after a reset)."""
        self.reset_count: int = 0
        """The number of resets so far (the sidecar's ``pool.reset_count``)."""
        self.last_reset_reason: Optional[str] = None
        """The reason given to the last :meth:`reset`."""

    def __len__(self):
        return len(self._candidates)

    def __iter__(self):
        return iter(list(self._candidates))

    def reset(self, reason):
        """Discard every candidate.

        Parameters
        ----------
        reason : str
            Why (kept in ``last_reset_reason``; the worker logs it).

        Returns
        -------
        int
            The number of candidates discarded.
        """
        n = len(self._candidates)
        self._candidates = []
        self._pop_rng = None
        self.network_sha256 = None
        self.reset_count += 1
        self.last_reset_reason = str(reason)
        return n

    def extend(self, candidates, network_sha256, pop_rng):
        """Add the candidates of one refill.

        Parameters
        ----------
        candidates : list of Candidate
            The refill's survivors (may be empty).
        network_sha256 : str
            sha256 of the network file whose weights generated them.
        pop_rng : numpy.random.Generator
            The refill's pop stream (:func:`refill_streams`), used by every later :meth:`pop`.

        Raises
        ------
        ValueError
            If the pool holds candidates of another network.
        """
        if self._candidates and network_sha256 != self.network_sha256:
            raise ValueError(f'the pool holds candidates of network {self.network_sha256!r}; reset '
                             f'it before adding candidates of network {network_sha256!r}')
        self._candidates.extend(candidates)
        self.network_sha256 = network_sha256
        self._pop_rng = pop_rng

    def pop(self):
        """Remove and return a uniformly random candidate (None if the pool is empty)."""
        if not self._candidates:
            return None
        index = int(self._pop_rng.integers(len(self._candidates)))
        return self._candidates.pop(index)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def _word63(data):
    """The first 8 bytes of sha256(`data`) as a little-endian int, shifted to 63 bits."""
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], 'little') >> 1


def base_seed(settings, directory):
    """The base seed of a worker's candidate generation.

    Parameters
    ----------
    settings : mapping
        C-GAIMMD settings; ``settings['generation_seed']`` is read.
    directory : str
        The run directory.

    Returns
    -------
    int
        ``generation_seed`` if it is set, else a 63-bit hash of ``os.path.abspath(directory)``.

    Notes
    -----
    Without ``generation_seed`` replicas of one system (different run
    folders) differ and a restarted run reproduces its seeds; with it, the
    seeds survive moving the run folder, and replicas need distinct values.
    """
    seed = settings.get('generation_seed')
    if seed is not None:
        return int(seed)
    return _word63(os.path.abspath(directory))


def refill_seed(base, folder, path_name, refill):
    """The seed of one refill.

    Parameters
    ----------
    base : int
        :func:`base_seed`.
    folder : str
        The chain folder (only its basename, e.g. ``'chainR0'``, enters the seed).
    path_name : str
        The path being prepared, e.g. ``'path000012'``.
    refill : int
        The number of refills made for this path before this one (0 for the first).

    Returns
    -------
    int
        The first 8 bytes (little-endian) of ``sha256(f'{base}:{folder_name}:{path_name}:{refill}')`` >> 1.
    """
    folder_name = os.path.basename(os.path.normpath(folder))
    return _word63(f'{int(base)}:{folder_name}:{path_name}:{int(refill)}')


def stream_seed(rng):
    """The integer seed of a stream (``CandidateContext.seed``).

    Parameters
    ----------
    rng : numpy.random.Generator
        A stream built by :func:`refill_streams` or :func:`preparation_stream`.

    Returns
    -------
    int
        The first 63-bit word of its ``SeedSequence``, ``generate_state(1, np.uint64)[0] >> 1``.
    """
    return _first_word(rng.bit_generator.seed_seq)


def _first_word(seed_sequence):
    return int(seed_sequence.generate_state(1, np.uint64)[0] >> np.uint64(1))


def refill_streams(refill_seed, n):
    """The random streams of one refill.

    Parameters
    ----------
    refill_seed : int
        :func:`refill_seed`.
    n : int
        The number of candidates the refill generates.

    Returns
    -------
    tuple
        ``(law, pop, validity, prep_seeds)``: three ``numpy.random.Generator``, the ``n`` preparation seeds.

    Notes
    -----
    ``law, pop, prep, validity = SeedSequence(refill_seed).spawn(4)``: the law
    stream (lambda* draws and ``cgaimmd_lambda_sampler``), the pop stream (every
    pop after this refill), the preparation streams (``prep.spawn(n)``, one per
    candidate, rebuilt by :func:`preparation_stream`) and the validity stream
    (``cgaimmd_validity_function``). The fourth child leaves the first three as
    ``spawn(3)`` gives them, and a stream that draws more numbers never moves
    another.
    """
    law, pop, prep, validity = np.random.SeedSequence(int(refill_seed)).spawn(4)
    prep_seeds = [_first_word(child) for child in prep.spawn(int(n))]
    return (np.random.default_rng(law), np.random.default_rng(pop),
            np.random.default_rng(validity), prep_seeds)


def preparation_stream(refill_seed, index):
    """Candidate `index`'s preparation stream of the refill with seed `refill_seed`.

    Parameters
    ----------
    refill_seed : int
        :func:`refill_seed`.
    index : int
        The candidate's index within its refill.

    Returns
    -------
    numpy.random.Generator
        ``default_rng`` of child `index` of the refill's third ``SeedSequence`` child.
    """
    child = np.random.SeedSequence(int(refill_seed), spawn_key=(2, int(index)))
    return np.random.default_rng(child)
