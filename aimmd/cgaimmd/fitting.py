"""
aimmd.cgaimmd.fitting
=====================

What the online fits of C-GAIMMD networks share: the trainer's run folder,
the round counter of a run and the base seed of its fits.

Both fits train every round again from fixed weights (the flow-matching fit
of PaiNN networks, :mod:`aimmd.cgaimmd.flow_matching.fit`, and the sequential
fit of BioEmu2 networks, :mod:`aimmd.cgaimmd.bioemu2.fit`). What they carry
from one round to the next is bookkeeping: the round counter, read from the
``network{states}.json`` that AIMMD's trainer writes next to the network
(:func:`previous_round`), and a seed that stays the same for a run and differs
between replicates (:func:`fit_seed`).

Moved here from :mod:`aimmd.cgaimmd.bioemu2.fit` (which still exports
``previous_round`` and ``fit_seed``), so that the flow-matching fit does not
import the BioEmu2 adaptor.
"""

import hashlib
import json
import os

__all__ = ['run_directory', 'previous_round', 'fit_seed']


def run_directory(worker):
    """The run folder of the trainer `worker`, or None (no worker).

    Inside a worker task the run is read through ``_directory``
    (``Worker.run`` changes into the params folder; ``directory`` is the name
    the worker was given), as the shoot and free tasks do; ``directory`` is
    the fallback.
    """
    if worker is None:
        return None
    return getattr(worker, '_directory', None) or getattr(worker, 'directory', None)


def previous_round(run_directory, states):
    """The round of the network the run holds, from ``network{states}.json``; -1 without one.

    AIMMD's trainer stores the fit's ``network_metadata`` at the top level of
    the file (``{'round': ..., ...}``); the old BioEmu2 fit looked for
    ``network_metadata.round`` and so restarted its count at every submission.
    Both layouts are read.
    """
    if run_directory is None:
        return -1
    fname = os.path.join(os.fspath(run_directory), f'network{states}.json')
    try:
        with open(fname) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return -1
    if not isinstance(record, dict):
        return -1
    value = record.get('round', (record.get('network_metadata') or {}).get('round'))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _word62(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 2


def fit_seed(params, run_directory, seed=None):
    """The base seed of a run's fits and where it came from.

    Returns
    -------
    (int, str)
        `seed` (``'argument'``) if given; else
        ``params.cgaimmd_parameters['generation_seed']`` (``'generation_seed'``)
        if set; else a 62-bit hash of the absolute run folder (``'run folder'``),
        so that replicates in different folders differ and a run restarted in
        the same folder reproduces its seeds; 0 (``'default'``) without a run
        folder.

    Notes
    -----
    The hash is of the folder's path: a run moved between machines or disks
    (``/tmp``, an archive, a cluster) gets other seeds after the move. Set
    ``generation_seed`` for a run that will move.
    """
    if seed is not None:
        return int(seed), 'argument'
    settings = getattr(params, 'cgaimmd_settings', None) or dict(getattr(params, 'cgaimmd_parameters', None) or {})
    if settings.get('generation_seed') is not None:
        return int(settings['generation_seed']), 'generation_seed'
    if run_directory is not None:
        return _word62(os.path.abspath(os.fspath(run_directory))), 'run folder'
    return 0, 'default'
