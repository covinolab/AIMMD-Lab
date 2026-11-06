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

The preflight (:mod:`aimmd.cgaimmd.preflight`) checks a params file's fit
keywords before a job starts: a fit's keywords live in the params file's
``fit`` wrapper and cannot be read back, so the preflight calls that wrapper
inside :func:`settings_only`, where both fits validate their keywords and
raise :class:`FitSettings` with them instead of training
(:func:`cgaimmd_fits_behind` tells whether a ``fit`` reaches one of them).
"""

import contextlib
import functools
import hashlib
import json
import os
import threading
import types

__all__ = ['run_directory', 'previous_round', 'fit_seed', 'FitSettings', 'settings_only',
           'settings_only_active', 'CGAIMMD_FITS', 'cgaimmd_fits_behind']


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


# ---------------------------------------------------------------------------
# the fit keywords, checked without training (the preflight)
# ---------------------------------------------------------------------------

class FitSettings(Exception):
    """Raised by a C-GAIMMD fit called inside :func:`settings_only`: its keywords, nothing trained.

    Attributes
    ----------
    fit : str
        The fit (``'aimmd.cgaimmd.flow_matching.fit'``, ``'aimmd.cgaimmd.bioemu2.fit'``).
    settings : dict
        Its keywords after validation, defaults included.
    keywords : dict
        The keywords the call passed (what the params file sets).
    problems : list of str
        Settings that are valid one by one but would not do what C-GAIMMD
        needs (each names the keyword and what to set instead); the
        preflight fails on them.
    notes : list of str
        Plain facts for the report (e.g. where every round starts from).
    """

    def __init__(self, fit, settings, keywords=None, problems=(), notes=()):
        self.fit = str(fit)
        self.settings = dict(settings)
        self.keywords = dict(keywords or {})
        self.problems = list(problems)
        self.notes = list(notes)
        super().__init__(f'{self.fit}: keywords checked, nothing trained')


_SETTINGS_ONLY = threading.local()


@contextlib.contextmanager
def settings_only():
    """Within this block a C-GAIMMD fit checks its keywords and raises :class:`FitSettings`.

    The fits (:func:`aimmd.cgaimmd.flow_matching.fit.flow_matching_fit`,
    :func:`aimmd.cgaimmd.bioemu2.fit.sequential_fit`) validate their keywords
    and their start as in a real round, then raise instead of reading data or
    training. Bad keywords raise as they would in the trainer. Per thread.
    """
    previous = getattr(_SETTINGS_ONLY, 'active', False)
    _SETTINGS_ONLY.active = True
    try:
        yield
    finally:
        _SETTINGS_ONLY.active = previous


def settings_only_active():
    """Whether the calling thread is inside :func:`settings_only`."""
    return getattr(_SETTINGS_ONLY, 'active', False)


CGAIMMD_FITS = {
    'aimmd.cgaimmd.flow_matching.fit.flow_matching_fit': 'the flow-matching fit',
    'aimmd.cgaimmd.bioemu2.fit.sequential_fit': 'the BioEmu2 sequential fit',
}
"""The C-GAIMMD fits that answer :func:`settings_only`, by module and name."""

FOLLOWED_CALL_LEVELS = 2
"""How many levels of module-level helper functions :func:`cgaimmd_fits_behind` follows."""


def _qualified_name(item):
    return f"{getattr(item, '__module__', '')}.{getattr(item, '__qualname__', '')}"


def _referenced(function):
    """The objects a Python function names: its globals (by name) and its closure cells."""
    code = getattr(function, '__code__', None)
    names = getattr(function, '__globals__', None)
    if code is None or names is None:
        return []
    out = [names[name] for name in code.co_names if name in names]
    for cell in getattr(function, '__closure__', None) or ():
        try:
            out.append(cell.cell_contents)
        except ValueError:                                  # an empty cell
            pass
    return out


def _fits_behind(item, level, seen, found):
    if item is None or id(item) in seen:
        return
    seen.add(id(item))
    name = _qualified_name(item)
    if name in CGAIMMD_FITS:
        found.add(name)
        return
    if isinstance(item, functools.partial):
        for inner in (item.func, *item.args, *item.keywords.values()):
            _fits_behind(inner, level, seen, found)
        return
    item = getattr(item, '__func__', item)                  # a bound method's function
    for inner in _referenced(item):
        if _qualified_name(inner) in CGAIMMD_FITS:
            found.add(_qualified_name(inner))
        elif level < FOLLOWED_CALL_LEVELS and isinstance(inner, (types.FunctionType,
                                                                 functools.partial)):
            _fits_behind(inner, level + 1, seen, found)


def cgaimmd_fits_behind(function):
    """The C-GAIMMD fits (:data:`CGAIMMD_FITS` names) that `function` is or calls.

    Followed: the function itself, ``functools.partial``, the names its code
    uses from its module's globals, its closure, and module-level helper
    functions it calls by a global name (:data:`FOLLOWED_CALL_LEVELS` deep):
    the params-file pattern ``def fit(params, pathensemble, verbose=False,
    worker=None): return flow_matching_fit(..., steps_per_round=4000)``.

    Returns
    -------
    list of str
        Sorted; empty when `function` reaches none (AIMMD's own fit, a custom fit).
    """
    found = set()
    _fits_behind(function, 0, set(), found)
    return sorted(found)
