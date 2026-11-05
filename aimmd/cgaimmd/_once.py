"""
aimmd.cgaimmd._once
===================

Warn-once and log-once flags: the only process-global memory of C-GAIMMD code.

C-GAIMMD code prints some warnings once per process (for example the
``window_after_trained_shots`` warning). Keeping those flags in one place is
what lets the test suite reset them between tests (``tests/conftest.py``), so
that no test depends on the collection order.

Standard library only.
"""

import threading

_SEEN = set()
_LOCK = threading.Lock()


def once(key):
    """Return True the first time `key` is seen in this process.

    Parameters
    ----------
    key : hashable
        Any hashable identifier of the event (a string, or a tuple such as
        ``('clash_fallback', topology_file)``).

    Returns
    -------
    bool
        True on the first call with `key` since the process started (or since
        the last :func:`_reset`), False afterwards.

    Examples
    --------
    >>> if once('window_after_trained_shots default'):
    ...     print('WARNING: ...')
    """
    with _LOCK:
        if key in _SEEN:
            return False
        _SEEN.add(key)
        return True


def _reset():
    """Forget every key (test isolation; not for production code)."""
    with _LOCK:
        _SEEN.clear()
