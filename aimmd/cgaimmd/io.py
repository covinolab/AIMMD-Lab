"""
aimmd.cgaimmd.io
================

Atomic writes and file hashing: the one write protocol of C-GAIMMD code.

Every file C-GAIMMD code writes into a run directory (a generator fit's side
checkpoint, network metadata, in-flight records, per-shot sidecars) goes
through :func:`atomic_write_bytes`: the bytes are written into a temporary
file next to the target, flushed and ``fsync``-ed, and then renamed over the
target with :func:`os.replace`, so a reader sees either the old or the new
file, never a partial one. Readers match files to each other by content hash
(:func:`file_sha256`) instead of by locks.

Only the standard library and numpy are imported at module level; torch is
imported inside :func:`atomic_torch_save`.
"""

import hashlib
import io as _io
import json
import os
import time
from numbers import Integral
from pathlib import PurePath

import numpy as np


_CHUNK = 1 << 20


def atomic_write_bytes(fname, data, *, keep_previous=False):
    """Write `data` to `fname` atomically.

    Parameters
    ----------
    fname : str or os.PathLike
        Target file.
    data : bytes
        The complete new content.
    keep_previous : bool, optional
        If True and `fname` exists, it is first renamed to ``fname + '.prev'``.

    Notes
    -----
    The bytes go to ``f'{fname}.tmp.{os.getpid()}'``, which is flushed and
    ``os.fsync``-ed before ``os.replace(tmp, fname)``. With `keep_previous`
    an older ``.prev`` is replaced; readers that find no `fname` then try the
    ``.prev`` file. On any error the temporary file is removed and the error
    re-raised; `fname` is then unchanged (or, with `keep_previous`, possibly
    already moved to ``.prev``).
    """
    fname = os.fspath(fname)
    tmp = f'{fname}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if keep_previous and os.path.exists(fname):
            os.replace(fname, fname + '.prev')
        os.replace(tmp, fname)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _json_default(obj):
    """Convert the non-JSON types C-GAIMMD records commonly hold."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def atomic_write_json(fname, obj, **kw):
    """Write `obj` as JSON to `fname` atomically.

    Parameters
    ----------
    fname : str or os.PathLike
        Target file.
    obj : object
        JSON-serializable object (numpy values and paths are converted).
    **kw
        Forwarded to :func:`atomic_write_bytes` (``keep_previous``).

    Notes
    -----
    The text is ``json.dumps(obj, indent=1, sort_keys=True)``, so equal
    records give equal files; numpy scalars and arrays are converted with
    ``.item()`` / ``.tolist()`` and paths with ``str``. Serialization happens
    before anything is written: an unserializable object raises
    ``TypeError`` and leaves `fname` untouched.
    """
    text = json.dumps(obj, indent=1, sort_keys=True, default=_json_default)
    atomic_write_bytes(fname, text.encode('utf-8'), **kw)


def atomic_torch_save(obj, fname, **kw):
    """``torch.save`` `obj` to `fname` atomically and return the sha256 of the bytes.

    Parameters
    ----------
    obj : object
        What to save (for network files: a state dict of tensors).
    fname : str or os.PathLike
        Target file.
    **kw
        Forwarded to :func:`atomic_write_bytes` (``keep_previous``).

    Returns
    -------
    str
        Hex sha256 of the bytes written (equal to :func:`file_sha256` of `fname` afterwards).
    """
    import torch
    buffer = _io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()
    atomic_write_bytes(fname, data, **kw)
    return hashlib.sha256(data).hexdigest()


def file_sha256(fname, *, retries=5, wait=0.2):
    """Hex sha256 digest of a file's bytes, robust to the file being replaced.

    Parameters
    ----------
    fname : str or os.PathLike
        File to hash.
    retries : int, optional
        Number of attempts (>= 1); an ``OSError`` while reading, a missing file included, is retried.
    wait : float, optional
        Seconds between attempts.

    Returns
    -------
    str or None
        The digest, or None if the file does not exist at the last attempt.

    Raises
    ------
    OSError
        The last error, if the file exists but could not be read in `retries` attempts.
    ValueError
        If `retries` is not an int >= 1.

    Notes
    -----
    Retries cover a file being replaced on a network file system or a
    ``keep_previous`` write in progress. A missing file is retried too, so it
    costs ``(retries - 1) * wait`` seconds (0.8 s by default) before None:
    code that polls for a file that may not exist yet checks
    :func:`os.path.exists` first or passes ``retries=1``.
    """
    if isinstance(retries, bool) or not isinstance(retries, Integral) or retries < 1:
        raise ValueError(f'retries must be an int >= 1, got {retries!r}')
    fname = os.fspath(fname)
    for attempt in range(int(retries)):
        last = attempt == retries - 1
        try:
            digest = hashlib.sha256()
            with open(fname, 'rb') as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b''):
                    digest.update(chunk)
            return digest.hexdigest()
        except FileNotFoundError:
            if last:
                return None
        except OSError:
            if last:
                raise
        time.sleep(wait)
