"""
aimmd.core.graphkey
===================

The graph-cache key, pinned.

The graph cache (``aimmd.network.graph_utils``) stores one graph per frame
under ``sha256(pickle.dumps(row))`` of the frame's descriptor row -- a 1-D
float32 coordinate row, written by numpy >= 2 under pickle protocol 4 (the
default protocol up to Python 3.13). Production caches are multi-GB and
addressed by exactly these digests, so the digest must not move.

Computed through ``pickle.dumps``, it silently would:

- Python 3.14 made protocol 5 the default, and numpy pickles differently
  under it;
- numpy 1.x names its reconstructor ``numpy.core.multiarray`` instead of
  ``numpy._core.multiarray``;
- an ``np.memmap`` row (e.g. from ``np.load(..., mmap_mode='r')``) pickles as
  a memmap, not as an ndarray.

Each of these re-keys every frame and turns every cache lookup into a miss.

:func:`graph_key` therefore reproduces the protocol-4 / numpy-2 bytes
WITHOUT calling pickle for the rows AIMMD actually stores: a 1-D
little-endian float32 row of at least 16384 elements is hashed as a fixed
prefix, the raw row bytes, and a fixed suffix (see :func:`_framing`). Any
other row falls back to ``pickle.dumps(row, protocol=4)`` of the plain
ndarray. The digest equals the historical one for every row the caches hold.

The key scheme is named by :data:`KEY_SCHEME`. A different scheme needs a
different name (and a different cache).

This module needs only numpy and the standard library, so key files can be
written and checked without the graph stack.
"""

# external
import hashlib
import pickle
import struct
import numpy as np


#: Size of one key in bytes (a SHA-256 digest).
KEY_BYTES = 32

#: Name of the key scheme: SHA-256 of the pickle protocol-4 bytes of the row
#: as numpy >= 2 writes them, version 1.
KEY_SCHEME = 'sha256-pickle4-np2-v1'

#: Smallest float32 row hashed without pickle. From 16384 elements (64 KiB)
#: on, pickle protocol 4 writes the row bytes outside its frame, so prefix
#: and suffix do not depend on the row contents.
_FAST_MIN_SIZE = 16384

_FLOAT32 = np.dtype('<f4')

# (prefix, suffix) per row length, built on first use
_FRAMING = {}


def _framing(n):
    """Pickle bytes around the data of a 1-D '<f4' row of ``n`` elements.

    Returns ``(prefix, suffix)`` such that, under numpy >= 2,

        pickle.dumps(row, protocol=4) == prefix + row.tobytes() + suffix

    for every 1-D little-endian float32 ndarray ``row`` of ``n >= 16384``
    elements. The bytes are spelled out rather than taken from a reference
    pickle, so that they do not depend on the running interpreter or numpy.

    Layout (pickle opcodes):

    - ``PROTO 4``, then ``FRAME`` with the length of the frame body;
    - frame body: the ``numpy._core.multiarray._reconstruct`` and
      ``numpy.ndarray`` globals, the ``(ndarray, (0,), b'b')`` call, and the
      state tuple ``(1, (n,), dtype('f4') with state
      (3, '<', None, None, None, -1, -1, 0), False, ...``. The shape uses
      ``BININT2`` (``M``) below 65536 elements and ``BININT`` (``J``) from
      65536 on;
    - outside the frame: ``BINBYTES`` (``B``) with the 4-byte data length,
      then the data itself;
    - a second ``FRAME`` of 5 bytes closing the state tuple, ``BUILD`` and
      ``STOP``.
    """
    framing = _FRAMING.get(n)
    if framing is not None:
        return framing
    module = b'numpy._core.multiarray'
    shape = (b'J' + struct.pack('<i', n) if n >= 65536
             else b'M' + struct.pack('<H', n))
    body = (b'\x8c' + bytes([len(module)]) + module
            + b'\x94\x8c\x0c_reconstruct\x94\x93\x94'
            b'\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94'
            b'K\x00\x85\x94C\x01b\x94\x87\x94R\x94'
            b'(K\x01' + shape + b'\x85\x94'
            b'h\x03\x8c\x05dtype\x94\x93\x94'
            b'\x8c\x02f4\x94\x89\x88\x87\x94R\x94'
            b'(K\x03\x8c\x01<\x94NNN'
            b'J\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89')
    prefix = (b'\x80\x04\x95' + struct.pack('<Q', len(body)) + body
              + b'B' + struct.pack('<I', 4 * n))
    suffix = b'\x95' + struct.pack('<Q', 5) + b'\x94t\x94b.'
    framing = _FRAMING[n] = (prefix, suffix)
    return framing


def graph_key(row):
    """Graph-cache key of one descriptor row.

    Parameters
    ----------
    row : array-like
        One descriptor row, normally a 1-D float32 array of flattened
        atomic coordinates. Subclasses (``np.memmap``), strided views and
        read-only arrays are keyed like the plain contiguous ndarray with
        the same contents.

    Returns
    -------
    bytes
        The 32-byte digest: ``sha256(pickle.dumps(row, protocol=4))`` of the
        plain ndarray under numpy >= 2 (``bytes.hex()`` of it is the key
        stored in the graph cache).
    """
    row = np.asarray(row)       # strips np.memmap and other subclasses
    if (row.ndim == 1 and row.dtype == _FLOAT32
            and row.size >= _FAST_MIN_SIZE):
        prefix, suffix = _framing(row.size)
        digest = hashlib.sha256(prefix)
        digest.update(memoryview(np.ascontiguousarray(row)).cast('B'))
        digest.update(suffix)
        return digest.digest()
    # small rows, other dtypes or shapes: the pickle bytes themselves, with
    # the protocol pinned (never the interpreter's default)
    return hashlib.sha256(
        pickle.dumps(row.view(np.ndarray), protocol=4)).digest()


def graph_keys(rows):
    """Graph-cache keys of a batch of descriptor rows.

    Parameters
    ----------
    rows : array-like
        Two-dimensional array (or sequence) of descriptor rows, one per
        frame. A ``np.memmap`` is read row by row, without a full copy.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, 32)`` uint8 array; row ``i`` is ``graph_key(rows[i])``.

    Raises
    ------
    ValueError
        If ``rows`` is a single row (fewer than two dimensions).
    """
    rows = np.asarray(rows)
    if rows.size == 0 and rows.ndim < 2:
        return np.zeros((0, KEY_BYTES), dtype=np.uint8)
    if rows.ndim < 2:
        raise ValueError('graph_keys expects a batch of rows (2-D); '
                         'use graph_key for a single row')
    result = np.empty((len(rows), KEY_BYTES), dtype=np.uint8)
    for i, row in enumerate(rows):
        result[i] = np.frombuffer(graph_key(row), dtype=np.uint8)
    return result


def is_key_batch(x):
    """Whether ``x`` is a batch of graph keys: an ``(n, 32)`` uint8 array.

    Descriptor rows are floating point, so this tells a batch of keys apart
    from a batch of coordinate rows.
    """
    return (isinstance(x, np.ndarray) and x.dtype == np.uint8
            and x.ndim == 2 and x.shape[1] == KEY_BYTES)


def pad_keys(stored, length, source='graph-key array'):
    """Stored key rows, padded with zero rows to at least ``length`` rows.

    A per-frame key file may be missing or shorter than its trajectory (a
    frame not keyed yet, a half that was never ingested); its absent rows
    are zero rows, which mean "not computed".

    Parameters
    ----------
    stored : numpy.ndarray or None
        The stored rows, e.g. the content of ``<traj>.graphkeys.npy``; None
        if there are none.
    length : int
        Minimum number of rows of the result.
    source : str, optional
        What ``stored`` is, for the error message (e.g. the file name).

    Returns
    -------
    numpy.ndarray
        ``(max(len(stored), length), 32)`` uint8: ``stored`` itself if it is
        long enough, otherwise a padded copy.

    Raises
    ------
    RuntimeError
        If ``stored`` is not an ``(n, 32)`` uint8 array.
    """
    if stored is None:
        return np.zeros((max(int(length), 0), KEY_BYTES), dtype=np.uint8)
    if (getattr(stored, 'dtype', None) != np.uint8
            or getattr(stored, 'ndim', 0) != 2
            or stored.shape[1] != KEY_BYTES):
        raise RuntimeError(
            f'{source!r} is not a graph-key file (dtype '
            f'{getattr(stored, "dtype", None)}, shape '
            f'{getattr(stored, "shape", None)}; expected uint8 rows of '
            f'{KEY_BYTES})')
    if len(stored) >= length:
        return stored
    result = np.zeros((int(length), KEY_BYTES), dtype=np.uint8)
    result[:len(stored)] = stored
    return result


def keys_to_hex(keys):
    """Hex strings (the graph-cache keys) of an ``(n, 32)`` uint8 key batch."""
    keys = np.ascontiguousarray(keys, dtype=np.uint8).reshape(-1, KEY_BYTES)
    return [row.tobytes().hex() for row in keys]


def hex_to_keys(hexes):
    """Inverse of :func:`keys_to_hex`: an ``(n, 32)`` uint8 key batch."""
    hexes = list(hexes)
    if not hexes:
        return np.zeros((0, KEY_BYTES), dtype=np.uint8)
    raw = b''.join(bytes.fromhex(h) for h in hexes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, KEY_BYTES).copy()
