"""Tests for the pinned graph-cache key (aimmd.core.graphkey).

The graph cache is addressed by ``sha256(pickle.dumps(row))`` of each
descriptor row, as written by numpy >= 2 under pickle protocol 4. These tests
pin that digest without the graph stack (no torch_geometric needed), so they
run in the default suite:

- the golden digests of the existing caches;
- bitwise identity with the protocol-4 pickle digest, for the float32 fast
  path in both shape encodings (``M`` below 65536 elements, ``J`` above) and
  for the pickle fallback (small rows, float64, big-endian);
- memmap, strided and read-only rows give the key of the plain ndarray;
- the key does not follow the interpreter's default pickle protocol.
"""

import hashlib
import pickle

import numpy as np
import pytest

from aimmd.core import graphkey


#: Float32 descriptor rows and their graph-cache keys, as computed by the
#: implementation the existing caches were written with (Python 3.13, numpy 2,
#: pickle protocol 4). Production caches are multi-GB and addressed by exactly
#: these digests.
GOLDEN_DESCRIPTORS = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 1.25, 0.0],
    ],
    dtype=np.float32,
)
GOLDEN_KEYS = [
    "f66d06949ad999a67ee8c6109595f0eda6cb4b866d4aec961970b1015b914c38",
    "c945a700bc74b9e7c0e067049451cccfb6cfd8a89c4bdb7faf2645dba7d8713c",
]

#: Row lengths that exercise the fast path: the smallest one (16384 float32 =
#: 64 KiB, the size from which pickle writes the bytes outside its frame), both
#: sides of the BININT2/BININT shape encoding at 65536, and the production row
#: of kcmpd09 (56970 atoms x 3).
FAST_SIZES = (16384, 20000, 65535, 65536, 170910)


def _legacy(row, protocol=4):
    """The historical key: sha256 of the pickled row."""
    return hashlib.sha256(pickle.dumps(row, protocol=protocol)).digest()


def _row(n, dtype=np.float32, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(dtype)


def test_key_constants():
    assert graphkey.KEY_BYTES == 32
    assert graphkey.KEY_SCHEME == 'sha256-pickle4-np2-v1'


def test_golden_digests():
    """Cache keys must never change by accident (runs without the graph stack)."""
    assert [graphkey.graph_key(r).hex() for r in GOLDEN_DESCRIPTORS] == GOLDEN_KEYS
    assert graphkey.keys_to_hex(graphkey.graph_keys(GOLDEN_DESCRIPTORS)) == GOLDEN_KEYS


@pytest.mark.parametrize('n', FAST_SIZES)
def test_fast_path_equals_the_pickle_digest(n):
    row = _row(n)
    key = graphkey.graph_key(row)
    assert isinstance(key, bytes) and len(key) == 32
    assert key == _legacy(row)


@pytest.mark.parametrize('n', FAST_SIZES)
def test_fast_path_does_not_call_pickle(n, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError('pickle called on the fast path')
    expected = _legacy(_row(n))
    monkeypatch.setattr(graphkey.pickle, 'dumps', _forbidden)
    assert graphkey.graph_key(_row(n)) == expected


@pytest.mark.parametrize('n, dtype', [
    (9, np.float32), (100, np.float32), (16383, np.float32),
    (9, np.float64), (20000, np.float64), (170910, np.float64),
    (50, np.int64), (0, np.float32),
])
def test_fallback_equals_the_protocol4_digest(n, dtype):
    row = _row(n, dtype=dtype) if n else np.zeros(0, dtype=dtype)
    assert graphkey.graph_key(row) == _legacy(row)


def test_big_endian_rows_fall_back_to_pickle():
    row = _row(20000).astype('>f4')
    assert graphkey.graph_key(row) == _legacy(row)
    # same values, different bytes: a different key, as before
    assert graphkey.graph_key(row) != graphkey.graph_key(row.astype('<f4'))


@pytest.mark.parametrize('n', (9, 20000, 170910))
def test_memmap_rows_give_the_ndarray_key(n, tmp_path):
    rows = np.stack([_row(n, seed=s) for s in range(3)])
    fname = tmp_path / 'rows.npy'
    np.save(fname, rows)
    mapped = np.load(fname, mmap_mode='r')
    assert isinstance(mapped[0], np.memmap)
    for i in range(3):
        assert graphkey.graph_key(mapped[i]) == _legacy(rows[i])
    assert np.array_equal(graphkey.graph_keys(mapped), graphkey.graph_keys(rows))
    if n >= 16384:
        # the historical pickle digest of a memmap row was a different key
        assert _legacy(mapped[0]) != _legacy(rows[0])


@pytest.mark.parametrize('n', (9, 20000, 170910))
def test_strided_and_read_only_rows(n):
    wide = _row(2 * n)
    strided = wide[::2]
    assert not strided.flags.c_contiguous
    assert graphkey.graph_key(strided) == _legacy(np.ascontiguousarray(strided))

    block = np.stack([_row(2 * n, seed=s) for s in range(2)])[:, ::2]
    assert np.array_equal(graphkey.graph_keys(block),
                          graphkey.graph_keys(np.ascontiguousarray(block)))

    frozen = _row(n)
    frozen.flags.writeable = False
    assert graphkey.graph_key(frozen) == _legacy(frozen.copy())


@pytest.mark.parametrize('n', (9, 20000, 170910))
def test_key_does_not_follow_the_default_protocol(n):
    """Python 3.14 pickles with protocol 5 by default; the key stays protocol 4."""
    row = _row(n)
    key = graphkey.graph_key(row)
    assert key == _legacy(row, protocol=4)
    if n >= 16384:
        # numpy hands protocol 5 an out-of-band-capable buffer: other bytes
        assert key != _legacy(row, protocol=5)


def test_graph_keys_shapes_and_inputs():
    rows = np.stack([_row(20000, seed=s) for s in range(4)])
    keys = graphkey.graph_keys(rows)
    assert keys.shape == (4, 32) and keys.dtype == np.uint8
    for row, key in zip(rows, keys):
        assert bytes(key) == _legacy(row)
    # a list of rows works too
    assert np.array_equal(graphkey.graph_keys(list(rows)), keys)
    # empty inputs give (0, 32)
    assert graphkey.graph_keys(np.zeros((0, 20000), np.float32)).shape == (0, 32)
    assert graphkey.graph_keys([]).shape == (0, 32)
    # a single row is not a batch of rows
    with pytest.raises(ValueError):
        graphkey.graph_keys(rows[0])


def test_hex_round_trip_and_key_batches():
    keys = graphkey.graph_keys(GOLDEN_DESCRIPTORS)
    hexes = graphkey.keys_to_hex(keys)
    assert hexes == GOLDEN_KEYS
    assert np.array_equal(graphkey.hex_to_keys(hexes), keys)
    assert graphkey.hex_to_keys([]).shape == (0, 32)

    assert graphkey.is_key_batch(keys)
    assert graphkey.is_key_batch(np.zeros((0, 32), np.uint8))
    assert not graphkey.is_key_batch(keys.astype(np.int16))
    assert not graphkey.is_key_batch(keys[0])
    assert not graphkey.is_key_batch(np.zeros((2, 31), np.uint8))
    assert not graphkey.is_key_batch(GOLDEN_DESCRIPTORS)
    assert not graphkey.is_key_batch(keys.tolist())
