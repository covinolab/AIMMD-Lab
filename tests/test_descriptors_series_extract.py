"""Paths read a named descriptors series like the historical one.

`Path._extract` serves 'descriptors' and 'states' from `NPY_CACHE` (they only
grow, so a long enough resident copy is reused), and reloads every other
`.npy` series, whose values can change. A named descriptors series
('descriptors-*') is the same kind of data as 'descriptors' and is cached the
same way.

A run that uses a named series has no 'descriptors' file. Asking such a file
for 'descriptors' used to return zeros silently (`path.descriptors`, or
`raise_if_missing=False`); it now raises an error that names the series that
does exist.
"""
import os

import numpy as np
import pytest

from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy
from aimmd.path.utils import get_cache_fname
from tests._helpers_unit import build_path

SERIES = "descriptors-gn0123456789"
X = np.array([-0.9, -0.4, 0.1, 0.6, 0.95])


def _positions(x):
    x = np.asarray(x, dtype=np.float32)
    zeros = np.zeros_like(x)
    return np.stack([np.stack([x, zeros, zeros], 1),
                     np.stack([x + 2.0, zeros, zeros], 1)], 1)


def _rows(n):
    """Three-column rows, distinct per frame."""
    return np.arange(3.0 * n).reshape(n, 3) + 1.0


@pytest.fixture
def path(tmp_path):
    """A path whose descriptors live in the named series only."""
    path = build_path(tmp_path, stem="path000001", positions=_positions(X))
    os.remove(get_cache_fname(path.fname, "descriptors"))
    save_npy(get_cache_fname(path.fname, SERIES), _rows(len(X)))
    NPY_CACHE.clear()
    yield path
    NPY_CACHE.clear()


def test_named_series_is_read_like_descriptors(path):
    rows = _rows(len(X))
    np.testing.assert_array_equal(getattr(path, SERIES), rows)
    np.testing.assert_array_equal(path.get(SERIES), rows)
    np.testing.assert_array_equal(path[1:4]._get(SERIES), rows[1:4])
    np.testing.assert_array_equal(path[::-1]._get(SERIES), rows[::-1])
    assert path._extract(0, SERIES, raise_if_missing=True).shape == (5, 3)


def test_named_series_is_served_from_npy_cache(path, monkeypatch):
    """Resident rows are reused, as for 'descriptors' (values are not)."""
    first = path._extract(0, SERIES)

    def no_reload(fname):
        raise AssertionError(f"reloaded {fname}")

    monkeypatch.setattr(NPY_CACHE, "load", no_reload)
    np.testing.assert_array_equal(path._extract(0, SERIES), first)


def test_named_series_feeds_values(path):
    values = path.compute(lambda rows: rows[:, 0] * 10.0, "values",
                          SERIES, overwrite=True)
    assert values == len(X)
    NPY_CACHE.clear()
    np.testing.assert_array_equal(path.values, _rows(len(X))[:, 0] * 10.0)


def test_missing_named_series(path):
    os.remove(get_cache_fname(path.fname, SERIES))
    NPY_CACHE.clear()
    with pytest.raises(TypeError, match="could not obtain"):
        path._extract(0, SERIES, raise_if_missing=True)
    assert not path._extract(0, SERIES, raise_if_missing=False).any()


@pytest.mark.parametrize("raise_if_missing", [True, False])
def test_descriptors_raises_when_only_a_named_series_exists(
        path, raise_if_missing):
    with pytest.raises(TypeError) as error:
        path._extract(0, "descriptors", raise_if_missing=raise_if_missing)
    message = str(error.value)
    assert os.path.basename(get_cache_fname(path.fname, SERIES)) in message
    assert "descriptors_series" in message


def test_descriptors_attribute_names_the_series(path):
    with pytest.raises(AttributeError, match=SERIES):
        path.descriptors
    with pytest.raises(TypeError, match=SERIES):
        path._get("descriptors")          # raise_if_missing=False


def test_value_pass_from_descriptors_skips_such_a_file(path):
    """Path.compute skips a source it cannot read, unless raise_if_error."""
    assert path.compute(lambda rows: rows[:, 0], "values", "descriptors",
                        overwrite=True) == 0
    with pytest.raises(TypeError, match=SERIES):
        path.compute(lambda rows: rows[:, 0], "values", "descriptors",
                     overwrite=True, raise_if_error=True)


def test_descriptors_without_any_series_still_gives_zeros(path):
    os.remove(get_cache_fname(path.fname, SERIES))
    NPY_CACHE.clear()
    assert not path._extract(0, "descriptors", raise_if_missing=False).any()
    assert path.descriptors.shape == (len(X),)
    with pytest.raises(TypeError, match="could not obtain"):
        path._extract(0, "descriptors", raise_if_missing=True)


def test_descriptors_is_read_when_both_series_exist(path):
    legacy = _rows(len(X)) * -1.0
    save_npy(get_cache_fname(path.fname, "descriptors"), legacy)
    NPY_CACHE.clear()
    np.testing.assert_array_equal(path.descriptors, legacy)
    np.testing.assert_array_equal(getattr(path, SERIES), _rows(len(X)))
