"""Training on graph keys: aimmd.network.fit with descriptor_cache='graphkeys'.

fit reads every batch through one helper, ``_transform_batch``. Without
graph keys it is ``descriptor_transform(_load_batch_descriptors(...))``, as
fit has always done; these tests pin that down first.
"""

import importlib

import numpy as np

from aimmd._config import NPY_CACHE
from aimmd.cache.npy import save_npy


fit_module = importlib.import_module("aimmd.network.fit")


# --------------------------------------------------------------- npy mode --
def _descriptor_files(tmp_path):
    """Two descriptor files and a batch that interleaves their frames."""
    files = []
    for k, n in enumerate((3, 5)):
        fname = str(tmp_path / f'traj{k}.xtc.descriptors.npy')
        save_npy(fname, np.arange(n * 4, dtype=np.float64).reshape(n, 4) + 100 * k)
        files.append(fname)
    NPY_CACHE.clear()
    batch = np.array([files[1], files[0], files[1], files[1]])
    return batch, np.array([4, 2, 0, 4])


def test_transform_batch_without_keys_transforms_the_loaded_rows(tmp_path):
    """The transform gets exactly what the loader returns, in one call."""
    files, locs = _descriptor_files(tmp_path)
    calls = []

    def transform(x, **kwargs):
        calls.append((x, kwargs))
        return 2 * x

    out = fit_module._transform_batch(transform, files, locs)
    expected = fit_module._load_batch_descriptors(files, locs)
    (x, kwargs), = calls
    assert kwargs == {}
    assert x.dtype == expected.dtype and x.shape == expected.shape
    assert np.array_equal(x, expected)
    assert np.array_equal(out, 2 * expected)

    calls.clear()
    fit_module._transform_batch(transform, files, locs, system_id='s2')
    (x, kwargs), = calls
    assert kwargs == {'system_id': 's2'}
    assert np.array_equal(x, expected)
