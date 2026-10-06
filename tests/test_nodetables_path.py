"""Node tables through `Path.compute`, as the workers and the trainer use them.

The rows of a trajectory are cached in ``{trajectory}.{series}.npy``, where
the compute ledger treats a zero row as missing: an overflowing frame is
featurized (and reported) again by the next ledger pass, never served. Values
come from the cached series through `NodeTableFeaturizer.batch_dict`. Toy
xtc with 370 atoms (compressed, so it re-reads exactly).
"""
import os

import numpy as np
import pytest

import aimmd
from aimmd._config import NPY_CACHE
from aimmd.network.nodetables import NodeTableFeaturizer
from aimmd.path.utils import get_cache_fname
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames, toy_universe,
                                   write_toy_xtc)

FRAMES = toy_frames(10, seed=5)


def _featurizer(**kwargs):
    return NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                               ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF,
                               **kwargs)


@pytest.fixture
def path(tmp_path):
    NPY_CACHE.clear()
    yield aimmd.Path(write_toy_xtc(tmp_path / 'traj.xtc', FRAMES))
    NPY_CACHE.clear()


def test_rows_are_cached_in_the_series_file(path):
    featurizer = _featurizer()
    series = featurizer.series

    assert path.compute(featurizer.descriptors_function, series) == len(FRAMES)

    stored = np.load(get_cache_fname(path.fname, series))
    assert stored.dtype == np.float32
    assert stored.shape == (len(FRAMES), featurizer.width)
    np.testing.assert_array_equal(
        stored, featurizer.rows_from_coordinates(path.positions))
    np.testing.assert_array_equal(getattr(path, series), stored)
    assert path.compute(featurizer.descriptors_function, series) == 0


def test_the_ledger_featurizes_zero_rows_again(path, capsys):
    counts = _featurizer().rows_from_coordinates(FRAMES)[:, 0]
    featurizer = _featurizer(n_max=int(np.median(counts)))
    overflow = int((counts > featurizer.n_max).sum())
    assert 0 < overflow < len(FRAMES)

    assert path.compute(featurizer.descriptors_function,
                        featurizer.series) == len(FRAMES)
    assert capsys.readouterr().out.count('ERROR') == 1
    # the overflowing frames stay missing: computed (and reported) again
    assert path.compute(featurizer.descriptors_function,
                        featurizer.series) == overflow
    assert capsys.readouterr().out.count('ERROR') == 1


@pytest.mark.graph
def test_values_come_from_the_cached_series(path):
    pytest.importorskip('torch_geometric')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import torch
    featurizer = _featurizer()
    series = featurizer.series
    torch.manual_seed(0)
    embed = torch.nn.Linear(len(ATOM_TYPES), 1)

    def network(batch):
        h = embed(batch['node_attrs'])[:, 0]
        h = h + torch.zeros_like(h).index_add(
            0, batch['edge_index'][1], h[batch['edge_index'][0]])
        return torch.zeros(len(batch['ptr']) - 1).index_add(
            0, batch['batch'], h)

    def values_function(rows, batchsize=4):
        with torch.no_grad():
            return np.concatenate([
                network(featurizer.batch_dict(rows[i:i + batchsize])).numpy()
                for i in range(0, len(rows), batchsize)])

    path.compute(featurizer.descriptors_function, series)
    assert path.compute(values_function, 'values', series) == len(FRAMES)

    expected = values_function(featurizer.rows_from_coordinates(
        path.positions))
    np.testing.assert_array_equal(path.values, expected)
    np.testing.assert_array_equal(
        path.compute(values_function, source=series), expected)
