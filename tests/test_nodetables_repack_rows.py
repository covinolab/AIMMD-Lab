"""Widening node-table rows: `NodeTableFeaturizer.with_n_max` and
`repack_rows`.

A frame with more graph nodes than ``n_max`` gets a zero row. The way out is
a wider layout: `with_n_max` gives the featurizer of the same graphs with
another row capacity (hence another series name), and `repack_rows` rewrites
stored rows into that layout without reading any trajectory. Rewritten rows
must equal featurizing the frames with the wider featurizer, bit for bit;
zero rows stay zero, for the ledger (or ``prefill --only-missing``) to fill.
Needs only numpy and MDAnalysis.
"""
import numpy as np
import pytest

from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      NodeTableFeaturizer,
                                      NodeTableOverflowError)
from aimmd.network.nodetables._featurizer import repack_rows
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, reference_nodes,
                                   toy_frames, toy_universe)

FRAMES = toy_frames(12, seed=7)


def _featurizer(**kwargs):
    return NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                               ENVIRONMENT_SELECTION, ATOM_TYPES,
                               cutoff=CUTOFF, **kwargs)


def _node_counts():
    universe = toy_universe()
    return np.array([len(reference_nodes(universe, frame)[1])
                     for frame in FRAMES])


def _bits(rows):
    return np.ascontiguousarray(rows).view(np.uint32)


def test_with_n_max_is_the_same_featurizer_with_another_capacity():
    featurizer = _featurizer(n_max=64, max_num_neighbors=16)
    wider = featurizer.with_n_max(96)

    assert isinstance(wider, NodeTableFeaturizer)
    assert wider.n_max == 96 and wider.width == 2 + 4 * 96
    assert (wider.system_selection, wider.environment_selection,
            wider.atom_types, wider.cutoff, wider.max_num_neighbors) == (
        featurizer.system_selection, featurizer.environment_selection,
        featurizer.atom_types, featurizer.cutoff,
        featurizer.max_num_neighbors)
    assert wider.series == _featurizer(n_max=96).series != featurizer.series
    assert featurizer.with_n_max(64).series == featurizer.series
    spec = featurizer.spec()
    spec['n_max'] = 96
    assert wider.spec() == spec


def test_with_n_max_shares_the_universe_without_changing_rows():
    featurizer = _featurizer(n_max=64)
    before = featurizer.rows_from_coordinates(FRAMES)
    wider = featurizer.with_n_max(96)
    assert wider.universe is featurizer.universe
    wider.rows_from_coordinates(FRAMES)
    assert np.array_equal(_bits(featurizer.rows_from_coordinates(FRAMES)),
                          _bits(before))


def test_repacked_rows_equal_featurizing_with_the_wider_layout(capsys):
    counts = _node_counts()
    n_max = int(np.median(counts))
    overflow = counts > n_max
    assert overflow.any() and not overflow.all()
    narrow = _featurizer(n_max=n_max)
    wide = narrow.with_n_max(int(counts.max()) + 3)
    rows = narrow.rows_from_coordinates(FRAMES)
    capsys.readouterr()

    repacked = repack_rows(rows, wide.n_max)

    assert repacked.dtype == np.float32
    assert repacked.shape == (len(FRAMES), wide.width)
    expected = wide.rows_from_coordinates(FRAMES)
    assert np.array_equal(_bits(repacked[~overflow]),
                          _bits(expected[~overflow]))
    # the overflowing frames stay zero: missing rows for the ledger
    assert not repacked[overflow].any()


def test_repack_rows_can_narrow_rows_that_fit():
    counts = _node_counts()
    wide = _featurizer(n_max=int(counts.max()) + 10)
    narrow = wide.with_n_max(int(counts.max()))
    rows = wide.rows_from_coordinates(FRAMES)

    assert np.array_equal(_bits(repack_rows(rows, narrow.n_max)),
                          _bits(narrow.rows_from_coordinates(FRAMES)))
    with pytest.raises(NodeTableOverflowError, match='n_max'):
        repack_rows(rows, int(counts.max()) - 1)


def test_repack_rows_keeps_zero_rows_and_handles_no_rows():
    featurizer = _featurizer(n_max=64)
    rows = featurizer.rows_from_coordinates(FRAMES[:3])
    rows[1] = 0
    repacked = repack_rows(rows, 80)
    assert not repacked[1].any() and repacked[[0, 2], 0].all()
    assert repack_rows(np.zeros((0, featurizer.width), np.float32),
                       80).shape == (0, 2 + 4 * 80)


def _rows_of_layout(layout):
    rows = np.zeros((2, 2 + 4 * 8), np.float32)
    rows[:, 0] = 1
    rows[:, 1] = layout
    return rows


@pytest.mark.parametrize('rows, match', [
    (np.zeros((2, 7), np.float32), 'width'),
    (np.zeros(10, np.float32), 'shape'),
    (_rows_of_layout(7), 'layout'),
    (_rows_of_layout(1).astype(np.float64), 'float32'),
])
def test_repack_rows_refuses_what_is_not_node_table_rows(rows, match):
    with pytest.raises(ValueError, match=match):
        repack_rows(rows, 16)


def test_repack_rows_refuses_an_invalid_capacity():
    rows = _featurizer(n_max=64).rows_from_coordinates(FRAMES[:1])
    with pytest.raises(ValueError, match='n_max'):
        repack_rows(rows, 0)


def test_multi_system_with_n_max_widens_every_system():
    featurizers = MultiSystemNodeTableFeaturizer({
        'a': _featurizer(n_max=64), 'b': _featurizer(n_max=80)})
    wider = featurizers.with_n_max(96)

    assert isinstance(wider, MultiSystemNodeTableFeaturizer)
    assert wider.system_ids == ['a', 'b']
    assert wider['a'].n_max == wider['b'].n_max == 96
    assert wider.series == MultiSystemNodeTableFeaturizer({
        'a': _featurizer(n_max=96), 'b': _featurizer(n_max=96)}).series
    assert wider.series != featurizers.series
