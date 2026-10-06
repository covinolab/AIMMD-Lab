"""`MultiSystemNodeTableFeaturizer`: one node-table featurizer per system.

Multi-system runs call the data functions with a ``system_id`` keyword. The
dispatcher routes every call to that system's `NodeTableFeaturizer` (own
topology, selections and n_max, so its own row width), keeps one atom-type
table for the shared network, and names one series for the campaign that
changes when any system's featurizer changes. Two toy systems: the toy of
tests/_nodetables_toy.py, and the same without its last ion, with a smaller
n_max.
"""
import os

import MDAnalysis as mda
import numpy as np
import pytest

from aimmd.core.utils import accepts_system_id
from aimmd.network.nodetables import (MultiSystemNodeTableFeaturizer,
                                      NodeTableFeaturizer, SERIES_PREFIX)
from tests._nodetables_toy import (ATOM_TYPES, BOX, CUTOFF,
                                   ENVIRONMENT_SELECTION, SYSTEM_SELECTION,
                                   memory_trajectory, toy_frames, toy_universe)

FRAMES = toy_frames(6, seed=3)


def _small_universe():
    universe = mda.Merge(toy_universe().atoms[:-1])
    universe.dimensions = BOX
    return universe


def _featurizers(n_max_1=128, atom_types_1=ATOM_TYPES, environment_1=None):
    first = NodeTableFeaturizer(toy_universe(), SYSTEM_SELECTION,
                                ENVIRONMENT_SELECTION, ATOM_TYPES, CUTOFF)
    second = NodeTableFeaturizer(
        _small_universe(), SYSTEM_SELECTION,
        environment_1 or ENVIRONMENT_SELECTION, atom_types_1, CUTOFF,
        n_max=n_max_1)
    return MultiSystemNodeTableFeaturizer({'ligA': first, 'ligB': second})


def test_rows_come_from_the_featurizer_of_the_system():
    featurizers = _featurizers()
    rows_a = featurizers.descriptors_function(memory_trajectory(FRAMES),
                                              system_id='ligA')
    rows_b = featurizers.rows_from_coordinates(FRAMES[:, :-1], 'ligB')

    assert rows_a.shape == (len(FRAMES), featurizers['ligA'].width)
    assert rows_b.shape == (len(FRAMES), 2 + 4 * 128)
    assert np.array_equal(
        rows_a, featurizers['ligA'].rows_from_coordinates(FRAMES))
    assert np.array_equal(
        rows_b, featurizers['ligB'].rows_from_coordinates(FRAMES[:, :-1]))


def test_system_ids_are_compared_as_strings():
    featurizers = MultiSystemNodeTableFeaturizer(
        {0: _featurizers()['ligA'], 1: _featurizers()['ligB']})
    assert featurizers.system_ids == ['0', '1']
    assert featurizers[1] is featurizers['1']


def test_an_unknown_system_raises_a_clear_key_error():
    with pytest.raises(KeyError, match="'ligC'.*ligA"):
        _featurizers().descriptors_function(memory_trajectory(FRAMES),
                                            system_id='ligC')


def test_the_wrapper_signature_carries_the_system_id():
    featurizers = _featurizers()

    def descriptors_function(trajectory, system_id):
        return featurizers.descriptors_function(trajectory, system_id)

    assert accepts_system_id(descriptors_function)


def test_one_series_name_for_the_campaign():
    featurizers = _featurizers()
    assert featurizers.series.startswith(SERIES_PREFIX)
    assert len(featurizers.series) == len(SERIES_PREFIX) + 10
    assert featurizers.series not in (featurizers['ligA'].series,
                                      featurizers['ligB'].series)
    assert featurizers.series == _featurizers().series
    assert featurizers.check_series(featurizers.series) == featurizers.series
    assert featurizers.spec() == {'ligA': featurizers['ligA'].spec(),
                                  'ligB': featurizers['ligB'].spec()}
    assert featurizers.atom_types == ATOM_TYPES


@pytest.mark.parametrize('change', [
    dict(n_max_1=256),
    dict(environment_1='not type H and around 6.0 (resname LIG)')])
def test_any_system_change_changes_the_series_name(change):
    featurizers = _featurizers(**change)
    assert featurizers.series != _featurizers().series
    with pytest.raises(ValueError, match='ligB'):
        featurizers.check_series(_featurizers().series)


def test_the_systems_must_share_the_atom_types():
    with pytest.raises(ValueError, match='atom_types'):
        _featurizers(atom_types_1=ATOM_TYPES + ['CL'])


def test_no_systems_raise():
    with pytest.raises(ValueError):
        MultiSystemNodeTableFeaturizer({})


@pytest.mark.graph
def test_graphs_and_batches_come_from_the_featurizer_of_the_system():
    pytest.importorskip("torch_geometric")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import torch
    featurizers = _featurizers()
    rows_b = featurizers.rows_from_coordinates(FRAMES[:, :-1], 'ligB')

    graphs = featurizers.graphs(rows_b, system_id='ligB')
    expected = featurizers['ligB'].graphs(rows_b)
    for graph, reference in zip(graphs, expected):
        for key in ('positions', 'edge_index', 'node_attrs', 'shifts'):
            assert torch.equal(graph[key], reference[key])
    batch = featurizers.batch_dict(rows_b, system_id='ligB')
    reference = featurizers['ligB'].batch_dict(rows_b)
    assert all(torch.equal(batch[key], reference[key]) for key in reference)
    assert np.array_equal(featurizers.row_from_graph(graphs[0], 'ligB'),
                          rows_b[0])
    # rows of one system are refused by the other one
    with pytest.raises(ValueError, match='shape'):
        featurizers.graphs(rows_b, system_id='ligA')
