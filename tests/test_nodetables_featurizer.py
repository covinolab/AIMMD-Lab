"""`NodeTableFeaturizer`: frames to node-table rows, and the series name.

A node-table row holds the graph nodes of a frame (positions after unwrap ->
center_in_box -> wrap, and atom-type indices) in a fixed-width float32 row.
These tests cover the worker side, which needs only numpy and MDAnalysis: the
row layout, the periodic-boundary handling of a ligand split across the box,
the overflow policy (never raise, zero row, logged error, warning at 80 % of
n_max) and the fingerprinted series name. The graph side is in
test_nodetables_graphs.py (``--rungraph``).
"""
import json
import re

import numpy as np
import pytest

from aimmd.network.nodetables import (NodeTableFeaturizer, NODE_TABLE_LAYOUT,
                                      DEFAULT_N_MAX, SERIES_PREFIX)
from aimmd.params.utils import check_descriptors_series
from tests._nodetables_toy import (ATOM_TYPES, BOX, CUTOFF,
                                   ENVIRONMENT_SELECTION, IONS, LIGAND,
                                   N_ATOMS, SYSTEM_SELECTION, memory_trajectory,
                                   reference_nodes, toy_frames, toy_universe)

FRAMES = toy_frames(12)

# the series of the toy featurizer with the default settings; a change means
# that every pinned production series name changes too
TOY_SERIES = 'descriptors-gn8732b6b2f8'


def _featurizer(universe=None, **kwargs):
    settings = dict(system_selection=SYSTEM_SELECTION,
                    environment_selection=ENVIRONMENT_SELECTION,
                    atom_types=ATOM_TYPES, cutoff=CUTOFF)
    settings.update(kwargs)
    return NodeTableFeaturizer(universe or toy_universe(), **settings)


def _node_counts(frames=FRAMES):
    universe = toy_universe()
    return np.array([len(reference_nodes(universe, frame)[1])
                     for frame in frames])


def test_rows_hold_the_nodes_after_the_pbc_handling():
    featurizer = _featurizer()
    rows = featurizer.rows_from_coordinates(FRAMES)

    assert rows.dtype == np.float32
    assert rows.shape == (len(FRAMES), 2 + 4 * DEFAULT_N_MAX)
    assert featurizer.width == rows.shape[1]
    universe = toy_universe()
    for row, frame in zip(rows, FRAMES):
        positions, types = reference_nodes(universe, frame)
        n = len(types)
        assert row[0] == n and row[1] == NODE_TABLE_LAYOUT
        p0, t0 = 2, 2 + 3 * DEFAULT_N_MAX
        assert np.array_equal(row[p0:p0 + 3 * n], positions.ravel())
        assert row[t0:t0 + n].tolist() == [ATOM_TYPES.index(t) for t in types]
        # zero padding after n_nodes in both blocks
        assert not row[p0 + 3 * n:t0].any() and not row[t0 + n:].any()


def test_a_ligand_split_across_the_box_is_made_whole_and_centered():
    featurizer = _featurizer()
    assert np.ptp(FRAMES[:, LIGAND, 0], axis=1).max() > 15   # split
    rows = featurizer.rows_from_coordinates(FRAMES)

    ligand = rows[:, 2:2 + 3 * 6].reshape(len(rows), 6, 3)   # system first
    bonds = np.linalg.norm(np.diff(ligand, axis=1), axis=2)
    assert np.allclose(bonds, np.hypot(1.25, 0.7), atol=1e-4)
    assert np.allclose(ligand.mean(axis=1), BOX[:3] / 2, atol=1e-4)
    # every node inside the box
    for row in rows:
        n = int(row[0])
        positions = row[2:2 + 3 * n].reshape(n, 3)
        assert (positions >= 0).all() and (positions < BOX[:3]).all()


def test_descriptors_function_reads_a_trajectory():
    featurizer = _featurizer()
    rows = featurizer.descriptors_function(memory_trajectory(FRAMES))
    assert np.array_equal(rows, _featurizer().rows_from_coordinates(FRAMES))
    # flat coordinate rows, as in a 'descriptors' series, are accepted too
    flat = FRAMES.reshape(len(FRAMES), -1)
    assert np.array_equal(featurizer.rows_from_coordinates(flat), rows)


def test_an_empty_trajectory_gives_no_rows():
    featurizer = _featurizer()
    rows = featurizer.descriptors_function(memory_trajectory(FRAMES)[0:0])
    assert rows.shape == (0, featurizer.width)
    assert rows.dtype == np.float32


def test_a_frame_with_another_atom_count_raises():
    with pytest.raises(ValueError, match='atoms'):
        _featurizer().rows_from_coordinates(FRAMES[:, :-1])


def test_overflowing_frames_get_zero_rows_and_an_error(capsys):
    counts = _node_counts()
    n_max = int(np.median(counts))
    overflow = counts > n_max
    assert overflow.any() and not overflow.all()

    rows = _featurizer(n_max=n_max).rows_from_coordinates(FRAMES)  # no raise

    assert not rows[overflow].any()
    assert (rows[~overflow, 0] == counts[~overflow]).all()
    out = capsys.readouterr().out
    assert out.count('ERROR') == 1                  # one line per call
    assert f'{overflow.sum()} frame(s)' in out
    assert f'largest {counts.max()}' in out
    assert 'python -m aimmd.network.nodetables repack --n-max N' in out


def test_frames_near_n_max_give_a_warning(capsys):
    counts = _node_counts()
    n_max = counts.max()                            # nothing overflows
    crowded = counts > int(0.8 * n_max)
    assert crowded.any() and not crowded.all()

    rows = _featurizer(n_max=n_max).rows_from_coordinates(FRAMES)

    assert rows[:, 0].all()
    out = capsys.readouterr().out
    assert 'ERROR' not in out
    assert out.count('WARNING') == 1
    assert f'{crowded.sum()} frame(s)' in out and '80%' in out


def test_no_message_far_below_n_max(capsys):
    _featurizer().rows_from_coordinates(FRAMES)
    assert capsys.readouterr().out == ''


def test_an_environment_atom_of_unknown_type_gives_a_zero_row(capsys):
    universe = toy_universe()
    no_ion = [i for i, frame in enumerate(FRAMES)
              if 'NA' not in reference_nodes(universe, frame)[1]]
    frames = FRAMES[no_ion[:3]].copy()
    frames[1, IONS.start] = frames[1, LIGAND.start + 2] + [0.0, 0.0, 3.0]
    atom_types = [t for t in ATOM_TYPES if t != 'NA']
    rows = _featurizer(atom_types=atom_types).rows_from_coordinates(frames)

    assert not rows[1].any() and rows[[0, 2], 0].all()
    out = capsys.readouterr().out
    assert 'ERROR' in out and "['NA']" in out


def test_atom_types_default_to_those_of_the_universe():
    featurizer = _featurizer(atom_types=None)
    assert featurizer.atom_types == sorted(set(toy_universe().atoms.types))


@pytest.mark.parametrize('kwargs, match', [
    (dict(system_selection='resname XYZ'), 'selects no atoms'),
    (dict(n_max=4), 'n_max'),
    (dict(atom_types=['C', 'O', 'S', 'NA', 'H']), r"\['N'\]"),
    (dict(cutoff=0.0), 'cutoff'),
    (dict(n_max=0), 'n_max'),
    (dict(n_max=10.5), 'n_max'),
    (dict(max_num_neighbors=0), 'max_num_neighbors')])
def test_invalid_settings_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _featurizer(**kwargs)


def test_a_universe_without_a_box_or_bonds_raises():
    universe = toy_universe()
    universe.dimensions = None
    with pytest.raises(ValueError, match='box'):
        _featurizer(universe)

    with pytest.raises(ValueError, match='bonds'):
        _featurizer(toy_universe(bonds=False))


# ----------------------------------------------------------------------------
# series name

def test_the_series_name_is_a_valid_pinned_fingerprint():
    featurizer = _featurizer()
    series = featurizer.series
    assert re.fullmatch(SERIES_PREFIX + '[0-9a-f]{10}', series)
    assert check_descriptors_series(series) == series
    assert series == SERIES_PREFIX + featurizer.fingerprint[:10]
    assert series == TOY_SERIES


def test_the_series_name_is_stable():
    featurizer = _featurizer()
    series = featurizer.series
    featurizer.rows_from_coordinates(FRAMES)        # moves the atoms
    assert featurizer.series == series
    assert _featurizer().series == series
    assert _featurizer(toy_universe(FRAMES[5])).series == series


def _renamed(attribute, index, value):
    universe = toy_universe()
    setattr(universe.atoms[index], attribute, value)
    return universe


def _relabeled_residue():
    universe = toy_universe()
    universe.residues[-1].resname = 'K'
    return universe


def _other_box():
    return toy_universe(box=np.array([20.0, 20.0, 21.0, 90.0, 90.0, 90.0],
                                     dtype=np.float32))


@pytest.mark.parametrize('change', [
    dict(system_selection='resname LIG and not type H and not name O5'),
    dict(environment_selection='not type H and around 6.0 (resname LIG)'),
    dict(n_max=1024),
    dict(atom_types=ATOM_TYPES + ['CL']),
    dict(atom_types=ATOM_TYPES[::-1]),
    dict(universe=_renamed('name', 10, 'CB')),
    dict(universe=_renamed('type', 10, 'O')),
    dict(universe=_relabeled_residue()),
    dict(universe=_other_box())],
    ids=['system', 'environment', 'n_max', 'atom_types', 'atom_type_order',
         'atom_name', 'atom_type', 'resname', 'box'])
def test_the_series_name_changes_with_what_the_rows_hold(change):
    assert _featurizer(**change).series != TOY_SERIES


def test_the_series_name_changes_with_the_atom_count():
    import MDAnalysis as mda
    universe = toy_universe()
    smaller = mda.Merge(universe.atoms[:-1])
    smaller.dimensions = BOX
    assert _featurizer(smaller).n_atoms == N_ATOMS - 1
    assert _featurizer(smaller).series != TOY_SERIES


@pytest.mark.parametrize('change', [dict(cutoff=5.0),
                                    dict(max_num_neighbors=32)])
def test_the_series_name_ignores_the_edge_settings(change):
    assert _featurizer(**change).series == TOY_SERIES


def test_spec_holds_what_the_series_name_hashes():
    featurizer = _featurizer()
    spec = featurizer.spec()
    assert set(spec) == {'layout', 'n_max', 'system', 'environment',
                         'atom_types', 'n_atoms', 'topology'}
    assert spec['layout'] == NODE_TABLE_LAYOUT
    assert spec['n_max'] == DEFAULT_N_MAX and spec['n_atoms'] == N_ATOMS
    assert spec['atom_types'] == ATOM_TYPES
    assert spec == json.loads(json.dumps(spec))
    spec['n_max'] = 1                               # a copy
    assert featurizer.spec()['n_max'] == DEFAULT_N_MAX


def test_check_series_accepts_the_pinned_name():
    featurizer = _featurizer()
    assert featurizer.check_series(TOY_SERIES) == TOY_SERIES


@pytest.mark.parametrize('pinned', ['descriptors-gn0000000000', 'descriptors',
                                    None])
def test_check_series_rejects_another_name_loudly(pinned):
    featurizer = _featurizer()
    with pytest.raises(ValueError) as info:
        featurizer.check_series(pinned)
    message = str(info.value)
    assert repr(pinned) in message and TOY_SERIES in message
    assert 'n_max' in message                       # names the spec
