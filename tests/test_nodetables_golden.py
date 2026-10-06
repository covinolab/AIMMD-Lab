"""Opt-in golden test: node tables of real frames rebuild the cached graphs.

Point ``AIMMD_NODETABLES_GOLDEN_DATA`` at a kcmpd09 run folder holding
``run.gro``, ``graphs_cache.sqlite`` and ``run_local/`` (the HSP90 compound-9
campaign: ligand INH, the selections and atom types below) and run with
``--rungraph``::

    AIMMD_NODETABLES_GOLDEN_DATA=/path/to/NHSP90/kcmpd09 \\
        pytest tests/test_nodetables_golden.py --rungraph

Skipped without the variable. The test copies the structure, the graph cache
and a few trajectories into its temporary directory (about 320 MB) and never
writes to the data folder. For frames of every trajectory it computes the
graph-cache key of the frame's coordinates (as `process_descriptors_pyg` does
at ingestion), reads the graph cached for it, and checks that the node-table
row of the frame

- rebuilds that graph bitwise (`NodeTableFeaturizer.graphs`),
- gives the batch ``Batch.from_data_list`` makes of the cached graphs
  (`NodeTableFeaturizer.batch_dict`),
- equals `NodeTableFeaturizer.row_from_graph` of the cached graph,

and that the series name is the one pinned for the production campaigns. Run
it on every host that will featurize (e.g. the arm64 nodes of JUPITER) before
switching a campaign to node tables.
"""
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

DATA = os.environ.get('AIMMD_NODETABLES_GOLDEN_DATA')

pytestmark = [
    pytest.mark.graph,
    pytest.mark.skipif(not DATA, reason='set AIMMD_NODETABLES_GOLDEN_DATA to '
                                        'a kcmpd09 run folder to run it')]

SERIES = 'descriptors-gne38bf950a1'
CUTOFF = 4.0
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F', 'NA', 'P', 'S', 'CL', 'BR', 'I']
SYSTEM_SELECTION = '(resname INH) and not type H'
ENVIRONMENT_SELECTION = 'not type H and around 8.0 (resname INH)'
TRAJECTORIES = ['run_local/initialARB/initial_selected.xtc',
                'run_local/chainR1/path000001.xtc',
                'run_local/freeB/traj000001.part0000.xtc']
FRAMES_PER_TRAJECTORY = 16
KEYS = ('positions', 'edge_index', 'node_attrs', 'shifts')


@pytest.fixture(scope='module')
def golden(tmp_path_factory):
    """Scratch copies of the structure, the graph cache and trajectories."""
    pytest.importorskip('torch_geometric')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    source = Path(DATA)
    scratch = tmp_path_factory.mktemp('nodetables_golden')
    for name in ('run.gro', 'graphs_cache.sqlite'):
        if not (source / name).is_file():
            pytest.fail(f'{source / name} not found: '
                        f'AIMMD_NODETABLES_GOLDEN_DATA must be a kcmpd09 run '
                        f'folder')
        shutil.copyfile(source / name, scratch / name)
    trajectories = []
    for name in TRAJECTORIES:
        if (source / name).is_file():
            target = scratch / name.replace('/', '_')
            shutil.copyfile(source / name, target)
            trajectories.append(target)
    if not trajectories:
        pytest.fail(f'none of {TRAJECTORIES} found in {source}')
    return scratch, trajectories


def test_node_tables_rebuild_the_cached_graphs(golden):
    import MDAnalysis as mda
    import torch
    from torch_geometric.data import Batch
    from aimmd.network import graph_utils
    from aimmd.network.nodetables import NodeTableFeaturizer

    scratch, trajectories = golden
    structure = str(scratch / 'run.gro')
    featurizer = NodeTableFeaturizer(
        mda.Universe(structure, to_guess=['bonds', 'masses', 'types']),
        SYSTEM_SELECTION, ENVIRONMENT_SELECTION, ATOM_TYPES, cutoff=CUTOFF)
    assert featurizer.series == SERIES
    database = sqlite3.connect(
        f"file:{scratch / 'graphs_cache.sqlite'}?mode=ro", uri=True)

    checked = 0
    for trajectory in trajectories:
        universe = mda.Universe(structure, str(trajectory))
        n_frames = len(universe.trajectory)
        frames = np.unique(np.linspace(
            0, n_frames - 1, min(FRAMES_PER_TRAJECTORY, n_frames)).astype(int))
        # the coordinate descriptors of these frames, as at ingestion
        coordinates = np.array([universe.trajectory[i].positions.ravel().copy()
                                for i in frames])
        cached = []
        for i, row in zip(frames, coordinates):
            found = database.execute(
                'SELECT data FROM graphs_cache WHERE key = ?',
                (graph_utils.get_stable_hash(row),)).fetchone()
            assert found, f'{trajectory.name} frame {i}: no cached graph'
            cached.append(graph_utils._decode(found[0]))

        rows = featurizer.descriptors_function(universe.trajectory[frames])
        assert np.array_equal(rows,
                              featurizer.rows_from_coordinates(coordinates))
        assert rows[:, 0].all() and rows[:, 0].max() < 0.8 * featurizer.n_max

        for i, graph, expected in zip(frames, featurizer.graphs(rows),
                                      cached):
            for key in KEYS:
                assert graph[key].dtype == expected[key].dtype, (i, key)
                assert torch.equal(graph[key], expected[key]), (i, key)
        batch = featurizer.batch_dict(rows)
        reference = Batch.from_data_list(cached).to_dict()
        assert set(batch) == set(reference)
        for key in reference:
            assert torch.equal(batch[key], reference[key]), key
        for row, graph in zip(rows, cached):
            assert np.array_equal(featurizer.row_from_graph(graph), row)
        checked += len(frames)

    database.close()
    assert checked >= FRAMES_PER_TRAJECTORY
