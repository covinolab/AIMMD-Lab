"""The graph-network params template, examples/graph_network/params.py.

The template chooses the graph input with ``GRAPH_INPUT``: node tables
(module-level wrappers around a `NodeTableFeaturizer`, a pinned series name)
or the graph cache of the coordinate-descriptor mode. These tests fill in
the template for the toy system (topology, selections, atom types, states,
the toy engine, a small stand-in for painn.py) and check that

- an unpinned or wrongly pinned series stops the load with the name to pin;
- in node-table mode Params.load works, writes no graph cache, and its
  params1.py loads again; the command-line prefill takes its featurizer;
- both graph inputs give the same values, bit for bit, for the same frames
  and network weights.

Loading evaluates the values function, so most tests need torch_geometric
(``--rungraph``).
"""
import re
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.network.nodetables import _cli, _tool
from tests._nodetables_run import (bits, expected_rows, series_file,
                                   toy_featurizer, trajectories, write_run)
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames,
                                   write_toy_gro, write_toy_xtc)

TEMPLATE = Path(__file__).resolve().parents[1] / 'examples' / \
    'graph_network' / 'params.py'

# the ligand moves along y: A (y < 7) -> R -> B (y > 13)
FRAMES = toy_frames(12, seed=4, ligand_y=np.linspace(5.0, 15.0, 12))

TOY_STATES = '''# --- states (toy) ---
def states_function(trajectory):
    y = np.array([ts.positions[:6, 1].mean() for ts in trajectory])
    labels = np.full(len(y), 'R', dtype='<U1')
    labels[y < 7.0] = 'A'
    labels[y > 13.0] = 'B'
    return labels


def toy_mdrun(ts):
    ts.positions[:] = ts.positions + 0.1


'''

# a stand-in for painn.py: per-graph sums over nodes and edges, so that
# positions, atom types and edges all reach the output
PAINN = '''
import torch


class PaiNNModel(torch.nn.Module):
    def __init__(self, n_out=1, **settings):
        super().__init__()
        torch.manual_seed(0)
        self.nodes = torch.nn.Linear({n_types} + 3, n_out)
        self.edges = torch.nn.Linear(1, n_out)

    def forward(self, batch):
        per_node = self.nodes(torch.cat([batch['node_attrs'],
                                         batch['positions']], dim=1))
        n_graphs = len(batch['ptr']) - 1
        nodes = torch.zeros(n_graphs, per_node.shape[1]).index_add_(
            0, batch['batch'], per_node)
        source = batch['batch'][batch['edge_index'][0]]
        edges = torch.zeros(n_graphs).index_add_(
            0, source, torch.ones(len(source)))
        return nodes + self.edges(edges[:, None])
'''


def _set(source, name, value):
    """Replace the assignment ``name = ...`` (one line) in the template."""
    pattern = re.compile(rf'^{name} = .*$', re.MULTILINE)
    assert len(pattern.findall(source)) == 1, name
    return pattern.sub(f'{name} = {value!r}', source)


def _fill_template(folder, graph_input='nodetables', series=None):
    """The template filled in for the toy system, in `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_toy_gro(folder / 'run.gro', FRAMES[0])
    write_toy_xtc(folder / 'initial.xtc', FRAMES)
    (folder / 'painn.py').write_text(PAINN.format(n_types=len(ATOM_TYPES)))
    source = TEMPLATE.read_text()
    source = _set(source, 'engine', 'toy')
    source = _set(source, 'CUTOFF', CUTOFF)
    source = _set(source, 'ATOM_TYPES', ATOM_TYPES)
    source = _set(source, 'SYSTEM_SELECTION', SYSTEM_SELECTION)
    source = _set(source, 'ENVIRONMENT_SELECTION', ENVIRONMENT_SELECTION)
    source = _set(source, 'GRAPH_INPUT', graph_input)
    start = source.index('# --- states')
    end = source.index('# --- graph input')
    source = source[:start] + TOY_STATES + source[end:]
    if series is None:
        series = toy_featurizer(folder, structure='run.gro').series
    assert "check_series('descriptors-gn0000000000')" in source
    source = source.replace("'descriptors-gn0000000000'", repr(series))
    (folder / 'params.py').write_text(source)
    return folder / 'params.py'


def test_the_template_names_the_series_to_pin(tmp_path):
    params = _fill_template(tmp_path, series='descriptors-gn0000000000')
    with pytest.raises(_tool.UsageError) as info:
        _tool.load_featurizer(params)
    series = toy_featurizer(tmp_path, structure='run.gro').series
    assert f'pin descriptors_series = {series!r}' in str(info.value)


@pytest.mark.graph
def test_the_template_loads_in_node_table_mode(tmp_path, monkeypatch):
    pytest.importorskip('torch_geometric')
    monkeypatch.chdir(tmp_path)
    params_file = _fill_template(tmp_path)
    featurizer = toy_featurizer(tmp_path, structure='run.gro')

    params = aimmd.Params.load(params_file)

    assert params.descriptors_series == featurizer.series
    path, = params.initial_paths
    rows = getattr(path, featurizer.series)
    assert np.array_equal(bits(rows), bits(path.compute(
        featurizer.descriptors_function)))
    assert params.values_function(rows).shape == (len(path),)
    assert not list(Path(tmp_path).glob('graphs_cache.sqlite*'))
    reloaded = aimmd.Params.load(params.path, save=False)
    assert reloaded.descriptors_series == featurizer.series
    assert not list(Path(tmp_path).glob('graphs_cache.sqlite*'))


@pytest.mark.graph
def test_both_graph_inputs_give_the_same_values(tmp_path, monkeypatch):
    pytest.importorskip('torch_geometric')
    nodetables = _fill_template(tmp_path / 'nodetables')
    sqlite = _fill_template(tmp_path / 'sqlite', graph_input='sqlite')
    monkeypatch.chdir(tmp_path)

    with_rows = aimmd.Params.load(nodetables, save=False)
    with_graphs = aimmd.Params.load(sqlite, save=False)
    assert with_graphs.descriptors_series == 'descriptors'
    assert (tmp_path / 'sqlite' / 'graphs_cache.sqlite').exists()
    with_graphs.network.load_state_dict(with_rows.network.state_dict())

    rows_path, = with_rows.initial_paths
    graphs_path, = with_graphs.initial_paths
    rows = getattr(rows_path, with_rows.descriptors_series)
    coordinates = graphs_path.descriptors
    assert len(rows) == len(coordinates) > 1
    for batchsize in (32, 3):
        from_rows = with_rows.values_function(rows, batchsize=batchsize)
        from_graphs = with_graphs.values_function(coordinates,
                                                  batchsize=batchsize)
        assert np.array_equal(from_rows.view(np.uint32),
                              from_graphs.view(np.uint32))


@pytest.mark.graph
def test_the_command_line_takes_the_featurizer_of_the_template(tmp_path):
    pytest.importorskip('torch_geometric')
    params = _fill_template(tmp_path)
    run = tmp_path / 'run1'
    write_run(run)
    featurizer = toy_featurizer(tmp_path, structure='run.gro')

    assert _cli.main(['prefill', '--params', str(params), '--run',
                      str(run), '--verify', '2']) == 0

    for trajectory in trajectories(run):
        stored = np.load(series_file(trajectory, featurizer.series))
        assert np.array_equal(bits(stored),
                              bits(expected_rows(featurizer, trajectory)))
    assert not list(Path(tmp_path).glob('graphs_cache.sqlite*'))
