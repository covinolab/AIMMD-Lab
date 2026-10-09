"""A toy campaign for the tests of ``python -m aimmd.network.nodetables``.

`make_campaign` writes a params file in node-table mode and a run folder with
the trajectory layout of an AIMMD run: the exported initial path
(``initialARB``), registered chain paths and the in-flight halves of a shot
(``chainR0``), and the parts of a free simulation (``freeA``). Every
trajectory of the run has a states series, and a stale coordinate series
``*.descriptors.npy`` (garbage: the tools must never open it). One more
trajectory has no states series and is not part of the run. Only numpy and
MDAnalysis are needed.
"""
import json
import os
from collections import namedtuple
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.core import reader

from aimmd.network.nodetables import DEFAULT_N_MAX, NodeTableFeaturizer
from tests._nodetables_toy import (ATOM_TYPES, CUTOFF, ENVIRONMENT_SELECTION,
                                   SYSTEM_SELECTION, toy_frames, write_toy_gro,
                                   write_toy_xtc)

# trajectory of the run -> (frames, seed of toy_frames)
LAYOUT = {
    'initialARB/initial.xtc': (6, 10),
    'chainR0/path000001.xtc': (9, 11),
    'chainR0/path000002.xtc': (7, 12),
    'chainR0/back.xtc': (5, 13),
    'chainR0/forw.xtc': (4, 14),
    'freeA/traj000001.part0000.xtc': (8, 15),
    'freeA/traj000001.part0001.xtc': (3, 16),
}
# a trajectory without states series: not part of the run
UNTRACKED = 'chainR0/temp.xtc'

PARAMS_SOURCE = '''
import MDAnalysis as mda
from aimmd.network.nodetables import NodeTableFeaturizer

GRAPH_INPUT = 'nodetables'

FEATURIZER = NodeTableFeaturizer(
    mda.Universe('toy.gro', to_guess=['types', 'bonds']),
    {system!r}, {environment!r}, {atom_types!r}, cutoff={cutoff!r},
    n_max={n_max!r})
descriptors_series = FEATURIZER.check_series({series!r})


def descriptors_function(trajectory):
    return FEATURIZER.descriptors_function(trajectory)
'''

Campaign = namedtuple('Campaign', 'folder params run featurizer')


def toy_featurizer(folder, n_max=DEFAULT_N_MAX, structure='toy.gro',
                   environment=ENVIRONMENT_SELECTION, refill=False):
    """The featurizer of the campaign's params file."""
    universe = mda.Universe(str(Path(folder) / structure),
                            to_guess=['types', 'bonds'])
    return NodeTableFeaturizer(universe, SYSTEM_SELECTION, environment,
                               ATOM_TYPES, cutoff=CUTOFF, n_max=n_max,
                               refill=refill)


def write_params(folder, n_max=DEFAULT_N_MAX, series=None,
                 name='params.py'):
    """Write the node-table params file (pinned series) into `folder`."""
    if series is None:
        series = toy_featurizer(folder, n_max).series
    source = PARAMS_SOURCE.format(
        system=SYSTEM_SELECTION, environment=ENVIRONMENT_SELECTION,
        atom_types=ATOM_TYPES, cutoff=CUTOFF, n_max=n_max, series=series)
    params = Path(folder) / name
    params.write_text(source)
    return str(params)


def write_run(run, layout=LAYOUT, untracked=UNTRACKED):
    """Write the toy run folder `run`; returns its trajectories."""
    trajectories = []
    for name, (n_frames, seed) in layout.items():
        trajectory = Path(run) / name
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        write_toy_xtc(trajectory, toy_frames(n_frames, seed=seed))
        np.save(f'{trajectory}.states.npy', np.full(n_frames, 'R'))
        np.save(f'{trajectory}.values.npy', np.zeros(n_frames))
        Path(f'{trajectory}.descriptors.npy').write_bytes(b'stale' * 100)
        trajectories.append(str(trajectory))
    if untracked:
        write_toy_xtc(Path(run) / untracked, toy_frames(2, seed=99))
    return trajectories


def make_campaign(folder, n_max=DEFAULT_N_MAX):
    """A params folder with toy.gro, params.py and the run folder run1/."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_toy_gro(folder / 'toy.gro', toy_frames(1, seed=10)[0])
    params = write_params(folder, n_max)
    run = folder / 'run1'
    write_run(run)
    return Campaign(str(folder), params, str(run),
                    toy_featurizer(folder, n_max))


def trajectories(run, layout=LAYOUT):
    """The trajectories of the toy run, absolute paths."""
    return [os.path.abspath(Path(run) / name) for name in layout]


def expected_rows(featurizer, trajectory):
    """Node-table rows of every frame of `trajectory`, featurized directly."""
    frames = reader(str(trajectory))
    try:
        return featurizer.descriptors_function(frames)
    finally:
        frames.close()


def series_file(trajectory, series):
    return f'{trajectory}.{series}.npy'


def bits(rows):
    """The rows as raw bits, for bitwise comparisons."""
    return np.ascontiguousarray(rows, dtype=np.float32).view(np.uint32)


def npy_bytes(rows):
    """The bytes ``np.save`` writes for `rows`."""
    from io import BytesIO
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(rows, dtype=np.float32))
    return buffer.getvalue()


def load_report(fname):
    """The JSON report of a command, with its files keyed by trajectory."""
    with open(fname) as file:
        report = json.load(file)
    report['by_trajectory'] = {item['trajectory']: item
                               for item in report['files']}
    return report


def hidden_temporaries(run):
    """Temporary files the tools left in the run folder."""
    return sorted(str(path) for path in Path(run).rglob('*')
                  if path.name.endswith('.tmp'))
