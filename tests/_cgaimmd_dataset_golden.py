"""The port-equivalence golden of the dataset extraction.

The extraction of the flow branch, ``aimmd.network.flow_fit.extract_flow_dataset``
at flowdev ``bb98943``, was run once on the fixture below, and what it returned
is stored in ``tests/data/cgaimmd_dataset_flowdev_bb98943.npz``.
``tests/test_cgaimmd_dataset.py`` checks that
:func:`aimmd.cgaimmd.dataset.extract_dataset` with the default series returns
the same, array by array and bit for bit, for every configuration of
:data:`CONFIGS`.

The fixture is a run folder of :func:`tests._cgaimmd_helpers.build_cgaimmd_run_dir`
(two atoms, so that the all-zero rows of a descriptor hole count) with
everything the extraction treats specially: an internal-shooting chain
(``chainA0``), free trajectories in three part files each, holes in a shot
and in a free part, a free part whose descriptor cache is two frames short (a
part being written) and one without a descriptor cache.

Recording it again is only needed when the fixture or the configurations
change. From the root of this checkout, with ``<export>`` an empty folder::

    git archive bb98943 aimmd | tar -x -C <export>
    python tests/_cgaimmd_dataset_golden.py fixture <folder>
    PYTHONPATH=<export> python tests/_cgaimmd_dataset_golden.py record <folder> \\
        tests/data/cgaimmd_dataset_flowdev_bb98943.npz

The first step writes the fixture with this checkout; the second runs the old
code on it (``aimmd`` is imported from ``<export>``). This module therefore
imports only the standard library and numpy at module level.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


GOLDEN = Path(__file__).resolve().parent / 'data' / 'cgaimmd_dataset_flowdev_bb98943.npz'
"""The recorded output of the old extraction."""

N_ATOMS = 2
STATES = 'ARB'

CONFIGS = {
    'uniform_free_stride_2': dict(weights='ones', free_stride=2),
    'weighted_strided': dict(weights='ramp', free_stride=5, shot_stride=2),
    'shot_stride_min_frames': dict(weights='ones', shot_stride=2, shot_stride_min_frames=7),
    'exclusions': dict(weights='ones', free_stride=1, exclude=True),
    'first_path_twice': dict(weights='ones', free_stride=1, first_path_twice=True),
    'zero_weights': dict(weights='zeros'),
    'reuse_same_stride': dict(weights='ones', free_stride=2, previous='uniform_free_stride_2'),
    'reuse_denser_stride': dict(weights='ones', free_stride=1, previous='uniform_free_stride_2'),
}
"""Name -> settings of one extraction, run in this order (a ``previous`` names an earlier one)."""

FIELDS = ('sp_rows', 'sp_r_first', 'sp_r_last', 'sp_group', 'sp_fname', 'fr_rows', 'fr_state',
          'fr_w', 'fr_group', 'fr_fname', 'fr_loc')
"""The dataset fields of the old code (file names are stored relative to the run)."""

NOTES = ('n_paths', 'n_files', 'files_missing', 'frames_beyond_cache', 'reused_files',
         'free_stride', 'shot_stride', 'shot_stride_min_frames', 'excluded_files',
         'excluded_path_frames', 'excluded_density_weight', 'excluded_shooting_points',
         'zero_rows', 'zero_row_shooting_points')
"""The notes of the old code, except the timing ``extract_seconds``."""


def write_fixture(directory):
    """Write the run folder of the golden into `directory`; return ``(run, manifest)``.

    The manifest lists the shot files (with the shooting index of all of
    them) and the part files of every free trajectory, relative to the run.
    It is also written to ``run/manifest.json``.
    """
    from tests._cgaimmd_helpers import build_cgaimmd_run_dir

    rd = build_cgaimmd_run_dir(directory, n_chains=2, n_shots=3, n_free=2, n_atoms=N_ATOMS,
                               n_parts=3)
    run = Path(rd.run)
    shutil.copytree(run / 'chainR0', run / 'chainA0')      # an internal-shooting chain
    holes = {'chainR1/path000001.xtc': [0, rd.shooting_index], 'freeB/traj000001.part0001.xtc': [6]}
    for name, frames in holes.items():                      # an interrupted descriptor pass
        cache = run / f'{name}.descriptors.npy'
        rows = np.load(cache)
        rows[frames] = 0.0
        np.save(cache, rows)
    cache = run / 'freeB/traj000002.part0003.xtc.descriptors.npy'
    np.save(cache, np.load(cache)[:-2])                     # a part being written
    os.remove(run / 'freeA/traj000002.part0003.xtc.descriptors.npy')
    shots = sorted(os.path.relpath(f, run) for f in run.glob('chain*/path*.xtc'))
    free = [[os.path.relpath(str(f), run) for f in path.fnames] for path in rd.free]
    manifest = {'shots': shots, 'shooting_index': rd.shooting_index, 'free': free}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return str(run), manifest


def fixture_ensemble(run, manifest):
    """The path ensemble the trainer would pass: shot paths whole, free trajectories split."""
    import aimmd
    from aimmd.pathensemble import PathEnsemble
    from aimmd.pathensemble.utils import assemble_pathensemble

    shots = PathEnsemble(*[aimmd.Path(os.path.join(run, f),
                                      shooting_index=manifest['shooting_index'])
                           for f in manifest['shots']])
    free = [aimmd.Path([os.path.join(run, f) for f in parts], remove_overlapping_frames=True)
            for parts in manifest['free']]
    return assemble_pathensemble(shots, free)


def exclusions(run):
    """The ``exclude_frames`` of the ``'exclusions'`` configuration (absolute file names)."""
    return {os.path.normpath(os.path.join(run, name)): frames for name, frames in
            {'chainR0/path000001.xtc': 'all', 'chainR1/path000002.xtc': {0, 3},
             'freeA/traj000001.part0002.xtc': {2, 4, 5}}.items()}


def run_configs(extract, pathensemble, run):
    """``{name: dataset}`` of every configuration, extracted with `extract`."""
    out = {}
    for name, config in CONFIGS.items():
        ensemble = (pathensemble + pathensemble[:1] if config.get('first_path_twice')
                    else pathensemble)
        n = len(ensemble)
        weights = {'ones': np.ones(n), 'zeros': np.zeros(n),
                   'ramp': np.linspace(0.0, 2.0, n)}[config['weights']]
        kwargs = {key: config[key] for key in ('free_stride', 'shot_stride',
                                                'shot_stride_min_frames') if key in config}
        out[name] = extract(ensemble, states=STATES, n_atoms=N_ATOMS, weights=weights,
                            exclude_frames=exclusions(run) if config.get('exclude') else None,
                            previous=out[config['previous']] if 'previous' in config else None,
                            **kwargs)
    return out


def flatten(data, run):
    """``{field: array}`` of a dataset as stored in the golden (``notes`` as JSON)."""
    out = {}
    for name in FIELDS:
        value = getattr(data, name)
        if name.endswith('fname'):
            value = np.array([os.path.relpath(str(f), run) for f in value], dtype=str)
        out[name] = np.asarray(value)
    out['notes'] = np.array(json.dumps({key: data.notes[key] for key in NOTES}, sort_keys=True))
    return out


def _record(folder, output):
    import aimmd
    if Path(aimmd.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]:
        raise SystemExit(f'aimmd is imported from this checkout ({aimmd.__file__}); put the '
                         f'exported old tree first on PYTHONPATH')
    from aimmd.network.flow_fit import extract_flow_dataset

    run = os.path.join(os.path.abspath(folder), 'run')
    with open(os.path.join(run, 'manifest.json')) as handle:
        manifest = json.load(handle)
    datasets = run_configs(extract_flow_dataset, fixture_ensemble(run, manifest), run)
    arrays = {f'{name}/{key}': value for name, data in datasets.items()
              for key, value in flatten(data, run).items()}
    arrays['source'] = np.array('aimmd.network.flow_fit.extract_flow_dataset, flowdev bb98943')
    np.savez_compressed(output, **arrays)
    print(f'{len(datasets)} configurations from {aimmd.__file__} -> {output}')


if __name__ == '__main__':
    mode, folder = sys.argv[1], sys.argv[2]
    if mode == 'fixture':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        print(write_fixture(folder)[0])
    elif mode == 'record':
        _record(folder, sys.argv[3])
    else:
        raise SystemExit(f'unknown mode {mode!r}: fixture or record')
