"""The training data of C-GAIMMD fits, ``aimmd.cgaimmd.dataset``.

A toy run folder (``tests._cgaimmd_helpers.build_cgaimmd_run_dir``) and the
path ensemble the trainer would pass. The extraction with the default series
is checked against the recorded output of the flow branch's
``extract_flow_dataset`` (``tests._cgaimmd_dataset_golden``).
"""

import json
import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import aimmd
from aimmd.cgaimmd import dataset as ds
from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble import PathEnsemble
from aimmd.pathensemble.utils import assemble_pathensemble
from tests import _cgaimmd_dataset_golden as golden
from tests._cgaimmd_helpers import (_write_with_caches, build_cgaimmd_run_dir,
                                    cgaimmd_params_namespace)


def _ensemble(params, run):
    """What the trainer passes: the shot chains whole, the free trajectories split."""
    return assemble_pathensemble(params.shot_chains(run, None), params.free_trajectories(run))


def _setup(tmp_path, *, n_chains=3, n_shots=4, n_free=1, n_atoms=1, n_parts=1):
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=n_chains, n_shots=n_shots, n_free=n_free,
                               n_atoms=n_atoms, n_parts=n_parts)
    params = cgaimmd_params_namespace()
    return SimpleNamespace(rd=rd, run=rd.run, params=params, pe=_ensemble(params, rd.run))


def _copy_series(pathensemble, series, transform=lambda rows: rows):
    """Write ``{trajectory}.{series}.npy`` next to every descriptor cache of the ensemble."""
    for fname in {str(f) for path in pathensemble for f in path.fnames}:
        rows = np.load(get_cache_fname(fname, 'descriptors'))
        np.save(get_cache_fname(fname, series), transform(rows))


# --------------------------------------------------------------------------------------------
# extraction (ported from the flow branch)
# --------------------------------------------------------------------------------------------

def test_extract_dataset(tmp_path):
    """Frames with positive weight, de-duplicated by file, strided; SPs from chain folders."""
    s = _setup(tmp_path, n_chains=2, n_shots=3, n_free=1)
    weights = np.ones(len(s.pe))
    data = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, weights=weights, free_stride=2)
    assert len(data.sp_rows) == 6 and data.sp_rows.dtype == np.float32
    assert set(data.sp_group.tolist()) == {0, 1}
    assert (data.sp_r_first == 1).all() and (data.sp_r_last == 1).all()
    assert data.sp_x.shape == (6, 1, 3)
    # the shooting frame is R and x of its row is the value near 0
    assert np.abs(data.sp_x[:, 0, 0]).max() < 0.1
    shot_frames = data.fr_group >= 0
    assert shot_frames.sum() == 6 * 7                       # stride 1 inside shots
    free = ~shot_frames
    assert free.sum() == 2 * 5 and (data.fr_w[free] == 2.0).all()     # free stride 2, w x 2
    assert set(np.unique(data.fr_state).tolist()) == {0, 1, 2}
    assert (data.fr_loc[free] % 2 == 0).all()
    summary = data.summary()
    assert summary['n_sp'] == 6 and summary['frames_by_state'][0] > 0
    assert summary['series'] == 'descriptors'
    json.dumps(summary)
    # a frame shared by two paths gets the sum of their weights
    doubled = s.pe + s.pe[:1]
    twice = ds.extract_dataset(doubled, states='ARB', n_atoms=1, weights=np.ones(len(doubled)))
    first = twice.fr_fname == s.pe[0].fnames[0]
    assert (twice.fr_w[first] == 2.0).all()
    # zero weights drop frames; SPs are kept (their paths are shots)
    zero = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, weights=np.zeros(len(s.pe)))
    assert len(zero.fr_rows) == 0 and len(zero.sp_rows) == 6
    assert len(zero.fr_trajectory) == len(zero.fr_frame) == 0 and zero.trajectories == []
    # reuse: an unchanged file is not read again
    again = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, weights=weights, free_stride=2,
                               previous=data)
    assert again.notes['reused_files'] == again.notes['n_files']
    assert np.array_equal(again.fr_rows, data.fr_rows) and np.array_equal(again.fr_w, data.fr_w)
    with pytest.raises(ValueError, match='3 weights for'):
        ds.extract_dataset(s.pe, states='ARB', n_atoms=1, weights=np.ones(3))


def test_path_weights_both_directions(tmp_path):
    s = _setup(tmp_path)
    w = ds.path_weights(s.pe, 'ARB', {'free_threshold': 20})
    w1 = s.pe.reweight('ARB', free_threshold=20, source='values')[0]
    w2 = s.pe.reweight('BRA', free_threshold=20, source='values')[0]
    assert np.allclose(w, np.where(np.isfinite(w1 + w2), w1 + w2, 0.0))


def test_stale_descriptor_cache_raises(tmp_path):
    """A cache of another width names the series and the way out."""
    s = _setup(tmp_path)
    fname = s.rd.shots[0].fnames[0]
    np.save(get_cache_fname(fname, 'descriptors'), np.zeros((7, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="series 'descriptors' has shape \\(7, 2\\), expected "
                                         "width 3") as info:
        ds.extract_dataset(s.pe, states='ARB', n_atoms=1)
    assert 'migrate_descriptors' in str(info.value) and 'descriptors_series' in str(info.value)


def test_exclude_frames(tmp_path):
    """exclude_frames (resolved from a JSON next to the params) removes frames and SPs."""
    s = _setup(tmp_path)
    whole = s.rd.shots[0].fnames[0]
    partial = s.rd.shots[1].fnames[0]
    spec = {'exclude': {os.path.relpath(whole, s.run): 'all',
                        os.path.relpath(partial, s.run): [0, 3]}}
    params_dir = tmp_path / 'params_dir'
    params_dir.mkdir()
    (params_dir / 'exclude.json').write_text(json.dumps(spec))
    exclusions = ds.resolve_exclusions('exclude.json', s.run, params_dir)
    data = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, exclude_frames=exclusions)
    assert whole not in set(data.fr_fname.tolist()) and whole not in data.sp_fname
    locs = data.fr_loc[data.fr_fname == partial]
    assert 0 not in locs and 3 not in locs and 1 in locs
    assert partial not in data.sp_fname                   # its shooting frame (3) is excluded
    assert data.notes['excluded_shooting_points'] == 2
    assert data.notes['excluded_files'] == 2 and data.notes['excluded_path_frames'] == 7 + 2
    with pytest.raises(FileNotFoundError, match='exclude.json'):
        ds.resolve_exclusions('exclude.json', s.run, s.run)    # not relative to the run


def test_all_zero_descriptor_rows_are_skipped(tmp_path):
    """Holes of a descriptor cache (all-zero rows, M >= 2) are neither frames nor shooting points; counted, never reused."""
    s = _setup(tmp_path, n_atoms=3)
    one_atom = _setup(tmp_path / 'one_atom')                  # the origin is a position there
    kept = ds.extract_dataset(one_atom.pe, states='ARB', n_atoms=1, free_stride=1)
    assert (~kept.fr_rows.any(axis=1)).any() and kept.notes['zero_rows'] == 0
    full = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1)
    assert full.notes['zero_rows'] == 0 and full.notes['zero_row_shooting_points'] == 0
    path = s.rd.shots[0]
    shot = path.fnames[0]
    assert shot in full.sp_fname
    si = int(np.asarray(path.locs)[path.shooting_index])      # the shooting frame's local index
    assert si != 0
    cache = get_cache_fname(shot, 'descriptors')
    original = np.load(cache)
    rows = original.copy()
    rows[[0, si]] = 0.0                                   # an in-state frame and the shooting frame
    np.save(cache, rows)
    data = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1)
    assert data.notes['zero_rows'] == 2 and data.notes['zero_row_shooting_points'] == 1
    assert shot not in data.sp_fname and len(data.sp_rows) == len(full.sp_rows) - 1
    assert len(data.fr_rows) == len(full.fr_rows) - 2 and data.fr_rows.any(axis=1).all()
    kept = data.fr_loc[data.fr_fname == shot]
    assert 0 not in kept and si not in kept
    again = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1, previous=data)
    assert again.notes['reused_files'] == again.notes['n_files'] - 1  # the file is read again
    assert again.notes['zero_rows'] == 2 and np.array_equal(again.fr_rows, data.fr_rows)
    np.save(cache, original)                              # a later descriptor pass filled them
    filled = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1, previous=again)
    assert filled.notes['zero_rows'] == 0 and shot in filled.sp_fname
    assert np.array_equal(filled.fr_rows, full.fr_rows) and np.array_equal(filled.sp_rows,
                                                                          full.sp_rows)


def test_shot_stride_min_frames(tmp_path):
    """shot_stride applies to shot files longer than shot_stride_min_frames; free files keep free_stride."""
    s = _setup(tmp_path, n_chains=1, n_shots=2)
    long_file = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, shot_stride=2,
                                   shot_stride_min_frames=6)     # 7 frames > 6: strided
    short_file = ds.extract_dataset(s.pe, states='ARB', n_atoms=1, shot_stride=2,
                                    shot_stride_min_frames=7)    # 7 frames <= 7: stride 1
    in_shots = [d.fr_group >= 0 for d in (long_file, short_file)]
    assert (long_file.fr_loc[in_shots[0]] % 2 == 0).all()
    assert (long_file.fr_w[in_shots[0]] == 2.0).all() and in_shots[0].sum() == 2 * 4
    assert in_shots[1].sum() == 2 * 7 and (short_file.fr_w[in_shots[1]] == 1.0).all()
    for data in (long_file, short_file):
        free = data.fr_group < 0
        assert (data.fr_loc[free] % 5 == 0).all() and (data.fr_w[free] == 5.0).all()


def test_dataset_keeps_one_copy_of_the_rows(tmp_path):
    """The reuse bookkeeping holds indices only; rows reused from the previous round are identical."""
    s = _setup(tmp_path, n_chains=2, n_shots=3, n_atoms=3)
    data = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=2)
    cache = data._file_cache
    assert cache and all(v.ndim <= 1 for entry in cache.values() for v in entry.values()
                         if isinstance(v, np.ndarray))
    cache_bytes = sum(v.nbytes for entry in cache.values() for v in entry.values()
                      if isinstance(v, np.ndarray))
    assert cache_bytes <= 0.1 * data.fr_rows.nbytes + 16 * len(data.sp_rows) + 64 * len(cache)
    for fname, entry in cache.items():                     # offsets point at the file's rows
        rows = data.fr_rows[entry['fr_start']:entry['fr_start'] + len(entry['fr_locs'])]
        stored = np.load(get_cache_fname(fname, 'descriptors'))
        assert np.array_equal(rows, stored[entry['fr_locs']])
    again = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=2, previous=data)
    assert again.notes['reused_files'] == again.notes['n_files'] == data.notes['n_files']
    for name in ('sp_rows', 'fr_rows', 'fr_state', 'fr_w', 'fr_group', 'fr_loc',
                 'fr_trajectory', 'fr_frame'):
        assert np.array_equal(getattr(again, name), getattr(data, name)), name
    assert again.trajectories == data.trajectories
    assert again.fr_rows.dtype == np.float32
    assert not np.shares_memory(again.fr_rows, data.fr_rows)
    # a stride that needs frames the previous round did not keep reads the file again
    denser = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1, previous=data)
    assert denser.notes['reused_files'] < denser.notes['n_files']
    fresh = ds.extract_dataset(s.pe, states='ARB', n_atoms=3, free_stride=1)
    assert np.array_equal(denser.fr_rows, fresh.fr_rows) and np.array_equal(denser.sp_rows,
                                                                            fresh.sp_rows)


def test_reused_shooting_points_outside_the_training_frames(tmp_path):
    """shot_stride=2 leaves the shooting frame (index 3) out of the training frames: its row is
    taken from the previous round's shooting points, and the result equals a fresh extraction."""
    s = _setup(tmp_path, n_chains=2, n_shots=3, n_free=1, n_atoms=2, n_parts=2)
    assert s.rd.shooting_index % 2 == 1
    settings = dict(states='ARB', n_atoms=2, shot_stride=2, free_stride=3)
    first = ds.extract_dataset(s.pe, **settings)
    shots = first.fr_group >= 0
    assert shots.any() and (first.fr_loc[shots] % 2 == 0).all()
    assert len(first.sp_rows) == 6
    again = ds.extract_dataset(s.pe, previous=first, **settings)
    assert again.notes['reused_files'] == again.notes['n_files'] == first.notes['n_files']
    fresh = ds.extract_dataset(s.pe, **settings)
    for name in ('sp_rows', 'sp_r_first', 'sp_r_last', 'sp_group', 'fr_rows', 'fr_state',
                 'fr_w', 'fr_group', 'fr_loc', 'fr_trajectory', 'fr_frame'):
        assert np.array_equal(getattr(again, name), getattr(fresh, name)), name
    assert list(again.sp_fname) == list(fresh.sp_fname)


def test_internal_shooting_chains_are_not_held_out(tmp_path):
    """Groups are reactive-chain numbers: chain<A>0 frames are group -1, never held out with chainR0.

    The flow branch also checked here that the trainer keeps such frames out
    of the held-out split; that part moves with the trainer
    (``aimmd.cgaimmd.flow_matching.fit``).
    """
    s = _setup(tmp_path, n_chains=2, n_shots=2)
    shutil.copytree(f'{s.run}/chainR0', f'{s.run}/chainA0')
    s.pe = _ensemble(s.params, s.run)
    data = ds.extract_dataset(s.pe, states='ARB', n_atoms=1)
    internal = np.array(['chainA0' in f for f in data.fr_fname])
    assert internal.any() and (data.fr_group[internal] == -1).all()
    assert set(data.fr_group[~internal & (data.fr_group >= 0)].tolist()) == {0, 1}
    sp_internal = np.array(['chainA0' in f for f in data.sp_fname])
    assert sp_internal.sum() == 2 and (data.sp_group[sp_internal] == -1).all()


def test_is_initial_path(tmp_path):
    s = _setup(tmp_path, n_chains=1, n_shots=1)
    assert not any(ds.is_initial_path(path) for path in s.pe)
    shutil.copytree(f'{s.run}/chainR0', f'{s.run}/initialARB')
    assert ds.is_initial_path(aimmd.Path(f'{s.run}/initialARB/path000001.xtc'))


# --------------------------------------------------------------------------------------------
# the extraction of the flow branch, the old names
# --------------------------------------------------------------------------------------------

def test_default_series_reproduces_the_flow_branch_extraction(tmp_path):
    """Every field and note of the flow branch's extract_flow_dataset, bit for bit (bb98943)."""
    run, manifest = golden.write_fixture(tmp_path)
    pathensemble = golden.fixture_ensemble(run, manifest)
    # the fixture's ensemble is the one the trainer builds from the run folder
    params = cgaimmd_params_namespace()

    def described(pe):
        return sorted((tuple(os.path.relpath(str(f), run) for f in path.fnames),
                       tuple(int(i) for i in path.locs), path.shooting_index) for path in pe)
    assert described(pathensemble) == described(_ensemble(params, run))
    recorded = np.load(golden.GOLDEN)
    for series in (None, 'descriptors'):
        extract = (ds.extract_dataset if series is None else
                   lambda *args, **kwargs: ds.extract_dataset(*args, series=series, **kwargs))
        datasets = golden.run_configs(extract, pathensemble, run)
        assert list(datasets) == list(golden.CONFIGS)
        for name, data in datasets.items():
            for key, value in golden.flatten(data, run).items():
                stored = recorded[f'{name}/{key}']
                if key == 'notes':
                    assert json.loads(str(value)) == json.loads(str(stored)), name
                    continue
                assert value.dtype == stored.dtype, (name, key)
                assert value.shape == stored.shape and np.array_equal(value, stored), (name, key)
    # what the golden exercises
    notes = {name: json.loads(str(recorded[f'{name}/notes'])) for name in golden.CONFIGS}
    assert notes['exclusions']['excluded_shooting_points'] == 2
    assert notes['uniform_free_stride_2']['zero_rows'] > 0
    assert notes['uniform_free_stride_2']['zero_row_shooting_points'] == 1
    assert notes['uniform_free_stride_2']['files_missing'] == 1
    assert notes['first_path_twice']['frames_beyond_cache'] > 0
    assert notes['reuse_same_stride']['reused_files'] > 0


def test_old_names_and_call_signatures(tmp_path):
    """The archived BioEmu2 fits call the old names with these arguments."""
    s = _setup(tmp_path, n_chains=1, n_shots=2)
    (tmp_path / 'exclude.json').write_text(json.dumps(
        {os.path.relpath(s.rd.shots[0].fnames[0], s.run): [1]}))
    import aimmd.network.flow_fit as flow_fit
    with pytest.warns(DeprecationWarning, match='extract_flow_dataset is deprecated; use '
                                                'aimmd.cgaimmd.dataset.extract_dataset'):
        extract_flow_dataset = flow_fit.extract_flow_dataset
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        resolve = flow_fit._resolve_exclusions
        assert flow_fit.FlowDataset is ds.TrainingDataset
        assert flow_fit.path_weights is ds.path_weights
    assert extract_flow_dataset is ds.extract_dataset and resolve is ds.resolve_exclusions
    exclusions = resolve(str(tmp_path / 'exclude.json'), s.run, os.getcwd())
    data = extract_flow_dataset(s.pe, states='ARB', n_atoms=1, weights=np.ones(len(s.pe)),
                                free_stride=5, shot_stride=1, shot_stride_min_frames=0,
                                exclude_frames=exclusions, previous=None)
    assert data.notes['excluded_path_frames'] == 1


# --------------------------------------------------------------------------------------------
# descriptor series
# --------------------------------------------------------------------------------------------

def test_extraction_under_a_named_series(tmp_path):
    """series= reads {trajectory}.{series}.npy; the default still reads the old caches."""
    s = _setup(tmp_path, n_chains=2, n_shots=2, n_atoms=2)
    _copy_series(s.pe, 'descriptors-cgtest', lambda rows: rows + np.float32(100.0))
    default = ds.extract_dataset(s.pe, states='ARB', n_atoms=2, free_stride=1)
    named = ds.extract_dataset(s.pe, states='ARB', n_atoms=2, free_stride=1,
                               series='descriptors-cgtest')
    assert named.notes['series'] == 'descriptors-cgtest'
    assert default.notes['series'] == 'descriptors'
    for name in ('fr_fname', 'fr_loc', 'fr_state', 'fr_w', 'sp_fname', 'fr_trajectory',
                 'fr_frame'):
        assert np.array_equal(np.asarray(getattr(named, name)),
                              np.asarray(getattr(default, name))), name
    assert np.array_equal(named.fr_rows, default.fr_rows + np.float32(100.0))
    assert np.array_equal(named.sp_rows, default.sp_rows + np.float32(100.0))
    # rows are reused only from a dataset of the same series
    switched = ds.extract_dataset(s.pe, states='ARB', n_atoms=2, free_stride=1,
                                  series='descriptors-cgtest', previous=default)
    assert switched.notes['reused_files'] == 0
    assert np.array_equal(switched.fr_rows, named.fr_rows)
    same = ds.extract_dataset(s.pe, states='ARB', n_atoms=2, free_stride=1,
                              series='descriptors-cgtest', previous=named)
    assert same.notes['reused_files'] == same.notes['n_files']
    # a file without the series is missing, as a file without descriptors is
    os.remove(get_cache_fname(s.rd.shots[0].fnames[0], 'descriptors-cgtest'))
    missing = ds.extract_dataset(s.pe, states='ARB', n_atoms=2, series='descriptors-cgtest')
    assert missing.notes['files_missing'] == 1
    assert s.rd.shots[0].fnames[0] not in set(missing.fr_fname.tolist())
    # a cache of the series with another width names the series
    np.save(get_cache_fname(s.rd.shots[1].fnames[0], 'descriptors-cgtest'),
            np.zeros((7, 9), np.float32))
    with pytest.raises(ValueError, match="series 'descriptors-cgtest' has shape \\(7, 9\\)"):
        ds.extract_dataset(s.pe, states='ARB', n_atoms=2, series='descriptors-cgtest')


@pytest.mark.parametrize('series', ['values', 'states', 'descriptors-', 'descriptors/x', '', 3])
def test_series_must_be_a_descriptor_series(tmp_path, series):
    s = _setup(tmp_path, n_chains=1, n_shots=1)
    with pytest.raises(ValueError, match='descriptors_series must be'):
        ds.extract_dataset(s.pe, states='ARB', n_atoms=1, series=series)


# --------------------------------------------------------------------------------------------
# resolve_exclusions
# --------------------------------------------------------------------------------------------

def test_resolve_exclusions(tmp_path):
    """Names relative to the run, a JSON file relative to the params folder, 'all' or frames."""
    run, params_dir = tmp_path / 'run', tmp_path / 'params'
    run.mkdir()
    params_dir.mkdir()
    absolute = str(tmp_path / 'elsewhere' / 'traj.xtc')
    mapping = {'chainR0/path000001.xtc': 'all', 'chainR0/../chainR1/path000002.xtc': [3, '4', 3],
               absolute: (0,)}
    expected = {os.path.join(str(run), 'chainR0', 'path000001.xtc'): 'all',
                os.path.join(str(run), 'chainR1', 'path000002.xtc'): {3, 4},
                absolute: {0}}
    assert ds.resolve_exclusions(None, run, params_dir) is None
    assert ds.resolve_exclusions(mapping, run, params_dir) == expected
    assert ds.resolve_exclusions({}, run, params_dir) == {}
    # a JSON file: on its own or under 'exclude'; relative to the params folder, or absolute
    (params_dir / 'bare.json').write_text(json.dumps(mapping))
    (params_dir / 'wrapped.json').write_text(json.dumps({'exclude': mapping}))
    assert ds.resolve_exclusions('bare.json', str(run), str(params_dir)) == expected
    assert ds.resolve_exclusions(params_dir / 'wrapped.json', run, tmp_path) == expected
    assert ds.resolve_exclusions('params/wrapped.json', run, tmp_path) == expected
    with pytest.raises(FileNotFoundError, match='bare.json'):
        ds.resolve_exclusions('bare.json', run, run)
    # the keys are the file keys of the extraction
    key = next(iter(ds.resolve_exclusions({'a/./b.xtc': [1]}, run, params_dir)))
    assert key == os.path.normpath(os.path.abspath(os.path.join(str(run), 'a', 'b.xtc')))


# --------------------------------------------------------------------------------------------
# trajectories across part files
# --------------------------------------------------------------------------------------------

def _crossing_trajectory(run, n_parts=3, n_atoms=2):
    """``freeA/traj000001.part*``: x runs A -> R -> B -> R -> A..., so the trainer splits it.

    Every later part starts with the frame its predecessor ended with (a
    restart). Returns the part files and the joined :class:`aimmd.Path`.
    """
    x = 1.2 * np.sin(np.linspace(0.0, 3.0 * np.pi, 9 * n_parts + 1))
    fnames = []
    for part in range(n_parts):
        frames = slice(9 * part, 9 * part + 10)
        positions = np.zeros((10, n_atoms, 3), np.float32)
        positions[:, :, 0] = x[frames, None]
        positions[:, :, 1] = 1.5 * np.arange(n_atoms)[None, :]
        fnames.append(_write_with_caches(
            os.path.join(run, 'freeA', f'traj000001.part{part + 1:04d}.xtc'), positions,
            np.arange(9 * part, 9 * part + 10, dtype=float)))
    return fnames, aimmd.Path(fnames, remove_overlapping_frames=True)


def test_trajectory_ids_join_part_files(tmp_path):
    """Part files of one free trajectory share a trajectory; the frame index runs across them."""
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2, n_free=0, n_atoms=2)
    parts, whole = _crossing_trajectory(rd.run)
    params = cgaimmd_params_namespace()
    pe = _ensemble(params, rd.run)
    # the trainer's ensemble holds the trajectory in pieces: some inside one part file, some
    # across two, none with all three
    pieces = [path for path in pe if 'freeA' in str(path.fnames[0])]
    assert len(pieces) > 3
    assert any(len(path.fnames) == 1 for path in pieces)
    assert any(len(path.fnames) == 2 for path in pieces)
    assert all(len(path.fnames) < 3 for path in pieces)
    # the frame index of every frame of the whole trajectory (AIMMD drops the repeated frame)
    expected = {}
    for index, (fname, loc) in enumerate(zip(whole.filenames, whole.locs)):
        expected[(str(fname), int(loc))] = index
    assert len(expected) == 9 * 3 + 1
    for stride in (1, 4):
        data = ds.extract_dataset(pe, states='ARB', n_atoms=2, free_stride=stride)
        free = np.isin(data.fr_fname, parts)
        assert free.sum() == len([loc for (_, loc) in expected if loc % stride == 0])
        trajectory = np.unique(data.fr_trajectory[free])
        assert len(trajectory) == 1
        assert data.trajectories[trajectory[0]] == tuple(parts)
        frames = [expected[(f, int(loc))] for f, loc in zip(data.fr_fname[free],
                                                           data.fr_loc[free])]
        assert data.fr_frame[free].tolist() == frames
        # a shot file is a trajectory of its own, starting at its first frame
        shots = ~free
        assert len(np.unique(data.fr_trajectory[shots])) == 2
        assert np.array_equal(data.fr_frame[shots], data.fr_loc[shots])
        assert sorted(data.trajectories) == sorted([tuple(parts)] + [(str(path.fnames[0]),)
                                                                     for path in rd.shots])
        assert data.notes['n_trajectories'] == 3 and data.notes['trajectory_conflicts'] == 0
    # the order of the paths does not matter
    reversed_pe = assemble_pathensemble(*[pe[i:i + 1] for i in reversed(range(len(pe)))])
    again = ds.extract_dataset(reversed_pe, states='ARB', n_atoms=2, free_stride=1)
    data = ds.extract_dataset(pe, states='ARB', n_atoms=2, free_stride=1)
    assert again.trajectories == data.trajectories
    assert np.array_equal(again.fr_frame, data.fr_frame)
    assert np.array_equal(again.fr_trajectory, data.fr_trajectory)


def test_frame_index_starts_at_the_first_frame_a_path_holds(tmp_path):
    """No path holds the first frames of the trajectory: its frame index still starts at 0."""
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2, n_free=0, n_atoms=2)
    parts, whole = _crossing_trajectory(rd.run)
    order = {(str(f), int(loc)): i for i, (f, loc) in enumerate(zip(whole.filenames,
                                                                    whole.locs))}
    # frames 5-14 (part files 1 and 2) and 20-23 (only part file 3, so a trajectory of its own)
    pe = assemble_pathensemble(PathEnsemble(whole[5:15]), PathEnsemble(whole[20:24]))
    data = ds.extract_dataset(pe, states='ARB', n_atoms=2, free_stride=1)
    assert data.trajectories == [tuple(parts[:2]), (parts[2],)]
    joined = data.fr_trajectory == 0
    assert data.fr_frame[joined].tolist() == [
        order[(f, int(loc))] - 5 for f, loc in zip(data.fr_fname[joined], data.fr_loc[joined])]
    assert sorted(data.fr_frame[joined].tolist()) == list(range(10))
    assert sorted(data.fr_frame[~joined].tolist()) == list(range(4))


def test_trajectory_of_paths_that_run_backwards(tmp_path):
    """A path through part files backwards joins them in time order, too."""
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=0, n_shots=0, n_free=1, n_atoms=1, n_parts=2)
    whole = rd.free[0]
    pe = assemble_pathensemble(PathEnsemble(whole[::-1]))
    data = ds.extract_dataset(pe, states='ARB', n_atoms=1, free_stride=1)
    assert data.trajectories == [tuple(str(f) for f in whole.fnames)]
    order = {(str(f), int(loc)): i for i, (f, loc) in enumerate(zip(whole.filenames,
                                                                    whole.locs))}
    assert data.fr_frame.tolist() == [order[(f, int(loc))] for f, loc in
                                      zip(data.fr_fname, data.fr_loc)]


def test_hand_built_dataset_has_one_trajectory_per_file():
    rows = np.zeros((3, 3), np.float32)
    data = ds.TrainingDataset(
        sp_rows=np.zeros((0, 3), np.float32), sp_r_first=np.zeros(0, np.float32),
        sp_r_last=np.zeros(0, np.float32), sp_group=np.zeros(0, np.int64), sp_fname=[],
        fr_rows=rows, fr_state=np.zeros(3, np.int8), fr_w=np.ones(3),
        fr_group=np.full(3, -1), fr_fname=np.array(['b.xtc', 'a.xtc', 'b.xtc'], dtype=object),
        fr_loc=np.array([4, 0, 5]))
    assert data.fr_trajectory.tolist() == [1, 0, 1]
    assert data.trajectories == [('a.xtc',), ('b.xtc',)]
    assert data.fr_frame.tolist() == [4, 0, 5]


# --------------------------------------------------------------------------------------------
# labels averaged over time
# --------------------------------------------------------------------------------------------

def _logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def _brute_force(labels, trajectory, time, window, include):
    out = np.array(labels, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-np.asarray(labels, dtype=np.float64)))
    for i in np.flatnonzero(include):
        near = include & (trajectory == trajectory[i]) & (np.abs(time - time[i]) <= window)
        out[i] = _logit(p[near].mean())
    return out


def test_time_averaged_labels_known_answer():
    """Committor probabilities 0.2, 0.4, 0.6, 0.8 at t = 0..3, +-1: means 0.3, 0.4, 0.6, 0.7."""
    labels = _logit([0.2, 0.4, 0.6, 0.8])
    out = ds.time_averaged_labels(labels, np.zeros(4), np.arange(4.0), 1.0)
    assert np.allclose(out, _logit([0.3, 0.4, 0.6, 0.7]), rtol=0, atol=1e-12)
    # a window wider than the trajectory: every label becomes the mean, 0.5
    assert np.allclose(ds.time_averaged_labels(labels, np.zeros(4), np.arange(4.0), 10.0), 0.0,
                       rtol=0, atol=1e-12)
    # window 0: a frame alone keeps its label
    assert np.allclose(ds.time_averaged_labels(labels, np.zeros(4), np.arange(4.0), 0.0),
                       labels, rtol=0, atol=1e-12)


def test_time_averaged_labels_by_trajectory_and_time():
    """Trajectories never mix; the window is in time, not in rows; left-out frames keep labels."""
    rng = np.random.default_rng(3)
    n = 200
    labels = rng.normal(0.0, 3.0, n)
    trajectory = rng.integers(0, 4, n)
    time = rng.choice(np.arange(0.0, 500.0, 2.5), n)       # unsorted, gaps, repeated times
    include = rng.random(n) < 0.7
    labels[5] = np.nan                                     # never averaged, never enters a mean
    out = ds.time_averaged_labels(labels, trajectory, time, 12.5, include=include)
    use = include & np.isfinite(labels)
    assert np.allclose(out[use], _brute_force(labels, trajectory, time, 12.5, use)[use],
                       rtol=0, atol=1e-10)
    assert np.array_equal(out[~use], labels[~use], equal_nan=True)
    # without include every finite frame takes part
    every = ds.time_averaged_labels(labels, trajectory, time, 12.5)
    finite = np.isfinite(labels)
    assert np.allclose(every[finite], _brute_force(labels, trajectory, time, 12.5,
                                                   finite)[finite], rtol=0, atol=1e-10)
    # string trajectory ids work as well
    named = ds.time_averaged_labels(labels, np.array([f'traj{k}' for k in trajectory]), time,
                                    12.5, include=include)
    assert np.allclose(named, out, rtol=0, atol=0, equal_nan=True)


def test_time_averaged_labels_window_none_is_bit_for_bit():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        labels = rng.normal(0.0, 5.0, 50).astype(dtype)
        labels[[3, 7]] = [np.nan, np.inf]
        out = ds.time_averaged_labels(labels, np.zeros(50), np.arange(50.0), None)
        assert out is not labels and out.dtype == labels.dtype
        assert out.tobytes() == labels.tobytes()
    assert ds.time_averaged_labels(labels, None, None, None).tobytes() == labels.tobytes()


def test_time_averaged_labels_keep_the_dtype():
    labels = np.array([1.0, -1.0, 0.5], np.float32)
    out = ds.time_averaged_labels(labels, [0, 0, 0], [0.0, 1.0, 2.0], 1.0)
    assert out.dtype == np.float32
    assert ds.time_averaged_labels(np.array([1, 2]), [0, 0], [0.0, 1.0], 1.0).dtype == np.float64


def test_time_averaged_labels_of_any_size_stay_finite():
    """Averaged in log space: no clipping of large committor logits."""
    out = ds.time_averaged_labels(np.array([500.0, 500.0, -500.0, 500.0, 400.0]),
                                  [0, 0, 1, 1, 2], [0.0, 1.0, 0.0, 1.0, 0.0], 1.0)
    assert np.allclose(out[:2], 500.0, rtol=0, atol=1e-9)
    assert np.allclose(out[2:4], 0.0, rtol=0, atol=1e-9)
    assert out[4] == pytest.approx(400.0, abs=1e-9)
    # mean of 1 - e^-400 and 1 - e^-500: logit = 400 + log 2 (to e^-100)
    pair = ds.time_averaged_labels(np.array([400.0, 500.0]), [0, 0], [0.0, 1.0], 1.0)
    assert np.allclose(pair, 400.0 + np.log(2.0), rtol=0, atol=1e-9)


def test_time_averaged_labels_across_part_files(tmp_path):
    """On a dataset: the window runs across the part files of one trajectory."""
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1, n_free=0, n_atoms=2)
    parts, _ = _crossing_trajectory(rd.run)
    data = ds.extract_dataset(_ensemble(cgaimmd_params_namespace(), rd.run), states='ARB',
                              n_atoms=2, free_stride=1)
    labels = np.sin(0.7 * data.fr_frame) * 3.0 + data.fr_trajectory
    frame_interval_ps = 5.0
    time = data.fr_frame * frame_interval_ps
    include = data.fr_state == 0
    out = ds.time_averaged_labels(labels, data.fr_trajectory, time, 10.0, include=include)
    expected = _brute_force(labels, data.fr_trajectory, time, 10.0, include)
    assert np.allclose(out, expected, rtol=0, atol=1e-10)
    # the last reactive frame of part 1 averages with the first frames of part 2
    first_part = np.flatnonzero((data.fr_fname == parts[0]) & include)
    second_part = (data.fr_fname == parts[1]) & include
    last = first_part[np.argmax(data.fr_frame[first_part])]
    neighbours = include & (np.abs(data.fr_frame - data.fr_frame[last]) <= 2) & \
        (data.fr_trajectory == data.fr_trajectory[last])
    assert (neighbours & second_part).any()
    assert out[last] == pytest.approx(_logit(np.mean(1 / (1 + np.exp(-labels[neighbours])))),
                                      abs=1e-10)


@pytest.mark.parametrize('kwargs, message', [
    (dict(window=-1.0), 'window must be'),
    (dict(window=float('nan')), 'window must be'),
    (dict(window=float('inf')), 'window must be'),
    (dict(trajectory=[0, 0]), 'trajectory has 2 entries for 3 labels'),
    (dict(time=[0.0, 1.0]), 'time has 2 entries for 3 labels'),
    (dict(include=[True, False]), 'include has 2 entries for 3 labels'),
    (dict(time=[0.0, np.nan, 2.0]), 'finite time'),
    (dict(labels=np.zeros((3, 1))), 'one-dimensional'),
])
def test_time_averaged_labels_checks(kwargs, message):
    arguments = dict(labels=np.zeros(3), trajectory=[0, 0, 0], time=[0.0, 1.0, 2.0], window=1.0)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ds.time_averaged_labels(arguments.pop('labels'), arguments.pop('trajectory'),
                                arguments.pop('time'), arguments.pop('window'), **arguments)
