"""Post-training on an existing run folder and RFPS -> C-GAIMMD restarts:
``aimmd.cgaimmd.restart``.

``derive_run`` mirrors a finished run into a new folder that a C-GAIMMD
trainer can write into without touching the source; ``check_descriptor_caches``,
``migrate_descriptors`` and ``check_cgaimmd_restart`` catch and migrate the
descriptor caches of a run that sampled with another network, under the
params' descriptor series; ``check_descriptor_record`` refuses caches of the
default series written with other descriptor settings (or, with a cluster of
waters, with no record of their settings). The recipes run in process with
the real trainer and a fake fit.
"""

import json
import os
import sys
import types

import numpy as np
import pytest

from aimmd.cache.npy import save_npy
from aimmd.cgaimmd import restart
from aimmd.cgaimmd.restart import (check_cgaimmd_restart, check_descriptor_caches,
                                   check_descriptor_record, derive_run, migrate_descriptors)
from aimmd.path.utils import get_cache_fname
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """``Params.load`` leaves the params file's folder in ``sys.path``; take it out after the test.

    Otherwise a later ``import params`` (``tests/test_retinal.py`` imports its
    ``params.py`` by name) finds a ``params.py`` of this module's runs first.
    """
    monkeypatch.setattr(sys, 'path', list(sys.path))


def snapshot(folder):
    """{relative path: (size, mtime_ns, is_link)} of every file under `folder`."""
    out = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            fname = os.path.join(root, name)
            stat = os.lstat(fname)
            out[os.path.relpath(fname, folder)] = (stat.st_size, stat.st_mtime_ns,
                                                   os.path.islink(fname))
    return out


def decorate_source(run):
    """Add what a real run folder holds besides trajectories and caches."""
    touch = {
        'networkARB.h5': b'net', 'networkARB.json': b'{}', 'binsARB.npy': b'bins',
        'densitiesARB.npy': b'dens', 'trainARB.log': b'log',
        'chainR0/pool.log': b'pool', 'chainR0/worker.log': b'log',
        'chainR0/back.xtc': b'leftover', 'chainR0/forw.tpr': b'tpr',
        'chainR0/path000001.xtc.cgaimmd.json': b'{}', 'chainR0/.cgaimmd_shooting_point.json': b'{}',
        'chainR0/.cgaimmd_shooting_point.npy': b'x', 'chainR0/path000002.xtc.flow.json': b'{}',
        'chainR0/.flow_shooting_point.json': b'{}', 'chainR1/tps_weights.npy': b'w',
        'cgaimmd_descriptors.json': b'{}',
        'freeA/traj000001.part0001.edr': b'edr', 'freeA/traj000001.cpt': b'cpt',
        'freeA/traj000001.part0001.log': b'mdlog', 'flow_fitARB/state.pt': b'state',
        'chainR0/path000001.xtc.values.new.npy': b'new',
    }
    for name, data in touch.items():
        fname = os.path.join(run, name)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as file:
            file.write(data)
    initial = os.path.join(run, 'initialARB')
    os.makedirs(initial, exist_ok=True)
    return touch


# ---------------------------------------------------------------------------
# derive_run
# ---------------------------------------------------------------------------

def test_derive_run_does_not_touch_source(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path)
    decorate_source(layout.run)
    before = snapshot(layout.run)
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target)
    assert snapshot(layout.run) == before
    # trajectories (and .edr) are symlinks into the source, absolute
    for shot in layout.shots + layout.free:
        rel = os.path.relpath(shot.fname, layout.run)
        linked = os.path.join(target, rel)
        assert os.path.islink(linked)
        assert os.path.realpath(linked) == os.path.realpath(shot.fname)
        assert os.path.isabs(os.readlink(linked))
    assert os.path.islink(os.path.join(target, 'freeA', 'traj000001.part0001.edr'))
    assert os.path.isfile(os.path.join(target, 'chainR1', 'tps_weights.npy'))
    assert not os.path.islink(os.path.join(target, 'chainR1', 'tps_weights.npy'))
    for skipped in ('networkARB.h5', 'networkARB.json', 'binsARB.npy', 'densitiesARB.npy',
                    'trainARB.log', 'chainR0/pool.log', 'chainR0/worker.log', 'chainR0/back.xtc',
                    'chainR0/forw.tpr', 'chainR0/path000001.xtc.cgaimmd.json',
                    'chainR0/.cgaimmd_shooting_point.json', 'chainR0/.cgaimmd_shooting_point.npy',
                    'chainR0/path000002.xtc.flow.json', 'chainR0/.flow_shooting_point.json',
                    'cgaimmd_descriptors.json',
                    'freeA/traj000001.cpt', 'freeA/traj000001.part0001.log',
                    'flow_fitARB/state.pt', 'chainR0/path000001.xtc.values.new.npy'):
        assert not os.path.lexists(os.path.join(target, skipped)), skipped
    assert os.path.isdir(os.path.join(target, 'initialARB'))
    assert report['source'] == os.path.abspath(layout.run)
    assert report['target'] == os.path.abspath(target)
    assert report['linked'] == len(layout.shots) + len(layout.free) + 1      # + the .edr
    assert report['copied'] == len(layout.shots) + len(layout.free) + 1      # states + tps weights
    assert report['linked_bytes'] > 0 and report['copied_bytes'] > 0
    assert 'chainR0/pool.log' in report['skipped']
    assert 'networkARB.h5' not in report['skipped']                  # top level: not mirrored at all
    # writing into the target (as the trainer does) never reaches the source
    save_npy(get_cache_fname(os.path.join(target, 'chainR0', 'path000001.xtc'), 'descriptors'),
             np.zeros((7, 3), dtype=np.float32))
    assert snapshot(layout.run) == before


def test_derive_run_refuses_nonempty(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    target = tmp_path / 'post'
    target.mkdir()
    (target / 'something').write_text('x')
    with pytest.raises(ValueError, match='not empty'):
        derive_run(layout.run, str(target))
    empty = tmp_path / 'empty'
    empty.mkdir()
    derive_run(layout.run, str(empty))                                # an empty folder is fine
    with pytest.raises(ValueError, match='inside the source'):
        derive_run(layout.run, os.path.join(layout.run, 'post'))
    with pytest.raises(ValueError, match='does not exist'):
        derive_run(str(tmp_path / 'missing'), str(tmp_path / 'post2'))


def add_bias_caches(layout):
    """A ``<traj>.bias.npy`` for every trajectory, as a ``record_bias`` run leaves them."""
    for path in layout.shots + layout.free:
        save_npy(get_cache_fname(path.fname, 'bias'), np.linspace(0.0, 1.0, len(path)))


def test_derive_run_copies_states_and_bias_by_default(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    add_bias_caches(layout)
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target)
    for path in layout.shots + layout.free:
        linked = os.path.join(target, os.path.relpath(path.fname, layout.run))
        for attribute in ('states', 'bias'):              # neither depends on the network
            cache = get_cache_fname(linked, attribute)
            assert os.path.isfile(cache) and not os.path.islink(cache)
            np.testing.assert_array_equal(np.load(cache),
                                          np.load(get_cache_fname(path.fname, attribute)))
        assert not os.path.lexists(get_cache_fname(linked, 'descriptors'))
        assert not os.path.lexists(get_cache_fname(linked, 'values'))
    assert report['bias_caches_left_out'] == 0
    other = str(tmp_path / 'post_values')
    derive_run(layout.run, other, copy_caches=('states', 'bias', 'values'))
    linked = os.path.join(other, os.path.relpath(layout.shots[0].fname, layout.run))
    assert os.path.isfile(get_cache_fname(linked, 'values'))


def test_derive_run_warns_about_uncopied_bias(tmp_path, capsys):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    add_bias_caches(layout)
    capsys.readouterr()
    # the bias cache of a chain's in-flight segment is not counted: the segment is not mirrored
    from tests._helpers_unit import write_trajectory
    back = write_trajectory(os.path.join(layout.run, 'chainR0'), stem='back',
                            positions=np.zeros((3, 1, 3), dtype=np.float32))
    save_npy(get_cache_fname(back, 'bias'), np.zeros(3))
    report = derive_run(layout.run, str(tmp_path / 'post'), copy_caches=('states',))
    assert report['bias_caches_left_out'] == len(layout.shots) + len(layout.free)
    out = capsys.readouterr().out
    assert f"WARNING: derive_run left out {report['bias_caches_left_out']} bias cache(s)" in out
    # no warning without bias caches in the source
    quiet = build_cgaimmd_run_dir(tmp_path / 'quiet', n_chains=1, n_shots=1)
    assert derive_run(quiet.run, str(tmp_path / 'post_quiet'),
                      copy_caches=('states',))['bias_caches_left_out'] == 0
    assert 'bias cache' not in capsys.readouterr().out


def test_derive_run_links_plumed_outputs(tmp_path):
    """The PLUMED outputs a bias_source='file' trainer rebuilds bias caches from are linked."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    outputs = {'chainR0/path000001_COLVAR': True, 'freeA/traj000001.part0001_COLVAR': True,
               'freeA/COLVAR': True,                        # the cumulative output of a free folder
               'chainR0/back_COLVAR': False,                # a shot in flight
               'chainR0/COLVAR': False,                     # the unrenamed output of one
               'freeA/bck.1.COLVAR': False, 'freeA/orphan_COLVAR': False}
    for name in outputs:
        with open(os.path.join(layout.run, name), 'w') as file:
            file.write('#! FIELDS time opes.bias\n0 0.5\n')
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target)
    for name, linked in outputs.items():
        fname = os.path.join(target, name)
        assert os.path.islink(fname) == linked, name
        if linked:
            assert os.path.realpath(fname) == os.path.realpath(os.path.join(layout.run, name))
        else:
            assert name in report['skipped']
            assert not os.path.lexists(fname)


def test_derive_run_keeps_back_forw_names_outside_chains(tmp_path):
    """Only chain folders hold segments; an initial file called seeds_backbone is mirrored."""
    from tests._helpers_unit import write_trajectory
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    initial = os.path.join(layout.run, 'initialARB')
    os.makedirs(initial)
    positions = np.zeros((3, 1, 3), dtype=np.float32)
    names = []
    for stem in ('seeds_backbone', 'free_md_forward'):
        fname = write_trajectory(initial, stem=stem, positions=positions)
        save_npy(get_cache_fname(fname, 'states'), np.array(['A', 'A', 'A']))
        names.append(os.path.basename(fname))
    for name in ('back.xtc', 'forw.part0001.xtc', 'back.xtc.states.npy'):
        with open(os.path.join(layout.run, 'chainR0', name), 'wb') as file:
            file.write(b'segment')
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target)
    for name in names:
        assert os.path.islink(os.path.join(target, 'initialARB', name))
        assert os.path.isfile(get_cache_fname(os.path.join(target, 'initialARB', name), 'states'))
    for name in ('back.xtc', 'forw.part0001.xtc', 'back.xtc.states.npy'):
        assert f'chainR0/{name}' in report['skipped']
        assert not os.path.lexists(os.path.join(target, 'chainR0', name))


def test_derive_run_needs_a_trajectory(tmp_path):
    source = tmp_path / 'source'
    (source / 'chainR0').mkdir(parents=True)
    (source / 'chainR0' / 'pool.log').write_text('')
    (source / 'chainR0' / 'path000001.dcd').write_bytes(b'traj')
    save_npy(str(source / 'chainR0' / 'path000001.dcd.states.npy'), np.array(['A', 'R', 'B']))
    target = tmp_path / 'post'
    with pytest.raises(ValueError, match=r'no trajectory \(\.xtc, \.trr\).*params='):
        derive_run(str(source), str(target))
    assert not target.exists()                                        # refused before writing
    # the params' trajectory_extension is looked for as well
    report = derive_run(str(source), str(target),
                        params=types.SimpleNamespace(trajectory_extension='.dcd'))
    assert report['linked'] == 1 and report['copied'] == 1
    assert os.path.islink(target / 'chainR0' / 'path000001.dcd')
    assert (target / 'chainR0' / 'path000001.dcd.states.npy').is_file()


@pytest.mark.parametrize('link', ['hardlink', 'copy'])
def test_derive_run_link_modes(tmp_path, link):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    target = str(tmp_path / f'post_{link}')
    derive_run(layout.run, target, link=link)
    source = layout.shots[0].fname
    linked = os.path.join(target, os.path.relpath(source, layout.run))
    assert not os.path.islink(linked)
    same_inode = os.stat(linked).st_ino == os.stat(source).st_ino
    assert same_inode == (link == 'hardlink')
    with open(linked, 'rb') as a, open(source, 'rb') as b:
        assert a.read() == b.read()
    with pytest.raises(ValueError, match='link'):
        derive_run(layout.run, str(tmp_path / 'bad'), link='junction')


def test_derive_run_without_initial(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    decorate_source(layout.run)
    target = str(tmp_path / 'post')
    derive_run(layout.run, target, include_initial=False)
    assert not os.path.exists(os.path.join(target, 'initialARB'))


def test_derive_run_links_checked_descriptors(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    params = cgaimmd_params_namespace()
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target, descriptors='link', params=params)
    linked = os.path.join(target, os.path.relpath(layout.shots[0].fname, layout.run))
    assert os.path.islink(get_cache_fname(linked, 'descriptors'))
    assert report['descriptors'] == 'link'
    assert report['descriptor_check']['n_files'] > 0
    with pytest.raises(TypeError, match='params'):
        derive_run(layout.run, str(tmp_path / 'noparams'), descriptors='link')
    with pytest.raises(ValueError, match="'recompute' or 'link'"):
        derive_run(layout.run, str(tmp_path / 'bad'), descriptors='keep')


def test_derive_run_removes_linked_descriptors_that_fail_the_check(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    for path in layout.shots + layout.free:                           # another network's width
        fname = get_cache_fname(path.fname, 'descriptors')
        save_npy(fname, np.zeros((len(np.load(fname)), 2), dtype=np.float32))
    target = str(tmp_path / 'post')
    with pytest.raises(ValueError, match='migrate_descriptors'):
        derive_run(layout.run, target, descriptors='link', params=cgaimmd_params_namespace())
    linked = os.path.join(target, os.path.relpath(layout.shots[0].fname, layout.run))
    assert os.path.islink(linked)                                     # the rest of the mirror stays
    assert not os.path.lexists(get_cache_fname(linked, 'descriptors'))


# ---------------------------------------------------------------------------
# check_descriptor_caches
# ---------------------------------------------------------------------------

def test_check_descriptor_caches(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path)
    params = cgaimmd_params_namespace()
    report = check_descriptor_caches(params, layout.run)
    assert report['n_files'] == 8 and report['n_frames'] == 16
    assert report['width'] == 3
    assert report['max_abs_diff'] <= 1e-3
    assert report == check_descriptor_caches(params, layout.run)       # seeded: reproducible
    small = check_descriptor_caches(params, layout.run, n_files=2, n_frames=1)
    assert small['n_files'] == 2 and small['n_frames'] == 2
    # an empty or missing run folder has nothing to check
    assert check_descriptor_caches(params, str(tmp_path / 'missing'))['n_files'] == 0


def test_check_descriptor_caches_width_mismatch(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    save_npy(fname, np.zeros((7, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='migrate_descriptors') as info:
        check_descriptor_caches(cgaimmd_params_namespace(), layout.run, n_files=100)
    assert 'width 2' in str(info.value)


def test_check_descriptor_caches_value_mismatch(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    save_npy(fname, np.load(fname) + 0.5)
    with pytest.raises(ValueError, match='migrate_descriptors') as info:
        check_descriptor_caches(cgaimmd_params_namespace(), layout.run, n_files=100)
    assert 'differ' in str(info.value)


def test_check_descriptor_caches_skips_uncomputed_rows(tmp_path):
    """Rows the trainer has not filled yet (NaN) and frames past the cache end are skipped."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1, n_free=0)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    rows = np.load(fname)
    rows[1:] = np.nan
    save_npy(fname, rows[:4])
    report = check_descriptor_caches(cgaimmd_params_namespace(), layout.run, n_frames=5)
    assert report['n_frames'] == 1


def test_check_descriptor_caches_skips_rows_past_the_trajectory(tmp_path):
    """A cache longer than its trajectory (a rewritten file): only the trajectory's frames are sampled."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1, n_free=0)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    rows = np.load(fname)
    save_npy(fname, np.concatenate([rows, np.zeros((3, rows.shape[1]), dtype=rows.dtype)]))
    report = check_descriptor_caches(cgaimmd_params_namespace(), layout.run, n_frames=100)
    assert report['n_frames'] == len(rows)


# ---------------------------------------------------------------------------
# migrate_descriptors
# ---------------------------------------------------------------------------

def test_migrate_descriptors(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    params = cgaimmd_params_namespace()
    caches = [get_cache_fname(p.fname, 'descriptors') for p in layout.shots + layout.free]
    dry = migrate_descriptors(params, layout.run, dry_run=True)
    assert dry['renamed'] == len(caches) and dry['dry_run']
    assert all(os.path.isfile(c) for c in caches)
    report = migrate_descriptors(params, layout.run)
    assert report['renamed'] == len(caches) and not report['dry_run']
    for cache in caches:
        assert not os.path.exists(cache)
        assert os.path.isfile(cache[:-len('.npy')] + '.pre_cgaimmd.npy')
    # nothing left to migrate; an existing backup is never overwritten
    assert migrate_descriptors(params, layout.run)['renamed'] == 0
    save_npy(caches[0], np.zeros((7, 3), dtype=np.float32))
    with pytest.raises(ValueError, match='exist already'):
        migrate_descriptors(params, layout.run)
    assert os.path.isfile(caches[0])
    assert migrate_descriptors(params, layout.run, backup_suffix='second')['renamed'] == 1


def test_migrated_run_passes_the_check(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    for path in layout.shots:
        fname = get_cache_fname(path.fname, 'descriptors')
        save_npy(fname, np.zeros((7, 2), dtype=np.float32))
    params = cgaimmd_params_namespace()
    with pytest.raises(ValueError):
        check_cgaimmd_restart(params, layout.run, descriptors=True)
    migrate_descriptors(params, layout.run)
    check_cgaimmd_restart(params, layout.run, descriptors=True)        # no stale cache is left


# ---------------------------------------------------------------------------
# check_cgaimmd_restart
# ---------------------------------------------------------------------------

def test_check_cgaimmd_restart_tps_raises(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=1)
    params = cgaimmd_params_namespace()
    check_cgaimmd_restart(params, layout.run, descriptors=True)        # RFPS history: fine
    save_npy(os.path.join(layout.run, 'chainR1', 'tps_weights.npy'), np.ones(1))
    with pytest.raises(NotImplementedError, match="continuing a TPS run as chain_type='cgaimmd' "
                                                  "is not supported"):
        check_cgaimmd_restart(params, layout.run)
    # other chain types are not checked
    check_cgaimmd_restart(cgaimmd_params_namespace(chain_type='rfps'), layout.run)
    # only chain folders hold TPS history: a tps_weights.npy elsewhere is not one
    os.remove(os.path.join(layout.run, 'chainR1', 'tps_weights.npy'))
    save_npy(os.path.join(layout.run, 'freeA', 'tps_weights.npy'), np.ones(1))
    save_npy(os.path.join(layout.run, 'tps_weights.npy'), np.ones(1))
    check_cgaimmd_restart(params, layout.run)


def test_tps_restart_switch(tmp_path, monkeypatch):
    """The refusal of TPS histories lives in one flag."""
    assert restart.TPS_RESTART_AS_CGAIMMD is False
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    save_npy(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'), np.ones(1))
    monkeypatch.setattr(restart, 'TPS_RESTART_AS_CGAIMMD', True)
    check_cgaimmd_restart(cgaimmd_params_namespace(), layout.run)


def test_check_cgaimmd_restart_descriptors_switch(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    save_npy(fname, np.zeros((7, 2), dtype=np.float32))
    params = cgaimmd_params_namespace()
    check_cgaimmd_restart(params, layout.run)                          # the seam's call
    with pytest.raises(ValueError, match='migrate_descriptors'):
        check_cgaimmd_restart(params, layout.run, descriptors=True)


# ---------------------------------------------------------------------------
# the recipes, in process (the trainer is real, the fit a fake returning
# (losses, info) as the C-GAIMMD fits do)
# ---------------------------------------------------------------------------

RECIPE_PARAMS = '''\
import numpy as np
import torch
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork

engine = 'toy'
states = 'ARB'
chain_type = {chain_type!r}
initial_paths = ['seeds_A.xtc', 'seeds_B.xtc']
initial_network = 'pretrained.h5'
round_shooting_index = True
nbins = 3
cgaimmd_parameters = {{'placement': 'none', 'lambda_clip': 10.0, 'window_after_trained_shots': 5,
                   'training_device': 'cpu', 'values_device': 'cpu'}}

def toy_mdrun(ts):
    ts.positions = ts.positions + 0.01

def states_function(trajectory):
    x = np.array([ts.positions[0, 0] for ts in trajectory])
    return np.where(x <= -0.5, 'A', np.where(x >= 0.5, 'B', 'R'))

def descriptors_function(trajectory):
    frames = [ts.positions.copy() for ts in trajectory]
    return np.array(frames, dtype=np.float32).reshape(len(frames), -1)

class Network(TinyFlowNetwork):
    pass

network = Network()

FIT_INPUTS = []

def fit(params, pathensemble, verbose=False, worker=None):
    FIT_INPUTS.append({{k: v.detach().clone() for k, v in params.network.state_dict().items()}})
    with torch.no_grad():
        params.network.committor_net[-1].bias.add_(0.25)
    return [0.5], {{'status': 'trained', 'reason': '', 'round': 1, 'record': {{}},
                   'network_metadata': {{'round': 1, 'n_trained_shots': 6, 'step': 5,
                                        'fit': 'tests.fake', 'ema_decay': None}}}}
'''


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def rfps_source(tmp_path):
    """A finished 'RFPS' run whose descriptor caches are another network's (x, x^2).

    Like a Launcher-made run it holds ``initialARB/`` (the seed files with
    their caches), so that a trainer on it never builds that folder lazily.
    """
    from tests._helpers_unit import TinyNetwork, write_trajectory
    import torch
    layout = build_cgaimmd_run_dir(tmp_path / 'source')
    initial = os.path.join(layout.run, 'initialARB')
    os.makedirs(initial)
    for kind, x in (('A', [-1.0, -1.1, -0.9]), ('B', [1.0, 1.1, 0.9])):
        positions = np.zeros((len(x), 1, 3), dtype=np.float32)
        positions[:, 0, 0] = x
        fname = write_trajectory(initial, stem=f'seeds_{kind}', positions=positions)
        save_npy(get_cache_fname(fname, 'states'), np.full(len(x), kind))
        save_npy(get_cache_fname(fname, 'descriptors'),
                 np.zeros((len(x), 2), dtype=np.float32))
    for path in layout.shots + layout.free:
        fname = get_cache_fname(path.fname, 'descriptors')
        x = np.load(fname)[:, 0]
        save_npy(fname, np.stack([x, x ** 2], axis=1).astype(np.float32))
    torch.save(TinyNetwork().state_dict(), os.path.join(layout.run, 'networkARB.h5'))
    return layout


@pytest.fixture
def no_lazy_build(monkeypatch):
    """The recipes must take the trainer's own path, never the lazy Launcher._build."""
    def refuse(self, *args, **kwargs):
        raise AssertionError('Worker.initial_paths built initial<s>/ lazily')
    monkeypatch.setattr('aimmd.launcher.Launcher._build', refuse)


def recipe_params(tmp_path, chain_type):
    import torch
    import aimmd
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    from tests._helpers_unit import write_trajectory
    folder = tmp_path / f'params_{chain_type}'
    folder.mkdir()
    for kind, x in (('A', [-1.0, -1.1, -0.9]), ('B', [1.0, 1.1, 0.9])):
        positions = np.zeros((len(x), 1, 3), dtype=np.float32)
        positions[:, 0, 0] = x
        write_trajectory(folder, stem=f'seeds_{kind}', positions=positions)
    torch.save(TinyFlowNetwork().state_dict(), folder / 'pretrained.h5')
    (folder / 'params.py').write_text(RECIPE_PARAMS.format(chain_type=chain_type))
    return aimmd.Params.load(str(folder / 'params.py'), save=False)


def pretrained_weights(tmp_path, chain_type):
    import torch
    return torch.load(tmp_path / f'params_{chain_type}' / 'pretrained.h5', weights_only=True)


def same_weights(a, b):
    import torch
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_posttrain_recipe_in_process(tmp_path, restore_signals, no_lazy_build):
    """Post-training: derive_run, then a trainer with a C-GAIMMD network under the source's chain type."""
    import aimmd
    from aimmd.cgaimmd.io import file_sha256
    layout = rfps_source(tmp_path)
    before = snapshot(layout.run)
    target = str(tmp_path / 'post')
    derive_run(layout.run, target)
    assert os.path.islink(os.path.join(target, 'initialARB', 'seeds_A.xtc'))
    params = recipe_params(tmp_path, 'rfps')
    aimmd.Worker(params, target, walltime=120).train(nrounds=1)
    # the trainer wrote the initial network before its first round
    fit_inputs = params.fit.__globals__['FIT_INPUTS']
    assert len(fit_inputs) == 1
    assert same_weights(fit_inputs[0], pretrained_weights(tmp_path, 'rfps'))
    assert snapshot(layout.run) == before                             # the source is untouched
    for path in layout.shots + layout.free:                           # descriptors recomputed
        linked = os.path.join(target, os.path.relpath(path.fname, layout.run))
        descriptors = np.load(get_cache_fname(linked, 'descriptors'))
        assert descriptors.shape[1] == 3
        assert not os.path.islink(get_cache_fname(linked, 'descriptors'))
    record = json.load(open(os.path.join(target, 'networkARB.json')))
    assert record['sha256'] == file_sha256(os.path.join(target, 'networkARB.h5'))
    assert record['round'] == 1 and record['fit'] == 'tests.fake'
    assert os.path.isfile(os.path.join(target, 'binsARB.npy'))
    assert not os.path.exists(os.path.join(target, 'networkARB.replaced0.h5'))


def test_restart_rfps_as_cgaimmd_in_process(tmp_path, restore_signals, no_lazy_build):
    """The trainer catches the stale caches, they are migrated, the run continues as 'cgaimmd'."""
    import aimmd
    from aimmd.cgaimmd.io import file_sha256
    layout = rfps_source(tmp_path)
    params = recipe_params(tmp_path, 'cgaimmd')
    weights_before = [p.weight for chain in params.shot_chains(layout.run, 'R') for p in chain]
    network_sha = file_sha256(os.path.join(layout.run, 'networkARB.h5'))
    # the trainer's own check_cgaimmd_restart refuses the folder before it writes anything
    with pytest.raises(ValueError, match='migrate_descriptors'):
        aimmd.Worker(params, layout.run, walltime=120).train(nrounds=1)
    assert file_sha256(os.path.join(layout.run, 'networkARB.h5')) == network_sha
    assert not os.path.exists(os.path.join(layout.run, 'networkARB.replaced0.h5'))
    assert not params.fit.__globals__['FIT_INPUTS']
    migrate_descriptors(params, layout.run)
    aimmd.Worker(params, layout.run, walltime=120).train(nrounds=1)
    assert os.path.isfile(os.path.join(layout.run, 'networkARB.replaced0.h5'))
    assert same_weights(params.fit.__globals__['FIT_INPUTS'][0],
                        pretrained_weights(tmp_path, 'cgaimmd'))
    record = json.load(open(os.path.join(layout.run, 'networkARB.json')))
    assert record['round'] == 1
    for path in layout.shots:
        descriptors = np.load(get_cache_fname(path.fname, 'descriptors'))
        assert descriptors.shape[1] == 3
    weights_after = [p.weight for chain in params.shot_chains(layout.run, 'R') for p in chain]
    assert weights_after == weights_before                            # RFPS history unchanged


# ---------------------------------------------------------------------------
# a named descriptor series
# ---------------------------------------------------------------------------

def _copy_series(layout, series):
    """Write every trajectory's 'descriptors' rows as the series `series` as well."""
    for path in layout.shots + layout.free:
        rows = np.load(get_cache_fname(path.fname, 'descriptors'))
        save_npy(get_cache_fname(path.fname, series), rows)


def test_named_series_is_the_one_checked_and_migrated(tmp_path):
    """Only the params' series is read: the 'descriptors' caches of an RFPS history (another
    network's width) never fail the check, and migrate_descriptors renames only the series."""
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    named = 'descriptors-cgtest'
    _copy_series(layout, named)
    for path in layout.shots + layout.free:                       # another network's width
        fname = get_cache_fname(path.fname, 'descriptors')
        save_npy(fname, np.zeros((len(np.load(fname)), 2), dtype=np.float32))
    params = cgaimmd_params_namespace(descriptors_series=named)
    report = check_descriptor_caches(params, layout.run)
    assert report['series'] == named and report['n_frames'] > 0 and report['width'] == 3
    check_cgaimmd_restart(params, layout.run, descriptors=True)
    assert not os.path.exists(os.path.join(layout.run, restart.DESCRIPTOR_RECORD))
    # a stale cache of the named series is caught, and only that series is set aside
    save_npy(get_cache_fname(layout.shots[0].fname, named), np.zeros((7, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='migrate_descriptors'):
        check_descriptor_caches(params, layout.run, n_files=100)
    report = migrate_descriptors(params, layout.run)
    assert report['series'] == named
    assert report['renamed'] == len(layout.shots) + len(layout.free)
    for path in layout.shots + layout.free:
        assert not os.path.exists(get_cache_fname(path.fname, named))
        assert os.path.isfile(get_cache_fname(path.fname, f'{named}.pre_cgaimmd'))
        assert os.path.isfile(get_cache_fname(path.fname, 'descriptors'))      # untouched


def test_derive_run_links_the_named_series(tmp_path):
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    named = 'descriptors-cgtest'
    _copy_series(layout, named)
    params = cgaimmd_params_namespace(descriptors_series=named)
    target = str(tmp_path / 'post')
    report = derive_run(layout.run, target, descriptors='link', params=params)
    assert report['series'] == named and report['descriptor_check']['series'] == named
    linked = os.path.join(target, os.path.relpath(layout.shots[0].fname, layout.run))
    assert os.path.islink(get_cache_fname(linked, named))
    assert not os.path.lexists(get_cache_fname(linked, 'descriptors'))   # not the params' series


# ---------------------------------------------------------------------------
# the record of the default series' descriptor settings
# ---------------------------------------------------------------------------

_DESCRIPTORS = None
"""The module-level object the wrapper below calls (the params-file pattern)."""


def wrapped_descriptors(trajectory):
    return _DESCRIPTORS(trajectory)


@pytest.fixture
def generator_descriptors():
    """``make(generator_atoms, **kw)``: a GeneratorDescriptors of solvated NANMA behind the wrapper."""
    from pathlib import Path
    from aimmd.cgaimmd.preparation.descriptors import GeneratorDescriptors
    structure = Path(__file__).resolve().parent / 'cgaimmd_nanma_water' / 'conf.gro'

    def make(generator_atoms, **kwargs):
        global _DESCRIPTORS
        _DESCRIPTORS = GeneratorDescriptors(str(structure), generator_atoms, **kwargs)
        return _DESCRIPTORS

    yield make
    global _DESCRIPTORS
    _DESCRIPTORS = None


def _record_params(**overrides):
    return cgaimmd_params_namespace(descriptors_function=wrapped_descriptors, **overrides)


def test_descriptor_record_written_then_checked(tmp_path, generator_descriptors):
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms
    descriptors = generator_descriptors(GeneratorAtoms(n_solute=22))
    run = tmp_path / 'run'
    run.mkdir()
    record = check_descriptor_record(_record_params(), str(run))
    fname = run / restart.DESCRIPTOR_RECORD
    assert json.loads(fname.read_text()) == record
    assert record['schema'] == restart.DESCRIPTOR_RECORD_SCHEMA
    assert record['fingerprint'] == descriptors.fingerprint and record['series'] == 'descriptors'
    assert record['spec'] == descriptors.spec()
    assert check_descriptor_record(_record_params(), str(run)) == record     # same settings
    generator_descriptors(GeneratorAtoms(n_solute=22), whole=False)
    with pytest.raises(ValueError, match='other descriptor settings') as caught:
        check_descriptor_record(_record_params(), str(run))
    assert 'descriptors_series = DESCRIPTORS.series' in str(caught.value)
    assert 'migrate_descriptors' in str(caught.value)
    # a series of its own needs no record; a plain descriptors function cannot be recorded
    assert check_descriptor_record(_record_params(descriptors_series='descriptors-cgx'),
                                   str(tmp_path / 'other')) is None
    assert check_descriptor_record(cgaimmd_params_namespace(), str(tmp_path / 'plain')) is None
    assert not os.path.exists(tmp_path / 'other') and not os.path.exists(tmp_path / 'plain')


def test_cluster_caches_without_a_record_are_refused(tmp_path, generator_descriptors):
    """A cluster run whose 'descriptors' caches predate the record (possibly the placement of
    the development branch) is refused with the ways out; without caches the record is
    written; the solute alone continues and gets its record."""
    from aimmd.cgaimmd.preparation.descriptors import ClusterSelection, GeneratorAtoms
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    generator_descriptors(GeneratorAtoms(n_solute=22, cluster=ClusterSelection(n_waters=4)))
    params = _record_params()
    with pytest.raises(ValueError, match='water placement 1') as caught:
        check_descriptor_record(params, layout.run)
    assert 'refill=True' in str(caught.value)
    assert not os.path.exists(os.path.join(layout.run, restart.DESCRIPTOR_RECORD))
    fresh = tmp_path / 'fresh'
    fresh.mkdir()
    assert check_descriptor_record(params, str(fresh))['spec']['generator_atoms']['cluster'][
        'water_placement'] == 2
    # migrating the old caches lets the cluster run continue
    migrate_descriptors(params, layout.run)
    assert check_descriptor_record(params, layout.run) is not None
    # the solute alone: the caches of an older version continue
    vacuum = build_cgaimmd_run_dir(tmp_path / 'vacuum', n_chains=1, n_shots=1)
    generator_descriptors(GeneratorAtoms(n_solute=22))
    assert check_descriptor_record(_record_params(), vacuum.run) is not None
    assert os.path.isfile(os.path.join(vacuum.run, restart.DESCRIPTOR_RECORD))


def test_migrate_sets_the_record_aside(tmp_path, generator_descriptors):
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1)
    generator_descriptors(GeneratorAtoms(n_solute=22))
    params = _record_params()
    check_descriptor_record(params, layout.run)
    report = migrate_descriptors(params, layout.run)
    assert report['record'] == os.path.join(layout.run, restart.DESCRIPTOR_RECORD)
    assert not os.path.exists(report['record'])
    assert os.path.isfile(os.path.join(layout.run, 'cgaimmd_descriptors.pre_cgaimmd.json'))


# ---------------------------------------------------------------------------
# every worker task records the settings before it writes caches
# ---------------------------------------------------------------------------

def _run_worker(tmp_path, params, run):
    from aimmd.worker._run import WorkerRun

    class Worker(WorkerRun):
        def __init__(self):
            self.directory = self._directory = str(run)
            self.params = params
            self.log_file = self.original_stdout = 'stdout'
            self.localid = 0
            self.ran = []
            self.termination_signal = 0

        def _bind_resources(self):
            pass

        def _update_stop_condition(self, **kwargs):
            pass

        def _terminate_operations(self):
            pass

        def _reset_stop_condition(self):
            pass

        def _free(self, *args, **kwargs):
            self.ran.append('free')

        def _shoot(self, *args, **kwargs):
            self.ran.append('shoot')

    return Worker()


def test_every_worker_task_records_the_descriptor_settings(tmp_path, generator_descriptors):
    """The first task of a new cluster run writes the record before it writes any cache, so
    that a later start never finds caches without one; a run whose caches predate the record
    is refused before the task runs."""
    from aimmd.cgaimmd.preparation.descriptors import ClusterSelection, GeneratorAtoms
    generator_descriptors(GeneratorAtoms(n_solute=22, cluster=ClusterSelection(n_waters=4)))
    params = _record_params(path=tmp_path)
    run = tmp_path / 'new_run'
    worker = _run_worker(tmp_path, params, run)
    worker.run('free')
    assert worker.ran == ['free']
    assert (run / restart.DESCRIPTOR_RECORD).is_file()
    old = build_cgaimmd_run_dir(tmp_path / 'old', n_chains=1, n_shots=1)
    worker = _run_worker(tmp_path, params, old.run)
    with pytest.raises(ValueError, match='water placement 1'):
        worker.run('shoot')
    assert worker.ran == []
    rfps = _run_worker(tmp_path, cgaimmd_params_namespace(chain_type='rfps',
                                                          descriptors_function=wrapped_descriptors,
                                                          path=tmp_path), old.run)
    rfps.run('shoot')                                         # other chain types: nothing
    assert rfps.ran == ['shoot']
    assert not os.path.exists(os.path.join(old.run, restart.DESCRIPTOR_RECORD))


def test_launcher_records_the_settings_of_a_new_run(tmp_path, generator_descriptors):
    """The Launcher's check writes the record of a run folder that does not exist yet."""
    from aimmd.cgaimmd.preparation.descriptors import ClusterSelection, GeneratorAtoms
    generator_descriptors(GeneratorAtoms(n_solute=22, cluster=ClusterSelection(n_waters=4)))
    run = tmp_path / 'not_yet'
    check_cgaimmd_restart(_record_params(), str(run), descriptors=True)
    assert (run / restart.DESCRIPTOR_RECORD).is_file()
