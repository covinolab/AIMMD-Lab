"""The Launcher's run-folder checks and its ``initial_network`` write.

Before any process is built (and before ``initial<s>/`` is rewritten), for
every run that will start: under ``chain_type='cgaimmd'`` the restart check
(TPS history, stale descriptor caches), the transition requirement of
internal shooting chains, the initial network and the missing-network check
of a sampling-only launch with a C-GAIMMD shooter; under every chain type the
``initial_network`` write. The lazy build of ``Worker.initial_paths`` inside a
worker task (``aimmd._config.WORKER_TASK`` set) does none of it.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.launcher import Launcher
from aimmd.path.utils import get_cache_fname
from tests._cgaimmd_helpers import build_cgaimmd_run_dir, cgaimmd_params_namespace
from tests._helpers_unit import build_path


class FlowNet(TinyFlowNetwork):
    """A C-GAIMMD network of a local class."""


def one_atom(x):
    positions = np.zeros((len(x), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = x
    return positions


@pytest.fixture
def launcher_env(tmp_path, monkeypatch):
    """No CPU/GPU discovery and no Params.save (the params are a namespace)."""
    (tmp_path / 'params.py').write_text('# placeholder params file\n')
    monkeypatch.setattr('aimmd.launcher._helpers.get_num_cpus', lambda: 4)
    monkeypatch.setattr('aimmd.launcher._helpers.get_num_gpus', lambda: 0)
    monkeypatch.setattr('aimmd.params._io.ParamsIO.save',
                        lambda self, *args, **kwargs: str(tmp_path / 'params.py'))
    return tmp_path


def cgaimmd_params(tmp_path, *, transitions=False, **overrides):
    if transitions:
        seeds = [build_path(tmp_path, stem='transition',
                            positions=one_atom([-1.0, -0.3, 0.0, 0.3, 1.0]))]
    else:
        seeds = [build_path(tmp_path, stem='seeds_A', positions=one_atom([-1.0, -1.1, -0.9])),
                 build_path(tmp_path, stem='seeds_B', positions=one_atom([1.0, 1.1, 0.9]))]
    overrides.setdefault('network', FlowNet())
    params = cgaimmd_params_namespace(**overrides)
    params.__dict__['initial_paths'] = aimmd.PathEnsemble(seeds)
    params.__dict__['path'] = tmp_path / 'params.py'
    return params


def build(params, run, **update):
    launcher = Launcher([params], [str(run)])
    settings = dict(n=1, n1=1, n2=1, cpus_per_task=1, gpus_per_task=0, ntasks_per_node=1)
    settings.update(update)
    launcher._update(**settings)
    return launcher._build()


def write_network(run, network):
    os.makedirs(run, exist_ok=True)
    torch.save(network.state_dict(), os.path.join(run, 'networkARB.h5'))


# ---------------------------------------------------------------------------
# C-GAIMMD checks
# ---------------------------------------------------------------------------

def test_cgaimmd_launch_builds(launcher_env):
    params = cgaimmd_params(launcher_env)
    run = launcher_env / 'run'
    args, descriptions = build(params, run)
    assert len(args) == 4                                             # shoot, free A, free B, train
    assert (run / 'initialARB').is_dir()


def refused_folder_params(launcher_env, run):
    """C-GAIMMD params with an initial_network, and an incompatible network in `run`.

    A launch that refuses `run` must refuse it before the network write: the
    old network stays where it is and nothing is moved aside or written.
    """
    torch.save(FlowNet().state_dict(), launcher_env / 'pretrained.h5')
    write_network(run, TinyFlowNetwork(hidden=4))
    old = torch.load(os.path.join(run, 'networkARB.h5'), weights_only=True)
    return cgaimmd_params(launcher_env, initial_network=str(launcher_env / 'pretrained.h5')), old


def assert_network_untouched(run, old):
    loaded = torch.load(os.path.join(run, 'networkARB.h5'), weights_only=True)
    assert loaded.keys() == old.keys() and all(torch.equal(loaded[k], old[k]) for k in old)
    assert not os.path.exists(os.path.join(run, 'networkARB.json'))
    assert not os.path.exists(os.path.join(run, 'networkARB.replaced0.h5'))


def test_launch_checks_before_processes_stale_caches(launcher_env):
    layout = build_cgaimmd_run_dir(launcher_env)
    fname = get_cache_fname(layout.shots[0].fname, 'descriptors')
    save_npy(fname, np.zeros((7, 2), dtype=np.float32))              # another network's width
    params, old = refused_folder_params(launcher_env, layout.run)
    with pytest.raises(ValueError, match='migrate_descriptors'):
        build(params, layout.run)
    assert not os.path.exists(os.path.join(layout.run, 'initialARB'))  # nothing written yet
    assert_network_untouched(layout.run, old)                         # a refused folder gets no network


def test_launch_checks_before_processes_tps_history(launcher_env):
    layout = build_cgaimmd_run_dir(launcher_env)
    save_npy(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'), np.ones(3))
    params, old = refused_folder_params(launcher_env, layout.run)
    with pytest.raises(NotImplementedError, match='TPS_RESTART_AS_CGAIMMD'):
        build(params, layout.run)
    assert not os.path.exists(os.path.join(layout.run, 'initialARB'))
    assert_network_untouched(layout.run, old)


def test_trainer_only_launch_is_checked(launcher_env):
    """The trainer-only launch of the post-training recipe (run(0, 0, 0, nrounds=K)) starts a process too."""
    layout = build_cgaimmd_run_dir(launcher_env)
    save_npy(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'), np.ones(3))
    params, old = refused_folder_params(launcher_env, layout.run)
    with pytest.raises(NotImplementedError, match='TPS_RESTART_AS_CGAIMMD'):
        build(params, layout.run, n=0, n1=0, n2=0, nrounds=1)
    assert_network_untouched(layout.run, old)
    os.remove(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'))
    args, descriptions = build(params, layout.run, n=0, n1=0, n2=0, nrounds=1)
    assert descriptions == [f'"{layout.run}" ARB trainer']
    assert json.loads((Path(layout.run) / 'networkARB.json').read_text())['source'] == \
        'initial_network'
    assert os.path.isfile(os.path.join(layout.run, 'networkARB.replaced0.h5'))


def test_lazy_build_in_a_worker_task_skips_the_checks(launcher_env, monkeypatch):
    """Worker.initial_paths' lazy _build inside a task writes initial<s>/ and nothing else.

    The folder would be refused by a launch (TPS history) and its network
    replaced (initial_network set, incompatible network): inside a worker
    task neither happens, so concurrent direct workers never move the same
    network aside; the trainer and the C-GAIMMD shooting worker check for
    themselves.
    """
    from aimmd import _config
    import aimmd.cgaimmd.restart as restart
    layout = build_cgaimmd_run_dir(launcher_env)
    save_npy(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'), np.ones(3))
    params, old = refused_folder_params(launcher_env, layout.run)
    checks = []
    real_check = restart.check_cgaimmd_restart

    def spy(*args, **kwargs):
        checks.append(args)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(restart, 'check_cgaimmd_restart', spy)
    initial = os.path.join(layout.run, 'initialARB')
    for task in ('free', 'shoot', 'train'):
        monkeypatch.setattr(_config, 'WORKER_TASK', task)
        # as Worker.initial_paths builds: n=0 (nrounds then becomes 1)
        launcher = Launcher([params], [str(layout.run)])
        launcher._update(n=0, cpus_per_task=1, gpus_per_task=0, ntasks_per_node=1)
        launcher._build()
        assert checks == []
        assert_network_untouched(layout.run, old)
        assert sorted(name for name in os.listdir(initial)
                      if name.endswith('.xtc')) == ['seeds_A.xtc', 'seeds_B.xtc']
        shutil.rmtree(initial)
    monkeypatch.setattr(_config, 'WORKER_TASK', None)                # a launch checks
    with pytest.raises(NotImplementedError, match='TPS_RESTART_AS_CGAIMMD'):
        build(params, layout.run)
    assert len(checks) == 1


def test_rfps_launch_not_checked(launcher_env):
    layout = build_cgaimmd_run_dir(launcher_env)
    save_npy(os.path.join(layout.run, 'chainR0', 'tps_weights.npy'), np.ones(3))
    params = cgaimmd_params(launcher_env, chain_type='rfps')
    args, _ = build(params, layout.run)
    assert args


def test_internal_shooting_needs_transitions(launcher_env):
    params = cgaimmd_params(launcher_env)
    run = launcher_env / 'run'
    with pytest.raises(TypeError, match=r"state1_mode.*'shoot'.*seeds_A.*seeds_B"):
        build(params, run, state1_mode='shoot')
    with pytest.raises(TypeError, match="state2_mode"):
        build(params, run, state2_mode='shoot')
    other = launcher_env / 'with_transition'
    other.mkdir()
    (other / 'params.py').write_text('# placeholder\n')
    params = cgaimmd_params(other, transitions=True)
    args, descriptions = build(params, launcher_env / 'run2', state1_mode='shoot')
    assert any('chainA0' in description for description in descriptions)


def test_nrounds0_needs_network(launcher_env):
    params = cgaimmd_params(launcher_env)
    run = launcher_env / 'run'
    with pytest.raises(RuntimeError, match="chain_type='cgaimmd' with nrounds=0 needs "
                                           "networkARB.h5 or initial_network"):
        build(params, run, nrounds=0)
    write_network(run, params.network)
    args, _ = build(params, run, nrounds=0)
    assert len(args) == 3                                             # no trainer


@pytest.mark.parametrize('launch', [
    dict(n=0, n1=1, n2=1),                                           # free workers only
    dict(n=1, n1=0, n2=0, reactive_region_mode='sweep'),              # a committor sweep
    dict(n=1, n1=1, n2=1, reactive_region_mode='free'),              # free R workers
    dict(n=0, n1=1, n2=0, state1_mode='shoot'),                      # internal shooting
], ids=['free-only', 'sweep', 'free-reactive-region', 'internal-shooting'])
def test_nrounds0_without_cgaimmd_shooter(launcher_env, launch):
    """Only a C-GAIMMD shooter waits for the network: other sampling-only launches need none."""
    params = cgaimmd_params(launcher_env, transitions=launch.get('state1_mode') == 'shoot')
    run = launcher_env / 'run'
    args, descriptions = build(params, run, nrounds=0, **launch)
    assert args and not any('trainer' in description for description in descriptions)
    assert not (run / 'networkARB.h5').exists()
    with pytest.raises(RuntimeError, match='nrounds=0 needs'):     # a C-GAIMMD shooter does
        build(params, run, nrounds=0, **{**launch, 'n': 1, 'reactive_region_mode': 'chain'})


def test_nrounds0_with_initial_network(launcher_env):
    torch.save(FlowNet().state_dict(), launcher_env / 'pretrained.h5')
    params = cgaimmd_params(launcher_env, initial_network=str(launcher_env / 'pretrained.h5'))
    run = launcher_env / 'run'
    build(params, run, nrounds=0)
    assert (run / 'networkARB.h5').is_file()


def test_incompatible_network_without_initial_network(launcher_env):
    """An RFPS network left in a folder relaunched as C-GAIMMD: RuntimeError at launch."""
    params = cgaimmd_params(launcher_env)
    run = launcher_env / 'run'
    write_network(run, TinyFlowNetwork(hidden=4))
    with pytest.raises(RuntimeError, match='initial_network'):
        build(params, run)


@pytest.mark.parametrize('chain_type', ['rfps', 'cgaimmd'])
def test_launcher_writes_initial_network(launcher_env, chain_type):
    source = FlowNet()
    torch.save(source.state_dict(), launcher_env / 'pretrained.h5')
    params = cgaimmd_params(launcher_env, chain_type=chain_type,
                         initial_network=str(launcher_env / 'pretrained.h5'))
    run = launcher_env / 'run'
    write_network(run, TinyFlowNetwork(hidden=4))                    # incompatible: moved aside
    build(params, run)
    loaded = torch.load(run / 'networkARB.h5', weights_only=True)
    assert all(torch.equal(loaded[k], v) for k, v in source.state_dict().items())
    assert json.loads((run / 'networkARB.json').read_text())['source'] == 'initial_network'
    assert (run / 'networkARB.replaced0.h5').is_file()


def test_skipped_run_is_not_touched(launcher_env):
    """A run with nothing to do (no workers, no rounds) is not checked or written."""
    torch.save(FlowNet().state_dict(), launcher_env / 'pretrained.h5')
    params = cgaimmd_params(launcher_env, initial_network=str(launcher_env / 'pretrained.h5'))
    active = cgaimmd_params(launcher_env, chain_type='rfps')
    runs = [launcher_env / 'idle', launcher_env / 'active']
    launcher = Launcher([params, active], [str(run) for run in runs])
    launcher._update(n=[0, 1], n1=0, n2=0, nrounds=[0, 1], cpus_per_task=1, gpus_per_task=0,
                     ntasks_per_node=1)
    launcher._build()
    assert not (runs[0] / 'networkARB.h5').exists()
