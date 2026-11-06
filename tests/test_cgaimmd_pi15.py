"""Post-training a C-GAIMMD model on an RFPS run, and RFPS -> C-GAIMMD restarts, end to end.

The PI's decision 15 (29 Sep): no hybrid C-GAIMMD/RFPS mixing within a run,
but post-training a C-GAIMMD model with a trainer on an existing RFPS (or TPS)
run folder must be possible, and continuing as ``chain_type='cgaimmd'`` from a
run that was something else must be possible when a restart changes the
params.

The source is a small, real RFPS run of ``tests/toy_1d`` (its engine, states,
network and fit; free simulations, three shots and the trainer's rounds, in
this process) whose ``descriptors_function`` is another network's,
``(x, x**2)``: width 2, where the C-GAIMMD network of ``tests/toy_1d_cgaimmd``
needs the flat coordinates of its one atom (width 3). So every recipe goes
through the descriptor checks and their migration.

- :func:`test_posttrain_rfps_folder`: ``derive_run`` and a C-GAIMMD trainer
  under the source's ``chain_type='rfps'``;
- :func:`test_posttrain_rfps_values_follow_device`: the trainer's post-fit
  value pass hands its inputs to the network's device at the time of the call;
- :func:`test_restart_rfps_as_cgaimmd`: the stale caches refused, migrated,
  and the run continued as ``'cgaimmd'``;
- :func:`test_restart_tps_as_cgaimmd_refused`: a run with TPS history is
  refused by every entry point (``TPS_RESTART_AS_CGAIMMD = False``).

Each test with a fit runs with the stand-in ``round_contract`` fit and with the
flow-matching fit (``tests/_cgaimmd_toy_campaign.py``). Only the C-GAIMMD
contract is asserted.
The tests are ``slow`` (the RFPS source takes about half a minute).
"""

import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.io import file_sha256
from aimmd.cgaimmd.restart import check_cgaimmd_restart, derive_run, migrate_descriptors
from aimmd.path.utils import get_cache_fname
from tests._cgaimmd_toy_campaign import (FIT_KINDS, FLOW_MATCHING_FIT, ROUND_CONTRACT_FIT,
                                         fit_kind)
from tests._helpers_unit import write_trajectory


pytestmark = pytest.mark.slow

TESTS = Path(__file__).resolve().parent
STATES = 'ARB'
# the C-GAIMMD shooting worker names the R frames of the initial paths it does not use (the
# RFPS source's initial transition of tests/toy_1d has 5)
R_FRAME_WARNING = (r'WARNING: C-GAIMMD does not shoot from initial paths: their 5 frames in '
                   r'state R \([^)]*run/initialARB/initial\.xtc\) are not used')

SOURCE_DESCRIPTORS = '''

def descriptors_function(trajectory):
    """Another network's descriptors: (x, x**2) of the one atom (width 2; C-GAIMMD's is 3)."""
    x = np.array([ts.positions[0, 0] for ts in trajectory], dtype=np.float32)
    return np.stack([x, x ** 2], axis=1)
'''


@pytest.fixture(scope='module')
def short_sys_path_module(tmp_path_factory):
    """``sys.path`` without the folders earlier tests' ``Params.load`` left in it, per module."""
    base = str(tmp_path_factory.getbasetemp())
    kept, seen = [], set()
    for entry in sys.path:
        if entry not in seen and not entry.startswith(base):
            seen.add(entry)
            kept.append(entry)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, 'path', kept)
        yield


@pytest.fixture(scope='module')
def restore_signals_module():
    """``aimmd.Worker`` installs SIGINT/SIGTERM handlers; give pytest its own back."""
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture
def restore_signals():
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture(params=FIT_KINDS)
def kind(request):
    return fit_kind(request)


def snapshot(folder):
    """{relative path: (size, mtime_ns)} of every file under `folder` (links not followed)."""
    out = {}
    for root, dirs, files in os.walk(folder):
        for name in files:
            fname = os.path.join(root, name)
            stat = os.lstat(fname)
            out[os.path.relpath(fname, folder)] = (stat.st_size, stat.st_mtime_ns)
    return out


def trajectories(run):
    """Every trajectory file of the chain and free folders of `run`."""
    return sorted(glob.glob(os.path.join(run, 'chain*', 'path??????.xtc')) +
                  glob.glob(os.path.join(run, 'free*', '*.xtc')))


@pytest.fixture(scope='module')
def rfps_source(tmp_path_factory, restore_signals_module, short_sys_path_module):
    """A finished RFPS run of tests/toy_1d with (x, x**2) descriptors: {'folder', 'run', 'params'}.

    Built once per module, with the working directory set to its folder, as a
    Worker runs in ``params.parent``. Tests that change the run work on a copy.
    """
    folder = tmp_path_factory.mktemp('rfps_source')
    text = (TESTS / 'toy_1d' / 'params.py').read_text() + SOURCE_DESCRIPTORS
    (folder / 'params.py').write_text(text)
    shutil.copy(TESTS / 'toy_1d' / 'initial.xtc', folder / 'initial.xtc')
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        np.random.seed(1)
        torch.manual_seed(1)
        params = aimmd.Params.load(str(folder / 'params.py'), save=False)
        assert params.chain_type == 'rfps'
        run = str(folder / 'run')
        aimmd.Worker(params, run, walltime=600).train(nrounds=1)
        for state in (0, 2):
            aimmd.Worker(params, run, nframes=150, walltime=600).free(state, 0, 1)
        aimmd.Worker(params, run, nsteps=3, walltime=600).shoot(1, 0)
        aimmd.Worker(params, run, walltime=600).train(nrounds=1)
    finally:
        os.chdir(cwd)
    assert len(params.shot_chains(run, 'R', 0)) == 3
    for fname in trajectories(run):
        assert np.load(get_cache_fname(fname, 'descriptors')).shape[1] == 2
    return {'folder': folder, 'run': run, 'params': params}


def cgaimmd_params(folder, chain_type, kind):
    """The C-GAIMMD params of tests/toy_1d_cgaimmd under `chain_type`, with the fit `kind`.

    With per-basin initial files and a seeded ``TinyFlowNetwork()`` as the
    ``initial_network`` (the pretrained network of the recipes).
    """
    folder = Path(folder)
    folder.mkdir(parents=True)
    text = (TESTS / 'toy_1d_cgaimmd' / 'params.py').read_text()
    assert "chain_type = 'cgaimmd'" in text
    text = text.replace("chain_type = 'cgaimmd'", f'chain_type = {chain_type!r}')
    text += {'round_contract': ROUND_CONTRACT_FIT, 'flow_matching': FLOW_MATCHING_FIT}[kind]
    (folder / 'params.py').write_text(text)
    for state, x in (('A', 0.9), ('B', 2.1)):
        xs = x + 0.03 * np.sin(np.arange(10))
        positions = np.repeat(xs[:, None, None], 3, axis=2).astype(np.float32)
        write_trajectory(folder, stem=f'initial_{state}', positions=positions)
    torch.manual_seed(2)
    torch.save(TinyFlowNetwork().state_dict(), folder / 'initial_network.h5')
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert params.chain_type == chain_type
    return params


def network_record(run):
    record = json.loads(Path(run, f'network{STATES}.json').read_text())
    assert record['sha256'] == file_sha256(os.path.join(run, f'network{STATES}.h5'))
    return record


def test_posttrain_rfps_folder(tmp_path, monkeypatch, capfd, rfps_source, restore_signals,
                               kind):
    """derive_run, then a C-GAIMMD trainer on the derived folder under the source's chain type."""
    source = rfps_source['run']
    before = snapshot(source)
    target = str(tmp_path / 'post')
    assert derive_run(source, target)['linked'] > 0
    params = cgaimmd_params(tmp_path / 'params_cgaimmd', 'rfps', kind)
    monkeypatch.chdir(params.parent)
    initial_sha = file_sha256(Path(params.parent, 'initial_network.h5'))

    aimmd.Worker(params, target, walltime=600).train(nrounds=1)
    first = network_record(target)
    aimmd.Worker(params, target, walltime=600).train(nrounds=1)
    record = network_record(target)

    assert snapshot(source) == before                      # no file of the source changed
    for fname in trajectories(target):                     # descriptors recomputed at width 3
        cache = get_cache_fname(fname, 'descriptors')
        assert not os.path.islink(cache)
        assert np.load(cache).shape[1] == 3, fname
    assert record['round'] > first['round']
    assert record['n_trained_shots'] == 3                  # the source's three shots
    assert record['sha256'] != initial_sha
    # derive_run copies no network: the initial network went into an empty slot
    assert not os.path.exists(os.path.join(target, f'network{STATES}.replaced0.h5'))
    assert 'reweighting failed' not in capfd.readouterr().out


def test_posttrain_rfps_values_follow_device(tmp_path, monkeypatch, rfps_source,
                                             restore_signals):
    """The trainer's post-fit value pass hands its inputs to the network's device of that time.

    AIMMD's default values function fixes its device when it is created (at
    params load, on the CPU); a C-GAIMMD fit may move the network to a GPU.
    Here the network reports another device (``meta``) once the fit has
    returned: the value pass that follows must send its inputs there.
    """
    from aimmd.cgaimmd import values as cgaimmd_values
    target = str(tmp_path / 'post')
    derive_run(rfps_source['run'], target)
    params = cgaimmd_params(tmp_path / 'params_cgaimmd', 'rfps', 'round_contract')
    monkeypatch.chdir(params.parent)
    assert isinstance(params.values_function, cgaimmd_values.ProtocolValuesFunction)
    fitted = []
    real_fit = params.fit

    def fit_then_report_another_device(*args, **kwargs):
        result = real_fit(*args, **kwargs)
        fitted.append(True)
        return result

    params.__dict__['fit'] = fit_then_report_another_device
    real_device = cgaimmd_values.network_device
    other = torch.device('meta')
    monkeypatch.setattr(cgaimmd_values, 'network_device',
                        lambda network: other if fitted else real_device(network))
    seen = []
    real_input = cgaimmd_values._values_input

    def values_input(rows, device, transform):
        # the device the values function hands its inputs to; a meta tensor holds no data, so
        # the network itself is still given the CPU input
        seen.append((bool(fitted), torch.device(device)))
        return real_input(rows, torch.device('cpu') if device == other else device, transform)

    monkeypatch.setattr(cgaimmd_values, '_values_input', values_input)
    aimmd.Worker(params, target, walltime=600).train(nrounds=1)
    assert fitted
    assert {device for after, device in seen if not after} == {torch.device('cpu')}
    assert {device for after, device in seen if after} == {other}


def test_restart_rfps_as_cgaimmd(tmp_path, monkeypatch, capfd, rfps_source, restore_signals,
                                 kind):
    """An RFPS run continued as 'cgaimmd' after migrating its descriptor caches."""
    run = str(tmp_path / 'run')
    shutil.copytree(rfps_source['run'], run, symlinks=True)
    params_rfps = rfps_source['params']
    old_paths = sorted(glob.glob(os.path.join(run, 'chainR0', 'path??????.xtc')))
    weights_before = [p.weight for p in params_rfps.shot_chains(run, 'R', 0)]
    rfps_network_sha = file_sha256(os.path.join(run, f'network{STATES}.h5'))
    params = cgaimmd_params(tmp_path / 'params_cgaimmd', 'cgaimmd', kind)
    monkeypatch.chdir(params.parent)

    with pytest.raises(ValueError, match='migrate_descriptors'):
        aimmd.Worker(params, run, walltime=60).train(nrounds=1)    # the trainer's own check
    assert file_sha256(os.path.join(run, f'network{STATES}.h5')) == rfps_network_sha
    assert migrate_descriptors(params, run)['renamed'] > 0
    assert glob.glob(os.path.join(run, 'chainR0', '*.descriptors.pre_cgaimmd.npy'))
    check_cgaimmd_restart(params, run, descriptors=True)        # nothing stale is left

    aimmd.Worker(params, run, walltime=600).train(nrounds=1)
    replaced = os.path.join(run, f'network{STATES}.replaced0.h5')
    assert file_sha256(replaced) == rfps_network_sha              # the RFPS network, moved aside
    assert network_record(run)['n_trained_shots'] == 3
    aimmd.Worker(params, run, nsteps=len(old_paths) + 1, walltime=600).shoot(1, 0)
    out = capfd.readouterr().out
    # the RFPS initial transition stays in the run; C-GAIMMD shoots from none of its frames
    assert re.search(R_FRAME_WARNING, out), [line for line in out.splitlines()
                                             if 'initial' in line]
    paths = sorted(glob.glob(os.path.join(run, 'chainR0', 'path??????.xtc')))
    assert paths[:-1] == old_paths and len(paths) == len(old_paths) + 1
    with_sidecar = [p for p in paths if os.path.exists(f'{p}.cgaimmd.json')]
    assert with_sidecar == paths[-1:]                             # exactly the new shot
    record = json.loads(Path(f'{paths[-1]}.cgaimmd.json').read_text())
    assert record['source'] == 'cgaimmd' and record['acceptance']['rule'] == 'bootstrap_any_R'
    for fname in paths:
        assert np.load(get_cache_fname(fname, 'descriptors')).shape[1] == 3
    weights_after = [p.weight for p in params.shot_chains(run, 'R', 0)]
    assert weights_after[:-1] == weights_before                   # RFPS history unchanged
    assert not os.path.exists(os.path.join(run, 'chainR0', 'tps_weights.npy'))


def test_restart_tps_as_cgaimmd_refused(tmp_path, monkeypatch, rfps_source, restore_signals):
    """A run with TPS history does not continue as 'cgaimmd', whatever starts it."""
    run = str(tmp_path / 'run')
    shutil.copytree(rfps_source['run'], run, symlinks=True)
    # one TPS chain next to the RFPS one: a real TPS shot writes chainR1/tps_weights.npy
    tps_folder = tmp_path / 'params_tps'
    tps_folder.mkdir()
    shutil.copy(rfps_source['folder'] / 'initial.xtc', tps_folder / 'initial.xtc')
    text = (rfps_source['folder'] / 'params.py').read_text()
    (tps_folder / 'params.py').write_text(text + "\nchain_type = 'tps'\nselection_pool_size = 1\n")
    monkeypatch.chdir(tps_folder)
    params_tps = aimmd.Params.load(str(tps_folder / 'params.py'), save=False)
    assert params_tps.chain_type == 'tps'
    aimmd.Worker(params_tps, run, nsteps=1, walltime=600).shoot(1, 1)
    assert os.path.isfile(os.path.join(run, 'chainR1', 'tps_weights.npy'))

    params = cgaimmd_params(tmp_path / 'params_cgaimmd', 'cgaimmd', 'round_contract')
    monkeypatch.chdir(params.parent)
    migrate_descriptors(params, run)                         # not the reason for the refusal
    before = snapshot(run)
    network_sha = file_sha256(os.path.join(run, f'network{STATES}.h5'))
    refused = 'TPS_RESTART_AS_CGAIMMD'
    with pytest.raises(NotImplementedError, match=refused):
        check_cgaimmd_restart(params, run)
    with pytest.raises(NotImplementedError, match=refused):
        aimmd.Worker(params, run, walltime=30).train(nrounds=1)
    with pytest.raises(NotImplementedError, match=refused):
        aimmd.Worker(params, run, nsteps=10, walltime=30).shoot(1, 0)
    with pytest.raises(NotImplementedError, match=refused):
        aimmd.Launcher(params, run).run(1, 1, 1, nsteps=1, walltime=30)
    assert not os.path.exists(os.path.join(run, f'network{STATES}.replaced0.h5'))
    assert not glob.glob(os.path.join(run, 'chain*', '*.cgaimmd.json'))
    # nothing written but logs and the readers' own hidden files (locks, MDAnalysis offsets)
    changed = {name for name, stat in snapshot(run).items() if before.get(name) != stat}
    assert not {name for name in changed
                if not name.endswith('.log') and not os.path.basename(name).startswith('.')}
    assert file_sha256(os.path.join(run, f'network{STATES}.h5')) == network_sha
