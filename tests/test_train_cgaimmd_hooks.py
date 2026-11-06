"""The trainer's C-GAIMMD hooks: ``aimmd.worker._train`` and
``aimmd.worker.utils.{save_network_atomically, write_initial_network}``.

The trainer writes an ``initial_network`` before its first round, checks a
C-GAIMMD restart before its first value pass, accepts a fit that returns
``(losses, info)``, says why a round was skipped, saves the network
atomically, with ``network<s>.json`` written before ``network<s>.h5`` when the
fit gives network metadata, and saves the network of a round a stop
interrupted as ``network<s>.final.h5``. The fits are faked (functions
returning ``(losses, info)``, as the C-GAIMMD fits do); the trainer loop is
real and runs on a stand-in path ensemble (as
``tests/test_worker_runtime_unit.py`` does). The round counter of the
BioEmu2 fit reads the json the trainer writes. One test runs the real
flow-matching fit (``aimmd.cgaimmd.flow_matching.fit``) under the real trainer.
"""

import json
import os

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd.io import file_sha256
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.worker._train import WorkerTrain
from aimmd.worker.utils import save_network_atomically, write_initial_network
from tests._cgaimmd_helpers import cgaimmd_params_namespace
from tests._helpers_unit import TinyNetwork, build_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FlowNet(TinyFlowNetwork):
    """A C-GAIMMD network of a local class."""


class OtherFlowNet(TinyFlowNetwork):
    """Another architecture: its state dict does not load into FlowNet(hidden=16)."""

    def __init__(self):
        super().__init__(hidden=4)


class TinyTrainWorker(WorkerTrain):
    """Concrete host for `_train` (as in tests/test_worker_runtime_unit.py)."""

    def __init__(self, params, initial_paths, root):
        self.params = params
        self.initial_paths = initial_paths
        self._directory = str(root)
        self.directory = str(root)
        self.termination_signal = 0
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self.task = 'train'


class FakeEnsemble:
    """Pathensemble-like object with the exact API `_train` needs; records its calls."""

    def __init__(self, fname, events):
        self.weights = np.array([1.0])
        self.fnames = np.array([fname])
        self.n_frames = np.array([3])
        self.events = events

    def __len__(self):
        return 1

    def __add__(self, other):
        return self

    def compute(self, *args, **kwargs):
        self.events.append(('compute', kwargs.get('target', args[1] if len(args) > 1 else None)))
        return 1

    def reweight(self, *args, **kwargs):
        return (np.array([1.0]), None, None, None, np.array([-1.0, 1.0]), np.array([0.2, 0.8]))

    def project(self, bins, source='values'):
        return np.array([2.0, 1.0], dtype=float)

    def types(self, pattern=None):
        return np.array([True])


def make_trainer(tmp_path, monkeypatch, fit, *, network=None, chain_type='cgaimmd', **overrides):
    """A TinyTrainWorker on `tmp_path/run` whose fit is `fit` and whose ensemble is fake."""
    run = tmp_path / 'run'
    run.mkdir(exist_ok=True)
    initial = build_path(tmp_path, stem='train_initial',
                         positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                                            dtype=np.float32))
    events = []
    ensemble = FakeEnsemble(initial.fname, events)
    network = FlowNet() if network is None else network
    params = cgaimmd_params_namespace(
        network=network, chain_type=chain_type, fit=fit, nbins=2, network_save_interval=1,
        shot_chains=lambda directory, target_state=None, old=None: [ensemble],
        free_trajectories=lambda directory, old=None: [], **overrides)
    worker = TinyTrainWorker(params, aimmd.PathEnsemble(initial), run)
    monkeypatch.setattr('aimmd.worker._train.assemble_pathensemble',
                        lambda *args, **kwargs: ensemble)
    monkeypatch.setattr('aimmd.worker._train.compute_bins',
                        lambda *args, **kwargs: np.array([-np.inf, 0.0, np.inf]))
    monkeypatch.setattr('aimmd.worker._train.replace_in_cache', lambda *args, **kwargs: None)
    return worker, events, str(run)


def metadata(round=1, n_trained_shots=5):
    return {'round': round, 'n_trained_shots': n_trained_shots, 'step': 10 * round,
            'fit': 'tests.fake_fit', 'seed': 7}


def fit_info(**meta):
    return {'status': 'trained', 'reason': '', 'round': meta.get('round', 1), 'record': {},
            'network_metadata': metadata(**meta)}


def state_dict_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# ---------------------------------------------------------------------------
# save_network_atomically
# ---------------------------------------------------------------------------

def test_save_network_atomically_without_metadata(tmp_path):
    fname = str(tmp_path / 'networkARB.h5')
    network = FlowNet()
    sha = save_network_atomically(network.state_dict(), fname)
    assert sha == file_sha256(fname)
    assert not os.path.exists(tmp_path / 'networkARB.json')
    loaded = torch.load(fname, weights_only=True)
    assert state_dict_equal(loaded, network.state_dict())
    assert [name for name in os.listdir(tmp_path) if '.tmp' in name] == []


def test_save_network_atomically_writes_json_first(tmp_path, monkeypatch):
    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst, *args, **kwargs):
        replaced.append(os.path.basename(os.fspath(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', recording_replace)
    fname = str(tmp_path / 'networkARB.h5')
    sha = save_network_atomically(FlowNet().state_dict(), fname,
                                  metadata={**metadata(), 'total_steps': 7})
    assert replaced == ['networkARB.json', 'networkARB.h5']
    record = json.loads((tmp_path / 'networkARB.json').read_text())
    assert record['sha256'] == sha == file_sha256(fname)
    assert record['total_steps'] == 7 and record['round'] == 1
    assert record['n_trained_shots'] == 5 and 'saved' in record


# ---------------------------------------------------------------------------
# write_initial_network
# ---------------------------------------------------------------------------

def initial_network_file(tmp_path, network=None, name='pretrained.h5'):
    network = FlowNet() if network is None else network
    fname = tmp_path / name
    torch.save(network.state_dict(), fname)
    return str(fname), network


def test_write_initial_network_unset(tmp_path):
    params = cgaimmd_params_namespace(network=FlowNet(), chain_type='rfps')
    assert write_initial_network(params, str(tmp_path)) == 'unset'
    assert not os.path.exists(tmp_path / 'networkARB.h5')


def test_initial_network_written(tmp_path, capsys):
    fname, source = initial_network_file(tmp_path)
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    run = tmp_path / 'run'
    run.mkdir()
    assert write_initial_network(params, str(run)) == 'written'
    assert state_dict_equal(params.network.state_dict(), source.state_dict())
    record = json.loads((run / 'networkARB.json').read_text())
    assert record['sha256'] == file_sha256(run / 'networkARB.h5')
    assert record['round'] == 0 and record['n_trained_shots'] == 0
    assert record['source'] == 'initial_network'
    assert record['initial_network'] == os.path.abspath(fname)
    assert f'Writing initial network {fname} to {run}/networkARB.h5' in capsys.readouterr().out


def test_existing_network_kept(tmp_path):
    fname, _ = initial_network_file(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    existing = FlowNet()
    torch.save(existing.state_dict(), run / 'networkARB.h5')
    sha = file_sha256(run / 'networkARB.h5')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    assert write_initial_network(params, str(run)) == 'kept'
    assert file_sha256(run / 'networkARB.h5') == sha
    assert not os.path.exists(run / 'networkARB.json')


def test_incompatible_network_moved_aside(tmp_path, capsys):
    fname, source = initial_network_file(tmp_path)
    run = tmp_path / 'run'
    (run / 'flow_fitARB').mkdir(parents=True)
    (run / 'flow_fitARB' / 'state.pt').write_bytes(b'state')
    torch.save(OtherFlowNet().state_dict(), run / 'networkARB.h5')
    (run / 'networkARB.json').write_text('{"round": 3}')
    old_sha = file_sha256(run / 'networkARB.h5')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    assert write_initial_network(params, str(run)) == 'written'
    assert file_sha256(run / 'networkARB.replaced0.h5') == old_sha
    assert json.loads((run / 'networkARB.replaced0.json').read_text()) == {'round': 3}
    assert (run / 'flow_fitARB.replaced0' / 'state.pt').read_bytes() == b'state'
    assert not (run / 'flow_fitARB').exists()
    assert state_dict_equal(torch.load(run / 'networkARB.h5', weights_only=True),
                            source.state_dict())
    out = capsys.readouterr().out
    assert (f'WARNING: {run}/networkARB.h5 does not load into params.network (' in out)
    assert (f'moved to {run}/networkARB.replaced0.h5 (and its fit state to '
            f'{run}/flow_fitARB.replaced0) and replaced by initial_network {fname}') in out
    # the next incompatible file takes the next free k, with the json next to it (here the
    # initial network's, written above); there is no fit state to move this time
    torch.save(OtherFlowNet().state_dict(), run / 'networkARB.h5')
    assert write_initial_network(params, str(run)) == 'written'
    assert (run / 'networkARB.replaced1.h5').exists()
    assert json.loads((run / 'networkARB.replaced1.json').read_text())['round'] == 0
    assert not (run / 'flow_fitARB.replaced1').exists()
    assert 'and its fit state' not in capsys.readouterr().out


def test_read_error_is_not_an_incompatible_network(tmp_path, monkeypatch):
    """An OSError reading network<s>.h5 propagates: the trained network is not moved aside."""
    fname, _ = initial_network_file(tmp_path)
    run = tmp_path / 'run'
    (run / 'flow_fitARB').mkdir(parents=True)
    torch.save(FlowNet().state_dict(), run / 'networkARB.h5')
    sha = file_sha256(run / 'networkARB.h5')
    real_load = torch.load

    def stale_handle(f, *args, **kwargs):
        if os.fspath(f).endswith('networkARB.h5'):
            raise OSError(116, 'Stale file handle')
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', stale_handle)
    for params in (cgaimmd_params_namespace(network=FlowNet(), initial_network=fname),
                   cgaimmd_params_namespace(network=FlowNet())):
        with pytest.raises(OSError, match='Stale file handle'):
            write_initial_network(params, str(run))
    assert file_sha256(run / 'networkARB.h5') == sha
    assert (run / 'flow_fitARB').is_dir()
    assert not (run / 'networkARB.replaced0.h5').exists()


def test_unreadable_network_file_is_replaced(tmp_path):
    """A file that torch cannot deserialize does not load (case 3): kept aside, then replaced."""
    fname, source = initial_network_file(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'networkARB.h5').write_bytes(b'not a state dict')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    assert write_initial_network(params, str(run)) == 'written'
    assert (run / 'networkARB.replaced0.h5').read_bytes() == b'not a state dict'
    assert state_dict_equal(torch.load(run / 'networkARB.h5', weights_only=True),
                            source.state_dict())


def test_incompatible_network_without_initial_network_under_cgaimmd(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    torch.save(OtherFlowNet().state_dict(), run / 'networkARB.h5')
    params = cgaimmd_params_namespace(network=FlowNet())
    with pytest.raises(RuntimeError, match='initial_network'):
        write_initial_network(params, str(run))
    assert (run / 'networkARB.h5').exists()                          # nothing is moved
    rfps = cgaimmd_params_namespace(network=FlowNet(), chain_type='rfps')
    assert write_initial_network(rfps, str(run)) == 'unset'          # old behaviour elsewhere


def test_initial_network_that_does_not_load_raises(tmp_path):
    fname, _ = initial_network_file(tmp_path, OtherFlowNet())
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    with pytest.raises(RuntimeError):
        write_initial_network(params, str(tmp_path))


def test_initial_network_that_does_not_load_moves_nothing(tmp_path):
    """A bad initial_network is refused before the existing network is moved aside."""
    fname, _ = initial_network_file(tmp_path, OtherFlowNet())
    run = tmp_path / 'run'
    run.mkdir()
    torch.save(TinyFlowNetwork(hidden=8).state_dict(), run / 'networkARB.h5')
    (run / 'networkARB.json').write_text('{"round": 2}')
    sha = file_sha256(run / 'networkARB.h5')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    with pytest.raises(RuntimeError, match='does not load'):
        write_initial_network(params, str(run))
    assert file_sha256(run / 'networkARB.h5') == sha
    assert json.loads((run / 'networkARB.json').read_text()) == {'round': 2}
    assert not (run / 'networkARB.replaced0.h5').exists()


def test_initial_network_multi_system_not_implemented(tmp_path):
    fname, _ = initial_network_file(tmp_path)
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname, multi_system=True)
    with pytest.raises(NotImplementedError, match='multi_system'):
        write_initial_network(params, str(tmp_path))


def test_initial_network_relative_to_params_folder(tmp_path, monkeypatch):
    folder = tmp_path / 'params_dir'
    folder.mkdir()
    fname, _ = initial_network_file(folder, name='rel.h5')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network='rel.h5')
    params.__dict__['path'] = folder
    monkeypatch.chdir(tmp_path)
    assert write_initial_network(params, str(tmp_path / 'run')) == 'written'
    # a params file loaded by a relative name: the record still holds an absolute path
    params.__dict__['path'] = type(folder)('params_dir')
    assert write_initial_network(params, str(tmp_path / 'run2')) == 'written'
    record = json.loads((tmp_path / 'run2' / 'networkARB.json').read_text())
    assert record['initial_network'] == str(folder / 'rel.h5')


def test_replaced_index_skips_an_existing_json(tmp_path):
    """A leftover network<s>.replaced0.json alone takes k = 0: the next free k is 1."""
    fname, _ = initial_network_file(tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'networkARB.replaced0.json').write_text('{"round": 7}')
    torch.save(OtherFlowNet().state_dict(), run / 'networkARB.h5')
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    assert write_initial_network(params, str(run)) == 'written'
    assert (run / 'networkARB.replaced1.h5').is_file()
    assert json.loads((run / 'networkARB.replaced0.json').read_text()) == {'round': 7}


def test_step_backup_of_a_fit_without_metadata_has_no_json(tmp_path, monkeypatch):
    """After an initial_network write (whose json stays), a default fit's step backup gets no json."""
    fname, _ = initial_network_file(tmp_path, TinyNetwork())

    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], [1.0], np.array([0.0]), np.array([1.0]), np.array([[1.0, 0.0]])

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, network=TinyNetwork(),
                                       chain_type='rfps', initial_network=fname)
    worker._train(nrounds=1, keep_running=False)
    assert json.loads(open(f'{run}/networkARB.json').read())['source'] == 'initial_network'
    assert os.path.exists(f'{run}/networkARB.step000001.h5')
    assert not os.path.exists(f'{run}/networkARB.step000001.json')


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------

def test_initial_network_written_before_round1(tmp_path, monkeypatch):
    fname, source = initial_network_file(tmp_path)
    seen = []

    def fit(params, pathensemble, verbose=False, worker=None):
        seen.append({k: v.clone() for k, v in params.network.state_dict().items()})
        return [1.0], fit_info()

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, initial_network=fname)
    worker._train(nrounds=1, keep_running=False)
    assert state_dict_equal(seen[0], source.state_dict())
    assert os.path.exists(f'{run}/networkARB.replaced0.h5') is False


def test_trainer_writes_initial_network_before_loading(tmp_path, monkeypatch):
    order = []
    fname, _ = initial_network_file(tmp_path)

    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], fit_info()

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, initial_network=fname)
    real_update = worker.params.update_network
    monkeypatch.setattr('aimmd.worker._train.write_initial_network',
                        lambda params, directory: order.append(('write', directory)) or 'written')
    worker.params.__dict__['update_network'] = (
        lambda *args, **kwargs: order.append(('update', args[0])) or real_update(*args, **kwargs))
    worker._train(nrounds=1, keep_running=False)
    assert order[:2] == [('write', run), ('update', run)]


def test_atomic_save_writes_json_first(tmp_path, monkeypatch):
    def fit(params, pathensemble, verbose=False, worker=None):
        with torch.no_grad():
            params.network.committor_net[-1].bias.fill_(0.5)
        return [1.0, 0.5], fit_info(round=4, n_trained_shots=12)

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst, *args, **kwargs):
        replaced.append(os.path.basename(os.fspath(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', recording_replace)
    worker._train(nrounds=1, keep_running=False)
    network_files = [name for name in replaced if name.startswith('networkARB.')]
    assert network_files[:2] == ['networkARB.json', 'networkARB.h5']
    record = json.loads(open(f'{run}/networkARB.json').read())
    assert record['sha256'] == file_sha256(f'{run}/networkARB.h5')
    assert record['round'] == 4 and record['n_trained_shots'] == 12
    assert record['total_steps'] == worker.total_steps == 1
    # the step backup copies the .h5 and then its .json
    assert file_sha256(f'{run}/networkARB.step000001.h5') == record['sha256']
    assert json.loads(open(f'{run}/networkARB.step000001.json').read()) == record
    loaded = torch.load(f'{run}/networkARB.h5', weights_only=True)
    assert float(loaded['committor_net.2.bias']) == 0.5


def test_default_fit_tuple_ignored(tmp_path, monkeypatch):
    """The default fit's 5-tuple has no info dict: no metadata, the old save."""
    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], [1.0], np.array([0.0]), np.array([1.0]), np.array([[1.0, 0.0]])

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, network=TinyNetwork(),
                                       chain_type='rfps')
    worker._train(nrounds=1, keep_running=False)
    assert os.path.exists(f'{run}/networkARB.h5')
    assert not os.path.exists(f'{run}/networkARB.json')
    assert os.path.exists(f'{run}/networkARB.step000001.h5')
    assert not os.path.exists(f'{run}/networkARB.step000001.json')


def test_rfps_trainer_still_reloads_after_atomic_save(tmp_path, monkeypatch, capsys):
    calls = []
    network = TinyNetwork()

    def fit(params, pathensemble, verbose=False, worker=None):
        calls.append(1)
        with torch.no_grad():
            params.network.linear.weight.fill_(float(len(calls)))
        if len(calls) == 1:
            return [1.0], [1.0]
        worker.must_stop = True                    # the failed round is the last one
        return [], []

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, network=network,
                                       chain_type='rfps')
    worker._train(nrounds=2, keep_running=False)
    assert len(calls) == 2
    assert '!!! training failed, reloading' in capsys.readouterr().out
    # the reload after the failed round brought back the weights saved after round 1
    np.testing.assert_array_equal(network.linear.weight.detach().numpy(), [[1.0, 1.0]])


def test_skip_message(tmp_path, monkeypatch, capsys):
    reloads = []

    def fit(params, pathensemble, verbose=False, worker=None):
        worker.must_stop = True
        return [], {'status': 'skipped', 'reason': 'no data (0 anchor frames in B)',
                    'round': 1, 'record': {}}

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    real_update = worker.params.update_network
    worker.params.__dict__['update_network'] = (
        lambda *args, **kwargs: reloads.append(args[0]) or real_update(*args, **kwargs))
    worker._train(nrounds=1, keep_running=False)
    out = capsys.readouterr().out
    assert '!!! training skipped: no data (0 anchor frames in B)' in out
    assert 'training failed' not in out
    assert len(reloads) == 2                    # at start, and the reload after the skip
    assert not os.path.exists(f'{run}/networkARB.h5')


def test_info_without_reason_keeps_failed_message(tmp_path, monkeypatch, capsys):
    def fit(params, pathensemble, verbose=False, worker=None):
        worker.must_stop = True
        return [], {'status': 'stopped'}

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    worker._train(nrounds=1, keep_running=False)
    assert '!!! training failed, reloading' in capsys.readouterr().out


def _two_rounds_last_stopped(stop, *, info=None, losses=(1.0, 0.5)):
    """A fit whose round 1 trains and whose round 2 changes the network and is stopped.

    `stop(worker)` signals the stop during round 2 (sets ``must_stop`` or
    ``termination_signal``); `info` is round 2's info (default: a stopped round).
    """
    seen = {}

    def fit(params, pathensemble, verbose=False, worker=None):
        seen['calls'] = seen.get('calls', 0) + 1
        if seen['calls'] == 1:
            return [1.0], fit_info(round=1, n_trained_shots=3)
        seen['main_sha'] = file_sha256(f'{worker._directory}/networkARB.h5')
        seen['main_json'] = open(f'{worker._directory}/networkARB.json').read()
        with torch.no_grad():
            params.network.committor_net[-1].bias.fill_(0.25)
        seen['trained'] = {k: v.clone() for k, v in params.network.state_dict().items()}
        stop(worker)
        round2 = info if info is not None else {
            **fit_info(round=2, n_trained_shots=4), 'status': 'stopped',
            'reason': 'stop requested'}
        return list(losses), round2
    return fit, seen


@pytest.mark.parametrize('how', ['must_stop', 'termination_signal'])
def test_last_round_saved_as_final(tmp_path, monkeypatch, capsys, how):
    """A stop during the last fit: its network goes to network<s>.final.h5, the main file stays."""
    def stop(worker):
        if how == 'must_stop':
            worker.must_stop = True              # the trainer's must_stop() after the fit
        else:
            worker.termination_signal = 2        # a signal while the fit ran

    fit, seen = _two_rounds_last_stopped(stop)
    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    worker._train(nrounds=3, keep_running=False)
    assert seen['calls'] == 2
    # the main file and its json are those of round 1
    assert file_sha256(f'{run}/networkARB.h5') == seen['main_sha']
    assert open(f'{run}/networkARB.json').read() == seen['main_json']
    # the final file holds round 2's network, with its metadata
    final = torch.load(f'{run}/networkARB.final.h5', weights_only=True)
    assert state_dict_equal(final, seen['trained'])
    record = json.loads(open(f'{run}/networkARB.final.json').read())
    assert record['sha256'] == file_sha256(f'{run}/networkARB.final.h5')
    assert record['round'] == 2 and record['n_trained_shots'] == 4
    assert record['final'] is True and record['fit_status'] == 'stopped'
    assert record['fit_reason'] == 'stop requested'
    assert record['total_steps'] == worker.total_steps
    assert not os.path.exists(f'{run}/networkARB.final.step000001.h5')   # no step backups
    assert (f'Saving the network of the stopped last round to {run}/networkARB.final.h5'
            in capsys.readouterr().out)


def test_final_network_of_a_fit_without_metadata(tmp_path, monkeypatch):
    """A fit without info (the default fit's tuple): the final file still gets a json."""
    def stop(worker):
        worker.must_stop = True

    fit, seen = _two_rounds_last_stopped(stop, info=[1.0])
    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    worker._train(nrounds=3, keep_running=False)
    record = json.loads(open(f'{run}/networkARB.final.json').read())
    assert record['sha256'] == file_sha256(f'{run}/networkARB.final.h5')
    assert record['final'] is True and record['fit_status'] is None
    assert 'round' not in record
    assert file_sha256(f'{run}/networkARB.h5') == seen['main_sha']


def test_no_final_network_without_a_trained_last_round(tmp_path, monkeypatch):
    """No final file after a stop when the last fit trained nothing: no losses, or the no-op."""
    def stop(worker):
        worker.must_stop = True

    empty, _ = _two_rounds_last_stopped(stop, losses=())
    worker, events, run = make_trainer(tmp_path, monkeypatch, empty)
    worker._train(nrounds=3, keep_running=False)
    assert not os.path.exists(f'{run}/networkARB.final.h5')

    noop, _ = _two_rounds_last_stopped(stop, losses=(0.0,), info={
        **fit_info(round=2), 'status': 'no-op', 'reason': 'first round of a fresh run'})
    (tmp_path / 'b').mkdir()
    worker, events, run = make_trainer(tmp_path / 'b', monkeypatch, noop)
    worker._train(nrounds=3, keep_running=False)
    assert os.path.exists(f'{run}/networkARB.h5')
    assert not os.path.exists(f'{run}/networkARB.final.h5')


def test_no_final_network_when_the_last_round_ends_normally(tmp_path, monkeypatch):
    """nrounds reached, or a stop outside the fit: the trainer writes no final file."""
    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], fit_info()

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    worker._train(nrounds=1, keep_running=False)
    assert os.path.exists(f'{run}/networkARB.h5')
    assert not os.path.exists(f'{run}/networkARB.final.h5')


def test_check_cgaimmd_restart_before_first_value_pass(tmp_path, monkeypatch):
    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], fit_info()

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    monkeypatch.setattr('aimmd.cgaimmd.restart.check_cgaimmd_restart',
                        lambda params, directory, **kw: events.append(('restart', directory, kw)))
    worker._train(nrounds=1, keep_running=False)
    assert events[0] == ('restart', run, {'descriptors': True})
    assert ('compute', 'values') in events[1:]


def test_check_cgaimmd_restart_failure_stops_the_trainer(tmp_path, monkeypatch):
    """A refused folder: no value pass, no fit, and no initial network written (the check comes first)."""
    def fit(params, pathensemble, verbose=False, worker=None):
        raise AssertionError('the fit must not run')

    fname, _ = initial_network_file(tmp_path)
    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, initial_network=fname)
    os.makedirs(f'{run}/chainR0')
    np.save(f'{run}/chainR0/tps_weights.npy', np.ones(1))
    with pytest.raises(NotImplementedError, match='TPS_RESTART_AS_CGAIMMD'):
        worker._train(nrounds=1, keep_running=False)
    assert not [event for event in events if event[0] == 'compute']
    assert not os.path.exists(f'{run}/networkARB.h5')
    assert not os.path.exists(f'{run}/networkARB.json')


def test_no_restart_check_under_rfps(tmp_path, monkeypatch):
    def fit(params, pathensemble, verbose=False, worker=None):
        return [1.0], [1.0]

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, chain_type='rfps')
    os.makedirs(f'{run}/chainR0')
    np.save(f'{run}/chainR0/tps_weights.npy', np.ones(1))
    worker._train(nrounds=1, keep_running=False)                     # no refusal


# ---------------------------------------------------------------------------
# kinetics convergence under C-GAIMMD
# ---------------------------------------------------------------------------

def test_kinetics_convergence_cgaimmd_needs_initial_network(tmp_path, monkeypatch):
    """Every fraction starts from initial_network: without one, a clear error before any work."""
    def fit(params, pathensemble, verbose=False, worker=None):
        raise AssertionError('the fit must not run')

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit)
    worker.task = 'kinetics_convergence'
    with pytest.raises(ValueError, match="chain_type='cgaimmd' trains every data fraction from "
                                         "params.initial_network"):
        worker._kinetics_convergence()
    assert not events                                      # refused before any value pass


def test_kinetics_convergence_fractions_start_from_initial_network(tmp_path, monkeypatch):
    """Each fraction's fit starts from initial_network in a fit-state folder of its own; the
    run's network file is not touched and the network in memory is restored."""
    from tests._cgaimmd_helpers import build_cgaimmd_run_dir
    layout = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3)
    run = layout.run
    fname, source = initial_network_file(tmp_path)
    run_network = FlowNet()
    with torch.no_grad():
        run_network.committor_net[-1].bias.fill_(3.0)
    torch.save(run_network.state_dict(), f'{run}/networkARB.h5')
    run_sha = file_sha256(f'{run}/networkARB.h5')
    seen = []

    def fit(params, pathensemble, verbose=False, worker=None):
        state_dir = worker.kinetics_convergence_state_dir
        seen.append(({k: v.clone() for k, v in params.network.state_dict().items()}, state_dir))
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, 'rounds.jsonl'), 'a') as handle:
            handle.write('{"round": 1}\n')
        if os.path.exists(os.path.join(state_dir, 'state.json')):     # an earlier call's state
            raise AssertionError('the fraction started on an earlier call\'s fit state')
        with open(os.path.join(state_dir, 'state.json'), 'w') as handle:
            handle.write('{"round": 1}')
        with torch.no_grad():
            params.network.committor_net[-1].bias.add_(1.0)   # what a round would learn
        return [1.0], fit_info()

    network = FlowNet()
    params = cgaimmd_params_namespace(network=network, fit=fit, nbins=2,
                                      initial_network=fname, path=tmp_path / 'params.py')
    params.__dict__['path'] = tmp_path / 'params.py'
    worker = TinyTrainWorker(params, aimmd.PathEnsemble(layout.free), run)
    worker.task = 'kinetics_convergence'
    in_memory = {k: v.clone() for k, v in network.state_dict().items()}
    monkeypatch.chdir(tmp_path)
    worker._kinetics_convergence(fractions=[0.5, 1.0], network_save_pattern=None)
    assert [os.path.basename(state_dir) for _, state_dir in seen] == ['flow_fitARB.kcv050',
                                                                      'flow_fitARB.kcv100']
    for weights, _ in seen:                                   # never the run's or another's
        assert state_dict_equal(weights, source.state_dict())
    assert worker.kinetics_convergence_state_dir is None
    assert file_sha256(f'{run}/networkARB.h5') == run_sha
    assert state_dict_equal(network.state_dict(), in_memory)
    # a second call starts the fraction's fit state afresh
    worker._kinetics_convergence(fractions=[1.0], network_save_pattern=None)
    lines = open(os.path.join(run, 'flow_fitARB.kcv100', 'rounds.jsonl')).read().splitlines()
    assert lines == ['{"round": 1}']
    assert state_dict_equal(seen[-1][0], source.state_dict())


def test_kinetics_convergence_releases_the_flow_matching_trainer(tmp_path, monkeypatch):
    """The flow-matching fit's in-process trainer of a fraction's folder is released before and
    after the fraction, when that fit is imported; without it nothing is called."""
    import sys
    import types
    from aimmd.worker._train import _release_flow_matching_trainer
    monkeypatch.delitem(sys.modules, 'aimmd.cgaimmd.flow_matching.fit', raising=False)
    _release_flow_matching_trainer('nothing')                # the fit was never imported
    released = []
    module = types.ModuleType('aimmd.cgaimmd.flow_matching.fit')
    module.release_trainer = released.append
    monkeypatch.setitem(sys.modules, 'aimmd.cgaimmd.flow_matching.fit', module)
    fname, _ = initial_network_file(tmp_path)
    params = cgaimmd_params_namespace(network=FlowNet(), initial_network=fname)
    worker = TinyTrainWorker(params, None, tmp_path)
    state_dir = worker._kinetics_convergence_cgaimmd_start(str(tmp_path), 'ARB', 0.2, fname)
    worker._kinetics_convergence_cgaimmd_end(state_dir)
    assert released == [f'{tmp_path}/flow_fitARB.kcv020'] * 2


# ---------------------------------------------------------------------------
# the BioEmu2 fit's round counter reads what the trainer writes
# ---------------------------------------------------------------------------

def test_bioemu2_round_counter_reads_the_trainer_json(tmp_path, monkeypatch):
    """sequential_fit numbers its rounds from network<s>.json: 0 after the initial network, then
    the round the trainer stored from the fit's network_metadata (also across trainer starts)."""
    fit_module = pytest.importorskip('aimmd.cgaimmd.bioemu2.fit')
    fname, _ = initial_network_file(tmp_path)
    rounds = []

    def fit(params, pathensemble, verbose=False, worker=None):
        r = fit_module.previous_round(worker._directory, 'ARB') + 1
        rounds.append(r)
        with torch.no_grad():
            params.network.committor_net[-1].bias.add_(0.1)
        return [1.0], fit_info(round=r, n_trained_shots=3 * r)

    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, initial_network=fname)
    worker._train(nrounds=1, keep_running=False)
    assert json.loads(open(f'{run}/networkARB.json').read())['round'] == 1
    worker, events, run = make_trainer(tmp_path, monkeypatch, fit, initial_network=fname)
    worker._train(nrounds=1, keep_running=False)              # a new submission
    assert rounds == [1, 2]
    assert fit_module.previous_round(run, 'ARB') == 2


# ---------------------------------------------------------------------------
# the real flow-matching fit
# ---------------------------------------------------------------------------

class StoppedDuringFitWorker(WorkerTrain):
    """A host of the real `_train` whose must_stop fires at its `stop_at`-th poll once armed.

    It sets ``termination_signal`` as ``WorkerProperties.must_stop`` does when
    a stop condition fires; the flow-matching fit polls it before every step.
    """

    def __init__(self, params, initial_paths, root, stop_at):
        self.params = params
        self.initial_paths = initial_paths
        self._directory = self.directory = str(root)
        self.termination_signal = 0
        self.total_steps = 0
        self.total_frames = 0
        self.task = 'train'
        self.stop_at = stop_at
        self.polls = 0
        self.armed = False

    @property
    def must_stop(self):
        if self.armed and not self.termination_signal:
            self.polls += 1
            if self.polls >= self.stop_at:
                self.termination_signal = 2
        return bool(self.termination_signal)


def test_last_round_of_the_flow_matching_fit_saved_as_final(tmp_path, monkeypatch):
    """The real trainer and flow-matching fit: a stop after two steps of round 2 keeps round 1's
    network as network<s>.h5 and writes round 2's as network<s>.final.h5."""
    from aimmd.cgaimmd.flow_matching import fit as fit_module
    from tests._cgaimmd_helpers import build_cgaimmd_run_dir
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3)
    torch.manual_seed(0)
    network = FlowNet(lambda_clip=10.0)
    fname, _ = initial_network_file(tmp_path, network)
    seen = {}

    def fit(params, pathensemble, verbose=False, worker=None):
        seen['calls'] = seen.get('calls', 0) + 1
        if seen['calls'] == 2:
            seen['main_sha'] = file_sha256(f'{worker._directory}/networkARB.h5')
            worker.armed = True                 # steps 1 and 2 run, the 3rd poll stops
        return fit_module.flow_matching_fit(
            params, pathensemble, verbose=verbose, worker=worker, steps_per_round=5,
            warmup_steps=0, n_sp=4, n_anchor=4, n_fm=4, idle_sleep_s=0.0,
            normalizer_warmup=0)

    params = cgaimmd_params_namespace(network=network, fit=fit, nbins=2, network_save_interval=1,
                                      initial_network=fname)
    initial = build_path(tmp_path, stem='train_initial',
                         positions=np.array([[[-1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]],
                                            dtype=np.float32),
                         descriptors=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float))
    worker = StoppedDuringFitWorker(params, aimmd.PathEnsemble(initial), rd.run, stop_at=3)
    worker._train(nrounds=3, keep_running=False)
    run = rd.run
    assert seen['calls'] == 2
    assert file_sha256(f'{run}/networkARB.h5') == seen['main_sha']        # round 1's network
    main = json.loads(open(f'{run}/networkARB.json').read())
    record = json.loads(open(f'{run}/networkARB.final.json').read())
    assert record['sha256'] == file_sha256(f'{run}/networkARB.final.h5')
    assert record['round'] == main['round'] + 1
    assert record['fit_status'] == 'stopped' and record['final'] is True
