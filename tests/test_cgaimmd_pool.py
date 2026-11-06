"""The candidate pool, the network watcher and the seeds of C-GAIMMD shooting."""

import hashlib
import io
import os

import numpy as np
import pytest
import torch

from aimmd.cgaimmd import pool as cgaimmd_pool
from aimmd.cgaimmd.io import atomic_write_bytes, atomic_write_json
from aimmd.cgaimmd.pool import (Candidate, CandidatePool, NetworkWatcher, base_seed,
                             preparation_stream, refill_seed, refill_streams, stream_seed)
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from tests._cgaimmd_helpers import cgaimmd_params_namespace, write_fake_network_file


def _network(value=None, **kwargs):
    """A TinyFlowNetwork; `value` fills every parameter (distinct weights, distinct sha256)."""
    network = TinyFlowNetwork(**kwargs)
    if value is not None:
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(value)
    return network


def _sha_of(network):
    buffer = io.BytesIO()
    torch.save(network.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                 b.state_dict().values()))


def _candidate(index, refill=0):
    return Candidate(positions=np.full((1, 3), float(index)), lambda_target=0.0, lambda_raw=0.0,
                     refill=refill, index=index, refill_seed=0, prep_seed=0, validity_info=None,
                     law_record={'name': 'uniform'})


@pytest.fixture
def params():
    return cgaimmd_params_namespace(network=_network(0.0))


# ---------------------------------------------------------------------------
# the network watcher
# ---------------------------------------------------------------------------

def test_watcher_hash_and_metadata(tmp_path, params):
    """A new sha256 loads the weights (strict); network{s}.json supplies round and trained shots."""
    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    assert watcher.info is None and not watcher.loaded and watcher.n_trained_shots() == 0
    assert watcher.refresh() is False                      # no run directory yet

    trained = _network(0.25)
    sha = write_fake_network_file(run, trained, round=3, n_trained_shots=40)
    assert watcher.refresh() is True
    assert watcher.loaded and _same_weights(params.network, trained)
    info = watcher.info
    fname = f'{run}/networkARB.h5'
    assert info.file == fname
    assert info.sha256 == sha == hashlib.sha256(open(fname, 'rb').read()).hexdigest()
    assert (info.round, info.n_trained_shots, info.metadata_matched) == (3, 40, True)
    assert info.metadata['sha256'] == sha and info.metadata['round'] == 3
    assert watcher.n_trained_shots() == 40

    assert watcher.refresh() is False                      # unchanged
    stat = os.stat(fname)
    os.utime(fname, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert watcher.refresh() is False                      # touched, same bytes: no reload
    assert watcher.info.sha256 == sha

    newer = _network(0.5)
    sha2 = write_fake_network_file(run, newer, round=4, n_trained_shots=90)
    assert watcher.refresh() is True
    assert _same_weights(params.network, newer)
    assert (watcher.info.sha256, watcher.info.round, watcher.n_trained_shots()) == (sha2, 4, 90)


def test_metadata_window_keeps_enforcement(tmp_path, params):
    """The trainer writes the json first: a json that does not match keeps the last matched count."""
    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    write_fake_network_file(run, _network(0.25), round=1, n_trained_shots=100)
    assert watcher.refresh() and watcher.n_trained_shots() == 100

    # the json of the next network is written, its .h5 not yet
    nxt = _network(0.5)
    atomic_write_json(run / 'networkARB.json', {'round': 2, 'n_trained_shots': 200,
                                                'sha256': _sha_of(nxt)})
    assert watcher.refresh() is False
    assert watcher.n_trained_shots() == 100 and watcher.info.n_trained_shots == 100

    # the .h5 follows: the json now matches the loaded weights
    write_fake_network_file(run, nxt, round=2, n_trained_shots=200)
    assert watcher.refresh() is True
    assert watcher.n_trained_shots() == 200 and watcher.info.metadata_matched

    # a network without a json (hand-replaced): the last matched metadata is kept
    third = _network(0.75)
    sha3 = write_fake_network_file(run, third, round=9, n_trained_shots=999, write_json=False)
    assert watcher.refresh() is True
    info = watcher.info
    assert info.sha256 == sha3 and not info.metadata_matched
    assert (info.round, info.n_trained_shots) == (2, 200)
    assert info.metadata['n_trained_shots'] == 200         # the last json that matched
    assert watcher.n_trained_shots() == 200


def test_json_first_after_an_unmatched_network(tmp_path, params):
    """The trainer's hand-off (json, then the .h5 alone) also after a network without its json.

    A new load forgets which json it parsed: the json of the next network, parsed while the
    previous (unmatched) weights were loaded, is parsed again against the new sha256.
    """
    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    write_fake_network_file(run, _network(0.25), write_json=False)       # hand-replaced, no json
    assert watcher.refresh() and not watcher.info.metadata_matched
    nxt = _network(0.5)
    buffer = io.BytesIO()
    torch.save(nxt.state_dict(), buffer)
    data = buffer.getvalue()
    atomic_write_json(run / 'networkARB.json', {'round': 2, 'n_trained_shots': 200,
                                                'sha256': hashlib.sha256(data).hexdigest()})
    assert watcher.refresh() is False                     # polled between the trainer's writes
    atomic_write_bytes(run / 'networkARB.h5', data)       # the .h5 alone; the json untouched
    assert watcher.refresh() is True
    assert watcher.info.metadata_matched and watcher.n_trained_shots() == 200
    assert watcher.info.round == 2


def test_metadata_missing_warns_after_60s(tmp_path, params, monkeypatch, capsys):
    """No matching json 60 s after the load: one WARNING; a later match is taken up."""
    clock = [1000.0]
    monkeypatch.setattr(cgaimmd_pool, '_now', lambda: clock[0])
    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    network = _network(0.25)
    sha = write_fake_network_file(run, network, write_json=False)
    assert watcher.refresh() is True
    assert watcher.info.loaded_at == 1000.0 and not watcher.info.metadata_matched
    clock[0] = 1059.0
    watcher.refresh()
    assert 'WARNING' not in capsys.readouterr().out
    clock[0] = 1061.0
    watcher.refresh()
    out = capsys.readouterr().out
    assert out.strip() == (f'WARNING: networkARB.json has not described the loaded network '
                           f'(sha256 {sha[:8]}) for 60 s; trained shots kept at 0 (last matched '
                           f'metadata)')
    clock[0] = 1200.0
    watcher.refresh()
    assert capsys.readouterr().out == ''                   # once per loaded network
    atomic_write_json(run / 'networkARB.json', {'round': 5, 'n_trained_shots': 12,
                                                'sha256': sha})
    assert watcher.refresh() is False                      # same weights, json re-read
    assert watcher.info.metadata_matched and watcher.n_trained_shots() == 12
    assert watcher.info.round == 5


def test_missing_files_cost_no_waits(tmp_path, params, monkeypatch):
    """Polling before the first network never sleeps (stat / exists checks, not retries)."""
    def no_sleep(seconds):
        raise AssertionError(f'slept {seconds} s')
    monkeypatch.setattr(cgaimmd_pool, '_sleep', no_sleep)
    run = tmp_path / 'run'
    run.mkdir()
    watcher = NetworkWatcher(params, str(run))
    assert watcher.refresh() is False
    atomic_write_json(run / 'networkARB.json', {'round': 1, 'n_trained_shots': 3,
                                                'sha256': 'f' * 64})
    assert watcher.refresh() is False and not watcher.loaded


def test_changed_file_read_retried(tmp_path, params, monkeypatch):
    """An OSError reading a changed network file is retried (READ_RETRIES x READ_WAIT s)."""
    sleeps = []
    monkeypatch.setattr(cgaimmd_pool, '_sleep', sleeps.append)
    failures = [1]
    real_open = open

    def flaky_open(fname, *args, **kwargs):
        if str(fname).endswith('.h5') and failures[0] > 0:
            failures[0] -= 1
            raise PermissionError('busy (a test)')
        return real_open(fname, *args, **kwargs)

    monkeypatch.setattr(cgaimmd_pool, 'open', flaky_open, raising=False)
    run = tmp_path / 'run'
    write_fake_network_file(run, _network(0.25))
    watcher = NetworkWatcher(params, str(run))
    assert watcher.refresh() is True and sleeps == [cgaimmd_pool.READ_WAIT]

    # unreadable in every attempt: not loaded, not marked as failed, read again next time
    write_fake_network_file(run, _network(0.5))
    sleeps.clear()
    failures[0] = cgaimmd_pool.READ_RETRIES
    assert watcher.refresh() is False
    assert sleeps == [cgaimmd_pool.READ_WAIT] * (cgaimmd_pool.READ_RETRIES - 1)
    sleeps.clear()
    assert watcher.refresh() is True and sleeps == []
    assert _same_weights(params.network, _network(0.5))


def test_unloadable_network_warned_once(tmp_path, params, capsys):
    """A network file that does not load (strict) is not taken; warned once per file version."""
    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    before = {key: value.clone() for key, value in params.network.state_dict().items()}
    write_fake_network_file(run, _network(0.25, hidden=8))
    assert watcher.refresh() is False and not watcher.loaded
    out = capsys.readouterr().out
    assert out.count('WARNING') == 1 and 'networkARB.h5' in out and 'does not load' in out
    assert watcher.refresh() is False
    assert capsys.readouterr().out == ''
    assert all(torch.equal(before[key], value)
               for key, value in params.network.state_dict().items())
    write_fake_network_file(run, _network(0.25))
    assert watcher.refresh() is True and watcher.loaded


def test_watcher_moves_network_after_load(tmp_path, params, monkeypatch):
    """device: where the network is put after a load; None leaves it where it is."""
    moved = []
    network = params.network
    monkeypatch.setattr(network, 'to', lambda device: moved.append(device) or network)
    run = tmp_path / 'run'
    write_fake_network_file(run, _network(0.25))
    assert NetworkWatcher(params, str(run)).refresh() and moved == []
    write_fake_network_file(run, _network(0.5))
    assert NetworkWatcher(params, str(run), device=torch.device('cpu')).refresh()
    assert moved == [torch.device('cpu')]


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------

def test_pool_resets_on_new_network(tmp_path, params):
    """The pool holds one network's candidates; a new sha256 (checked before a pop) resets it."""
    pool = CandidatePool()
    assert len(pool) == 0 and pool.pop() is None
    assert pool.network_sha256 is None and pool.reset_count == 0

    run = tmp_path / 'run'
    watcher = NetworkWatcher(params, str(run))
    write_fake_network_file(run, _network(0.25))
    assert watcher.refresh()
    pool.extend([_candidate(i) for i in range(5)], watcher.info.sha256,
                np.random.default_rng(1))
    assert len(pool) == 5 and pool.network_sha256 == watcher.info.sha256
    with pytest.raises(ValueError, match='network'):
        pool.extend([_candidate(9)], 'another sha', np.random.default_rng(1))

    # the source's rule before every pop
    write_fake_network_file(run, _network(0.5))
    assert watcher.refresh()
    assert pool.reset('network changed') == 5
    assert len(pool) == 0 and pool.network_sha256 is None and pool.reset_count == 1
    assert pool.pop() is None
    pool.extend([_candidate(7)], watcher.info.sha256, np.random.default_rng(2))
    assert pool.network_sha256 == watcher.info.sha256 and pool.pop().index == 7


def test_random_pop_reproducible():
    """Pops are uniformly random with the pop stream of the refill that last extended the pool."""
    def order(pool):
        popped = []
        while len(pool):
            popped.append(pool.pop().index)
        return popped

    pools = []
    for _ in range(2):
        pool = CandidatePool()
        pool.extend([_candidate(i) for i in range(20)], 'sha', np.random.default_rng(7))
        pools.append(pool)
    first, second = order(pools[0]), order(pools[1])
    rng, remaining = np.random.default_rng(7), list(range(20))
    expected = [remaining.pop(int(rng.integers(len(remaining)))) for _ in range(20)]
    assert first == second == expected
    assert first != list(range(20)) and sorted(first) == list(range(20))

    # a later extend brings its own pop stream
    pool = CandidatePool()
    pool.extend([_candidate(i) for i in range(3)], 'sha', np.random.default_rng(1))
    pool.pop()
    pool.extend([_candidate(i) for i in range(3, 6)], 'sha', np.random.default_rng(99))
    rng = np.random.default_rng(99)
    remaining = [candidate.index for candidate in pool]
    assert pool.pop().index == remaining[int(rng.integers(len(remaining)))]


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_seed_streams_independent():
    """More draws from one stream do not move the others (four SeedSequence children)."""
    seed = refill_seed(123, 'run/chainR0', 'path000004', 0)
    law, pop, validity, prep = refill_streams(seed, 8)
    law.random(1000)
    law2, pop2, validity2, prep2 = refill_streams(seed, 8)
    law2.random(3)
    np.testing.assert_array_equal(pop.random(5), pop2.random(5))
    np.testing.assert_array_equal(validity.random(5), validity2.random(5))
    assert prep == prep2

    children = np.random.SeedSequence(seed).spawn(3)       # the fourth child leaves these as they are
    law3, pop3, validity3, _ = refill_streams(seed, 2)
    np.testing.assert_array_equal(law3.random(4), np.random.default_rng(children[0]).random(4))
    np.testing.assert_array_equal(pop3.random(4), np.random.default_rng(children[1]).random(4))
    validity_ss = np.random.SeedSequence(seed).spawn(4)[3]
    np.testing.assert_array_equal(validity3.random(4), np.random.default_rng(validity_ss).random(4))

    prep_children = np.random.SeedSequence(seed).spawn(4)[2].spawn(8)
    for index in (0, 5, 7):
        child = prep_children[index]
        assert prep[index] == int(child.generate_state(1, np.uint64)[0] >> 1)
        np.testing.assert_array_equal(preparation_stream(seed, index).random(4),
                                      np.random.default_rng(child).random(4))
        assert stream_seed(preparation_stream(seed, index)) == prep[index]
    assert refill_streams(seed, 3)[3] == prep[:3]          # a candidate's stream ignores n
    assert all(isinstance(value, int) and 0 <= value < 2 ** 63 for value in prep)
    law_ss = np.random.SeedSequence(seed).spawn(4)[0]
    assert stream_seed(refill_streams(seed, 1)[0]) == int(law_ss.generate_state(1, np.uint64)[0] >> 1)


def test_replicas_differ_without_generation_seed(tmp_path, monkeypatch):
    """base_seed: generation_seed, else a hash of the absolute run directory."""
    monkeypatch.chdir(tmp_path)
    for name in ('runA', 'runB'):
        (tmp_path / name).mkdir()
    settings = {'generation_seed': None}
    a, b = base_seed(settings, 'runA'), base_seed(settings, 'runB')
    assert a != b
    digest = hashlib.sha256(os.path.abspath('runA').encode()).digest()
    assert a == int.from_bytes(digest[:8], 'little') >> 1
    assert base_seed(settings, str(tmp_path / 'runA')) == a      # relative or absolute
    assert base_seed(settings, './runA/') == a
    assert base_seed({'generation_seed': 5}, 'runA') == base_seed({'generation_seed': 5}, 'runB') == 5
    assert isinstance(a, int) and 0 <= a < 2 ** 63

    expected = int.from_bytes(hashlib.sha256(b'5:chainR0:path000002:1').digest()[:8], 'little') >> 1
    assert refill_seed(5, 'runA/chainR0', 'path000002', 1) == expected
    assert refill_seed(5, str(tmp_path / 'runA' / 'chainR0') + '/', 'path000002', 1) == expected
    seeds = {refill_seed(5, 'runA/chainR0', path, refill)
             for path in ('path000001', 'path000002') for refill in range(3)}
    assert len(seeds) == 6
    assert refill_seed(5, 'runA/chainR1', 'path000002', 1) != expected
