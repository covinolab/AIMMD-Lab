"""The toy 1-D C-GAIMMD chain, in this process: generation, the final check, two-way shots,
registration and sidecars on the toy engine.

``tests/toy_1d_cgaimmd/params.py`` runs ``chain_type='cgaimmd'`` on the toy
1-D engine of ``tests/toy_1d`` with the minimal flow-matching network
``TinyFlowNetwork`` and the ``'ode'`` generator picked by name in the params.
That the shooting worker runs the whole loop with it (generation through the
generator interface, the validity and minimization hooks, the final check,
the shot, registration, the sidecar) shows that the loop is independent of
the network's architecture.

The trainer is not part of this test: the network files are written here, as
the trainer writes them (``network<s>.json`` first, then ``network<s>.h5``),
so that the acceptance rule switches at a known shot and the pool must reset
at a known point. Five shots in the bootstrap phase (no trained shots yet:
any R frame is accepted), then a network that names 5 trained shots (the
target rule is enforced), and after the sixth shot another network (the pool
resets before the seventh).

The task sequence is deterministic; the trajectories are not (the toy
engine's stop is polled by another thread, so segment lengths depend on
timing, although ``np.random`` and torch are seeded): no assertion depends on
them. The toy engine spends about 3.5 s per segment, so the test takes about
a minute. Late in a test session ``sys.path`` holds a folder for every params
file loaded before, which slows the worker down; the test runs without those
entries (:func:`short_sys_path`).
"""

import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.source import GenerativeShootingPointSource
from aimmd.cgaimmd.sidecar import SIDECAR_KEYS, software_versions
from tests._cgaimmd_helpers import write_fake_network_file
from tests._helpers_unit import write_trajectory


FOLDER = Path(__file__).resolve().parent / 'toy_1d_cgaimmd'
STATES = 'ARB'
WINDOW_AFTER = 5                          # cgaimmd_parameters['window_after_trained_shots']
GENERATOR = {'method': 'ode', 'settings': {'sampler': 'midpoint', 'n_steps': 8, 'chunk': 16},
             'batch_size': 16}

SIDECAR_SUBKEYS = {
    'acceptance': {'rule', 'enforced', 'tolerance', 'window', 'window_after_trained_shots',
                   'n_trained_shots'},
    'network': {'file', 'sha256', 'round', 'n_trained_shots', 'metadata_matched', 'provenance'},
    'generator': {'method', 'settings', 'batch_size'},
    'pool': {'refill', 'candidate', 'refill_size', 'n_targets', 'candidates_per_target',
             'passed_stage1', 'pool_left_after_pop', 'reset_count', 'stale_refill_pop',
             'refill_network_sha256'},
    'candidates': {'tried', 'rejected'},
    'validity': {'function', 'info'},
    'prepare': {'function', 'placement', 'ok', 'reason', 'seconds', 'record'},
    'minimize': {'function', 'ok', 'reason', 'seconds', 'record'},
    'cost_seconds': {'generate', 'validity', 'raw_lambda', 'prepare', 'minimize',
                     'final_check', 'total'},
    'seed': {'base', 'refill', 'refill_path', 'refill_index', 'candidate_index', 'prep'},
    'device': {'generation', 'values'},
    'aimmd': {'version', 'git_commit', 'git_dirty', 'git_reason', 'git_source', 'source_sha256'},
    'registration': {'n_frames', 'path_type', 'shooting_index', 'complete',
                     'lambda_registered_missing'},
}
NO_SHOT_WARNING = 'no acceptable C-GAIMMD shooting point'
R_FRAME_WARNING = 'WARNING: C-GAIMMD does not shoot from initial paths'
TOY_WINDOW = (1.1, 1.9)                              # the params file's validity window
XTC_TOLERANCE = 0.01                                 # Angstrom: XTC rounds to 0.01 A


def make_campaign(folder):
    """The campaign folder: the params file and the files it names.

    The initial paths are end-state frames only, one file per basin, close
    to the reactive region (A just below ``x = 1``, B just above ``x = 2``).
    The initial network is a seeded ``TinyFlowNetwork()``, the constructor of
    the params file's ``Network`` class.
    """
    folder = Path(folder)
    folder.mkdir(parents=True)
    shutil.copy(FOLDER / 'params.py', folder / 'params.py')
    for kind, x in (('A', 0.9), ('B', 2.1)):
        xs = x + 0.03 * np.sin(np.arange(10))
        positions = np.repeat(xs[:, None, None], 3, axis=2).astype(np.float32)   # x == y == z
        write_trajectory(folder, stem=f'initial_{kind}', positions=positions)
    torch.manual_seed(0)
    torch.save(TinyFlowNetwork().state_dict(), folder / 'initial_network.h5')
    return folder


@pytest.fixture
def short_sys_path(monkeypatch, tmp_path_factory):
    """``sys.path`` without the folders earlier tests' ``Params.load`` left in it.

    ``Params.load`` leaves the folder of every params file it loads in
    ``sys.path``; late in a test session every failed import then stats all
    of them, which slows the in-process workers down. The test runs without
    those entries (every entry under the session's base temporary directory)
    and without duplicates; ``monkeypatch`` puts the old list back.
    """
    base = str(tmp_path_factory.getbasetemp())
    kept, seen = [], set()
    for entry in sys.path:
        if entry not in seen and not entry.startswith(base):
            seen.add(entry)
            kept.append(entry)
    monkeypatch.setattr(sys, 'path', kept)


@pytest.fixture
def restore_signals():
    """``aimmd.Worker`` installs SIGINT/SIGTERM handlers; give pytest its own back."""
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def sidecars(run, pattern='chainR*'):
    """{path file: sidecar} of every shot of the reactive chains, in chain order."""
    out = {}
    for folder in sorted(glob.glob(os.path.join(run, pattern))):
        for fname in sorted(glob.glob(os.path.join(folder, 'path??????.xtc'))):
            sidecar = f'{fname}.cgaimmd.json'
            out[fname] = json.loads(Path(sidecar).read_text()) if os.path.exists(sidecar) else None
    return out


def check_sidecar_keys(record):
    assert set(record) >= set(SIDECAR_KEYS), set(SIDECAR_KEYS) - set(record)
    for key, subkeys in SIDECAR_SUBKEYS.items():
        assert set(record[key]) >= subkeys, (key, subkeys - set(record[key]))
    assert record['schema'] == 'aimmd.cgaimmd.shot/1' and record['source'] == 'cgaimmd'
    assert record['stale_record'] is None and record['recovered_after_restart'] is False
    assert math.isfinite(record['lambda_registered'])
    assert record['registration']['lambda_registered_missing'] is None
    assert record['state_final'] == 'R'
    assert record['generator'] == GENERATOR                    # picked by name in the params
    assert record['network']['provenance'] is None             # TinyFlowNetwork has none
    assert record['prepare']['function'] == 'place_in_box'
    assert record['prepare']['placement'] == 'none'
    assert record['registration']['complete'] is True
    assert record['aimmd']['git_commit'] or record['aimmd']['git_reason']
    assert record['aimmd']['source_sha256'] and record['software'] == software_versions()
    assert record['bias'] is None
    # the params file's hooks: the validity window and the minimization's record
    assert record['validity']['function'].endswith('cgaimmd_validity_function')
    assert TOY_WINDOW[0] <= record['validity']['info']['x'] <= TOY_WINDOW[1]
    assert set(record['stage1_rejected']) <= {'toy_window'}, record['stage1_rejected']
    minimize = record['minimize']
    assert minimize['function'].endswith('cgaimmd_minimize_function') and minimize['ok'] is True
    after = np.array(minimize['record']['after'])
    assert after.shape == (1, 3) and np.allclose(after[0], after[0, 0])      # y == z == x


def check_shooting_frames(chain, records):
    """Every shot starts from the frame the minimization hook returned (to XTC precision)."""
    assert len(chain) == len(records)
    for path, record in zip(chain, records):
        assert os.path.basename(path.fnames[0]) == record['path']
        assert path.shooting_index == record['registration']['shooting_index']
        frame = np.asarray(path.positions[path.shooting_index], dtype=float)
        after = np.array(record['minimize']['record']['after'])
        assert np.abs(frame - after).max() <= XTC_TOLERANCE, (record['path'], frame, after)


def check_pool_resets(records):
    """A candidate comes from a refill of the network it names, or is a stale pop.

    `records` are consecutive shots of one shooting worker (its counters
    start at 0). ``pool.refill_network_sha256`` names the network that
    generated a shot's candidate: the network of its final check
    (``network.sha256``), except for a stale pop, the first pop of a refill
    whose network was replaced while it was generated. Two shots whose
    candidates come from different networks must show a pool reset in
    between (``pool.reset_count`` up), a new refill (``pool.refill`` up) and
    that refill made for the later shot's own path (``seed.refill_path``).
    """
    for record in records:
        pool = record['pool']
        if pool['stale_refill_pop']:
            assert pool['refill_network_sha256'] != record['network']['sha256'], record
            assert pool['pool_left_after_pop'] == pool['passed_stage1'] - 1, record   # first pop
        else:
            assert pool['refill_network_sha256'] == record['network']['sha256'], record
    for before, after in zip(records, records[1:]):
        if after['pool']['refill_network_sha256'] == before['pool']['refill_network_sha256']:
            continue
        assert after['pool']['reset_count'] > before['pool']['reset_count'], (before, after)
        assert after['pool']['refill'] > before['pool']['refill'], (before, after)
        assert after['seed']['refill_path'] == os.path.splitext(after['path'])[0], after


def logs_of(run):
    """The text of every log file of the run folder."""
    text = []
    for fname in glob.glob(os.path.join(run, '**', '*.log'), recursive=True):
        text.append(Path(fname).read_text(errors='replace'))
    return '\n'.join(text)


def published_network(run, value, round, n_trained_shots):
    """Write a network file as the trainer does (json first): the initial network with every
    velocity weight moved by `value` (the committor head is untouched, so lambda stays 0)."""
    network = TinyFlowNetwork()
    network.load_state_dict(torch.load(os.path.join(os.path.dirname(run), 'initial_network.h5'),
                                       weights_only=True))
    with torch.no_grad():
        network.velocity_net[-1].bias += value
    return write_fake_network_file(run, network, STATES, round=round,
                                   n_trained_shots=n_trained_shots)


def test_toy_1d_cgaimmd_chain_in_process(tmp_path, monkeypatch, capfd, restore_signals,
                                         short_sys_path):
    """Bootstrap shots, a network naming trained shots, target shots, and a pool reset."""
    folder = make_campaign(tmp_path / 'campaign')
    monkeypatch.chdir(folder)                            # a Worker runs in params.parent
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert params.chain_type == 'cgaimmd'
    assert params.cgaimmd_generator == {'method': 'ode', 'sampler': 'midpoint', 'n_steps': 8,
                                        'chunk': 16}
    run = str(folder / 'run')

    # the trainer's first network: the initial one, no trained shots yet
    initial_sha = published_network(run, 0.0, round=1, n_trained_shots=0)

    # five shots in the bootstrap phase: no trained shots yet, any R configuration is accepted
    aimmd.Worker(params, run, nsteps=5, walltime=600).shoot(1, 0)
    assert len(sidecars(run)) == 5

    # a network that names 5 trained shots: the target rule is enforced from the sixth shot on;
    # after the sixth shot the test publishes another network, so the pool must reset before
    # the seventh
    trained_sha = published_network(run, 1e-4, round=2, n_trained_shots=5)
    real_note_registered = GenerativeShootingPointSource.note_registered
    published = []

    def note_registered(self):
        real_note_registered(self)
        if not published:
            published.append(published_network(run, 2e-4, round=3, n_trained_shots=6))
    monkeypatch.setattr(GenerativeShootingPointSource, 'note_registered', note_registered)
    aimmd.Worker(params, run, nsteps=8, walltime=600).shoot(1, 0)
    monkeypatch.setattr(GenerativeShootingPointSource, 'note_registered', real_note_registered)
    out = capfd.readouterr().out
    assert len(published) == 1
    updated_sha = published[0]
    assert len({initial_sha, trained_sha, updated_sha}) == 3

    records = sidecars(run)
    assert len(records) == 8 and all(r is not None for r in records.values())
    rules = [r['acceptance']['rule'] for r in records.values()]
    assert rules == ['bootstrap_any_R'] * 5 + ['target'] * 3
    for n, r in enumerate(records.values(), start=1):
        check_sidecar_keys(r)
        assert r['path'] == f'path{n:06d}.xtc' and r['chain'] == 'chainR0'
        assert r['pool']['stale_refill_pop'] is False      # the networks change between shots
        acceptance = r['acceptance']
        assert acceptance['window'] == [-3.0, 3.0]
        assert -3.0 <= r['lambda_target'] <= 3.0
        if n <= 5:
            assert acceptance['enforced'] is False and acceptance['n_trained_shots'] == 0
            assert r['network']['sha256'] == initial_sha
        elif n == 6:
            assert acceptance['enforced'] is True and acceptance['n_trained_shots'] == 5
            assert r['network']['sha256'] == trained_sha
            assert r['network']['round'] == 2 and r['network']['metadata_matched'] is True
            assert r['pool']['reset_count'] == 0
        else:                              # after the test's network update
            assert acceptance['enforced'] is True and acceptance['n_trained_shots'] == 6
            assert r['network']['sha256'] == updated_sha
            assert r['network']['round'] == 3 and r['network']['metadata_matched'] is True
            assert r['pool']['reset_count'] == 1
        if n > 5:
            assert abs(r['lambda_final'] - r['lambda_target']) <= acceptance['tolerance']
    records = list(records.values())
    check_pool_resets(records[5:])                          # the second shooting worker's shots
    assert records[6]['seed']['refill_path'] == 'path000007'          # a fresh refill for it
    assert records[6]['pool']['refill'] > records[5]['pool']['refill']
    text = out + logs_of(run)
    assert 'C-GAIMMD pool reset: network changed' in text
    assert "generator ode {'sampler': 'midpoint', 'n_steps': 8, 'chunk': 16}" in text
    # the validity hook filtered candidates: the first refill is reproducible (generation_seed
    # 0, and the initial network's zero velocity output returns its seeded prior samples)
    assert records[0]['stage1_rejected'].get('toy_window', 0) > 0, records[0]['stage1_rejected']
    # the RFPS selection pool is not used under C-GAIMMD
    for pool_log in glob.glob(os.path.join(run, 'chainR*', 'pool.log')):
        assert os.path.getsize(pool_log) == 0
    assert NO_SHOT_WARNING not in text
    assert R_FRAME_WARNING not in text                      # the initial files hold no R frame
    # no in-flight marker is left behind
    assert not glob.glob(os.path.join(run, 'chainR*', '.cgaimmd_shooting_point.*'))
    # the paths are ordinary RFPS-style shots of the chain, started from the minimized frames
    chain = params.shot_chains(run, 'R', 0)
    assert len(chain) == 8 and all(p.shooting_index is not None for p in chain)
    check_shooting_frames(chain, records)
