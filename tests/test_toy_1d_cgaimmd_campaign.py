"""The toy 1-D C-GAIMMD campaign with its trainer: in process, launched, kinetics
convergence, under a recorded bias, and resumed from an old-name in-flight marker.

The campaign is that of ``tests/test_toy_1d_cgaimmd.py`` (the toy engine,
``TinyFlowNetwork``, the ``'ode'`` generator picked by name, per-basin initial
files) with a trainer fit appended (``tests/_cgaimmd_toy_campaign.py``). Each
test runs with two fits:

- ``round_contract``: a stand-in that keeps the contract of a C-GAIMMD round
  (start from ``initial_network``, metadata with the round and the trained
  shots) without learning, so that the trainer's and the launcher's hooks run
  end to end on this branch;
- ``flow_matching``: the flow-matching fit of stream S4; skipped until
  ``aimmd.cgaimmd.flow_matching.fit`` exists, and to be run once S4 is merged.

Only the C-GAIMMD contract is asserted (network files and their json, the
sidecars, the acceptance rule, the restart from ``initial_network``), never a
fit's internals. The toy engine's trajectories are not reproducible (its stop
is polled by another thread), so no assertion depends on them. Every test
here is ``slow`` (each one runs a campaign of 30 s to two minutes).
"""

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import aimmd
from aimmd.cgaimmd import _legacy
from aimmd.cgaimmd.io import file_sha256
from aimmd.worker.utils import write_initial_network
from tests._cgaimmd_toy_campaign import FIT_KINDS, fit_kind, make_trained_campaign
from tests.test_toy_1d_cgaimmd import (STATES, check_pool_resets, check_shooting_frames,  # noqa: F401
                                       make_campaign, restore_signals, short_sys_path, sidecars)


pytestmark = pytest.mark.slow

WINDOW_AFTER = 2
SHORT_WINDOW = f'''
cgaimmd_parameters = {{**cgaimmd_parameters, 'window_after_trained_shots': {WINDOW_AFTER}}}
'''
"""Appended to the params: the target rule after two trained shots (the toy's five take long)."""

R_FRAME_WARNING = 'WARNING: C-GAIMMD does not shoot from initial paths'
NO_SHOT_WARNING = 'no acceptable C-GAIMMD shooting point'


@pytest.fixture(params=FIT_KINDS)
def kind(request):
    return fit_kind(request)


def network_record(run):
    """``network<s>.json``, checked to describe ``network<s>.h5``."""
    record = json.loads(Path(run, f'network{STATES}.json').read_text())
    assert record['sha256'] == file_sha256(os.path.join(run, f'network{STATES}.h5'))
    return record


def state_dict_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def logs_of(run):
    return '\n'.join(Path(fname).read_text(errors='replace')
                     for fname in glob.glob(os.path.join(run, '**', '*.log'), recursive=True))


def run_campaign(folder, params, *, n_bootstrap=WINDOW_AFTER, n_target=1, free_frames=40):
    """In process: initial network, free MD in A and B, bootstrap shots, one round, shots."""
    run = str(folder / 'run')
    assert write_initial_network(params, run) == 'written'
    for state in (0, 2):
        aimmd.Worker(params, run, nframes=free_frames, walltime=600).free(state, 0, 1)
    aimmd.Worker(params, run, nsteps=n_bootstrap, walltime=600).shoot(1, 0)
    aimmd.Worker(params, run, walltime=600).train(nrounds=1)
    if n_target:
        aimmd.Worker(params, run, nsteps=n_bootstrap + n_target, walltime=600).shoot(1, 0)
    return run


def test_campaign_in_process(tmp_path, monkeypatch, capfd, restore_signals, short_sys_path,
                             kind):
    """Free MD from the per-basin files, bootstrap shots, a trained round whose metadata the
    trainer writes, then shots under the target rule with that network."""
    folder = make_trained_campaign(tmp_path / 'campaign', kind, SHORT_WINDOW)
    monkeypatch.chdir(folder)
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    initial = torch.load(folder / 'initial_network.h5', weights_only=True)
    run = run_campaign(folder, params)
    out = capfd.readouterr().out

    record = network_record(run)
    assert record['n_trained_shots'] == WINDOW_AFTER and record['round'] >= 1, record
    assert record['total_steps'] >= 1
    trained = torch.load(os.path.join(run, f'network{STATES}.h5'), weights_only=True)
    assert not state_dict_equal(trained, initial)
    assert not os.path.exists(os.path.join(run, f'network{STATES}.final.h5'))
    for state in 'AB':                                    # each free run from its own basin
        assert glob.glob(os.path.join(run, f'free{state}', 'traj000001.part0001.xtc'))
    records = list(sidecars(run).values())
    assert len(records) == WINDOW_AFTER + 1 and all(r is not None for r in records)
    rules = [r['acceptance']['rule'] for r in records]
    assert rules == ['bootstrap_any_R'] * WINDOW_AFTER + ['target']
    last = records[-1]
    assert last['network']['sha256'] == record['sha256']
    assert last['network']['round'] == record['round']
    assert last['network']['metadata_matched'] is True
    assert last['acceptance']['n_trained_shots'] == WINDOW_AFTER
    assert abs(last['lambda_final'] - last['lambda_target']) <= last['acceptance']['tolerance']
    check_shooting_frames(params.shot_chains(run, 'R', 0), records)
    for pool_log in glob.glob(os.path.join(run, 'chainR*', 'pool.log')):
        assert os.path.getsize(pool_log) == 0
    text = out + logs_of(run)
    assert NO_SHOT_WARNING not in text and R_FRAME_WARNING not in text


def test_campaign_launched(tmp_path, monkeypatch, short_sys_path, kind):
    """One Launcher run (shooter, free A, free B, trainer): only invariants, the timing is free."""
    folder = make_trained_campaign(tmp_path / 'launch', kind, SHORT_WINDOW)
    monkeypatch.chdir(folder)
    aimmd.Launcher('params.py', 'run').run(1, 1, 1, nsteps=4, walltime=240)
    run = str(folder / 'run')
    records = list(sidecars(run).values())
    assert records, f'no C-GAIMMD shot in the launch; run folder: {sorted(os.listdir(run))}'
    assert all(r is not None for r in records)
    shas = set()
    for r in records:
        acceptance = r['acceptance']
        assert acceptance['enforced'] == (acceptance['n_trained_shots'] >= WINDOW_AFTER)
        assert r['source'] == 'cgaimmd' and r['registration']['complete'] is True
        assert r['generator']['method'] == 'ode'
        shas.add(r['network']['sha256'])
    shoot_log = Path(run, 'chainR0', 'worker.log').read_text(errors='replace')
    if len(shas) > 1:
        assert 'C-GAIMMD pool reset' in shoot_log or 'pool reset' in shoot_log
    check_pool_resets(records)
    text = logs_of(run)
    assert R_FRAME_WARNING not in text
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    check_shooting_frames(params.shot_chains(run, 'R', 0), records)
    # the launcher wrote the initial network; the json describes the network or a step backup
    record = json.loads(Path(run, f'network{STATES}.json').read_text())
    candidates = [os.path.join(run, f'network{STATES}.h5')]
    candidates += glob.glob(os.path.join(run, f'network{STATES}.step*.h5'))
    assert record['sha256'] in {file_sha256(f) for f in candidates}
    for pool_log in glob.glob(os.path.join(run, 'chainR*', 'pool.log')):
        assert os.path.getsize(pool_log) == 0
    # the free workers of both states started from their own basin's file and skipped the other
    for state, other in ('AB', 'BA'):
        log = '\n'.join(Path(f).read_text(errors='replace')
                        for f in glob.glob(os.path.join(run, f'free{state}', 'worker*.log')))
        assert f'initialARB/initial_{state}.xtc' in log, log[:2000]
        assert (f'skipped the initial paths without an end frame in the state: '
                f'run/initialARB/initial_{other}.xtc') in log
        assert glob.glob(os.path.join(run, f'free{state}', 'traj000001.part0001.xtc'))


def _run_files(run):
    """sha256 of what kinetics convergence must leave alone: the network, its json, bins,
    densities, the flow-matching fit state (if any) and every values cache."""
    names = [f'network{STATES}.h5', f'network{STATES}.json', f'bins{STATES}.npy',
             f'densities{STATES}.npy']
    names += [os.path.relpath(f, run) for f in glob.glob(os.path.join(run, f'flow_fit{STATES}',
                                                                       '*'))]
    state = {name: file_sha256(os.path.join(run, name)) for name in names
             if os.path.isfile(os.path.join(run, name))}
    for fname in sorted(glob.glob(os.path.join(run, '**', '*.values.npy'), recursive=True)):
        state[os.path.relpath(fname, run)] = file_sha256(fname)
    return state


def test_kinetics_convergence(tmp_path, monkeypatch, restore_signals, short_sys_path, kind):
    """Every fraction trains from initial_network in a fit-state folder of its own; the run is
    untouched; a fraction trained again in a second call gives the same network."""
    folder = make_trained_campaign(tmp_path / 'campaign', kind, SHORT_WINDOW)
    monkeypatch.chdir(folder)
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    run = run_campaign(folder, params, n_bootstrap=3, n_target=0)
    before = _run_files(run)
    in_memory = {k: v.clone() for k, v in params.network.state_dict().items()}
    initial = torch.load(folder / 'initial_network.h5', weights_only=True)

    results = aimmd.Worker(params, run, walltime=600).kinetics_convergence(fractions=[0.5, 1.0])
    assert results['fraction'].tolist() == [0.5, 1.0]
    assert np.all(results['n_frames'] > 0)
    assert results['n_frames'][0] < results['n_frames'][1]
    assert np.isfinite(results['k12']).all() and np.isfinite(results['k21']).all(), results
    networks = {}
    for pct in ('050', '100'):
        if kind == 'round_contract':                       # its fit state in the fraction's folder
            lines = Path(run, f'flow_fit{STATES}.kcv{pct}', 'rounds.jsonl').read_text()
            assert len(lines.splitlines()) == 1
        network = torch.load(os.path.join(run, f'network{STATES}.kcv{pct}.h5'),
                             weights_only=True)
        assert not state_dict_equal(network, initial)
        networks[pct] = network
    assert not state_dict_equal(networks['050'], networks['100'])
    assert _run_files(run) == before
    assert not glob.glob(os.path.join(run, '**', '*.kcv.npy'), recursive=True)
    assert state_dict_equal(params.network.state_dict(), in_memory)
    # independence: fraction 1.0 alone, after the first call, gives the same network
    aimmd.Worker(params, run, walltime=600).kinetics_convergence(fractions=[1.0])
    again = torch.load(os.path.join(run, f'network{STATES}.kcv100.h5'), weights_only=True)
    assert state_dict_equal(again, networks['100'])
    if kind == 'round_contract':                           # the earlier fit state was removed
        lines = Path(run, f'flow_fit{STATES}.kcv100', 'rounds.jsonl').read_text()
        assert len(lines.splitlines()) == 1
    assert _run_files(run) == before


BIAS = '''
record_bias = True
bias_source = 'reader'
bias_reactive_threshold = 0.5


def bias_function(trajectory):
    """A static wall in A: 2 kT per Angstrom below x = 1 (and above x = 9), zero elsewhere."""
    x = np.array([ts.positions[0, 0] for ts in trajectory], dtype=float)
    return 2.0 * (np.clip(1.0 - x, 0.0, None) + np.clip(x - 9.0, 0.0, None))
'''


def test_campaign_under_a_recorded_bias(tmp_path, monkeypatch, capfd, restore_signals,
                                        short_sys_path):
    """A C-GAIMMD run with record_bias: bias caches for shots and free MD, the trainer's bias
    check and bias-reweighted rates, the bias in every sidecar; kinetics convergence gives
    finite bias-reweighted rates too."""
    folder = make_trained_campaign(tmp_path / 'campaign', 'round_contract', SHORT_WINDOW + BIAS)
    monkeypatch.chdir(folder)
    np.random.seed(0)
    torch.manual_seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    assert params.record_bias is True
    run = run_campaign(folder, params, n_bootstrap=WINDOW_AFTER, n_target=0)
    out = capfd.readouterr().out
    assert "Bias check passed: mean |bias| in 'R'" in out
    assert 'k12 bias-reweighted:' in out and 'k21 bias-reweighted:' in out
    shots = params.shot_chains(run, 'R', 0)
    for path in list(shots) + [aimmd.Path(f) for f in glob.glob(os.path.join(run, 'free*',
                                                                             '*.xtc'))]:
        fname = path.fnames[0]
        bias = np.load(fname + '.bias.npy')
        x = np.asarray(path.positions)[:, 0, 0]
        assert len(bias) == len(x)
        np.testing.assert_allclose(bias, 2.0 * (np.clip(1.0 - x, 0, None)
                                                + np.clip(x - 9.0, 0, None)), atol=1e-5)
    free_a = np.concatenate([np.load(f) for f in glob.glob(os.path.join(run, 'freeA',
                                                                        '*.bias.npy'))])
    assert free_a.max() > 0                                 # the wall acts in A
    for record in sidecars(run).values():
        assert record['bias']['record_bias'] is True and record['bias']['bias_source'] == 'reader'
        assert record['bias']['bias_function'].endswith('bias_function')
        assert record['bias']['plumed_file'] is None and record['bias']['preparation_unbiased']
    results = aimmd.Worker(params, run, walltime=600).kinetics_convergence(fractions=[1.0])
    assert np.isfinite(results['k12_rw']).all() and np.isfinite(results['k21_rw']).all()


def test_resume_from_an_old_name_marker(tmp_path, monkeypatch, restore_signals, short_sys_path):
    """A shot in flight under the old names (``.flow_shooting_point.{json,npy}`` of the October
    campaigns) resumes without a new candidate: it is registered with a sidecar under the new
    name that carries the old record translated, and the old marker goes."""
    from tests._cgaimmd_helpers import write_fake_network_file
    folder = make_campaign(tmp_path / 'campaign')
    monkeypatch.chdir(folder)
    np.random.seed(0)
    params = aimmd.Params.load(str(folder / 'params.py'), save=False)
    run = str(folder / 'run')
    write_fake_network_file(run, params.network, round=1, n_trained_shots=0)
    chain = Path(run, 'chainR0')
    chain.mkdir(parents=True)
    # the in-flight shot: its segments initialized from x = 1.5 (in R), its old-name marker
    x = 1.5
    from types import SimpleNamespace
    frame = SimpleNamespace(positions=np.full((1, 3), x, dtype=np.float32),
                            dimensions=np.array([10, 10, 10, 90, 90, 90], dtype=np.float32))
    params.initialize_simulation(frame, str(chain / 'back'), str(chain / 'forw'))
    archived = Path(__file__).resolve().parent / 'data' / 'legacy_cgaimmd'
    record = json.loads((archived / 'archived_chignolin_flow_shooting_point.json').read_text())
    record.update(path_name='path000001', chain='chainR0', lambda_target=0.25)
    (chain / '.flow_shooting_point.json').write_text(json.dumps(record))
    np.save(chain / '.flow_shooting_point.npy', frame.positions.astype(np.float64))
    _legacy._reset()
    from aimmd.cgaimmd.source import GenerativeShootingPointSource
    asked = []
    real_next = GenerativeShootingPointSource.next_shooting_point

    def next_shooting_point(self, **kwargs):
        asked.append(kwargs['path_name'])
        return real_next(self, **kwargs)

    monkeypatch.setattr(GenerativeShootingPointSource, 'next_shooting_point', next_shooting_point)
    with pytest.warns(DeprecationWarning):
        aimmd.Worker(params, run, nsteps=1, walltime=600).shoot(1, 0)
    assert 'path000001' not in asked                        # resumed, not generated anew
    fname = chain / 'path000001.xtc'
    assert fname.is_file()
    sidecar = json.loads(Path(f'{fname}.cgaimmd.json').read_text())
    assert sidecar['schema'] == 'aimmd.cgaimmd.shot/1' and sidecar['source'] == 'cgaimmd'
    assert sidecar['lambda_target'] == 0.25 and sidecar['path_name'] == 'path000001'
    assert sidecar['generator']['method'] == 'bioemu2_edm'   # 'network.generate', translated
    assert math.isfinite(sidecar['lambda_registered'])
    assert not glob.glob(str(chain / '.flow_shooting_point.*'))
    assert not glob.glob(str(chain / '.cgaimmd_shooting_point.*'))
    assert not glob.glob(str(chain / '*.flow.json'))
    path = params.shot_chains(run, 'R', 0)[0]
    shooting_frame = np.asarray(path.positions[path.shooting_index], dtype=float)
    assert np.abs(shooting_frame - x).max() <= 0.01         # the marker's configuration
