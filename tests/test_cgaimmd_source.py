"""The C-GAIMMD shooting-point source: start-up, the two stages, the failure policy, the pool.

Generation goes through a generator registered by this test module ('test_scripted') and picked
by name in the params (``cgaimmd_generator``), as any generation method is.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import aimmd
from aimmd import _config
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd import lambda_law
from aimmd.cgaimmd import source as cgaimmd_source
from aimmd.cgaimmd._once import once
from aimmd.cgaimmd.generation import register_generator
from aimmd.cgaimmd.pool import preparation_stream, refill_seed, refill_streams, stream_seed
from aimmd.cgaimmd.sidecar import (REGISTRATION_KEYS, SIDECAR_KEYS, read_shot_sidecar,
                                   software_versions, write_in_flight, write_shot_sidecar)
from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
from aimmd.cgaimmd.types import NetworkInfo, PreparedFrame, ShootingPoint, ValidityResult
from aimmd.cgaimmd.source import (GenerativeShootingPointSource, candidate_context, final_check,
                                  place_in_box, prepare_candidate)
from aimmd.path.chainreader import ChainReader
from aimmd.pathensemble import PathEnsemble
from tests._cgaimmd_helpers import (build_cgaimmd_run_dir, cgaimmd_params_namespace,
                                    make_candidate_context, write_fake_network_file,
                                    x_states_function)

GAIN = 10.0                  # the scripted committor: lambda = GAIN * x of atom 0
FOLDER = 'run/chainR0'
PATH = 'path000003'          # build_cgaimmd_run_dir(n_shots=2) holds path000001 and path000002
LATE = 1                     # PATH's first refill (seed 0) pops 7, 5, 2, 6, 4, 3, 1, 0: sample 1 7th
MAX_FAKE_SLEEPS = 50         # the default fake sleep fails a test that waits more often
SCRIPTED = 'test_scripted'   # the generation method of this module


class ScriptedNetwork(TinyFlowNetwork):
    """lambda = 10 x; the generator 'test_scripted' puts sample k at x = script[k](lambda*) / 10.

    Without a script entry a sample is put at lambda* itself. Committor NaN for
    -0.48 < x < -0.46 (lambda near -4.7, still in R). ``oom_above``: batches larger than this
    raise a CUDA out-of-memory error; ``on_generate(network)`` runs at every call; ``broken``
    makes the generator return one atom too many. The start-up checks generate too:
    ``_started`` rewinds the script afterwards.
    """

    def __init__(self, script=(), *, oom_above=None, on_generate=None, **kwargs):
        kwargs.setdefault('lambda_clip', 10.0)
        super().__init__(**kwargs)
        self.script = list(script)
        self.oom_above = oom_above
        self.on_generate = on_generate
        self.broken = False
        self.generate_calls = []
        self.counter = 0

    def forward(self, inp):
        x = torch.flatten(torch.as_tensor(inp, dtype=torch.float32), start_dim=1)[:, :1]
        band = (x > -0.48) & (x < -0.46)
        return torch.where(band, torch.full_like(x, math.nan), GAIN * x)

    def scripted_samples(self, lam):
        out = torch.zeros((len(lam), self.n_atoms, 3))
        for i, target in enumerate(lam.tolist()):
            k = self.counter
            self.counter += 1
            value = self.script[k](target) if k < len(self.script) else target
            out[i, :, 0] = value / GAIN
        return out


class ScriptedGenerator:
    """The generation method 'test_scripted': the samples of a ScriptedNetwork's script."""

    method = SCRIPTED

    def __init__(self, chunk=128):
        self.chunk = int(chunk)

    def check(self, network):
        return [] if isinstance(network, ScriptedNetwork) else [
            f'{SCRIPTED} needs a ScriptedNetwork, got {type(network).__name__}']

    def generate(self, network, lam, *, generator=None):
        lam = torch.as_tensor(lam, dtype=torch.float32).reshape(-1)
        network.generate_calls.append({'n': len(lam), 'lam': lam.clone(),
                                       'generator': generator})
        if network.on_generate is not None:
            network.on_generate(network)
        if network.oom_above is not None and len(lam) > network.oom_above:
            raise torch.cuda.OutOfMemoryError('CUDA out of memory (a test)')
        out = network.scripted_samples(lam)
        return torch.cat([out, out[:, :1]], dim=1) if network.broken else out

    def settings(self):
        return {'chunk': self.chunk}


@pytest.fixture(autouse=True)
def scripted_method():
    """Register 'test_scripted' (the conftest restores the registry after every test)."""
    register_generator(SCRIPTED, ScriptedGenerator)


class OOMPriorNetwork(TinyFlowNetwork):
    """The 'ode' generator's network; sample_prior runs out of memory for more than `limit` rows.

    It draws its prior noise first and raises afterwards, as a real run does part of the way
    through a batch, so that a retry that does not re-seed the generator draws other numbers.
    """

    def __init__(self, limit, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def sample_prior(self, lam, n, *, uncond_mask=None, generator=None):
        prior = super().sample_prior(lam, n, uncond_mask=uncond_mask, generator=generator)
        if n > self.limit:
            raise torch.cuda.OutOfMemoryError('CUDA out of memory (a test)')
        return prior


class FakeWorker:
    """What the source reads of a Worker: must_stop (a property), initial_paths, _location."""

    def __init__(self, initial_paths=()):
        self.initial_paths = PathEnsemble(list(initial_paths))
        self.stop = False
        self._location = './run/chainR0'

    @property
    def must_stop(self):
        return self.stop


def _setup(tmp_path, monkeypatch, *, network=None, n_trained_shots=0, initial=None,
           write_network=True, params_overrides=None, generator=None, chunk=None, **settings):
    """A run folder, params, a network file and a source for chain folder run/chainR0.

    The params pick the generator by name: ``{'method': 'test_scripted'}`` for a
    ScriptedNetwork (with ``chunk`` if given), else `generator` (default None: the
    network's own default, 'ode' for a TinyFlowNetwork; ``chunk`` then overrides its
    chunk).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    run = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=2)
    network = ScriptedNetwork() if network is None else network
    cgaimmd_parameters = {'placement': 'none', 'lambda_window': (-4.0, 4.0),
                          'window_after_trained_shots': 0, 'n_targets': 4,
                          'candidates_per_target': 2, 'max_refills': 3, 'wait_seconds': 5.0,
                          'generation_seed': 0}
    cgaimmd_parameters.update(settings)
    if generator is None and isinstance(network, ScriptedNetwork):
        generator = {'method': SCRIPTED}
    if chunk is not None:
        generator = {**(generator or {}), 'chunk': chunk}
    overrides = {'cgaimmd_generator': generator, **(params_overrides or {})}
    params = cgaimmd_params_namespace(network=network, cgaimmd_parameters=cgaimmd_parameters,
                                      **overrides)
    sha = (write_fake_network_file('run', network, n_trained_shots=n_trained_shots)
           if write_network else None)
    worker = FakeWorker(run.free if initial is None else initial(run))
    sleeps = []

    def bounded_sleep(seconds):                 # a regression that never shoots fails, not hangs
        sleeps.append(seconds)
        if len(sleeps) > MAX_FAKE_SLEEPS:
            raise AssertionError(f'the source waited {MAX_FAKE_SLEEPS} times without a result')

    monkeypatch.setattr(cgaimmd_source, '_sleep', bounded_sleep)
    source = GenerativeShootingPointSource(params, worker, 'run')
    return SimpleNamespace(run=run, params=params, network=network, worker=worker, source=source,
                           sha=sha, sleeps=sleeps, chain=PathEnsemble(run.chains[0]))


def _next(setup, path_name=PATH):
    return setup.source.next_shooting_point(chain=setup.chain, folder=FOLDER, path_name=path_name)


def _started(tmp_path, monkeypatch, **kwargs):
    setup = _setup(tmp_path, monkeypatch, **kwargs)
    scripted = isinstance(setup.network, ScriptedNetwork)
    if scripted:                                            # the numerics' draws are not scripted
        script, setup.network.script = setup.network.script, []
    assert setup.source.wait_for_network() is True
    if scripted:
        setup.network.script, setup.network.counter = script, 0
        setup.network.generate_calls = []
    return setup


def _targets(path_name=PATH, refill=0, n=4, lo=-4.0, hi=4.0, base=0):
    """The uniform law's lambda* of a refill, recomputed from the seed formulas."""
    law = refill_streams(refill_seed(base, FOLDER, path_name, refill), 8)[0]
    return lo + (hi - lo) * law.random(n)


def _lines(out, start):
    return [line for line in out.splitlines() if line.startswith(start)]


def _only(good, bad, accepted=lambda target: target, n=8):
    """A script: sample `good` at ``accepted(lambda*)``, the other samples at ``bad(lambda*)``."""
    return [accepted if k == good else bad for k in range(n)]


# ---------------------------------------------------------------------------
# start-up
# ---------------------------------------------------------------------------

def test_wait_for_network(tmp_path, monkeypatch, capsys):
    """Blocks (1 s polls) until network{s}.h5 loads; prints the waiting line once; must_stop -> False."""
    setup = _setup(tmp_path, monkeypatch, write_network=False)

    def sleep(seconds):
        setup.sleeps.append(seconds)
        if len(setup.sleeps) == 3:
            write_fake_network_file('run', setup.network)

    monkeypatch.setattr(cgaimmd_source, '_sleep', sleep)
    assert setup.source.wait_for_network() is True
    assert setup.sleeps == [1.0, 1.0, 1.0] and setup.source.watcher.loaded
    out = capsys.readouterr().out
    assert len(_lines(out, 'Waiting for networkARB.h5 (C-GAIMMD shooting) ')) == 1

    stopped = _setup(tmp_path / 'other', monkeypatch, write_network=False)
    stopped.worker.stop = True
    assert stopped.source.wait_for_network() is False


def test_startup_line(tmp_path, monkeypatch, capsys):
    """The start-up line (devices resolved inside a worker task) and the numerics report."""
    moved = []
    network = ScriptedNetwork()
    real_to = network.to
    monkeypatch.setattr(network, 'to', lambda device: moved.append(str(device)) or real_to(device))
    monkeypatch.setattr(_config, 'WORKER_TASK', 'shoot')
    setup = _started(tmp_path, monkeypatch, network=network, generation_device='cpu',
                     values_device='cpu', window_after_trained_shots=7, n_trained_shots=3)
    out = capsys.readouterr().out
    assert _lines(out, 'C-GAIMMD shooting: ') == [
        f"C-GAIMMD shooting: ScriptedNetwork n_atoms=1, generator test_scripted {{'chunk': 128}}, "
        f"devices generation=cpu values=cpu, network {setup.sha[:8]} (round 1, 3 trained "
        f"shots), law uniform, window -4.0, 4.0, acceptance target after 7 trained shots"]
    assert setup.source.numerics['n_atoms'] == 1 and setup.source.numerics['n_frames'] == 4
    # the numerics drew through the run's generator
    assert setup.source.numerics['generation'] == {'method': SCRIPTED,
                                                   'settings': {'chunk': 128}}
    assert moved and set(moved) == {'cpu'}
    moved.clear()
    assert _next(setup).record['device'] == {'generation': 'cpu', 'values': 'cpu'}
    assert moved and set(moved) == {'cpu'}                    # moved at use, not cached

    _started(tmp_path / 'b', monkeypatch,
             network=TinyFlowNetwork(prior_center=0.0, prior_width=0.1, lambda_clip=10.0),
             lambda_law='bins', lambda_window='bins')
    out = capsys.readouterr().out
    line = _lines(out, 'C-GAIMMD shooting: ')[0]
    assert ("generator ode {'sampler': 'midpoint', 'n_steps': 16, 'chunk': 128, "
            "'guidance_scale': 1.0}") in line
    assert 'law bins, window bins (no bins yet: -10.0, 10.0), ' in line

    # the window printed is the resolved one: a pair clipped to [-L, L], the bins' cutoffs
    _started(tmp_path / 'c', monkeypatch, network=ScriptedNetwork(lambda_clip=4.0),
             lambda_window=(-6.0, 6.0))
    assert 'law uniform, window -4.0, 4.0, ' in _lines(capsys.readouterr().out,
                                                       'C-GAIMMD shooting: ')[0]
    binned = _setup(tmp_path / 'd', monkeypatch, lambda_window='bins',
                    params_overrides={'reweight_parameters': {'sp_cutoff_max': 2.0}})
    save_npy('run/binsARB.npy', np.array([-np.inf, -3.0, 0.0, 3.0, np.inf]))
    save_npy('run/densitiesARB.npy', np.full(4, 0.25))
    assert binned.source.wait_for_network()
    assert 'law uniform, window bins (-3.0, 2.0), ' in _lines(capsys.readouterr().out,
                                                              'C-GAIMMD shooting: ')[0]


def test_descriptor_width_checked(tmp_path, monkeypatch):
    """Descriptors that are not the flat coordinates of the M generator atoms stop the start-up."""
    def two_columns(trajectory):
        return np.array([ts.positions[:, :2].reshape(-1) for ts in trajectory], dtype=np.float32)

    setup = _setup(tmp_path, monkeypatch, params_overrides={'descriptors_function': two_columns})
    with pytest.raises(ValueError, match='migrate_descriptors'):
        setup.source.wait_for_network()


def test_startup_configuration_errors(tmp_path, monkeypatch):
    """An empty window or an impossible placement stops the start-up, before any refill."""
    import MDAnalysis as mda
    empty = _setup(tmp_path / 'a', monkeypatch, lambda_window=(11.0, 12.0))      # L = 10
    with pytest.raises(ValueError, match='empty after clipping'):
        empty.source.wait_for_network()

    cutoffs = _setup(tmp_path / 'b', monkeypatch, lambda_window='bins',
                     params_overrides={'reweight_parameters': {'sp_cutoff_min': 20.0}})
    save_npy('run/binsARB.npy', np.array([-np.inf, -3.0, 0.0, 3.0, np.inf]))
    save_npy('run/densitiesARB.npy', np.full(4, 0.25))
    with pytest.raises(ValueError, match='empty after clipping'):
        cutoffs.source.wait_for_network()

    boxless = _setup(tmp_path / 'c', monkeypatch, placement='box_center')
    with pytest.raises(TypeError, match='box'):
        boxless.source.wait_for_network()

    universe = mda.Universe.empty(2, trajectory=True)
    universe.dimensions = [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]
    larger = _setup(tmp_path / 'd', monkeypatch, params_overrides={'_universe': universe})
    with pytest.raises(ValueError, match='cgaimmd_prepare_function'):
        larger.source.wait_for_network()

    # a prepare function takes over the placement: no universe needed
    prepared = _setup(tmp_path / 'e', monkeypatch, placement='box_center',
                      params_overrides={'cgaimmd_prepare_function': cgaimmd_prepare_function})
    assert prepared.source.wait_for_network() is True


def test_r_frame_warning(tmp_path, monkeypatch, capsys):
    """Initial paths with R frames: a WARNING with the count and the files (they seed free MD only)."""
    setup = _started(tmp_path, monkeypatch, initial=lambda run: [run.free[0], run.shots[0]])
    out = capsys.readouterr().out
    assert _lines(out, 'WARNING: C-GAIMMD does not shoot') == [
        f'WARNING: C-GAIMMD does not shoot from initial paths: their 5 frames in state R '
        f'({setup.run.shots[0].fname}) are not used; initial paths seed free simulations only.']
    _started(tmp_path / 'b', monkeypatch)
    assert not _lines(capsys.readouterr().out, 'WARNING: C-GAIMMD does not shoot')


def test_default_window_after_warning(tmp_path, monkeypatch, capsys):
    """window_after_trained_shots left at its default: the WARNING, once per process."""
    setup = _setup(tmp_path, monkeypatch)
    setup.params.__dict__['cgaimmd_parameters'] = {
        key: value for key, value in setup.params.cgaimmd_parameters.items()
        if key != 'window_after_trained_shots'}
    assert setup.source.wait_for_network()
    GenerativeShootingPointSource(setup.params, setup.worker, 'run').wait_for_network()
    warnings = _lines(capsys.readouterr().out, 'WARNING: window_after_trained_shots')
    assert warnings == ['WARNING: window_after_trained_shots is at its default 1000; this is '
                        'highly system dependent (it was 8000 for the NANMA test system), set '
                        'it explicitly']
    # the worker's own key; the Params checks use 'params: ...'
    assert once(cgaimmd_source.WINDOW_AFTER_WARNING_KEY) is False
    assert cgaimmd_source.WINDOW_AFTER_WARNING_KEY == 'window_after_trained_shots default'
    from aimmd.cgaimmd import _once
    _once._reset()
    _started(tmp_path / 'b', monkeypatch)                           # the key given: no warning
    assert not _lines(capsys.readouterr().out, 'WARNING: window_after')


# ---------------------------------------------------------------------------
# one shooting point
# ---------------------------------------------------------------------------

def test_first_valid_candidate_returned(tmp_path, monkeypatch, capsys):
    """Stage 1 fills the pool, stage 2 pops until a candidate passes; its record is complete."""
    in_b = _only(LATE, lambda target: 30.0 + target)     # x = 3 + lambda*/10: state B
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(in_b))
    capsys.readouterr()
    point = _next(setup)
    assert isinstance(point, ShootingPoint)
    assert point.positions.dtype == np.float32 and point.positions.shape == (1, 3)
    assert point.dimensions is None                     # placement 'none', no universe
    lam = setup.network.generate_calls[0]['lam'].numpy()
    np.testing.assert_array_equal(lam[0::2], lam[1::2])  # each target repeated, not tiled
    record = point.record
    targets = _targets()
    assert record['lambda_target'] == targets[0]        # sample 1 = the 2nd candidate of target 0
    np.testing.assert_allclose(point.positions[0, 0], targets[0] / GAIN, rtol=1e-6)
    assert abs(record['lambda_raw'] - targets[0]) < 1e-4
    assert abs(record['lambda_final'] - targets[0]) <= 0.05 + 1e-6   # rounded to 0.01 A
    assert record['state_final'] == 'R' and record['source'] == 'cgaimmd'
    assert record['path_name'] == PATH and record['chain'] == 'chainR0'
    tried = 7
    assert record['candidates'] == {'tried': tried, 'rejected': {'state_B': 6}}
    assert record['pool'] == {'refill': 1, 'candidate': LATE, 'refill_size': 8, 'n_targets': 4,
                              'candidates_per_target': 2, 'passed_stage1': 8,
                              'pool_left_after_pop': 8 - tried, 'reset_count': 0,
                              'stale_refill_pop': False, 'refill_network_sha256': setup.sha}
    assert record['stage1_rejected'] == {}
    assert record['acceptance'] == {'rule': 'target', 'enforced': True, 'tolerance': 1.0,
                                    'window': [-4.0, 4.0], 'window_after_trained_shots': 0,
                                    'n_trained_shots': 0}
    assert record['network'] == {'file': 'run/networkARB.h5', 'sha256': setup.sha, 'round': 1,
                                 'n_trained_shots': 0, 'metadata_matched': True,
                                 'provenance': None}       # the network has no provenance()
    # what the run used, recorded once per worker
    assert record['aimmd']['source_sha256'] and (record['aimmd']['git_commit']
                                                 or record['aimmd']['git_reason'])
    assert record['software'] == software_versions() and record['bias'] is None
    # the generator picked by name in the params, its settings and the batch size used
    assert record['generator'] == {'method': SCRIPTED, 'settings': {'chunk': 128},
                                   'batch_size': 128}
    seed = refill_seed(0, FOLDER, PATH, 0)
    assert record['seed'] == {'base': 0, 'refill': seed, 'refill_path': PATH, 'refill_index': 0,
                              'candidate_index': LATE,
                              'prep': stream_seed(preparation_stream(seed, LATE))}
    assert record['device'] == {'generation': 'cpu', 'values': 'cpu'}
    assert record['lambda_law'] == {'name': 'uniform', 'window': [-4.0, 4.0],
                                    'window_source': 'setting'}
    assert record['prepare']['function'] == 'place_in_box' and record['prepare']['ok']
    assert record['minimize'] == {'function': None, 'ok': None, 'reason': None, 'seconds': None,
                                  'record': None}
    assert set(record['cost_seconds']) == {'generate', 'validity', 'raw_lambda', 'prepare',
                                           'minimize', 'final_check', 'total'}
    assert all(value >= 0 for value in record['cost_seconds'].values())
    assert len(setup.source.pool) == 8 - tried
    out = capsys.readouterr().out
    [refill] = _lines(out, 'C-GAIMMD pool refill ')
    assert refill.startswith(f'C-GAIMMD pool refill 1 for ./run/chainR0/{PATH}: 8 candidates for 4 '
                             f'targets (network {setup.sha[:8]}, uniform); 8 passed stage 1 (none) '
                             f'in ') and refill.endswith(' s')
    [line] = _lines(out, 'C-GAIMMD shooting point for ')
    assert line.startswith(f'C-GAIMMD shooting point for ./run/chainR0/{PATH}: '
                           f'lambda*={targets[0]:.3f} lambda={record["lambda_final"]:.3f} '
                           f'(target), 7 candidates tried (state_B: 6), network ')
    assert f'network {setup.sha[:8]}, ' in line and line.endswith(' s')


def test_bootstrap_before_n(tmp_path, monkeypatch):
    """Before window_after_trained_shots, any R configuration is accepted ('bootstrap_any_R')."""
    far = [lambda target: target + 3.0] * 8
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(far), lambda_window=(-4.0, 1.0),
                     window_after_trained_shots=5, n_trained_shots=4)
    record = _next(setup).record
    assert record['candidates'] == {'tried': 1, 'rejected': {}}
    assert abs(record['lambda_final'] - record['lambda_target'] - 3.0) < 0.06
    assert record['acceptance'] == {'rule': 'bootstrap_any_R', 'enforced': False,
                                    'tolerance': 1.0, 'window': [-4.0, 1.0],
                                    'window_after_trained_shots': 5, 'n_trained_shots': 4}


def test_target_rule_after_n(tmp_path, monkeypatch):
    """From window_after_trained_shots on, |lambda - lambda*| <= lambda_tolerance ('target')."""
    script = _only(LATE, lambda target: target + 3.0, lambda target: target + 0.2)
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(script),
                     lambda_window=(-4.0, 1.0), window_after_trained_shots=5, n_trained_shots=5)
    record = _next(setup).record
    assert record['pool']['candidate'] == LATE
    assert record['acceptance']['rule'] == 'target' and record['acceptance']['enforced']
    assert record['candidates'] == {'tried': 7, 'rejected': {'lambda_window': 6}}
    assert abs(record['lambda_final'] - record['lambda_target'] - 0.2) < 0.06


def test_window_rule(tmp_path, monkeypatch):
    """acceptance='window': lambda inside the resolved window, whatever lambda*."""
    script = _only(LATE, lambda target: 3.0, lambda target: 1.5)
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(script),
                     lambda_window=(-2.0, 2.0), acceptance='window')
    record = _next(setup).record
    assert record['pool']['candidate'] == LATE and abs(record['lambda_final'] - 1.5) < 1e-4
    assert record['acceptance']['rule'] == 'window'
    assert record['acceptance']['window'] == [-2.0, 2.0]
    assert record['candidates'] == {'tried': 7, 'rejected': {'lambda_window': 6}}


def _near(positions, x):
    return abs(float(positions[0, 0]) - x) < 5e-3


def cgaimmd_validity_function(positions, context):
    ok = np.array([not _near(p, -0.35) for p in positions])
    reason = np.array(['' if good else 'clash: 0.9 A' for good in ok], dtype=object)
    return ValidityResult(positions=np.array(positions, dtype=float), ok=ok, reason=reason)


def cgaimmd_prepare_function(positions, context):
    if _near(positions, -0.33):
        return PreparedFrame(np.array(positions), None, ok=False)
    return PreparedFrame(np.array(positions), None, record={'prepared': True})


def cgaimmd_minimize_function(frame, context):
    positions = np.array(frame.positions)
    if _near(positions, -0.31):
        return PreparedFrame(positions, None, ok=False, reason='minimize_failed: boom')
    if _near(positions, -0.29):
        positions[:, 0] = 0.7                                   # the minimizer lands in B
    return PreparedFrame(positions, None, record={'steps': 3})


def test_rejection_counts(tmp_path, monkeypatch, capsys):
    """Every reason is counted: per refill (stage 1), per path (stage 2) and cumulatively."""
    fixed = [(lambda value: (lambda target: value))(v) for v in
             (math.nan, math.nan, -3.5, -4.7, -3.3, -3.1, -2.9, -2.9) + (-1.0,) * 8]
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(fixed),
                     lambda_window=(-4.0, 1.0), window_after_trained_shots=10 ** 6,
                     placement='box_center',
                     params_overrides={'cgaimmd_validity_function': cgaimmd_validity_function,
                                       'cgaimmd_prepare_function': cgaimmd_prepare_function,
                                       'cgaimmd_minimize_function': cgaimmd_minimize_function})
    record = _next(setup).record
    assert record['pool']['refill'] == 2                  # refill 1 had no survivor of stage 2
    assert record['stage1_rejected'] == {}
    assert record['candidates'] == {'tried': 5, 'rejected': {'prepare_failed': 1,
                                                             'minimize_failed': 1, 'state_B': 2}}
    assert setup.source.counters == {'non_finite': 2, 'clash': 1, 'lambda_nonfinite': 1,
                                     'prepare_failed': 1, 'minimize_failed': 1, 'state_B': 2}
    function = f'{__name__}.cgaimmd_prepare_function'
    assert record['prepare'] == {'function': function, 'placement': None, 'ok': True, 'reason': '',
                                 'seconds': record['prepare']['seconds'],
                                 'record': {'prepared': True}}
    assert record['minimize']['function'] == f'{__name__}.cgaimmd_minimize_function'
    assert record['minimize']['record'] == {'steps': 3} and record['minimize']['ok'] is True
    assert record['validity'] == {'function': f'{__name__}.cgaimmd_validity_function', 'info': None}
    out = capsys.readouterr().out
    first = _lines(out, 'C-GAIMMD pool refill 1 ')[0]
    assert ('4 passed stage 1 (clash: 1, lambda_nonfinite: 1, non_finite: 2)' in first)


def test_raw_lambda_prefilter(tmp_path, monkeypatch):
    """raw_lambda_tolerance rejects in stage 1 on |lambda_raw - lambda*| once the rule is enforced."""
    script = [lambda target: target + 2.0] * 7
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(script),
                     lambda_window=(-4.0, 1.0), raw_lambda_tolerance=0.5)
    record = _next(setup).record
    assert record['stage1_rejected'] == {'raw_lambda': 7}
    assert record['pool']['passed_stage1'] == 1 and record['pool']['candidate'] == 7
    assert record['candidates'] == {'tried': 1, 'rejected': {}}

    # the bootstrap phase (window not enforced yet): no prefilter either
    boot = _started(tmp_path / 'b', monkeypatch, network=ScriptedNetwork(script),
                    lambda_window=(-4.0, 1.0), raw_lambda_tolerance=0.5,
                    window_after_trained_shots=5, n_trained_shots=4)
    record = _next(boot).record
    assert record['stage1_rejected'] == {} and record['pool']['passed_stage1'] == 8
    assert record['acceptance']['rule'] == 'bootstrap_any_R'
    enforced = _started(tmp_path / 'c', monkeypatch, network=ScriptedNetwork(script),
                        lambda_window=(-4.0, 1.0), raw_lambda_tolerance=0.5,
                        window_after_trained_shots=5, n_trained_shots=5)
    assert _next(enforced).record['stage1_rejected'] == {'raw_lambda': 7}


def test_hooks_get_their_streams(tmp_path, monkeypatch):
    """Validity gets the refill's validity stream, preparation the candidate's own stream."""
    seen = {}

    def validity(positions, context):
        seen['validity'] = (context.seed, context.rng.random())
        return ValidityResult(np.array(positions), np.ones(len(positions), bool),
                              np.array([''] * len(positions), dtype=object),
                              info=[{'k': i} for i in range(len(positions))])

    def prepare(positions, context):
        seen.setdefault('prepare', []).append((context.seed, context.rng.random(), context.path_name,
                                               context.scratch))
        return PreparedFrame(np.array(positions), None)

    def minimize(frame, context):
        seen.setdefault('minimize', []).append(context.seed)
        return frame

    # sample 0 is not finite: the validity function sees samples 1..7, its info[j] is sample j + 1's
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork([lambda target: math.nan]),
                     placement='box_center',
                     params_overrides={'cgaimmd_validity_function': validity,
                                       'cgaimmd_prepare_function': prepare,
                                       'cgaimmd_minimize_function': minimize})
    record = _next(setup).record
    seed = refill_seed(0, FOLDER, PATH, 0)
    validity_rng = refill_streams(seed, 8)[2]
    assert seen['validity'] == (stream_seed(validity_rng), validity_rng.random())
    index = record['seed']['candidate_index']
    expected = preparation_stream(seed, index)
    assert seen['prepare'][-1][:2] == (stream_seed(expected), expected.random())
    assert seen['prepare'][-1][2:] == (PATH, f'{FOLDER}/cgaimmd_scratch')
    assert seen['minimize'][-1] == record['seed']['prep']
    assert index > 0 and record['validity']['info'] == {'k': index - 1}
    assert record['stage1_rejected'] == {'non_finite': 1}


def test_relabelled_positions_enter_the_pool(tmp_path, monkeypatch):
    """The candidates are the validity function's positions (after relabelling)."""
    def relabel(positions, context):
        out = np.array(positions)
        out[:, :, 1] = 0.25                              # 'relabelled': other coordinates
        return ValidityResult(out, np.ones(len(out), bool), np.array([''] * len(out), dtype=object))

    setup = _started(tmp_path, monkeypatch, params_overrides={'cgaimmd_validity_function': relabel})
    point = _next(setup)
    np.testing.assert_allclose(point.positions[:, 1], 0.25, atol=1e-6)
    assert len(setup.source.pool) and all(candidate.positions[0, 1] == 0.25
                                          for candidate in setup.source.pool)


def test_default_placement_reaches_the_shooting_point(tmp_path, monkeypatch):
    """The default placement 'box_center' without a prepare hook: the centered coordinates and the
    box reach the ShootingPoint (what grompp or the toy engine receives) and the final check."""
    import MDAnalysis as mda
    box = np.array([0.6, 0.6, 0.6, 90.0, 90.0, 90.0])        # center x = 0.3: in R (lambda 3)
    universe = mda.Universe.empty(1, trajectory=True)
    universe.dimensions = box
    seen = []

    def states(trajectory):
        seen.extend(None if ts.dimensions is None else ts.dimensions.tolist() for ts in trajectory)
        return x_states_function(trajectory)

    setup = _setup(tmp_path, monkeypatch, window_after_trained_shots=5,
                   params_overrides={'_universe': universe, 'states_function': states})
    setup.params.__dict__['cgaimmd_parameters'] = {
        key: value for key, value in setup.params.cgaimmd_parameters.items() if key != 'placement'}
    assert setup.source.wait_for_network()
    assert setup.source.settings['placement'] == 'box_center'           # the default
    seen.clear()
    point = _next(setup)
    np.testing.assert_allclose(point.positions, [[0.3, 0.3, 0.3]], atol=1e-6)
    assert point.positions.dtype == np.float32
    assert point.dimensions.dtype == np.float32
    np.testing.assert_array_equal(point.dimensions, box.astype(np.float32))
    prepare = point.record['prepare']
    assert prepare['function'] == 'place_in_box' and prepare['placement'] == 'box_center'
    x = point.record['lambda_target'] / GAIN                            # the generated sample
    np.testing.assert_allclose(prepare['record']['translation'], [0.3 - x, 0.3, 0.3], atol=1e-6)
    np.testing.assert_allclose(prepare['record']['box'], box, rtol=1e-6)
    assert seen == [box.astype(np.float32).tolist()]                    # the final check's box


def test_user_sp_cutoffs_reach_the_law_and_the_final_check(tmp_path, monkeypatch):
    """lambda_window='bins' with a user sp_cutoff_min: the targets and the acceptance window."""
    script = _only(LATE, lambda target: -2.0)        # lambda -2: inside the bins' cutoffs only
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(script), lambda_window='bins',
                     acceptance='window',
                     params_overrides={'reweight_parameters': {'sp_cutoff_min': -1.0}})
    save_npy('run/binsARB.npy', np.array([-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf]))
    save_npy('run/densitiesARB.npy', np.full(5, 0.2))
    record = _next(setup).record
    assert record['lambda_law']['window'] == [-1.0, 3.0]              # trainer's (-3, 3), user -1
    assert record['lambda_law']['window_source'] == 'bins'
    assert record['acceptance']['window'] == [-1.0, 3.0]
    assert record['candidates'] == {'tried': 7, 'rejected': {'lambda_window': 6}}
    lam = setup.network.generate_calls[0]['lam'].numpy()
    assert lam.min() >= -1.0 and lam.max() <= 3.0


def test_path_bookkeeping_keeps_one_path_per_folder(tmp_path, monkeypatch):
    """A folder selects one path at a time: the source keeps the state of its current path only."""
    setup = _started(tmp_path, monkeypatch)
    _next(setup)
    _next(setup, 'path000004')
    assert list(setup.source._paths) == [('chainR0', 'path000004')]


def test_network_on_the_values_device_for_the_final_check(tmp_path, monkeypatch):
    """Stage 2 moves the network to the values device right before the final check."""
    events = []
    real_move = cgaimmd_source._move_network
    real_check = cgaimmd_source.final_check
    monkeypatch.setattr(cgaimmd_source, '_move_network',
                        lambda params, network, role: events.append(role)
                        or real_move(params, network, role))
    monkeypatch.setattr(cgaimmd_source, 'final_check',
                        lambda *args: events.append('final_check') or real_check(*args))
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(_only(LATE, lambda t: 70.0)))
    events.clear()
    _next(setup)
    checks = [k for k, event in enumerate(events) if event == 'final_check']
    assert len(checks) == 7 and all(events[k - 1] == 'values' for k in checks)


# ---------------------------------------------------------------------------
# failure policy
# ---------------------------------------------------------------------------

def test_wait_after_max_refills(tmp_path, monkeypatch, capsys):
    """max_refills refills without a shot: WARNING and a wait of wait_seconds; then again."""
    network = ScriptedNetwork([lambda target: 7.0] * 1000)          # all in B
    setup = _started(tmp_path, monkeypatch, network=network)
    capsys.readouterr()

    def sleep(seconds):
        setup.sleeps.append(seconds)
        if sum(setup.sleeps) >= 10:                                 # two full waits
            setup.worker.stop = True

    monkeypatch.setattr(cgaimmd_source, '_sleep', sleep)
    assert _next(setup) is None
    assert len(network.generate_calls) == 6 and sum(setup.sleeps) == 10
    warnings = _lines(capsys.readouterr().out, 'WARNING: no acceptable')
    assert warnings == ['WARNING: no acceptable C-GAIMMD shooting point after 3 pool refills '
                        '(24 candidates: state_B: 24); waiting 5 s'] * 2
    assert setup.source.counters == {'state_B': 48}
    assert len(setup.source._refills) <= 1         # at most the last refill's bookkeeping


def test_waits_poll_must_stop_every_second(tmp_path, monkeypatch):
    """Both waits sleep in polls of at most 1 s and end at the first poll after must_stop."""
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork([lambda target: 7.0] * 1000))

    def sleep_then_stop(seconds):
        setup.sleeps.append(seconds)
        setup.worker.stop = True

    monkeypatch.setattr(cgaimmd_source, '_sleep', sleep_then_stop)
    assert _next(setup) is None
    assert setup.sleeps == [1.0]                          # wait_seconds = 5, one poll

    readback = _started(tmp_path / 'b', monkeypatch, max_refills=1)
    point = _next(readback)

    def sleep_then_stop_readback(seconds):
        readback.sleeps.append(seconds)
        readback.worker.stop = True

    monkeypatch.setattr(cgaimmd_source, '_sleep', sleep_then_stop_readback)
    readback.source.note_glitch(FOLDER, RuntimeError('should be in R'), point.record)
    assert readback.sleeps == [1.0]


def test_must_stop_returns_none(tmp_path, monkeypatch):
    """must_stop before a pop, between generation batches and after a hook: None."""
    setup = _started(tmp_path, monkeypatch)
    setup.worker.stop = True
    assert _next(setup) is None and setup.network.generate_calls == []

    other = _started(tmp_path / 'b', monkeypatch, chunk=4)
    other.network.on_generate = lambda network: setattr(other.worker, 'stop', True)
    assert _next(other) is None
    assert len(other.network.generate_calls) == 1                # the second batch never ran
    assert len(other.source.pool) == 0

    seen = []

    def prepare(positions, context):
        third.worker.stop = True
        seen.append(context.must_stop())
        return PreparedFrame(np.array(positions), None, ok=False, reason='prepare_failed')

    third = _started(tmp_path / 'c', monkeypatch, placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': prepare})
    assert _next(third) is None
    # a hook stopped by must_stop reports ok=False: not a real rejection, not counted
    assert seen == [True] and third.source.counters == {}

    # a hook that succeeds just as must_stop flips: None, not a shooting point; no final check
    checks = []
    monkeypatch.setattr(cgaimmd_source, 'final_check',
                        lambda *args: checks.append(args) or pytest.fail('checked'))
    for hook, placement in (('cgaimmd_prepare_function', 'box_center'),
                            ('cgaimmd_minimize_function', 'none')):
        def stop_and_succeed(value, context):
            late.worker.stop = True
            return (PreparedFrame(np.array(value), None) if hook == 'cgaimmd_prepare_function'
                    else value)

        late = _started(tmp_path / hook, monkeypatch, placement=placement,
                        params_overrides={hook: stop_and_succeed})
        assert _next(late) is None, hook
        assert late.source.counters == {}
    assert checks == []


def test_oom_halves_the_batch(tmp_path, monkeypatch, capsys):
    """CUDA OOM: halve the batch and retry the refill from its seed; the batch stays halved."""
    setup = _started(tmp_path, monkeypatch, chunk=8)
    setup.network.oom_above = 2
    capsys.readouterr()
    record = _next(setup).record
    assert [call['n'] for call in setup.network.generate_calls] == [8, 4, 2, 2, 2, 2]
    assert record['generator'] == {'method': SCRIPTED, 'settings': {'chunk': 8},
                                   'batch_size': 2}
    assert _lines(capsys.readouterr().out, 'C-GAIMMD generation ran out') == [
        'C-GAIMMD generation ran out of CUDA memory (batch 8); retrying with batch 4',
        'C-GAIMMD generation ran out of CUDA memory (batch 4); retrying with batch 2']
    while len(setup.source.pool):
        setup.source.pool.pop()
    _next(setup, 'path000004')
    assert setup.network.generate_calls[-1]['n'] == 2             # kept for later refills
    assert not _lines(capsys.readouterr().out, 'C-GAIMMD generation ran out')

    # the 'ode' generator: the retried refill equals a refill made at the smaller batch
    def tiny(**kwargs):
        return OOMPriorNetwork(limit=4, prior_center=0.0, prior_width=0.1, lambda_clip=10.0,
                               **kwargs)
    torch.manual_seed(0)
    retried = _started(tmp_path / 'b', monkeypatch, network=tiny(), chunk=8,
                       lambda_window=(-0.5, 0.5))
    torch.manual_seed(0)
    direct = _started(tmp_path / 'c', monkeypatch, network=tiny(), chunk=4,
                      lambda_window=(-0.5, 0.5))
    a, b = _next(retried), _next(direct)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.record['seed'] == b.record['seed']
    assert a.record['generator']['method'] == 'ode'
    assert a.record['generator']['batch_size'] == b.record['generator']['batch_size'] == 4
    assert a.record['generator']['settings']['chunk'] == 8           # the setting, as given

    # out of memory after every halving: the refill is empty ('generation_oom')
    starved = _started(tmp_path / 'd', monkeypatch, generation_oom_halvings=1, max_refills=2)
    starved.network.oom_above = 0
    monkeypatch.setattr(cgaimmd_source, '_sleep',
                        lambda seconds: setattr(starved.worker, 'stop', True))
    capsys.readouterr()
    assert _next(starved) is None
    assert starved.source.counters == {'generation_oom': 16}
    out = capsys.readouterr().out
    # the WARNING lists the stage-1 reasons too, not '(16 candidates: none)'
    assert _lines(out, 'WARNING: no acceptable') == [
        'WARNING: no acceptable C-GAIMMD shooting point after 2 pool refills (16 candidates: '
        'generation_oom: 16); waiting 5 s']
    assert _lines(out, 'C-GAIMMD generation ran out') == [
        'C-GAIMMD generation ran out of CUDA memory (batch 128); retrying with batch 64',
        'C-GAIMMD generation ran out of CUDA memory (batch 64); retrying with batch 32']
    assert [call['n'] for call in starved.network.generate_calls] == [8, 8, 8, 8]
    assert starved.source._chunk == 32
    assert starved.source._refills == {}          # no bookkeeping kept for empty refills


def test_oom_as_runtime_error_text(tmp_path, monkeypatch, capsys):
    """A RuntimeError whose text says 'out of memory' is a CUDA OOM as well (batch halving)."""
    setup = _started(tmp_path, monkeypatch, chunk=8)

    def out_of_memory(network):
        if network.generate_calls[-1]['n'] > 2:
            raise RuntimeError('CUDA error: out of memory')

    setup.network.on_generate = out_of_memory
    capsys.readouterr()
    assert _next(setup) is not None
    assert [call['n'] for call in setup.network.generate_calls] == [8, 4, 2, 2, 2, 2]
    assert len(_lines(capsys.readouterr().out, 'C-GAIMMD generation ran out of CUDA')) == 2


def test_generator_without_a_chunk_setting_gets_the_whole_refill(tmp_path, monkeypatch):
    """A generator whose settings have no 'chunk' is called once per refill."""
    class Whole(ScriptedGenerator):
        method = 'test_whole_refill'

        def settings(self):
            return {}

    register_generator('test_whole_refill', Whole)
    setup = _started(tmp_path, monkeypatch, generator='test_whole_refill', n_targets=6)
    record = _next(setup).record
    assert [call['n'] for call in setup.network.generate_calls] == [12]
    assert record['generator'] == {'method': 'test_whole_refill', 'settings': {},
                                   'batch_size': 12}


def not_a_frame(positions, context):
    return np.array(positions)


def minimize_not_a_frame(frame, context):
    return frame.positions


def validity_not_a_result(positions, context):
    return positions


def validity_wrong_shape(positions, context):
    return ValidityResult(np.array(positions), np.ones(len(positions) + 1, bool),
                          np.array([''] * len(positions), dtype=object))


def test_bugs_in_hooks_and_generators_propagate(tmp_path, monkeypatch):
    """Hooks, validity results and generators that break the contract raise."""
    cases = [
        ({'placement': 'box_center', 'params_overrides': {'cgaimmd_prepare_function': not_a_frame}},
         TypeError, 'cgaimmd_prepare_function must return'),
        ({'params_overrides': {'cgaimmd_minimize_function': minimize_not_a_frame}},
         TypeError, 'cgaimmd_minimize_function must return'),
        ({'params_overrides': {'cgaimmd_validity_function': validity_not_a_result}},
         TypeError, 'ValidityResult'),
        ({'params_overrides': {'cgaimmd_validity_function': validity_wrong_shape}},
         ValueError, 'cgaimmd_validity_function returned'),
    ]
    for k, (kwargs, error, match) in enumerate(cases):
        setup = _started(tmp_path / f'hook{k}', monkeypatch, **kwargs)
        with pytest.raises(error, match=match):
            _next(setup)

    shaped = _started(tmp_path / 'shape', monkeypatch)
    shaped.network.broken = True
    with pytest.raises(ValueError, match="the generator 'test_scripted' returned shape"):
        _next(shaped)

    def bug(network):
        raise RuntimeError('a bug in the generator')

    failing = _started(tmp_path / 'bug', monkeypatch)
    failing.network.on_generate = bug
    with pytest.raises(RuntimeError, match='a bug in the generator'):
        _next(failing)                                  # not an OOM: no halving, no rejection
    assert failing.source.counters == {} and len(failing.network.generate_calls) == 1


def test_note_glitch_counts_state_readback(tmp_path, monkeypatch, capsys):
    """A read-back rejection counts in the next shot's stage-2 counters; the pool is kept."""
    setup = _started(tmp_path, monkeypatch)
    point = _next(setup)
    left = len(setup.source.pool)
    capsys.readouterr()
    setup.source.note_glitch(FOLDER, RuntimeError('frame 0 should be in R'), point.record)
    assert setup.source.counters == {'state_readback': 1}
    assert setup.source.consecutive_readback == 1
    assert _lines(capsys.readouterr().out, '...') == [
        '... C-GAIMMD: candidate discarded, next one from the pool']
    assert len(setup.source.pool) == left
    record = _next(setup).record
    assert record['candidates'] == {'tried': 2, 'rejected': {'state_readback': 1}}
    assert record['seed']['refill'] == point.record['seed']['refill']   # the same refill
    setup.source.note_glitch(FOLDER, RuntimeError('should be in R'), None)   # no record: this path
    assert _next(setup).record['candidates'] == {'tried': 3, 'rejected': {'state_readback': 2}}
    setup.source.note_registered()
    assert setup.source.consecutive_readback == 0
    assert _next(setup, 'path000004').record['candidates'] == {'tried': 1, 'rejected': {}}

    # a shot resumed after a restart (selected by an earlier process) counts as tried
    resumed = {'path_name': 'path000005', 'source': 'cgaimmd'}
    setup.source.note_glitch(FOLDER, RuntimeError('should be in R'), resumed)
    assert _next(setup, 'path000005').record['candidates'] == {
        'tried': 2, 'rejected': {'state_readback': 1}}


def test_note_glitch_after_a_restart(tmp_path, monkeypatch):
    """A resumed shot's path continues its refill counter; a record without a path name is kept."""
    first = _started(tmp_path / 'a', monkeypatch)
    point = _next(first)
    assert point.record['seed']['refill_index'] == 0

    # the restarted worker: the resumed shot fails the read-back check; the next refill of the
    # path is not refill 0 again (which would pop the same candidate first)
    restarted = _started(tmp_path / 'a', monkeypatch)
    restarted.source.note_glitch(FOLDER, RuntimeError('should be in R'), point.record)
    record = _next(restarted).record
    assert record['seed']['refill_index'] == 1
    assert record['seed']['refill'] == refill_seed(0, FOLDER, PATH, 1)
    assert record['candidates'] == {'tried': 2, 'rejected': {'state_readback': 1}}

    # a candidate the pool carried over from an earlier path's refill: the record names that
    # refill's path; after a restart the path starts its own refills at 0 (it made none)
    carrier = _started(tmp_path / 'd', monkeypatch)
    _next(carrier)
    carried = _next(carrier, 'path000004').record
    assert carried['path_name'] == 'path000004' and carried['pool']['refill'] == 1
    assert carried['seed']['refill_path'] == PATH and carried['seed']['refill_index'] == 0
    assert carried['seed']['refill'] == refill_seed(0, FOLDER, PATH, 0)
    again = _started(tmp_path / 'd', monkeypatch)
    again.source.note_glitch(FOLDER, RuntimeError('should be in R'), carried)
    record = _next(again, 'path000004').record
    assert record['seed']['refill_path'] == 'path000004' and record['seed']['refill_index'] == 0
    assert record['seed']['refill'] == refill_seed(0, FOLDER, 'path000004', 0)

    # an unreadable in-flight marker has no path name: counted for the folder's next path
    fresh = _started(tmp_path / 'b', monkeypatch)
    unreadable = {'schema': 'aimmd.cgaimmd.shot/1', 'source': 'unknown',
                  'note': 'unreadable marker'}
    fresh.source.note_glitch(FOLDER, RuntimeError('should be in R'), unreadable)
    fresh.source.note_glitch(FOLDER, RuntimeError('should be in R'), None)
    record = _next(fresh).record
    assert record['candidates'] == {'tried': 2, 'rejected': {'state_readback': 2}}
    assert record['seed']['refill_index'] == 0

    # before the start-up: a bug of the caller
    idle = _setup(tmp_path / 'c', monkeypatch)
    with pytest.raises(RuntimeError, match='wait_for_network'):
        idle.source.note_glitch(FOLDER, RuntimeError('should be in R'), None)


def test_readback_wait_after_max_refills(tmp_path, monkeypatch, capsys):
    """max_refills consecutive read-back rejections: WARNING and a wait; a registered shot resets."""
    setup = _started(tmp_path, monkeypatch, max_refills=3, wait_seconds=2.0)
    point = _next(setup)
    capsys.readouterr()
    for _ in range(2):
        setup.source.note_glitch(FOLDER, RuntimeError('should be in R'), point.record)
    setup.source.note_registered()
    for _ in range(2):
        setup.source.note_glitch(FOLDER, RuntimeError('should be in R'), point.record)
    assert setup.sleeps == []
    setup.source.note_glitch(FOLDER, RuntimeError('should be in R'), point.record)
    assert sum(setup.sleeps) == 2.0 and setup.source.consecutive_readback == 0
    assert _lines(capsys.readouterr().out, 'WARNING') == [
        'WARNING: 3 consecutive C-GAIMMD shooting points failed the read-back state check '
        '(states_function on back.xtc frame 0); waiting 2 s']


# ---------------------------------------------------------------------------
# pool, seeds, laws
# ---------------------------------------------------------------------------

def test_pool_reset_on_network_change(tmp_path, monkeypatch, capsys):
    """A new network file before a pop: the pool is reset (logged) and refilled by the new network."""
    setup = _started(tmp_path, monkeypatch)
    first = _next(setup)
    left = len(setup.source.pool)
    assert left > 0
    newer = ScriptedNetwork()
    with torch.no_grad():
        for parameter in newer.velocity_net.parameters():
            parameter.fill_(0.5)
    sha = write_fake_network_file('run', newer, round=2, n_trained_shots=6)
    capsys.readouterr()
    record = _next(setup, 'path000004').record
    assert _lines(capsys.readouterr().out, 'C-GAIMMD pool reset') == [
        f'C-GAIMMD pool reset: network changed {setup.sha[:8]} -> {sha[:8]} (round 2, 6 trained '
        f'shots); {left} unused candidates discarded']
    assert record['network']['sha256'] == sha and record['network']['round'] == 2
    assert record['pool']['refill'] == 2 and record['pool']['reset_count'] == 1
    assert first.record['network']['sha256'] == setup.sha


def _write_other_network(value, **metadata):
    """A network file with weights filled with `value` (a new sha256 for every value)."""
    other = ScriptedNetwork()
    with torch.no_grad():
        for parameter in other.velocity_net.parameters():
            parameter.fill_(value)
    return write_fake_network_file('run', other, **metadata)


def test_network_change_during_a_call_resets_before_the_next_pop(tmp_path, monkeypatch):
    """The watcher is refreshed before every pop, also between the pops of one call."""
    shas = []

    def prepare(positions, context):
        if not shas:                                  # the trainer replaces the network meanwhile
            shas.append(_write_other_network(0.5, round=2, n_trained_shots=6))
            return PreparedFrame(np.array(positions), None, ok=False, reason='prepare_failed')
        return PreparedFrame(np.array(positions), None)

    setup = _started(tmp_path, monkeypatch, placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': prepare})
    record = _next(setup).record
    assert record['network']['sha256'] == shas[0]
    assert record['pool']['refill'] == 2 and record['pool']['reset_count'] == 1
    assert record['candidates'] == {'tried': 2, 'rejected': {'prepare_failed': 1}}
    assert setup.source.counters == {'prepare_failed': 1, 'pool_reset': 7}


def _change_network_during_first_refill(setup):
    """The trainer publishes a new network while the source's first refill generates."""
    shas = []

    def once_during_generation(network):
        if not shas:
            shas.append(_write_other_network(0.5, round=2, n_trained_shots=6))

    setup.network.on_generate = once_during_generation
    return shas


def test_refill_outdated_while_generated_serves_one_pop(tmp_path, monkeypatch, capsys):
    """A network written while a refill runs: the refill serves one pop, then the pool resets."""
    setup = _started(tmp_path, monkeypatch)
    shas = _change_network_during_first_refill(setup)
    capsys.readouterr()
    record = _next(setup).record
    out = capsys.readouterr().out
    assert _lines(out, 'C-GAIMMD pool: network changed') == [
        f'C-GAIMMD pool: network changed {setup.sha[:8]} -> {shas[0][:8]} (round 2, 6 trained shots) '
        f'while refill 1 was generated; its 8 candidates serve 1 pop before the pool resets']
    assert not _lines(out, 'C-GAIMMD pool reset')
    assert record['network']['sha256'] == shas[0]          # the final check's network: the new one
    assert record['network']['round'] == 2 and record['acceptance']['n_trained_shots'] == 6
    assert record['pool']['stale_refill_pop'] is True
    assert record['pool']['refill_network_sha256'] == setup.sha          # the refill's network
    assert record['pool']['refill'] == 1 and record['pool']['reset_count'] == 0
    assert record['pool']['pool_left_after_pop'] == 7
    assert setup.source.counters == {}
    assert len(setup.network.generate_calls) == 1

    # the next pop: the pool resets as before, the refill's 7 other candidates are discarded
    again = _next(setup, 'path000004').record
    assert _lines(capsys.readouterr().out, 'C-GAIMMD pool reset') == [
        f'C-GAIMMD pool reset: network changed {setup.sha[:8]} -> {shas[0][:8]} (round 2, 6 trained '
        f'shots); 7 unused candidates discarded']
    assert again['pool']['stale_refill_pop'] is False
    assert again['pool']['refill_network_sha256'] == again['network']['sha256'] == shas[0]
    assert again['pool']['refill'] == 2 and again['pool']['reset_count'] == 1
    assert setup.source.counters == {'pool_reset': 7}
    assert len(setup.network.generate_calls) == 2


def _failing_once():
    """A prepare function that rejects its first candidate and passes the others."""
    calls = []

    def prepare(positions, context):
        calls.append(context.path_name)
        if len(calls) == 1:
            return PreparedFrame(np.array(positions), None, ok=False, reason='prepare_failed')
        return PreparedFrame(np.array(positions), None)
    return prepare, calls


def test_stale_pop_rejected_then_the_pool_resets(tmp_path, monkeypatch):
    """A stale pop that fails stage 2: the next candidate of the call comes from a fresh refill."""
    prepare, calls = _failing_once()
    setup = _started(tmp_path, monkeypatch, placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': prepare})
    shas = _change_network_during_first_refill(setup)
    record = _next(setup).record
    assert len(calls) == 2
    assert record['pool']['stale_refill_pop'] is False
    assert record['pool']['refill'] == 2 and record['pool']['reset_count'] == 1
    assert record['pool']['refill_network_sha256'] == record['network']['sha256'] == shas[0]
    assert record['candidates'] == {'tried': 2, 'rejected': {'prepare_failed': 1}}
    assert setup.source.counters == {'prepare_failed': 1, 'pool_reset': 7}


def test_network_change_during_a_stale_pop_resets_at_once(tmp_path, monkeypatch, capsys):
    """A second network update while the stale pop is in stage 2: the next check resets the pool."""
    second = []

    def prepare(positions, context):
        if not second:                                 # the trainer publishes again meanwhile
            second.append(_write_other_network(0.7, round=3, n_trained_shots=7))
            return PreparedFrame(np.array(positions), None, ok=False, reason='prepare_failed')
        return PreparedFrame(np.array(positions), None)

    setup = _started(tmp_path, monkeypatch, placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': prepare})
    shas = _change_network_during_first_refill(setup)
    capsys.readouterr()
    record = _next(setup).record
    assert _lines(capsys.readouterr().out, 'C-GAIMMD pool reset') == [
        f'C-GAIMMD pool reset: network changed {shas[0][:8]} -> {second[0][:8]} (round 3, 7 trained '
        f'shots); 7 unused candidates discarded']
    assert record['network']['sha256'] == second[0] and record['pool']['stale_refill_pop'] is False
    assert record['pool']['refill'] == 2 and record['pool']['reset_count'] == 1
    assert setup.source.counters == {'prepare_failed': 1, 'pool_reset': 7}


def test_network_changing_during_every_refill(tmp_path, monkeypatch, capsys):
    """Every refill outdated while it runs: one stale pop each; the WARNING counts the rest."""
    def always_fails(positions, context):
        return PreparedFrame(np.array(positions), None, ok=False, reason='prepare_failed')

    starved = _started(tmp_path, monkeypatch, placement='box_center',
                       params_overrides={'cgaimmd_prepare_function': always_fails})
    starved.network.on_generate = lambda network: _write_other_network(
        0.1 * len(starved.network.generate_calls))
    monkeypatch.setattr(cgaimmd_source, '_sleep', lambda seconds: setattr(starved.worker, 'stop', True))
    capsys.readouterr()
    assert _next(starved) is None
    out = capsys.readouterr().out
    assert len(_lines(out, 'C-GAIMMD pool: network changed')) == 3
    assert _lines(out, 'WARNING: no acceptable') == [
        'WARNING: no acceptable C-GAIMMD shooting point after 3 pool refills (24 candidates: '
        'pool_reset: 21, prepare_failed: 3); waiting 5 s']
    assert starved.source.counters == {'pool_reset': 21, 'prepare_failed': 3}
    assert starved.source.pool.reset_count == 3

    # and with an accepting prepare function the first stale pop is the shot
    fed = _started(tmp_path / 'b', monkeypatch)
    fed.network.on_generate = lambda network: _write_other_network(
        0.1 * len(fed.network.generate_calls))
    record = _next(fed).record
    assert record['pool']['stale_refill_pop'] is True and record['pool']['refill'] == 1


def test_pi4_strictly_without_stale_pops(tmp_path, monkeypatch, capsys):
    """STALE_REFILL_POPS = 0: a refill outdated while it runs is discarded unused (strict resets)."""
    monkeypatch.setattr(cgaimmd_source, 'STALE_REFILL_POPS', 0)
    setup = _started(tmp_path, monkeypatch)
    shas = _change_network_during_first_refill(setup)
    record = _next(setup).record
    assert record['network']['sha256'] == shas[0] and record['pool']['stale_refill_pop'] is False
    assert record['pool']['refill'] == 2 and record['pool']['reset_count'] == 1
    assert setup.source.counters == {'pool_reset': 8}
    assert len(setup.network.generate_calls) == 2

    # a network that changes during every refill: no shot, and the WARNING says why
    starved = _started(tmp_path / 'b', monkeypatch)
    starved.network.on_generate = lambda network: _write_other_network(
        0.1 * len(starved.network.generate_calls))
    monkeypatch.setattr(cgaimmd_source, '_sleep', lambda seconds: setattr(starved.worker, 'stop', True))
    capsys.readouterr()
    assert _next(starved) is None
    out = capsys.readouterr().out
    assert not _lines(out, 'C-GAIMMD pool: network changed')
    assert _lines(out, 'WARNING: no acceptable') == [
        'WARNING: no acceptable C-GAIMMD shooting point after 3 pool refills (24 candidates: '
        'pool_reset: 24); waiting 5 s']
    assert starved.source.counters == {'pool_reset': 24} and starved.source.pool.reset_count == 3


def test_seeds_reproducible(tmp_path, monkeypatch):
    """Same generation_seed, folder and path: the same shooting point; refills count per path."""
    networks = []
    for _ in range(2):
        torch.manual_seed(0)
        networks.append(TinyFlowNetwork(prior_center=0.0, prior_width=0.1, lambda_clip=10.0))
    a = _started(tmp_path / 'a', monkeypatch, network=networks[0], lambda_window=(-0.5, 0.5))
    b = _started(tmp_path / 'b', monkeypatch, network=networks[1], lambda_window=(-0.5, 0.5))
    pa, pb = _next(a), _next(b)
    np.testing.assert_array_equal(pa.positions, pb.positions)
    assert pa.record['lambda_target'] == pb.record['lambda_target']

    # the second refill of one path uses refill index 1 of the seed formula
    while len(a.source.pool):
        a.source.pool.pop()
    again = _next(a)
    assert again.record['seed']['refill'] == refill_seed(0, FOLDER, PATH, 1)
    assert again.record['seed']['refill_index'] == 1 and again.record['pool']['refill'] == 2

    # a new path counts its own refills: index 0 of its seed, the worker's refill 3
    while len(a.source.pool):
        a.source.pool.pop()
    other = _next(a, 'path000004')
    assert other.record['seed']['refill'] == refill_seed(0, FOLDER, 'path000004', 0)
    assert other.record['seed']['refill_index'] == 0 and other.record['pool']['refill'] == 3

    # without generation_seed the run directory decides
    c = _started(tmp_path / 'c', monkeypatch, generation_seed=None)
    d = _started(tmp_path / 'd', monkeypatch, generation_seed=None)
    assert _next(c).record['seed']['base'] != _next(d).record['seed']['base']


def test_bins_law_gets_loaded_bins_and_chains(tmp_path, monkeypatch):
    """Bins and densities are loaded (timeout 0) before each refill; the law gets the chains."""
    calls = []
    real = lambda_law.draw_targets

    def spy(n, context, *, chain=None, shooting_chains=()):
        calls.append({'bins': context.bins, 'densities': context.densities, 'chain': chain,
                      'shooting_chains': list(shooting_chains), 'rng': context.rng,
                      'seed': context.seed})
        return real(n, context, chain=chain, shooting_chains=shooting_chains)

    monkeypatch.setattr(cgaimmd_source, 'draw_targets', spy)
    setup = _started(tmp_path, monkeypatch, lambda_law='bins', lambda_window='bins')
    record = _next(setup).record
    assert calls[0]['bins'] is None and record['lambda_law']['name'] == 'bins (uniform: no bins yet)'
    assert record['lambda_law']['window'] == [-10.0, 10.0]
    assert calls[0]['chain'] is setup.chain and calls[0]['shooting_chains'] == [setup.chain]
    seed = refill_seed(0, FOLDER, PATH, 0)
    assert calls[0]['seed'] == stream_seed(refill_streams(seed, 8)[0])

    bins = np.array([-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf])
    save_npy('run/binsARB.npy', bins)
    save_npy('run/densitiesARB.npy', np.full(5, 0.2))
    while len(setup.source.pool):
        setup.source.pool.pop()
    record = _next(setup, 'path000004').record
    np.testing.assert_array_equal(calls[-1]['bins'], bins)
    assert record['lambda_law']['name'] == 'bins'
    assert record['lambda_law']['window'] == [-3.0, 3.0]        # the trainer's sp cutoffs
    assert -3.0 <= record['lambda_target'] <= 3.0

    save_npy('run/densitiesARB.npy', np.full(3, 0.2))            # a mismatched pair: no bins
    while len(setup.source.pool):
        setup.source.pool.pop()
    record = _next(setup, 'path000005').record
    assert calls[-1]['bins'] is None


def test_generation_device_restored_after_the_values_function(tmp_path, monkeypatch):
    """The law and the validity hook may call the values function (which moves the network):
    the network is moved to the generation device after them, before generation and raw lambda."""
    events = []
    real_move = cgaimmd_source._move_network
    real_committor = cgaimmd_source.coordinate_committor

    def move(params, network, role):
        events.append(('move', role))
        return real_move(params, network, role)

    def committor(*args, **kwargs):
        events.append(('raw_lambda',))
        return real_committor(*args, **kwargs)

    def values_function(descriptors):
        events.append(('values_function',))      # the protocol values function moves the network here
        return GAIN * np.asarray(descriptors, dtype=float).reshape(len(descriptors), -1)[:, 0]

    def validity(positions, context):
        setup.params.values_function(positions.reshape(len(positions), -1))
        return ValidityResult(np.array(positions), np.ones(len(positions), bool),
                              np.array([''] * len(positions), dtype=object))

    monkeypatch.setattr(cgaimmd_source, '_move_network', move)
    monkeypatch.setattr(cgaimmd_source, 'coordinate_committor', committor)
    network = ScriptedNetwork(on_generate=lambda net: events.append(('generate',)))
    setup = _started(tmp_path, monkeypatch, network=network, lambda_law='bins',
                     params_overrides={'shared_density_adjustment': True, 'nbins': 5,
                                       'values_function': values_function,
                                       'cgaimmd_validity_function': validity})
    save_npy('run/binsARB.npy', np.array([-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf]))
    save_npy('run/densitiesARB.npy', np.full(5, 0.2))
    events.clear()
    assert _next(setup).record['lambda_law']['name'] == 'bins'
    generate = events.index(('generate',))
    assert ('values_function',) in events[:generate]            # the shared density adjustment
    assert events[generate - 1] == ('move', 'generation')
    raw = events.index(('raw_lambda',))
    assert events[generate + 1:raw] == [('values_function',), ('move', 'generation')]


def _strict_loads(text):
    """json.loads that rejects Infinity and NaN, as strict JSON readers (jq, JavaScript) do."""
    def constant(name):
        raise ValueError(f'not strict JSON: {name}')
    return json.loads(text, parse_constant=constant)


def test_record_is_a_sidecar_record(tmp_path, monkeypatch):
    """The record is strict JSON, carries every sidecar key registration does not add, registers."""
    setup = _started(tmp_path, monkeypatch, lambda_law='bins')
    save_npy('run/binsARB.npy', np.array([-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf]))
    save_npy('run/densitiesARB.npy', np.full(5, 0.2))
    point = _next(setup, 'path000002')
    record = point.record
    assert record['lambda_law']['name'] == 'bins'                # infinite edges, lorentzian inf
    json.dumps(record, allow_nan=False)
    assert set(SIDECAR_KEYS) - set(REGISTRATION_KEYS) <= set(record)
    assert not set(REGISTRATION_KEYS) & set(record)
    write_in_flight(FOLDER, record, point.positions)
    path = aimmd.Path(f'{FOLDER}/path000002.xtc', shooting_index=3)
    write_shot_sidecar(path, record, params=setup.params, states='ARB', target_state='R')
    sidecar = read_shot_sidecar(f'{FOLDER}/path000002.xtc')
    assert sidecar['source'] == 'cgaimmd' and sidecar['lambda_target'] == record['lambda_target']
    assert set(sidecar) >= set(SIDECAR_KEYS)
    with open(f'{FOLDER}/path000002.xtc.cgaimmd.json') as handle:
        assert _strict_loads(handle.read())['lambda_law']['bins'][0] == '-inf'


# ---------------------------------------------------------------------------
# final check and placement
# ---------------------------------------------------------------------------

def _check_context(tmp_path, **settings):
    settings.setdefault('lambda_clip', 10.0)
    settings.setdefault('window_after_trained_shots', 0)
    settings.setdefault('lambda_window', (-6.0, 6.0))
    return make_candidate_context(tmp_path, **settings)


def test_final_check_uses_values_function(tmp_path, monkeypatch):
    """lambda is params.values_function of params.descriptors_function; the rule follows N."""
    calls = []

    def values(descriptors):
        calls.append(np.array(descriptors))
        return np.full(len(descriptors), 0.75)

    context = _check_context(tmp_path)
    context.params.__dict__['values_function'] = values

    def no_forward(*args, **kwargs):
        raise AssertionError('final_check must go through params.values_function')

    monkeypatch.setattr(context.network, 'forward', no_forward)
    frame = PreparedFrame(np.array([[0.1, 0.0, 0.0]]), None)
    check = final_check(frame, context, 0.5)
    assert check['ok'] and check['reason'] == '' and check['lambda'] == 0.75
    assert check['state'] == 'R' and check['rule'] == 'target' and check['enforced']
    assert check['n_trained_shots'] == 0 and check['window'] == [-6.0, 6.0]
    assert check['tolerance'] == 1.0 and check['window_after_trained_shots'] == 0
    np.testing.assert_allclose(calls[0], [[0.1, 0.0, 0.0]], atol=1e-6)
    assert set(check) == {'ok', 'reason', 'state', 'lambda', 'rule', 'enforced', 'tolerance',
                          'window', 'window_after_trained_shots', 'n_trained_shots', 'decimals',
                          'seconds'}

    assert final_check(frame, context, 2.0)['reason'] == 'lambda_window'
    in_b = final_check(PreparedFrame(np.array([[0.7, 0.0, 0.0]]), None), context, 0.5)
    assert (in_b['ok'], in_b['reason'], in_b['state'], in_b['lambda']) == (False, 'state_B', 'B',
                                                                          None)
    assert in_b['rule'] == 'target' and in_b['window'] == [-6.0, 6.0] and len(calls) == 2

    context.params.__dict__['values_function'] = lambda d: np.full(len(d), np.nan)
    assert final_check(frame, context, 0.5)['reason'] == 'lambda_nonfinite'

    boot = _check_context(tmp_path / 'b', window_after_trained_shots=5)
    boot.params.__dict__['values_function'] = values
    boot.network_info = NetworkInfo(file='f', sha256='s', round=1, n_trained_shots=3, metadata={},
                                    metadata_matched=True, loaded_at=0.0)
    check = final_check(frame, boot, 4.0)
    assert check['ok'] and check['rule'] == 'bootstrap_any_R' and not check['enforced']
    assert check['n_trained_shots'] == 3
    window = _check_context(tmp_path / 'c', acceptance='window', lambda_window=(-1.0, 0.5))
    window.params.__dict__['values_function'] = values
    check = final_check(frame, window, 0.0)
    assert (check['ok'], check['rule'], check['window']) == (False, 'window', [-1.0, 0.5])
    binned = _check_context(tmp_path / 'd', acceptance='window', lambda_window='bins')
    binned.params.__dict__['values_function'] = values
    binned.bins = np.array([-np.inf, -1.0, 0.5, np.inf])     # the trainer's cutoffs: -1 and 0.5
    check = final_check(frame, binned, 0.0)
    assert (check['ok'], check['reason'], check['window']) == (False, 'lambda_window', [-1.0, 0.5])


def test_final_check_rejects_an_unknown_acceptance(tmp_path):
    """An acceptance rule outside ACCEPTANCE_RULES raises, enforced or not (it never passes all R)."""
    assert cgaimmd_source.ACCEPTANCE_RULES == ('target', 'window')
    frame = PreparedFrame(np.array([[0.0, 0.0, 0.0]]), None)
    for after in (0, 5):                                     # enforced, bootstrap phase
        context = _check_context(tmp_path, window_after_trained_shots=after)
        context.settings['acceptance'] = 'windows'
        with pytest.raises(ValueError, match="unknown acceptance 'windows'"):
            final_check(frame, context, -9.0)


def test_final_check_rounds_to_xtc_precision(tmp_path):
    """'auto' rounds to 2 decimals (0.01 A) for .xtc frames of ten or more atoms, not for other
    formats and not for frames of at most nine atoms (XTC stores those as plain floats, as an
    MDAnalysis round trip shows); ints and None too."""
    import MDAnalysis as mda
    seen = []

    def descriptors(trajectory):
        out = np.array([ts.positions.reshape(-1) for ts in trajectory])
        seen.append(out)
        return out

    for n_atoms, compressed in ((9, False), (10, True)):           # what XTC itself does
        universe = mda.Universe.empty(n_atoms, trajectory=True)
        universe.atoms.positions = np.full((n_atoms, 3), 0.123456, dtype=np.float32)
        fname = str(tmp_path / f'atoms{n_atoms}.xtc')
        with mda.Writer(fname, n_atoms) as writer:
            writer.write(universe.atoms)
        stored = mda.coordinates.XTC.XTCReader(fname)[0].positions[0, 0]
        assert abs(stored - (0.12 if compressed else 0.123456)) < 1e-6

    large = np.zeros((10, 3))
    large[0, 0] = 0.123456
    small = large[:1]
    expected = {('.xtc', 'auto', 10): (2, 0.12), ('.xtc', 'auto', 1): (None, 0.123456),
                ('.trr', 'auto', 10): (None, 0.123456), ('.xtc', 3, 10): (3, 0.123),
                ('.xtc', 3, 1): (3, 0.123), ('.xtc', None, 10): (None, 0.123456)}
    for (ext, setting, n_atoms), (decimals, x) in expected.items():
        frame = PreparedFrame(large if n_atoms == 10 else small, None)
        context = _check_context(tmp_path, final_check_round=setting)
        context.params.__dict__.update(trajectory_extension=ext, descriptors_function=descriptors)
        check = final_check(frame, context, 0.0)
        assert check['decimals'] == decimals, (ext, setting, n_atoms)
        assert seen[-1].dtype == np.float32
        assert abs(float(seen[-1][0, 0]) - x) < 1e-6


def test_final_check_passes_a_chainreader(tmp_path):
    """states_function and descriptors_function get the reader type Path.compute hands them."""
    seen = []

    def states(trajectory):
        seen.append(('states', type(trajectory), len(trajectory),
                     [(ts.positions.dtype, ts.positions.shape, None if ts.dimensions is None
                       else ts.dimensions.tolist()) for ts in trajectory]))
        return np.array(['R'] * len(trajectory))

    def descriptors(trajectory):
        seen.append(('descriptors', type(trajectory), len(trajectory)))
        return np.array([ts.positions.reshape(-1) for ts in trajectory], dtype=np.float32)

    context = _check_context(tmp_path)
    context.params.__dict__.update(states_function=states, descriptors_function=descriptors)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    final_check(PreparedFrame(np.array([[0.1, 0.2, 0.3]]), box), context, 0.0)
    assert seen[0][:3] == ('states', ChainReader, 1)
    assert seen[0][3] == [(np.float32, (1, 3), box.tolist())]
    assert seen[1] == ('descriptors', ChainReader, 1)


def test_placement_box_center_and_none(tmp_path):
    """'box_center': centroid to the box center (orthorhombic only); 'none': coordinates kept."""
    import MDAnalysis as mda
    universe = mda.Universe.empty(2, trajectory=True)
    universe.dimensions = [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]
    params = cgaimmd_params_namespace()
    params.__dict__['_universe'] = universe
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frame = place_in_box(positions, params, 'box_center')
    np.testing.assert_allclose(frame.positions, [[9.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    np.testing.assert_allclose(frame.dimensions, [20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    assert frame.ok and frame.record == {'placement': 'box_center',
                                         'box': [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
                                         'translation': [9.0, 10.0, 10.0]}
    kept = place_in_box(positions, params, 'none')
    np.testing.assert_array_equal(kept.positions, positions)
    np.testing.assert_allclose(kept.dimensions, [20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    assert kept.record['translation'] is None
    with pytest.raises(ValueError, match='cgaimmd_prepare_function'):
        place_in_box(np.zeros((3, 3)), params, 'none')
    with pytest.raises(ValueError, match='placement'):
        place_in_box(positions, params, 'somewhere')

    universe.dimensions = [20.0, 20.0, 20.0, 60.0, 90.0, 90.0]
    with pytest.raises(TypeError, match='orthorhombic'):
        place_in_box(positions, params, 'box_center')
    params.__dict__['_universe'] = None
    with pytest.raises(TypeError, match='box'):
        place_in_box(positions, params, 'box_center')
    frame = place_in_box(positions, params, 'none')
    assert frame.dimensions is None and frame.record['box'] is None
    np.testing.assert_array_equal(frame.positions, positions)


def test_scripted_source_has_the_real_interface():
    """tests._cgaimmd_helpers.ScriptedShootingPointSource and the real source have one interface."""
    import inspect
    from tests._cgaimmd_helpers import ScriptedShootingPointSource
    methods = ('wait_for_network', 'next_shooting_point', 'note_glitch', 'note_registered')
    for name in methods:
        real = inspect.signature(getattr(GenerativeShootingPointSource, name))
        fake = inspect.signature(getattr(ScriptedShootingPointSource, name))
        assert [(p.name, p.kind) for p in real.parameters.values()] == \
            [(p.name, p.kind) for p in fake.parameters.values()], name
    params = cgaimmd_params_namespace()
    real = GenerativeShootingPointSource(params, FakeWorker(), 'run')
    fake = ScriptedShootingPointSource(params, FakeWorker(), 'run')
    for name in ('watcher', 'pool', 'counters', 'consecutive_readback'):
        assert hasattr(real, name) and hasattr(fake, name), name
    assert real.counters == {} and real.consecutive_readback == 0
    assert real.watcher.loaded is False and len(real.pool) == 0     # nothing loaded at init
    assert real.generator is None                                  # nor built


# ---------------------------------------------------------------------------
# the generator: picked by name, checked strictly at start-up
# ---------------------------------------------------------------------------

class CommittorOnly(torch.nn.Module):
    """A committor-only C-GAIMMD network (nothing any generation method needs)."""

    n_atoms = 1
    lambda_clip = 10.0

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, inp):
        x = torch.flatten(torch.as_tensor(inp, dtype=torch.float32), start_dim=1)[:, :1]
        return GAIN * self.scale * x


class OldBioEmu2LikeNetwork(ScriptedNetwork):
    """What the archived BioEmu2 networks looked like: their own generate, no default_generator."""

    default_generator = None

    def generate(self, lam, n=None, **kwargs):            # never called by the new worker
        raise AssertionError("the network's own generate must not be called")


def test_generator_is_picked_by_name_in_the_params(tmp_path, monkeypatch):
    """params.cgaimmd_generator names a method registered by a test module; the source builds it
    with its settings at start-up and generates through it only."""
    setup = _started(tmp_path, monkeypatch, chunk=4)
    assert setup.params.cgaimmd_generator == {'method': SCRIPTED, 'chunk': 4}
    assert isinstance(setup.source.generator, ScriptedGenerator)
    assert setup.source.generator.chunk == 4
    record = _next(setup).record
    assert record['generator'] == {'method': SCRIPTED, 'settings': {'chunk': 4}, 'batch_size': 4}
    assert [call['n'] for call in setup.network.generate_calls] == [4, 4]
    # a fresh torch generator of the refill, seeded from the refill seed, reaches the method
    seed = refill_seed(0, FOLDER, PATH, 0)
    expected = torch.Generator().manual_seed(seed).initial_seed()
    assert setup.network.generate_calls[0]['generator'].initial_seed() == expected


@pytest.mark.parametrize('case', ['committor_only_with_ode', 'no_default_and_unset',
                                  'unknown_method', 'old_bioemu2_network'])
def test_generator_check_is_strict_at_start_up(tmp_path, monkeypatch, case):
    """Every generator problem stops the worker at start-up with a TypeError naming
    params.cgaimmd_generator; for params read through old names too (Params.load only warns for
    them, so that archived runs load for analysis, but no worker generates with them)."""
    from aimmd.cgaimmd import _legacy
    if case == 'committor_only_with_ode':
        network, generator, match = CommittorOnly(), 'ode', 'velocity'
    elif case == 'no_default_and_unset':
        network, generator, match = CommittorOnly(), None, 'default_generator'
    elif case == 'unknown_method':
        network, generator, match = ScriptedNetwork(), 'no_such_method', 'unknown generation'
    else:
        network = OldBioEmu2LikeNetwork()
        # what Params stores for such a network read through old names
        with pytest.warns(DeprecationWarning, match='generation method'):
            generator = _legacy.generator_spec({'chunk': 64}, network)
        assert generator == {'method': 'bioemu2_edm', 'chunk': 64}
        match = 'bioemu2_edm'
    setup = _setup(tmp_path, monkeypatch, network=network, generator=generator,
                   params_overrides={'_read_old_names': case == 'old_bioemu2_network'})
    with pytest.raises(TypeError, match='cgaimmd_generator') as caught:
        setup.source.wait_for_network()
    assert match in str(caught.value)
    assert setup.source.generator is None and setup.source.settings is not None


def test_atom_count_checked_against_the_generator_atoms(tmp_path, monkeypatch):
    """A GeneratorDescriptors that selects other atoms than the network generates stops the start-up."""
    from pathlib import Path
    from aimmd.cgaimmd.preparation.descriptors import GeneratorAtoms, GeneratorDescriptors
    nanma = Path(__file__).resolve().parent / 'cgaimmd_nanma' / 'conf.gro'
    descriptors = GeneratorDescriptors(str(nanma), GeneratorAtoms(n_solute=22))
    setup = _setup(tmp_path, monkeypatch, params_overrides={'descriptors_function': descriptors})
    with pytest.raises(ValueError, match='generates 1 atoms but the descriptors function '
                                         r'\(GeneratorDescriptors\) selects 22'):
        setup.source.wait_for_network()


def test_network_provenance_reaches_the_record(tmp_path, monkeypatch):
    """A network's provenance() is called once at start-up and recorded in every shot."""
    calls = []

    class Recorded(ScriptedNetwork):
        def provenance(self):
            calls.append(1)
            return {'checkpoint': 'tiny', 'parameter_checksum': 'abc'}

    setup = _started(tmp_path, monkeypatch, network=Recorded())
    first, second = _next(setup), _next(setup, 'path000004')
    for point in (first, second):
        assert point.record['network']['provenance'] == {'checkpoint': 'tiny',
                                                         'parameter_checksum': 'abc'}
    assert calls == [1]


# ---------------------------------------------------------------------------
# stage 2 in public: prepare_candidate
# ---------------------------------------------------------------------------

def _seeded_prepare(positions, context):
    """A prepare hook that uses the candidate's random stream and seed."""
    out = np.array(positions, dtype=float)
    out[:, 1] = context.rng.random() + 0.001 * (context.seed % 7)
    return PreparedFrame(out, None, record={'seed': context.seed, 'path': context.path_name})


def _seeded_minimize(frame, context):
    out = np.array(frame.positions)
    out[:, 2] = 0.5 * out[:, 1]
    return PreparedFrame(out, frame.dimensions, record={'steps': 3, 'seed': context.seed})


def test_prepare_candidate_is_the_workers_stage_2(tmp_path, monkeypatch):
    """prepare_candidate on the candidate and context of a worker's pop gives the worker's
    shooting point, record for record."""
    seen = []
    real = cgaimmd_source.prepare_candidate

    def spy(positions, lambda_target, context):
        outcome = real(positions, lambda_target, context)
        seen.append((np.array(positions), lambda_target, context, outcome))
        return outcome

    monkeypatch.setattr(cgaimmd_source, 'prepare_candidate', spy)
    script = _only(LATE, lambda target: 30.0 + target)          # all but LATE land in B
    setup = _started(tmp_path, monkeypatch, network=ScriptedNetwork(script),
                     placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': _seeded_prepare,
                                       'cgaimmd_minimize_function': _seeded_minimize})
    point = _next(setup)
    positions, lambda_target, context, outcome = seen[-1]
    assert outcome.ok and not outcome.stopped and outcome.reason == ''
    assert [s[3].reason for s in seen[:-1]] == ['state_B'] * (len(seen) - 1)
    # again, with the candidate's own stream rebuilt
    import dataclasses
    index = point.record['seed']['candidate_index']
    again = real(positions, lambda_target, dataclasses.replace(
        context, rng=preparation_stream(point.record['seed']['refill'], index)))
    np.testing.assert_array_equal(np.asarray(again.frame.positions, dtype=np.float32),
                                  point.positions)
    assert again.check['lambda'] == point.record['lambda_final']
    for key in ('prepare', 'minimize'):
        expected = {k: v for k, v in point.record[key].items() if k != 'seconds'}
        assert {k: v for k, v in getattr(again, key).items() if k != 'seconds'} == expected
    assert {k: again.check[k] for k in point.record['acceptance']} == point.record['acceptance']
    assert again.values_device == point.record['device']['values']


def test_prepare_candidate_outside_a_worker(tmp_path, monkeypatch):
    """candidate_context + prepare_candidate: a validation tool prepares a generated structure as
    a run would (hooks that use the seed give the same frame), and every outcome is reported."""
    setup = _started(tmp_path, monkeypatch, placement='box_center',
                     params_overrides={'cgaimmd_prepare_function': _seeded_prepare,
                                       'cgaimmd_minimize_function': _seeded_minimize})
    positions = np.array([[0.12, 0.0, 0.0]])
    context = candidate_context(setup.params, tmp_path / 'validation', seed=11,
                                network_info=setup.source.watcher.info)
    assert context.scratch == f'{tmp_path / "validation"}/cgaimmd_scratch'
    assert context.settings == setup.params.cgaimmd_settings and context.seed == 11
    outcome = prepare_candidate(positions, 1.2, context)
    assert outcome.ok and outcome.check['rule'] == 'target' and outcome.check['enforced']
    assert outcome.check['lambda'] == pytest.approx(1.2, abs=1e-6)
    assert outcome.prepare['record'] == {'seed': 11, 'path': 'candidate'}
    assert outcome.minimize['record'] == {'steps': 3, 'seed': 11}
    assert set(outcome.seconds) == {'prepare', 'minimize', 'final_check'}
    assert (tmp_path / 'validation' / 'cgaimmd_scratch').is_dir()
    # a target far from the committor: rejected with the final check's reason
    far = prepare_candidate(positions, -3.0, candidate_context(
        setup.params, tmp_path / 'validation', seed=11, network_info=setup.source.watcher.info))
    assert (far.ok, far.reason, far.stopped) == (False, 'lambda_window', False)
    # a frame in B and a stop request
    in_b = prepare_candidate(np.array([[0.7, 0.0, 0.0]]), 0.0, context)
    assert (in_b.ok, in_b.reason, in_b.check['state']) == (False, 'state_B', 'B')
    stopped = prepare_candidate(positions, 1.2, candidate_context(
        setup.params, tmp_path / 'validation', must_stop=lambda: True))
    assert (stopped.ok, stopped.stopped, stopped.reason, stopped.check) == (False, True,
                                                                           'stopped', None)
    # without network_info the bootstrap rule applies (unless window_after_trained_shots is 0)
    setup.params.__dict__['cgaimmd_parameters'] = {
        **setup.params.cgaimmd_parameters, 'window_after_trained_shots': 5}
    boot = prepare_candidate(positions, -3.0, candidate_context(setup.params, tmp_path / 'v2'))
    assert boot.ok and boot.check['rule'] == 'bootstrap_any_R'
