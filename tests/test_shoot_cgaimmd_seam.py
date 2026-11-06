"""The C-GAIMMD seam of the shoot worker: ``aimmd.worker._shoot``.

Under ``chain_type='cgaimmd'`` a reactive-region chain takes its shooting
points from the C-GAIMMD shooting-point source
(``aimmd.cgaimmd.source.make_cgaimmd_source``), writes the in-flight record
before the segments are initialized, runs the glitch branch for read-back
failures (a candidate rejection), writes the per-shot sidecar after
registration, and leaves ``pool.log`` alone; RFPS chains, internal shooting
and sweeps are unchanged.

The source is ``ScriptedShootingPointSource``; the MD is faked as in
``tests/test_worker_runtime_unit.py`` (``_simulate`` returns scripted results,
``register_path`` writes a real registered path, under the run's descriptor
series, so that the sidecar can read it).
"""

import itertools
import os

import numpy as np
import pytest

import aimmd
from aimmd.cache.npy import save_npy
from aimmd.cgaimmd.sidecar import (REGISTRATION_KEYS, in_flight_fname, in_flight_positions_fname,
                                read_shot_sidecar, write_in_flight)
from aimmd.cgaimmd.types import ShootingPoint
from aimmd.path.utils import get_cache_fname
from aimmd.worker import _shoot
from aimmd.worker._shoot import WorkerShoot
from aimmd.cgaimmd import source as cgaimmd_source
from tests._cgaimmd_helpers import ScriptedShootingPointSource, cgaimmd_params_namespace
from tests._helpers_unit import build_path, state_labels_from_positions, write_trajectory


class TinyShootWorker(WorkerShoot):
    """Concrete host for `_shoot` (as in tests/test_worker_runtime_unit.py)."""

    def __init__(self, params, initial_paths, root, directory=None):
        self.params = params
        self.initial_paths = initial_paths
        self.directory = str(root) if directory is None else directory
        self._directory = str(root)
        self.must_stop = False
        self.total_steps = 0
        self.total_frames = 0
        self._location = ''
        self.log_file = ''
        self.original_stdout = ''


def one_atom(x):
    positions = np.zeros((len(x), 1, 3), dtype=np.float32)
    positions[:, 0, 0] = np.asarray(x, dtype=np.float32)
    return positions


class Harness:
    """A TinyShootWorker on `tmp_path/run` with fake MD, a fake source and recorders."""

    def __init__(self, tmp_path, monkeypatch, *, points=(), chain_type='cgaimmd', n_shots=1,
                 initialized=(), simulate=None, wait_result=True, retry=False,
                 initial_x=(-1.0, -1.1, -0.9), directory=None, drop_rows=False, **overrides):
        self.tmp_path = tmp_path
        self.series = []
        self.bias_functions = []
        self.drop_rows = drop_rows
        self.run = tmp_path / 'run'
        self.run.mkdir(exist_ok=True)
        self.events = []
        self.registered = []
        self.sources = []
        self.n_shots = n_shots
        self._initialized = list(initialized)
        segments = tmp_path / 'segments'
        segments.mkdir(exist_ok=True)
        self.back = build_path(segments, stem='back_seg', positions=one_atom([0.1, -0.3, -1.0]),
                               shooting_index=0)
        self.forw = build_path(segments, stem='forw_seg', positions=one_atom([0.1, 0.4, 1.0]),
                               shooting_index=0)
        initial = build_path(tmp_path, stem='seed', positions=one_atom(initial_x),
                             shooting_index=0)
        values = dict(chain_type=chain_type, nbins=1, max_length=100, selection_pool_size=2,
                      free_overriding_states='', at_least_one_transition_in_pool=False,
                      retry_with_state_definition_glitches=retry,
                      check_if_initialized=self.check_if_initialized,
                      initialize_simulation=self.initialize_simulation,
                      free_trajectories=lambda directory, old=None: [])
        values.update(overrides)
        self.params = cgaimmd_params_namespace(**values)
        self.worker = TinyShootWorker(self.params, aimmd.PathEnsemble(initial), self.run,
                                      directory=directory)
        harness = self

        def make_cgaimmd_source(params, worker, directory):
            source = ScriptedShootingPointSource(params, worker, directory, points=points,
                                    wait_result=wait_result)
            harness.sources.append(source)
            return source

        monkeypatch.setattr(cgaimmd_source, 'make_cgaimmd_source', make_cgaimmd_source)
        monkeypatch.setattr(_shoot, 'register_path', self.register_path)
        results = simulate if simulate is not None else self.default_simulate()
        monkeypatch.setattr(self.worker, '_simulate', lambda *args, **kwargs: self._next(results),
                            raising=False)
        cycle = itertools.cycle([self.back, self.forw])
        monkeypatch.setattr(_shoot, 'Path',
                            lambda *args, **kwargs: next(cycle) if not args
                            else aimmd.Path(*args, **kwargs))

    def default_simulate(self):
        return itertools.cycle([(0, 3, 'A', 3), (0, 3, 'B', 3)])

    def _next(self, results):
        item = next(results)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def folder(self):
        return f'{self.run}/chainR0'

    @property
    def source(self):
        return self.sources[0] if self.sources else None

    def check_if_initialized(self, *deffnms):
        return self._initialized.pop(0) if self._initialized else False

    def initialize_simulation(self, shooting_point, *deffnms, **kwargs):
        folder = os.path.dirname(deffnms[0])
        self.events.append(('initialize', {
            'shooting_point': shooting_point, 'deffnms': deffnms,
            'in_flight': os.path.isfile(in_flight_fname(folder)),
            'in_flight_positions': os.path.isfile(in_flight_positions_fname(folder))}))

    def register_path(self, path, chain, eneconv=None, bias_function=None,
                      descriptors_series='descriptors'):
        """Write a registered path (5 frames, shooting frame 2 in R) and mutate `path` into it.

        Its descriptor rows go to `descriptors_series`, as the real ``register_path`` writes
        them from the segments; with ``drop_rows`` the shooting frame's row is zero (a half
        registered without its series: the ledger counts it as missing).
        """
        self.series.append(descriptors_series)
        self.bias_functions.append(bias_function)
        folder = self.worker.folder                  # the chain folder _shoot activated
        fname = f'{folder}/path{len(chain) + 1:06g}.xtc'
        positions = one_atom([-1.0, -0.2, 0.1, 0.3, 1.0])
        write_trajectory(folder, stem=f'path{len(chain) + 1:06g}', positions=positions,
                         times=[2.0, 1.0, 0.0, 1.0, 2.0])
        stored = aimmd.Path(fname).positions
        save_npy(get_cache_fname(fname, 'states'), state_labels_from_positions(stored))
        rows = np.array(stored, dtype=np.float32).reshape(len(stored), -1)
        if self.drop_rows:
            rows[2] = 0.0
        save_npy(get_cache_fname(fname, descriptors_series), rows)
        registered = aimmd.Path(fname, shooting_index=2)
        path.__dict__.clear()
        path.__dict__.update(registered.__dict__)
        chain._paths.append(path)
        self.registered.append(fname)
        self.events.append(('register', fname))
        if len(self.registered) >= self.n_shots:
            self.worker.must_stop = True

    def shoot(self, target_state='R', **kwargs):
        return self.worker._shoot(target_state=target_state, k=0, sweep=False, **kwargs)


def point(x=0.1):
    return one_atom([x])[0]


def forbid(name):
    def raising(*args, **kwargs):
        raise AssertionError(f'{name} must not be called under C-GAIMMD')
    return raising


# ---------------------------------------------------------------------------
# C-GAIMMD chains
# ---------------------------------------------------------------------------

def test_source_called_not_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(_shoot, 'select_shooting_point', forbid('select_shooting_point'))
    monkeypatch.setattr(_shoot, 'update_selection_pool', forbid('update_selection_pool'))
    monkeypatch.setattr(_shoot, 'get_initial_transitions_for_shooting_chain',
                        forbid('get_initial_transitions_for_shooting_chain'))
    harness = Harness(tmp_path, monkeypatch, points=[point(0.1)])
    harness.shoot()
    source = harness.source
    assert [call[0] for call in source.calls] == ['__init__', 'wait_for_network',
                                                  'next_shooting_point', 'note_registered']
    kwargs = source.called('next_shooting_point')[0]
    assert kwargs['folder'] == harness.folder and kwargs['path_name'] == 'path000001'
    assert len(kwargs['chain']) == 1                 # the chain object, registered into afterwards
    shooting_point = harness.events[0][1]['shooting_point']
    assert isinstance(shooting_point, ShootingPoint)
    np.testing.assert_allclose(shooting_point.positions, point(0.1))
    assert harness.worker.total_steps == 1


def test_in_flight_before_initialize(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point(0.2)])
    harness.shoot()
    initialize = harness.events[0]
    assert initialize[0] == 'initialize'
    assert initialize[1]['in_flight'] and initialize[1]['in_flight_positions']
    assert initialize[1]['deffnms'] == (f'{harness.folder}/back', f'{harness.folder}/forw')


def test_sidecar_after_register(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point(0.1), point(0.2)], n_shots=2)
    harness.shoot()
    assert len(harness.registered) == 2
    for n, fname in enumerate(harness.registered, start=1):
        sidecar = read_shot_sidecar(fname)
        assert sidecar['source'] == 'cgaimmd'
        assert sidecar['path_name'] == f'path{n:06d}'
        assert sidecar['note'] == 'ScriptedShootingPointSource'
        assert sidecar['path'] == os.path.basename(fname)
        assert sidecar['registration']['shooting_index'] == 2
        assert sidecar['lambda_registered'] is not None
    # in-flight records cleared after each registration, the source told each time
    assert not os.path.exists(in_flight_fname(harness.folder))
    assert not os.path.exists(in_flight_positions_fname(harness.folder))
    assert len(harness.source.called('note_registered')) == 2
    assert [call[1]['path_name'] for call in harness.source.calls
            if call[0] == 'next_shooting_point'] == ['path000001', 'path000002']


def test_shot_under_a_named_series_has_lambda_registered(tmp_path, monkeypatch):
    """A C-GAIMMD chain registers its shot under the run's named descriptor series and the
    sidecar has a lambda_registered (here the shooting frame's row is missing at registration:
    the sidecar featurizes it into that series, never into 'descriptors')."""
    def values_function(descriptors):
        return 10.0 * np.asarray(descriptors, dtype=float).reshape(len(descriptors), -1)[:, 0]

    harness = Harness(tmp_path, monkeypatch, points=[point(0.1)],
                      descriptors_series='descriptors-cgtest', drop_rows=True,
                      values_function=values_function)
    harness.shoot()
    assert harness.series == ['descriptors-cgtest']
    fname = harness.registered[0]
    sidecar = read_shot_sidecar(fname)
    x = aimmd.Path(fname).positions[2, 0, 0]
    assert sidecar['registration']['lambda_registered_missing'] is None
    assert sidecar['lambda_registered'] == pytest.approx(10.0 * x)    # not the zero row's 0
    rows = np.load(get_cache_fname(fname, 'descriptors-cgtest'))
    np.testing.assert_allclose(rows[2], [x, 0.0, 0.0], atol=1e-6)        # featurized
    assert not os.path.exists(get_cache_fname(fname, 'descriptors'))


def test_shot_under_a_static_plumed_bias(tmp_path, monkeypatch):
    """Under a recorded PLUMED bias a C-GAIMMD shot is registered with the bias function (its
    COLVAR and bias cache, as for RFPS) and its sidecar records the bias settings."""
    plumed = tmp_path / 'opes_static.dat'
    plumed.write_text('OPES_METAD ... (a test)\n')

    def bias_function(fname):
        return np.zeros(5)

    harness = Harness(tmp_path, monkeypatch, points=[point(0.1)], engine='gromacs',
                      gmx_mdrun=f'gmx mdrun -ntomp 1 -plumed {plumed}', gmx_eneconv=None,
                      record_bias=True, bias_source='file', bias_function=bias_function)
    harness.shoot()
    assert harness.bias_functions == [bias_function]
    bias = read_shot_sidecar(harness.registered[0])['bias']
    assert bias['record_bias'] is True and bias['bias_source'] == 'file'
    assert bias['plumed_file'] == str(plumed) and bias['plumed_sha256']
    assert bias['bias_function'].endswith('bias_function')
    assert bias['preparation_unbiased'] is True                     # no preparation hooks
    assert bias['preparation_unverified_hooks'] == {}


def test_pool_log_untouched(tmp_path, monkeypatch):
    """Under C-GAIMMD the pool is empty; pool.log is neither saved nor created."""
    harness = Harness(tmp_path, monkeypatch, points=[point()])
    os.makedirs(harness.folder)
    with open(f'{harness.folder}/pool.log', 'w') as file:
        file.write('path000001.xtc\n')                               # an RFPS history's pool
    harness.shoot()
    assert open(f'{harness.folder}/pool.log').read() == 'path000001.xtc\n'


def test_pool_log_not_created(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point()])
    harness.shoot()
    assert harness.registered                                         # the shot ran
    assert not os.path.exists(f'{harness.folder}/pool.log')


def test_must_stop_during_generation(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[None])
    harness.shoot()
    assert harness.events == []                                       # nothing initialized
    assert not os.path.exists(in_flight_fname(harness.folder))


def test_wait_for_network_false_returns(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point()], wait_result=False)
    harness.shoot()
    assert [call[0] for call in harness.source.calls] == ['__init__', 'wait_for_network']
    assert harness.events == []


def test_cgaimmd_source_gets_underscore_directory(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point()], directory='launch/relative/run')
    harness.worker._directory = str(harness.run)
    harness.shoot()
    assert harness.source.directory == str(harness.run)
    assert harness.source.calls[0] == ('__init__', {'directory': str(harness.run)})


def test_resume_does_not_regenerate(tmp_path, monkeypatch):
    """Segments initialized at start: the shot resumes with its in-flight record."""
    harness = Harness(tmp_path, monkeypatch, points=[point()], initialized=[True, True])
    os.makedirs(harness.folder)
    write_in_flight(harness.folder, {'schema': 'aimmd.cgaimmd.shot/1', 'source': 'cgaimmd',
                                     'path_name': 'path000001', 'lambda_target': 1.5,
                                     'note': 'before the restart'}, point(0.1))
    harness.shoot()
    assert harness.source.called('next_shooting_point') == []
    assert [event[0] for event in harness.events] == ['register']
    sidecar = read_shot_sidecar(harness.registered[0])
    assert sidecar['source'] == 'cgaimmd' and sidecar['lambda_target'] == 1.5
    assert sidecar['note'] == 'before the restart'


def test_resume_without_record(tmp_path, monkeypatch):
    """A shot in flight when an RFPS run was continued as C-GAIMMD: 'resumed_without_cgaimmd_record'."""
    harness = Harness(tmp_path, monkeypatch, points=[point()], initialized=[True])
    harness.shoot()
    sidecar = read_shot_sidecar(harness.registered[0])
    assert sidecar['source'] == 'resumed_without_cgaimmd_record'
    assert sidecar['path_name'] == 'path000001'


class NetworkLoadedAtWait:
    """A values function that scores like the params file's network until the source's
    ``wait_for_network`` has run, and like the run's network file after it.

    The params file's network gives 0 everywhere (an untrained head); the run's network gives
    ``10 x``. ``calls`` records which network scored each call.
    """

    def __init__(self, monkeypatch):
        self.loaded = False
        self.calls = []
        real_wait = ScriptedShootingPointSource.wait_for_network
        harness_values = self

        def wait_for_network(source):
            result = real_wait(source)
            harness_values.loaded = result
            return result
        monkeypatch.setattr(ScriptedShootingPointSource, 'wait_for_network', wait_for_network)

    def __call__(self, descriptors):
        self.calls.append('run network' if self.loaded else 'params-file network')
        x = np.asarray(descriptors, dtype=float).reshape(len(descriptors), -1)[:, 0]
        return 10.0 * x if self.loaded else np.zeros_like(x)


def crash_before_the_sidecar(tmp_path, monkeypatch):
    """One registered shot whose sidecar was not written: its in-flight marker is still there
    (the crash window between ``register_path`` and the sidecar write). Returns the path file
    and its shooting frame's x."""
    first = Harness(tmp_path, monkeypatch, points=[point(0.1)])
    first.shoot()
    fname = first.registered[0]
    sidecar = read_shot_sidecar(fname)
    os.remove(f'{fname}.cgaimmd.json')
    record = {key: value for key, value in sidecar.items() if key not in REGISTRATION_KEYS}
    path = aimmd.Path(fname, shooting_index=sidecar['registration']['shooting_index'])
    write_in_flight(first.folder, record, np.asarray(path.positions[path.shooting_index]))
    return fname, float(path.positions[path.shooting_index][0, 0])


def test_recovered_sidecar_scored_with_the_run_network(tmp_path, monkeypatch):
    """A sidecar recovered at a restart gets lambda_registered from network<s>.h5, which the
    source loads in wait_for_network; not from the network the params file builds."""
    fname, x = crash_before_the_sidecar(tmp_path, monkeypatch)
    values = NetworkLoadedAtWait(monkeypatch)
    harness = Harness(tmp_path, monkeypatch, points=[], values_function=values)
    harness.shoot()                          # recovers, then finds no shooting point and stops
    assert harness.registered == []
    recovered = read_shot_sidecar(fname)
    assert recovered['recovered_after_restart'] is True
    assert recovered['note'] == 'ScriptedShootingPointSource'          # the record is attached
    assert recovered['registration']['lambda_registered_missing'] is None
    assert recovered['lambda_registered'] == pytest.approx(10.0 * x)   # not the untrained 0
    assert x != 0.0 and values.calls == ['run network']
    assert not os.path.exists(in_flight_fname(harness.folder))


def test_recovery_waits_for_the_network(tmp_path, monkeypatch):
    """A worker that must stop before any network file exists leaves the in-flight record for
    the next start, which recovers the sidecar with the network."""
    fname, _ = crash_before_the_sidecar(tmp_path, monkeypatch)
    values = NetworkLoadedAtWait(monkeypatch)
    harness = Harness(tmp_path, monkeypatch, points=[], values_function=values,
                      wait_result=False)
    harness.shoot()
    assert read_shot_sidecar(fname) is None and values.calls == []
    assert os.path.exists(in_flight_fname(harness.folder))


def test_glitch_clears_in_flight(tmp_path, monkeypatch, capsys):
    glitch = RuntimeError('... frame 0 in state A, should be in R; consider deleting the '
                          'trajectory file to allow AIMMD to recreate it')
    simulate = iter([glitch, (0, 3, 'A', 3), (0, 3, 'B', 3)])
    harness = Harness(tmp_path, monkeypatch, points=[point(0.1), point(0.2)],
                      simulate=simulate, retry=True)
    in_flight_after_glitch = []
    real_note = ScriptedShootingPointSource.note_glitch

    def note_glitch(self, folder, exc, record):
        in_flight_after_glitch.append(os.path.exists(in_flight_fname(folder)))
        return real_note(self, folder, exc, record)

    monkeypatch.setattr(ScriptedShootingPointSource, 'note_glitch', note_glitch)
    harness.shoot()
    assert in_flight_after_glitch == [True]          # note_glitch first, then the record is cleared
    source = harness.source
    glitches = source.called('note_glitch')
    assert len(glitches) == 1
    assert glitches[0]['folder'] == harness.folder and glitches[0]['exc'] is glitch
    assert glitches[0]['record']['path_name'] == 'path000001'
    assert source.counters == {'state_readback': 1}
    # the next candidate is shot for the same path; its record reaches the sidecar
    assert [c['path_name'] for c in source.called('next_shooting_point')] == ['path000001'] * 2
    assert len(harness.registered) == 1
    assert read_shot_sidecar(harness.registered[0])['path_name'] == 'path000001'
    assert not os.path.exists(in_flight_fname(harness.folder))
    assert 'WARNING: state-definition glitch in' in capsys.readouterr().out


def test_glitch_branch_runs_under_cgaimmd_without_flag(tmp_path, monkeypatch, capsys):
    """A read-back failure under C-GAIMMD is a candidate rejection, whatever the flag says."""
    assert cgaimmd_source.READBACK_FAILURE_IS_REJECTION is True
    glitch = RuntimeError('... frame 0 in state B, should be in R')
    simulate = iter([glitch, (0, 3, 'A', 3), (0, 3, 'B', 3)])
    harness = Harness(tmp_path, monkeypatch, points=[point(0.1), point(0.2)],
                      simulate=simulate, retry=False)
    harness.shoot()
    assert harness.source.counters == {'state_readback': 1}
    assert len(harness.registered) == 1
    out = capsys.readouterr().out
    assert 'retry_with_state_definition_glitches=True' not in out
    assert "chain_type='cgaimmd'" in out


def test_readback_switch_off_reraises(tmp_path, monkeypatch):
    monkeypatch.setattr(cgaimmd_source, 'READBACK_FAILURE_IS_REJECTION', False)
    glitch = RuntimeError('... frame 0 in state B, should be in R')
    harness = Harness(tmp_path, monkeypatch, points=[point()], simulate=iter([glitch]))
    with pytest.raises(RuntimeError, match='should be in R'):
        harness.shoot()
    assert harness.source.called('note_glitch') == []


def test_other_runtime_errors_propagate(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, points=[point()],
                      simulate=iter([RuntimeError('mdrun crashed')]))
    with pytest.raises(RuntimeError, match='mdrun crashed'):
        harness.shoot()
    assert harness.source.called('note_glitch') == []


def test_tps_history_raises(tmp_path, monkeypatch):
    """The TPS-history check globs the run the worker chdir'ed into (self._directory, K19)."""
    import aimmd.cgaimmd.restart as restart
    harness = Harness(tmp_path, monkeypatch, points=[point()], directory='launch/relative/run')
    harness.worker._directory = str(harness.run)
    calls = []
    real_check = restart.check_cgaimmd_restart

    def spy(params, run_dir, **kwargs):
        calls.append((run_dir, kwargs))
        return real_check(params, run_dir, **kwargs)

    monkeypatch.setattr(restart, 'check_cgaimmd_restart', spy)
    os.makedirs(harness.folder)
    np.save(f'{harness.folder}/tps_weights.npy', np.ones(1))
    with pytest.raises(NotImplementedError, match='TPS_RESTART_AS_CGAIMMD'):
        harness.shoot()
    assert harness.sources == []
    # no descriptor check at the seam: the C-GAIMMD source checks the width itself
    assert calls == [(str(harness.run), {})]


# ---------------------------------------------------------------------------
# unchanged branches
# ---------------------------------------------------------------------------

def rfps_harness(tmp_path, monkeypatch, **kwargs):
    selected = []
    kwargs.setdefault('initial_x', (-1.0, -0.3, 0.0, 0.3, 1.0))     # RFPS needs a transition
    harness = Harness(tmp_path, monkeypatch, **kwargs)
    monkeypatch.setattr(cgaimmd_source, 'make_cgaimmd_source', forbid('make_cgaimmd_source'))
    monkeypatch.setattr(_shoot, 'update_selection_pool',
                        lambda pool, *args, **kw: selected.append('pool') or pool)

    def select(pool, params, folder, chain, **kw):
        selected.append(('select', folder))
        return harness.back[0:1]

    monkeypatch.setattr(_shoot, 'select_shooting_point', select)
    return harness, selected


def test_rfps_branch_unchanged(tmp_path, monkeypatch):
    harness, selected = rfps_harness(tmp_path, monkeypatch, chain_type='rfps')
    harness.shoot()
    assert ('select', harness.folder) in selected
    assert harness.sources == []
    assert read_shot_sidecar(harness.registered[0]) is None
    assert not os.path.exists(in_flight_fname(harness.folder))
    assert os.path.exists(f'{harness.folder}/pool.log')             # RFPS saves its pool


def test_internal_shooting_unchanged(tmp_path, monkeypatch):
    """chain_type='cgaimmd' shoots internal chains (t != states[1]) as before."""
    harness, selected = rfps_harness(tmp_path, monkeypatch)
    harness.shoot(target_state='A')
    assert ('select', f'{harness.run}/chainA0') in selected
    assert harness.sources == []
    assert not os.path.exists(f'{harness.run}/chainR0')


def test_rfps_readback_glitch_still_raises_without_flag(tmp_path, monkeypatch):
    glitch = RuntimeError('... frame 0 in state B, should be in R')
    harness, selected = rfps_harness(tmp_path, monkeypatch, chain_type='rfps',
                                     simulate=iter([glitch]))
    with pytest.raises(RuntimeError, match='should be in R'):
        harness.shoot()
