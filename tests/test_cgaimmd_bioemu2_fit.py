"""The sequential fit of the BioEmu2 network (aimmd.cgaimmd.bioemu2.fit) on the tiny model.

- the goldens: one whole round of the old ``sequential_round`` (chignolin code)
  and ``holo_sequential_round`` (trypsin code) on the same synthetic data are
  reproduced (head losses, final head and adapters, label bins, label probe);
- ``sequential_fit`` end to end on a toy run folder whose descriptor caches
  hold peptide coordinates: head and adapters train, every round starts from
  reset (its result does not depend on the weights it started from), the seed
  reproduces and replicates differ, the round counter reads AIMMD's metadata,
  and the time-averaged labels change the generator's labels only when set;
- the bin floor, the frame intervals of the trajectories and the halving of the
  micro-batch on running out of CUDA memory.
"""

import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aimmd.path.utils import get_cache_fname
from aimmd.pathensemble.utils import assemble_pathensemble

from ._bioemu2_fixtures import (golden, golden_state, tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_peptide_system, tiny_protein_ligand_system)
from ._cgaimmd_bioemu2_golden import HEAD_HIDDEN, ROUND_SETTINGS, ROUND_SETTINGS_LIGAND, ROUND_SETTINGS_PEPTIDE
from ._cgaimmd_helpers import _write_with_caches, build_cgaimmd_run_dir, cgaimmd_params_namespace

pytestmark = pytest.mark.bioemu2

FAST = dict(min_shots_for_generator=2, adapter_steps=3, head_steps_without_holdout=5, batch_sp=4, batch_anchor=4,
            batch_den=4, micro_batch=2, n_anchor_max=12, pool=12, noisy_copies=1, feature_batch=16,
            label_probe=False)
"""Round settings small enough for the CPU (the recipe is the production one)."""


@pytest.fixture(autouse=True)
def _fresh_dataset_cache():
    from aimmd.cgaimmd.bioemu2.fit import clear_dataset_cache
    clear_dataset_cache()
    yield
    clear_dataset_cache()


def _network(system, golden=None, which=None, **kwargs):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    scaler = None if golden is None else {key: golden[f'{which}.network.scaler.{key}']
                                          for key in ('mean', 'std', 'log_sigma_mean', 'log_sigma_std')}
    return BioEmu2Network(system, hidden=HEAD_HIDDEN, scaler=scaler, **kwargs)


def _state(network):
    return {key: value.detach().clone() for key, value in network.state_dict().items()}


def _equal_states(a, b, prefixes=None):
    return all(torch.equal(a[key], b[key]) for key in a if prefixes is None or key.split('.')[0] in prefixes)


# --------------------------------------------------------------------------- goldens of the old rounds

@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_goldens_of_the_old_rounds(which, request, golden):
    """One round on the synthetic dataset of the goldens, with the recorded settings."""
    from aimmd.cgaimmd.bioemu2.fit import sequential_round
    system = request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')
    network = _network(system, golden, which)
    data = types.SimpleNamespace(**{key: golden[f'{which}.dataset.{key}']
                                    for key in ('sp_rows', 'sp_r_first', 'sp_r_last', 'fr_rows', 'fr_state')})
    settings = {**ROUND_SETTINGS, **(ROUND_SETTINGS_PEPTIDE if which == 'peptide' else ROUND_SETTINGS_LIGAND)}
    settings.pop('pool_weighting', None)                       # the old 'uniform' draw is the only one
    settings['label_probe'] = settings.pop('probe', which == 'ligand')
    settings['verbose'] = False
    losses, info = sequential_round(network, data, 0, **settings)
    head_losses = np.array([[rec.get('L_sp', np.nan), rec.get('L_anchor', np.nan)] for rec in losses])
    np.testing.assert_allclose(head_losses, golden[f'{which}.round.head_losses'], rtol=1e-5, atol=1e-5)
    old = json.loads(str(golden[f'{which}.round.info']))
    assert info['label_counts'] == golden[f'{which}.round.label_counts'].tolist() == old['label_counts']
    assert info['head']['steps'] == old['head']['steps'] and info['generator_trained'] == old['generator_trained']
    assert info['adapter_final_loss'] == pytest.approx(old['adapter_final_loss'], rel=1e-5)
    np.testing.assert_allclose(info['adapter_final_loss'], np.mean(golden[f'{which}.round.adapter_losses']),
                               rtol=1e-5)
    modules = [('head_ema', network.head_ema), ('adapter', network.single_adapter)]
    if which == 'ligand':
        modules.append(('pair_adapter', network.pair_adapter))
    for name, module in modules:
        for key, value in golden_state(golden, f'{which}.round.{name}').items():
            np.testing.assert_allclose(module.state_dict()[key].numpy(), value.numpy(), rtol=1e-5, atol=1e-6,
                                       err_msg=f'{name}.{key}')
    assert _equal_states(network.head.state_dict(), network.head_ema.state_dict())
    if which == 'ligand':
        probe = json.loads(str(golden['ligand.round.label_probe']))
        for sigma, values in probe.items():
            for key in ('own', 'shuffled', 'none'):
                assert info['label_probe'][sigma][key] == pytest.approx(values[key], rel=1e-5)
    network.engine().close()


# --------------------------------------------------------------------------- a toy run folder

def _crossing_free_trajectory(run, n_parts=3):
    """``freeA/traj000002.part*``: x runs A -> R -> B -> R -> A..., in part files (restarts repeat a frame)."""
    x = 1.2 * np.sin(np.linspace(0.0, 3.0 * np.pi, 9 * n_parts + 1))
    for part in range(n_parts):
        frames = slice(9 * part, 9 * part + 10)
        positions = np.zeros((10, 1, 3), np.float32)
        positions[:, 0, 0] = x[frames]
        _write_with_caches(Path(run) / 'freeA' / f'traj000002.part{part + 1:04d}.xtc', positions,
                           np.arange(9 * part, 9 * part + 10, dtype=float))


def _peptide_run(tmp_path, system):
    """A run folder (shots, free MD in two parts, a crossing free trajectory) whose descriptors are peptides.

    The state of every frame comes from a one-atom toy coordinate x (as in
    ``build_cgaimmd_run_dir``); the descriptor caches are rewritten with the
    peptide's template moved along a fixed displacement by x, plus noise, so
    the BioEmu2 features see different structures in A, R and B.
    """
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=2, n_shots=3, n_free=1, n_atoms=1, n_parts=2)
    _crossing_free_trajectory(rd.run)
    rng = np.random.default_rng(0)
    template = system.template
    direction = rng.normal(size=template.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    for fname in sorted(Path(rd.run).rglob('*.xtc')):
        x = np.load(get_cache_fname(str(fname), 'descriptors'))[:, 0]
        rows = template[None] + 1.5 * x[:, None, None] * direction[None] + 0.05 * rng.normal(
            size=(len(x),) + template.shape)
        np.save(get_cache_fname(str(fname), 'descriptors'), rows.reshape(len(x), -1).astype(np.float32))
    return rd.run


def _fit_inputs(tmp_path, system, network, **params):
    run = _peptide_run(tmp_path, system)
    params = cgaimmd_params_namespace(network=network, **params)
    pathensemble = assemble_pathensemble(params.shot_chains(run, None), params.free_trajectories(run))
    return SimpleNamespace(run=run, params=params, pathensemble=pathensemble,
                           worker=SimpleNamespace(_directory=run, directory=run))


def _fit(inputs, **settings):
    from aimmd.cgaimmd.bioemu2.fit import sequential_fit
    return sequential_fit(inputs.params, inputs.pathensemble, worker=inputs.worker, **{**FAST, **settings})


def test_one_round_end_to_end_on_a_toy_run(tmp_path, tiny_peptide_system):
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    losses, info = _fit(inputs, seed=7, label_probe=True)
    assert losses and info['status'] == 'trained' and info['round'] == 0
    record = info['record']
    assert record['n_shooting_points'] == 6 and record['n_with_outcome'] == 6 and record['n_anchor'] == 12
    assert record['generator_trained'] and record['adapter_steps'] == 3 and record['seed'] == 7
    assert record['round_seed'] == 7 and record['seed_source'] == 'argument'
    assert set(record['label_probe']) == {'sigma4', 'sigma16'}
    for adapter in network.adapters():                           # the adapters left the identity
        assert adapter.out.weight.abs().sum() > 0
    trained = _state(network)
    network.reset_parameters(seed=7)
    assert not _equal_states(trained, _state(network), prefixes={'head'})   # the head trained
    metadata = info['network_metadata']
    json.dumps(metadata)
    assert metadata['round'] == 0 and metadata['n_trained_shots'] == 6 and metadata['seed'] == 7
    assert metadata['fit'] == 'aimmd.cgaimmd.bioemu2.fit' and metadata['settings']['adapter_steps'] == 3
    assert metadata['settings']['series'] == 'descriptors' and metadata['network']['readout'] == 'flat'
    network.engine().close()


def test_every_round_starts_from_reset(tmp_path, tiny_peptide_system):
    """Round 1 gives the same network whatever weights it started from: no warm start."""
    from aimmd.cgaimmd.bioemu2.fit import clear_dataset_cache
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    _, info0 = _fit(inputs, seed=3)
    after_round0 = _state(network)
    _, info1 = _fit(inputs, seed=3)                               # the counter of this process
    assert (info0['round'], info1['round']) == (0, 1)
    after_round1 = _state(network)
    assert not _equal_states(after_round0, after_round1, prefixes={'single_adapter'})
    # the same round from other weights: AIMMD's metadata says round 0 is done, nothing is cached
    clear_dataset_cache()
    (Path(inputs.run) / 'networkARB.json').write_text(json.dumps({'round': 0, 'n_trained_shots': 6}))
    with torch.no_grad():
        for p in network.parameters():
            p.add_(torch.randn_like(p))
    _, again = _fit(inputs, seed=3)
    assert again['round'] == 1
    assert _equal_states(after_round1, _state(network))
    network.engine().close()


def test_seeds_reproduce_and_replicates_differ(tmp_path, tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2.fit import clear_dataset_cache, fit_seed
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    states = {}
    for seed in (1, 1, 2):
        clear_dataset_cache()
        _fit(inputs, seed=seed)
        states.setdefault(seed, []).append(_state(network))
    assert _equal_states(states[1][0], states[1][1])
    assert not _equal_states(states[1][0], states[2][0], prefixes={'single_adapter'})
    assert not _equal_states(states[1][0], states[2][0], prefixes={'head'})
    # without a seed: the run folder decides (replicates in other folders differ), or generation_seed
    params = SimpleNamespace(cgaimmd_parameters=None)
    a, source = fit_seed(params, tmp_path / 'replicate1' / 'run')
    assert source == 'run folder' and a == fit_seed(params, tmp_path / 'replicate1' / 'run')[0]
    assert a != fit_seed(params, tmp_path / 'replicate2' / 'run')[0] and 0 <= a < 2 ** 62
    assert fit_seed(SimpleNamespace(cgaimmd_parameters={'generation_seed': 12}), 'run') == (12, 'generation_seed')
    assert fit_seed(params, None) == (0, 'default') and fit_seed(params, 'run', 5) == (5, 'argument')
    clear_dataset_cache()
    _, info = _fit(inputs)
    assert info['record']['seed_source'] == 'run folder'
    assert info['record']['seed'] == fit_seed(params, inputs.run)[0]
    network.engine().close()


def test_the_round_counter_reads_aimmds_metadata(tmp_path):
    from aimmd.cgaimmd.bioemu2.fit import previous_round
    assert previous_round(None, 'ARB') == -1 and previous_round(tmp_path, 'ARB') == -1
    (tmp_path / 'networkARB.json').write_text(json.dumps({'round': 4, 'n_trained_shots': 10}))
    assert previous_round(tmp_path, 'ARB') == 4
    (tmp_path / 'networkARB.json').write_text(json.dumps({'network_metadata': {'round': 2}}))   # old layout
    assert previous_round(tmp_path, 'ARB') == 2
    (tmp_path / 'networkARB.json').write_text('{not json')
    assert previous_round(tmp_path, 'ARB') == -1


def test_time_averaged_labels_change_the_generator_labels_only_when_set(tmp_path, tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2.fit import clear_dataset_cache
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    runs = {}
    for name, extra in (('default', {}), ('none', dict(label_time_window_ps=None)),
                        ('window', dict(label_time_window_ps=2.0))):
        clear_dataset_cache()
        _, info = _fit(inputs, seed=4, pool=40, **extra)
        runs[name] = (_state(network), info['record'])
    assert _equal_states(runs['default'][0], runs['none'][0])
    assert runs['default'][1]['time_averaged_labels'] is None
    averaged = runs['window'][1]['time_averaged_labels']
    assert averaged['time_window_ps'] == 2.0 and averaged['frame_intervals_ps'] == [1.0]
    assert averaged['n_averaged'] > 0 and averaged['n_without_interval'] == 0
    assert averaged['n_pool_changed'] > 0
    assert _equal_states(runs['default'][0], runs['window'][0], prefixes={'head', 'head_ema'})
    assert not _equal_states(runs['default'][0], runs['window'][0], prefixes={'single_adapter'})
    network.engine().close()


def test_frame_intervals_come_from_the_trajectories(tmp_path):
    from aimmd.cgaimmd.bioemu2.fit import trajectory_frame_intervals
    from aimmd.cgaimmd.dataset import extract_dataset
    rd = build_cgaimmd_run_dir(tmp_path, n_chains=1, n_shots=1, n_free=1, n_atoms=1, n_parts=2)
    positions = np.zeros((6, 1, 3), np.float32)
    positions[:, 0, 0] = np.linspace(1.5, 1.8, 6)
    _write_with_caches(Path(rd.run) / 'freeB' / 'traj000002.part0001.xtc', positions, 2.5 * np.arange(6.0))
    params = cgaimmd_params_namespace()
    pathensemble = assemble_pathensemble(params.shot_chains(rd.run, None), params.free_trajectories(rd.run))
    data = extract_dataset(pathensemble, states='ARB', n_atoms=1, free_stride=1)
    intervals = trajectory_frame_intervals(data)
    by_file = {files[0].split('/')[-1]: value for files, value in zip(data.trajectories, intervals)
               if 'traj000002' in files[0] or 'path' in files[0]}
    assert by_file == {'path000001.xtc': 1.0, 'traj000002.part0001.xtc': 2.5}
    assert np.isfinite(intervals).all()


def test_the_bin_floor():
    from aimmd.cgaimmd.bioemu2.fit import _bin_draw_probabilities
    labels = torch.tensor([-9.5, -9.0, -8.6, 0.5, 0.6, 0.7, 0.8, 0.9, 9.9])     # bins 0 (3), 5 (5), 9 (1)
    probs, counts = _bin_draw_probabilities(labels)
    assert counts.tolist() == [3, 0, 0, 0, 0, 5, 0, 0, 0, 1]
    np.testing.assert_allclose(probs.numpy(), [1 / 9] * 3 + [1 / 15] * 5 + [1 / 3], rtol=1e-6)
    probs, _ = _bin_draw_probabilities(labels, bin_min=4)                       # shares 0.75, 1, 0.25
    np.testing.assert_allclose(probs.numpy(), np.array([0.25] * 3 + [0.2] * 5 + [0.25]) / 2.0, rtol=1e-6)


def test_running_out_of_cuda_memory_halves_the_micro_batch(tmp_path, tiny_peptide_system, monkeypatch):
    from aimmd.cgaimmd.bioemu2.adapters import ConditionedDenoiser
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    real = ConditionedDenoiser.denoise
    calls = {'oom': 0}

    def denoise(self, x, sigma, lam=None, uncond=None):
        if len(x) > 1 and torch.is_grad_enabled():
            calls['oom'] += 1
            raise torch.OutOfMemoryError('CUDA out of memory (simulated)')
        return real(self, x, sigma, lam, uncond)
    monkeypatch.setattr(ConditionedDenoiser, 'denoise', denoise)
    losses, info = _fit(inputs, seed=1, micro_batch=4)
    record = info['record']
    assert losses and record['micro_batch'] == 1 and calls['oom'] == 2
    assert record['out_of_memory'] == [dict(step=1, micro_batch=2), dict(step=1, micro_batch=1)]
    assert record['generator_trained'] and len(record['adapter_out_weight_norm']) == 1
    network.engine().close()


def test_below_min_shots_the_generator_stays_plain_bioemu2(tmp_path, tiny_peptide_system):
    network = _network(tiny_peptide_system)
    inputs = _fit_inputs(tmp_path, tiny_peptide_system, network)
    losses, info = _fit(inputs, seed=1, min_shots_for_generator=100)
    assert losses and not info['record']['generator_trained'] and info['record']['label_counts'] is None
    for adapter in network.adapters():
        assert torch.equal(adapter.out.weight, torch.zeros_like(adapter.out.weight))
    network.engine().close()


def test_settings_it_refuses(tiny_peptide_system):
    from aimmd.cgaimmd.bioemu2.fit import sequential_round
    network = _network(tiny_peptide_system)
    data = SimpleNamespace(sp_rows=np.zeros((0, 498)), sp_r_first=np.zeros(0), sp_r_last=np.zeros(0),
                           fr_rows=np.zeros((0, 498)), fr_state=np.zeros(0))
    for settings, message in ((dict(p_uncond=1.0), 'p_uncond must be in'), (dict(bin_min=-1), 'bin_min'),
                              (dict(label_time_window_ps=float('nan')), 'label_time_window_ps'),
                              (dict(adapter_lr_schedule='linear'), 'adapter_lr_schedule')):
        with pytest.raises(ValueError, match=message):
            sequential_round(network, data, **settings)
    losses, info = sequential_round(network, data)                 # nothing to train on: no losses
    assert losses == [] and info['head']['steps'] == 0
