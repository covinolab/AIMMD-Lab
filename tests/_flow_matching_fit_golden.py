"""The port-equivalence golden of the flow-matching trainer and of one fit round.

The trainer and the fit of the flow branch (``aimmd.network.flow_fit`` at
flowdev ``bb98943``: ``FlowTrainer`` and ``flow_fit``) were run once on the
cases below, and what they returned is stored in
``tests/data/flow_matching_fit_flowdev_bb98943.pt``.
``tests/test_flow_matching_fit.py`` runs the same cases with
:mod:`aimmd.cgaimmd.flow_matching.fit` and checks the results bit for bit.

Two kinds of cases:

- :data:`STEP_CASES`: steps of a ``FlowTrainer`` on a hand-built dataset
  (shooting points, anchors and reactive frames of known rows), with the
  toy network (an OT pairing that swaps the two atoms) and with a tiny
  ``JointPaiNN``, over several recipes (the loss weights of the Goethe-dataset
  sweep, a held-out chain, density-weighted basin draws with a cosine
  schedule and an EMA). Every step's total and loss terms, the final live and
  EMA weights, the optimizer and normalizer states and both random streams
  are stored.
- :data:`ROUND_CASES`: one call of the old ``flow_fit`` on a fresh run folder
  (:func:`write_fixture`), the network starting from stored weights, with
  ``trainer_seed`` 5 and, again from the same weights in a fresh trainer,
  with ``trainer_seed`` 1005. The new fit restarts every round from its
  start weights and seeds round ``r`` with ``seed + 1000 (r - 1)``: its
  first round must equal the old fit's first call, and its second round the
  old fit's fresh call with seed 1005 (the old fit warm-started, so its own
  second call is not comparable). The path weights of AIMMD's reweighting
  are replaced by a fixed ramp in both (:func:`ramp_weights`), so that the
  golden checks the fit and not the reweighting of the two bases.

Recording it again is only needed when the cases change. From the root of
this checkout, with ``<export>`` an empty folder::

    git archive bb98943 aimmd | tar -x -C <export>
    python tests/_flow_matching_fit_golden.py fixture <folder>
    PYTHONPATH=<export> python tests/_flow_matching_fit_golden.py record <folder> \\
        tests/data/flow_matching_fit_flowdev_bb98943.pt

The first step writes the run folder with this checkout; the second runs the
old code (``aimmd`` is then imported from ``<export>``; the script refuses to
record from this checkout). This module imports only the standard library,
numpy and torch at module level; the old and the new package are reached
through the namespaces of :func:`old_api` and :func:`new_api`.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'data' / 'flow_matching_fit_flowdev_bb98943.pt'
"""The recorded outputs of the old trainer and fit."""

STATES = 'ARB'
N_ATOMS = 3
"""Atoms of the run folder of the round cases."""

TINY_PAINN = dict(coord_scale=1.5, lambda_clip=10.0, n_channels=8, n_layers=2, n_basis=6,
                  cutoff=4.0, cond_dim=8, lambda_embedding_scale=4.0, committor_hidden=(8,),
                  velocity_hidden=8, velocity_radial=6, velocity_min_distance=0.7,
                  parity_breaking=False, edge_backend='torch')
"""A tiny ``JointPaiNN`` (8 channels x 2 layers) on three atoms with tokens ``[6, 6, 8]``."""

TRAINER = dict(warmup_steps=0, normalizer_warmup=0, n_sp=6, n_anchor=4, n_fm=6,
               fm_max_sample_loss=None)
"""The settings of the hand-built trainer of the old tests (``tests/test_flow_fit.py``)."""

STEP_CASES = {
    'paired_all_terms': dict(network='paired', m=2, steps=6, held=(),
                             settings=dict(p_uncond=0.5, instate_fm_share=1 / 3,
                                           committor_weight=1.3, anchor_weight=0.7,
                                           grad_penalty=0.2, fm_weight=0.9)),
    'paired_goethe_weights_held_chain': dict(network='paired', m=2, steps=6, held=(1,),
                                             settings=dict(p_uncond=0.5, grad_penalty=2.0,
                                                           fm_weight=4.0)),
    'tiny_density_cosine_ema': dict(network='tiny', m=1, steps=8, held=(),
                                    settings=dict(basin_draw='density', lr_schedule='cosine',
                                                  total_steps=8, warmup_steps=2, ema_decay=0.9,
                                                  normalizer_warmup=2, lr=1e-2,
                                                  fm_max_sample_loss=1e3)),
    'painn': dict(network='painn', m=3, steps=3, held=(),
                  settings=dict(p_uncond=0.5, grad_penalty=0.1)),
}
"""Name -> network (``'paired'``, ``'tiny'`` or ``'painn'``), atoms, steps, held-out chains, settings."""

ROUND_SETTINGS = dict(steps_per_round=7, warmup_steps=2, n_sp=4, n_anchor=4, n_fm=6,
                      normalizer_warmup=2, idle_sleep_s=0.0, fill_missing_values=False,
                      relabel_every=3, relabel_max_frames=5, grad_penalty=0.1, ema_decay=0.9,
                      lr=1e-2)
"""The fit keywords of the round cases (besides ``trainer_seed``)."""

ROUND_CASES = {
    'tiny_seed5': dict(network='tiny', seed=5),
    'tiny_seed1005': dict(network='tiny', seed=1005),
    'painn_seed5': dict(network='painn', seed=5, steps_per_round=4),
}
"""Name -> network and ``trainer_seed`` of one fresh call of the old fit."""


# ---------------------------------------------------------------------------
# the two packages
# ---------------------------------------------------------------------------

def old_api():
    """The flow branch's names (``aimmd`` must be the exported old tree)."""
    from aimmd.flow.toy import TinyFlowNetwork
    from aimmd.network import flow_fit as ff
    from aimmd.network.joint_painn import JointPaiNN
    return SimpleNamespace(
        module=ff, FlowTrainer=ff.FlowTrainer, settings_from=ff._settings_from,
        frames_input=ff._MDFrameInput, Dataset=ff.FlowDataset, TinyFlowNetwork=TinyFlowNetwork,
        JointPaiNN=JointPaiNN, fit=ff.flow_fit, clear=ff._TRAINERS.clear)


def new_api():
    """The same names in this checkout."""
    from aimmd.cgaimmd.dataset import TrainingDataset
    from aimmd.cgaimmd.flow_matching import fit as ff
    from aimmd.cgaimmd.flow_matching.painn import JointPaiNN
    from aimmd.cgaimmd.flow_matching.toy import TinyFlowNetwork
    return SimpleNamespace(
        module=ff, FlowTrainer=ff.FlowTrainer, settings_from=ff._settings_from,
        frames_input=ff._MDFrameInput, Dataset=TrainingDataset, TinyFlowNetwork=TinyFlowNetwork,
        JointPaiNN=JointPaiNN, fit=ff.flow_matching_fit, clear=ff._RUNS.clear)


def randomize(net, seed=0, scale=0.5):
    """Seeded random weights of scale `scale` (no zero-initialized term hides)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return net


def build_network(api, kind, m):
    """The network of a case, with its random start weights."""
    if kind == 'painn':
        return randomize(api.JointPaiNN(m, [6, 6, 8][:m], **TINY_PAINN), seed=1, scale=0.2)

    class Paired(api.TinyFlowNetwork):
        """The toy network with an OT pairing the steps can see (the atoms swapped)."""

        def flow_pair(self, x1, x0):
            return x1.flip(1)

    cls = Paired if kind == 'paired' else api.TinyFlowNetwork
    return randomize(cls(n_atoms=m, lambda_clip=10.0))


# ---------------------------------------------------------------------------
# trainer steps on a hand-built dataset
# ---------------------------------------------------------------------------

def rows(xs, m):
    """Flat Å rows of m-atom frames: atom i at (x, 1.5 i, 0)."""
    x = np.zeros((len(xs), m, 3), np.float32)
    x[:, :, 0] = np.asarray(xs, np.float32)[:, None]
    x[:, :, 1] = 1.5 * np.arange(m, dtype=np.float32)[None, :]
    return x.reshape(len(xs), -1)


def hand_dataset(api, m):
    """Shooting points at -0.3 (outcome (2, 0)) and +0.3 ((0, 2)), anchors near -+1, reactive frames between.

    Chain 1 is the held-out chain where one is held out: its shooting points
    sit at -0.35 and +0.35.
    """
    sp_x = [-0.3, -0.31, 0.3, 0.31, -0.35, 0.35]
    fr_x = [-1.0, -1.05, -0.95, 1.0, 1.05, 0.95, -0.2, 0.0, 0.2, 0.1]
    fr_state = [1, 1, 1, 2, 2, 2, 0, 0, 0, 0]
    return api.Dataset(
        sp_rows=rows(sp_x, m), sp_r_first=np.array([2, 2, 0, 0, 2, 0], np.float32),
        sp_r_last=np.array([0, 0, 2, 2, 0, 2], np.float32),
        sp_group=np.array([0, 0, 0, 0, 1, 1], np.int64), sp_fname=['sp'] * 6,
        fr_rows=rows(fr_x, m), fr_state=np.array(fr_state, np.int8),
        fr_w=np.array([1.0, 6, 1, 2, 1, 1, 1, 3, 1, 2]),
        fr_group=np.array([-1, -1, -1, -1, -1, -1, 0, 0, 1, 0], np.int64),
        fr_fname=np.array(['fr'] * 10, dtype=object), fr_loc=np.arange(10, dtype=np.int64))


def hand_trainer(api, network, m, held=(), **settings):
    """A trainer on :func:`hand_dataset` with targets ``10 x`` on its reactive frames."""
    s = api.settings_from({**TRAINER, **settings})
    committor_fn = api.frames_input(SimpleNamespace(descriptor_transform=None), False)
    trainer = api.FlowTrainer(network, s, lambda_clip=10.0, device='cpu',
                              committor_fn=committor_fn)
    data = hand_dataset(api, m)
    trainer.set_data(data, held)
    pool = trainer.fm_candidates
    trainer.set_fm_targets(pool, data.fr_x[pool, 0, 0] * 10.0)
    return trainer


def state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def trainer_outcome(trainer, results):
    """What a step case stores: the steps' results and the trainer's final state."""
    optimizer = trainer.optimizer.state_dict()['state']
    return {
        'totals': [float(total) for total, _ in results],
        'terms': [dict(terms) for _, terms in results],
        'live': state(trainer.live), 'ema': state(trainer.network),
        'optimizer': {int(k): {name: torch.as_tensor(v).clone() for name, v in entry.items()}
                      for k, entry in optimizer.items()},
        'normalizer': trainer.normalizer.state_dict(),
        'skipped_steps': int(trainer.skipped_steps),
        'rng': json.dumps(trainer.rng.bit_generator.state),
        'generator': trainer.generator.get_state().clone(),
    }


def run_step_case(api, name):
    case = STEP_CASES[name]
    torch.manual_seed(0)
    net = build_network(api, case['network'], case['m'])
    trainer = hand_trainer(api, net, case['m'], held=case['held'], **case['settings'])
    results = [trainer.step() for _ in range(case['steps'])]
    return trainer_outcome(trainer, results)


# ---------------------------------------------------------------------------
# one round of the fit on a run folder
# ---------------------------------------------------------------------------

def write_fixture(directory):
    """Write the run folder of the round cases into `directory`; return ``(run, manifest)``."""
    from tests._cgaimmd_helpers import build_cgaimmd_run_dir

    rd = build_cgaimmd_run_dir(directory, n_chains=4, n_shots=3, n_free=1, n_atoms=N_ATOMS)
    run = Path(rd.run)
    shots = sorted(os.path.relpath(f, run) for f in run.glob('chain*/path*.xtc'))
    free = [[os.path.relpath(str(f), run) for f in path.fnames] for path in rd.free]
    manifest = {'shots': shots, 'shooting_index': rd.shooting_index, 'free': free}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return str(run), manifest


def fixture_ensemble(run, manifest):
    """The path ensemble the trainer would pass: shot paths whole, free trajectories split."""
    import aimmd
    from aimmd.pathensemble import PathEnsemble
    from aimmd.pathensemble.utils import assemble_pathensemble

    shots = PathEnsemble(*[aimmd.Path(os.path.join(run, f),
                                      shooting_index=manifest['shooting_index'])
                           for f in manifest['shots']])
    free = [aimmd.Path([os.path.join(run, f) for f in parts], remove_overlapping_frames=True)
            for parts in manifest['free']]
    return assemble_pathensemble(shots, free)


def ramp_weights(pathensemble, states, reweight_parameters=None):
    """The path weights of the round cases: ``1 + 0.1 k`` for the k-th path (in place of AIMMD's reweighting)."""
    return 1.0 + 0.1 * np.arange(len(pathensemble), dtype=np.float64)


class FakeWorker:
    """The members of AIMMD's trainer the fits read."""

    def __init__(self, directory, task='train'):
        self._directory = str(directory)
        self.directory = str(directory)
        self.task = task
        self.total_steps = 0
        self.must_stop = False


def round_params(network, initial_network=None):
    """The params the fits read (a namespace: no params file)."""
    return SimpleNamespace(network=network, states=STATES, sorted_states=STATES,
                           descriptor_transform=None, reweight_parameters={},
                           multi_system=False, rescale_committor=False,
                           initial_network=initial_network)


def round_settings(name):
    case = ROUND_CASES[name]
    settings = dict(ROUND_SETTINGS, trainer_seed=case['seed'])
    if 'steps_per_round' in case:
        settings['steps_per_round'] = case['steps_per_round']
    return settings


def round_outcome(losses, info, network, dataset, pool, lam, run):
    """What a round case stores."""
    record = info['record']
    return {
        'losses': [float(x) for x in losses], 'network': state(network),
        'held_out_chains': [int(c) for c in record['held_out_chains']],
        'n_trained_shots': int(record['n_trained_shots']),
        'loss_terms': dict(record['loss_terms']),
        'holdout_nll_ema': record['holdout_nll_ema'],
        'holdout_nll_live': record['holdout_nll_live'],
        'relabels_in_round': int(record['relabels_in_round']),
        'fm_pool': torch.as_tensor(np.asarray(pool, dtype=np.int64)),
        'fm_lambda': torch.as_tensor(np.asarray(lam, dtype=np.float64)),
        'sp_fname': [os.path.relpath(str(f), run) for f in dataset.sp_fname],
        'fr_w': torch.as_tensor(np.asarray(dataset.fr_w, dtype=np.float64)),
    }


def _record_rounds(api, folder):
    """The round cases with the old fit: every case a fresh call on a fresh copy of the run."""
    out = {'start': {}}
    source = os.path.join(os.path.abspath(folder), 'run')
    with open(os.path.join(source, 'manifest.json')) as handle:
        manifest = json.load(handle)
    api.module.path_weights = ramp_weights
    for name, case in ROUND_CASES.items():
        work = os.path.join(os.path.abspath(folder), f'work_{name}')
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(source, os.path.join(work, 'run'))
        run = os.path.join(work, 'run')
        torch.manual_seed(0)
        network = build_network(api, case['network'], N_ATOMS)
        out['start'][case['network']] = state(network)
        api.clear()
        params = round_params(network)
        worker = FakeWorker(run)
        losses, info = api.fit(params, fixture_ensemble(run, manifest), worker=worker,
                               **round_settings(name))
        trainer = next(iter(api.module._TRAINERS.values()))
        out[name] = round_outcome(losses, info, network, trainer.dataset, trainer.fm_pool,
                                  trainer.fm_lambda, run)
    return out


def _record(folder, output):
    import aimmd
    if Path(aimmd.__file__).resolve().parents[1] == ROOT:
        raise SystemExit(f'aimmd is imported from this checkout ({aimmd.__file__}); put the '
                         f'exported old tree first on PYTHONPATH')
    api = old_api()
    golden = {'steps': {name: run_step_case(api, name) for name in STEP_CASES},
              'rounds': _record_rounds(api, folder),
              'source': 'aimmd.network.flow_fit (FlowTrainer, flow_fit), flowdev bb98943',
              'torch': str(torch.__version__), 'numpy': str(np.__version__)}
    torch.save(golden, output)
    print(f'{len(STEP_CASES)} step cases and {len(ROUND_CASES)} round cases from '
          f'{aimmd.__file__} -> {output}')


if __name__ == '__main__':
    mode, folder = sys.argv[1], sys.argv[2]
    if mode == 'fixture':
        sys.path.insert(0, str(ROOT))
        print(write_fixture(folder)[0])
    elif mode == 'record':
        _record(folder, sys.argv[3])
    else:
        raise SystemExit(f'unknown mode {mode!r}: fixture or record')
