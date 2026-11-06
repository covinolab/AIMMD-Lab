"""The toy 1-D C-GAIMMD campaign with a trainer, for the end-to-end tests.

``tests/toy_1d_cgaimmd/params.py`` (part of the in-process chain test) has no
``fit``: these helpers copy it into a campaign folder (with
:func:`tests.test_toy_1d_cgaimmd.make_campaign`) and append one of two fits.

- ``'round_contract'``: :data:`ROUND_CONTRACT_FIT`, a stand-in that follows
  the contract of a C-GAIMMD round without learning (every round starts from
  ``initial_network``; without shooting points it is a no-op round; the
  metadata name the round and the trained shots). It needs nothing beyond
  this branch, so the trainer, launcher and restart hooks are tested end to
  end today.
- ``'flow_matching'``: :data:`FLOW_MATCHING_FIT`, the params-file wrapper of
  ``aimmd.cgaimmd.flow_matching.fit.flow_matching_fit`` (stream S4), five
  steps per round. Tests that use it skip without that module
  (:func:`fit_kind`), and must run once S4 is merged.
"""

from pathlib import Path

import pytest


FIT_KINDS = ('round_contract', 'flow_matching')
"""The fits of the campaign tests."""

ROUND_CONTRACT_FIT = '''

# --- the trainer's fit of the campaign tests (tests/_cgaimmd_toy_campaign.py) ---
import json as _json
import os as _os

import torch as _torch


def fit(params, pathensemble, verbose=False, worker=None):
    """A stand-in for a C-GAIMMD fit: the contract of a round, without learning.

    Every round starts from initial_network (no warm start). Without shooting
    points it is a no-op round (the network stays the initial one);
    otherwise the committor's output bias moves by 1e-3 per shooting point,
    so that each round's network differs while the committor stays about 0.
    The metadata name the round (one past the round of the run's
    network<states>.json) and the trained shots.
    """
    # the shots: one-file paths of the chain folders with a shooting index (free MD and the
    # initial-path margins carry shooting indices too)
    n_shots = sum(1 for path in pathensemble
                  if getattr(path, 'shooting_index', None) is not None
                  and len(path.fnames) == 1
                  and _os.path.basename(_os.path.dirname(str(path.fnames[0]))).startswith('chain'))
    params.network.load_state_dict(
        _torch.load(params._initial_network_fname(), weights_only=True))
    if n_shots:
        with _torch.no_grad():
            params.network.committor_net[-1].bias.add_(1e-3 * n_shots)
    previous = 0
    record = _os.path.join(worker._directory, f'network{params.sorted_states}.json')
    if _os.path.exists(record):
        with open(record) as handle:
            previous = int(_json.load(handle).get('round') or 0)
    metadata = {'round': previous + 1, 'n_trained_shots': n_shots, 'fit': 'round_contract',
                'seed': 0}
    # a fit that keeps state between calls keeps it where the trainer says (the fraction's own
    # folder in a kinetics convergence)
    state_dir = getattr(worker, 'kinetics_convergence_state_dir', None)
    if state_dir is not None:
        _os.makedirs(state_dir, exist_ok=True)
        with open(_os.path.join(state_dir, 'rounds.jsonl'), 'a') as handle:
            handle.write(_json.dumps(metadata) + '\\n')
    return [0.0], {'status': 'trained' if n_shots else 'no-op',
                   'reason': '' if n_shots else 'no shooting points yet',
                   'round': previous + 1, 'record': {}, 'network_metadata': metadata}
'''

FLOW_MATCHING_FIT = '''

# --- the trainer's fit of the campaign tests (tests/_cgaimmd_toy_campaign.py) ---
from aimmd.cgaimmd.flow_matching.fit import flow_matching_fit


def fit(params, pathensemble, verbose=False, worker=None):
    """The params-file wrapper of the flow-matching fit, five steps per round."""
    return flow_matching_fit(params, pathensemble, verbose=verbose, worker=worker,
                             steps_per_round=5)
'''


def fit_kind(request):
    """The fit of a parametrized test; skips the flow-matching fit without stream S4's module."""
    kind = request.param
    if kind == 'flow_matching':
        pytest.importorskip('aimmd.cgaimmd.flow_matching.fit',
                            reason='the flow-matching fit (stream S4) is not on this branch')
    return kind


def make_trained_campaign(folder, kind, extra=''):
    """The toy campaign folder (:func:`tests.test_toy_1d_cgaimmd.make_campaign`) with a fit.

    Parameters
    ----------
    folder : path-like
        The campaign folder (created).
    kind : str
        One of :data:`FIT_KINDS`.
    extra : str, optional
        Python appended after the fit (e.g. a bias).

    Returns
    -------
    pathlib.Path
        The folder.
    """
    from tests.test_toy_1d_cgaimmd import make_campaign
    folder = make_campaign(folder)
    text = {'round_contract': ROUND_CONTRACT_FIT, 'flow_matching': FLOW_MATCHING_FIT}[kind]
    with open(Path(folder) / 'params.py', 'a') as handle:
        handle.write(text + extra)
    return Path(folder)
