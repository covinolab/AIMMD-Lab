"""
aimmd.cgaimmd.bioemu2.pretrain
==============================

The initial network of a BioEmu2 C-GAIMMD run: identity adapters (the
generator is plain BioEmu2) and a committor head trained on unbiased in-state
frames only. Before the first shot there are no committor labels, but the
frames of the two states are known (C-GAIMMD's bootstrap, as for PaiNN).

:func:`pretrain_initial_network` also fits the head's feature scaler on these
frames (their clean and noised features); every later round keeps it, so the
initial network's frames set the scale of every head of the run.

Save the result as the run's ``initial_network`` (``pretrained/network{states}.h5``)::

    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, pretrain_initial_network
    network = BioEmu2Network('system')
    record = pretrain_initial_network(network, frames, states, network_file='pretrained/networkARB.h5')

Ported from the two pretraining scripts of the archived campaigns
(``faimmd/pretrain_flow_network.py``, ``faimmd/pretrain_holo_flow_network.py``),
made generic: the frames and their states are arguments, the trypsin
script's balanced A/B draws and its scaler on every frame are kept.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .head import SIGMA_RANGE, fit_scaler, in_state_nll

__all__ = ['pretrain_initial_network']


def pretrain_initial_network(network, frames, states, *, steps=3000, batch=128, lr=3e-4, weight_decay=1e-4,
                             ema=0.999, seed=0, feature_batch=16, network_file=None, verbose=False):
    """Pretrain `network` in place: identity adapters, a scaler and a head fitted on in-state frames.

    Parameters
    ----------
    network : BioEmu2Network
    frames : array-like
        ``(n, N, 3)`` or ``(n, 3 N)`` generator-atom coordinates (Ångström)
        of unbiased in-state frames.
    states : array-like
        ``(n,)`` the state of every frame: 1 = ``sorted_states[0]``, 2 =
        ``sorted_states[2]`` (the codes of :class:`aimmd.cgaimmd.dataset.TrainingDataset`).
        Both states must occur.
    steps, batch, lr, weight_decay, ema : optional
        AdamW with 100 warm-up steps and cosine decay; every step draws
        ``batch / 2`` frames of each state, each clean and noised.
    seed : int, optional
        Seeds the head (``seed + 1``), the noise and the draws.
    feature_batch : int, optional
        Structures per BioEmu2 pass.
    network_file : str or Path, optional
        Save ``network.state_dict()`` there (atomically) and the record next
        to it (``networkARB.h5`` -> ``networkARB.pretrain.json``).
    verbose : bool, optional
        Print the losses every 250 steps.

    Returns
    -------
    dict
        The record: frame counts, settings, final loss, the EMA head's
        median logits per state, seconds (and the file and its sha256).
    """
    t0 = time.time()
    states = np.asarray(states).reshape(-1)
    if not set(np.unique(states).tolist()) <= {1, 2} or not ((states == 1).any() and (states == 2).any()):
        raise ValueError('states must be 1 (first state) or 2 (last state), with frames of both')
    device = network.device
    x = network.to_bioemu2(np.asarray(frames, np.float32))
    if len(x) != len(states):
        raise ValueError(f'{len(x)} frames for {len(states)} states')
    x = x - x.mean(1, keepdim=True)
    network.reset_parameters(seed)
    features = network.engine().features
    rng = np.random.default_rng(seed)
    lo, hi = np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1])
    sigma_noisy = torch.as_tensor(np.exp(rng.uniform(lo, hi, len(x))).astype(np.float32), device=device)
    noise = torch.as_tensor(rng.standard_normal(tuple(x.shape)).astype(np.float32), device=device)
    clean, noisy = [], []
    with torch.no_grad():
        for i in range(0, len(x), feature_batch):
            xi = x[i:i + feature_batch]
            clean.append(features.features(xi, torch.full((len(xi),), network.sigma_ref, device=device)).half())
            xn = features.canonical(xi) + sigma_noisy[i:i + feature_batch, None, None] * noise[i:i + feature_batch]
            noisy.append(features.features(xn, sigma_noisy[i:i + feature_batch]).half())
    clean, noisy = torch.cat(clean), torch.cat(noisy)
    t_features = time.time() - t0
    scaler = fit_scaler(torch.cat([clean, noisy]).float().cpu().numpy(),
                        np.concatenate([np.full(len(clean), network.sigma_ref), sigma_noisy.cpu().numpy()]))
    for head in (network.head, network.head_ema):
        head.set_scaler(scaler)
    network.head_ema.load_state_dict(network.head.state_dict())
    head = network.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    state = torch.as_tensor(states, device=device)
    first, last = torch.nonzero(state == 1).reshape(-1), torch.nonzero(state == 2).reshape(-1)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    half = max(1, int(batch) // 2)
    log = []
    head.train()
    for step in range(int(steps)):
        f = min(1.0, (step + 1) / 100) * 0.5 * (1 + np.cos(np.pi * step / steps))
        optimizer.param_groups[0]['lr'] = lr * f
        pick = torch.cat([first[torch.randint(len(first), (half,), generator=generator, device=device)],
                          last[torch.randint(len(last), (half,), generator=generator, device=device)]])
        sigma = torch.cat([torch.full((len(pick),), network.sigma_ref, device=device), sigma_noisy[pick]])
        lam = head(torch.cat([clean[pick], noisy[pick]]).float(), sigma)
        loss = in_state_nll(lam, torch.cat([state[pick], state[pick]]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p_ema, p in zip(network.head_ema.parameters(), head.parameters()):
                p_ema.mul_(ema).add_(p.detach(), alpha=1 - ema)
        if step % 250 == 0 or step == steps - 1:
            log.append(dict(step=step, loss=float(loss.detach())))
            if verbose:
                print(json.dumps(log[-1]), flush=True)
    head.load_state_dict(network.head_ema.state_dict())
    head.eval()
    network.eval()
    with torch.no_grad():
        lam = network.head_ema(clean.float(), torch.full((len(clean),), network.sigma_ref, device=device))
    lam = lam.cpu().numpy()
    record = dict(n_first_state=int(len(first)), n_last_state=int(len(last)), steps=int(steps), batch=int(batch),
                  lr=float(lr), weight_decay=float(weight_decay), ema=float(ema), seed=int(seed),
                  final_loss=log[-1]['loss'] if log else None,
                  ema_logit_median_first_state=float(np.median(lam[states == 1])),
                  ema_logit_median_last_state=float(np.median(lam[states == 2])),
                  adapters='identity (plain BioEmu2)', network=network.settings(),
                  feature_seconds=round(t_features, 1), seconds=round(time.time() - t0, 1), log=log)
    if network_file is not None:
        from ..io import atomic_torch_save, atomic_write_json
        network_file = Path(network_file)
        network_file.parent.mkdir(parents=True, exist_ok=True)
        record['network_file'] = str(network_file)
        record['sha256'] = atomic_torch_save(network.state_dict(), str(network_file))
        atomic_write_json(network_file.with_suffix('.pretrain.json'), record)
    return record
