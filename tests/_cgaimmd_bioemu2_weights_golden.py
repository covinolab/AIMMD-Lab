"""The real-weights goldens of the BioEmu2 adaptor, recorded from the old code with the real BioEmu2 weights.

The tiny-model goldens (``tests/_cgaimmd_bioemu2_golden.py``) check the
width-agnostic parts of the port on CPU. These goldens check what only the
real model can show: the networks of the archived production campaigns,
read by :func:`aimmd.cgaimmd.bioemu2.load_archived_network`, give the values,
denoised coordinates and samples the old project code gave, with the real
``bioemu2-preview`` weights on a GPU.

The BioEmu2 code of the archived campaigns (the project folder
``committor_bioemu2_aimmd``, not under git; the md5 of every file used is
stored with the goldens) was run once on the workstation GPU, with the real
weights from the Hugging Face cache, on:

- the archived chignolin campaign of 27 Oct 2026 (equilibrium-free free MD,
  one trunk adapter, flat token features): its ``pretrained/networkARB.h5``
  (identity adapter, pretrained head) and its final ``networkARB.h5``;
- the archived trypsin-benzamidine campaign of 28 Oct 2026 (single and pair
  adapters, pooled features): its ``pretrained/networkARB.h5`` and its final
  ``networkARB.h5``.

What was recorded, per system (key prefixes ``chignolin.`` and ``trypsin.``)
and network (``pretrained.``, ``final.``):

- ``values``: the old network's ``forward`` (the logit committor of its EMA
  head) on :data:`N_FRAMES` fixed frames of that campaign: shooting points and
  in-state frames of its free MD. The chignolin frames are stored (100 x 166
  atoms); the trypsin frames are rows of the run's ``*.descriptors.npy`` files,
  stored as (file, row) and read from the archive by the tests;
- ``features``: the head's input features of the first two frames;
- ``denoiser.*``: one pass of the old ``ConditionedDenoiser`` with the
  network's adapters on three noisy copies of a frame at
  :data:`DENOISER_SIGMAS` with labels :data:`DENOISER_LAMBDAS` (the second row
  without a label), and the same pass without any label;
- ``sample``: one seeded sample per requested logit of :data:`SAMPLER_LAMBDAS`
  (chignolin) or :data:`TRYPSIN_SAMPLER_LAMBDAS` (trypsin), 200 steps of the
  old sampler as the old networks drove it.

Also stored: the model's widths, parameter count and checksum, the plain
BioEmu2 denoiser on the chignolin copies (``chignolin.plain``), the trypsin
system's 3PTB reference pose and core residues (the archived choices of the
canonical pose and the pocket), and the old atom map of trypsin.

Recording again is only needed when the archive or the settings below change.
From the root of this checkout, in the ``bioemu2`` conda env (bioemu2 0.1.0rc0,
cuEquivariance), on a GPU, with the old project folder ``<BE>``, the archive of
the campaigns ``<ARCHIVE>`` and the chignolin system folder of the old code
``<CHIGNOLIN>``::

    OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<BE> \\
        python tests/_cgaimmd_bioemu2_weights_golden.py record <BE> <ARCHIVE> <CHIGNOLIN> \\
        <md_reference_structures.npz> tests/data/cgaimmd_bioemu2_weights_goldens.npz

This module imports only the standard library and numpy at module level; the
tests import its constants.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
GOLDEN = DATA / 'cgaimmd_bioemu2_weights_goldens.npz'
"""The recorded outputs of the old code with the real weights."""
PEPTIDE = DATA / 'cgaimmd_bioemu2_peptide'
"""CLN025: its ``md.gro`` holds the coordinates of the old code's reference (``conf_A_box.gro``)."""
TRYPSIN_SYSTEM_YAML = DATA / 'cgaimmd_bioemu2_trypsin_benzamidine.yaml'
"""The BioEmu2 system of the archived trypsin campaign (copied from the old project)."""

ARCHIVE = Path('/mnt/plutus/lichtinger_committor_flow_archive/bioemu2_campaign_backups')
"""The backups of the archived campaigns (read only). The tests that need it skip without it."""
CHIGNOLIN_CAMPAIGN = '2026-10-27_chignolin-faimmd-equilibrium-free_full_run'
CHIGNOLIN_RUN = 'chignolin__faimmd-online__bioemu2-trunk-single__sequential-fresh-earlystop-equilibrium-free__s0'
TRYPSIN_CAMPAIGN = '2026-10-28_trypsin-faimmd_full_run'
TRYPSIN_RUN = ('trypsin-benzamidine__faimmd-online__bioemu2-holo-trunk-single-pair-ligw5__'
               'sequential-fresh-earlystop-equilibrium-free__s0')
NETWORKS = {'pretrained': 'pretrained/networkARB.h5', 'final': '{run}/networkARB.h5'}
"""The archived network files of a campaign, relative to the campaign folder."""
OLD_PROJECT = Path('/home/covino-shared/data/lichtinger/committor_bioemu2_aimmd')
"""The old project folder (read only); the trypsin tests need its saved trunk embeddings and skip without it."""
OLD_TRYPSIN_TRUNK = OLD_PROJECT / 'trypsin' / 'system' / 'embeddings_holo_equilibrium' / 'embeddings.pt'
"""The trunk embeddings the archived trypsin campaign used (bf16 autocast, not query-only; md5 in the goldens' meta)."""
TRYPSIN_N_ATOMS = 3239
"""The generator atoms of the trypsin campaign: protein (with hydrogens), benzamidinium, Ca2+."""

CHECKPOINT = 'bioemu2-preview'
HF_SNAPSHOT = 'dd4b0c1b69984b68edcb6200a544213be4d75242'
PARAMETER_CHECKSUM = '09c744acec703a153d62842f989d9ec57269d5e4'
"""sha1 of the bioemu2-preview parameters, recorded by every BioEmu2 campaign since 20 Oct 2026."""
N_PARAMETERS = 511684478
WIDTHS = {'token': 768, 'atom': 128, 'single': 384, 'pair': 256}
"""Widths of bioemu2-preview: readout token and atom features, trunk single and pair embeddings."""

N_FRAMES = 100
FRAME_SEED = 20261106
N_SHOOTING_POINTS = 70
"""Chignolin: shooting points of the campaign; the rest are in-state frames of A and B (15 each)."""
N_TRYPSIN_PATH_ROWS = 60
"""Trypsin: rows of the shooting chains' path descriptors; the rest are free-MD frames of A and B (20 each)."""
SIGMA_REF = 0.05
DENOISER_SIGMAS = (0.5, 4.0, 16.0)
DENOISER_LAMBDAS = (-3.0, 0.0, 2.5)
DENOISER_UNCOND = (False, True, False)
DENOISER_SEED = 17
SAMPLER_SEED = 11
SAMPLER_STEPS = 200
SAMPLER_LAMBDAS = (-4.0, 0.0, 4.0)
TRYPSIN_SAMPLER_LAMBDAS = (-4.0, 4.0)
VALUE_BATCH = 16
OLD_FILES = ('bioemu2_committor/finetune.py', 'bioemu2_committor/guided_sampler.py',
             'bioemu2_committor/extractor.py', 'bioemu2_committor/holo.py', 'bioemu2_committor/joint.py',
             'bioemu2_committor/aimmd_flow.py', 'bioemu2_committor/aimmd_flow_holo.py',
             'trypsin/trypsin_flow.py', 'trypsin/trypsin_features.py', 'trypsin/trypsin_atom_map.py',
             'trypsin/system/trypsin_atom_map.json', 'trypsin/system/trypsin_benzamidine.yaml',
             'trypsin/system/embeddings_holo_equilibrium/embeddings.pt',
             'results/trypsin/trypsin_holo_constants.npz', 'results/trypsin/native_pose_check.json')


def campaign_file(campaign, relative, archive=ARCHIVE):
    """A file of an archived campaign (``{run}`` in `relative` is the campaign's run folder)."""
    run = CHIGNOLIN_RUN if campaign == CHIGNOLIN_CAMPAIGN else TRYPSIN_RUN
    return Path(archive) / campaign / relative.format(run=run)


def trypsin_frames(sources, archive=ARCHIVE):
    """``(n, 3239, 3)`` float32: the trypsin frames named by ``sources`` (``'relative/path.npy:row'``)."""
    out, cache = [], {}
    for source in sources:
        path, _, row = str(source).rpartition(':')
        if path not in cache:
            cache[path] = np.load(campaign_file(TRYPSIN_CAMPAIGN, path, archive), mmap_mode='r')
        out.append(np.asarray(cache[path][int(row)], dtype=np.float32))
    return np.stack(out).reshape(len(out), TRYPSIN_N_ATOMS, 3)


def denoiser_input(reference, n_pad, rng_seed=DENOISER_SEED):
    """``(3, n_pad, 3)`` float32: `reference` (centred, ``(M, 3)``) padded with zeros, plus noise at each sigma."""
    rng = np.random.default_rng(rng_seed)
    k = len(DENOISER_SIGMAS)
    x0 = np.zeros((k, n_pad, 3), np.float32)
    x0[:, :len(reference)] = reference
    noise = rng.normal(size=x0.shape).astype(np.float32)
    return (x0 + np.asarray(DENOISER_SIGMAS, np.float32)[:, None, None] * noise).astype(np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------- recording
def _np(t):
    return t.detach().float().cpu().numpy().copy()


def _chignolin_frames(archive, rng):
    folder = Path(archive) / CHIGNOLIN_CAMPAIGN
    data = np.load(next(folder.glob('campaign_head_data__*.npz')))
    sp = np.sort(rng.choice(len(data['sp_rows']), N_SHOOTING_POINTS, replace=False))
    n_each = (N_FRAMES - N_SHOOTING_POINTS) // 2
    anchors = []
    for state in (1, 2):
        rows = np.flatnonzero(data['an_state'] == state)
        anchors.append(np.sort(rng.choice(rows, n_each, replace=False)))
    frames = np.concatenate([data['sp_rows'][sp], data['an_rows'][anchors[0]], data['an_rows'][anchors[1]]])
    sources = [f'sp_rows:{i}' for i in sp] + [f'an_rows:{i}' for i in np.concatenate(anchors)]
    return frames.reshape(N_FRAMES, -1, 3).astype(np.float32), np.array(sources)


def _trypsin_sources(archive, rng):
    run = Path(archive) / TRYPSIN_CAMPAIGN / TRYPSIN_RUN
    rows = []
    for path in sorted(run.glob('chainR*/path*.xtc.descriptors.npy')):
        n = len(np.load(path, mmap_mode='r'))
        rows += [f'{{run}}/{path.relative_to(run).as_posix()}:{i}' for i in range(n)]
    picked = [rows[i] for i in np.sort(rng.choice(len(rows), N_TRYPSIN_PATH_ROWS, replace=False))]
    n_each = (N_FRAMES - N_TRYPSIN_PATH_ROWS) // 2
    for state in ('freeA', 'freeB'):
        parts = sorted((run / state).glob('traj*.part*.xtc.descriptors.npy'))
        biggest = max(parts, key=lambda p: len(np.load(p, mmap_mode='r')))
        n = len(np.load(biggest, mmap_mode='r'))
        index = np.sort(rng.choice(n, n_each, replace=False))
        picked += [f'{{run}}/{biggest.relative_to(run).as_posix()}:{i}' for i in index]
    return np.array(picked)


def _values(net, frames):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), VALUE_BATCH):
            out.append(_np(net(torch.as_tensor(frames[i:i + VALUE_BATCH]).reshape(-1, frames[0].size))).reshape(-1))
    return np.concatenate(out)


def _record_denoiser(out, prefix, cond, x, trunk):
    import torch
    sigma = torch.tensor(DENOISER_SIGMAS, device=cond.model.device)
    xt = torch.as_tensor(x, device=cond.model.device)
    with torch.no_grad():
        out[f'{prefix}.denoiser.labelled'] = _np(cond.denoise(None, trunk, xt, sigma,
                                                              torch.tensor(DENOISER_LAMBDAS, device=xt.device),
                                                              torch.tensor(DENOISER_UNCOND, device=xt.device)))
        out[f'{prefix}.denoiser.no_label'] = _np(cond.denoise(None, trunk, xt, sigma, None))


def _plain_denoiser(model, batch, trunk, x):
    """Plain BioEmu2 on k copies, as its sampler calls it (``compute_static_context`` with multiplicity k)."""
    import torch
    k = len(x)
    xt = torch.as_tensor(x, device=model.device)
    with torch.no_grad():
        context = model.compute_static_context(batch, trunk, k)
        return _np(model.denoise(context, trunk, xt, torch.tensor(DENOISER_SIGMAS, device=xt.device)).denoised_coords)


def record(be_root, archive, chignolin_dir, md_reference, out_path):
    import os
    import torch
    be_root, archive, chignolin_dir = Path(be_root).resolve(), Path(archive), Path(chignolin_dir).resolve()
    sys.path.insert(0, str(be_root))
    sys.path.insert(0, str(be_root / 'trypsin'))
    import bioemu2
    from bioemu2.structure.preparation import prepare_sampling_inputs

    from bioemu2_committor import aimmd_flow, aimmd_flow_holo, extractor, guided_sampler, joint
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    device = torch.device('cuda')
    out, meta_times = {}, {}
    real_load_model = bioemu2.load_model
    t0 = time.time()
    model = real_load_model(CHECKPOINT, device=device, precision='32-true', matmul_precision=None)
    meta_times['load_model_s'] = round(time.time() - t0, 1)

    def shared_loader(*args, **kwargs):                     # one model for every old network of this recording
        return model
    bioemu2.load_model = shared_loader
    out['model.parameter_checksum'] = np.array(extractor.parameter_checksum(model))
    out['model.n_parameters'] = np.array(sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(FRAME_SEED)

    # ------------------------------------------------------------------ chignolin (old BioEmu2FlowNetwork)
    extractor.SYSTEM_DIR = chignolin_dir
    extractor.REFERENCE_GRO = chignolin_dir / 'conf_A_box.gro'
    frames, sources = _chignolin_frames(archive, rng)
    out['chignolin.frames'] = frames
    out['chignolin.frame_sources'] = sources
    files = {}
    for which, relative in NETWORKS.items():
        path = campaign_file(CHIGNOLIN_CAMPAIGN, relative, archive)
        files[f'chignolin.{which}'] = dict(file=str(path), sha256=_sha256(path))
        net = aimmd_flow.BioEmu2FlowNetwork(shared=False, cfg_weight=1.0, gen_batch=len(SAMPLER_LAMBDAS),
                                            value_batch=VALUE_BATCH)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        net = net.to(device).eval()
        eng = net.engine()
        ex, cond = eng['ex'], eng['cond']
        t0 = time.time()
        out[f'chignolin.{which}.values'] = _values(net, frames)
        meta_times[f'chignolin.{which}.values_s'] = round(time.time() - t0, 2)
        out[f'chignolin.{which}.values_repeat'] = _values(net, frames)
        with torch.no_grad():
            out[f'chignolin.{which}.features'] = _np(net.features(torch.as_tensor(frames[:2], device=device)))
        if which == 'pretrained':
            with torch.no_grad():
                ex.denoise(ex.canonical(torch.as_tensor(frames[:1], device=device)),
                           torch.tensor([SIGMA_REF], device=device))
            out['chignolin.widths'] = np.array([int(ex._store['a'].shape[-1]), int(ex._store['q'].shape[-1]),
                                                int(ex.trunk.single.shape[-1]), int(ex.trunk.pair.shape[-1])])
            for name in ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram'):
                value = getattr(ex.trunk, name)
                out[f'chignolin.trunk_sha1.{name}'] = np.array(hashlib.sha1(
                    value.detach().float().cpu().numpy().tobytes()).hexdigest())
            x = denoiser_input(_np(ex.reference), ex.n_pad)
            out['chignolin.denoiser.x'] = x
            _, batch = prepare_sampling_inputs(ex.system, device=device)
            out['chignolin.plain'] = _plain_denoiser(ex.model, batch, ex.trunk, x)
        _record_denoiser(out, f'chignolin.{which}', cond, out['chignolin.denoiser.x'], ex.trunk)
        lam = torch.tensor(SAMPLER_LAMBDAS).clamp(-joint.LAMBDA_CLIP, joint.LAMBDA_CLIP).to(device)
        t0 = time.time()
        coords, _ = guided_sampler.sample(ex.model, ex.system, ex.trunk, n=len(lam), batch_size=len(lam),
                                          seed=SAMPLER_SEED, num_steps=SAMPLER_STEPS, denoise=net._denoise_hook(lam))
        meta_times[f'chignolin.{which}.sample_s'] = round(time.time() - t0, 1)
        out[f'chignolin.{which}.sample'] = np.asarray(coords, np.float32)[:, :extractor.N_HEAVY]
        ex._hook.remove()
        del net, eng, ex, cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ trypsin (old HoloFlowNetwork)
    import trypsin_atom_map
    import trypsin_flow
    campaign = archive / TRYPSIN_CAMPAIGN
    system = trypsin_flow.TrypsinSystem(campaign / 'run.gro', campaign / 'topol.top')
    sources = _trypsin_sources(archive, rng)
    out['trypsin.frame_sources'] = sources
    frames = trypsin_frames(sources, archive)
    out['trypsin.frames_sha1'] = np.array(hashlib.sha1(frames.tobytes()).hexdigest())
    out['trypsin.bioemu2_to_md'] = np.asarray(system.be2_index, np.int64)
    constants = np.load(be_root / 'results' / 'trypsin' / 'trypsin_holo_constants.npz')
    out['trypsin.pocket_tokens'] = constants['pocket_tokens']
    out['trypsin.core_bioemu2'] = constants['core_be2']
    core_residues = json.loads((be_root / 'results' / 'trypsin' / 'native_pose_check.json').read_text())[
        'definitions']['core_residues_sequential']
    out['trypsin.core_residues'] = np.asarray(core_residues, np.int64)
    reference = np.load(md_reference)
    names = list(reference['crystal_names'])
    out['trypsin.reference_3ptb'] = np.asarray(reference['crystal_xyz'][names.index('3PTB')], np.float32)
    assert trypsin_atom_map.N_BIOEMU == len(out['trypsin.reference_3ptb'])
    for which, relative in NETWORKS.items():
        path = campaign_file(TRYPSIN_CAMPAIGN, relative, archive)
        files[f'trypsin.{which}'] = dict(file=str(path), sha256=_sha256(path))
        net = aimmd_flow_holo.HoloFlowNetwork(system, kind='trunk_single_pair', cfg_weight=1.0,
                                              gen_batch=len(TRYPSIN_SAMPLER_LAMBDAS), value_batch=VALUE_BATCH)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        net = net.to(device).eval()
        eng = net.engine()
        ex, cond = eng['ex'], eng['cond']
        t0 = time.time()
        out[f'trypsin.{which}.values'] = _values(net, frames)
        meta_times[f'trypsin.{which}.values_s'] = round(time.time() - t0, 2)
        out[f'trypsin.{which}.values_repeat'] = _values(net, frames)
        with torch.no_grad():
            out[f'trypsin.{which}.features'] = _np(net.features(torch.as_tensor(frames[:2], device=device)))
        if which == 'pretrained':
            for name in ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram'):
                value = getattr(ex.trunk, name)
                out[f'trypsin.trunk_sha1.{name}'] = np.array(hashlib.sha1(
                    value.detach().float().cpu().numpy().tobytes()).hexdigest())
            with torch.no_grad():
                canonical = _np(ex.canonical_be2(net.to_be2(torch.as_tensor(frames[:1], device=device)))[0])
            out['trypsin.denoiser.x'] = denoiser_input(canonical, ex.n_pad)
        _record_denoiser(out, f'trypsin.{which}', cond, out['trypsin.denoiser.x'], ex.trunk)
        t0 = time.time()
        out[f'trypsin.{which}.sample'] = net.generate_be2(
            torch.tensor(TRYPSIN_SAMPLER_LAMBDAS, device=device), seed=SAMPLER_SEED).astype(np.float32)
        meta_times[f'trypsin.{which}.sample_s'] = round(time.time() - t0, 1)
        print(json.dumps(meta_times), flush=True)
        del net, eng, ex, cond
        torch.cuda.empty_cache()

    bioemu2.load_model = real_load_model
    from importlib.metadata import version
    meta = dict(torch=torch.__version__, bioemu2=version('bioemu2'),
                cuequivariance_torch=version('cuequivariance-torch'), bioemu2_file=bioemu2.__file__,
                numpy=np.__version__, python=sys.version.split()[0], device=torch.cuda.get_device_name(0),
                old_files={name: hashlib.md5((be_root / name).read_bytes()).hexdigest() for name in OLD_FILES},
                networks=files, chignolin_system=str(chignolin_dir), md_reference=str(md_reference),
                times=meta_times, recorded=time.strftime('%Y-%m-%d %H:%M:%S'),
                note='recorded by tests/_cgaimmd_bioemu2_weights_golden.py from the old BioEmu2 project code')
    out['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **out)
    print(json.dumps(meta, indent=1))
    print(f'{len(out)} arrays, {Path(out_path).stat().st_size} bytes -> {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != 'record':
        sys.exit(__doc__)
    record(*sys.argv[2:])
