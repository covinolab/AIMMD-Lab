"""The port-equivalence goldens of the BioEmu2 adaptor, recorded from the old code on the tiny model.

The BioEmu2 code of the archived production campaigns (the project folder
``committor_bioemu2_aimmd``, not under git; the md5 of every file used is
stored with the goldens) was run once, on the CPU, on the tiny random-weight
BioEmu2 of ``tests/data/bioemu2_tiny_production_shaped.yaml`` (seed
:data:`MODEL_SEED`) and the two test systems of
``tests/_cgaimmd_bioemu2_test_systems.py``. What it returned is stored in
``tests/data/cgaimmd_bioemu2_goldens.npz``; the ``aimmd.cgaimmd.bioemu2``
tests compare the new modules with it.

What was recorded, per system (key prefixes ``peptide.`` and ``ligand.``):

- the trunk embeddings of the query-only MSA in equilibrium mode;
- the atom map of the old code: for the peptide (CLN025 = chignolin, the
  archived chignolin campaign's system) ``extractor.heavy_atom_map`` on the
  project's ``systems/chignolin`` folder; for the protein-ligand system
  (trypsin residues 187-206 + benzamidinium) the stored trypsin map
  ``trypsin/system/trypsin_atom_map.json`` restricted to these atoms;
- the features of the committor head (``extractor.BioEmu2Features`` via
  ``joint.token_features``: flat token features; ``holo.HoloFeatures``:
  pooled ligand / pocket / protein features) of four frames, at sigma_ref and
  at one sigma per frame, and the canonical poses;
- ``finetune.TrunkAdapter`` (single and pair; with and without a learned
  null) and ``finetune.ConditionedDenoiser`` (kinds ``trunk_single`` and
  ``trunk_single_pair``) with seeded, non-zero adapter weights: one denoiser
  pass with labels, unlabelled rows and no label, the plain BioEmu2 pass, and
  the classifier-free-guidance hook;
- ``guided_sampler.sample`` driven as the old networks drive it (chunks of
  ``gen_batch`` labels, seed + chunk start, the network's denoise hook), with
  :data:`SAMPLER_STEPS` steps instead of 200, at guidance weight 1 and 2;
- the values (``forward``) of the old networks (``aimmd_flow.BioEmu2FlowNetwork``,
  ``aimmd_flow_holo.HoloFlowNetwork``) with a seeded head;
- one whole training round of the old fits
  (``aimmd_flow_fit.sequential_round``, ``aimmd_flow_holo_fit.holo_sequential_round``)
  on a synthetic dataset, with :data:`ROUND_SETTINGS`: the head losses of every
  step, the denoising losses of the three adapter steps, the label probe, and
  the final head and adapter weights.

The old code hard-codes the production widths (768 / 384 / 256, 166 atoms,
...). Where it does, the recorder sets the module constants to the tiny
model's widths before building the old objects; nothing else is changed, and
``bioemu2.load_model`` is replaced by a loader of the tiny checkpoint.

Recording again is only needed when the tiny model, the test systems or the
settings below change. From the root of this checkout, in the ``bioemu2``
conda env (bioemu2 0.1.0rc0), with the old project folder ``<BE>``::

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<BE> \\
        python tests/_cgaimmd_bioemu2_golden.py record <BE> <committor_flow>/systems/chignolin \\
        tests/data/cgaimmd_bioemu2_goldens.npz

This module imports only the standard library and numpy at module level; the
tests import its constants and :func:`write_tiny_checkpoint`.
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / 'data'
GOLDEN = DATA / 'cgaimmd_bioemu2_goldens.npz'
"""The recorded outputs of the old code."""
TINY_CONFIG = DATA / 'bioemu2_tiny_production_shaped.yaml'
"""The vendored config of the tiny model."""
PEPTIDE = DATA / 'cgaimmd_bioemu2_peptide'
PROTEIN_LIGAND = DATA / 'cgaimmd_bioemu2_protein_ligand'

MODEL_SEED = 0
"""``torch.manual_seed`` before the tiny score model is instantiated."""
SIGMA_REF = 0.05
"""The noise level (Angstrom) of a clean structure's features (the old ``joint.SIGMA_REF``)."""
FRAME_SIGMAS = (0.05, 0.7, 4.0, 16.0)
"""One noise level per frame for the second feature call."""
DENOISER_SIGMAS = (0.5, 4.0, 16.0)
DENOISER_LAMBDAS = (-3.0, 0.0, 2.5)
DENOISER_UNCOND = (False, True, False)
ADAPTER_LAMBDAS = (-15.0, -1.0, 4.0)
"""Labels of the direct TrunkAdapter calls (the first is clipped at -10)."""
ADAPTER_KEEP = (True, False, True)
ADAPTER_SEED = 5
"""``torch.manual_seed`` before the adapter weights are perturbed: p += 0.05 N(0, 1)."""
CFG_TARGET, CFG_WEIGHT = 1.5, 2.0
"""The classifier-free-guidance hook call (``ConditionedDenoiser.cfg_denoise``)."""
SAMPLER_LAMBDAS = (-12.0, 0.0, 3.0)
SAMPLER_SEED = 11
SAMPLER_STEPS = 5
GEN_BATCH = 2
HEAD_SEED = 101
"""``torch.manual_seed`` before the old network (and its head) is built."""
HEAD_HIDDEN = (16,)
ROUND_SETTINGS = dict(batch_sp=4, batch_anchor=4, batch_den=4, micro_batch=2, n_anchor_max=8, pool=8,
                      noisy_copies=2, head_steps_without_holdout=5, adapter_steps=3, p_uncond=0.2, seed=0,
                      verbose=True, adapter_log_every=1)
"""Settings of the recorded training round (shared by both systems; 3 adapter steps)."""
ROUND_SETTINGS_PEPTIDE = dict(min_shots_for_generator=2)
ROUND_SETTINGS_LIGAND = dict(min_shots_for_generator=0, min_frames_for_generator=0, ligand_weight=5.0,
                             feature_batch=4, probe=True, pool_weighting='uniform')
DATASET_OUTCOMES = ((1, 1), (2, 0), (0, 2), (1, 1))
"""``(n_first, n_last)`` of the synthetic dataset's shooting points."""
DATASET_STATES = (1, 1, 1, 2, 2, 2, 0, 0)
"""``fr_state`` of the synthetic dataset's frames (0 = in no state)."""
FRAME_SEED = 20261105
"""numpy seed of the synthetic frames (they are stored, so the stream need not stay stable)."""
LIGAND_FRAGMENT_FIRST_RESIDUE = 187
"""The trypsin residue (MD numbering) of the protein-ligand system's first residue."""
OLD_FILES = ('bioemu2_committor/finetune.py', 'bioemu2_committor/guided_sampler.py',
             'bioemu2_committor/extractor.py', 'bioemu2_committor/holo.py', 'bioemu2_committor/joint.py',
             'bioemu2_committor/head.py', 'bioemu2_committor/aimmd_flow.py',
             'bioemu2_committor/aimmd_flow_holo.py', 'bioemu2_committor/aimmd_flow_fit.py',
             'bioemu2_committor/aimmd_flow_holo_fit.py', 'trypsin/system/trypsin_atom_map.json')


def write_tiny_checkpoint(folder):
    """Write the tiny model as ``folder/config.yaml`` + ``folder/model.ckpt``; return the checkpoint path.

    The weights are ``AveragedModel(instantiate(cfg.score_model)).state_dict()``
    after ``torch.manual_seed(MODEL_SEED)``: the layout of a released
    checkpoint (``model_ema``), so ``bioemu2.load_model(checkpoint_path=...)``
    loads it.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.optim.swa_utils import AveragedModel
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.yaml').write_text(TINY_CONFIG.read_text())
    cfg = OmegaConf.load(folder / 'config.yaml')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        score_model = instantiate(cfg.score_model)
    torch.save({'model_ema': AveragedModel(score_model).state_dict()}, folder / 'model.ckpt')
    return folder / 'model.ckpt'


def load_tiny_model(checkpoint):
    """The tiny model as the tests and the old code load it (fp32, no TF32 setting)."""
    from bioemu2.model import load_model            # not bioemu2.load_model, which the recorder replaces
    return load_model(checkpoint_path=Path(checkpoint), device='cpu', precision='32-true',
                      matmul_precision=None)


def read_gro(path):
    """``(atoms [(resid, resname, name)], xyz (n, 3) Angstrom, box (3,) Angstrom)`` of a .gro file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1])
    atoms, xyz = [], []
    for line in lines[2:2 + n]:
        atoms.append((int(line[0:5]), line[5:10].strip(), line[10:15].strip()))
        xyz.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
    box = np.array([float(v) for v in lines[2 + n].split()[:3]]) * 10.0
    return atoms, np.array(xyz) * 10.0, box


def read_top_bonds(path):
    """``(masses (n,), bonds [(i, j)] 0-based)`` of a test topology (molecules in file order, one copy each)."""
    masses, bonds, offset, n_mol, section = [], [], 0, 0, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.strip('[] ').lower()
            if section == 'moleculetype':
                offset += n_mol
                n_mol = 0
            continue
        fields = line.split()
        if section == 'atoms':
            masses.append(float(fields[7]))
            n_mol += 1
        elif section == 'bonds':
            bonds.append((offset + int(fields[0]) - 1, offset + int(fields[1]) - 1))
    return np.array(masses), bonds


def whole_orthorhombic(xyz, bonds, box):
    """Make each bonded molecule whole (minimum image per bond along a BFS tree) and put every molecule
    after the first at the periodic image nearest the first one (the test inputs; not the code under test)."""
    n = len(xyz)
    neighbours = [[] for _ in range(n)]
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)
    out = np.array(xyz, dtype=np.float64)
    seen = np.zeros(n, bool)
    molecules = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        order, frontier = [root], [root]
        while frontier:
            nxt = []
            for p in frontier:
                for c in sorted(neighbours[p]):
                    if not seen[c]:
                        seen[c] = True
                        d = out[c] - out[p]
                        out[c] = out[p] + d - box * np.round(d / box)
                        order.append(c)
                        nxt.append(c)
            frontier = nxt
        molecules.append(sorted(order))
    first = molecules[0]
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * box
    for mol in molecules[1:]:
        best = min(shifts, key=lambda s: np.min(np.linalg.norm(
            (out[mol] + s)[:, None] - out[first][None], axis=-1)))
        out[mol] += best
    return out


def random_frames(reference, n, rng, noise=0.3):
    """``reference`` and ``n - 1`` copies with Gaussian noise, a random rotation and a translation (float32)."""
    frames = [np.asarray(reference, np.float64)]
    for _ in range(n - 1):
        q = rng.normal(size=4)
        w, x, y, z = q / np.linalg.norm(q)
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        moved = (reference + rng.normal(scale=noise, size=reference.shape)) @ rot.T + rng.normal(scale=5.0,
                                                                                               size=3)
        frames.append(moved)
    return np.stack(frames).astype(np.float32)


# ------------------------------------------------------------------------------------------------- recording
def _seed_adapter(module, seed):
    import torch
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for p in module.parameters():
            p.add_(0.05 * torch.randn_like(p))


def _state(out, prefix, module):
    for name, value in module.state_dict().items():
        out[f'{prefix}.{name}'] = value.detach().cpu().numpy()


def _trunk(out, prefix, trunk):
    for name in ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram'):
        out[f'{prefix}.{name}'] = getattr(trunk, name).detach().cpu().float().numpy()


def _np(t):
    return t.detach().cpu().numpy()


def _round_log(text):
    """The JSON lines a round printed: (head lines, adapter step lines, label probe or None, the rest)."""
    head, steps, probe, rest = [], [], None, []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get('phase') == 'adapter' and 'step' in rec:
            steps.append(rec)
        elif rec.get('phase') == 'label probe':
            probe = rec
        elif rec.get('phase') == 'head':
            head.append(rec)
        else:
            rest.append(rec)
    return head, steps, probe, rest


def _dataset(reference_md, rng):
    """A synthetic round dataset (the fields the old rounds read) from noisy copies of the reference."""
    n_sp, n_fr = len(DATASET_OUTCOMES), len(DATASET_STATES)
    rows = random_frames(reference_md, n_sp + n_fr + 1, rng, noise=0.4)[1:].reshape(n_sp + n_fr, -1)
    return types.SimpleNamespace(
        sp_rows=rows[:n_sp], sp_r_first=np.array([a for a, _ in DATASET_OUTCOMES], np.float64),
        sp_r_last=np.array([b for _, b in DATASET_OUTCOMES], np.float64),
        fr_rows=rows[n_sp:], fr_state=np.array(DATASET_STATES, np.int64))


def _record_adapters(out, prefix, cond, trunk, k_lam, *, pair):
    """Direct TrunkAdapter calls on expanded trunk tensors (the denoiser's own adapters)."""
    import torch
    lam = torch.tensor(ADAPTER_LAMBDAS)
    keep = torch.tensor(ADAPTER_KEEP)
    gen = torch.Generator().manual_seed(k_lam)
    x = trunk.single.expand(3, *trunk.single.shape[1:]).float()
    x = x + 0.1 * torch.randn(x.shape, generator=gen)
    out[f'{prefix}.single.x'] = _np(x)
    with torch.no_grad():
        out[f'{prefix}.single.out'] = _np(cond.single_adapter(x, lam, keep))
    if pair:
        z = trunk.pair.expand(3, *trunk.pair.shape[1:]).float()
        z = z + 0.1 * torch.randn(z.shape, generator=gen)
        out[f'{prefix}.pair.x'] = _np(z)
        with torch.no_grad():
            out[f'{prefix}.pair.out'] = _np(cond.pair_adapter(z, lam, keep))


def _record_denoiser(out, prefix, cond, model, trunk, batch, reference_be2, n_pad):
    """One conditioned pass, an unlabelled pass, the plain BioEmu2 pass and the CFG hook."""
    import torch
    from bioemu2.biomoa.models.score_model import TrunkEmbeddings
    from bioemu2_committor.finetune import expand_batch
    k = len(DENOISER_SIGMAS)
    gen = torch.Generator().manual_seed(17)
    x0 = torch.zeros(k, n_pad, 3)
    x0[:, :len(reference_be2)] = torch.as_tensor(reference_be2, dtype=torch.float32)
    sigma = torch.tensor(DENOISER_SIGMAS)
    x = x0 + sigma[:, None, None] * torch.randn(x0.shape, generator=gen)
    out[f'{prefix}.x'] = _np(x)
    with torch.no_grad():
        out[f'{prefix}.labelled'] = _np(cond.denoise(None, trunk, x, sigma, torch.tensor(DENOISER_LAMBDAS),
                                                     torch.tensor(DENOISER_UNCOND)))
        out[f'{prefix}.no_label'] = _np(cond.denoise(None, trunk, x, sigma, None))
        rep = lambda v: v.expand(k, *v.shape[1:])                              # noqa: E731
        trunk_k = TrunkEmbeddings(s_inputs=rep(trunk.s_inputs), single=rep(trunk.single).float(),
                                  pair=rep(trunk.pair).float(),
                                  relative_position_encoding=rep(trunk.relative_position_encoding),
                                  distogram=rep(trunk.distogram))
        context_k = model.score_model.compute_static_context(expand_batch(batch, k), trunk_k)
        out[f'{prefix}.plain'] = _np(model.denoise(context_k, trunk_k, x, sigma).denoised_coords)
        out[f'{prefix}.cfg'] = _np(cond.cfg_denoise(CFG_TARGET, CFG_WEIGHT)(None, trunk, x, sigma, 0.5))


def record(be_root, chignolin_dir, out_path):
    import os
    import torch
    be_root, chignolin_dir = Path(be_root).resolve(), Path(chignolin_dir).resolve()
    sys.path.insert(0, str(be_root))
    import bioemu2
    from bioemu2 import MSA, System, TrunkEmbeddings
    from bioemu2.structure.preparation import prepare_sampling_inputs

    from bioemu2_committor import (aimmd_flow, aimmd_flow_fit, aimmd_flow_holo, aimmd_flow_holo_fit, extractor,
                                   finetune, guided_sampler, holo, joint)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    out = {}
    tmp = Path(tempfile.mkdtemp(prefix='cgaimmd_bioemu2_golden_'))
    checkpoint = write_tiny_checkpoint(tmp / 'tiny')
    real_load_model = bioemu2.load_model

    def tiny_loader(*args, **kwargs):                       # what the old code loads as bioemu2-preview
        return load_tiny_model(checkpoint)
    bioemu2.load_model = tiny_loader
    model = load_tiny_model(checkpoint)
    out['model.parameter_checksum'] = np.array(extractor.parameter_checksum(model))
    out['model.n_parameters'] = np.array(sum(p.numel() for p in model.parameters()))
    single_width = int(model.score_model.structure_module.single_conditioner.single_embed.out_features)
    rng = np.random.default_rng(FRAME_SEED)

    # ------------------------------------------------------------------ the peptide (old chignolin code)
    from omegaconf import OmegaConf
    aimmd_flow.TRUNK_SINGLE_DIM = int(OmegaConf.load(TINY_CONFIG).score_model.token_s)
    extractor.SYSTEM_DIR = chignolin_dir
    extractor.REFERENCE_GRO = chignolin_dir / 'conf_A_box.gro'
    n_token_features = 10 * single_width
    joint.JointHead.__init__.__defaults__ = (HEAD_HIDDEN, None, None, 0.0, 1.0, n_token_features)
    atoms, xyz, _ = read_gro(PEPTIDE / 'md.gro')
    assert len(atoms) == extractor.N_MD_ATOMS
    frames = random_frames(xyz, 4, rng)
    out['peptide.x_md'] = frames
    out['peptide.sigmas'] = np.array(FRAME_SIGMAS, np.float32)
    ds = _dataset(xyz, rng)
    for name in ('sp_rows', 'sp_r_first', 'sp_r_last', 'fr_rows', 'fr_state'):
        out[f'peptide.dataset.{name}'] = getattr(ds, name)

    torch.manual_seed(HEAD_SEED)
    net = aimmd_flow.BioEmu2FlowNetwork(shared=False, hidden=HEAD_HIDDEN, cfg_weight=1.0, gen_batch=GEN_BATCH,
                                        value_batch=2)
    eng = net.engine()
    ex, cond = eng['ex'], eng['cond']
    assert aimmd_flow.TRUNK_SINGLE_DIM == ex.trunk.single.shape[-1]
    out['peptide.md_index'] = ex.md_index
    out['peptide.reference'] = _np(ex.reference)
    _trunk(out, 'peptide.trunk', ex.trunk)
    with torch.no_grad():
        out['peptide.canonical'] = _np(ex.canonical(torch.as_tensor(frames)))
        out['peptide.features_ref'] = _np(net.features(frames))
        out['peptide.features_sigmas'] = _np(net.features(frames, torch.tensor(FRAME_SIGMAS)))
        tok = net.features(ds.fr_rows)
    mean, std, ls_mean, ls_std = joint.fit_scaler(_np(tok), np.full(len(tok), SIGMA_REF))
    scaler = dict(mean=mean, std=std, log_sigma_mean=ls_mean, log_sigma_std=ls_std)
    for key, value in scaler.items():
        out[f'peptide.network.scaler.{key}'] = np.asarray(value)
    torch.manual_seed(HEAD_SEED)
    net = aimmd_flow.BioEmu2FlowNetwork(shared=False, hidden=HEAD_HIDDEN, scaler=scaler, cfg_weight=1.0,
                                        gen_batch=GEN_BATCH, value_batch=2)
    eng = net.engine()
    ex, cond = eng['ex'], eng['cond']
    _state(out, 'peptide.network.head_ema', net.head_ema)
    with torch.no_grad():
        out['peptide.network.values'] = _np(net(torch.as_tensor(frames).reshape(len(frames), -1)))
    # the denoiser and the adapters, with seeded weights (net.adapter is the denoiser's single adapter)
    _seed_adapter(net.adapter, ADAPTER_SEED)
    _state(out, 'peptide.denoiser.state.single_adapter', net.adapter)
    _, b_inputs = prepare_sampling_inputs(ex.system, device='cpu')
    reference_be2 = ex.canonical(torch.as_tensor(xyz[None], dtype=torch.float32))[0]
    _record_denoiser(out, 'peptide.denoiser', cond, ex.model, ex.trunk, b_inputs, _np(reference_be2), ex.n_pad)
    _record_adapters(out, 'peptide.adapter', cond, ex.trunk, 23, pair=False)
    # a learned null (an option): a second denoiser of the same kind
    cond_null = finetune.ConditionedDenoiser(ex.model, kind='trunk_single', trunk=ex.trunk, batch_inputs=b_inputs,
                                             learned_null=True)
    _seed_adapter(cond_null.single_adapter, ADAPTER_SEED + 1)
    _state(out, 'peptide.denoiser_null.state.single_adapter', cond_null.single_adapter)
    _record_denoiser(out, 'peptide.denoiser_null', cond_null, ex.model, ex.trunk, b_inputs, _np(reference_be2),
                     ex.n_pad)
    _record_adapters(out, 'peptide.adapter_null', cond_null, ex.trunk, 29, pair=False)
    # sampling as BioEmu2FlowNetwork.generate drives it (without the pdb2gmx hydrogens)
    lam_all = torch.tensor(SAMPLER_LAMBDAS).clamp(-joint.LAMBDA_CLIP, joint.LAMBDA_CLIP)
    for weight in (1.0, 2.0):
        net.cfg_weight = weight
        coords = []
        for s in range(0, len(lam_all), net.gen_batch):
            lam_k = lam_all[s:s + net.gen_batch]
            c, _ = guided_sampler.sample(ex.model, ex.system, ex.trunk, n=len(lam_k), batch_size=len(lam_k),
                                         seed=SAMPLER_SEED + s, num_steps=SAMPLER_STEPS,
                                         denoise=net._denoise_hook(lam_k))
            coords.append(np.asarray(c, np.float32)[:, :extractor.N_HEAVY])
        out[f'peptide.sample.cfg{weight:g}'] = np.concatenate(coords)
    net.cfg_weight = 1.0
    # one training round
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        losses, info = aimmd_flow_fit.sequential_round(net, ds, 0, **ROUND_SETTINGS, **ROUND_SETTINGS_PEPTIDE)
    head_log, steps, _, _ = _round_log(log.getvalue())
    assert len(steps) == ROUND_SETTINGS['adapter_steps'], steps
    out['peptide.round.head_losses'] = np.array([[rec.get('L_sp', np.nan), rec.get('L_anchor', np.nan)]
                                                 for rec in losses])
    out['peptide.round.adapter_losses'] = np.array([rec['L_den'] for rec in steps])
    out['peptide.round.label_counts'] = np.array(info['label_counts'])
    out['peptide.round.info'] = np.array(json.dumps(info, default=str))
    _state(out, 'peptide.round.head_ema', net.head_ema)
    _state(out, 'peptide.round.adapter', net.adapter)

    # ------------------------------------------------------------------ the protein-ligand system (old holo code)
    atoms, xyz_raw, box = read_gro(PROTEIN_LIGAND / 'md.gro')
    masses, bonds = read_top_bonds(PROTEIN_LIGAND / 'md.top')
    assert len(masses) == len(atoms)
    xyz = whole_orthorhombic(xyz_raw, bonds, box)
    system = System.from_yaml(PROTEIN_LIGAND / 'bioemu2_system.yaml')
    topology, batch = prepare_sampling_inputs(system, device='cpu')
    n_be2 = topology.n_atoms
    # the old trypsin map, restricted to these atoms
    old = json.loads((be_root / 'trypsin' / 'system' / 'trypsin_atom_map.json').read_text())
    md_atoms_full = old['md_atoms']
    first_full = next(i for i, a in enumerate(md_atoms_full) if a[0] == LIGAND_FRAGMENT_FIRST_RESIDUE)
    n_protein_toy = sum(1 for a in atoms if a[1] != 'BEN')
    ligand_first_full = next(i for i, a in enumerate(md_atoms_full) if a[1] == 'BEN')
    full_index = {(a[0], a[1] == 'L', a[3]): k for k, a in enumerate(
        [(row[1], row[0], row[2], row[3]) for row in old['bioemu_atoms']])}
    b2m_full = old['bioemu_to_md']
    b2m = []
    for atom in topology.atoms:
        is_ligand = atom.residue.chain.chain_id == 'L'
        res_seq = 1 if is_ligand else atom.residue.resSeq + LIGAND_FRAGMENT_FIRST_RESIDUE - 1
        g = b2m_full[full_index[(res_seq, is_ligand, atom.name)]]
        b2m.append(n_protein_toy + g - ligand_first_full if is_ligand else g - first_full)
    b2m = np.array(b2m, np.int64)
    for k, atom in enumerate(topology.atoms):                 # same atom names (protein) under the old map
        if atom.residue.chain.chain_id == 'A':
            assert atoms[b2m[k]][2] == atom.name, (atom, atoms[b2m[k]])
    out['ligand.bioemu2_to_md'] = b2m
    out['ligand.x_md_reference'] = xyz.astype(np.float32)
    reference_be2 = xyz[b2m]
    # constants as trypsin_features.build_constants makes them (pocket within 6 A of the ligand in the reference)
    tok = _np(batch.token_idx_at_atom[0, :n_be2]).astype(np.int64)
    is_ligand = np.array([a.residue.chain.chain_id == 'L' for a in topology.atoms])
    names = np.array([a.name for a in topology.atoms])
    core = np.flatnonzero((names == 'CA') & ~is_ligand)
    lig = np.flatnonzero(is_ligand)
    d = np.linalg.norm(reference_be2[~is_ligand][:, None] - reference_be2[lig][None], axis=-1).min(1)
    constants = dict(token_idx_at_atom=tok, ref_core=reference_be2[core].astype(np.float32), core_be2=core,
                     ligand_tokens=np.unique(tok[lig]), pocket_tokens=np.unique(tok[~is_ligand][d < 6.0]),
                     protein_tokens=np.unique(tok[~is_ligand]), ligand_atoms=lig)
    for key, value in constants.items():
        out[f'ligand.constants.{key}'] = value
    system_dir = tmp / 'ligand_system'
    system_dir.mkdir()
    (system_dir / 'system.yaml').write_text((PROTEIN_LIGAND / 'bioemu2_system.yaml').read_text())
    seq = system.entities[0].sequence
    trunk = model.embed(system, msa=MSA({'A': f'>query\n{seq}\n'}), mode='equilibrium')
    trunk.save(system_dir / 'embeddings.pt')
    np.savez(tmp / 'ligand_constants.npz', **constants)
    _trunk(out, 'ligand.trunk', trunk)
    holo.SYSTEM_YAML, holo.EMBEDDINGS, holo.N_ATOMS = 'system.yaml', 'embeddings.pt', n_be2
    aimmd_flow_holo.TRUNK_SINGLE_DIM = int(trunk.single.shape[-1])
    aimmd_flow_holo.TRUNK_PAIR_DIM = int(trunk.pair.shape[-1])
    aimmd_flow_holo.N_POOLED = len(holo.POOLS) * single_width

    class OldSystem:                                          # the members HoloFlowNetwork reads
        n_atoms = len(atoms)
        ligand_atoms_be2 = lig

        def __init__(self):
            self.n_be2, self.be2_index = n_be2, b2m

        @staticmethod
        def make_extractor(device='cuda'):
            return holo.HoloFeatures(holo.SIGMA_REF, device=device, system_dir=system_dir,
                                     constants=tmp / 'ligand_constants.npz')

    frames_md = random_frames(xyz, 4, rng)
    out['ligand.x_md'] = frames_md
    ds = _dataset(xyz, rng)
    for name in ('sp_rows', 'sp_r_first', 'sp_r_last', 'fr_rows', 'fr_state'):
        out[f'ligand.dataset.{name}'] = getattr(ds, name)
    torch.manual_seed(HEAD_SEED + 1)
    net = aimmd_flow_holo.HoloFlowNetwork(OldSystem(), kind='trunk_single_pair', hidden=HEAD_HIDDEN, cfg_weight=1.0,
                                          gen_batch=GEN_BATCH, value_batch=2)
    eng = net.engine()
    ex, cond = eng['ex'], eng['cond']
    with torch.no_grad():
        frames_be2 = net.to_be2(frames_md)
        out['ligand.canonical'] = _np(ex.canonical_be2(frames_be2))
        out['ligand.features_ref'] = _np(net.features(frames_md))
        out['ligand.features_sigmas'] = _np(net.features(frames_md, torch.tensor(FRAME_SIGMAS)))
        feats = net.features(ds.fr_rows)
    mean, std, ls_mean, ls_std = joint.fit_scaler(_np(feats), np.full(len(feats), SIGMA_REF))
    scaler = dict(mean=mean, std=std, log_sigma_mean=ls_mean, log_sigma_std=ls_std)
    for key, value in scaler.items():
        out[f'ligand.network.scaler.{key}'] = np.asarray(value)
    torch.manual_seed(HEAD_SEED)
    net = aimmd_flow_holo.HoloFlowNetwork(OldSystem(), kind='trunk_single_pair', hidden=HEAD_HIDDEN,
                                          scaler=scaler, cfg_weight=1.0, gen_batch=GEN_BATCH, value_batch=2)
    eng = net.engine()
    ex, cond = eng['ex'], eng['cond']
    _state(out, 'ligand.network.head_ema', net.head_ema)
    with torch.no_grad():
        out['ligand.network.values'] = _np(net(torch.as_tensor(frames_md).reshape(len(frames_md), -1)))
    _seed_adapter(net.adapter, ADAPTER_SEED + 2)
    _seed_adapter(net.pair_adapter, ADAPTER_SEED + 3)
    _state(out, 'ligand.denoiser.state.single_adapter', net.adapter)
    _state(out, 'ligand.denoiser.state.pair_adapter', net.pair_adapter)
    with torch.no_grad():
        reference_canonical = _np(ex.canonical_be2(torch.as_tensor(reference_be2[None], dtype=torch.float32))[0])
    _record_denoiser(out, 'ligand.denoiser', cond, ex.model, ex.trunk, ex.batch, reference_canonical, ex.n_pad)
    _record_adapters(out, 'ligand.adapter', cond, ex.trunk, 31, pair=False)
    # sampling through HoloFlowNetwork.generate_be2 (old code), with SAMPLER_STEPS steps
    real_sample = guided_sampler.sample

    def short_sample(*args, **kwargs):
        kwargs.setdefault('num_steps', SAMPLER_STEPS)
        return real_sample(*args, **kwargs)
    guided_sampler.sample = short_sample
    try:
        for weight in (1.0, 2.0):
            net.cfg_weight = weight
            out[f'ligand.sample.cfg{weight:g}'] = net.generate_be2(torch.tensor(SAMPLER_LAMBDAS),
                                                                   seed=SAMPLER_SEED).astype(np.float32)
    finally:
        guided_sampler.sample = real_sample
    net.cfg_weight = 1.0
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        losses, info = aimmd_flow_holo_fit.holo_sequential_round(net, ds, 0, **ROUND_SETTINGS,
                                                                 **ROUND_SETTINGS_LIGAND)
    head_log, steps, probe, _ = _round_log(log.getvalue())
    assert len(steps) == ROUND_SETTINGS['adapter_steps'], steps
    out['ligand.round.head_losses'] = np.array([[rec.get('L_sp', np.nan), rec.get('L_anchor', np.nan)]
                                                for rec in losses])
    out['ligand.round.adapter_losses'] = np.array([rec['L_den'] for rec in steps])
    out['ligand.round.label_counts'] = np.array(info['label_counts'])
    out['ligand.round.label_probe'] = np.array(json.dumps(info['label_probe']))
    out['ligand.round.info'] = np.array(json.dumps(info, default=str))
    _state(out, 'ligand.round.head_ema', net.head_ema)
    _state(out, 'ligand.round.adapter', net.adapter)
    _state(out, 'ligand.round.pair_adapter', net.pair_adapter)

    bioemu2.load_model = real_load_model
    from importlib.metadata import version
    meta = dict(torch=torch.__version__, bioemu2=version('bioemu2'),
                bioemu2_file=bioemu2.__file__, numpy=np.__version__, python=sys.version.split()[0],
                old_files={name: hashlib.md5((be_root / name).read_bytes()).hexdigest() for name in OLD_FILES},
                chignolin_system=str(chignolin_dir),
                note='recorded by tests/_cgaimmd_bioemu2_golden.py from the old BioEmu2 project code')
    out['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **out)
    print(json.dumps(meta, indent=1))
    print(f'{len(out)} arrays, {Path(out_path).stat().st_size} bytes -> {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != 'record':
        sys.exit(__doc__)
    record(sys.argv[2], sys.argv[3], sys.argv[4])
