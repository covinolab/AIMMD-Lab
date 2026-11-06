"""The BioEmu2 adaptor with the real bioemu2-preview weights on a GPU (the weights lane).

Run by ``devtools/run_weights_tests.sh`` (``pytest --runweights -m "bioemu2_weights or gpu"``)
before every campaign, on the machine and in the environment of the campaign.
Skipped without ``--runweights``, without CUDA, and when the checkpoint is
not in the Hugging Face cache (nothing is downloaded).

What it checks:

- the checkpoint: its widths (token and atom features at the readout, trunk
  single and pair embeddings), parameter count and checksum, the Hugging Face
  snapshot, and what a network records about it (``provenance``);
- BioEmu2 with identity adapters is plain BioEmu2: one denoiser pass on
  chignolin and on a trypsin-benzamidine system, and the generator
  ``'bioemu2_edm'`` against BioEmu2's own sampler at the same seed;
- the networks of the archived chignolin and trypsin-benzamidine campaigns,
  read by ``load_archived_network``, against the goldens recorded with the
  old project code and the same weights (``tests/_cgaimmd_bioemu2_weights_golden.py``):
  committor values on 100 frames of each campaign, the head's features,
  denoiser passes with the trained adapters, and seeded 200-step samples.
  These need the archive of the campaigns (network files, the trypsin MD
  files and descriptors) and skip without it.

Tolerances are absolute, in the units of each quantity, and each is next to
the error measured when it was set (Quadro RTX 5000, torch 2.11, cuEquivariance
0.11.1; the goldens were recorded on the same card with torch 2.13). On another
GPU the goldens are compared with looser tolerances (see ``_tolerance``).
"""

import json

import numpy as np
import pytest

from ._cgaimmd_bioemu2_weights_golden import (
    ARCHIVE, CHECKPOINT, CHIGNOLIN_CAMPAIGN, DENOISER_LAMBDAS, DENOISER_SIGMAS, DENOISER_UNCOND, GOLDEN,
    HF_SNAPSHOT, N_FRAMES, N_PARAMETERS, NETWORKS, OLD_TRYPSIN_TRUNK, PARAMETER_CHECKSUM, PEPTIDE,
    SAMPLER_LAMBDAS, SAMPLER_SEED, SAMPLER_STEPS, TRYPSIN_CAMPAIGN, TRYPSIN_N_ATOMS, TRYPSIN_SAMPLER_LAMBDAS,
    TRYPSIN_SYSTEM_YAML, WIDTHS, campaign_file, denoiser_input, trypsin_frames)
from ._cgaimmd_bioemu2_golden import PROTEIN_LIGAND

pytestmark = [pytest.mark.bioemu2_weights, pytest.mark.gpu]

ZERO_ADAPTER_ATOL = 1e-4
"""Ångström: identity adapters against plain BioEmu2, one denoiser pass (DESIGN 6.1; production measured 4.8e-5;
here 8.6e-6 chignolin, 2.0e-5 trypsin fragment)."""
PLAIN_SAMPLE_ATOL = 1e-3
"""Ångström: the generator with identity adapters against BioEmu2's own sampler, same seed, 200 steps (2.1e-4)."""

# Against the goldens: (on the card the goldens were recorded on, on any other card). Another GPU computes in
# float32 with other kernels; the head is that sensitive (a change of 1e-7 Å in the canonical pose moves the
# chignolin logits by 1.7e-4) that the goldens are reproduced exactly only on the recording card. The second values
# are not measured yet: the first weights-lane run on another GPU should report its errors.
VALUE_ATOL = (1e-5, 1e-2)
"""Logit committor: the archived networks' values against the old code's (DESIGN 6.1: 1e-5). Measured 0 (trypsin;
chignolin through the old chignolin code's Kabsch formula)."""
CHIGNOLIN_POSE_VALUE_ATOL = (1e-3, 1e-2)
"""Logit committor: the archived chignolin networks with the merged canonical pose (measured 1.7e-4; see
``_old_chignolin_pose_values``)."""
FEATURE_RTOL = (1e-5, 1e-3)
"""Relative to the largest feature: the head's input features against the old code's (measured 0 trypsin, 7.8e-6
chignolin)."""
DENOISER_ATOL = (1e-4, 1e-3)
"""Ångström: denoised coordinates with the archived adapters against the old code's (measured 0)."""
SAMPLE_ATOL = (2e-3, 1e-2)
"""Ångström: seeded 200-step samples against the old sampler's (measured: max 7.8e-4 for one atom of one trypsin
sample, RMSD at most 5e-5; the old sampler orders its float32 Euler update differently)."""
AS_BUILT_VALUE_ATOL = (1e-3, 1e-2)
"""Logit committor: the archived chignolin networks on a folder built from its files, whose reference MDAnalysis
reads in float32 (5e-6 Å from the old code's float64 read of the same .gro): measured 2.2e-4."""


def _tolerance(golden, pair):
    """The first value of `pair` on the card the goldens were recorded on, else the second."""
    import torch
    recorded = json.loads(str(golden['meta']))['device']
    return pair[0] if torch.cuda.get_device_name(0) == recorded else pair[1]


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='session')
def real_model():
    pytest.importorskip('bioemu2')
    import torch
    from bioemu2.checkpoints import get_checkpoint_files
    try:
        get_checkpoint_files(CHECKPOINT)
    except Exception as exc:                                  # noqa: BLE001 - any cache miss
        pytest.skip(f'the {CHECKPOINT} checkpoint is not in the Hugging Face cache ({type(exc).__name__})')
    from aimmd.cgaimmd.bioemu2 import _api
    torch.cuda.reset_peak_memory_stats()
    return _api.load_model(CHECKPOINT, device='cuda')


@pytest.fixture(scope='session')
def golden_weights():
    return np.load(GOLDEN)


def _build(folder, data, model, **kwargs):
    from aimmd.cgaimmd.bioemu2 import build_system
    return build_system(data / 'bioemu2_system.yaml', data / 'md.gro', data / 'md.top', folder, model=model,
                        **kwargs)


@pytest.fixture(scope='session')
def chignolin_system(real_model, tmp_path_factory):
    """CLN025 built from its files (``md.gro`` holds the coordinates of the old code's ``conf_A_box.gro``)."""
    return _build(tmp_path_factory.mktemp('weights_chignolin') / 'system', PEPTIDE, real_model)


@pytest.fixture(scope='session')
def chignolin_exact_system(real_model, tmp_path_factory):
    """CLN025 with the old code's reference exactly as it read it (the .gro text in float64; every atom is core).

    Built from the files (:func:`chignolin_system`), MDAnalysis reads the
    reference in float32, 5e-6 Å from the old code's: enough to move the
    archived head's logits by 2e-4 (test below).
    """
    from ._cgaimmd_bioemu2_golden import read_gro
    _, reference, _ = read_gro(PEPTIDE / 'md.gro')
    return _build(tmp_path_factory.mktemp('weights_chignolin_exact') / 'system', PEPTIDE, real_model,
                  reference_structure=reference)


@pytest.fixture(scope='session')
def trypsin_fragment_system(real_model, tmp_path_factory):
    """Trypsin residues 187-206 with benzamidinium (the tiny tests' system), on the real model."""
    return _build(tmp_path_factory.mktemp('weights_trypsin_fragment') / 'system', PROTEIN_LIGAND, real_model,
                  core_residues=tuple(range(1, 21)))


def _needs_archive(*extra):
    for path in (ARCHIVE / CHIGNOLIN_CAMPAIGN, ARCHIVE / TRYPSIN_CAMPAIGN) + extra:
        if not path.exists():
            pytest.skip(f'{path} (the archive of the BioEmu2 campaigns or the old project) is not here')


@pytest.fixture(scope='session')
def trypsin_system(real_model, golden_weights, tmp_path_factory):
    """The archived trypsin campaign's system with its archived choices.

    The 3,239 generator atoms of its ``run.gro``; the canonical pose from the
    3PTB coordinates of the BioEmu2 atoms (float32, as the old code kept them)
    and the 180 core residues; and the trunk embeddings the campaign used,
    copied from the old project. They were saved by a sampling run of 27 Oct
    2026 with bf16 autocast and from an alignment other than the query-only
    one (the MSA profile features differ in every token); a query-only trunk
    computed here differs from them by up to 14 in the single embedding.
    """
    _needs_archive(OLD_TRYPSIN_TRUNK)
    from aimmd.cgaimmd.bioemu2 import build_system, read_md_atoms
    campaign = ARCHIVE / TRYPSIN_CAMPAIGN
    md = read_md_atoms(campaign / 'run.gro', campaign / 'topol.top', n_atoms=TRYPSIN_N_ATOMS)
    reference = np.asarray(md.positions, np.float64)
    reference[golden_weights['trypsin.bioemu2_to_md']] = golden_weights['trypsin.reference_3ptb']
    return build_system(TRYPSIN_SYSTEM_YAML, campaign / 'run.gro', campaign / 'topol.top',
                        tmp_path_factory.mktemp('weights_trypsin') / 'system', n_atoms=TRYPSIN_N_ATOMS,
                        reference_structure=reference, trunk_embeddings=OLD_TRYPSIN_TRUNK,
                        core_residues=[int(r) for r in golden_weights['trypsin.core_residues']], model=real_model)


def _systems(request, which):
    return request.getfixturevalue({'chignolin': 'chignolin_system', 'chignolin_exact': 'chignolin_exact_system',
                                    'trypsin': 'trypsin_system',
                                    'trypsin_fragment': 'trypsin_fragment_system'}[which])


def _plain_denoise(model, batch, trunk, x, sigmas):
    """Plain BioEmu2 on k structures, as its sampler calls it (static context with multiplicity k)."""
    import torch
    with torch.no_grad():
        context = model.compute_static_context(batch, trunk, len(x))
        return model.denoise(context, trunk, x, sigmas).denoised_coords


def _reference_input(network):
    """Three noisy copies of the system's reference pose (centred, BioEmu2 atoms), padded."""
    import torch
    engine = network.engine()
    reference = network.system.reference
    x = denoiser_input((reference - reference.mean(0)).astype(np.float32), engine.denoiser.n_pad)
    return torch.as_tensor(x, device=network.device), torch.tensor(DENOISER_SIGMAS, device=network.device)


def _report(name, value):
    print(f'[weights] {name}: {value:.3e}')


# ---------------------------------------------------------------------------------------------- the checkpoint
def test_the_checkpoint_widths_checksum_and_provenance(real_model, chignolin_system, golden_weights):
    import torch
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, _api
    model = real_model
    assert _api.token_feature_width(model) == WIDTHS['token']
    trunk = chignolin_system.data['trunk']
    assert (trunk['token_feature_width'], trunk['single_width'], trunk['pair_width']) == (
        WIDTHS['token'], WIDTHS['single'], WIDTHS['pair'])
    network = BioEmu2Network(chignolin_system, model=model).to('cuda')
    features = network.engine().features
    with torch.no_grad():
        features.denoise(features.canonical(torch.as_tensor(chignolin_system.reference[None], dtype=torch.float32,
                                                            device='cuda')))
    assert features._readout.a.shape[-1] == WIDTHS['token']
    assert features._readout.q.shape[-1] == WIDTHS['atom']
    assert list(golden_weights['chignolin.widths']) == [WIDTHS[k] for k in ('token', 'atom', 'single', 'pair')]
    n_parameters = sum(p.numel() for p in model.parameters())
    assert n_parameters == N_PARAMETERS == int(golden_weights['model.n_parameters'])
    checksum = _api.parameter_checksum(model)
    assert checksum == PARAMETER_CHECKSUM == str(golden_weights['model.parameter_checksum'])
    assert trunk['parameter_checksum'] == checksum
    assert _api.checkpoint_snapshot(CHECKPOINT) == HF_SNAPSHOT
    provenance = network.provenance()
    json.dumps(provenance)
    assert provenance['parameter_checksum'] == checksum
    assert provenance['hf_snapshot'] == HF_SNAPSHOT
    assert provenance['checkpoint'] == CHECKPOINT and provenance['checkpoint_path'] is None
    assert provenance['n_parameters'] == N_PARAMETERS
    assert provenance['bioemu2_version'] == _api.SUPPORTED_VERSION
    network.engine().close()


# ---------------------------------------------------------------------------------------------- identity adapters
@pytest.mark.parametrize('which', ['chignolin', 'trypsin_fragment'])
def test_identity_adapters_denoise_as_plain_bioemu2(which, request, real_model):
    import torch
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network
    network = BioEmu2Network(_systems(request, which), model=real_model).to('cuda')
    denoiser = network.engine().denoiser
    x, sigmas = _reference_input(network)
    lam = torch.tensor(DENOISER_LAMBDAS, device='cuda')
    with torch.no_grad():
        conditioned = denoiser.denoise(x, sigmas, lam)
        unlabelled = denoiser.denoise(x, sigmas, None)
    plain = _plain_denoise(real_model, denoiser.batch, denoiser.trunk, x, sigmas)
    n = network.system.n_bioemu2_atoms
    for name, out in (('labelled', conditioned), ('no label', unlabelled)):
        error = float((out - plain)[:, :n].abs().max())
        _report(f'{which} identity adapters ({name}) vs plain BioEmu2, max |dx| A', error)
        assert error <= ZERO_ADAPTER_ATOL
    network.engine().close()


# ---------------------------------------------------------------------------------------------- archived networks
def _archived(which, system_fixture, request, real_model, network_name):
    _needs_archive()
    from aimmd.cgaimmd.bioemu2 import load_archived_network
    campaign = CHIGNOLIN_CAMPAIGN if which == 'chignolin' else TRYPSIN_CAMPAIGN
    path = campaign_file(campaign, NETWORKS[network_name])
    network = load_archived_network(path, request.getfixturevalue(system_fixture), model=real_model)
    return network.to('cuda').eval()


def _frames(which, golden):
    if which == 'chignolin':
        return golden['chignolin.frames']
    _needs_archive()
    return trypsin_frames(golden['trypsin.frame_sources'])


ARCHIVED = [(which, name) for which in ('chignolin', 'trypsin') for name in NETWORKS]
_EXACT = {'chignolin': 'chignolin_exact_system', 'trypsin': 'trypsin_system'}


def test_the_trypsin_system_has_the_archived_map_pose_and_pocket(trypsin_system, golden_weights):
    """The folder built with the archived choices maps, aligns and pools as the old trypsin code did."""
    np.testing.assert_array_equal(trypsin_system.bioemu2_to_md, golden_weights['trypsin.bioemu2_to_md'])
    np.testing.assert_array_equal(trypsin_system.core, golden_weights['trypsin.core_bioemu2'])
    np.testing.assert_array_equal(trypsin_system.token_pools['pocket'], golden_weights['trypsin.pocket_tokens'])
    np.testing.assert_array_equal(trypsin_system.reference, golden_weights['trypsin.reference_3ptb'])
    assert trypsin_system.data['trunk']['imported_from'] == str(OLD_TRYPSIN_TRUNK)


def _old_chignolin_pose_values(network, frames):
    """The archived chignolin network's values with the old chignolin code's canonical pose.

    The old chignolin code (``extractor.kabsch_onto``) fitted onto its
    reference as it was stored (centred in float64, then float32); the old
    trypsin code (``holo.kabsch_rotation``), and with it the merged
    :func:`~aimmd.cgaimmd.bioemu2.features.canonical_pose`, centres the stored
    reference once more in float32. The two poses differ by float32 rounding
    (about 1e-7 Å); everything after the pose is the network's own.
    """
    import torch
    features = network.engine().features
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            x = features.to_bioemu2(torch.as_tensor(frames[i:i + 16], device=network.device))[:, :features.n_atoms]
            xc = x - x.mean(dim=1, keepdim=True)
            h = xc.double().transpose(1, 2) @ features.reference_core.double()
            u, _, vh = torch.linalg.svd(h)
            diag = torch.ones(len(x), 3, dtype=h.dtype, device=h.device)
            diag[:, 2] = torch.sign(torch.linalg.det(u @ vh))
            canonical = (xc.double() @ (u @ torch.diag_embed(diag) @ vh)).to(torch.float32)
            features.denoise(canonical)
            tokens = features._readout.a[:, :features.n_tokens].reshape(len(x), -1)
            sigma = torch.full((len(x),), network.sigma_ref, device=network.device)
            out.append(network.head_ema(tokens, sigma).reshape(-1))
    return torch.cat(out).cpu().numpy()


@pytest.mark.parametrize('which,network_name', ARCHIVED)
def test_archived_networks_reproduce_the_old_values(which, network_name, request, real_model, golden_weights):
    import torch
    network = _archived(which, _EXACT[which], request, real_model, network_name)
    assert network.readout == ('flat' if which == 'chignolin' else 'pooled')
    assert (network.pair_adapter is not None) == (which == 'trypsin')
    frames = _frames(which, golden_weights)
    assert len(frames) == N_FRAMES
    values = _values(network, frames)
    with torch.no_grad():
        features = network.features(torch.as_tensor(frames[:2])).cpu().numpy()
    want = golden_weights[f'{which}.{network_name}.values']
    noise = float(np.abs(golden_weights[f'{which}.{network_name}.values_repeat'] - want).max())
    error = float(np.abs(values - want).max())
    _report(f'{which} {network_name} values vs old code, max |d logit| (old code against itself {noise:.1e}; '
            f'logits {want.min():.2f} to {want.max():.2f})', error)
    feature_want = golden_weights[f'{which}.{network_name}.features']
    feature_error = float(np.abs(features - feature_want).max() / np.abs(feature_want).max())
    _report(f'{which} {network_name} features vs old code, max relative', feature_error)
    assert feature_error <= _tolerance(golden_weights, FEATURE_RTOL)
    if which == 'chignolin':
        assert error <= _tolerance(golden_weights, CHIGNOLIN_POSE_VALUE_ATOL)
        error = float(np.abs(_old_chignolin_pose_values(network, frames) - want).max())
        _report(f'chignolin {network_name} values with the old chignolin pose vs old code, max |d logit|', error)
    assert error <= _tolerance(golden_weights, VALUE_ATOL)
    network.engine().close()


def _values(network, frames):
    import torch
    with torch.no_grad():
        return torch.cat([network(torch.as_tensor(frames[i:i + 16]).reshape(len(frames[i:i + 16]), -1))
                          for i in range(0, len(frames), 16)]).reshape(-1).cpu().numpy()


@pytest.mark.parametrize('network_name', list(NETWORKS))
def test_archived_chignolin_networks_on_a_folder_built_from_its_files(network_name, request, real_model,
                                                                        golden_weights):
    """What a user gets with build-system on the .gro: the same values up to the float32 read of the reference."""
    network = _archived('chignolin', 'chignolin_system', request, real_model, network_name)
    want = golden_weights[f'chignolin.{network_name}.values']
    error = float(np.abs(_values(network, golden_weights['chignolin.frames']) - want).max())
    _report(f'chignolin {network_name} values, folder built from the files, vs old code, max |d logit|', error)
    assert error <= _tolerance(golden_weights, AS_BUILT_VALUE_ATOL)
    network.engine().close()


@pytest.mark.parametrize('which,network_name', ARCHIVED)
def test_archived_adapters_denoise_as_the_old_code(which, network_name, request, real_model, golden_weights):
    import torch
    network = _archived(which, _EXACT[which], request, real_model, network_name)
    denoiser = network.engine().denoiser
    x = torch.as_tensor(golden_weights[f'{which}.denoiser.x'], device='cuda')
    sigmas = torch.tensor(DENOISER_SIGMAS, device='cuda')
    with torch.no_grad():
        labelled = denoiser.denoise(x, sigmas, torch.tensor(DENOISER_LAMBDAS, device='cuda'),
                                    torch.tensor(DENOISER_UNCOND, device='cuda'))
        unlabelled = denoiser.denoise(x, sigmas, None)
    n = network.system.n_bioemu2_atoms
    for name, out in (('labelled', labelled), ('no_label', unlabelled)):
        want = golden_weights[f'{which}.{network_name}.denoiser.{name}'][:, :n]
        error = float(np.abs(out[:, :n].cpu().numpy() - want).max())
        _report(f'{which} {network_name} denoiser ({name}) vs old code, max |dx| A', error)
        assert error <= _tolerance(golden_weights, DENOISER_ATOL)
    if which == 'chignolin' and network_name == 'pretrained':
        plain = _plain_denoise(real_model, denoiser.batch, denoiser.trunk, x, sigmas)[:, :n].cpu().numpy()
        error = float(np.abs(plain - golden_weights['chignolin.plain'][:, :n]).max())
        _report('chignolin plain BioEmu2 vs the old code\'s plain pass, max |dx| A', error)
        assert error <= _tolerance(golden_weights, DENOISER_ATOL)
    network.engine().close()


# ---------------------------------------------------------------------------------------------- samples
@pytest.mark.parametrize('which,network_name', ARCHIVED)
def test_archived_networks_sample_as_the_old_sampler(which, network_name, request, real_model, golden_weights):
    import torch
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    network = _archived(which, _EXACT[which], request, real_model, network_name)
    lam = SAMPLER_LAMBDAS if which == 'chignolin' else TRYPSIN_SAMPLER_LAMBDAS
    generator = BioEmu2Generator(n_steps=SAMPLER_STEPS, chunk=len(lam))
    samples = generator.sample_bioemu2(network, torch.tensor(lam), seed=SAMPLER_SEED).cpu().numpy()
    want = golden_weights[f'{which}.{network_name}.sample']
    error = np.abs(samples - want).max(axis=(1, 2))
    rmsd = np.sqrt(((samples - want) ** 2).sum(-1).mean(-1))
    _report(f'{which} {network_name} seeded 200-step sample vs old sampler, max |dx| A per sample '
            f'{np.array2string(error, precision=2)}, RMSD {np.array2string(rmsd, precision=2)}', float(error.max()))
    assert error.max() <= _tolerance(golden_weights, SAMPLE_ATOL)
    network.engine().close()


@pytest.mark.parametrize('which', ['chignolin', 'trypsin_fragment'])
def test_identity_adapters_sample_as_plain_bioemu2(which, request, real_model):
    import torch
    from aimmd.cgaimmd.bioemu2 import BioEmu2Network, _api
    from aimmd.cgaimmd.bioemu2.sampling import BioEmu2Generator
    network = BioEmu2Network(_systems(request, which), model=real_model).to('cuda')
    denoiser = network.engine().denoiser
    lam = torch.tensor(SAMPLER_LAMBDAS)
    samples = BioEmu2Generator(n_steps=SAMPLER_STEPS, chunk=len(lam)).sample_bioemu2(network, lam,
                                                                                     seed=SAMPLER_SEED)
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.manual_seed(SAMPLER_SEED)
        plain = _api.edm_sample(real_model, denoiser.batch, denoiser.trunk, len(lam), SAMPLER_STEPS)
    n = network.system.n_bioemu2_atoms
    plain = plain.reshape(len(lam), -1, 3)[:, :n].float()
    plain = plain - plain.mean(dim=1, keepdim=True)
    error = (samples - plain).abs().amax(dim=(1, 2)).cpu().numpy()
    _report(f'{which} identity-adapter generator vs plain BioEmu2 sampler, same seed, max |dx| A per sample '
            f'{np.array2string(error, precision=2)}', float(error.max()))
    assert error.max() <= PLAIN_SAMPLE_ATOL
    network.engine().close()
