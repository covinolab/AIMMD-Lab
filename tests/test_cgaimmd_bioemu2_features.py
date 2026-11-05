"""The frozen BioEmu2 features of the committor head (aimmd.cgaimmd.bioemu2.features).

On the tiny model: shapes and widths read from the model, invariance to rigid
motions, and the goldens of the old feature extractors (``extractor.BioEmu2Features``
with ``joint.token_features``: flat; ``holo.HoloFeatures``: pooled), which the
new code reproduces to 1e-6.
"""

import numpy as np
import pytest

from ._bioemu2_fixtures import (tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_peptide_system, tiny_protein_ligand_system)
from ._cgaimmd_bioemu2_golden import FRAME_SIGMAS, GOLDEN, SIGMA_REF

pytestmark = pytest.mark.bioemu2

TRUNK_FIELDS = ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram')


@pytest.fixture(scope='module')
def golden():
    return np.load(GOLDEN)


def _features(system, model, **kwargs):
    from aimmd.cgaimmd.bioemu2 import BioEmu2Features
    return BioEmu2Features(system, model, **kwargs)


def _systems(request, which):
    return request.getfixturevalue('tiny_peptide_system' if which == 'peptide' else 'tiny_protein_ligand_system')


@pytest.mark.parametrize('which, readout', [('peptide', 'flat'), ('peptide', 'pooled'), ('ligand', 'pooled'),
                                            ('ligand', 'flat')])
def test_shapes_and_widths(which, readout, request, tiny_bioemu2_model):
    import torch
    system = _systems(request, which)
    features = _features(system, tiny_bioemu2_model, readout=readout)
    width = 32                                           # a_s of the tiny model, read from the model
    assert features.token_width == width and features.n_tokens == system.n_tokens
    pools = ('protein',) if which == 'peptide' else ('ligand', 'pocket', 'protein')
    assert features.pools == pools
    assert features.n_features == (system.n_tokens if readout == 'flat' else len(pools)) * width
    x = torch.as_tensor(system.template[None].repeat(3, 0), dtype=torch.float32)
    with torch.no_grad():
        out = features(x)
        tokens = features.token_features(features.to_bioemu2(x))
    assert out.shape == (3, features.n_features) and out.dtype == torch.float32
    assert tokens.shape == (3, system.n_tokens, width)
    assert torch.isfinite(out).all()
    # flat input and BioEmu2 padding rows are accepted
    padded = torch.cat([features.to_bioemu2(x), torch.zeros(3, 7, 3)], dim=1)
    with torch.no_grad():
        assert torch.equal(features(x.reshape(3, -1)), out)
        assert torch.equal(features.features(padded), out)
    features.close()


@pytest.mark.parametrize('which', ['peptide', 'ligand'])
def test_invariant_to_rigid_motions(which, request, tiny_bioemu2_model):
    import torch
    system = _systems(request, which)
    features = _features(system, tiny_bioemu2_model, readout='pooled')
    rng = np.random.default_rng(5)
    q = rng.normal(size=4)
    w, a, b, c = q / np.linalg.norm(q)
    rotation = np.array([[1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                         [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                         [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)]])
    x = system.template + rng.normal(scale=0.2, size=system.template.shape)
    moved = x @ rotation.T + np.array([12.0, -30.0, 7.5])
    with torch.no_grad():
        f = features(torch.as_tensor(np.stack([x, moved]), dtype=torch.float32))
    assert torch.allclose(f[0], f[1], rtol=1e-4, atol=1e-4)
    # without the alignment only translations are removed
    unaligned = _features(system, tiny_bioemu2_model, readout='pooled', align=False)
    with torch.no_grad():
        g = unaligned(torch.as_tensor(np.stack([x, x + 40.0, moved]), dtype=torch.float32))
    assert torch.allclose(g[0], g[1], rtol=1e-4, atol=1e-4) and not torch.allclose(g[0], g[2], atol=1e-3)
    features.close()
    unaligned.close()


@pytest.mark.parametrize('which, readout', [('peptide', 'flat'), ('ligand', 'pooled')])
def test_goldens_of_the_old_extractors(which, readout, request, tiny_bioemu2_model, golden):
    """The old chignolin (flat) and holo (pooled) features on the same frames, poses and noise levels."""
    import torch
    system = _systems(request, which)
    for field in TRUNK_FIELDS:                           # trunk of the build step = the old code's (torch 2.13)
        from aimmd.cgaimmd.bioemu2 import _api
        trunk = getattr(_api.load_trunk(system.trunk_path, 'cpu'), field).numpy()
        np.testing.assert_allclose(trunk, golden[f'{which}.trunk.{field}'], rtol=1e-5, atol=1e-5)
    features = _features(system, tiny_bioemu2_model, readout=readout)
    x = torch.as_tensor(golden[f'{which}.x_md'])
    with torch.no_grad():
        canonical = features.canonical(features.to_bioemu2(x)).numpy()
        at_reference = features(x).numpy()
        at_sigmas = features(x, torch.tensor(FRAME_SIGMAS)).numpy()
    np.testing.assert_allclose(canonical, golden[f'{which}.canonical'], rtol=0, atol=1e-5)
    np.testing.assert_allclose(at_reference, golden[f'{which}.features_ref'], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(at_sigmas, golden[f'{which}.features_sigmas'], rtol=1e-6, atol=1e-6)
    assert features.sigma_ref == SIGMA_REF
    identity = features.identity()
    assert identity['parameter_checksum'] == str(golden['model.parameter_checksum'])
    assert identity['readout'] == readout and identity['n_features'] == features.n_features
    features.close()


def test_the_readout_hook_goes_with_close(tiny_peptide_system, tiny_bioemu2_model):
    import torch
    features = _features(tiny_peptide_system, tiny_bioemu2_model)
    x = torch.as_tensor(tiny_peptide_system.template[None], dtype=torch.float32)
    with torch.no_grad():
        features(x)
    assert features._readout.a is not None
    features.close()
    hooks = tiny_bioemu2_model.score_model.structure_module.readout._forward_pre_hooks
    assert len(hooks) == 0


def test_what_the_features_refuse(tiny_peptide_system, tiny_protein_ligand_system, tiny_bioemu2_model):
    from aimmd.cgaimmd.bioemu2 import _api
    with pytest.raises(ValueError, match="readout must be one of \\('flat', 'pooled'\\)"):
        _features(tiny_peptide_system, tiny_bioemu2_model, readout='mean')
    with pytest.raises(ValueError, match='sigma_ref must be > 0'):
        _features(tiny_peptide_system, tiny_bioemu2_model, sigma_ref=0.0)
    trunk = _api.load_trunk(tiny_peptide_system.trunk_path, 'cpu')
    with pytest.raises(ValueError, match='the trunk embeddings have 10 tokens, the system 29'):
        _features(tiny_protein_ligand_system, tiny_bioemu2_model, trunk=trunk)
    tiny_bioemu2_model.train()
    with pytest.raises(ValueError, match='must be frozen and in eval mode'):
        _features(tiny_peptide_system, tiny_bioemu2_model)
