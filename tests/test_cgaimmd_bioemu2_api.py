"""The bioemu2 boundary (aimmd.cgaimmd.bioemu2._api): version policy, the contract of the internal touch
points, and the ligand atom-order check.

The contract tests pin what the adaptor reads inside bioemu2 (DESIGN section
5.2); the weekly canary job runs them against bioemu2's main branch.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import aimmd

from ._bioemu2_fixtures import (tiny_bioemu2_checkpoint, tiny_bioemu2_model,  # noqa: F401
                                tiny_protein_ligand_system)
from ._cgaimmd_bioemu2_golden import GOLDEN

pytestmark = pytest.mark.bioemu2

PERMUTED_SMILES = 'c1ccc(cc1)C(=[NH2+])N'
"""Benzamidinium written ring first: the same molecule as the test system's NC(=[NH2+])c1ccccc1."""


@pytest.fixture
def api():
    pytest.importorskip('bioemu2')
    from aimmd.cgaimmd.bioemu2 import _api
    return _api


def test_bioemu2_is_the_installed_package(api):
    """A subpackage named bioemu2 cannot shadow the real package (absolute imports); made explicit."""
    root = Path(aimmd.__file__).resolve().parent
    package = Path(api.bioemu2.__file__).resolve()
    assert root not in package.parents
    assert package.parent.name == 'bioemu2'


def test_other_versions_warn_once(api):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert api.check_version(api.SUPPORTED_VERSION) == '0.1.0rc0'
    with pytest.warns(RuntimeWarning, match='bioemu2 0.2.0 is installed; C-GAIMMD is tested with bioemu2 '
                                            '0.1.0rc0'):
        api.check_version('0.2.0')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        api.check_version('0.2.0')                     # once per process and version


def test_the_model_is_frozen_float32(api, tiny_bioemu2_model, golden):
    model = tiny_bioemu2_model
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {__import__('torch').float32}
    assert api.parameter_checksum(model) == str(golden['model.parameter_checksum'])
    assert api.token_feature_width(model) == 32


@pytest.fixture(scope='module')
def golden():
    return np.load(GOLDEN)


def test_contract_of_the_touch_points(api, tiny_bioemu2_model, tiny_protein_ligand_system):
    """Readout hook (a, q), the static context with gradients, the batch internals, the widths."""
    import torch
    model, system = tiny_bioemu2_model, tiny_protein_ligand_system
    topology, batch = api.sampling_inputs(system.bioemu2_system(), 'cpu')
    assert topology.n_atoms == system.n_bioemu2_atoms
    for name in ('coords', 'atom_mask', 'token_pad_mask', 'token_idx_at_atom'):
        assert getattr(batch, name).shape[0] == 1, name
    assert callable(batch.replace) and callable(batch.items)
    trunk = api.load_trunk(system.trunk_path, 'cpu')
    assert {f for f in ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram')
            if getattr(trunk, f).dtype == torch.float32} == {'s_inputs', 'single', 'pair',
                                                              'relative_position_encoding', 'distogram'}
    n_tokens = system.n_tokens
    assert trunk.single.shape[:2] == (1, n_tokens) and trunk.pair.shape[:3] == (1, n_tokens, n_tokens)
    assert float(model.edm_schedule.sigma_min) < float(model.edm_schedule.sigma_max)
    # the readout receives the token features a and atom features q as keyword arguments
    capture = api.ReadoutCapture(model)
    context = model.compute_static_context(batch, trunk, 2)
    x = torch.randn(2, batch.coords.shape[1], 3) * 5
    model.denoise(context, trunk, x, torch.full((2,), 1.0))
    assert capture.a.shape == (2, n_tokens, api.token_feature_width(model))
    assert capture.q.shape[:2] == (2, batch.coords.shape[1])
    with pytest.raises(RuntimeError, match="no longer receives the token features 'a' and the atom features 'q'"):
        capture._capture(None, (capture.a,), {})
    capture.remove()
    capture.clear()
    model.denoise(context, trunk, x, torch.full((2,), 1.0))
    assert capture.a is None
    # k copies of the batch; the static context keeps the gradient to a trunk that needs one
    expanded = api.expand_batch(batch, 3)
    assert expanded.coords.shape[0] == 3 and type(expanded) is type(batch)
    single = trunk.single.expand(3, -1, -1).clone().requires_grad_(True)
    pair = trunk.pair.expand(3, -1, -1, -1).clone().requires_grad_(True)
    rep = lambda v: v.expand(3, *v.shape[1:])                               # noqa: E731
    trunk_k = api.TrunkEmbeddings(s_inputs=rep(trunk.s_inputs), single=single, pair=pair,
                                  relative_position_encoding=rep(trunk.relative_position_encoding),
                                  distogram=rep(trunk.distogram))
    context_k = api.static_context_with_grad(model, expanded, trunk_k)
    out = model.denoise(context_k, trunk_k, torch.randn(3, batch.coords.shape[1], 3) * 5,
                        torch.full((3,), 2.0)).denoised_coords
    out.square().mean().backward()
    assert pair.grad is not None and float(pair.grad.abs().sum()) > 0


def test_query_only_msa(api, tiny_protein_ligand_system):
    msa = api.query_only_msa(tiny_protein_ligand_system.bioemu2_system())
    assert msa.chain_ids == ['A']
    assert msa.a3m('A') == '>query\nLQGIVSWGSGCAQKNKPGVY\n'


def _topology_in_smiles_order(smiles, chain='L'):
    """An mdtraj topology of the ligand with its heavy atoms in the order of `smiles` (as bioemu2 builds
    SMILES ligands after 0.1.0rc0)."""
    import mdtraj
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    topology = mdtraj.Topology()
    residue = topology.add_residue('UNK', topology.add_chain(chain_id=chain), resSeq=1)
    atoms = [topology.add_atom(f'{a.GetSymbol()}{k + 1}', mdtraj.element.get_by_symbol(a.GetSymbol()), residue)
             for k, a in enumerate(mol.GetAtoms())]
    for bond in mol.GetBonds():
        topology.add_bond(atoms[bond.GetBeginAtomIdx()], atoms[bond.GetEndAtomIdx()])
    return topology


def test_the_ligand_order_check_refuses_a_permuted_smiles(api, tiny_protein_ligand_system):
    system = tiny_protein_ligand_system
    stored = {'L': {key: system.ligands['L'][key] for key in ('elements', 'bonds')}}
    # 0.1.0rc0 orders SMILES ligands through InChI: the permuted SMILES gives the stored order
    spec = system.bioemu2_system().model_dump(exclude_none=True)
    spec['entities'][1]['smiles'] = PERMUTED_SMILES
    topology, _ = api.sampling_inputs(spec, 'cpu')
    assert api.check_ligand_order(stored, topology) == stored
    # a bioemu2 that keeps the SMILES order builds the same molecule in another order: refused (here the
    # element sequence happens to be the same, CCCCCCCNN, and the bonds tell)
    with pytest.raises(api.LigandOrderError, match=r"builds ligand chain 'L' differently from the folder: "
                                                   r'the bonds differ \(for example \[\[0, 2\], \[1, 3\], '
                                                   r'\[2, 4\]\] are missing\)'):
        api.check_ligand_order(stored, _topology_in_smiles_order(PERMUTED_SMILES), where='the folder')
    with pytest.raises(api.LigandOrderError, match='atom 0 is N instead of C'):
        api.check_ligand_order(stored, _topology_in_smiles_order('NC(=[NH2+])c1ccccc1'))
    with pytest.raises(api.LigandOrderError, match='7 atoms instead of 9'):
        api.check_ligand_order(stored, _topology_in_smiles_order('Oc1ccccc1'))


def test_a_folder_built_in_another_ligand_order_is_refused(api, tiny_protein_ligand_system, tmp_path):
    """A system folder whose ligand order came from a bioemu2 that keeps the SMILES order (simulated by
    rewriting the stored order) does not load."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2System
    folder = tmp_path / 'copy'
    shutil.copytree(tiny_protein_ligand_system.folder, folder)
    data = json.loads((folder / 'system.json').read_text())
    signature = api.ligand_signature(_topology_in_smiles_order(PERMUTED_SMILES), ['L'])['L']
    data['ligands']['L'].update(signature)
    (folder / 'system.json').write_text(json.dumps(data))
    with pytest.raises(api.LigandOrderError, match=f"differently from the system folder {folder}"):
        BioEmu2System.load(folder)
    assert BioEmu2System.load(folder, check=False).ligands['L']['elements'] == signature['elements']


def test_entities_the_map_does_not_support_are_named(api):
    spec = {'entities': [{'entity_type': 'protein', 'asym_ids': ['A'], 'sequence': 'GG'},
                         {'entity_type': 'rna', 'asym_ids': ['B'], 'sequence': 'GCAU'}]}
    with pytest.raises(NotImplementedError, match="entities of type 'rna' \\(chains \\['B'\\]\\) are not supported"):
        api.ligand_chains(spec)


def test_a_trunk_that_needs_cuequivariance_says_how_to_install_it(api, tiny_bioemu2_checkpoint, tmp_path,
                                                                     monkeypatch):
    """The released checkpoint's trunk uses cuEquivariance's triangle multiplication (trimul_variant 'cueq').

    Without cuEquivariance (the 'bioemu2' extra instead of 'bioemu2-cuda'),
    bioemu2 builds the model and fails inside the first trunk pass with
    ``NameError: name 'cueq_torch' is not defined``. The embedding says what
    to install instead.
    """
    from bioemu2.biomoa.ops import triangle_multiplication
    folder = tmp_path / 'cueq'
    folder.mkdir()
    text = (tiny_bioemu2_checkpoint.parent / 'config.yaml').read_text()
    assert 'trimul_variant: full' in text
    (folder / 'config.yaml').write_text(text.replace('trimul_variant: full', 'trimul_variant: cueq'))
    shutil.copy(tiny_bioemu2_checkpoint, folder / 'model.ckpt')
    monkeypatch.setattr(triangle_multiplication, 'HAS_CUEQ', False)
    model = api.load_model(checkpoint_path=folder / 'model.ckpt', device='cpu')
    system = api.make_system({'entities': [{'entity_type': 'protein', 'asym_ids': ['A'], 'sequence': 'YYDPETGTWY'}]})
    with pytest.raises(ImportError, match=r"aimmd-lab\[bioemu2-cuda\]") as info:
        api.embed(model, system, api.query_only_msa(system))
    assert 'cuEquivariance' in str(info.value)
    # a model without those kernels (the tiny one) embeds as before
    plain = api.load_model(checkpoint_path=tiny_bioemu2_checkpoint, device='cpu')
    assert api.embed(plain, system, api.query_only_msa(system)).single.shape[1] == 10
