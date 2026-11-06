"""Building a BioEmu2 system folder (aimmd.cgaimmd.bioemu2.build and ``python -m aimmd.cgaimmd.bioemu2``).

Reading the MD files (MDAnalysis) and the command-line parsing need no
bioemu2; the build itself does (marker ``bioemu2``).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aimmd
from aimmd.cgaimmd.bioemu2.__main__ import _msa, _residues
from aimmd.cgaimmd.bioemu2.build import read_md_atoms

from ._bioemu2_fixtures import LIGAND_CORE_RESIDUES, tiny_bioemu2_checkpoint  # noqa: F401
from ._cgaimmd_bioemu2_golden import GOLDEN, PEPTIDE, PROTEIN_LIGAND

ROOT = Path(aimmd.__file__).resolve().parents[1]


def test_reading_the_md_makes_molecules_whole():
    """The test system's coordinates are the MD's (wrapped); bonds come out short, the ligand in its pocket."""
    md = read_md_atoms(PROTEIN_LIGAND / 'md.gro', PROTEIN_LIGAND / 'md.top')
    assert md.n_atoms == 309 and len(md.bonds) == 312
    assert md.elements[:3] == ['N', 'H', 'C'] and md.resnames[-1] == 'BEN' and md.resids[0] == 187
    assert md.residues[0] == 0 and md.residues[-1] == 20
    lengths = np.linalg.norm(md.positions[md.bonds[:, 0]] - md.positions[md.bonds[:, 1]], axis=1)
    assert lengths.max() < 2.1
    golden = np.load(GOLDEN)
    assert np.allclose(md.positions, golden['ligand.x_md_reference'], atol=1e-5)
    peptide = read_md_atoms(PEPTIDE / 'md.gro', PEPTIDE / 'md.top')
    assert peptide.n_atoms == 166 and peptide.names[-2:] == ['OT1', 'OT2']
    assert peptide.elements.count('H') == 166 - 93


def test_generator_atoms_are_whole_molecules():
    with pytest.raises(ValueError, match='n_atoms = 300 cuts a molecule in two'):
        read_md_atoms(PROTEIN_LIGAND / 'md.gro', PROTEIN_LIGAND / 'md.top', n_atoms=300)
    assert read_md_atoms(PROTEIN_LIGAND / 'md.gro', PROTEIN_LIGAND / 'md.top', n_atoms=291).n_atoms == 291


def test_command_line_values():
    assert _residues(['16-18', 'A:30', 'B:1-2,5']) == [16, 17, 18, ('A', 30), ('B', 1), ('B', 2), 5]
    assert _msa(['query-only']) == 'query-only' and _msa([]) == 'query-only'
    assert _msa(['A=a.a3m', 'B=b.a3m']) == {'A': 'a.a3m', 'B': 'b.a3m'}
    with pytest.raises(SystemExit, match="--msa takes 'query-only' or CHAIN=FILE.a3m"):
        _msa(['a.a3m'])


@pytest.mark.bioemu2
def test_build_system_from_the_command_line(tiny_bioemu2_checkpoint, tmp_path):
    """``python -m aimmd.cgaimmd.bioemu2 build-system`` writes a folder that loads, with the old trunk."""
    from aimmd.cgaimmd.bioemu2 import BioEmu2System, _api
    out = tmp_path / 'system'
    args = ['build-system', '--bioemu2-system', str(PROTEIN_LIGAND / 'bioemu2_system.yaml'),
            '--md-structure', str(PROTEIN_LIGAND / 'md.gro'), '--topology', str(PROTEIN_LIGAND / 'md.top'),
            '--out', str(out), '--checkpoint-path', str(tiny_bioemu2_checkpoint), '--device', 'cpu',
            '--core-residues', f'{LIGAND_CORE_RESIDUES[0]}-{LIGAND_CORE_RESIDUES[-1]}', '--pocket-cutoff', '6']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([str(ROOT)] + ([os.environ['PYTHONPATH']]
                                                         if os.environ.get('PYTHONPATH') else [])))
    proc = subprocess.run([sys.executable, '-m', 'aimmd.cgaimmd.bioemu2'] + args, cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout[proc.stdout.index('{'):])
    assert summary['n_atoms'] == 309 and summary['n_bioemu2_atoms'] == 155 and summary['ligands'] == ['L']
    assert summary['pools'] == {'ligand': 9, 'pocket': 13, 'protein': 20}
    assert sorted(p.name for p in out.iterdir()) == ['bioemu2_system.yaml', 'system.json', 'trunk_embeddings.pt']
    system = BioEmu2System.load(out)
    golden = np.load(GOLDEN)
    trunk_record = system.data['trunk']
    assert trunk_record['msa'] == 'query-only' and trunk_record['mode'] == 'equilibrium'
    assert trunk_record['checkpoint_path'] == str(tiny_bioemu2_checkpoint) and trunk_record['checkpoint'] is None
    assert trunk_record['parameter_checksum'] == str(golden['model.parameter_checksum'])
    assert trunk_record['token_feature_width'] == 32 and trunk_record['single_width'] == 16
    assert system.data['sources']['topology_md5'] and system.data['sources']['core_residues'] == list(
        LIGAND_CORE_RESIDUES)
    assert np.array_equal(system.core, golden['ligand.constants.core_be2'])
    trunk = _api.load_trunk(system.trunk_path, 'cpu')
    for field in ('s_inputs', 'single', 'pair', 'relative_position_encoding', 'distogram'):
        np.testing.assert_allclose(getattr(trunk, field).numpy(), golden[f'ligand.trunk.{field}'], atol=1e-5)
    assert _api.make_system(out / 'bioemu2_system.yaml') == system.bioemu2_system()
    # an existing folder is not overwritten unless asked
    from aimmd.cgaimmd.bioemu2.__main__ import main
    with pytest.raises(FileExistsError, match='pass overwrite=True \\(--overwrite\\) to rebuild it'):
        main(args)
    before = (out / 'trunk_embeddings.pt').stat().st_mtime_ns
    assert main(args + ['--overwrite']) == 0
    assert (out / 'trunk_embeddings.pt').stat().st_mtime_ns != before
    assert BioEmu2System.load(out).data['bioemu2_to_md'] == system.data['bioemu2_to_md']


def test_a_folder_reproduces_an_earlier_trunk_and_an_exact_reference(tiny_bioemu2_checkpoint, tmp_path):
    """The archived choices of an earlier campaign: its saved trunk embeddings and its reference pose, exactly.

    The archived trypsin campaign used trunk embeddings saved by another tool
    (bf16 autocast), and its canonical pose came from crystal coordinates kept
    as float32 arrays. A folder must be able to take both as they were: the
    trunk file is copied, not recomputed, and the reference may be given as
    coordinates (MD order, Ångström) instead of a structure file, which
    MDAnalysis would read in float32.
    """
    import torch
    from aimmd.cgaimmd.bioemu2 import _api, build_system
    from aimmd.cgaimmd.bioemu2.__main__ import main
    model = _api.load_model(checkpoint_path=tiny_bioemu2_checkpoint, device='cpu')
    first = build_system(PEPTIDE / 'bioemu2_system.yaml', PEPTIDE / 'md.gro', PEPTIDE / 'md.top', tmp_path / 'first',
                         model=model)
    earlier = _api.load_trunk(first.trunk_path, 'cpu')
    changed = _api.TrunkEmbeddings(s_inputs=earlier.s_inputs, single=earlier.single + 1.0,
                                   pair=earlier.pair.to(torch.bfloat16),
                                   relative_position_encoding=earlier.relative_position_encoding,
                                   distogram=earlier.distogram.to(torch.bfloat16))
    saved = tmp_path / 'earlier' / 'embeddings.pt'
    saved.parent.mkdir()
    _api.save_trunk(changed, saved)
    md = read_md_atoms(PEPTIDE / 'md.gro', PEPTIDE / 'md.top')
    reference = np.asarray(md.positions, np.float64) + 1e-3 * np.arange(md.n_atoms)[:, None]  # not float32-exact
    system = build_system(PEPTIDE / 'bioemu2_system.yaml', PEPTIDE / 'md.gro', PEPTIDE / 'md.top', tmp_path / 'second',
                          model=model, trunk_embeddings=saved, reference_structure=reference)
    assert (tmp_path / 'second' / 'trunk_embeddings.pt').read_bytes() == saved.read_bytes()
    record = system.data['trunk']
    assert record['imported_from'] == str(saved) and record['imported_md5'] == record['md5']
    assert record['msa'] == 'imported' and record['dtypes']['pair'] == 'torch.bfloat16'
    assert record['parameter_checksum'] == first.data['trunk']['parameter_checksum']
    assert system.data['sources']['reference_structure'] == 'coordinates'
    np.testing.assert_array_equal(system.reference, reference[system.bioemu2_to_md])
    loaded = _api.load_trunk(system.trunk_path, 'cpu')
    torch.testing.assert_close(loaded.single, earlier.single + 1.0, rtol=0, atol=0)
    # the command line takes the trunk file too
    args = ['build-system', '--bioemu2-system', str(PEPTIDE / 'bioemu2_system.yaml'), '--md-structure',
            str(PEPTIDE / 'md.gro'), '--topology', str(PEPTIDE / 'md.top'), '--out', str(tmp_path / 'third'),
            '--checkpoint-path', str(tiny_bioemu2_checkpoint), '--device', 'cpu', '--trunk-embeddings', str(saved)]
    assert main(args) == 0
    assert (tmp_path / 'third' / 'trunk_embeddings.pt').read_bytes() == saved.read_bytes()
    # a trunk of another system is refused
    other = tmp_path / 'other' / 'embeddings.pt'
    other.parent.mkdir()
    _api.save_trunk(_api.TrunkEmbeddings(s_inputs=earlier.s_inputs[:, :5], single=earlier.single[:, :5],
                                         pair=earlier.pair[:, :5, :5],
                                         relative_position_encoding=earlier.relative_position_encoding[:, :5, :5],
                                         distogram=earlier.distogram[:, :5, :5]), other)
    with pytest.raises(ValueError, match='5 tokens, the system 10'):
        build_system(PEPTIDE / 'bioemu2_system.yaml', PEPTIDE / 'md.gro', PEPTIDE / 'md.top', tmp_path / 'fourth',
                     model=model, trunk_embeddings=other)
    with pytest.raises(ValueError, match=r'reference coordinates must be \(166, 3\)'):
        build_system(PEPTIDE / 'bioemu2_system.yaml', PEPTIDE / 'md.gro', PEPTIDE / 'md.top', tmp_path / 'fifth',
                     model=model, reference_structure=reference[:10])
