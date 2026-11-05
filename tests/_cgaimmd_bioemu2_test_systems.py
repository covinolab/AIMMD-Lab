"""The two small MD systems of the BioEmu2 adaptor tests, and how they were cut.

Both live in ``tests/data/`` as three files each: ``md.gro`` (the structure),
``md.top`` (a self-contained GROMACS topology with the ``[ atoms ]`` and
``[ bonds ]`` of the force field, no parameters, no includes) and
``bioemu2_system.yaml`` (the BioEmu2 ``System`` of the same molecules).

``cgaimmd_bioemu2_peptide``
    A 10-residue protein: CLN025 (chignolin, YYDPETGTWY), CHARMM22* names
    (N-terminal H1/H2/H3, C-terminal OT1/OT2), 166 atoms, from the
    project's ``systems/chignolin`` folder (``conf_A_box.gro``, folded, and
    ``topol.top``). It is the system of the archived chignolin campaign, so
    the atom map can be compared with that campaign's map.
``cgaimmd_bioemu2_protein_ligand``
    A 20-residue protein with a ligand: residues 187-206 of bovine
    beta-trypsin (LQGIVSWGSGCAQKNKPGVY, the MD's sequential numbering kept;
    they line the S1 pocket) and benzamidinium (BEN, 9 heavy atoms and 9
    hydrogens, SMILES ``NC(=[NH2+])c1ccccc1``) from the trypsin-benzamidine
    MD (Amber ff14SB + GAFF2; ``run.gro`` and ``topol.top`` of
    ``jupiter_aimmd_freshbuild_trial1``). The fragment's first residue keeps
    its single backbone H and its last residue has no OXT (the chain was
    cut there); Cys197 keeps its force-field name CYX without its partner.
    The coordinates are the MD's as they are, not made whole.

They are test data, not MD systems: nothing here is meant to be simulated.

Recreate them (the sources are read only) from the root of this checkout::

    python tests/_cgaimmd_bioemu2_test_systems.py \\
        <committor_flow>/systems/chignolin \\
        /mnt/plutus/new_aimmd_runs/trypsin-benzamidine/jupiter_aimmd_freshbuild_trial1

Standard library only.
"""

import hashlib
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
PEPTIDE = DATA / 'cgaimmd_bioemu2_peptide'
PROTEIN_LIGAND = DATA / 'cgaimmd_bioemu2_protein_ligand'

PEPTIDE_YAML = """\
# CLN025 (chignolin) for BioEmu2: one protein chain, no ligand.
entities:
- entity_type: protein
  asym_ids: [A]
  sequence: YYDPETGTWY
"""

PROTEIN_LIGAND_YAML = """\
# Residues 187-206 of bovine beta-trypsin (one chain) and benzamidinium, for BioEmu2.
entities:
- entity_type: protein
  asym_ids: [A]
  sequence: LQGIVSWGSGCAQKNKPGVY
- entity_type: ligand
  asym_ids: [L]
  smiles: NC(=[NH2+])c1ccccc1
"""

FRAGMENT = (187, 206)
"""First and last residue (MD numbering) of the trypsin fragment."""


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _sections(top_path):
    """``{moleculetype: {'atoms': [fields], 'bonds': [(i, j)]}}`` of a standalone topology."""
    blocks, current, section = {}, None, None
    for raw in Path(top_path).read_text().splitlines():
        line = raw.split(';')[0].strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            section = line.strip('[] ').lower()
            continue
        if section == 'moleculetype':
            current = line.split()[0]
            blocks[current] = {'atoms': [], 'bonds': []}
        elif current is not None and section == 'atoms':
            blocks[current]['atoms'].append(line.split()[:8])
        elif current is not None and section == 'bonds':
            fields = line.split()
            blocks[current]['bonds'].append((int(fields[0]), int(fields[1])))
    return blocks


def _write_top(path, header, molecules):
    """``molecules``: [(name, atoms (fields), bonds (1-based, renumbered))]."""
    lines = [f'; {text}' if text else ';' for text in header] + ['']
    for name, atoms, bonds in molecules:
        lines += ['[ moleculetype ]', '; name  nrexcl', f'{name}  3', '', '[ atoms ]',
                  ';   nr  type  resnr residue  atom  cgnr  charge  mass']
        lines += ['{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>12} {:>10}'.format(*atom) for atom in atoms]
        lines += ['', '[ bonds ]', ';  ai    aj funct']
        lines += [f'{i:>6} {j:>6}     1' for i, j in bonds]
        lines.append('')
    lines += ['[ system ]', 'test data of the C-GAIMMD BioEmu2 adaptor', '', '[ molecules ]']
    lines += [f'{name}  1' for name, _, _ in molecules]
    Path(path).write_text('\n'.join(lines) + '\n')


def _write_gro(path, title, atom_lines, box_line):
    out = [title, f'{len(atom_lines):>5}']
    for k, line in enumerate(atom_lines, start=1):
        out.append(line[:15] + f'{k % 100000:>5}' + line[20:])
    out.append(box_line)
    Path(path).write_text('\n'.join(out) + '\n')


def make_peptide(chignolin_dir):
    """The 10-residue protein, from ``systems/chignolin``."""
    chignolin_dir = Path(chignolin_dir)
    gro, top = chignolin_dir / 'conf_A_box.gro', chignolin_dir / 'topol.top'
    blocks = _sections(top)
    (name, block), = [(n, b) for n, b in blocks.items() if n.startswith('Protein')]
    PEPTIDE.mkdir(parents=True, exist_ok=True)
    lines = gro.read_text().splitlines()
    n = int(lines[1])
    assert n == len(block['atoms']) == 166
    _write_gro(PEPTIDE / 'md.gro', 'CLN025 (chignolin), folded, protein only (test data)',
               lines[2:2 + n], lines[2 + n])
    _write_top(PEPTIDE / 'md.top',
               ['CLN025 (chignolin), protein only, CHARMM22* atom names and masses; test data of the',
                'C-GAIMMD BioEmu2 adaptor: [ atoms ] (8 fields) and [ bonds ] of the source, no parameters.',
                f'source: {top} (md5 {_md5(top)})', f'structure: {gro} (md5 {_md5(gro)})'],
               [(name, block['atoms'], block['bonds'])])
    (PEPTIDE / 'bioemu2_system.yaml').write_text(PEPTIDE_YAML)


def make_protein_ligand(trypsin_dir, first=FRAGMENT[0], last=FRAGMENT[1]):
    """Trypsin residues ``first``-``last`` and benzamidinium, from the trypsin MD folder."""
    trypsin_dir = Path(trypsin_dir)
    gro, top = trypsin_dir / 'run.gro', trypsin_dir / 'topol.top'
    blocks = _sections(top)
    protein, ligand = blocks['system1'], blocks['BEN']
    keep = [k for k, atom in enumerate(protein['atoms']) if first <= int(atom[2]) <= last]
    new_index = {k + 1: i + 1 for i, k in enumerate(keep)}
    atoms = []
    for i, k in enumerate(keep):
        fields = list(protein['atoms'][k])
        fields[0], fields[5] = str(i + 1), str(i + 1)
        atoms.append(fields)
    bonds = sorted((new_index[a], new_index[b]) for a, b in protein['bonds']
                   if a in new_index and b in new_index)
    lines = gro.read_text().splitlines()
    n_protein = len(protein['atoms'])
    ligand_lines = lines[2 + n_protein:2 + n_protein + len(ligand['atoms'])]
    assert all(line[5:10].strip() == 'BEN' for line in ligand_lines)
    n = int(lines[1])
    PROTEIN_LIGAND.mkdir(parents=True, exist_ok=True)
    _write_gro(PROTEIN_LIGAND / 'md.gro',
               f'trypsin residues {first}-{last} + benzamidinium (test data; MD coordinates as they are)',
               [lines[2 + k] for k in keep] + ligand_lines, lines[2 + n])
    _write_top(PROTEIN_LIGAND / 'md.top',
               [f'Bovine beta-trypsin residues {first}-{last} (sequential MD numbering) and benzamidinium,',
                'Amber ff14SB + GAFF2 atom names and masses; test data of the C-GAIMMD BioEmu2 adaptor:',
                '[ atoms ] (8 fields) and [ bonds ] of the source restricted to these atoms, no parameters.',
                f'source: {top} (md5 {_md5(top)})', f'structure: {gro} (md5 {_md5(gro)})'],
               [('protein', atoms, bonds), ('BEN', ligand['atoms'], sorted(ligand['bonds']))])
    (PROTEIN_LIGAND / 'bioemu2_system.yaml').write_text(PROTEIN_LIGAND_YAML)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    make_peptide(sys.argv[1])
    make_protein_ligand(sys.argv[2])
