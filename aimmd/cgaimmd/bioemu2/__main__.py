"""
``python -m aimmd.cgaimmd.bioemu2``: command-line tools of the BioEmu2 adaptor.

``build-system``
    Write a BioEmu2 system folder (:func:`aimmd.cgaimmd.bioemu2.build.build_system`)::

        python -m aimmd.cgaimmd.bioemu2 build-system --bioemu2-system system.yaml \\
            --md-structure run.gro --topology topol.top --n-atoms 3239 --out <system folder> \\
            [--msa query-only | --msa A=chain_A.a3m ...] [--pocket-cutoff 6.0] \\
            [--core-residues 16-245 | A:16-30 A:40] [--reference-structure crystal.gro] \\
            [--checkpoint bioemu2-preview | --checkpoint-path model.ckpt] [--device cuda]
            [--trunk-embeddings embeddings.pt]
"""

import argparse
import json
import sys


def _residues(specs):
    """``['16-20', 'A:30', 'B:1-3']`` -> ``[16, 17, ..., ('A', 30), ('B', 1), ...]``."""
    out = []
    for spec in specs:
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            chain, _, numbers = part.rpartition(':')
            first, _, last = numbers.partition('-')
            for number in range(int(first), int(last or first) + 1):
                out.append((chain, number) if chain else number)
    return out


def _msa(values):
    if values == ['query-only'] or not values:
        return 'query-only'
    out = {}
    for value in values:
        chain, sep, path = value.partition('=')
        if not sep:
            raise SystemExit(f"--msa takes 'query-only' or CHAIN=FILE.a3m entries, got {value!r}")
        out[chain] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m aimmd.cgaimmd.bioemu2',
                                     description='Tools of the C-GAIMMD BioEmu2 adaptor.')
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build-system', help='write a BioEmu2 system folder (atom map, trunk embeddings)')
    build.add_argument('--bioemu2-system', required=True, help='BioEmu2 System YAML (proteins and ligands)')
    build.add_argument('--md-structure', required=True, help='MD structure (the completion template)')
    build.add_argument('--topology', required=True, help='MD topology with bonds and masses (.top, .tpr, ...)')
    build.add_argument('--out', required=True, help='the system folder to write')
    build.add_argument('--n-atoms', type=int, default=None,
                       help='the generator atoms are the first N MD atoms (default: all)')
    build.add_argument('--reference-structure', default=None,
                       help='structure for the canonical pose and the pocket (default: --md-structure)')
    build.add_argument('--msa', nargs='*', default=['query-only'],
                       help="'query-only' (default) or CHAIN=FILE.a3m per protein chain")
    build.add_argument('--mode', default='equilibrium', help="BioEmu2 sampling mode of the trunk embeddings")
    build.add_argument('--pocket-cutoff', type=float, default=6.0,
                       help='Angstrom; protein tokens this close to the ligand form the pocket pool')
    build.add_argument('--ion-cutoff', type=float, default=3.0,
                       help='Angstrom; an unbonded atom is placed from the known heavy atoms this close')
    build.add_argument('--core-residues', nargs='*', default=None,
                       help='protein residues (BioEmu2 numbers, e.g. 16-245 or A:16-30) whose CA atoms set the '
                            'canonical pose (default: every BioEmu2 atom)')
    build.add_argument('--checkpoint', default='bioemu2-preview', help='Hugging Face checkpoint name')
    build.add_argument('--checkpoint-path', default=None, help='a local checkpoint file (config.yaml next to it)')
    build.add_argument('--device', default=None, help='default: cuda if available, else cpu')
    build.add_argument('--include-dir', default=None, help="folder of a .top's #include files")
    build.add_argument('--trunk-embeddings', default=None,
                       help='a saved TrunkEmbeddings file of this system to copy instead of computing the trunk '
                            '(e.g. the embeddings an earlier campaign used)')
    build.add_argument('--overwrite', action='store_true', help='rebuild an existing folder')
    args = parser.parse_args(argv)
    if args.command == 'build-system':
        from .build import build_system
        system = build_system(args.bioemu2_system, args.md_structure, args.topology, args.out,
                              n_atoms=args.n_atoms, reference_structure=args.reference_structure,
                              core_residues=None if args.core_residues is None else _residues(args.core_residues),
                              pocket_cutoff=args.pocket_cutoff, ion_cutoff=args.ion_cutoff, msa=_msa(args.msa),
                              mode=args.mode, checkpoint=args.checkpoint, checkpoint_path=args.checkpoint_path,
                              device=args.device, include_dir=args.include_dir,
                              trunk_embeddings=args.trunk_embeddings, overwrite=args.overwrite)
        print(json.dumps(system.summary(), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
