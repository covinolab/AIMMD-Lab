"""
aimmd.cgaimmd.bioemu2.build
===========================

Building a BioEmu2 system folder: ``python -m aimmd.cgaimmd.bioemu2 build-system``.

A system folder holds everything about a system that a BioEmu2 C-GAIMMD run
needs besides the params file:

``system.json``
    The :class:`~aimmd.cgaimmd.bioemu2.system.BioEmu2System`: the BioEmu2
    system and the bioemu2 version, the MD <-> BioEmu2 atom map, the
    completion of the other generator atoms, stereocenters, the reference
    pose and core atoms, the token pools, the ligand atom order and where
    the files came from.
``trunk_embeddings.pt``
    ``model.embed(system, msa, mode)`` saved with ``TrunkEmbeddings.save``.
    The MSA (query-only unless alignments are given) and the mode are
    recorded in ``system.json``, with the checkpoint and its parameter
    checksum.
``bioemu2_system.yaml``
    The BioEmu2 system again, for reading.

The params file names the folder; nothing is read from environment variables.
The trunk is computed once here, so every process of a run uses the same
embeddings.

MDAnalysis reads the MD files (any topology it reads with bonds; a GROMACS
``.top`` is read with its ``#include`` files from the topology's folder or
``include_dir``).
"""

import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

from .system import SYSTEM_FILE, TRUNK_FILE, BioEmu2System, MDAtoms

__all__ = ['build_system', 'read_md_atoms', 'core_atoms_of_residues', 'MSA_QUERY_ONLY']

MSA_QUERY_ONLY = 'query-only'
"""The MSA of every protein chain is its own sequence (no ColabFold call; the chignolin campaign's choice)."""


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _universe(structure, topology, include_dir=None):
    import MDAnalysis as mda
    topology = Path(topology)
    kwargs = {}
    if topology.suffix in ('.top', '.itp'):
        kwargs['topology_format'] = 'ITP'
        kwargs['include_dir'] = str(include_dir if include_dir is not None else topology.parent)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')               # element guessing and missing attributes
        return mda.Universe(str(topology), str(structure), **kwargs)


def _whole(universe, n_atoms):
    """Positions ``(n_atoms, 3)`` of the first atoms, every molecule whole and next to the first one."""
    from MDAnalysis.lib.mdamath import make_whole, triclinic_vectors
    atoms = universe.atoms[:n_atoms]
    positions = atoms.positions.astype(np.float64)
    dimensions = universe.dimensions
    if dimensions is None or not np.all(np.asarray(dimensions[:3]) > 0):
        return positions
    fragments = []
    for fragment in atoms.fragments:
        if fragment.indices.max() >= n_atoms:
            raise ValueError(f'n_atoms = {n_atoms} cuts a molecule in two (atoms {fragment.indices.min()} to '
                             f'{fragment.indices.max()}): the generator atoms must be whole molecules')
        if len(fragment) > 1:
            positions[fragment.indices] = make_whole(fragment, inplace=False)
        fragments.append(np.sort(fragment.indices))
    fragments.sort(key=lambda f: f[0])
    box = triclinic_vectors(dimensions).astype(np.float64)
    shifts = np.array([i * box[0] + j * box[1] + k * box[2]
                       for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    placed = list(fragments[0])
    for fragment in fragments[1:]:
        best = min(range(len(shifts)), key=lambda s: (
            np.min(np.linalg.norm((positions[fragment] + shifts[s])[:, None] - positions[placed][None], axis=-1)),
            np.abs(shifts[s]).sum()))
        positions[fragment] += shifts[best]
        placed += list(fragment)
    return positions


def read_md_atoms(structure, topology, *, n_atoms=None, include_dir=None):
    """The generator atoms of an MD system as :class:`MDAtoms`, every molecule whole.

    Parameters
    ----------
    structure : str or Path
        A structure file of the MD system (``.gro``, ``.pdb``, ...).
    topology : str or Path
        Its topology with bonds and masses (a GROMACS ``.top`` or ``.tpr``, ...).
    n_atoms : int, optional
        The generator atoms are the first `n_atoms` atoms (the solute; whole
        molecules). Default: every atom.
    include_dir : str or Path, optional
        Where a ``.top``'s ``#include`` files are (default: its folder).

    Notes
    -----
    Molecules are made whole along their bonds, and every molecule after the
    first is moved by the lattice vector that brings it closest to the
    molecules before it (a bound ligand into its pocket). An unbound ligand is
    moved next to the protein, which is the right image for the completion
    but not a pocket: give a reference structure with the ligand bound.
    """
    universe = _universe(structure, topology, include_dir)
    n = len(universe.atoms) if n_atoms is None else int(n_atoms)
    if not 0 < n <= len(universe.atoms):
        raise ValueError(f'n_atoms must be between 1 and {len(universe.atoms)}, got {n_atoms}')
    atoms = universe.atoms[:n]
    try:
        bonds = universe.bonds.indices
    except Exception:                                  # noqa: BLE001 - topologies without bonds
        raise ValueError(f'the topology {topology} holds no bonds; the atom map needs them') from None
    bonds = bonds[(bonds < n).all(axis=1)]
    _, residues = np.unique(atoms.resindices, return_inverse=True)
    return MDAtoms(names=list(atoms.names), resnames=list(atoms.resnames), residues=residues,
                   resids=atoms.resids, masses=atoms.masses, positions=_whole(universe, n), bonds=bonds)


def core_atoms_of_residues(system, residues):
    """The BioEmu2 CA atoms of protein residues.

    Parameters
    ----------
    system : bioemu2.System or dict or path-like
    residues : sequence
        ``(chain, residue number)`` pairs, or residue numbers of the first
        protein chain; numbers count from 1 along each chain (BioEmu2's).
    """
    from . import _api
    topology, _ = _api.sampling_inputs(system, 'cpu')
    atoms = _api.topology_atoms(topology)
    first_chain = next(iter(_api.polymer_sequences(system)))
    wanted = []
    for item in residues:
        chain, number = (item if isinstance(item, (tuple, list)) else (first_chain, item))
        wanted.append((str(chain), int(number)))
    index = {(a['chain'], a['res_seq']): k for k, a in enumerate(atoms) if a['name'] == 'CA'}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ValueError(f'no CA atom for residues {missing[:5]} in the BioEmu2 system')
    return sorted(index[w] for w in wanted)


def build_system(bioemu2_system, md_structure, topology, out, *, n_atoms=None, reference_structure=None,
                 core_residues=None, pocket_cutoff=6.0, ion_cutoff=3.0, placement_references=None,
                 msa=MSA_QUERY_ONLY, mode='equilibrium', model=None, checkpoint='bioemu2-preview',
                 checkpoint_path=None, device=None, include_dir=None, trunk_embeddings=None, overwrite=False,
                 verbose=False):
    """Write a BioEmu2 system folder and return its checked :class:`BioEmu2System`.

    Parameters
    ----------
    bioemu2_system : str, Path, dict or bioemu2.System
        The BioEmu2 system (YAML file as ``System.from_yaml`` reads it). Its
        protein chains and ligands must be the generator atoms' (in order).
    md_structure, topology : str or Path
        The MD structure and topology; the structure is the template of the
        completion.
    out : str or Path
        The system folder (created).
    n_atoms : int, optional
        The generator atoms are the first `n_atoms` MD atoms (default all).
    reference_structure : str or Path or array_like, optional
        A structure of the same MD system for the canonical pose and the
        pocket (default `md_structure`), e.g. a crystal-based one with the
        ligand bound; or its coordinates, ``(n_atoms, 3)`` Ångström in MD
        order (generator atoms), taken as they are (a structure file is read
        by MDAnalysis in float32).
    core_residues : sequence, optional
        Protein residues whose CA atoms are fitted onto the reference pose
        (:func:`core_atoms_of_residues`); default every BioEmu2 atom.
    pocket_cutoff, ion_cutoff, placement_references
        See :meth:`BioEmu2System.build`.
    msa : str or dict, optional
        ``'query-only'`` or ``{chain: a3m file}``.
    mode : str, optional
        BioEmu2's sampling mode of the trunk embeddings (``'equilibrium'``).
    model : InferenceModel, optional
        A loaded model (else :func:`._api.load_model` with `checkpoint` /
        `checkpoint_path` on `device`).
    trunk_embeddings : str or Path, optional
        A saved ``TrunkEmbeddings`` file of this system to use instead of
        computing the trunk, for example the embeddings an earlier campaign
        used (the archived trypsin campaign's came from a sampling run with
        bf16 autocast and an alignment). The file is copied unchanged; its
        token count must be the system's. The record says ``msa='imported'``
        with the file and its md5, because how it was made is not known here.
        Default: computed with `model`, `msa` and `mode`.
    overwrite : bool, optional
        Replace an existing folder's files.
    """
    from . import _api
    out = Path(out)
    if (out / SYSTEM_FILE).exists() and not overwrite:
        raise FileExistsError(f'{out / SYSTEM_FILE} exists; pass overwrite=True (--overwrite) to rebuild it')
    system = _api.make_system(bioemu2_system)
    md = read_md_atoms(md_structure, topology, n_atoms=n_atoms, include_dir=include_dir)
    reference = None
    if isinstance(reference_structure, (str, os.PathLike)):
        ref = read_md_atoms(reference_structure, topology, n_atoms=md.n_atoms, include_dir=include_dir)
        reference = ref.positions
    elif reference_structure is not None:
        reference = np.asarray(reference_structure, dtype=np.float64)
        if reference.shape != (md.n_atoms, 3) or not np.isfinite(reference).all():
            raise ValueError(f'the reference coordinates must be ({md.n_atoms}, 3) finite Angstrom values of the '
                             f'generator atoms in MD order, got shape {reference.shape}')
    core = None if core_residues is None else core_atoms_of_residues(system, core_residues)
    sources = dict(bioemu2_system=None if isinstance(bioemu2_system, (dict, _api.System)) else str(bioemu2_system),
                   md_structure=str(md_structure), md_structure_md5=_md5(md_structure),
                   topology=str(topology), topology_md5=_md5(topology),
                   reference_structure=(None if reference_structure is None else str(reference_structure)
                                        if isinstance(reference_structure, (str, os.PathLike)) else 'coordinates'),
                   core_residues=None if core_residues is None else [list(r) if isinstance(r, (tuple, list))
                                                                     else int(r) for r in core_residues])
    built = BioEmu2System.build(system, md, reference=reference, core=core, pocket_cutoff=pocket_cutoff,
                                ion_cutoff=ion_cutoff, placement_references=placement_references,
                                sources=sources)
    if model is None:
        model = _api.load_model(checkpoint, checkpoint_path=checkpoint_path, device=device)
    imported = None
    if trunk_embeddings is not None:
        imported = Path(trunk_embeddings)
        trunk = _api.TrunkEmbeddings.load(imported, device=_api.torch.device('cpu'))
        msa_record = 'imported'
        if int(trunk.single.shape[1]) != built.n_tokens:
            raise ValueError(f'the trunk embeddings {imported} have {trunk.single.shape[1]} tokens, the system '
                             f'{built.n_tokens}: they belong to another system')
    elif msa == MSA_QUERY_ONLY:
        alignment, msa_record = _api.query_only_msa(system), MSA_QUERY_ONLY
    elif isinstance(msa, dict):
        alignment = _api.read_msa(msa)
        msa_record = {chain: dict(file=str(path), md5=_md5(path)) for chain, path in msa.items()}
    else:
        raise ValueError(f"msa must be {MSA_QUERY_ONLY!r} or {{chain: a3m file}}, got {msa!r}")
    if imported is None:
        trunk = _api.embed(model, system, alignment, mode)
    if int(trunk.single.shape[1]) != built.n_tokens:
        raise RuntimeError(f'the trunk embeddings have {trunk.single.shape[1]} tokens, the system {built.n_tokens}')
    out.mkdir(parents=True, exist_ok=True)
    for name in (SYSTEM_FILE, TRUNK_FILE):
        if (out / name).exists():
            os.remove(out / name)                       # overwrite: system.json goes first, so a half-written
    if imported is None:                                # folder never looks complete
        _api.save_trunk(trunk, out / TRUNK_FILE)
    else:
        shutil.copyfile(imported, out / TRUNK_FILE)
    built.data['trunk'] = dict(
        file=TRUNK_FILE, msa=msa_record, mode=str(mode) if imported is None else 'imported',
        checkpoint=None if checkpoint_path is not None else checkpoint,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        parameter_checksum=_api.parameter_checksum(model),
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        token_feature_width=_api.token_feature_width(model),
        single_width=int(trunk.single.shape[-1]), pair_width=int(trunk.pair.shape[-1]),
        dtypes={name: str(getattr(trunk, name).dtype) for name in ('s_inputs', 'single', 'pair',
                                                                   'relative_position_encoding', 'distogram')},
        md5=_md5(out / TRUNK_FILE))
    if imported is not None:
        built.data['trunk'].update(imported_from=str(imported), imported_md5=_md5(imported))
    yaml_path = out / 'bioemu2_system.yaml'
    if yaml_path.exists():
        os.remove(yaml_path)
    system.to_yaml(yaml_path)
    built.save(out)
    if verbose:
        print(json.dumps(built.summary(), indent=1))
    return BioEmu2System.load(out)
