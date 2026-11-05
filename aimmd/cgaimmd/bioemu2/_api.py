"""
aimmd.cgaimmd.bioemu2._api
==========================

The one module of AIMMD-Lab that imports ``bioemu2`` (and RDKit, which comes
with it). Every other module of the adaptor reaches BioEmu2 through the
names defined here, so that a change of BioEmu2 has one place to be absorbed
and one contract test to fail.

What it holds:

- the import guard: without bioemu2 (or RDKit) importing this module raises
  one :class:`ImportError` with :data:`INSTALL_HINT`;
- the version policy: AIMMD-Lab is tested with bioemu2 :data:`SUPPORTED_VERSION`
  (= upstream commit e0b9903). :func:`check_version` warns once per process
  about any other version and does not raise (the real guard is the
  data-level check of the ligand atom order, below), so that a canary job can
  run newer versions;
- the public BioEmu2 calls the adaptor uses (``load_model``, ``System``,
  ``MSA``, ``TrunkEmbeddings``, ``EDMSampler`` (:func:`edm_sample`),
  ``InferenceModel.embed`` / ``compute_static_context`` / ``denoise`` /
  ``edm_schedule``);
- the internal touch points, each in one function (DESIGN section 5.2):
  ``prepare_sampling_inputs`` (:func:`sampling_inputs`); the forward pre-hook
  on ``score_model.structure_module.readout`` that reads the token features
  ``a`` and atom features ``q`` (:class:`ReadoutCapture`);
  ``score_model.compute_static_context`` with gradients
  (:func:`static_context_with_grad`); the batch internals behind
  :func:`expand_batch`; the widths read from the model
  (:func:`token_feature_width`); ``weighted_rigid_align``;
- the ligand atom-order check (:func:`ligand_signature`,
  :func:`check_ligand_order`). bioemu2 0.1.0rc0 builds a SMILES ligand
  through InChI, so its atoms come in InChI order; later versions build it
  in SMILES order. A system folder stores the element sequence and the bond
  graph of every ligand as BioEmu2 ordered them when the atom map and the
  trunk embeddings were made, and refuses to be used with a BioEmu2 that
  orders them differently;
- the RDKit graph helpers of the ligand atom map (:func:`element_graph`,
  :func:`graph_matches`, :func:`symmetry_classes`).

Nothing here patches bioemu2.
"""

import hashlib
import importlib.metadata
import warnings
import weakref
from pathlib import Path

from .. import _once

INSTALL_HINT = ("C-GAIMMD's BioEmu2 adaptor needs bioemu2 0.1.0rc0 (Python >= 3.12): "
                "pip install 'aimmd-lab[bioemu2]' (CUDA: 'aimmd-lab[bioemu2-cuda]')")
"""The message of the ImportError when bioemu2 is missing."""

SUPPORTED_VERSION = '0.1.0rc0'
"""The bioemu2 release this adaptor is tested with."""

try:
    import torch

    import bioemu2
    from bioemu2 import MSA, EDMSampler, System, TrunkEmbeddings
    from bioemu2.denoising.rigid_align import weighted_rigid_align
    from bioemu2.model import load_model as _load_model
    from bioemu2.structure.preparation import prepare_sampling_inputs
    from rdkit import Chem
except ImportError as _exc:                                   # pragma: no cover - depends on the install
    raise ImportError(f'{INSTALL_HINT} ({type(_exc).__name__}: {_exc})',
                      name=getattr(_exc, 'name', None) or 'bioemu2') from _exc

__all__ = ['INSTALL_HINT', 'SUPPORTED_VERSION', 'bioemu2', 'Chem', 'MSA', 'System', 'TrunkEmbeddings',
           'weighted_rigid_align', 'bioemu2_version', 'check_version', 'load_model', 'parameter_checksum',
           'model_parameter_checksum', 'checkpoint_snapshot',
           'make_system', 'system_to_dict', 'polymer_sequences', 'ligand_chains', 'query_only_msa', 'read_msa',
           'embed', 'save_trunk', 'load_trunk', 'sampling_inputs', 'topology_atoms', 'topology_bonds',
           'token_feature_width', 'ReadoutCapture', 'static_context_with_grad', 'expand_batch',
           'edm_sample', 'ligand_signature', 'check_ligand_order', 'LigandOrderError', 'element_graph',
           'graph_matches', 'symmetry_classes']

POLYMER_ENTITY_TYPES = ('protein',)
"""Entity types mapped residue by residue (atom names); nucleic acids and branched entities are not yet."""
LIGAND_ENTITY_TYPES = ('ligand',)
"""Entity types mapped by element graph."""


# ---------------------------------------------------------------------------
# versions and the model
# ---------------------------------------------------------------------------

def bioemu2_version():
    """The installed bioemu2 version (distribution metadata)."""
    try:
        return importlib.metadata.version('bioemu2')
    except importlib.metadata.PackageNotFoundError:          # pragma: no cover - source checkout
        return getattr(bioemu2, '__version__', 'unknown')


def check_version(version=None):
    """Warn (once per process) when bioemu2 is not :data:`SUPPORTED_VERSION`; return the version.

    It does not raise: a system folder refuses a BioEmu2 that orders its
    ligands differently (:func:`check_ligand_order`), and the saved trunk
    embeddings refuse a version mismatch unless allowed explicitly.
    """
    version = bioemu2_version() if version is None else str(version)
    if version != SUPPORTED_VERSION and _once.once(('bioemu2 version', version)):
        warnings.warn(f'bioemu2 {version} is installed; C-GAIMMD is tested with bioemu2 '
                      f'{SUPPORTED_VERSION}. Rebuild system folders with this version '
                      f'(python -m aimmd.cgaimmd.bioemu2 build-system) and check the results.',
                      RuntimeWarning, stacklevel=2)
    return version


def load_model(checkpoint='bioemu2-preview', *, checkpoint_path=None, device=None):
    """A frozen BioEmu2 model in full float32 precision.

    Parameters
    ----------
    checkpoint : str, optional
        The checkpoint name in the ``microsoft/bioemu2`` Hugging Face
        repository (the cached snapshot is used; set ``HF_HUB_OFFLINE=1`` on
        compute nodes). Ignored when `checkpoint_path` is given.
    checkpoint_path : str or Path, optional
        A local checkpoint file with its ``config.yaml`` next to it (tests,
        offline use).
    device : str or torch.device, optional
        Default: CUDA when available, else the CPU.

    Notes
    -----
    ``precision='32-true'`` and ``matmul_precision=None`` (no TF32), as in
    every production campaign: TF32 matmuls moved the pooled features of two
    identical poses by up to 1.5 % (smoke test of 23 Oct 2026).
    """
    check_version()
    if checkpoint_path is not None:
        return _load_model(checkpoint_path=Path(checkpoint_path), device=device, precision='32-true',
                           matmul_precision=None)
    return _load_model(checkpoint, device=device, precision='32-true', matmul_precision=None)


def parameter_checksum(model):
    """sha1 over every parameter and buffer of `model` (name and float32 bytes, sorted by name)."""
    digest = hashlib.sha1()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


_CHECKSUMS = weakref.WeakKeyDictionary()
"""Parameter checksums of loaded models (computed once per model object)."""


def model_parameter_checksum(model):
    """:func:`parameter_checksum` of `model`, computed once per model object (the model is frozen)."""
    try:
        return _CHECKSUMS[model]
    except (KeyError, TypeError):
        pass
    checksum = parameter_checksum(model)
    try:
        _CHECKSUMS[model] = checksum
    except TypeError:                                        # pragma: no cover - not weak-referenceable
        pass
    return checksum


def checkpoint_snapshot(checkpoint=None, checkpoint_path=None, *, cache_dir=None):
    """The Hugging Face snapshot (commit) a BioEmu2 checkpoint comes from, without any download.

    Parameters
    ----------
    checkpoint : str, optional
        A checkpoint name of the ``microsoft/bioemu2`` repository (``'bioemu2-preview'``): the
        snapshot of the local Hugging Face cache that holds it, found as
        ``bioemu2.checkpoints.get_checkpoint_files`` finds it.
    checkpoint_path : str or Path, optional
        A local checkpoint file instead: its snapshot when the path lies in a Hugging Face cache
        (read from the path as bioemu2 reads it), else None.
    cache_dir : str or Path, optional
        The Hugging Face cache (default: ``huggingface_hub.constants.HF_HUB_CACHE``).

    Returns
    -------
    str or None
        The snapshot folder name (the repository commit), or None when it cannot be told.
    """
    from bioemu2.checkpoints import HF_CHECKPOINT_REPO
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name
    folder = repo_folder_name(repo_id=HF_CHECKPOINT_REPO, repo_type='model')
    if checkpoint_path is not None:
        _, found, rest = Path(checkpoint_path).expanduser().absolute().as_posix().partition(
            f'{folder}/snapshots/')
        return rest.split('/')[0] if found and rest else None
    if checkpoint is None:
        return None
    hub_cache = Path(constants.HF_HUB_CACHE if cache_dir is None else cache_dir)
    for snapshot in (hub_cache / folder / 'snapshots').glob('*'):
        files = snapshot / 'checkpoints' / str(checkpoint)
        if (files / 'checkpoint.ckpt').is_file() and (files / 'config.yaml').is_file():
            return snapshot.name
    return None


def token_feature_width(model):
    """Width of the token features at the readout (768 for bioemu2-preview), read from the model."""
    return int(model.score_model.structure_module.single_conditioner.single_embed.out_features)


# ---------------------------------------------------------------------------
# systems, MSAs and trunk embeddings
# ---------------------------------------------------------------------------

def make_system(spec):
    """A ``bioemu2.System`` from a System, its dict (``System.model_dump``) or a YAML file."""
    if isinstance(spec, System):
        return spec
    if isinstance(spec, dict):
        return System.model_validate(spec)
    return System.from_yaml(Path(spec))


def system_to_dict(system):
    """The JSON-able dict of a ``bioemu2.System`` (``make_system`` rebuilds it)."""
    return make_system(system).model_dump(mode='json', exclude_none=True)


def polymer_sequences(system):
    """``{chain id: sequence}`` of the protein chains of `system`, in chain order."""
    out = {}
    for entity in make_system(system).entities:
        if entity.entity_type in POLYMER_ENTITY_TYPES:
            for chain in entity.asym_ids:
                out[chain] = entity.sequence
    return out


def ligand_chains(system):
    """``{chain id: {'smiles' | 'inchi' | 'ccd_code': ...}}`` of the ligand chains of `system`."""
    out = {}
    for entity in make_system(system).entities:
        if entity.entity_type in LIGAND_ENTITY_TYPES:
            spec = {key: getattr(entity, key) for key in ('smiles', 'inchi', 'ccd_code')
                    if getattr(entity, key, None) is not None}
            for chain in entity.asym_ids:
                out[chain] = spec
        elif entity.entity_type not in POLYMER_ENTITY_TYPES:
            raise NotImplementedError(
                f'BioEmu2 entities of type {entity.entity_type!r} (chains {list(entity.asym_ids)}) are not '
                f'supported by the C-GAIMMD atom map yet: proteins are mapped by residue and atom name, '
                f'ligands by their element graph')
    return out


def query_only_msa(system):
    """An MSA with only the query sequence of every protein chain (no ColabFold call)."""
    return MSA({chain: f'>query\n{sequence}\n' for chain, sequence in polymer_sequences(system).items()})


def read_msa(files):
    """An MSA from ``{chain id: a3m file}`` (unpaired alignments)."""
    return MSA({chain: Path(path).read_text() for chain, path in files.items()})


def embed(model, system, msa, mode='equilibrium'):
    """``model.embed(system, msa=msa, mode=mode)``: the trunk embeddings of `system`."""
    return model.embed(make_system(system), msa=msa, mode=mode)


def save_trunk(trunk, path):
    """``TrunkEmbeddings.save`` (refuses to overwrite an existing file)."""
    trunk.save(Path(path))


def load_trunk(path, device):
    """``TrunkEmbeddings.load``, every tensor as float32 (a saved bf16 distogram is cast exactly)."""
    trunk = TrunkEmbeddings.load(Path(path), device=torch.device(device))
    return TrunkEmbeddings(s_inputs=trunk.s_inputs.float(), single=trunk.single.float(),
                           pair=trunk.pair.float(),
                           relative_position_encoding=trunk.relative_position_encoding.float(),
                           distogram=trunk.distogram.float())


def sampling_inputs(system, device):
    """``(topology, batch)`` of `system` as BioEmu2's sampler builds them (``prepare_sampling_inputs``).

    The topology (mdtraj) lists the atoms BioEmu2 generates, in the order of
    its coordinates; the batch holds the model inputs of one system (padded
    atoms at the end).
    """
    return prepare_sampling_inputs(make_system(system), device=torch.device(device))


def topology_atoms(topology):
    """One dict per atom of a BioEmu2 topology, in order.

    Keys: ``chain`` (asym id), ``residue`` (0-based index over the whole
    topology), ``res_seq`` (1-based within its chain), ``resname``, ``name``,
    ``element``.
    """
    return [dict(chain=atom.residue.chain.chain_id, residue=int(atom.residue.index),
                 res_seq=int(atom.residue.resSeq), resname=atom.residue.name, name=atom.name,
                 element=atom.element.symbol)
            for atom in topology.atoms]


def topology_bonds(topology):
    """The bonds of a BioEmu2 topology as sorted ``(i, j)`` pairs, ``i < j``."""
    return sorted(tuple(sorted((int(a.index), int(b.index)))) for a, b in topology.bonds)


# ---------------------------------------------------------------------------
# internal touch points of the model
# ---------------------------------------------------------------------------

class ReadoutCapture:
    """The inputs of ``score_model.structure_module.readout``: token features ``a`` and atom features ``q``.

    A forward pre-hook (with keyword arguments) on the readout module. There
    is no public route to the token features: ``DenoiseOutput`` holds only
    the score and the denoised coordinates. If bioemu2 stops passing ``a``
    and ``q`` as keyword arguments the hook raises instead of capturing
    nothing.

    Parameters
    ----------
    model : InferenceModel
        A loaded BioEmu2 model. The hook stays installed until
        :meth:`remove`.

    Attributes
    ----------
    a, q : torch.Tensor or None
        What the readout received in the last denoiser pass (cleared by
        :meth:`clear`).
    """

    def __init__(self, model):
        self.a = self.q = None
        readout = model.score_model.structure_module.readout
        self._handle = readout.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs):
        if 'a' not in kwargs or 'q' not in kwargs:
            raise RuntimeError(
                'bioemu2 changed: score_model.structure_module.readout no longer receives the token '
                "features 'a' and the atom features 'q' as keyword arguments (it got "
                f'{sorted(kwargs)} and {len(args)} positional arguments); the C-GAIMMD features '
                f'need them (tested with bioemu2 {SUPPORTED_VERSION})')
        self.a, self.q = kwargs['a'], kwargs['q']

    def clear(self):
        self.a = self.q = None

    def remove(self):
        """Remove the hook from the model."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def static_context_with_grad(model, batch, trunk):
    """``model.score_model.compute_static_context(batch, trunk)``, which keeps gradients.

    ``InferenceModel.compute_static_context`` is ``@torch.no_grad()``, which
    would cut the gradient to an adapter of the pair embedding (it reaches
    the denoiser only through the static context).
    """
    return model.score_model.compute_static_context(batch, trunk)


def edm_sample(score_model, batch, trunk, n, num_steps):
    """``(n, n_pad, 3)``: BioEmu2's public ``EDMSampler`` (default churn and scales) on one system, `n` samples.

    `score_model` is any object with BioEmu2's ``InferenceModel`` protocol
    (the conditioned :class:`~aimmd.cgaimmd.bioemu2.sampling.GuidedModel`);
    the noise comes from torch's default random stream of the batch's device.
    """
    sampler = EDMSampler(num_steps=int(num_steps))
    return sampler(batch=batch, score_model=score_model, trunk_embeddings=trunk, batch_size=int(n)).coords


def expand_batch(batch, k):
    """``k`` copies of a one-system batch: every tensor with a leading system axis of size 1 repeated."""
    out = {}
    for key, value in batch.items():
        out[key] = (value.repeat(k, *([1] * (value.ndim - 1)))
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == 1 else value)
    return type(batch)(**out)


# ---------------------------------------------------------------------------
# the ligand atom order
# ---------------------------------------------------------------------------

class LigandOrderError(ValueError):
    """A BioEmu2 topology orders (or builds) a ligand differently from a system folder."""


def ligand_signature(topology, chains):
    """``{chain: {'elements': [...], 'bonds': [[i, j], ...]}}`` of the ligand `chains` of a BioEmu2 topology.

    Indices are local to the chain (0 = its first atom), bonds sorted with
    ``i < j``: the order BioEmu2 gives the ligand atoms, which the atom map and
    the trunk embeddings of a system folder depend on.
    """
    atoms = topology_atoms(topology)
    out = {}
    for chain in chains:
        index = [k for k, atom in enumerate(atoms) if atom['chain'] == chain]
        local = {k: i for i, k in enumerate(index)}
        bonds = sorted([local[i], local[j]] for i, j in topology_bonds(topology) if i in local and j in local)
        out[chain] = {'elements': [atoms[k]['element'] for k in index], 'bonds': bonds}
    return out


def check_ligand_order(stored, topology, *, where='the system folder'):
    """Raise :class:`LigandOrderError` unless `topology` orders every ligand as `stored` says.

    Parameters
    ----------
    stored : dict
        ``{chain: {'elements': [...], 'bonds': [[i, j], ...]}}`` (as
        :func:`ligand_signature` returns it) recorded when the atom map was
        built.
    topology : mdtraj.Topology
        The topology the installed bioemu2 builds for the same system.
    """
    current = ligand_signature(topology, list(stored))
    for chain, want in stored.items():
        have = current[chain]
        problem = None
        if len(have['elements']) != len(want['elements']):
            problem = f"{len(have['elements'])} atoms instead of {len(want['elements'])}"
        elif have['elements'] != list(want['elements']):
            k = next(i for i, (a, b) in enumerate(zip(have['elements'], want['elements'])) if a != b)
            problem = (f"atom {k} is {have['elements'][k]} instead of {want['elements'][k]} "
                       f"(elements {''.join(have['elements'])} instead of {''.join(want['elements'])})")
        elif [list(b) for b in have['bonds']] != [list(b) for b in want['bonds']]:
            missing = [b for b in want['bonds'] if list(b) not in have['bonds']]
            problem = f'the bonds differ (for example {missing[:3]} are missing)'
        if problem is not None:
            raise LigandOrderError(
                f'bioemu2 {bioemu2_version()} builds ligand chain {chain!r} differently from {where}: '
                f'{problem}. The atom map and the trunk embeddings of the folder were made for another '
                f'atom order (bioemu2 {SUPPORTED_VERSION} builds SMILES ligands through InChI, later '
                f'versions in SMILES order); rebuild the folder with this bioemu2 '
                f'(python -m aimmd.cgaimmd.bioemu2 build-system) or install bioemu2=={SUPPORTED_VERSION}')
    return current


# ---------------------------------------------------------------------------
# element graphs (RDKit)
# ---------------------------------------------------------------------------

def element_graph(elements, bonds):
    """An RDKit molecule of an element-labelled graph (single bonds, no hydrogens added, not sanitised).

    For graph matching only: bond orders and charges are not known from a
    force-field topology, and they do not matter for the atom map.
    """
    mol = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(str(element).capitalize())
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in bonds:
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def graph_matches(query, target, max_matches=100000):
    """Every isomorphism of two element graphs, sorted: ``match[i]`` = the target atom of query atom ``i``.

    Empty when the graphs differ (different sizes count as different).
    """
    if query.GetNumAtoms() != target.GetNumAtoms() or query.GetNumBonds() != target.GetNumBonds():
        return []
    return sorted(tuple(int(i) for i in match) for match in
                  target.GetSubstructMatches(query, uniquify=False, useChirality=False,
                                             maxMatches=int(max_matches)))


def symmetry_classes(mol):
    """The graph-symmetry class of every atom (equal for atoms an automorphism exchanges)."""
    return [int(rank) for rank in Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False)]
