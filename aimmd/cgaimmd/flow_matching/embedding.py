"""
aimmd.cgaimmd.flow_matching.embedding
=====================================

Embedding tokens of the generator atoms: what a network is told about each
atom besides its position.

Schemes (:data:`SCHEMES`):

- ``'element'``: the atomic number;
- ``'residue_atomname'`` (the default of ``JointPaiNN.from_topology``): one
  token per ``(element, force-field atom name, residue number, chain id)``.
  The chain id is the molecule-instance index among the non-solvent
  molecules, used when there is more than one of them (``chain_token``), so
  two copies of a ligand are told apart. Waters and ions get ``(element, atom
  name, 'none')`` and share their tokens across copies: they are
  interchangeable, and a per-copy residue number is not chemistry.

Ties (:data:`TIES`): ``'orbits'`` merges the tokens of the atoms of each
orbit of the labelled bond-graph automorphism group
(:func:`aimmd.cgaimmd.preparation.symmetry.analyse`), so the tokens are
exactly as symmetric as the topology; ``'none'`` merges nothing. Merging is transitive over token
values and never joins atoms of different elements.

Token values: ``'element'`` tokens are atomic numbers; the others are
``1001 + rank``, ranked by first appearance in atom order. Values above 118
are never atomic numbers, which is how a consumer tells a token list from an
element list.
"""

# Provenance: Port of ``cfaimmd.embedding`` (the prototype of C-GAIMMD) without the
# schemes and ties C-GAIMMD does not use. Standard library and numpy only.

import numpy as np

__all__ = ['SCHEMES', 'TIES', 'TOKEN_OFFSET', 'token_keys', 'atom_tokens']

SCHEMES = ('element', 'residue_atomname')
"""The embedding schemes."""

TIES = ('orbits', 'none')
"""How tokens are merged."""

TOKEN_OFFSET = 1001
"""The first non-element token."""


def _elements(records, elements):
    z = [int(r.element) for r in records] if elements is None else [int(v) for v in elements]
    if len(z) != len(records):
        raise ValueError(f'{len(z)} elements for {len(records)} atoms')
    return z


def token_keys(records, scheme, *, elements=None, chain_token=True):
    """Per-atom token keys of a scheme, before tying.

    Parameters
    ----------
    records : sequence of aimmd.cgaimmd.preparation.topology.AtomRecord
        The generator atoms, in network order.
    scheme : str
        One of :data:`SCHEMES`.
    elements : sequence of int, optional
        Atomic numbers overriding those of the records.
    chain_token : bool, optional
        Add the chain id when there is more than one non-solvent molecule.

    Returns
    -------
    list
        One hashable key per atom.

    Raises
    ------
    ValueError
        For an unknown scheme or an elements list of the wrong length.
    """
    if scheme not in SCHEMES:
        raise ValueError(f'embedding scheme must be one of {SCHEMES}, got {scheme!r}')
    z = _elements(records, elements)
    solute = sorted({r.molecule for r in records if not (r.is_solvent or r.is_ion)})
    chain = {m: k for k, m in enumerate(solute)} if (chain_token and len(solute) > 1) else None
    keys = []
    for i, record in enumerate(records):
        if scheme == 'element':
            keys.append(('element', z[i]))
        elif record.is_solvent or record.is_ion:
            keys.append(('solvent', z[i], record.name, 'none'))
        else:
            keys.append((z[i], record.name, int(record.resnr),
                         None if chain is None else chain[record.molecule]))
    return keys


def atom_tokens(records, scheme, tie='orbits', *, orbits=(), elements=None, chain_token=True):
    """Embedding tokens of the generator atoms.

    Parameters
    ----------
    records : sequence of aimmd.cgaimmd.preparation.topology.AtomRecord
        The generator atoms, in network order.
    scheme : str
        One of :data:`SCHEMES`.
    tie : str, optional
        One of :data:`TIES`.
    orbits : sequence of sequence of int, optional
        Non-trivial orbits (``SymmetryReport.orbits``), used with ``tie='orbits'``.
    elements : sequence of int, optional
        Atomic numbers overriding those of the records.
    chain_token : bool, optional
        Add the chain id when there is more than one non-solvent molecule.

    Returns
    -------
    numpy.ndarray
        ``(M,)`` int tokens: atomic numbers for ``'element'``, else ``1001 + rank``.

    Raises
    ------
    ValueError
        For an unknown scheme or tie, or an orbit that joins atoms of different elements.

    Examples
    --------
    >>> atom_tokens(records, 'residue_atomname', 'orbits', orbits=report.orbits)  # doctest: +SKIP
    """
    if tie not in TIES:
        raise ValueError(f'embedding tie must be one of {TIES}, got {tie!r}')
    if scheme not in SCHEMES:
        raise ValueError(f'embedding scheme must be one of {SCHEMES}, got {scheme!r}')
    z = np.asarray(_elements(records, elements), dtype=int)
    if scheme == 'element':
        return z.copy()
    keys = token_keys(records, scheme, elements=z, chain_token=chain_token)
    classes = orbits if tie == 'orbits' else ()
    for cls in classes:
        found = {int(z[int(i)]) for i in cls}
        if len(found) > 1:
            raise ValueError(f'tie {tie!r}: {list(cls)} would tie atoms of different elements '
                             f'{sorted(found)}')
    distinct = {key: j for j, key in enumerate(dict.fromkeys(keys))}
    parent = list(range(len(distinct)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cls in classes:
        cls = [int(i) for i in cls]
        for i in cls[1:]:
            a, b = find(distinct[keys[cls[0]]]), find(distinct[keys[i]])
            if a != b:
                parent[max(a, b)] = min(a, b)
    rank, out = {}, np.empty(len(records), dtype=int)
    for i, key in enumerate(keys):
        root = find(distinct[key])
        if root not in rank:
            rank[root] = len(rank)
        out[i] = TOKEN_OFFSET + rank[root]
    return out
