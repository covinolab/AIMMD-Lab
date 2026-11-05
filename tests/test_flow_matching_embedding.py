"""Embedding tokens of the generator atoms."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aimmd.cgaimmd.flow_matching import embedding as E
from aimmd.cgaimmd.preparation import topology as T

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'

TOP = """\
[ atomtypes ]
  CT   6  12.011  0.0  A  0.35  0.27
  HC   1   1.008  0.0  A  0.25  0.12
  OW   8  15.999  0.0  A  0.315 0.636
  HW   1   1.008  0.0  A  0.0   0.0
  NA   11 22.990  0.0  A  0.24  0.20

[ moleculetype ]
LIG 3
[ atoms ]
 1  CT  1  LIG  C1   1  -0.20  12.011
 2  HC  1  LIG  H1   1   0.10   1.008
 3  HC  1  LIG  H2   1   0.10   1.008
 4  CT  2  LIG  C1   2   0.00  12.011
[ bonds ]
 1 2 1
 1 3 1
 1 4 1

[ moleculetype ]
SOL 2
[ atoms ]
 1  OW  1  SOL  OW   1  -0.834
 2  HW  1  SOL  HW1  1   0.417
 3  HW  1  SOL  HW2  1   0.417
[ settles ]
 1  1  0.09572  0.15139

[ moleculetype ]
NA 1
[ atoms ]
 1  NA  1  NA  NA  1  1.0

[ system ]
x
[ molecules ]
"""


def _records(tmp_path, molecules='LIG 1\nSOL 2\nNA 2\n'):
    top = tmp_path / 'topol.top'
    top.write_text(TOP + molecules)
    return T.read_atom_records(top)


def test_element_scheme(tmp_path):
    """'element': the atomic numbers, whatever the tie; explicit elements override the records."""
    records = _records(tmp_path)
    tokens = E.atom_tokens(records, 'element')
    assert tokens.tolist() == [6, 1, 1, 6, 8, 1, 1, 8, 1, 1, 11, 11]
    assert E.atom_tokens(records, 'element', 'none').tolist() == tokens.tolist()
    assert E.atom_tokens(records, 'element', orbits=[[1, 2]]).tolist() == tokens.tolist()
    elements = np.arange(12) + 1
    assert E.atom_tokens(records, 'element', elements=elements).tolist() == elements.tolist()
    assert E.token_keys(records, 'element')[0] == ('element', 6)
    with pytest.raises(ValueError, match='scheme'):
        E.atom_tokens(records, 'identity')
    with pytest.raises(ValueError, match='tie'):
        E.atom_tokens(records, 'residue_atomname', 'groups')
    assert E.SCHEMES == ('element', 'residue_atomname') and E.TIES == ('orbits', 'none')


def test_residue_atomname_scheme():
    """(element, atom name, residue number): 1001 + rank of first appearance; NANMA all distinct."""
    records = T.read_atom_records(NANMA / 'topol.top')
    tokens = E.atom_tokens(records, 'residue_atomname', 'none')
    assert tokens.tolist() == list(range(E.TOKEN_OFFSET, E.TOKEN_OFFSET + 22))
    assert E.TOKEN_OFFSET == 1001 and tokens.min() > 118            # never atomic numbers
    keys = E.token_keys(records, 'residue_atomname')
    assert keys[0] == (6, 'CL', 1, None)


def test_solvent_none_token_shared(tmp_path):
    """Waters and ions share (element, name, 'none') across copies; the solute is per atom."""
    records = _records(tmp_path)
    tokens = E.atom_tokens(records, 'residue_atomname', 'none')
    # C1 of residue 1 and C1 of residue 2 differ (residue number)
    assert tokens[0] != tokens[3]
    assert tokens[4] == tokens[7] and tokens[5] == tokens[8] and tokens[6] == tokens[9]
    assert tokens[5] != tokens[6]                                   # HW1 and HW2 by name
    assert tokens[10] == tokens[11]
    assert E.token_keys(records, 'residue_atomname')[4] == ('solvent', 8, 'OW', 'none')
    assert tokens.tolist() == [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1005, 1006, 1007, 1008,
                               1008]


def test_chain_token(tmp_path):
    """Two copies of a non-solvent molecule are told apart by the chain token (unless off)."""
    records = _records(tmp_path, 'LIG 2\nSOL 1\n')
    with_chain = E.atom_tokens(records, 'residue_atomname', 'none')
    assert len(set(with_chain[:4]) & set(with_chain[4:8])) == 0
    without = E.atom_tokens(records, 'residue_atomname', 'none', chain_token=False)
    assert without[:4].tolist() == without[4:8].tolist()
    keys = E.token_keys(records, 'residue_atomname')
    assert keys[0][-1] == 0 and keys[4][-1] == 1
    # one solute molecule: no chain id in the key
    assert E.token_keys(_records(tmp_path, 'LIG 1\n'), 'residue_atomname')[0][-1] is None


def test_orbit_ties(tmp_path):
    """Orbit ties merge tokens transitively, never across elements."""
    records = _records(tmp_path)
    tied = E.atom_tokens(records, 'residue_atomname', 'orbits', orbits=[[1, 2], [5, 6]])
    assert tied[1] == tied[2] and tied[5] == tied[6] == tied[8] == tied[9]    # transitive
    assert tied.tolist() == [1001, 1002, 1002, 1003, 1004, 1005, 1005, 1004, 1005, 1005, 1006,
                             1006]
    assert E.atom_tokens(records, 'residue_atomname', 'none', orbits=[[1, 2]])[1] != \
        E.atom_tokens(records, 'residue_atomname', 'none', orbits=[[1, 2]])[2]
    with pytest.raises(ValueError, match='different elements'):
        E.atom_tokens(records, 'residue_atomname', 'orbits', orbits=[[0, 1]])
    # the NANMA methyl hydrogens tied along the symmetry orbits
    nanma = T.read_atom_records(NANMA / 'topol.top')
    tokens = E.atom_tokens(nanma, 'residue_atomname', orbits=[[1, 2, 3], [11, 12, 13],
                                                             [19, 20, 21]])
    assert len(set(tokens.tolist())) == 22 - 6
    assert E.atom_tokens([replace(r, element=1) for r in nanma], 'element').tolist() == [1] * 22
