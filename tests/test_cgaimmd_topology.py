"""GROMACS topology reader and derived topologies of the C-GAIMMD preparation tools.

The synthetic topologies are written into ``tmp_path`` and carry every mass
they need, so no GROMACS installation is required; the NANMA topology of
``tests/cgaimmd_nanma`` has its masses inline too (its force-field include is
skipped when GROMACS' share directory is not found).
"""

from pathlib import Path

import numpy as np
import pytest

from aimmd.cgaimmd.preparation import topology as T

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'

# a three-atom "solute" (X1-X2-X3 chain), a SETTLE water and a sodium ion
SOLUTE_ITP = """\
[ moleculetype ]
; name nrexcl
LIG   3

[ atoms ]
    1   CT   1  LIG   C1   1   -0.20   12.011
    2   CT   1  LIG   C2   2    0.10   12.011
    3   HC   1  LIG   H3   3    0.10    1.008

[ bonds ]
    1   2   1
    2   3   1

[ angles ]
    1   2   3   1
"""

WATER = """\
[ moleculetype ]
SOL   2

[ atoms ]
#ifdef _MY_FF
    1   OW   1  SOL   OW   1   -0.834
    2   HW   1  SOL  HW1   1    0.417
    3   HW   1  SOL  HW2   1    0.417
#endif

#ifdef FLEXIBLE
[ bonds ]
    1   2   1   0.09572   502416.0
    1   3   1   0.09572   502416.0
#else
[ settles ]
    1   1   0.09572   0.15139

[ exclusions ]
1 2 3
2 1 3
3 1 2
#endif
"""

HEADER = """\
; a test topology
#define _MY_FF

[ defaults ]
1   2   yes   0.5   0.8333

[ atomtypes ]
; name  at.num  mass     charge  ptype  sigma  epsilon
  OW    8       15.9994  0.000   A      0.315  0.636
  HW    1        1.008   0.000   A      0.0    0.0
  CT    6       12.011   0.000   A      0.35   0.27
  HC    1        1.008   0.000   A      0.25   0.12
  NA    11      22.990   0.000   A      0.24   0.20
"""

ION = """\
[ moleculetype ]
NA   1

[ atoms ]
    1   NA   1  NA   NA   1   1.0
"""


def _write_system(tmp_path, *, n_waters=4, n_ions=1, water_in_include=True, extra=''):
    """topol.top = header + lig.itp (included) + water (included or inline) + ion + molecules."""
    (tmp_path / 'lig.itp').write_text(SOLUTE_ITP)
    parts = [HEADER, '#include "lig.itp"\n']
    if water_in_include:
        (tmp_path / 'water.itp').write_text(WATER)
        parts.append('#include "water.itp"\n')
    else:
        parts.append(WATER)
    parts += [ION, extra, '[ system ]\ntest system\n\n[ molecules ]\n',
              'LIG 1\n', f'SOL {n_waters}\n', f'NA {n_ions}\n' if n_ions else '']
    top = tmp_path / 'topol.top'
    top.write_text(''.join(parts))
    return top


def test_read_topology_includes_defines(tmp_path):
    """Local includes are followed, #ifdef/#else/#define honoured, continuations joined."""
    top = _write_system(tmp_path, extra=(
        '[ moleculetype ]\nCONT 1\n\n[ atoms ]\n'
        '    1   CT   1  CON   C1   1 \\\n   0.0   12.011\n'))
    parsed = T.read_topology(top)
    assert isinstance(parsed, T.Topology)
    assert set(parsed.moleculetypes) == {'LIG', 'SOL', 'NA', 'CONT'}
    assert parsed.molecules == [('LIG', 1), ('SOL', 4), ('NA', 1)]
    water = parsed.moleculetypes['SOL']
    assert [a['name'] for a in water.atoms] == ['OW', 'HW1', 'HW2']      # under #ifdef _MY_FF
    assert water.atoms[0]['mass'] is None                                  # no mass column
    assert 'settles' in water.interactions and 'bonds' not in water.interactions
    assert water.bonds() == [(0, 1), (0, 2)]
    assert parsed.moleculetypes['CONT'].atoms[0]['mass'] == 12.011         # continuation joined
    assert parsed.atomtypes['OW']['mass'] == pytest.approx(15.9994)
    assert parsed.n_atoms == 3 + 4 * 3 + 1
    # a define given by the caller switches the branch
    flexible = T.read_topology(top, defines=('FLEXIBLE',)).moleculetypes['SOL']
    assert 'bonds' in flexible.interactions and 'settles' not in flexible.interactions
    offsets = [(k, mt.name, offset) for k, mt, offset in parsed.instances()]
    assert offsets[:3] == [(0, 'LIG', 0), (1, 'SOL', 3), (2, 'SOL', 6)]
    assert offsets[-1] == (5, 'NA', 15)


def test_include_search_path(tmp_path, monkeypatch):
    """An include is looked up next to the file, then in $GMXLIB (and GMXDATA/top)."""
    lib = tmp_path / 'lib'
    (lib / 'my.ff').mkdir(parents=True)
    (lib / 'my.ff' / 'lig.itp').write_text(SOLUTE_ITP)
    top = tmp_path / 'sys' / 'topol.top'
    top.parent.mkdir()
    top.write_text('#include "my.ff/lig.itp"\n[ system ]\nx\n[ molecules ]\nLIG 2\n')
    monkeypatch.delenv('GMXDATA', raising=False)
    monkeypatch.setenv('GMXLIB', str(lib))
    assert T.read_topology(top).n_atoms == 6
    monkeypatch.delenv('GMXLIB')
    monkeypatch.setenv('GMXDATA', str(tmp_path / 'data'))
    (tmp_path / 'data').mkdir()
    (lib).rename(tmp_path / 'data' / 'top')
    assert T.read_topology(top).n_atoms == 6
    # found nowhere: skipped, and the moleculetype it would have defined raises by name
    monkeypatch.delenv('GMXDATA')
    monkeypatch.setattr(T, '_gromacs_share_top', lambda: None)
    with pytest.raises(ValueError, match="'LIG'.*my.ff/lig.itp"):
        T.read_topology(top)


def test_read_masses(tmp_path):
    """[ atoms ] masses, [ atomtypes ] for the missing ones, multiplicities expanded."""
    top = _write_system(tmp_path, n_waters=2)
    masses = T.read_masses(top)
    np.testing.assert_allclose(masses, [12.011, 12.011, 1.008] + [15.9994, 1.008, 1.008] * 2
                               + [22.990])
    np.testing.assert_array_equal(T.read_atomic_numbers(top), [6, 6, 1, 8, 1, 1, 8, 1, 1, 11])
    with pytest.raises(ValueError, match='10 atoms but 11'):
        T.read_masses(top, expected=11)
    # NANMA: the CHARMM carbons named CL* are carbons, not chlorine
    nanma = T.read_masses(NANMA / 'topol.top', expected=22)
    assert nanma[0] == pytest.approx(12.011) and nanma[4] == pytest.approx(12.011)
    elements = T.read_atomic_numbers(NANMA / 'topol.top')
    assert elements.tolist().count(6) == 6 and elements.tolist().count(1) == 12
    # a mass that matches no element is not guessed; a virtual site is element 0
    bad = tmp_path / 'bad.top'
    bad.write_text('[ moleculetype ]\nX 1\n[ atoms ]\n1 A 1 X A1 1 0.0 3.024\n'
                   '2 V 1 X M 1 0.0 0.0\n[ molecules ]\nX 1\n')
    with pytest.raises(ValueError, match='3.024'):
        T.read_atomic_numbers(bad)
    assert T.read_masses(bad).tolist() == [3.024, 0.0]


def test_masses_to_elements():
    """Force-field roundings map to their element; united atoms and far masses raise."""
    rounded = {12.01: 6, 14.01: 7, 16.00: 8, 15.9994: 8, 32.065: 16, 30.97: 15, 65.40: 30,
               65.37: 30, 39.10: 19, 132.91: 55, 28.08: 14, 1.00794: 1, 35.453: 17}
    assert T._masses_to_atomic_numbers(list(rounded)).tolist() == list(rounded.values())
    # united-atom groups lie 0.02 u (CH2) and 0.044 u (CH4) from N and O; a HEAVY_H hydrogen
    # (4.032 u) near helium; AMBER's iron at 55.00 u is 0.06 u from manganese
    for mass, match in ((14.027, 'CH2'), (16.043, 'CH4'), (13.019, 'CH1'), (15.035, 'CH3'),
                        (4.032, 'HEAVY_H'), (55.0, '55'), (12.1, '12.1')):
        with pytest.raises(ValueError, match=match):
            T._masses_to_atomic_numbers([12.011, mass])
    # the tolerant mode of the descriptors' bond rule: the element a group is built on, else 0
    assert T._masses_to_atomic_numbers([14.027, 15.035, 4.032, 3.024, 12.011, 0.0],
                                       strict=False).tolist() == [6, 6, 1, 0, 6, 0]


def test_read_atom_records(tmp_path):
    """Records carry molecule instance, solvent and ion flags and the element."""
    top = _write_system(tmp_path, n_waters=2)
    records = T.read_atom_records(top)
    assert len(records) == 10 and all(isinstance(r, T.AtomRecord) for r in records)
    assert [r.index for r in records] == list(range(10))
    assert [r.molecule for r in records] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert [r.is_solvent for r in records] == [False] * 3 + [True] * 6 + [False]
    assert records[-1].is_ion and records[-1].element == 11
    assert records[3].residue_token == 'none' and records[0].residue_token == 'LIG'
    assert records[0].label == ('CT', -0.2, 12.011)
    # water residue name of another model
    other = T.read_atom_records(top, water_resname='HOH')
    assert not any(r.is_solvent for r in other)
    # ions by residue name
    named = T.read_atom_records(top, ion_resnames=('LIG',))
    assert named[0].is_ion and not named[-1].is_ion


def test_model_topology_solute_plus_waters(tmp_path):
    """The first n_solute atoms (whole molecules) plus n_waters copies of the water."""
    top = _write_system(tmp_path, n_waters=5)
    mt = T.model_topology(top, n_solute=3, n_waters=2)
    assert isinstance(mt, T.ModelTopology)
    assert mt.n_atoms == 9 and [r.index for r in mt.records] == list(range(9))
    assert [r.name for r in mt.records] == ['C1', 'C2', 'H3'] + ['OW', 'HW1', 'HW2'] * 2
    assert mt.bonds.tolist() == [[0, 1], [1, 2], [3, 4], [3, 5], [6, 7], [6, 8]]
    assert mt.molecules == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert mt.moleculetypes == ['LIG', 'SOL', 'SOL']
    assert [r.molecule for r in mt.records] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert mt.elements.tolist() == [6, 6, 1, 8, 1, 1, 8, 1, 1]
    assert [a for _, a, _ in mt.interactions['settles']] == [(3,), (6,)]
    assert mt.interactions['angles'] == [(1, (0, 1, 2), ())]
    # solute only (a solvated system without cluster waters)
    solute = T.model_topology(top, n_solute=3)
    assert solute.n_atoms == 3 and solute.moleculetypes == ['LIG']
    # vacuum: every atom
    nanma = T.model_topology(NANMA / 'topol.top', n_solute=22)
    assert nanma.n_atoms == 22 and len(nanma.bonds) == 21
    with pytest.raises(ValueError, match='cuts'):
        T.model_topology(top, n_solute=2)
    with pytest.raises(ValueError, match='atoms_per_water'):
        T.model_topology(top, n_solute=3, n_waters=1, atoms_per_water=4)
    with pytest.raises(ValueError, match="'HOH'"):
        T.model_topology(top, n_solute=3, n_waters=1, water_resname='HOH')
    with pytest.raises(ValueError, match='expands to 22'):
        T.model_topology(NANMA / 'topol.top', n_solute=23)


def test_topological_distances():
    """BFS distances clipped at max_distance; disconnected pairs get max_distance."""
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)]
    d = T.topological_distances(7, bonds, max_distance=4)
    assert d.dtype == np.int16 and d.shape == (7, 7)
    assert d[0].tolist() == [0, 1, 2, 3, 4, 4, 4]
    assert d[5, 6] == 1 and d[6, 5] == 1
    assert (d == d.T).all() and (np.diag(d) == 0).all()
    assert T.topological_distances(3, np.zeros((0, 2), int)).tolist() == [[0, 4, 4], [4, 0, 4],
                                                                           [4, 4, 0]]


def _posres(text, define):
    """Parse the position restraints of a derived topology text, per moleculetype."""
    out, current, active = {}, None, [True]
    section = None
    for raw in text.splitlines():
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('#ifdef'):
            active.append(line.split()[1] == define)
            continue
        if line.startswith('#endif'):
            active.pop()
            continue
        if line.startswith('#'):
            continue
        if line.startswith('['):
            section = line.strip('[] ').lower()
            continue
        if section == 'moleculetype':
            current = line.split()[0]
        elif section == 'position_restraints' and all(active):
            out.setdefault(current, []).append(line.split())
    return out


def test_restrained_topology_posres(tmp_path):
    """Restrained moleculetypes get a position-restraint block under the define."""
    text = T.restrained_topology(NANMA / 'topol.top', [0, 4, 5, 6], 1000.0)
    assert '#ifdef AIMMD_CGAIMMD_POSRES' in text
    blocks = _posres(text, 'AIMMD_CGAIMMD_POSRES')
    assert blocks == {'Other': [[str(i), '1', '1000', '1000', '1000'] for i in (1, 5, 6, 7)]}
    # the block sits inside the moleculetype, before [ system ], outside the POSRES include
    lines = text.splitlines()
    ours = lines.index('#ifdef AIMMD_CGAIMMD_POSRES')
    assert ours < lines.index('[ system ]')
    assert ours < lines.index('#ifdef POSRES') or ours > lines.index('#endif', lines.index(
        '#ifdef POSRES'))
    assert '#include "charmm27.ff/forcefield.itp"' in text             # includes kept as written
    # the derived file parses to the same system, with the restraints under the define
    derived = tmp_path / 'derived.top'
    derived.write_text(text)
    parsed = T.read_topology(derived, defines=('AIMMD_CGAIMMD_POSRES',))
    assert parsed.n_atoms == 22
    assert [a for _, a, _ in parsed.moleculetypes['Other'].interactions['position_restraints']] == [
        (0,), (4,), (5,), (6,)]
    assert 'position_restraints' not in T.read_topology(derived).moleculetypes[
        'Other'].interactions
    # a moleculetype defined in an include: the block follows the #include line
    top = _write_system(tmp_path, n_waters=3)
    text = T.restrained_topology(top, [0, 1, 2], 10.0, define='MYDEF')
    lines = text.splitlines()
    at = lines.index('#include "lig.itp"')
    assert lines[at + 1] == '#ifdef MYDEF'
    derived = tmp_path / 'derived_lig.top'           # next to lig.itp, so the include resolves
    derived.write_text(text)
    posres = T.read_topology(derived, defines=('MYDEF',)).moleculetypes['LIG'].interactions[
        'position_restraints']
    assert posres == [(1, (0,), ('10', '10', '10')), (1, (1,), ('10', '10', '10')),
                      (1, (2,), ('10', '10', '10'))]


def test_restrained_topology_water_copy(tmp_path):
    """The first K waters become a restrained, constrained copy; bulk O a flat-bottom restraint."""
    top = _write_system(tmp_path, n_waters=5)
    restrained = [0, 1, 2, 3, 6]                  # the solute and the O of the first two waters
    text = T.restrained_topology(top, restrained, 500.0, n_restrained_waters=2,
                                 flat_bottom=('OW', 8.0, 250.0))
    derived = tmp_path / 'restrained.top'
    derived.write_text(text)
    parsed = T.read_topology(derived, defines=('AIMMD_CGAIMMD_POSRES',))
    assert parsed.molecules == [('LIG', 1), ('SOLR', 2), ('SOL', 3), ('NA', 1)]
    assert parsed.n_atoms == 3 + 15 + 1
    copy = parsed.moleculetypes['SOLR']
    assert [a['name'] for a in copy.atoms] == ['OW', 'HW1', 'HW2']
    assert 'settles' not in copy.interactions
    assert sorted((a, p[-1]) for _, a, p in copy.interactions['constraints']) == [
        ((0, 1), '0.09572'), ((0, 2), '0.09572'), ((1, 2), '0.15139')]
    assert [a for _, a, _ in copy.interactions['position_restraints']] == [(0,)]
    bulk = parsed.moleculetypes['SOL'].interactions['position_restraints']
    assert bulk == [(2, (0,), ('1', '-0.8000', '250'))]
    # only one moleculetype has [ settles ] (GROMACS' limit)
    assert sum('settles' in m.interactions for m in parsed.moleculetypes.values()) == 1
    # the flat-bottom block follows the water include
    lines = text.splitlines()
    assert lines[lines.index('#include "water.itp"') + 1] == '#ifdef AIMMD_CGAIMMD_POSRES'
    # an inline water: the flat-bottom block ends the moleculetype, the copy precedes [ system ]
    sub = tmp_path / 'inline'
    sub.mkdir()
    inline = _write_system(sub, n_waters=4, water_in_include=False)
    text = T.restrained_topology(inline, [0, 1, 2, 3, 4, 5], 1.0, n_restrained_waters=1,
                                 water_copy_suffix='C')
    derived.write_text(text)
    parsed = T.read_topology(derived, defines=('AIMMD_CGAIMMD_POSRES',))
    assert parsed.molecules == [('LIG', 1), ('SOLC', 1), ('SOL', 3), ('NA', 1)]
    assert [a for _, a, _ in parsed.moleculetypes['SOLC'].interactions['position_restraints']] == [
        (0,), (1,), (2,)]
    assert 'position_restraints' not in parsed.moleculetypes['SOL'].interactions
    # the copy keeps the charges and masses as written (a 3-site model with 6 decimals, OPC3):
    # same values, a neutral molecule, and no mass column where the original has none
    for atoms, with_mass in (("""\
    1   OW   1  SOL   OW   1   -0.895170   15.99940
    2   HW   1  SOL  HW1   1    0.447585    1.00800
    3   HW   1  SOL  HW2   1    0.447585    1.00800
""", True), ("""\
    1   OW   1  SOL   OW   1   -0.895170
    2   HW   1  SOL  HW1   1    0.447585
    3   HW   1  SOL  HW2   1    0.447585
""", False)):
        sub = tmp_path / f'opc3_{with_mass}'
        sub.mkdir()
        opc3 = _write_system(sub, n_waters=3)
        (sub / 'water.itp').write_text(WATER.replace(WATER.split('#ifdef _MY_FF\n')[1].split(
            '#endif')[0], atoms))
        derived = sub / 'restrained.top'
        derived.write_text(T.restrained_topology(opc3, [0, 1, 2, 3], 1.0, n_restrained_waters=1))
        parsed = T.read_topology(derived)
        copy, original = parsed.moleculetypes['SOLR'].atoms, parsed.moleculetypes['SOL'].atoms
        assert [a['charge'] for a in copy] == [a['charge'] for a in original] == [
            -0.89517, 0.447585, 0.447585]
        assert sum(a['charge'] for a in copy) == sum(a['charge'] for a in original)
        assert [a['mass'] for a in copy] == [a['mass'] for a in original]
        assert (copy[0]['mass'] is not None) == with_mass
        np.testing.assert_array_equal(T.read_masses(derived), T.read_masses(opc3))


def test_partial_restraint_not_implemented(tmp_path):
    """Restraints that are neither whole moleculetypes nor the first K waters raise."""
    top = _write_system(tmp_path, n_waters=4)
    with pytest.raises(NotImplementedError):
        T.restrained_topology(top, [0, 1, 2, 6], 1.0)                 # water 2 without K
    with pytest.raises(NotImplementedError):
        T.restrained_topology(top, [0, 1, 2, 9], 1.0, n_restrained_waters=1)   # water 3 > K
    with pytest.raises(NotImplementedError):
        T.restrained_topology(top, [0, 1, 2, 3, 7], 1.0, n_restrained_waters=2)  # O of 1, H of 2
    two = tmp_path / 'two.top'
    two.write_text(HEADER + '#include "lig.itp"\n[ system ]\nx\n[ molecules ]\nLIG 2\n')
    with pytest.raises(NotImplementedError):
        T.restrained_topology(two, [0, 1, 2], 1.0)                    # one of two copies
    with pytest.raises(ValueError, match='SOLR'):
        T.restrained_topology(_write_system(tmp_path, extra='[ moleculetype ]\nSOLR 1\n'),
                              [0, 1, 2, 3], 1.0, n_restrained_waters=1)
    with pytest.raises(ValueError, match='out of range'):
        T.restrained_topology(top, [0, 99], 1.0)
    # a moleculetype that is not the last one of its include
    both = tmp_path / 'both.itp'
    both.write_text(SOLUTE_ITP + ION)
    three = tmp_path / 'three.top'
    three.write_text(HEADER + '#include "both.itp"\n[ system ]\nx\n[ molecules ]\nLIG 1\nNA 1\n')
    with pytest.raises(NotImplementedError, match='last'):
        T.restrained_topology(three, [0, 1, 2], 1.0)


# an AMBER-style water: settles under #ifndef FLEXIBLE, and a symbol defined and undefined again
WATER_IFNDEF = """\
[ moleculetype ]
SOL   2

[ atoms ]
    1   OW   1  SOL   OW   1   -0.834
    2   HW   1  SOL  HW1   1    0.417
    3   HW   1  SOL  HW2   1    0.417

#define ANGLED_WATER
#undef ANGLED_WATER
#ifndef FLEXIBLE
[ settles ]
    1   1   0.09572   0.15139
#else
[ bonds ]
    1   2   1   0.09572   502416.0
    1   3   1   0.09572   502416.0
#endif
#ifdef ANGLED_WATER
[ angles ]
    2   1   3   1   104.52   628.02
#endif
"""


def test_ifndef_and_undef(tmp_path):
    """#ifndef selects the settles of an AMBER water, #undef removes a symbol; the copy works."""
    top = _write_system(tmp_path, n_waters=3)
    (tmp_path / 'water.itp').write_text(WATER_IFNDEF)
    water = T.read_topology(top).moleculetypes['SOL']
    assert 'settles' in water.interactions and 'bonds' not in water.interactions
    assert 'angles' not in water.interactions                          # undefined again
    flexible = T.read_topology(top, defines=('FLEXIBLE',)).moleculetypes['SOL']
    assert 'bonds' in flexible.interactions and 'settles' not in flexible.interactions
    derived = tmp_path / 'restrained.top'
    derived.write_text(T.restrained_topology(top, [0, 1, 2, 3], 1.0, n_restrained_waters=1))
    parsed = T.read_topology(derived, defines=('AIMMD_CGAIMMD_POSRES',))
    assert parsed.molecules[:3] == [('LIG', 1), ('SOLR', 1), ('SOL', 2)]
    assert len(parsed.moleculetypes['SOLR'].interactions['constraints']) == 3


def test_include_from_gromacs_share(tmp_path, monkeypatch):
    """Without GMXLIB and GMXDATA an include is found in the share directory of AIMMD's gmx."""
    import aimmd._config as config
    install = tmp_path / 'gromacs'
    (install / 'bin').mkdir(parents=True)
    gmx = install / 'bin' / 'gmx'
    gmx.write_text('#!/bin/sh\n')
    gmx.chmod(0o755)
    forcefield = install / 'share' / 'gromacs' / 'top' / 'my.ff'
    forcefield.mkdir(parents=True)
    (forcefield / 'x.itp').write_text('[ atomtypes ]\n  CX  6  12.011  0.0  A  0.35  0.27\n'
                                      '  HX  1   1.008  0.0  A  0.10  0.10\n')
    top = tmp_path / 'sys' / 'topol.top'
    top.parent.mkdir()
    top.write_text('#include "my.ff/x.itp"\n[ moleculetype ]\nM 3\n[ atoms ]\n'
                   '1 CX 1 M C1 1 0.0\n2 HX 1 M H1 1 0.0\n[ bonds ]\n1 2 1\n'
                   '[ system ]\nx\n[ molecules ]\nM 2\n')
    monkeypatch.delenv('GMXLIB', raising=False)
    monkeypatch.delenv('GMXDATA', raising=False)
    monkeypatch.setattr(config, 'GROMACS', str(gmx), raising=False)
    assert T.read_topology(top).unresolved_includes == []
    assert T.read_masses(top).tolist() == [12.011, 1.008, 12.011, 1.008]   # from [ atomtypes ]
    assert T.read_atomic_numbers(top).tolist() == [6, 1, 6, 1]


def test_restraints_before_a_conditional_moleculetype(tmp_path):
    """A block whose moleculetype ends where a conditional opens goes before that conditional."""
    top = tmp_path / 'topol.top'
    top.write_text(HEADER + SOLUTE_ITP + """
#ifdef HEAVY_IONS
[ moleculetype ]
NA   1
[ atoms ]
    1   NA   1  NA   NA   1   1.0   45.98
#else
""" + ION + """#endif

[ system ]
x

[ molecules ]
LIG 1
NA 1
""")
    text = T.restrained_topology(top, [0, 1, 2], 10.0)
    lines = text.splitlines()
    assert lines.index('#ifdef AIMMD_CGAIMMD_POSRES') < lines.index('#ifdef HEAVY_IONS')
    derived = tmp_path / 'derived.top'
    derived.write_text(text)
    for defines in (('AIMMD_CGAIMMD_POSRES',), ('AIMMD_CGAIMMD_POSRES', 'HEAVY_IONS')):
        parsed = T.read_topology(derived, defines=defines)
        assert [a for _, a, _ in parsed.moleculetypes['LIG'].interactions[
            'position_restraints']] == [(0,), (1,), (2,)]
        assert 'position_restraints' not in parsed.moleculetypes['NA'].interactions
    assert T.read_masses(derived, expected=4)[-1] == 22.990


def test_neutral_atom_and_split_water_entries(tmp_path):
    """A neutral single atom is no ion; the water copy needs K waters in the first entry."""
    top = _write_system(tmp_path, n_waters=3, extra='[ moleculetype ]\nAR 1\n[ atoms ]\n'
                        '    1   CT   1  AR   AR   1   0.0   39.948\n')
    top.write_text(top.read_text() + 'AR 1\n')
    records = T.read_atom_records(top)
    assert records[-2].is_ion and not records[-1].is_ion and records[-1].element == 18
    split = tmp_path / 'split.top'
    split.write_text(top.read_text().replace('SOL 3\n', 'SOL 1\nSOL 2\n'))
    with pytest.raises(NotImplementedError, match='first'):
        T.restrained_topology(split, [3, 6], 1.0, n_restrained_waters=2)
