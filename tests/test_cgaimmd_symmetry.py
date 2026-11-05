"""Permutation symmetries from the topology and the OT specification."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('networkx')

from aimmd.cgaimmd.preparation import symmetry as S                         # noqa: E402
from aimmd.cgaimmd.preparation import topology as T                         # noqa: E402

NANMA = Path(__file__).resolve().parent / 'cgaimmd_nanma'

TYPES = """\
[ atomtypes ]
  CT   6  12.011  0.0  A  0.35  0.27
  HC   1   1.008  0.0  A  0.25  0.12
  C    6  12.011  0.0  A  0.35  0.27
  OC   8  15.999  0.0  A  0.30  0.80
  OW   8  15.999  0.0  A  0.315 0.636
  HW   1   1.008  0.0  A  0.0   0.0
  S    16 32.060  0.0  A  0.35  1.0
"""

# acetate: a methyl and a carboxylate whose improper orders the two oxygens (as CHARMM's
# CG CB OD2 OD1 of Asp does)
ACETATE = """\
[ moleculetype ]
ACE 3
[ atoms ]
 1  CT  1  ACE  CB   1  -0.28  12.011
 2  HC  1  ACE  HB1  1   0.09   1.008
 3  HC  1  ACE  HB2  1   0.09   1.008
 4  HC  1  ACE  HB3  1   0.09   1.008
 5  C   1  ACE  CG   2   0.62  12.011
 6  OC  1  ACE  OD1  2  -0.76  15.999
 7  OC  1  ACE  OD2  2  -0.76  15.999
[ bonds ]
 1 2 1
 1 3 1
 1 4 1
 1 5 1
 5 6 1
 5 7 1
[ angles ]
 2 1 5 5
 3 1 5 5
 4 1 5 5
 1 5 6 5
 1 5 7 5
 6 5 7 5
[ dihedrals ]
 2 1 5 6 9
 3 1 5 6 9
 4 1 5 6 9
 2 1 5 7 9
 3 1 5 7 9
 4 1 5 7 9
 5 1 7 6 2
"""

# two identical arms S1-H and S2-H on a central carbon: a swap that moves bonded atoms together
# (found by the full analysis, not by the local twin swaps)
ARMS = """\
[ moleculetype ]
ARM 3
[ atoms ]
 1  CT  1  ARM  C1  1   0.0  12.011
 2  HC  1  ARM  H1  1   0.1   1.008
 3  HC  1  ARM  H2  1   0.1   1.008
 4  S   1  ARM  S1  1  -0.2  32.060
 5  HC  1  ARM  HS1 1   0.1   1.008
 6  S   1  ARM  S2  1  -0.2  32.060
 7  HC  1  ARM  HS2 1   0.1   1.008
[ bonds ]
 1 2 1
 1 3 1
 1 4 1
 4 5 1
 1 6 1
 6 7 1
"""

WATER = """\
[ moleculetype ]
SOL 2
[ atoms ]
 1  OW  1  SOL  OW   1  -0.834
 2  HW  1  SOL  HW1  1   0.417
 3  HW  1  SOL  HW2  1   0.417
[ settles ]
 1  1  0.09572  0.15139
"""


def _write(tmp_path, body, molecules, name='topol.top'):
    top = tmp_path / name
    top.write_text(TYPES + body + '[ system ]\nx\n[ molecules ]\n' + molecules)
    return top


def test_methyl_factors():
    """NANMA: three methyl groups, each an exact factor of order 6 (S3 on its hydrogens)."""
    mt = T.model_topology(NANMA / 'topol.top', n_solute=22)
    report = S.analyse(mt)
    assert isinstance(report, S.SymmetryReport)
    assert report.orbits == [[1, 2, 3], [11, 12, 13], [19, 20, 21]]
    assert [f.support for f in report.factors] == [(1, 2, 3), (11, 12, 13), (19, 20, 21)]
    assert all(isinstance(f, S.Factor) and f.order == 6 and f.exact for f in report.factors)
    assert [f.support for f in report.exact_factors] == [f.support for f in report.factors]
    first = report.factors[0]
    assert first.elements[0] == (0, 1, 2)                   # identity first
    assert sorted(first.elements) == sorted(
        {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)})
    assert report.molecule_blocks == []
    assert report.atom_names[0] == 'ALAD1.CL'
    summary = report.summary()
    assert summary['group_order'] == 216 and summary['exact_subgroup_order'] == 216
    assert summary['n_candidates_per_ot_pass'] == 18
    assert 'twins_only_min_atoms' not in summary
    assert report.select(('exact', 'approximate')) == report.factors
    spec = S.ot_spec(report)
    assert isinstance(spec, S.OTSpec) and spec.mode == 'topology'
    assert spec.groups == () and spec.molecule_blocks == ()
    assert [support for support, _ in spec.factors] == [(1, 2, 3), (11, 12, 13), (19, 20, 21)]
    kwargs = spec.kwargs()
    assert set(kwargs) == {'groups', 'factors', 'molecule_blocks'}
    assert kwargs['factors'] == spec.factors


def test_carboxylate_is_approximate(tmp_path):
    """A swap that turns an ordered improper into a term the topology lacks is approximate."""
    mt = T.model_topology(_write(tmp_path, ACETATE, 'ACE 1\n'), n_solute=7)
    report = S.analyse(mt)
    assert report.orbits == [[1, 2, 3], [5, 6]]
    kinds = {f.support: f.kind for f in report.factors}
    assert kinds == {(1, 2, 3): 'exact', (5, 6): 'approximate'}
    assert [f.support for f in report.exact_factors] == [(1, 2, 3)]
    assert report.summary()['exact_subgroup_order'] == 6
    assert report.summary()['group_order'] == 12
    exact = S.ot_spec(report, mode='topology')
    both = S.ot_spec(report, mode='topology+approximate')
    assert [s for s, _ in exact.factors] == [(1, 2, 3)]
    assert [s for s, _ in both.factors] == [(1, 2, 3), (5, 6)]
    assert S.ot_spec(report, mode='none') == S.OTSpec((), (), (), 'none')
    groups = S.ot_spec(None, mode='groups', groups=[[5, 6], (1, 2, 3)])
    assert groups == S.OTSpec(((5, 6), (1, 2, 3)), (), (), 'groups')
    with pytest.raises(ValueError, match='mode'):
        S.ot_spec(report, mode='orbits')
    with pytest.raises(ValueError, match='groups'):
        S.ot_spec(report, mode='topology', groups=[[5, 6]])
    assert S.SYMMETRY_MODES == ('topology', 'topology+approximate', 'groups', 'none')


def test_twins_only(tmp_path):
    """Large molecules get only their local twin swaps: a subgroup, tagged exact or not."""
    nanma = T.model_topology(NANMA / 'topol.top', n_solute=22)
    full, twins = S.analyse(nanma), S.analyse(nanma, twins_only_min_atoms=10)
    assert twins.orbits == full.orbits                         # methyl H are twins
    assert [(f.support, f.order, f.exact) for f in twins.factors] == [
        (f.support, f.order, f.exact) for f in full.factors]
    assert twins.twins_only_moleculetypes == ['Other']
    assert twins.summary()['twins_only_min_atoms'] == 10
    # moving bonded skeletons (the two S-H arms) are found by the full analysis only
    mt = T.model_topology(_write(tmp_path, ARMS, 'ARM 1\n'), n_solute=7)
    full = S.analyse(mt)
    assert full.orbits == [[1, 2], [3, 5], [4, 6]]
    coupled = [f for f in full.factors if 3 in f.support]
    assert len(coupled) == 1 and set(coupled[0].support) == {3, 4, 5, 6} and coupled[0].order == 2
    twins = S.analyse(mt, twins_only_min_atoms=7)
    assert twins.orbits == [[1, 2]]
    assert [f.support for f in twins.factors] == [(1, 2)]
    assert S.analyse(mt, twins_only_min_atoms=8).orbits == full.orbits    # below the threshold
    with pytest.raises(ValueError):
        S.analyse(mt, twins_only_min_atoms=0)


def test_factor_cap(tmp_path, capsys):
    """A factor above max_factor_order is not enumerated; its orbits go to the Hungarian groups."""
    mt = T.model_topology(NANMA / 'topol.top', n_solute=22)
    report = S.analyse(mt, max_factor_order=5)
    assert all(not f.enumerated and f.order == 6 and f.kind == 'unenumerated'
               for f in report.factors)
    spec = S.ot_spec(report)
    assert spec.factors == () and spec.groups == ((1, 2, 3), (11, 12, 13), (19, 20, 21))
    assert 'not enumerated' in capsys.readouterr().out


def test_water_blocks(tmp_path):
    """Identical waters are exchangeable whole blocks; their H swaps are exact factors."""
    top = _write(tmp_path, ACETATE + WATER, 'ACE 1\nSOL 5\n')
    mt = T.model_topology(top, n_solute=7, n_waters=3)
    report = S.analyse(mt)
    assert report.molecule_blocks == [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
    water_factors = [f for f in report.exact_factors if f.support[0] >= 7]
    assert [(f.support, f.order, f.molecule) for f in water_factors] == [
        ((8, 9), 2, 1), ((11, 12), 2, 2), ((14, 15), 2, 3)]
    spec = S.ot_spec(report)
    assert spec.molecule_blocks == ((7, 8, 9), (10, 11, 12), (13, 14, 15))
    assert S.ot_spec(report, mode='groups', groups=[[1, 2, 3]]).molecule_blocks == ()


def _frucht(tmp_path, copies=1):
    """A moleculetype of `copies` disconnected Frucht graphs of identical CT atoms."""
    import networkx as nx
    edges = sorted(nx.frucht_graph().edges())
    n = 12 * copies
    atoms = ''.join(f' {k + 1}  CT  1  FRU  C{k + 1}  1  0.0  12.011\n' for k in range(n))
    bonds = ''.join(f' {i + 1 + 12 * c} {j + 1 + 12 * c} 1\n'
                    for c in range(copies) for i, j in edges)
    top = _write(tmp_path, f'[ moleculetype ]\nFRU 3\n[ atoms ]\n{atoms}[ bonds ]\n{bonds}',
                 'FRU 1\n', name=f'frucht{copies}.top')
    return T.model_topology(top, n_solute=n)


def test_orbits_are_exact_not_wl_classes(tmp_path, capsys):
    """Orbits come from automorphisms, not from 1-WL colors (the Frucht graph tells them apart).

    The Frucht graph is 3-regular, so 1-WL gives all 12 atoms one color, but its automorphism
    group is trivial. Two copies are exchanged by one automorphism (order 2); counted without
    enumeration (max_factor_order 1), the group order must count automorphisms, not atoms of
    equal color.
    """
    report = S.analyse(_frucht(tmp_path))
    assert report.orbits == [] and report.factors == [] and report.wl_equals_orbits is False
    two = _frucht(tmp_path, copies=2)
    full = S.analyse(two)
    assert full.orbits == [[k, k + 12] for k in range(12)] and not full.wl_equals_orbits
    assert [(f.order, f.enumerated, f.exact, len(f.support)) for f in full.factors] == [
        (2, True, True, 24)]
    capped = S.analyse(two, max_factor_order=1)
    assert [(f.order, f.enumerated) for f in capped.factors] == [(2, False)]
    assert capped.summary()['group_order'] == 2
    assert 'not enumerated' in capsys.readouterr().out


def test_ot_token_mismatch(tmp_path):
    """An OT permutation that exchanges atoms with different tokens raises OTTokenMismatch."""
    mt = T.model_topology(_write(tmp_path, ACETATE + WATER, 'ACE 1\nSOL 2\n'), n_solute=7,
                          n_waters=2)
    report = S.analyse(mt)
    tokens = np.arange(13) + 1001                           # every atom its own token
    with pytest.raises(S.OTTokenMismatch, match='ACE1.HB'):
        S.ot_spec(report, tokens=tokens)
    assert issubclass(S.OTTokenMismatch, ValueError)
    tied = tokens.copy()
    tied[[2, 3]] = tied[1]                                   # methyl H tied
    tied[[9]] = tied[8]
    tied[[11, 12]] = tied[8]                                 # water H tied, water O shared
    tied[10] = tied[7]
    S.ot_spec(report, tokens=tied)                           # the exact factors: fine
    with pytest.raises(S.OTTokenMismatch):
        S.ot_spec(report, mode='topology+approximate', tokens=tied)     # OD1/OD2 differ
    with pytest.raises(S.OTTokenMismatch, match='groups'):
        S.ot_spec(None, mode='groups', groups=[[5, 6]], tokens=tied)
    blocks = tied.copy()
    blocks[10] = 5000                                        # the second water's O differs
    with pytest.raises(S.OTTokenMismatch, match='blocks'):
        S.check_ot_tokens(blocks, molecule_blocks=report.molecule_blocks)
    S.check_ot_tokens(tied, groups=[[1, 2, 3]], factors=[((1, 2, 3), [(0, 1, 2), (1, 0, 2)])],
                      molecule_blocks=report.molecule_blocks)
