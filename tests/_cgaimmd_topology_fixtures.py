"""Synthetic GROMACS topologies of the force-field reference geometry and the topology-feature tests.

Self-contained (every parameter in the file: ``[ bondtypes ]``, ``[ angletypes ]``,
``[ constrainttypes ]``, explicit line parameters, a ``#define`` macro, SETTLE
waters), so no GROMACS installation is needed. Ported from the flow branch's
``tests/_topology_fixtures.py`` (``bb98943``), unchanged. Standard library and
numpy only (the PaiNN golden recorder loads it with the old code).
"""

import numpy as np

# ethanol + a sodium ion + SETTLE waters, every parameter by type lookup
ETHANOL_TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
CT  6  12.011  0.0  A  0.35  0.27
HC  1   1.008  0.0  A  0.25  0.12
OH  8  15.999  0.0  A  0.31  0.71
HO  1   1.008  0.0  A  0.0   0.0
OW  8  15.9994 0.0  A  0.315 0.636
HW  1   1.008  0.0  A  0.0   0.0
NA 11  22.990  0.0  A  0.24  0.20

[ bondtypes ]
; i  j  func  b0  kb
CT  HC  1  0.1090  284512.0
CT  CT  1  0.1529  224262.4
OH  CT  1  0.1410  267776.0
OH  HO  1  0.0945  462750.4

[ angletypes ]
; i  j  k  func  th0  cth
HC  CT  HC  1  107.8  276.144
HC  CT  CT  1  110.7  313.800
CT  CT  OH  1  109.5  418.400
HC  CT  OH  1  109.5  292.880
CT  OH  HO  1  108.5  460.240

[ moleculetype ]
ETH 3

[ atoms ]
1 CT 1 ETH C1  1 -0.18 12.011
2 HC 1 ETH H11 1  0.06  1.008
3 HC 1 ETH H12 1  0.06  1.008
4 HC 1 ETH H13 1  0.06  1.008
5 CT 1 ETH C2  1  0.14 12.011
6 HC 1 ETH H21 1  0.03  1.008
7 HC 1 ETH H22 1  0.03  1.008
8 OH 1 ETH O   1 -0.68 15.999
9 HO 1 ETH HO  1  0.42  1.008

[ bonds ]
1 2 1
1 3 1
1 4 1
1 5 1
5 6 1
5 7 1
5 8 1
8 9 1

[ angles ]
2 1 3 1
2 1 4 1
3 1 4 1
2 1 5 1
3 1 5 1
4 1 5 1
1 5 6 1
1 5 7 1
6 5 7 1
1 5 8 1
6 5 8 1
7 5 8 1
5 8 9 1

[ moleculetype ]
NA 1

[ atoms ]
1 NA 1 NA NA 1 1.0 22.990

[ moleculetype ]
SOL 2

[ atoms ]
1 OW 1 SOL OW  1 -0.834
2 HW 1 SOL HW1 1  0.417
3 HW 1 SOL HW2 1  0.417

[ settles ]
1 1 0.09572 0.15139

[ exclusions ]
1 2 3
2 1 3
3 1 2

[ system ]
ethanol, an ion and water

[ molecules ]
ETH 1
NA  1
SOL 3
"""

# explicit line parameters, constraints, a GROMOS-style macro and an OPLS-style bond_type column
MIXED_TOP = """\
#define gb_test 0.1000 1.5700e+07

[ defaults ]
1 3 yes 0.5 0.5

[ atomtypes ]
; name  bond_type at.num mass charge ptype sigma epsilon
opls_c  CX  6  12.011  0.0  A  0.35  0.27
opls_h  HX  1   1.008  0.0  A  0.25  0.12
opls_n  NX  7  14.007  0.0  A  0.32  0.71

[ bondtypes ]
CX  HX  1  0.1100  300000.0

[ constrainttypes ]
NX  HX  1  0.1010

[ angletypes ]
HX  CX  HX  1  109.0  300.0
HX  CX  NX  1  110.0  300.0
CX  NX  HX  5  112.0  300.0  0.21  1000.0

[ moleculetype ]
MOL 3

[ atoms ]
1 opls_c 1 MOL C  1 0.0 12.011
2 opls_h 1 MOL H1 1 0.0  1.008
3 opls_h 1 MOL H2 1 0.0  1.008
4 opls_n 1 MOL N  1 0.0 14.007
5 opls_h 1 MOL HN 1 0.0  1.008
6 opls_h 1 MOL H3 1 0.0  1.008

[ bonds ]
1 2 1
1 3 1 0.1080 300000.0
1 4 2 gb_test
1 6 1

[ constraints ]
4 5 1

[ angles ]
2 1 3 1
2 1 4 1
3 1 4 1 111.0 300.0
2 1 6 1
3 1 6 1
4 1 6 1
1 4 5 5

[ system ]
mixed

[ molecules ]
MOL 1
"""

ETHANOL_NAMES = ['C1', 'H11', 'H12', 'H13', 'C2', 'H21', 'H22', 'O', 'HO']
ETHANOL_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                              [-0.63, 0.63, -0.63], [1.5, 0.0, 0.0], [1.9, 1.0, 0.0],
                              [1.9, -0.5, 0.9], [2.0, -0.7, -1.1], [2.9, -0.7, -1.1]])


def write(tmp_path, text, name='topol.top'):
    path = tmp_path / name
    path.write_text(text)
    return path


def write_ethanol_gro(tmp_path, positions=ETHANOL_POSITIONS, name='conf.gro'):
    lines = ['ethanol', str(len(positions))]
    for i, (atom, pos) in enumerate(zip(ETHANOL_NAMES, positions / 10.0 + 1.0)):
        lines.append(f'{1:5d}{"ETH":<5s}{atom:>5s}{i + 1:5d}{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}')
    lines.append('   3.00000   3.00000   3.00000')
    gro = tmp_path / name
    gro.write_text('\n'.join(lines) + '\n')
    return str(gro)
