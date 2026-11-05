# NANMA in TIP3P water: solvated test data of the C-GAIMMD preparation tools

The NANMA molecule of `../cgaimmd_nanma` (CHARMM22 without CMAP, `charmm27.ff`
from the GROMACS share directory) in a 2.8 nm cubic box of 695 TIP3P waters
with one Na+ and one Cl-, energy-minimized. Built once by `build.sh`
(`gmx editconf`, `gmx solvate`, `gmx genion -seed 20260930`, steepest-descent
EM) with GROMACS 2024.6; its outputs are committed and the tests only read
them.

| File | Content |
|------|---------|
| `topol_base.top` | `../cgaimmd_nanma/topol.top` plus the `tip3p.itp` and `ions.itp` includes (the input of `build.sh`) |
| `topol.top` | the topology of the built system (`[ molecules ]`: Other 1, SOL 695, NA 1, CL 1) |
| `conf.gro` | the built, minimized system (2109 atoms) |
| `em.mdp` | steepest-descent EM (PME, 1 nm cut-offs, h-bond constraints) |
| `nvt.mdp` | NVT MD (2 fs, V-rescale 300 K, PME); the template of the restrained NVT stage |
| `build.sh` | how `conf.gro` and `topol.top` were made |

Used by `tests/test_cgaimmd_solvate_gromacs.py` (marker `gromacs`).
