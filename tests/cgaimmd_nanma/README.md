# NANMA in vacuum: test data of the C-GAIMMD preparation tools

N-acetyl-N'-methylalaninamide (alanine dipeptide), CHARMM22 without CMAP
(`charmm27.ff` from the GROMACS share directory), in vacuum: the system of the
C-GAIMMD development project (`systems/nanma_vacuum/`). The topology, structure and
mdp files are copied unchanged; `states.py` is a trimmed rewrite of the
project's state definition (the same A and B discs).

| File | Content |
|------|---------|
| `topol.top` | `pdb2gmx -ff charmm27 -water none -nocmap` topology (one molecule, 22 atoms) |
| `conf.gro` | the C7eq (A) structure in an 8 nm cubic box (`conf_A_box.gro` of the project) |
| `em.mdp` | steepest-descent EM, plain 3 nm cut-offs in the large box |
| `run.mdp` | production MD (`sd`, 1 fs, frames every 50 fs) |
| `states.py` | A = C7eq and B = C7ax discs in the (phi, psi) plane (the project's definition) |
| `free_md_A.xtc`, `free_md_B.xtc` | 10 frames of unbiased free MD in each basin, 1 ns apart |
| `make_free_md.py` | how the two `.xtc` files were cut from the project's unbiased per-basin MD |

The frames are unbiased MD only (no umbrella, OPES, steered or reweighted
frames): they serve as clash references and stand-in generated frames in the
C-GAIMMD tests. The tests that run GROMACS on this folder carry the `gromacs` marker.
