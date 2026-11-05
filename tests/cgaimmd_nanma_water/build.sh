#!/usr/bin/env bash
# Build tests/cgaimmd_nanma_water once: NANMA (tests/cgaimmd_nanma/conf.gro) in a 2.8 nm cubic box of
# TIP3P water with one Na+ and one Cl-, energy-minimized. The outputs (conf.gro, topol.top) are
# committed; the C-GAIMMD tests only read them. Run from this folder with GROMACS on PATH (or GMX set):
#   GMX=/path/to/gmx bash build.sh
set -euo pipefail
GMX=${GMX:-gmx}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/topol_base.top" "$WORK/topol.top"
cp "$HERE/em.mdp" "$WORK/"
cd "$WORK"
"$GMX" -quiet editconf -f "$HERE/../cgaimmd_nanma/conf.gro" -o box.gro -c -box 2.8 2.8 2.8 -bt cubic
"$GMX" -quiet solvate -cp box.gro -cs spc216.gro -p topol.top -o solvated.gro
"$GMX" -quiet grompp -f em.mdp -c solvated.gro -p topol.top -o ions.tpr -po ions_mdout.mdp -maxwarn 1
printf 'SOL\n' | "$GMX" -quiet genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL \
    -np 1 -nn 1 -seed 20260930
"$GMX" -quiet grompp -f em.mdp -c ions.gro -p topol.top -o em.tpr -po em_mdout.mdp
"$GMX" -quiet mdrun -deffnm em -ntmpi 1 -ntomp 2 -nb cpu
cp em.gro "$HERE/conf.gro"
cp topol.top "$HERE/topol.top"
grep -A10 '\[ molecules \]' topol.top
