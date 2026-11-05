"""Write free_md_A.xtc and free_md_B.xtc: 10 frames of unbiased free MD per basin.

Provenance of the committed files (run once, 2026-09-30, with the aimmd-flow-dev
environment). The source trajectories are the unbiased per-basin runs of the
C-GAIMMD development project (plain GROMACS ``sd`` MD of 10 ns started in each basin,
no bias, no restraints)::

    committor_flow/runs/md_10ns_per_basin/A/md.xtc   (started from conf_A_box.gro)
    committor_flow/runs/md_10ns_per_basin/B/md.xtc   (started from conf_B_box.gro)

Both used this folder's ``topol.top`` (identical file) and 50 fs frames. From
each file, the first frame in the file's own basin (``states.py``) at or after
t = 20 ps + k * 1 ns, k = 0..9, is kept, so the frames are 1 ns apart and
decorrelated. Coordinates are written as stored (a frame may be split across
the periodic boundary, as MD frames are).

Usage (paths of the development machine)::

    python make_free_md.py /path/to/committor_flow/runs/md_10ns_per_basin
"""

import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from states import states_function  # noqa: E402

N_FRAMES = 10
FIRST_PS = 20.0
SPACING_PS = 1000.0


def main(source):
    for state in 'AB':
        universe = mda.Universe(str(HERE / 'conf.gro'), str(Path(source) / state / 'md.xtc'))
        times = np.array([ts.time for ts in universe.trajectory])
        labels = states_function(universe.trajectory)
        picked = []
        for k in range(N_FRAMES):
            candidates = np.flatnonzero((times >= FIRST_PS + k * SPACING_PS) & (labels == state))
            picked.append(int(candidates[0]))
        with mda.Writer(str(HERE / f'free_md_{state}.xtc'), universe.atoms.n_atoms) as writer:
            for index in picked:
                universe.trajectory[index]
                writer.write(universe.atoms)
        print(state, [float(times[i]) for i in picked])


if __name__ == '__main__':
    main(sys.argv[1])
