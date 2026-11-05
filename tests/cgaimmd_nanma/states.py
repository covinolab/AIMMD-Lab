"""Metastable states of NANMA (alanine dipeptide, CHARMM22 without CMAP) in vacuum.

The state definition of the C-GAIMMD development project (``systems/nanma_vacuum/states.py``,
2026-09-23): discs in the periodic (phi, psi) plane,

    A = C7eq: centre (-83, 73) deg, radius 25 deg
    B = C7ax: centre (71, -71) deg, radius 20 deg

everything else is R. Atom indices are 0-based into ``topol.top``/``conf.gro``
(phi = CLP-NL-CA-CRP, psi = NL-CA-CRP-NR). Bond vectors are minimum-imaged
against the frame's box, so a molecule split across the periodic boundary gives
the right angles.
"""

import numpy as np
from MDAnalysis.lib.distances import calc_dihedrals

PHI, PSI = [4, 6, 8, 14], [6, 8, 14, 16]
CENTRE_A, RADIUS_A = (-83.0, 73.0), 25.0
CENTRE_B, RADIUS_B = (71.0, -71.0), 20.0


def _wrap(x):
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _disc(phi, psi, centre, radius):
    return np.hypot(_wrap(phi - centre[0]), _wrap(psi - centre[1])) <= radius


def dihedrals(trajectory):
    """``(n_frames, 2)`` (phi, psi) in degrees."""
    out = np.empty((len(trajectory), 2))
    for i, frame in enumerate(trajectory):
        p = frame.positions.astype(np.float64)
        box = frame.dimensions
        box = box if box is not None and np.all(box[:3] > 0) else None
        out[i] = [calc_dihedrals(p[a], p[b], p[c], p[d], box=box) for a, b, c, d in (PHI, PSI)]
    return np.degrees(out)


def states_function(trajectory):
    """AIMMD's state function: ``'A'`` / ``'B'`` / ``'R'`` per frame."""
    d = dihedrals(trajectory)
    phi, psi = d[:, 0], d[:, 1]
    s = np.full(len(d), 'R', dtype='<U1')
    s[_disc(phi, psi, CENTRE_A, RADIUS_A)] = 'A'
    s[_disc(phi, psi, CENTRE_B, RADIUS_B)] = 'B'
    return s
