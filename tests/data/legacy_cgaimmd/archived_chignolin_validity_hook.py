"""The validity hook of the archived chignolin production params, verbatim.

Source: ``params_chignolin_faimmd_equilibrium_free.py`` of the online BioEmu2
chignolin campaign of 27 Oct 2026 (archived on plutus in
``bioemu2_campaign_backups/2026-10-27_chignolin-faimmd-equilibrium-free_full_run/``,
md5 07b257c75fee8c3d869ff2d0bf1973c8), lines 122-160: the class ``_AtomCount``
and the function ``flow_validity_function``, copied unchanged. They call the
polish by its old private name ``aimmd.flow.validate._polish``.

The params file built the objects the hook uses from a system folder on
JUPITER; here the test sets them as module attributes before calling the
hook:

- ``N_PROTEIN``: the number of generator atoms (166 for chignolin; the class
  ``_AtomCount`` reads it when this module is loaded, so set
  ``_AtomCount.n_atoms`` as well);
- ``_validity``: the ``TopologyValidity`` of the generator atoms;
- ``_polisher``: the minimizer of the polish;
- ``_SYSTEM``: the folder of ``conf_A_box.gro``, the structure of the
  generator atoms alone (names and box of the polish);
- ``_POLISH_WORKERS``: the number of polish threads.

``tests/test_cgaimmd_polished_validity.py`` checks that
``aimmd.cgaimmd.preparation.validity.PolishedValidity`` gives what this hook
gives, and that this hook still runs through the old names.
"""
import os

import numpy as np

N_PROTEIN = None
_validity = None
_polisher = None
_SYSTEM = None
_POLISH_WORKERS = 1

# ---- verbatim from the archived params file (lines 122-160) ----------------------------------

class _AtomCount:
    n_atoms = N_PROTEIN                  # GromacsMinimizer asks the network only for n_atoms


def flow_validity_function(positions, context):
    """Validity of the raw sample (recorded), restrained-EM polish, validity of the polished sample (decides)."""
    import types
    import MDAnalysis as mda
    from aimmd.flow import validate
    from aimmd.flow.types import ValidityResult
    x = np.asarray(positions, dtype=np.float64)
    raw = _validity(x, context)
    raw_pos = np.asarray(raw.positions, dtype=np.float64)
    finite = np.isfinite(raw_pos).reshape(len(x), -1).all(axis=1)
    out_pos, ok = raw_pos.copy(), np.zeros(len(x), bool)
    reason = np.array(['polish: not finite'] * len(x), dtype=object)
    info = [dict(raw_ok=bool(raw.ok[i]), raw_reason=str(raw.reason[i])) for i in range(len(x))]
    idx = np.flatnonzero(finite)
    if len(idx):
        scratch = os.path.join(context.scratch, 'polish')
        os.makedirs(scratch, exist_ok=True)
        uni = mda.Universe(os.path.join(_SYSTEM, 'conf_A_box.gro'))
        p = types.SimpleNamespace(universe=uni, parent=_SYSTEM)
        res = validate._polish(raw_pos[idx], _polisher, _AtomCount(), p, uni.dimensions, scratch, _POLISH_WORKERS,
                               int(context.seed))
        done = np.array([r[1] for r in res], bool)
        for j, r in enumerate(res):
            info[idx[j]].update(polish_ok=bool(r[1]),
                                heavy_rms_displacement_A=(r[3] or {}).get('heavy_rms_displacement_A'))
            if not r[1]:
                reason[idx[j]] = f'polish: {r[2]}'
        if done.any():
            polished = np.stack([res[j][0] for j in np.flatnonzero(done)])
            again = _validity(polished, context)
            sel = idx[done]
            out_pos[sel] = np.asarray(again.positions, dtype=np.float64)
            ok[sel] = np.asarray(again.ok, bool)
            reason[sel] = np.asarray(again.reason, dtype=object)
    return ValidityResult(positions=out_pos, ok=ok, reason=reason, info=info)
