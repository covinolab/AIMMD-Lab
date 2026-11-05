"""
aimmd.cgaimmd.restart
=====================

Continuing a run folder as ``chain_type='cgaimmd'``.

:func:`check_cgaimmd_restart` is what a C-GAIMMD shooting worker calls before
it loads its chains: it refuses a run folder with TPS history
(:data:`TPS_RESTART_AS_CGAIMMD`).

The tools for post-training a C-GAIMMD network on an existing run and for
continuing an RFPS run as C-GAIMMD (mirroring a run folder, checking and
setting aside descriptor caches written for another network) are ported
with the trainer's restart hooks (stream S5, part B), and join this module.

Standard library only.
"""

import glob
import os


TPS_RESTART_AS_CGAIMMD = False
"""Whether a run with TPS history may continue as ``chain_type='cgaimmd'``.

False, so :func:`check_cgaimmd_restart` raises ``NotImplementedError`` for a
chain folder holding ``tps_weights.npy``: such a chain would mix TPS
multiplicities on its old paths with RFPS-style weights on the new generated
shots, and whether AIMMD's reweighting of that mixture is right is not
established. Post-training on a TPS folder (which keeps the source's
``chain_type``) is unaffected. Setting it to True only removes the refusal; it
does not answer the question.
"""


def check_cgaimmd_restart(params, run_dir):
    """Refuse a run folder that cannot continue as ``chain_type='cgaimmd'``.

    Parameters
    ----------
    params : aimmd.Params
        The params of the relaunch; nothing is checked unless ``chain_type == 'cgaimmd'``.
    run_dir : str
        The run folder (``worker._directory`` inside a worker).

    Raises
    ------
    NotImplementedError
        If a ``chain*/`` folder holds ``tps_weights.npy`` (TPS history) and
        :data:`TPS_RESTART_AS_CGAIMMD` is False.
    """
    if getattr(params, 'chain_type', None) != 'cgaimmd':
        return
    if (not TPS_RESTART_AS_CGAIMMD
            and glob.glob(os.path.join(glob.escape(str(run_dir)), 'chain*', 'tps_weights.npy'))):
        raise NotImplementedError(
            "continuing a TPS run as chain_type='cgaimmd' is not supported: the chains would "
            "mix TPS multiplicities with RFPS-style weights, and whether AIMMD's reweighting "
            "of that mixture is right is not established (aimmd.cgaimmd.restart."
            "TPS_RESTART_AS_CGAIMMD)")
