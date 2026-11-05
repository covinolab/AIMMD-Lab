"""
aimmd.network.joint_painn
=========================

Compatibility module, for one release: the PaiNN joint model (now
:mod:`aimmd.cgaimmd.flow_matching.painn`).

Every name resolves to the object of its new module, with one
``DeprecationWarning`` per old name that names the new spelling. The tables
are in :mod:`aimmd.cgaimmd._legacy`.
"""

from ..cgaimmd._legacy import module_getattr as _module_getattr

__getattr__ = _module_getattr(__name__)
