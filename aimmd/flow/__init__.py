"""
aimmd.flow
==========

Compatibility package: the old name of :mod:`aimmd.cgaimmd`, for one release.

C-GAIMMD was developed as "flow shooting" (F-AIMMD) in this package. Params
files of that time import from it (``from aimmd.flow.types import
ValidityResult``, ``from aimmd.flow import validate``). Its submodules are
compatibility modules: every name resolves to the object of its new module,
with one ``DeprecationWarning`` per old name that names the new spelling. The
tables are in :mod:`aimmd.cgaimmd._legacy`. New code imports
:mod:`aimmd.cgaimmd`.
"""

import sys as _sys

from ..cgaimmd import _legacy

_legacy.install_package_shim(_sys.modules[__name__])
__getattr__ = _legacy.module_getattr(__name__)
