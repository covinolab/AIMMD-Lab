"""
...
"""

from ._io import PathEnsembleIO
from ._magic import PathEnsembleMagic
from ._report import PathEnsembleReport
from ._helpers import PathEnsembleHelpers
from ._methods import PathEnsembleMethods
from ._project import PathEnsembleProject
from ._reweight import PathEnsembleReweight
from ._positions import PathEnsemblePositions
from ._properties import PathEnsembleProperties

class PathEnsemble(
    PathEnsembleHelpers,
    PathEnsembleMagic,
    PathEnsembleProperties,
    PathEnsembleMethods,
    PathEnsemblePositions,
    PathEnsembleProject,
    PathEnsembleReweight,
    PathEnsembleIO,
    PathEnsembleReport):

    __init__ = PathEnsembleHelpers._init

__all__ = ['PathEnsemble']
