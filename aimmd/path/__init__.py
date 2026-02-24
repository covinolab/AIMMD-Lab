"""
...
"""

from ._io import PathIO
from ._get import PathGet
from ._magic import PathMagic
from ._extract import PathExtract
from ._helpers import PathHelpers
from ._compute import PathCompute
from ._methods import PathMethods
from ._positions import PathPositions
from ._properties import PathProperties

class Path(
    PathHelpers,
    PathMagic,
    PathProperties,
    PathMethods,
    PathExtract,
    PathGet,
    PathPositions,
    PathCompute,
    PathIO):
    
    __init__ = PathHelpers._init

# visible objects
__all__ = ['Path']
