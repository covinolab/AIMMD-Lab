"""
...
"""

from ._run import LauncherRun
from ._build import LauncherBuild
from ._magic import LauncherMagic
from ._helpers import LauncherHelpers
from ._methods import LauncherMethods
from ._properties import LauncherProperties

class Launcher(
    LauncherHelpers,
    LauncherMagic,
    LauncherProperties,
    LauncherMethods,
    LauncherBuild,
    LauncherRun):
    
    __init__ = LauncherHelpers._init

__all__ = ['Launcher']
