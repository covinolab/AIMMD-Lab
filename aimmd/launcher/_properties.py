"""
...
"""

# external
import os
import numpy as np
from abc import ABC

# Launcher properties
class LauncherProperties(ABC):
    
    @property
    def directories(self):
        return list(self._directories)

    @property
    def params(self):
        return list(self._params)

    @property
    def paths(self):
        return [params.path for params in self._params]

    @property
    def processes(self):
        return self._processes
