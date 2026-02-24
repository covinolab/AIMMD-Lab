'''
aimmd — AI for molecular mechanism discovery
'''

# initialize
from ._init import initialize
initialize()

# basic
from ._config import *

# imports
from .core import utils
from .path import Path
from .params import Params
from .worker import Worker
from .launcher import Launcher
from .pathensemble import PathEnsemble

# all/version
__all__ = ['__version__']
__all__.extend(dir(_config))
__all__.remove('_initialized')
__all__.extend(['utils',
    'Path', 'Params', 'PathEnsemble',
    'Worker', 'Launcher'
])
__version__ = '0.1.0'

# summary/representation
def _summary() -> str:
    return f'AIMMD v{__version__} — AI for molecular mechanism discovery'

def __repr__():
    return _summary()
