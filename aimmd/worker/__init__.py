"""
...
"""

from ._run import WorkerRun
from ._free import WorkerFree
from ._shoot import WorkerShoot
from ._magic import WorkerMagic
from ._train import WorkerTrain
from ._helpers import WorkerHelpers
from ._simulate import WorkerSimulate
from ._properties import WorkerProperties

class Worker(
    WorkerHelpers,
    WorkerMagic,
    WorkerProperties,
    WorkerRun,
    WorkerSimulate,
    WorkerFree,
    WorkerShoot,
    WorkerTrain):
    
    __init__ = WorkerHelpers._init

__all__ = ['Worker']
