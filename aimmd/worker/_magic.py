"""
...
"""

# external
from abc import ABC

# Worker magic methods
class WorkerMagic(ABC):
    def __repr__(self):
        return f'Worker of {self.directory!r}'
