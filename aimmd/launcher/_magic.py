"""
...
"""

# external
from abc import ABC

# Launcher magic methods
class LauncherMagic(ABC):
    def __len__(self):
        return len(self._params)

    def __add__(self, instance):
        return Launcher(self.params + instance.params,
                        self.directories + instance.directories,
                        self.termination_timeout)
    
    def __repr__(self):
        directories = [f'{directory!r}' for directory in self.directories]
        return (f'Launcher of {len(self)} '
                f'run{"s" if len(self) != 1 else ""} '
                f'({", ".join(directories)})')
