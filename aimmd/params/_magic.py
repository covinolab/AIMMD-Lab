"""
...
"""

# external
import os
from abc import ABC
from math import inf

# aimmd imports
from ..path import Path
from ._properties import ParamsProperties

# params' magic methods
class ParamsMagic(ABC):
    
    def __repr__(self):
        return f'Params {os.path.relpath(self.path)}'
    
    def __setattr__(self, name, value):
        """Enforce data types when reassigning params."""
        
        # can't set properties
        if isinstance(getattr(ParamsProperties, name, None), property):
            raise AttributeError(
                f"can't set aimmd.Params property {name!r}")
        
        # backup old value
        if hasattr(self, name):
            backup = getattr(self, name)
        else:
            backup = None
        
        try:
            self._setattr(name, value)
        
        # in case of errors, back to the old value
        except Exception as exception:
            if backup is not None:
                self.__dict__[name] = backup
            raise exception

    def __setitem__(self, name, value):
        """Same as setting an attribute."""
        return self.__setattr__(name, value)
    
    def __getitem__(self, name):
        """Same as getting from dict."""
        return self.__dict__[name]
    
    def __eq__(self, params):
        
        # different working directory
        if self.parent != params.parent:
            return False
        
        # simple string check
        return str(self) == str(params)
    
    def __str__(self, go_to=None):
        """Verbose string representation of params with descriptions and
        function bodies."""
        
        lines = []
        
        for name in self.__dataclass_fields__:
            if name == 'path':
                continue

            if self.engine == 'toy' and name.startswith('gmx'):
                continue

            if self.engine == 'gromacs' and name.startswith('toy'):
                continue

            # field and value
            field = self.__dataclass_fields__[name]
            if name == 'values_function' and self._default_values_function:
                value = None
            else:
                value = getattr(self, name)
            
            # function or network
            if hasattr(value, '__source__'):
                lines.append(value.__source__.rstrip('\n'))

            elif name in ('topology', 'gmx_mdp'):
                lines.append(f'{name} = {os.path.relpath(value, go_to)!r}')
            
            # initial paths
            elif name == 'initial_paths':
                filenames = []
                if self._reload_initial_paths:
                    initial_paths = value
                    for path in initial_paths:
                        if isinstance(path, Path):
                            path = path.fnames[0]
                        filename = os.path.relpath(path, go_to)
                        filenames.append(f'"{filename}"')
                lines.append(f'{name} = [{", ".join(filenames)}]')
            
            # all the rest
            elif value == inf:
                lines.append(f'{name} = float(\'inf\')')
            else:
                lines.append(f'{name} = {repr(value)}')
            
            # print description
            if desc := field.metadata.get("description", ""):
                lines.append(f"\"\"\"{desc}\"\"\"\n")
        
        return "\n".join(lines)
